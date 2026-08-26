function test_gradients()
%TEST_GRADIENTS  Finite-difference verification of Du & Olhoff eq. (19) and of
%   the subeigenvalue problem (25d) in its ERRATUM form.
maxNumCompThreads(1); addpath('fem'); addpath('algo');
rng(0);
p = 3; mi = '4';
base = struct('a',8,'b',1,'t',1,'E',1e7,'nu',0.3,'rhom',1,'massType','consistent', ...
              'axial','both','support','mid','bc','a','elemType','Q4', ...
              'nelx',40,'nely',6);
mdl = model2D(base);
rho = 0.4 + 0.2*rand(mdl.nele,1);
[K,M] = assemble2D(mdl,rho,p,mi);
[~,Phi,lam] = eigSolve(K,M,4,'dense');

% ================= (a) simple eigenvalue:  f_jj == grad(lambda_j) =========
h = 1e-4;
fprintf('(a) simple eigenvalue gradients, eq.(19)-(21), central FD h=%g\n',h);
fprintf('    %5s %5s %16s %16s %10s\n','mode','elem','analytic','central FD','rel err');
worst = 0;
for j = 1:3
    Fj = genGrad(mdl,rho,p,mi,Phi,lam(j),j);
    fj = Fj(:,1,1);
    for e = [1 37 100 155 199]
        r1=rho; r1(e)=r1(e)+h; [K1,M1]=assemble2D(mdl,r1,p,mi); l1=eigSolve(K1,M1,4,'dense').^2;
        r2=rho; r2(e)=r2(e)-h; [K2,M2]=assemble2D(mdl,r2,p,mi); l2=eigSolve(K2,M2,4,'dense').^2;
        fd = (l1(j)-l2(j))/(2*h);
        re = abs(fj(e)-fd)/max(abs(fd),eps);
        worst = max(worst,re);
        fprintf('    %5d %5d %16.8e %16.8e %10.2e\n',j,e,fj(e),fd,re);
    end
end
fprintf('    worst relative error: %.2e\n', worst);

% ================= (b) TRUE degenerate pair ==============================
% Square domain clamped on all four edges -> mode pair 2,3 is degenerate by
% the 90-degree rotational symmetry.  This is where the off-diagonal terms of
% (25d) actually matter.
cfg2 = base; cfg2.a=4; cfg2.b=4; cfg2.nelx=24; cfg2.nely=24;
m2 = model2D(cfg2);
nn = m2.nodenrs;
edge = unique([nn(1,:) nn(end,:) nn(:,1)' nn(:,end)']);
fixed = unique([2*edge(:)-1; 2*edge(:)]);
m2.free  = setdiff((1:m2.ndof)',fixed);
m2.fixed = fixed;
rho2 = 0.5*ones(m2.nele,1);
[Ka,Ma] = assemble2D(m2,rho2,p,mi);
[~,Phia,lama] = eigSolve(Ka,Ma,6,'dense');
relgap = abs(diff(lama))./lama(1:end-1);
j = find(relgap < 1e-8, 1);
if isempty(j)
    fprintf('\n(b) SKIPPED: no degenerate pair (min rel gap %.2e)\n',min(relgap)); return
end
idx = [j j+1];
fprintf('\n(b) degenerate pair: modes %d,%d  lambda=%.10g,%.10g  rel gap %.1e\n', ...
        j,j+1,lama(j),lama(j+1),relgap(j));
lamT = mean(lama(idx));
Fd = genGrad(m2,rho2,p,mi,Phia,lamT,idx);
A0 = [Fd(:,1,1) Fd(:,1,2) Fd(:,2,2)];
for hh = [1e-5 1e-6]
    drho = hh*(2*rand(m2.nele,1)-1);
    A = [A0(:,1)'*drho, A0(:,2)'*drho; A0(:,2)'*drho, A0(:,3)'*drho];
    pred     = sort(eig(A));
    predDiag = sort([A(1,1); A(2,2)]);
    [Kb,Mb] = assemble2D(m2,rho2+drho,p,mi);
    lamb = eigSolve(Kb,Mb,6,'dense').^2;
    act  = sort(lamb(idx)-lama(idx));
    fprintf('   |drho|_inf = %g\n',hh);
    fprintf('     actual        %15.7e %15.7e\n',act(1),act(2));
    fprintf('     (25d) full    %15.7e %15.7e   rel err %8.1e %8.1e\n', ...
            pred(1),pred(2),abs(pred(1)-act(1))/abs(act(1)),abs(pred(2)-act(2))/abs(act(2)));
    fprintf('     diagonal only %15.7e %15.7e   rel err %8.1e %8.1e\n', ...
            predDiag(1),predDiag(2), ...
            abs(predDiag(1)-act(1))/abs(act(1)),abs(predDiag(2)-act(2))/abs(act(2)));
    fprintf('     off-diag A_12 = %.4e  (vs diag %.4e, %.4e)\n',A(1,2),A(1,1),A(2,2));
end
end
