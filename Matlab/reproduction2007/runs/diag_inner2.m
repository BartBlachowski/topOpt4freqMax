function diag_inner2(moves, maxInner)
%DIAG_INNER2  How far does the inner loop actually travel as a function of the
%   move limit?  Exposes whether the MMA subproblem is properly scaled.
maxNumCompThreads(1);
if nargin<1, moves=[0.1 0.05 0.02 0.01 0.005]; end
if nargin<2, maxInner=200; end
cfg = defaultCfg();
mdl = model2D(cfg); NE = mdl.nele; rho = 0.5*ones(NE,1);
flt = prepFilter(cfg.nelx,cfg.nely,cfg.rminEl);
[K,M] = assemble2D(mdl,rho,cfg.p,cfg.massInterp);
[~,Phi,lam] = eigSolve(K,M,5,'eigs');
F  = genGrad(mdl,rho,cfg.p,cfg.massInterp,Phi,lam(1),1);
FJ = genGrad(mdl,rho,cfg.p,cfg.massInterp,Phi,lam(2),2);
F(:,1,1) = applyFilter(flt,rho,F(:,1,1));
fJJ = applyFilter(flt,rho,FJ(:,1,1));
fprintf('scale check: lam1=%.4g  max|f11|=%.4g  max|f11|/lam1=%.3g\n', ...
        lam(1), max(abs(F(:,1,1))), max(abs(F(:,1,1)))/lam(1));
fprintf('%-8s %8s %10s %12s %12s %8s\n','move','nInner','conv','max|drho|','sum drho','sqrt(b)');
for mv = moves
    ctx = struct('F',F,'fJJ',fJJ,'lam',lam(1),'lamJ',lam(2),'rho',rho, ...
       'rhomin',cfg.rhomin,'volfrac',cfg.volfrac,'move',mv, ...
       'maxInner',maxInner,'tolInner',1e-2,'minInner',5,'offDiag',true);
    [drho,st] = innerLoop(ctx);
    fprintf('%-8.4f %8d %10d %12.3e %12.3e %8.2f\n', ...
        mv, st.nInner, st.conv, max(abs(drho)), sum(drho), sqrt(max(st.beta,0)));
end
end
