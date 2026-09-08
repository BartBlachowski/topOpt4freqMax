function P = dr_dyn(RHO, move, rho0)
%DR_DYN  The preregistered dynamical quantities for a RETAINED trajectory.
%   Formulas are character-identical to the dynamical section of dr_telemetry;
%   this variant exists because retained archives carry RHO and a telemetry CSV
%   but not the solver `hist` struct.

NE = size(RHO,1); nO = size(RHO,2);
n2 = @(v) norm(v)/sqrt(NE);
n1 = @(v) sum(abs(v))/NE;
X = [rho0*ones(NE,1), RHO];
D = diff(X,1,2);
W = 10;

P = struct('W',W);
f = {'d1','d2','q2','d1_1','d2_1','q2_1','cosT','cos2','r2','r4', ...
     'stepNorm','stepNorm1','boundFrac','revFrac','path_W','net_W','net_ratio', ...
     'path_W1','net_W1','net_ratio1','q2_unsat','cosT_unsat','net_ratio_unsat'};
for i=1:numel(f), P.(f{i}) = nan(nO,1); end
P.undefQ2 = false(nO,1); P.undefCos = false(nO,1);

for k = 1:nO
    P.stepNorm(k)=n2(D(:,k)); P.stepNorm1(k)=n1(D(:,k));
    P.boundFrac(k)=mean(abs(D(:,k)) > 0.9*move(k));
end
for k = 2:nO
    P.d1(k)=n2(X(:,k+1)-X(:,k)); P.d1_1(k)=n1(X(:,k+1)-X(:,k));
    P.revFrac(k)=mean(D(:,k).*D(:,k-1) < 0);
    a=D(:,k); b=D(:,k-1); na=norm(a); nb=norm(b);
    if na>0 && nb>0, P.cosT(k)=(a.'*b)/(na*nb); else, P.undefCos(k)=true; end
end
for k = 3:nO
    P.d2(k)=n2(X(:,k+1)-X(:,k-1)); P.d2_1(k)=n1(X(:,k+1)-X(:,k-1)); P.r2(k)=P.d2(k);
    if P.d1(k)>0, P.q2(k)=P.d2(k)/P.d1(k); P.q2_1(k)=P.d2_1(k)/P.d1_1(k); else, P.undefQ2(k)=true; end
    a=D(:,k); b=D(:,k-2); na=norm(a); nb=norm(b);
    if na>0 && nb>0, P.cos2(k)=(a.'*b)/(na*nb); end
    sat = abs(D(:,k))>0.9*move(k) | abs(D(:,k-1))>0.9*move(k-1); m=~sat;
    if nnz(m)>0
        d1u=norm(X(m,k+1)-X(m,k)); d2u=norm(X(m,k+1)-X(m,k-1));
        if d1u>0, P.q2_unsat(k)=d2u/d1u; end
        a=D(m,k); b=D(m,k-1);
        if norm(a)>0 && norm(b)>0, P.cosT_unsat(k)=(a.'*b)/(norm(a)*norm(b)); end
    end
end
for k = 5:nO, P.r4(k)=n2(X(:,k+1)-X(:,k-3)); end
for k = W:nO
    pw = sum(arrayfun(@(j) n2(D(:,j)), (k-W+1):k));
    pw1= sum(arrayfun(@(j) n1(D(:,j)), (k-W+1):k));
    nw = n2(X(:,k+1)-X(:,k-W+1)); nw1 = n1(X(:,k+1)-X(:,k-W+1));
    P.path_W(k)=pw; P.net_W(k)=nw;   if pw>0,  P.net_ratio(k)=nw/pw;   end
    P.path_W1(k)=pw1;P.net_W1(k)=nw1;if pw1>0, P.net_ratio1(k)=nw1/pw1; end
    sat = any(abs(D(:,(k-W+1):k)) > 0.9*max(move((k-W+1):k)), 2); m=~sat;
    if nnz(m)>0
        pwu = sum(arrayfun(@(j) norm(D(m,j)), (k-W+1):k));
        nwu = norm(X(m,k+1)-X(m,k-W+1));
        if pwu>0, P.net_ratio_unsat(k)=nwu/pwu; end
    end
end
P.cancellation = 1 - P.net_ratio;
end
