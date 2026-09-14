function E = fi_eval(S, rho, want)
%FI_EVAL  Evaluate the physical and filtered gradient fields at one density.
%
%   READ-ONLY.  Assembles, eigensolves and forms the first-mode generalized
%   gradient exactly as olhoffSolve does, then applies the production
%   sensitivity filter.  Nothing is written back to any density.
%
%   E.gPhys  = F(:,1,1) raw   = d(lambda_1)/d(rho)      (M-orthonormal modes)
%   E.gFilt  = applyFilter(flt, rho, gPhys)             = A(rho)*gPhys
%   E.omega  = omega_1..omega_Jcalc                     (multiplicity control)
%   E.gap12  = (omega_2-omega_1)/omega_1
%
%   'want' may be 'full' to also return the complete N x N tensor and fJJ, as
%   olhoffSolve builds them.

if nargin < 3, want = 'mode1'; end
rho = rho(:);

[K,M] = assemble2D(S.mdl, rho, S.p, S.massCfg);
[w, Phi, lam] = eigSolve(K, M, S.Jcalc, S.solver);

E = struct();
E.omega = w(1:S.Jcalc);
E.lam   = lam(1:S.Jcalc);
E.gap12 = (w(2)-w(1))/w(1);

if strcmp(want,'full')
    [N, ~] = olh.multi.detect(S.cfg, w, S.n, S.Jcalc, []);
    idx = S.n:(S.n+N-1);
    lamTild = lam(S.n);
    F = genGrad(S.mdl, rho, S.p, S.massCfg, Phi, lamTild, idx);
    for j = 1:N
        Gj = genGrad(S.mdl, rho, S.p, S.massCfg, Phi, lam(idx(j)), idx(j));
        F(:,j,j) = Gj(:,1,1);
    end
    FJ = genGrad(S.mdl, rho, S.p, S.massCfg, Phi, lam(S.n+N), S.n+N);
    E.Fraw = F;  E.fJJraw = FJ(:,1,1);
    E.N = N;  E.idx = idx;  E.dOff = lam(idx) - lam(idx(1));
    Ff = F;
    for s = 1:N
        for k = s:N
            v = applyFilter(S.flt, rho, Ff(:,s,k));
            Ff(:,s,k) = v;  Ff(:,k,s) = v;
        end
    end
    E.F = Ff;
    E.fJJ = applyFilter(S.flt, rho, E.fJJraw);
    E.gPhys = F(:,1,1);
    E.gFilt = Ff(:,1,1);
else
    G = genGrad(S.mdl, rho, S.p, S.massCfg, Phi, lam(1), 1);
    E.gPhys = G(:,1,1);
    E.gFilt = applyFilter(S.flt, rho, E.gPhys);
end
end
