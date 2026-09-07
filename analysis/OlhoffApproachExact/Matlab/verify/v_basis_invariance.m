function ok = v_basis_invariance()
% V_BASIS_INVARIANCE  Phase 3 acid test for the multiple-eigenvalue path.
%
%   For an N-fold eigenvalue the eigenvectors are NOT unique: "the multiplicity
%   of the eigenvalue lam~ in (11) implies that any linear combination of the
%   eigenvectors phi_j ... will satisfy the generalized eigenvalue problem (1b),
%   which implies that the eigenvectors are not unique" (Olhoff & Du 2014,
%   p. 281).  The increment subproblem (19) must therefore give the SAME
%   Delta_rho whichever orthonormal basis of the invariant subspace the
%   eigensolver happened to return.
%
%   Mechanism: replacing Phi_c by Phi_c*Q for orthogonal Q maps F -> Q'FQ, so
%   the sub-eigenvalue problem (12) -- whose unknowns are the eigenvalues of F,
%   Eq. (19d) -- is unchanged, and so is the LMI of SUBPROBLEM_LP.
%
%   T1  F transforms as Q'FQ under Phi_c -> Phi_c*Q, elementwise.
%   T2  the eigenvalues of F are invariant.
%   T3  the exact subproblem solution Delta_rho is invariant.
%   T4  the MMA subproblem solution Delta_rho is invariant.
%
%   A failure here means the cluster machinery depends on an arbitrary basis
%   choice and every bimodal result computed with it is meaningless.

fprintf('\n=== Phase 3: cluster basis invariance ===\n');
ok = true;
rng(7);

% ---- Build a genuine near-degenerate cluster on a symmetric structure ------
% A square domain with symmetric supports gives an exactly double mode.
nelx = 20; nely = 20;
mdl  = build_sq_model(nelx, nely);
rho  = 0.5*ones(mdl.nEl,1);

[lam, Phi] = solve_modes(mdl, rho, 6);
% find the tightest adjacent pair
[~, j0] = min(diff(lam(1:5))./lam(1:4));
cl   = j0:j0+1;
N    = 2;
fprintf('  cluster modes %d-%d:  lam = [%.6g %.6g],  relative spread = %.3e\n', ...
    cl(1), cl(2), lam(cl(1)), lam(cl(2)), (lam(cl(2))-lam(cl(1)))/lam(cl(1)));

lref = lam(cl(1));
Fe   = generalized_gradients(rho, lref, Phi(:,cl), mdl.cMat, mdl.Ke_phys, ...
                             mdl.Me_phys, 3, 'du2007_c1');

% ---- Random orthogonal rotation of the cluster basis ----------------------
[Q, ~] = qr(randn(N));
Phi_rot = Phi(:,cl) * Q;
Fe_rot  = generalized_gradients(rho, lref, Phi_rot, mdl.cMat, mdl.Ke_phys, ...
                                mdl.Me_phys, 3, 'du2007_c1');

%% T1 elementwise F -> Q'FQ
worst = 0;
for e = 1:mdl.nEl
    A = squeeze(Fe(e,:,:));
    B = squeeze(Fe_rot(e,:,:));
    worst = max(worst, max(max(abs(B - Q'*A*Q))) / max(max(max(abs(A))),1e-30));
end
ok = check('T1 F -> Q''FQ elementwise ', worst < 1e-12, ok, sprintf('%.3e', worst));

%% T2 eigenvalues of F invariant for a random Delta_rho
d  = 0.05*(2*rand(mdl.nEl,1)-1);
Fa = reshape(reshape(Fe,     mdl.nEl, N*N)' * d, N, N);
Fb = reshape(reshape(Fe_rot, mdl.nEl, N*N)' * d, N, N);
ea = sort(real(eig((Fa+Fa')/2)));
eb = sort(real(eig((Fb+Fb')/2)));
rel = max(abs(ea-eb))/max(max(abs(ea)),1e-30);
ok = check('T2 eig(F) invariant      ', rel < 1e-12, ok, sprintf('%.3e', rel));

%% T3 / T4 subproblem solutions invariant
sp = struct('mode','nth','rho',rho,'volfrac',0.5,'rho_min',1e-3,'move',0.05);
sp.up = struct('L', lref*ones(N,1), 'Fe', Fe, 'guard', ...
               struct('lam', lam(cl(2)+1), 'grad', ...
               compute_elem_sensitivity(rho, lam(cl(2)+1), Phi(:,cl(2)+1), ...
                   mdl.cMat, mdl.Ke_phys, mdl.Me_phys, mdl.free, mdl.nDof, 3, 'du2007_c1')));
sp_rot    = sp;
sp_rot.up.Fe = Fe_rot;

a = subproblem_lp(sp);
b = subproblem_lp(sp_rot);
dd = norm(a.drho - b.drho)/max(norm(a.drho),1e-30);
db = abs(a.beta - b.beta)/max(abs(a.beta),1e-30);
ok = check('T3 LP  drho invariant    ', dd < 1e-8 && db < 1e-10, ok, ...
           sprintf('drho %.3e, beta %.3e', dd, db));

% MMA is basis-invariant only in the LIMIT.  At a degenerate cluster the
% eigenvectors of G(0) = lam~ I are arbitrary, so the "N smooth constraints"
% embedding starts from an arbitrary subgradient of mu_min and the first steps
% depend on the basis.  Measured decay of the basis difference with the inner
% budget (20 x 20 plate, N = 2, move = 0.05):
%     budget    20      60     200     600    2000
%     |ddrho|  1.4e-1  5.1e-2  5.8e-5  1.1e-4  9.4e-5
% i.e. it is non-convergence, not a formulation defect, and it plateaus near
% 1e-4 where MMA's own asymptote termination floors out.  The tolerance below
% is set from that measurement.  The LP/LMI path (T3) has no such dependence,
% which is one reason it is the default solver.
o = struct('max_iter', 300, 'verbose', false);
a = subproblem_mma(sp,     o);
b = subproblem_mma(sp_rot, o);
dd = norm(a.drho - b.drho)/max(norm(a.drho),1e-30);
db = abs(a.beta - b.beta)/max(abs(a.beta),1e-30);
ok = check('T4 MMA drho invariant    ', dd < 1e-3 && db < 1e-5, ok, ...
           sprintf('drho %.3e, beta %.3e (limit-only, see comment)', dd, db));

% And report how far MMA is from the exact optimum at that budget.
ex   = subproblem_lp(sp);
gapa = (ex.obj - a.obj)/ex.lam_ref;
fprintf('       MMA optimality gap vs exact LP: %.3e (scaled), stop = %s after %d iters\n', ...
    gapa, a.stop_reason, a.n_iters);

fprintf('\n=== v_basis_invariance: %s ===\n\n', tf2s(ok));
end

%% =======================================================================
function mdl = build_sq_model(nelx, nely)
% Square plate, pinned at all four corners: symmetry gives exactly double modes.
    L = 1; H = 1; nu = 0.3; t = 1; E0 = 1e7; rho0 = 1;
    dx = L/nelx; dy = H/nely;
    mdl.nEl  = nelx*nely;
    mdl.nDof = 2*(nelx+1)*(nely+1);
    [Ke, Me] = fe_q4_exact(nu, t, dx, dy);
    mdl.Ke_phys = E0*Ke;  mdl.Me_phys = rho0*Me;
    nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
    cVec = reshape(2*nodeNrs(1:nely,1:nelx)+1, mdl.nEl, 1);
    mdl.cMat = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
                cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];
    [Il, Jl] = find(tril(ones(8)));
    mdl.iK = reshape(mdl.cMat(:,Il)', [], 1);
    mdl.jK = reshape(mdl.cMat(:,Jl)', [], 1);
    mdl.Ke_l = mdl.Ke_phys(sub2ind([8,8],Il,Jl));
    mdl.Me_l = mdl.Me_phys(sub2ind([8,8],Il,Jl));
    corners = [nodeNrs(1,1) nodeNrs(1,end) nodeNrs(end,1) nodeNrs(end,end)];
    fixed = [2*corners-1, 2*corners];
    mdl.free = setdiff(1:mdl.nDof, fixed);
end

function [lam, Phi] = solve_modes(mdl, rho, nm)
    [K, M] = assemble_KM_exact(rho, mdl.Ke_l, mdl.Me_l, mdl.iK, mdl.jK, ...
                               mdl.nDof, 3, 'du2007_c1');
    Kf = K(mdl.free,mdl.free);  Mf = M(mdl.free,mdl.free);
    o = struct('tol',eps,'maxit',5000);
    o.v0 = ones(numel(mdl.free),1); o.v0(2:2:end) = -1; o.v0 = o.v0/norm(o.v0);
    [V, D] = eigs(Kf, Mf, nm, 'SM', o);
    [lam, ix] = sort(real(diag(D)));  V = real(V(:,ix));
    for j = 1:nm
        s = sqrt(abs(V(:,j)'*(Mf*V(:,j))));
        if s > 0, V(:,j) = V(:,j)/s; end
    end
    Phi = zeros(mdl.nDof, nm);  Phi(mdl.free,:) = V;
end

function ok = check(name, cond, ok, extra)
    if cond, s = 'PASS'; else, s = 'FAIL'; end
    fprintf('  %s  %s   %s\n', name, s, extra);
    ok = ok && cond;
end

function s = tf2s(tf)
    if tf, s = 'PASS'; else, s = 'FAIL'; end
end
