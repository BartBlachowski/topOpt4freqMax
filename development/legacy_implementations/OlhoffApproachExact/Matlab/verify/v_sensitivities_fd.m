function ok = v_sensitivities_fd()
% V_SENSITIVITIES_FD  Phase 2/3: finite-difference audit of all derivatives.
%
%   Checks, on a small CC beam with a deliberately non-uniform density so that
%   both branches of the mass law are exercised:
%
%   T1  d m / d rho for every mass interpolation mode, on BOTH sides of the
%       rho = 0.1 kink of Du & Olhoff (2007) Eq. (4)/(4a)/(4b).
%   T2  the simple-eigenvalue sensitivity of Olhoff & Du (2014) Eq. (4)/(5),
%       against central differences on lambda_j.
%   T3  the DIAGONAL generalized gradients f_ss, which Eqs. (14)/(15) require to
%       equal the simple sensitivity of mode s.
%   T4  the OFF-DIAGONAL generalized gradients f_sk, s != k, against central
%       differences of the projected matrix Phi_c' (K - lam~ M) Phi_c.  This is
%       the check that was missing from every previous campaign: without it the
%       multiple-eigenvalue path (Eq. 12/13) is untested.
%   T5  exact symmetry f_sk == f_ks.
%
%   Returns true if every test passes its tolerance.

fprintf('\n=== Phase 2/3: finite-difference audit of derivatives ===\n');
ok = true;

%% ---------------------------------------------------------------- T1
fprintf('\n T1  mass-law derivatives across the rho = 0.1 kink\n');
modes = {'olhoff2014_pow','linear','du2007_step','du2007_c0','du2007_c1'};
pts   = [0.02 0.05 0.09 0.11 0.3 0.7 1.0];
hfd   = 1e-7;
for k = 1:numel(modes)
    worst = 0;
    for p = pts
        [~, dm] = mass_interp(p, modes{k});
        mp = mass_interp(min(p+hfd,1), modes{k});
        mm = mass_interp(max(p-hfd,0), modes{k});
        num = (mp - mm)/( min(p+hfd,1) - max(p-hfd,0) );
        worst = max(worst, abs(num - dm)/max(abs(dm),1e-12));
    end
    st = pass(worst < 1e-5);
    fprintf('     %-16s max rel err = %.3e   %s\n', modes{k}, worst, st);
    ok = ok && worst < 1e-5;
end

%% -------------------------------------------------- model for T2..T5
nelx = 24; nely = 6;
mdl  = build_model(nelx, nely, 'CC', 3, 'du2007_c1');
rng(0);
rho  = 0.05 + 0.9*rand(mdl.nEl,1);          % spans both mass branches
rho(1:10) = 0.03;                           % force the low-density branch

[lam, Phi] = solve_modes(mdl, rho, 6);
els = round(linspace(1, mdl.nEl, 12));
h   = 1e-5;     % best-conditioned central-difference step for lambda ~ 1e3-1e5

% NOTE on the error measure.  The element gradients of one mode span a huge
% dynamic range here (max |dlam/drho_e| ~ 5.6e3, min ~ 0.22, i.e. 2.6e4:1), so a
% per-element relative error is dominated by finite-difference roundoff on the
% smallest entries and says nothing about correctness.  The tests below use the
% standard FD gradient-check measure -- error normalized by the SCALE of the
% gradient (its max magnitude) -- and additionally report the per-element
% relative error restricted to entries above 1 % of that scale.

%% ---------------------------------------------------------------- T2
fprintf('\n T2  simple-eigenvalue sensitivity, Eq. (4)/(5)\n');
worst_scaled = 0; worst_rel = 0;
for j = 1:3
    ana = compute_elem_sensitivity(rho, lam(j), Phi(:,j), mdl.cMat, ...
              mdl.Ke_phys, mdl.Me_phys, mdl.free, mdl.nDof, 3, 'du2007_c1');
    scale = max(abs(ana));
    for e = els
        rp = rho; rp(e) = rp(e) + h;
        rm = rho; rm(e) = rm(e) - h;
        lp = solve_modes(mdl, rp, 6);
        lm = solve_modes(mdl, rm, 6);
        num = (lp(j) - lm(j)) / (2*h);
        err = abs(num - ana(e));
        worst_scaled = max(worst_scaled, err/scale);
        if abs(ana(e)) > 0.01*scale
            worst_rel = max(worst_rel, err/abs(ana(e)));
        end
    end
end
fprintf('     modes 1-3, 12 elements: err/scale = %.3e, rel err (entries > 1%% of scale) = %.3e   %s\n', ...
    worst_scaled, worst_rel, pass(worst_scaled < 1e-5 && worst_rel < 1e-4));
ok = ok && worst_scaled < 1e-5 && worst_rel < 1e-4;

%% ---------------------------------------------------------------- T3
fprintf('\n T3  f_ss == grad lambda_s  (Eqs. 14, 15)\n');
cl   = 1:3;
lref = lam(1);
Fe   = generalized_gradients(rho, lref, Phi(:,cl), mdl.cMat, mdl.Ke_phys, ...
                             mdl.Me_phys, 3, 'du2007_c1');
worst = 0;
for s = 1:numel(cl)
    ana = compute_elem_sensitivity(rho, lref, Phi(:,cl(s)), mdl.cMat, ...
              mdl.Ke_phys, mdl.Me_phys, mdl.free, mdl.nDof, 3, 'du2007_c1');
    worst = max(worst, max(abs(Fe(:,s,s) - ana)) / max(max(abs(ana)),1e-12));
end
fprintf('     max rel err = %.3e   %s\n', worst, pass(worst<1e-12));
ok = ok && worst < 1e-12;

%% ---------------------------------------------------------------- T4
fprintf('\n T4  off-diagonal f_sk vs central differences of Phi_c''(K - lam~ M)Phi_c\n');
worst_scaled = 0; worst_rel = 0;
Fscale = max(max(max(abs(Fe))));
for e = els
    rp = rho; rp(e) = rp(e) + h;
    rm = rho; rm(e) = rm(e) - h;
    Ap = proj_matrix(mdl, rp, Phi(:,cl), lref);
    Am = proj_matrix(mdl, rm, Phi(:,cl), lref);
    num = (Ap - Am) / (2*h);                       % N x N, = F_e of Eq. (13)
    for s = 1:numel(cl)
        for kk = 1:numel(cl)
            if s == kk, continue, end
            err = abs(num(s,kk) - Fe(e,s,kk));
            worst_scaled = max(worst_scaled, err/Fscale);
            if abs(Fe(e,s,kk)) > 0.01*Fscale
                worst_rel = max(worst_rel, err/abs(Fe(e,s,kk)));
            end
        end
    end
end
fprintf('     err/scale = %.3e, rel err (entries > 1%% of scale) = %.3e   %s\n', ...
    worst_scaled, worst_rel, pass(worst_scaled < 1e-6 && worst_rel < 1e-4));
ok = ok && worst_scaled < 1e-6 && worst_rel < 1e-4;

%% ---------------------------------------------------------------- T5
fprintf('\n T5  symmetry f_sk == f_ks (exact)\n');
asym = 0;
for s = 1:numel(cl)
    for kk = 1:numel(cl)
        asym = max(asym, max(abs(Fe(:,s,kk) - Fe(:,kk,s))));
    end
end
fprintf('     max |f_sk - f_ks| = %.3e   %s\n', asym, pass(asym == 0));
ok = ok && asym == 0;

fprintf('\n=== v_sensitivities_fd: %s ===\n\n', pass(ok));
end

%% =======================================================================
function A = proj_matrix(mdl, rho, Phi_c, lref)
% Phi_c' (K(rho) - lref M(rho)) Phi_c, with Phi_c held FIXED.  Its derivative
% wrt rho_e is exactly the f_sk matrix of Eq. (13) for element e.
    [K, M] = assemble_KM_exact(rho, mdl.Ke_l, mdl.Me_l, mdl.iK, mdl.jK, ...
                               mdl.nDof, 3, 'du2007_c1');
    A = Phi_c' * (K - lref*M) * Phi_c;
end

function [lam, Phi] = solve_modes(mdl, rho, nm)
    [K, M] = assemble_KM_exact(rho, mdl.Ke_l, mdl.Me_l, mdl.iK, mdl.jK, ...
                               mdl.nDof, 3, 'du2007_c1');
    Kf = K(mdl.free, mdl.free);  Mf = M(mdl.free, mdl.free);
    o = struct('tol',1e-14,'maxit',2000);
    o.v0 = ones(numel(mdl.free),1); o.v0(2:2:end) = -1; o.v0 = o.v0/norm(o.v0);
    [V, D] = eigs(Kf, Mf, nm, 'SM', o);
    [lam, ix] = sort(real(diag(D)));  V = real(V(:,ix));
    for j = 1:nm
        s = sqrt(abs(V(:,j)'*(Mf*V(:,j))));
        if s > 0, V(:,j) = V(:,j)/s; end
    end
    Phi = zeros(mdl.nDof, nm);  Phi(mdl.free,:) = V;
end

function mdl = build_model(nelx, nely, bc, ~, ~)
    L = 8; H = 1; nu = 0.3; t = 1; E0 = 1e7; rho0 = 1;
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
    fixed = build_supports_exact(bc, nodeNrs);
    mdl.free = setdiff(1:mdl.nDof, fixed);
end

function s = pass(tf)
    if tf, s = 'PASS'; else, s = 'FAIL'; end
end
