function [rho_final, hist] = topopt_freq_exact(cfg)
% TOPOPT_FREQ_EXACT  Paper-faithful Du & Olhoff (2007) frequency maximization.
%
%   [rho_final, hist] = topopt_freq_exact(cfg)
%
%   Implements the nested increment formulation of Section 3.5:
%     Outer loop: assemble K,M; solve eigenproblem; detect multiplicity N;
%                 compute and filter generalized gradients fsk; call inner loop.
%     Inner loop: MMA increment subproblem on [beta; Delta_rho] (inner_loop_mma).
%     Update:     rho := rho + alpha * Delta_rho  (under-relaxed to damp 2-cycles)
%     Stop:       norm(Delta_rho)/sqrt(nEl) < outer_tol.
%
%   Returns the FINAL CONVERGED rho, not a best-seen tracking.
%
%   cfg struct fields (all optional; defaults match paper Section 4.1):
%     .L, .H          geometry (default 8, 1)
%     .nelx, .nely    mesh (default 40, 5)
%     .E0             Young's modulus (default 1e7)
%     .nu             Poisson ratio (default 0.3)
%     .rho0           mass density (default 1)
%     .t              thickness (default 1)
%     .volfrac        volume fraction upper bound (default 0.5)
%     .rho_min        minimum density (default 1e-3)
%     .penal          SIMP stiffness power (default 3)
%     .mass_mode      mass interpolation mode (default 'du2007_c1')
%     .rmin_elem      filter radius in element units (default 2.5)
%     .support_type   boundary conditions: 'SS','CS','CC' (default 'CC')
%     .n_target       target mode index (default 1)
%     .n_modes        modes to compute each outer iter (default n_target+3)
%     .mult_tol       cluster detection tolerance (default 1e-3)
%     .outer_max_iter outer loop limit (default 300)
%     .outer_tol      convergence: norm(drho)/sqrt(nEl) < outer_tol (default 1e-3)
%     .inner_max_iter inner MMA iteration limit (default 30)
%     .inner_tol      inner convergence tolerance (default 1e-4)
%     .move_lim       trust-region radius per inner MMA iteration (default 0.2)
%     .outer_move     outer trust-region: max |Delta_rho_e| per outer iter (default 0.2)
%     .alpha          under-relaxation for outer update: rho += alpha*drho (default 0.5)
%     .verbose        print progress table (default true)
%
%   Outputs:
%     rho_final   nEl x 1   converged physical density
%     hist        struct with fields omega, beta, volume, N, inner_iters,
%                 drho_norm, outer_iters
%
%   Reference: Du & Olhoff (2007), Struct Multidisc Optim 34:91-110.

if nargin < 1 || isempty(cfg), cfg = struct(); end
cfg = set_defaults(cfg);

L    = cfg.L;   H    = cfg.H;
nelx = cfg.nelx; nely = cfg.nely;
E0   = cfg.E0;  nu   = cfg.nu; rho0 = cfg.rho0; t = cfg.t;
volfrac  = cfg.volfrac;   rho_min  = cfg.rho_min;
penal    = cfg.penal;     mass_mode = cfg.mass_mode;
rmin_elem = cfg.rmin_elem;
support_type = cfg.support_type;
n_target = cfg.n_target;  n_modes  = cfg.n_modes;
mult_tol = cfg.mult_tol;
outer_max_iter = cfg.outer_max_iter;
outer_tol      = cfg.outer_tol;
inner_max_iter = cfg.inner_max_iter;
inner_tol      = cfg.inner_tol;
move_lim       = cfg.move_lim;
outer_move     = cfg.outer_move;
alpha          = cfg.alpha;
verbose        = cfg.verbose;

%% ------------------------------------------------------------------
%  Mesh and FE setup
% ------------------------------------------------------------------
dx = L/nelx;  dy = H/nely;
nEl  = nelx * nely;
nDof = 2 * (nelx+1) * (nely+1);

[Ke_star, Me_star] = fe_q4_exact(nu, t, dx, dy);
Ke_phys = E0   * Ke_star;
Me_phys = rho0 * Me_star;

nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
cVec    = reshape(2*nodeNrs(1:nely,1:nelx)+1, nEl, 1);
cMat    = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
           cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];

[Il, Jl] = find(tril(ones(8)));
iK = reshape(cMat(:,Il)', [], 1);
jK = reshape(cMat(:,Jl)', [], 1);
Ke_phys_l = Ke_phys(sub2ind([8,8], Il, Jl));
Me_phys_l = Me_phys(sub2ind([8,8], Il, Jl));

fixed = build_supports_exact(support_type, nodeNrs);
free  = setdiff(1:nDof, fixed);
nFree = numel(free);

[h_filt, Hs_filt] = build_filter(nelx, nely, rmin_elem);

%% ------------------------------------------------------------------
%  Initialise density and history
% ------------------------------------------------------------------
rho = volfrac * ones(nEl, 1);

hist.omega       = nan(outer_max_iter, n_modes);
hist.beta        = nan(outer_max_iter, 1);
hist.volume      = nan(outer_max_iter, 1);
hist.N           = nan(outer_max_iter, 1);
hist.inner_iters = nan(outer_max_iter, 1);
hist.drho_norm   = nan(outer_max_iter, 1);
hist.outer_iters = 0;

if verbose
    fprintf('\n');
    fprintf(' %-4s  %-10s  %-10s  %-3s  %-6s  %-9s  %-10s  %-5s\n', ...
        'iter', 'omega_1', 'omega_2', ' N ', 'vol', 'drho/sqrt', 'beta(rad/s)', 'in');
    fprintf(' %s\n', repmat('-', 1, 72));
end

opts_eig.tol   = 1e-10;
opts_eig.maxit = 600;

%% ------------------------------------------------------------------
%  Outer loop
% ------------------------------------------------------------------
for out_it = 1:outer_max_iter

    %% --- Assemble K, M ---
    [K, M] = assemble_KM_exact(rho, Ke_phys_l, Me_phys_l, iK, jK, nDof, penal, mass_mode);
    Kf = K(free, free);
    Mf = M(free, free);

    %% --- Eigensolve ---
    [V, D, flag] = eigs(Kf, Mf, n_modes, 'SM', opts_eig);
    if flag ~= 0
        opts_r.tol   = 1e-8;
        opts_r.maxit = 1500;
        opts_r.p     = min(nFree-1, max(40, 4*n_modes));
        [V, D, flag] = eigs(Kf, Mf, n_modes, 'SM', opts_r);
        if flag ~= 0
            if verbose
                fprintf(' WARN iter %d: eigs failed (flag=%d), skipping.\n', out_it, flag);
            end
            hist.outer_iters = out_it;
            continue
        end
    end

    [lam, idx] = sort(real(diag(D)));
    V = real(V(:, idx));

    for j = 1:n_modes
        v = V(:,j);
        sc = sqrt(abs(v' * (Mf * v)));
        if sc > 1e-14, V(:,j) = v / sc; end
    end

    omega = sqrt(max(lam, 0));
    Phi   = zeros(nDof, n_modes);
    for j = 1:n_modes, Phi(free, j) = V(:, j); end

    %% --- Multiplicity detection ---
    [N, J_idx, cluster_idx] = detect_multiplicity(omega, n_target, mult_tol);
    lambda_bar = mean(lam(cluster_idx));

    %% --- Generalized gradients and sensitivity filter ---
    Phi_cluster = Phi(:, cluster_idx);
    fsk_raw = compute_generalized_gradients(rho, lambda_bar, Phi_cluster, ...
                  cMat, Ke_phys, Me_phys, penal, mass_mode);

    fsk_filt = zeros(size(fsk_raw));
    for s = 1:N
        for k = 1:N
            fsk_filt(:,s,k) = apply_sensitivity_filter( ...
                fsk_raw(:,s,k), rho, h_filt, Hs_filt, nely, nelx);
        end
    end

    %% --- J-mode gradient ---
    if J_idx > 0
        lambda_J = lam(J_idx);
        dlam_J_raw = compute_elem_sensitivity(rho, lambda_J, Phi(:,J_idx), ...
            cMat, Ke_phys, Me_phys, free, nDof, penal, mass_mode);
        dlam_J = apply_sensitivity_filter(dlam_J_raw, rho, h_filt, Hs_filt, nely, nelx);
    else
        lambda_J = Inf;
        dlam_J   = [];
    end

    %% --- Inner loop (MMA increment subproblem) ---
    [drho, beta_fin, i_hist] = inner_loop_mma(rho, lambda_bar, fsk_filt, ...
        lambda_J, dlam_J, volfrac, rho_min, inner_max_iter, inner_tol, ...
        move_lim, outer_move);

    %% --- Update with under-relaxation ---
    % alpha < 1 damps the 2-cycle oscillation common in nested increment
    % methods without changing the fixed-point condition (drho=0 at optimum).
    rho_new   = max(rho_min, min(1, rho + alpha * drho));
    drho_norm = norm(rho_new - rho) / sqrt(nEl);

    hist.omega(out_it, :)    = omega(:)';
    hist.beta(out_it)        = beta_fin;
    hist.volume(out_it)      = mean(rho_new);
    hist.N(out_it)           = N;
    hist.inner_iters(out_it) = i_hist.n_iters;
    hist.drho_norm(out_it)   = drho_norm;
    hist.outer_iters         = out_it;

    if verbose
        o1 = omega(n_target);
        o2 = omega(min(n_target+1, n_modes));
        fprintf(' %-4d  %-10.4f  %-10.4f  %-3d  %-6.4f  %-9.4e  %-10.4f  %-5d\n', ...
            out_it, o1, o2, N, mean(rho_new), drho_norm, sqrt(max(beta_fin,0)), i_hist.n_iters);
    end

    rho = rho_new;

    if drho_norm < outer_tol
        if verbose
            fprintf(' Converged at outer iter %d  (drho_norm = %.2e < %.2e)\n', ...
                out_it, drho_norm, outer_tol);
        end
        break
    end
end

%% ------------------------------------------------------------------
%  Trim history
% ------------------------------------------------------------------
ni = hist.outer_iters;
fns = fieldnames(hist);
for fi = 1:numel(fns)
    fn = fns{fi};
    v  = hist.(fn);
    if isnumeric(v) && size(v,1) == outer_max_iter
        hist.(fn) = v(1:ni, :);
    end
end

rho_final = rho;
end

%% ------------------------------------------------------------------
function cfg = set_defaults(cfg)
    function cfg = def(cfg, f, v)
        if ~isfield(cfg, f) || isempty(cfg.(f)), cfg.(f) = v; end
    end
    cfg = def(cfg, 'L',             8.0);
    cfg = def(cfg, 'H',             1.0);
    cfg = def(cfg, 'nelx',          40);
    cfg = def(cfg, 'nely',          5);
    cfg = def(cfg, 'E0',            1e7);
    cfg = def(cfg, 'nu',            0.3);
    cfg = def(cfg, 'rho0',          1.0);
    cfg = def(cfg, 't',             1.0);
    cfg = def(cfg, 'volfrac',       0.5);
    cfg = def(cfg, 'rho_min',       1e-3);
    cfg = def(cfg, 'penal',         3.0);
    cfg = def(cfg, 'mass_mode',     'du2007_c1');
    cfg = def(cfg, 'rmin_elem',     2.5);
    cfg = def(cfg, 'support_type',  'CC');
    cfg = def(cfg, 'n_target',      1);
    n_t = cfg.n_target;
    cfg = def(cfg, 'n_modes',       max(n_t + 3, 4));
    cfg = def(cfg, 'mult_tol',      1e-3);
    cfg = def(cfg, 'outer_max_iter', 300);
    cfg = def(cfg, 'outer_tol',     1e-3);
    cfg = def(cfg, 'inner_max_iter', 30);
    cfg = def(cfg, 'inner_tol',     1e-4);
    cfg = def(cfg, 'move_lim',      0.2);
    cfg = def(cfg, 'outer_move',    0.2);
    cfg = def(cfg, 'alpha',         0.5);
    cfg = def(cfg, 'verbose',       true);
end
