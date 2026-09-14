function [rho_final, hist] = topopt_freq_exact_persist_mma_experiment(cfg)
% TOPOPT_FREQ_EXACT_PERSIST_MMA_EXPERIMENT  Experiment-only MMA persistence test.
%
%   [rho_final, hist] = topopt_freq_exact(cfg)
%
%   Implements the nested increment formulation of Section 3.5 (Fig. 1):
%     Outer loop: assemble K,M; solve eigenproblem; detect multiplicity N;
%                 compute generalized gradients fsk; call inner loop.
%     Inner loop: MMA increment subproblem on [beta; Delta_rho] (inner_loop_mma).
%     Update:     rho := rho + Delta_rho  (paper Fig. 1 step 4, damped by alpha)
%     Stop:       norm(Delta_rho)/sqrt(nEl) < outer_tol.
%
%   Returns the FINAL CONVERGED rho, not a best-seen tracking.
%
%   cfg struct fields (all optional; defaults follow the 2007 benchmark
%   unless noted as numerical safeguards):
%     .L, .H               geometry (default 8, 1)
%     .nelx, .nely         mesh (default 40, 5)
%     .E0                  Young's modulus (default 1e7)
%     .nu                  Poisson ratio (default 0.3)
%     .rho0                mass density (default 1)
%     .t                   thickness (default 1)
%     .volfrac             volume fraction upper bound (default 0.5)
%     .rho_min             minimum density (default 1e-3)
%     .penal               SIMP penalization exponent (default 3)
%     .mass_mode           mass interpolation (default 'du2007_c1', paper Eq. 4b)
%     .rmin_elem           filter radius in element units (default 2.5)
%     .sensitivity_filter  apply sensitivity filter to fsk (default true)
%     .support_type        boundary conditions: 'SS','CS','CC' (default 'CC')
%     .n_target            target mode index (default 1)
%     .n_modes             modes to compute each outer iter (default n_target+3)
%     .mult_tol            cluster detection tolerance (default 1e-3)
%     .outer_max_iter      outer loop limit (default 300)
%     .outer_tol           convergence: norm(drho)/sqrt(nEl) < outer_tol (default 1e-3)
%     .inner_max_iter      inner MMA iteration limit (default 30)
%     .inner_tol           inner convergence tolerance (default 1e-4)
%     .move_lim            per-iter inner trust region (default Inf = disabled; not in paper)
%     .outer_move          outer trust region on |Delta_rho_e| (default Inf = disabled; not in paper)
%     .alpha               outer update: rho += alpha*drho (default 1.0 = full update, paper Fig.1 step 4)
%     .acceptance_check    backtrack updates that collapse omega_n (default false; not in paper)
%     .max_freq_drop       max accepted relative omega_n drop per outer update (default 0.01; inactive)
%     .min_alpha           minimum backtracking alpha before accepting a step (default 1e-3; inactive)
%     .verbose             print progress table (default true)
%
%   Outputs:
%     rho_final   nEl x 1   converged physical density
%     hist        struct with fields omega, beta, volume, N, inner_iters,
%                 drho_norm, outer_iters.  omega/N are pre-update diagnostics;
%                 omega_trial/N_trial and final_omega/final_N describe the
%                 accepted post-update designs.
%
%   Reference: Du & Olhoff (2007), Struct Multidisc Optim 34:91-110,
%              DOI 10.1007/s00158-007-0101-y, plus erratum
%              DOI 10.1007/s00158-007-0167-6.

if nargin < 1 || isempty(cfg), cfg = struct(); end
cfg = set_defaults(cfg);

L    = cfg.L;   H    = cfg.H;
nelx = cfg.nelx; nely = cfg.nely;
E0   = cfg.E0;  nu   = cfg.nu; rho0 = cfg.rho0; t = cfg.t;
volfrac  = cfg.volfrac;   rho_min  = cfg.rho_min;
penal    = cfg.penal;     mass_mode = cfg.mass_mode;
rmin_elem = cfg.rmin_elem;
sensitivity_filter = cfg.sensitivity_filter;
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
acceptance_check = cfg.acceptance_check;
max_freq_drop    = cfg.max_freq_drop;
min_alpha        = cfg.min_alpha;
verbose        = cfg.verbose;
asyinit        = cfg.asyinit;

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

if isfield(cfg, 'fixed_dofs') && ~isempty(cfg.fixed_dofs)
    fixed = unique(double(cfg.fixed_dofs(:)));
else
    fixed = build_supports_exact(support_type, nodeNrs);
end
free  = setdiff(1:nDof, fixed);
nFree = numel(free);

if sensitivity_filter
    [h_filt, Hs_filt] = build_filter(nelx, nely, rmin_elem);
end

%% ------------------------------------------------------------------
%  Initialise density and history
% ------------------------------------------------------------------
rho = volfrac * ones(nEl, 1);

hist.omega       = nan(outer_max_iter, n_modes);
hist.beta        = nan(outer_max_iter, 1);
hist.volume      = nan(outer_max_iter, 1);
hist.N           = nan(outer_max_iter, 1);
hist.N_trial     = nan(outer_max_iter, 1);
hist.inner_iters = nan(outer_max_iter, 1);
hist.drho_norm   = nan(outer_max_iter, 1);
hist.omega_trial = nan(outer_max_iter, n_modes);
hist.step_alpha  = nan(outer_max_iter, 1);
hist.max_abs_drho = nan(outer_max_iter, 1);
hist.asym_width_min = nan(outer_max_iter, 1);
hist.asym_width_mean = nan(outer_max_iter, 1);
hist.asym_width_max = nan(outer_max_iter, 1);
hist.asym_expand_count = nan(outer_max_iter, 1);
hist.asym_contract_count = nan(outer_max_iter, 1);
hist.asym_same_count = nan(outer_max_iter, 1);
hist.outer_iters = 0;
mma_state = struct();

if verbose
    fprintf('\n');
    fprintf(' %-4s  %-10s  %-10s  %-3s  %-6s  %-9s  %-10s  %-9s  %-8s  %-8s\n', ...
        'iter', 'omega_1', 'omega_2', ' N ', 'vol', 'max|drho|', ...
        'beta(rad/s)', 'asy_mean', 'expand', 'contract');
    fprintf(' %s\n', repmat('-', 1, 98));
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

    %% --- Generalized gradients (paper Fig. 1 step 2, Eq. 13) ---
    Phi_cluster = Phi(:, cluster_idx);
    fsk_raw = compute_generalized_gradients(rho, lambda_bar, Phi_cluster, ...
                  cMat, Ke_phys, Me_phys, penal, mass_mode);

    if sensitivity_filter
        fsk_use = zeros(size(fsk_raw));
        for s = 1:N
            for k = 1:N
                fsk_use(:,s,k) = apply_sensitivity_filter( ...
                    fsk_raw(:,s,k), rho, h_filt, Hs_filt, nely, nelx);
            end
        end
    else
        fsk_use = fsk_raw;
    end

    %% --- J-mode gradient ---
    if J_idx > 0
        lambda_J = lam(J_idx);
        dlam_J_raw = compute_elem_sensitivity(rho, lambda_J, Phi(:,J_idx), ...
            cMat, Ke_phys, Me_phys, free, nDof, penal, mass_mode);
        if sensitivity_filter
            dlam_J = apply_sensitivity_filter(dlam_J_raw, rho, h_filt, Hs_filt, nely, nelx);
        else
            dlam_J = dlam_J_raw;
        end
    else
        lambda_J = Inf;
        dlam_J   = [];
    end

    %% --- Experiment-only one-step MMA with cross-outer asymptote persistence ---
    [drho, beta_fin, i_hist, mma_state] = inner_loop_mma_persist_step_experiment( ...
        rho, lambda_bar, fsk_use, lambda_J, dlam_J, volfrac, rho_min, ...
        mma_state, asyinit);

    %% --- Update design variables (paper Fig. 1 step 4: rho := rho + Delta_rho) ---
    step_alpha = 1.0;
    rho_new    = max(rho_min, min(1, rho + drho));
    omega_trial = nan(n_modes, 1);

    if acceptance_check
        while true
            rho_trial = max(rho_min, min(1, rho + step_alpha * drho));
            [omega_trial, trial_flag] = eval_omega_only(rho_trial, Ke_phys_l, Me_phys_l, ...
                iK, jK, nDof, free, n_modes, opts_eig, penal, mass_mode);

            if trial_flag == 0
                rel_drop = (omega(n_target) - omega_trial(n_target)) / max(omega(n_target), eps);
                if rel_drop <= max_freq_drop || step_alpha <= min_alpha
                    rho_new = rho_trial;
                    break
                end
            elseif step_alpha <= min_alpha
                rho_new = rho;
                omega_trial = omega;
                step_alpha = 0;
                break
            end

            step_alpha = 0.5 * step_alpha;
        end
    else
        [omega_trial, ~] = eval_omega_only(rho_new, Ke_phys_l, Me_phys_l, ...
            iK, jK, nDof, free, n_modes, opts_eig, penal, mass_mode);
    end

    drho_norm = norm(rho_new - rho) / sqrt(nEl);
    if all(isfinite(omega_trial))
        [N_trial, ~, ~] = detect_multiplicity(omega_trial, n_target, mult_tol);
    else
        N_trial = NaN;
    end

    hist.omega(out_it, :)    = omega(:)';
    hist.beta(out_it)        = beta_fin;
    hist.volume(out_it)      = mean(rho_new);
    hist.N(out_it)           = N;
    hist.N_trial(out_it)     = N_trial;
    hist.inner_iters(out_it) = i_hist.n_iters;
    hist.drho_norm(out_it)   = drho_norm;
    hist.omega_trial(out_it, :) = omega_trial(:)';
    hist.step_alpha(out_it)  = step_alpha;
    hist.max_abs_drho(out_it) = max(abs(rho_new - rho));
    hist.asym_width_min(out_it) = i_hist.asym_width_min;
    hist.asym_width_mean(out_it) = i_hist.asym_width_mean;
    hist.asym_width_max(out_it) = i_hist.asym_width_max;
    hist.asym_expand_count(out_it) = i_hist.asym_expand_count;
    hist.asym_contract_count(out_it) = i_hist.asym_contract_count;
    hist.asym_same_count(out_it) = i_hist.asym_same_count;
    hist.outer_iters         = out_it;

    if verbose
        o1 = omega_trial(n_target);
        o2 = omega_trial(min(n_target+1, n_modes));
        if ~isfinite(o1), o1 = omega(n_target); end
        if ~isfinite(o2), o2 = omega(min(n_target+1, n_modes)); end
        if isfinite(N_trial), N_disp = N_trial; else, N_disp = N; end
        fprintf(' %-4d  %-10.4f  %-10.4f  %-3d  %-6.4f  %-9.3g  %-10.4f  %-9.3g  %-8d  %-8d\n', ...
            out_it, o1, o2, N_disp, mean(rho_new), hist.max_abs_drho(out_it), ...
            sqrt(max(beta_fin,0)), i_hist.asym_width_mean, ...
            i_hist.asym_expand_count, i_hist.asym_contract_count);
    end

    rho = rho_new;

    full_step_taken = (~acceptance_check) || step_alpha >= alpha * (1 - 1e-12);
    if drho_norm < outer_tol && full_step_taken
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
[final_omega, final_flag] = eval_omega_only(rho_final, Ke_phys_l, Me_phys_l, ...
    iK, jK, nDof, free, n_modes, opts_eig, penal, mass_mode);
hist.final_omega = final_omega(:)';
hist.final_flag  = final_flag;
if final_flag == 0
    [hist.final_N, hist.final_J_idx, hist.final_cluster_idx] = ...
        detect_multiplicity(final_omega, n_target, mult_tol);
else
    hist.final_N = NaN;
    hist.final_J_idx = NaN;
    hist.final_cluster_idx = [];
end
hist.final_volume = mean(rho_final);
end

%% ------------------------------------------------------------------
function cfg = set_defaults(cfg)
    function cfg = def(cfg, f, v)
        if ~isfield(cfg, f) || isempty(cfg.(f)), cfg.(f) = v; end
    end
    cfg = def(cfg, 'L',                  8.0);
    cfg = def(cfg, 'H',                  1.0);
    cfg = def(cfg, 'nelx',               40);
    cfg = def(cfg, 'nely',               5);
    cfg = def(cfg, 'E0',                 1e7);
    cfg = def(cfg, 'nu',                 0.3);
    cfg = def(cfg, 'rho0',               1.0);
    cfg = def(cfg, 't',                  1.0);
    cfg = def(cfg, 'volfrac',            0.5);
    cfg = def(cfg, 'rho_min',            1e-3);
    cfg = def(cfg, 'penal',              3.0);
    cfg = def(cfg, 'mass_mode',          'du2007_c1'); % paper Eq.4b: d=6 below rho=0.1
    cfg = def(cfg, 'rmin_elem',          2.5);
    cfg = def(cfg, 'sensitivity_filter', true);
    cfg = def(cfg, 'support_type',       'CC');
    cfg = def(cfg, 'n_target',           1);
    n_t = cfg.n_target;
    cfg = def(cfg, 'n_modes',            max(n_t + 3, 4));
    cfg = def(cfg, 'mult_tol',           1e-3);
    cfg = def(cfg, 'outer_max_iter',     300);
    cfg = def(cfg, 'outer_tol',          1e-3);
    cfg = def(cfg, 'inner_max_iter',     30);
    cfg = def(cfg, 'inner_tol',          1e-4);
    % PHASE A (paper-exact): the following non-paper safeguards are DISABLED by
    % default.  Inf disables the extra Delta_rho trust regions, leaving only the
    % box bounds implied by Eq. (25f): rho_min <= rho_e + Delta_rho_e <= 1.
    cfg = def(cfg, 'move_lim',           Inf);   % disabled: extra inner MMA move-limit (not in paper)
    cfg = def(cfg, 'outer_move',         Inf);   % disabled: extra outer Delta_rho trust region (not in paper)
    % Paper Fig.1 step 4 uses the full update rho := rho + Delta_rho (alpha = 1).
    cfg = def(cfg, 'alpha',              1.0);
    cfg = def(cfg, 'acceptance_check',   false); % disabled: backtracking/frequency-drop acceptance (not in paper)
    cfg = def(cfg, 'max_freq_drop',      0.01);  % inactive unless acceptance_check = true
    cfg = def(cfg, 'min_alpha',          1e-3);  % inactive unless acceptance_check = true
    cfg = def(cfg, 'verbose',            true);
    cfg = def(cfg, 'asyinit',            0.02);
end

%% ------------------------------------------------------------------
function [omega, flag] = eval_omega_only(rho, Ke_phys_l, Me_phys_l, ...
                                         iK, jK, nDof, free, n_modes, ...
                                         opts_eig, penal, mass_mode)
    [K, M] = assemble_KM_exact(rho, Ke_phys_l, Me_phys_l, iK, jK, nDof, penal, mass_mode);
    Kf = K(free, free);
    Mf = M(free, free);
    nFree = numel(free);

    [~, D, flag] = eigs(Kf, Mf, n_modes, 'SM', opts_eig);
    if flag ~= 0
        opts_r.tol   = 1e-8;
        opts_r.maxit = 1500;
        opts_r.p     = min(nFree-1, max(40, 4*n_modes));
        [~, D, flag] = eigs(Kf, Mf, n_modes, 'SM', opts_r);
    end

    if flag == 0
        lam = sort(real(diag(D)));
        omega = sqrt(max(lam, 0));
    else
        omega = nan(n_modes, 1);
    end
end
