function [rho_final, hist] = topopt_freq_exact(cfg)
% TOPOPT_FREQ_EXACT  Du & Olhoff (2007) frequency maximization.
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
%     .mass_matrix         element mass matrix: 'consistent' or 'lumped'
%                           (default 'consistent')
%     .rmin_elem           filter radius in element units (default 2.5)
%     .sensitivity_filter  apply sensitivity filter to fsk (default true)
%     .filter_type         diagnostic filter path: '', 'none',
%                           'sensitivity', 'density', or
%                           'density_projection'.  Empty preserves
%                           .sensitivity_filter behavior.
%     .projection_beta     Heaviside beta for density_projection
%     .projection_eta      Heaviside threshold for density_projection
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
%     .rho_snapshot_interval
%                           diagnostic-only: store rho every N iterations
%                           (default 0 = disabled; no effect on optimizer)
%     .forensic_enabled     diagnostic-only: after the first post-update
%                           omega(1:2) relative gap falls below
%                           .forensic_gap_tol, record every following outer
%                           update without changing the accepted step
%                           (default false)
%     .forensic_gap_tol     relative omega(1:2) gap trigger (default 0.005)
%     .forensic_active_tol  active-constraint residual tolerance (default 1e-4)
%     .globalization_enabled
%                           diagnostic/experimental outer-update globalization
%                           switch (default false = original update path)
%     .globalization_monotone_cluster
%                           backtrack until the real post-update value
%                           min(lambda(n:n+N-1)) does not decrease
%     .globalization_low_mode_guard
%                           backtrack if a new lower mode has low MAC with
%                           the pre-update target cluster subspace
%     .globalization_alpha_start
%                           first backtracking alpha (default 1.0)
%     .globalization_alpha_min
%                           reject the outer step if no alpha down to this
%                           value is acceptable (default 1/128)
%     .globalization_low_mode_mac_threshold
%                           subspace MAC threshold for low-mode guard
%                           (default 0.25)
%     .post_coalescence_trust_enabled
%                           reduce move_lim and outer_move when omega1/omega2
%                           are near-coalesced (default false)
%     .post_coalescence_trust_factor
%                           move-limit multiplier after coalescence trigger
%                           (default 1.0)
%     .density_symmetry     diagnostic-only density projection: 'none',
%                           'midspan', 'midheight', or 'both'
%                           (default 'none')
%     .forced_solid_mask    diagnostic-only passive solid element mask/indices
%                           applied after each update (default empty)
%     .forced_solid_value   diagnostic-only density assigned to forced mask
%                           (default 1)
%     .forced_solid_preserve_volume
%                           diagnostic-only non-passive density rescaling to
%                           keep mean density at volfrac after forcing
%                           (default false)
%     .initial_rho          optional nEl x 1 starting density.  If omitted,
%                           starts from uniform volfrac as in the paper.
%
%   Outputs:
%     rho_final   nEl x 1   converged physical density
%     hist        struct with fields omega, lambda, beta, volume, N, inner_iters,
%                 drho_norm, outer_iters.  omega/N are pre-update diagnostics;
%                 omega_trial/lambda_trial/N_trial and final_omega/final_N
%                 describe the accepted post-update designs.
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
mass_matrix = lower(strtrim(cfg.mass_matrix));
rmin_elem = cfg.rmin_elem;
sensitivity_filter = cfg.sensitivity_filter;
filter_type = lower(strtrim(cfg.filter_type));
if isempty(filter_type)
    if sensitivity_filter
        filter_type = 'sensitivity';
    else
        filter_type = 'none';
    end
end
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
rho_snapshot_interval = cfg.rho_snapshot_interval;
persistent_mma_state = cfg.persistent_mma_state;
forensic_enabled = cfg.forensic_enabled;
globalization_enabled = cfg.globalization_enabled;
density_symmetry = cfg.density_symmetry;

%% ------------------------------------------------------------------
%  Mesh and FE setup
% ------------------------------------------------------------------
dx = L/nelx;  dy = H/nely;
nEl  = nelx * nely;
nDof = 2 * (nelx+1) * (nely+1);

[Ke_star, Me_star] = fe_q4_exact(nu, t, dx, dy);
Ke_phys = E0   * Ke_star;
Me_phys = rho0 * Me_star;
if strcmp(mass_matrix, 'lumped')
    Me_phys = diag(sum(Me_phys, 2));
elseif ~strcmp(mass_matrix, 'consistent')
    error('topopt_freq_exact:InvalidMassMatrix', ...
        'mass_matrix must be ''consistent'' or ''lumped''.');
end

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

h_filt = [];
Hs_filt = [];
uses_filter_kernel = any(strcmp(filter_type, {'sensitivity','density','density_projection'}));
if uses_filter_kernel
    [h_filt, Hs_filt] = build_filter(nelx, nely, rmin_elem);
end

%% ------------------------------------------------------------------
%  Initialise density and history
% ------------------------------------------------------------------
if isfield(cfg, 'initial_rho') && ~isempty(cfg.initial_rho)
    rho = cfg.initial_rho(:);
    if numel(rho) ~= nEl
        error('topopt_freq_exact:InvalidInitialRho', ...
            'initial_rho has %d entries, expected nelx*nely = %d.', numel(rho), nEl);
    end
    rho = max(rho_min, min(1, rho));
else
    rho = volfrac * ones(nEl, 1);
end
rho = apply_density_symmetry(rho, nely, nelx, density_symmetry);
rho = apply_forced_solid(rho, cfg, volfrac);

hist.omega       = nan(outer_max_iter, n_modes);
hist.lambda      = nan(outer_max_iter, n_modes);
hist.beta        = nan(outer_max_iter, 1);
hist.volume      = nan(outer_max_iter, 1);
hist.N           = nan(outer_max_iter, 1);
hist.N_trial     = nan(outer_max_iter, 1);
hist.inner_iters = nan(outer_max_iter, 1);
hist.inner_cpu_time = nan(outer_max_iter, 1);
hist.inner_converged = false(outer_max_iter, 1);
hist.inner_hit_max_iter = false(outer_max_iter, 1);
hist.inner_termination_reason = cell(outer_max_iter, 1);
hist.mma_reused_previous_state = false(outer_max_iter, 1);
hist.asym_width_min = nan(outer_max_iter, 1);
hist.asym_width_mean = nan(outer_max_iter, 1);
hist.asym_width_max = nan(outer_max_iter, 1);
hist.asym_expand_count = nan(outer_max_iter, 1);
hist.asym_contract_count = nan(outer_max_iter, 1);
hist.asym_same_count = nan(outer_max_iter, 1);
hist.drho_norm   = nan(outer_max_iter, 1);
hist.drho_max    = nan(outer_max_iter, 1);
hist.omega_trial = nan(outer_max_iter, n_modes);
hist.lambda_trial = nan(outer_max_iter, n_modes);
hist.step_alpha  = nan(outer_max_iter, 1);
hist.globalization_rejected_outer_step = false(outer_max_iter, 1);
hist.globalization_rejected_trial_count = nan(outer_max_iter, 1);
hist.globalization_trial_count = nan(outer_max_iter, 1);
hist.globalization_reason = cell(outer_max_iter, 1);
hist.globalization_low_mode_mac_min = nan(outer_max_iter, 1);
hist.move_lim_effective = nan(outer_max_iter, 1);
hist.outer_move_effective = nan(outer_max_iter, 1);
hist.outer_iters = 0;
hist.rho_initial = rho;
if rho_snapshot_interval > 0
    nSnapMax = ceil(outer_max_iter / rho_snapshot_interval);
    hist.rho_snapshot_iters = nan(nSnapMax, 1);
    hist.rho_snapshots = nan(nEl, nSnapMax);
    hist.rho_snapshot_count = 0;
end
hist.forensic = init_forensic_history(cfg, nEl, n_modes);

if verbose
    fprintf('\n');
    fprintf(' %-4s  %-10s  %-10s  %-3s  %-6s  %-9s  %-10s  %-6s  %-5s\n', ...
        'iter', 'omega_1', 'omega_2', ' N ', 'vol', 'drho/sqrt', 'beta(rad/s)', 'alpha', 'in');
    fprintf(' %s\n', repmat('-', 1, 82));
end

opts_eig.tol   = 1e-10;
opts_eig.maxit = 600;
mma_state = [];

%% ------------------------------------------------------------------
%  Outer loop
% ------------------------------------------------------------------
for out_it = 1:outer_max_iter
    forensic_record_this_iter = forensic_enabled && hist.forensic.active && ...
        out_it > hist.forensic.trigger_iter;

    [rho_phys, density_map] = design_to_physical_density( ...
        rho, filter_type, h_filt, Hs_filt, nely, nelx, cfg);

    %% --- Assemble K, M ---
    [K, M] = assemble_KM_exact(rho_phys, Ke_phys_l, Me_phys_l, iK, jK, nDof, penal, mass_mode);
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

    move_lim_eff = move_lim;
    outer_move_eff = outer_move;
    if cfg.post_coalescence_trust_enabled && numel(omega) >= n_target + 1
        coalescence_gap = abs(omega(n_target+1) - omega(n_target)) / ...
            max(omega(n_target), eps);
        if coalescence_gap < cfg.post_coalescence_gap_tol
            if isfinite(move_lim_eff), move_lim_eff = cfg.post_coalescence_trust_factor * move_lim_eff; end
            if isfinite(outer_move_eff), outer_move_eff = cfg.post_coalescence_trust_factor * outer_move_eff; end
        end
    end

    %% --- Generalized gradients (paper Fig. 1 step 2, Eq. 13) ---
    Phi_cluster = Phi(:, cluster_idx);
    fsk_raw = compute_generalized_gradients(rho_phys, lambda_bar, Phi_cluster, ...
                  cMat, Ke_phys, Me_phys, penal, mass_mode);

    if strcmp(filter_type, 'sensitivity')
        fsk_use = zeros(size(fsk_raw));
        for s = 1:N
            for k = 1:N
                fsk_use(:,s,k) = apply_sensitivity_filter( ...
                    fsk_raw(:,s,k), rho_phys, h_filt, Hs_filt, nely, nelx);
            end
        end
    elseif any(strcmp(filter_type, {'density','density_projection'}))
        fsk_use = zeros(size(fsk_raw));
        for s = 1:N
            for k = 1:N
                fsk_use(:,s,k) = physical_to_design_sensitivity( ...
                    fsk_raw(:,s,k), density_map, h_filt, Hs_filt, nely, nelx);
            end
        end
    else
        fsk_use = fsk_raw;
    end

    %% --- J-mode gradient ---
    if J_idx > 0
        lambda_J = lam(J_idx);
        dlam_J_raw = compute_elem_sensitivity(rho_phys, lambda_J, Phi(:,J_idx), ...
            cMat, Ke_phys, Me_phys, free, nDof, penal, mass_mode);
        if strcmp(filter_type, 'sensitivity')
            dlam_J = apply_sensitivity_filter(dlam_J_raw, rho_phys, h_filt, Hs_filt, nely, nelx);
        elseif any(strcmp(filter_type, {'density','density_projection'}))
            dlam_J = physical_to_design_sensitivity( ...
                dlam_J_raw, density_map, h_filt, Hs_filt, nely, nelx);
        else
            dlam_J = dlam_J_raw;
        end
    else
        lambda_J = Inf;
        dlam_J   = [];
    end

    %% --- Inner loop (MMA increment subproblem, paper Eq. 19) ---
    inner_tic = tic;
    reused_mma_state = persistent_mma_state && ~isempty(mma_state) && ...
        isfield(mma_state, 'initialized') && mma_state.initialized;
    if persistent_mma_state
        [drho, beta_fin, i_hist, mma_state] = inner_loop_mma_persistent_phase2( ...
            rho, lambda_bar, fsk_use, lambda_J, dlam_J, volfrac, rho_min, ...
            inner_max_iter, inner_tol, move_lim_eff, outer_move_eff, mma_state);
    else
        [drho, beta_fin, i_hist] = inner_loop_mma(rho, lambda_bar, fsk_use, ...
            lambda_J, dlam_J, volfrac, rho_min, inner_max_iter, inner_tol, ...
            move_lim_eff, outer_move_eff);
    end
    inner_cpu_time = toc(inner_tic);

    %% --- Update design variables (paper Fig. 1 step 4: rho := rho + Delta_rho) ---
    step_alpha = alpha;
    rho_new    = max(rho_min, min(1, rho + step_alpha * drho));
    rho_new = apply_forced_solid(rho_new, cfg, volfrac);
    omega_trial = nan(n_modes, 1);
    lam_trial = nan(n_modes, 1);
    glob_info = default_globalization_info();

    if globalization_enabled
        [rho_new, omega_trial, lam_trial, step_alpha, glob_info] = globalized_outer_update( ...
            rho, drho, lam, Phi_cluster, cluster_idx, Ke_phys_l, Me_phys_l, ...
            iK, jK, nDof, free, n_modes, opts_eig, penal, mass_mode, rho_min, cfg, ...
            filter_type, h_filt, Hs_filt, nely, nelx, volfrac);
    elseif acceptance_check
        while true
            rho_trial = max(rho_min, min(1, rho + step_alpha * drho));
            rho_trial = apply_forced_solid(rho_trial, cfg, volfrac);
            rho_trial_phys = design_to_physical_density(rho_trial, filter_type, h_filt, Hs_filt, nely, nelx, cfg);
            [omega_trial, trial_flag] = eval_omega_only(rho_trial_phys, Ke_phys_l, Me_phys_l, ...
                iK, jK, nDof, free, n_modes, opts_eig, penal, mass_mode);

            if trial_flag == 0
                lam_trial = omega_trial(:).^2;
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
        rho_new_phys = design_to_physical_density(rho_new, filter_type, h_filt, Hs_filt, nely, nelx, cfg);
        [omega_trial, ~] = eval_omega_only(rho_new_phys, Ke_phys_l, Me_phys_l, ...
            iK, jK, nDof, free, n_modes, opts_eig, penal, mass_mode);
        lam_trial = omega_trial(:).^2;
    end
    if all(isnan(lam_trial)) && all(isfinite(omega_trial))
        lam_trial = omega_trial(:).^2;
    end
    if ~strcmpi(density_symmetry, 'none')
        rho_new = apply_density_symmetry(rho_new, nely, nelx, density_symmetry);
        rho_new = apply_forced_solid(rho_new, cfg, volfrac);
        rho_new_phys = design_to_physical_density(rho_new, filter_type, h_filt, Hs_filt, nely, nelx, cfg);
        [omega_trial, lam_trial, ~, ~, trial_flag_sym] = eval_modes_exact( ...
            rho_new_phys, Ke_phys_l, Me_phys_l, iK, jK, nDof, free, ...
            n_modes, opts_eig, penal, mass_mode);
        if trial_flag_sym ~= 0
            omega_trial = nan(n_modes, 1);
            lam_trial = nan(n_modes, 1);
        end
    end

    drho_norm = norm(rho_new - rho) / sqrt(nEl);
    drho_max  = max(abs(rho_new - rho));
    if all(isfinite(omega_trial))
        [N_trial, ~, ~] = detect_multiplicity(omega_trial, n_target, mult_tol);
    else
        N_trial = NaN;
    end

    prediction = [];
    if forensic_enabled
        prediction = forensic_linear_prediction(beta_fin, lambda_bar, fsk_use, ...
            lambda_J, dlam_J, drho, volfrac, rho);
    end

    if forensic_record_this_iter
        rho_proposed = max(rho_min, min(1, rho + drho));
        rho_proposed = apply_forced_solid(rho_proposed, cfg, volfrac);
        rho_proposed_phys = design_to_physical_density(rho_proposed, filter_type, h_filt, Hs_filt, nely, nelx, cfg);
        [omega_prop, lam_prop, Phi_prop, M_prop, prop_flag] = eval_modes_exact( ...
            rho_proposed_phys, Ke_phys_l, Me_phys_l, iK, jK, nDof, free, ...
            n_modes, opts_eig, penal, mass_mode);
        if prop_flag ~= 0
            omega_prop = nan(n_modes, 1);
            lam_prop = nan(n_modes, 1);
            Phi_prop = nan(nDof, n_modes);
        end
        mac_pre_mass = forensic_mac(Phi, Phi_prop, M);
        mac_post_mass = forensic_mac(Phi, Phi_prop, M_prop);
        hist.forensic = append_forensic_record(hist.forensic, out_it, rho, ...
            drho, rho_new - rho, beta_fin, lambda_bar, lam, omega, prediction, ...
            lam_prop, omega_prop, omega_trial(:).^2, omega_trial, N, N_trial, ...
            mean(rho), mean(rho_proposed), mean(rho_new), mac_pre_mass, ...
            mac_post_mass, i_hist, step_alpha);
    end

    hist.omega(out_it, :)    = omega(:)';
    hist.lambda(out_it, :)   = lam(:)';
    hist.beta(out_it)        = beta_fin;
    rho_new_phys_hist = design_to_physical_density(rho_new, filter_type, h_filt, Hs_filt, nely, nelx, cfg);
    hist.volume(out_it)      = mean(rho_new_phys_hist);
    hist.N(out_it)           = N;
    hist.N_trial(out_it)     = N_trial;
    hist.inner_iters(out_it) = i_hist.n_iters;
    hist.inner_cpu_time(out_it) = inner_cpu_time;
    hist.mma_reused_previous_state(out_it) = reused_mma_state;
    hist.inner_converged(out_it) = isfield(i_hist, 'converged') && i_hist.converged;
    hist.inner_hit_max_iter(out_it) = isfield(i_hist, 'hit_max_iter') && i_hist.hit_max_iter;
    if isfield(i_hist, 'termination_reason')
        hist.inner_termination_reason{out_it} = i_hist.termination_reason;
    else
        hist.inner_termination_reason{out_it} = 'unknown';
    end
    hist.drho_norm(out_it)   = drho_norm;
    hist.drho_max(out_it)    = drho_max;
    if isfield(i_hist, 'asym_width_min')
        hist.asym_width_min(out_it) = i_hist.asym_width_min;
        hist.asym_width_mean(out_it) = i_hist.asym_width_mean;
        hist.asym_width_max(out_it) = i_hist.asym_width_max;
        hist.asym_expand_count(out_it) = i_hist.asym_expand_count;
        hist.asym_contract_count(out_it) = i_hist.asym_contract_count;
        hist.asym_same_count(out_it) = i_hist.asym_same_count;
    end
    hist.omega_trial(out_it, :) = omega_trial(:)';
    hist.lambda_trial(out_it, :) = lam_trial(:)';
    hist.step_alpha(out_it)  = step_alpha;
    hist.globalization_rejected_outer_step(out_it) = glob_info.rejected_outer_step;
    hist.globalization_rejected_trial_count(out_it) = glob_info.rejected_trial_count;
    hist.globalization_trial_count(out_it) = glob_info.trial_count;
    hist.globalization_reason{out_it} = glob_info.reason;
    hist.globalization_low_mode_mac_min(out_it) = glob_info.low_mode_mac_min;
    hist.move_lim_effective(out_it) = move_lim_eff;
    hist.outer_move_effective(out_it) = outer_move_eff;
    hist.outer_iters         = out_it;
    if rho_snapshot_interval > 0 && (mod(out_it, rho_snapshot_interval) == 0 || out_it == 1)
        hist.rho_snapshot_count = hist.rho_snapshot_count + 1;
        si = hist.rho_snapshot_count;
        hist.rho_snapshot_iters(si) = out_it;
        hist.rho_snapshots(:, si) = rho_new;
    end

    if forensic_enabled && ~hist.forensic.active && all(isfinite(omega_trial)) && ...
            numel(omega_trial) >= n_target + 1
        forensic_gap = abs(omega_trial(n_target+1) - omega_trial(n_target)) / ...
            max(omega_trial(n_target), eps);
        if forensic_gap < cfg.forensic_gap_tol
            hist.forensic.active = true;
            hist.forensic.trigger_iter = out_it;
            hist.forensic.trigger_gap = forensic_gap;
            hist.forensic.trigger_omega = omega_trial(:)';
            hist.forensic.trigger_lambda = omega_trial(:)'.^2;
            hist.forensic.trigger_volume = mean(rho_new);
            hist.forensic.trigger_rho = rho_new;
        end
    end

    if verbose
        o1 = omega_trial(n_target);
        o2 = omega_trial(min(n_target+1, n_modes));
        if ~isfinite(o1), o1 = omega(n_target); end
        if ~isfinite(o2), o2 = omega(min(n_target+1, n_modes)); end
        if isfinite(N_trial), N_disp = N_trial; else, N_disp = N; end
        fprintf(' %-4d  %-10.4f  %-10.4f  %-3d  %-6.4f  %-9.4e  %-10.4f  %-6.3g  %-5d\n', ...
            out_it, o1, o2, N_disp, mean(rho_new), drho_norm, sqrt(max(beta_fin,0)), ...
            step_alpha, i_hist.n_iters);
    end

    rho = rho_new;

    if globalization_enabled
        full_step_taken = ~glob_info.rejected_outer_step && step_alpha >= ...
            cfg.globalization_alpha_start * (1 - 1e-12);
    else
        full_step_taken = (~acceptance_check) || step_alpha >= alpha * (1 - 1e-12);
    end
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
    elseif islogical(v) && size(v,1) == outer_max_iter
        hist.(fn) = v(1:ni, :);
    elseif iscell(v) && size(v,1) == outer_max_iter
        hist.(fn) = v(1:ni, :);
    end
end

hist.rho_design_final = rho;
rho_final = design_to_physical_density(rho, filter_type, h_filt, Hs_filt, nely, nelx, cfg);
hist.rho_final = rho_final;
[final_omega, final_flag] = eval_omega_only(rho_final, Ke_phys_l, Me_phys_l, ...
    iK, jK, nDof, free, n_modes, opts_eig, penal, mass_mode);
hist.final_omega = final_omega(:)';
hist.final_lambda = final_omega(:)'.^2;
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
    cfg = def(cfg, 'mass_matrix',        'consistent');
    cfg = def(cfg, 'rmin_elem',          2.5);
    cfg = def(cfg, 'sensitivity_filter', true);
    cfg = def(cfg, 'filter_type',        '');
    cfg = def(cfg, 'projection_beta',    1.0);
    cfg = def(cfg, 'projection_eta',     0.5);
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
    cfg = def(cfg, 'rho_snapshot_interval', 0);
    cfg = def(cfg, 'persistent_mma_state', false);
    cfg = def(cfg, 'forensic_enabled',   false);
    cfg = def(cfg, 'forensic_gap_tol',   0.005);
    cfg = def(cfg, 'forensic_active_tol', 1e-4);
    cfg = def(cfg, 'globalization_enabled', false);
    cfg = def(cfg, 'globalization_monotone_cluster', false);
    cfg = def(cfg, 'globalization_low_mode_guard', false);
    cfg = def(cfg, 'globalization_alpha_start', 1.0);
    cfg = def(cfg, 'globalization_alpha_min', 1/128);
    cfg = def(cfg, 'globalization_value_tol', 1e-10);
    cfg = def(cfg, 'globalization_low_mode_mac_threshold', 0.25);
    cfg = def(cfg, 'globalization_low_mode_lambda_tol', 1e-10);
    cfg = def(cfg, 'post_coalescence_trust_enabled', false);
    cfg = def(cfg, 'post_coalescence_gap_tol', 0.005);
    cfg = def(cfg, 'post_coalescence_trust_factor', 1.0);
    cfg = def(cfg, 'density_symmetry', 'none');
    cfg = def(cfg, 'forced_solid_mask', []);
    cfg = def(cfg, 'forced_solid_value', 1.0);
    cfg = def(cfg, 'forced_solid_preserve_volume', false);
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

%% ------------------------------------------------------------------
function rho = apply_density_symmetry(rho, nely, nelx, mode)
    if nargin < 4 || isempty(mode) || strcmpi(mode, 'none')
        rho = rho(:);
        return
    end
    R = reshape(rho(:), nely, nelx);
    switch lower(mode)
        case 'midspan'
            R = 0.5 * (R + fliplr(R));
        case 'midheight'
            R = 0.5 * (R + flipud(R));
        case 'both'
            R = 0.25 * (R + fliplr(R) + flipud(R) + flipud(fliplr(R)));
        otherwise
            error('topopt_freq_exact:InvalidDensitySymmetry', ...
                'Unknown density_symmetry mode: %s', mode);
    end
    rho = R(:);
end

%% ------------------------------------------------------------------
function [rho_phys, map] = design_to_physical_density(rho, filter_type, h, Hs, nely, nelx, cfg)
    rho = rho(:);
    map = struct('type', filter_type, 'dproj', []);
    switch lower(strtrim(filter_type))
        case {'', 'none', 'sensitivity'}
            rho_phys = rho;
        case 'density'
            rho_phys = apply_density_filter(rho, h, Hs, nely, nelx);
        case 'density_projection'
            rho_bar = apply_density_filter(rho, h, Hs, nely, nelx);
            beta = cfg.projection_beta;
            eta = cfg.projection_eta;
            denom = tanh(beta * eta) + tanh(beta * (1 - eta));
            t = tanh(beta * (rho_bar - eta));
            rho_phys = (tanh(beta * eta) + t) ./ denom;
            map.dproj = beta * (1 - t.^2) ./ denom;
        otherwise
            error('topopt_freq_exact:InvalidFilterType', ...
                'filter_type must be none, sensitivity, density, or density_projection.');
    end
    rho_phys = max(cfg.rho_min, min(1, rho_phys(:)));
end

%% ------------------------------------------------------------------
function rho_f = apply_density_filter(rho, h, Hs, nely, nelx)
    rho_f = conv2(reshape(rho, nely, nelx), h, 'same') ./ Hs;
    rho_f = rho_f(:);
end

%% ------------------------------------------------------------------
function s_design = physical_to_design_sensitivity(s_phys, map, h, Hs, nely, nelx)
    q = s_phys(:);
    if strcmp(map.type, 'density_projection')
        q = q .* map.dproj(:);
    end
    q_mat = reshape(q ./ Hs(:), nely, nelx);
    s_design = conv2(q_mat, h, 'same');
    s_design = s_design(:);
end

%% ------------------------------------------------------------------
function rho = apply_forced_solid(rho, cfg, volfrac)
    rho = rho(:);
    if ~isfield(cfg, 'forced_solid_mask') || isempty(cfg.forced_solid_mask)
        return
    end
    mask = false(size(rho));
    fs = cfg.forced_solid_mask;
    if islogical(fs)
        mask = fs(:);
    else
        mask(fs(:)) = true;
    end
    forced_value = max(cfg.rho_min, min(1, cfg.forced_solid_value));
    rho(mask) = forced_value;
    if isfield(cfg, 'forced_solid_preserve_volume') && cfg.forced_solid_preserve_volume
        free = ~mask;
        target_sum = volfrac * numel(rho) - forced_value * nnz(mask);
        if target_sum > cfg.rho_min * nnz(free) && any(free)
            vals = max(cfg.rho_min, min(1, rho(free)));
            for it = 1:20
                diff_sum = sum(vals) - target_sum;
                if abs(diff_sum) <= 1e-10 * numel(rho), break; end
                movable = vals > cfg.rho_min + 1e-12 | diff_sum < 0;
                movable = movable & (vals < 1 - 1e-12 | diff_sum > 0);
                if ~any(movable), break; end
                vals(movable) = vals(movable) - diff_sum / nnz(movable);
                vals = max(cfg.rho_min, min(1, vals));
            end
            rho(free) = vals;
        end
    end
end

%% ------------------------------------------------------------------
function info = default_globalization_info()
    info = struct();
    info.rejected_outer_step = false;
    info.rejected_trial_count = 0;
    info.trial_count = 0;
    info.reason = 'not_enabled';
    info.low_mode_mac_min = NaN;
end

%% ------------------------------------------------------------------
function [rho_new, omega_trial, lam_trial, accepted_alpha, info] = globalized_outer_update( ...
        rho, drho, lam_pre, Phi_cluster, cluster_idx, Ke_phys_l, Me_phys_l, ...
        iK, jK, nDof, free, n_modes, opts_eig, penal, mass_mode, rho_min, cfg, ...
        filter_type, h_filt, Hs_filt, nely, nelx, volfrac)

    info = default_globalization_info();
    info.reason = 'accepted';

    alpha_try = cfg.globalization_alpha_start;
    protected_pre = min(lam_pre(cluster_idx));
    protected_idx = cluster_idx(:)';
    value_tol = cfg.globalization_value_tol * max(abs(protected_pre), 1);
    low_mode_tol = cfg.globalization_low_mode_lambda_tol * max(abs(protected_pre), 1);

    last_omega = nan(n_modes, 1);
    last_lam = nan(n_modes, 1);
    last_reason = 'not_evaluated';
    last_low_mode_mac_min = NaN;

    while alpha_try >= cfg.globalization_alpha_min * (1 - 1e-12)
        info.trial_count = info.trial_count + 1;
        rho_trial = max(rho_min, min(1, rho + alpha_try * drho));
        rho_trial = apply_forced_solid(rho_trial, cfg, volfrac);
        rho_trial_phys = design_to_physical_density(rho_trial, filter_type, h_filt, Hs_filt, nely, nelx, cfg);
        [omega_cand, lam_cand, Phi_cand, M_cand, trial_flag] = eval_modes_exact( ...
            rho_trial_phys, Ke_phys_l, Me_phys_l, iK, jK, nDof, free, ...
            n_modes, opts_eig, penal, mass_mode);
        last_omega = omega_cand;
        last_lam = lam_cand;

        accept = trial_flag == 0;
        reasons = {};
        low_mode_mac_min = NaN;

        if trial_flag ~= 0
            accept = false;
            reasons{end+1} = 'eigs_failed';
        end

        if accept && cfg.globalization_monotone_cluster
            if max(protected_idx) > numel(lam_cand)
                accept = false;
                reasons{end+1} = 'missing_protected_modes';
            else
                protected_cand = min(lam_cand(protected_idx));
                if protected_cand < protected_pre - value_tol
                    accept = false;
                    reasons{end+1} = 'monotone_cluster_decrease';
                end
            end
        end

        if trial_flag == 0 && cfg.globalization_low_mode_guard
            [violates_low_mode, low_mode_mac_min] = low_mode_guard_violation( ...
                lam_cand, Phi_cand, M_cand, protected_pre, Phi_cluster, ...
                cfg.globalization_low_mode_mac_threshold, low_mode_tol);
            if violates_low_mode
                accept = false;
                reasons{end+1} = 'low_mode_guard';
            end
        end

        if accept
            rho_new = rho_trial;
            omega_trial = omega_cand;
            lam_trial = lam_cand;
            accepted_alpha = alpha_try;
            info.low_mode_mac_min = low_mode_mac_min;
            if isempty(reasons)
                info.reason = 'accepted';
            else
                info.reason = strjoin(reasons, '+');
            end
            return
        end

        info.rejected_trial_count = info.rejected_trial_count + 1;
        if isempty(reasons)
            last_reason = 'rejected';
        else
            last_reason = strjoin(reasons, '+');
        end
        last_low_mode_mac_min = low_mode_mac_min;
        alpha_try = 0.5 * alpha_try;
    end

    rho_new = rho;
    omega_trial = sqrt(max(lam_pre(:), 0));
    lam_trial = lam_pre(:);
    accepted_alpha = 0;
    info.rejected_outer_step = true;
    info.reason = ['rejected_all_alpha:' last_reason];
    info.low_mode_mac_min = last_low_mode_mac_min;
    if all(isfinite(last_omega))
        info.last_rejected_omega = last_omega;
        info.last_rejected_lambda = last_lam;
    end
end

%% ------------------------------------------------------------------
function [violates, low_mode_mac_min] = low_mode_guard_violation( ...
        lam_trial, Phi_trial, M_trial, protected_pre, Phi_cluster, ...
        mac_threshold, lambda_tol)

    low_idx = find(lam_trial(:) < protected_pre - lambda_tol);
    if isempty(low_idx)
        violates = false;
        low_mode_mac_min = NaN;
        return
    end

    mac = forensic_mac(Phi_cluster, Phi_trial(:, low_idx), M_trial);
    subspace_mac = max(mac, [], 1);
    low_mode_mac_min = min(subspace_mac);
    violates = any(subspace_mac < mac_threshold);
end

%% ------------------------------------------------------------------
function forensic = init_forensic_history(cfg, nEl, n_modes)
    forensic = struct();
    forensic.enabled = cfg.forensic_enabled;
    forensic.gap_tol = cfg.forensic_gap_tol;
    forensic.active_tol = cfg.forensic_active_tol;
    forensic.active = false;
    forensic.trigger_iter = NaN;
    forensic.trigger_gap = NaN;
    forensic.trigger_omega = nan(1, n_modes);
    forensic.trigger_lambda = nan(1, n_modes);
    forensic.trigger_volume = NaN;
    forensic.trigger_rho = [];
    forensic.count = 0;
    forensic.iter = [];
    forensic.rho_pre = zeros(nEl, 0);
    forensic.drho_proposed = zeros(nEl, 0);
    forensic.drho_accepted = zeros(nEl, 0);
    forensic.beta = [];
    forensic.beta_omega = [];
    forensic.lambda_bar = [];
    forensic.lambda_pre = zeros(0, n_modes);
    forensic.omega_pre = zeros(0, n_modes);
    forensic.lambda_after_proposed = zeros(0, n_modes);
    forensic.omega_after_proposed = zeros(0, n_modes);
    forensic.lambda_after_accepted = zeros(0, n_modes);
    forensic.omega_after_accepted = zeros(0, n_modes);
    forensic.N_pre = [];
    forensic.N_after = [];
    forensic.volume_pre = [];
    forensic.volume_after_proposed = [];
    forensic.volume_after_accepted = [];
    forensic.predicted_cluster_lambda = {};
    forensic.predicted_cluster_dlambda = {};
    forensic.predicted_J_lambda = [];
    forensic.predicted_J_dlambda = [];
    forensic.constraint_names = {};
    forensic.constraint_values = {};
    forensic.active_constraints = {};
    forensic.mac_pre_mass = {};
    forensic.mac_post_mass = {};
    forensic.predicted_improvement = [];
    forensic.real_decrease = [];
    forensic.predicted_improvement_real_decrease = [];
    forensic.inner_iters = [];
    forensic.inner_converged = [];
    forensic.inner_hit_max_iter = [];
    forensic.step_alpha = [];
end

%% ------------------------------------------------------------------
function prediction = forensic_linear_prediction(beta, lambda_bar, fsk, ...
                                                 lambda_J, dlam_J, drho, ...
                                                 volfrac, rho)
    N = size(fsk, 2);
    fsk2D = reshape(fsk, numel(rho), N*N);
    F_mat = reshape(fsk2D' * drho(:), N, N);
    mu = sort(real(eig(F_mat)), 'ascend');

    prediction = struct();
    prediction.cluster_dlambda = mu(:)';
    prediction.cluster_lambda = lambda_bar + mu(:)';
    prediction.has_J = isfinite(lambda_J) && ~isempty(dlam_J);
    if prediction.has_J
        prediction.J_dlambda = dlam_J(:)' * drho(:);
        prediction.J_lambda = lambda_J + prediction.J_dlambda;
    else
        prediction.J_dlambda = NaN;
        prediction.J_lambda = NaN;
    end

    n_con = N + 1 + prediction.has_J;
    values = zeros(n_con, 1);
    names = cell(n_con, 1);
    for i = 1:N
        names{i} = sprintf('cluster_%d', i);
        values(i) = beta - prediction.cluster_lambda(i);
    end
    row = N + 1;
    if prediction.has_J
        names{row} = 'J_mode';
        values(row) = beta - prediction.J_lambda;
        row = row + 1;
    end
    names{row} = 'volume';
    values(row) = mean(rho(:) + drho(:)) - volfrac;

    prediction.constraint_names = names;
    prediction.constraint_values = values;
end

%% ------------------------------------------------------------------
function forensic = append_forensic_record(forensic, out_it, rho_pre, ...
                                           drho_proposed, drho_accepted, ...
                                           beta, lambda_bar, lam_pre, ...
                                           omega_pre, prediction, ...
                                           lam_prop, omega_prop, ...
                                           lam_acc, omega_acc, N_pre, N_after, ...
                                           volume_pre, volume_prop, volume_acc, ...
                                           mac_pre_mass, mac_post_mass, ...
                                           inner_hist, step_alpha)
    rec = forensic.count + 1;
    forensic.count = rec;
    forensic.iter(rec, 1) = out_it;
    forensic.rho_pre(:, rec) = rho_pre(:);
    forensic.drho_proposed(:, rec) = drho_proposed(:);
    forensic.drho_accepted(:, rec) = drho_accepted(:);
    forensic.beta(rec, 1) = beta;
    forensic.beta_omega(rec, 1) = sqrt(max(beta, 0));
    forensic.lambda_bar(rec, 1) = lambda_bar;
    forensic.lambda_pre(rec, :) = lam_pre(:)';
    forensic.omega_pre(rec, :) = omega_pre(:)';
    forensic.lambda_after_proposed(rec, :) = lam_prop(:)';
    forensic.omega_after_proposed(rec, :) = omega_prop(:)';
    forensic.lambda_after_accepted(rec, :) = lam_acc(:)';
    forensic.omega_after_accepted(rec, :) = omega_acc(:)';
    forensic.N_pre(rec, 1) = N_pre;
    forensic.N_after(rec, 1) = N_after;
    forensic.volume_pre(rec, 1) = volume_pre;
    forensic.volume_after_proposed(rec, 1) = volume_prop;
    forensic.volume_after_accepted(rec, 1) = volume_acc;
    forensic.predicted_cluster_lambda{rec, 1} = prediction.cluster_lambda;
    forensic.predicted_cluster_dlambda{rec, 1} = prediction.cluster_dlambda;
    forensic.predicted_J_lambda(rec, 1) = prediction.J_lambda;
    forensic.predicted_J_dlambda(rec, 1) = prediction.J_dlambda;
    forensic.constraint_names{rec, 1} = prediction.constraint_names;
    forensic.constraint_values{rec, 1} = prediction.constraint_values;

    active = false(size(prediction.constraint_values));
    for i = 1:numel(active)
        if strcmp(prediction.constraint_names{i}, 'volume')
            active(i) = abs(prediction.constraint_values(i)) <= forensic.active_tol;
        else
            active(i) = abs(prediction.constraint_values(i)) <= ...
                forensic.active_tol * max(abs(lambda_bar), 1);
        end
    end
    forensic.active_constraints{rec, 1} = prediction.constraint_names(active);
    forensic.mac_pre_mass{rec, 1} = mac_pre_mass;
    forensic.mac_post_mass{rec, 1} = mac_post_mass;

    lambda_tol = forensic.active_tol * max(abs(lam_pre(1)), 1);
    predicted_min = min([prediction.cluster_lambda(:); prediction.J_lambda]);
    if ~isfinite(prediction.J_lambda)
        predicted_min = min(prediction.cluster_lambda(:));
    end
    forensic.predicted_improvement(rec, 1) = beta > lam_pre(1) + lambda_tol || ...
        predicted_min > lam_pre(1) + lambda_tol;
    forensic.real_decrease(rec, 1) = isfinite(lam_prop(1)) && ...
        lam_prop(1) < lam_pre(1) - lambda_tol;
    forensic.predicted_improvement_real_decrease(rec, 1) = ...
        forensic.predicted_improvement(rec) && forensic.real_decrease(rec);

    forensic.inner_iters(rec, 1) = inner_hist.n_iters;
    forensic.inner_converged(rec, 1) = isfield(inner_hist, 'converged') && inner_hist.converged;
    forensic.inner_hit_max_iter(rec, 1) = isfield(inner_hist, 'hit_max_iter') && inner_hist.hit_max_iter;
    forensic.step_alpha(rec, 1) = step_alpha;
end

%% ------------------------------------------------------------------
function mac = forensic_mac(phi, psi, M)
    if isempty(psi) || any(~isfinite(psi(:)))
        mac = nan(size(phi, 2), size(psi, 2));
        return
    end
    mass_phi = real(sum(phi .* (M * phi), 1));
    mass_psi = real(sum(psi .* (M * psi), 1));
    cross = real(phi' * (M * psi));
    mac = cross.^2 ./ max(mass_phi' * mass_psi, eps);
    mac = min(max(mac, 0), 1);
end

%% ------------------------------------------------------------------
function [omega, lam, Phi, M, flag] = eval_modes_exact(rho, Ke_phys_l, Me_phys_l, ...
                                                       iK, jK, nDof, free, ...
                                                       n_modes, opts_eig, ...
                                                       penal, mass_mode)
    [K, M] = assemble_KM_exact(rho, Ke_phys_l, Me_phys_l, iK, jK, nDof, penal, mass_mode);
    Kf = K(free, free);
    Mf = M(free, free);
    nFree = numel(free);

    [V, D, flag] = eigs(Kf, Mf, n_modes, 'SM', opts_eig);
    if flag ~= 0
        opts_r.tol = 1e-8;
        opts_r.maxit = 1500;
        opts_r.p = min(nFree-1, max(40, 4*n_modes));
        [V, D, flag] = eigs(Kf, Mf, n_modes, 'SM', opts_r);
    end

    if flag ~= 0
        lam = nan(n_modes, 1);
        omega = nan(n_modes, 1);
        Phi = nan(nDof, n_modes);
        return
    end

    [lam, idx] = sort(real(diag(D)));
    V = real(V(:, idx));
    for j = 1:n_modes
        v = V(:, j);
        sc = sqrt(abs(v' * (Mf * v)));
        if sc > 1e-14, V(:, j) = v / sc; end
    end
    omega = sqrt(max(lam, 0));
    Phi = zeros(nDof, n_modes);
    for j = 1:n_modes, Phi(free, j) = V(:, j); end
end
