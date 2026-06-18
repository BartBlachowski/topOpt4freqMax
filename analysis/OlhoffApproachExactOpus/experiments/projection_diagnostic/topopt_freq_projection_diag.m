function [rho_final, x_final, hist] = topopt_freq_projection_diag(cfg)
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
%     .rho_snapshot_interval
%                           diagnostic-only: store rho every N iterations
%                           (default 0 = disabled; no effect on optimizer)
%     .initial_rho          optional nEl x 1 starting density.  If omitted,
%                           starts from uniform volfrac as in the paper.
%
%   Outputs:
%     rho_final   nEl x 1   converged projected physical density
%     x_final     nEl x 1   converged design density
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
rho_snapshot_interval = cfg.rho_snapshot_interval;
projection_beta = cfg.projection_beta;
projection_eta = cfg.projection_eta;
projection_beta_schedule = cfg.projection_beta_schedule;
projection_beta_interval = cfg.projection_beta_interval;

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

[h_filt, Hs_filt] = build_filter(nelx, nely, rmin_elem);

%% ------------------------------------------------------------------
%  Initialise density and history
% ------------------------------------------------------------------
if isfield(cfg, 'initial_rho') && ~isempty(cfg.initial_rho)
    x = cfg.initial_rho(:);
    if numel(x) ~= nEl
        error('topopt_freq_exact:InvalidInitialRho', ...
            'initial_rho has %d entries, expected nelx*nely = %d.', numel(x), nEl);
    end
    x = max(rho_min, min(1, x));
else
    x = volfrac * ones(nEl, 1);
end

hist.omega       = nan(outer_max_iter, n_modes);
hist.beta        = nan(outer_max_iter, 1);
hist.volume      = nan(outer_max_iter, 1);
hist.N           = nan(outer_max_iter, 1);
hist.N_trial     = nan(outer_max_iter, 1);
hist.inner_iters = nan(outer_max_iter, 1);
hist.drho_norm   = nan(outer_max_iter, 1);
hist.drho_max    = nan(outer_max_iter, 1);
hist.design_volume = nan(outer_max_iter, 1);
hist.omega_trial = nan(outer_max_iter, n_modes);
hist.step_alpha  = nan(outer_max_iter, 1);
hist.outer_iters = 0;
hist.volume_field = 'physical_projected_density';
hist.design_filter = 'heaviside_projection_only';
hist.projection_beta = projection_beta;
hist.projection_eta = projection_eta;
hist.projection_beta_schedule = projection_beta_schedule;
hist.projection_beta_interval = projection_beta_interval;
hist.projection_beta_iter = nan(outer_max_iter, 1);
if rho_snapshot_interval > 0
    nSnapMax = ceil(outer_max_iter / rho_snapshot_interval);
    hist.rho_snapshot_iters = nan(nSnapMax, 1);
    hist.rho_snapshots = nan(nEl, nSnapMax);
    hist.x_snapshots = nan(nEl, nSnapMax);
    hist.rho_snapshot_count = 0;
end

if verbose
    fprintf('\n');
    fprintf(' %-4s  %-10s  %-10s  %-3s  %-6s  %-9s  %-10s  %-6s  %-5s\n', ...
        'iter', 'omega_1', 'omega_2', ' N ', 'vol', 'drho/sqrt', 'beta(rad/s)', 'alpha', 'in');
    fprintf(' %s\n', repmat('-', 1, 82));
end

opts_eig.tol   = 1e-10;
opts_eig.maxit = 600;

%% ------------------------------------------------------------------
%  Outer loop
% ------------------------------------------------------------------
for out_it = 1:outer_max_iter

    beta_it = projection_beta_at_iter(out_it, projection_beta, ...
        projection_beta_schedule, projection_beta_interval);

    %% --- Assemble K, M ---
    [rho, dHdx] = heaviside_projection(x, beta_it, projection_eta, rho_min);
    dvol_dx = dHdx / nEl;
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
    fsk_design = zeros(size(fsk_use));
    for s = 1:N
        for k = 1:N
            fsk_design(:,s,k) = fsk_use(:,s,k) .* dHdx;
        end
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
        dlam_J = dlam_J .* dHdx;
    else
        lambda_J = Inf;
        dlam_J   = [];
    end

    %% --- Inner loop (MMA increment subproblem, paper Eq. 19) ---
    [dx_step, beta_fin, i_hist] = inner_loop_mma_projection_diag(x, lambda_bar, fsk_design, ...
        lambda_J, dlam_J, volfrac, rho_min, inner_max_iter, inner_tol, ...
        move_lim, outer_move, mean(rho), dvol_dx);

    %% --- Update design variables (paper Fig. 1 step 4: rho := rho + Delta_rho) ---
    step_alpha = alpha;
    x_new      = max(rho_min, min(1, x + step_alpha * dx_step));
    rho_new    = heaviside_projection(x_new, beta_it, projection_eta, rho_min);
    omega_trial = nan(n_modes, 1);

    if acceptance_check
        while true
            x_trial = max(rho_min, min(1, x + step_alpha * dx_step));
            rho_trial = heaviside_projection(x_trial, beta_it, projection_eta, rho_min);
            [omega_trial, trial_flag] = eval_omega_only(rho_trial, Ke_phys_l, Me_phys_l, ...
                iK, jK, nDof, free, n_modes, opts_eig, penal, mass_mode);

            if trial_flag == 0
                rel_drop = (omega(n_target) - omega_trial(n_target)) / max(omega(n_target), eps);
                if rel_drop <= max_freq_drop || step_alpha <= min_alpha
                    rho_new = rho_trial;
                    break
                end
            elseif step_alpha <= min_alpha
                x_new = x;
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

    drho_norm = norm(x_new - x) / sqrt(nEl);
    drho_max  = max(abs(x_new - x));
    if all(isfinite(omega_trial))
        [N_trial, ~, ~] = detect_multiplicity(omega_trial, n_target, mult_tol);
    else
        N_trial = NaN;
    end

    hist.omega(out_it, :)    = omega(:)';
    hist.beta(out_it)        = beta_fin;
    hist.volume(out_it)      = mean(rho_new);
    hist.design_volume(out_it) = mean(x_new);
    hist.N(out_it)           = N;
    hist.N_trial(out_it)     = N_trial;
    hist.inner_iters(out_it) = i_hist.n_iters;
    hist.drho_norm(out_it)   = drho_norm;
    hist.drho_max(out_it)    = drho_max;
    hist.projection_beta_iter(out_it) = beta_it;
    hist.omega_trial(out_it, :) = omega_trial(:)';
    hist.step_alpha(out_it)  = step_alpha;
    hist.outer_iters         = out_it;
    if rho_snapshot_interval > 0 && (mod(out_it, rho_snapshot_interval) == 0 || out_it == 1)
        hist.rho_snapshot_count = hist.rho_snapshot_count + 1;
        si = hist.rho_snapshot_count;
        hist.rho_snapshot_iters(si) = out_it;
        hist.rho_snapshots(:, si) = rho_new;
        hist.x_snapshots(:, si) = x_new;
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

    x = x_new;

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

x_final = x;
final_beta = projection_beta_at_iter(max(1, hist.outer_iters), projection_beta, ...
    projection_beta_schedule, projection_beta_interval);
rho_final = heaviside_projection(x_final, final_beta, projection_eta, rho_min);
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
hist.final_design_volume = mean(x_final);
hist.final_projection_beta = final_beta;
end

%% ------------------------------------------------------------------
function beta = projection_beta_at_iter(iter, beta_default, beta_schedule, beta_interval)
    if isempty(beta_schedule)
        beta = beta_default;
        return
    end
    idx = min(numel(beta_schedule), floor((iter - 1) / beta_interval) + 1);
    beta = beta_schedule(idx);
end

%% ------------------------------------------------------------------
function [rho_phys, dHdx] = heaviside_projection(x, beta, eta, rho_min)
    denom = tanh(beta * eta) + tanh(beta * (1 - eta));
    rho_raw = (tanh(beta * eta) + tanh(beta * (x - eta))) / denom;
    dHdx = (beta * (1 - tanh(beta * (x - eta)).^2)) / denom;
    rho_phys = max(rho_min, min(1, rho_raw));
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
    cfg = def(cfg, 'rho_snapshot_interval', 0);
    cfg = def(cfg, 'projection_beta',    1.0);
    cfg = def(cfg, 'projection_eta',     0.5);
    cfg = def(cfg, 'projection_beta_schedule', []);
    cfg = def(cfg, 'projection_beta_interval', 25);
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
