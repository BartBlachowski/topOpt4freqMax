% AUDIT_OPTIMIZER_NOCHANGE  Optimizer-side audit for Du & Olhoff CC case.
%
% This script is diagnostic only. It does not modify the production solver.

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'tools', 'Matlab'));
old_warning_state = warning('off', 'all');
cleanup_warning_state = onCleanup(@() warning(old_warning_state));

out_dir = fullfile(fileparts(mfilename('fullpath')), 'results', 'optimizer_audit');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

base = struct();
base.support_type = 'CC';
base.nelx = 40;
base.nely = 5;
base.volfrac = 0.5;
base.mass_mode = 'du2007_c1';
base.sensitivity_filter = true;
base.rmin_elem = 2.5;
base.n_target = 1;
base.n_modes = 4;
base.mult_tol = 1e-3;
base.outer_max_iter = 80;
base.outer_tol = 1e-6;
base.inner_max_iter = 30;
base.inner_tol = 1e-4;
base.move_lim = 0.2;
base.outer_move = 0.2;
base.alpha = 0.5;
base.acceptance_check = false;
base.verbose = false;

paper_cc = 456.4;

fprintf('\n==========================================================\n');
fprintf(' Optimizer audit, CC, no production changes\n');
fprintf(' Baseline: inner_max_iter=30, alpha=0.5, move/outer_move=0.2, no acceptance\n');
fprintf('==========================================================\n\n');

[~, base_hist, base_trace] = audit_run_trace(base);
base_summary = audit_summarize_trace(base_trace, paper_cc);
fprintf('BASE final: iter=%d omega1=%.4f omega2=%.4f omega3=%.4f N=%g vol=%.5f\n', ...
    base_summary.final_iter, base_summary.final_omega1, base_summary.final_omega2, ...
    base_summary.final_omega3, base_summary.final_N, base_summary.final_volume);
fprintf('BASE best near-paper: iter=%d omega1=%.4f omega2=%.4f N=%g vol=%.5f gap=%.3g\n\n', ...
    base_summary.best_iter, base_summary.best_omega1, base_summary.best_omega2, ...
    base_summary.best_N, base_summary.best_volume, base_summary.best_gap);

near_diag = audit_snapshot_diagnostics(base_trace, base_summary.best_iter);
final_diag = audit_snapshot_diagnostics(base_trace, base_summary.final_iter);

save(fullfile(out_dir, 'baseline_trace.mat'), 'base', 'base_hist', 'base_trace', ...
    'base_summary', 'near_diag', 'final_diag');
audit_write_diag_text(fullfile(out_dir, 'baseline_diagnostics.txt'), near_diag, final_diag);

rows = {};
rows(end+1,:) = audit_row('base', 'base', base_summary);

mass_modes = {'linear', 'du2007_step', 'du2007_c0', 'du2007_c1'};
for i = 1:numel(mass_modes)
    cfg = base;
    cfg.mass_mode = mass_modes{i};
    rows(end+1,:) = audit_run_case(sprintf('mass_%s', mass_modes{i}), 'mass_mode', cfg, paper_cc, out_dir);
end

mult_tols = [1e-4, 1e-3, 1e-2, 5e-2];
for i = 1:numel(mult_tols)
    cfg = base;
    cfg.mult_tol = mult_tols(i);
    rows(end+1,:) = audit_run_case(sprintf('mult_%g', mult_tols(i)), 'mult_tol', cfg, paper_cc, out_dir);
end

rmins = [1.5, 2.0, 2.5, 3.0];
for i = 1:numel(rmins)
    cfg = base;
    cfg.rmin_elem = rmins(i);
    rows(end+1,:) = audit_run_case(sprintf('rmin_%.1f', rmins(i)), 'rmin_elem', cfg, paper_cc, out_dir);
end

n_modes_values = [base.n_target + 2, base.n_target + 3, base.n_target + 5, base.n_target + 8];
for i = 1:numel(n_modes_values)
    cfg = base;
    cfg.n_modes = n_modes_values(i);
    rows(end+1,:) = audit_run_case(sprintf('nmodes_%d', n_modes_values(i)), 'n_modes', cfg, paper_cc, out_dir);
end

summary = cell2table(rows, 'VariableNames', {'case_name','factor','final_iter', ...
    'final_omega1','final_omega2','final_omega3','final_N','final_volume', ...
    'best_iter','best_omega1','best_omega2','best_omega3','best_N', ...
    'best_volume','best_gap','best_score','n2_count','near_paper_count', ...
    'low_density_final','low_density_best'});
writetable(summary, fullfile(out_dir, 'sweep_summary.csv'));
save(fullfile(out_dir, 'sweep_summary.mat'), 'summary');

fprintf('\nWrote audit outputs to %s\n', out_dir);

function row = audit_run_case(case_name, factor, cfg, paper_cc, out_dir)
    fprintf('Running %-18s (%s) ...\n', case_name, factor);
    [~, hist, trace] = audit_run_trace(cfg);
    summary = audit_summarize_trace(trace, paper_cc);
    save(fullfile(out_dir, [sanitize(case_name) '.mat']), 'cfg', 'hist', 'trace', 'summary');
    fprintf('  final omega1=%.4f omega2=%.4f N=%g vol=%.5f | best iter=%d omega1=%.4f omega2=%.4f N=%g vol=%.5f gap=%.3g\n', ...
        summary.final_omega1, summary.final_omega2, summary.final_N, summary.final_volume, ...
        summary.best_iter, summary.best_omega1, summary.best_omega2, summary.best_N, ...
        summary.best_volume, summary.best_gap);
    row = audit_row(case_name, factor, summary);
end

function row = audit_row(case_name, factor, s)
    row = {case_name, factor, s.final_iter, s.final_omega1, s.final_omega2, ...
        s.final_omega3, s.final_N, s.final_volume, s.best_iter, s.best_omega1, ...
        s.best_omega2, s.best_omega3, s.best_N, s.best_volume, s.best_gap, ...
        s.best_score, s.n2_count, s.near_paper_count, s.low_density_final, ...
        s.low_density_best};
end

function [rho_final, hist, trace] = audit_run_trace(cfg)
    cfg = audit_defaults(cfg);

    L = cfg.L; H = cfg.H; nelx = cfg.nelx; nely = cfg.nely;
    E0 = cfg.E0; nu = cfg.nu; rho0 = cfg.rho0; t = cfg.t;
    n_target = cfg.n_target; n_modes = cfg.n_modes;
    dx = L / nelx; dy = H / nely;
    nEl = nelx * nely;
    nDof = 2 * (nelx+1) * (nely+1);

    [Ke_star, Me_star] = fe_q4_exact(nu, t, dx, dy);
    Ke_phys = E0 * Ke_star;
    Me_phys = rho0 * Me_star;

    nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
    cVec = reshape(2*nodeNrs(1:nely,1:nelx)+1, nEl, 1);
    cMat = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
            cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];

    [Il, Jl] = find(tril(ones(8)));
    iK = reshape(cMat(:,Il)', [], 1);
    jK = reshape(cMat(:,Jl)', [], 1);
    Ke_phys_l = Ke_phys(sub2ind([8,8], Il, Jl));
    Me_phys_l = Me_phys(sub2ind([8,8], Il, Jl));

    fixed = build_supports_exact(cfg.support_type, nodeNrs);
    free = setdiff(1:nDof, fixed);

    if cfg.sensitivity_filter
        [h_filt, Hs_filt] = build_filter(nelx, nely, cfg.rmin_elem);
    end

    rho = cfg.volfrac * ones(nEl, 1);
    hist.omega = nan(cfg.outer_max_iter, n_modes);
    hist.omega_trial = nan(cfg.outer_max_iter, n_modes);
    hist.N = nan(cfg.outer_max_iter, 1);
    hist.N_trial = nan(cfg.outer_max_iter, 1);
    hist.beta = nan(cfg.outer_max_iter, 1);
    hist.volume = nan(cfg.outer_max_iter, 1);
    hist.drho_norm = nan(cfg.outer_max_iter, 1);
    hist.outer_iters = 0;

    trace = struct();
    trace.cfg = cfg;
    trace.rho_pre = nan(nEl, cfg.outer_max_iter);
    trace.rho_post = nan(nEl, cfg.outer_max_iter);
    trace.drho = nan(nEl, cfg.outer_max_iter);
    trace.omega_pre = nan(cfg.outer_max_iter, n_modes);
    trace.omega_post = nan(cfg.outer_max_iter, n_modes);
    trace.lambda_pre = nan(cfg.outer_max_iter, n_modes);
    trace.lambda_post = nan(cfg.outer_max_iter, n_modes);
    trace.N_pre = nan(cfg.outer_max_iter, 1);
    trace.N_post = nan(cfg.outer_max_iter, 1);
    trace.N_pre_lambda_tol = nan(cfg.outer_max_iter, 1);
    trace.N_post_lambda_tol = nan(cfg.outer_max_iter, 1);
    trace.cluster = cell(cfg.outer_max_iter, 1);
    trace.J_idx = nan(cfg.outer_max_iter, 1);
    trace.beta = nan(cfg.outer_max_iter, 1);
    trace.lambda_bar = nan(cfg.outer_max_iter, 1);
    trace.lambda_J = nan(cfg.outer_max_iter, 1);
    trace.constraint_values = cell(cfg.outer_max_iter, 1);
    trace.constraint_names = cell(cfg.outer_max_iter, 1);
    trace.volume = nan(cfg.outer_max_iter, 1);
    trace.low_count = nan(cfg.outer_max_iter, 1);
    trace.max_abs_drho = nan(cfg.outer_max_iter, 1);
    trace.final_iter = 0;

    opts_eig.tol = 1e-10;
    opts_eig.maxit = 600;

    for it = 1:cfg.outer_max_iter
        rho_pre = rho;
        [omega, lam, Phi, flag] = audit_eigs(rho_pre, Ke_phys_l, Me_phys_l, ...
            iK, jK, nDof, free, n_modes, opts_eig, cfg.penal, cfg.mass_mode);
        if flag ~= 0, break; end

        [N, J_idx, cluster_idx] = detect_multiplicity(omega, n_target, cfg.mult_tol);
        [N_lambda_tol, ~, ~] = audit_detect_multiplicity_lambda(lam, n_target, cfg.mult_tol);
        lambda_bar = mean(lam(cluster_idx));
        Phi_cluster = Phi(:, cluster_idx);
        fsk_raw = compute_generalized_gradients(rho_pre, lambda_bar, Phi_cluster, ...
            cMat, Ke_phys, Me_phys, cfg.penal, cfg.mass_mode);
        if cfg.sensitivity_filter
            fsk_use = zeros(size(fsk_raw));
            for s = 1:N
                for k = 1:N
                    fsk_use(:,s,k) = apply_sensitivity_filter(fsk_raw(:,s,k), ...
                        rho_pre, h_filt, Hs_filt, nely, nelx);
                end
            end
        else
            fsk_use = fsk_raw;
        end

        if J_idx > 0
            lambda_J = lam(J_idx);
            dlam_J_raw = compute_elem_sensitivity(rho_pre, lambda_J, Phi(:,J_idx), ...
                cMat, Ke_phys, Me_phys, free, nDof, cfg.penal, cfg.mass_mode);
            if cfg.sensitivity_filter
                dlam_J = apply_sensitivity_filter(dlam_J_raw, rho_pre, h_filt, Hs_filt, nely, nelx);
            else
                dlam_J = dlam_J_raw;
            end
        else
            lambda_J = Inf;
            dlam_J = [];
        end

        [drho, beta_fin, ~] = inner_loop_mma(rho_pre, lambda_bar, fsk_use, ...
            lambda_J, dlam_J, cfg.volfrac, cfg.rho_min, cfg.inner_max_iter, ...
            cfg.inner_tol, cfg.move_lim, cfg.outer_move);

        rho_post = max(cfg.rho_min, min(1, rho_pre + cfg.alpha * drho));
        [omega_post, lam_post, ~, flag_post] = audit_eigs(rho_post, Ke_phys_l, ...
            Me_phys_l, iK, jK, nDof, free, n_modes, opts_eig, cfg.penal, cfg.mass_mode);
        if flag_post ~= 0
            omega_post(:) = NaN;
            lam_post(:) = NaN;
            N_post = NaN;
            N_post_lambda_tol = NaN;
        else
            [N_post, ~, ~] = detect_multiplicity(omega_post, n_target, cfg.mult_tol);
            [N_post_lambda_tol, ~, ~] = audit_detect_multiplicity_lambda(lam_post, n_target, cfg.mult_tol);
        end

        [cvals, cnames] = audit_constraints(beta_fin, lambda_bar, fsk_use, ...
            lambda_J, dlam_J, drho, cfg.volfrac, rho_pre);

        trace.rho_pre(:,it) = rho_pre;
        trace.rho_post(:,it) = rho_post;
        trace.drho(:,it) = rho_post - rho_pre;
        trace.omega_pre(it,:) = omega(:)';
        trace.omega_post(it,:) = omega_post(:)';
        trace.lambda_pre(it,:) = lam(:)';
        trace.lambda_post(it,:) = lam_post(:)';
        trace.N_pre(it) = N;
        trace.N_post(it) = N_post;
        trace.N_pre_lambda_tol(it) = N_lambda_tol;
        trace.N_post_lambda_tol(it) = N_post_lambda_tol;
        trace.cluster{it} = cluster_idx;
        trace.J_idx(it) = J_idx;
        trace.beta(it) = beta_fin;
        trace.lambda_bar(it) = lambda_bar;
        trace.lambda_J(it) = lambda_J;
        trace.constraint_values{it} = cvals;
        trace.constraint_names{it} = cnames;
        trace.volume(it) = mean(rho_post);
        trace.low_count(it) = nnz(rho_post < 0.1);
        trace.max_abs_drho(it) = max(abs(rho_post - rho_pre));

        hist.omega(it,:) = omega(:)';
        hist.omega_trial(it,:) = omega_post(:)';
        hist.N(it) = N;
        hist.N_trial(it) = N_post;
        hist.beta(it) = beta_fin;
        hist.volume(it) = mean(rho_post);
        hist.drho_norm(it) = norm(rho_post - rho_pre) / sqrt(nEl);
        hist.outer_iters = it;

        rho = rho_post;
        if hist.drho_norm(it) < cfg.outer_tol
            break;
        end
    end

    ni = hist.outer_iters;
    hist = audit_trim_numeric(hist, ni, cfg.outer_max_iter);
    trace = audit_trim_trace(trace, ni, cfg.outer_max_iter);
    trace.final_iter = ni;
    rho_final = rho;
end

function [cvals, cnames] = audit_constraints(beta, lambda_bar, fsk, lambda_J, dlam_J, drho, volfrac, rho)
    N = size(fsk, 2);
    fsk2D = reshape(fsk, numel(rho), N*N);
    F = reshape(fsk2D' * drho(:), N, N);
    mu = sort(real(eig(F)), 'ascend');
    cvals = zeros(N + 1 + isfinite(lambda_J), 1);
    cnames = cell(size(cvals));
    for i = 1:N
        cvals(i) = beta - lambda_bar - mu(i);
        cnames{i} = sprintf('cluster_%d', i);
    end
    row = N + 1;
    if isfinite(lambda_J)
        cvals(row) = beta - lambda_J - dlam_J(:)' * drho(:);
        cnames{row} = 'J_mode';
        row = row + 1;
    end
    cvals(row) = mean(rho(:) + drho(:)) - volfrac;
    cnames{row} = 'volume';
end

function s = audit_summarize_trace(trace, paper_cc)
    ni = trace.final_iter;
    omega = trace.omega_post(1:ni,:);
    gap = abs(omega(:,2) - omega(:,1)) ./ max(omega(:,1), eps);
    score = abs(omega(:,1) - paper_cc) + 1000 * gap;
    [best_score, best_iter] = min(score);
    final_iter = ni;
    s = struct();
    s.final_iter = final_iter;
    s.final_omega1 = omega(final_iter,1);
    s.final_omega2 = omega(final_iter,2);
    s.final_omega3 = omega(final_iter,min(3,size(omega,2)));
    s.final_N = trace.N_post(final_iter);
    s.final_volume = trace.volume(final_iter);
    s.best_iter = best_iter;
    s.best_omega1 = omega(best_iter,1);
    s.best_omega2 = omega(best_iter,2);
    s.best_omega3 = omega(best_iter,min(3,size(omega,2)));
    s.best_N = trace.N_post(best_iter);
    s.best_volume = trace.volume(best_iter);
    s.best_gap = gap(best_iter);
    s.best_score = best_score;
    s.n2_count = nnz(trace.N_post(1:ni) >= 2);
    s.near_paper_count = nnz(abs(omega(:,1) - paper_cc) < 25);
    s.low_density_final = trace.low_count(final_iter);
    s.low_density_best = trace.low_count(best_iter);
end

function diag = audit_snapshot_diagnostics(trace, iter)
    cfg = trace.cfg;
    rho = trace.rho_post(:,iter);
    rho_pre = trace.rho_pre(:,iter);
    drho = trace.drho(:,iter);
    diag = struct();
    diag.iter = iter;
    diag.omega = trace.omega_post(iter,:);
    diag.lambda = trace.lambda_post(iter,:);
    diag.N = trace.N_post(iter);
    diag.N_lambda_relative = trace.N_post_lambda_tol(iter);
    diag.cluster = trace.cluster{iter};
    diag.beta = trace.beta(iter);
    diag.lambda_bar = trace.lambda_bar(iter);
    diag.lambda_J = trace.lambda_J(iter);
    diag.constraint_names = trace.constraint_names{iter};
    diag.constraint_values = trace.constraint_values{iter};
    active = false(size(diag.constraint_values));
    for i = 1:numel(active)
        if strcmp(diag.constraint_names{i}, 'volume')
            active(i) = abs(diag.constraint_values(i)) < 1e-4;
        else
            active(i) = abs(diag.constraint_values(i)) < 1e-4 * max(abs(diag.lambda_bar), 1);
        end
    end
    diag.active_constraints = diag.constraint_names(active);
    diag.volume = trace.volume(iter);
    diag.mean_density = mean(rho);
    diag.min_density = min(rho);
    diag.max_density = max(rho);
    diag.n_below_01 = nnz(rho < 0.1);
    diag.max_abs_drho = max(abs(drho));
    diag.rho_pre_mean = mean(rho_pre);

    [mcoef, ~] = mass_interp(rho, cfg.mass_mode);
    kcoef = rho(:).^cfg.penal;
    low = rho < 0.1;
    diag.low_k_coeff_fraction = sum(kcoef(low)) / max(sum(kcoef), eps);
    diag.low_m_coeff_fraction = sum(mcoef(low)) / max(sum(mcoef), eps);

    local = audit_local_sensitivity(cfg, rho);
    diag.dlambda_min = min(local.dlam1);
    diag.dlambda_max = max(local.dlam1);
    diag.dlambda_mean = mean(local.dlam1);
    diag.dlambda_pos_count = nnz(local.dlam1 > 0);
    diag.dlambda_neg_count = nnz(local.dlam1 < 0);
    diag.domega_min = min(local.domega1);
    diag.domega_max = max(local.domega1);
    diag.domega_mean = mean(local.domega1);
    diag.low_modal_K_fraction = local.low_modal_K_fraction;
    diag.low_modal_M_fraction = local.low_modal_M_fraction;
    diag.adding_material_increases_omega1_count = nnz(local.domega1 > 0);
    diag.adding_material_decreases_omega1_count = nnz(local.domega1 < 0);
end

function local = audit_local_sensitivity(cfg, rho)
    nelx = cfg.nelx; nely = cfg.nely; nEl = nelx*nely;
    dx = cfg.L/nelx; dy = cfg.H/nely;
    nDof = 2*(nelx+1)*(nely+1);
    [Ke_star, Me_star] = fe_q4_exact(cfg.nu, cfg.t, dx, dy);
    Ke_phys = cfg.E0 * Ke_star;
    Me_phys = cfg.rho0 * Me_star;
    nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
    cVec = reshape(2*nodeNrs(1:nely,1:nelx)+1, nEl, 1);
    cMat = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
            cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];
    [Il, Jl] = find(tril(ones(8)));
    iK = reshape(cMat(:,Il)', [], 1);
    jK = reshape(cMat(:,Jl)', [], 1);
    Ke_phys_l = Ke_phys(sub2ind([8,8], Il, Jl));
    Me_phys_l = Me_phys(sub2ind([8,8], Il, Jl));
    fixed = build_supports_exact(cfg.support_type, nodeNrs);
    free = setdiff(1:nDof, fixed);
    opts.tol = 1e-10; opts.maxit = 600;
    [omega, lam, Phi, ~] = audit_eigs(rho, Ke_phys_l, Me_phys_l, iK, jK, ...
        nDof, free, cfg.n_modes, opts, cfg.penal, cfg.mass_mode);
    dlam1 = compute_elem_sensitivity(rho, lam(1), Phi(:,1), cMat, ...
        Ke_phys, Me_phys, free, nDof, cfg.penal, cfg.mass_mode);
    if cfg.sensitivity_filter
        [h, Hs] = build_filter(nelx, nely, cfg.rmin_elem);
        dlam1 = apply_sensitivity_filter(dlam1, rho, h, Hs, nely, nelx);
    end
    domega1 = dlam1 / (2 * max(omega(1), eps));
    [mcoef, ~] = mass_interp(rho, cfg.mass_mode);
    kcoef = rho(:).^cfg.penal;
    phi = Phi(:,1);
    Kterm = zeros(nEl,1);
    Mterm = zeros(nEl,1);
    for e = 1:nEl
        dofs = cMat(e,:);
        pe = phi(dofs);
        Kterm(e) = kcoef(e) * (pe' * Ke_phys * pe);
        Mterm(e) = mcoef(e) * (pe' * Me_phys * pe);
    end
    low = rho < 0.1;
    local.dlam1 = dlam1;
    local.domega1 = domega1;
    local.low_modal_K_fraction = sum(Kterm(low)) / max(sum(Kterm), eps);
    local.low_modal_M_fraction = sum(Mterm(low)) / max(sum(Mterm), eps);
end

function [omega, lam, Phi, flag] = audit_eigs(rho, Ke_phys_l, Me_phys_l, iK, jK, nDof, free, n_modes, opts, penal, mass_mode)
    [K, M] = assemble_KM_exact(rho, Ke_phys_l, Me_phys_l, iK, jK, nDof, penal, mass_mode);
    Kf = K(free,free);
    Mf = M(free,free);
    [V, D, flag] = eigs(Kf, Mf, n_modes, 'SM', opts);
    if flag ~= 0
        opts2 = opts;
        opts2.tol = 1e-8;
        opts2.maxit = 1500;
        opts2.p = min(numel(free)-1, max(40, 4*n_modes));
        [V, D, flag] = eigs(Kf, Mf, n_modes, 'SM', opts2);
    end
    if flag ~= 0
        omega = nan(n_modes,1);
        lam = nan(n_modes,1);
        Phi = nan(nDof,n_modes);
        return
    end
    [lam, idx] = sort(real(diag(D)));
    V = real(V(:,idx));
    for j = 1:n_modes
        sc = sqrt(abs(V(:,j)' * (Mf * V(:,j))));
        if sc > 1e-14, V(:,j) = V(:,j) / sc; end
    end
    omega = sqrt(max(lam,0));
    Phi = zeros(nDof,n_modes);
    for j = 1:n_modes
        Phi(free,j) = V(:,j);
    end
end

function cfg = audit_defaults(cfg)
    cfg = def(cfg, 'L', 8.0);
    cfg = def(cfg, 'H', 1.0);
    cfg = def(cfg, 'E0', 1e7);
    cfg = def(cfg, 'nu', 0.3);
    cfg = def(cfg, 'rho0', 1.0);
    cfg = def(cfg, 't', 1.0);
    cfg = def(cfg, 'rho_min', 1e-3);
    cfg = def(cfg, 'penal', 3.0);
end

function cfg = def(cfg, name, value)
    if ~isfield(cfg, name) || isempty(cfg.(name)), cfg.(name) = value; end
end

function s = audit_trim_numeric(s, ni, maxit)
    f = fieldnames(s);
    for i = 1:numel(f)
        v = s.(f{i});
        if isnumeric(v) && size(v,1) == maxit
            s.(f{i}) = v(1:ni,:);
        end
    end
end

function trace = audit_trim_trace(trace, ni, maxit)
    f = fieldnames(trace);
    for i = 1:numel(f)
        v = trace.(f{i});
        if isnumeric(v) && size(v,2) == maxit && size(v,1) ~= maxit
            trace.(f{i}) = v(:,1:ni);
        elseif isnumeric(v) && size(v,1) == maxit
            trace.(f{i}) = v(1:ni,:);
        elseif iscell(v) && numel(v) == maxit
            trace.(f{i}) = v(1:ni);
        end
    end
end

function audit_write_diag_text(path, near_diag, final_diag)
    fid = fopen(path, 'w');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, 'NEAR-PAPER SNAPSHOT\n');
    audit_print_diag(fid, near_diag);
    fprintf(fid, '\nFINAL SNAPSHOT\n');
    audit_print_diag(fid, final_diag);
end

function audit_print_diag(fid, d)
    fprintf(fid, 'iter: %d\n', d.iter);
    fprintf(fid, 'omega: %.10g %.10g %.10g\n', d.omega(1), d.omega(2), d.omega(3));
    fprintf(fid, 'lambda: %.10g %.10g %.10g\n', d.lambda(1), d.lambda(2), d.lambda(3));
    fprintf(fid, 'N: %g\n', d.N);
    fprintf(fid, 'N with eigenvalue-relative tolerance: %g\n', d.N_lambda_relative);
    fprintf(fid, 'cluster: %s\n', mat2str(d.cluster));
    fprintf(fid, 'beta: %.10g\n', d.beta);
    fprintf(fid, 'lambda_bar: %.10g\n', d.lambda_bar);
    fprintf(fid, 'lambda_J: %.10g\n', d.lambda_J);
    fprintf(fid, 'constraints:\n');
    for i = 1:numel(d.constraint_values)
        fprintf(fid, '  %s: %.10g\n', d.constraint_names{i}, d.constraint_values(i));
    end
    fprintf(fid, 'active constraints: %s\n', strjoin(d.active_constraints(:)', ', '));
    fprintf(fid, 'volume/mean density: %.10g\n', d.volume);
    fprintf(fid, 'min/max density: %.10g %.10g\n', d.min_density, d.max_density);
    fprintf(fid, 'elements below 0.1: %d\n', d.n_below_01);
    fprintf(fid, 'low-density K/M coefficient fractions: %.10g %.10g\n', ...
        d.low_k_coeff_fraction, d.low_m_coeff_fraction);
    fprintf(fid, 'low-density modal K/M fractions: %.10g %.10g\n', ...
        d.low_modal_K_fraction, d.low_modal_M_fraction);
    fprintf(fid, 'dlam/drho min/mean/max: %.10g %.10g %.10g\n', ...
        d.dlambda_min, d.dlambda_mean, d.dlambda_max);
    fprintf(fid, 'domega/drho min/mean/max: %.10g %.10g %.10g\n', ...
        d.domega_min, d.domega_mean, d.domega_max);
    fprintf(fid, 'adding material increases/decreases omega1 counts: %d %d\n', ...
        d.adding_material_increases_omega1_count, d.adding_material_decreases_omega1_count);
end

function [N, J_idx, cluster_idx] = audit_detect_multiplicity_lambda(lam, n, mult_tol)
    lam = lam(:);
    nModes = numel(lam);
    ref = max(abs(lam(n)), eps);
    N = 1;
    for j = n+1:nModes
        if abs(lam(j) - lam(n)) / ref <= mult_tol
            N = N + 1;
        else
            break
        end
    end
    cluster_idx = n:n+N-1;
    J_idx = n + N;
    if J_idx > nModes, J_idx = 0; end
end

function s = sanitize(s)
    s = regexprep(s, '[^A-Za-z0-9_]+', '_');
end
