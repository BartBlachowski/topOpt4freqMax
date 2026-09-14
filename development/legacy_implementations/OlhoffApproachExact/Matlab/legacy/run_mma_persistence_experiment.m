% RUN_MMA_PERSISTENCE_EXPERIMENT  Controlled CC benchmark experiment.
%
% Tests whether standard MMA low/upp/xold1/xold2 asymptote persistence across
% outer iterations stabilizes the Du & Olhoff CC benchmark without adding
% damping, line search, trust regions, move limits, acceptance checks, or beta
% caps. This is an experiment-only runner.

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'tools', 'Matlab'));

out_dir = fullfile(fileparts(mfilename('fullpath')), 'results', 'mma_persistence_experiment');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

asy_values = [0.02, 0.05];
paper_cc = 456.4;
all_results = struct([]);

fprintf('\n==========================================================\n');
fprintf('  MMA persistence experiment, CC benchmark\n');
fprintf('  inner_max_iter=1, alpha=1, no acceptance/move/trust/beta cap\n');
fprintf('==========================================================\n\n');

for ia = 1:numel(asy_values)
    cfg = struct();
    cfg.support_type = 'CC';
    cfg.nelx = 40;
    cfg.nely = 5;
    cfg.volfrac = 0.5;
    cfg.mass_mode = 'du2007_c1';
    cfg.sensitivity_filter = true;
    cfg.rmin_elem = 2.5;
    cfg.outer_max_iter = 300;
    cfg.outer_tol = 1e-3;
    cfg.inner_max_iter = 1;
    cfg.alpha = 1.0;
    cfg.acceptance_check = false;
    cfg.move_lim = Inf;
    cfg.outer_move = Inf;
    cfg.asyinit = asy_values(ia);
    cfg.verbose = false;

    fprintf('Running asyinit = %.3f ...\n', cfg.asyinit);
    t0 = tic;
    [rho_final, hist] = topopt_freq_exact_persist_mma_experiment(cfg);
    elapsed = toc(t0);

    ni = hist.outer_iters;
    iter = (1:ni)';
    omega_pre = hist.omega(:, 1:2);
    omega_post = hist.omega_trial(:, 1:2);
    gap_rel = abs(omega_post(:,2) - omega_post(:,1)) ./ max(omega_post(:,1), eps);

    T = table(iter, ...
        omega_pre(:,1), omega_pre(:,2), hist.N, ...
        omega_post(:,1), omega_post(:,2), hist.N_trial, ...
        hist.volume, hist.max_abs_drho, hist.drho_norm, ...
        hist.asym_width_min, hist.asym_width_mean, hist.asym_width_max, ...
        hist.asym_expand_count, hist.asym_contract_count, hist.asym_same_count, ...
        gap_rel, ...
        'VariableNames', {'iter', 'omega1_pre', 'omega2_pre', 'N_pre', ...
        'omega1', 'omega2', 'N', 'volume', 'max_abs_Delta_rho', ...
        'drho_norm', 'asym_width_min', 'asym_width_mean', 'asym_width_max', ...
        'asym_expand_count', 'asym_contract_count', 'asym_same_count', ...
        'relative_gap_omega2_omega1'});

    tag = sprintf('asyinit_%03d', round(1000 * cfg.asyinit));
    csv_path = fullfile(out_dir, [tag '.csv']);
    mat_path = fullfile(out_dir, [tag '.mat']);
    writetable(T, csv_path);
    save(mat_path, 'cfg', 'hist', 'rho_final', 'elapsed', 'paper_cc');

    bimodal = T.N >= 2;
    near_paper = abs(T.omega1 - paper_cc);
    score = near_paper + 100 * T.relative_gap_omega2_omega1;
    if any(bimodal)
        idx_candidates = find(bimodal);
        [~, local_idx] = min(score(idx_candidates));
        best_idx = idx_candidates(local_idx);
    else
        [~, best_idx] = min(score);
    end

    tail_n = min(30, ni);
    tail = (ni-tail_n+1):ni;
    tail_omega1_std = std(T.omega1(tail));
    tail_max_step = max(T.max_abs_Delta_rho(tail));
    tail_mean_width = mean(T.asym_width_mean(tail));

    all_results(ia).asyinit = cfg.asyinit;
    all_results(ia).elapsed = elapsed;
    all_results(ia).outer_iters = ni;
    all_results(ia).final_omega = hist.final_omega;
    all_results(ia).final_N = hist.final_N;
    all_results(ia).final_volume = hist.final_volume;
    all_results(ia).best_iter = best_idx;
    all_results(ia).best_omega1 = T.omega1(best_idx);
    all_results(ia).best_omega2 = T.omega2(best_idx);
    all_results(ia).best_N = T.N(best_idx);
    all_results(ia).best_volume = T.volume(best_idx);
    all_results(ia).best_gap_rel = T.relative_gap_omega2_omega1(best_idx);
    all_results(ia).tail_omega1_std = tail_omega1_std;
    all_results(ia).tail_max_step = tail_max_step;
    all_results(ia).tail_mean_width = tail_mean_width;
    all_results(ia).csv_path = csv_path;
    all_results(ia).mat_path = mat_path;

    fprintf(['  final: iter=%d omega1=%.4f omega2=%.4f N=%g vol=%.5f ', ...
             'tail_std=%.4g tail_max_step=%.4g tail_width=%.4g\n'], ...
        ni, hist.final_omega(1), hist.final_omega(2), hist.final_N, ...
        hist.final_volume, tail_omega1_std, tail_max_step, tail_mean_width);
    fprintf('  best bimodal/near-paper: iter=%d omega1=%.4f omega2=%.4f N=%g gap=%.3g vol=%.5f\n', ...
        best_idx, T.omega1(best_idx), T.omega2(best_idx), T.N(best_idx), ...
        T.relative_gap_omega2_omega1(best_idx), T.volume(best_idx));
    fprintf('  wrote %s\n\n', csv_path);
end

summary_path = fullfile(out_dir, 'summary.mat');
save(summary_path, 'all_results');

fprintf('Summary saved to %s\n', summary_path);
