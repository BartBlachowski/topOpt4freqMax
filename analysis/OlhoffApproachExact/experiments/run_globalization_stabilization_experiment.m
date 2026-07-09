% RUN_GLOBALIZATION_STABILIZATION_EXPERIMENT
%
% Outer-update globalization/stabilization experiment for the Du & Olhoff
% (2007) clamped-clamped 40x5 benchmark.
%
% Scope: optimizer update controls only.  FE, interpolation, sensitivities,
% filters, boundary conditions, and generalized gradients are not modified.

clearvars;
close all;
clc;

this_dir = fileparts(mfilename('fullpath'));
repo_root = fullfile(this_dir, '..', '..', '..');
matlab_dir = fullfile(this_dir, '..', 'Matlab');
tools_dir = fullfile(repo_root, 'tools', 'Matlab');
addpath(matlab_dir);
addpath(tools_dir);

out_dir = fullfile(this_dir, 'globalization_stabilization_results');
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
base.outer_max_iter = 100;
base.outer_tol = 1e-6;
base.inner_max_iter = 30;
base.inner_tol = 1e-4;
base.move_lim = 0.2;
base.outer_move = 0.2;
base.alpha = 0.5;
base.acceptance_check = false;
base.verbose = false;
base.rho_snapshot_interval = 1;

base.globalization_alpha_start = 1.0;
base.globalization_alpha_min = 1/128;
base.globalization_low_mode_mac_threshold = 0.25;
base.post_coalescence_gap_tol = 0.005;

basin_omega_min = 450.0;
basin_gap_tol = 0.005;
required_retention = 20;
paper_target_omega1 = 456.4;
paper_close_rel_tol = 0.02;

variants = build_variants(base);
summaries = cell(numel(variants), 1);

fprintf('\n==========================================================\n');
fprintf(' OlhoffApproachExact globalization/stabilization experiment\n');
fprintf(' Output: %s\n', out_dir);
fprintf(' Basin: omega1 >= %.1f and gap12 <= %.4g; retain >= %d iters\n', ...
    basin_omega_min, basin_gap_tol, required_retention);
fprintf(' Paper-like guard: |omega1 - %.1f|/%.1f <= %.3g and retain >= %d iters\n', ...
    paper_target_omega1, paper_target_omega1, paper_close_rel_tol, required_retention);
fprintf('==========================================================\n\n');

for iv = 1:numel(variants)
    v = variants(iv);
    variant_dir = fullfile(out_dir, v.tag);
    if ~exist(variant_dir, 'dir'), mkdir(variant_dir); end

    fprintf('Running %-18s  %s\n', v.tag, v.description);
    elapsed_tic = tic;
    [rho_final, hist] = topopt_freq_exact(v.cfg);
    elapsed_s = toc(elapsed_tic);

    summary = summarize_variant(v, hist, rho_final, elapsed_s, ...
        basin_omega_min, basin_gap_tol, required_retention, ...
        paper_target_omega1, paper_close_rel_tol, variant_dir);
    summaries{iv} = summary;

    T = iteration_table(hist, summary);
    writetable(T, fullfile(variant_dir, [v.tag '_iterations.csv']));
    writematrix(rho_final(:), fullfile(variant_dir, [v.tag '_rho_final.csv']));
    export_topology_png(rho_final, v.cfg, summary, fullfile(variant_dir, [v.tag '_topology.png']));
    save(fullfile(variant_dir, [v.tag '_result.mat']), ...
        'v', 'hist', 'rho_final', 'summary', 'T', 'elapsed_s');

    fprintf(['  final omega=(%.4f, %.4f) N=%g; entry=%s exit=%s ', ...
        'max_streak=%d reject_outer=%d reject_trials=%d retained20=%d\n'], ...
        summary.final_omega1, summary.final_omega2, summary.final_N, ...
        fmt_iter(summary.basin_entry_iter), fmt_iter(summary.basin_exit_iter), ...
        summary.max_basin_streak, summary.rejected_outer_steps, ...
        summary.rejected_trial_count, summary.paper_like_bimodal_retained);
end

summary_struct = [summaries{:}];
summary_table = summaries_to_table(summary_struct);
writetable(summary_table, fullfile(out_dir, 'globalization_summary.csv'));
save(fullfile(out_dir, 'globalization_stabilization_results.mat'), ...
    'base', 'variants', 'summary_struct', 'summary_table', ...
    'basin_omega_min', 'basin_gap_tol', 'required_retention', ...
    'paper_target_omega1', 'paper_close_rel_tol');
write_report(fullfile(this_dir, 'globalization_stabilization_report.md'), ...
    summary_struct, base, basin_omega_min, basin_gap_tol, required_retention, ...
    paper_target_omega1, paper_close_rel_tol);

fprintf('\nReport written to %s\n', fullfile(this_dir, 'globalization_stabilization_report.md'));

function variants = build_variants(base)
variants = repmat(struct('tag','','description','','cfg',base), 9, 1);
k = 0;

k = k + 1;
cfg = base;
cfg.globalization_enabled = false;
cfg.post_coalescence_trust_enabled = false;
variants(k) = make_variant('A_baseline', 'current behavior', cfg);

k = k + 1;
cfg = base;
cfg.globalization_enabled = true;
cfg.globalization_monotone_cluster = true;
cfg.globalization_low_mode_guard = false;
cfg.post_coalescence_trust_enabled = false;
variants(k) = make_variant('B_monotone_cluster', 'monotone cluster acceptance', cfg);

k = k + 1;
cfg = base;
cfg.globalization_enabled = true;
cfg.globalization_monotone_cluster = false;
cfg.globalization_low_mode_guard = true;
cfg.post_coalescence_trust_enabled = false;
variants(k) = make_variant('C_low_mode_guard', 'low-mode MAC guard', cfg);

factors = [0.5, 0.25, 0.1];
for i = 1:numel(factors)
    k = k + 1;
    cfg = base;
    cfg.globalization_enabled = false;
    cfg.post_coalescence_trust_enabled = true;
    cfg.post_coalescence_trust_factor = factors(i);
    variants(k) = make_variant(sprintf('D_trust_%s', factor_tag(factors(i))), ...
        sprintf('post-coalescence trust factor %.3g', factors(i)), cfg);
end

for i = 1:numel(factors)
    k = k + 1;
    cfg = base;
    cfg.globalization_enabled = true;
    cfg.globalization_monotone_cluster = true;
    cfg.globalization_low_mode_guard = true;
    cfg.post_coalescence_trust_enabled = true;
    cfg.post_coalescence_trust_factor = factors(i);
    variants(k) = make_variant(sprintf('E_combined_%s', factor_tag(factors(i))), ...
        sprintf('B+C+D with trust factor %.3g', factors(i)), cfg);
end
end

function v = make_variant(tag, description, cfg)
v = struct();
v.tag = tag;
v.description = description;
v.cfg = cfg;
end

function summary = summarize_variant(v, hist, rho_final, elapsed_s, ...
    basin_omega_min, basin_gap_tol, required_retention, ...
    paper_target_omega1, paper_close_rel_tol, variant_dir)

ni = hist.outer_iters;
omega = hist.omega_trial(1:ni, :);
fallback = ~isfinite(omega(:,1));
omega(fallback, :) = hist.omega(fallback, :);
lambda = omega.^2;
N_post = hist.N_trial(1:ni);
missing_N = ~isfinite(N_post);
N_post(missing_N) = hist.N(missing_N);
volume = hist.volume(1:ni);
gap12 = abs(omega(:,2) - omega(:,1)) ./ max(omega(:,1), eps);
basin = omega(:,1) >= basin_omega_min & gap12 <= basin_gap_tol;
paper_like = abs(omega(:,1) - paper_target_omega1) / paper_target_omega1 <= ...
    paper_close_rel_tol & gap12 <= basin_gap_tol;

[entry_iter, exit_iter] = entry_exit_iters(basin);
[max_streak, streak_start, streak_end] = max_true_streak(basin);
[paper_streak, paper_streak_start, paper_streak_end] = max_true_streak(paper_like);
paper_like_bimodal_retained = paper_streak >= required_retention;

rejected_outer = hist.globalization_rejected_outer_step(1:ni);
rejected_trial_count = hist.globalization_rejected_trial_count(1:ni);
rejected_trial_count(~isfinite(rejected_trial_count)) = 0;
step_alpha = hist.step_alpha(1:ni);

summary = struct();
summary.tag = v.tag;
summary.description = v.description;
summary.elapsed_s = elapsed_s;
summary.variant_dir = variant_dir;
summary.outer_iters = ni;
summary.final_omega1 = hist.final_omega(1);
summary.final_omega2 = hist.final_omega(2);
summary.final_N = hist.final_N;
summary.final_volume = hist.final_volume;
summary.basin_entry_iter = entry_iter;
summary.basin_exit_iter = exit_iter;
summary.max_basin_streak = max_streak;
summary.max_basin_streak_start = streak_start;
summary.max_basin_streak_end = streak_end;
summary.max_paper_like_streak = paper_streak;
summary.max_paper_like_streak_start = paper_streak_start;
summary.max_paper_like_streak_end = paper_streak_end;
summary.paper_like_bimodal_retained = paper_like_bimodal_retained;
summary.rejected_outer_steps = nnz(rejected_outer);
summary.rejected_trial_count = sum(rejected_trial_count);
summary.accepted_alpha_min = min(step_alpha);
summary.accepted_alpha_median = median(step_alpha);
summary.accepted_alpha_values = unique(step_alpha(:))';
summary.final_topology_png = fullfile(variant_dir, [v.tag '_topology.png']);
summary.final_rho_csv = fullfile(variant_dir, [v.tag '_rho_final.csv']);
summary.per_iteration_csv = fullfile(variant_dir, [v.tag '_iterations.csv']);
summary.config = v.cfg;
summary.per_iteration = struct('omega', omega, 'lambda', lambda, 'N_post', N_post, ...
    'volume', volume, 'gap12', gap12, 'basin', basin, 'paper_like', paper_like);
summary.rho_final = rho_final;
end

function T = iteration_table(hist, summary)
ni = summary.outer_iters;
iter = (1:ni)';
omega = summary.per_iteration.omega;
gap12 = summary.per_iteration.gap12;
alpha = hist.step_alpha(1:ni);
rejected_outer = hist.globalization_rejected_outer_step(1:ni);
rejected_trials = hist.globalization_rejected_trial_count(1:ni);
rejected_trials(~isfinite(rejected_trials)) = 0;
trial_count = hist.globalization_trial_count(1:ni);
trial_count(~isfinite(trial_count)) = 0;
move_eff = hist.move_lim_effective(1:ni);
outer_eff = hist.outer_move_effective(1:ni);
reason = hist.globalization_reason(1:ni);
T = table(iter, omega(:,1), omega(:,2), gap12, summary.per_iteration.N_post, ...
    hist.volume(1:ni), hist.drho_norm(1:ni), hist.drho_max(1:ni), alpha, ...
    rejected_outer, rejected_trials, trial_count, move_eff, outer_eff, ...
    summary.per_iteration.basin, reason, ...
    'VariableNames', {'iter','omega1','omega2','gap12','N_post','volume', ...
    'drho_norm','drho_max','accepted_alpha','rejected_outer_step', ...
    'rejected_trial_count','trial_count','move_lim_effective', ...
    'outer_move_effective','coalesced_basin','globalization_reason'});
T.paper_like_basin = summary.per_iteration.paper_like;
end

function T = summaries_to_table(s)
n = numel(s);
tag = cell(n,1); description = cell(n,1); entry = nan(n,1); exit_iter = nan(n,1);
outer_iters = nan(n,1); final_omega1 = nan(n,1); final_omega2 = nan(n,1);
final_N = nan(n,1); final_volume = nan(n,1); max_streak = nan(n,1);
paper_streak = nan(n,1);
retained20 = false(n,1); rejected_outer = nan(n,1); rejected_trials = nan(n,1);
alpha_min = nan(n,1); alpha_median = nan(n,1); topology = cell(n,1);
for i = 1:n
    tag{i} = s(i).tag;
    description{i} = s(i).description;
    outer_iters(i) = s(i).outer_iters;
    entry(i) = s(i).basin_entry_iter;
    exit_iter(i) = s(i).basin_exit_iter;
    final_omega1(i) = s(i).final_omega1;
    final_omega2(i) = s(i).final_omega2;
    final_N(i) = s(i).final_N;
    final_volume(i) = s(i).final_volume;
    max_streak(i) = s(i).max_basin_streak;
    paper_streak(i) = s(i).max_paper_like_streak;
    retained20(i) = s(i).paper_like_bimodal_retained;
    rejected_outer(i) = s(i).rejected_outer_steps;
    rejected_trials(i) = s(i).rejected_trial_count;
    alpha_min(i) = s(i).accepted_alpha_min;
    alpha_median(i) = s(i).accepted_alpha_median;
    topology{i} = s(i).final_topology_png;
end
T = table(tag, description, outer_iters, entry, exit_iter, final_omega1, ...
    final_omega2, final_N, final_volume, max_streak, paper_streak, retained20, ...
    rejected_outer, rejected_trials, alpha_min, alpha_median, topology);
end

function export_topology_png(rho, cfg, summary, path)
fig = figure('Visible', 'off', 'Color', 'w');
rho_img = reshape(rho, cfg.nely, cfg.nelx);
imagesc(1 - rho_img);
colormap(gray);
axis equal tight off;
title(sprintf('%s | omega=(%.1f, %.1f), N=%g', ...
    strrep(summary.tag, '_', '\_'), summary.final_omega1, ...
    summary.final_omega2, summary.final_N), 'Interpreter', 'tex');
exportgraphics(fig, path, 'Resolution', 180);
close(fig);
end

function write_report(path, summaries, base, basin_omega_min, basin_gap_tol, required_retention, ...
    paper_target_omega1, paper_close_rel_tol)
fid = fopen(path, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '# Globalization/Stabilization Experiment Report\n\n');
fprintf(fid, 'Generated: `%s`\n\n', datestr(now, 31));
fprintf(fid, '## Scope\n\n');
fprintf(fid, ['CC 40x5 Du & Olhoff benchmark. FE, interpolation, sensitivities, ', ...
    'filters, boundary conditions, and generalized gradients were not changed. ', ...
    'Only outer-update acceptance/globalization and post-coalescence optimizer ', ...
    'move limits were varied.\n\n']);
fprintf(fid, ['Base numerical layer: `move_lim=%.3g`, `outer_move=%.3g`, ', ...
    '`alpha=%.3g`, `inner_max_iter=%d`, low-mode MAC threshold `%.3g`, ', ...
    '`alpha_min=%.6g`.\n\n'], base.move_lim, base.outer_move, base.alpha, ...
    base.inner_max_iter, base.globalization_low_mode_mac_threshold, ...
    base.globalization_alpha_min);
fprintf(fid, ['Coalesced basin: `omega1 >= %.1f` and ', ...
    '`abs(omega2-omega1)/omega1 <= %.4g`.\n'], ...
    basin_omega_min, basin_gap_tol);
fprintf(fid, ['Paper-like guard: `abs(omega1 - %.1f)/%.1f <= %.3g` and ', ...
    '`abs(omega2-omega1)/omega1 <= %.4g`. Paper reproduction is not claimed ', ...
    'unless this stricter guard holds for `%d` consecutive outer iterations.\n\n'], ...
    paper_target_omega1, paper_target_omega1, paper_close_rel_tol, ...
    basin_gap_tol, required_retention);

fprintf(fid, '## Summary\n\n');
fprintf(fid, ['| variant | basin entry | basin exit | max basin streak | ', ...
    'paper-like streak | final omega1 | final omega2 | final N | rejected outer | rejected trials | ', ...
    'alpha min/median | retained paper-like? |\n']);
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n');
for i = 1:numel(summaries)
    s = summaries(i);
    fprintf(fid, ['| `%s` | %s | %s | %d | %d | %.4f | %.4f | %g | %d | %d | ', ...
        '%.4g / %.4g | %s |\n'], s.tag, fmt_iter(s.basin_entry_iter), ...
        fmt_iter(s.basin_exit_iter), s.max_basin_streak, ...
        s.max_paper_like_streak, s.final_omega1, s.final_omega2, ...
        s.final_N, s.rejected_outer_steps, ...
        s.rejected_trial_count, s.accepted_alpha_min, s.accepted_alpha_median, ...
        yesno(s.paper_like_bimodal_retained));
end

fprintf(fid, '\n## Topology Outputs\n\n');
for i = 1:numel(summaries)
    s = summaries(i);
    fprintf(fid, '- `%s`: `%s`\n', s.tag, s.final_topology_png);
end

fprintf(fid, '\n## Interpretation Guard\n\n');
if any([summaries.paper_like_bimodal_retained])
    kept = summaries([summaries.paper_like_bimodal_retained]);
    fprintf(fid, 'Paper-like bimodal retention was observed for: ');
    fprintf(fid, '`%s` ', kept.tag);
    fprintf(fid, '\n');
else
    fprintf(fid, ['No paper-reproduction claim is allowed: no variant stayed ', ...
        'inside the strict paper-like guard for `%d` consecutive outer iterations.\n'], ...
        required_retention);
end

fprintf(fid, '\n## Evidence Files\n\n');
fprintf(fid, '- `globalization_stabilization_results/globalization_summary.csv`\n');
fprintf(fid, '- `globalization_stabilization_results/<variant>/<variant>_iterations.csv`\n');
fprintf(fid, '- `globalization_stabilization_results/<variant>/<variant>_rho_final.csv`\n');
fprintf(fid, '- `globalization_stabilization_results/<variant>/<variant>_topology.png`\n');
fprintf(fid, '- `globalization_stabilization_results/<variant>/<variant>_result.mat`\n');
end

function [entry_iter, exit_iter] = entry_exit_iters(basin)
entry_iter = find(basin, 1, 'first');
if isempty(entry_iter)
    entry_iter = NaN;
    exit_iter = NaN;
    return
end
exit_rel = find(~basin(entry_iter+1:end), 1, 'first');
if isempty(exit_rel)
    exit_iter = NaN;
else
    exit_iter = entry_iter + exit_rel;
end
end

function [best_len, best_start, best_end] = max_true_streak(v)
best_len = 0; best_start = NaN; best_end = NaN;
cur_len = 0; cur_start = NaN;
for i = 1:numel(v)
    if v(i)
        if cur_len == 0, cur_start = i; end
        cur_len = cur_len + 1;
        if cur_len > best_len
            best_len = cur_len;
            best_start = cur_start;
            best_end = i;
        end
    else
        cur_len = 0;
    end
end
end

function s = fmt_iter(v)
if isempty(v) || ~isfinite(v)
    s = 'NA';
else
    s = sprintf('%d', v);
end
end

function s = yesno(v)
if v, s = 'yes'; else, s = 'no'; end
end

function s = factor_tag(x)
s = strrep(sprintf('%.3g', x), '.', 'p');
end
