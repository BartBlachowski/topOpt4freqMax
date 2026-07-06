% RUN_STABILIZATION_BASIN_RETENTION
%
% Targeted Du & Olhoff (2007) CC benchmark stabilization experiment.
%
% Scope: four controlled runs only:
%   1. baseline: current topopt_freq_exact behavior for the observed basin setup
%   2. persistent_mma: preserve MMA low/upp/xold history across outer loops
%   3. converged_inner: allow substantially more inner MMA iterations
%   4. combined: persistent MMA + more inner iterations
%
% The FE formulation, interpolation, sensitivities, supports, and objective are
% not changed by this runner.

clearvars;
close all;
clc;

this_dir = fileparts(mfilename('fullpath'));
repo_root = fullfile(this_dir, '..', '..', '..');
matlab_dir = fullfile(this_dir, '..', 'Matlab');
tools_dir = fullfile(repo_root, 'tools', 'Matlab');
addpath(matlab_dir);
addpath(tools_dir);

out_dir = fullfile(this_dir, 'stabilization_basin_retention_results');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

report_path = fullfile(this_dir, 'stabilization_basin_retention_report.md');
paper_target_omega1 = 456.4;
basin_omega_min = 450.0;
basin_gap_tol = 0.005;
retain_iters = 10;
paper_close_rel_tol = 0.02;
prior_basin_iter = 24;

base_cfg = struct();
base_cfg.support_type = 'CC';
base_cfg.nelx = 40;
base_cfg.nely = 5;
base_cfg.volfrac = 0.5;
base_cfg.mass_mode = 'du2007_c1';
base_cfg.sensitivity_filter = true;
base_cfg.rmin_elem = 2.5;
base_cfg.n_target = 1;
base_cfg.n_modes = 4;
base_cfg.mult_tol = 1e-3;
base_cfg.outer_max_iter = 120;
base_cfg.outer_tol = 1e-6;
base_cfg.inner_tol = 1e-4;
base_cfg.move_lim = 0.2;
base_cfg.outer_move = 0.2;
base_cfg.alpha = 0.5;
base_cfg.acceptance_check = false;
base_cfg.verbose = false;
base_cfg.rho_snapshot_interval = 1;

variants = {
    make_variant('baseline', ...
        'Current topopt_freq_exact behavior; MMA state resets every outer iteration.', ...
        false, 30);
    make_variant('persistent_mma', ...
        'Preserve MMA asymptote/history state across outer iterations.', ...
        true, 30);
    make_variant('converged_inner', ...
        'Reset MMA each outer iteration, but allow many more inner MMA iterations.', ...
        false, 300);
    make_variant('combined', ...
        'Persistent MMA state plus many more inner MMA iterations.', ...
        true, 300);
};

summaries = cell(numel(variants), 1);

fprintf('\n==========================================================\n');
fprintf(' Stabilization basin-retention experiment, CC benchmark\n');
fprintf(' Output: %s\n', out_dir);
fprintf('==========================================================\n\n');

for iv = 1:numel(variants)
    v = variants{iv};
    cfg = base_cfg;
    cfg.persistent_mma_state = v.persistent_mma_state;
    cfg.inner_max_iter = v.inner_max_iter;

    variant_dir = fullfile(out_dir, v.tag);
    if ~exist(variant_dir, 'dir'), mkdir(variant_dir); end

    fprintf('Running %-16s persistent=%d inner_max_iter=%d ...\n', ...
        v.tag, cfg.persistent_mma_state, cfg.inner_max_iter);
    elapsed_tic = tic;
    [rho_final, hist] = topopt_freq_exact(cfg);
    elapsed_s = toc(elapsed_tic);

    summary = analyze_variant(v, cfg, hist, rho_final, elapsed_s, ...
        paper_target_omega1, basin_omega_min, basin_gap_tol, retain_iters, ...
        paper_close_rel_tol, variant_dir);
    summaries{iv} = summary;

    per_iteration = build_per_iteration_table(hist, summary);
    writetable(per_iteration, fullfile(variant_dir, [v.tag '_per_iteration.csv']));
    write_rho_csv(hist, cfg, fullfile(variant_dir, [v.tag '_rho_by_iteration.csv']));
    save(fullfile(variant_dir, [v.tag '_result.mat']), ...
        'cfg', 'hist', 'rho_final', 'elapsed_s', 'summary', 'per_iteration');
    export_topology_images(hist, cfg, summary, variant_dir);

    fprintf(['  final omega1=%.4f omega2=%.4f N=%g gap=%.4g; ', ...
             'best iter=%d omega1=%.4f; entry=%s retained=%d\n'], ...
        summary.final_omega1, summary.final_omega2, summary.final_N, ...
        summary.final_freq_gap, summary.best_iter, summary.best_omega1, ...
        fmt_iter(summary.basin_entry_iter), summary.retained_basin);
end

write_report(report_path, summaries, base_cfg, paper_target_omega1, ...
    basin_omega_min, basin_gap_tol, retain_iters, paper_close_rel_tol, ...
    prior_basin_iter);

fprintf('\nReport written to %s\n', report_path);

function v = make_variant(tag, description, persistent_mma_state, inner_max_iter)
v = struct();
v.tag = tag;
v.description = description;
v.persistent_mma_state = persistent_mma_state;
v.inner_max_iter = inner_max_iter;
end

function summary = analyze_variant(v, cfg, hist, ~, elapsed_s, ...
    paper_target_omega1, basin_omega_min, basin_gap_tol, retain_iters, ...
    paper_close_rel_tol, variant_dir)

ni = hist.outer_iters;
iters = (1:ni)';
omega = hist.omega_trial(1:ni, :);
fallback = ~isfinite(omega(:,1));
omega(fallback, :) = hist.omega(fallback, :);
lambda = omega.^2;
N_freq = hist.N_trial(1:ni);
missing_N = ~isfinite(N_freq);
N_freq(missing_N) = hist.N(missing_N);
N_lambda = detect_N_from_lambda_rows(lambda, cfg.mult_tol);

freq_gap = abs(omega(:,2) - omega(:,1)) ./ max(omega(:,1), eps);
lambda_gap = abs(lambda(:,2) - lambda(:,1)) ./ max(lambda(:,1), eps);
basin = omega(:,1) >= basin_omega_min & freq_gap <= basin_gap_tol;

[entry_iter, exit_iter] = entry_exit_iters(basin);
[retained, retain_start, retain_end] = has_consecutive_true(basin, retain_iters);

volume = hist.volume(1:ni);
feasible = isfinite(omega(:,1)) & volume <= cfg.volfrac + 1e-6;
if any(feasible)
    feasible_idx = find(feasible);
    [~, local_best] = max(omega(feasible_idx, 1));
    best_iter = feasible_idx(local_best);
else
    [~, best_iter] = max(omega(:,1));
end

final_iter = ni;
entry_topology = topology_or_empty(hist, cfg, entry_iter);
best_topology = topology_or_empty(hist, cfg, best_iter);
final_topology = topology_or_empty(hist, cfg, final_iter);
retained_topology = topology_or_empty(hist, cfg, retain_end);

entry_conn = connectedness(entry_topology, cfg);
best_conn = connectedness(best_topology, cfg);
final_conn = connectedness(final_topology, cfg);
retained_conn = connectedness(retained_topology, cfg);

final_close = abs(omega(final_iter,1) - paper_target_omega1) / paper_target_omega1 <= paper_close_rel_tol;
best_close = abs(omega(best_iter,1) - paper_target_omega1) / paper_target_omega1 <= paper_close_rel_tol;
retained_close = retained && abs(omega(retain_end,1) - paper_target_omega1) / paper_target_omega1 <= paper_close_rel_tol;

final_bimodal = N_freq(final_iter) >= 2 || freq_gap(final_iter) <= cfg.mult_tol;
best_bimodal = N_freq(best_iter) >= 2 || freq_gap(best_iter) <= cfg.mult_tol;
retained_bimodal = retained && (N_freq(retain_end) >= 2 || freq_gap(retain_end) <= cfg.mult_tol);

summary = struct();
summary.tag = v.tag;
summary.description = v.description;
summary.elapsed_s = elapsed_s;
summary.outer_iters = ni;
summary.persistent_mma_state = cfg.persistent_mma_state;
summary.inner_max_iter = cfg.inner_max_iter;
summary.inner_converged_count = nnz(hist.inner_converged(1:ni));
summary.inner_max_iter_hit_count = nnz(hist.inner_hit_max_iter(1:ni));
summary.all_inner_converged = all(hist.inner_converged(1:ni));
summary.any_N2_frequency_tol = any(N_freq >= 2);
summary.any_N2_lambda_tol = any(N_lambda >= 2);
summary.first_N2_frequency_iter = first_or_nan(find(N_freq >= 2, 1));
summary.first_N2_lambda_iter = first_or_nan(find(N_lambda >= 2, 1));
summary.basin_entry_iter = entry_iter;
summary.basin_exit_iter = exit_iter;
summary.retained_basin = retained;
summary.retain_start_iter = retain_start;
summary.retain_end_iter = retain_end;
summary.basin_iter_count = nnz(basin);
summary.best_iter = best_iter;
summary.best_omega1 = omega(best_iter, 1);
summary.best_omega2 = omega(best_iter, 2);
summary.best_lambda1 = lambda(best_iter, 1);
summary.best_lambda2 = lambda(best_iter, 2);
summary.best_freq_gap = freq_gap(best_iter);
summary.best_lambda_gap = lambda_gap(best_iter);
summary.best_N = N_freq(best_iter);
summary.best_N_lambda = N_lambda(best_iter);
summary.best_volume = volume(best_iter);
summary.best_connected = best_conn.connected;
summary.best_support_connected = best_conn.support_connected;
summary.best_largest_component_fraction = best_conn.largest_component_fraction;
summary.final_iter = final_iter;
summary.final_omega1 = omega(final_iter, 1);
summary.final_omega2 = omega(final_iter, 2);
summary.final_lambda1 = lambda(final_iter, 1);
summary.final_lambda2 = lambda(final_iter, 2);
summary.final_freq_gap = freq_gap(final_iter);
summary.final_lambda_gap = lambda_gap(final_iter);
summary.final_N = N_freq(final_iter);
summary.final_N_lambda = N_lambda(final_iter);
summary.final_volume = volume(final_iter);
summary.final_connected = final_conn.connected;
summary.final_support_connected = final_conn.support_connected;
summary.final_largest_component_fraction = final_conn.largest_component_fraction;
summary.entry_connected = entry_conn.connected;
summary.entry_support_connected = entry_conn.support_connected;
summary.retained_connected = retained_conn.connected;
summary.retained_support_connected = retained_conn.support_connected;
summary.paper_claim_final_allowed = final_connected_enough(final_conn) && final_bimodal && final_close;
summary.paper_claim_best_allowed = final_connected_enough(best_conn) && best_bimodal && best_close;
summary.paper_claim_retained_allowed = retained && final_connected_enough(retained_conn) && retained_bimodal && retained_close;
summary.pre24_omega1 = omega(1:min(24, ni), 1);
summary.variant_dir = variant_dir;
summary.per_iteration = struct('iter', iters, 'omega', omega, 'lambda', lambda, ...
    'N_frequency', N_freq, 'N_lambda', N_lambda, 'freq_gap', freq_gap, ...
    'lambda_gap', lambda_gap, 'basin', basin);
end

function T = build_per_iteration_table(hist, summary)
ni = summary.outer_iters;
iter = (1:ni)';
omega = summary.per_iteration.omega;
lambda = summary.per_iteration.lambda;
T = table(iter, ...
    omega(:,1), omega(:,2), omega(:,3), omega(:,4), ...
    lambda(:,1), lambda(:,2), lambda(:,3), lambda(:,4), ...
    summary.per_iteration.N_frequency, summary.per_iteration.N_lambda, ...
    hist.N(1:ni), hist.beta(1:ni), hist.volume(1:ni), hist.drho_norm(1:ni), ...
    hist.inner_iters(1:ni), hist.inner_converged(1:ni), hist.inner_hit_max_iter(1:ni), ...
    summary.per_iteration.freq_gap, summary.per_iteration.lambda_gap, ...
    summary.per_iteration.basin, ...
    'VariableNames', {'iter', 'omega1', 'omega2', 'omega3', 'omega4', ...
    'lambda1', 'lambda2', 'lambda3', 'lambda4', 'N_frequency_post', ...
    'N_lambda_post', 'N_frequency_pre', 'beta', 'volume', 'drho_norm', ...
    'inner_iters', 'inner_converged', 'inner_hit_max_iter', ...
    'freq_relative_gap_1_2', 'lambda_relative_gap_1_2', 'near_paper_basin'});
end

function write_rho_csv(hist, cfg, path)
count = hist.rho_snapshot_count;
iters = hist.rho_snapshot_iters(1:count);
rhos = hist.rho_snapshots(:, 1:count)';
names = cell(1, cfg.nelx * cfg.nely + 1);
names{1} = 'iter';
for e = 1:(cfg.nelx * cfg.nely)
    names{e+1} = sprintf('rho_%03d', e);
end
T = array2table([iters, rhos], 'VariableNames', names);
writetable(T, path);
end

function export_topology_images(hist, cfg, summary, variant_dir)
rho_initial = cfg.volfrac * ones(cfg.nelx * cfg.nely, 1);
write_topology_png(rho_initial, cfg, fullfile(variant_dir, [summary.tag '_topology_initial.png']));
if isfinite(summary.basin_entry_iter)
    write_topology_png(rho_at_iter(hist, summary.basin_entry_iter), cfg, ...
        fullfile(variant_dir, [summary.tag '_topology_first_basin_entry.png']));
end
write_topology_png(rho_at_iter(hist, summary.best_iter), cfg, ...
    fullfile(variant_dir, [summary.tag '_topology_best_feasible.png']));
write_topology_png(rho_at_iter(hist, summary.final_iter), cfg, ...
    fullfile(variant_dir, [summary.tag '_topology_final.png']));
end

function write_topology_png(rho, cfg, path)
if isempty(rho), return; end
img = reshape(rho, cfg.nely, cfg.nelx);
img = uint8(round(255 * (1 - img)));
scale = max(1, floor(600 / max(size(img))));
img = kron(img, ones(scale, scale, 'uint8'));
imwrite(img, path);
end

function write_report(report_path, summaries, base_cfg, paper_target_omega1, ...
    basin_omega_min, basin_gap_tol, retain_iters, paper_close_rel_tol, prior_basin_iter)

fid = fopen(report_path, 'w');
if fid < 0
    error('run_stabilization_basin_retention:ReportOpenFailed', ...
        'Could not open report path: %s', report_path);
end
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, '# Stabilization Basin-Retention Report\n\n');
fprintf(fid, 'Generated: `%s`\n\n', char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd HH:mm:ss Z')));
fprintf(fid, '## Scope\n\n');
fprintf(fid, ['Targeted Du & Olhoff 2007 clamped-clamped benchmark test. ', ...
    'The FE formulation, mass interpolation, sensitivities, boundary conditions, ', ...
    'and objective are unchanged. The controlled numerical-layer settings are ', ...
    'MMA state persistence and inner MMA iteration budget only.\n\n']);
fprintf(fid, 'Base benchmark: `%dx%d`, `volfrac=%.3g`, `mass_mode=%s`, `rmin_elem=%.3g`, `mult_tol=%.1e`, `alpha=%.3g`, `move_lim=%.3g`, `outer_move=%.3g`.\n\n', ...
    base_cfg.nelx, base_cfg.nely, base_cfg.volfrac, base_cfg.mass_mode, ...
    base_cfg.rmin_elem, base_cfg.mult_tol, base_cfg.alpha, base_cfg.move_lim, base_cfg.outer_move);
fprintf(fid, 'Near-paper basin definition: `omega_1 >= %.1f` and `abs(omega_2 - omega_1)/omega_1 <= %.4g`. Retention requires `%d` consecutive outer iterations. Published CC target: `omega_1 = %.1f`, bimodal optimum.\n\n', ...
    basin_omega_min, basin_gap_tol, retain_iters, paper_target_omega1);

fprintf(fid, '## Variant Comparison\n\n');
fprintf(fid, '| Variant | Persistent MMA | Inner max | Outer iters | Inner converged | Inner cap hits | Basin entry | Basin exit | Retained 10? | Any N=2 freq tol | Any N=2 lambda tol | Best omega1 | Best gap f | Final omega1 | Final gap f | Final N | Final support-connected | Paper reproduction claim |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n');
for i = 1:numel(summaries)
    s = summaries{i};
    fprintf(fid, '| `%s` | %d | %d | %d | %d/%d | %d | %s | %s | %d | %d | %d | %.4f | %.4g | %.4f | %.4g | %.0f | %s | %s |\n', ...
        s.tag, s.persistent_mma_state, s.inner_max_iter, s.outer_iters, ...
        s.inner_converged_count, s.outer_iters, s.inner_max_iter_hit_count, ...
        fmt_iter(s.basin_entry_iter), fmt_iter(s.basin_exit_iter), ...
        s.retained_basin, s.any_N2_frequency_tol, s.any_N2_lambda_tol, ...
        s.best_omega1, s.best_freq_gap, s.final_omega1, s.final_freq_gap, ...
        s.final_N, yesno(s.final_support_connected), paper_claim_text(s));
end

fprintf(fid, '\n## Basin Entry And Exit\n\n');
fprintf(fid, '| Variant | Entry iter | Exit iter | Retain window | Basin iter count | N=2 freq first | N=2 lambda first |\n');
fprintf(fid, '|---|---:|---:|---|---:|---:|---:|\n');
for i = 1:numel(summaries)
    s = summaries{i};
    if s.retained_basin
        retain_txt = sprintf('%d-%d', s.retain_start_iter, s.retain_end_iter);
    else
        retain_txt = 'none';
    end
    fprintf(fid, '| `%s` | %s | %s | %s | %d | %s | %s |\n', ...
        s.tag, fmt_iter(s.basin_entry_iter), fmt_iter(s.basin_exit_iter), ...
        retain_txt, s.basin_iter_count, fmt_iter(s.first_N2_frequency_iter), ...
        fmt_iter(s.first_N2_lambda_iter));
end

fprintf(fid, '\n## Best Snapshot Vs Final Snapshot\n\n');
fprintf(fid, '| Variant | Best iter | Best omega1 | Best omega2 | Best freq gap | Best lambda gap | Best N | Best support-connected | Final iter | Final omega1 | Final omega2 | Final freq gap | Final lambda gap | Final N | Final support-connected |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|\n');
for i = 1:numel(summaries)
    s = summaries{i};
    fprintf(fid, '| `%s` | %d | %.4f | %.4f | %.4g | %.4g | %.0f | %s | %d | %.4f | %.4f | %.4g | %.4g | %.0f | %s |\n', ...
        s.tag, s.best_iter, s.best_omega1, s.best_omega2, s.best_freq_gap, ...
        s.best_lambda_gap, s.best_N, yesno(s.best_support_connected), ...
        s.final_iter, s.final_omega1, s.final_omega2, s.final_freq_gap, ...
        s.final_lambda_gap, s.final_N, yesno(s.final_support_connected));
end

fprintf(fid, '\n## Transition Notes\n\n');
for i = 1:numel(summaries)
    s = summaries{i};
    fprintf(fid, '- `%s`: %s\n', s.tag, transition_note(s));
end

fprintf(fid, '\n## Transition Detail\n\n');
fprintf(fid, '| Variant | Entry iter | Entry omega1 | Entry omega2 | Entry freq gap | Exit iter | Exit omega1 | Exit omega2 | Exit freq gap | Exit reason |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n');
for i = 1:numel(summaries)
    s = summaries{i};
    fprintf(fid, '%s\n', transition_detail_row(s, basin_omega_min, basin_gap_tol));
end

fprintf(fid, '\n## Attribution\n\n');
fprintf(fid, '%s\n\n', attribution_text(summaries, prior_basin_iter));

fprintf(fid, '## Evidence Files\n\n');
for i = 1:numel(summaries)
    s = summaries{i};
    fprintf(fid, '- `%s`: `%s`\n', s.tag, s.variant_dir);
end
fprintf(fid, '\nEach variant directory contains the MAT result, per-iteration CSV, full rho-by-iteration CSV, and topology PNGs for the initial, first basin-entry when present, best feasible, and final designs.\n\n');

fprintf(fid, '## Reproduction Guard\n\n');
fprintf(fid, 'No paper-reproduction claim is allowed unless a final or retained design is support-connected, bimodal under the declared tolerance, and within %.1f%% of `omega_1 = %.1f`. The table above reports this as a separate guard rather than assuming it from basin entry alone.\n', ...
    100 * paper_close_rel_tol, paper_target_omega1);
end

function txt = attribution_text(summaries, prior_basin_iter)
baseline = summaries{1};
any_retained = false;
for i = 1:numel(summaries)
    any_retained = any_retained || summaries{i}.retained_basin;
end
if any_retained
    retained_tags = {};
    for i = 1:numel(summaries)
        if summaries{i}.retained_basin
            retained_tags{end+1} = summaries{i}.tag; %#ok<AGROW>
        end
    end
    txt = sprintf('At least one variant retained the near-paper basin for 10 consecutive iterations (`%s`). Under the requested acceptance logic, the earlier failure is attributable to path-control/MMA-state related numerical-layer behavior. Compare which retained variants use persistent MMA and/or a larger inner budget to separate restart from inner-convergence effects.', strjoin(retained_tags, '`, `'));
    return;
end

any_entry = false;
for i = 1:numel(summaries)
    any_entry = any_entry || isfinite(summaries{i}.basin_entry_iter);
end
if any_entry
    txt = 'At least one variant enters the near-paper basin but no variant retains it for 10 consecutive iterations. Under the requested acceptance logic, this remains unresolved path instability; the transition-detail table identifies the exit iteration and failed basin condition for each entry.';
    return;
end

parts = {};
for i = 2:numel(summaries)
    s = summaries{i};
    n = min([numel(baseline.pre24_omega1), numel(s.pre24_omega1), prior_basin_iter]);
    if n > 0
        delta = max(abs(s.pre24_omega1(1:n) - baseline.pre24_omega1(1:n)));
        parts{end+1} = sprintf('`%s` max |Delta omega1| through iter %d = %.4g', s.tag, n, delta); %#ok<AGROW>
    end
end
txt = sprintf('No variant enters the near-paper basin. The pre-%d trajectory comparison is: %s. This means the tested numerical-layer changes altered or failed to recover the previously observed basin before it could be retained.', ...
    prior_basin_iter, strjoin(parts, '; '));
end

function txt = transition_note(s)
if ~isfinite(s.basin_entry_iter)
    txt = 'never enters the near-paper basin.';
elseif s.retained_basin
    txt = sprintf('enters at iteration %d and retains the basin through at least iterations %d-%d.', ...
        s.basin_entry_iter, s.retain_start_iter, s.retain_end_iter);
elseif isfinite(s.basin_exit_iter)
    txt = sprintf('enters at iteration %d but exits at iteration %d; inspect that transition.', ...
        s.basin_entry_iter, s.basin_exit_iter);
else
    txt = sprintf('enters at iteration %d but the run ends before a 10-iteration retention window is proven.', ...
        s.basin_entry_iter);
end
end

function row = transition_detail_row(s, basin_omega_min, basin_gap_tol)
if ~isfinite(s.basin_entry_iter)
    row = sprintf('| `%s` | NA | NA | NA | NA | NA | NA | NA | NA | no basin entry |', s.tag);
    return;
end

entry = s.basin_entry_iter;
omega = s.per_iteration.omega;
gap = s.per_iteration.freq_gap;
entry_o1 = omega(entry, 1);
entry_o2 = omega(entry, 2);
entry_gap = gap(entry);

if isfinite(s.basin_exit_iter)
    exit_it = s.basin_exit_iter;
    exit_o1 = omega(exit_it, 1);
    exit_o2 = omega(exit_it, 2);
    exit_gap = gap(exit_it);
    reasons = {};
    if exit_o1 < basin_omega_min
        reasons{end+1} = sprintf('omega1 %.4f < %.1f', exit_o1, basin_omega_min);
    end
    if exit_gap > basin_gap_tol
        reasons{end+1} = sprintf('gap %.4g > %.4g', exit_gap, basin_gap_tol);
    end
    if isempty(reasons), reasons = {'left basin by definition'}; end
    reason = strjoin(reasons, '; ');
    row = sprintf('| `%s` | %d | %.4f | %.4f | %.4g | %d | %.4f | %.4f | %.4g | %s |', ...
        s.tag, entry, entry_o1, entry_o2, entry_gap, exit_it, exit_o1, exit_o2, exit_gap, reason);
else
    row = sprintf('| `%s` | %d | %.4f | %.4f | %.4g | NA | NA | NA | NA | no exit before run ended |', ...
        s.tag, entry, entry_o1, entry_o2, entry_gap);
end
end

function txt = paper_claim_text(s)
if s.paper_claim_final_allowed || s.paper_claim_best_allowed || s.paper_claim_retained_allowed
    txt = 'allowed';
else
    txt = 'not allowed';
end
end

function b = final_connected_enough(c)
b = c.support_connected && c.largest_component_fraction >= 0.95;
end

function [entry_iter, exit_iter] = entry_exit_iters(mask)
idx = find(mask, 1);
if isempty(idx)
    entry_iter = NaN;
    exit_iter = NaN;
    return;
end
entry_iter = idx;
exit_local = find(~mask(idx:end), 1);
if isempty(exit_local)
    exit_iter = NaN;
else
    exit_iter = idx + exit_local - 1;
end
end

function [has_run, start_iter, end_iter] = has_consecutive_true(mask, n)
has_run = false;
start_iter = NaN;
end_iter = NaN;
if isempty(mask), return; end
run_len = 0;
for i = 1:numel(mask)
    if mask(i)
        run_len = run_len + 1;
        if run_len >= n
            has_run = true;
            end_iter = i;
            start_iter = i - n + 1;
            return;
        end
    else
        run_len = 0;
    end
end
end

function N_lambda = detect_N_from_lambda_rows(lambda, tol)
N_lambda = ones(size(lambda, 1), 1);
for i = 1:size(lambda, 1)
    row = lambda(i, :);
    if ~isfinite(row(1)), N_lambda(i) = NaN; continue; end
    n = 1;
    ref = max(row(1), eps);
    for j = 2:numel(row)
        if isfinite(row(j)) && abs(row(j) - row(1)) / ref <= tol
            n = n + 1;
        else
            break;
        end
    end
    N_lambda(i) = n;
end
end

function rho = topology_or_empty(hist, cfg, iter)
if ~isfinite(iter)
    rho = [];
else
    rho = rho_at_iter(hist, iter);
    if isempty(rho)
        rho = cfg.volfrac * ones(cfg.nelx * cfg.nely, 1);
    end
end
end

function rho = rho_at_iter(hist, iter)
rho = [];
if ~isfield(hist, 'rho_snapshot_iters') || ~isfield(hist, 'rho_snapshots')
    return;
end
idx = find(hist.rho_snapshot_iters(1:hist.rho_snapshot_count) == iter, 1);
if ~isempty(idx)
    rho = hist.rho_snapshots(:, idx);
end
end

function c = connectedness(rho, cfg)
c = struct('connected', false, 'support_connected', false, ...
    'largest_component_fraction', NaN, 'component_count', 0);
if isempty(rho), return; end
solid = reshape(rho, cfg.nely, cfg.nelx) >= 0.5;
visited = false(size(solid));
total_solid = nnz(solid);
if total_solid == 0, return; end

largest = 0;
largest_touches_left = false;
largest_touches_right = false;
component_count = 0;
for r = 1:size(solid, 1)
    for col = 1:size(solid, 2)
        if solid(r, col) && ~visited(r, col)
            component_count = component_count + 1;
            [count, touches_left, touches_right, visited] = flood_component(solid, visited, r, col);
            if count > largest
                largest = count;
                largest_touches_left = touches_left;
                largest_touches_right = touches_right;
            end
        end
    end
end
c.component_count = component_count;
c.largest_component_fraction = largest / total_solid;
c.connected = component_count == 1;
c.support_connected = largest_touches_left && largest_touches_right && c.largest_component_fraction >= 0.95;
end

function [count, touches_left, touches_right, visited] = flood_component(solid, visited, r0, c0)
nr = size(solid, 1);
nc = size(solid, 2);
queue = zeros(numel(solid), 2);
head = 1;
tail = 1;
queue(tail, :) = [r0, c0];
visited(r0, c0) = true;
count = 0;
touches_left = false;
touches_right = false;
while head <= tail
    r = queue(head, 1);
    c = queue(head, 2);
    head = head + 1;
    count = count + 1;
    touches_left = touches_left || c == 1;
    touches_right = touches_right || c == nc;
    neigh = [r-1 c; r+1 c; r c-1; r c+1];
    for k = 1:4
        rr = neigh(k, 1);
        cc = neigh(k, 2);
        if rr >= 1 && rr <= nr && cc >= 1 && cc <= nc && solid(rr, cc) && ~visited(rr, cc)
            tail = tail + 1;
            queue(tail, :) = [rr, cc];
            visited(rr, cc) = true;
        end
    end
end
end

function x = first_or_nan(x)
if isempty(x), x = NaN; end
end

function s = fmt_iter(x)
if isfinite(x)
    s = sprintf('%d', x);
else
    s = 'NA';
end
end

function s = yesno(v)
if v
    s = 'yes';
else
    s = 'no';
end
end
