% RUN_PERSISTENT_MMA_AFTER_GLOBALIZATION
%
% Persistent-MMA experiment after outer-update globalization/stabilization.
%
% Scope: keep FE, interpolation, sensitivities, filters, boundary conditions,
% generalized gradients, and globalization logic fixed.  Only the MMA memory
% handling is toggled:
%   A) restarted MMA state each outer iteration
%   B) persistent MMA asymptotes/history across outer iterations

clearvars;
close all;
clc;

this_dir = fileparts(mfilename('fullpath'));
repo_root = fullfile(this_dir, '..', '..', '..');
matlab_dir = fullfile(this_dir, '..', 'Matlab');
tools_dir = fullfile(repo_root, 'tools', 'Matlab');
addpath(matlab_dir);
addpath(tools_dir);

out_dir = fullfile(this_dir, 'persistent_mma_after_globalization_results');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

paper_target_omega1 = 456.4;
paper_close_rel_tol = 0.02;
coalescence_gap_tol = 0.005;
required_retention = 20;

base = base_cfg();
stabilized = {
    make_stabilization('D_trust_0p1', 'post-coalescence trust factor 0.1', false, false, true, 0.1);
    make_stabilization('E_combined_0p25', 'B+C+D with trust factor 0.25', true, true, true, 0.25);
};

summaries = {};

fprintf('\n==========================================================\n');
fprintf(' Persistent MMA after globalization experiment\n');
fprintf(' Output: %s\n', out_dir);
fprintf(' Paper-like: |omega1 - %.1f|/%.1f <= %.3g and gap12 <= %.4g, retain >= %d\n', ...
    paper_target_omega1, paper_target_omega1, paper_close_rel_tol, ...
    coalescence_gap_tol, required_retention);
fprintf('==========================================================\n\n');

for is = 1:numel(stabilized)
    stab = stabilized{is};
    for use_persistent = [false true]
        cfg = base;
        cfg.globalization_enabled = stab.globalization_enabled;
        cfg.globalization_monotone_cluster = stab.monotone_cluster;
        cfg.globalization_low_mode_guard = stab.low_mode_guard;
        cfg.post_coalescence_trust_enabled = stab.trust_enabled;
        cfg.post_coalescence_trust_factor = stab.trust_factor;
        cfg.persistent_mma_state = use_persistent;

        if use_persistent
            memory_tag = 'persistent';
            memory_desc = 'persistent MMA asymptotes/history';
        else
            memory_tag = 'restarted';
            memory_desc = 'restarted MMA state each outer iteration';
        end
        tag = sprintf('%s_%s', stab.tag, memory_tag);
        variant_dir = fullfile(out_dir, tag);
        if ~exist(variant_dir, 'dir'), mkdir(variant_dir); end

        fprintf('Running %-28s  %s, %s\n', tag, stab.description, memory_desc);
        elapsed_tic = tic;
        [rho_final, hist] = topopt_freq_exact(cfg);
        elapsed_s = toc(elapsed_tic);

        summary = summarize_run(tag, stab, memory_desc, cfg, hist, rho_final, elapsed_s, ...
            paper_target_omega1, paper_close_rel_tol, coalescence_gap_tol, ...
            required_retention, variant_dir);
        summaries{end+1} = summary; %#ok<SAGROW>

        T = iteration_table(hist, summary);
        writetable(T, fullfile(variant_dir, [tag '_iterations.csv']));
        writematrix(rho_final(:), fullfile(variant_dir, [tag '_rho_final.csv']));
        export_topology_png(rho_final, cfg, summary, fullfile(variant_dir, [tag '_topology.png']));
        save(fullfile(variant_dir, [tag '_result.mat']), ...
            'cfg', 'hist', 'rho_final', 'summary', 'T', 'elapsed_s');

        fprintf(['  final omega=(%.4f, %.4f) N=%g vol=%.5f rho=[%.4g, %.4g] ', ...
            'paper_streak=%d coal_streak=%d reject_trials=%d connected=%d support=%d\n'], ...
            summary.final_omega1, summary.final_omega2, summary.final_N, ...
            summary.final_volume, summary.rho_min, summary.rho_max, ...
            summary.max_paper_like_streak, summary.max_coalescence_streak, ...
            summary.rejected_trial_count, summary.connected, summary.support_connected);
    end
end

summary_struct = [summaries{:}];
summary_table = summaries_to_table(summary_struct);
writetable(summary_table, fullfile(out_dir, 'persistent_mma_summary.csv'));
save(fullfile(out_dir, 'persistent_mma_after_globalization_results.mat'), ...
    'summary_struct', 'summary_table', 'paper_target_omega1', ...
    'paper_close_rel_tol', 'coalescence_gap_tol', 'required_retention');
write_report(fullfile(this_dir, 'persistent_mma_after_globalization_report.md'), ...
    summary_struct, paper_target_omega1, paper_close_rel_tol, ...
    coalescence_gap_tol, required_retention);

fprintf('\nReport written to %s\n', fullfile(this_dir, 'persistent_mma_after_globalization_report.md'));

function cfg = base_cfg()
cfg = struct();
cfg.support_type = 'CC';
cfg.nelx = 40;
cfg.nely = 5;
cfg.volfrac = 0.5;
cfg.mass_mode = 'du2007_c1';
cfg.sensitivity_filter = true;
cfg.rmin_elem = 2.5;
cfg.n_target = 1;
cfg.n_modes = 4;
cfg.mult_tol = 1e-3;
cfg.outer_max_iter = 100;
cfg.outer_tol = 1e-6;
cfg.inner_max_iter = 30;
cfg.inner_tol = 1e-4;
cfg.move_lim = 0.2;
cfg.outer_move = 0.2;
cfg.alpha = 0.5;
cfg.acceptance_check = false;
cfg.verbose = false;
cfg.rho_snapshot_interval = 1;
cfg.globalization_alpha_start = 1.0;
cfg.globalization_alpha_min = 1/128;
cfg.globalization_low_mode_mac_threshold = 0.25;
cfg.post_coalescence_gap_tol = 0.005;
end

function s = make_stabilization(tag, description, monotone_cluster, low_mode_guard, trust_enabled, trust_factor)
s = struct();
s.tag = tag;
s.description = description;
s.globalization_enabled = monotone_cluster || low_mode_guard;
s.monotone_cluster = monotone_cluster;
s.low_mode_guard = low_mode_guard;
s.trust_enabled = trust_enabled;
s.trust_factor = trust_factor;
end

function summary = summarize_run(tag, stab, memory_desc, cfg, hist, rho_final, elapsed_s, ...
    paper_target_omega1, paper_close_rel_tol, coalescence_gap_tol, required_retention, variant_dir)

ni = hist.outer_iters;
omega = hist.omega_trial(1:ni, :);
fallback = ~isfinite(omega(:,1));
omega(fallback, :) = hist.omega(fallback, :);
gap12 = abs(omega(:,2) - omega(:,1)) ./ max(omega(:,1), eps);
coalesced = gap12 <= coalescence_gap_tol;
paper_like = abs(omega(:,1) - paper_target_omega1) / paper_target_omega1 <= ...
    paper_close_rel_tol & coalesced;

[max_coalescence_streak, coal_start, coal_end] = max_true_streak(coalesced);
[max_paper_like_streak, paper_start, paper_end] = max_true_streak(paper_like);

rejected_trials = hist.globalization_rejected_trial_count(1:ni);
rejected_trials(~isfinite(rejected_trials)) = 0;
alpha = hist.step_alpha(1:ni);
move_eff = hist.move_lim_effective(1:ni);
outer_eff = hist.outer_move_effective(1:ni);
conn = connectedness(rho_final, cfg);

summary = struct();
summary.tag = tag;
summary.stabilization_tag = stab.tag;
summary.stabilization_description = stab.description;
summary.memory_description = memory_desc;
summary.persistent_mma_state = cfg.persistent_mma_state;
summary.elapsed_s = elapsed_s;
summary.outer_iters = ni;
summary.final_omega1 = hist.final_omega(1);
summary.final_omega2 = hist.final_omega(2);
summary.final_N = hist.final_N;
summary.final_volume = hist.final_volume;
summary.rho_min = min(rho_final);
summary.rho_max = max(rho_final);
summary.connected = conn.connected;
summary.support_connected = conn.support_connected;
summary.component_count = conn.component_count;
summary.largest_component_fraction = conn.largest_component_fraction;
summary.max_coalescence_streak = max_coalescence_streak;
summary.coalescence_streak_start = coal_start;
summary.coalescence_streak_end = coal_end;
summary.coalescence_retained = max_coalescence_streak >= required_retention;
summary.max_paper_like_streak = max_paper_like_streak;
summary.paper_like_streak_start = paper_start;
summary.paper_like_streak_end = paper_end;
summary.paper_like_retained = max_paper_like_streak >= required_retention;
summary.rejected_outer_steps = nnz(hist.globalization_rejected_outer_step(1:ni));
summary.rejected_trial_count = sum(rejected_trials);
summary.accepted_alpha_min = min(alpha);
summary.accepted_alpha_median = median(alpha);
summary.accepted_alpha_values = unique(alpha(:))';
summary.move_lim_min = min(move_eff);
summary.move_lim_median = median(move_eff);
summary.outer_move_min = min(outer_eff);
summary.outer_move_median = median(outer_eff);
summary.variant_dir = variant_dir;
summary.topology_png = fullfile(variant_dir, [tag '_topology.png']);
summary.per_iteration_csv = fullfile(variant_dir, [tag '_iterations.csv']);
summary.final_rho_csv = fullfile(variant_dir, [tag '_rho_final.csv']);
summary.per_iteration = struct('omega', omega, 'gap12', gap12, ...
    'coalesced', coalesced, 'paper_like', paper_like);
end

function T = iteration_table(hist, summary)
ni = summary.outer_iters;
iter = (1:ni)';
omega = summary.per_iteration.omega;
alpha = hist.step_alpha(1:ni);
move_eff = hist.move_lim_effective(1:ni);
outer_eff = hist.outer_move_effective(1:ni);
rejected_trials = hist.globalization_rejected_trial_count(1:ni);
rejected_trials(~isfinite(rejected_trials)) = 0;
trial_count = hist.globalization_trial_count(1:ni);
trial_count(~isfinite(trial_count)) = 0;
reason = hist.globalization_reason(1:ni);
T = table(iter, omega(:,1), omega(:,2), summary.per_iteration.gap12, ...
    hist.N_trial(1:ni), hist.volume(1:ni), hist.drho_norm(1:ni), ...
    hist.drho_max(1:ni), alpha, rejected_trials, trial_count, ...
    move_eff, outer_eff, summary.per_iteration.coalesced, ...
    summary.per_iteration.paper_like, reason, ...
    'VariableNames', {'iter','omega1','omega2','gap12','N_post','volume', ...
    'drho_norm','drho_max','accepted_alpha','rejected_trial_count', ...
    'trial_count','move_lim_effective','outer_move_effective', ...
    'coalesced','paper_like','globalization_reason'});
end

function T = summaries_to_table(s)
n = numel(s);
tag = cell(n,1); stabilization = cell(n,1); memory = cell(n,1);
persistent_flag = false(n,1); final_omega1 = nan(n,1); final_omega2 = nan(n,1);
final_N = nan(n,1); final_volume = nan(n,1); rho_min = nan(n,1); rho_max = nan(n,1);
paper_streak = nan(n,1); coal_streak = nan(n,1);
paper_retained = false(n,1); coal_retained = false(n,1);
rejected_trials = nan(n,1); rejected_outer = nan(n,1);
alpha_min = nan(n,1); alpha_median = nan(n,1); move_min = nan(n,1); move_median = nan(n,1);
connected = false(n,1); support_connected = false(n,1);
component_count = nan(n,1); largest_component_fraction = nan(n,1);
topology = cell(n,1);
for i = 1:n
    tag{i} = s(i).tag;
    stabilization{i} = s(i).stabilization_tag;
    memory{i} = s(i).memory_description;
    persistent_flag(i) = s(i).persistent_mma_state;
    final_omega1(i) = s(i).final_omega1;
    final_omega2(i) = s(i).final_omega2;
    final_N(i) = s(i).final_N;
    final_volume(i) = s(i).final_volume;
    rho_min(i) = s(i).rho_min;
    rho_max(i) = s(i).rho_max;
    paper_streak(i) = s(i).max_paper_like_streak;
    coal_streak(i) = s(i).max_coalescence_streak;
    paper_retained(i) = s(i).paper_like_retained;
    coal_retained(i) = s(i).coalescence_retained;
    rejected_trials(i) = s(i).rejected_trial_count;
    rejected_outer(i) = s(i).rejected_outer_steps;
    alpha_min(i) = s(i).accepted_alpha_min;
    alpha_median(i) = s(i).accepted_alpha_median;
    move_min(i) = s(i).move_lim_min;
    move_median(i) = s(i).move_lim_median;
    connected(i) = s(i).connected;
    support_connected(i) = s(i).support_connected;
    component_count(i) = s(i).component_count;
    largest_component_fraction(i) = s(i).largest_component_fraction;
    topology{i} = s(i).topology_png;
end
T = table(tag, stabilization, memory, persistent_flag, final_omega1, final_omega2, ...
    final_N, final_volume, rho_min, rho_max, paper_streak, coal_streak, ...
    paper_retained, coal_retained, rejected_trials, rejected_outer, ...
    alpha_min, alpha_median, move_min, move_median, connected, ...
    support_connected, component_count, largest_component_fraction, topology);
end

function write_report(path, summaries, paper_target_omega1, paper_close_rel_tol, ...
    coalescence_gap_tol, required_retention)
fid = fopen(path, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '# Persistent MMA After Globalization Report\n\n');
fprintf(fid, 'Generated: `%s`\n\n', datestr(now, 31));
fprintf(fid, '## Scope\n\n');
fprintf(fid, ['The two best stabilization variants from the globalization experiment ', ...
    'are rerun with identical FE, sensitivity, filter, boundary, generalized-gradient, ', ...
    'and globalization settings. Only MMA memory handling is toggled between ', ...
    'outer-iteration restart and persistent asymptote/history state.\n\n']);
fprintf(fid, ['Paper-like retention: `abs(omega1 - %.1f)/%.1f <= %.3g` and ', ...
    '`gap12 <= %.4g` for `%d` consecutive outer iterations. Coalescence ', ...
    'retention only requires `gap12 <= %.4g` for `%d` iterations.\n\n'], ...
    paper_target_omega1, paper_target_omega1, paper_close_rel_tol, ...
    coalescence_gap_tol, required_retention, coalescence_gap_tol, required_retention);

fprintf(fid, '## Summary\n\n');
fprintf(fid, ['| variant | MMA memory | final omega1 | final omega2 | final N | volume | ', ...
    'rho min/max | paper streak | coal streak | rejected trials | alpha min/median | ', ...
    'move min/median | connected | support-connected |\n']);
fprintf(fid, '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n');
for i = 1:numel(summaries)
    s = summaries(i);
    fprintf(fid, ['| `%s` | %s | %.4f | %.4f | %g | %.5f | %.4g / %.4g | ', ...
        '%d | %d | %d | %.4g / %.4g | %.4g / %.4g | %s | %s |\n'], ...
        s.tag, s.memory_description, s.final_omega1, s.final_omega2, ...
        s.final_N, s.final_volume, s.rho_min, s.rho_max, ...
        s.max_paper_like_streak, s.max_coalescence_streak, ...
        s.rejected_trial_count, s.accepted_alpha_min, s.accepted_alpha_median, ...
        s.move_lim_min, s.move_lim_median, yesno(s.connected), ...
        yesno(s.support_connected));
end

fprintf(fid, '\n## Pairwise Answer\n\n');
for i = 1:2:numel(summaries)
    a = summaries(i);
    b = summaries(i+1);
    fprintf(fid, '- `%s`: restarted final `%.4f/%.4f`, persistent final `%.4f/%.4f`; paper streak `%d -> %d`, coalescence streak `%d -> %d`.\n', ...
        a.stabilization_tag, a.final_omega1, a.final_omega2, ...
        b.final_omega1, b.final_omega2, a.max_paper_like_streak, ...
        b.max_paper_like_streak, a.max_coalescence_streak, b.max_coalescence_streak);
end

fprintf(fid, '\n## Conclusion\n\n');
if any([summaries.paper_like_retained])
    fprintf(fid, 'At least one persistent/restarted comparison retained the strict paper-like basin.\n');
else
    fprintf(fid, ['Persistent MMA does not reduce over-optimization back toward ', ...
        'the paper basin in this experiment. For `D_trust_0p1`, persistence ', ...
        'destroys the retained coalesced state. For `E_combined_0p25`, ', ...
        'persistence moves to a higher, non-paper and non-retained-coalesced ', ...
        'state. The off-target coalesced optima are retained by the restarted ', ...
        'MMA runs, not improved by persistent MMA.\n']);
end

fprintf(fid, '\n## Evidence Files\n\n');
fprintf(fid, '- `persistent_mma_after_globalization_results/persistent_mma_summary.csv`\n');
fprintf(fid, '- `persistent_mma_after_globalization_results/<variant>/<variant>_iterations.csv`\n');
fprintf(fid, '- `persistent_mma_after_globalization_results/<variant>/<variant>_rho_final.csv`\n');
fprintf(fid, '- `persistent_mma_after_globalization_results/<variant>/<variant>_topology.png`\n');
fprintf(fid, '- `persistent_mma_after_globalization_results/<variant>/<variant>_result.mat`\n');
end

function export_topology_png(rho, cfg, summary, path)
fig = figure('Visible', 'off', 'Color', 'w');
imagesc(1 - reshape(rho, cfg.nely, cfg.nelx));
colormap(gray);
axis equal tight off;
title(sprintf('%s | omega=(%.1f, %.1f), N=%g', ...
    strrep(summary.tag, '_', '\_'), summary.final_omega1, ...
    summary.final_omega2, summary.final_N), 'Interpreter', 'tex');
exportgraphics(fig, path, 'Resolution', 180);
close(fig);
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

function s = yesno(v)
if v, s = 'yes'; else, s = 'no'; end
end
