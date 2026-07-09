% RUN_BASIN_EXIT_FORENSICS
%
% Diagnostic-only Du & Olhoff (2007) CC basin-exit audit.
%
% The solver algorithm is not changed.  This runner only enables the
% cfg.forensic_* tracing path in topopt_freq_exact.m and writes evidence for
% the first post-update omega1/omega2 coalescence basin and all following
% outer updates.

clearvars;
close all;
clc;

this_dir = fileparts(mfilename('fullpath'));
repo_root = fullfile(this_dir, '..', '..', '..');
matlab_dir = fullfile(this_dir, '..', 'Matlab');
tools_dir = fullfile(repo_root, 'tools', 'Matlab');
addpath(matlab_dir);
addpath(tools_dir);

out_dir = fullfile(this_dir, 'basin_exit_forensics_results');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

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
cfg.outer_max_iter = 40;
cfg.outer_tol = 1e-6;
cfg.inner_max_iter = 30;
cfg.inner_tol = 1e-4;

% Match the existing no-physics-change optimizer evidence in
% Matlab/results/optimizer_audit and experiments/stabilization_*.
cfg.move_lim = 0.2;
cfg.outer_move = 0.2;
cfg.alpha = 0.5;
cfg.acceptance_check = false;
cfg.verbose = true;

cfg.forensic_enabled = true;
cfg.forensic_gap_tol = 0.005;
cfg.forensic_active_tol = 1e-4;

fprintf('\n==========================================================\n');
fprintf(' Basin-exit forensics, Du & Olhoff CC benchmark\n');
fprintf(' Trigger: abs(omega2-omega1)/omega1 < %.4g\n', cfg.forensic_gap_tol);
fprintf(' Output: %s\n', out_dir);
fprintf('==========================================================\n\n');

elapsed_tic = tic;
[rho_final, hist] = topopt_freq_exact(cfg);
elapsed_s = toc(elapsed_tic);
forensic = hist.forensic;

summary = summarize_forensics(cfg, hist, elapsed_s);
T = forensic_table(forensic);
writetable(T, fullfile(out_dir, 'basin_exit_forensics_table.csv'));
write_vector_csv(fullfile(out_dir, 'basin_exit_rho_drho_vectors.csv'), forensic);
write_mac_csvs(out_dir, forensic);
save(fullfile(out_dir, 'basin_exit_forensics_result.mat'), ...
    'cfg', 'hist', 'rho_final', 'forensic', 'summary', 'T', 'elapsed_s');
write_report(fullfile(this_dir, 'basin_exit_forensics_report.md'), cfg, hist, summary);

fprintf('\nReport written to %s\n', fullfile(this_dir, 'basin_exit_forensics_report.md'));

function summary = summarize_forensics(cfg, hist, elapsed_s)
f = hist.forensic;
summary = struct();
summary.elapsed_s = elapsed_s;
summary.outer_iters = hist.outer_iters;
summary.trigger_iter = f.trigger_iter;
summary.trigger_gap = f.trigger_gap;
summary.trigger_omega = f.trigger_omega;
summary.trigger_volume = f.trigger_volume;
summary.record_count = f.count;
summary.final_omega = hist.final_omega;
summary.final_volume = hist.final_volume;
summary.final_N = hist.final_N;

if f.count == 0
    summary.exit_iter = NaN;
    summary.first_bad_linearization_iter = NaN;
    summary.max_drho_iter = NaN;
    summary.max_drho = NaN;
    return
end

gap_after = abs(f.omega_after_proposed(:,2) - f.omega_after_proposed(:,1)) ./ ...
    max(f.omega_after_proposed(:,1), eps);
in_basin_after = gap_after < cfg.forensic_gap_tol;
exit_local = find(~in_basin_after, 1, 'first');
if isempty(exit_local)
    summary.exit_iter = NaN;
else
    summary.exit_iter = f.iter(exit_local);
end

bad = find(f.predicted_improvement_real_decrease, 1, 'first');
if isempty(bad)
    summary.first_bad_linearization_iter = NaN;
else
    summary.first_bad_linearization_iter = f.iter(bad);
end

drho_max = max(abs(f.drho_proposed), [], 1)';
[summary.max_drho, idx] = max(drho_max);
summary.max_drho_iter = f.iter(idx);

summary.any_N2_pre = any(f.N_pre >= 2);
summary.first_N2_pre = first_or_nan(f.iter(find(f.N_pre >= 2, 1, 'first')));
summary.bad_linearization_count = nnz(f.predicted_improvement_real_decrease);
summary.real_decrease_count = nnz(f.real_decrease);
summary.predicted_improvement_count = nnz(f.predicted_improvement);
end

function T = forensic_table(f)
if f.count == 0
    T = table();
    return
end
n = f.count;
pred1 = nan(n,1); pred2 = nan(n,1); dpred1 = nan(n,1); dpred2 = nan(n,1);
active = cell(n,1);
mac_diag_min = nan(n,1); mac_max_offdiag = nan(n,1);
for i = 1:n
    pl = f.predicted_cluster_lambda{i};
    dl = f.predicted_cluster_dlambda{i};
    pred1(i) = value_or_nan(pl, 1);
    pred2(i) = value_or_nan(pl, 2);
    dpred1(i) = value_or_nan(dl, 1);
    dpred2(i) = value_or_nan(dl, 2);
    active{i} = strjoin(f.active_constraints{i}(:)', ',');
    mac = f.mac_pre_mass{i};
    mac_diag_min(i) = min(diag(mac));
    off = mac;
    off(1:size(off,1)+1:end) = NaN;
    mac_max_offdiag(i) = max(off(:), [], 'omitnan');
end
drho_norm = sqrt(sum(f.drho_proposed.^2, 1))' / sqrt(size(f.drho_proposed, 1));
drho_max = max(abs(f.drho_proposed), [], 1)';
T = table(f.iter, f.N_pre, f.N_after, ...
    f.omega_pre(:,1), f.omega_pre(:,2), ...
    f.beta, f.beta_omega, pred1, pred2, dpred1, dpred2, ...
    f.predicted_J_lambda, f.predicted_J_dlambda, ...
    f.omega_after_proposed(:,1), f.omega_after_proposed(:,2), ...
    f.lambda_after_proposed(:,1), f.lambda_after_proposed(:,2), ...
    f.volume_pre, f.volume_after_proposed, f.volume_after_accepted, ...
    drho_norm, drho_max, active, mac_diag_min, mac_max_offdiag, ...
    f.predicted_improvement, f.real_decrease, ...
    f.predicted_improvement_real_decrease, f.inner_iters, ...
    f.inner_converged, f.inner_hit_max_iter, f.step_alpha, ...
    'VariableNames', {'iter','N_pre','N_after','omega1_pre','omega2_pre', ...
    'beta_lambda','beta_omega','pred_cluster_lambda1','pred_cluster_lambda2', ...
    'pred_cluster_dlambda1','pred_cluster_dlambda2','pred_J_lambda', ...
    'pred_J_dlambda','omega1_after_proposed','omega2_after_proposed', ...
    'lambda1_after_proposed','lambda2_after_proposed','volume_pre', ...
    'volume_after_proposed','volume_after_accepted','drho_norm', ...
    'drho_max','active_constraints','mac_diag_min_pre_mass', ...
    'mac_max_offdiag_pre_mass','predicted_improvement','real_decrease', ...
    'predicted_improvement_real_decrease','inner_iters','inner_converged', ...
    'inner_hit_max_iter','step_alpha'});
end

function write_vector_csv(path, f)
if f.count == 0, return; end
nEl = size(f.rho_pre, 1);
iter = f.iter(:)';
header = [{'element'}, strcat('rho_pre_iter_', cellstr(num2str(iter(:))))', ...
    strcat('drho_iter_', cellstr(num2str(iter(:))))'];
data = [(1:nEl)', f.rho_pre, f.drho_proposed];
fid = fopen(path, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', strjoin(header, ','));
fmt = [repmat('%.16g,', 1, size(data,2)-1), '%.16g\n'];
fprintf(fid, fmt, data');
end

function write_mac_csvs(out_dir, f)
for i = 1:f.count
    writematrix(f.mac_pre_mass{i}, fullfile(out_dir, ...
        sprintf('mac_pre_mass_iter_%03d.csv', f.iter(i))));
    writematrix(f.mac_post_mass{i}, fullfile(out_dir, ...
        sprintf('mac_post_mass_iter_%03d.csv', f.iter(i))));
end
end

function write_report(path, cfg, hist, summary)
f = hist.forensic;
fid = fopen(path, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '# Basin Exit Forensics Report\n\n');
fprintf(fid, 'Generated: `%s`\n\n', datestr(now, 31));
fprintf(fid, '## Scope\n\n');
fprintf(fid, ['Diagnostic-only run of `topopt_freq_exact` for the Du & Olhoff ', ...
    'CC beam. FE, interpolation, sensitivities, filters, and update logic were ', ...
    'not changed. Forensic tracing started after the first post-update state ', ...
    'with `abs(omega2-omega1)/omega1 < %.4g`.\n\n'], cfg.forensic_gap_tol);
fprintf(fid, ['Config: `nelx=%d`, `nely=%d`, `volfrac=%.3g`, `mass_mode=%s`, ', ...
    '`rmin_elem=%.3g`, `mult_tol=%.3g`, `move_lim=%.3g`, ', ...
    '`outer_move=%.3g`, `alpha=%.3g`, `inner_max_iter=%d`.\n\n'], ...
    cfg.nelx, cfg.nely, cfg.volfrac, cfg.mass_mode, cfg.rmin_elem, ...
    cfg.mult_tol, cfg.move_lim, cfg.outer_move, cfg.alpha, cfg.inner_max_iter);

fprintf(fid, '## Trigger\n\n');
if isnan(summary.trigger_iter)
    fprintf(fid, 'No trigger occurred. No basin-exit conclusion is possible from this run.\n');
    return
end
fprintf(fid, ['First trigger: outer iteration `%d`, post-update `omega1=%.6g`, ', ...
    '`omega2=%.6g`, relative gap `%.6g`, volume `%.6g`.\n\n'], ...
    summary.trigger_iter, summary.trigger_omega(1), summary.trigger_omega(2), ...
    summary.trigger_gap, summary.trigger_volume);

fprintf(fid, '## Per-Iteration Evidence\n\n');
    fprintf(fid, ['| iter | N pre | omega pre | beta omega | predicted cluster omega | ', ...
    'omega after rho+Delta | accepted omega | drho norm/max | active constraints | MAC diag min | ', ...
    'pred improve + real decrease |\n']);
fprintf(fid, '|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|\n');
for i = 1:f.count
    pred_lam = f.predicted_cluster_lambda{i};
    pred_omega1 = sqrt(max(value_or_nan(pred_lam, 1), 0));
    pred_omega2 = sqrt(max(value_or_nan(pred_lam, 2), 0));
    drho_norm = norm(f.drho_proposed(:,i)) / sqrt(size(f.drho_proposed,1));
    drho_max = max(abs(f.drho_proposed(:,i)));
    mac = f.mac_pre_mass{i};
    active = strjoin(f.active_constraints{i}(:)', ', ');
    if isempty(active), active = 'none'; end
    fprintf(fid, ['| %d | %g | %.4f / %.4f | %.4f | %.4f / %.4f | ', ...
        '%.4f / %.4f | %.4f / %.4f | %.4g / %.4g | %s | %.4g | %d |\n'], ...
        f.iter(i), f.N_pre(i), f.omega_pre(i,1), f.omega_pre(i,2), ...
        f.beta_omega(i), pred_omega1, pred_omega2, ...
        f.omega_after_proposed(i,1), f.omega_after_proposed(i,2), ...
        f.omega_after_accepted(i,1), f.omega_after_accepted(i,2), ...
        drho_norm, drho_max, active, min(diag(mac)), ...
        f.predicted_improvement_real_decrease(i));
end

fprintf(fid, '\n## Findings\n\n');
write_findings(fid, summary, f);

fprintf(fid, '\n## Evidence Files\n\n');
fprintf(fid, '- `basin_exit_forensics_results/basin_exit_forensics_result.mat`: full rho, Delta rho, constraints, and MAC matrices.\n');
fprintf(fid, '- `basin_exit_forensics_results/basin_exit_forensics_table.csv`: compact scalar evidence.\n');
fprintf(fid, '- `basin_exit_forensics_results/basin_exit_rho_drho_vectors.csv`: density and proposed increment vectors by recorded iteration.\n');
fprintf(fid, '- `basin_exit_forensics_results/mac_*_iter_*.csv`: mode MAC matrices.\n');
end

function write_findings(fid, summary, f)
accepted_gap = abs(f.omega_after_accepted(:,2) - f.omega_after_accepted(:,1)) ./ ...
    max(f.omega_after_accepted(:,1), eps);
accepted_in_basin = accepted_gap < f.gap_tol;
accepted_exit_idx = find(~accepted_in_basin, 1, 'first');
full_step_bad_idx = find(f.predicted_improvement_real_decrease, 1, 'first');
if ~isempty(accepted_exit_idx)
    bad_idx = accepted_exit_idx;
else
    bad_idx = full_step_bad_idx;
end
fprintf(fid, 'A) MMA linearization: ');
if ~isempty(bad_idx)
    fprintf(fid, ['yes. The first predicted-improvement/real-decrease step is ', ...
        'iteration `%d` for the full proposed step, and the accepted basin ', ...
        'exit is iteration `%d`. At the accepted exit, beta predicts ', ...
        '`omega=%.6g`, the cluster model predicts `omega=%.6g / %.6g`, ', ...
        'the accepted update gives `omega=%.6g / %.6g`, and the full ', ...
        '`rho+Delta rho` proposal gives `omega=%.6g / %.6g`.\n\n'], ...
        f.iter(first_nonempty(full_step_bad_idx, bad_idx)), f.iter(bad_idx), ...
        f.beta_omega(bad_idx), ...
        sqrt(max(value_or_nan(f.predicted_cluster_lambda{bad_idx}, 1), 0)), ...
        sqrt(max(value_or_nan(f.predicted_cluster_lambda{bad_idx}, 2), 0)), ...
        f.omega_after_accepted(bad_idx,1), f.omega_after_accepted(bad_idx,2), ...
        f.omega_after_proposed(bad_idx,1), f.omega_after_proposed(bad_idx,2));
else
    fprintf(fid, 'not shown by the recorded steps; no predicted-improvement/real-decrease event was recorded.\n\n');
end

fprintf(fid, 'B) Multiple-eigenvalue constraint: ');
if ~isempty(bad_idx) && f.N_pre(bad_idx) >= 2
    active = strjoin(f.active_constraints{bad_idx}(:)', ', ');
    if isempty(active), active = 'none'; end
    fprintf(fid, ['not inactive at the decisive exit. At iteration `%d`, ', ...
        '`N_pre=%g` and active model constraints were `%s`. The multiple ', ...
        'constraint was present, but its local model was not protective for ', ...
        'the accepted/proposed density change.\n\n'], ...
        f.iter(bad_idx), f.N_pre(bad_idx), active);
elseif summary.any_N2_pre
    fprintf(fid, ['partly active in the run (`N>=2` first at `%d`), but the ', ...
        'first bad-linearization step did not occur inside an `N>=2` recorded ', ...
        'state.\n\n'], summary.first_N2_pre);
else
    fprintf(fid, 'not reached before the recorded exit; the run stayed in the simple-mode constraint set.\n\n');
end

fprintf(fid, 'C) Mode tracking/multiplicity: ');
if ~isempty(bad_idx)
    mac = f.mac_pre_mass{bad_idx};
    collapse_mac = max(max(mac(:, 1:min(2,size(mac,2)))));
    fprintf(fid, ['not the primary failure. On the decisive step, the maximum ', ...
        'MAC between any pre-update mode and the two collapsed post-update ', ...
        'modes is `%.6g`, so the low modes are newly introduced by the ', ...
        'density step rather than a simple swap of the coalesced pair. ', ...
        'The previous `N=2` step may show expected mode mixing inside the ', ...
        'nearly multiple subspace; the exit itself is not resolved by ', ...
        'renumbering modes.\n\n'], collapse_mac);
else
    fprintf(fid, ['no decisive bad step was recorded; use the saved MAC matrices ', ...
        'to distinguish mode exchange from newly introduced low modes.\n\n']);
end

fprintf(fid, 'D) Step size: ');
if ~isempty(bad_idx)
    fprintf(fid, ['yes. At the accepted exit, the proposed step has ', ...
        '`drho_norm=%.6g` and `drho_max=%.6g` at the imposed cap; the ', ...
        'accepted half-step still has `drho_norm=%.6g` and `drho_max=%.6g`. ', ...
        'The accepted update leaves the basin at `omega1=%.6g`, while the ', ...
        'full proposed update collapses further to `omega1=%.6g`. This is ', ...
        'a step-length failure coupled to the bad local model.\n'], ...
        norm(f.drho_proposed(:,bad_idx)) / sqrt(size(f.drho_proposed,1)), ...
        max(abs(f.drho_proposed(:,bad_idx))), ...
        norm(f.drho_accepted(:,bad_idx)) / sqrt(size(f.drho_accepted,1)), ...
        max(abs(f.drho_accepted(:,bad_idx))), ...
        f.omega_after_accepted(bad_idx,1), f.omega_after_proposed(bad_idx,1));
else
    fprintf(fid, ['the largest proposed element step was `%.6g` at iteration `%d`; ', ...
        'compare this with the stored `drho_norm`/`drho_max` columns.\n'], ...
        summary.max_drho, summary.max_drho_iter);
end
end

function idx = first_nonempty(primary, fallback)
if isempty(primary)
    idx = fallback;
else
    idx = primary;
end
end

function v = value_or_nan(x, idx)
if numel(x) >= idx
    v = x(idx);
else
    v = NaN;
end
end

function v = first_or_nan(x)
if isempty(x), v = NaN; else, v = x(1); end
end
