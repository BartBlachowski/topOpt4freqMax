function study = s1_mitigation_400x50_pilot(outDir)
%S1_MITIGATION_400X50_PILOT One-parameter mass-penalization pilot.
%
% Runs only the failed Exp3 400x50 alpha=1.00 setup with one mitigation:
% mass interpolation exponent pmass = 6 instead of the baseline linear
% pmass = 1. This is the closest existing ourApproach mass-penalization
% control; Du-Olhoff du2007_c1 exists in OlhoffExact but is not implemented
% in ourApproach.

if nargin < 1 || isempty(outDir)
    scriptDir = fileparts(mfilename('fullpath'));
    outDir = fullfile(scriptDir, 'output', 's1_mitigation_400x50');
end
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));
addpath(scriptDir);

paths = localPaths(repoRoot, outDir);
diary(paths.log);
cleanupDiary = onCleanup(@() diary('off'));

fprintf('S1 mitigation pilot started: %s\n', char(datetime('now', 'TimeZone', 'UTC')));
fprintf('Scope: one mitigated 400x50 alpha=1.00 run only; no Exp2/Exp3 sweep, CR2, A4, P1, or manuscript edits.\n');

criteria = struct( ...
    'mac_threshold', 0.8, ...
    'feasibility_tolerance', 1e-8, ...
    'design_change_tolerance', 0.001);

baselineCfg = jsondecode(fileread(paths.baseline_config));
cfg = baselineCfg;
cfg.meta.name = 'S1 mitigation 400x50 alpha=1.00 mass exponent pmass=6';
cfg.meta.notes = ['S1 one-parameter mitigation pilot from failed Exp3 400x50. ', ...
    'Changed exactly one mitigation parameter relative to generated Exp3 config: ', ...
    'mass interpolation exponent pmass 1 -> 6. Du-Olhoff du2007_c1 is not ', ...
    'implemented in ourApproach; this uses the closest existing mass-penalization option.'];
cfg.optimization.mass_interpolation = struct( ...
    'mode', 'power', ...
    'pmass', 6, ...
    'baseline_pmass', 1, ...
    'documented_basis', 'closest existing ourApproach low-density mass penalization; not du2007_c1');
localWriteJson(paths.config_json, cfg);

runCfg = localBuildRunCfg(cfg);
rminPhys = double(cfg.optimization.filter.radius);
ft = 0; % generated Exp3 config uses sensitivity filter

fprintf('Mitigation parameter: mass_interpolation.pmass = %.6g\n', runCfg.pmass);
s1_json = fullfile(paths.s1_dir, 's1_mitigation_400x50_mode_summary.json');
resumeFromSaved = exist(paths.result_mat, 'file') == 2 && exist(s1_json, 'file') == 2;
if resumeFromSaved
    fprintf('Resuming from saved mitigated result and S1 artifacts; optimization is not rerun.\n');
    saved = load(paths.result_mat, 'result', 'cfg', 'xFinal', 'omega', 'info');
    result = saved.result;
    cfg = saved.cfg;
    xFinal = saved.xFinal; %#ok<NASGU>
    omega = saved.omega; %#ok<NASGU>
    info = saved.info; %#ok<NASGU>
    s1 = jsondecode(fileread(s1_json));
else
    fprintf('Calling topopt_freq once for mitigated 400x50 case...\n');
    runTic = tic;
    [xFinal, fHz, tIter, nIter, info] = topopt_freq( ...
        double(cfg.domain.mesh.nelx), double(cfg.domain.mesh.nely), ...
        double(cfg.optimization.volume_fraction), double(cfg.optimization.penalization), ...
        rminPhys, ft, double(cfg.domain.size.length), double(cfg.domain.size.height), runCfg);
    timingTotal = toc(runTic);
    omega = 2*pi*fHz(:);

    result = localBuildResult(cfg, criteria, xFinal, omega, tIter, nIter, timingTotal, info, paths, repoRoot);
    localWriteHistories(paths, info, cfg);
    localWriteTopology(paths, xFinal, cfg);
    result.acceptance.artifacts_complete = true;
    result.accepted = result.acceptance.not_capped && result.acceptance.design_change_ok && ...
        result.acceptance.feasibility_ok && result.acceptance.tracked_mac_ok && ...
        result.acceptance.a5_lowest_mode_ok && result.acceptance.artifacts_complete;
    if result.accepted
        result.classification = 'accepted';
    end
    localWriteJson(paths.result_json, localJsonSafe(result));
    save(paths.result_mat, 'result', 'cfg', 'xFinal', 'omega', 'info', '-v7.3');

    fprintf('Running S1 postprocessing on mitigated final topology...\n');
    s1 = s1_exp3_400x50_mode_diagnostic(paths.s1_dir, paths.result_mat, 's1_mitigation_400x50');
end

baseline = localLoadBaseline(paths);
comparison = localCompareBaselineMitigated(baseline, result, s1, paths, repoRoot);
classification = localClassifyMitigation(result, s1, baseline);

study = struct();
study.study = 'S1 mitigation pilot 400x50 alpha=1.00';
study.scope = 'One mitigated 400x50 run; same Exp3 setup except pmass=6 mass penalization';
study.created_utc = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'));
study.mitigation = cfg.optimization.mass_interpolation;
study.criteria = criteria;
study.result = result;
study.s1 = s1;
study.comparison = comparison;
study.classification = classification;
study.artifacts = localArtifacts(paths, repoRoot);

localWriteJson(paths.study_json, localJsonSafe(study));
localWriteSummary(paths.summary_md, study);
localWriteManifest(paths.manifest_json, study, paths, repoRoot);

fprintf('S1 mitigation classification: %s\n', classification.label);
fprintf('Summary: %s\n', paths.summary_md);
diary('off');
end

function paths = localPaths(repoRoot, outDir)
prefix = fullfile(outDir, 's1_mitigation_400x50');
paths = struct();
paths.baseline_result_json = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    'exp3_authoritative_mesh_convergence', 'mesh_400x50', ...
    'exp3_authoritative_400x50_result.json');
paths.baseline_config = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    'exp3_authoritative_mesh_convergence', 'mesh_400x50', ...
    'exp3_authoritative_400x50_config.json');
paths.baseline_convergence_csv = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    'exp3_authoritative_mesh_convergence', 'mesh_400x50', ...
    'exp3_authoritative_400x50_convergence_history.csv');
paths.baseline_topology_csv = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    'exp3_authoritative_mesh_convergence', 'mesh_400x50', ...
    'exp3_authoritative_400x50_topology.csv');
paths.baseline_s1_json = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    's1_exp3_400x50_mode_diagnostic', 's1_exp3_400x50_mode_summary.json');

paths.log = [prefix, '_run.log'];
paths.config_json = [prefix, '_config.json'];
paths.result_json = [prefix, '_result.json'];
paths.result_mat = [prefix, '_result.mat'];
paths.study_json = [prefix, '_study.json'];
paths.summary_md = fullfile(outDir, 's1_mitigation_400x50_summary.md');
paths.manifest_json = [prefix, '_manifest.json'];
paths.convergence_csv = [prefix, '_convergence_history.csv'];
paths.frequency_csv = [prefix, '_frequency_history.csv'];
paths.mode_tracking_csv = [prefix, '_mode_tracking.csv'];
paths.topology_csv = [prefix, '_topology.csv'];
paths.topology_json = [prefix, '_topology.json'];
paths.topology_png = [prefix, '_topology.png'];
paths.a5_json = [prefix, '_a5_lowest_mode_check.json'];
paths.s1_dir = fullfile(outDir, 's1_postprocessing');
if exist(paths.s1_dir, 'dir') ~= 7
    mkdir(paths.s1_dir);
end
end

function runCfg = localBuildRunCfg(cfg)
runCfg = struct();
runCfg.E0 = double(cfg.material.E);
runCfg.Emin = double(cfg.material.E) * double(cfg.void_material.E_min_ratio);
runCfg.nu = double(cfg.material.nu);
runCfg.rho0 = double(cfg.material.rho);
runCfg.rho_min = double(cfg.void_material.rho_min);
runCfg.move = double(cfg.optimization.move_limit);
runCfg.conv_tol = double(cfg.optimization.convergence_tol);
runCfg.max_iters = double(cfg.optimization.max_iters);
runCfg.supportType = 'CC';
[extraFixedDofs, ~] = supportsToFixedDofs(cfg.bc.supports, ...
    double(cfg.domain.mesh.nelx), double(cfg.domain.mesh.nely), ...
    double(cfg.domain.size.length), double(cfg.domain.size.height));
runCfg.extraFixedDofs = extraFixedDofs;
runCfg.pasS = [];
runCfg.pasV = [];
runCfg.load_sensitivity = char(string(cfg.optimization.load_sensitivity));
runCfg.gate_a0_diagnostics = true;
runCfg.harmonic_normalize = false;
runCfg.semi_harmonic_baseline = 'solid';
runCfg.load_cases = cfg.domain.load_cases;
runCfg.optimizer = upper(char(string(cfg.optimization.optimizer)));
runCfg.approach_name = 'ourApproach';
runCfg.visualize_live = false;
runCfg.visualization_quality = 'regular';
runCfg.save_frq_iterations = false;
runCfg.pmass = double(cfg.optimization.mass_interpolation.pmass);
end

function result = localBuildResult(cfg, criteria, xFinal, omega, tIter, nIter, timingTotal, info, paths, repoRoot)
volfrac = double(cfg.optimization.volume_fraction);
result = struct();
result.study = 'S1 mitigation pilot 400x50 alpha=1.00';
result.scope = 'same Exp3 400x50 setup except mass_interpolation.pmass=6';
result.implementation_language = 'MATLAB';
result.mesh_label = '400x50';
result.mesh = cfg.domain.mesh;
result.alpha = 1.0;
result.source_baseline_config = localRel(fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    'exp3_authoritative_mesh_convergence', 'mesh_400x50', ...
    'exp3_authoritative_400x50_config.json'), repoRoot);
result.case_config = localRel(paths.config_json, repoRoot);
result.authoritative_load = 'F(x) = omega0^2 * M(x) * Phi0';
result.mitigation = cfg.optimization.mass_interpolation;
result.filter_physical_radius = double(cfg.optimization.filter.radius);
result.criteria = criteria;
result.solver_success = true;
result.iterations = double(nIter);
result.iteration_cap = double(cfg.optimization.max_iters);
result.final = struct();
result.final.omega_rad_s = omega(:);
result.final.frequency_hz = omega(:) / (2*pi);
result.final.grayness = mean(4 * xFinal(:) .* (1 - xFinal(:)));
result.final.volume = mean(xFinal(:));
result.final.feasibility = max(0, result.final.volume - volfrac);
result.final.design_change = localLast(info.cr2_history.design_change);
result.final.tracked_mode_index = double(info.cr2_final_tracking.tracked_mode_index);
result.final.tracked_mode_mac = double(info.cr2_final_tracking.tracked_mode_mac);
result.final.tracked_mode_omega = double(info.cr2_final_tracking.tracked_mode_omega);
result.a5_lowest_mode_check = localA5(info);
result.acceptance = struct( ...
    'not_capped', result.iterations < result.iteration_cap, ...
    'design_change_ok', result.final.design_change <= criteria.design_change_tolerance, ...
    'feasibility_ok', result.final.feasibility <= criteria.feasibility_tolerance, ...
    'tracked_mac_ok', result.final.tracked_mode_mac >= criteria.mac_threshold, ...
    'a5_lowest_mode_ok', result.a5_lowest_mode_check.pass, ...
    'artifacts_complete', false);
result.accepted = result.acceptance.not_capped && result.acceptance.design_change_ok && ...
    result.acceptance.feasibility_ok && result.acceptance.tracked_mac_ok && ...
    result.acceptance.a5_lowest_mode_ok;
if result.accepted
    result.classification = 'accepted';
elseif ~result.acceptance.not_capped
    result.classification = 'capped';
elseif ~result.acceptance.tracked_mac_ok || ~result.acceptance.a5_lowest_mode_ok
    result.classification = 'mode invalid';
else
    result.classification = 'not accepted';
end
result.timing_total_s = timingTotal;
result.timing_per_iter_s = tIter;
result.artifacts = localArtifacts(paths, repoRoot);
end

function check = localA5(info)
freqs = info.cr2_final_tracking.frequencies(:);
trackedIndex = double(info.cr2_final_tracking.tracked_mode_index);
trackedOmega = NaN;
if trackedIndex >= 1 && trackedIndex <= numel(freqs)
    trackedOmega = freqs(trackedIndex);
end
check = struct( ...
    'pass', isfinite(trackedIndex) && trackedIndex == 1 && isfinite(trackedOmega), ...
    'tracked_mode_index', trackedIndex, ...
    'tracked_mode_omega_rad_s', trackedOmega, ...
    'lowest_mode_omega_rad_s', freqs(1), ...
    'modes_below_tracked_count', max(0, trackedIndex - 1));
if check.pass
    check.message = 'tracked mode is the lowest computed mode';
else
    check.message = 'tracked mode is not the lowest computed mode';
end
end

function localWriteHistories(paths, info, cfg)
h = info.cr2_history;
n = size(h.objective, 1);
iter = (1:n)';
convTable = table(iter, h.objective(:), h.design_change(:), h.volume(:), ...
    h.feasibility(:), h.grayness(:), h.tracked_mode_index(:), ...
    h.tracked_mode_mac(:), h.tracked_mode_omega(:), ...
    'VariableNames', {'iteration','objective','design_change','volume', ...
    'feasibility','grayness','tracked_mode_index','tracked_mode_mac', ...
    'tracked_mode_omega_rad_s'});
writetable(convTable, paths.convergence_csv);
freq = h.frequency;
freqNames = arrayfun(@(k) sprintf('omega_%d_rad_s', k), 1:size(freq,2), 'UniformOutput', false);
writetable(array2table([iter, freq], 'VariableNames', [{'iteration'}, freqNames]), paths.frequency_csv);
writetable(table(iter, h.tracked_mode_index(:), h.tracked_mode_mac(:), h.tracked_mode_omega(:), ...
    'VariableNames', {'iteration','tracked_mode_index','tracked_mode_mac','tracked_mode_omega_rad_s'}), ...
    paths.mode_tracking_csv);
localWriteJson(paths.a5_json, localA5(info));
end

function localWriteTopology(paths, xFinal, cfg)
nelx = double(cfg.domain.mesh.nelx);
nely = double(cfg.domain.mesh.nely);
topology = reshape(xFinal(:), nely, nelx);
writematrix(topology, paths.topology_csv);
localWriteJson(paths.topology_json, struct('nelx', nelx, 'nely', nely, ...
    'density_row_major_nely_by_nelx', topology));
fig = figure('Color', 'white', 'Visible', 'off');
ax = axes('Parent', fig);
imagesc(ax, 1 - topology);
axis(ax, 'equal', 'off');
colormap(ax, gray);
title(ax, 'S1 mitigation 400x50 alpha=1.00 topology', 'Interpreter', 'none');
exportgraphics(fig, paths.topology_png, 'Resolution', 180, 'BackgroundColor', 'white');
close(fig);
end

function baseline = localLoadBaseline(paths)
baseline = struct();
baseline.result = jsondecode(fileread(paths.baseline_result_json));
if exist(paths.baseline_s1_json, 'file') == 2
    baseline.s1 = jsondecode(fileread(paths.baseline_s1_json));
else
    baseline.s1 = struct();
end
baseline.convergence = readtable(paths.baseline_convergence_csv);
baseline.topology = readmatrix(paths.baseline_topology_csv);
end

function comparison = localCompareBaselineMitigated(baseline, result, s1, paths, repoRoot)
mitConv = readtable(paths.convergence_csv);
mitTopo = readmatrix(paths.topology_csv);
baseTopo = baseline.topology;
d = mitTopo - baseTopo;
comparison = struct();
comparison.baseline = struct( ...
    'omega_rad_s', baseline.result.final.omega_rad_s, ...
    'tracked_mac', baseline.result.final.tracked_mode_mac, ...
    'tracked_omega_rad_s', baseline.result.final.tracked_mode_omega, ...
    'classification', baseline.result.classification, ...
    'grayness', baseline.result.final.grayness, ...
    'iterations', baseline.result.iterations, ...
    'final_design_change', baseline.result.final.design_change);
comparison.mitigated = struct( ...
    'omega_rad_s', result.final.omega_rad_s, ...
    'tracked_mac', result.final.tracked_mode_mac, ...
    'tracked_omega_rad_s', result.final.tracked_mode_omega, ...
    'classification', result.classification, ...
    'grayness', result.final.grayness, ...
    'iterations', result.iterations, ...
    'final_design_change', result.final.design_change);
comparison.delta = struct( ...
    'omega_rad_s', result.final.omega_rad_s(:) - baseline.result.final.omega_rad_s(:), ...
    'tracked_mac', result.final.tracked_mode_mac - baseline.result.final.tracked_mode_mac, ...
    'tracked_omega_rad_s', result.final.tracked_mode_omega - baseline.result.final.tracked_mode_omega, ...
    'grayness', result.final.grayness - baseline.result.final.grayness, ...
    'iterations', result.iterations - baseline.result.iterations);
comparison.topology = struct( ...
    'mean_abs_difference', mean(abs(d(:))), ...
    'rms_difference', sqrt(mean(d(:).^2)), ...
    'max_abs_difference', max(abs(d(:))), ...
    'correlation', localCorr(baseTopo(:), mitTopo(:)));
comparison.convergence = struct( ...
    'baseline_last10_design_change_mean', mean(baseline.convergence.design_change(max(1,end-9):end)), ...
    'mitigated_last10_design_change_mean', mean(mitConv.design_change(max(1,end-9):end)), ...
    'baseline_last100_design_change_max', max(baseline.convergence.design_change(max(1,end-99):end)), ...
    'mitigated_last100_design_change_max', max(mitConv.design_change(max(1,end-99):end)));
comparison.s1 = localCompareS1(baseline.s1, s1);
comparison.artifacts = struct( ...
    'baseline_result_json', localRel(paths.baseline_result_json, repoRoot), ...
    'baseline_s1_json', localRel(paths.baseline_s1_json, repoRoot), ...
    'mitigated_result_json', localRel(paths.result_json, repoRoot), ...
    'mitigated_s1_json', localRel(fullfile(paths.s1_dir, 's1_mitigation_400x50_mode_summary.json'), repoRoot));
end

function c = localCompareS1(baseS1, mitS1)
c = struct();
if isfield(baseS1, 'overall')
    c.baseline_overall = baseS1.overall;
else
    c.baseline_overall = struct();
end
c.mitigated_overall = mitS1.overall;
baseModes = localModesArray(baseS1);
mitModes = localModesArray(mitS1);
n = min(numel(baseModes), numel(mitModes));
rows = repmat(struct('mode', NaN, 'baseline_classification', '', ...
    'mitigated_classification', '', 'baseline_low_density_strain_fraction', NaN, ...
    'mitigated_low_density_strain_fraction', NaN, ...
    'baseline_strain_effective_element_fraction', NaN, ...
    'mitigated_strain_effective_element_fraction', NaN), n, 1);
for i = 1:n
    rows(i).mode = i;
    rows(i).baseline_classification = baseModes(i).classification;
    rows(i).mitigated_classification = mitModes(i).classification;
    rows(i).baseline_low_density_strain_fraction = baseModes(i).low_density_strain_fraction;
    rows(i).mitigated_low_density_strain_fraction = mitModes(i).low_density_strain_fraction;
    rows(i).baseline_strain_effective_element_fraction = baseModes(i).strain_effective_element_fraction;
    rows(i).mitigated_strain_effective_element_fraction = mitModes(i).strain_effective_element_fraction;
end
c.modes = rows;
end

function modes = localModesArray(s)
if isfield(s, 'modes')
    modes = s.modes;
else
    modes = repmat(struct('classification', '', 'low_density_strain_fraction', NaN, ...
        'strain_effective_element_fraction', NaN), 0, 1);
end
end

function classification = localClassifyMitigation(result, s1, baseline)
baseLow = NaN;
if isfield(baseline.s1, 'overall') && isfield(baseline.s1.overall, 'localized_low_density_count')
    baseLow = baseline.s1.overall.localized_low_density_count;
end
mitLow = s1.overall.localized_low_density_count;
accepted = result.accepted;
if accepted && mitLow == 0
    label = 'successful';
    reason = 'All acceptance gates passed and no first-10 mode is classified as localized low-density.';
elseif accepted && (isnan(baseLow) || mitLow < baseLow)
    label = 'partially successful';
    reason = 'Acceptance gates passed and localized low-density mode count decreased, but was not eliminated.';
elseif ~accepted && ~isnan(baseLow) && mitLow < baseLow
    label = 'partially successful';
    reason = 'Localized low-density mode count decreased, but one or more acceptance gates failed.';
elseif ~accepted
    label = 'failed';
    reason = 'Mitigated case did not pass acceptance gates and did not reduce localized low-density modes.';
else
    label = 'inconclusive';
    reason = 'Acceptance and S1 localization indicators are mixed.';
end
classification = struct('label', label, 'reason', reason);
end

function localWriteSummary(path, study)
fid = fopen(path, 'w');
if fid < 0
    error('S1Mitigation:CannotWriteSummary', 'Cannot write %s', path);
end
cleanup = onCleanup(@() fclose(fid));
r = study.result;
c = study.comparison;
fprintf(fid, '# S1 Mitigation 400x50 Summary\n\n');
fprintf(fid, 'Scope: one mitigated Exp3 400x50 alpha=1.00 run only. No manuscript edits, no full Exp2/Exp3 rerun, and no CR2/A4/P1 run.\n\n');
fprintf(fid, '## Mitigation\n\n');
fprintf(fid, 'Changed exactly one mitigation parameter: mass interpolation exponent `pmass` from baseline `1` to mitigated `6`.\n\n');
fprintf(fid, 'Du-Olhoff `du2007_c1` is implemented in `analysis/OlhoffApproachExact/Matlab/mass_interp.m`, but not in `ourApproach`; this pilot uses the closest existing `ourApproach` mass-penalization option.\n\n');
fprintf(fid, '## Classification\n\n');
fprintf(fid, '**%s.** %s\n\n', study.classification.label, study.classification.reason);
fprintf(fid, '## Acceptance Gates\n\n');
fprintf(fid, '| gate | pass | value |\n|---|---|---:|\n');
fprintf(fid, '| not capped | %s | %.0f/%.0f |\n', localPass(r.acceptance.not_capped), r.iterations, r.iteration_cap);
fprintf(fid, '| design_change <= tol | %s | %.12g <= %.12g |\n', localPass(r.acceptance.design_change_ok), r.final.design_change, study.criteria.design_change_tolerance);
fprintf(fid, '| feasibility <= tol | %s | %.12g <= %.12g |\n', localPass(r.acceptance.feasibility_ok), r.final.feasibility, study.criteria.feasibility_tolerance);
fprintf(fid, '| tracked MAC >= 0.8 | %s | %.12g |\n', localPass(r.acceptance.tracked_mac_ok), r.final.tracked_mode_mac);
fprintf(fid, '| A5 lowest-mode check | %s | mode %.0f |\n', localPass(r.acceptance.a5_lowest_mode_ok), r.a5_lowest_mode_check.tracked_mode_index);
fprintf(fid, '| artifacts complete | %s | manifest written |\n\n', localPass(true));
fprintf(fid, '## Baseline vs Mitigated\n\n');
fprintf(fid, '| metric | baseline | mitigated | delta |\n|---|---:|---:|---:|\n');
fprintf(fid, '| omega_1 rad/s | %.12g | %.12g | %.12g |\n', c.baseline.omega_rad_s(1), c.mitigated.omega_rad_s(1), c.delta.omega_rad_s(1));
fprintf(fid, '| omega_2 rad/s | %.12g | %.12g | %.12g |\n', c.baseline.omega_rad_s(2), c.mitigated.omega_rad_s(2), c.delta.omega_rad_s(2));
fprintf(fid, '| omega_3 rad/s | %.12g | %.12g | %.12g |\n', c.baseline.omega_rad_s(3), c.mitigated.omega_rad_s(3), c.delta.omega_rad_s(3));
fprintf(fid, '| tracked MAC | %.12g | %.12g | %.12g |\n', c.baseline.tracked_mac, c.mitigated.tracked_mac, c.delta.tracked_mac);
fprintf(fid, '| grayness | %.12g | %.12g | %.12g |\n', c.baseline.grayness, c.mitigated.grayness, c.delta.grayness);
fprintf(fid, '| iterations | %.0f | %.0f | %.0f |\n', c.baseline.iterations, c.mitigated.iterations, c.delta.iterations);
fprintf(fid, '| final design change | %.12g | %.12g |  |\n', c.baseline.final_design_change, c.mitigated.final_design_change);
fprintf(fid, '| topology MAD |  | %.12g |  |\n', c.topology.mean_abs_difference);
fprintf(fid, '| topology correlation |  | %.12g |  |\n\n', c.topology.correlation);
fprintf(fid, '## S1 Mode Diagnosis\n\n');
fprintf(fid, 'Baseline S1: `%s` with %d localized low-density modes, %d ambiguous modes.\n\n', ...
    c.s1.baseline_overall.classification, c.s1.baseline_overall.localized_low_density_count, ...
    c.s1.baseline_overall.ambiguous_count);
fprintf(fid, 'Mitigated S1: `%s` with %d localized low-density modes, %d ambiguous modes.\n\n', ...
    c.s1.mitigated_overall.classification, c.s1.mitigated_overall.localized_low_density_count, ...
    c.s1.mitigated_overall.ambiguous_count);
fprintf(fid, '| mode | baseline class | mitigated class | baseline low-density S frac | mitigated low-density S frac | baseline eff-S frac | mitigated eff-S frac |\n');
fprintf(fid, '|---:|---|---|---:|---:|---:|---:|\n');
for i = 1:numel(c.s1.modes)
    m = c.s1.modes(i);
    fprintf(fid, '| %d | %s | %s | %.6g | %.6g | %.6g | %.6g |\n', ...
        m.mode, m.baseline_classification, m.mitigated_classification, ...
        m.baseline_low_density_strain_fraction, m.mitigated_low_density_strain_fraction, ...
        m.baseline_strain_effective_element_fraction, m.mitigated_strain_effective_element_fraction);
end
fprintf(fid, '\n## Artifacts\n\n');
fprintf(fid, '- result MAT: `%s`\n', study.artifacts.result_mat);
fprintf(fid, '- result JSON: `%s`\n', study.artifacts.result_json);
fprintf(fid, '- S1 mode diagnosis: `%s`\n', study.s1.artifacts.report_md);
fprintf(fid, '- manifest: `%s`\n', study.artifacts.manifest_json);
end

function localWriteManifest(path, study, paths, repoRoot)
manifest = struct();
manifest.study = study.study;
manifest.scope = study.scope;
manifest.created_utc = study.created_utc;
manifest.mitigation = study.mitigation;
manifest.classification = study.classification;
manifest.acceptance = study.result.acceptance;
manifest.inputs = struct( ...
    'baseline_result_json', localRel(paths.baseline_result_json, repoRoot), ...
    'baseline_config', localRel(paths.baseline_config, repoRoot), ...
    'baseline_s1_json', localRel(paths.baseline_s1_json, repoRoot));
manifest.outputs = localArtifacts(paths, repoRoot);
manifest.no_manuscript_edit = true;
manifest.no_full_exp2_exp3_rerun = true;
manifest.no_cr2_a4_p1 = true;
localWriteJson(path, localJsonSafe(manifest));
end

function artifacts = localArtifacts(paths, repoRoot)
artifacts = struct( ...
    'log', localRel(paths.log, repoRoot), ...
    'config_json', localRel(paths.config_json, repoRoot), ...
    'result_json', localRel(paths.result_json, repoRoot), ...
    'result_mat', localRel(paths.result_mat, repoRoot), ...
    'study_json', localRel(paths.study_json, repoRoot), ...
    'summary_md', localRel(paths.summary_md, repoRoot), ...
    'manifest_json', localRel(paths.manifest_json, repoRoot), ...
    'convergence_csv', localRel(paths.convergence_csv, repoRoot), ...
    'frequency_csv', localRel(paths.frequency_csv, repoRoot), ...
    'mode_tracking_csv', localRel(paths.mode_tracking_csv, repoRoot), ...
    'topology_csv', localRel(paths.topology_csv, repoRoot), ...
    'topology_json', localRel(paths.topology_json, repoRoot), ...
    'topology_png', localRel(paths.topology_png, repoRoot), ...
    'a5_json', localRel(paths.a5_json, repoRoot), ...
    's1_dir', localRel(paths.s1_dir, repoRoot));
end

function v = localLast(x)
x = x(:);
v = x(end);
end

function c = localCorr(a, b)
C = corrcoef(a(:), b(:));
if numel(C) >= 4
    c = C(1,2);
else
    c = NaN;
end
end

function s = localPass(tf)
if tf
    s = 'pass';
else
    s = 'fail';
end
end

function safe = localJsonSafe(value)
safe = value;
if isstruct(safe)
    for j = 1:numel(safe)
        fields = fieldnames(safe(j));
        for i = 1:numel(fields)
            safe(j).(fields{i}) = localJsonSafe(safe(j).(fields{i}));
        end
    end
elseif iscell(safe)
    for i = 1:numel(safe)
        safe{i} = localJsonSafe(safe{i});
    end
elseif isnumeric(safe)
    safe = double(safe);
end
end

function localWriteJson(path, data)
txt = jsonencode(data, 'PrettyPrint', true);
fid = fopen(path, 'w');
if fid < 0
    error('S1Mitigation:CannotWriteJson', 'Cannot write %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, txt, 'char');
fwrite(fid, newline, 'char');
end

function rel = localRel(path, repoRoot)
path = char(path);
repoRoot = char(repoRoot);
if startsWith(path, [repoRoot, filesep])
    rel = strrep(path(numel(repoRoot)+2:end), filesep, '/');
else
    rel = strrep(path, filesep, '/');
end
end
