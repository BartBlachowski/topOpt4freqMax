function study = exp3_authoritative_mesh_convergence(outDir)
%EXP3_AUTHORITATIVE_MESH_CONVERGENCE  Exp3 mesh convergence for Exp2 alpha=1.
%
% MATLAB-only runner. Scope is limited to the accepted nontrivial Exp2
% authoritative case alpha=1.00 on 200x25 and 400x50 meshes.

if nargin < 1 || isempty(outDir)
    scriptDir = fileparts(mfilename('fullpath'));
    outDir = fullfile(scriptDir, 'output', 'exp3_authoritative_mesh_convergence');
end
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));

rootPaths = localRootPaths(outDir);
diary(rootPaths.log);
cleanupDiary = onCleanup(@() diary('off'));

criteria = struct( ...
    'mac_threshold', 0.8, ...
    'feasibility_tolerance', 1e-8, ...
    'design_change_tolerance', 0.001, ...
    'tracked_omega_relative_tolerance', 0.05, ...
    'topology_correlation_minimum', 0.8, ...
    'topology_mean_abs_difference_maximum', 0.15);

fprintf('Exp3 authoritative mesh convergence started: %s\n', ...
    char(datetime('now', 'TimeZone', 'UTC')));
fprintf('Implementation language: MATLAB\n');
fprintf('Scope: Exp3 alpha=1.00 mesh convergence only; no Exp2 sweep, CR2, A4, S1, P1, Python, or manuscript edits.\n');
fprintf(['Predeclared convergence criterion: accepted gates on both meshes, ', ...
    'same tracked mode index, tracked MAC >= %.3f, relative tracked omega change <= %.3f, ', ...
    'topology correlation >= %.3f, topology MAD <= %.3f.\n'], ...
    criteria.mac_threshold, criteria.tracked_omega_relative_tolerance, ...
    criteria.topology_correlation_minimum, criteria.topology_mean_abs_difference_maximum);

meshCases = [ ...
    struct('label', '200x25', 'config', fullfile(scriptDir, 'clamped_beam_200x25.json')); ...
    struct('label', '400x50', 'config', fullfile(scriptDir, 'clamped_beam_400x50.json'))];

caseResults = repmat(localInitCase(), numel(meshCases), 1);
for i = 1:numel(meshCases)
    mc = meshCases(i);
    caseDir = fullfile(outDir, ['mesh_', mc.label]);
    if exist(caseDir, 'dir') ~= 7
        mkdir(caseDir);
    end
    paths = localCasePaths(caseDir, mc.label);
    cfg = localBuildConfig(mc.config, criteria);
    localWriteJson(paths.config, cfg);

    result = localInitCase();
    result.study = 'Exp3 authoritative alpha=1 mesh convergence';
    result.implementation_language = 'MATLAB';
    result.scope = 'Exp3 alpha=1.00 mesh convergence case';
    result.mesh_label = mc.label;
    result.mesh = cfg.domain.mesh;
    result.alpha = 1.0;
    result.source_config = localRelativePath(mc.config, repoRoot);
    result.case_config = localRelativePath(paths.config, repoRoot);
    result.authoritative_load = 'F(x) = omega0^2 * M(x) * Phi0';
    result.filter_physical_radius = cfg.optimization.filter.radius;
    result.criteria = criteria;
    result.iteration_cap = double(cfg.optimization.max_iters);
    result.artifacts = localRelativeStruct(paths, repoRoot);

    xFinal = [];
    omega = NaN(3, 1);
    info = struct();
    tIter = NaN; %#ok<NASGU>
    memUsage = NaN; %#ok<NASGU>
    runTic = tic;

    try
        fprintf('\n=== Exp3 authoritative mesh %s alpha=1.00 ===\n', mc.label);
        [xFinal, omega, tIter, nIter, memUsage, info] = run_topopt_from_json(cfg);
        result.solver_success = true;
        result.iterations = double(nIter);
        result.final.omega_rad_s = omega(:);
        result.final.frequency_hz = omega(:) / (2*pi);
        result.final.grayness = mean(4 * xFinal(:) .* (1 - xFinal(:)));
        result.final.volume = mean(xFinal(:));
        result.final.feasibility = max(0, result.final.volume - double(cfg.optimization.volume_fraction));
        result.final.design_change = localFinalHistoryValue(info, {'cr2_history','design_change'});
        result.final.tracked_mode_index = localFinalHistoryValue(info, {'cr2_final_tracking','tracked_mode_index'});
        result.final.tracked_mode_mac = localFinalHistoryValue(info, {'cr2_final_tracking','tracked_mode_mac'});
        result.final.tracked_mode_omega = localFinalHistoryValue(info, {'cr2_final_tracking','tracked_mode_omega'});
        result.timing_total_s = toc(runTic);
        result.timing_per_iter_s = tIter;
        result.peak_memory_MB = memUsage;
        result.topology_quality = checkTopologyQuality( ...
            xFinal, double(cfg.domain.mesh.nelx), double(cfg.domain.mesh.nely));

        localValidateDiagnostics(info, nIter);
        result.a5_lowest_mode_check = localA5LowestModeCheck(info);
        localWriteHistories(paths, info, cfg);
        localWriteTopology(paths, xFinal, cfg, mc.label);
        localWriteA5(paths, result.a5_lowest_mode_check);
        localWriteJson(paths.topology_quality_json, localJsonSafe(result.topology_quality));

        result.classification = localClassifyCase(result, cfg, criteria);
        result.accepted = strcmp(result.classification, 'accepted');
    catch ME
        result.timing_total_s = toc(runTic);
        result.exception = getReport(ME, 'extended', 'hyperlinks', 'off');
        result.classification = 'implementation failure';
        result.accepted = false;
        fprintf(2, '\nExp3 mesh %s exception preserved:\n%s\n', mc.label, result.exception);
    end

    save(paths.mat, 'result', 'cfg', 'xFinal', 'omega', 'info', '-v7.3');
    localWriteJson(paths.result_json, localJsonSafe(result));
    localWriteCaseManifest(paths.manifest, result, paths, repoRoot);
    caseResults(i) = result;
    fprintf('Exp3 mesh %s classification: %s\n', mc.label, result.classification);
end

comparison = localCompareMeshes(caseResults, outDir, repoRoot);
localWriteJson(rootPaths.topology_metrics_json, comparison.topology_difference_metrics);

study = struct();
study.study = 'Exp3 authoritative alpha=1 mesh convergence';
study.implementation_language = 'MATLAB';
study.scope = 'Exp3 alpha=1.00 mesh convergence only';
study.excluded = {'Exp2 alpha sweep','CR2','A4','S1','P1','Python','manuscript edits'};
study.alpha = 1.0;
study.criteria = criteria;
study.cases = caseResults;
study.comparison = comparison;
study.classification = localClassifyStudy(caseResults, comparison, criteria);
study.created_utc = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'));
study.artifacts = localRelativeStruct(rootPaths, repoRoot);

save(rootPaths.mat, 'study', '-v7.3');
localWriteJson(rootPaths.result_json, localJsonSafe(study));
localWriteRootManifest(rootPaths.manifest, study, rootPaths, repoRoot);
localWriteSummary(rootPaths.summary, study);

fprintf('\nExp3 classification: %s\n', study.classification);
fprintf('Summary: %s\n', rootPaths.summary);
fprintf('Manifest: %s\n', rootPaths.manifest);
diary('off');
end

function cfg = localBuildConfig(configPath, criteria)
cfg = jsondecode(fileread(configPath));
cfg.meta.name = sprintf('Exp3 authoritative alpha=1.00 mesh convergence %dx%d', ...
    cfg.domain.mesh.nelx, cfg.domain.mesh.nely);
cfg.meta.notes = ['Exp3 scoped mesh-convergence case for accepted nontrivial ', ...
    'Exp2 alpha=1.00. Authoritative load F(x)=omega0^2*M(x)*Phi0, ', ...
    'solid reference, load_sensitivity=complete, Gate A0 diagnostics enabled.'];

cfg.domain.load_cases(1).name = 'alpha1.00_solid_reference_mode_1';
cfg.domain.load_cases(1).factor = 1.0;
cfg.domain.load_cases(1).loads(1).type = 'semi_harmonic';
cfg.domain.load_cases(1).loads(1).mode = 1;
cfg.domain.load_cases(1).loads(1).factor = 1.0;
cfg.domain.load_cases(2).name = 'alpha0.00_solid_reference_mode_2';
cfg.domain.load_cases(2).factor = 0.0;
cfg.domain.load_cases(2).loads(1).type = 'semi_harmonic';
cfg.domain.load_cases(2).loads(1).mode = 2;
cfg.domain.load_cases(2).loads(1).factor = 1.0;

cfg.optimization.semi_harmonic_baseline = 'solid';
if isfield(cfg.optimization, 'semi_harmonic_rho_source')
    cfg.optimization = rmfield(cfg.optimization, 'semi_harmonic_rho_source');
end
cfg.optimization.harmonic_normalize = false;
cfg.optimization.load_sensitivity = 'complete';
cfg.optimization.gate_a0_diagnostics = true;
cfg.optimization.convergence_tol = criteria.design_change_tolerance;

coarseHx = 8.0 / 200.0;
cfg.optimization.filter.radius = 2.0 * coarseHx;
cfg.optimization.filter.radius_units = 'physical';

cfg.postprocessing.compute_modes = 1;
cfg.postprocessing.compute_modes_initial = 0;
cfg.postprocessing.visualize_live = false;
cfg.postprocessing.visualize_modes.enabled = false;
cfg.postprocessing.visualize_modes.count = 0;
cfg.postprocessing.visualize_topology_modes.enabled = false;
cfg.postprocessing.visualize_topology_modes.count = 0;
cfg.postprocessing.save_snapshot_image = false;
cfg.postprocessing.save_final_image = false;
cfg.postprocessing.save_frequency_iterations = false;
cfg.postprocessing.write_correlation_table = false;
cfg.postprocessing.correlation.enabled = false;
cfg.postprocessing.correlation.initial_count = 0;
cfg.postprocessing.correlation.topology_count = 0;
cfg.postprocessing.correlation.metric = 'mac';
cfg.postprocessing.correlation.write_csv = false;
end

function result = localInitCase()
result = struct();
result.study = '';
result.implementation_language = 'MATLAB';
result.scope = '';
result.mesh_label = '';
result.mesh = struct('nelx', NaN, 'nely', NaN);
result.alpha = NaN;
result.source_config = '';
result.case_config = '';
result.authoritative_load = '';
result.filter_physical_radius = NaN;
result.criteria = struct();
result.solver_success = false;
result.accepted = false;
result.classification = 'implementation failure';
result.exception = '';
result.iterations = NaN;
result.iteration_cap = NaN;
result.final = struct( ...
    'omega_rad_s', NaN(3,1), ...
    'frequency_hz', NaN(3,1), ...
    'grayness', NaN, ...
    'volume', NaN, ...
    'feasibility', NaN, ...
    'design_change', NaN, ...
    'tracked_mode_index', NaN, ...
    'tracked_mode_mac', NaN, ...
    'tracked_mode_omega', NaN);
result.topology_quality = struct('pass', false, 'issues', {{'not evaluated'}}, ...
    'metrics', struct(), 'thresholds', struct());
result.a5_lowest_mode_check = struct( ...
    'pass', false, ...
    'tracked_mode_index', NaN, ...
    'tracked_mode_omega_rad_s', NaN, ...
    'lowest_mode_omega_rad_s', NaN, ...
    'modes_below_tracked_count', NaN, ...
    'message', '');
result.artifacts = struct();
result.timing_total_s = NaN;
result.timing_per_iter_s = NaN;
result.peak_memory_MB = NaN;
end

function localValidateDiagnostics(info, nIter)
if ~isfield(info, 'cr2_history') || ~isfield(info, 'cr2_final_tracking') || ...
        ~isfield(info, 'gate_a0')
    error('Exp3Mesh:MissingDiagnostics', ...
        'Required Gate A0 / history / mode tracking diagnostics are missing.');
end
required = {'objective','frequency','design_change','feasibility','grayness', ...
    'volume','tracked_mode_index','tracked_mode_mac','tracked_mode_omega'};
for k = 1:numel(required)
    field = required{k};
    if ~isfield(info.cr2_history, field) || size(info.cr2_history.(field), 1) ~= nIter
        error('Exp3Mesh:InvalidHistory', ...
            'History %s is missing or has the wrong length.', field);
    end
end
if ~isfield(info.cr2_final_tracking, 'frequencies') || ...
        ~isfield(info.cr2_final_tracking, 'mac_values')
    error('Exp3Mesh:InvalidTracking', ...
        'Final tracking frequencies or MAC values are missing.');
end
end

function classification = localClassifyCase(result, cfg, criteria)
if ~result.solver_success
    classification = 'implementation failure';
    return;
end
if ~isfield(result, 'topology_quality') || ~isfield(result.topology_quality, 'pass') || ...
        ~result.topology_quality.pass
    classification = 'topology invalid';
    return;
end
if ~isfinite(result.final.tracked_mode_mac) || ...
        result.final.tracked_mode_mac < criteria.mac_threshold || ...
        ~isfinite(result.final.tracked_mode_index) || result.final.tracked_mode_index < 1 || ...
        ~result.a5_lowest_mode_check.pass
    classification = 'mode invalid';
    return;
end
if result.iterations >= double(cfg.optimization.max_iters)
    classification = 'capped';
    return;
end
if result.final.design_change <= criteria.design_change_tolerance && ...
        result.final.feasibility <= criteria.feasibility_tolerance
    classification = 'accepted';
else
    classification = 'capped';
end
end

function comparison = localCompareMeshes(caseResults, outDir, repoRoot)
comparison = struct();
comparison.available = false;
comparison.reason = '';
comparison.tracked_mode_match = false;
comparison.relative_tracked_omega_change = NaN;
comparison.relative_omega_change = NaN(3,1);
comparison.iteration_count_ratio_fine_over_coarse = NaN;
comparison.grayness_difference = NaN;
comparison.constraint_residual_difference = NaN;
comparison.topology_difference_metrics = struct();
comparison.artifacts = struct();

if numel(caseResults) ~= 2 || any(~[caseResults.solver_success])
    comparison.reason = 'one or both mesh cases did not complete successfully';
    return;
end

coarse = caseResults(1);
fine = caseResults(2);
comparison.available = true;
comparison.tracked_mode_match = coarse.final.tracked_mode_index == fine.final.tracked_mode_index;
comparison.relative_tracked_omega_change = abs(fine.final.tracked_mode_omega - ...
    coarse.final.tracked_mode_omega) / max(abs(coarse.final.tracked_mode_omega), eps);
comparison.relative_omega_change = abs(fine.final.omega_rad_s(:) - coarse.final.omega_rad_s(:)) ./ ...
    max(abs(coarse.final.omega_rad_s(:)), eps);
comparison.iteration_count_ratio_fine_over_coarse = fine.iterations / max(coarse.iterations, eps);
comparison.grayness_difference = fine.final.grayness - coarse.final.grayness;
comparison.constraint_residual_difference = fine.final.feasibility - coarse.final.feasibility;

coarseTopo = localReadTopologyFromArtifact(coarse, repoRoot);
fineTopo = localReadTopologyFromArtifact(fine, repoRoot);
metrics = localTopologyMetrics(coarseTopo, fineTopo);
comparison.topology_difference_metrics = metrics;
comparison.artifacts.topology_difference_metrics_json = localRelativePath( ...
    fullfile(outDir, 'exp3_authoritative_topology_difference_metrics.json'), repoRoot);
end

function metrics = localTopologyMetrics(coarseTopo, fineTopo)
fineMapped = localMapFineToCoarse(fineTopo, size(coarseTopo));
d = fineMapped - coarseTopo;
c = corrcoef(coarseTopo(:), fineMapped(:));
if numel(c) >= 4
    corrVal = c(1,2);
else
    corrVal = NaN;
end
coarseBin = coarseTopo >= 0.5;
fineBin = fineMapped >= 0.5;
unionCount = nnz(coarseBin | fineBin);
if unionCount == 0
    jaccard = NaN;
else
    jaccard = nnz(coarseBin & fineBin) / unionCount;
end
metrics = struct( ...
    'coarse_shape_nely_by_nelx', size(coarseTopo), ...
    'fine_shape_nely_by_nelx', size(fineTopo), ...
    'mapped_fine_shape_nely_by_nelx', size(fineMapped), ...
    'mean_abs_difference', mean(abs(d(:))), ...
    'root_mean_square_difference', sqrt(mean(d(:).^2)), ...
    'max_abs_difference', max(abs(d(:))), ...
    'correlation', corrVal, ...
    'binary_threshold_0_5_jaccard', jaccard, ...
    'coarse_volume', mean(coarseTopo(:)), ...
    'mapped_fine_volume', mean(fineMapped(:)), ...
    'volume_difference', mean(fineMapped(:)) - mean(coarseTopo(:)));
end

function mapped = localMapFineToCoarse(fineTopo, coarseSize)
fineSize = size(fineTopo);
ry = fineSize(1) / coarseSize(1);
rx = fineSize(2) / coarseSize(2);
if abs(ry - round(ry)) > 1e-12 || abs(rx - round(rx)) > 1e-12
    error('Exp3Mesh:NonIntegerMeshRatio', ...
        'Fine mesh cannot be block-averaged to coarse mesh with integer ratios.');
end
ry = round(ry);
rx = round(rx);
mapped = zeros(coarseSize);
for iy = 1:coarseSize(1)
    rows = (iy-1)*ry + (1:ry);
    for ix = 1:coarseSize(2)
        cols = (ix-1)*rx + (1:rx);
        block = fineTopo(rows, cols);
        mapped(iy, ix) = mean(block(:));
    end
end
end

function topo = localReadTopologyFromArtifact(result, repoRoot)
path = fullfile(repoRoot, result.artifacts.topology_csv);
topo = readmatrix(path);
end

function classification = localClassifyStudy(caseResults, comparison, criteria)
classes = string({caseResults.classification});
if any(classes == "implementation failure")
    classification = 'inconclusive/capped/mode/topology invalid';
    return;
end
if any(classes == "capped") || any(classes == "mode invalid") || any(classes == "topology invalid")
    classification = 'inconclusive/capped/mode/topology invalid';
    return;
end
if ~comparison.available
    classification = 'inconclusive/capped/mode/topology invalid';
    return;
end
if all(classes == "accepted") && comparison.tracked_mode_match && ...
        comparison.relative_tracked_omega_change <= criteria.tracked_omega_relative_tolerance && ...
        comparison.topology_difference_metrics.correlation >= criteria.topology_correlation_minimum && ...
        comparison.topology_difference_metrics.mean_abs_difference <= criteria.topology_mean_abs_difference_maximum
    classification = 'passed mesh convergence';
else
    classification = 'failed mesh convergence';
end
end

function check = localA5LowestModeCheck(info)
freqs = info.cr2_final_tracking.frequencies(:);
trackedIndex = double(info.cr2_final_tracking.tracked_mode_index);
trackedOmega = NaN;
if isfinite(trackedIndex) && trackedIndex >= 1 && trackedIndex <= numel(freqs)
    trackedOmega = freqs(trackedIndex);
end
finiteFreqs = freqs(isfinite(freqs));
if isempty(finiteFreqs)
    lowestOmega = NaN;
else
    lowestOmega = finiteFreqs(1);
end
countBelow = max(0, trackedIndex - 1);
passes = isfinite(trackedIndex) && trackedIndex == 1 && isfinite(trackedOmega);
if passes
    msg = 'tracked mode is the lowest computed mode';
else
    msg = sprintf('tracked mode index %.0f leaves %.0f computed mode(s) below it', ...
        trackedIndex, countBelow);
end
check = struct('pass', passes, ...
    'tracked_mode_index', trackedIndex, ...
    'tracked_mode_omega_rad_s', trackedOmega, ...
    'lowest_mode_omega_rad_s', lowestOmega, ...
    'modes_below_tracked_count', countBelow, ...
    'message', msg);
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
freqNames = arrayfun(@(k) sprintf('omega_%d_rad_s', k), 1:size(freq,2), ...
    'UniformOutput', false);
freqTable = array2table([iter, freq], 'VariableNames', [{'iteration'}, freqNames]);
writetable(freqTable, paths.frequency_csv);

modeTable = table(iter, h.tracked_mode_index(:), h.tracked_mode_mac(:), ...
    h.tracked_mode_omega(:), ...
    'VariableNames', {'iteration','tracked_mode_index','tracked_mode_mac', ...
    'tracked_mode_omega_rad_s'});
writetable(modeTable, paths.mode_tracking_csv);

feasTable = table(iter, h.feasibility(:), ...
    repmat(double(cfg.optimization.volume_fraction), n, 1), h.volume(:), ...
    'VariableNames', {'iteration','feasibility','volume_fraction_target','volume'});
writetable(feasTable, paths.feasibility_csv);

grayTable = table(iter, h.grayness(:), 'VariableNames', {'iteration','grayness'});
writetable(grayTable, paths.grayness_csv);
end

function localWriteTopology(paths, xFinal, cfg, label)
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
title(ax, sprintf('Exp3 authoritative %s alpha=1.00 topology', label), ...
    'Interpreter', 'none');
try
    exportgraphics(fig, paths.topology_png, 'Resolution', 180, 'BackgroundColor', 'white');
catch
    print(fig, paths.topology_png, '-dpng', '-r180');
end
close(fig);
end

function localWriteA5(paths, check)
t = table(check.pass, check.tracked_mode_index, check.tracked_mode_omega_rad_s, ...
    check.lowest_mode_omega_rad_s, check.modes_below_tracked_count, string(check.message), ...
    'VariableNames', {'pass','tracked_mode_index','tracked_mode_omega_rad_s', ...
    'lowest_mode_omega_rad_s','modes_below_tracked_count','message'});
writetable(t, paths.a5_lowest_mode_check_csv);
localWriteJson(paths.a5_lowest_mode_check_json, check);
end

function paths = localRootPaths(outDir)
prefix = fullfile(outDir, 'exp3_authoritative_mesh_convergence');
paths = struct( ...
    'log', [prefix, '_run.log'], ...
    'mat', [prefix, '_result.mat'], ...
    'result_json', [prefix, '_result.json'], ...
    'summary', [prefix, '_summary.md'], ...
    'manifest', [prefix, '_manifest.json'], ...
    'topology_metrics_json', fullfile(outDir, 'exp3_authoritative_topology_difference_metrics.json'));
end

function paths = localCasePaths(caseDir, label)
tag = strrep(label, 'x', 'x');
prefix = fullfile(caseDir, ['exp3_authoritative_', tag]);
paths = struct( ...
    'config', [prefix, '_config.json'], ...
    'mat', [prefix, '_result.mat'], ...
    'result_json', [prefix, '_result.json'], ...
    'convergence_csv', [prefix, '_convergence_history.csv'], ...
    'frequency_csv', [prefix, '_frequency_history.csv'], ...
    'mode_tracking_csv', [prefix, '_mode_tracking.csv'], ...
    'feasibility_csv', [prefix, '_feasibility_history.csv'], ...
    'grayness_csv', [prefix, '_grayness_history.csv'], ...
    'topology_csv', [prefix, '_topology.csv'], ...
    'topology_json', [prefix, '_topology.json'], ...
    'topology_png', [prefix, '_topology.png'], ...
    'topology_quality_json', [prefix, '_topology_quality.json'], ...
    'a5_lowest_mode_check_csv', [prefix, '_a5_lowest_mode_check.csv'], ...
    'a5_lowest_mode_check_json', [prefix, '_a5_lowest_mode_check.json'], ...
    'manifest', [prefix, '_manifest.json']);
end

function localWriteCaseManifest(path, result, paths, repoRoot)
manifest = struct();
manifest.study = result.study;
manifest.implementation_language = result.implementation_language;
manifest.scope = result.scope;
manifest.excluded = {'Exp2 alpha sweep','CR2','A4','S1','P1','Python','manuscript edits'};
manifest.mesh_label = result.mesh_label;
manifest.mesh = result.mesh;
manifest.alpha = result.alpha;
manifest.classification = result.classification;
manifest.accepted = result.accepted;
manifest.acceptance = struct( ...
    'not_capped', result.iterations < result.iteration_cap, ...
    'design_change_ok', result.final.design_change <= result.criteria.design_change_tolerance, ...
    'feasibility_ok', result.final.feasibility <= result.criteria.feasibility_tolerance, ...
    'tracked_mac_ok', result.final.tracked_mode_mac >= result.criteria.mac_threshold, ...
    'a5_lowest_mode_ok', result.a5_lowest_mode_check.pass);
manifest.authoritative_load = result.authoritative_load;
manifest.filter_physical_radius = result.filter_physical_radius;
manifest.artifacts = localRelativeStruct(paths, repoRoot);
manifest.created_utc = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'));
localWriteJson(path, manifest);
end

function localWriteRootManifest(path, study, rootPaths, repoRoot)
manifest = struct();
manifest.study = study.study;
manifest.implementation_language = study.implementation_language;
manifest.scope = study.scope;
manifest.excluded = study.excluded;
manifest.alpha = study.alpha;
manifest.criteria = study.criteria;
manifest.classification = study.classification;
manifest.created_utc = study.created_utc;
manifest.artifacts = localRelativeStruct(rootPaths, repoRoot);
manifest.cases = localManifestCaseEntries(study.cases);
manifest.comparison = study.comparison;
localWriteJson(path, manifest);
end

function entries = localManifestCaseEntries(caseResults)
entries = repmat(struct('mesh_label', '', 'classification', '', ...
    'accepted', false, 'iterations', NaN, 'iteration_cap', NaN, ...
    'omega_rad_s', NaN(3,1), 'design_change', NaN, 'feasibility', NaN, ...
    'grayness', NaN, 'tracked_mode_index', NaN, 'tracked_mac', NaN, ...
    'artifacts', struct()), numel(caseResults), 1);
for i = 1:numel(caseResults)
    r = caseResults(i);
    entries(i).mesh_label = r.mesh_label;
    entries(i).classification = r.classification;
    entries(i).accepted = r.accepted;
    entries(i).iterations = r.iterations;
    entries(i).iteration_cap = r.iteration_cap;
    entries(i).omega_rad_s = r.final.omega_rad_s;
    entries(i).design_change = r.final.design_change;
    entries(i).feasibility = r.final.feasibility;
    entries(i).grayness = r.final.grayness;
    entries(i).tracked_mode_index = r.final.tracked_mode_index;
    entries(i).tracked_mac = r.final.tracked_mode_mac;
    entries(i).artifacts = r.artifacts;
end
end

function localWriteSummary(path, study)
lines = {
    '# Exp3 Authoritative Mesh Convergence Summary'
    ''
    'Implementation language: MATLAB.'
    'Scope: Exp3 alpha=1.00 mesh convergence only. No Exp2 alpha sweep, CR2, A4, S1, P1, Python, or manuscript edits.'
    ''
    sprintf('Classification: %s', study.classification)
    sprintf('Authoritative load: %s', study.cases(1).authoritative_load)
    sprintf('Filter physical radius: %.12g', study.cases(1).filter_physical_radius)
    sprintf('Predeclared criteria: both meshes accepted; tracked mode match; tracked MAC >= %.3g; relative tracked omega change <= %.3g; topology correlation >= %.3g; topology MAD <= %.3g.', ...
        study.criteria.mac_threshold, study.criteria.tracked_omega_relative_tolerance, ...
        study.criteria.topology_correlation_minimum, study.criteria.topology_mean_abs_difference_maximum)
    ''
    '| mesh | classification | iterations | omega rad/s | tracked mode | MAC | design_change | feasibility | grayness | A5 |'
    '|---|---|---:|---|---:|---:|---:|---:|---:|---|'
    };
for i = 1:numel(study.cases)
    r = study.cases(i);
    lines{end+1} = sprintf('| %s | %s | %.0f/%.0f | [%.6f, %.6f, %.6f] | %.0f | %.6f | %.6g | %.3g | %.6g | %s |', ...
        r.mesh_label, r.classification, r.iterations, r.iteration_cap, ...
        r.final.omega_rad_s(1), r.final.omega_rad_s(2), r.final.omega_rad_s(3), ...
        r.final.tracked_mode_index, r.final.tracked_mode_mac, ...
        r.final.design_change, r.final.feasibility, r.final.grayness, ...
        localPassFail(r.a5_lowest_mode_check.pass)); %#ok<AGROW>
end
lines{end+1} = '';
if study.comparison.available
    m = study.comparison.topology_difference_metrics;
    lines{end+1} = sprintf('Relative tracked omega change: %.12g', study.comparison.relative_tracked_omega_change);
    lines{end+1} = sprintf('Tracked mode match: %d', study.comparison.tracked_mode_match);
    lines{end+1} = sprintf('Topology correlation: %.12g', m.correlation);
    lines{end+1} = sprintf('Topology mean absolute difference: %.12g', m.mean_abs_difference);
    lines{end+1} = sprintf('Topology RMS difference: %.12g', m.root_mean_square_difference);
    lines{end+1} = sprintf('Grayness difference fine-minus-coarse: %.12g', study.comparison.grayness_difference);
    lines{end+1} = sprintf('Constraint residual difference fine-minus-coarse: %.12g', study.comparison.constraint_residual_difference);
end
lines{end+1} = '';
lines{end+1} = 'This is experiment evidence only and makes no manuscript claim.';
localWriteText(path, strjoin(lines, newline));
end

function value = localFinalHistoryValue(s, path)
value = NaN;
cur = s;
for k = 1:numel(path)
    if ~isstruct(cur) || ~isfield(cur, path{k})
        return;
    end
    cur = cur.(path{k});
end
if isnumeric(cur) && ~isempty(cur)
    value = double(cur(end));
end
end

function safe = localJsonSafe(data)
safe = data;
if isfield(safe, 'exception') && strlength(string(safe.exception)) > 4000
    safe.exception = extractBefore(string(safe.exception), 4001);
end
if isfield(safe, 'cases')
    for i = 1:numel(safe.cases)
        if isfield(safe.cases(i), 'exception') && ...
                strlength(string(safe.cases(i).exception)) > 4000
            safe.cases(i).exception = extractBefore(string(safe.cases(i).exception), 4001);
        end
    end
end
end

function localWriteJson(path, data)
try
    txt = jsonencode(data, PrettyPrint=true);
catch
    txt = jsonencode(data);
end
localWriteText(path, txt);
end

function localWriteText(path, txt)
fid = fopen(path, 'w');
if fid < 0
    error('Exp3Mesh:WriteFailed', 'Could not open file for writing: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', char(txt));
end

function rels = localRelativeStruct(paths, repoRoot)
fields = fieldnames(paths);
rels = struct();
for k = 1:numel(fields)
    rels.(fields{k}) = localRelativePath(paths.(fields{k}), repoRoot);
end
end

function rel = localRelativePath(path, repoRoot)
path = char(string(path));
repoRoot = char(string(repoRoot));
if startsWith(path, [repoRoot, filesep])
    rel = path(numel(repoRoot)+2:end);
else
    rel = path;
end
end

function txt = localPassFail(tf)
if tf
    txt = 'pass';
else
    txt = 'fail';
end
end
