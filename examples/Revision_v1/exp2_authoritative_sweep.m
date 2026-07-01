function sweep = exp2_authoritative_sweep(outDir)
%EXP2_AUTHORITATIVE_SWEEP  Full Exp2 authoritative alpha sweep.
%
% Implementation language: MATLAB only.
%
% Runs only Exp2 clamped-beam cases:
%   alpha = [1, 0.75, 0.5, 0.25, 0]
% using the authoritative load F(x)=omega0^2*M(x)*Phi0 with a solid
% reference, omitted load sensitivity, and Gate A0 diagnostics enabled.

if nargin < 1 || isempty(outDir)
    scriptDir = fileparts(mfilename('fullpath'));
    outDir = fullfile(scriptDir, 'output', 'exp2_authoritative_sweep');
end
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));

rootPaths = localRootArtifactPaths(outDir);
diary(rootPaths.log);
cleanupDiary = onCleanup(@() diary('off'));

fprintf('Exp2 authoritative alpha sweep started: %s\n', ...
    char(datetime('now', 'TimeZone', 'UTC')));
fprintf('Implementation language: MATLAB\n');
fprintf('Scope: Exp2 only; no Exp3, A4, S1, P1, CR2, Python, or manuscript edits.\n');

sourceConfig = fullfile(scriptDir, 'clamped_beam_200x25.json');
if ~isfile(sourceConfig)
    error('Exp2Sweep:MissingConfig', 'Missing source config: %s', sourceConfig);
end

alphaVals = [1.0, 0.75, 0.5, 0.25, 0.0];
baseCfg = jsondecode(fileread(sourceConfig));
criteria = struct( ...
    'mac_threshold', 0.8, ...
    'feasibility_tolerance', 1e-8, ...
    'design_change_tolerance', double(baseCfg.optimization.convergence_tol));

caseResults = repmat(localInitializeResult(), numel(alphaVals), 1);
for i = 1:numel(alphaVals)
    alpha = alphaVals(i);
    caseDir = fullfile(outDir, localAlphaTag(alpha));
    if exist(caseDir, 'dir') ~= 7
        mkdir(caseDir);
    end
    paths = localCaseArtifactPaths(caseDir, alpha);
    cfg = localBuildCaseConfig(sourceConfig, alpha);
    localWriteJson(paths.config, cfg);

    result = localInitializeResult();
    result.study = 'Exp2 authoritative clamped-beam alpha sweep';
    result.implementation_language = 'MATLAB';
    result.source_config = localRelativePath(sourceConfig, repoRoot);
    result.case_config = localRelativePath(paths.config, repoRoot);
    result.authoritative_load = 'F(x) = omega0^2 * M(x) * Phi0';
    result.alpha = alpha;
    result.mesh = cfg.domain.mesh;
    result.load_sensitivity = char(cfg.optimization.load_sensitivity);
    result.gate_a0_diagnostics = cfg.optimization.gate_a0_diagnostics;
    result.criteria = criteria;
    result.iteration_cap = double(cfg.optimization.max_iters);
    result.artifacts = localRelativeArtifactStruct(paths, repoRoot);

    xFinal = [];
    omega = NaN(3,1);
    info = struct();
    tIter = NaN; %#ok<NASGU>
    memUsage = NaN; %#ok<NASGU>
    runTic = tic;

    try
        fprintf('\n=== Exp2 authoritative alpha=%.2f ===\n', alpha);
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

        localValidateDiagnostics(info, nIter);
        result.a5_lowest_mode_check = localA5LowestModeCheck(info);
        result.spectrum_below_tracked = localSpectrumBelowTracked(info);

        localWriteHistories(paths, info, cfg);
        localWriteTopology(paths, xFinal, cfg, alpha);
        localWriteSpectrumArtifacts(paths, result.spectrum_below_tracked);
        localWriteA5Artifacts(paths, result.a5_lowest_mode_check);

        result.classification = localClassify(result, cfg, criteria);
        result.accepted = strcmp(result.classification, 'accepted');
    catch ME
        result.timing_total_s = toc(runTic);
        result.exception = getReport(ME, 'extended', 'hyperlinks', 'off');
        result.classification = 'implementation failure';
        result.accepted = false;
        fprintf(2, '\nExp2 alpha=%.2f exception preserved:\n%s\n', alpha, result.exception);
    end

    save(paths.mat, 'result', 'cfg', 'xFinal', 'omega', 'info', '-v7.3');
    localWriteJson(paths.result_json, localJsonSafeResult(result));
    localWriteCaseManifest(paths.manifest, result, paths, repoRoot);

    caseResults(i) = result;
    fprintf('Exp2 alpha=%.2f classification: %s\n', alpha, result.classification);
end

sweep = struct();
sweep.study = 'Exp2 authoritative clamped-beam alpha sweep';
sweep.implementation_language = 'MATLAB';
sweep.scope = 'Exp2 full authoritative alpha sweep only';
sweep.excluded = {'Exp3','A4','S1','P1','CR2','Python','manuscript edits'};
sweep.alpha_values = alphaVals(:);
sweep.criteria = criteria;
sweep.cases = caseResults;
sweep.all_accepted = all(strcmp({caseResults.classification}, 'accepted'));
sweep.created_utc = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'));
sweep.artifacts = localRelativeArtifactStruct(rootPaths, repoRoot);

save(rootPaths.mat, 'sweep', '-v7.3');
localWriteJson(rootPaths.result_json, localJsonSafeResult(sweep));
localWriteRootManifest(rootPaths.manifest, sweep, caseResults, rootPaths, repoRoot);
localWriteSummary(rootPaths.summary, sweep, caseResults);

fprintf('\nExp2 authoritative sweep all accepted: %d\n', sweep.all_accepted);
fprintf('Summary: %s\n', rootPaths.summary);
fprintf('Manifest: %s\n', rootPaths.manifest);
diary('off');
end

function result = localInitializeResult()
result = struct();
result.study = '';
result.implementation_language = 'MATLAB';
result.source_config = '';
result.case_config = '';
result.authoritative_load = '';
result.alpha = NaN;
result.mesh = struct('nelx', NaN, 'nely', NaN);
result.load_sensitivity = '';
result.gate_a0_diagnostics = false;
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
result.a5_lowest_mode_check = struct( ...
    'pass', false, ...
    'tracked_mode_index', NaN, ...
    'tracked_mode_omega_rad_s', NaN, ...
    'lowest_mode_omega_rad_s', NaN, ...
    'modes_below_tracked_count', NaN, ...
    'message', '');
result.spectrum_below_tracked = struct( ...
    'tracked_mode_index', NaN, ...
    'modes', struct('mode_index', {}, 'omega_rad_s', {}, 'frequency_hz', {}, 'mac_to_reference_mode1', {}));
result.artifacts = struct();
result.timing_total_s = NaN;
result.timing_per_iter_s = NaN;
result.peak_memory_MB = NaN;
end

function cfg = localBuildCaseConfig(sourceConfig, alpha)
cfg = jsondecode(fileread(sourceConfig));
cfg.meta.name = sprintf('Exp2 authoritative clamped beam alpha=%.2f (200x25)', alpha);
cfg.meta.notes = ['Full Exp2 alpha sweep case. Authoritative load ', ...
    'F(x)=omega0^2*M(x)*Phi0, solid reference, load_sensitivity=complete, ', ...
    'Gate A0 diagnostics enabled.'];

cfg.domain.load_cases(1).name = sprintf('alpha%.2f_solid_reference_mode_1', alpha);
cfg.domain.load_cases(1).factor = alpha;
cfg.domain.load_cases(1).loads(1).type = 'semi_harmonic';
cfg.domain.load_cases(1).loads(1).mode = 1;
cfg.domain.load_cases(1).loads(1).factor = 1.0;
cfg.domain.load_cases(2).name = sprintf('alpha%.2f_solid_reference_mode_2', 1 - alpha);
cfg.domain.load_cases(2).factor = 1 - alpha;
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

function localValidateDiagnostics(info, nIter)
if ~isfield(info, 'cr2_history') || ~isfield(info, 'cr2_final_tracking') || ...
        ~isfield(info, 'gate_a0')
    error('Exp2Sweep:MissingDiagnostics', ...
        'Sweep case did not return required Gate A0 / mode-tracking diagnostics.');
end
required = {'objective','frequency','design_change','feasibility','grayness', ...
    'volume','tracked_mode_index','tracked_mode_mac','tracked_mode_omega'};
for k = 1:numel(required)
    field = required{k};
    if ~isfield(info.cr2_history, field) || size(info.cr2_history.(field), 1) ~= nIter
        error('Exp2Sweep:InvalidHistory', ...
            'History %s is missing or has the wrong length.', field);
    end
end
if ~isfield(info.cr2_final_tracking, 'frequencies') || ...
        ~isfield(info.cr2_final_tracking, 'mac_values')
    error('Exp2Sweep:InvalidTracking', ...
        'Final tracking frequencies or MAC values are missing.');
end
end

function classification = localClassify(result, cfg, criteria)
if ~result.solver_success
    classification = 'implementation failure';
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

function check = localA5LowestModeCheck(info)
freqs = info.cr2_final_tracking.frequencies(:);
trackedIndex = double(info.cr2_final_tracking.tracked_mode_index);
trackedOmega = NaN;
if isfinite(trackedIndex) && trackedIndex >= 1 && trackedIndex <= numel(freqs)
    trackedOmega = freqs(trackedIndex);
end
lowestOmega = NaN;
finiteFreqs = freqs(isfinite(freqs));
if ~isempty(finiteFreqs)
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
check = struct( ...
    'pass', passes, ...
    'tracked_mode_index', trackedIndex, ...
    'tracked_mode_omega_rad_s', trackedOmega, ...
    'lowest_mode_omega_rad_s', lowestOmega, ...
    'modes_below_tracked_count', countBelow, ...
    'message', msg);
end

function spectrum = localSpectrumBelowTracked(info)
freqs = info.cr2_final_tracking.frequencies(:);
macVals = info.cr2_final_tracking.mac_values(:);
trackedIndex = double(info.cr2_final_tracking.tracked_mode_index);
if ~isfinite(trackedIndex) || trackedIndex < 1
    below = zeros(0,1);
else
    below = (1:max(0, trackedIndex - 1))';
end
modes = struct('mode_index', {}, 'omega_rad_s', {}, 'frequency_hz', {}, 'mac_to_reference_mode1', {});
for i = 1:numel(below)
    idx = below(i);
    macVal = NaN;
    if idx <= numel(macVals)
        macVal = macVals(idx);
    end
    modes(i).mode_index = idx;
    modes(i).omega_rad_s = freqs(idx);
    modes(i).frequency_hz = freqs(idx) / (2*pi);
    modes(i).mac_to_reference_mode1 = macVal;
end
spectrum = struct('tracked_mode_index', trackedIndex, 'modes', modes);
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

grayTable = table(iter, h.grayness(:), ...
    'VariableNames', {'iteration','grayness'});
writetable(grayTable, paths.grayness_csv);
end

function localWriteTopology(paths, xFinal, cfg, alpha)
nelx = double(cfg.domain.mesh.nelx);
nely = double(cfg.domain.mesh.nely);
topology = reshape(xFinal(:), nely, nelx);
writematrix(topology, paths.topology_csv);
localWriteJson(paths.topology_json, struct('nelx', nelx, 'nely', nely, ...
    'density_row_major_nely_by_nelx', topology));

img = buildTopologyDisplayImage(xFinal(:), nelx, nely, 'regular', true);
fig = figure('Color', 'white', 'Visible', 'off');
ax = axes('Parent', fig);
imagesc(ax, 1 - img);
axis(ax, 'equal', 'off');
colormap(ax, gray);
title(ax, sprintf('Exp2 authoritative alpha=%.2f topology', alpha), 'Interpreter', 'none');
try
    exportgraphics(fig, paths.topology_png, 'Resolution', 180, 'BackgroundColor', 'white');
catch
    print(fig, paths.topology_png, '-dpng', '-r180');
end
close(fig);
end

function localWriteSpectrumArtifacts(paths, spectrum)
n = numel(spectrum.modes);
modeIndex = zeros(n,1);
omega = zeros(n,1);
freqHz = zeros(n,1);
mac = zeros(n,1);
for i = 1:n
    modeIndex(i) = spectrum.modes(i).mode_index;
    omega(i) = spectrum.modes(i).omega_rad_s;
    freqHz(i) = spectrum.modes(i).frequency_hz;
    mac(i) = spectrum.modes(i).mac_to_reference_mode1;
end
t = table(modeIndex, omega, freqHz, mac, ...
    'VariableNames', {'mode_index','omega_rad_s','frequency_hz','mac_to_reference_mode1'});
writetable(t, paths.spectrum_below_tracked_csv);
localWriteJson(paths.spectrum_below_tracked_json, spectrum);
end

function localWriteA5Artifacts(paths, check)
t = table(check.pass, check.tracked_mode_index, check.tracked_mode_omega_rad_s, ...
    check.lowest_mode_omega_rad_s, check.modes_below_tracked_count, string(check.message), ...
    'VariableNames', {'pass','tracked_mode_index','tracked_mode_omega_rad_s', ...
    'lowest_mode_omega_rad_s','modes_below_tracked_count','message'});
writetable(t, paths.a5_lowest_mode_check_csv);
localWriteJson(paths.a5_lowest_mode_check_json, check);
end

function paths = localRootArtifactPaths(outDir)
prefix = fullfile(outDir, 'exp2_authoritative_sweep');
paths = struct( ...
    'log', [prefix, '_run.log'], ...
    'mat', [prefix, '_result.mat'], ...
    'result_json', [prefix, '_result.json'], ...
    'summary', fullfile(outDir, 'exp2_authoritative_summary.md'), ...
    'manifest', [prefix, '_manifest.json']);
end

function paths = localCaseArtifactPaths(caseDir, alpha)
prefix = fullfile(caseDir, ['exp2_authoritative_', localAlphaTag(alpha)]);
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
    'spectrum_below_tracked_csv', [prefix, '_spectrum_below_tracked.csv'], ...
    'spectrum_below_tracked_json', [prefix, '_spectrum_below_tracked.json'], ...
    'a5_lowest_mode_check_csv', [prefix, '_a5_lowest_mode_check.csv'], ...
    'a5_lowest_mode_check_json', [prefix, '_a5_lowest_mode_check.json'], ...
    'manifest', [prefix, '_manifest.json']);
end

function tag = localAlphaTag(alpha)
tag = sprintf('alpha_%0.2f', alpha);
tag = strrep(tag, '.', '_');
end

function rel = localRelativeArtifactStruct(paths, repoRoot)
fields = fieldnames(paths);
rel = struct();
for k = 1:numel(fields)
    rel.(fields{k}) = localRelativePath(paths.(fields{k}), repoRoot);
end
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

function localWriteCaseManifest(path, result, paths, repoRoot)
manifest = struct();
manifest.study = result.study;
manifest.implementation_language = result.implementation_language;
manifest.scope = 'Exp2 authoritative alpha sweep case';
manifest.excluded = {'Exp3','A4','S1','P1','CR2','Python','manuscript edits'};
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
manifest.mesh = result.mesh;
manifest.artifacts = localRelativeArtifactStruct(paths, repoRoot);
manifest.created_utc = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'));
localWriteJson(path, manifest);
end

function localWriteRootManifest(path, sweep, caseResults, rootPaths, repoRoot)
manifest = struct();
manifest.study = sweep.study;
manifest.implementation_language = sweep.implementation_language;
manifest.scope = sweep.scope;
manifest.excluded = sweep.excluded;
manifest.alpha_values = sweep.alpha_values;
manifest.criteria = sweep.criteria;
manifest.all_accepted = sweep.all_accepted;
manifest.created_utc = sweep.created_utc;
manifest.artifacts = localRelativeArtifactStruct(rootPaths, repoRoot);
manifest.cases = localManifestCaseEntries(caseResults);
localWriteJson(path, manifest);
end

function entries = localManifestCaseEntries(caseResults)
entries = repmat(struct( ...
    'alpha', NaN, ...
    'classification', '', ...
    'accepted', false, ...
    'iterations', NaN, ...
    'iteration_cap', NaN, ...
    'design_change', NaN, ...
    'feasibility', NaN, ...
    'tracked_mac', NaN, ...
    'tracked_mode_index', NaN, ...
    'a5_lowest_mode_ok', false, ...
    'artifacts', struct()), numel(caseResults), 1);
for i = 1:numel(caseResults)
    r = caseResults(i);
    entries(i).alpha = r.alpha;
    entries(i).classification = r.classification;
    entries(i).accepted = r.accepted;
    entries(i).iterations = r.iterations;
    entries(i).iteration_cap = r.iteration_cap;
    entries(i).design_change = r.final.design_change;
    entries(i).feasibility = r.final.feasibility;
    entries(i).tracked_mac = r.final.tracked_mode_mac;
    entries(i).tracked_mode_index = r.final.tracked_mode_index;
    entries(i).a5_lowest_mode_ok = r.a5_lowest_mode_check.pass;
    entries(i).artifacts = r.artifacts;
end
end

function localWriteSummary(path, sweep, caseResults)
lines = {
    '# Exp2 Authoritative Alpha Sweep Summary'
    ''
    'Implementation language: MATLAB.'
    'Scope: Exp2 authoritative clamped-beam alpha sweep only. No Python, Exp3, A4, S1, P1, CR2, or manuscript edits.'
    ''
    sprintf('Authoritative load: %s', caseResults(1).authoritative_load)
    sprintf('Mesh/settings: %dx%d, max_iters=%g, design_change_tolerance=%.12g, feasibility_tolerance=%.12g, MAC threshold=%.3g', ...
        caseResults(1).mesh.nelx, caseResults(1).mesh.nely, caseResults(1).iteration_cap, ...
        sweep.criteria.design_change_tolerance, sweep.criteria.feasibility_tolerance, sweep.criteria.mac_threshold)
    ''
    '| alpha | classification | iterations | design_change | feasibility | tracked_mode | tracked_MAC | A5 lowest-mode | grayness | omega1 rad/s |'
    '|---:|---|---:|---:|---:|---:|---:|---|---:|---:|'
    };
for i = 1:numel(caseResults)
    r = caseResults(i);
    lines{end+1} = sprintf('| %.2f | %s | %.0f/%.0f | %.6g | %.3g | %.0f | %.6f | %s | %.6g | %.6f |', ...
        r.alpha, r.classification, r.iterations, r.iteration_cap, ...
        r.final.design_change, r.final.feasibility, r.final.tracked_mode_index, ...
        r.final.tracked_mode_mac, localPassFail(r.a5_lowest_mode_check.pass), ...
        r.final.grayness, r.final.omega_rad_s(1)); %#ok<AGROW>
end
lines{end+1} = '';
lines{end+1} = sprintf('All cases accepted: %d', sweep.all_accepted);
lines{end+1} = '';
lines{end+1} = 'Alpha=0.75 diagnosis:';
lines{end+1} = localAlpha075Diagnosis(caseResults);
lines{end+1} = '';
lines{end+1} = 'No monotonic trend is assumed or used for classification; each alpha is judged only against the declared convergence, feasibility, tracked-MAC, cap, artifact, and A5 lowest-mode checks.';
lines{end+1} = '';
lines{end+1} = 'This is experiment evidence only and makes no manuscript claim.';
localWriteText(path, strjoin(lines, newline));
end

function text = localAlpha075Diagnosis(caseResults)
idx = find(abs([caseResults.alpha] - 0.75) < 1e-12, 1);
if isempty(idx)
    text = '- alpha=0.75 was not run.';
    return;
end
r = caseResults(idx);
text = sprintf(['- alpha=0.75 classification=%s, iterations=%.0f/%.0f, ', ...
    'design_change=%.12g, feasibility=%.12g, tracked_mode_index=%.0f, ', ...
    'tracked_MAC=%.12g, A5_lowest_mode=%d, grayness=%.12g. ', ...
    'The diagnosis is standalone for this case and does not rely on monotonicity across alpha.'], ...
    r.classification, r.iterations, r.iteration_cap, r.final.design_change, ...
    r.final.feasibility, r.final.tracked_mode_index, r.final.tracked_mode_mac, ...
    r.a5_lowest_mode_check.pass, r.final.grayness);
end

function txt = localPassFail(tf)
if tf
    txt = 'pass';
else
    txt = 'fail';
end
end

function safe = localJsonSafeResult(result)
safe = result;
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
    error('Exp2Sweep:WriteFailed', 'Could not open file for writing: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', char(txt));
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
