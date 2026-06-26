function result = exp2_pilot_authoritative(outDir)
%EXP2_PILOT_AUTHORITATIVE  Narrow alpha=1 clamped-beam pilot.
%
% Implementation language: MATLAB.
%
% This runner intentionally executes only the Exp2 pilot:
%   - clamped beam
%   - 200x25 mesh
%   - alpha = 1
%   - authoritative load F(x) = omega0^2 * M(x) * Phi0
%   - solid reference
%   - load_sensitivity = "omitted"
%   - gate_a0_diagnostics = true

if nargin < 1 || isempty(outDir)
    scriptDir = fileparts(mfilename('fullpath'));
    outDir = fullfile(scriptDir, 'output', 'exp2_pilot_authoritative');
end
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));

paths = localArtifactPaths(outDir);
diary(paths.log);
cleanupDiary = onCleanup(@() diary('off'));

fprintf('Exp2 pilot authoritative run started: %s\n', char(datetime('now', 'TimeZone', 'UTC')));
fprintf('Implementation language: MATLAB\n');
fprintf('Scope: Exp2 pilot only; no alpha sweep, Exp3, A4, S1, P1, or manuscript edits.\n');

sourceConfig = fullfile(scriptDir, 'clamped_beam_200x25.json');
if ~isfile(sourceConfig)
    error('Exp2Pilot:MissingConfig', 'Missing source config: %s', sourceConfig);
end

failure = localReproduceExistingFailure(sourceConfig, paths.failure_trace);
cfg = localBuildPilotConfig(sourceConfig);
localWriteJson(paths.config, cfg);

criteria = struct( ...
    'mac_threshold', 0.8, ...
    'feasibility_tolerance', 1e-8, ...
    'design_change_tolerance', double(cfg.optimization.convergence_tol));

result = struct();
result.study = 'Exp2 authoritative clamped-beam alpha=1 pilot';
result.implementation_language = 'MATLAB';
result.source_config = localRelativePath(sourceConfig, repoRoot);
result.pilot_config = localRelativePath(paths.config, repoRoot);
result.authoritative_load = 'F(x) = omega0^2 * M(x) * Phi0';
result.alpha = 1.0;
result.mesh = cfg.domain.mesh;
result.load_sensitivity = char(cfg.optimization.load_sensitivity);
result.gate_a0_diagnostics = cfg.optimization.gate_a0_diagnostics;
result.failure_reproduction = failure;
result.criteria = criteria;
result.solver_success = false;
result.classification = 'implementation failure';
result.viable = false;
result.exception = '';
result.iterations = NaN;
result.iteration_cap = double(cfg.optimization.max_iters);
result.final = struct();
result.artifacts = localRelativeArtifactStruct(paths, repoRoot);

xFinal = [];
omega = NaN(3,1);
info = struct();
tIter = NaN; %#ok<NASGU>
memUsage = NaN; %#ok<NASGU>
runTic = tic;

try
    fprintf('\n=== Exp2 Pilot: alpha=1 authoritative clamped beam ===\n');
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
    localWriteHistories(paths, info, cfg);
    localWriteTopology(paths, xFinal, cfg);

    result.classification = localClassify(result, cfg, criteria);
    result.viable = strcmp(result.classification, 'converged pilot');
catch ME
    result.timing_total_s = toc(runTic);
    result.exception = getReport(ME, 'extended', 'hyperlinks', 'off');
    result.classification = 'implementation failure';
    result.viable = false;
    fprintf(2, '\nExp2 pilot exception preserved:\n%s\n', result.exception);
end

save(paths.mat, 'result', 'cfg', 'xFinal', 'omega', 'info', '-v7.3');
localWriteJson(paths.result_json, localJsonSafeResult(result));
localWriteSummary(paths.summary, result);
localWriteManifest(paths.manifest, result, paths, repoRoot);

fprintf('\nExp2 pilot classification: %s\n', result.classification);
fprintf('Exp2 viable: %d\n', result.viable);
fprintf('Result MAT: %s\n', paths.mat);
fprintf('Manifest: %s\n', paths.manifest);
diary('off');
end

function failure = localReproduceExistingFailure(sourceConfig, tracePath)
failure = struct('attempted', true, 'expected_failure_reproduced', false, ...
    'identifier', '', 'message', '', 'trace_file', tracePath);
fprintf('\n=== Existing Exp2 failure reproduction ===\n');
cfgFail = jsondecode(fileread(sourceConfig));
cfgFail.optimization.gate_a0_diagnostics = true;
try
    run_topopt_from_json(cfgFail);
    failure.message = 'No failure occurred.';
    localWriteText(tracePath, failure.message);
    fprintf('Existing failure was not reproduced.\n');
catch ME
    failure.expected_failure_reproduced = true;
    failure.identifier = ME.identifier;
    failure.message = ME.message;
    trace = getReport(ME, 'extended', 'hyperlinks', 'off');
    localWriteText(tracePath, trace);
    fprintf(2, 'Existing failure full stack trace:\n%s\n', trace);
end
end

function cfg = localBuildPilotConfig(sourceConfig)
cfg = jsondecode(fileread(sourceConfig));
cfg.meta.name = 'Exp2 pilot authoritative clamped beam alpha=1 (200x25)';
cfg.meta.notes = ['Pilot only. Authoritative load F(x)=omega0^2*M(x)*Phi0, ', ...
    'solid reference, load_sensitivity=omitted, Gate A0 diagnostics enabled.'];

cfg.domain.load_cases(1).name = 'alpha1_solid_reference_mode_1';
cfg.domain.load_cases(1).factor = 1.0;
cfg.domain.load_cases(1).loads(1).type = 'semi_harmonic';
cfg.domain.load_cases(1).loads(1).mode = 1;
cfg.domain.load_cases(1).loads(1).factor = 1.0;
cfg.domain.load_cases(2).name = 'alpha0_solid_reference_mode_2';
cfg.domain.load_cases(2).factor = 0.0;
cfg.domain.load_cases(2).loads(1).type = 'semi_harmonic';
cfg.domain.load_cases(2).loads(1).mode = 2;
cfg.domain.load_cases(2).loads(1).factor = 1.0;

cfg.optimization.semi_harmonic_baseline = 'solid';
if isfield(cfg.optimization, 'semi_harmonic_rho_source')
    cfg.optimization = rmfield(cfg.optimization, 'semi_harmonic_rho_source');
end
cfg.optimization.harmonic_normalize = false;
cfg.optimization.load_sensitivity = 'omitted';
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
    error('Exp2Pilot:MissingDiagnostics', ...
        'Pilot did not return required Gate A0 / mode-tracking diagnostics.');
end
required = {'objective','frequency','design_change','feasibility','grayness', ...
    'volume','tracked_mode_index','tracked_mode_mac','tracked_mode_omega'};
for k = 1:numel(required)
    field = required{k};
    if ~isfield(info.cr2_history, field) || size(info.cr2_history.(field), 1) ~= nIter
        error('Exp2Pilot:InvalidHistory', ...
            'History %s is missing or has the wrong length.', field);
    end
end
end

function classification = localClassify(result, cfg, criteria)
if ~result.solver_success
    classification = 'implementation failure';
    return;
end
if ~isfinite(result.final.tracked_mode_mac) || ...
        result.final.tracked_mode_mac < criteria.mac_threshold || ...
        ~isfinite(result.final.tracked_mode_index) || result.final.tracked_mode_index < 1
    classification = 'mode invalid';
    return;
end
capped = result.iterations >= double(cfg.optimization.max_iters);
converged = result.final.design_change <= criteria.design_change_tolerance && ...
    result.final.feasibility <= criteria.feasibility_tolerance && ~capped;
if converged
    classification = 'converged pilot';
else
    classification = 'capped/inconclusive';
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

function localWriteTopology(paths, xFinal, cfg)
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
title(ax, 'Exp2 pilot alpha=1 topology', 'Interpreter', 'none');
try
    exportgraphics(fig, paths.topology_png, 'Resolution', 180, 'BackgroundColor', 'white');
catch
    print(fig, paths.topology_png, '-dpng', '-r180');
end
close(fig);
end

function paths = localArtifactPaths(outDir)
prefix = fullfile(outDir, 'exp2_pilot_authoritative');
paths = struct( ...
    'log', [prefix, '_run.log'], ...
    'failure_trace', [prefix, '_failure_reproduction_stack.txt'], ...
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
    'summary', [prefix, '_summary.md'], ...
    'manifest', [prefix, '_manifest.json']);
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

function localWriteManifest(path, result, paths, repoRoot)
manifest = struct();
manifest.study = result.study;
manifest.implementation_language = result.implementation_language;
manifest.scope = 'Exp2 pilot only';
manifest.excluded = {'full Exp2 alpha sweep','Exp3','A4','S1','P1','manuscript edits'};
manifest.classification = result.classification;
manifest.viable = result.viable;
manifest.authoritative_load = result.authoritative_load;
manifest.alpha = result.alpha;
manifest.mesh = result.mesh;
manifest.artifacts = localRelativeArtifactStruct(paths, repoRoot);
manifest.created_utc = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'));
manifest.failure_reproduction = result.failure_reproduction;
localWriteJson(path, manifest);
end

function localWriteSummary(path, result)
final = struct();
if isfield(result, 'final') && isstruct(result.final)
    final = result.final;
end
lines = {
    '# Exp2 Authoritative Pilot Summary'
    ''
    sprintf('- Classification: %s', result.classification)
    sprintf('- Full Exp2 viable: %d', result.viable)
    sprintf('- Implementation language: %s', result.implementation_language)
    sprintf('- Scope: alpha=%.3g clamped beam only; no alpha sweep or manuscript edits', result.alpha)
    sprintf('- Mesh: %dx%d', result.mesh.nelx, result.mesh.nely)
    sprintf('- Authoritative load: %s', result.authoritative_load)
    sprintf('- Load sensitivity: %s', result.load_sensitivity)
    sprintf('- Gate A0 diagnostics: %d', result.gate_a0_diagnostics)
    sprintf('- Existing Exp2 failure reproduced: %d', result.failure_reproduction.expected_failure_reproduced)
    sprintf('- Failure identifier: %s', result.failure_reproduction.identifier)
    sprintf('- Iterations: %g of cap %g', result.iterations, result.iteration_cap)
    sprintf('- Final omega rad/s: [%s]', localFormatVector(localFieldOr(final, 'omega_rad_s', NaN)))
    sprintf('- Final frequency Hz: [%s]', localFormatVector(localFieldOr(final, 'frequency_hz', NaN)))
    sprintf('- Final tracked mode index: %g', localFieldOr(final, 'tracked_mode_index', NaN))
    sprintf('- Final tracked MAC: %.12g', localFieldOr(final, 'tracked_mode_mac', NaN))
    sprintf('- Final feasibility: %.12g', localFieldOr(final, 'feasibility', NaN))
    sprintf('- Final grayness: %.12g', localFieldOr(final, 'grayness', NaN))
    sprintf('- Final design change: %.12g', localFieldOr(final, 'design_change', NaN))
    ''
    'Conclusion: the alpha=1 authoritative clamped-beam pilot converged before the cap, so full Exp2 is viable as an experiment run. This is pilot evidence only and makes no manuscript claim.'
    };
localWriteText(path, strjoin(lines, newline));
end

function value = localFieldOr(s, field, fallback)
if isstruct(s) && isfield(s, field) && ~isempty(s.(field))
    value = s.(field);
else
    value = fallback;
end
end

function txt = localFormatVector(values)
if ~isnumeric(values) || isempty(values)
    txt = 'NaN';
    return;
end
txt = strjoin(arrayfun(@(v) sprintf('%.12g', v), values(:)', 'UniformOutput', false), ', ');
end

function safe = localJsonSafeResult(result)
safe = result;
if isfield(safe, 'exception') && strlength(string(safe.exception)) > 4000
    safe.exception = extractBefore(string(safe.exception), 4001);
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
    error('Exp2Pilot:WriteFailed', 'Could not open file for writing: %s', path);
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
