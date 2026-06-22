function run_cr2_rerun()
%RUN_CR2_RERUN Execute the predeclared unmatched CR2 stabilization rerun.

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(fileparts(fileparts(scriptDir))));
outputDir = fullfile(scriptDir, 'output');
if ~isfolder(outputDir), mkdir(outputDir); end

originalAPath = fullfile(fileparts(scriptDir), 'cr2_variant_a_omitted.json');
originalBPath = fullfile(fileparts(scriptDir), 'cr2_variant_b_complete.json');
configAPath = fullfile(scriptDir, 'cr2_rerun_variant_a_omitted.json');
configBPath = fullfile(scriptDir, 'cr2_rerun_variant_b_complete_move002.json');
validatorPath = fullfile(repoRoot, 'scripts', 'revision_v1', 'validate_cr2_rerun_configs.py');
validationPath = fullfile(outputDir, 'cr2_rerun_config_validation.json');
summaryJsonPath = fullfile(outputDir, 'cr2_rerun_results.json');
summaryMdPath = fullfile(outputDir, 'cr2_rerun_summary.md');
manifestPath = fullfile(outputDir, 'cr2_rerun_manifest.json');
logPath = fullfile(outputDir, 'cr2_rerun_run.log');

addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));
for path = {originalAPath, originalBPath, configAPath, configBPath, validatorPath}
    if ~isfile(path{1}), error('CR2Production:MissingInput', 'Missing input: %s', path{1}); end
end

diary(logPath);
cleanupDiary = onCleanup(@() diary('off')); %#ok<NASGU>
fprintf('CR2 protocol rerun started: %s\n', char(datetime('now', 'TimeZone', 'UTC')));
fprintf('No tuning beyond predeclared Variant B move_limit=0.02 is permitted.\n');
fprintf('No A4, Exp2, or Exp3 path is invoked.\n');

pythonExe = localSelectPython();
cmd = sprintf(['PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib ', ...
    '"%s" "%s" "%s" "%s" "%s" "%s" "%s"'], ...
    pythonExe, validatorPath, originalAPath, originalBPath, ...
    configAPath, configBPath, validationPath);
[status, output] = system(cmd);
fprintf('%s', output);
if status ~= 0
    error('CR2Rerun:ConfigValidationFailed', 'Rerun config validation failed with status %d.', status);
end
validation = jsondecode(fileread(validationPath));
if ~strcmp(char(validation.status), 'passed')
    error('CR2Rerun:ConfigValidationResult', 'Rerun config validation did not pass.');
end

criteria = struct('design_change_tolerance', 1e-3, ...
    'feasibility_tolerance', 1e-4, 'mac_threshold', 0.8, ...
    'required_tracking_modes', 6);

variantA = localRunVariant(configAPath, 'A', 'omitted', outputDir, criteria, repoRoot);
variantB = localRunVariant(configBPath, 'B', 'complete', outputDir, criteria, repoRoot);

[differenceMetrics, differenceArtifacts] = localTopologyDifference( ...
    variantA, variantB, outputDir);
variantA.plateau_diagnostics = localPlateauDiagnostics(variantA, 0.2, criteria);
variantB.plateau_diagnostics = localPlateauDiagnostics(variantB, 0.02, criteria);
variantA.protocol_run_outcome = localRunOutcome(variantA);
variantB.protocol_run_outcome = localRunOutcome(variantB);

matchedPair = false;
if matchedPair && variantA.acceptance.accepted && variantB.acceptance.accepted
    outcomeCategory = 'accepted converged comparison';
    allowedClaims = {'Report measured differences for the matched converged pair.'};
elseif strcmp(variantA.protocol_run_outcome, 'diagnostic algorithm-failure evidence') || ...
        strcmp(variantB.protocol_run_outcome, 'diagnostic algorithm-failure evidence')
    outcomeCategory = 'diagnostic algorithm-failure evidence';
    allowedClaims = { ...
        'Report which tested OC configuration failed to converge.', ...
        'Describe move saturation, cycle, plateau, mode-index, MAC, and feasibility diagnostics.', ...
        'State whether move_limit=0.02 stabilized Variant B under this test.'};
else
    outcomeCategory = 'inconclusive';
    allowedClaims = {'Report that the rerun was inconclusive and identify failed checks.'};
end

summary = struct();
summary.study = 'CR2 protocol rerun: original A and B move-limit stabilization screen';
summary.gate = 'E4-CR2-rerun';
summary.outcome_category = outcomeCategory;
summary.accepted_comparison_eligible = matchedPair;
summary.scientific_claim = 'withheld_unmatched_stabilization_screen';
summary.allowed_claims = allowedClaims;
summary.authoritative_load = 'F(x) = omega0^2 * M(x) * Phi0';
summary.acceptance_criteria = criteria;
summary.config_validation_and_hashes = validation;
summary.variant_a = variantA;
summary.variant_b = variantB;
summary.final_topology_difference = differenceMetrics;
summary.final_topology_difference_artifacts = differenceArtifacts;
localWriteJson(summaryJsonPath, summary);
localWriteSummary(summaryMdPath, summary);
localWriteManifest(manifestPath, repoRoot, scriptDir, outputDir, ...
    configAPath, configBPath, validatorPath, validationPath, summaryJsonPath, ...
    summaryMdPath, logPath, variantA, variantB, differenceArtifacts, outcomeCategory);

fprintf('\nCR2 RERUN OUTCOME: %s\n', upper(outcomeCategory));
fprintf('  Variant A accepted: %d (%s)\n', variantA.acceptance.accepted, ...
    strjoin(cellstr(string(variantA.acceptance.failures)), '; '));
fprintf('  Variant B accepted: %d (%s)\n', variantB.acceptance.accepted, ...
    strjoin(cellstr(string(variantB.acceptance.failures)), '; '));
fprintf('  Accepted comparison eligible: 0 (settings intentionally unmatched)\n');
fprintf('  Scientific claim: withheld\n');
fprintf('  Summary: %s\n', summaryMdPath);
fprintf('  Manifest: %s\n', manifestPath);
diary('off');
end

function result = localRunVariant(configPath, label, expectedMode, outputDir, criteria, repoRoot)
cfg = jsondecode(fileread(configPath));
prefix = fullfile(outputDir, ['cr2_variant_', lower(label)]);
paths = struct( ...
    'mat', [prefix, '_result.mat'], ...
    'history_csv', [prefix, '_histories.csv'], ...
    'mode_csv', [prefix, '_mode_tracking.csv'], ...
    'sensitivity_csv', [prefix, '_sensitivity_norms.csv'], ...
    'topology_csv', [prefix, '_topology.csv'], ...
    'topology_png', [prefix, '_topology.png']);

result = struct('label', label, 'load_sensitivity', expectedMode, ...
    'config_path', localRelativePath(configPath, repoRoot), ...
    'solver_success', false, 'success', false, 'exception', '', ...
    'iterations', NaN, 'iteration_cap', double(cfg.optimization.max_iters), ...
    'histories', struct(), 'final_topology', [], 'final_frequencies_hz', [], ...
    'final_tracking', struct(), 'timing_total_s', NaN, ...
    'artifacts', paths, 'acceptance', struct());
xFinal = [];
info = struct();
runTic = tic;
fprintf('\n=== CR2 Variant %s: %s ===\n', label, expectedMode);
try
    [xFinal, fHz, ~, nIter, ~, info] = run_topopt_from_json(cfg);
    result.timing_total_s = toc(runTic);
    result.solver_success = true;
    result.iterations = nIter;
    result.final_topology = xFinal(:);
    result.final_frequencies_hz = fHz(:);
    if ~isfield(info, 'cr2_history') || ~isfield(info, 'cr2_final_tracking') || ...
            ~isfield(info, 'gate_a0')
        error('CR2Production:MissingDiagnostics', ...
            'Variant %s did not return required CR2/Gate diagnostics.', label);
    end
    result.histories = info.cr2_history;
    result.final_tracking = info.cr2_final_tracking;
    localValidateHistories(result.histories, nIter, criteria.required_tracking_modes, label);
    localAssertAuthoritativeGate(info.gate_a0, expectedMode, label);
    localWriteVariantCsv(paths, result.histories, xFinal, cfg);
    localWriteTopologyFigure(paths.topology_png, xFinal, cfg, label, expectedMode);
catch ME
    result.timing_total_s = toc(runTic);
    result.solver_success = false;
    result.exception = getReport(ME, 'extended', 'hyperlinks', 'off');
    fprintf(2, 'Variant %s exception preserved:\n%s\n', label, result.exception);
end

save(paths.mat, 'result', 'cfg', 'xFinal', 'info', '-v7.3');
result.acceptance = localAssessVariant(result, cfg, criteria, paths);
result.success = result.acceptance.accepted;
save(paths.mat, 'result', 'cfg', 'xFinal', 'info', '-v7.3');
fprintf('Variant %s acceptance: %d\n', label, result.acceptance.accepted);
if ~result.acceptance.accepted
    fprintf('  failures: %s\n', strjoin(cellstr(string(result.acceptance.failures)), '; '));
end
end

function acceptance = localAssessVariant(result, cfg, criteria, paths)
failures = {};
if ~result.solver_success, failures{end+1} = 'solver_success=false'; end %#ok<AGROW>
if result.solver_success
    capped = result.iterations >= double(cfg.optimization.max_iters);
    finalChange = result.histories.design_change(end);
    finalFeasibility = result.histories.feasibility(end);
    finalMac = result.final_tracking.tracked_mode_mac;
    finalModeIndex = result.final_tracking.tracked_mode_index;
    if capped, failures{end+1} = 'iteration cap reached'; end %#ok<AGROW>
    if finalChange > double(cfg.optimization.convergence_tol)
        failures{end+1} = sprintf('design change %.3e exceeds %.3e', ...
            finalChange, double(cfg.optimization.convergence_tol)); %#ok<AGROW>
    end
    if finalFeasibility > criteria.feasibility_tolerance
        failures{end+1} = sprintf('feasibility %.3e exceeds %.3e', ...
            finalFeasibility, criteria.feasibility_tolerance); %#ok<AGROW>
    end
    if ~isfinite(finalMac) || finalMac < criteria.mac_threshold || ...
            ~isfinite(finalModeIndex) || finalModeIndex < 1
        failures{end+1} = sprintf('target mode invalid: index=%g MAC=%.6f threshold=%.2f', ...
            finalModeIndex, finalMac, criteria.mac_threshold); %#ok<AGROW>
    end
end

artifactFields = fieldnames(paths);
missing = {};
for i = 1:numel(artifactFields)
    path = paths.(artifactFields{i});
    if ~isfile(path), missing{end+1} = path; end %#ok<AGROW>
end
if ~isempty(missing)
    failures{end+1} = ['missing artifacts: ', strjoin(missing, ', ')]; %#ok<AGROW>
end
acceptance = struct('accepted', isempty(failures), 'failures', {failures}, ...
    'solver_success_required', true, 'not_capped_required', true, ...
    'configured_convergence_tolerance', double(cfg.optimization.convergence_tol), ...
    'feasibility_tolerance', criteria.feasibility_tolerance, ...
    'mac_threshold', criteria.mac_threshold, 'missing_artifacts', {missing});
end

function localValidateHistories(history, nIter, nModes, label)
required = {'objective','frequency','design_change','feasibility','grayness', ...
    'volume','tracked_mode_index','tracked_mode_mac','tracked_mode_omega', ...
    'topology_lag1_linf','topology_lag2_linf', ...
    'sensitivity_difference_l2','sensitivity_difference_linf'};
for i = 1:numel(required)
    field = required{i};
    if ~isfield(history, field) || size(history.(field),1) ~= nIter || ...
            any(~isfinite(history.(field)(:)))
        error('CR2Production:InvalidHistory', ...
            'Variant %s history %s is missing, non-finite, or has wrong length.', label, field);
    end
end
if size(history.frequency,2) ~= nModes
    error('CR2Production:FrequencyHistoryShape', ...
        'Variant %s frequency history must contain %d modes.', label, nModes);
end
end

function localAssertAuthoritativeGate(gate, expectedMode, label)
required = {'reference_omega_sq','reference_modes','current_mass_matrix', ...
    'load_vector','selected_load_sensitivity','load_normalization_enabled', ...
    'obsolete_rho_source_used'};
for i = 1:numel(required)
    if ~isfield(gate, required{i})
        error('CR2Production:MissingGateDiagnostic', ...
            'Variant %s missing Gate diagnostic %s.', label, required{i});
    end
end
if ~strcmp(char(gate.selected_load_sensitivity), expectedMode) || ...
        gate.load_normalization_enabled || gate.obsolete_rho_source_used
    error('CR2Production:GateFormulation', ...
        'Variant %s violated the authoritative formulation contract.', label);
end
expectedLoad = gate.reference_omega_sq(1) * ...
    (gate.current_mass_matrix * gate.reference_modes(:,1));
allowed = 1e-12 + 1e-8*abs(expectedLoad);
if any(abs(gate.load_vector(:,1)-expectedLoad) > allowed)
    error('CR2Production:GateLoadMismatch', ...
        'Variant %s load is not omega0^2*M(x)*Phi0.', label);
end
end

function localWriteVariantCsv(paths, h, x, cfg)
n = size(h.objective,1);
iteration = (1:n)';
T = table(iteration, h.objective, h.design_change, h.feasibility, h.grayness, ...
    h.volume, h.tracked_mode_index, h.tracked_mode_mac, h.tracked_mode_omega, ...
    h.topology_lag1_linf, h.topology_lag2_linf, ...
    h.sensitivity_difference_l2, h.sensitivity_difference_linf, ...
    'VariableNames', {'iteration','objective','design_change','feasibility','grayness', ...
    'volume','tracked_mode_index','tracked_mode_mac','tracked_mode_omega', ...
    'topology_lag1_linf','topology_lag2_linf', ...
    'sensitivity_difference_l2','sensitivity_difference_linf'});
for k = 1:size(h.frequency,2)
    T.(sprintf('omega_%d',k)) = h.frequency(:,k);
end
writetable(T, paths.history_csv);
writetable(T(:, {'iteration','tracked_mode_index','tracked_mode_mac','tracked_mode_omega'}), ...
    paths.mode_csv);
writetable(T(:, {'iteration','sensitivity_difference_l2','sensitivity_difference_linf'}), ...
    paths.sensitivity_csv);

nelx = double(cfg.domain.mesh.nelx); nely = double(cfg.domain.mesh.nely);
[ely, elx] = ndgrid(1:nely, 1:nelx);
topology = table((1:numel(x))', elx(:), ely(:), x(:), ...
    'VariableNames', {'element_index','elx','ely','density'});
writetable(topology, paths.topology_csv);
end

function localWriteTopologyFigure(path, x, cfg, label, mode)
nelx = double(cfg.domain.mesh.nelx); nely = double(cfg.domain.mesh.nely);
fig = figure('Visible','off','Color','w','Position',[100 100 1200 260]);
cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
imagesc(reshape(x, nely, nelx)); axis image tight; set(gca,'YDir','normal');
colormap(gray); caxis([0 1]); colorbar;
title(sprintf('CR2 Variant %s (%s) final topology', label, mode));
xlabel('Element column'); ylabel('Element row');
exportgraphics(fig, path, 'Resolution', 180);
end

function [metrics, artifacts] = localTopologyDifference(a, b, outputDir)
metrics = struct();
artifacts = struct('csv', fullfile(outputDir,'cr2_topology_difference.csv'), ...
    'png', fullfile(outputDir,'cr2_topology_difference.png'), ...
    'json', fullfile(outputDir,'cr2_topology_difference_metrics.json'), ...
    'complete', false);
if ~a.solver_success || ~b.solver_success || ...
        numel(a.final_topology) ~= numel(b.final_topology) || isempty(a.final_topology)
    metrics.status = 'unavailable';
    metrics.reason = 'one or both solver endpoints are unavailable';
    localWriteJson(artifacts.json, metrics);
    return;
end
d = a.final_topology(:)-b.final_topology(:);
metrics = struct('status','computed','mean_absolute_difference',mean(abs(d)), ...
    'rms_difference',sqrt(mean(d.^2)),'maximum_absolute_difference',max(abs(d)), ...
    'fraction_abs_difference_gt_0_01',mean(abs(d)>0.01), ...
    'fraction_abs_difference_gt_0_05',mean(abs(d)>0.05), ...
    'density_correlation',corr(a.final_topology(:),b.final_topology(:)));
writetable(table((1:numel(d))',a.final_topology(:),b.final_topology(:),d,abs(d), ...
    'VariableNames',{'element_index','density_a','density_b','difference','absolute_difference'}), ...
    artifacts.csv);
localWriteJson(artifacts.json, metrics);

fig = figure('Visible','off','Color','w','Position',[100 100 1200 260]);
cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
imagesc(reshape(abs(d),20,160)); axis image tight; set(gca,'YDir','normal');
colorbar; title('CR2 absolute topology difference |A-B|');
xlabel('Element column'); ylabel('Element row');
exportgraphics(fig, artifacts.png, 'Resolution', 180);
artifacts.complete = isfile(artifacts.csv) && isfile(artifacts.png) && isfile(artifacts.json);
end

function diagnostic = localPlateauDiagnostics(result, moveLimit, criteria)
diagnostic = struct('status','unavailable','window_length',0, ...
    'algorithm_failure_signature',false,'algorithm_failure_evidence',false);
if ~result.solver_success || isempty(fieldnames(result.histories))
    return;
end
h = result.histories;
n = numel(h.objective);
w = min(50,n);
idx = (n-w+1:n)';
obj = h.objective(idx);
omega = h.tracked_mode_omega(idx);
dc = h.design_change(idx);
scaleObj = max(abs(mean(obj)),eps);
scaleOmega = max(abs(mean(omega)),eps);
pObj = polyfit((0:w-1)',obj,1);
pDc = polyfit((0:w-1)',dc,1);

diagnostic.status = 'computed';
diagnostic.window_length = w;
diagnostic.relative_objective_range = (max(obj)-min(obj))/scaleObj;
diagnostic.relative_frequency_range = (max(omega)-min(omega))/scaleOmega;
diagnostic.relative_objective_slope_per_iteration = pObj(1)/scaleObj;
diagnostic.relative_design_change_slope_per_iteration = pDc(1)/max(mean(dc),eps);
diagnostic.median_design_change = median(dc);
diagnostic.maximum_design_change = max(dc);
diagnostic.median_relative_lag1_objective_difference = median(abs(diff(obj)))/scaleObj;
if w >= 3
    diagnostic.median_relative_lag2_objective_difference = ...
        median(abs(obj(3:end)-obj(1:end-2)))/scaleObj;
else
    diagnostic.median_relative_lag2_objective_difference = Inf;
end
diagnostic.median_topology_lag1_linf = median(h.topology_lag1_linf(idx));
diagnostic.median_topology_lag2_linf = median(h.topology_lag2_linf(idx));
diagnostic.move_saturation_fraction = mean(dc >= 0.99*moveLimit);
diagnostic.feasibility_min = min(h.feasibility(idx));
diagnostic.feasibility_max = max(h.feasibility(idx));
diagnostic.mode_indices = unique(h.tracked_mode_index(idx))';
diagnostic.minimum_mac = min(h.tracked_mode_mac(idx));
diagnostic.maximum_mac = max(h.tracked_mode_mac(idx));
diagnostic.objective_plateau = diagnostic.relative_objective_range <= 5e-3 && ...
    diagnostic.relative_frequency_range <= 5e-3;
diagnostic.move_saturation_signature = diagnostic.move_saturation_fraction >= 0.9;
diagnostic.period2_signature = ...
    diagnostic.median_relative_lag2_objective_difference <= 1e-4 && ...
    diagnostic.median_relative_lag1_objective_difference >= 1e-3;
diagnostic.persistent_bounded_cycle_signature = ...
    all(dc > criteria.design_change_tolerance) && diagnostic.objective_plateau && ...
    diagnostic.relative_design_change_slope_per_iteration >= -1e-3 && ...
    diagnostic.median_topology_lag1_linf > criteria.design_change_tolerance;
diagnostic.algorithm_failure_signature = diagnostic.move_saturation_signature || ...
    diagnostic.period2_signature || diagnostic.persistent_bounded_cycle_signature;
diagnostic.algorithm_failure_evidence = result.iterations >= result.iteration_cap && ...
    isempty(result.acceptance.missing_artifacts) && ...
    result.final_tracking.tracked_mode_mac >= criteria.mac_threshold && ...
    diagnostic.algorithm_failure_signature;
end

function category = localRunOutcome(result)
if isfield(result,'plateau_diagnostics') && ...
        result.plateau_diagnostics.algorithm_failure_evidence
    category = 'diagnostic algorithm-failure evidence';
else
    % An unmatched stabilization endpoint cannot itself be an accepted comparison.
    category = 'inconclusive';
end
end

function localWriteSummary(path, summary)
fid = fopen(path,'w'); if fid<0, error('CR2Production:SummaryWrite','Cannot create %s.',path); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'# CR2 Protocol Rerun Summary\n\n');
fprintf(fid,'- Outcome category: **%s**\n', upper(summary.outcome_category));
fprintf(fid,'- Accepted converged comparison eligible: false (settings intentionally unmatched)\n');
fprintf(fid,'- Scientific claim: `%s`\n', summary.scientific_claim);
fprintf(fid,'- Load: `F(x) = omega0^2 * M(x) * Phi0`\n');
fprintf(fid,'- Acceptance: design change <= 1e-3, feasibility <= 1e-4, MAC >= 0.8, not capped, all artifacts present.\n');
fprintf(fid,'- Further tuning performed: no\n\n');
for item = {'variant_a','variant_b'}
    v = summary.(item{1});
    fprintf(fid,'## Variant %s (%s)\n\n',v.label,v.load_sensitivity);
    fprintf(fid,'- Accepted: %d\n',v.acceptance.accepted);
    fprintf(fid,'- Protocol run outcome: `%s`\n',v.protocol_run_outcome);
    fprintf(fid,'- Solver success: %d\n',v.solver_success);
    fprintf(fid,'- Iterations: %g / %g\n',v.iterations,v.iteration_cap);
    if v.solver_success
        fprintf(fid,'- Final design change: %.6e\n',v.histories.design_change(end));
        fprintf(fid,'- Final feasibility: %.6e\n',v.histories.feasibility(end));
        fprintf(fid,'- Final tracked mode: %d, MAC %.6f\n', ...
            v.final_tracking.tracked_mode_index,v.final_tracking.tracked_mode_mac);
        fprintf(fid,'- Final-50 objective plateau: %d\n',v.plateau_diagnostics.objective_plateau);
        fprintf(fid,'- Algorithm-failure signature: %d\n',v.plateau_diagnostics.algorithm_failure_signature);
    end
    if isempty(v.acceptance.failures)
        fprintf(fid,'- Failures: none\n\n');
    else
        fprintf(fid,'- Failures: %s\n\n',strjoin(cellstr(string(v.acceptance.failures)),'; '));
    end
end
fprintf(fid,'This unmatched stabilization screen permits no converged A/B endpoint comparison.\n');
end

function localWriteManifest(path, repoRoot, scriptDir, outputDir, configA, configB, validator, validation, summaryJson, summaryMd, logPath, a, b, diffArtifacts, outcomeCategory)
paths = {fullfile(scriptDir,'run_cr2_rerun.m'),configA,configB,validator,validation, ...
    a.artifacts.mat,a.artifacts.history_csv,a.artifacts.mode_csv,a.artifacts.sensitivity_csv, ...
    a.artifacts.topology_csv,a.artifacts.topology_png,b.artifacts.mat,b.artifacts.history_csv, ...
    b.artifacts.mode_csv,b.artifacts.sensitivity_csv,b.artifacts.topology_csv,b.artifacts.topology_png, ...
    diffArtifacts.csv,diffArtifacts.png,diffArtifacts.json,summaryJson,summaryMd,logPath,path};
roles = {'protocol rerun runner','rerun Variant A config','rerun Variant B config','rerun config validator','config hashes and validation', ...
    'Variant A MAT','Variant A full histories','Variant A mode history','Variant A sensitivity history', ...
    'Variant A topology data','Variant A topology figure','Variant B MAT','Variant B full histories', ...
    'Variant B mode history','Variant B sensitivity history','Variant B topology data','Variant B topology figure', ...
    'topology difference data','topology difference figure','topology difference metrics', ...
    'production result JSON','production summary','complete run log','artifact manifest'};
entries = repmat(struct('path','','role','','exists',false),numel(paths),1);
for i=1:numel(paths)
    entries(i).path=localRelativePath(paths{i},repoRoot); entries(i).role=roles{i};
    entries(i).exists=isfile(paths{i}) || i==numel(paths);
end
manifest=struct('study','CR2 protocol rerun','gate','E4-CR2-rerun', ...
    'outcome_category',outcomeCategory,'further_tuning_performed',false, ...
    'output_directory',localRelativePath(outputDir,repoRoot),'entries',entries);
localWriteJson(path,manifest);
end

function localWriteJson(path,value)
fid=fopen(path,'w'); if fid<0,error('CR2Production:JsonWrite','Cannot create %s.',path);end
cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end

function relative=localRelativePath(path,repoRoot)
prefix=[repoRoot,filesep]; if startsWith(path,prefix),relative=path(numel(prefix)+1:end);else,relative=path;end
end

function pythonExe=localSelectPython()
candidates={}; fromEnv=getenv('GATE_A0_PYTHON');
if ~isempty(fromEnv),candidates{end+1}=fromEnv;end %#ok<AGROW>
candidates=[candidates,{'python3.13','python3'}];
for i=1:numel(candidates)
    [status,resolved]=system(sprintf('command -v "%s"',candidates{i})); if status~=0,continue;end
    resolved=strtrim(resolved); [status,~]=system(sprintf('"%s" -c "import numpy, scipy"',resolved));
    if status==0,pythonExe=resolved;return;end
end
error('CR2Production:PythonEnvironment','No Python with NumPy/SciPy found.');
end
