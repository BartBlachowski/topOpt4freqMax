%RUN_CR2_SMOKE Small-mesh structural smoke validation for CR2 A/B configs.

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
configDir = fullfile(repoRoot, 'examples', 'Revision_v1', 'cr2');
configAPath = fullfile(configDir, 'cr2_variant_a_omitted.json');
configBPath = fullfile(configDir, 'cr2_variant_b_complete.json');
validatorPath = fullfile(scriptDir, 'validate_v1c_cr2_configs.py');
validationPath = fullfile(scriptDir, 'v1c_cr2_validation.json');
jsonResultPath = fullfile(scriptDir, 'cr2_smoke_results.json');
matResultPath = fullfile(scriptDir, 'cr2_smoke_results.mat');
manifestPath = fullfile(scriptDir, 'cr2_smoke_manifest.json');
summaryPath = fullfile(scriptDir, 'cr2_smoke_summary.md');

addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));
for path = {configAPath, configBPath, validatorPath}
    if ~isfile(path{1}), error('CR2Smoke:MissingInput', 'Missing input: %s', path{1}); end
end

pythonExe = localSelectPython();
cmd = sprintf('PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib "%s" "%s" "%s" "%s" "%s"', ...
    pythonExe, validatorPath, configAPath, configBPath, validationPath);
[status, output] = system(cmd);
fprintf('%s', output);
if status ~= 0, error('CR2Smoke:V1cFailed', 'V1c validator failed with status %d.', status); end
validation = jsondecode(fileread(validationPath));
if ~isfield(validation, 'status') || ~strcmp(char(validation.status), 'passed')
    error('CR2Smoke:V1cResult', 'V1c validation did not report passed status.');
end

cfgA = localSmokeOverride(jsondecode(fileread(configAPath)));
cfgB = localSmokeOverride(jsondecode(fileread(configBPath)));
variantA = localRunVariant(cfgA, 'A', 'omitted');
variantB = localRunVariant(cfgB, 'B', 'complete');

results = struct();
results.gate = 'CR2-smoke';
results.status = 'passed';
results.scientific_effect_claim = 'not_evaluated';
results.authoritative_load = 'F(x) = omega0^2 * M(x) * Phi0';
results.v1c_validation = validation;
results.smoke_overrides = struct('nelx', 16, 'nely', 2, ...
    'max_iters', 30, 'convergence_tol', 0.02);
results.variant_a = variantA;
results.variant_b = variantB;
results.remaining_before_production = { ...
    'Run both unmodified production-intent configs at 160x20.', ...
    'Require both variants to meet the predeclared 1e-3 design-change and feasibility rules without hitting the 400-iteration cap.', ...
    'Apply V1a central finite-difference evidence to the complete-gradient implementation used for production.', ...
    'Add production mode-validity tracking with squared mass-weighted MAC and frequency continuity.', ...
    'Save converged topologies and compare objective, tracked frequency, feasibility, grayness, iteration history, and sensitivity differences.', ...
    'Run the independent Gate E4 audit before making any effect-size claim.'};

localWriteJson(jsonResultPath, results);
save(matResultPath, 'results', 'cfgA', 'cfgB');
localWriteSummary(summaryPath, results);
localWriteManifest(manifestPath, repoRoot, configAPath, configBPath, validatorPath, ...
    validationPath, jsonResultPath, matResultPath, summaryPath);

fprintf('\nCR2 SMOKE PASSED\n');
fprintf('  V1c configuration validation: passed\n');
fprintf('  Variant A termination: %s (%d iterations)\n', ...
    variantA.termination.reason, variantA.iterations);
fprintf('  Variant B termination: %s (%d iterations)\n', ...
    variantB.termination.reason, variantB.iterations);
fprintf('  Scientific effect claim: not evaluated\n');
fprintf('  Results: %s\n', jsonResultPath);
fprintf('  Manifest: %s\n', manifestPath);

function cfg = localSmokeOverride(cfg)
cfg.meta.name = [char(cfg.meta.name), ' [small-mesh smoke override]'];
cfg.domain.mesh.nelx = 16;
cfg.domain.mesh.nely = 2;
cfg.optimization.max_iters = 30;
cfg.optimization.convergence_tol = 0.02;
cfg.postprocessing.save_frequency_iterations = false;
end

function result = localRunVariant(cfg, label, expectedSensitivity)
[xFinal, fHz, ~, nIter, ~, info] = run_topopt_from_json(cfg);
if ~isstruct(info) || ~isfield(info, 'gate_a0') || ~isfield(info, 'cr2_history')
    error('CR2Smoke:MissingDiagnostics', 'Variant %s is missing Gate A0 or CR2 histories.', label);
end
gate = info.gate_a0;
history = info.cr2_history;
if ~strcmp(char(gate.selected_load_sensitivity), expectedSensitivity)
    error('CR2Smoke:SensitivityMode', 'Variant %s selected the wrong sensitivity.', label);
end
if gate.load_normalization_enabled || gate.obsolete_rho_source_used
    error('CR2Smoke:FormulationViolation', 'Variant %s violated authoritative-load constraints.', label);
end
expectedLoad = gate.reference_omega_sq(1) * ...
    (gate.current_mass_matrix * gate.reference_modes(:,1));
localAssertClose(['Variant ' label ' load'], gate.load_vector(:,1), expectedLoad, 1e-12, 1e-8);

required = {'objective', 'frequency', 'design_change', 'feasibility', ...
    'grayness', 'volume', 'sensitivity_difference_l2', 'sensitivity_difference_linf'};
for i = 1:numel(required)
    field = required{i};
    if ~isfield(history, field) || size(history.(field),1) ~= nIter || ...
            any(~isfinite(history.(field)(:)))
        error('CR2Smoke:InvalidHistory', ...
            'Variant %s history %s is missing, non-finite, or has wrong length.', label, field);
    end
end
if nIter < 1 || nIter > cfg.optimization.max_iters || any(~isfinite(xFinal)) || ...
        any(xFinal < -1e-12) || any(xFinal > 1+1e-12) || any(~isfinite(fHz))
    error('CR2Smoke:InvalidTermination', 'Variant %s did not terminate structurally validly.', label);
end

designChangeStopped = nIter < cfg.optimization.max_iters && ...
    history.design_change(end) <= cfg.optimization.convergence_tol;
feasibilitySatisfied = history.feasibility(end) <= 1e-10;
productionConverged = designChangeStopped && feasibilitySatisfied && ...
    cfg.optimization.convergence_tol <= 1e-3;
if designChangeStopped
    reason = 'smoke_design_change_stop_structurally_valid';
elseif nIter == cfg.optimization.max_iters
    reason = 'smoke_iteration_cap_structurally_valid';
else
    error('CR2Smoke:UnknownTermination', 'Variant %s has an unexplained termination.', label);
end
result = struct();
result.label = label;
result.load_sensitivity = expectedSensitivity;
result.structurally_valid = true;
result.iterations = nIter;
result.termination = struct('reason', reason, ...
    'design_change_stopped', designChangeStopped, ...
    'feasibility_satisfied', feasibilitySatisfied, ...
    'production_converged', productionConverged, ...
    'capped', nIter == cfg.optimization.max_iters);
result.histories = history;
result.final_frequency_hz = fHz(:);
result.final_design = xFinal(:);
result.final_objective = history.objective(end);
result.final_design_change = history.design_change(end);
result.final_feasibility = history.feasibility(end);
result.final_grayness = history.grayness(end);
result.sensitivity_difference_norms = struct( ...
    'l2_history', history.sensitivity_difference_l2, ...
    'linf_history', history.sensitivity_difference_linf);
end

function localWriteSummary(path, results)
fid = fopen(path, 'w');
if fid < 0, error('CR2Smoke:SummaryWrite', 'Unable to create %s.', path); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# CR2 Smoke Summary\n\n');
fprintf(fid, '- Status: PASS\n');
fprintf(fid, '- Pass scope: structural smoke validation only; this is not Gate E4 or production convergence.\n');
fprintf(fid, '- V1c configuration validation: PASS\n');
fprintf(fid, '- Authoritative load: `F(x) = omega0^2 * M(x) * Phi0`\n');
fprintf(fid, '- Variant A: `%s`, %d iterations, `%s`\n', ...
    results.variant_a.load_sensitivity, results.variant_a.iterations, results.variant_a.termination.reason);
fprintf(fid, '- Variant B: `%s`, %d iterations, `%s`\n', ...
    results.variant_b.load_sensitivity, results.variant_b.iterations, results.variant_b.termination.reason);
fprintf(fid, '- Variant A production-converged evidence: false\n');
fprintf(fid, '- Variant B production-converged evidence: false\n');
fprintf(fid, '- Scientific effect claim: not evaluated by this smoke run.\n\n');
fprintf(fid, '## Remaining Before Production CR2\n\n');
for i = 1:numel(results.remaining_before_production)
    fprintf(fid, '%d. %s\n', i, results.remaining_before_production{i});
end
end

function localWriteManifest(path, repoRoot, configA, configB, validator, validation, jsonResult, matResult, summary)
paths = {configA, configB, validator, fullfile(fileparts(path), 'run_cr2_smoke.m'), ...
    validation, jsonResult, matResult, summary, path};
roles = {'CR2 Variant A config', 'CR2 Variant B config', 'V1c config validator', ...
    'small-mesh smoke runner', 'V1c validation result', 'smoke JSON result', ...
    'smoke MAT result', 'pass/fail summary and remaining steps', 'artifact manifest'};
entries = repmat(struct('path', '', 'role', '', 'status', 'present'), numel(paths), 1);
for i = 1:numel(paths)
    entries(i).path = localRelativePath(paths{i}, repoRoot);
    entries(i).role = roles{i};
    if i < numel(paths) && ~isfile(paths{i})
        error('CR2Smoke:ManifestMissing', 'Manifest input missing: %s', paths{i});
    end
end
manifest = struct('study', 'CR2 preparation and smoke validation', ...
    'status', 'passed', 'scientific_effect_claim', 'not_evaluated', 'entries', entries);
localWriteJson(path, manifest);
end

function relative = localRelativePath(path, repoRoot)
prefix = [repoRoot, filesep];
if startsWith(path, prefix), relative = path(numel(prefix)+1:end); else, relative = path; end
end

function localWriteJson(path, value)
fid = fopen(path, 'w');
if fid < 0, error('CR2Smoke:JsonWrite', 'Unable to create %s.', path); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(value, PrettyPrint=true));
end

function localAssertClose(name, actual, reference, absTolerance, relTolerance)
actual = double(actual(:)); reference = double(reference(:));
if ~isequal(size(actual), size(reference)), error('CR2Smoke:ShapeMismatch', '%s shape mismatch.', name); end
allowed = absTolerance + relTolerance*abs(reference);
if any(~isfinite(actual)) || any(abs(actual-reference) > allowed)
    error('CR2Smoke:ToleranceExceeded', '%s exceeds tolerance.', name);
end
end

function pythonExe = localSelectPython()
candidates = {};
fromEnv = getenv('GATE_A0_PYTHON');
if ~isempty(fromEnv), candidates{end+1} = fromEnv; end %#ok<AGROW>
candidates = [candidates, {'python3.13', 'python3'}];
for i = 1:numel(candidates)
    [status, resolved] = system(sprintf('command -v "%s"', candidates{i}));
    if status ~= 0, continue; end
    resolved = strtrim(resolved);
    [status, ~] = system(sprintf('"%s" -c "import numpy, scipy"', resolved));
    if status == 0, pythonExe = resolved; return; end
end
error('CR2Smoke:PythonEnvironment', 'No Python interpreter with working NumPy/SciPy was found.');
end
