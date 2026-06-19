%TEST_A0_PARITY End-to-end MATLAB/Python authoritative-load parity gate.

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
fixturePath = fullfile(scriptDir, 'gate_a0_fixture.json');
pythonScript = fullfile(scriptDir, 'gate_a0_python_diagnostics.py');
pythonOutput = [tempname, '.json'];
cleanupOutput = onCleanup(@() localDeleteIfExists(pythonOutput)); %#ok<NASGU>

addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));

if ~isfile(fixturePath)
    error('GateA0:MissingFixture', 'Missing fixture: %s', fixturePath);
end
if ~isfile(pythonScript)
    error('GateA0:MissingPythonScript', 'Missing Python diagnostics script: %s', pythonScript);
end

cfgBase = jsondecode(fileread(fixturePath));
matlabRuns = struct();
for modeCell = {'omitted', 'complete'}
    mode = modeCell{1};
    cfg = cfgBase;
    cfg.optimization.load_sensitivity = mode;
    [~, ~, ~, ~, ~, info] = run_topopt_from_json(cfg);
    if ~isstruct(info) || ~isfield(info, 'gate_a0')
        error('GateA0:MissingDiagnostics', ...
            'MATLAB solver did not return gate_a0 diagnostics for %s.', mode);
    end
    localValidateDiagnostics(info.gate_a0, mode);
    matlabRuns.(mode) = localSerializableDiagnostics(info.gate_a0);
end

fields = {'reference_omega', 'reference_omega_sq', 'reference_modes', ...
    'reference_modal_mass', 'load_vector', 'objective', ...
    'omitted_sensitivity', 'complete_sensitivity'};
for i = 1:numel(fields)
    field = fields{i};
    localAssertClose(['MATLAB omitted/complete invariant ' field], ...
        matlabRuns.complete.(field), matlabRuns.omitted.(field));
end

pythonExe = localSelectPython();
cmd = sprintf('"%s" "%s" "%s" "%s"', pythonExe, pythonScript, fixturePath, pythonOutput);
[status, output] = system(cmd);
fprintf('%s', output);
if status ~= 0
    error('GateA0:PythonFailed', ...
        'Python Gate A0 diagnostics failed with exit status %d.', status);
end
if ~isfile(pythonOutput)
    error('GateA0:MissingPythonOutput', 'Python diagnostics output was not created.');
end
pythonRuns = jsondecode(fileread(pythonOutput));

for modeCell = {'omitted', 'complete'}
    mode = modeCell{1};
    for i = 1:numel(fields)
        field = fields{i};
        localAssertClose(sprintf('MATLAB/Python %s %s', mode, field), ...
            matlabRuns.(mode).(field), pythonRuns.(mode).(field));
    end
end

fprintf('\nGATE A0 PASSED\n');
fprintf('  Formula: F = omega0^2 * M(x) * Phi0\n');
fprintf('  Reference: fully solid domain, mass-normalized and phase-oriented\n');
fprintf('  Sensitivity modes: omitted and complete\n');
fprintf('  Tolerance: abs_error <= 1e-12 + 1e-8*abs(reference)\n');

function localValidateDiagnostics(diag, selectedMode)
required = {'reference_omega', 'reference_omega_sq', 'reference_modes', ...
    'reference_modal_mass', 'current_mass_matrix', 'load_vector', ...
    'objective', 'omitted_sensitivity', 'complete_sensitivity', ...
    'selected_sensitivity', 'selected_load_sensitivity', ...
    'load_normalization_enabled', 'obsolete_rho_source_used'};
for i = 1:numel(required)
    if ~isfield(diag, required{i})
        error('GateA0:MissingDiagnostic', 'Missing Gate A0 diagnostic: %s', required{i});
    end
end
if diag.load_normalization_enabled
    error('GateA0:LoadNormalization', 'Load normalization is enabled.');
end
if diag.obsolete_rho_source_used
    error('GateA0:ObsoleteRhoSource', 'Obsolete rho-source behavior was used.');
end
if ~strcmp(char(diag.selected_load_sensitivity), selectedMode)
    error('GateA0:SensitivityMode', 'Selected load-sensitivity mode was not propagated.');
end

localAssertClose('reference modal mass', diag.reference_modal_mass, ...
    ones(size(diag.reference_modal_mass)));
omegaSq = diag.reference_omega_sq(1);
phi0 = diag.reference_modes(:, 1);
expectedLoad = omegaSq * (diag.current_mass_matrix * phi0);
localAssertClose('independent omega0^2*M(x)*Phi0 load', ...
    diag.load_vector(:, 1), expectedLoad);

if strcmp(selectedMode, 'complete')
    expectedSensitivity = diag.complete_sensitivity;
else
    expectedSensitivity = diag.omitted_sensitivity;
end
localAssertClose('selected analytical sensitivity', ...
    diag.selected_sensitivity, expectedSensitivity);
end

function out = localSerializableDiagnostics(diag)
fields = {'reference_omega', 'reference_omega_sq', 'reference_modes', ...
    'reference_modal_mass', 'load_vector', 'omitted_sensitivity', ...
    'complete_sensitivity'};
out = struct();
for i = 1:numel(fields)
    field = fields{i};
    value = diag.(field);
    out.(field) = double(value(:));
end
out.objective = double(diag.objective);
end

function localAssertClose(name, actual, reference)
actual = double(actual(:));
reference = double(reference(:));
if ~isequal(size(actual), size(reference))
    error('GateA0:ShapeMismatch', '%s shape mismatch.', name);
end
allowed = 1e-12 + 1e-8 * abs(reference);
err = abs(actual - reference);
if any(~isfinite(actual)) || any(err > allowed)
    error('GateA0:ToleranceExceeded', ...
        '%s max error %.3e exceeds tolerance %.3e.', ...
        name, max(err), max(allowed));
end
end

function localDeleteIfExists(path)
if isfile(path)
    delete(path);
end
end

function pythonExe = localSelectPython()
candidates = {};
fromEnv = getenv('GATE_A0_PYTHON');
if ~isempty(fromEnv)
    candidates{end+1} = fromEnv; %#ok<AGROW>
end
candidates = [candidates, {'python3.13', 'python3'}];
for i = 1:numel(candidates)
    candidate = candidates{i};
    [status, resolved] = system(sprintf('command -v "%s"', candidate));
    if status ~= 0
        continue;
    end
    resolved = strtrim(resolved);
    [status, ~] = system(sprintf('"%s" -c "import numpy, scipy"', resolved));
    if status == 0
        pythonExe = resolved;
        return;
    end
end
error('GateA0:PythonEnvironment', ...
    ['No Python interpreter with working NumPy/SciPy was found. ', ...
     'Set GATE_A0_PYTHON to a compatible interpreter.']);
end
