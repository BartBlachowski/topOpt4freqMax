%TEST_V1B_MAC Modal normalization, phase, MAC, and MATLAB/Python parity.

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
fixturePath = fullfile(scriptDir, 'gate_a0_fixture.json');
pythonScript = fullfile(scriptDir, 'verify_v1b_mac.py');
resultPath = fullfile(scriptDir, 'v1b_mac_results.json');
pythonOutput = [tempname, '.json'];
cleanupOutput = onCleanup(@() localDeleteIfExists(pythonOutput)); %#ok<NASGU>
localTolerance = 1e-12;
parityAbsTolerance = 1e-12;
parityRelTolerance = 1e-8;

addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));
if ~isfile(fixturePath), error('V1b:MissingFixture', 'Missing fixture: %s', fixturePath); end
if ~isfile(pythonScript), error('V1b:MissingPythonScript', 'Missing verifier: %s', pythonScript); end

cfg = jsondecode(fileread(fixturePath));
cfg.optimization.load_sensitivity = 'complete';
[~, ~, ~, ~, ~, info] = run_topopt_from_json(cfg);
if ~isstruct(info) || ~isfield(info, 'gate_a0')
    error('V1b:MissingDiagnostics', 'Run returned no Gate A0 diagnostics.');
end
gateDiag = info.gate_a0;
for field = {'current_mass_matrix', 'reference_modes', 'reference_modal_mass'}
    if ~isfield(gateDiag, field{1})
        error('V1b:MissingDiagnostic', 'Missing Gate A0 diagnostic: %s', field{1});
    end
end
M = gateDiag.current_mass_matrix;
if isempty(gateDiag.reference_modes)
    error('V1b:EmptyReferenceModes', 'Reference mode diagnostic is empty.');
end
localAssertClose('reference modal mass', gateDiag.reference_modal_mass, ...
    ones(size(gateDiag.reference_modal_mass)), localTolerance, 0);

first = double(gateDiag.reference_modes(:,1));
seed = linspace(1, 2, numel(first))';
seed = seed - first * ((first' * M * seed) / (first' * M * first));
rawModes = [-3.25*first, 2.75*seed];
normalized = mass_normalize_modes(rawModes, M);
normalized = orient_modes_deterministic(normalized);
modalMassMatrix = normalized' * M * normalized;
modalMasses = diag(modalMassMatrix);
[~, largestIndices] = max(abs(normalized), [], 1);
largestValues = normalized(sub2ind(size(normalized), largestIndices, 1:size(normalized,2)));

identicalMac = squared_mass_weighted_mac(normalized(:,1), normalized(:,1), M);
signInvariantMac = squared_mass_weighted_mac(normalized(:,1), -normalized(:,1), M);
scaleInvariantMac = squared_mass_weighted_mac(3*normalized(:,1), -7*normalized(:,1), M);
orthogonalMac = squared_mass_weighted_mac(normalized(:,1), normalized(:,2), M);
macMatrix = squared_mass_weighted_mac(normalized, normalized, M);

localAssertClose('unit modal mass', modalMasses, ones(2,1), localTolerance, 0);
if any(largestValues < 0)
    error('V1b:PhaseConvention', 'A phase-defining largest-magnitude DOF is negative.');
end
localAssertClose('identical-mode MAC', identicalMac, 1, localTolerance, 0);
localAssertClose('sign-invariant MAC', signInvariantMac, 1, localTolerance, 0);
localAssertClose('scale-invariant MAC', scaleInvariantMac, 1, localTolerance, 0);
localAssertClose('M-orthogonal MAC', orthogonalMac, 0, localTolerance, 0);
localAssertClose('pairwise MAC matrix', macMatrix, eye(2), localTolerance, 0);

matlabResult = struct();
matlabResult.status = 'passed';
matlabResult.modal_mass_tolerance = localTolerance;
matlabResult.mac_tolerance = localTolerance;
matlabResult.normalization_definition = 'phi'' * M * phi = 1';
matlabResult.phase_definition = 'largest-magnitude DOF is nonnegative';
matlabResult.mac_definition = '(phi'' * M * psi)^2 / ((phi'' * M * phi) * (psi'' * M * psi))';
matlabResult.modal_masses = modalMasses;
matlabResult.largest_magnitude_dof_indices = largestIndices(:);
matlabResult.largest_magnitude_dof_values = largestValues(:);
matlabResult.normalized_oriented_modes = normalized(:);
matlabResult.identical_mode_mac = identicalMac;
matlabResult.sign_invariant_mac = signInvariantMac;
matlabResult.scale_invariant_mac = scaleInvariantMac;
matlabResult.orthogonal_mode_mac = orthogonalMac;
matlabResult.pairwise_mac_matrix = macMatrix(:);

pythonExe = localSelectPython();
cmd = sprintf('PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib "%s" "%s" "%s" "%s"', ...
    pythonExe, pythonScript, fixturePath, pythonOutput);
[status, output] = system(cmd);
fprintf('%s', output);
if status ~= 0, error('V1b:PythonFailed', 'Python verifier failed with status %d.', status); end
if ~isfile(pythonOutput), error('V1b:MissingPythonOutput', 'Python result was not created.'); end
pythonResult = jsondecode(fileread(pythonOutput));
if ~isfield(pythonResult, 'status') || ~strcmp(char(pythonResult.status), 'passed')
    error('V1b:PythonResult', 'Python result did not report passed status.');
end

parityFields = {'modal_masses', 'largest_magnitude_dof_indices', ...
    'largest_magnitude_dof_values', 'normalized_oriented_modes', ...
    'identical_mode_mac', 'sign_invariant_mac', 'scale_invariant_mac', ...
    'orthogonal_mode_mac', 'pairwise_mac_matrix'};
for i = 1:numel(parityFields)
    field = parityFields{i};
    if ~isfield(pythonResult, field)
        error('V1b:MissingPythonDiagnostic', 'Missing Python diagnostic: %s', field);
    end
    localAssertClose(['MATLAB/Python ' field], matlabResult.(field), ...
        pythonResult.(field), parityAbsTolerance, parityRelTolerance);
end

combined = struct('gate', 'V1b', 'status', 'passed', ...
    'fixture', 'scripts/revision_v1/gate_a0_fixture.json', ...
    'parity_absolute_tolerance', parityAbsTolerance, ...
    'parity_relative_tolerance', parityRelTolerance, ...
    'matlab', matlabResult, 'python', pythonResult);
fid = fopen(resultPath, 'w');
if fid < 0, error('V1b:ResultWrite', 'Unable to create %s.', resultPath); end
cleanupFid = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(combined, PrettyPrint=true));

fprintf('\nV1b PASSED\n');
fprintf('  Unit modal masses: [%.16g, %.16g]\n', modalMasses);
fprintf('  MAC identical/sign/scale: %.16g / %.16g / %.16g\n', ...
    identicalMac, signInvariantMac, scaleInvariantMac);
fprintf('  M-orthogonal MAC: %.3e\n', orthogonalMac);
fprintf('  MATLAB/Python parity: abs %.1e + rel %.1e\n', ...
    parityAbsTolerance, parityRelTolerance);
fprintf('  Saved: %s\n', resultPath);

function localAssertClose(name, actual, reference, absTolerance, relTolerance)
actual = double(actual(:));
reference = double(reference(:));
if ~isequal(size(actual), size(reference))
    error('V1b:ShapeMismatch', '%s shape mismatch.', name);
end
allowed = absTolerance + relTolerance * abs(reference);
errors = abs(actual-reference);
if any(~isfinite(actual)) || any(errors > allowed)
    error('V1b:ToleranceExceeded', ...
        '%s max error %.3e exceeds allowed %.3e.', name, max(errors), max(allowed));
end
end

function localDeleteIfExists(path)
if isfile(path), delete(path); end
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
error('V1b:PythonEnvironment', 'No Python interpreter with working NumPy/SciPy was found.');
end
