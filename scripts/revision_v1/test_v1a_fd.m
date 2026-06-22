%TEST_V1A_FD Verify complete authoritative sensitivity by central differences.

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
fixturePath = fullfile(scriptDir, 'gate_a0_fixture.json');
pythonScript = fullfile(scriptDir, 'verify_v1a_fd.py');
resultPath = fullfile(scriptDir, 'v1a_fd_results.json');
pythonOutput = [tempname, '.json'];
cleanupOutput = onCleanup(@() localDeleteIfExists(pythonOutput)); %#ok<NASGU>

fdStep = 1e-6;
relativeTolerance = 1e-5;
relativeFloor = 1e-14;

addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));
if ~isfile(fixturePath)
    error('V1a:MissingFixture', 'Missing Gate A0 fixture: %s', fixturePath);
end
if ~isfile(pythonScript)
    error('V1a:MissingPythonScript', 'Missing Python V1a verifier: %s', pythonScript);
end

cfg = jsondecode(fileread(fixturePath));
cfg.optimization.load_sensitivity = 'complete';
[~, ~, ~, ~, ~, info] = run_topopt_from_json(cfg);
if ~isstruct(info) || ~isfield(info, 'gate_a0')
    error('V1a:MissingDiagnostics', 'Complete run returned no Gate A0 diagnostics.');
end
diag = info.gate_a0;
required = {'current_x', 'complete_sensitivity', 'omitted_sensitivity', ...
    'reference_omega_sq', 'reference_modes', 'selected_load_sensitivity'};
for i = 1:numel(required)
    if ~isfield(diag, required{i})
        error('V1a:MissingDiagnostic', 'Missing Gate A0 diagnostic: %s', required{i});
    end
end
if ~strcmp(char(diag.selected_load_sensitivity), 'complete')
    error('V1a:WrongSensitivityBranch', 'Complete sensitivity branch was not selected.');
end

x = double(diag.current_x(:));
completeAnalytical = double(diag.complete_sensitivity(:));
omittedAnalytical = double(diag.omitted_sensitivity(:));
if numel(completeAnalytical) ~= numel(x) || numel(omittedAnalytical) ~= numel(x)
    error('V1a:DiagnosticSize', 'Sensitivity diagnostic has the wrong size.');
end
testedIndices = unique(round(linspace(1, numel(x), 6)), 'stable');
if any(x(testedIndices) - fdStep <= 0) || any(x(testedIndices) + fdStep >= 1)
    error('V1a:PerturbationBounds', 'Perturbation leaves the open density interval.');
end

fdValues = zeros(numel(testedIndices), 1);
for i = 1:numel(testedIndices)
    index = testedIndices(i);
    xPlus = x;
    xMinus = x;
    xPlus(index) = xPlus(index) + fdStep;
    xMinus(index) = xMinus(index) - fdStep;
    fdValues(i) = (localObjective(xPlus, cfg, diag) - ...
        localObjective(xMinus, cfg, diag)) / (2 * fdStep);
end
completeValues = completeAnalytical(testedIndices);
omittedValues = omittedAnalytical(testedIndices);
relativeErrors = abs(fdValues - completeValues) ./ ...
    max(max(abs(fdValues), abs(completeValues)), relativeFloor);
omittedErrors = abs(fdValues - omittedValues) ./ ...
    max(max(abs(fdValues), abs(omittedValues)), relativeFloor);
if any(~isfinite(relativeErrors))
    error('V1a:NonfiniteError', 'Complete relative errors contain non-finite values.');
end
if any(relativeErrors > relativeTolerance)
    [worstError, worst] = max(relativeErrors);
    error('V1a:ToleranceExceeded', ...
        'Complete FD relative error %.3e at element %d exceeds %.1e.', ...
        worstError, testedIndices(worst), relativeTolerance);
end
if all(abs(completeValues - omittedValues) <= ...
        relativeFloor + relativeTolerance * abs(completeValues))
    error('V1a:IndistinguishableBranches', ...
        'Omitted and complete sensitivities are unexpectedly indistinguishable.');
end

matlabResult = struct();
matlabResult.status = 'passed';
matlabResult.formulation = 'F(x) = omega0^2 * M(x) * Phi0';
matlabResult.perturbation_size = fdStep;
matlabResult.relative_error_tolerance = relativeTolerance;
matlabResult.tested_element_indices = testedIndices(:);
matlabResult.finite_difference_values = fdValues;
matlabResult.complete_analytical_values = completeValues;
matlabResult.complete_relative_errors = relativeErrors;
matlabResult.omitted_analytical_values = omittedValues;
matlabResult.omitted_relative_errors_vs_complete_fd = omittedErrors;
matlabResult.omitted_expected_to_match_complete_fd = false;
matlabResult.omitted_confirmation = ['The omitted branch excludes the nonzero load ', ...
    'derivative and is not expected to match the complete finite-difference derivative.'];

pythonExe = localSelectPython();
cmd = sprintf('PYTHONDONTWRITEBYTECODE=1 "%s" "%s" "%s" "%s"', ...
    pythonExe, pythonScript, fixturePath, pythonOutput);
[status, output] = system(cmd);
fprintf('%s', output);
if status ~= 0
    error('V1a:PythonFailed', 'Python V1a verifier failed with exit status %d.', status);
end
if ~isfile(pythonOutput)
    error('V1a:MissingPythonOutput', 'Python V1a result was not created.');
end
pythonResult = jsondecode(fileread(pythonOutput));
if ~isfield(pythonResult, 'status') || ~strcmp(char(pythonResult.status), 'passed')
    error('V1a:PythonResult', 'Python V1a result did not report passed status.');
end

combined = struct();
combined.gate = 'V1a';
combined.status = 'passed';
combined.fixture = 'scripts/revision_v1/gate_a0_fixture.json';
combined.matlab = matlabResult;
combined.python = pythonResult;
fid = fopen(resultPath, 'w');
if fid < 0
    error('V1a:ResultWrite', 'Unable to create result artifact: %s', resultPath);
end
cleanupFid = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(combined, PrettyPrint=true));

fprintf('\nV1a PASSED\n');
fprintf('  Formula: F(x) = omega0^2 * M(x) * Phi0\n');
fprintf('  MATLAB max complete relative error: %.3e\n', max(relativeErrors));
fprintf('  Python max complete relative error: %.3e\n', ...
    max(double(pythonResult.complete_relative_errors(:))));
fprintf('  Omitted sensitivity is recorded and is not expected to match complete FD.\n');
fprintf('  Saved: %s\n', resultPath);

function objective = localObjective(x, cfg, diag)
nelx = double(cfg.domain.mesh.nelx);
nely = double(cfg.domain.mesh.nely);
L = double(cfg.domain.size.length);
H = double(cfg.domain.size.height);
ndof = 2 * (nelx + 1) * (nely + 1);
hx = L / nelx;
hy = H / nely;
KE = localLk(hx, hy, double(cfg.material.nu));
ME = localLm(hx, hy);
[edofMat, iK, jK] = localElementData(nelx, nely);
E0 = double(cfg.material.E);
Emin = E0 * double(cfg.void_material.E_min_ratio);
rho0 = double(cfg.material.rho);
rhoMin = double(cfg.void_material.rho_min);
penal = double(cfg.optimization.penalization);
pmass = 1.0;
sK = reshape(KE(:) * (Emin + x'.^penal * (E0 - Emin)), [], 1);
sM = reshape(ME(:) * (rhoMin + x'.^pmass * (rho0 - rhoMin)), [], 1);
K = sparse(iK, jK, sK, ndof, ndof);
M = sparse(iK, jK, sM, ndof, ndof);
phi0 = double(diag.reference_modes(:, 1));
omega0Sq = double(diag.reference_omega_sq(1));
F = omega0Sq * (M * phi0);
fixed = localFixedDofs(cfg, nelx, nely, L, H);
free = setdiff((1:ndof)', fixed);
U = zeros(ndof, 1);
U(free) = K(free, free) \ F(free);
objective = U' * K * U;
if ~isfinite(objective)
    error('V1a:NonfiniteObjective', 'Perturbed objective is not finite.');
end
end

function [edofMat, iK, jK] = localElementData(nelx, nely)
edofMat = zeros(nelx*nely, 8);
for elx = 0:nelx-1
    for ely = 0:nely-1
        el = ely + elx*nely + 1;
        n1 = (nely+1)*elx + ely;
        n2 = (nely+1)*(elx+1) + ely;
        edofMat(el,:) = [2*n1+1, 2*n1+2, 2*n2+1, 2*n2+2, ...
            2*(n2+1)+1, 2*(n2+1)+2, 2*(n1+1)+1, 2*(n1+1)+2];
    end
end
iK = reshape(kron(edofMat, ones(1,8))', [], 1);
jK = reshape(kron(edofMat, ones(8,1))', [], 1);
end

function fixed = localFixedDofs(cfg, nelx, nely, L, H)
[X, Y] = meshgrid(linspace(0, L, nelx+1), linspace(0, H, nely+1));
nodeX = X(:);
nodeY = Y(:);
fixed = [];
supports = cfg.bc.supports;
for i = 1:numel(supports)
    if iscell(supports)
        support = supports{i};
    else
        support = supports(i);
    end
    switch char(support.type)
        case 'vertical_line'
            tol = 1e-9;
            if isfield(support, 'tol'), tol = double(support.tol); end
            nodes = find(abs(nodeX - double(support.x)) <= tol);
        case 'closest_point'
            location = double(support.location(:));
            [~, node] = min((nodeX-location(1)).^2 + (nodeY-location(2)).^2);
            nodes = node;
        otherwise
            error('V1a:UnsupportedSupport', ...
                'V1a fixture uses unsupported support type: %s', char(support.type));
    end
    dofs = cellstr(string(support.dofs));
    if any(strcmp(dofs, 'ux')), fixed = [fixed; 2*nodes-1]; end %#ok<AGROW>
    if any(strcmp(dofs, 'uy')), fixed = [fixed; 2*nodes]; end %#ok<AGROW>
end
fixed = unique(fixed);
if isempty(fixed)
    error('V1a:NoFixedDofs', 'V1a fixture produced no fixed degrees of freedom.');
end
end

function KE = localLk(hx, hy, nu)
D = (1/(1-nu^2)) * [1, nu, 0; nu, 1, 0; 0, 0, (1-nu)/2];
invJ = [2/hx, 0; 0, 2/hy];
detJ = hx*hy/4;
gp = 1/sqrt(3);
KE = zeros(8,8);
for xi = [-gp, gp]
    for eta = [-gp, gp]
        dNdxi = 0.25 * [-(1-eta), (1-eta), (1+eta), -(1+eta)];
        dNdeta = 0.25 * [-(1-xi), -(1+xi), (1+xi), (1-xi)];
        dNxy = invJ * [dNdxi; dNdeta];
        B = zeros(3,8);
        B(1,1:2:end) = dNxy(1,:);
        B(2,2:2:end) = dNxy(2,:);
        B(3,1:2:end) = dNxy(2,:);
        B(3,2:2:end) = dNxy(1,:);
        KE = KE + B' * D * B * detJ;
    end
end
end

function ME = localLm(hx, hy)
Ms = (hx*hy/36) * [4,2,1,2; 2,4,2,1; 1,2,4,2; 2,1,2,4];
ME = kron(Ms, eye(2));
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
error('V1a:PythonEnvironment', ...
    'No Python interpreter with working NumPy/SciPy was found.');
end
