%TEST_V1_4_SENSITIVITY V1-4: Sensitivity variant regression.
%
% Verifies that:
%   1. Both the "omitted" and "complete" load-sensitivity branches execute
%      without error and produce finite, deterministic outputs.
%   2. The two branches produce numerically distinguishable sensitivities
%      (the load-derivative contribution is nonzero).
%   3. The complete branch is internally consistent: both runs (omitted-mode
%      and complete-mode) return the same complete_sensitivity vector.
%   4. The complete sensitivity matches central finite differences to the
%      declared tolerance of 1e-5 (consistent with V1a evidence).
%   5. Determinism: running the complete branch twice yields bitwise-identical
%      sensitivity vectors.
%
% Output: scripts/revision_v1/v1_4_sensitivity_results.json

scriptDir  = fileparts(mfilename('fullpath'));
repoRoot   = fileparts(fileparts(scriptDir));
fixturePath = fullfile(scriptDir, 'gate_a0_fixture.json');
resultPath  = fullfile(scriptDir, 'v1_4_sensitivity_results.json');

addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));

if ~isfile(fixturePath)
    error('V1_4:MissingFixture', 'Missing fixture: %s', fixturePath);
end

cfgBase = jsondecode(fileread(fixturePath));
fdStep  = 1e-6;
fdRelTol = 1e-5;
fdRelFloor = 1e-14;
branchDistinctRelTol = 1e-3;  % branches must differ by at least 0.1%

% -----------------------------------------------------------------
% Run 1: omitted sensitivity (gate_a0_diagnostics=true)
% -----------------------------------------------------------------
cfgOmit = cfgBase;
cfgOmit.optimization.load_sensitivity = 'omitted';
[~,~,~,~,~, infoOmit] = run_topopt_from_json(cfgOmit);
if ~isstruct(infoOmit) || ~isfield(infoOmit, 'gate_a0')
    error('V1_4:MissingDiag_Omit', 'Omitted run returned no gate_a0 diagnostics.');
end
diagOmit = infoOmit.gate_a0;

% -----------------------------------------------------------------
% Run 2: complete sensitivity (gate_a0_diagnostics=true)
% -----------------------------------------------------------------
cfgComp = cfgBase;
cfgComp.optimization.load_sensitivity = 'complete';
[~,~,~,~,~, infoComp] = run_topopt_from_json(cfgComp);
if ~isstruct(infoComp) || ~isfield(infoComp, 'gate_a0')
    error('V1_4:MissingDiag_Comp', 'Complete run returned no gate_a0 diagnostics.');
end
diagComp = infoComp.gate_a0;

omittedSens  = double(diagComp.omitted_sensitivity(:));
completeSens = double(diagComp.complete_sensitivity(:));

% Check 1: Both sensitivity vectors are finite.
if any(~isfinite(omittedSens))
    error('V1_4:NonfiniteOmitted', 'Omitted sensitivity contains non-finite values.');
end
if any(~isfinite(completeSens))
    error('V1_4:NonfiniteComplete', 'Complete sensitivity contains non-finite values.');
end
fprintf('[V1-4] Both sensitivity vectors are finite (%d elements).\n', numel(omittedSens));

% Check 2: Branches are numerically distinguishable.
maxDiff = max(abs(omittedSens - completeSens));
refScale = max(abs(completeSens));
relDiff  = maxDiff / max(refScale, fdRelFloor);
if relDiff < branchDistinctRelTol
    error('V1_4:BranchesIndistinguishable', ...
        'Omitted and complete sensitivities are within %.3e relative — load derivative appears zero.', ...
        relDiff);
end
fprintf('[V1-4] Branches are distinguishable: max relative diff = %.3e.\n', relDiff);

% Check 3: cross-run consistency — complete_sensitivity is the same regardless
% of which sensitivity mode was active.
omittedFromOmitRun   = double(diagOmit.complete_sensitivity(:));
omittedFromCompleteRun = completeSens;
crossErr = max(abs(omittedFromOmitRun - omittedFromCompleteRun)) / max(max(abs(omittedFromOmitRun)), fdRelFloor);
if crossErr > 1e-10
    error('V1_4:CrossRunInconsistency', ...
        'complete_sensitivity differs between omitted-mode and complete-mode runs (rel err %.3e).', crossErr);
end
fprintf('[V1-4] Cross-run consistency: complete_sensitivity rel err = %.3e.\n', crossErr);

% Check 4: Complete sensitivity matches central FD.
x = double(diagComp.current_x(:));
testedIndices = unique(round(linspace(1, numel(x), 6)), 'stable');
if any(x(testedIndices) - fdStep <= 0) || any(x(testedIndices) + fdStep >= 1)
    error('V1_4:PerturbationBounds', 'Perturbation leaves the open density interval (0, 1).');
end

fdValues = zeros(numel(testedIndices), 1);
for i = 1:numel(testedIndices)
    idx    = testedIndices(i);
    xPlus  = x; xPlus(idx)  = xPlus(idx)  + fdStep;
    xMinus = x; xMinus(idx) = xMinus(idx) - fdStep;
    fdValues(i) = (localObjective(xPlus, cfgComp, diagComp) - ...
        localObjective(xMinus, cfgComp, diagComp)) / (2 * fdStep);
end

completePts = completeSens(testedIndices);
fdRelErrors = abs(fdValues - completePts) ./ ...
    max(max(abs(fdValues), abs(completePts)), fdRelFloor);

if any(~isfinite(fdRelErrors))
    error('V1_4:NonfiniteRelError', 'FD relative errors contain non-finite values.');
end
maxFdRelError = max(fdRelErrors);
if maxFdRelError > fdRelTol
    [~, worst] = max(fdRelErrors);
    error('V1_4:FDToleranceExceeded', ...
        'Complete FD relative error %.3e at element %d exceeds tolerance %.1e.', ...
        maxFdRelError, testedIndices(worst), fdRelTol);
end
fprintf('[V1-4] Complete sensitivity FD check: max rel err = %.3e (tol %.1e). PASS.\n', ...
    maxFdRelError, fdRelTol);

% Check 5: Determinism — run complete branch a second time.
[~,~,~,~,~, infoComp2] = run_topopt_from_json(cfgComp);
completeSens2 = double(infoComp2.gate_a0.complete_sensitivity(:));
deterministicErr = max(abs(completeSens - completeSens2));
if deterministicErr ~= 0
    error('V1_4:Nondeterminism', ...
        'Two identical complete runs produced different sensitivities (max diff %.3e).', ...
        deterministicErr);
end
fprintf('[V1-4] Determinism: two identical runs produce bitwise-equal sensitivities.\n');

% -----------------------------------------------------------------
% Collect and write result.
% -----------------------------------------------------------------
result = struct();
result.gate    = 'V1-4';
result.status  = 'passed';
result.fixture = 'scripts/revision_v1/gate_a0_fixture.json';
result.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
result.matlab_version = version;

result.check_1_both_finite = true;
result.check_2_branches_distinguishable = struct( ...
    'passed', true, ...
    'max_relative_difference', relDiff, ...
    'min_required', branchDistinctRelTol);
result.check_3_cross_run_consistency = struct( ...
    'passed', true, ...
    'complete_sensitivity_cross_run_rel_error', crossErr);
result.check_4_complete_fd = struct( ...
    'passed', true, ...
    'perturbation_size', fdStep, ...
    'relative_error_tolerance', fdRelTol, ...
    'tested_element_indices', testedIndices(:), ...
    'finite_difference_values', fdValues, ...
    'complete_analytical_values', completePts, ...
    'complete_relative_errors', fdRelErrors, ...
    'max_relative_error', maxFdRelError);
result.check_5_determinism = struct( ...
    'passed', true, ...
    'max_difference_second_run', deterministicErr);

fid = fopen(resultPath, 'w');
if fid < 0
    error('V1_4:ResultWrite', 'Unable to create result file: %s', resultPath);
end
cleanupFid = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(result, PrettyPrint=true));

fprintf('\nV1-4 PASSED\n');
fprintf('  Both sensitivity branches execute, finite, deterministic.\n');
fprintf('  Max relative branch difference: %.3e\n', relDiff);
fprintf('  Complete FD max relative error: %.3e (tol %.1e)\n', maxFdRelError, fdRelTol);
fprintf('  Saved: %s\n', resultPath);

% =========================================================================
% Perturbed objective for central-difference FD (no solver call).
function objective = localObjective(x, cfg, diag)
nelx  = double(cfg.domain.mesh.nelx);
nely  = double(cfg.domain.mesh.nely);
L     = double(cfg.domain.size.length);
H     = double(cfg.domain.size.height);
ndof  = 2 * (nelx + 1) * (nely + 1);
hx    = L / nelx;
hy    = H / nely;
KE    = localLk(hx, hy, double(cfg.material.nu));
ME    = localLm(hx, hy);
[~, iK, jK] = localEdofData(nelx, nely);
E0    = double(cfg.material.E);
Emin  = E0 * double(cfg.void_material.E_min_ratio);
rho0  = double(cfg.material.rho);
rhoMin = double(cfg.void_material.rho_min);
penal = double(cfg.optimization.penalization);
pmass = 1.0;
sK = reshape(KE(:) * (Emin + x'.^penal * (E0 - Emin)), [], 1);
sM = reshape(ME(:) * (rhoMin + x'.^pmass * (rho0 - rhoMin)), [], 1);
K  = sparse(iK, jK, sK, ndof, ndof);
M  = sparse(iK, jK, sM, ndof, ndof);
phi0     = double(diag.reference_modes(:, 1));
omega0Sq = double(diag.reference_omega_sq(1));
F    = omega0Sq * (M * phi0);
fixed = localFixedDofs(cfg, nelx, nely, L, H);
free  = setdiff((1:ndof)', fixed);
U     = zeros(ndof, 1);
U(free) = K(free, free) \ F(free);
objective = U' * K * U;
if ~isfinite(objective)
    error('V1_4:NonfiniteObjective', 'Perturbed objective is not finite.');
end
end

function [edofMat, iK, jK] = localEdofData(nelx, nely)
edofMat = zeros(nelx * nely, 8);
for elx = 0:nelx-1
    for ely = 0:nely-1
        el  = ely + elx * nely + 1;
        n1  = (nely + 1) * elx + ely;
        n2  = (nely + 1) * (elx + 1) + ely;
        edofMat(el, :) = [2*n1+1, 2*n1+2, 2*n2+1, 2*n2+2, ...
            2*(n2+1)+1, 2*(n2+1)+2, 2*(n1+1)+1, 2*(n1+1)+2];
    end
end
iK = reshape(kron(edofMat, ones(1, 8))', [], 1);
jK = reshape(kron(edofMat, ones(8, 1))', [], 1);
end

function fixed = localFixedDofs(cfg, nelx, nely, L, H)
[X, Y] = meshgrid(linspace(0, L, nelx+1), linspace(0, H, nely+1));
nodeX = X(:); nodeY = Y(:);
fixed = [];
supports = cfg.bc.supports;
for i = 1:numel(supports)
    if iscell(supports), sup = supports{i}; else, sup = supports(i); end
    switch char(sup.type)
        case 'vertical_line'
            tol = 1e-9; if isfield(sup, 'tol'), tol = double(sup.tol); end
            nodes = find(abs(nodeX - double(sup.x)) <= tol);
        case 'closest_point'
            loc = double(sup.location(:));
            [~, node] = min((nodeX - loc(1)).^2 + (nodeY - loc(2)).^2);
            nodes = node;
        otherwise
            error('V1_4:UnsupportedSupport', 'Unsupported support type: %s', char(sup.type));
    end
    dofs = cellstr(string(sup.dofs));
    if any(strcmp(dofs, 'ux')), fixed = [fixed; 2*nodes - 1]; end %#ok<AGROW>
    if any(strcmp(dofs, 'uy')), fixed = [fixed; 2*nodes];     end %#ok<AGROW>
end
fixed = unique(fixed);
if isempty(fixed), error('V1_4:NoFixedDofs', 'No fixed DOFs produced.'); end
end

function KE = localLk(hx, hy, nu)
D = (1/(1-nu^2)) * [1, nu, 0; nu, 1, 0; 0, 0, (1-nu)/2];
invJ = [2/hx, 0; 0, 2/hy]; detJ = hx*hy/4; gp = 1/sqrt(3);
KE = zeros(8,8);
for xi = [-gp, gp], for eta = [-gp, gp]
    dNdxi  = 0.25*[-(1-eta),(1-eta),(1+eta),-(1+eta)];
    dNdeta = 0.25*[-(1-xi),-(1+xi),(1+xi),(1-xi)];
    dNxy   = invJ*[dNdxi; dNdeta];
    B = zeros(3,8); B(1,1:2:end)=dNxy(1,:); B(2,2:2:end)=dNxy(2,:);
    B(3,1:2:end)=dNxy(2,:); B(3,2:2:end)=dNxy(1,:);
    KE = KE + B'*D*B*detJ;
end, end
end

function ME = localLm(hx, hy)
Ms = (hx*hy/36)*[4,2,1,2;2,4,2,1;1,2,4,2;2,1,2,4];
ME = kron(Ms, eye(2));
end
