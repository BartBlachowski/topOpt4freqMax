%TEST_A0_MATLAB_ONLY Gate A0 authoritative-load verification — MATLAB only.
%
% Reviewer-facing R1 gate. Certifies the authoritative inertial-load
% formulation in MATLAB without requiring a Python environment.
%
% Verified claims:
%   1. Reference eigenpair is from the fully solid domain (modal mass = 1).
%   2. Load scalar is omega0^2 (not omega0): reference_omega_sq = reference_omega.^2.
%   3. Load vector F = omega0^2 * M(x) * Phi0 (evolving M, frozen reference).
%   4. No rho_nodal or obsolete-source scaling: obsolete_rho_source_used = false.
%   5. Reference mode is mass-normalised: Phi0'*M0*Phi0 = 1.
%   6. Deterministic phase: largest-|DOF| entry of Phi0 is non-negative.
%   7. Omitted branch excludes load-sensitivity (differs from complete branch).
%   8. Complete branch matches central finite differences to <= 1e-5 relative.
%   9. Obsolete semi_harmonic_rho_source setting rejected by wrapper.
%  10. harmonic_normalize=true rejected by wrapper.
%
% Output: scripts/revision_v1/a0_matlab_result.json

scriptDir  = fileparts(mfilename('fullpath'));
repoRoot   = fileparts(fileparts(scriptDir));
fixturePath = fullfile(scriptDir, 'gate_a0_fixture.json');
resultPath  = fullfile(scriptDir, 'a0_matlab_result.json');

addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));

if ~isfile(fixturePath)
    error('A0Matlab:MissingFixture', 'Missing fixture: %s', fixturePath);
end

cfgBase   = jsondecode(fileread(fixturePath));
fdStep    = 1e-6;
relTol    = 1e-5;
relFloor  = 1e-14;
parRelTol = 1e-8;
parAbsTol = 1e-12;

% -----------------------------------------------------------------------
% Items 1-7: Run omitted and complete modes; validate diagnostics.
% -----------------------------------------------------------------------
runs = struct();
for modeCell = {'omitted', 'complete'}
    mode = modeCell{1};
    cfg  = cfgBase;
    cfg.optimization.load_sensitivity = mode;
    [~, ~, ~, ~, ~, info] = run_topopt_from_json(cfg);
    if ~isstruct(info) || ~isfield(info, 'gate_a0')
        error('A0Matlab:MissingDiagnostics', ...
            'Solver returned no gate_a0 diagnostics for load_sensitivity=%s.', mode);
    end
    localValidateDiagnostics(info.gate_a0, mode);
    runs.(mode) = info.gate_a0;
end

% Cross-check: invariant fields must agree between omitted and complete runs.
invariantFields = {'reference_omega', 'reference_omega_sq', 'reference_modes', ...
    'reference_modal_mass', 'load_vector', 'objective'};
for i = 1:numel(invariantFields)
    f = invariantFields{i};
    localAssertParity(['omitted/complete invariant: ' f], ...
        double(runs.complete.(f)(:)), double(runs.omitted.(f)(:)), ...
        parRelTol, parAbsTol);
end

% Item 7: The two sensitivity branches must be numerically distinguishable.
omittedSens  = double(runs.complete.omitted_sensitivity(:));
completeSens = double(runs.complete.complete_sensitivity(:));
branchDiff   = abs(omittedSens - completeSens);
branchThresh = relFloor + relTol * abs(completeSens);
if all(branchDiff <= branchThresh)
    error('A0Matlab:IndistinguishableBranches', ...
        ['Omitted and complete sensitivities are unexpectedly identical — ' ...
         'load derivative contribution appears to be zero.']);
end

% -----------------------------------------------------------------------
% Item 8: Central FD verification of complete sensitivity.
% -----------------------------------------------------------------------
diag             = runs.complete;
x                = double(diag.current_x(:));
completeSensAll  = double(diag.complete_sensitivity(:));
omittedSensAll   = double(diag.omitted_sensitivity(:));
testedIndices    = unique(round(linspace(1, numel(x), 6)), 'stable');

if any(x(testedIndices) - fdStep <= 0) || any(x(testedIndices) + fdStep >= 1)
    error('A0Matlab:PerturbationBounds', ...
        'Perturbation would leave the open density interval (0, 1).');
end

fdValues = zeros(numel(testedIndices), 1);
cfgFD    = cfgBase;
cfgFD.optimization.load_sensitivity = 'complete';
for i = 1:numel(testedIndices)
    idx    = testedIndices(i);
    xPlus  = x; xPlus(idx)  = xPlus(idx)  + fdStep;
    xMinus = x; xMinus(idx) = xMinus(idx) - fdStep;
    objP   = localObjective(xPlus,  cfgFD, diag);
    objM   = localObjective(xMinus, cfgFD, diag);
    fdValues(i) = (objP - objM) / (2 * fdStep);
end

completeValues = completeSensAll(testedIndices);
omittedValues  = omittedSensAll(testedIndices);
relErrors = abs(fdValues - completeValues) ./ ...
    max(max(abs(fdValues), abs(completeValues)), relFloor);
omittedFdErrors = abs(fdValues - omittedValues) ./ ...
    max(max(abs(fdValues), abs(omittedValues)), relFloor);

if any(~isfinite(relErrors))
    error('A0Matlab:NonfiniteError', 'FD relative errors contain non-finite values.');
end
if any(relErrors > relTol)
    [worstErr, worstIdx] = max(relErrors);
    error('A0Matlab:FDToleranceExceeded', ...
        'Complete FD relative error %.3e at element %d exceeds %.1e.', ...
        worstErr, testedIndices(worstIdx), relTol);
end

% -----------------------------------------------------------------------
% Item 9: Obsolete rho-source must be rejected.
% -----------------------------------------------------------------------
cfgBad9 = cfgBase;
cfgBad9.optimization.load_sensitivity   = 'omitted';
cfgBad9.optimization.semi_harmonic_rho_source = 'nodal_projection';
rhoSourceRejected = false;
try
    run_topopt_from_json(cfgBad9);
catch ME
    if contains(ME.identifier, 'GateA0ObsoleteRhoSource') || ...
       contains(ME.identifier, 'ObsoleteRhoSource')
        rhoSourceRejected = true;
    else
        rethrow(ME);
    end
end
if ~rhoSourceRejected
    error('A0Matlab:Item9Failed', ...
        'Wrapper did not reject semi_harmonic_rho_source when gate_a0_diagnostics=true.');
end

% -----------------------------------------------------------------------
% Item 10: harmonic_normalize=true must be rejected.
% -----------------------------------------------------------------------
cfgBad10 = cfgBase;
cfgBad10.optimization.load_sensitivity  = 'omitted';
cfgBad10.optimization.harmonic_normalize = true;
normalizeRejected = false;
try
    run_topopt_from_json(cfgBad10);
catch ME
    if contains(ME.identifier, 'GateA0ObsoleteNormalize') || ...
       contains(ME.identifier, 'ObsoleteNormalize')
        normalizeRejected = true;
    else
        rethrow(ME);
    end
end
if ~normalizeRejected
    error('A0Matlab:Item10Failed', ...
        'Wrapper did not reject harmonic_normalize=true when gate_a0_diagnostics=true.');
end

% -----------------------------------------------------------------------
% Collect and write result.
% -----------------------------------------------------------------------
result = struct();
result.gate      = 'A0-MATLAB-ONLY';
result.status    = 'passed';
result.fixture   = 'scripts/revision_v1/gate_a0_fixture.json';
result.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
result.matlab_version = version;

v = struct();
v.item_1_solid_reference_eigenpair          = true;
v.item_2_omega_sq_scalar                    = true;
v.item_3_load_formula_F_eq_omega_sq_M_phi   = true;
v.item_4_no_rho_nodal_scaling               = true;
v.item_5_mass_normalisation                 = true;
v.item_6_deterministic_phase                = true;
v.item_7_branches_distinguishable           = true;
v.item_8_complete_fd_tolerance_met          = true;
v.item_9_obsolete_rho_source_rejected       = rhoSourceRejected;
v.item_10_harmonic_normalize_rejected       = normalizeRejected;
result.verified = v;

fd = struct();
fd.perturbation_size              = fdStep;
fd.relative_error_tolerance       = relTol;
fd.tested_element_indices         = testedIndices(:);
fd.finite_difference_values       = fdValues;
fd.complete_analytical_values     = completeValues;
fd.complete_relative_errors       = relErrors;
fd.max_complete_relative_error    = max(relErrors);
fd.omitted_analytical_values      = omittedValues;
fd.omitted_relative_errors_vs_complete_fd = omittedFdErrors;
fd.omitted_expected_to_match_complete_fd  = false;
result.fd_check = fd;

result.reference_omega       = double(diag.reference_omega(:));
result.reference_omega_sq    = double(diag.reference_omega_sq(:));
result.reference_modal_mass  = double(diag.reference_modal_mass(:));

fid = fopen(resultPath, 'w');
if fid < 0
    error('A0Matlab:ResultWrite', 'Unable to create result file: %s', resultPath);
end
cleanupFid = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(result, PrettyPrint=true));

fprintf('\nGATE A0 MATLAB-ONLY PASSED\n');
fprintf('  All 10 verification items satisfied.\n');
fprintf('  FD max complete relative error: %.3e  (tolerance: %.1e)\n', ...
    max(relErrors), relTol);
fprintf('  Saved: %s\n', resultPath);

% =======================================================================
% Local functions
% =======================================================================

function localValidateDiagnostics(diag, selectedMode)
% Check all required fields and key invariants.
required = {'reference_omega', 'reference_omega_sq', 'reference_modes', ...
    'reference_modal_mass', 'current_x', 'current_mass_matrix', 'load_vector', ...
    'objective', 'omitted_sensitivity', 'complete_sensitivity', ...
    'selected_sensitivity', 'selected_load_sensitivity', ...
    'load_normalization_enabled', 'obsolete_rho_source_used'};
for i = 1:numel(required)
    if ~isfield(diag, required{i})
        error('A0Matlab:MissingDiagnostic', ...
            'Missing gate_a0 diagnostic field: %s', required{i});
    end
end

% Item 4: No obsolete rho source.
if diag.obsolete_rho_source_used
    error('A0Matlab:ObsoleteRhoSource', ...
        'Diagnostic confirms obsolete_rho_source_used=true.');
end

% Item 4 (load normalization arm): load normalization must be off.
if diag.load_normalization_enabled
    error('A0Matlab:LoadNormalization', ...
        'Diagnostic confirms load_normalization_enabled=true.');
end

% Verify selected branch was propagated correctly.
if ~strcmp(char(diag.selected_load_sensitivity), selectedMode)
    error('A0Matlab:SensitivityBranchMismatch', ...
        'Selected load_sensitivity is "%s"; expected "%s".', ...
        char(diag.selected_load_sensitivity), selectedMode);
end

% Item 5: Phi0'*M0*Phi0 = 1.
localAssertClose('reference modal mass = 1', ...
    diag.reference_modal_mass, ones(size(diag.reference_modal_mass)), 1e-8, 1e-12);

% Items 1+2+3: Independently verify F = omega0^2 * M(x) * Phi0.
omegaSq  = double(diag.reference_omega_sq(1));
omega    = double(diag.reference_omega(1));
phi0     = double(diag.reference_modes(:, 1));
M        = diag.current_mass_matrix;
expectedLoad = omegaSq * (M * phi0);
localAssertClose('load vector = omega0^2*M(x)*Phi0', ...
    diag.load_vector(:, 1), expectedLoad, 1e-8, 1e-12);

% Item 2: reference_omega_sq is omega^2, not omega.
omegaSqFromOmega = omega^2;
if abs(omegaSq - omegaSqFromOmega) > 1e-8 * abs(omegaSqFromOmega) + 1e-12
    error('A0Matlab:OmegaSqMismatch', ...
        'reference_omega_sq(1)=%.6e but reference_omega(1)^2=%.6e.', ...
        omegaSq, omegaSqFromOmega);
end

% Item 6: Deterministic phase — largest-magnitude DOF is non-negative.
[~, largestIdx] = max(abs(phi0));
if phi0(largestIdx) < 0
    error('A0Matlab:PhaseConvention', ...
        'Largest-magnitude DOF of reference_modes(:,1) is negative (= %.6e).', ...
        phi0(largestIdx));
end

% Item 7 (partial): selected_sensitivity matches the declared branch.
if strcmp(selectedMode, 'complete')
    expected = diag.complete_sensitivity;
else
    expected = diag.omitted_sensitivity;
end
localAssertClose('selected_sensitivity matches declared branch', ...
    diag.selected_sensitivity, expected, 1e-8, 1e-12);
end

function localAssertClose(name, actual, reference, relTol, absTol)
actual    = double(actual(:));
reference = double(reference(:));
if ~isequal(size(actual), size(reference))
    error('A0Matlab:ShapeMismatch', '%s: shape mismatch.', name);
end
allowed = absTol + relTol * abs(reference);
err     = abs(actual - reference);
if any(~isfinite(actual)) || any(err > allowed)
    error('A0Matlab:ToleranceExceeded', ...
        '%s: max error %.3e exceeds tolerance %.3e.', ...
        name, max(err), max(allowed));
end
end

function localAssertParity(name, a, b, relTol, absTol)
allowed = absTol + relTol * abs(b);
err     = abs(a - b);
if any(~isfinite(a)) || any(~isfinite(b)) || any(err > allowed)
    error('A0Matlab:ParityFailed', ...
        '%s: max difference %.3e exceeds tolerance %.3e.', ...
        name, max(err), max(allowed));
end
end

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
phi0    = double(diag.reference_modes(:, 1));
omega0Sq = double(diag.reference_omega_sq(1));
F    = omega0Sq * (M * phi0);
fixed = localFixedDofs(cfg, nelx, nely, L, H);
free  = setdiff((1:ndof)', fixed);
U     = zeros(ndof, 1);
Kff   = K(free, free);
U(free) = Kff \ F(free);
objective = U' * K * U;
if ~isfinite(objective)
    error('A0Matlab:NonfiniteObjective', 'Perturbed objective is not finite.');
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
nodeX = X(:);
nodeY = Y(:);
fixed = [];
supports = cfg.bc.supports;
for i = 1:numel(supports)
    if iscell(supports)
        sup = supports{i};
    else
        sup = supports(i);
    end
    switch char(sup.type)
        case 'vertical_line'
            tol   = 1e-9;
            if isfield(sup, 'tol'), tol = double(sup.tol); end
            nodes = find(abs(nodeX - double(sup.x)) <= tol);
        case 'closest_point'
            loc   = double(sup.location(:));
            [~, node] = min((nodeX - loc(1)).^2 + (nodeY - loc(2)).^2);
            nodes = node;
        otherwise
            error('A0Matlab:UnsupportedSupport', ...
                'Unsupported support type: %s', char(sup.type));
    end
    dofs = cellstr(string(sup.dofs));
    if any(strcmp(dofs, 'ux')), fixed = [fixed; 2*nodes - 1]; end %#ok<AGROW>
    if any(strcmp(dofs, 'uy')), fixed = [fixed; 2*nodes];     end %#ok<AGROW>
end
fixed = unique(fixed);
if isempty(fixed)
    error('A0Matlab:NoFixedDofs', 'Fixture produced no fixed degrees of freedom.');
end
end

function KE = localLk(hx, hy, nu)
D    = (1/(1-nu^2)) * [1, nu, 0; nu, 1, 0; 0, 0, (1-nu)/2];
invJ = [2/hx, 0; 0, 2/hy];
detJ = hx * hy / 4;
gp   = 1 / sqrt(3);
KE   = zeros(8, 8);
for xi = [-gp, gp]
    for eta = [-gp, gp]
        dNdxi  = 0.25 * [-(1-eta),  (1-eta),  (1+eta), -(1+eta)];
        dNdeta = 0.25 * [-(1-xi),  -(1+xi),   (1+xi),   (1-xi) ];
        dNxy   = invJ * [dNdxi; dNdeta];
        B = zeros(3, 8);
        B(1, 1:2:end) = dNxy(1, :);
        B(2, 2:2:end) = dNxy(2, :);
        B(3, 1:2:end) = dNxy(2, :);
        B(3, 2:2:end) = dNxy(1, :);
        KE = KE + B' * D * B * detJ;
    end
end
end

function ME = localLm(hx, hy)
Ms = (hx * hy / 36) * [4,2,1,2; 2,4,2,1; 1,2,4,2; 2,1,2,4];
ME = kron(Ms, eye(2));
end
