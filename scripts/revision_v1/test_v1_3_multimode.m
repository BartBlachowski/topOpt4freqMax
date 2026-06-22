%TEST_V1_3_MULTIMODE V1-3: Weighted multi-mode objective assembly regression.
%
% Verifies that the solver correctly assembles the compliance objective as an
% unweighted sum over load cases, and that the per-case factor enters as
% factor^2 (because F_case = factor * F_raw, so U = K^-1 F_case = factor * U_raw,
% and obj_case = U^T K U = factor^2 * U_raw^T K U_raw).
%
% Test design (6x2 mesh, gate_a0_fixture basis):
%   Run A: mode-1 load case only (factor=1)   -> obj_A
%   Run B: mode-2 load case only (factor=1)   -> obj_B
%   Run C: both load cases (factors 1, 1)     -> obj_C  must equal obj_A + obj_B
%   Run D: mode-1 factor=2, mode-2 factor=3   -> obj_D  must equal 4*obj_A + 9*obj_B
%
% Output: scripts/revision_v1/v1_3_multimode_results.json

scriptDir  = fileparts(mfilename('fullpath'));
repoRoot   = fileparts(fileparts(scriptDir));
fixturePath = fullfile(scriptDir, 'gate_a0_fixture.json');
resultPath  = fullfile(scriptDir, 'v1_3_multimode_results.json');

addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));

if ~isfile(fixturePath)
    error('V1_3:MissingFixture', 'Missing fixture: %s', fixturePath);
end

cfgBase = jsondecode(fileread(fixturePath));

% Disable gate_a0 diagnostics (not needed here) and verbose output.
cfgBase.optimization.gate_a0_diagnostics = false;
cfgBase.optimization.load_sensitivity    = 'omitted';  % doesn't matter at max_iters=1

relTol = 1e-10;  % for objective additivity check (floating-point arithmetic only)

% -----------------------------------------------------------------
% Run A: mode-1 load case only, factor=1
% -----------------------------------------------------------------
cfgA = cfgBase;
cfgA.domain.load_cases = localMakeCase('mode_1_only', 1, 1.0);
[~,~,~,~,~, infoA] = run_topopt_from_json(cfgA);
obj_A = double(infoA.last_obj);
if ~isfinite(obj_A) || obj_A <= 0
    error('V1_3:RunA', 'Run A objective is not a positive finite scalar (%.6e).', obj_A);
end
fprintf('[V1-3] obj_A (mode 1, factor=1) = %.10e\n', obj_A);

% -----------------------------------------------------------------
% Run B: mode-2 load case only, factor=1
% -----------------------------------------------------------------
cfgB = cfgBase;
cfgB.domain.load_cases = localMakeCase('mode_2_only', 2, 1.0);
[~,~,~,~,~, infoB] = run_topopt_from_json(cfgB);
obj_B = double(infoB.last_obj);
if ~isfinite(obj_B) || obj_B <= 0
    error('V1_3:RunB', 'Run B objective is not a positive finite scalar (%.6e).', obj_B);
end
fprintf('[V1-3] obj_B (mode 2, factor=1) = %.10e\n', obj_B);

% -----------------------------------------------------------------
% Run C: both load cases, factors 1 and 1 -> must equal obj_A + obj_B
% -----------------------------------------------------------------
cfgC = cfgBase;
cfgC.domain.load_cases = localMakeTwoCases(1, 1.0, 2, 1.0);
[~,~,~,~,~, infoC] = run_topopt_from_json(cfgC);
obj_C = double(infoC.last_obj);
expected_C = obj_A + obj_B;
err_C = abs(obj_C - expected_C) / max(expected_C, 1e-12);
fprintf('[V1-3] obj_C (modes 1+2, factors 1,1) = %.10e\n', obj_C);
fprintf('[V1-3] expected_C                      = %.10e\n', expected_C);
fprintf('[V1-3] relative error (additivity)     = %.3e  (tol %.1e)\n', err_C, relTol);
if err_C > relTol
    error('V1_3:AdditivityFailed', ...
        'Multi-mode objective %.6e != obj_A + obj_B = %.6e  (rel err %.3e).', ...
        obj_C, expected_C, err_C);
end

% -----------------------------------------------------------------
% Run D: mode-1 factor=2, mode-2 factor=3 -> must equal 4*obj_A + 9*obj_B
% -----------------------------------------------------------------
cfgD = cfgBase;
cfgD.domain.load_cases = localMakeTwoCases(1, 2.0, 2, 3.0);
[~,~,~,~,~, infoD] = run_topopt_from_json(cfgD);
obj_D = double(infoD.last_obj);
expected_D = 4.0 * obj_A + 9.0 * obj_B;
err_D = abs(obj_D - expected_D) / max(expected_D, 1e-12);
fprintf('[V1-3] obj_D (modes 1+2, factors 2,3) = %.10e\n', obj_D);
fprintf('[V1-3] expected_D (4*A + 9*B)         = %.10e\n', expected_D);
fprintf('[V1-3] relative error (factor scaling) = %.3e  (tol %.1e)\n', err_D, relTol);
if err_D > relTol
    error('V1_3:FactorScalingFailed', ...
        'Weighted objective %.6e != 4*obj_A + 9*obj_B = %.6e  (rel err %.3e).', ...
        obj_D, expected_D, err_D);
end

% -----------------------------------------------------------------
% Collect and write result.
% -----------------------------------------------------------------
result = struct();
result.gate    = 'V1-3';
result.status  = 'passed';
result.fixture = 'scripts/revision_v1/gate_a0_fixture.json';
result.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
result.matlab_version = version;

result.objective_additivity = struct( ...
    'description', 'obj(case1+case2) == obj(case1) + obj(case2) with factors=1', ...
    'obj_A_mode1_factor1', obj_A, ...
    'obj_B_mode2_factor1', obj_B, ...
    'obj_C_both_factors_1_1', obj_C, ...
    'expected_C', expected_C, ...
    'relative_error', err_C, ...
    'tolerance', relTol, ...
    'passed', err_C <= relTol);

result.factor_square_scaling = struct( ...
    'description', 'obj(mode1_f2 + mode2_f3) == 4*obj_A + 9*obj_B', ...
    'obj_D_factors_2_3', obj_D, ...
    'expected_D', expected_D, ...
    'relative_error', err_D, ...
    'tolerance', relTol, ...
    'passed', err_D <= relTol);

fid = fopen(resultPath, 'w');
if fid < 0
    error('V1_3:ResultWrite', 'Unable to create result file: %s', resultPath);
end
cleanupFid = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(result, PrettyPrint=true));

fprintf('\nV1-3 PASSED\n');
fprintf('  Objective additivity verified: rel err %.3e (tol %.1e)\n', err_C, relTol);
fprintf('  Factor^2 scaling verified:     rel err %.3e (tol %.1e)\n', err_D, relTol);
fprintf('  Saved: %s\n', resultPath);

% =========================================================================
function lc = localMakeCase(name, mode, factor)
%LOCALMAKECASE  Build a single-entry load_cases struct for run_topopt_from_json.
ld.type   = 'semi_harmonic';
ld.mode   = mode;
ld.factor = 1.0;
lc.name   = name;
lc.factor = factor;
lc.loads  = ld;
end

function lcs = localMakeTwoCases(mode1, factor1, mode2, factor2)
%LOCALMAKETWO  Build a two-entry load_cases struct array.
lcs = [localMakeCase(sprintf('mode_%d', mode1), mode1, factor1), ...
       localMakeCase(sprintf('mode_%d', mode2), mode2, factor2)];
end
