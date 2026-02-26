% test_multi_load_cases_ourApproach.m
% Validates multi-load-case RHS/solution assembly for ourApproach.

fprintf('=== test_multi_load_cases_ourApproach ===\n');

thisDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
addpath(fullfile(repoRoot, 'tools'));
addpath(fullfile(repoRoot, 'ourApproach', 'Matlab'));

% Small mesh for fast regression check.
nelx = 24;
nely = 8;
volfrac = 0.5;
penal = 3.0;
rmin = 0.25;
ft = 0;
L = 8.0;
H = 1.0;

% Common runtime config.
runCfgBase = struct();
runCfgBase.visualize_live = false;
runCfgBase.save_frq_iterations = false;
runCfgBase.max_iters = 1;
runCfgBase.conv_tol = 1e-9;
runCfgBase.supportType = 'SS';
runCfgBase.approach_name = 'ourApproach';

% Two load cases:
%   case1: self-weight + nodal load + harmonic mode 1 + semi_harmonic mode 1
%   case2: nodal load only
case1 = struct( ...
    'name', 'case1', ...
    'factor', 1.0, ...
    'loads', {{ ...
        struct('type', 'self_weight', 'factor', 1.0), ...
        struct('type', 'closest_node', 'location', [4.0, 0.0], 'force', [0.0, -10.0]), ...
        struct('type', 'harmonic', 'mode', 1), ...
        struct('type', 'semi_harmonic', 'mode', 1) ...
    }});

case2 = struct( ...
    'name', 'case2', ...
    'factor', 1.5, ...
    'loads', {{ ...
        struct('type', 'closest_node', 'location', [8.0, 0.5], 'force', [0.0, -10.0]) ...
    }});

runCfgMulti = runCfgBase;
runCfgMulti.load_cases = [case1, case2];

[~, ~, ~, ~, infoMulti] = topopt_freq(nelx, nely, volfrac, penal, rmin, ft, L, H, runCfgMulti);

ndof = 2 * (nelx + 1) * (nely + 1);
assert(isfield(infoMulti, 'last_F') && isequal(size(infoMulti.last_F), [ndof, 2]), ...
    'Expected info.last_F to have size [ndof, 2].');
assert(isfield(infoMulti, 'last_U') && isequal(size(infoMulti.last_U), [ndof, 2]), ...
    'Expected info.last_U to have size [ndof, 2].');

fprintf('PASS: F size = [%d, %d], U size = [%d, %d]\n', ...
    size(infoMulti.last_F,1), size(infoMulti.last_F,2), ...
    size(infoMulti.last_U,1), size(infoMulti.last_U,2));

% Multi-case compliance should equal sum of per-case compliance contributions.
cFromCases = sum(infoMulti.last_obj_cases);
cFromDot = sum(sum(infoMulti.last_F .* infoMulti.last_U));
scale = max(1.0, abs(infoMulti.last_obj));
assert(abs(infoMulti.last_obj - cFromCases) <= 1e-8 * scale, ...
    'Total compliance does not match sum(info.last_obj_cases).');
assert(abs(infoMulti.last_obj - cFromDot) <= 1e-8 * scale, ...
    'Total compliance does not match sum(F(:,i)^T U(:,i)).');

% Compare with two independent single-case runs (same initial design, 1 iteration).
runCfgCase1 = runCfgBase;
runCfgCase1.load_cases = case1;
[~, ~, ~, ~, infoCase1] = topopt_freq(nelx, nely, volfrac, penal, rmin, ft, L, H, runCfgCase1);

runCfgCase2 = runCfgBase;
runCfgCase2.load_cases = case2;
[~, ~, ~, ~, infoCase2] = topopt_freq(nelx, nely, volfrac, penal, rmin, ft, L, H, runCfgCase2);

cSingleSum = infoCase1.last_obj + infoCase2.last_obj;
assert(abs(infoMulti.last_obj - cSingleSum) <= 1e-8 * max(1.0, abs(cSingleSum)), ...
    'Multi-case compliance does not match sum of independent single-case compliances.');

fprintf('PASS: compliance aggregation check succeeded.\n');
fprintf('=== test_multi_load_cases_ourApproach PASSED ===\n');
