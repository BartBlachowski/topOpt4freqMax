% check_edof_and_harmonic_sensitivity_optionB.m
% Quick deterministic checks for:
%   1) edofMat indexing/connectivity (indirect: harmonic smoke run must assemble/solve),
%   2) ld.factor scaling for closest_node and harmonic loads,
%   3) Option B harmonic sensitivity activation (dM/dx term only, no eigen-derivatives).

clear; clc;
fprintf('=== check_edof_and_harmonic_sensitivity_optionB ===\n');

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot, 'tools'));
addpath(fullfile(repoRoot, 'ourApproach', 'Matlab'));

% Tiny fast setup.
nelx = 6; nely = 2;
volfrac = 0.5;
penal = 3.0;
rmin = 0.4;
ft = 0;            % sensitivity-like branch in this code
L = 3.0; H = 1.0;
ndof = 2 * (nelx + 1) * (nely + 1);

base = struct();
base.visualize_live = false;
base.save_frq_iterations = false;
base.max_iters = 1;
base.conv_tol = 1e-12;
base.supportType = 'CC';
base.approach_name = 'ourApproach';
base.optimizer = 'OC';
base.harmonic_normalize = false;  % keep factor-scaling checks direct
base.debug_return_dc = true;
base.E0 = 1e7;
base.Emin = 1e-2;
base.rho0 = 1.0;
base.rho_min = 1e-6;
base.pmass = 1.0;

% -------------------------------------------------------------------------
% 1) Smoke test: harmonic run should execute without indexing errors.
% -------------------------------------------------------------------------
cfg = base;
cfg.load_cases = makeSingleCase(struct( ...
    'type', 'harmonic', ...
    'mode', 1, ...
    'update_after', 1, ...
    'factor', 1.0), ...
    'harmonic_smoke');
[~, ~, ~, ~, infoSmoke] = topopt_freq(nelx, nely, volfrac, penal, rmin, ft, L, H, cfg);
assert(isfield(infoSmoke, 'last_F'), 'Smoke test failed: info.last_F missing.');
assert(isequal(size(infoSmoke.last_F), [ndof, 1]), ...
    'Smoke test failed: unexpected F size, possible edof/connectivity issue.');
fprintf('[OK] Harmonic smoke run completed and F has expected size.\n');

% -------------------------------------------------------------------------
% 2) ld.factor scaling for closest_node.
% -------------------------------------------------------------------------
cfgC1 = base;
cfgC1.load_cases = makeSingleCase(struct( ...
    'type', 'closest_node', ...
    'location', [L/2, H/2], ...
    'force', [0.0, -1.0], ...
    'factor', 1.0), ...
    'closest_factor1');
[~, ~, ~, ~, infoC1] = topopt_freq(nelx, nely, volfrac, penal, rmin, ft, L, H, cfgC1);

cfgC3 = cfgC1;
cfgC3.load_cases = makeSingleCase(struct( ...
    'type', 'closest_node', ...
    'location', [L/2, H/2], ...
    'force', [0.0, -1.0], ...
    'factor', 3.0), ...
    'closest_factor3');
[~, ~, ~, ~, infoC3] = topopt_freq(nelx, nely, volfrac, penal, rmin, ft, L, H, cfgC3);

rClosest = norm(infoC3.last_F(:,1)) / norm(infoC1.last_F(:,1));
assert(abs(rClosest - 3.0) < 1e-12, ...
    'closest_node ld.factor scaling failed: expected ratio 3.');
fprintf('[OK] closest_node ld.factor scaling ratio = %.16f\n', rClosest);

% -------------------------------------------------------------------------
% 3) ld.factor scaling for harmonic.
% -------------------------------------------------------------------------
cfgH1 = base;
cfgH1.load_cases = makeSingleCase(struct( ...
    'type', 'harmonic', ...
    'mode', 1, ...
    'update_after', 1, ...
    'factor', 1.0), ...
    'harmonic_factor1');
[~, ~, ~, ~, infoH1] = topopt_freq(nelx, nely, volfrac, penal, rmin, ft, L, H, cfgH1);

cfgH2 = cfgH1;
cfgH2.load_cases = makeSingleCase(struct( ...
    'type', 'harmonic', ...
    'mode', 1, ...
    'update_after', 1, ...
    'factor', 2.0), ...
    'harmonic_factor2');
[~, ~, ~, ~, infoH2] = topopt_freq(nelx, nely, volfrac, penal, rmin, ft, L, H, cfgH2);

rHarm = norm(infoH2.last_F(:,1)) / norm(infoH1.last_F(:,1));
assert(abs(rHarm - 2.0) < 1e-10, ...
    'harmonic ld.factor scaling failed: expected ratio 2.');
fprintf('[OK] harmonic ld.factor scaling ratio = %.16f\n', rHarm);

% -------------------------------------------------------------------------
% 4) Option B harmonic sensitivity active.
%
% Make stiffness interpolation flat (E0 == Emin), so stiffness-part dc is zero.
% Then:
%   - closest_node load should give near-zero dc (no dF/dx),
%   - harmonic load should give non-zero dc via Option B mass-term dF/dx.
% -------------------------------------------------------------------------
cfgSensBase = base;
cfgSensBase.E0 = 1e5;
cfgSensBase.Emin = 1e5;

cfgSensClosest = cfgSensBase;
cfgSensClosest.load_cases = makeSingleCase(struct( ...
    'type', 'closest_node', ...
    'location', [L/2, H/2], ...
    'force', [0.0, -1.0], ...
    'factor', 1.0), ...
    'closest_no_mass_sens');
[~, ~, ~, ~, infoSensClosest] = topopt_freq(nelx, nely, volfrac, penal, rmin, ft, L, H, cfgSensClosest);

cfgSensHarm = cfgSensBase;
cfgSensHarm.load_cases = makeSingleCase(struct( ...
    'type', 'harmonic', ...
    'mode', 1, ...
    'update_after', 1, ...
    'factor', 1.0), ...
    'harmonic_optionB');
[~, ~, ~, ~, infoSensHarm] = topopt_freq(nelx, nely, volfrac, penal, rmin, ft, L, H, cfgSensHarm);

dcClosestInf = norm(infoSensClosest.last_dc, inf);
dcHarmInf = norm(infoSensHarm.last_dc, inf);
assert(dcClosestInf < 1e-12, ...
    'Option B check failed: closest_node dc should be ~0 when E0==Emin.');
assert(dcHarmInf > 1e-12, ...
    'Option B check failed: harmonic dc should be non-zero (mass-term sensitivity).');
fprintf('[OK] Option B sensitivity active: ||dc||_inf closest=%.3e, harmonic=%.3e\n', ...
    dcClosestInf, dcHarmInf);

fprintf('=== check_edof_and_harmonic_sensitivity_optionB PASSED ===\n');

function caseStruct = makeSingleCase(loadStruct, caseName)
caseStruct = struct('name', caseName, 'factor', 1.0, 'loads', loadStruct);
end
