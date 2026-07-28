function results = test_a4_phase4()
%TEST_A4_PHASE4  Regression tests for Phase-3 blockers P3-1 and P3-2.
%
% Uses the authoritative immutable topology fixture for V-P2-2 and a tiny real-driver
% run for the HALTED control flow. No production sweep is executed.

fprintf('\n=== test_a4_phase4 (blocker regressions; no production sweep) ===\n');
nPass = 0;
nFail = 0;

thisDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(thisDir));
rv1 = fullfile(repoRoot, 'examples', 'Revision_v1');
addpath(thisDir);
addpath(rv1);
addpath(fullfile(repoRoot, 'tools', 'Matlab'));

tmpDir = tempname;
mkdir(tmpDir);
cleanup = onCleanup(@() rmdir(tmpDir, 's')); %#ok<NASGU>

% ---- P3-1: exact scalars + exact topology-file SHA, not lossy reload ----
reference = a4_frozen_baseline_reference(tmpDir);
preservedPath = reference.path;
csvValues = readmatrix(preservedPath);
fullPrecision = csvValues(:);
fractional = find(fullPrecision > 0 & fullPrecision < 1);
for i = 1:numel(fractional)
    index = fractional(i);
    candidate = fullPrecision(index) + 4 * eps(fullPrecision(index));
    if strcmp(sprintf('%.15g', candidate), sprintf('%.15g', fullPrecision(index)))
        fullPrecision(index) = candidate;
    end
end

matchingPath = fullfile(tmpDir, 'matching_topology.csv');
copyfile(preservedPath, matchingPath);
baseline = a4_validate_frozen_baseline( ...
    preservedPath, reference.expected_sha256, tmpDir, matchingPath);
arm = localExpectedArm(fullPrecision);
gate = a4_validate_frozen_identity(arm, matchingPath, baseline);

[nPass,nFail] = ck( ...
    'P3-1 matching topology SHA passes despite several-ulp CSV reload loss', ...
    gate.pass && gate.exact_scalar_identity && gate.topology_file_identity && ...
    ~gate.numerical_diagnostics.csv_reload_exactly_equals_in_memory && ...
    gate.numerical_diagnostics.n_different_values > 0, nPass, nFail);

wrongTopology = fullPrecision;
wrongTopology(find(wrongTopology > 0, 1, 'first')) = ...
    wrongTopology(find(wrongTopology > 0, 1, 'first')) + 1e-3;
wrongPath = fullfile(tmpDir, 'wrong_topology.csv');
localWrite15g(wrongPath, wrongTopology);
wrongBaseline = a4_validate_frozen_baseline( ...
    preservedPath, reference.expected_sha256, tmpDir, wrongPath);
wrongHashGate = a4_validate_frozen_identity(arm, wrongPath, wrongBaseline);
[nPass,nFail] = ck( ...
    'P3-1 wrong topology CSV SHA fails V-P2-2', ...
    ~wrongHashGate.pass && wrongHashGate.exact_scalar_identity && ...
    ~wrongHashGate.topology_file_identity && ...
    any(contains(wrongHashGate.failures, 'SHA-256')), nPass, nFail);

wrongScalarArm = arm;
wrongScalarArm.omega1_tracked = arm.omega1_tracked + eps(arm.omega1_tracked);
wrongScalarGate = a4_validate_frozen_identity(wrongScalarArm, matchingPath, baseline);
[nPass,nFail] = ck( ...
    'P3-1 wrong exact scalar invariant fails V-P2-2', ...
    ~wrongScalarGate.pass && ~wrongScalarGate.exact_scalar_identity && ...
    wrongScalarGate.topology_file_identity && ...
    ~wrongScalarGate.scalar_identity.omega1_tracked, nPass, nFail);

% ---- P3-2: real tiny driver, forced naturally through V-P2-2 mismatch ----
haltDir = fullfile(tmpDir, 'halt');
mkdir(haltDir);
localWriteText(fullfile(haltDir, 'a4_manifest.json'), '{"stale":true}');
localWriteText(fullfile(haltDir, 'a4_stage_manifest.json'), '{"stale":true}');

base = jsondecode(fileread(fullfile(rv1, 'a4_ss_400x50_base.json')));
base.domain.mesh.nelx = 40;
base.domain.mesh.nely = 5;
base.optimization.max_iters = 4;
base.optimization.convergence_tol = 1e-16;
base.optimization.volume_fraction = 0.9;
tinyPath = fullfile(tmpDir, 'a4_tiny_halt_base.json');
localWriteText(tinyPath, jsonencode(base, PrettyPrint=true));

opts = struct( ...
    'base_config', tinyPath, ...
    'n_levels', [Inf, 2], ...
    'n_modes', 6, ...
    'enforce_frozen_bit_identity', true, ...
    'run_nonperturbation_replay', false, ...
    'run_finite_replay', false, ...
    'run_fixture_validators', false);

caught = [];
try
    evalc('a4_eigenpair_refresh(haltDir, opts);');
catch ME
    caught = ME;
end

[nPass,nFail] = ck( ...
    'P3-2 original HALTED reason propagates without missing-field replacement', ...
    ~isempty(caught) && strcmp(caught.identifier, 'a4:FrozenBitIdentityFailed') && ...
    ~contains(caught.message, 'acceptance_checks') && ...
    ~contains(caught.message, 'Unrecognized field'), nPass, nFail);

required = { ...
    'A4_RECOVERY_PHASE2_REPORT.md', ...
    'A4_RECOVERY_PHASE2_VALIDATION.md', ...
    'a4_manifest.json', ...
    'a4_stage_manifest.json', ...
    'a4_stage_result.json', ...
    'a4_eigenpair_refresh_results.mat', ...
    'a4_result.json', ...
    'a4_screening_events.json', ...
    'a4_candidate_telemetry.csv', ...
    'a4_iteration_histories.csv'};
present = cellfun(@(name) localNonempty(fullfile(haltDir, name)), required);
[nPass,nFail] = ck( ...
    'P3-2 HALTED path writes reports, manifests, result, and telemetry artifacts', ...
    all(present), nPass, nFail);

persistedOk = false;
telemetryOk = false;
manifestOk = false;
reasonOk = false;
try
    saved = load(fullfile(haltDir, 'a4_eigenpair_refresh_results.mat'), 'res');
    persisted = saved.res;
    persistedOk = strcmp(persisted.run_verdict, 'HALTED') && ...
        persisted.halt.halted && ...
        strcmp(persisted.halt.identifier, 'a4:FrozenBitIdentityFailed') && ...
        numel(persisted.arms) == 1;
    telemetryOk = ~isempty(persisted.arms(1).screening_events) && ...
        ~isempty(persisted.arms(1).candidate_telemetry) && ...
        ~isempty(persisted.arms(1).iteration_histories);

    manifest = jsondecode(fileread(fullfile(haltDir, 'a4_manifest.json')));
    stageManifest = jsondecode(fileread(fullfile(haltDir, 'a4_stage_manifest.json')));
    manifestOk = isequal(manifest.files, stageManifest.files) && ...
        strcmp(manifest.run_verdict, 'HALTED') && ...
        ~isfield(manifest, 'stale') && ~isfield(stageManifest, 'stale');

    reportText = fileread(fullfile(haltDir, 'A4_RECOVERY_PHASE2_REPORT.md'));
    resultJson = jsondecode(fileread(fullfile(haltDir, 'a4_result.json')));
    reasonOk = contains(reportText, 'a4:FrozenBitIdentityFailed') && ...
        strcmp(resultJson.run_verdict, 'HALTED') && ...
        strcmp(resultJson.halt.identifier, 'a4:FrozenBitIdentityFailed');
catch
end

[nPass,nFail] = ck( ...
    'P3-2 accumulated arm, event, candidate, and history telemetry persists', ...
    persistedOk && telemetryOk, nPass, nFail);
[nPass,nFail] = ck( ...
    'P3-2 matched manifests replace stale manifests', manifestOk, nPass, nFail);
[nPass,nFail] = ck( ...
    'P3-2 true halt reason is present in persisted report and JSON', ...
    reasonOk, nPass, nFail);

results = struct('passed', nPass, 'failed', nFail);
fprintf('  passed: %d   failed: %d\n', nPass, nFail);
if nFail > 0
    error('test_a4_phase4:Failed', '%d Phase-4 regression(s) failed.', nFail);
end
end

function arm = localExpectedArm(topology)
arm = struct( ...
    'omega1_tracked', 159.56562699328325, ...
    'final_design_change', 3.034903639330122e-03, ...
    'iterations', 2000, ...
    'mode_index_jstar', 1, ...
    'mac_to_phi0', 0.9996284251363903, ...
    'omega1_omega2_gap', 67.37267502573462, ...
    'topology', topology(:));
end

function localWrite15g(path, values)
fid = fopen(path, 'w');
if fid < 0, error('test_a4_phase4:WriteFailed', 'Cannot write %s', path); end
cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%.15g\n', values);
end

function localWriteText(path, text)
fid = fopen(path, 'w');
if fid < 0, error('test_a4_phase4:WriteFailed', 'Cannot write %s', path); end
cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', text);
end

function tf = localNonempty(path)
tf = isfile(path) && dir(path).bytes > 0;
end

function [passed,failed] = ck(name,condition,passed,failed)
if condition
    fprintf('  [PASS] %s\n', name);
    passed = passed + 1;
else
    fprintf(2, '  [FAIL] %s\n', name);
    failed = failed + 1;
end
end
