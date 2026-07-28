function results = test_a4_phase5a()
%TEST_A4_PHASE5A  Immutable frozen-baseline lifecycle regressions.
%
% Destructive lifecycle checks operate only in temporary directories. The
% authoritative reference fixture is read-only throughout. No production
% optimization is executed.

fprintf('\n=== test_a4_phase5a (immutable baseline lifecycle) ===\n');
nPass = 0;
nFail = 0;

thisDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(thisDir));
rv1 = fullfile(repoRoot, 'examples', 'Revision_v1');
productionOutput = fullfile(rv1, 'output', 'a4');
addpath(thisDir);

tmpDir = tempname;
mkdir(tmpDir);
cleanup = onCleanup(@() rmdir(tmpDir, 's')); %#ok<NASGU>

expectedSha = ...
    '9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806';
baseline = a4_frozen_baseline_reference(productionOutput);

[nPass,nFail] = ck('immutable reference exists with the declared SHA-256', ...
    isfile(baseline.path) && strcmp(baseline.actual_sha256, expectedSha), ...
    nPass, nFail);
[nPass,nFail] = ck('immutable reference is outside mutable output', ...
    ~startsWith(baseline.path, [baseline.mutable_output_dir filesep]), ...
    nPass, nFail);
[nPass,nFail] = ck('reference and produced topology paths are distinct', ...
    ~strcmp(baseline.path, baseline.produced_topology_path), nPass, nFail);

outputEntries = dir(productionOutput);
outputEntries = outputEntries(~ismember({outputEntries.name}, {'.','..'}));
[nPass,nFail] = ck('production output/a4 is empty at pre-run', ...
    isempty(outputEntries), nPass, nFail);
[nPass,nFail] = ck('no immutable baseline prerequisite exists in output/a4', ...
    ~isfile(fullfile(productionOutput, 'a4_topology_Ninf.csv')), nPass, nFail);

% Moving/deleting mutable output must not affect the immutable fixture.
lifecycleOutput = fullfile(tmpDir, 'mutable_output');
mkdir(lifecycleOutput);
copyfile(baseline.path, fullfile(lifecycleOutput, 'disposable.csv'));
movedOutput = fullfile(tmpDir, 'moved_output');
movefile(lifecycleOutput, movedOutput);
rmdir(movedOutput, 's');
[nPass,nFail] = ck('moving or deleting mutable output preserves baseline', ...
    isfile(baseline.path) && strcmp(a4_sha256_file(baseline.path), expectedSha), ...
    nPass, nFail);

% A byte-identical produced topology passes V-P2-2.
producedDir = fullfile(tmpDir, 'produced');
mkdir(producedDir);
producedPath = fullfile(producedDir, 'a4_topology_inf.csv');
copyfile(baseline.path, producedPath);
tempBaseline = a4_validate_frozen_baseline( ...
    baseline.path, expectedSha, producedDir, producedPath);
arm = localExpectedArm(readmatrix(baseline.path));
matchingGate = a4_validate_frozen_identity(arm, producedPath, tempBaseline);
[nPass,nFail] = ck('byte-identical produced topology passes V-P2-2', ...
    matchingGate.pass && matchingGate.topology_file_identity, nPass, nFail);

% A changed produced file must fail V-P2-2.
modifiedProduced = fullfile(producedDir, 'modified_topology.csv');
copyfile(baseline.path, modifiedProduced);
localAppendNewline(modifiedProduced);
modifiedBaseline = a4_validate_frozen_baseline( ...
    baseline.path, expectedSha, producedDir, modifiedProduced);
modifiedGate = a4_validate_frozen_identity( ...
    arm, modifiedProduced, modifiedBaseline);
[nPass,nFail] = ck('modified produced topology fails V-P2-2', ...
    ~modifiedGate.pass && ~modifiedGate.topology_file_identity && ...
    any(contains(modifiedGate.failures, 'SHA-256')), nPass, nFail);

% A corrupt reference copy fails the pre-run lifecycle gate. The
% authoritative fixture is never modified.
badReference = fullfile(tmpDir, 'bad_reference.csv');
copyfile(baseline.path, badReference);
localAppendNewline(badReference);
badReferenceError = localCapture(@() a4_validate_frozen_baseline( ...
    badReference, expectedSha, producedDir, producedPath));
[nPass,nFail] = ck( ...
    'modified immutable-baseline copy fails before arm execution', ...
    strcmp(badReferenceError, 'a4:FrozenBaselineHashMismatch') && ...
    ~isfile(fullfile(producedDir, 'a4_checkpoint_inf.mat')), nPass, nFail);

missingError = localCapture(@() a4_validate_frozen_baseline( ...
    fullfile(tmpDir, 'missing.csv'), expectedSha, producedDir, producedPath));
[nPass,nFail] = ck('missing immutable baseline fails loud', ...
    strcmp(missingError, 'a4:FrozenBaselineMissing'), nPass, nFail);

aliasOutput = fullfile(tmpDir, 'alias_output');
mkdir(aliasOutput);
aliasError = localCapture(@() a4_validate_frozen_baseline( ...
    baseline.path, expectedSha, aliasOutput, baseline.path));
[nPass,nFail] = ck('reference and produced paths cannot alias', ...
    strcmp(aliasError, 'a4:FrozenBaselinePathAlias'), nPass, nFail);

insideOutput = fullfile(tmpDir, 'inside_output');
mkdir(insideOutput);
insideReference = fullfile(insideOutput, 'a4_topology_Ninf.csv');
copyfile(baseline.path, insideReference);
insideProduced = fullfile(insideOutput, 'a4_topology_inf.csv');
insideError = localCapture(@() a4_validate_frozen_baseline( ...
    insideReference, expectedSha, insideOutput, insideProduced));
[nPass,nFail] = ck('reference inside mutable output fails loud', ...
    strcmp(insideError, 'a4:FrozenBaselineInsideMutableOutput'), nPass, nFail);

results = struct('passed', nPass, 'failed', nFail);
fprintf('  passed: %d   failed: %d\n', nPass, nFail);
if nFail > 0
    error('test_a4_phase5a:Failed', ...
        '%d immutable-baseline lifecycle regression(s) failed.', nFail);
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

function localAppendNewline(path)
fid = fopen(path, 'a');
if fid < 0
    error('test_a4_phase5a:WriteFailed', 'Cannot modify temporary file: %s', path);
end
cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '\n');
end

function identifier = localCapture(action)
identifier = '';
try
    action();
catch ME
    identifier = ME.identifier;
end
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
