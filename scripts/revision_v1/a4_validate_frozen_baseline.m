function baseline = a4_validate_frozen_baseline( ...
    referencePath, expectedSha256, mutableOutputDir, producedTopologyPath)
%A4_VALIDATE_FROZEN_BASELINE  Fail-loud immutable V-P2-2 baseline gate.
%
% This gate validates reference lifecycle and path separation only. It must
% run before any production arm. The reference may never live in the mutable
% output directory or alias the topology that the current run will produce.

referencePath = localChar(referencePath);
expectedSha256 = lower(localChar(expectedSha256));
mutableOutputDir = localChar(mutableOutputDir);
producedTopologyPath = localChar(producedTopologyPath);

if ~isfile(referencePath)
    error('a4:FrozenBaselineMissing', ...
        'Immutable V-P2-2 baseline is missing: %s', referencePath);
end

actualSha256 = a4_sha256_file(referencePath);
if ~strcmp(actualSha256, expectedSha256)
    error('a4:FrozenBaselineHashMismatch', ...
        ['Immutable V-P2-2 baseline SHA-256 mismatch at %s: ' ...
         'expected %s, got %s. No production arm was started.'], ...
        referencePath, expectedSha256, actualSha256);
end

referenceCanonical = localCanonical(referencePath);
outputCanonical = localCanonical(mutableOutputDir);
producedCanonical = localCanonical(producedTopologyPath);

if localInside(referenceCanonical, outputCanonical)
    error('a4:FrozenBaselineInsideMutableOutput', ...
        ['Immutable V-P2-2 baseline must not be inside mutable output. ' ...
         'Reference: %s; output: %s'], ...
        referenceCanonical, outputCanonical);
end

if strcmp(referenceCanonical, producedCanonical)
    error('a4:FrozenBaselinePathAlias', ...
        ['Immutable V-P2-2 baseline and produced topology paths must be ' ...
         'distinct. Both resolve to: %s'], referenceCanonical);
end

baseline = struct( ...
    'validated', true, ...
    'path', referenceCanonical, ...
    'expected_sha256', expectedSha256, ...
    'actual_sha256', actualSha256, ...
    'mutable_output_dir', outputCanonical, ...
    'produced_topology_path', producedCanonical);
end

function value = localChar(value)
if isstring(value), value = char(value); end
if ~ischar(value)
    error('a4:FrozenBaselineBadPath', ...
        'Frozen-baseline paths and hashes must be character vectors or strings.');
end
end

function path = localCanonical(path)
path = char(java.io.File(path).getCanonicalPath());
end

function tf = localInside(path, directory)
tf = strcmp(path, directory) || startsWith(path, [directory filesep]);
end
