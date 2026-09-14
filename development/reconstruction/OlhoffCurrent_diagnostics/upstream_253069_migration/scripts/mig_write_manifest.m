function mig_write_manifest(expectedTree)
%MIG_WRITE_MANIFEST  The deliberate SOURCE_MANIFEST.json rewrite of this promotion.
%   Writes the manifest with the production function (olhoffcurrent_source_manifest
%   'Write') and refuses to leave it in place unless the tree hash equals the one
%   computed independently (Python, before MATLAB saw the tree) and every file is
%   byte-identical to the 253069 snapshot.
P = mig_paths();
restoredefaultpath; addpath(P.scripts); addpath(P.oc);
man = olhoffcurrent_source_manifest('Write', true);
assert(strcmp(man.treeHash, expectedTree), 'mig:manifest', ...
    'MATLAB tree hash %s differs from the independent computation %s', man.treeHash, expectedTree);
assert(man.nFiles == 79, 'mig:manifest', 'expected 79 files, got %d', man.nFiles);
for k = 1:numel(man.files)
    up = fullfile(P.up, man.files{k});
    assert(exist(up, 'file') == 2, 'mig:manifest', '%s is not in the 253069 snapshot', man.files{k});
    assert(strcmp(olhoffcurrent_sha256_file(up), man.hashes{k}), 'mig:manifest', ...
        '%s differs from the 253069 snapshot', man.files{k});
end
v = olhoffcurrent_source_manifest('Verify', true);
assert(v.ok, 'mig:manifest', 'the written manifest does not verify');
fprintf('MANIFEST written: %d files, tree %s, all byte-identical to 253069, verify ok=%d\n', ...
    man.nFiles, man.treeHash, v.ok);
end
