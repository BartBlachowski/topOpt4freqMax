function test_a4_phase1()
%TEST_A4_PHASE1  Recovery Phase 1 regressions for C-3 and C-4.
%
%   Postprocessing/provenance only. No optimization is executed.

fprintf('\n=== test_a4_phase1 (C-3 endpoint floor; C-4 hashing) ===\n');
nPass = 0; nFail = 0;

tmp = tempname;
mkdir(tmp);
cleanup = onCleanup(@() rmdir(tmp, 's'));

emptyPath = fullfile(tmp, 'empty.bin');
aPath = fullfile(tmp, 'a.bin');
foobarPath = fullfile(tmp, 'foobar.bin');
foobarCopyPath = fullfile(tmp, 'foobar_copy.bin');
localWriteBytes(emptyPath, uint8([]));
localWriteBytes(aPath, uint8('a'));
localWriteBytes(foobarPath, uint8('foobar'));
localWriteBytes(foobarCopyPath, uint8('foobar'));

[nPass,nFail] = ck('C-4 empty reference = 811c9dc5', ...
    strcmp(a4_hash_file(emptyPath), 'fnv1a32_811c9dc5'), nPass,nFail);
[nPass,nFail] = ck('C-4 known vector "a" = e40c292c', ...
    strcmp(a4_hash_file(aPath), 'fnv1a32_e40c292c'), nPass,nFail);
[nPass,nFail] = ck('C-4 known vector "foobar" = bf9cf968', ...
    strcmp(a4_hash_file(foobarPath), 'fnv1a32_bf9cf968'), nPass,nFail);
[nPass,nFail] = ck('C-4 identical bytes reproduce identical hashes', ...
    strcmp(a4_hash_file(foobarPath), a4_hash_file(foobarCopyPath)), nPass,nFail);
[nPass,nFail] = ck('C-4 different files produce different hashes', ...
    ~strcmp(a4_hash_file(aPath), a4_hash_file(foobarPath)), nPass,nFail);

s1 = struct('z', 2, 'a', struct('y', 4, 'b', 3));
s2 = struct('a', struct('b', 3, 'y', 4), 'z', 2);
s3 = s2; s3.z = 5;
[nPass,nFail] = ck('C-4 canonical struct hash ignores field insertion order', ...
    strcmp(fnv1a32_canonical_struct(s1), fnv1a32_canonical_struct(s2)), nPass,nFail);
[nPass,nFail] = ck('C-4 canonical struct hash changes with content', ...
    ~strcmp(fnv1a32_canonical_struct(s2), fnv1a32_canonical_struct(s3)), nPass,nFail);

x = [0.9; 0.8; 0.2; 0.1];
rhoMin = 1e-9;
xt = a4_volume_preserving_threshold(x, 0.5, rhoMin);
[nPass,nFail] = ck('C-3 threshold preserves requested solid count', ...
    isequal(xt(1:2), ones(2,1)), nPass,nFail);
[nPass,nFail] = ck('C-3 threshold uses configured rho_min exactly', ...
    isequal(xt(3:4), rhoMin*ones(2,1)), nPass,nFail);
[nPass,nFail] = ck('C-3 undeclared 1e-3 floor is absent', ...
    ~any(xt == 1e-3), nPass,nFail);

fprintf('\n  passed: %d   failed: %d\n', nPass, nFail);
if nFail > 0
    error('test_a4_phase1:Failed', '%d Recovery Phase 1 test(s) failed.', nFail);
end
fprintf('  ALL RECOVERY PHASE 1 TESTS PASSED.\n\n');
end

function localWriteBytes(path, bytes)
fid = fopen(path, 'wb');
if fid < 0, error('test_a4_phase1:WriteFailed', 'Cannot write %s', path); end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, bytes, 'uint8');
end

function [nPass,nFail] = ck(label, ok, nPass,nFail)
if ok
    nPass = nPass + 1;
    fprintf('  [PASS] %s\n', label);
else
    nFail = nFail + 1;
    fprintf('  [FAIL] %s\n', label);
end
end
