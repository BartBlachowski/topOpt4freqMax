function mig_make_fixtures()
%MIG_MAKE_FIXTURES  Compact tracked reference digests for the new equivalence tests.
%
%   tests/fixtures/S160x20_reference.json
%       from the COMMITTED upstream result repro/results/S160x20/res.mat
%       (blob 93c65509... in both 6b08708 and 253069), read from the snapshot
%   tests/fixtures/EX3_160_reference.json
%       from the PRE-MIGRATION OlhoffCurrent run PRE_EX3 (tree edbfe47...), after
%       proving it bitwise equal to the independent upstream-audit record
%       case_TARGET_EX3_160.mat produced from the same tree on 2026-09-13
%   tests/fixtures/schema_rows_pre_253069.json
%       the 81 configuration schema rows of the pre-migration tree, for
%       recomputing historical configuration hashes
%
%   Digests: olhoffcurrent_test_digest (timing excluded).
P = mig_paths();
restoredefaultpath; addpath(P.scripts); addpath(fullfile(P.oc, 'tests'));
fx = fullfile(P.oc, 'tests', 'fixtures');
if ~isfolder(fx), mkdir(fx); end

% ---- S160x20 committed upstream result -----------------------------------
C = load(P.ref.S160, 'res');
dS = olhoffcurrent_test_digest(C.res);
dS.source = struct( ...
    'artifact', 'Olhoff repro/results/S160x20/res.mat', ...
    'git_blob', '93c6550980f06fe3535cc1780bc0317517b6a0c0', ...
    'commits', {{'6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7', '253069262407885a8b759a9e721c4f0a7d3a397d'}}, ...
    'file_sha256', olhoffcurrent_sha256_file_local(P.ref.S160), ...
    'upstream_preset', 'duOlhoffAdaptivePedersen', ...
    'olhoffcurrent_preset', P.name.ped, ...
    'note', ['committed with runtime.verbose = true and runtime.name = S160x20; neither is a ' ...
             'scientific field. hist.tOuter did not exist at 6b08708 and is excluded anyway.']);
local_write(fullfile(fx, 'S160x20_reference.json'), dS);

% ---- EX3_160 pre-migration target run --------------------------------------
A = load(fullfile(P.ev, 'PRE_EX3.mat'));
B = load(P.ref.targetEX3, 'r');
dA = olhoffcurrent_test_digest(A.res);
rb = B.r; rb.lambda = B.r.lambda;
dB = olhoffcurrent_test_digest(rb);
same = isequal(rmfield(dA, {'logLines'}), rmfield(dB, {'logLines'}));
assert(same, 'mig:fixtures', 'PRE_EX3 is not bitwise equal to case_TARGET_EX3_160');
dA.source = struct( ...
    'artifact', 'analysis/OlhoffCurrent/evidence/upstream_253069_migration/PRE_EX3.mat', ...
    'artifact_sha256', olhoffcurrent_sha256_file_local(fullfile(P.ev, 'PRE_EX3.mat')), ...
    'tree', A.meta.treeHash, ...
    'crosscheck', ['bitwise equal (all digests) to Olhoff-upstream-capabilities-evidence/runs/' ...
                   'case_TARGET_EX3_160.mat, produced from the same OlhoffCurrent tree by the ' ...
                   'upstream capability audit'], ...
    'crosscheck_sha256', olhoffcurrent_sha256_file_local(P.ref.targetEX3), ...
    'olhoffcurrent_preset', P.name.ex3, ...
    'runtime', 'cap 1600, diagnostics on, single thread');
local_write(fullfile(fx, 'EX3_160_reference.json'), dA);

% ---- pre-migration schema rows ---------------------------------------------
Q = load(fullfile(P.ev, 'cfg_pre.mat'), 'rows');
rows = struct('source', 'olh.config.schema() at OlhoffCurrent tree edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb', ...
    'n', numel(Q.rows), 'rows', {Q.rows(:).'});
local_write(fullfile(fx, 'schema_rows_pre_253069.json'), rows);
fprintf('fixtures written: S160x20 (nOuter %d), EX3_160 (nOuter %d), %d pre-migration schema rows\n', ...
    dS.nOuter, dA.nOuter, numel(Q.rows));
end

function local_write(p, s)
fid = fopen(p, 'w'); fwrite(fid, jsonencode(s, 'PrettyPrint', true)); fprintf(fid, '\n'); fclose(fid);
end

function h = olhoffcurrent_sha256_file_local(p)
fid = fopen(p, 'r'); b = fread(fid, Inf, '*uint8'); fclose(fid);
md = java.security.MessageDigest.getInstance('SHA-256'); md.update(b);
x = typecast(md.digest(), 'uint8'); h = lower(reshape(dec2hex(x, 2).', 1, []));
end
