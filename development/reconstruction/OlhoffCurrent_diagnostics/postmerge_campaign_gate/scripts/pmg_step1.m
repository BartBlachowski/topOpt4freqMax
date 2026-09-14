function S = pmg_step1(outDir)
%PMG_STEP1  Step 1 re-review evidence for the repaired Amendment A2, from COMMITTED code.
%   A. the preregistered probe suite, run from a fresh clone of the migration branch
%      tip with THAT clone's gate and harness (not any working copy);
%   B. R1 (two_branch_controller_validation) in that clone;
%   C. the ORIGINAL attempt-1 attack scripts (attempt1_blocked/scripts/gate_probe*.m,
%      copied with their output path redirected) against a second fresh clone.
%   Throwaway clones only; nothing in either checkout is written except outDir.
tip = 'b21483b158f58e05e7b56957f2fbe8e1d2891395';
W = '/Users/piotrek/Programming/topOpt4freqMax-migration-253069';
orig = '/private/tmp/claude-501/-Users-piotrek-Programming-topOpt4freqMax/84bbf569-dc04-4771-be4d-d9d2da2b8566/scratchpad/step1_orig';
setenv('GIT_PAGER', 'cat'); setenv('PMG_OUT', outDir);
S = struct('tip', tip, 'when', char(datetime('now')), 'matlab', version);

C = [tempname() '_step1A'];
c1 = onCleanup(@() local_rmrf(C));
local_sh(sprintf('git --no-pager clone -q --shared --no-checkout "%s" "%s" && git --no-pager -C "%s" -c advice.detachedHead=false checkout -q --detach %s', W, C, C, tip));
oc = fullfile(C, 'analysis', 'OlhoffCurrent');
restoredefaultpath; addpath(fileparts(mfilename('fullpath'))); addpath(oc); addpath(fullfile(oc, 'tests'));
assert(strcmp(which('olhoffcurrent_finalization_gate'), fullfile(oc, 'olhoffcurrent_finalization_gate.m')));
assert(strcmp(which('gate_provenance_probes'), fullfile(oc, 'tests', 'gate_provenance_probes.m')));
[~, h] = system(sprintf('git --no-pager -C "%s" rev-parse HEAD', C)); S.cloneHead = strtrim(h);
[T, meta] = gate_provenance_probes(C);
S.A = struct('probes', T, 'meta', meta, 'nAsExpected', sum([T.pass]), 'n', numel(T));
fprintf('STEP1-A committed probes: %d/%d as expected\n', sum([T.pass]), numel(T));

st = olhoffcurrent_finalization_gate(fullfile(oc, 'diagnostics', 'two_branch_controller_validation'), 'Verbose', true, 'RepoRoot', C);
hv = st.sourceLines(strcmp({st.sourceLines.verdict}, 'HISTORICAL_VERIFIED'));
S.B = struct('ok', st.ok, 'nHistorical', numel(hv), 'paths', {{hv.path}}, 'digests', {{hv.digest}}, ...
    'freeze', unique({hv.freezeCommit}), 'source', unique({hv.sourceCommit}), ...
    'declaredTree', st.historicalSource.declaredTree, 'freezeTree', st.historicalSource.freezeTree, ...
    'historical', st.historicalSource.status, 'current', st.currentSource.status, 'headTree', st.currentSource.treeHash);
fprintf('STEP1-B R1 ok=%d historical=%d %s %s\n', st.ok, numel(hv), st.historicalSource.status, st.currentSource.status);

C2 = [tempname() '_step1C'];
c2 = onCleanup(@() local_rmrf(C2));
local_sh(sprintf('git --no-pager clone -q --shared --no-checkout "%s" "%s" && git --no-pager -C "%s" -c advice.detachedHead=false checkout -q --detach %s', W, C2, C2, tip));
local_sh(sprintf('git --no-pager -C "%s" config user.email probe@invalid && git --no-pager -C "%s" config user.name probe && git --no-pager -C "%s" config core.hooksPath /dev/null', C2, C2, C2));
restoredefaultpath; addpath(orig);
evalc('gate_probe(C2)');
restoredefaultpath; addpath(orig);
local_sh(sprintf('git --no-pager -C "%s" -c advice.detachedHead=false checkout -q -f --detach %s', C2, tip));
evalc('gate_probe_p10(C2)');
S.C = struct('original', jsondecode(fileread(fullfile(outDir, 'orig_gate_probe_results.json'))), ...
             'p10', jsondecode(fileread(fullfile(outDir, 'orig_gate_probe_p10_result.json'))));
R = S.C.original.results;
for i = 1:numel(R)
    fprintf('STEP1-C original %-3s expect=%d got=%d  %s\n', R(i).id, R(i).expect, R(i).got, R(i).desc);
end
fprintf('STEP1-C original P10 got=%d\n', S.C.p10.got);
fid = fopen(fullfile(outDir, 'step1_amendment_evidence.json'), 'w'); fwrite(fid, jsonencode(S, 'PrettyPrint', true)); fclose(fid);
end

function local_sh(cmd)
[s, o] = system([cmd ' 2>&1']); if s ~= 0; error('pmg:sh', '%s\n%s', cmd, o); end
end
function local_rmrf(p)
if exist(p, 'dir') == 7; rmdir(p, 's'); end
end
