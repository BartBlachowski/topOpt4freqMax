function gate_probe(repo)
%GATE_PROBE  Adversarial probes of the SUPERSEDED_PRODUCTION_SOURCE rule in
%   olhoffcurrent_finalization_gate at migration commit 9b30ec4.
%   Runs entirely inside a throwaway shared clone in the session scratchpad.
if nargin < 1; repo = '/private/tmp/claude-501/-Users-piotrek-Programming-topOpt4freqMax/84bbf569-dc04-4771-be4d-d9d2da2b8566/scratchpad/gateprobe_repo'; end   % throwaway shared clone at 9b30ec4
root = fullfile(repo, 'analysis', 'OlhoffCurrent');
addpath(root);
fprintf('which olhoffcurrent_finalization_gate -all:\n'); which('olhoffcurrent_finalization_gate', '-all');
fprintf('root = %s\n', olhoffcurrent_root());
assert(strcmp(olhoffcurrent_root(), root), 'wrong root on path');

sandbox = fullfile(root, 'evidence', '_gate_probe');
if exist(sandbox, 'dir'); rmdir(sandbox, 's'); end
study = fullfile(sandbox, 'study'); evAbs = fullfile(sandbox, 'data');
mkdir(study); mkdir(evAbs);
RHO = rand(8, 4); %#ok<NASGU>
save(fullfile(evAbs, 'arm_probe_trajectory.mat'), 'RHO', '-v7.3');
olhoffcurrent_evidence_declare(study, '_gate_probe', ...
    {'arm_probe_trajectory.mat', 'required', 'probe trajectory'}, ...
    'EvidenceRoot', 'analysis/OlhoffCurrent/evidence/_gate_probe/data', 'RepoRoot', repo);
fs = fullfile(study, 'FINAL_SHA256.txt');
ev = fullfile(study, 'EVIDENCE.json');
base = sprintf('%s  EVIDENCE.json', olhoffcurrent_sha256_file(ev));
w(fs, base);

src    = 'analysis/OlhoffCurrent/+impl/architecture/olhoffSolve.m';
src2   = 'analysis/OlhoffCurrent/+impl/architecture/+olh/+move/limit.m';
hOld   = digestAt(repo, '013cc48', src);   % real pre-promotion content
hOld2  = digestAt(repo, '013cc48', src2);
cur    = olhoffcurrent_sha256_file(fullfile(repo, src));
fab    = repmat('c', 1, 64);
EMPTY  = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';
fprintf('hOld(src)=%s\nhOld(src2)=%s\ncur(src)=%s\n', hOld, hOld2, cur);
assert(~strcmp(hOld, cur));

R = struct('id', {}, 'desc', {}, 'expect', {}, 'got', {}, 'nSup', {}, 'supRecorded', {});
R = probe(R, 'P0', 'baseline compliant study', true, study, repo, fs, base);
R = probe(R, 'P1', 'J1 replica: real historical digest, same path', true, study, repo, fs, ...
    sprintf('%s\n%s  %s', base, hOld, src));
R = probe(R, 'P2', 'J2 replica: fabricated digest', false, study, repo, fs, ...
    sprintf('%s\n%s  %s', base, fab, src));
R = probe(R, 'P3', 'historical digest of a DIFFERENT source path (limit.m digest on olhoffSolve.m line)', false, study, repo, fs, ...
    sprintf('%s\n%s  %s', base, hOld2, src));
R = probe(R, 'P4', 'historical digest of a NON-source file (README.md @013cc48)', false, study, repo, fs, ...
    sprintf('%s\n%s  %s', base, digestAt(repo, '013cc48', 'analysis/OlhoffCurrent/README.md'), 'analysis/OlhoffCurrent/README.md'));
R = probe(R, 'P5', 'DUPLICATE path: fabricated line FOLLOWED by real historical line', false, study, repo, fs, ...
    sprintf('%s\n%s  %s\n%s  %s', base, fab, src, hOld, src));
R = probe(R, 'P6', 'DUPLICATE path: real historical line followed by fabricated line', false, study, repo, fs, ...
    sprintf('%s\n%s  %s\n%s  %s', base, hOld, src, fab, src));
R = probe(R, 'P7', 'empty-content digest, path never deleted in history', false, study, repo, fs, ...
    sprintf('%s\n%s  %s', base, EMPTY, src));

% P8: local edit of +impl (manifest verification fails) with real historical digest
bak = fileread(fullfile(repo, src));
w(fullfile(repo, src), [bak sprintf('\n%% probe edit\n')]);
R = probe(R, 'P8', 'real historical digest but +impl locally edited (manifest fails)', false, study, repo, fs, ...
    sprintf('%s\n%s  %s', base, hOld, src));
w(fullfile(repo, src), bak);
assert(strcmp(olhoffcurrent_sha256_file(fullfile(repo, src)), cur));

% P9: pipeline exit status -- make a commit that DELETES src, then re-add it
% byte-identically (manifest still verifies).  git show <deletionCommit>:src
% fails, shasum hashes empty stdin.
sh(repo, sprintf('git rm -q "%s" && git commit -q -m probe-delete', src));
w(fullfile(repo, src), bak);
sh(repo, sprintf('git add "%s" && git commit -q -m probe-readd', src));
assert(strcmp(olhoffcurrent_sha256_file(fullfile(repo, src)), cur));
R = probe(R, 'P9', 'empty-content digest, path deleted+re-added in history (fabricated: content never empty)', false, study, repo, fs, ...
    sprintf('%s\n%s  %s', base, EMPTY, src));
sh(repo, 'git reset -q --hard 9b30ec45b038fb36e7cf20d57679b71cfd099fb3');

fprintf('\n==== GATE PROBE RESULTS ====\n');
nBad = 0;
for i = 1:numel(R)
    okp = R(i).got == R(i).expect;
    nBad = nBad + ~okp;
    fprintf('%s  expect=%s got=%s  nSuperseded=%d  recorded=%s  [%s]  %s\n', R(i).id, pf(R(i).expect), pf(R(i).got), ...
        R(i).nSup, R(i).supRecorded, ternary(okp, 'FAIL-CLOSED OK', 'FAIL-OPEN'), R(i).desc);
end
fprintf('fail-open cases: %d\n', nBad);
S = struct('results', R, 'nFailOpen', nBad, 'commit', '9b30ec45b038fb36e7cf20d57679b71cfd099fb3');
fid = fopen(fullfile(fileparts(mfilename('fullpath')), '..', 'evidence', 'gate_probe_results.json'), 'w'); fwrite(fid, jsonencode(S, 'PrettyPrint', true)); fclose(fid);
rmdir(sandbox, 's');
end

function R = probe(R, id, desc, expect, study, repo, fs, txt)
w(fs, txt);
st = olhoffcurrent_finalization_gate(study, 'Verbose', false, 'RepoRoot', repo);
rec = '';
if ~isempty(st.supersededSource); rec = strjoin({st.supersededSource.recorded}, ','); end
R(end+1) = struct('id', id, 'desc', desc, 'expect', expect, 'got', st.ok, ...
    'nSup', numel(st.supersededSource), 'supRecorded', rec);
end

function h = digestAt(repo, c, p)
[s, o] = system(sprintf('git --no-pager -C "%s" cat-file blob "%s:%s" | shasum -a 256', repo, c, p));
assert(s == 0); h = strtok(strtrim(o));
[s2, ~] = system(sprintf('git --no-pager -C "%s" cat-file -e "%s:%s"', repo, c, p)); assert(s2 == 0);
end

function sh(repo, cmd)
[s, o] = system(sprintf('cd "%s" && %s', repo, cmd)); assert(s == 0, o);
end

function w(p, txt), fid = fopen(p, 'w'); fwrite(fid, txt); fclose(fid); end
function s = pf(t), if t, s = 'PASS'; else, s = 'FAIL'; end, end
function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
