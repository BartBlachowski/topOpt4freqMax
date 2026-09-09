function nFail = test_evidence_retention()
%TEST_EVIDENCE_RETENTION  The evidence gate must fail exactly when evidence is lost.
%
%   Regression cover for the defect that destroyed three studies' raw density
%   trajectories: files written under a git-ignored path, never declared, never
%   existence-checked, while the studies reported themselves frozen and complete.
%
%   TEST R1  all required evidence present and matching        -> PASS
%   TEST R2  a required trajectory file is missing             -> FAIL
%   TEST R3  a required trajectory file is modified            -> FAIL
%   TEST R4  optional and scratch artifacts missing            -> PASS
%   TEST R5  git-IGNORED required evidence present + hashes    -> PASS
%   TEST R6  git-IGNORED required evidence disappears          -> FAIL
%   TEST R7  study declares no EVIDENCE.json at all            -> FAIL
%   TEST R8  declaring a required artifact that does not exist -> REFUSED
%   TEST R9  unknown artifact class is treated as required     -> FAIL
%   TEST R10 source-integrity mechanism still intact           -> CURRENT
%
%   R5/R6 are the ones that matter most: they assert the gate works on exactly
%   the storage arrangement that failed before -- a large file that git does not
%   and should not track.  The test verifies with `git check-ignore` that the
%   path really is ignored, so it cannot silently degrade into a tracked-file
%   test if the ignore rules change.

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
repo = fileparts(fileparts(root));
addpath(root);

nFail = 0;
fprintf('\n%s\nTEST_EVIDENCE_RETENTION\n%s\n', repmat('=',1,72), repmat('=',1,72));

% The +impl tree hash BEFORE any of this runs.  R10b compares against it, so the
% test asks the question it is for -- "did the retention mechanism disturb
% production source?" -- rather than pinning one historical digest, which would
% have to be edited every time production legitimately changes and would then no
% longer be a check at all.
treeHash0 = olhoffcurrent_source_manifest('Verify', false).treeHash;

sandbox   = fullfile(root, 'evidence', '_gate_selftest');
studyDir  = fullfile(sandbox, 'study');
evDirRel  = 'analysis/OlhoffCurrent/evidence/_gate_selftest/data';
evDirAbs  = fullfile(root, 'evidence', '_gate_selftest', 'data');

cleanup = onCleanup(@() local_rmrf(sandbox));
local_rmrf(sandbox);
mkdir(studyDir); mkdir(evDirAbs);

% ---- the evidence: a stand-in element-level trajectory -------------------
traj = fullfile(evDirAbs, 'arm_test_trajectory.mat');
RHO  = rand(64, 20);                                  %#ok<NASGU>
save(traj, 'RHO', '-v7.3');
opt  = fullfile(evDirAbs, 'optional_extra.mat');
X = 1; save(opt, 'X', '-v7.3');                       %#ok<NASGU>
scratch = fullfile(evDirAbs, 'scratch_tmp.mat');
save(scratch, 'X', '-v7.3');

% R5/R6 depend on this path genuinely being git-ignored.  Assert it.
ignored = local_gitIgnored(repo, traj);
nFail = nFail + chk('R5a git actually ignores the evidence path', ignored, true);

decl = { 'arm_test_trajectory.mat', 'required', 'element-level trajectory' ; ...
         'optional_extra.mat',      'optional', 'nice to have' ; ...
         'scratch_tmp.mat',         'scratch',  'disposable' };
olhoffcurrent_evidence_declare(studyDir, '_gate_selftest', decl, ...
    'EvidenceRoot', evDirRel, 'RepoRoot', repo);

% ------------------------------------------------------------------ R1
st = olhoffcurrent_evidence_gate(studyDir, 'Verbose', false, 'RepoRoot', repo);
nFail = nFail + chk('R1  all required present            -> PASS', st.ok, true);
nFail = nFail + chk('R1b required counted as present', st.nPresent == 1 && st.nRequired == 1, true);

% ------------------------------------------------------------------ R5
% Same assertion, stated as the git-ignored case it actually is.
nFail = nFail + chk('R5  git-ignored required evidence OK -> PASS', st.ok && ignored, true);

% ------------------------------------------------------------------ R4
delete(opt); delete(scratch);
st = olhoffcurrent_evidence_gate(studyDir, 'Verbose', false, 'RepoRoot', repo);
nFail = nFail + chk('R4  optional+scratch missing         -> PASS', st.ok, true);
save(opt, 'X', '-v7.3'); save(scratch, 'X', '-v7.3');

% ------------------------------------------------------------------ R3
bytes = local_read(traj);
fid = fopen(traj, 'r+'); fseek(fid, 512, 'bof'); fwrite(fid, uint8(255)); fclose(fid);
st = olhoffcurrent_evidence_gate(studyDir, 'Verbose', false, 'RepoRoot', repo);
nFail = nFail + chk('R3  required trajectory MODIFIED     -> FAIL', st.ok, false);
nFail = nFail + chk('R3b status is REQUIRED_HASH_MISMATCH', ...
                    strcmp(st.artifacts(1).status, 'REQUIRED_HASH_MISMATCH'), true);
local_write(traj, bytes);

% ------------------------------------------------------------------ R2 / R6
delete(traj);
st = olhoffcurrent_evidence_gate(studyDir, 'Verbose', false, 'RepoRoot', repo);
nFail = nFail + chk('R2  required trajectory MISSING      -> FAIL', st.ok, false);
nFail = nFail + chk('R6  git-ignored evidence DISAPPEARS  -> FAIL', ...
                    ~st.ok && ignored, true);
nFail = nFail + chk('R2b status is REQUIRED_MISSING', ...
                    strcmp(st.artifacts(1).status, 'REQUIRED_MISSING'), true);

% ------------------------------------------------------------------ R8
% declare() must refuse to record a required artifact that is not on disk.
refused = false;
try
    olhoffcurrent_evidence_declare(studyDir, '_gate_selftest', ...
        {'arm_test_trajectory.mat','required',''}, 'EvidenceRoot', evDirRel, 'RepoRoot', repo);
catch ME
    refused = strcmp(ME.identifier, 'olhoffcurrent_evidence_declare:RequiredMissing');
end
nFail = nFail + chk('R8  declaring absent required file   -> REFUSED', refused, true);
local_write(traj, bytes);

% ------------------------------------------------------------------ R7
noDecl = fullfile(sandbox, 'undeclared');
mkdir(noDecl);
st = olhoffcurrent_evidence_gate(noDecl, 'Verbose', false, 'RepoRoot', repo);
nFail = nFail + chk('R7  no EVIDENCE.json at all          -> FAIL', st.ok, false);

% ------------------------------------------------------------------ R9
% An unrecognised class must fail closed, never be treated as harmless.
D = jsondecode(fileread(fullfile(studyDir,'EVIDENCE.json')));
A = D.artifacts; if ~iscell(A); A = num2cell(A); end
A{1}.class = 'archival';
D.artifacts = A;
fid = fopen(fullfile(studyDir,'EVIDENCE.json'),'w');
fwrite(fid, jsonencode(D,'PrettyPrint',true)); fclose(fid);
st = olhoffcurrent_evidence_gate(studyDir, 'Verbose', false, 'RepoRoot', repo);
nFail = nFail + chk('R9  unknown class fails closed       -> FAIL', st.ok, false);

% ------------------------------------------------------------------ R10
% This mechanism must not have disturbed source integrity.
cur = olhoffcurrent_currentness('Verbose', false);
man = olhoffcurrent_source_manifest('Verify', true);
nFail = nFail + chk('R10 source integrity still CURRENT', ...
                    man.ok && ~strcmp(cur.state,'LOCAL_MODIFIED'), true);
nFail = nFail + chk('R10b +impl tree hash unchanged by the evidence mechanism', ...
    strcmp(man.treeHash, treeHash0), true);

fprintf('%s\nTEST_EVIDENCE_RETENTION: %d failure(s)\n%s\n', ...
        repmat('-',1,72), nFail, repmat('=',1,72));
end

% ---------------------------------------------------------------- helpers
function n = chk(name, got, want)
ok = isequal(got, want);
fprintf('  [%s] %s\n', ternary(ok,'PASS','FAIL'), name);
n = double(~ok);
end

function s = ternary(c,a,b)
if c; s = a; else; s = b; end
end

function tf = local_gitIgnored(repo, p)
[status, ~] = system(sprintf('cd %s && git check-ignore -q %s', ...
                             local_q(repo), local_q(p)));
tf = (status == 0);
end

function s = local_q(p)
s = ['''' strrep(p,'''','''\''''') ''''];
end

function b = local_read(p)
fid = fopen(p,'r','n'); c = onCleanup(@() fclose(fid));
b = fread(fid, Inf, '*uint8');
end

function local_write(p, b)
fid = fopen(p,'w','n'); c = onCleanup(@() fclose(fid));
fwrite(fid, b, 'uint8');
end

function local_rmrf(d)
if exist(d,'dir') == 7
    rmdir(d, 's');
end
end
