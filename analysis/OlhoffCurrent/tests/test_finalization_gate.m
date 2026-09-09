function nFail = test_finalization_gate()
%TEST_FINALIZATION_GATE  The finalization gate must fail closed, and must not
%   quietly pass the state that lost five studies' trajectories.
%
%   The gate's whole purpose is that "declares nothing" is not a passing state.
%   So the destructive cases below are the point of the test, not decoration.
%
%   TEST A  a compliant study                              -> PASS
%   TEST B  no EVIDENCE.json at all                        -> FAIL  (the lost-study state)
%   TEST C  a required declared artifact removed           -> FAIL
%   TEST D  a required declared artifact altered           -> FAIL
%   TEST E  no FINAL_SHA256.txt                            -> FAIL
%   TEST F  FINAL_SHA256.txt gone stale (one digest wrong) -> FAIL
%   TEST G  FINAL_SHA256.txt names an absent .mat          -> FAIL
%   TEST H  the three compliant real studies               -> PASS
%   TEST I  the known-deficient legacy set has not GROWN   -> ledger check
%
%   Scientifically inert: nothing here touches the optimizer, a configuration or
%   a trajectory.  All destructive cases run in a sandbox under evidence/.

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
repo = fileparts(fileparts(root));
addpath(root);

nFail = 0;
fprintf('\n%s\nTEST_FINALIZATION_GATE\n%s\n', repmat('=',1,72), repmat('=',1,72));

sandbox = fullfile(root, 'evidence', '_finalization_selftest');
cleanup = onCleanup(@() local_rmrf(sandbox)); %#ok<NASGU>
local_rmrf(sandbox);

% ---- build a minimal COMPLIANT study in the sandbox ---------------------
study = fullfile(sandbox, 'study');
evAbs = fullfile(root, 'evidence', '_finalization_selftest', 'data');
mkdir(study); mkdir(evAbs);
traj = fullfile(evAbs, 'arm_test_trajectory.mat');
RHO = rand(32, 12); %#ok<NASGU>
save(traj, 'RHO', '-v7.3');
evRel = 'analysis/OlhoffCurrent/evidence/_finalization_selftest/data';
olhoffcurrent_evidence_declare(study, '_finalization_selftest', ...
    {'arm_test_trajectory.mat', 'required', 'sandbox trajectory'}, ...
    'EvidenceRoot', evRel, 'RepoRoot', repo);
local_writeHashFile(study, {fullfile(study,'EVIDENCE.json')}, repo);

nFail = nFail + chk('A  compliant study                        -> PASS', ...
    local_ok(study, repo), true);

% ---- B: remove the declaration (the exact state that lost the data) ----
ev = fullfile(study, 'EVIDENCE.json');
bak = fileread(ev); delete(ev);
nFail = nFail + chk('B  no EVIDENCE.json (the lost-study state) -> FAIL', ...
    local_ok(study, repo), false);
local_write(ev, bak);

% ---- C: required artifact removed --------------------------------------
tbak = fullfile(sandbox, 'traj.bak'); copyfile(traj, tbak); delete(traj);
nFail = nFail + chk('C  required artifact removed               -> FAIL', ...
    local_ok(study, repo), false);
copyfile(tbak, traj);

% ---- D: required artifact altered --------------------------------------
fid = fopen(traj, 'a'); fwrite(fid, uint8(0)); fclose(fid);
nFail = nFail + chk('D  required artifact altered               -> FAIL', ...
    local_ok(study, repo), false);
copyfile(tbak, traj);
nFail = nFail + chk('D2 restored                                -> PASS', ...
    local_ok(study, repo), true);

% ---- E: no hash file ----------------------------------------------------
fs = fullfile(study, 'FINAL_SHA256.txt');
hbak = fileread(fs); delete(fs);
nFail = nFail + chk('E  no FINAL_SHA256.txt                     -> FAIL', ...
    local_ok(study, repo), false);
local_write(fs, hbak);

% ---- F: hash file gone stale -------------------------------------------
stale = regexprep(hbak, '^[0-9a-f]{64}', repmat('0',1,64), 'lineanchors', 'once');
local_write(fs, stale);
nFail = nFail + chk('F  FINAL_SHA256.txt stale (one wrong)      -> FAIL', ...
    local_ok(study, repo), false);
local_write(fs, hbak);

% ---- G: hash file names an absent .mat ---------------------------------
local_write(fs, [hbak sprintf('\n%s  runs/never_existed_trajectory.mat\n', repmat('a',1,64))]);
nFail = nFail + chk('G  names an absent .mat                    -> FAIL', ...
    local_ok(study, repo), false);
local_write(fs, hbak);

% ---- H: the real compliant studies -------------------------------------
DIAG = fullfile(root, 'diagnostics');
for s = {'move_activity_400', 'beta_transition_mechanism', 'two_branch_controller_validation'}
    nFail = nFail + chk(sprintf('H  real compliant study %-28s -> PASS', s{1}), ...
        local_ok(fullfile(DIAG, s{1}), repo), true);
end

% ---- I: the known-deficient legacy set must not GROW -------------------
% History is NOT rewritten.  These five studies predate EVIDENCE_POLICY.md and
% lost raw trajectories; they are recorded here so a REGRESSION is caught while
% the past stays as it was.
LEGACY = {'dynamical_regime','fixedmove_400_dynamics','move_stop', ...
          'topology_maturity_transition','two_branch_maturity_240', ...
          'admission_rule','move_transition','move_activity_offline'};
d = dir(DIAG); d = d([d.isdir] & ~ismember({d.name}, {'.','..'}));
failing = {};
for i = 1:numel(d)
    if ~local_ok(fullfile(DIAG, d(i).name), repo); failing{end+1} = d(i).name; end %#ok<AGROW>
end
newFail = setdiff(failing, LEGACY);
nFail = nFail + chk(sprintf('I  no NEW study fails the gate (new: %s)', ...
    local_join(newFail)), isempty(newFail), true);
fprintf('     legacy deficient set (%d, unchanged history): %s\n', ...
    numel(intersect(failing, LEGACY)), local_join(intersect(failing, LEGACY)));

fprintf('%s\nTEST_FINALIZATION_GATE: %d failure(s)\n%s\n', ...
        repmat('-',1,72), nFail, repmat('=',1,72));
end

% ---------------------------------------------------------------- helpers
function ok = local_ok(studyDir, repo)
st = olhoffcurrent_finalization_gate(studyDir, 'Verbose', false, 'RepoRoot', repo);
ok = st.ok;
end

function n = chk(name, got, want)
ok = isequal(logical(got), logical(want));
fprintf('  [%s] %s\n', local_pf(ok), name);
n = double(~ok);
end

function local_write(p, txt)
fid = fopen(p, 'w'); fwrite(fid, txt); fclose(fid);
end

function local_writeHashFile(study, extra, repo) %#ok<INUSD>
files = [{fullfile(study,'EVIDENCE.json')}, extra];
lines = {};
for i = 1:numel(files)
    if exist(files{i},'file') ~= 2; continue; end
    [~, b, e] = fileparts(files{i});
    lines{end+1} = sprintf('%s  %s', olhoffcurrent_sha256_file(files{i}), [b e]); %#ok<AGROW>
end
local_write(fullfile(study,'FINAL_SHA256.txt'), strjoin(unique(lines), newline));
end

function s = local_join(c)
if isempty(c); s = '(none)'; else; s = strjoin(c, ', '); end
end

function s = local_pf(t), if t, s = 'PASS'; else, s = 'FAIL'; end, end

function local_rmrf(p)
if exist(p, 'dir') == 7; rmdir(p, 's'); end
end
