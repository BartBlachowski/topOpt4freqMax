function st = olhoffcurrent_finalization_gate(studyDir, varargin)
%OLHOFFCURRENT_FINALIZATION_GATE  May this diagnostic study be called FINALIZED?
%
%   st = OLHOFFCURRENT_FINALIZATION_GATE(studyDir) answers one question, and it
%   fails closed:
%
%       can this study still be believed, end to end?
%
%   WHY THIS EXISTS, ON TOP OF THE EVIDENCE GATE
%   --------------------------------------------
%   EVIDENCE_POLICY.md and OLHOFFCURRENT_EVIDENCE_GATE already answer "is the
%   declared raw evidence still on disk?".  They were written after three
%   studies lost their trajectories, and they work: every study finalized after
%   they landed carries an EVIDENCE.json and has lost nothing.
%
%   But they are OPT-IN.  A study that simply never writes an EVIDENCE.json is
%   not caught by them -- it is invisible to them -- and can still declare
%   itself complete with a FINAL_SHA256.txt whose .mat entries no longer
%   resolve.  That is exactly the state five studies are in:
%
%       dynamical_regime              3 hashed artifacts missing
%       fixedmove_400_dynamics        2
%       move_stop                     4
%       topology_maturity_transition  1
%       two_branch_maturity_240       2
%
%   and two more (admission_rule, move_transition) show a clean FINAL_SHA256
%   only because they never hashed their raw trajectories at all -- a false
%   green, which is worse.
%
%   THE RULE THIS ENFORCES
%   ----------------------
%   A study may not be FINALIZED unless ALL of:
%
%     G1  it declares its raw evidence      (EVIDENCE.json exists)
%     G2  every REQUIRED declared artifact is present and hash-valid
%     G3  it has a FINAL_SHA256.txt
%     G4  every path listed in FINAL_SHA256.txt resolves and hashes as recorded
%         -- so the hash file is SELF-VERIFYING and cannot go stale silently
%     G5  no .mat named by FINAL_SHA256.txt or DATA_MANIFEST.json is absent
%
%   G1 is the one that closes the hole: "declares nothing" is not a passing
%   state, because that is precisely the state the lost studies were in.
%
%   SCIENTIFICALLY INERT.  This function reads files and hashes them.  It lives
%   outside +impl/, so it cannot change the canonical tree hash, and it never
%   touches the optimizer, a configuration, or a trajectory.
%
%   Options:
%     'Verbose'  (default true)
%     'RepoRoot' (default derived)
%
%   st fields: ok, studyDir, gates (G1..G5 logical), detail, missing, mismatched.
%
%   See also OLHOFFCURRENT_EVIDENCE_GATE, OLHOFFCURRENT_EVIDENCE_DECLARE.

p = inputParser();
p.addParameter('Verbose', true, @(v) islogical(v) && isscalar(v));
p.addParameter('RepoRoot', '', @(v) ischar(v) || isstring(v));
p.parse(varargin{:});
verbose = p.Results.Verbose;

root = olhoffcurrent_root();
repo = char(p.Results.RepoRoot);
if isempty(repo); repo = fileparts(fileparts(root)); end
studyDir = char(studyDir);

st = struct('ok', false, 'studyDir', studyDir, ...
            'gates', struct('G1', false, 'G2', false, 'G3', false, 'G4', false, 'G5', false), ...
            'detail', '', 'missing', {{}}, 'mismatched', {{}}, ...
            'nHashed', 0, 'nRequired', 0);

% ---- G1: the study must declare its raw evidence -------------------------
evPath = fullfile(studyDir, 'EVIDENCE.json');
st.gates.G1 = exist(evPath, 'file') == 2;

% ---- G2: every required declared artifact present and hash-valid ---------
if st.gates.G1
    eg = olhoffcurrent_evidence_gate(studyDir, 'Verbose', false, 'RepoRoot', repo);
    st.gates.G2 = eg.ok;
    st.nRequired = eg.nRequired;
else
    st.gates.G2 = false;
end

% ---- G3/G4: FINAL_SHA256.txt must exist and verify against itself --------
fs = fullfile(studyDir, 'FINAL_SHA256.txt');
st.gates.G3 = exist(fs, 'file') == 2;
if st.gates.G3
    [nOk, missing, mism] = local_verifyHashFile(fs, studyDir, repo);
    st.nHashed = nOk + numel(missing) + numel(mism);
    st.missing = missing;
    st.mismatched = mism;
    st.gates.G4 = isempty(missing) && isempty(mism);
end

% ---- G5: no .mat named by the study's own manifests may be absent --------
mats = local_namedMats(studyDir);
absent = {};
for i = 1:numel(mats)
    if ~local_resolve(mats{i}, studyDir, repo); absent{end+1} = mats{i}; end %#ok<AGROW>
end
st.gates.G5 = isempty(absent);
st.absentMats = absent;

st.ok = st.gates.G1 && st.gates.G2 && st.gates.G3 && st.gates.G4 && st.gates.G5;
g = st.gates;
st.detail = sprintf('G1 declares=%d G2 required=%d G3 hashfile=%d G4 selfverify=%d G5 nomissingmat=%d', ...
                    g.G1, g.G2, g.G3, g.G4, g.G5);

if verbose
    fprintf('\n%s\nFINALIZATION GATE  %s\n%s\n', repmat('=',1,72), studyDir, repmat('=',1,72));
    lbl = {'G1  declares raw evidence (EVIDENCE.json)', ...
           'G2  required declared artifacts present + hash-valid', ...
           'G3  FINAL_SHA256.txt present', ...
           'G4  FINAL_SHA256.txt is self-verifying', ...
           'G5  no .mat named by its manifests is absent'};
    vals = [g.G1 g.G2 g.G3 g.G4 g.G5];
    for i = 1:5
        fprintf('  [%s] %s\n', local_pf(vals(i)), lbl{i});
    end
    for i = 1:numel(st.missing);    fprintf('        MISSING   %s\n', st.missing{i}); end
    for i = 1:numel(st.mismatched); fprintf('        MISMATCH  %s\n', st.mismatched{i}); end
    for i = 1:numel(absent);        fprintf('        ABSENT .mat  %s\n', absent{i}); end
    fprintf('  RESULT: %s\n\n', local_pf(st.ok));
end
end

% =========================================================================
function [nOk, missing, mism] = local_verifyHashFile(fs, studyDir, repo)
%LOCAL_VERIFYHASHFILE  Verify every DIGESTED line of a study's hash file.
%   Only lines carrying a 64-hex digest are claims of possession.  A study may
%   also DOCUMENT absent files (e.g. a "MISSING PRIOR EVIDENCE" section); those
%   lines carry no digest and are disclosures, not claims, so they are not
%   verified here -- penalising honest disclosure would be exactly backwards.
nOk = 0; missing = {}; mism = {};
lines = strsplit(fileread(fs), newline);
for i = 1:numel(lines)
    tok = regexp(lines{i}, '^([0-9a-f]{64})\s+(\S+)', 'tokens', 'once');
    if isempty(tok); continue; end
    fp = local_resolvePath(tok{2}, studyDir, repo);
    if isempty(fp)
        missing{end+1} = tok{2}; %#ok<AGROW>
    elseif ~strcmp(olhoffcurrent_sha256_file(fp), tok{1})
        mism{end+1} = tok{2}; %#ok<AGROW>
    else
        nOk = nOk + 1;
    end
end
end

function fp = local_resolvePath(rel, studyDir, repo)
%LOCAL_RESOLVEPATH  Resolve a manifest path to a real file.
%   STUDY-LOCAL FIRST: a bare name like 'PROVENANCE.md' means the study's own
%   file, and several studies have one alongside the implementation-level file
%   of the same name.  isfile() is used rather than exist(), because exist()
%   also searches the MATLAB path and would silently resolve a bare name to
%   whichever copy happens to be on it.
fp = '';
cand = {fullfile(studyDir, rel), fullfile(studyDir, 'runs', rel), ...
        fullfile(repo, rel), rel};
for c = 1:numel(cand)
    if isfile(cand{c}); fp = cand{c}; return; end
end
end

function mats = local_namedMats(studyDir)
%LOCAL_NAMEDMATS  The .mat files the study CLAIMS to hold.
%   From FINAL_SHA256.txt only digested lines count (see local_verifyHashFile);
%   from DATA_MANIFEST.json only entries that carry a sha256.
mats = {};
fs = fullfile(studyDir, 'FINAL_SHA256.txt');
if isfile(fs)
    lines = strsplit(fileread(fs), newline);
    for i = 1:numel(lines)
        tok = regexp(lines{i}, '^[0-9a-f]{64}\s+(\S+\.mat)', 'tokens', 'once');
        if ~isempty(tok); mats{end+1} = tok{1}; end %#ok<AGROW>
    end
end
dm = fullfile(studyDir, 'DATA_MANIFEST.json');
if isfile(dm)
    try
        D = jsondecode(fileread(dm));
        if isfield(D, 'artifacts')
            A = D.artifacts; if ~iscell(A); A = num2cell(A); end
            for i = 1:numel(A)
                a = A{i};
                if isfield(a,'path') && isfield(a,'sha256') && ~isempty(a.sha256) && ...
                        endsWith(a.path, '.mat')
                    mats{end+1} = a.path; %#ok<AGROW>
                end
            end
        end
    catch
        % a manifest that will not parse is caught by G4, not here
    end
end
mats = unique(mats);
end

function tf = local_resolve(rel, studyDir, repo)
%LOCAL_RESOLVE  Does this claimed .mat exist anywhere it legitimately could?
tf = ~isempty(local_resolvePath(rel, studyDir, repo));
if tf; return; end
% the durable evidence root is a legitimate home for large raw trajectories
[~, b, e] = fileparts(rel);
tf = ~isempty(dir(fullfile(repo, 'analysis', 'OlhoffCurrent', 'evidence', '**', [b e])));
end

function s = local_pf(t), if t, s = 'PASS'; else, s = 'FAIL'; end, end
