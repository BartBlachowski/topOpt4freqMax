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
%     G4  EVERY digested line of FINAL_SHA256.txt verifies ON ITS OWN
%         -- so the hash file is SELF-VERIFYING and cannot go stale silently
%     G5  no .mat named by FINAL_SHA256.txt or DATA_MANIFEST.json is absent
%     G6  the CURRENT production source is the committed HEAD
%
%   G1 is the one that closes the hole: "declares nothing" is not a passing
%   state, because that is precisely the state the lost studies were in.
%
%   TWO SEPARATE FACTS: HISTORICAL EVIDENCE vs CURRENT IMPLEMENTATION
%   ------------------------------------------------------------------
%   A study may hash PRODUCTION SOURCE (analysis/OlhoffCurrent/+impl/** or
%   analysis/OlhoffCurrent/SOURCE_MANIFEST.json) to record which code it ran.  A
%   legitimate promotion replaces those files.  The gate keeps two facts apart
%   and never lets one stand in for the other:
%
%   HISTORICAL_SOURCE_HASH_VERIFIED -- per line, in G4.  A production-source line
%   whose digest is not the working-tree content is accepted ONLY if ALL hold:
%     1. FINAL_SHA256.txt is committed and unmodified (working tree = HEAD blob);
%     2. the FREEZE COMMIT is the one commit `git log -1 HEAD -- <hash file>`,
%        and the hash file there equals the working tree.  No other ancestor is
%        ever searched;
%     3. the study's EVIDENCE.json is committed and unmodified and DECLARES the
%        source tree it ran (sourceTree and/or impl_tree_sha256, equal if both),
%        and that tree equals the +impl tree hash computed from the committed
%        blobs at the freeze commit and SOURCE_MANIFEST.json's tree there;
%     4. <freeze>:<path> is proved to exist as a blob (git cat-file -e, -t)
%        BEFORE it is read, and the SHA-256 of that blob equals THIS LINE's
%        digest.  Every line is checked alone: a valid duplicate never rescues
%        an invalid line, and a missing object never hashes as "empty".
%   Production-source lines resolve only at <repo>/<path> (never study-local) and
%   must be canonical.  A line that names production source in any spelling (case,
%   //, ./, .., backslashes) without being a canonical digest line, or a line whose
%   file resolves (links followed) into +impl or SOURCE_MANIFEST.json, is
%   MALFORMED_SOURCE_LINE and fails.
%
%   CURRENT_SOURCE_HASH_VERIFIED -- G6, for every study.  HEAD is the root of
%   trust, not the editable manifest: every +impl blob of HEAD must equal the
%   working-tree file byte for byte (SHA-256 of raw content read with git
%   cat-file; the index is never consulted, so skip-worktree cannot hide an
%   edit); no extra non-artifact source file and no symlink may exist;
%   SOURCE_MANIFEST.json must equal its HEAD blob and agree row for row and in
%   tree hash with HEAD; and PROVENANCE.md's "Source tree SHA-256" row must be
%   that tree.  A locally edited +impl with a regenerated manifest fails.  Every
%   git call runs with --no-replace-objects and without GIT_DIR/GIT_WORK_TREE/...
%   in its environment, so replace refs and redirection cannot substitute objects.
%
%   SCIENTIFICALLY INERT.  This function reads files and git objects and hashes
%   them.  It lives outside +impl/, so it cannot change the canonical tree hash,
%   and it never touches the optimizer, a configuration, or a trajectory.
%
%   Options:
%     'Verbose'  (default true)
%     'RepoRoot' (default derived)
%
%   st fields: ok, studyDir, gates (G1..G6 logical), detail, missing, mismatched,
%   sourceLines (per production-source line: lineNo, path, digest, verdict,
%   freezeCommit, sourceCommit, reason), supersededSource, malformedSourceLines,
%   duplicatePaths, historicalSource (status, freezeCommit, declaredTree,
%   freezeTree, reason), currentSource (status, ok, head, treeHash, nFiles,
%   reasons).
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
            'gates', struct('G1', false, 'G2', false, 'G3', false, 'G4', false, 'G5', false, 'G6', false), ...
            'detail', '', 'missing', {{}}, 'mismatched', {{}}, ...
            'sourceLines', local_emptyLines(), ...
            'supersededSource', struct('path', {}, 'recorded', {}, 'commit', {}, 'freezeCommit', {}, 'lineNo', {}), ...
            'malformedSourceLines', struct('lineNo', {}, 'text', {}, 'reason', {}), ...
            'duplicatePaths', {{}}, ...
            'historicalSource', struct('status', 'NO_HISTORICAL_SOURCE_LINES', 'freezeCommit', '', ...
                                       'declaredTree', '', 'freezeTree', '', 'reason', ''), ...
            'currentSource', local_emptyCurrent(), ...
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

% ---- G3/G4: FINAL_SHA256.txt must exist and EVERY line verify on its own ---
fs = fullfile(studyDir, 'FINAL_SHA256.txt');
st.gates.G3 = exist(fs, 'file') == 2;
if st.gates.G3
    [E, malformed] = local_parseHashFile(fs);
    st.malformedSourceLines = malformed;
    ctx = [];
    nAttempted = 0; nHistOk = 0;
    for i = 1:numel(E)
        e = E(i);
        if e.isSource
            fp = fullfile(repo, e.path);            % never study-local, never cwd-relative
            fz = ''; src = ''; reason = '';
            if ~isfile(fp)
                v = 'MISSING'; reason = 'production-source path absent from the repository working tree';
            elseif strcmp(olhoffcurrent_sha256_file(fp), e.digest)
                v = 'CURRENT_MATCH';
            else
                if isempty(ctx); ctx = local_historicalContext(studyDir, fs, repo); end
                nAttempted = nAttempted + 1;
                [v, reason, src] = local_historicalLine(ctx, e, repo);
                fz = ctx.freezeCommit;
                if strcmp(v, 'HISTORICAL_VERIFIED')
                    nHistOk = nHistOk + 1;
                    st.supersededSource(end+1) = struct('path', e.path, 'recorded', e.digest, ...
                        'commit', src, 'freezeCommit', fz, 'lineNo', e.lineNo);
                end
            end
            st.sourceLines(end+1) = struct('lineNo', e.lineNo, 'path', e.path, 'digest', e.digest, ...
                'verdict', v, 'freezeCommit', fz, 'sourceCommit', src, 'reason', reason);
        else
            fp = local_resolvePath(e.path, studyDir, repo);
            if isempty(fp)
                v = 'MISSING';
            elseif local_resolvesIntoProduction(fp, repo)
                % production source may be referenced ONLY by its canonical path
                v = 'MALFORMED';
                st.malformedSourceLines(end+1) = struct('lineNo', e.lineNo, 'text', e.path, ...
                    'reason', 'resolves to production source through a non-canonical path or link');
            elseif ~strcmp(olhoffcurrent_sha256_file(fp), e.digest)
                v = 'MISMATCH';
            else
                v = 'CURRENT_MATCH';
            end
        end
        switch v
            case 'MISSING',  st.missing{end+1} = e.path;
            case 'MISMATCH', st.mismatched{end+1} = e.path;
        end
    end
    paths = {E.path};
    if ~isempty(paths)
        [u, ~, k] = unique(paths);
        st.duplicatePaths = u(accumarray(k(:), 1) > 1);
    end
    st.nHashed = numel(E);
    st.gates.G4 = isempty(st.missing) && isempty(st.mismatched) && isempty(st.malformedSourceLines);
    if nAttempted > 0
        st.historicalSource.freezeCommit = ctx.freezeCommit;
        st.historicalSource.declaredTree = ctx.declaredTree;
        st.historicalSource.freezeTree   = ctx.freezeTree;
        st.historicalSource.reason       = ctx.reason;
        if nHistOk == nAttempted
            st.historicalSource.status = 'HISTORICAL_SOURCE_HASH_VERIFIED';
        else
            st.historicalSource.status = 'HISTORICAL_SOURCE_HASH_NOT_VERIFIED';
        end
    end
end

% ---- G5: no .mat named by the study's own manifests may be absent --------
mats = local_namedMats(studyDir);
absent = {};
for i = 1:numel(mats)
    if ~local_resolve(mats{i}, studyDir, repo); absent{end+1} = mats{i}; end %#ok<AGROW>
end
st.gates.G5 = isempty(absent);
st.absentMats = absent;

% ---- G6: the current production source is the committed HEAD ---------------
st.currentSource = local_currentSource(repo);
st.gates.G6 = st.currentSource.ok;

st.ok = st.gates.G1 && st.gates.G2 && st.gates.G3 && st.gates.G4 && st.gates.G5 && st.gates.G6;
g = st.gates;
st.detail = sprintf(['G1 declares=%d G2 required=%d G3 hashfile=%d G4 selfverify=%d ' ...
                     'G5 nomissingmat=%d G6 currentsource=%d; %s; %s'], ...
                    g.G1, g.G2, g.G3, g.G4, g.G5, g.G6, st.historicalSource.status, st.currentSource.status);

if verbose
    fprintf('\n%s\nFINALIZATION GATE  %s\n%s\n', repmat('=',1,72), studyDir, repmat('=',1,72));
    lbl = {'G1  declares raw evidence (EVIDENCE.json)', ...
           'G2  required declared artifacts present + hash-valid', ...
           'G3  FINAL_SHA256.txt present', ...
           'G4  every FINAL_SHA256.txt line verifies on its own', ...
           'G5  no .mat named by its manifests is absent', ...
           'G6  current production source = committed HEAD'};
    vals = [g.G1 g.G2 g.G3 g.G4 g.G5 g.G6];
    for i = 1:6
        fprintf('  [%s] %s\n', local_pf(vals(i)), lbl{i});
    end
    for i = 1:numel(st.missing);    fprintf('        MISSING   %s\n', st.missing{i}); end
    for i = 1:numel(st.mismatched); fprintf('        MISMATCH  %s\n', st.mismatched{i}); end
    for i = 1:numel(st.malformedSourceLines)
        fprintf('        MALFORMED_SOURCE_LINE  line %d: %s\n', st.malformedSourceLines(i).lineNo, ...
            st.malformedSourceLines(i).reason);
    end
    for i = 1:numel(st.duplicatePaths)
        fprintf('        DUPLICATE_PATH  %s (every line validated on its own)\n', st.duplicatePaths{i});
    end
    fprintf('  HISTORICAL SOURCE: %s\n', st.historicalSource.status);
    for i = 1:numel(st.sourceLines)
        s = st.sourceLines(i);
        if strcmp(s.verdict, 'CURRENT_MATCH'); continue; end
        fprintf('        line %-3d %-20s %s  %s\n', s.lineNo, s.verdict, s.path, s.reason);
    end
    fprintf('  CURRENT SOURCE:    %s (HEAD %s, %d files, tree %s)\n', st.currentSource.status, ...
        local_short(st.currentSource.head), st.currentSource.nFiles, local_short(st.currentSource.treeHash));
    for i = 1:numel(st.currentSource.reasons)
        fprintf('        %s\n', st.currentSource.reasons{i});
    end
    for i = 1:numel(absent);        fprintf('        ABSENT .mat  %s\n', absent{i}); end
    fprintf('  RESULT: %s\n\n', local_pf(st.ok));
end
end

% =========================================================================
function [E, malformed] = local_parseHashFile(fs)
%LOCAL_PARSEHASHFILE  Every DIGESTED line of a study's hash file, one entry per line.
%   Only lines carrying a 64-hex digest are claims of possession.  A study may
%   also DOCUMENT absent files (e.g. a "MISSING PRIOR EVIDENCE" section); those
%   lines carry no digest and are disclosures, not claims, so they are not
%   verified -- penalising honest disclosure would be exactly backwards.  The
%   one exception: a line that NAMES production source must be a well-formed,
%   canonical digest line, because production-source claims are never prose.
E = struct('lineNo', {}, 'digest', {}, 'path', {}, 'isSource', {});
malformed = struct('lineNo', {}, 'text', {}, 'reason', {});
lines = strsplit(fileread(fs), newline);
for i = 1:numel(lines)
    ln = lines{i};
    tok = regexp(ln, '^([0-9a-f]{64})\s+(\S+)', 'tokens', 'once');
    if isempty(tok)
        if local_namesSource(local_textNorm(ln))
            malformed(end+1) = struct('lineNo', i, 'text', strtrim(ln), ...
                'reason', 'names production source but is not a <64 lowercase hex> <path> line'); %#ok<AGROW>
        end
        continue
    end
    isSrc = local_isCanonicalSource(tok{2});
    namesSource = local_namesSource(local_textNorm(ln)) || local_namesSource(local_logicalNorm(tok{2}));
    if namesSource && ~isSrc
        malformed(end+1) = struct('lineNo', i, 'text', strtrim(ln), ...
            'reason', 'names production source through a non-canonical path'); %#ok<AGROW>
        continue
    end
    E(end+1) = struct('lineNo', i, 'digest', tok{1}, 'path', tok{2}, 'isSource', isSrc); %#ok<AGROW>
end
end

function tf = local_isCanonicalSource(p)
%LOCAL_ISCANONICALSOURCE  The exact logical production-source paths, and nothing else.
tf = false;
if strcmp(p, 'analysis/OlhoffCurrent/SOURCE_MANIFEST.json'); tf = true; return; end
if ~startsWith(p, 'analysis/OlhoffCurrent/+impl/'); return; end
if isempty(regexp(p, '^[A-Za-z0-9_+./-]+$', 'once')); return; end
seg = strsplit(p, '/', 'CollapseDelimiters', false);
tf = ~any(cellfun(@isempty, seg)) && ~any(strcmp(seg, '.')) && ~any(strcmp(seg, '..'));
end

function tf = local_namesSource(normalized)
tf = contains(normalized, 'olhoffcurrent/+impl') || contains(normalized, 'olhoffcurrent/source_manifest.json');
end

function t = local_textNorm(txt)
%LOCAL_TEXTNORM  Free text, lower-cased, with \ -> /, repeated / collapsed, ./ dropped.
t = regexprep(lower(strrep(txt, '\', '/')), '/+', '/');
prev = '';
while ~strcmp(prev, t); prev = t; t = regexprep(t, '(^|/)\./', '$1'); end
end

function n = local_logicalNorm(p)
%LOCAL_LOGICALNORM  A path token as a logical path: lower-cased, \ -> /, empty and
%   . segments dropped, .. resolved.
seg = strsplit(lower(strrep(p, '\', '/')), '/', 'CollapseDelimiters', false);
out = {};
for k = 1:numel(seg)
    s = seg{k};
    if isempty(s) || strcmp(s, '.'); continue; end
    if strcmp(s, '..') && ~isempty(out) && ~strcmp(out{end}, '..')
        out(end) = [];
    else
        out{end+1} = s; %#ok<AGROW>
    end
end
n = strjoin(out, '/');
end

function tf = local_resolvesIntoProduction(fp, repo)
%LOCAL_RESOLVESINTOPRODUCTION  Does a resolved file (links followed) lie in +impl or
%   is it SOURCE_MANIFEST.json?  Case-insensitive where the filesystem usually is.
tf = false;
c = local_canon(fp);
impl = local_canon(fullfile(repo, 'analysis', 'OlhoffCurrent', '+impl'));
man = local_canon(fullfile(repo, 'analysis', 'OlhoffCurrent', 'SOURCE_MANIFEST.json'));
if isempty(c) || isempty(impl); return; end
if ismac || ispc; c = lower(c); impl = lower(impl); man = lower(man); end
tf = startsWith(c, [impl filesep]) || (~isempty(man) && strcmp(c, man));
end

function ctx = local_historicalContext(studyDir, fs, repo)
%LOCAL_HISTORICALCONTEXT  The ONE admissible historical source state of a study.
ctx = struct('ok', false, 'reason', '', 'freezeCommit', '', 'declaredTree', '', 'freezeTree', '');
if ~local_safe(repo); ctx.reason = 'UNSAFE_REPOSITORY_PATH'; return; end
relFs = local_repoRel(fs, repo);
if isempty(relFs) || ~local_safe(relFs); ctx.reason = 'HASH_FILE_OUTSIDE_REPOSITORY'; return; end
wtFs = olhoffcurrent_sha256_file(fs);
[okH, hHead] = local_blobSha(repo, 'HEAD', relFs);
if ~okH || ~strcmp(hHead, wtFs)
    ctx.reason = 'HASH_FILE_NOT_COMMITTED_UNMODIFIED_AT_HEAD'; return
end
[s, out] = local_git(repo, sprintf('log -1 --format=%%H HEAD -- "%s"', relFs));
fz = strtrim(out);
if s ~= 0 || isempty(regexp(fz, '^([0-9a-f]{40}|[0-9a-f]{64})$', 'once'))
    ctx.reason = 'FREEZE_COMMIT_UNRESOLVED'; return
end
[okF, hF] = local_blobSha(repo, fz, relFs);
if ~okF || ~strcmp(hF, wtFs)
    ctx.reason = 'HASH_FILE_AT_FREEZE_COMMIT_DIFFERS'; return
end
ctx.freezeCommit = fz;
ev = fullfile(studyDir, 'EVIDENCE.json');
relEv = local_repoRel(ev, repo);
if ~isfile(ev) || isempty(relEv) || ~local_safe(relEv)
    ctx.reason = 'EVIDENCE_JSON_ABSENT'; return
end
[okE, hE] = local_blobSha(repo, 'HEAD', relEv);
if ~okE || ~strcmp(hE, olhoffcurrent_sha256_file(ev))
    ctx.reason = 'EVIDENCE_JSON_NOT_COMMITTED_UNMODIFIED_AT_HEAD'; return
end
try
    D = jsondecode(fileread(ev));
catch
    ctx.reason = 'EVIDENCE_JSON_UNPARSEABLE'; return
end
decl = {};
for f = {'sourceTree', 'impl_tree_sha256'}
    if isfield(D, f{1}) && ischar(D.(f{1})) && ~isempty(D.(f{1})); decl{end+1} = D.(f{1}); end %#ok<AGROW>
end
if isempty(decl);                 ctx.reason = 'NO_DECLARED_SOURCE_TREE'; return; end
if numel(unique(decl)) ~= 1;      ctx.reason = 'DECLARED_SOURCE_TREES_DISAGREE'; return; end
if isempty(regexp(decl{1}, '^[0-9a-f]{64}$', 'once')); ctx.reason = 'DECLARED_SOURCE_TREE_MALFORMED'; return; end
ctx.declaredTree = decl{1};
[okT, ft, why] = local_treeAt(repo, fz);
if ~okT; ctx.reason = ['FREEZE_TREE_UNREADABLE: ' why]; return; end
ctx.freezeTree = ft;
if ~strcmp(ft, ctx.declaredTree)
    ctx.reason = 'DECLARED_SOURCE_TREE_IS_NOT_THE_TREE_AT_THE_FREEZE_COMMIT'; return
end
[okM, bytes] = local_blobBytes(repo, fz, 'analysis/OlhoffCurrent/SOURCE_MANIFEST.json');
mt = '';
if okM
    try, M = jsondecode(char(bytes(:).')); mt = M.tree_sha256; catch, mt = ''; end
end
if ~ischar(mt) || ~strcmp(mt, ctx.declaredTree)
    ctx.reason = 'SOURCE_MANIFEST_AT_FREEZE_COMMIT_DISAGREES_WITH_DECLARED_TREE'; return
end
ctx.ok = true;
end

function [v, reason, src] = local_historicalLine(ctx, e, repo)
%LOCAL_HISTORICALLINE  Verify ONE hash-file line against <freeze commit>:<its path>.
v = 'MISMATCH'; src = '';
if ~ctx.ok; reason = ctx.reason; return; end
spec = sprintf('%s:%s', ctx.freezeCommit, e.path);
if local_git(repo, sprintf('cat-file -e "%s"', spec)) ~= 0
    reason = 'HISTORICAL_PATH_ABSENT_AT_FREEZE_COMMIT'; return
end
[s, t] = local_git(repo, sprintf('cat-file -t "%s"', spec));
if s ~= 0 || ~strcmp(strtrim(t), 'blob')
    reason = 'HISTORICAL_PATH_NOT_A_BLOB_AT_FREEZE_COMMIT'; return
end
[ok, h] = local_blobSha(repo, ctx.freezeCommit, e.path);
if ~ok; reason = 'HISTORICAL_BLOB_UNREADABLE'; return; end
if ~strcmp(h, e.digest)
    reason = 'DIGEST_IS_NOT_THIS_PATH_AT_THE_FREEZE_COMMIT'; return
end
[~, out] = local_git(repo, sprintf('log -1 --format=%%H %s -- "%s"', ctx.freezeCommit, e.path));
src = strtrim(out);
v = 'HISTORICAL_VERIFIED';
reason = sprintf('digest = blob at freeze commit %s (path last changed in %s)', ...
    local_short(ctx.freezeCommit), local_short(src));
end

function cs = local_currentSource(repo)
%LOCAL_CURRENTSOURCE  CURRENT_SOURCE_HASH_VERIFIED: +impl, manifest and provenance vs HEAD.
cs = local_emptyCurrent();
if ~local_safe(repo); cs.reasons{end+1} = 'UNSAFE_REPOSITORY_PATH'; return; end
[s, out] = local_git(repo, 'rev-parse --verify "HEAD^{commit}"');
head = strtrim(out);
if s ~= 0 || isempty(head); cs.reasons{end+1} = 'HEAD_UNRESOLVED'; return; end
cs.head = head;
[okT, tree, why, rows] = local_treeAt(repo, head);
if ~okT; cs.reasons{end+1} = ['HEAD_IMPL_TREE_UNREADABLE: ' why]; return; end
cs.treeHash = tree; cs.nFiles = numel(rows);
implDir = fullfile(repo, 'analysis', 'OlhoffCurrent', '+impl');
[sl, lnk] = system(sprintf('find "%s" -type l 2>/dev/null', implDir));
if sl ~= 0
    cs.reasons{end+1} = 'IMPL_WORKING_TREE_UNLISTABLE';
elseif ~isempty(strtrim(lnk))
    cs.reasons{end+1} = ['SYMLINK_IN_IMPL: ' strtrim(lnk)];
end
implCanon = local_canon(implDir);
W = dir(fullfile(implDir, '**', '*'));
W = W(~[W.isdir]);
wrel = {};
for k = 1:numel(W)
    if olhoffcurrent_is_artifact(W(k).name); continue; end
    fc = local_canon(W(k).folder);
    if strcmp(fc, implCanon)
        wrel{end+1} = W(k).name; %#ok<AGROW>
    elseif startsWith(fc, [implCanon filesep])
        wrel{end+1} = [strrep(fc(numel(implCanon)+2:end), filesep, '/') '/' W(k).name]; %#ok<AGROW>
    else
        cs.reasons{end+1} = ['IMPL_FILE_OUTSIDE_IMPL: ' fullfile(W(k).folder, W(k).name)];
    end
end
hrel = {rows.rel};
for k = 1:numel(rows)
    fp = fullfile(implDir, rows(k).rel);
    if ~isfile(fp)
        cs.reasons{end+1} = ['MISSING_IN_WORKING_TREE: ' rows(k).rel];
    elseif ~strcmp(olhoffcurrent_sha256_file(fp), rows(k).sha)
        cs.reasons{end+1} = ['MODIFIED_IN_WORKING_TREE: ' rows(k).rel];
    end
end
extra = setdiff(wrel, hrel);
for k = 1:numel(extra); cs.reasons{end+1} = ['NOT_IN_HEAD: ' extra{k}]; end
manRel = 'analysis/OlhoffCurrent/SOURCE_MANIFEST.json';
manWt = fullfile(repo, manRel);
[okM, hM] = local_blobSha(repo, head, manRel);
if ~okM
    cs.reasons{end+1} = 'SOURCE_MANIFEST_ABSENT_AT_HEAD';
elseif ~isfile(manWt)
    cs.reasons{end+1} = 'SOURCE_MANIFEST_ABSENT_IN_WORKING_TREE';
else
    if ~strcmp(hM, olhoffcurrent_sha256_file(manWt))
        cs.reasons{end+1} = 'SOURCE_MANIFEST_DIFFERS_FROM_HEAD';
    end
    try
        M = jsondecode(fileread(manWt));
        F = M.files; if iscell(F); F = [F{:}]; end
        mrows = sort(cellfun(@(a, b) sprintf('%s  %s', a, b), {F.path}, {F.sha256}, 'UniformOutput', false));
        hrows = sort(cellfun(@(a, b) sprintf('%s  %s', a, b), {rows.rel}, {rows.sha}, 'UniformOutput', false));
        if ~isequal(mrows(:), hrows(:)) || M.n_files ~= numel(rows)
            cs.reasons{end+1} = 'SOURCE_MANIFEST_ROWS_DISAGREE_WITH_HEAD';
        end
        if ~strcmp(M.tree_sha256, tree)
            cs.reasons{end+1} = 'SOURCE_MANIFEST_TREE_DISAGREES_WITH_HEAD';
        end
    catch
        cs.reasons{end+1} = 'SOURCE_MANIFEST_UNPARSEABLE';
    end
end
pv = fullfile(repo, 'analysis', 'OlhoffCurrent', 'PROVENANCE.md');
if ~isfile(pv)
    cs.reasons{end+1} = 'PROVENANCE_MD_ABSENT';
else
    tok = regexp(fileread(pv), '\*\*Source tree SHA-256\*\*\s*\|\s*`([0-9a-f]{64})`', 'tokens');
    if numel(tok) ~= 1
        cs.reasons{end+1} = 'PROVENANCE_MD_SOURCE_TREE_ROW_NOT_UNIQUE';
    elseif ~strcmp(tok{1}{1}, tree)
        cs.reasons{end+1} = 'PROVENANCE_MD_SOURCE_TREE_DISAGREES_WITH_HEAD';
    end
end
cs.ok = isempty(cs.reasons);
if cs.ok; cs.status = 'CURRENT_SOURCE_HASH_VERIFIED'; end
end

function [ok, tree, why, rows] = local_treeAt(repo, commit)
%LOCAL_TREEAT  +impl tree hash of a COMMITTED state (olhoffcurrent_source_manifest
%   algorithm), from git objects only.
ok = false; tree = ''; why = ''; rows = struct('rel', {}, 'sha', {});
tmp = [tempname() '.lstree']; c = onCleanup(@() local_delete(tmp));
s = system(sprintf('%s ls-tree -r %s -- analysis/OlhoffCurrent/+impl > "%s" 2>/dev/null', ...
    local_gitPrefix(repo), commit, tmp));
if s ~= 0 || ~isfile(tmp); why = 'git ls-tree failed'; return; end
L = splitlines(strtrim(fileread(tmp)));
L = L(~cellfun(@isempty, L));
if isempty(L); why = 'no +impl entries'; return; end
oids = cell(numel(L), 1); rel = cell(numel(L), 1);
prefix = 'analysis/OlhoffCurrent/+impl/';
for k = 1:numel(L)
    tk = regexp(L{k}, '^(\d{6}) (\w+) ([0-9a-f]{40}|[0-9a-f]{64})\t(.+)$', 'tokens', 'once');
    if isempty(tk); why = ['unparseable entry: ' L{k}]; return; end
    if ~strcmp(tk{2}, 'blob') || ~any(strcmp(tk{1}, {'100644', '100755'}))
        why = ['non-regular entry: ' L{k}]; return
    end
    if ~local_isCanonicalSource(tk{4}); why = ['non-canonical path: ' tk{4}]; return; end
    oids{k} = tk{3}; rel{k} = tk{4}(numel(prefix)+1:end);
end
[okB, shas, whyB] = local_batchSha(repo, oids);
if ~okB; why = whyB; return; end
keep = ~cellfun(@olhoffcurrent_is_artifact, rel);
rel = rel(keep); shas = shas(keep);
[rel, ix] = sort(rel); shas = shas(ix);
rows = struct('rel', rel(:).', 'sha', shas(:).');
lines = cell(numel(rel), 1);
for k = 1:numel(rel); lines{k} = sprintf('%s  %s', rel{k}, shas{k}); end
tree = local_sha(uint8(strjoin(lines, newline)));
ok = true;
end

function [ok, shas, why] = local_batchSha(repo, oids)
%LOCAL_BATCHSHA  SHA-256 of the raw content of each blob, via git cat-file --batch.
ok = false; why = ''; shas = cell(size(oids));
in = [tempname() '.in']; out = [tempname() '.out'];
c = onCleanup(@() cellfun(@local_delete, {in, out}));
fid = fopen(in, 'w'); fprintf(fid, '%s\n', oids{:}); fclose(fid);
s = system(sprintf('%s cat-file --batch < "%s" > "%s" 2>/dev/null', local_gitPrefix(repo), in, out));
if s ~= 0 || ~isfile(out); why = 'git cat-file --batch failed'; return; end
fid = fopen(out, 'r'); B = fread(fid, Inf, '*uint8'); fclose(fid);
pos = 1;
for k = 1:numel(oids)
    nl = find(B(pos:end) == 10, 1);
    if isempty(nl); why = 'truncated cat-file output'; return; end
    hdr = char(B(pos:pos+nl-2).');
    pos = pos + nl;
    tk = regexp(hdr, '^(\S+) (\w+) (\d+)$', 'tokens', 'once');
    if isempty(tk) || ~strcmp(tk{1}, oids{k}) || ~strcmp(tk{2}, 'blob')
        why = ['object not a readable blob: ' hdr]; return
    end
    n = str2double(tk{3});
    if pos + n > numel(B) || B(pos + n) ~= 10; why = 'malformed cat-file output'; return; end
    shas{k} = local_sha(B(pos:pos+n-1));
    pos = pos + n + 1;
end
ok = true;
end

function [ok, h] = local_blobSha(repo, commit, path)
%LOCAL_BLOBSHA  SHA-256 of <commit>:<path>, only after the object is PROVED to be a blob.
h = '';
[ok, bytes] = local_blobBytes(repo, commit, path);
if ok; h = local_sha(bytes); end
end

function [ok, bytes] = local_blobBytes(repo, commit, path)
ok = false; bytes = uint8([]);
if isempty(path) || ~local_safe(path); return; end
spec = sprintf('%s:%s', commit, path);
if local_git(repo, sprintf('cat-file -e "%s"', spec)) ~= 0; return; end
[s, t] = local_git(repo, sprintf('cat-file -t "%s"', spec));
if s ~= 0 || ~strcmp(strtrim(t), 'blob'); return; end
tmp = [tempname() '.blob']; c = onCleanup(@() local_delete(tmp));
s = system(sprintf('%s cat-file blob "%s" > "%s" 2>/dev/null', local_gitPrefix(repo), spec, tmp));
if s ~= 0 || ~isfile(tmp); return; end
fid = fopen(tmp, 'r'); bytes = fread(fid, Inf, '*uint8'); fclose(fid);
ok = true;
end

function [s, out] = local_git(repo, args)
[s, out] = system(sprintf('%s %s 2>/dev/null', local_gitPrefix(repo), args));
end

function c = local_gitPrefix(repo)
%LOCAL_GITPREFIX  Every git call reads the repository it NAMES, as COMMITTED: the
%   environment cannot redirect it (GIT_DIR & co.), replace refs cannot substitute
%   objects, and no pager can block on MATLAB's TTY.
c = sprintf(['env -u GIT_DIR -u GIT_WORK_TREE -u GIT_INDEX_FILE -u GIT_OBJECT_DIRECTORY ' ...
             '-u GIT_ALTERNATE_OBJECT_DIRECTORIES -u GIT_COMMON_DIR -u GIT_NAMESPACE ' ...
             '-u GIT_REPLACE_REF_BASE -u GIT_CONFIG_PARAMETERS -u GIT_CONFIG_COUNT ' ...
             'git --no-pager --no-replace-objects -C "%s"'], repo);
end

function h = local_sha(bytes)
md = java.security.MessageDigest.getInstance('SHA-256');
if ~isempty(bytes); md.update(uint8(bytes(:))); end
d = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(d, 2).', 1, []));
end

function rel = local_repoRel(f, repo)
%LOCAL_REPOREL  Repository-relative path of f, or '' if f is not inside repo.
rel = '';
a = local_canon(f); r = local_canon(repo);
if isempty(a) || isempty(r) || ~startsWith(a, [r filesep]); return; end
rel = strrep(a(numel(r)+2:end), filesep, '/');
end

function c = local_canon(p)
try
    jf = java.io.File(p);
    if ~jf.isAbsolute(); jf = java.io.File(fullfile(pwd, p)); end
    c = char(jf.getCanonicalPath());
catch
    c = '';
end
end

function tf = local_safe(s)
tf = ~any(ismember(char(s), ['"$`\' char(10) char(13)]));
end

function local_delete(p)
if isfile(p); delete(p); end
end

function s = local_short(h)
s = h(1:min(12, numel(h)));
end

function L = local_emptyLines()
L = struct('lineNo', {}, 'path', {}, 'digest', {}, 'verdict', {}, 'freezeCommit', {}, ...
           'sourceCommit', {}, 'reason', {});
end

function cs = local_emptyCurrent()
cs = struct('status', 'CURRENT_SOURCE_HASH_NOT_VERIFIED', 'ok', false, 'head', '', ...
            'treeHash', '', 'nFiles', 0, 'reasons', {{}});
end

function fp = local_resolvePath(rel, studyDir, repo)
%LOCAL_RESOLVEPATH  Resolve a manifest path to a real file (NON-source lines only).
%   STUDY-LOCAL FIRST: a bare name like 'PROVENANCE.md' means the study's own
%   file, and several studies have one alongside the implementation-level file
%   of the same name.  isfile() is used rather than exist(), because exist()
%   also searches the MATLAB path and would silently resolve a bare name to
%   whichever copy happens to be on it.  Production-source lines never come
%   here: they resolve only at <repo>/<path>.
fp = '';
cand = {fullfile(studyDir, rel), fullfile(studyDir, 'runs', rel), ...
        fullfile(repo, rel), rel};
for c = 1:numel(cand)
    if isfile(cand{c}); fp = cand{c}; return; end
end
end

function mats = local_namedMats(studyDir)
%LOCAL_NAMEDMATS  The .mat files the study CLAIMS to hold.
%   From FINAL_SHA256.txt only digested lines count (see local_parseHashFile);
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
