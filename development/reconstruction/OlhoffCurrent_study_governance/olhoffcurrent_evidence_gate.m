function st = olhoffcurrent_evidence_gate(studyDir, varargin)
%OLHOFFCURRENT_EVIDENCE_GATE  Is a diagnostic's REQUIRED raw evidence still there?
%
%   st = OLHOFFCURRENT_EVIDENCE_GATE(studyDir) reads EVIDENCE.json from a
%   diagnostic study directory and answers one question:
%
%       can this study still be believed?
%
%   WHY THIS EXISTS
%   ---------------
%   Three completed OlhoffCurrent diagnostics (move_stop, admission_rule,
%   move_transition) each built a full NE x nOuter density trajectory, saved it
%   to a .mat file, and cited conclusions drawn from it.  Every one of those
%   files is now gone.  They sat under a git-ignored path (`*.mat`, both
%   repo-wide and re-stated in diagnostics/.gitignore) and nothing ever checked
%   that they still existed.
%
%   move_stop had hashed its four files in FINAL_SHA256.txt, so its loss is
%   PROVABLE.  admission_rule and move_transition never manifested theirs at
%   all, so for those two the loss is merely inferable from the code that wrote
%   them.  All three studies nevertheless reported themselves complete and
%   frozen.  A later study then could not answer questions about element
%   identity, spatial location or persistence, and had to record them as
%   permanently unanswerable.
%
%   THE RULE THIS ENFORCES
%   ----------------------
%   A frozen scientific diagnostic may not claim integrity while evidence it
%   declared REQUIRED is missing or altered.  FINAL_SHA256.txt cannot do this
%   job: it is a flat hash list with no notion of required-versus-disposable,
%   and a missing entry there is indistinguishable from a file that was never
%   meant to be permanent.
%
%   Deliberately ORTHOGONAL to the source-integrity mechanism.  This function
%   does not read, modify or relax SOURCE_MANIFEST.json, currentness, the
%   dispatch gate or the artifact policy; it lives outside +impl/ so it cannot
%   change the canonical tree hash.  Source integrity asks "is production code
%   what we recorded?"; this asks "is the scientific evidence still on disk?".
%   Neither substitutes for the other.
%
%   PER-ARTIFACT STATUS  (st.artifacts(i).status)
%     REQUIRED_PRESENT_MATCH   required, on disk, SHA-256 matches   -> ok
%     REQUIRED_MISSING         required, absent                     -> FAIL
%     REQUIRED_HASH_MISMATCH   required, present, bytes differ      -> FAIL
%     OPTIONAL_PRESENT         optional, present (hash checked if recorded)
%     OPTIONAL_MISSING         optional, absent                     -> ok
%     OPTIONAL_HASH_MISMATCH   optional, present, bytes differ      -> ok, WARNED
%     SCRATCH                  disposable; existence never asserted -> ok
%
%   st.ok is TRUE iff no artifact is REQUIRED_MISSING or REQUIRED_HASH_MISMATCH.
%
%   Options:
%     'Verbose'  (default true)   print the per-artifact report
%     'RepoRoot' (default derived) repository root for resolving evidenceRoot
%
%   See also OLHOFFCURRENT_EVIDENCE_DECLARE, OLHOFFCURRENT_SOURCE_MANIFEST.

p = inputParser();
p.addParameter('Verbose', true, @(v) islogical(v) && isscalar(v));
p.addParameter('RepoRoot', '', @(v) ischar(v) || isstring(v));
p.parse(varargin{:});
verbose = p.Results.Verbose;

root = olhoffcurrent_root();
repo = char(p.Results.RepoRoot);
if isempty(repo); repo = fileparts(fileparts(root)); end

studyDir = char(studyDir);
manPath  = fullfile(studyDir, 'EVIDENCE.json');

st = struct('ok', false, 'studyDir', studyDir, 'manifestPath', manPath, ...
            'manifestPresent', false, 'nRequired', 0, 'nPresent', 0, ...
            'nMissing', 0, 'nMismatch', 0, 'nOptional', 0, 'nScratch', 0, ...
            'artifacts', struct([]), 'detail', '');

% A study that declares nothing cannot be gated.  That is itself a failure:
% "no declaration" is exactly the state the three lost studies were in.
if exist(manPath, 'file') ~= 2
    st.detail = sprintf('EVIDENCE.json absent in %s -- the study declares no evidence', studyDir);
    if verbose
        fprintf('[evidence] %-58s NO DECLARATION -> FAIL\n', studyDir);
    end
    return
end
st.manifestPresent = true;

D = jsondecode(fileread(manPath));
if ~isfield(D, 'artifacts') || isempty(D.artifacts)
    st.detail = 'EVIDENCE.json declares an empty artifact list';
    if verbose; fprintf('[evidence] empty artifact list -> FAIL\n'); end
    return
end

% jsondecode gives a struct array for homogeneous lists, a cell for ragged ones.
A = D.artifacts;
if iscell(A); items = A; else; items = num2cell(A); end

evRoot = '';
if isfield(D, 'evidenceRoot'); evRoot = char(D.evidenceRoot); end

recs = struct('path', {}, 'resolved', {}, 'class', {}, 'status', {}, ...
              'bytes', {}, 'bytesOnDisk', {}, 'sha256', {}, 'sha256OnDisk', {});

for i = 1:numel(items)
    a = items{i};
    rel = char(a.path);
    cls = lower(char(a.class));

    % Resolve: evidenceRoot-relative first (the durable external location),
    % then study-relative (compact tracked artifacts living beside the report).
    cand = {};
    if ~isempty(evRoot); cand{end+1} = fullfile(repo, evRoot, rel); end %#ok<AGROW>
    cand{end+1} = fullfile(studyDir, rel); %#ok<AGROW>
    cand{end+1} = fullfile(repo, rel);     %#ok<AGROW>
    resolved = ''; 
    for c = 1:numel(cand)
        if exist(cand{c}, 'file') == 2; resolved = cand{c}; break; end
    end
    if isempty(resolved); resolved = cand{1}; end   % report the canonical path

    r = struct('path', rel, 'resolved', resolved, 'class', cls, 'status', '', ...
               'bytes', NaN, 'bytesOnDisk', NaN, 'sha256', '', 'sha256OnDisk', '');
    if isfield(a, 'bytes');  r.bytes  = double(a.bytes);  end
    if isfield(a, 'sha256'); r.sha256 = char(a.sha256);   end

    present = exist(resolved, 'file') == 2;
    if present
        d = dir(resolved); r.bytesOnDisk = d(1).bytes;
        if ~isempty(r.sha256)
            r.sha256OnDisk = olhoffcurrent_sha256_file(resolved);
        end
    end

    switch cls
        case 'required'
            st.nRequired = st.nRequired + 1;
            if ~present
                r.status = 'REQUIRED_MISSING';   st.nMissing = st.nMissing + 1;
            elseif ~isempty(r.sha256) && ~strcmpi(r.sha256, r.sha256OnDisk)
                r.status = 'REQUIRED_HASH_MISMATCH'; st.nMismatch = st.nMismatch + 1;
            else
                r.status = 'REQUIRED_PRESENT_MATCH'; st.nPresent = st.nPresent + 1;
            end
        case 'optional'
            st.nOptional = st.nOptional + 1;
            if ~present
                r.status = 'OPTIONAL_MISSING';
            elseif ~isempty(r.sha256) && ~strcmpi(r.sha256, r.sha256OnDisk)
                r.status = 'OPTIONAL_HASH_MISMATCH';
            else
                r.status = 'OPTIONAL_PRESENT';
            end
        case 'scratch'
            st.nScratch = st.nScratch + 1;
            r.status = 'SCRATCH';
        otherwise
            % An unrecognised class must never be silently treated as harmless.
            r.status = 'REQUIRED_MISSING';
            st.nRequired = st.nRequired + 1; st.nMissing = st.nMissing + 1;
            r.class = sprintf('%s (UNKNOWN CLASS -- treated as required)', cls);
    end
    recs(end+1) = r; %#ok<AGROW>
end

st.artifacts = recs;
st.ok = (st.nMissing == 0) && (st.nMismatch == 0);
st.detail = sprintf(['%d required (%d present/match, %d missing, %d mismatch), ' ...
                     '%d optional, %d scratch'], st.nRequired, st.nPresent, ...
                    st.nMissing, st.nMismatch, st.nOptional, st.nScratch);

if verbose
    fprintf('\n%s\nEVIDENCE GATE  %s\n%s\n', repmat('=',1,72), studyDir, repmat('=',1,72));
    for i = 1:numel(recs)
        fprintf('  %-26s %s\n', recs(i).status, recs(i).path);
        if strcmp(recs(i).status, 'REQUIRED_HASH_MISMATCH')
            fprintf('      recorded %s\n      on disk  %s\n', recs(i).sha256, recs(i).sha256OnDisk);
        end
    end
    fprintf('  %s\n  RESULT: %s\n', st.detail, ternary(st.ok, 'PASS', 'FAIL'));
end
end

function s = ternary(c, a, b)
if c; s = a; else; s = b; end
end
