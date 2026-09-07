function st = olhoffcurrent_currentness(varargin)
%OLHOFFCURRENT_CURRENTNESS  Is analysis/OlhoffCurrent still current?
%
%   st = OLHOFFCURRENT_CURRENTNESS() answers the one question that must stay
%   easy to answer, and reports exactly one state:
%
%     CURRENT               the promoted source is intact and matches the
%                           upstream commit it was promoted from
%     LOCAL_MODIFIED        +impl/ no longer hashes to SOURCE_MANIFEST.json.
%                           Production source has been edited in place -- the
%                           most serious state, because it means the recorded
%                           provenance is now a lie
%     UPSTREAM_AHEAD        upstream has commits after the promoted one.  This
%                           is INFORMATIONAL, NOT OBSOLESCENCE -- see below
%     PROVENANCE_MISMATCH   the promoted commit is not in upstream's history,
%                           or PROVENANCE.json disagrees with the promoted files
%     UPSTREAM_UNREACHABLE  the development repository is not on this machine;
%                           the local integrity check still ran
%
%   WHY "UPSTREAM_AHEAD" IS NOT "OBSOLETE"
%   --------------------------------------
%   The upstream repository is DEVELOPMENT / RESEARCH UPSTREAM.  Experimental
%   commits land there constantly, and most of them will never be promoted --
%   they are audits, spikes and abandoned branches.  Production currentness is
%   changed by ONE event and one only: a human explicitly ACCEPTS an upstream
%   state and promotes it.  So an upstream commit that nobody has accepted does
%   not make production stale; it makes production DIFFERENT FROM A DRAFT.
%
%   Accordingly this function NEVER updates anything.  It reports, and a person
%   decides.
%
%   Options:
%     'Verbose'  (default true)  print the report
%
%   st fields: state, detail, localOk, upstream (repo/branch/head/promoted/
%   commitsAhead/promotedIsAncestor), manifest.

p = inputParser();
p.addParameter('Verbose', true, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});

root = olhoffcurrent_root();
prov = jsondecode(fileread(fullfile(root, 'PROVENANCE.json')));

st = struct('state','UNKNOWN','detail','','localOk',false, ...
            'upstream',struct(),'manifest',struct());

% ---- 1. local integrity: does the promoted source still hash as recorded?
man = olhoffcurrent_source_manifest('Verify', true);
st.manifest = man;
st.localOk  = man.ok;

% ---- 2. upstream state ---------------------------------------------------
up = struct('repo', prov.source.repository, 'reachable', false, ...
            'branch','', 'head','', 'promoted', prov.source.commit, ...
            'commitsAhead', NaN, 'promotedIsAncestor', false, 'dirty', '');
if exist(prov.source.repository, 'dir') == 7
    up.reachable = true;
    up.branch = local_git(prov.source.repository, 'rev-parse --abbrev-ref HEAD');
    up.head   = local_git(prov.source.repository, 'rev-parse HEAD');
    up.dirty  = local_git(prov.source.repository, 'status --porcelain');
    anc = local_gitStatus(prov.source.repository, ...
        sprintf('merge-base --is-ancestor %s HEAD', prov.source.commit));
    up.promotedIsAncestor = (anc == 0);
    n = local_git(prov.source.repository, ...
        sprintf('rev-list --count %s..HEAD', prov.source.commit));
    if ~isempty(n); up.commitsAhead = str2double(n); end
end
st.upstream = up;

% ---- 3. the state model --------------------------------------------------
if ~st.localOk
    st.state  = 'LOCAL_MODIFIED';
    bad = [man.mismatches, man.missing, man.extra];
    st.detail = sprintf(['+impl/ does not match SOURCE_MANIFEST.json (%d file(s) ' ...
        'differ/missing/extra): %s. The recorded provenance no longer describes ' ...
        'the code on disk.'], numel(bad), strjoin(bad, ', '));
elseif ~up.reachable
    st.state  = 'UPSTREAM_UNREACHABLE';
    st.detail = sprintf(['%s is not present on this machine. Local integrity ' ...
        'PASSED, so production is internally consistent; upstream comparison ' ...
        'could not be made.'], prov.source.repository);
elseif ~up.promotedIsAncestor
    st.state  = 'PROVENANCE_MISMATCH';
    st.detail = sprintf(['the promoted commit %s is not an ancestor of upstream ' ...
        'HEAD %s on branch %s. Upstream history was rewritten, or the branch ' ...
        'moved. Investigate before promoting anything further.'], ...
        prov.source.commit, up.head, up.branch);
elseif up.commitsAhead > 0
    st.state  = 'UPSTREAM_AHEAD';
    st.detail = sprintf(['upstream has %d commit(s) after the promoted state on ' ...
        'branch %s. This is INFORMATIONAL: production currentness changes only ' ...
        'when an upstream state is explicitly ACCEPTED and promoted. ' ...
        'analysis/OlhoffCurrent remains valid production.'], up.commitsAhead, up.branch);
else
    st.state  = 'CURRENT';
    st.detail = sprintf('promoted source is intact and matches upstream %s at %s.', ...
        up.branch, prov.source.commit);
end

if p.Results.Verbose
    fprintf('\nOLHOFF CURRENTNESS\n%s\n', repmat('=',1,64));
    fprintf('  state            : %s\n', st.state);
    fprintf('  local integrity  : %s (%d files, tree %s)\n', ...
        local_tf(st.localOk), man.nFiles, man.treeHash(1:16));
    fprintf('  promoted commit  : %s\n', prov.source.commit);
    if up.reachable
        fprintf('  upstream branch  : %s\n', up.branch);
        fprintf('  upstream HEAD    : %s\n', up.head);
        fprintf('  commits ahead    : %d\n', up.commitsAhead);
        if ~isempty(strtrim(up.dirty))
            fprintf('  upstream dirty   : YES (development tree; not a gate)\n');
        end
    else
        fprintf('  upstream         : NOT REACHABLE\n');
    end
    fprintf('  %s\n\n', st.detail);
end
end

function s = local_git(repo, args)
[stt, out] = system(sprintf('git -C "%s" %s 2>/dev/null', repo, args));
if stt == 0; s = strtrim(out); else; s = ''; end
end
function c = local_gitStatus(repo, args)
[c, ~] = system(sprintf('git -C "%s" %s 2>/dev/null', repo, args));
end
function s = local_tf(c), if c, s='PASS'; else, s='FAIL'; end, end
