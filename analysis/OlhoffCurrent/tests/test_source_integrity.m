function nFail = test_source_integrity()
%TEST_SOURCE_INTEGRITY  The integrity gate must ignore artifacts and ONLY artifacts.
%
%   The gate exists to answer one question -- "is the production source on disk
%   the source we recorded?" -- and it has failed that question in both
%   directions:
%
%     * it once said LOCAL_MODIFIED because macOS wrote two .DS_Store files,
%       blocking production while all 74 source files were byte identical;
%     * a careless repair would say CURRENT while an unexpected .m file sat in
%       the tree, which is the far worse failure.
%
%   So both directions are tested, and the destructive cases restore exact
%   bytes under onCleanup and verify the restoration.
%
%   TEST A  pristine tree                              -> CURRENT / PASS
%   TEST B  .DS_Store added                            -> CURRENT / PASS
%   TEST C  *.asv added                                -> CURRENT / PASS
%   TEST D  *.m~ added                                 -> CURRENT / PASS
%   TEST E  recorded production .m modified            -> LOCAL_MODIFIED / BLOCK
%   TEST F  recorded production source removed         -> BLOCK
%   TEST G  unexpected new .m under +impl              -> BLOCK
%   TEST H  unexpected .m inside a package directory   -> BLOCK
%   TEST I  competing Olhoff implementation on path    -> BLOCK
%   TEST J  tools/Matlab 'asfound' mmasub wins         -> BLOCK

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
repo = fileparts(fileparts(root));
addpath(root);
core = fullfile(root, '+impl');

nFail = 0;
fprintf('\n%s\nTEST_SOURCE_INTEGRITY\n%s\n', repmat('=',1,72), repmat('=',1,72));

% Guard the whole suite: whatever happens, the tree must end pristine.
finalCheck = onCleanup(@() local_assertPristine(root));

% ---------------------------------------------------------------- TEST A
nFail = nFail + chk('A  pristine tree -> CURRENT', @() local_isCurrent(), true);

% ------------------------------------------------------------ TESTS B-D
artifacts = { 'B  .DS_Store added -> CURRENT', fullfile(core,'.DS_Store_probe_placeholder'); ...
              'C  *.asv added     -> CURRENT', fullfile(core,'algo','probe_artifact.asv'); ...
              'D  *.m~ added      -> CURRENT', fullfile(core,'algo','probe_artifact.m~') };
% .DS_Store must be tested under its real name, not a placeholder.
artifacts{1,2} = fullfile(core,'algo','.DS_Store');
for i = 1:size(artifacts,1)
    nFail = nFail + chk(artifacts{i,1}, @() local_withFile(artifacts{i,2}, ...
        'artifact probe', @() local_isCurrent()), true);
end

% ---------------------------------------------------------------- TEST E
% Modify a recorded .m in place, then restore its exact bytes.
victim = fullfile(core, 'algo', 'textStamp.m');   % a non-numerical helper
nFail = nFail + chk('E  recorded .m modified -> BLOCK', ...
    @() local_withModified(victim, @() local_isBlocked('LOCAL_MODIFIED')), true);

% ---------------------------------------------------------------- TEST F
nFail = nFail + chk('F  recorded source removed -> BLOCK', ...
    @() local_withRemoved(victim, @() local_isBlocked('LOCAL_MODIFIED')), true);

% ---------------------------------------------------------------- TEST G
nFail = nFail + chk('G  unexpected new .m under +impl -> BLOCK', ...
    @() local_withFile(fullfile(core,'algo','probe_unexpected.m'), ...
        'function probe_unexpected(), end', @() local_isBlocked('LOCAL_MODIFIED')), true);

% ---------------------------------------------------------------- TEST H
nFail = nFail + chk('H  unexpected .m inside a package dir -> BLOCK', ...
    @() local_withFile(fullfile(core,'architecture','+olh','+config','probe_pkg.m'), ...
        'function probe_pkg(), end', @() local_isBlocked('LOCAL_MODIFIED')), true);

% ---------------------------------------------------------------- TEST I
nFail = nFail + chk('I  competing Olhoff implementation on path -> BLOCK', ...
    @() local_withPath({fullfile(repo,'Matlab','reproduction2007','algo'), ...
                        fullfile(repo,'Matlab','reproduction2007','fem')}, false, ...
        @() local_gateRefuses()), true);

% ---------------------------------------------------------------- TEST J
nFail = nFail + chk('J  tools/Matlab ''asfound'' mmasub wins -> BLOCK', ...
    @() local_withPath({fullfile(repo,'tools','Matlab')}, true, ...
        @() local_gateRefuses()), true);

fprintf('%s\n  failures: %d\n\n', repmat('-',1,72), nFail);
end

% =========================================================================
function ok = local_isCurrent()
m  = olhoffcurrent_source_manifest();
st = olhoffcurrent_currentness('Verbose', false);
ok = m.ok && isempty(m.mismatches) && isempty(m.missing) && isempty(m.extra) && ...
     strcmp(st.state, 'CURRENT') && m.nFiles == 74;
if ~ok
    fprintf('      (ok=%d nFiles=%d state=%s mism=%d miss=%d extra=%d)\n', ...
        m.ok, m.nFiles, st.state, numel(m.mismatches), numel(m.missing), numel(m.extra));
end
end

function ok = local_isBlocked(wantState)
m  = olhoffcurrent_source_manifest();
st = olhoffcurrent_currentness('Verbose', false);
ok = ~m.ok && strcmp(st.state, wantState);
fprintf('      state=%s  mismatches=%d missing=%d extra=%d\n', ...
    st.state, numel(m.mismatches), numel(m.missing), numel(m.extra));
end

function ok = local_gateRefuses()
rep = olhoffcurrent_assert_dispatch('Throw', false);
ok = ~rep.ok;
if ~isempty(rep.blockers)
    fprintf('      first blocker: %s\n', rep.blockers{1});
end
end

% ---- scaffolding: every one of these restores exactly -------------------
function ok = local_withFile(p, contents, fn)
if exist(p,'file') == 2
    error('test_source_integrity:ProbeExists', 'probe file already exists: %s', p);
end
fid = fopen(p,'w'); fprintf(fid,'%s\n',contents); fclose(fid);
c = onCleanup(@() local_deleteIfPresent(p));
ok = fn();
end

function ok = local_withModified(p, fn)
orig = local_readBytes(p);
h0 = olhoffcurrent_sha256_file(p);
c = onCleanup(@() local_restoreBytes(p, orig, h0));
fid = fopen(p,'a'); fprintf(fid,'%% integrity probe\n'); fclose(fid);
ok = fn();
end

function ok = local_withRemoved(p, fn)
orig = local_readBytes(p);
h0 = olhoffcurrent_sha256_file(p);
c = onCleanup(@() local_restoreBytes(p, orig, h0));
delete(p);
ok = fn();
end

function ok = local_withPath(dirs, atFront, fn)
old = path(); c = onCleanup(@() path(old));
d = olhoffcurrent_impl_dirs();
addpath(d.algo, d.fem, d.filter, d.architecture, d.mma_published);
for k = 1:numel(dirs)
    if exist(dirs{k},'dir') == 7
        if atFront; addpath(dirs{k}, '-begin'); else; addpath(dirs{k}); end
    end
end
ok = fn();
end

function b = local_readBytes(p)
fid = fopen(p,'r','n'); c = onCleanup(@() fclose(fid));
b = fread(fid, Inf, '*uint8');
end

function local_restoreBytes(p, bytes, wantHash)
fid = fopen(p,'w','n'); fwrite(fid, bytes, 'uint8'); fclose(fid);
got = olhoffcurrent_sha256_file(p);
if ~strcmp(got, wantHash)
    error('test_source_integrity:RestoreFailed', ...
        'FAILED TO RESTORE %s (hash %s, expected %s)', p, got, wantHash);
end
end

function local_deleteIfPresent(p)
if exist(p,'file') == 2; delete(p); end
end

function local_assertPristine(root)
addpath(root);
m = olhoffcurrent_source_manifest();
if ~(m.ok && m.nFiles == 74)
    error('test_source_integrity:TreeNotRestored', ...
        ['THE TREE WAS NOT RESTORED: ok=%d nFiles=%d mismatches=%d missing=%d ' ...
         'extra=%d. Investigate before running anything else.'], ...
        m.ok, m.nFiles, numel(m.mismatches), numel(m.missing), numel(m.extra));
end
fprintf('  [OK]   tree restored pristine: %d source files, tree %s\n', ...
    m.nFiles, m.treeHash(1:16));
end

function n = chk(label, fn, want)
try
    ok = fn();
catch ME
    fprintf('  [FAIL] %-56s (threw %s)\n', label, ME.identifier);
    n = 1; return
end
if isequal(ok, want); fprintf('  [PASS] %s\n', label); n = 0;
else;                 fprintf('  [FAIL] %s\n', label); n = 1;
end
end
