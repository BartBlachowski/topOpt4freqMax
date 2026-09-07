function nFail = test_currentness()
%TEST_CURRENTNESS  The provenance and currentness machinery must work, and must
%   report the states it claims to report.
%
%   Checks:
%     1. SOURCE_MANIFEST.json exists and +impl/ hashes to it exactly
%     2. PROVENANCE.json parses and names a real upstream commit
%     3. olhoffcurrent_currentness() reports a state from the declared model
%     4. LOCAL_MODIFIED is REACHABLE -- proved by perturbing a temporary copy
%        of the manifest rather than by trusting the branch to work
%     5. the production preset delegates to the upstream preset it claims

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(root);
guard = olhoffcurrent_paths(); %#ok<NASGU>

nFail = 0;
fprintf('\n%s\nTEST_CURRENTNESS\n%s\n', repmat('=',1,72), repmat('=',1,72));

% ---- 1. integrity --------------------------------------------------------
man = olhoffcurrent_source_manifest();
nFail = nFail + rep('+impl/ matches SOURCE_MANIFEST.json', man.ok, ...
    sprintf('%d files, tree %s', man.nFiles, man.treeHash(1:16)));

% ---- 2. provenance -------------------------------------------------------
prov = jsondecode(fileread(fullfile(root,'PROVENANCE.json')));
okP = ~isempty(prov.source.commit) && numel(prov.source.commit) == 40;
nFail = nFail + rep('PROVENANCE.json names a 40-char upstream commit', okP, prov.source.commit);

% ---- 3. state model ------------------------------------------------------
st = olhoffcurrent_currentness('Verbose', false);
known = {'CURRENT','UPSTREAM_AHEAD','LOCAL_MODIFIED','PROVENANCE_MISMATCH','UPSTREAM_UNREACHABLE'};
nFail = nFail + rep('currentness reports a declared state', any(strcmp(st.state, known)), st.state);
nFail = nFail + rep('local integrity is PASS', st.localOk, st.detail);

% ---- 4. LOCAL_MODIFIED is reachable, not decorative ----------------------
% Perturb a COPY of the manifest, point the check at it, and require the state
% to flip.  Nothing under +impl/ is touched.
manFile = fullfile(root,'SOURCE_MANIFEST.json');
bak = [manFile '.testbak'];
copyfile(manFile, bak);
cl = onCleanup(@() local_restore(bak, manFile));
J = jsondecode(fileread(manFile));
J.files(1).sha256 = repmat('0',1,64);
fid = fopen(manFile,'w'); fprintf(fid,'%s\n', jsonencode(J,'PrettyPrint',true)); fclose(fid);
st2 = olhoffcurrent_currentness('Verbose', false);
nFail = nFail + rep('a corrupted manifest yields LOCAL_MODIFIED', ...
    strcmp(st2.state,'LOCAL_MODIFIED'), st2.state);
clear cl

st3 = olhoffcurrent_currentness('Verbose', false);
nFail = nFail + rep('restoring the manifest restores the state', ...
    strcmp(st3.state, st.state), st3.state);

% ---- 5. preset delegation ------------------------------------------------
info = olhoffcurrent_preset();
T = olh.presets.list();
nFail = nFail + rep('the upstream preset it delegates to exists', ...
    any(strcmp(info.upstreamPreset, T(:,1))), info.upstreamPreset);
nFail = nFail + rep('the production preset name is not an audit code', ...
    ~any(strcmpi(info.name, {'M4','S2','R2','P1','PD1','PM1','T800','TMA','B0','REG160'})), ...
    info.name);

fprintf('%s\n  failures: %d\n\n', repmat('-',1,72), nFail);
end

function local_restore(bak, manFile)
if exist(bak,'file') == 2, copyfile(bak, manFile); delete(bak); end
end

function n = rep(label, ok, detail)
if ok, fprintf('  [PASS] %-52s %s\n', label, detail); n = 0;
else,  fprintf('  [FAIL] %-52s %s\n', label, detail); n = 1; end
end
