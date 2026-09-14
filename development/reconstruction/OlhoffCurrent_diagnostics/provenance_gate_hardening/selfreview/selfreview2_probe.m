function selfreview2_probe(repo)
%SELFREVIEW2_PROBE  Can git replace refs or a GIT_DIR environment redirect make an
%   edited +impl pass G6 of the COMMITTED gate (adf86a3)?
setenv('GIT_PAGER', 'cat');
A = [tempname() '_srA']; B = [tempname() '_srB'];
c = onCleanup(@() cellfun(@(d) rmdir(d, 's'), {A, B}));
sh(sprintf('git clone -q --shared "%s" "%s"', repo, A));
oc = fullfile(A, 'analysis', 'OlhoffCurrent'); addpath(oc);
study = fullfile(oc, 'diagnostics', 'two_branch_controller_validation');   % any compliant committed study
src = 'analysis/OlhoffCurrent/+impl/architecture/olhoffSolve.m';
% ---- dirty +impl with regenerated manifest and PROVENANCE row (uncommitted)
fid = fopen(fullfile(A, src), 'a'); fprintf(fid, '\n%% LOCAL EDIT\n'); fclose(fid);
man = olhoffcurrent_source_manifest('Write', true);                        % root is A (on path)
m2 = olhoffcurrent_source_manifest('Verify', true);
pv = fullfile(oc, 'PROVENANCE.md'); t = fileread(pv);
t = regexprep(t, '(\*\*Source tree SHA-256\*\*\s*\|\s*`)[0-9a-f]{64}', ['$1' m2.treeHash]);
fid = fopen(pv, 'w'); fwrite(fid, t); fclose(fid);
st0 = olhoffcurrent_finalization_gate(study, 'Verbose', false, 'RepoRoot', A);
fprintf('SR2 baseline dirty (no tricks): G6=%d %s\n', st0.gates.G6, strjoin(st0.currentSource.reasons, '; '));
% ---- P37: git replace the HEAD blobs of olhoffSolve.m and SOURCE_MANIFEST.json
for rel = {src, 'analysis/OlhoffCurrent/SOURCE_MANIFEST.json'}
    old = strtrim(shA(A, sprintf('git --no-pager rev-parse "HEAD:%s"', rel{1})));
    new = strtrim(shA(A, sprintf('git --no-pager hash-object -w -- "%s"', rel{1})));
    shA(A, sprintf('git replace -f %s %s', old, new));
end
st1 = olhoffcurrent_finalization_gate(study, 'Verbose', false, 'RepoRoot', A);
fprintf('SR2 P37 replace refs: ok=%d G6=%d reasons=%s\n', st1.ok, st1.gates.G6, strjoin(st1.currentSource.reasons, '; '));
shA(A, 'git for-each-ref --format="%(refname)" refs/replace | xargs -n1 git update-ref -d');
% ---- P36: GIT_DIR redirect to a repository whose HEAD commits the same edit
sh(sprintf('git clone -q --shared "%s" "%s"', A, B));
copyfile(fullfile(A, src), fullfile(B, src)); copyfile(fullfile(oc, 'SOURCE_MANIFEST.json'), fullfile(B, 'analysis/OlhoffCurrent/SOURCE_MANIFEST.json'));
shA(B, sprintf('git add -- "%s" analysis/OlhoffCurrent/SOURCE_MANIFEST.json && git -c user.name=p -c user.email=p@x -c core.hooksPath=/dev/null commit -q -m edit', src));
setenv('GIT_DIR', fullfile(B, '.git'));
st2 = olhoffcurrent_finalization_gate(study, 'Verbose', false, 'RepoRoot', A);
setenv('GIT_DIR', '');
fprintf('SR2 P36 GIT_DIR redirect: ok=%d G6=%d reasons=%s\n', st2.ok, st2.gates.G6, strjoin(st2.currentSource.reasons, '; '));
end
function sh(cmd), [s, o] = system([cmd ' 2>&1']); assert(s == 0, o); end
function o = shA(d, cmd), [s, o] = system(sprintf('cd "%s" && %s 2>&1', d, cmd)); assert(s == 0, o); end
