function prov = olhoffcurrent_provenance()
%OLHOFFCURRENT_PROVENANCE  Machine-readable provenance of this implementation.
%
%   prov = OLHOFFCURRENT_PROVENANCE() reads PROVENANCE.json -- the recorded,
%   committed facts about where this code came from -- and adds what can only
%   be measured now: the live source tree hash and the current main-repository
%   git commit.
%
%   Every production result embeds this, so "what implementation produced this
%   number?" is answerable from the artifact alone.
%
%   See also OLHOFFCURRENT_CURRENTNESS, OLHOFFCURRENT_SOURCE_MANIFEST.

root = olhoffcurrent_root();
prov = jsondecode(fileread(fullfile(root, 'PROVENANCE.json')));

man = olhoffcurrent_source_manifest('Verify', false);
prov.live_source_tree_sha256 = man.treeHash;
prov.live_source_n_files     = man.nFiles;
info = olhoffcurrent_preset();
prov.implementation                    = 'analysis/OlhoffCurrent';
prov.production_preset                 = info.name;
prov.production_preset_upstream_alias  = info.upstreamPreset;
prov.production_preset_historical_aliases = info.historicalAliases;
prov.main_repo_commit                  = local_gitHead(fileparts(fileparts(root)));
end

function h = local_gitHead(repoRoot)
h = 'unknown';
try
    [st, out] = system(sprintf('git -C "%s" rev-parse HEAD 2>/dev/null', repoRoot));
    if st == 0; h = strtrim(out); end
    [st2, out2] = system(sprintf('git -C "%s" status --porcelain 2>/dev/null', repoRoot));
    if st2 == 0 && ~isempty(strtrim(out2)); h = [h '-dirty']; end
catch
end
end
