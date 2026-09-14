function pmg_declare(varargin)
%PMG_DECLARE  EVIDENCE.json + FINAL_SHA256.txt for postmerge_campaign_gate (attempt 2).
%   Declares every evidence/*.json (study-local, committed-size) as REQUIRED, plus
%   every .mat of the git-ignored durable root analysis/OlhoffCurrent/evidence/
%   postmerge_campaign_gate (the two 160x20 anchor results) when present.  Hashes
%   every file of the study -- including the preserved attempt1_blocked/ record --
%   except the top-level FINAL_SHA256.txt itself.  Re-run after artifacts change.
here = fileparts(mfilename('fullpath'));
study = fileparts(here);
oc = fileparts(fileparts(study));
repo = fileparts(fileparts(oc));
addpath(oc);
evRoot = 'analysis/OlhoffCurrent/evidence/postmerge_campaign_gate';
items = {};
E = dir(fullfile(study, 'evidence', '*.json'));
for i = 1:numel(E)
    items(end+1, :) = {['evidence/' E(i).name], 'required', 'post-merge gate evidence (JSON)'}; %#ok<AGROW>
end
M = dir(fullfile(repo, evRoot, '*.mat'));
for i = 1:numel(M)
    items(end+1, :) = {M(i).name, 'required', '160x20 anchor: full olhoffSolve result + cfg + meta'}; %#ok<AGROW>
end
% study-local JSON is resolved through the study directory; .mat through the durable root
olhoffcurrent_evidence_declare(study, 'postmerge_campaign_gate', items, ...
    'EvidenceRoot', evRoot, 'RepoRoot', repo, ...
    'Extra', struct('attempt', 2, 'attempt1', 'attempt1_blocked/ (OLHOFF_NINE_MESH_CAMPAIGN_BLOCKED, preserved byte-identical)', ...
                    'no_mesh_above_160x20_solved', true));
L = dir(fullfile(study, '**', '*')); L = L(~[L.isdir]);
lines = {};
for i = 1:numel(L)
    fp = fullfile(L(i).folder, L(i).name);
    rel = strrep(fp(numel(study)+2:end), filesep, '/');
    if strcmp(rel, 'FINAL_SHA256.txt') || startsWith(L(i).name, '.'); continue; end
    lines{end+1} = sprintf('%s  %s', olhoffcurrent_sha256_file(fp), rel); %#ok<AGROW>
end
lines = sort(lines);
fid = fopen(fullfile(study, 'FINAL_SHA256.txt'), 'w'); fprintf(fid, '%s\n', lines{:}); fclose(fid);
st = olhoffcurrent_finalization_gate(study, 'Verbose', true, 'RepoRoot', repo);
assert(st.ok, 'pmg:declare', 'postmerge_campaign_gate does not pass its own finalization gate');
end
