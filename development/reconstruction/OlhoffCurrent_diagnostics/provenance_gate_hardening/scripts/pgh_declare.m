function pgh_declare()
%PGH_DECLARE  EVIDENCE.json + FINAL_SHA256.txt for provenance_gate_hardening.
%   Every JSON in evidence/ is declared REQUIRED; every file of the study except
%   FINAL_SHA256.txt itself is hashed.  Re-run after the last artifact changes.
here = fileparts(mfilename('fullpath'));
study = fileparts(here);
oc = fileparts(fileparts(study));
repo = fileparts(fileparts(oc));
addpath(oc);
E = dir(fullfile(study, 'evidence', '*.json'));
items = cell(numel(E), 3);
for i = 1:numel(E)
    items(i, :) = {E(i).name, 'required', 'hardening audit evidence (probe table / suite results / real-study verdicts)'};
end
olhoffcurrent_evidence_declare(study, 'provenance_gate_hardening', items, ...
    'EvidenceRoot', 'analysis/OlhoffCurrent/diagnostics/provenance_gate_hardening/evidence', 'RepoRoot', repo, ...
    'Extra', struct('preregistration_sha256', strtrim(strtok(fileread(fullfile(study, 'PREREGISTRATION.sha256')))), ...
                    'migration_commit', '9b30ec45b038fb36e7cf20d57679b71cfd099fb3', ...
                    'upstream_commit', '253069262407885a8b759a9e721c4f0a7d3a397d', ...
                    'no_scientific_run', true));
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
assert(st.ok, 'pgh:declare', 'hardening study does not pass its own finalization gate');
end
