function gate_probe_p10(repo)
%GATE_PROBE_P10  Is "+impl manifest-verified" independent of a local edit?
%   Edit +impl in the throwaway clone, regenerate SOURCE_MANIFEST.json, then
%   present a real historical digest for the edited path.
if nargin < 1; repo = '/private/tmp/claude-501/-Users-piotrek-Programming-topOpt4freqMax/84bbf569-dc04-4771-be4d-d9d2da2b8566/scratchpad/gateprobe_repo'; end   % throwaway shared clone at 9b30ec4
root = fullfile(repo, 'analysis', 'OlhoffCurrent');
addpath(root); assert(strcmp(olhoffcurrent_root(), root));
[s, o] = system(sprintf('git -C "%s" status --short', repo)); assert(s == 0 && isempty(strtrim(o)), o);
sandbox = fullfile(root, 'evidence', '_gate_probe'); if exist(sandbox, 'dir'); rmdir(sandbox, 's'); end
study = fullfile(sandbox, 'study'); evAbs = fullfile(sandbox, 'data'); mkdir(study); mkdir(evAbs);
RHO = rand(8, 4); %#ok<NASGU>
save(fullfile(evAbs, 'arm_probe_trajectory.mat'), 'RHO', '-v7.3');
olhoffcurrent_evidence_declare(study, '_gate_probe', {'arm_probe_trajectory.mat', 'required', 'probe'}, ...
    'EvidenceRoot', 'analysis/OlhoffCurrent/evidence/_gate_probe/data', 'RepoRoot', repo);
base = sprintf('%s  EVIDENCE.json', olhoffcurrent_sha256_file(fullfile(study, 'EVIDENCE.json')));
src = 'analysis/OlhoffCurrent/+impl/architecture/olhoffSolve.m';
[~, o] = system(sprintf('git -C "%s" cat-file blob 013cc48:%s | shasum -a 256', repo, src)); hOld = strtok(strtrim(o));
fid = fopen(fullfile(repo, src), 'a'); fwrite(fid, sprintf('\n%% LOCAL EDIT (probe)\n')); fclose(fid);
olhoffcurrent_source_manifest('Write', true);
man = olhoffcurrent_source_manifest('Verify', true);
fid = fopen(fullfile(study, 'FINAL_SHA256.txt'), 'w'); fwrite(fid, sprintf('%s\n%s  %s', base, hOld, src)); fclose(fid);
st = olhoffcurrent_finalization_gate(study, 'Verbose', false, 'RepoRoot', repo);
fprintf('P10 +impl locally edited AND manifest regenerated: man.ok=%d gate.ok=%d nSuperseded=%d  (fail-closed would be gate.ok=0)\n', ...
    man.ok, st.ok, numel(st.supersededSource));
S = struct('id', 'P10', 'desc', '+impl locally edited AND SOURCE_MANIFEST.json regenerated; real historical digest line', ...
    'expect', false, 'manifestOk', man.ok, 'got', st.ok, 'nSup', numel(st.supersededSource), 'failOpen', st.ok);
fid = fopen(fullfile(fileparts(mfilename('fullpath')), '..', 'evidence', 'gate_probe_p10_result.json'), 'w');
fwrite(fid, jsonencode(S, 'PrettyPrint', true)); fclose(fid);
system(sprintf('git -C "%s" checkout -q -- analysis/OlhoffCurrent', repo));
rmdir(sandbox, 's');
[~, o] = system(sprintf('git -C "%s" status --short', repo)); fprintf('clone status after restore: "%s"\n', strtrim(o));
end
