function selfreview_probe(repo)
%SELFREVIEW_PROBE  Do non-canonical spellings of a production-source path escape the
%   "names production source" detection and resolve to a study-local shadow copy?
%   Uses the COMMITTED gate of repo (adf86a3) and a throwaway clone.
setenv('GIT_PAGER', 'cat');
clone = [tempname() '_selfreview'];
c = onCleanup(@() rmdir(clone, 's'));
system(sprintf('git --no-pager clone -q --shared "%s" "%s"', repo, clone));
oc = fullfile(clone, 'analysis', 'OlhoffCurrent');
addpath(oc); assert(strcmp(olhoffcurrent_root(), oc));
study = fullfile(oc, 'diagnostics', '_selfreview'); mkdir(study);
evd = fullfile(oc, 'evidence', '_selfreview', 'data'); mkdir(evd);
RHO = 1; save(fullfile(evd, 'arm.mat'), 'RHO', '-v7.3'); %#ok<NASGU>
evalc(['olhoffcurrent_evidence_declare(study, ''_selfreview'', {''arm.mat'',''required'',''x''}, ' ...
       '''EvidenceRoot'', ''analysis/OlhoffCurrent/evidence/_selfreview/data'', ''RepoRoot'', clone);']);
fake = sprintf('%% fabricated shadow\n');
shadow = fullfile(study, 'analysis', 'OlhoffCurrent', '+impl', 'architecture', 'olhoffSolve.m');
mkdir(fileparts(shadow)); fid = fopen(shadow, 'w'); fwrite(fid, fake); fclose(fid);
md = java.security.MessageDigest.getInstance('SHA-256'); md.update(uint8(fake));
h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2).', 1, []));
evl = sprintf('%s  EVIDENCE.json', olhoffcurrent_sha256_file(fullfile(study, 'EVIDENCE.json')));
V = {'analysis/OlhoffCurrent//+impl/architecture/olhoffSolve.m', ...
     'analysis/OlhoffCurrent/./+impl/architecture/olhoffSolve.m', ...
     'analysis//OlhoffCurrent/+impl/architecture/olhoffSolve.m', ...
     'analysis/./OlhoffCurrent/+impl/architecture/olhoffSolve.m', ...
     'analysis/OlhoffCurrent/+impl//architecture/olhoffSolve.m', ...
     'analysis/OlhoffCurrent/+impl/architecture/olhoffSolve.m'};
for i = 1:numel(V)
    fid = fopen(fullfile(study, 'FINAL_SHA256.txt'), 'w'); fprintf(fid, '%s\n%s  %s\n', evl, h, V{i}); fclose(fid);
    st = olhoffcurrent_finalization_gate(study, 'Verbose', false, 'RepoRoot', clone);
    fprintf('SELFREVIEW ok=%d malformed=%d sourceLines=%d  %s\n', st.ok, numel(st.malformedSourceLines), numel(st.sourceLines), V{i});
end
end
