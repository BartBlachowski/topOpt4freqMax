function mig_gate_diag(outJson)
%MIG_GATE_DIAG  Read-only finalization-gate record for the studies this migration affects.
P = mig_paths();
restoredefaultpath; addpath(P.oc);
D = fullfile(P.repo, 'analysis', 'OlhoffCurrent', 'diagnostics');
S = struct();
for s = {'two_branch_controller_validation', 'upstream_253069_migration'}
    st = olhoffcurrent_finalization_gate(fullfile(D, s{1}), 'Verbose', true, 'RepoRoot', P.repo);
    S.(s{1}) = struct('ok', st.ok, 'detail', st.detail, 'gates', st.gates, 'missing', {st.missing}, ...
        'mismatched', {st.mismatched}, 'supersededSource', st.supersededSource, 'nHashed', st.nHashed);
end
fid = fopen(outJson, 'w'); fwrite(fid, jsonencode(S, 'PrettyPrint', true)); fclose(fid);
end
