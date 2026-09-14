function cs_evidence()
%CS_EVIDENCE  Declare the treatment's raw evidence (repository evidence policy)
%   by hashing the files on disk, then run the evidence gate on this study.
S = cs_setup();
items = {
  'C480x60_socp_trajectory.mat', 'required', 'treatment: RHO, DRHO, move, hist, cfg, meta, exh, log, socp records, final omega/lambda, status, treat'
  'C480x60_socp_state.mat',      'required', 'treatment terminal design + cfg + final spectrum'
  'C480x60_socp_diag.mat',       'required', 'treatment res.diag: per-iteration drho, predicted dlam, lam, beta, Vrot'
  'LAUNCHED.lock',               'optional', 'one-run launch lock (timestamp)'
  'checkpoint.mat',              'scratch',  'process-interruption checkpoint; not evidence'
};
extra = struct('preregistration', cs_filehash(fullfile(S.study,'AUDIT_PREREGISTRATION.md')), ...
    'amendment1', cs_filehash(fullfile(S.study,'PREREGISTRATION_AMENDMENT_1.md')), ...
    'control_trajectory_reused_read_only', 'analysis/OlhoffCurrent/evidence/three_rung_canary_preflight/C480x60_three_rung_trajectory.mat', ...
    'control_trajectory_sha256', S.expect.trajSha, 'scientific_runs', 1, ...
    'treatment_runs', 1, 'control_reruns', 0);
D = olhoffcurrent_evidence_declare(S.study, 'c480_socp_causal_run', items, 'Extra', extra); %#ok<NASGU>
st = olhoffcurrent_evidence_gate(S.study);
cs_json(fullfile(S.study,'evaluations','evidence_gate.json'), st);
disp(st);
end
