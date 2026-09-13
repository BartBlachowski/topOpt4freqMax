function mig_declare_evidence()
%MIG_DECLARE_EVIDENCE  EVIDENCE.json for this study, hashed from the files on disk.
%   Raw solver results live git-ignored under
%   analysis/OlhoffCurrent/evidence/upstream_253069_migration (EVIDENCE_POLICY.md).
P = mig_paths();
restoredefaultpath; addpath(P.scripts); addpath(P.oc);
req = {
 'PRE_BETA.mat',  'pre-migration OlhoffCurrent (tree edbfe47) historical beta-stall preset, 160x20: full olhoffSolve result + cfg + meta'
 'PRE_EX3.mat',   'pre-migration OlhoffCurrent three-rung stage-exhaustion configuration, 160x20, diagnostics on (includes every drho)'
 'POST_BETA.mat', 'migrated OlhoffCurrent (tree 4ba9a3ae) duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered, 160x20'
 'POST_EX3.mat',  'migrated duOlhoffSimpEq4bThreeRungStageExhaustionSensitivityFiltered, 160x20, diagnostics on'
 'POST_EX4.mat',  'migrated beta-stall preset + stage exhaustion (four rungs), cap 1600, diagnostics on, 160x20'
 'POST_PED.mat',  'migrated duOlhoffPedersenAdaptiveBoxSensitivityFiltered through the production wrapper, 160x20'
 'UP_BETA.mat',   'read-only Olhoff 253069 snapshot, duOlhoffFrozenM4 with the same runtime fields, 160x20'
 'UP_EX3.mat',    'read-only Olhoff 253069 snapshot, three-rung stage-exhaustion overrides, 160x20'
 'UP_PED.mat',    'read-only Olhoff 253069 snapshot, duOlhoffAdaptivePedersen, 160x20'
 'POST_A6.mat',   'migrated OlhoffCurrent, upstream anchor A6_pdecoupled160 (legacy route), 250 outer CAP_HIT, diagnostics on'
 'POST_A7.mat',   'migrated OlhoffCurrent, upstream anchor A7_massp160 (legacy route), 250 outer CAP_HIT, diagnostics on'
 'cfg_pre.mat',   'pre-migration resolution of the 14 recorded historical configurations (81-row schema)'
 'cfg_post.mat',  'migrated resolution of the same configurations plus the Pedersen preset at nine meshes (87-row schema)'
 'cfg_up.mat',    'upstream 253069 resolution of the same configurations'
};
items = [req, repmat({'required'}, size(req,1), 1)];
items = items(:, [1 3 2]);
opt = {
 'T_preset_equivalence.mat',        'suite result'
 'T_named_pedersen.mat',            'suite result'
 'T_named_stageExhaustion.mat',     'suite result'
 'T_preset_identity.mat',           'suite result'
 'T_cost_reporting.mat',            'suite result'
 'T_pedersen_adaptive_units.mat',   'suite result'
 'T_upstream_suites.mat',           'suite result (six upstream suites against the migrated +impl)'
 'T_upstream_suites_snapshot.mat',  'suite result (same suites against the 253069 snapshot)'
 'T_gates.mat',                     'suite result (path isolation, currentness, integrity, evidence and finalization gates)'
 'harness_selftest.json',           'confbench_selftest after migration'
 'harness_selftest_BASELINE_013cc48.json', 'confbench_selftest on a pristine git archive of 013cc48'
};
items = [items; [opt(:,1), repmat({'optional'}, size(opt,1), 1), opt(:,2)]];
pre = strtrim(fileread(fullfile(P.study, 'evidence', 'preregistration_sha256.txt')));
tok = regexp(pre, '^([0-9a-f]{64})', 'tokens', 'once');
extra = struct( ...
    'preregistration_sha256', tok{1}, ...
    'upstream_commit', P.upCommit, 'upstream_parent', P.upParent, ...
    'mesh', [160 20], 'NE', 3200, ...
    'no_mesh_above_160x20_solved', true, ...
    'external_inputs_note', ['External references are identified in DATA_MANIFEST.json ' ...
        '(external_inputs) by content hash; they are not re-declared here because they ' ...
        'belong to other studies or to the upstream repository.']);
olhoffcurrent_evidence_declare(P.study, 'upstream_253069_migration', items, 'Extra', extra);
end
