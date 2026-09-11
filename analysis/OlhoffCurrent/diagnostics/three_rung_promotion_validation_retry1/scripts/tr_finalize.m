function out = tr_finalize()
%TR_FINALIZE  Declare this study's raw evidence and write its manifests.
%
%   Writes EVIDENCE.json (via the repository's own declarer, which HASHES
%   REALITY rather than trusting the caller), DATA_MANIFEST.json and
%   METRICS.json.  FINAL_SHA256.txt is written last, by tr_seal, so it can
%   cover everything including these files.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
repo  = fileparts(fileparts(root));
addpath(root); addpath(here);
guard = olhoffcurrent_paths(); %#ok<NASGU>

STUDY = 'three_rung_promotion_validation_retry1';

% ---- EVIDENCE.json ------------------------------------------------------
items = {
 'C320x40_three_rung_trajectory.mat', 'required', ...
   'THE one authorized scientific run: 320x40, ladder [0.04 0.02 0.01], frozen A OR B'
 'runs/C320x40_three_rung_iterations.csv', 'required', ...
   'per-iteration telemetry of that run, 55 columns, cv_export format'
 'runs/C320x40_three_rung_record.json', 'required', ...
   'scalar record: status, event structure, digests'
 'evidence/oracle_C320x40_iterations_HEAD.csv', 'required', ...
   'the frozen four-rung C320 oracle telemetry, extracted from git HEAD (ff570d6e...)'
 'evidence/prefix_equivalence.json', 'required', ...
   'Part E/F machine-readable comparison against the oracle'
 'evidence/singlefactor.json', 'required', 'Part C 81-leaf configuration diff'
 'evidence/software_tests.json', 'required', 'Part C controller-mechanics suite'
 'evidence/inventory.json', 'required', 'Part A0/A1 inventory and oracle re-verification'
 'evidence/gates.json', 'required', 'Part H4 finalization gate output'
 'analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C320x40_trajectory.mat', ...
   'optional', ...
   ['the four-rung C320 oracle container.  OPTIONAL by deliberate classification: ' ...
    'its container digest is host-specific (Class A), while the scientific content ' ...
    'this study consumes is verified against git-committed digests -- final rho ' ...
    '0348b288..., RHO[:,1:352] b8c0f18d..., omega(1:2,1:352) fd636083...']
};
D = olhoffcurrent_evidence_declare(study, STUDY, items, 'RepoRoot', repo, ...
      'Extra', struct( ...
        'implTree', olhoffcurrent_source_manifest('Verify',false).treeHash, ...
        'oracle_study', 'two_branch_controller_validation arm C', ...
        'oracle_doc_sha256', 'd09579b80c0d187b3800e71e22538a1c482e60d79505d44457ffc9d57e8b82d9', ...
        'scientific_runs', 1, ...
        'candidate_cfg_hash', 'afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab', ...
        'four_rung_cfg_hash', '2359a1112fcec9edd0971aae9a89508cd703ea3899770e7e85c850138c2550f4'));
fprintf('[tr_finalize] EVIDENCE.json: %d artifacts\n', numel(D.artifacts));

% ---- METRICS.json -------------------------------------------------------
pre = jsondecode(fileread(fullfile(study,'evidence','prefix_equivalence.json')));
rec = jsondecode(fileread(fullfile(study,'runs','C320x40_three_rung_record.json')));
sf  = jsondecode(fileread(fullfile(study,'evidence','singlefactor.json')));
sw  = jsondecode(fileread(fullfile(study,'evidence','software_tests.json')));

M = struct();
M.study = STUDY;
M.attempt = 2;
M.previous_attempt = struct('dir','three_rung_promotion_validation', ...
    'verdict','THREE_RUNG_PROMOTION_PROVENANCE_FAIL','scientific_runs',0, ...
    'preserved_unchanged', true);
M.scientific_runs = 1;
M.run = struct('tag', rec.tag, 'mesh', rec.mesh(:).', 'NE', rec.NE, ...
    'status', rec.status, 'nOuter', rec.nOuter, 'cap', rec.cap, ...
    'levels', rec.levels(:).', 'tol', rec.tol, ...
    'innerTotal', rec.innerTotal, 'innerMax', rec.innerMax, ...
    'innerNonConv', rec.innerNonConv, 'stage_final', rec.stage_final, ...
    'move_final', rec.move_final, 'wall_s', rec.wall_s, ...
    'cfgHash', rec.cfgHash, 'implTree', rec.implTree, 'rho_sha256', rec.rho_sha256);
M.anchors = pre.anchors;
M.events = struct('S1', pre.S1, 'S2', pre.S2, 'S3', pre.S3, 'ok', pre.eventsOk);
M.terminal = pre.terminal;
M.cost = pre.cost;
M.single_factor = struct('verdict', sf.verdict, 'nSchemaRows', sf.nSchemaRows, ...
    'cfgHash3', sf.cfgHash3, 'cfgHash4', sf.cfgHash4);
M.software = struct('verdict', sw.verdict, 'nTests', sw.nTests, 'nPass', sw.nPass);
M.verdicts = struct( ...
  'dependency_provenance', 'DEPENDENCY_SPECIFIC_SCIENTIFIC_PROVENANCE_PASS', ...
  'single_factor',         sf.verdict, ...
  'software',              sw.verdict, ...
  'prefix',                pre.prefixVerdict, ...
  'termination',           pre.terminationVerdict, ...
  'policy',                local_policy(pre, sf, sw, rec), ...
  'promotion_provenance',  'PROMOTION_PROVENANCE_BLOCKED', ...
  'promotion',             'PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED', ...
  'campaign',              'NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED');
M.promotion_blockers = { ...
  'H1 C240x30_trajectory.mat REMOTE_REQUIRED_EVIDENCE_NOT_LOCAL (blocking, not closable on this host)'; ...
  'H2 two_branch_controller_validation/FINAL_SHA256.txt stale, repair prepared not applied'; ...
  'H4 finalization G1-G5 FAIL on all three load-bearing studies'; ...
  'H5 test_finalization_gate 4 failures (cases H and I)'};
M.nine_mesh_runs_executed = 0;
M.outcome_driven_repair = false;
fid = fopen(fullfile(study,'METRICS.json'),'w'); c1 = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n', jsonencode(M,'PrettyPrint',true));
fprintf('[tr_finalize] METRICS.json written\n');
clear c1

% ---- DATA_MANIFEST.json -------------------------------------------------
DM = struct('manifest_schema','olhoff_current_data_manifest/1', ...
    'study', STUDY, ...
    'generated', char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ssXXX','TimeZone','local')), ...
    'repo_head', local_head(repo), ...
    'implTree', M.run.implTree, ...
    'scientific_runs', 1);
files = local_listStudy(study);
ent = struct('path',{},'bytes',{},'sha256',{},'kind',{});
for k = 1:numel(files)
    f = files{k};
    d = dir(f);
    rel = strrep(f(numel(study)+2:end), filesep, '/');
    ent(end+1) = struct('path', rel, 'bytes', d.bytes, ...
        'sha256', olhoffcurrent_sha256_file(f), 'kind', local_kind(rel)); %#ok<AGROW>
end
% the git-ignored raw trajectory, recorded by its durable repo-relative path
traj = fullfile(root,'evidence',STUDY,'C320x40_three_rung_trajectory.mat');
if exist(traj,'file') == 2
    d = dir(traj);
    ent(end+1) = struct('path', strrep(traj,[repo filesep],''), 'bytes', d.bytes, ...
        'sha256', olhoffcurrent_sha256_file(traj), 'kind', 'raw_trajectory_gitignored');
end
DM.files = ent;
DM.n_files = numel(ent);
fid = fopen(fullfile(study,'DATA_MANIFEST.json'),'w'); c2 = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n', jsonencode(DM,'PrettyPrint',true));
fprintf('[tr_finalize] DATA_MANIFEST.json: %d files\n', numel(ent));

out = struct('evidence', numel(D.artifacts), 'manifest', numel(ent), 'metrics', M);
end

function v = local_policy(pre, sf, sw, rec)
ok = pre.prefixPass && pre.terminationPass && sf.pass && sw.pass && ...
     rec.innerNonConv == 0 && ~strcmp(rec.status,'CAP_HIT');
if ok, v = 'THREE_RUNG_PRODUCTION_POLICY_VALIDATED';
else,  v = 'THREE_RUNG_PRODUCTION_POLICY_PARTIALLY_VALIDATED'; end
end

function h = local_head(repo)
[s,o] = system(sprintf('git -C "%s" rev-parse HEAD', repo));
if s == 0, h = strtrim(o); else, h = '<unavailable>'; end
end

function k = local_kind(rel)
if startsWith(rel,'scripts/'),  k = 'script';
elseif startsWith(rel,'runs/'), k = 'run_output';
elseif startsWith(rel,'figures/'), k = 'figure';
elseif startsWith(rel,'evidence/'), k = 'evidence';
elseif endsWith(rel,'.json'),   k = 'manifest';
else,                           k = 'document'; end
end

function L = local_listStudy(study)
L = {};
d = dir(fullfile(study,'**','*'));
for k = 1:numel(d)
    if d(k).isdir, continue; end
    if strcmp(d(k).name,'FINAL_SHA256.txt'), continue; end   % written afterwards
    if strcmp(d(k).name,'.DS_Store'), continue; end
    L{end+1} = fullfile(d(k).folder, d(k).name); %#ok<AGROW>
end
L = sort(L);
end
