function st = tr3_finalize()
%TR3_FINALIZE  Fail-closed retention for the three-rung architecture audit.
%
%   Declares the raw evidence this ZERO-RUN study depends on, writes
%   DATA_MANIFEST.json and a self-verifying FINAL_SHA256.txt, re-records
%   provenance, and runs OLHOFFCURRENT_FINALIZATION_GATE.  Read-only with
%   respect to +impl/ and to every trajectory.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
repo  = fileparts(fileparts(root));
addpath(root); addpath(here);

M = jsondecode(fileread(fullfile(study,'METRICS.json')));

% ---- EVIDENCE.json: the raw trajectories, declared by immutable path -----
items = {
 'analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C160x20_trajectory.mat', 'required', ...
   '160x20 four-rung causal-controller trajectory: source of the frozen replay and of S1/S2/S3/F densities.'
 'analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C320x40_trajectory.mat', 'required', ...
   '320x40 four-rung causal-controller trajectory (CAP_HIT @1600).'
 'analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C400x50_trajectory.mat', 'required', ...
   '400x50 four-rung causal-controller trajectory.'
 'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C160x20_iterations.csv', 'required', ...
   '160x20 per-iteration telemetry incl. the ex* controller trace the solver acted on.'
 'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C320x40_iterations.csv', 'required', ...
   '320x40 per-iteration telemetry.'
 'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C400x50_iterations.csv', 'required', ...
   '400x50 per-iteration telemetry.'
 'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/evidence/baselines.json', 'required', ...
   'The three frozen production baselines P.'
 'analysis/OlhoffCurrent/evidence/move_activity_400/P400_400x50_trajectory.mat', 'optional', ...
   'Production 400x50 density field -- the only surviving P topology; 160x20 and 320x40 are lost.'
 'analysis/OlhoffCurrent/diagnostics/move_ladder_necessity/METRICS.json', 'optional', ...
   'Prior audit: S1 extraction and rung boundaries, used for cross-checking only.'
 'analysis/OlhoffCurrent/diagnostics/two_rung_architecture/METRICS.json', 'required', ...
   'Two-rung audit: the recorded S1/S2 events and the blocked +0.1139%% residual this study splits.'
 'analysis/OlhoffCurrent/diagnostics/two_rung_architecture/evidence/event_verification.json', 'required', ...
   'Two-rung audit: recorded kE1/kE2, reproduced here as a cross-check.'};

extra = struct( ...
  'scientific_runs_executed', 0, ...
  'note', ['This study executed zero optimization runs.  It declares pre-existing ' ...
           'durable trajectories by immutable path and SHA-256 rather than copying them.'], ...
  'preregistration_sha256', M.preregistration_sha256, ...
  'impl_tree_sha256', M.impl_tree_sha256, ...
  'architecture_verdict', M.verdicts.architecture, ...
  'next_step_verdict', M.verdicts.next_step, ...
  'counterfactual_verdict', M.verdicts.counterfactual, ...
  'evidence_gaps', {{ ...
     ['240x30: no causal trajectory exists (no runs/ dir; that arm was fixed-move and ' ...
      'never ran a 0.02 or 0.01 stage) -- UNAVAILABLE_FOR_THREE_RUNG_CAUSAL_COMPARISON, not inferred'], ...
     '160x20 and 320x40 production density fields: raw .mat lost, rho_available=false in baselines.json', ...
     'wall-clock time: s/inner drifts 4.1x-5.7x within each run; down-weighted, never decisive', ...
     'two_branch_maturity_240 finalization gate FAILS (pre-existing); only its hash-valid PREREGISTRATION.md is used'}});

olhoffcurrent_evidence_declare(study, 'three_rung_architecture', items, ...
    'EvidenceRoot', 'analysis/OlhoffCurrent/evidence/two_branch_controller_validation', ...
    'RepoRoot', repo, 'Extra', extra);

% ---- FINAL_SHA256.txt, pass 1: so the study's OWN finalization gate can pass
% before tr3_provenance('final') evaluates it.  Without this the final record
% would report a failure created purely by write ordering.
local_writeHashes(study);

% ---- provenance at end -------------------------------------------------
% Now G1-G5 are all satisfiable for this study, so test_finalization_gate sees
% no new deficient study and the recorded testsPass is a real pass.
tr3_provenance('final');

% ---- DATA_MANIFEST.json: every artifact THIS study produced -------------
% Everything except FINAL_SHA256.txt, which is the outer hash file and is
% rewritten last, over the manifest itself.
own = local_ownFiles(study);
own = own(~strcmp(own,'FINAL_SHA256.txt'));
arts = {};
for i = 1:numel(own)
    rel = own{i};
    f = fullfile(study, rel);
    d = dir(f);
    arts{end+1} = struct('path', ['analysis/OlhoffCurrent/diagnostics/three_rung_architecture/' rel], ...
        'role', local_role(rel), 'bytes', d(1).bytes, ...
        'sha256', olhoffcurrent_sha256_file(f), 'tracked', true); %#ok<AGROW>
end
DM = struct('schema','olhoff_three_rung_architecture_manifest/1', ...
    'study','three_rung_architecture', ...
    'generated', char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z''')), ...
    'scientific_runs_executed', 0, ...
    'note', ['No raw scientific data was produced.  The study depends on ' ...
             'pre-existing durable trajectories declared in EVIDENCE.json and ' ...
             'verified by olhoffcurrent_evidence_gate.'], ...
    'verdicts', struct('counterfactual', M.verdicts.counterfactual, ...
                       'architecture', M.verdicts.architecture, ...
                       'next_step', M.verdicts.next_step, ...
                       'production', 'PRODUCTION_CONTROLLER_NOT_CHANGED', ...
                       'campaign', 'NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED'), ...
    'preregistration_sha256', M.preregistration_sha256, ...
    'artifacts', {arts});
fid = fopen(fullfile(study,'DATA_MANIFEST.json'),'w');
fwrite(fid, jsonencode(DM,'PrettyPrint',true)); fclose(fid);
fprintf('[manifest] DATA_MANIFEST.json  (%d artifacts)\n', numel(arts));

% ---- FINAL_SHA256.txt, pass 2: now covering the manifest and the final
% provenance record as well.  Self-verifying over everything but itself.
n = local_writeHashes(study);
fprintf('[hashes]  FINAL_SHA256.txt  (%d files)\n', n);

% ---- the gate ------------------------------------------------------------
st = olhoffcurrent_finalization_gate(study, 'RepoRoot', repo);
end

function n = local_writeHashes(study)
%LOCAL_WRITEHASHES  Write FINAL_SHA256.txt over every file in the study except
%   the hash file itself, so that the hash file is self-verifying (gate G4).
own = local_ownFiles(study);
own = own(~strcmp(own,'FINAL_SHA256.txt'));
lines = {};
lines{end+1} = '# FINAL_SHA256 -- three_rung_architecture';
lines{end+1} = sprintf('# generated %s', ...
    char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z''')));
lines{end+1} = '# paths are relative to analysis/OlhoffCurrent/diagnostics/three_rung_architecture';
lines{end+1} = '# every path listed here must resolve and hash as recorded (gate G4)';
lines{end+1} = '';
for i = 1:numel(own)
    lines{end+1} = sprintf('%s  %s', ...
        olhoffcurrent_sha256_file(fullfile(study,own{i})), own{i}); %#ok<AGROW>
end
fid = fopen(fullfile(study,'FINAL_SHA256.txt'),'w');
fprintf(fid,'%s\n',lines{:}); fclose(fid);
n = numel(own);
end

function f = local_ownFiles(study)
d = dir(fullfile(study,'**','*'));
f = {};
for i = 1:numel(d)
    if d(i).isdir; continue; end
    if strcmp(d(i).name,'.DS_Store'); continue; end
    rel = strrep(erase(fullfile(d(i).folder,d(i).name), [study filesep]), filesep, '/');
    if contains(rel,'__pycache__'); continue; end
    f{end+1} = rel; %#ok<AGROW>
end
f = sort(f);
end

function r = local_role(rel)
if startsWith(rel,'figures/');       r = 'figure';
elseif startsWith(rel,'scripts/');   r = 'analysis code';
elseif startsWith(rel,'evidence/');  r = 'evidence';
elseif endsWith(rel,'.json');        r = 'metrics';
else;                                r = 'report';
end
end
