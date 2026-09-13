function P = mig_paths()
%MIG_PATHS  Fixed locations used by the upstream_253069_migration audit.
%   Every path is absolute and read-only except P.ev (the study's durable raw
%   evidence directory, git-ignored) and the study directory itself.

P = struct();
P.scripts  = fileparts(mfilename('fullpath'));
P.study    = fileparts(P.scripts);
P.oc       = fileparts(fileparts(P.study));                 % analysis/OlhoffCurrent (worktree)
P.repo     = fileparts(fileparts(P.oc));                    % worktree root
P.ev       = fullfile(P.oc, 'evidence', 'upstream_253069_migration');
P.main     = '/Users/piotrek/Programming/topOpt4freqMax';   % primary checkout, READ ONLY
P.up       = ['/private/tmp/claude-501/-Users-piotrek-Programming-topOpt4freqMax/' ...
              '0f7b4bbd-8b78-4751-ace2-9b1c6599be5c/scratchpad/mig/up253069'];  % git archive 253069, read-only
P.upCommit = '253069262407885a8b759a9e721c4f0a7d3a397d';
P.upParent = '6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7';
P.capEv    = '/Users/piotrek/Programming/Matlab/Olhoff-upstream-capabilities-evidence';

% historical evidence (read only)
P.ref.benchmarkRecords = fullfile(P.main, 'examples', 'Performance', 'conference_benchmark', ...
    'campaign_9mesh_r2', 'benchmark_records.mat');
P.ref.benchmarkResults = fullfile(P.main, 'examples', 'Performance', 'conference_benchmark', ...
    'campaign_9mesh_r2', 'benchmark_results.json');
P.ref.targetEX3 = fullfile(P.capEv, 'runs', 'case_TARGET_EX3_160.mat');
P.ref.candEX3   = fullfile(P.capEv, 'runs', 'case_CAND_EX3_160.mat');
P.ref.C160record = fullfile(P.main, 'analysis', 'OlhoffCurrent', 'diagnostics', ...
    'two_branch_controller_validation', 'runs', 'C160x20_record.json');
P.ref.C160csv = fullfile(P.main, 'analysis', 'OlhoffCurrent', 'diagnostics', ...
    'two_branch_controller_validation', 'runs', 'C160x20_iterations.csv');
P.ref.C160traj = fullfile(P.main, 'analysis', 'OlhoffCurrent', 'evidence', ...
    'two_branch_controller_validation', 'C160x20_trajectory.mat');
P.ref.S160 = fullfile(P.up, 'repro', 'results', 'S160x20', 'res.mat');
P.ref.anchorsRef = fullfile(P.up, 'architecture', 'anchors', 'reference');
P.ref.anchorCand = @(L) fullfile(P.capEv, 'runs', ['anchor_CAND_' L '.mat']);

% canonical OlhoffCurrent preset names fixed by the preregistration
P.name.beta = 'duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered';
P.name.ex3  = 'duOlhoffSimpEq4bThreeRungStageExhaustionSensitivityFiltered';
P.name.ped  = 'duOlhoffPedersenAdaptiveBoxSensitivityFiltered';
P.name.compat = 'duOlhoffFixedPenaltySensitivityFiltered';

P.timing = {'tEig', 'tGrad', 'tInner', 'tOuter'};
end
