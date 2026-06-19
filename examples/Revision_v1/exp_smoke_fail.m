function r = exp_smoke_fail(outDir)
%EXP_SMOKE_FAIL  Intentionally failing experiment for Gate I1 verification.
%
%   r = exp_smoke_fail(outDir)
%
%   Does NOT run any solver.  Always returns a fully schema-valid result
%   with termination.capped=true and success=false.
%
%   Purpose: verify that run_all_revision_experiments detects a capped run
%   and reports the exact failed acceptance condition.
%
%   Gate I1: an intentionally failing smoke experiment must make the master
%   run fail and identify the exact failed acceptance condition.

if nargin < 1 || isempty(outDir)
    outDir = fileparts(mfilename('fullpath'));
end

scriptDir = fileparts(mfilename('fullpath'));
localEnsurePaths(scriptDir);

fprintf('  [smoke] Building deliberately capped result (no solver run)...\n');

r = make_experiment_result();

n = 200;  % iteration count == cap

r.success               = false;
r.termination.reason    = 'iteration_cap';
r.termination.capped    = true;
r.termination.message   = sprintf( ...
    'smoke test: deliberately capped at %d iterations to verify fail-loud detection', n);

r.iterations.count = n;
r.iterations.cap   = n;

r.convergence.final_design_change = 5.0e-3;   % above any reasonable tol
r.convergence.constraint_residual = 0.0;

r.histories.objective   = linspace(1.00, 1.11, n)';
r.histories.frequency   = linspace(100,  110,  n)';
r.histories.feasibility = zeros(n, 1);
r.histories.grayness    = linspace(0.30, 0.16, n)';

r.mode_tracking.tracked_mode_indices = ones(n, 1);
r.mode_tracking.mac_history          = 0.87 * ones(n, 1);

r.timing.init_s          = 0.05;
r.timing.optim_loop_s    = 0.10;
r.timing.postproc_s      = 0.01;
r.timing.total_s         = 0.16;
r.timing.per_iter_mean_s = r.timing.optim_loop_s / n;
r.timing.peak_memory_MB  = 0.0;

r.provenance.config_hash    = 'smoke001';
r.provenance.commit         = localGitCommit();
r.provenance.random_seed    = NaN;
r.provenance.matlab_version = version();
r.provenance.platform       = computer();
r.provenance.timestamp_utc  = '';

% Write the mat artifact (so artifact checks see it)
if ~exist(outDir, 'dir'), mkdir(outDir); end
r.artifacts.mat_file   = fullfile(outDir, 'smoke_fail_result.mat');
r.artifacts.diary_file = fullfile(outDir, 'smoke_fail_diary.txt');
save(r.artifacts.mat_file, 'r');
fprintf('  [smoke] Result written to: %s\n', r.artifacts.mat_file);
end

% =========================================================================
function localEnsurePaths(scriptDir)
repoRoot  = fileparts(fileparts(scriptDir));
toolsDir  = fullfile(repoRoot, 'tools', 'Matlab');
schemaDir = fullfile(repoRoot, 'scripts', 'revision_v1');
if exist(toolsDir,  'dir') == 7, addpath(toolsDir);  end
if exist(schemaDir, 'dir') == 7, addpath(schemaDir); end
addpath(genpath(fullfile(repoRoot, 'analysis')));
end

% =========================================================================
function sha = localGitCommit()
sha = 'unknown';
try
    [s, out] = system('git rev-parse HEAD 2>/dev/null');
    if s == 0 && ~isempty(strtrim(out)), sha = strtrim(out); end
catch, end
end
