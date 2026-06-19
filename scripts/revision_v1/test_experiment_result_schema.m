function pass = test_experiment_result_schema()
%TEST_EXPERIMENT_RESULT_SCHEMA  Verify the three canonical result states.
%
%   pass = test_experiment_result_schema()
%
%   Does NOT run any solver or write any file.  Builds one result struct
%   for each scenario, calls check_experiment_result on it, then asserts
%   that the discriminating fields correctly identify the outcome.
%
%   Case 1: converged success          -- 80-iteration clamped-beam run
%   Case 2: capped failure             -- hit 200-iteration cap, still oscillating
%   Case 3: mode-invalid failure       -- MAC dropped below 0.80 at iteration 52
%
%   USAGE
%   -----
%   cd scripts/revision_v1
%   pass = test_experiment_result_schema()

localEnsurePaths();

pass = true;
fprintf('\n=== test_experiment_result_schema ===\n\n');

% ---- Case 1: converged success ------------------------------------------
r1 = localMakeSuccess();
[ok1, iss1] = check_experiment_result(r1);

fprintf('Case 1: converged success\n');
localPrintSummary(r1);
localPrintCheckResult(ok1, iss1);

pass = localCheck(pass, ok1,                           'Case1 struct is valid');
pass = localCheck(pass, r1.success,                    'Case1 r.success=true');
pass = localCheck(pass, r1.termination.converged,      'Case1 converged=true');
pass = localCheck(pass, ~r1.termination.capped,        'Case1 capped=false');
pass = localCheck(pass, ~r1.termination.mode_lost,     'Case1 mode_lost=false');
pass = localCheck(pass, r1.iterations.count < r1.iterations.cap, ...
                                                       'Case1 count < cap');
pass = localCheck(pass, r1.mode_tracking.mac_history(end) >= 0.80, ...
                                                       'Case1 final MAC >= 0.80');

% ---- Case 2: capped failure ---------------------------------------------
r2 = localMakeCapped();
[ok2, iss2] = check_experiment_result(r2);

fprintf('Case 2: capped failure\n');
localPrintSummary(r2);
localPrintCheckResult(ok2, iss2);

pass = localCheck(pass, ok2,                           'Case2 struct is valid');
pass = localCheck(pass, ~r2.success,                   'Case2 r.success=false');
pass = localCheck(pass, r2.termination.capped,         'Case2 capped=true');
pass = localCheck(pass, ~r2.termination.converged,     'Case2 converged=false');
pass = localCheck(pass, r2.iterations.count == r2.iterations.cap, ...
                                                       'Case2 count==cap');
pass = localCheck(pass, r2.convergence.final_design_change > 1e-3, ...
                                                       'Case2 design change above tol');

% ---- Case 3: mode-invalid failure ---------------------------------------
r3 = localMakeModeInvalid();
[ok3, iss3] = check_experiment_result(r3);

fprintf('Case 3: mode-invalid failure\n');
localPrintSummary(r3);
localPrintCheckResult(ok3, iss3);

pass = localCheck(pass, ok3,                           'Case3 struct is valid');
pass = localCheck(pass, ~r3.success,                   'Case3 r.success=false');
pass = localCheck(pass, r3.termination.mode_lost,      'Case3 mode_lost=true');
pass = localCheck(pass, ~r3.termination.capped,        'Case3 capped=false');
pass = localCheck(pass, r3.iterations.count < r3.iterations.cap, ...
                                                       'Case3 stopped before cap');
pass = localCheck(pass, r3.mode_tracking.mac_history(end) < 0.80, ...
                                                       'Case3 final MAC below threshold');

% ---- summary ------------------------------------------------------------
fprintf('\n');
if pass
    fprintf('=== ALL ASSERTIONS PASSED ===\n\n');
else
    fprintf('=== ONE OR MORE ASSERTIONS FAILED ===\n\n');
end
end

% =========================================================================
% Case 1: converged success after 80 iterations
% =========================================================================
function r = localMakeSuccess()
r = make_experiment_result();
n = 80;

r.success               = true;
r.termination.reason    = 'converged';
r.termination.converged = true;
r.termination.message   = sprintf( ...
    'design change %.2e < tol 1e-3; constraint residual 0', 8.4e-4);

r.iterations.count = n;
r.iterations.cap   = 200;

r.convergence.final_design_change = 8.4e-4;
r.convergence.constraint_residual = 0.0;

r.histories.objective   = linspace(1.00, 1.35, n)';
r.histories.frequency   = linspace(100,  145,  n)';
r.histories.feasibility = zeros(n, 1);
r.histories.grayness    = linspace(0.25, 0.04, n)';

r.mode_tracking.tracked_mode_indices = ones(n, 1);
r.mode_tracking.mac_history          = linspace(1.00, 0.92, n)';

r.timing.init_s          = 3.2;
r.timing.optim_loop_s    = 187.4;
r.timing.postproc_s      = 2.1;
r.timing.total_s         = 192.7;
r.timing.per_iter_mean_s = r.timing.optim_loop_s / n;
r.timing.peak_memory_MB  = 412.0;

r.provenance.config_hash    = 'ab12cd34';
r.provenance.commit         = 'abc1234';
r.provenance.random_seed    = NaN;
r.provenance.matlab_version = 'R2025b (test)';
r.provenance.platform       = 'MACI64';
r.provenance.timestamp_utc  = '2026-06-19T10:00:00Z';

r.artifacts.mat_file      = 'results/exp2_alpha1_result.mat';
r.artifacts.diary_file    = 'results/exp2_alpha1_diary.txt';
r.artifacts.topology_png  = 'results/exp2_alpha1_topo.png';
end

% =========================================================================
% Case 2: capped failure -- hit 200-iteration cap while still oscillating
% =========================================================================
function r = localMakeCapped()
r = make_experiment_result();
n = 200;  % count equals cap

r.success               = false;
r.termination.reason    = 'iteration_cap';
r.termination.capped    = true;
r.termination.message   = sprintf( ...
    'reached iteration cap %d; design change %.2e still above tol 1e-3', n, 4.1e-3);

r.iterations.count = n;
r.iterations.cap   = n;  % count == cap: cap was hit

r.convergence.final_design_change = 4.1e-3;
r.convergence.constraint_residual = 3.0e-4;

plateau  = linspace(1.00, 1.28, 150)';
wobble   = 1.28 + 0.02 * sin(linspace(0, 6*pi, 50)');
r.histories.objective   = [plateau; wobble];
r.histories.frequency   = linspace(100, 135, n)';
r.histories.feasibility = 1e-3 * abs(sin(linspace(0, 3*pi, n)'));
r.histories.grayness    = linspace(0.25, 0.08, n)';

r.mode_tracking.tracked_mode_indices = ones(n, 1);
r.mode_tracking.mac_history          = 0.88 + 0.04 * sin(linspace(0, 4*pi, n)');

r.timing.init_s          = 3.1;
r.timing.optim_loop_s    = 467.3;
r.timing.postproc_s      = 2.0;
r.timing.total_s         = 472.4;
r.timing.per_iter_mean_s = r.timing.optim_loop_s / n;
r.timing.peak_memory_MB  = 415.0;

r.provenance.config_hash    = 'ef56ab78';
r.provenance.commit         = 'abc1234';
r.provenance.random_seed    = NaN;
r.provenance.matlab_version = 'R2025b (test)';
r.provenance.platform       = 'MACI64';
r.provenance.timestamp_utc  = '2026-06-19T10:10:00Z';

r.artifacts.mat_file   = 'results/exp2_alpha075_result.mat';
r.artifacts.diary_file = 'results/exp2_alpha075_diary.txt';
end

% =========================================================================
% Case 3: mode-invalid failure -- MAC dropped below 0.80 at iteration 52
% =========================================================================
function r = localMakeModeInvalid()
r = make_experiment_result();
n = 52;  % stopped early when MAC fell below threshold

r.success               = false;
r.termination.reason    = 'mode_invalid';
r.termination.mode_lost = true;
r.termination.message   = sprintf( ...
    'MAC = %.2f at iteration %d dropped below threshold 0.80; mode tracking lost', ...
    0.61, n);

r.iterations.count = n;
r.iterations.cap   = 200;  % count < cap: run aborted early

r.convergence.final_design_change = 1.2e-2;  % not converged; aborted
r.convergence.constraint_residual = 0.0;

r.histories.objective   = linspace(1.00, 1.15, n)';
r.histories.frequency   = linspace(100,  122,  n)';
r.histories.feasibility = zeros(n, 1);
r.histories.grayness    = linspace(0.25, 0.14, n)';

% MAC traces a slow decline then collapses; tracked index switches to mode 2
mac_trace = [linspace(0.98, 0.85, 45)'; linspace(0.82, 0.61, 7)'];
r.mode_tracking.tracked_mode_indices = [ones(45, 1); 2 * ones(7, 1)];
r.mode_tracking.mac_history          = mac_trace;

r.timing.init_s          = 3.0;
r.timing.optim_loop_s    = 122.1;
r.timing.postproc_s      = 1.8;
r.timing.total_s         = 126.9;
r.timing.per_iter_mean_s = r.timing.optim_loop_s / n;
r.timing.peak_memory_MB  = 410.0;

r.provenance.config_hash    = '9900aabb';
r.provenance.commit         = 'abc1234';
r.provenance.random_seed    = NaN;
r.provenance.matlab_version = 'R2025b (test)';
r.provenance.platform       = 'MACI64';
r.provenance.timestamp_utc  = '2026-06-19T10:20:00Z';

r.artifacts.mat_file   = 'results/exp2_alpha050_result.mat';
r.artifacts.diary_file = 'results/exp2_alpha050_diary.txt';
end

% =========================================================================
% Helpers
% =========================================================================

function localPrintSummary(r)
fprintf('  success=%d  reason=''%s''  count=%d  cap=%d\n', ...
    r.success, r.termination.reason, r.iterations.count, r.iterations.cap);
fprintf('  dX=%.2e  cRes=%.2e  obj_len=%d  MAC_final=%.3f\n', ...
    r.convergence.final_design_change, r.convergence.constraint_residual, ...
    numel(r.histories.objective), r.mode_tracking.mac_history(end));
fprintf('  total=%.1fs  peak=%.0fMB\n', r.timing.total_s, r.timing.peak_memory_MB);
fprintf('  message: %s\n', r.termination.message);
end

function localPrintCheckResult(ok, issues)
if ok
    fprintf('  check_experiment_result => ok\n\n');
else
    fprintf('  check_experiment_result => INVALID\n');
    for k = 1:numel(issues)
        fprintf('    [!] %s\n', issues{k});
    end
    fprintf('\n');
end
end

function pass = localCheck(pass, cond, label)
if cond
    fprintf('  [PASS] %s\n', label);
else
    fprintf('  [FAIL] %s\n', label);
    pass = false;
end
end

function localEnsurePaths()
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(thisDir));
addpath(thisDir);
toolsDir = fullfile(repoRoot, 'tools', 'Matlab');
if exist(toolsDir, 'dir') == 7, addpath(toolsDir); end
end
