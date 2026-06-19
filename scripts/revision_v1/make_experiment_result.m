function r = make_experiment_result()
%MAKE_EXPERIMENT_RESULT  Allocate a blank experiment result struct.
%
%   r = make_experiment_result()
%
%   Returns a struct with every required schema field pre-initialised to
%   NaN / false / '' / [].  Callers fill fields during the experiment then
%   call check_experiment_result(r) before saving.
%
%   FIELD GROUPS
%   ------------
%   r.success                         logical  overall pass/fail
%   r.termination                     struct   reason string + flag bits
%   r.iterations.count / .cap         double   actual vs declared maximum
%   r.convergence.*                   double   final design change, feasibility
%   r.histories.*                     arrays   per-iteration scalars/vectors
%   r.mode_tracking.*                 arrays   tracked mode index + MAC
%   r.timing.*                        double   independently measured times (s)
%   r.provenance.*                    strings  config hash, commit, platform
%   r.artifacts.*                     strings  paths to saved output files
%
%   USAGE PATTERN
%   -------------
%   r = make_experiment_result();
%   r = fill_result_provenance(r, cfg);   % optional: fill system metadata
%   % ... run experiment, populate fields ...
%   [ok, issues] = check_experiment_result(r);
%   if ~ok, error('result incomplete: %s', strjoin(issues, '; ')); end
%   save(matFile, 'r');
%
%   See also FILL_RESULT_PROVENANCE, CHECK_EXPERIMENT_RESULT.

r = struct();

% ---- top-level -----------------------------------------------------------
r.success = false;

% ---- termination ---------------------------------------------------------
% reason must be set to one of:
%   'converged'      design change and feasibility criteria satisfied
%   'iteration_cap'  loop ended at cap without meeting convergence criteria
%   'mode_invalid'   MAC dropped below declared threshold; mode tracking lost
%   'exception'      a MATLAB exception was caught
%   'unknown'        (default) -- must not remain in an accepted result
r.termination           = struct();
r.termination.reason    = 'unknown';
r.termination.converged = false;
r.termination.capped    = false;   % true iff iterations.count == iterations.cap
r.termination.mode_lost = false;   % true iff MAC < declared threshold
r.termination.exception = false;   % true iff a caught exception caused termination
r.termination.message   = '';      % human-readable detail (values, thresholds)

% ---- iterations ----------------------------------------------------------
r.iterations       = struct();
r.iterations.count = NaN;   % actual iteration count at termination
r.iterations.cap   = NaN;   % declared maximum iterations (must match config)

% ---- convergence ---------------------------------------------------------
r.convergence                     = struct();
r.convergence.final_design_change = NaN;   % ||x_k - x_{k-1}||_inf at termination
r.convergence.constraint_residual = NaN;   % max(0, g(x)) at final iterate

% ---- per-iteration histories (row per iteration) -------------------------
r.histories             = struct();
r.histories.objective   = [];   % [n_iter x 1]        objective value
r.histories.frequency   = [];   % [n_iter x n_modes]  tracked frequencies (rad/s)
r.histories.feasibility = [];   % [n_iter x 1]        constraint residual
r.histories.grayness    = [];   % [n_iter x 1]        mean(4*x.*(1-x))

% ---- mode tracking -------------------------------------------------------
r.mode_tracking                      = struct();
r.mode_tracking.tracked_mode_indices = [];  % [n_iter x 1]  integer mode index
r.mode_tracking.mac_history          = [];  % [n_iter x 1]  MAC at each iteration

% ---- timing (each field independently measured; do NOT derive init_s -----
% by subtracting a one-iteration probe from loop time)
r.timing                 = struct();
r.timing.init_s          = NaN;   % K0/M0 assembly + reference eigenproblem
r.timing.optim_loop_s    = NaN;   % FEA + sensitivity + design update, total
r.timing.postproc_s      = NaN;   % figures, CSV/MAT export
r.timing.total_s         = NaN;   % wall-clock from entry to final save
r.timing.per_iter_mean_s = NaN;   % optim_loop_s / iterations.count
r.timing.peak_memory_MB  = NaN;   % peak RSS or MATLAB whos total

% ---- provenance (filled by fill_result_provenance) -----------------------
r.provenance                = struct();
r.provenance.config_hash    = '';    % hex fingerprint of serialised config
r.provenance.commit         = '';    % git rev-parse HEAD (or 'unknown')
r.provenance.random_seed    = NaN;   % integer seed used; NaN if deterministic
r.provenance.matlab_version = '';    % version()
r.provenance.platform       = '';    % computer()
r.provenance.cpu_info       = '';    % e.g. sysctl machdep.cpu.brand_string
r.provenance.ram_GB         = NaN;
r.provenance.timestamp_utc  = '';    % ISO-8601, e.g. '2026-06-19T10:00:00Z'

% ---- artifact paths (set after saving files) ----------------------------
r.artifacts                = struct();
r.artifacts.mat_file       = '';   % primary .mat result file
r.artifacts.diary_file     = '';   % experiment diary / console log
r.artifacts.topology_png   = '';   % final topology image
r.artifacts.mode_shape_png = '';   % tracked mode shape image
r.artifacts.mac_csv        = '';   % MAC correlation CSV
r.artifacts.history_csv    = '';   % per-iteration history CSV
r.artifacts.extra          = {};   % cell array of any additional file paths
end
