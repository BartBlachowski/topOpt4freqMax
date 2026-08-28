function s = repro2007_stopping(res, cfg)
%REPRO2007_STOPPING  Classify why an OLHOFFOPT run ended -- failure FIRST.
%
%   s = REPRO2007_STOPPING(res)        cfg taken from res.cfg
%   s = REPRO2007_STOPPING(res, cfg)   cfg given explicitly
%
%   RES is the struct OLHOFFOPT returns, verbatim.
%
%   The imported OLHOFFOPT breaks its outer loop on
%
%       if dxOuter < cfg.tolOuter,  break,  end
%
%   and the imported INNERLOOPLP returns drho = zeros(NE,1) when linprog fails.
%   A failed subproblem therefore produces dxOuter = 0, which satisfies ANY
%   positive tolerance, and an earlier version of this function labelled that
%   `outer_increment_below_tolerance` -- reporting a solver failure as
%   convergence.  Both facts are properties of the FROZEN implementation and
%   are deliberately left in place (SOURCE_SHA256.txt); what is fixed here is
%   the classification, which is the RUNNER's responsibility, not the
%   frozen solver's.
%
%   Precedence, in order, per BENCHMARK_PROTOCOL_R3.md section 4:
%
%       subproblem failed        -> SOLVER_FAILURE
%       native stop test met     -> CONVERGED
%       iteration budget reached -> CAP_HIT
%       otherwise                -> RUNNING
%
%   `status` carries that classification.  `stop_reason` keeps the repository's
%   existing vocabulary so that runs which did NOT fail report exactly what
%   they always reported; only the failure case gains a new label.
%
%   The raw native quantities the classification was made from are preserved
%   alongside it -- native_stop_reason, native_break_taken, final_lp_flag,
%   final_inner_converged, lp_failure_iters -- so that a reader can re-derive
%   the verdict instead of trusting it.
%
%   This lives in its own file, rather than inside RUN_REPRO2007, so that the
%   benchmark-path equivalence harness classifies the direct clean-room run
%   with the SAME code the dispatched run is classified by.  Two copies of a
%   precedence rule are two chances for the paths to disagree about a run they
%   both executed identically.
%
%   See also RUN_REPRO2007, REPRO2007_LP_FLAGS,
%            VERIFY_REPRO2007_BENCHMARK_EQUIVALENCE.

if nargin < 2 || isempty(cfg)
    cfg = res.cfg;
end

h = res.hist;
n = numel(h.N);

if n == 0
    s = struct('stop_reason', 'no_iterations', 'status', 'NO_ITERATIONS', ...
        'final_max_density_change', NaN, 'final_rms_density_change', NaN, ...
        'final_relative_objective_change', NaN, 'final_grayness', NaN, ...
        'convergence_tolerance', cfg.tolOuter, ...
        'native_stop_reason', 'no_iterations', 'native_break_taken', false, ...
        'final_inner_converged', false, 'final_lp_flag', NaN, ...
        'subproblem_failed', false, 'n_subproblem_failures', 0, ...
        'lp_failure_iters', [], 'move_limit', cfg.move, ...
        'final_multiplicity', NaN, 'final_eigengap_rel', NaN, ...
        'move_saturated_frac', NaN);
    return
end

% ---- raw native quantities, recorded before anything is decided ---------
lpFlags   = repro2007_lp_flags(res, n);
innerConv = logical(h.innerConv(:).');
failIters = find(~innerConv);
finalInnerConverged = innerConv(end);
finalLpFlag = lpFlags(end);

% What OLHOFFOPT itself did: its break fires iff the last increment is below
% the outer tolerance.  Recorded separately from the interpretation of it.
nativeBreakTaken = h.dxOuter(end) < cfg.tolOuter;
if nativeBreakTaken
    nativeReason = 'outer_increment_below_tolerance';
elseif n >= cfg.maxOuter
    nativeReason = 'max_outer_iterations';
else
    nativeReason = 'terminated_early';
end

% ---- classification: failure outranks convergence -----------------------
subproblemFailed = ~finalInnerConverged;
if subproblemFailed
    status = 'SOLVER_FAILURE';
    reason = 'solver_failure_subproblem';
elseif nativeBreakTaken
    status = 'CONVERGED';
    reason = 'outer_increment_below_tolerance';
elseif n >= cfg.maxOuter
    status = 'CAP_HIT';
    reason = 'max_outer_iterations';
else
    status = 'RUNNING';
    reason = 'terminated_early';
end

obj = sqrt(max(h.beta(:), 0));
if n >= 2 && obj(end-1) ~= 0
    relObj = abs(obj(end) - obj(end-1)) / abs(obj(end-1));
else
    relObj = NaN;
end

rho = res.rho(:);

s = struct( ...
    'stop_reason',                     reason, ...
    'status',                          status, ...
    'final_max_density_change',        h.dxOuter(end), ...
    'final_rms_density_change',        NaN, ...  % see repro2007_history
    'final_relative_objective_change', relObj, ...
    'final_grayness',                  mean(4 * rho .* (1 - rho)), ...
    'convergence_tolerance',           cfg.tolOuter, ...
    'move_limit',                      cfg.move, ...
    'final_multiplicity',              h.N(end), ...
    'final_eigengap_rel',              (res.omega(2) - res.omega(1)) / res.omega(1), ...
    'move_saturated_frac',             mean(abs(h.dxOuter - cfg.move) < 1e-12), ...
    ... % ---- raw native record, preserved (never overwritten) ----
    'native_stop_reason',              nativeReason, ...
    'native_break_taken',              nativeBreakTaken, ...
    'final_inner_converged',           finalInnerConverged, ...
    'final_lp_flag',                   finalLpFlag, ...
    'subproblem_failed',               subproblemFailed, ...
    'n_subproblem_failures',           numel(failIters), ...
    'lp_failure_iters',                failIters);
end
