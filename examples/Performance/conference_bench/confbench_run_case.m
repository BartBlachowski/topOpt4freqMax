function rec = confbench_run_case(methodKey, mcfg, opts)
%CONFBENCH_RUN_CASE  Run one (method, mesh) case and account for its cost natively.
%
%   rec = CONFBENCH_RUN_CASE(methodKey, mcfg, opts)
%
%   THE POINT OF THIS FILE is that the three methods are NOT forced into one
%   generic iteration count.  Total wall time is the common performance
%   quantity; the counts and the component times explain the architecture that
%   produced it, and they mean different things per method:
%
%     Proposed   Stage 1 is ONE reference eigenanalysis (a SOLVE, not an
%                optimization iteration) plus the preparation around it.
%                Stage 2 is the SIMP topology optimization.
%     Yuksel     two sequential optimization stages; N_total = N1 + N2 and
%                T_loop = T1 + T2, both asserted.
%     Olhoff     a NESTED scheme: outer iterations, each containing a complete
%                MMA sub-optimization.  Outer-exclusive time and nested-MMA
%                time are separated, and N_outer + N_inner is never summed.
%
%   TIMING BOUNDARIES.  total_wall_time_s is a caller-side tic/toc around the
%   solve and nothing else.  CSV/JSON/MAT writing, plotting, table formatting,
%   LaTeX generation, the common E1/E2/E3 evaluator and topology rendering all
%   happen outside it, in the driver.
%
%   opts fields (all optional):
%     .max_outer_override   smaller budget; marks the record non-scientific
%     .warmup               true marks the record as a discarded warm-up
%     .label                free text recorded in the record
%     .timing_tol_abs       accounting-identity tolerance, seconds  (default 1e-6)
%     .timing_tol_rel       accounting-identity tolerance, relative (default 1e-9)
%     .crosscheck_tol_rel   independent-cross-check tolerance       (default 0.05)
%
%   See also CONFBENCH_METHOD_CONFIG, OLHOFFCURRENT_RUN, CONFBENCH_TIMING_SCHEMA.

if nargin < 3 || isempty(opts); opts = struct(); end
tolAbs  = getOpt(opts, 'timing_tol_abs', 1e-6);
tolRel  = getOpt(opts, 'timing_tol_rel', 1e-9);
xTolRel = getOpt(opts, 'crosscheck_tol_rel', 0.05);

methodKey = lower(char(string(methodKey)));

rec = struct();
rec.method_key = methodKey;
rec.method = confbench_display_name(methodKey);
rec.method_label = rec.method;
rec.is_warmup = getOpt(opts, 'warmup', false);
rec.overridden = isfield(opts, 'max_outer_override') && ~isempty(opts.max_outer_override);
rec.label = char(string(getOpt(opts, 'label', '')));
rec.status = 'RUN_ERROR';
rec.status_note = '';
rec.ok = false;
rec.error = '';
rec.x = [];
rec.omega = NaN(3,1);
rec.omega1_native = NaN;
rec.counts = struct();
rec.times = struct();
rec.stopping = struct();
rec.caveat = '';
rec.resolved_implementation = struct('name', {}, 'file', {});
% Declared up front so every record -- whatever method produced it -- carries
% the SAME top-level field set and the driver can build one struct array.
rec.effective_config = struct();
rec.solver_log = {};
rec.telemetry = struct();

try
    switch methodKey
        case 'olhoff';                 rec = runOlhoff(rec, mcfg, opts);
        case {'proposed','ourapproach'}; rec = runProposed(rec, mcfg, opts);
        case 'yuksel';                 rec = runYuksel(rec, mcfg, opts);
        otherwise
            error('confbench_run_case:UnknownMethod', 'Unknown method "%s".', methodKey);
    end
catch ME
    rec.status = 'RUN_ERROR';
    rec.status_note = sprintf('%s: %s', ME.identifier, ME.message);
    rec.error = getReport(ME, 'extended', 'hyperlinks', 'off');
    rec.ok = false;
end

% ---- explicit timing accounting, identical shape for every method -------
rec.accounting = confbench_accounting(rec.times, tolAbs, tolRel, xTolRel);
rec = orderfields(rec);

if rec.is_warmup || rec.overridden
    rec.ok = false;
    rec.status_note = strtrim([rec.status_note ...
        ' | warm-up or overridden budget; NOT a benchmark observation']);
end
end

% =========================================================================
function rec = runOlhoff(rec, mcfg, opts)
%RUNOLHOFF  THE production Du-Olhoff implementation, analysis/OlhoffCurrent.
%   Nested accounting.  olhoffcurrent_run installs the fail-closed path guard
%   and proves that exactly one Olhoff implementation is visible before it
%   solves, so the numbers below are attributable to a PROVED implementation
%   rather than an intended one.  The timed region is the solver call only,
%   unchanged by the promotion.
nelx = mcfg.nelx; nely = mcfg.nely;
args = {};
if isfield(opts, 'max_outer_override') && ~isempty(opts.max_outer_override)
    args = [args, {'MaxOuter', double(opts.max_outer_override)}];
end
if getOpt(opts, 'warmup', false); args = [args, {'Warmup', true}]; end

o = olhoffcurrent_run(nelx, nely, args{:});

rec.status = o.status;
rec.status_note = o.status_note;
rec.ok = o.ok;
rec.error = o.error;
rec.x = o.x;
rec.omega = o.omega;
rec.omega1_native = o.omega(1);
rec.caveat = o.caveat;
rec.resolved_implementation = o.resolved_implementation;
% Result provenance: enough to answer "what implementation produced this
% number?" from the artifact alone -- implementation, named production preset,
% effective configuration hash, promoted source tree hash, repository commit.
rec.implementation = getOpt(o, 'implementation', '');
rec.production_preset = getOpt(o, 'production_preset', '');
rec.effective_config_hash = getOpt(o, 'effective_config_hash', '');
rec.implementation_provenance = orStruct(o, 'provenance');
rec.effective_config = orStruct(o, 'effective_cfg');
rec.stopping = orStruct(o, 'stopping');
rec.solver_log = orCell(o, 'log');

% A failed nested MMA solve must never reach the table as a clean convergence.
% olhoffcurrent_run counts them but gates SOLVER_FAILURE only on a nonfinite
% design or a nonpositive omega1, so the count is enforced here rather than by
% editing the promoted solver.  Inert whenever the count is zero, which it is
% for every row of the existing campaign.
nBadInner = 0;
if isstruct(rec.stopping) && isfield(rec.stopping, 'n_inner_not_converged') ...
        && ~isempty(rec.stopping.n_inner_not_converged) ...
        && isfinite(rec.stopping.n_inner_not_converged)
    nBadInner = double(rec.stopping.n_inner_not_converged);
end
if nBadInner > 0 && ~strcmp(rec.status, 'SOLVER_FAILURE')
    rec.status = 'SOLVER_FAILURE';
    rec.status_note = sprintf(['%d nested MMA sub-optimization(s) did not converge; ' ...
        'NOT a clean convergence | previous status: %s'], nBadInner, rec.status_note);
    rec.ok = false;
end

if ~isfield(o, 'accounting'); return; end
a = o.accounting;

rec.counts = struct( ...
    'count1_name', 'outer_iterations', ...
    'count1', a.outer_iterations, ...
    'count2_name', 'inner_mma_iterations_total', ...
    'count2', a.inner_iterations_total, ...
    'outer_iterations', a.outer_iterations, ...
    'inner_iterations_total', a.inner_iterations_total, ...
    'inner_iterations_per_outer_mean', a.inner_iterations_per_outer_mean, ...
    'iterations_total_generic', NaN);   % deliberately NOT N_outer + N_inner

rec.times = struct( ...
    'time1_name', 'outer_time_excluding_inner_s', ...
    'time1', a.outer_time_excluding_inner_s, ...
    'time2_name', 'inner_mma_time_total_s', ...
    'time2', a.inner_time_total_s, ...
    'outer_time_excluding_inner_s', a.outer_time_excluding_inner_s, ...
    'inner_time_total_s', a.inner_time_total_s, ...
    'inner_time_per_outer_mean_s', a.inner_time_per_outer_mean_s, ...
    'inner_time_per_inner_iteration_mean_s', a.inner_time_per_inner_iteration_mean_s, ...
    'inner_time_share_pct', a.inner_time_share_pct, ...
    'eigen_time_s', a.eigen_time_s, ...
    'gradient_time_s', a.gradient_time_s, ...
    'outer_bookkeeping_time_s', a.outer_bookkeeping_time_s, ...
    'overhead_time_s', a.overhead_time_s, ...
    'total_wall_time_s', a.total_wall_time_s, ...
    'solver_self_report_wall_s', a.solver_self_report_wall_s, ...
    'independent_crosscheck_residual_s', a.solver_self_report_residual_s);
end

% =========================================================================
function rec = runProposed(rec, mcfg, opts)
%RUNPROPOSED  One reference eigenanalysis, then a SIMP solve.
if isfield(opts, 'max_outer_override') && ~isempty(opts.max_outer_override)
    mcfg.optimization.max_iters = double(opts.max_outer_override);
end
% Memory is out of the benchmark contract, and the sampler it would otherwise
% start forks `ps` at 10 Hz inside the timed loop.  Off, explicitly.
mcfg.benchmark.measure_memory = false;

tCall = tic;
[x, omega, tIter, nIter, ~, nIterStage, tel] = run_topopt_from_json(mcfg); %#ok<ASGLU>
callWall = toc(tCall);

% The Proposed method has one iteration budget; compare the SIMP count to it.
rec = fillDispatched(rec, x, omega, nIter, nIterStage, tel, ...
    struct('total', cfgNum(mcfg, {'optimization','max_iters'})));

t1 = tel.timing.initialization_time;                 % preparation + the 1 eigensolve
t2 = tel.timing.optimization_loop_time;              % the SIMP solve
tEig = tel.timing.stage1_reference_eigen_time;
nSolves = tel.timing.stage1_reference_eigen_solves;

assert(nSolves == 1, 'confbench_run_case:ProposedStage1Solves', ...
    ['The Proposed method must perform exactly ONE reference eigenanalysis; ' ...
     'the solver reported %g.'], nSolves);
assert(isfinite(tEig) && tEig >= 0 && tEig <= t1 + 1e-9, ...
    'confbench_run_case:ProposedStage1Nesting', ...
    ['The reference eigenanalysis (%.6f s) is not a sub-interval of ' ...
     'Stage-1 preparation (%.6f s).'], tEig, t1);

rec.counts = struct( ...
    'count1_name', 'eigenanalysis_solves', ...
    'count1', nSolves, ...
    'count2_name', 'simp_iterations', ...
    'count2', nIter, ...
    'stage1_solves', nSolves, ...
    'stage2_iterations', nIter, ...
    'iterations_total_generic', nIter);

rec.times = struct( ...
    'time1_name', 'stage1_eigenanalysis_and_preparation_s', ...
    'time1', t1, ...
    'time2_name', 'stage2_simp_time_s', ...
    'time2', t2, ...
    'stage1_time_s', t1, ...
    'stage1_reference_eigen_time_s', tEig, ...
    'stage2_time_s', t2, ...
    'overhead_time_s', callWall - t1 - t2, ...
    'total_wall_time_s', callWall, ...
    'solver_self_report_wall_s', tel.timing.total_wall_time, ...
    'independent_crosscheck_residual_s', callWall - tel.timing.total_wall_time);
end

% =========================================================================
function rec = runYuksel(rec, mcfg, opts)
%RUNYUKSEL  Two sequential optimization stages.
% The per-stage safety budget, applied to BOTH stages so neither can be
% censored while the other runs free.  max_outer_override is applied after it
% and therefore wins: that knob is the smoke-test truncation, and a truncation
% must not be silently widened by a raised budget.
if isfield(opts, 'yuksel_max_iters') && ~isempty(opts.yuksel_max_iters)
    mcfg.optimization.max_iters = double(opts.yuksel_max_iters);
    mcfg.optimization.yuksel.stage1_max_iters = double(opts.yuksel_max_iters);
end
if isfield(opts, 'max_outer_override') && ~isempty(opts.max_outer_override)
    mcfg.optimization.max_iters = double(opts.max_outer_override);
    mcfg.optimization.yuksel.stage1_max_iters = double(opts.max_outer_override);
end
mcfg.benchmark.measure_memory = false;

tCall = tic;
[x, omega, tIter, nIter, ~, nIterStage, tel] = run_topopt_from_json(mcfg); %#ok<ASGLU>
callWall = toc(tCall);

% Both per-stage caps as this run actually received them.  Stage 1 prefers the
% value the solver echoed back (telemetry.yuksel.stage1_max_iters); Stage 2 is
% governed by optimization.max_iters.  Neither is restated as a constant.
yCaps = struct('stage1', telNum(tel, {'yuksel','stage1_max_iters'}), ...
               'stage2', cfgNum(mcfg, {'optimization','max_iters'}));
if isempty(yCaps.stage1)
    yCaps.stage1 = cfgNum(mcfg, {'optimization','yuksel','stage1_max_iters'});
end
rec = fillDispatched(rec, x, omega, nIter, nIterStage, tel, yCaps);
rec.stopping.stage1_max_iters_effective = yCaps.stage1;
rec.stopping.stage2_max_iters_effective = yCaps.stage2;

n1 = nIterStage.stage1; n2 = nIterStage.stage2;
t1 = tel.yuksel.stage1_loop_time; t2 = tel.yuksel.stage2_loop_time;

assert(isfinite(n1) && isfinite(n2) && n1 + n2 == nIter, ...
    'confbench_run_case:YukselIterationIdentity', ...
    'N1 + N2 = %g + %g = %g does not equal N_total = %g.', n1, n2, n1+n2, nIter);
assert(isfinite(t1) && isfinite(t2), 'confbench_run_case:YukselStageTimers', ...
    'Yuksel did not report both stage loop times (%g, %g).', t1, t2);
assert(abs((t1 + t2) - tel.timing.optimization_loop_time) <= ...
       max(1e-6, 1e-6*tel.timing.optimization_loop_time), ...
    'confbench_run_case:YukselStageTimersOverlap', ...
    ['Stage timers must be non-overlapping and exhaustive: T1 + T2 = %.9f s ' ...
     'but the loop time is %.9f s.'], t1 + t2, tel.timing.optimization_loop_time);

rec.counts = struct( ...
    'count1_name', 'stage1_iterations', ...
    'count1', n1, ...
    'count2_name', 'stage2_iterations', ...
    'count2', n2, ...
    'stage1_iterations', n1, ...
    'stage2_iterations', n2, ...
    'iterations_total_generic', nIter);

rec.times = struct( ...
    'time1_name', 'stage1_time_s', ...
    'time1', t1, ...
    'time2_name', 'stage2_time_s', ...
    'time2', t2, ...
    'stage1_time_s', t1, ...
    'stage2_time_s', t2, ...
    'stage1_share_pct', 100*n1/max(nIter,1), ...
    'stage2_share_pct', 100*n2/max(nIter,1), ...
    'overhead_time_s', callWall - t1 - t2, ...
    'total_wall_time_s', callWall, ...
    'solver_self_report_wall_s', tel.timing.total_wall_time, ...
    'independent_crosscheck_residual_s', callWall - tel.timing.total_wall_time);
end

% =========================================================================
function rec = fillDispatched(rec, x, omega, nIter, nIterStage, tel, caps)
%FILLDISPATCHED  Shared record fill for the two dispatched methods.
%
%   The status decision itself lives in CONFBENCH_CLASSIFY, which is a separate
%   file precisely so the precedence
%
%       SOLVER_FAILURE > CAP_HIT > NATIVE_CONVERGED > UNRECOGNIZED_STOP
%
%   can be regression-tested without a solve.  caps carries the ACTUAL numeric
%   iteration caps this run received; see CONFBENCH_CLASSIFY for why the
%   numeric test is required and the textual stop reason is not sufficient.
if nargin < 7 || isempty(caps); caps = struct(); end
rec.x = double(x(:));
rec.omega = double(omega(:));
rec.omega1_native = rec.omega(1);
rec.telemetry = tel;
s = tel.stopping;
rec.stopping = s;

% A nonfinite design or a nonpositive first frequency is a solver failure
% regardless of what the solver said; hand that to the classifier as data.
s.design_nonfinite = ~all(isfinite(rec.x)) || ~isfinite(rec.omega(1)) || rec.omega(1) <= 0;

cls = confbench_classify(s, nIter, nIterStage, caps);
rec.status = cls.status;
rec.status_note = cls.note;
rec.ok = cls.ok;
end

% =========================================================================
function v = getOpt(s, name, dflt)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name)); v = s.(name); else; v = dflt; end
end
function v = orStruct(s, name)
if isfield(s, name); v = s.(name); else; v = struct(); end
end
function v = orCell(s, name)
if isfield(s, name); v = s.(name); else; v = {}; end
end

% =========================================================================
function v = cfgNum(s, pathCells)
%CFGNUM  Numeric value at a nested config path, [] when absent.
v = [];
for i = 1:numel(pathCells)
    if ~isstruct(s) || ~isfield(s, pathCells{i}); return; end
    s = s.(pathCells{i});
end
if isnumeric(s) && isscalar(s) && isfinite(s); v = double(s); end
end

% =========================================================================
function v = telNum(t, pathCells)
%TELNUM  Same, for telemetry.
v = cfgNum(t, pathCells);
end
