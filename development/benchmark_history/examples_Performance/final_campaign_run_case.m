function out = final_campaign_run_case(method, cfg, opts)
%FINAL_CAMPAIGN_RUN_CASE Execute one final-campaign case under its frozen profile.
%
%   out = FINAL_CAMPAIGN_RUN_CASE(method, cfg) runs a single (method, mesh)
%   case and returns a uniform result struct, whatever runner the method's
%   frozen profile names.
%
%   The three methods do NOT share a runner:
%
%     Olhoff    analysis/olhoff_stabilization_audit/run_stabilization_case.m
%               (profile olhoff_fig3a_s1_native_bimodal_p100_move005_0025_fixed1600_v1)
%               This is the runner named in final_campaign_profile.json.  It
%               rebuilds repro2007_config('fig3a_best') itself -- rminEl = 1.3,
%               move = 0.005, 1600 outer iterations -- and applies the frozen
%               causal S1 policy (N == 2 and gap12 <= 0.01 for 100 consecutive
%               native evaluations -> move 0.0025, once).  It is deliberately
%               NOT the run_topopt_from_json -> OlhoffDu2007Repro path, which
%               carries the legacy S0 / r_min = 2 benchmark profile.
%
%     Yuksel    run_topopt_from_json (profile yuksel_practical_move01_tol001)
%     Proposed  run_topopt_from_json (profile proposed_practical_move02_tol001)
%
%   opts (optional) may carry:
%     .max_outer_override  smaller outer budget, warm-up/smoke use ONLY
%     .warmup              true marks the result as a discarded warm-up
%     .label               free text recorded in the result
%
%   Any override is recorded in out.overrides and makes out.ok false, so a
%   warm-up or smoke result can never be counted as a campaign observation.
%
%   out fields:
%     x, omega, tIter, nIter, mem, nIterStage, telemetry
%     total_wall_time_s   solver-side wall time around the complete solve
%     driver_wall_time_s  caller-side wall time, includes any evidence save
%     status              precedence-ordered verdict (see below)
%     status_note         human-readable reason
%     ok                  true only for the method's declared successful status
%
%   Status vocabulary, following final_campaign_profile.json status_precedence
%   SOLVER_FAILURE > VALID_STABILIZED_STATE_AT_FIXED_WORK > CAP_HIT > RUNNING:
%
%     RUN_ERROR                            the call raised; nothing measured
%     SOLVER_FAILURE                       subproblem/LP failed or state nonfinite
%     VALID_STABILIZED_STATE_AT_FIXED_WORK Olhoff reached k = 1600 solver-healthy
%     NATIVE_CONVERGED                     Yuksel/Proposed met their native test
%     CAP_HIT                              iteration cap reached; NOT convergence
%     UNRECOGNIZED_STOP                    stop reason not in the frozen vocabulary
%
%   Only VALID_STABILIZED_STATE_AT_FIXED_WORK (Olhoff) and NATIVE_CONVERGED
%   (Yuksel, Proposed) set out.ok.  Every other row stays visible in the tables
%   and is censored from the scaling fits by the caller.
%
%   See also FINAL_CAMPAIGN_CONFIG, FINAL_CAMPAIGN_PREFLIGHT,
%            PERFORMANCE_COMPARISON.

if nargin < 3 || isempty(opts); opts = struct(); end

out = struct();
out.method = char(string(method));
out.overrides = opts;
out.is_warmup = isfield(opts, 'warmup') && logical(opts.warmup);
out.x = [];
out.omega = NaN(3,1);
out.tIter = NaN;
out.nIter = NaN;
out.mem = NaN;
out.nIterStage = struct('stage1', NaN, 'stage2', NaN);
out.telemetry = struct();
out.total_wall_time_s = NaN;
out.driver_wall_time_s = NaN;
out.status = 'RUN_ERROR';
out.status_note = '';
out.ok = false;
out.error = '';

driverTic = tic;
try
    switch lower(out.method)
        case 'olhoff'
            out = run_olhoff(out, cfg, opts);
        case {'yuksel', 'ourapproach', 'proposed'}
            out = run_dispatched(out, cfg);
        otherwise
            error('final_campaign_run_case:UnknownMethod', ...
                'Unknown method "%s".', out.method);
    end
catch ME
    out.status = 'RUN_ERROR';
    out.status_note = sprintf('%s: %s', ME.identifier, ME.message);
    out.error = getReport(ME, 'extended', 'hyperlinks', 'off');
    out.ok = false;
end
out.driver_wall_time_s = toc(driverTic);

% An overridden or warm-up run is never a campaign observation, whatever it
% reports.  Stated here once so no caller has to remember it.
if out.is_warmup || isfield(opts, 'max_outer_override')
    out.ok = false;
    if isempty(out.status_note)
        out.status_note = 'warm-up / overridden budget; not a campaign observation';
    end
end
end

% ------------------------------------------------------------------------
function out = run_olhoff(out, cfg, opts)
st = cfg.optimization.stabilization;
nelx = cfg.domain.mesh.nelx;
nely = cfg.domain.mesh.nely;

maxOuter = st.max_iters_expected;
if isfield(opts, 'max_outer_override') && ~isempty(opts.max_outer_override)
    maxOuter = double(opts.max_outer_override);
end

outDir = st.output_dir;
if exist(outDir, 'dir') ~= 7; mkdir(outDir); end

sampler = start_rss_sampler();
try
    solveTic = tic;
    matFile = run_stabilization_case(st.policy_id, nelx, nely, outDir, maxOuter);
    driverSolveTime = toc(solveTic);
catch ME
    % Leave no sampler timer running behind a failed solve.
    stop_rss_sampler(sampler);
    rethrow(ME);
end
out.mem = stop_rss_sampler(sampler);

loaded = load(matFile, 'res');
res = loaded.res;

% ---- Effective-configuration check AT THE OPTIMIZER BOUNDARY -------------
% The preflight proves repro2007_config('fig3a_best') carries the frozen
% numbers before anything is solved.  This re-proves it against what the
% optimizer was actually handed, so a run whose settings drifted cannot be
% reported as this profile.
assert(res.cfg.rminEl == st.rmin_element, 'final_campaign_run_case:OlhoffRmin', ...
    'Olhoff ran with rminEl = %.17g, frozen profile requires %.17g.', ...
    res.cfg.rminEl, st.rmin_element);
assert(res.cfg.move == st.move_initial, 'final_campaign_run_case:OlhoffMove', ...
    'Olhoff ran with move = %.17g, frozen profile requires %.17g.', ...
    res.cfg.move, st.move_initial);
assert(res.cfg.threads == 1, 'final_campaign_run_case:OlhoffThreads', ...
    'Olhoff ran with %d computation threads, frozen profile requires 1.', res.cfg.threads);
assert(isequal(res.policy.move_sequence(:).', [st.move_initial st.move_stabilized]), ...
    'final_campaign_run_case:OlhoffMoveSequence', ...
    'Olhoff move sequence %s does not match the frozen S1 sequence.', ...
    mat2str(res.policy.move_sequence));
assert(res.policy.persistence == st.persistence && res.policy.gap_threshold == st.gap_threshold, ...
    'final_campaign_run_case:OlhoffTrigger', ...
    'Olhoff trigger (persistence %g, gap %g) does not match the frozen policy.', ...
    res.policy.persistence, res.policy.gap_threshold);
if ~isfield(opts, 'max_outer_override')
    assert(res.cfg.maxOuter == st.max_iters_expected, ...
        'final_campaign_run_case:OlhoffMaxOuter', ...
        'Olhoff ran with maxOuter = %d, frozen profile requires %d.', ...
        res.cfg.maxOuter, st.max_iters_expected);
end

h = res.hist;
n = res.nOuter;
loopTime = sum(h.tEig(:)) + sum(h.tGrad(:)) + sum(h.tInner(:));
eigTime  = sum(h.tEig(:));

out.x = double(res.rho(:));
w = double(res.omega(:));
out.omega = [w(1:min(3, numel(w))); NaN(max(0, 3 - numel(w)), 1)];
out.nIter = n;
out.tIter = loopTime / max(n, 1);
out.nIterStage = struct('stage1', NaN, 'stage2', NaN);
out.total_wall_time_s = res.wallclock;
out.solver_evidence_file = matFile;
out.driver_solve_time_s = driverSolveTime;

objectiveHistory = double(h.omega(1, :)).';
relObj = NaN;
if numel(objectiveHistory) >= 2
    relObj = abs(objectiveHistory(end) - objectiveHistory(end-1)) / ...
        max(abs(objectiveHistory(end-1)), eps);
end

tel = struct();
% init/post are NOT separable inside the frozen audit runner, which times only
% the three in-loop phases.  NaN rather than 0: zero would read as "no setup
% cost", which is false.  unattributed_time_s carries what is outside the loop.
tel.timing = struct( ...
    'initialization_time', NaN, ...
    'optimization_loop_time', loopTime, ...
    'postprocessing_time', NaN, ...
    'unattributed_time_s', res.wallclock - loopTime, ...
    'total_wall_time', res.wallclock, ...
    'average_iteration_time', out.tIter, ...
    'eigensolve_time', eigTime, ...
    'gradient_time', sum(h.tGrad(:)), ...
    'subproblem_time', sum(h.tInner(:)));
tel.stopping = struct( ...
    'stop_reason', lower(res.status), ...
    'total_iterations', n, ...
    'iter_stage1', NaN, ...
    'iter_stage2', NaN, ...
    'final_max_density_change', last_or_nan(h.dxOuter), ...
    'final_rms_density_change', last_or_nan(h.dRms), ...
    'final_relative_objective_change', relObj, ...
    'final_grayness', mean(4*out.x.*(1-out.x)), ...
    ... % There is no native convergence test in this profile: the endpoint is
    ... % a fixed work horizon.  A tolerance number here would invite exactly
    ... % the reading the audit forbids.
    'convergence_tolerance', NaN, ...
    'status', '', ...
    'native_stop_reason', res.status, ...
    'native_break_taken', ~strcmp(res.status, 'CAP_HIT'), ...
    'final_multiplicity', last_or_nan(h.N), ...
    'subproblem_failed', strcmp(res.status, 'SOLVER_FAILURE'), ...
    'n_subproblem_failures', double(strcmp(res.status, 'SOLVER_FAILURE')), ...
    'final_inner_converged', last_or_nan(h.innerConv), ...
    'final_lp_flag', last_or_nan(h.lpFlag), ...
    'lp_failure_iters', find(h.lpFlag(:).' ~= 1));
tel.iterations = struct( ...
    'total', n, 'outer', n, 'inner', sum(h.nInner(:)), ...
    'inner_per_outer', sum(h.nInner(:)) / max(n, 1), 'inner_solver', 'lp');
tel.objective_final = out.omega(1);
tel.objective_history = objectiveHistory;
tel.diagnostics_enabled = false;
tel.yuksel = struct('stage1_max_iters', NaN, 'stage1_tolerance', NaN, ...
    'stage2_tolerance', NaN, 'stage1_loop_time', NaN, 'stage2_loop_time', NaN, ...
    'total_loop_time', loopTime);
tel.stabilization = struct( ...
    'profile_id', st.profile_id, ...
    'policy_id', res.policy.id, ...
    'move_sequence', res.policy.move_sequence(:).', ...
    'gap_threshold', res.policy.gap_threshold, ...
    'persistence', res.policy.persistence, ...
    'trigger_iterations', res.trigger_iterations(:).', ...
    'final_policy_stage', res.final_policy_stage, ...
    'final_move_limit', last_or_nan(h.moveLimit), ...
    'final_move_bound_fraction', last_or_nan(h.moveBoundFraction), ...
    'final_gap12', last_or_nan(h.gap12), ...
    'evidence_file', matFile, ...
    'work_semantics', 'FIXED_TOTAL_OUTER_WORK; endpoint is NOT native convergence');

finiteOk = all(logical(h.finiteOk));
if strcmp(res.status, 'SOLVER_FAILURE') || ~finiteOk || any(h.lpFlag(:) ~= 1)
    out.status = 'SOLVER_FAILURE';
    out.status_note = sprintf('LP/state failure at outer iteration %g', res.failure_iteration);
elseif strcmp(res.status, 'CAP_HIT')
    out.status = 'VALID_STABILIZED_STATE_AT_FIXED_WORK';
    out.status_note = sprintf(['reached the frozen %d-outer work horizon; ' ...
        'this is NOT native convergence'], n);
    out.ok = true;
else
    out.status = 'UNRECOGNIZED_STOP';
    out.status_note = sprintf('runner reported status "%s"', res.status);
end
tel.stopping.status = out.status;
out.telemetry = tel;
end

% ------------------------------------------------------------------------
function out = run_dispatched(out, cfg)
solveTic = tic;
[x, omega, tIter, nIter, mem, nIterStage, telemetry] = run_topopt_from_json(cfg);
out.total_wall_time_s = toc(solveTic);

out.x = double(x(:));
out.omega = double(omega(:));
out.tIter = tIter;
out.nIter = nIter;
out.mem = mem;
out.nIterStage = nIterStage;
out.telemetry = telemetry;

s = telemetry.stopping;
failed = false;
if isfield(s, 'subproblem_failed') && ~isempty(s.subproblem_failed)
    failed = failed || logical(s.subproblem_failed);
end
if isfield(s, 'n_subproblem_failures') && isfinite(s.n_subproblem_failures)
    failed = failed || s.n_subproblem_failures > 0;
end
if ~all(isfinite(out.x)) || ~isfinite(out.omega(1)) || out.omega(1) <= 0
    failed = true;
end

reasons = {char(string(s.stop_reason))};
if isfield(s, 'stage1_stop_reason'); reasons{end+1} = char(string(s.stage1_stop_reason)); end
if isfield(s, 'stage2_stop_reason'); reasons{end+1} = char(string(s.stage2_stop_reason)); end
capHit = any(cellfun(@(r) contains(lower(r), 'max_iter'), reasons));
converged = contains(lower(reasons{1}), 'tolerance');

if failed
    out.status = 'SOLVER_FAILURE';
    out.status_note = 'solver reported a failed subproblem or a nonfinite result';
elseif capHit
    % Any stage that ran out of budget censors the row.  A capped stage is a
    % truncated run, not a faster one, and its cost is not the cost of the
    % native profile.
    out.status = 'CAP_HIT';
    out.status_note = sprintf('iteration cap reached (%s); NOT convergence', ...
        strjoin(unique(reasons), '|'));
elseif converged
    out.status = 'NATIVE_CONVERGED';
    out.status_note = sprintf('native stop test met (%s)', reasons{1});
    out.ok = true;
else
    out.status = 'UNRECOGNIZED_STOP';
    out.status_note = sprintf('stop reason "%s" is not in the frozen vocabulary', reasons{1});
end
out.telemetry.stopping.status = out.status;
end

% ------------------------------------------------------------------------
function v = last_or_nan(a)
a = a(:);
if isempty(a); v = NaN; else; v = double(a(end)); end
end

function s = start_rss_sampler()
%START_RSS_SAMPLER Peak-RSS sampler mirroring the one in run_topopt_from_json,
%   so the Olhoff memory column is measured the same way as the other two.
s = struct('baseline', rss_kb(), 'peak', rss_kb(), 'timer', []);
setappdata(0, 'final_campaign_peakRSS_KB', s.peak);
try
    t = timer('ExecutionMode', 'fixedSpacing', 'Period', 0.25, ...
        'TimerFcn', @(~,~) sample_rss());
    start(t);
    s.timer = t;
catch
    s.timer = [];
end
end

function memMB = stop_rss_sampler(s)
sample_rss();
if ~isempty(s.timer)
    try; stop(s.timer); catch; end %#ok<*NOSEMI>
    try; delete(s.timer); catch; end
end
peak = getappdata(0, 'final_campaign_peakRSS_KB');
if isempty(peak); peak = s.baseline; end
memMB = max(0, peak - s.baseline) / 1024;
end

function sample_rss()
cur = rss_kb();
peak = getappdata(0, 'final_campaign_peakRSS_KB');
if isempty(peak) || cur > peak
    setappdata(0, 'final_campaign_peakRSS_KB', cur);
end
end

function kb = rss_kb()
kb = 0;
if ismac || isunix
    [st, r] = system(sprintf('ps -o rss= -p %d', feature('getpid')));
    if st == 0
        kb = str2double(strtrim(r));
        if isnan(kb); kb = 0; end
    end
else
    try
        m = memory;
        kb = m.MemUsedMATLAB / 1024;
    catch
        kb = 0;
    end
end
end
