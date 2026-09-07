function out = olhoffcurrent_run(nelx, nely, varargin)
%OLHOFFCURRENT_RUN  THE production entry point for Du-Olhoff eigenfrequency runs.
%
%   out = OLHOFFCURRENT_RUN(nelx, nely) installs the fail-closed path guard,
%   PROVES that analysis/OlhoffCurrent is the only Olhoff implementation MATLAB
%   can see, resolves the production preset at that mesh, runs it through the
%   canonical route (olh.config.resolve -> olhoffSolve) and returns the
%   method-native computational decomposition the conference benchmark reports.
%
%   The method is NESTED.  Its cost is therefore not one iteration count:
%
%     outer_iterations                    Fig. 1 outer loop
%     inner_iterations_total              cumulative nested MMA sub-iterations
%     inner_iterations_per_outer_mean
%     outer_time_excluding_inner_s        FE assembly, eigenproblem, modal
%                                         processing, sensitivities, filtering,
%                                         subproblem construction, the step
%                                         controller, the design update,
%                                         bookkeeping and the convergence test
%     inner_time_total_s                  the nested MMA solve, and only that
%     inner_time_per_outer_mean_s
%     inner_time_per_inner_iteration_mean_s
%     inner_time_share_pct
%     overhead_time_s                     model build, filter preparation, the
%                                         final modal analysis, mode
%                                         classification, result assembly
%     total_wall_time_s                   caller-side, around the solve only
%
%   N_outer + N_inner is NOT reported as a generic iteration count: they are
%   different objects and adding them is meaningless.
%
%   TIMING BOUNDARY.  The timed region is exactly the solver call, as before:
%   the path guard and the configuration resolve happen OUTSIDE tCall, so the
%   measured quantity is unchanged by the promotion.
%
%   Name/value options are forwarded to OLHOFFCURRENT_CONFIG, plus:
%     'Label'   free text recorded in the result
%     'Warmup'  true marks the result as a discarded warm-up
%
%   See also OLHOFFCURRENT_CONFIG, OLHOFFCURRENT_PATHS, OLHOFFCURRENT_PRESET.

p = inputParser();
p.KeepUnmatched = true;
p.addParameter('Label', '', @(v) ischar(v) || isstring(v));
p.addParameter('Warmup', false, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
fwd = p.Unmatched;

fwdArgs = {};
fn = fieldnames(fwd);
for i = 1:numel(fn); fwdArgs = [fwdArgs, {fn{i}, fwd.(fn{i})}]; end %#ok<AGROW>

preset = olhoffcurrent_preset();

out = struct();
out.method = 'Olhoff';
out.method_label = preset.label;
out.implementation = 'analysis/OlhoffCurrent';
out.production_preset = preset.name;
out.upstream_preset = preset.upstreamPreset;
out.mesh = [nelx nely];
out.is_warmup = logical(p.Results.Warmup);
out.label = char(string(p.Results.Label));
out.status = 'RUN_ERROR';
out.status_note = '';
out.ok = false;
out.error = '';
out.x = [];
out.omega = NaN(3,1);
out.caveat = olhoffcurrent_caveat();

% ---- fail-closed dispatch: prove WHICH implementation runs ---------------
% Held for the life of this function; the path is restored on return, including
% on error.
[guard, gate] = olhoffcurrent_paths(); %#ok<ASGLU>
out.resolved_implementation = gate.resolved;

% ---- provenance of THIS result, before anything is solved ---------------
prov = olhoffcurrent_provenance();
out.provenance = prov;

try
    % Configuration resolution is OUTSIDE the timed region.
    [cfg, presetInfo] = olhoffcurrent_config(nelx, nely, fwdArgs{:});
    out.configuration = cfg;
    out.preset_info = presetInfo;
    out.effective_config_hash = olhoffcurrent_config_hash(cfg);
    out.formulation = olh.config.describe(cfg);

    maxOuter = olh.config.getPath(cfg, 'runtime.maxOuter');
    tolOuter = olh.config.getPath(cfg, 'stop.tolerance');

    tCall = tic;
    res = olhoffSolve(cfg);
    callWall = toc(tCall);

    out.x = double(res.rho(:));
    w = double(res.omega(:));
    out.omega = [w(1:min(3,numel(w))); NaN(max(0, 3-min(3,numel(w))), 1)];
    out.resolved_mmasub = which('mmasub');
    out.effective_cfg = res.cfg;
    out.log = res.log;
    out.solver_status = res.status;

    h = res.hist;
    nOuter = double(numel(h.N));
    tOuter = double(h.tOuter(:));
    tInner = double(h.tInner(:));
    tEig   = double(h.tEig(:));
    tGrad  = double(h.tGrad(:));
    nInner = double(h.nInner(:));

    % ---- nesting assertions.  These are NOT vacuous: they would fail if the
    % instrumentation double-counted or mis-nested a region.
    assert(numel(tOuter) == nOuter, 'olhoffcurrent_run:OuterTimerCount', ...
        'hist.tOuter has %d entries for %d outer iterations.', numel(tOuter), nOuter);
    assert(all(tInner <= tOuter + 1e-9), 'olhoffcurrent_run:InnerNotNested', ...
        'The nested MMA time exceeds its own outer iteration at %d iteration(s).', ...
        sum(tInner > tOuter + 1e-9));
    assert(all(tEig + tGrad + tInner <= tOuter + 1e-9), 'olhoffcurrent_run:PhasesNotNested', ...
        'The measured phases exceed their outer iteration at %d iteration(s).', ...
        sum(tEig + tGrad + tInner > tOuter + 1e-9));
    assert(sum(tOuter) <= callWall + 1e-6, 'olhoffcurrent_run:LoopNotNested', ...
        'The outer loop (%.6f s) exceeds the timed solver call (%.6f s).', ...
        sum(tOuter), callWall);

    acc = struct();
    acc.outer_iterations                   = nOuter;
    acc.inner_iterations_total             = sum(nInner);
    acc.inner_iterations_per_outer_mean    = sum(nInner)/max(nOuter,1);
    acc.outer_time_excluding_inner_s       = sum(tOuter) - sum(tInner);
    acc.inner_time_total_s                 = sum(tInner);
    acc.inner_time_per_outer_mean_s        = sum(tInner)/max(nOuter,1);
    acc.inner_time_per_inner_iteration_mean_s = sum(tInner)/max(sum(nInner),1);
    acc.total_wall_time_s                  = callWall;
    acc.overhead_time_s                    = callWall - sum(tOuter);
    acc.inner_time_share_pct               = 100*acc.inner_time_total_s/max(callWall, eps);
    acc.eigen_time_s        = sum(tEig);
    acc.gradient_time_s     = sum(tGrad);
    acc.outer_bookkeeping_time_s = sum(tOuter) - sum(tEig) - sum(tGrad) - sum(tInner);
    acc.timing_accounting_residual_s = callWall - ...
        (acc.outer_time_excluding_inner_s + acc.inner_time_total_s + acc.overhead_time_s);
    acc.timing_accounting_relative_residual = ...
        acc.timing_accounting_residual_s/max(callWall, eps);
    acc.solver_self_report_wall_s = res.wallclock;
    acc.solver_self_report_residual_s = callWall - res.wallclock;
    out.accounting = acc;

    out.stopping = struct( ...
        'outer_iterations', nOuter, ...
        'max_outer', maxOuter, ...
        'converged', any(contains(res.log, 'converged at outer')), ...
        'final_max_density_change', lastOr(h.dxOuter), ...
        'final_l2_density_change', lastOr(h.dxNorm2), ...
        'final_rms_density_change', lastOr(h.dxNorm2)/sqrt(nelx*nely), ...
        'eps_l2', tolOuter, ...
        'eps_rms', tolOuter/sqrt(nelx*nely), ...
        'final_move_limit', lastOr(h.move), ...
        'final_ladder_stage', lastOr(h.stage), ...
        'final_multiplicity', lastOr(h.N), ...
        'final_inner_converged', lastOr(h.innerConv), ...
        'n_inner_not_converged', sum(~logical(h.innerConv)), ...
        'volume', mean(out.x), ...
        'final_grayness', mean(4*out.x.*(1-out.x)), ...
        'gap12_pct', 100*(out.omega(2)-out.omega(1))/out.omega(1));

    if ~all(isfinite(out.x)) || ~isfinite(out.omega(1)) || out.omega(1) <= 0
        out.status = 'SOLVER_FAILURE';
        out.status_note = 'nonfinite design or nonpositive first eigenfrequency';
    elseif out.stopping.converged
        out.status = 'NATIVE_CONVERGED';
        out.status_note = sprintf(['||drho||_2 = %.3e < eps = %.3e with the ' ...
            'move limit settled at %.4g'], out.stopping.final_l2_density_change, ...
            tolOuter, out.stopping.final_move_limit);
        out.ok = true;
    elseif nOuter >= maxOuter
        out.status = 'CAP_HIT';
        out.status_note = sprintf(['reached the %d-outer safety cap; this is ' ...
            'NOT convergence'], maxOuter);
    else
        out.status = 'UNRECOGNIZED_STOP';
        out.status_note = sprintf('outer loop ended at %d without a convergence log line', nOuter);
    end
catch ME
    out.status = 'RUN_ERROR';
    out.status_note = sprintf('%s: %s', ME.identifier, ME.message);
    out.error = getReport(ME, 'extended', 'hyperlinks', 'off');
    out.ok = false;
end

if out.is_warmup
    out.ok = false;
    out.status_note = strtrim([out.status_note ' | warm-up; not a benchmark observation']);
end
end

function v = lastOr(a)
a = a(:);
if isempty(a); v = NaN; else; v = double(a(end)); end
end
