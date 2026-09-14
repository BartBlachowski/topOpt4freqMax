clear; clc;

% Paired extension test, plan section 6.3.
%
% Offline acceptance is only valid if extending a run past its native stopping
% point leaves the entire prefix unchanged.  For each (method, mesh):
%
%   1. run with the native stopping rule;
%   2. rerun from the identical initialization with only the FINAL native
%      termination disabled, leaving stage handoffs and continuation active;
%   3. compare every recorded scalar through the native stopping iteration;
%   4. compare the physical density at that iteration;
%   5. continue for a preregistered validation horizon beyond it.
%
% The static reading of the stop rules is in STOP_RULE_AUDIT.md.  That shows the
% control flow is unchanged; this shows the arithmetic is.

thisDir = fileparts(mfilename('fullpath'));
baseCfg = jsondecode(fileread(fullfile(thisDir, 'performance_comparison.json')));
baseCfg.postprocessing.visualize_live = false;
baseCfg.postprocessing.save_final_image = false;
baseCfg.postprocessing.save_snapshot_image = false;
baseCfg.postprocessing.save_frequency_iterations = false;
baseCfg.optimization.filter.radius = 2;
baseCfg.optimization.filter.radius_units = 'element';
baseCfg.benchmark.record_history = true;

meshes     = [160, 20; 320, 40];
approaches = {'Olhoff',         'Yuksel',         'OurApproach'};
labels     = {'OlhoffApproach', 'YukselApproach', 'ProposedApproach'};

% Preregistered bound.  Bitwise equality is required where execution is
% deterministic, which section 6.1 established for all three solvers; the
% numerical bound is the fallback and is reported either way.
PREFIX_BOUND = 1e-12;
VALIDATION_HORIZON = 60;   % iterations to continue beyond the native stop

failures = {};
rows = struct('method', {}, 'mesh', {}, 'shared_budget', {}, ...
    'native_iters', {}, 'extended_iters', {}, ...
    'native_stop_iter_in_extended', {}, 'prefix_rows_compared', {}, ...
    'max_abs_diff_xphys_at_stop', {}, 'scalar_prefix_identical', {}, ...
    'bitwise', {}, 'horizon_reached', {}, 'pass', {});

for r = 1:size(meshes,1)
    for m = 1:numel(approaches)
        meshStr = sprintf('%dx%d', meshes(r,1), meshes(r,2));
        cfg = baseCfg;
        cfg.domain.mesh.nelx = meshes(r,1);
        cfg.domain.mesh.nely = meshes(r,2);
        cfg.optimization.approach = approaches{m};

        % BOTH arms must run under the SAME iteration budget.  max_iters is not
        % a pure safety budget for every method: the Olhoff-Du continuous beta
        % ramp uses T = min(300, 0.7*maxiter), so a budget below 429 changes the
        % projection schedule and hence the whole trajectory.  Giving the two
        % arms different budgets compares two different problems.  A probe run
        % sizes the shared budget; only its iteration count is used.
        fprintf('%-18s %-9s probe ...\n', labels{m}, meshStr);
        [~, ~, ~, nProbe] = run_topopt_from_json(cfg);
        budget = nProbe + VALIDATION_HORIZON;

        cfgShared = cfg;
        cfgShared.optimization.max_iters = budget;
        if strcmpi(approaches{m}, 'Yuksel')
            cfgShared.optimization.yuksel.stage1_max_iters = budget;
        end

        fprintf('%-18s %-9s native   (budget %d) ...\n', labels{m}, meshStr, budget);
        cfgN = cfgShared;
        cfgN.benchmark.extend_beyond_native_stop = false;
        [xN, ~, ~, nN, ~, ~, telN] = run_topopt_from_json(cfgN);

        fprintf('%-18s %-9s extended (budget %d) ...\n', labels{m}, meshStr, budget);
        cfgE = cfgShared;
        cfgE.benchmark.extend_beyond_native_stop = true;
        [~, ~, ~, nE, ~, ~, telE] = run_topopt_from_json(cfgE);

        HN = telN.history;
        HE = telE.history;
        stopIter = telE.extension.native_stop_iter;

        rec = struct('method', labels{m}, 'mesh', meshStr, ...
            'shared_budget', budget, ...
            'native_iters', nN, 'extended_iters', nE, ...
            'native_stop_iter_in_extended', stopIter, ...
            'prefix_rows_compared', NaN, 'max_abs_diff_xphys_at_stop', NaN, ...
            'scalar_prefix_identical', false, 'bitwise', false, ...
            'horizon_reached', nE - nN, 'pass', false);

        if isnan(stopIter)
            failures{end+1} = sprintf(['%s %s: extended run never reported a ' ...
                'native stop iteration'], labels{m}, meshStr); %#ok<SAGROW>
            rows(end+1) = rec; %#ok<SAGROW>
            continue;
        end
        if stopIter ~= nN
            failures{end+1} = sprintf(['%s %s: extended run places the native ' ...
                'stop at %d, native run executed %d iterations'], ...
                labels{m}, meshStr, stopIter, nN); %#ok<SAGROW>
        end

        % 3. every recorded scalar through the native stopping iteration
        nCmp = min([nN, HN.n, HE.n]);
        rec.prefix_rows_compared = nCmp;
        cols = {'iter','stage','stage_iter','d_inf','d_rms','rV','grayness', ...
            'd_inf_design','d_rms_design','objective','move_active_frac'};
        identical = true;
        worst = 0;
        for c = 1:numel(cols)
            a = HN.(cols{c})(1:nCmp);
            b = HE.(cols{c})(1:nCmp);
            if ~isequaln(a, b)
                identical = false;
            end
            d = max(abs(a - b), [], 'omitnan');
            if ~isempty(d) && isfinite(d)
                worst = max(worst, d);
            end
        end
        rec.scalar_prefix_identical = identical;
        rec.bitwise = identical;

        % 4. physical density at the native stopping iteration
        xStop = telE.extension.xPhys_at_native_stop;
        if isempty(xStop)
            failures{end+1} = sprintf('%s %s: no density checkpoint recorded', ...
                labels{m}, meshStr); %#ok<SAGROW>
        else
            rec.max_abs_diff_xphys_at_stop = max(abs(xStop(:) - xN(:)));
            if rec.max_abs_diff_xphys_at_stop > PREFIX_BOUND
                failures{end+1} = sprintf(['%s %s: density at the native stop ' ...
                    'differs by %.3e, above the %.0e bound'], labels{m}, meshStr, ...
                    rec.max_abs_diff_xphys_at_stop, PREFIX_BOUND); %#ok<SAGROW>
            end
        end

        if ~identical && worst > PREFIX_BOUND
            failures{end+1} = sprintf(['%s %s: scalar prefix differs by %.3e, ' ...
                'above the %.0e bound'], labels{m}, meshStr, worst, PREFIX_BOUND); %#ok<SAGROW>
        end

        % 5. the run must actually have continued past the native stop
        if nE <= nN
            failures{end+1} = sprintf(['%s %s: extended run did not continue ' ...
                'past the native stop (%d vs %d)'], labels{m}, meshStr, nE, nN); %#ok<SAGROW>
        end

        rec.pass = identical && nE > nN && ...
            (isempty(xStop) == false && rec.max_abs_diff_xphys_at_stop <= PREFIX_BOUND);
        rows(end+1) = rec; %#ok<SAGROW>
    end
end

sep = repmat('-', 1, 124);
fprintf('\nPaired extension test, section 6.3\n%s\n', sep);
fprintf('%-18s %-9s %8s %8s %10s %8s %9s %16s %8s %5s\n', ...
    'Method', 'Mesh', 'budget', 'native', 'extended', 'stop@', 'rows cmp', ...
    'max|dxPhys|@stop', 'bitwise', 'pass');
fprintf('%s\n', sep);
for i = 1:numel(rows)
    fprintf('%-18s %-9s %8d %8d %10d %8d %9d %16.3e %8d %5d\n', ...
        rows(i).method, rows(i).mesh, rows(i).shared_budget, ...
        rows(i).native_iters, rows(i).extended_iters, ...
        rows(i).native_stop_iter_in_extended, rows(i).prefix_rows_compared, ...
        rows(i).max_abs_diff_xphys_at_stop, rows(i).bitwise, rows(i).pass);
end
fprintf('%s\n', sep);

passed = isempty(failures);
if passed
    fprintf(['\nPASS: every extended run reproduces its native prefix bitwise and ' ...
        'continues past\n      the native stopping point.  Offline acceptance is ' ...
        'sound for these methods.\n']);
else
    fprintf('\nFAIL: %d problem(s):\n', numel(failures));
    for k = 1:numel(failures)
        fprintf('  - %s\n', failures{k});
    end
end

validation = struct();
validation.description = 'Paired extension test, plan section 6.3';
validation.prefix_bound = PREFIX_BOUND;
validation.validation_horizon = VALIDATION_HORIZON;
validation.static_audit = 'examples/Performance/STOP_RULE_AUDIT.md';
validation.rows = rows;
validation.failures = failures(:);
validation.pass = passed;

validationPath = fullfile(thisDir, 'extension_invariance_validation.json');
fid = fopen(validationPath, 'w');
assert(fid >= 0, 'validate_extension_invariance:OpenFailed', ...
    'Cannot open %s for writing.', validationPath);
fprintf(fid, '%s\n', jsonencode(validation));
fclose(fid);
fprintf('Validation record saved to: %s\n', validationPath);

assert(passed, 'validate_extension_invariance:PrefixChanged', ...
    'Extending past the native stop changed the prefix; offline gating is blocked.');
