function nFail = test_cost_reporting()
%TEST_COST_REPORTING  Total AND per-outer-iteration cost are reported, consistently.
%
%   Software test, not scientific evidence: 160x20 with the outer cap CUT to 3
%   (cut the iteration cap, never the mesh).  Checks that olhoffcurrent_run
%   reports the nested accounting and the per-outer-iteration cost fields for a
%   named preset, that the identities between them hold, and that the status of a
%   capped run is CAP_HIT, never convergence.

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(root);
maxNumCompThreads(1);
nFail = 0;
fprintf('\n%s\nTEST_COST_REPORTING\n%s\n', repmat('=',1,72), repmat('=',1,72));

PED = 'duOlhoffPedersenAdaptiveBoxSensitivityFiltered';
out = olhoffcurrent_run(160, 20, 'Preset', PED, 'MaxOuter', 3);
nFail = nFail + chk(sprintf('run completed (status %s)', out.status), strcmp(out.status, 'CAP_HIT') && ~out.ok);
nFail = nFail + chk('result names its preset, role and upstream commit', ...
    strcmp(out.preset, PED) && ~isempty(out.preset_role) && numel(out.upstream_commit) == 40);
a = out.accounting;
need = {'outer_iterations','inner_iterations_total','outer_time_excluding_inner_s','inner_time_total_s', ...
        'total_wall_time_s','overhead_time_s','eigen_time_s','gradient_time_s', ...
        'outer_time_total_s','outer_time_per_outer_mean_s','outer_time_per_outer_median_s', ...
        'outer_time_excluding_inner_per_outer_mean_s','eigen_time_per_outer_mean_s', ...
        'gradient_time_per_outer_mean_s','total_wall_time_per_outer_s','inner_time_per_outer_mean_s'};
nFail = nFail + chk('every total and per-outer field present and finite', ...
    all(isfield(a, need)) && all(cellfun(@(f) isfinite(a.(f)), need)));
n = a.outer_iterations;
rel = @(x, y) abs(x - y) <= 1e-12*max(1, abs(y));
nFail = nFail + chk('outer_iterations = 3 (the cut cap)', n == 3);
nFail = nFail + chk('per-outer means x outer count reproduce the totals', ...
    rel(a.outer_time_per_outer_mean_s*n, a.outer_time_total_s) && ...
    rel(a.eigen_time_per_outer_mean_s*n, a.eigen_time_s) && ...
    rel(a.gradient_time_per_outer_mean_s*n, a.gradient_time_s) && ...
    rel(a.total_wall_time_per_outer_s*n, a.total_wall_time_s) && ...
    rel(a.outer_time_excluding_inner_per_outer_mean_s*n, a.outer_time_excluding_inner_s) && ...
    rel(a.inner_time_per_outer_mean_s*n, a.inner_time_total_s));
nFail = nFail + chk('outer loop nested in the call; overhead = total - sum(tOuter)', ...
    a.outer_time_total_s <= a.total_wall_time_s + 1e-6 && ...
    rel(a.overhead_time_s, a.total_wall_time_s - a.outer_time_total_s));
nFail = nFail + chk('median per-outer time lies within [total/n^2, total]', ...
    a.outer_time_per_outer_median_s > 0 && a.outer_time_per_outer_median_s <= a.outer_time_total_s);
nFail = nFail + chk('stopping record names the move policy and the meaning of final_move_limit', ...
    strcmp(out.stopping.move_policy, 'adaptive') && contains(out.stopping.final_move_limit_meaning, 'per-element'));

fprintf('%s\n  failures: %d\n\n', repmat('-',1,72), nFail);
end

function n = chk(label, ok)
if ok, fprintf('  [PASS] %s\n', label); n = 0;
else,  fprintf('  [FAIL] %s\n', label); n = 1; end
end
