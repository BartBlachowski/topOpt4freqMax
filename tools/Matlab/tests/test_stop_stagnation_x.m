function nFail = test_stop_stagnation_x()
%TEST_STOP_STAGNATION_X  Stop criterion 'stagnation_x' for Proposed and Yuksel.
%
%   'stagnation_x' is 'stagnation' with the objective test replaced by a test on
%   the physical density field xPhys (the field entering the FE analysis): over
%   the last W+1 analysed designs of the current stage, the W relative changes
%   ||xPhys_j - xPhys_{j-1}||_2/||xPhys_{j-1}||_2 must ALL be < relTol, and the
%   grayness range must be < grayTol, as for 'stagnation'.
%
%   A. the window function stop_stagnation_x, on constructed sequences:
%        1. W+1 designs / W changes: a change just outside the window is ignored,
%           the oldest change inside it is not
%        2. no stop before the window is complete (k - W >= 2)
%        3. the largest change of the window is tested, not the last one
%        4. the grayness condition is still required; both tests are strict
%   B. configuration paths accept 'stagnation_x' and its tolerance, and still
%      reject an unknown criterion
%   C. Proposed 160x20, sensitivity filter (production, xPhys = x) and density
%      filter (xPhys = (H x)./Hs ~= x): the run stops exactly where the rule,
%      recomputed from the recorded ANALYSED designs xPhys, first holds, and
%      reports that rule's window values
%   D. Yuksel 160x20, both stages on 'stagnation_x', sensitivity and density
%      filter: the same per stage, each stage's window built from that stage's
%      designs only (5. reset at the handoff)
%   6. the field: C and D recompute the rule from the recorded xPhys, and check
%      against the recorder's design-variable increments max|x_{k+1} - x_k|
%      that xPhys IS x under the sensitivity filter and is NOT x under the
%      density filter -- so the exact match identifies the field
%   (7. existing strategies: test_stop_strategies_unchanged.  Olhoff:
%   analysis/Olhoff/tests/test_olhoff_stagnation_x.)

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(here)));
addpath(fullfile(repo, 'tools', 'Matlab'));
addpath(fullfile(repo, 'examples', 'Performance', 'conference_bench'));
addpath(fullfile(repo, 'examples', 'Performance', 'benchmark_profile'));
addpath(fullfile(repo, 'analysis', 'Olhoff'));
olhoffcurrent_scrub_forbidden_paths(repo);

nFail = 0;
fprintf('\n%s\nTEST_STOP_STAGNATION_X\n%s\n', repmat('=',1,72), repmat('=',1,72));

% ---- A. the window function ------------------------------------------------
W = 3; tol = 1e-3; gtol = 1e-2;
n = 12; d = [NaN; zeros(n-1,1)]; g = 0.5*ones(n,1);
k = 8;   % window designs 5..8, changes d(6..8)
dd = d; dd(5) = 1;
nFail = nFail + chk('A1 change d(k-W) (designs k-W-1 -> k-W, outside the window) is ignored', ...
    stop_stagnation_x(dd, g, k, W, tol, gtol));
dd = d; dd(6) = 1;
nFail = nFail + chk('A1 change d(k-W+1) (first change inside the window) is tested', ...
    ~stop_stagnation_x(dd, g, k, W, tol, gtol));
[~, r] = stop_stagnation_x(d, g, k, W, tol, gtol);
nFail = nFail + chk('A1 window spans W+1 designs: grayness of design k-W is in, k-W-1 is out', ...
    r(2) == 0 && ...
    ~stop_stagnation_x(d, setEl(g, 5, 0.9), k, W, tol, gtol) && ...
    stop_stagnation_x(d, setEl(g, 4, 0.9), k, W, tol, gtol));
met = arrayfun(@(kk) stop_stagnation_x(d, g, kk, W, tol, gtol), 1:n);
nFail = nFail + chk('A2 no stop while k - W < 2; first possible stop at k = W + 2', ...
    isequal(find(met, 1), W + 2));
[m2, r2] = stop_stagnation_x(d, g, W + 1, W, tol, gtol);
nFail = nFail + chk('A2 incomplete window returns met = false, r = [NaN NaN]', ...
    ~m2 && all(isnan(r2)));
dd = d; dd(k-W+1) = 10*tol;
[m3, r3] = stop_stagnation_x(dd, g, k, W, tol, gtol);
nFail = nFail + chk('A3 the largest change of the window decides (last-step-only would stop)', ...
    ~m3 && r3(1) == 10*tol && dd(k) < tol);
gg = g; gg(k) = 0.5 + 2*gtol;
nFail = nFail + chk('A4 grayness range >= grayTol blocks the stop although xPhys is still', ...
    ~stop_stagnation_x(d, gg, k, W, tol, gtol));
nFail = nFail + chk('A4 both tests strict: r(1) == relTol and r(2) == grayTol do not stop', ...
    ~stop_stagnation_x(setEl(d, k, tol), g, k, W, tol, gtol) && ...
    ~stop_stagnation_x(d, setEl(g, k, 0.5 + gtol), k, W, tol, gtol) && ...
    stop_stagnation_x(setEl(d, k, 0.999*tol), setEl(g, k, 0.5 + 0.999*gtol), k, W, tol, gtol));

% ---- B. configuration paths ------------------------------------------------
stopX = struct('criterion', 'stagnation_x', 'window', 7, 'relativeChangeTol', 2e-3, 'graynessTol', 4e-3);
mp = confbench_method_config('proposed', 160, 20, '', stopX);
nFail = nFail + chk('B confbench: Proposed stop_criterion and optimization.stagnation', ...
    strcmp(mp.optimization.stop_criterion, 'stagnation_x') && ...
    isequal(mp.optimization.stagnation, struct('window', 7, 'grayness_tol', 4e-3, 'relative_change_tol', 2e-3)));
my = confbench_method_config('yuksel', 160, 20, '', stopX);
nFail = nFail + chk('B confbench: Yuksel, both stages (no stage1_stop_criterion set)', ...
    strcmp(my.optimization.stop_criterion, 'stagnation_x') && ...
    ~isfield(my.optimization.yuksel, 'stage1_stop_criterion') && ...
    my.optimization.stagnation.relative_change_tol == 2e-3);
my1 = confbench_method_config('yuksel', 160, 20, '', setField(stopX, 'stage1Criterion', 'stagnation'));
nFail = nFail + chk('B confbench: Yuksel stage1Criterion may differ from stage 2', ...
    strcmp(my1.optimization.yuksel.stage1_stop_criterion, 'stagnation') && ...
    strcmp(my1.optimization.stop_criterion, 'stagnation_x'));
mo = confbench_method_config('olhoff', 160, 20, '', stopX);
nFail = nFail + chk('B confbench: Olhoff forwards StopStagnationX and resolves stop.rule', ...
    isequal(mo.olhoff_stop_stagnation_x, struct('window', 7, ...
        'relativeChangeTolerance', 2e-3, 'graynessTolerance', 4e-3)) && ...
    isempty(mo.olhoff_stop_stagnation) && ...
    strcmp(mo.canonical.stop.rule, 'stagnation_x') && ...
    mo.canonical.stop.stagnation.relativeChangeTolerance == 2e-3);
bad = mp; bad.optimization.stop_criterion = 'stagnation_y';
nFail = nFail + chk('B run_topopt_from_json still rejects an unknown criterion', ...
    throwsId(@() run_topopt_from_json(bad), 'run_topopt_from_json:InvalidStopCriterion'));
nFail = nFail + chk('B confbench still rejects an unknown criterion', ...
    throwsId(@() confbench_method_config('proposed', 160, 20, '', struct('criterion', 'stagnation_y')), ...
        'confbench_method_config:UnknownCriterion'));
sch = jsondecode(fileread(fullfile(repo, 'docs', 'topopt_config.schema.json')));
po = sch.properties.optimization.properties;
nFail = nFail + chk('B docs/topopt_config.schema.json lists stagnation_x and relative_change_tol', ...
    any(strcmp(po.stop_criterion.enum, 'stagnation_x')) && ...
    any(strcmp(po.yuksel.properties.stage1_stop_criterion.enum, 'stagnation_x')) && ...
    isfield(po.stagnation.properties, 'relative_change_tol'));

% ---- C. Proposed -----------------------------------------------------------
% Tolerances chosen so the rule fires well inside the budget at 160x20; they
% exercise the mechanics and are not calibrated values.
cases = {'sensitivity', 5e-3, 1e-2; 'density', 5e-3, 1e-2};
for c = 1:size(cases, 1)
    W = 10; relTol = cases{c,2}; grayTol = cases{c,3};
    mcfg = confbench_method_config('proposed', 160, 20, '', struct('criterion', 'stagnation_x', ...
        'window', W, 'relativeChangeTol', relTol, 'graynessTol', grayTol, 'maxIters', 400));
    mcfg.optimization.filter.type = cases{c,1};
    mcfg.postprocessing.record_design_history = true;
    mcfg.benchmark.record_history = true;   % recorder: design-variable increments
    [~, ~, ~, nIter, ~, ~, tel] = run_topopt_from_json(mcfg);
    X = tel.design_history;
    [kStar, rStar, dRec] = oracleStop(X, W, relTol, grayTol);
    tag = sprintf('C Proposed %s filter', cases{c,1});
    fprintf('  %s: stopped at %d (%s), oracle k* = %d, max change %.4g, gray range %.4g\n', ...
        tag, nIter, tel.stopping.stop_reason, kStar, rStar(1), rStar(2));
    nFail = nFail + chk([tag ': design history holds one analysed design per iteration'], ...
        size(X, 2) == nIter);
    nFail = nFail + chk([tag ': stops exactly at the first k where the rule holds on xPhys'], ...
        strcmp(tel.stopping.stop_reason, 'stagnation_x_tolerance') && nIter == kStar);
    nFail = nFail + chk([tag ': reported window values are the rule''s, recomputed from xPhys'], ...
        relEq(tel.stopping.final_stagnation_relative_change, rStar(1)) && ...
        relEq(tel.stopping.final_stagnation_grayness_range, rStar(2)) && ...
        isnan(tel.stopping.final_stagnation_objective_range) && ...
        tel.stopping.stagnation_relative_change_tol == relTol);
    nFail = nFail + fieldCheck(tag, cases{c,1}, X, tel.history.d_inf_design(1:nIter), true);
    reportBinding(tag, dRec, X, W, relTol, grayTol, kStar);
end

% ---- D. Yuksel -------------------------------------------------------------
% Third case: stage 1 hands over early on max|dx| < 0.05, so that stage 2 still
% moves under the density filter and its field can be identified (with both
% stages on stagnation_x, stage 1 runs to a design stage 2 no longer changes).
s1Early = struct('stage1Criterion', 'max_change', 'stage1Tol', 0.05);
cases = {'sensitivity', 5e-3, 1e-2, []; 'density', 5e-3, 1e-2, []; 'density', 5e-3, 1e-2, s1Early};
for c = 1:size(cases, 1)
    W = 10; relTol = cases{c,2}; grayTol = cases{c,3};
    stopY = struct('criterion', 'stagnation_x', 'window', W, 'relativeChangeTol', relTol, ...
        'graynessTol', grayTol, 'maxIters', 400);
    s1x = isempty(cases{c,4});   % stage 1 also on stagnation_x
    if ~s1x
        stopY.stage1Criterion = cases{c,4}.stage1Criterion;
        stopY.stage1Tol = cases{c,4}.stage1Tol;
    end
    mcfg = confbench_method_config('yuksel', 160, 20, '', stopY);
    mcfg.optimization.filter.type = cases{c,1};
    mcfg.postprocessing.record_design_history = true;
    mcfg.benchmark.record_history = true;   % recorder: design-variable increments
    [~, ~, ~, nIter, ~, nStage, tel] = run_topopt_from_json(mcfg);
    X = tel.design_history;
    n1 = nStage.stage1; n2 = nStage.stage2;
    [k1, r1] = oracleStop(X(:, 1:n1), W, relTol, grayTol);
    [k2, r2] = oracleStop(X(:, n1+1:end), W, relTol, grayTol);
    tag = sprintf('D Yuksel %s filter', cases{c,1});
    if ~s1x, tag = [tag ', stage 1 on max_change 0.05']; end
    fprintf('  %s: stage 1 %d (%s, oracle %d), stage 2 %d (%s, oracle %d)\n', tag, ...
        n1, tel.stopping.stage1_stop_reason, k1, n2, tel.stopping.stop_reason, k2);
    nFail = nFail + chk([tag ': design history holds one analysed design per iteration'], ...
        size(X, 2) == nIter && nIter == n1 + n2);
    if s1x
        nFail = nFail + chk([tag ': stage 1 stops exactly at its own rule on its own xPhys'], ...
            strcmp(tel.stopping.stage1_stop_reason, 'stagnation_x_tolerance') && n1 == k1);
    else
        nFail = nFail + chk([tag ': stage 1 keeps its own criterion'], ...
            strcmp(tel.stopping.stage1_stop_reason, 'density_change_tolerance'));
    end
    nFail = nFail + chk([tag ': stage 2 stops exactly at its own rule on its own xPhys'], ...
        strcmp(tel.stopping.stop_reason, 'stagnation_x_tolerance') && n2 == k2);
    nFail = nFail + chk([tag ': stage 2 window restarts at the handoff (>= W+2 stage-2 designs)'], ...
        n2 >= W + 2);
    nFail = nFail + chk([tag ': reported stage-2 window values are the rule''s'], ...
        relEq(tel.stopping.final_stagnation_relative_change, r2(1)) && ...
        relEq(tel.stopping.final_stagnation_grayness_range, r2(2)));
    % Without the reset the window could fire on stage-2 iteration k < W+2,
    % mixing stage-1 designs into it: report whether that would have happened.
    kJoint = oracleStopFrom(X, W, relTol, grayTol, n1 + 1);
    fprintf('      a window across the handoff would have stopped at stage-2 iteration %s\n', ...
        num2str(kJoint - n1));
    dDesign = tel.history.d_inf_design(1:nIter);
    if s1x
        nFail = nFail + fieldCheck([tag ' stage 1'], cases{c,1}, X(:, 1:n1), dDesign(1:n1), true);
    end
    nFail = nFail + fieldCheck([tag ' stage 2'], cases{c,1}, X(:, n1+1:end), dDesign(n1+1:end), ~s1x);
end

fprintf('%s\n  failures: %d\n\n', repmat('-',1,72), nFail);
end

% =========================================================================
function [kStar, r, d] = oracleStop(X, W, relTol, grayTol)
% The rule recomputed from the analysed designs X(:,j), j = 1..n of one stage.
kStar = oracleStopFrom(X, W, relTol, grayTol, 1);
[d, g] = series(X);
r = [NaN NaN];
if ~isnan(kStar)
    r = [max(d(kStar-W+1:kStar)), max(g(kStar-W:kStar)) - min(g(kStar-W:kStar))];
end
end

function kStar = oracleStopFrom(X, W, relTol, grayTol, kMin)
% First k >= kMin at which the rule holds over designs k-W..k of X (no reset).
[d, g] = series(X);
kStar = NaN;
for k = max(W + 2, kMin):size(X, 2)
    if max(d(k-W+1:k)) < relTol && max(g(k-W:k)) - min(g(k-W:k)) < grayTol
        kStar = k; return
    end
end
end

function [d, g] = series(X)
n = size(X, 2);
d = NaN(n, 1);
for j = 2:n
    d(j) = norm(X(:,j) - X(:,j-1)) / max(norm(X(:,j-1)), realmin);
end
g = mean(4*X.*(1-X), 1).';
end

function reportBinding(tag, d, X, W, relTol, grayTol, kStar)
% Informational: where a last-change-only or a no-grayness variant would stop.
[~, g] = series(X);
kLast = NaN; kNoGray = NaN;
for k = W + 2:size(X, 2)
    gr = max(g(k-W:k)) - min(g(k-W:k));
    if isnan(kLast) && d(k) < relTol && gr < grayTol, kLast = k; end
    if isnan(kNoGray) && max(d(k-W+1:k)) < relTol, kNoGray = k; end
end
fprintf('      k* = %d; last-change-only variant: %s; without the grayness test: %s\n', ...
    kStar, num2str(kLast), num2str(kNoGray));
end

function n = fieldCheck(tag, filterType, X, dDesign, mustMove)
% X(:,k) is the design analysed at iteration k of one stage; the recorder's
% dDesign(k) = max|x_{k+1} - x_k| is the design-variable update of the same
% iteration, which moves the analysed field from X(:,k) to X(:,k+1).  Under
% the sensitivity filter the analysed field IS x (the two series coincide);
% under the density filter it is the filtered field (they differ).
m = size(X, 2) - 1;
dPhys = max(abs(diff(X, 1, 2)), [], 1).';
gap = max(abs(dPhys - dDesign(1:m)));
if max(dDesign(1:m)) == 0
    % A stage whose design never moves cannot tell x from xPhys.
    if mustMove
        n = chk(sprintf('%s: the design must move to identify the field', tag), false);
    else
        fprintf('      %s: design did not move, field not identifiable here (NOTE)\n', tag);
        n = 0;
    end
    return
end
if strcmp(filterType, 'sensitivity')
    n = chk(sprintf('%s: analysed field = design variable x (max gap %.3g)', tag, gap), gap == 0);
else
    n = chk(sprintf('%s: analysed field = filtered xPhys ~= x (max gap %.3g)', tag, gap), gap > 1e-6);
end
end

function v = setEl(v, i, val)
v(i) = val;
end

function s = setField(s, f, v)
s.(f) = v;
end

function tf = relEq(a, b)
tf = isfinite(a) && isfinite(b) && abs(a - b) <= 1e-12 * max(abs(b), realmin);
end

function tf = throwsId(fn, id)
try
    fn(); tf = false;
catch ME
    tf = strcmp(ME.identifier, id);
    if ~tf, fprintf('      threw %s (expected %s)\n', ME.identifier, id); end
end
end

function n = chk(label, ok)
if ok, fprintf('  [PASS] %s\n', label); n = 0;
else,  fprintf('  [FAIL] %s\n', label); n = 1; end
end
