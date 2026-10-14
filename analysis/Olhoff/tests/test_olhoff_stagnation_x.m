function nFail = test_olhoff_stagnation_x()
%TEST_OLHOFF_STAGNATION_X  stop.rule = 'stagnation_x' in the Du-Olhoff solver.
%
%   'stagnation_x' is 'stagnation' with the omega_n test replaced by one on the
%   density field rho that assemble2D receives (the field entering stiffness,
%   mass and the eigenanalysis): over the W+1 designs analysed at outer k-W..k
%   (never the initial design), each of the W relative changes
%   ||rho_j - rho_{j-1}||_2/||rho_{j-1}||_2 must be < relativeChangeTolerance,
%   AND range(4*mean(rho(1-rho))) < graynessTolerance, as for 'stagnation'.
%
%   No solve unless stated.
%     1. schema: stop.rule admits 'stagnation_x'; one new row
%        stop.stagnation.relativeChangeTolerance (default 1e-3, class D)
%     2. olhoffcurrent_config 'StopStagnationX': true -> schema defaults, a
%        struct sets window / relativeChangeTolerance / graynessTolerance, every
%        guard off; an unknown field and a second Stop* option are refused
%     3. validate refuses stagnation_x with a guard and under projection, the
%        two cases refused for 'stagnation'
%     4. describe names the rule and its thresholds
%     5. SOLVE 160x20 (production preset, diagnostics recorder on): the designs
%        analysed at each outer iteration are replayed from the recorded
%        increments and must reproduce res.rho; the rule recomputed from them
%        stops exactly where the solver stopped, and every recorded window value
%        (aux.stagXRel, aux.stagMndRange) equals the recomputed one -- which
%        also identifies rho as the tested field and the W+1 / W window, the
%        max over the window and the grayness condition (a window value differs
%        from the recomputed one if any of these is wrong)
%   The 'stagnation' rule and the default rule keep their trajectories: see
%   tools/Matlab/tests/test_stop_strategies_unchanged.

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(root); addpath(here);
guard = olhoffcurrent_paths(); %#ok<NASGU>
PED = 'duOlhoffPedersenAdaptiveBoxSensitivityFiltered';

nFail = 0;
fprintf('\n%s\nTEST_OLHOFF_STAGNATION_X\n%s\n', repmat('=',1,72), repmat('=',1,72));

% ---- 1. schema ---------------------------------------------------------------
S = olh.config.schema();
iRule = find(strcmp(S(:,1), 'stop.rule'));
iRel  = find(strcmp(S(:,1), 'stop.stagnation.relativeChangeTolerance'));
nFail = nFail + chk('stop.rule admits stagnation and stagnation_x', ...
    all(ismember({'stagnation', 'stagnation_x'}, S{iRule, 4})));
nFail = nFail + chk('row stop.stagnation.relativeChangeTolerance: double, default 1e-3, class D', ...
    isscalar(iRel) && strcmp(S{iRel,2}, 'double') && S{iRel,3} == 1e-3 && strcmp(S{iRel,5}, 'D'));

% ---- 2. olhoffcurrent_config ---------------------------------------------------
c0 = olhoffcurrent_config(160, 20, 'Preset', PED);
cT = olhoffcurrent_config(160, 20, 'Preset', PED, 'StopStagnationX', true);
nFail = nFail + chk('default preset keeps stop.rule = designChange', strcmp(c0.stop.rule, 'designChange'));
nFail = nFail + chk('StopStagnationX true: rule stagnation_x, schema defaults 10 / 1e-3 / 5e-3', ...
    strcmp(cT.stop.rule, 'stagnation_x') && cT.stop.stagnation.window == 10 && ...
    cT.stop.stagnation.relativeChangeTolerance == 1e-3 && cT.stop.stagnation.graynessTolerance == 5e-3);
nFail = nFail + chk('StopStagnationX: every guard off', ...
    ~cT.stop.guards.settledMove && cT.stop.guards.boxInactiveFraction == 0 && ...
    ~cT.stop.guards.ladderExhausted && ~cT.stop.guards.maxDesignChange);
cS = olhoffcurrent_config(160, 20, 'Preset', PED, 'StopStagnationX', ...
    struct('window', 6, 'relativeChangeTolerance', 2e-3, 'graynessTolerance', 4e-3));
nFail = nFail + chk('StopStagnationX struct sets window, relativeChangeTolerance, graynessTolerance', ...
    cS.stop.stagnation.window == 6 && cS.stop.stagnation.relativeChangeTolerance == 2e-3 && ...
    cS.stop.stagnation.graynessTolerance == 4e-3 && cS.stop.stagnation.objectiveTolerance == 1e-3);
nFail = nFail + chk('StopStagnationX refuses an unknown field (objectiveTolerance is not read)', ...
    throwsId(@() olhoffcurrent_config(160, 20, 'Preset', PED, 'StopStagnationX', ...
        struct('objectiveTolerance', 1e-3)), 'olhoffcurrent_config:UnknownStagnationField'));
nFail = nFail + chk('StopStagnationX with StopStagnation is a conflict', ...
    throwsId(@() olhoffcurrent_config(160, 20, 'Preset', PED, 'StopStagnationX', true, ...
        'StopStagnation', true), 'olhoffcurrent_config:StopRuleConflict'));
nFail = nFail + chk('StopStagnationX with StopRelativeChangeTolerance is a conflict', ...
    throwsId(@() olhoffcurrent_config(160, 20, 'Preset', PED, 'StopStagnationX', true, ...
        'StopRelativeChangeTolerance', 1e-3), 'olhoffcurrent_config:StopRuleConflict'));

% ---- 3. validate ---------------------------------------------------------------
cG = cT; cG.stop.guards.settledMove = true;
nFail = nFail + chk('validate refuses stagnation_x with a stop guard', ...
    throwsId(@() olh.config.validate(cG), 'olh:config:stagnationWithGuard'));
% An otherwise valid projection configuration: density filter, a beta
% schedule, the MMA increment inner loop.  The control proves that only the
% stop rule is refused.
cP = cT; cP.projection.enabled = true; cP.filter.type = 'density';
if isempty(cP.projection.beta.levels), cP.projection.beta.levels = [1 2 4]; end
if strcmp(cP.optimizer.inner.type, 'lp'), cP.optimizer.inner.type = 'mma'; end
cP.optimizer.inner.variable = 'increment';
cC = cP; cC.stop.rule = 'designChange';
nFail = nFail + chk('validate refuses stagnation_x under projection (control: designChange accepted)', ...
    ~throwsAny(@() olh.config.validate(cC)) && ...
    throwsId(@() olh.config.validate(cP), 'olh:config:stagnationUnderProjection'));
nFail = nFail + chk('validate accepts the StopStagnationX configuration', ...
    ~throwsAny(@() olh.config.validate(cT)));

% ---- 4. describe -----------------------------------------------------------------
dT = strjoin(cellstr(olh.config.describe(cT)), newline);
nFail = nFail + chk('describe names STAGNATION_X and its thresholds', ...
    contains(dT, 'STAGNATION_X over the last 10+1 analysed designs') && contains(dT, '< 0.001'));

% ---- 5. solve 160x20 --------------------------------------------------------------
% Thresholds chosen so the rule fires inside the cap at 160x20; they exercise
% the mechanics and are not calibrated values.
W = 10; relTol = 5e-3; grayTol = 1e-2;
cfg = olhoffcurrent_config(160, 20, 'Preset', PED, 'Diagnostics', true, 'MaxOuter', 200, ...
    'StopStagnationX', struct('window', W, 'relativeChangeTolerance', relTol, 'graynessTolerance', grayTol));
res = olhoffSolve(cfg);
nOuter = numel(res.hist.tOuter);
rho = cfg.design.initial*ones(numel(res.rho), 1);
rhoHist = zeros(numel(rho), nOuter);
for k = 1:nOuter
    rhoHist(:,k) = rho;   % the design analysed (assembled, eigensolved) at outer k
    rho = min(1, max(cfg.design.minimum, rho + res.diag.drho{k}));
end
nFail = nFail + chk('replayed analysed designs reproduce res.rho', isequal(rho, res.rho));
d = NaN(nOuter, 1); m = mean(4*rhoHist.*(1-rhoHist), 1).';
for j = 2:nOuter
    d(j) = norm(rhoHist(:,j) - rhoHist(:,j-1))/max(norm(rhoHist(:,j-1)), realmin);
end
kStar = NaN; rRec = NaN(nOuter, 2);
for k = W + 2:nOuter
    rRec(k,:) = [max(d(k-W+1:k)), max(m(k-W:k)) - min(m(k-W:k))];
    if isnan(kStar) && rRec(k,1) < relTol && rRec(k,2) < grayTol, kStar = k; end
end
fprintf('  solver: %s after %d outer; recomputed rule first holds at outer %d\n', ...
    res.status, nOuter, kStar);
nFail = nFail + chk('solver stops exactly where the recomputed rule first holds', ...
    nOuter == kStar && contains(res.log{end}, 'stagnation_x'));
nFail = nFail + chk('window values NaN while incomplete (outer < W+2)', ...
    all(isnan(res.aux.stagXRel(1:W+1))) && all(isnan(res.aux.stagMndRange(1:W+1))));
nFail = nFail + chk('every recorded window value equals the one recomputed from the analysed rho', ...
    maxRel(res.aux.stagXRel(W+2:nOuter), rRec(W+2:nOuter,1)) <= 1e-12 && ...
    maxRel(res.aux.stagMndRange(W+2:nOuter), rRec(W+2:nOuter,2)) <= 1e-12);
nFail = nFail + chk('the omega_n window of stagnation is not created', ...
    ~isfield(res.aux, 'stagObjRange'));

fprintf('%s\n  failures: %d\n\n', repmat('-',1,72), nFail);
end

% =========================================================================
function e = maxRel(a, b)
e = max(abs(a(:) - b(:)) ./ max(abs(b(:)), realmin));
if isempty(e), e = Inf; end
end

function tf = throwsAny(fn)
try, fn(); tf = false; catch, tf = true; end
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
