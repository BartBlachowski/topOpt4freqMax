function nFail = test_continuation()
%TEST_CONTINUATION  The three controllers are independent and each records its
%   own state, transitions and terminal state.
%
%   Fast structural checks.  The full trajectories are pinned bitwise by the
%   behavioural anchors; this suite exists so a break is caught in seconds.
nFail = 0;
ws = warning('off','olh:config:suspicious');  c = onCleanup(@() warning(ws));
N = 12;   % enough outer iterations to see the records, few enough to be quick

% ---- 1. fixed move: no move transitions at all --------------------------
cfg = olh.config.resolve('noDescentFixedMove','runtime.maxOuter',N,'runtime.diagnostics',true);
r = olhoffSolve(cfg);
nFail = nFail + chk('fixed move: the move limit never changes', all(r.hist.move == 0.04));
nFail = nFail + chk('fixed move: the ladder stage stays 1',      all(r.hist.stage == 1));

% ---- 2. ladder: stage and move are recorded and consistent ---------------
cfg = olh.config.resolve('duOlhoffFrozenM4','runtime.maxOuter',N,'runtime.diagnostics',true);
r = olhoffSolve(cfg);
lv = cfg.move.levels;
nFail = nFail + chk('ladder: hist.move is always the level of hist.stage', ...
    isequal(r.hist.move, lv(r.hist.stage)));
nFail = nFail + chk('ladder: the stage never decreases', all(diff(r.hist.stage) >= 0));

% ---- 3. p continuation: the two drivers are genuinely different ----------
cfgC = olh.config.resolve('pContinuationCoupled','runtime.maxOuter',N,'runtime.diagnostics',true);
rc = olhoffSolve(cfgC);
nFail = nFail + chk('coupled p: p is indexed by the ladder stage', ...
    isequal(rc.hist.pPen, cfgC.material.stiffness.continuation.schedule(min(rc.hist.stage,3))));
nFail = nFail + chk('coupled p: its own counter never advances', all(rc.hist.pStage == 1));
nFail = nFail + chk('coupled p: no interception events', all(rc.hist.pEvent == 0));

cfgD = olh.config.resolve('pContinuationDecoupled','runtime.maxOuter',N,'runtime.diagnostics',true);
rd = olhoffSolve(cfgD);
nFail = nFail + chk('decoupled p: p is indexed by its OWN counter', ...
    isequal(rd.hist.pPen, cfgD.material.stiffness.continuation.schedule(rd.hist.pStage)));
nFail = nFail + chk('decoupled p: pStage never decreases', all(diff(rd.hist.pStage) >= 0));

% ---- 4. blockStopUntilFinal is what keeps a low-p run going --------------
nFail = nFail + chk('p continuation: a run at p<final is never CONVERGED', ...
    ~(strcmp(rc.status,'CONVERGED') && rc.hist.pPen(end) < 3));

% ---- 5. mass continuation follows the p schedule, nothing else -----------
cfgM = olh.config.resolve('pMassCompatible','runtime.maxOuter',N,'runtime.diagnostics',true);
rm = olhoffSolve(cfgM);
pend = cfgM.material.stiffness.continuation.schedule(end);
nFail = nFail + chk('mass continuation: low-p model in force exactly while p<final', ...
    isequal(logical(rm.hist.massLow), rm.hist.pPen < pend));

% ---- 6. projection continuation is driven by the convergence event -------
cfgP = olh.config.resolve('projected','runtime.maxOuter',N,'runtime.diagnostics',true);
rp = olhoffSolve(cfgP);
lev = cfgP.projection.beta.levels;
nFail = nFail + chk('projection: hist.projBeta is the level of hist.projStage', ...
    isequal(rp.hist.projBeta, lev(rp.hist.projStage)));
nFail = nFail + chk('projection: projStage advances only on a projEvent', ...
    all(diff(rp.hist.projStage) == rp.hist.projEvent(1:end-1)));
nFail = nFail + chk('projection: an unfinished schedule cannot report CONVERGED', ...
    ~(strcmp(rp.status,'CONVERGED') && rp.hist.projStage(end) < numel(lev)));

% ---- 7. the controllers are independent ---------------------------------
nFail = nFail + chk('no projection controller state in a non-projection run', ...
    all(r.hist.projBeta == 0) && all(r.hist.projEvent == 0));
nFail = nFail + chk('no p controller state in a fixed-p run', ...
    all(r.hist.pPen == 3) && all(r.hist.pStage == 1) && all(r.hist.massLow == 0));
end

function n = chk(name, cond)
if cond
    fprintf('  ok   %s\n', name); n = 0;
else
    fprintf('  FAIL %s\n', name); n = 1;
end
end
