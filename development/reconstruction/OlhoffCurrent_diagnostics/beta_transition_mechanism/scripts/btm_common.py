"""btm_common -- loaders and the EXACT production stall predicate, re-derived.

The stall predicate here is transcribed from
+impl/architecture/+olh/+move/limit.m, case 'ladder', signal 'boundVariable':

    if numel(b) >= 2*W  and  (outer - lastStage) > W
        w2  = mean(b(end-W+1:end))          % the last W COMPLETED iterations
        w1  = mean(b(end-2*W+1:end-W))      % the W before those
        rel = (w2 - w1) / max(|w1|, eps)    % relative INCREASE of beta
        if rel < tol  ->  descend one rung

with W = 10 and tol = 5e-3 (schema defaults, both provenance class 'C').

Two details that matter and are easy to get wrong:

1. The controller is called BEFORE iteration `outer` is recorded, so `hist.beta`
   holds iterations 1..outer-1.  rel(outer) is therefore formed from COMPLETED
   iterations only.
2. `(outer - lastStage) > W` is a DWELL GUARD: after a stage change at iteration
   L, the earliest possible next descent is outer = L + W + 1.  With W = 10 that
   is L + 11.
"""
import csv, os

REPO = '/Users/piotrek/Programming/topOpt4freqMax'
DIAG = os.path.join(REPO, 'analysis/OlhoffCurrent/diagnostics')
OUT  = os.path.join(DIAG, 'beta_transition_mechanism')
W, TOL = 10, 5e-3


def _f(s):
    if s in (None, '', 'NaN'): return float('nan')
    return float(s)


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    R = {'n': len(rows), 'path': path}
    for k in rows[0].keys():
        R[k] = [_f(r[k]) for r in rows]
    R['outer'] = [int(x) for x in R['outer']]
    return R


def stall_rel(beta, W=W):
    """rel(outer) for outer = 1..n, using COMPLETED iterations 1..outer-1.

    Returns NaN where fewer than 2W completed iterations exist.  This is the
    raw metric; the dwell guard is applied separately by stall_fires().
    """
    n = len(beta)
    rel = [float('nan')]*n
    for k in range(n):            # k is 0-based index of outer = k+1
        nb = k                    # completed iterations available
        if nb < 2*W: continue
        b = beta[:nb]
        w2 = sum(b[-W:])/W
        w1 = sum(b[-2*W:-W])/W
        d = abs(w1) if abs(w1) > 2.22e-16 else 2.22e-16
        rel[k] = (w2 - w1)/d
    return rel


def stall_fires(beta, W=W, tol=TOL, nlevels=4):
    """Replay the production ladder exactly: returns (fires, stage, lastStage)."""
    n = len(beta)
    rel = stall_rel(beta, W)
    fires = [False]*n
    stage, last = 1, 0
    stages = [1]*n
    for k in range(n):
        outer = k + 1
        if rel[k] == rel[k] and (outer - last) > W and rel[k] < tol:
            if stage < nlevels:
                stage = min(stage+1, nlevels)
                last = outer
                fires[k] = True
            else:
                last = outer      # limit.m still updates lastStage at the floor
                fires[k] = True
        stages[k] = stage
    return fires, stages, rel


RUNS = {
    # production-ladder runs carried past the production stop (600 iters)
    '160x20': dict(NE=3200, mesh=(160, 20),
                   prod=os.path.join(DIAG, 'move_transition/runs/armP_160x20_iterations.csv'),
                   fixed=os.path.join(DIAG, 'move_stop/runs/fixedmove_160x20_iterations.csv')),
    '320x40': dict(NE=12800, mesh=(320, 40),
                   prod=os.path.join(DIAG, 'move_transition/runs/armP_320x40_iterations.csv'),
                   fixed=os.path.join(DIAG, 'move_stop/runs/fixedmove_320x40_iterations.csv')),
    '400x50': dict(NE=20000, mesh=(400, 50),
                   prod=os.path.join(DIAG, 'move_activity_400/runs/P400_400x50_iterations.csv'),
                   fixed=os.path.join(DIAG, 'move_activity_400/runs/F400_400x50_iterations.csv')),
}
