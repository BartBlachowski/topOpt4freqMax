"""mao_common -- loading and derived statistics for the offline move-activity study.

OFFLINE ONLY.  Nothing here runs an optimizer, touches +impl/, or writes into a
previous diagnostic directory.  Every quantity is either read verbatim from a
committed per-iteration CSV or derived from it by an identity stated below.

--------------------------------------------------------------------------
MOVE-INDEXING CONVENTION  (brief sec. 3)
--------------------------------------------------------------------------
Established by reading +impl/architecture/olhoffSolve.m, not assumed:

    line 267  [mvNow, mvState] = olh.move.limit(cfg, outer, hist, mvState)
    line 298  ctx = struct(..., 'move', mvNow, ...)      -> bounds drho
    line 341  dxOuter = max(abs(drho))
    line 342  dxNorm2 = norm(drho)
    line 373  hist.dxOuter(outer) = dxOuter
    line 380  hist.dxNorm2(outer) = dxNorm2
    line 381  hist.move(outer)    = mvNow

mvNow is computed BEFORE the update and is the box that bounded THAT update;
it is stored at the same index as the increment it bounded.  Therefore

    move(k) governs the transition rho(k-1) -> rho(k),

and u_e(k) = |rho_e(k)-rho_e(k-1)| / move(k) uses move(k), NOT move(k-1).
This matches mt_spatial.m (`d = abs(RHO(:,k)-prev)/P.move(k)`) and the recorded
`r_rho` column (`P.ratio = h.dxOuter./h.move`).  No off-by-one correction is
applied, and none is needed.

Because production has projection.enabled = false (scope-locked), the branch at
olhoffSolve line 325 is the non-projection branch, so `drho` IS the physical
density increment and dxOuter/dxNorm2 are formed on it.  Under projection this
would not hold; it does hold for every run used here.

--------------------------------------------------------------------------
WHAT IS EXACT AND WHAT IS A BOUND
--------------------------------------------------------------------------
Per-element density history (the RHO matrices) DOES NOT SURVIVE -- see
DATA_INVENTORY.md.  The surviving CSVs carry, per iteration, exactly two
functionals of the increment distribution:

    maxAbs(k) = max_e |drho_e(k)|
    l2(k)     = ||drho(k)||_2        (rms = l2/sqrt(NE), an identity, not new info)

Dividing by move(k) gives two EXACT statistics of the utilization u_e(k):

    maxU(k) = max_e u_e(k)          = maxAbs/move        [= recorded r_rho]
    rmsU(k) = sqrt(mean_e u_e(k)^2) = rms/move

From these two, one further quantity is EXACT and move-free:

    PARTICIPATION NUMBER
        N_eff(k) = (sum_e drho_e^2) / (max_e drho_e^2) = (l2(k)/maxAbs(k))^2

    Proof: sum_e drho_e^2 = l2^2 and max_e drho_e^2 = maxAbs^2.  The move cancels
    identically, so N_eff is invariant to the move level -- unlike r_rho, which
    the previous study found is NOT invariant near a move change.

    N_eff is the effective number of elements carrying the design increment.
    N_eff = 1 means one element holds all of it; N_eff = NE means all elements
    contribute equally.  phi_eff = N_eff/NE is its fraction form.

Anything about the SHAPE of the distribution between max and RMS is not exact.
The honest form is a one-sided Markov bound on u^2, which IS exact:

    frac(u >= t) <= rmsU^2 / t^2                     for any t > 0
    frac(u >= t*maxU) <= phi_eff / t^2               (same bound, relative form)

This CAPS the active fraction.  It cannot lower-bound it beyond the trivial
1/NE, because a distribution with one element at maxU and the rest arbitrarily
small is consistent with any (maxU, rmsU) pair satisfying rmsU^2 >= maxU^2/NE.
That asymmetry is stated wherever the bound is used, and no bound is ever
reported as if it were a measurement.

Percentiles (P75/P90/P95/P97.5/P99) are NOT recoverable per-iteration and are
NOT estimated here.  Element identity is not recoverable at all.
"""

import csv, os

REPO = '/Users/piotrek/Programming/topOpt4freqMax'
DIAG = os.path.join(REPO, 'analysis/OlhoffCurrent/diagnostics')
OUT  = os.path.join(DIAG, 'move_activity_offline')

# ---------------------------------------------------------------------------
# Run registry.  'policy' records what actually drove the move ladder.
#   prodLadder = production boundVariableStall detector (the thing under study)
#   fixedMove  = ladder disabled, move pinned at 0.04
#   maxUtil    = the previous study's ARM U gate (r_rho < 0.5 for 10 consecutive)
# 'prodStop' is the iteration production WOULD have stopped at; rows beyond it
# exist only because the arm was run with the stopping rule relaxed.
# ---------------------------------------------------------------------------
RUNS = [
  dict(key='ms_baseline_160x20',  study='move_stop',      csv='runs/baseline_160x20_iterations.csv',
       mesh=(160,20), policy='prodLadder', prodStop=91,  nactive=True),
  dict(key='ms_baseline_320x40',  study='move_stop',      csv='runs/baseline_320x40_iterations.csv',
       mesh=(320,40), policy='prodLadder', prodStop=131, nactive=True),
  dict(key='ms_fixedmove_160x20', study='move_stop',      csv='runs/fixedmove_160x20_iterations.csv',
       mesh=(160,20), policy='fixedMove',  prodStop=None, nactive=True),
  dict(key='ms_fixedmove_320x40', study='move_stop',      csv='runs/fixedmove_320x40_iterations.csv',
       mesh=(320,40), policy='fixedMove',  prodStop=None, nactive=True),
  dict(key='ar_unstopped_160x20', study='admission_rule', csv='runs/unstopped_160x20_iterations.csv',
       mesh=(160,20), policy='prodLadder', prodStop=91,  nactive=False),
  dict(key='ar_unstopped_320x40', study='admission_rule', csv='runs/unstopped_320x40_iterations.csv',
       mesh=(320,40), policy='prodLadder', prodStop=131, nactive=False),
  dict(key='mt_armP_160x20',      study='move_transition', csv='runs/armP_160x20_iterations.csv',
       mesh=(160,20), policy='prodLadder', prodStop=91,  nactive=False),
  dict(key='mt_armP_320x40',      study='move_transition', csv='runs/armP_320x40_iterations.csv',
       mesh=(320,40), policy='prodLadder', prodStop=131, nactive=False),
  dict(key='mt_armU_160x20',      study='move_transition', csv='runs/armU_160x20_iterations.csv',
       mesh=(160,20), policy='maxUtil',   prodStop=91,  nactive=False),
  dict(key='mt_armU_320x40',      study='move_transition', csv='runs/armU_320x40_iterations.csv',
       mesh=(320,40), policy='maxUtil',   prodStop=131, nactive=False),
]

# Diagnostic bins from brief sec. 4.  These are BINS, never controller thresholds.
TAUS = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]


def _f(s):
    if s is None or s == '' or s == 'NaN':
        return float('nan')
    return float(s)


def load(run):
    """Read one run's CSV and attach exactly-derived activity statistics."""
    path = os.path.join(DIAG, run['study'], run['csv'])
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    NE = run['mesh'][0] * run['mesh'][1]
    R = dict(run)
    R['NE'] = NE
    R['path'] = path
    R['n'] = len(rows)
    col = lambda name: [_f(r.get(name)) for r in rows]

    R['outer']  = [int(float(r['outer'])) for r in rows]
    R['move']   = col('move')
    R['stage']  = [int(float(r['stage'])) for r in rows]
    R['maxAbs'] = col('maxAbs')
    R['l2']     = col('l2')
    R['rms']    = col('rms')
    R['Mnd']    = col('Mnd_pct')
    R['gray']   = col('gray_frac')
    R['mid']    = col('mid_frac')
    R['omega1'] = col('omega1')
    R['volume'] = col('volume')
    R['beta']   = col('beta')
    R['descent'] = [int(float(r['moveDescent'])) for r in rows]

    # --- identity check: rms must equal l2/sqrt(NE) -------------------------
    ne_sqrt = NE ** 0.5
    R['rmsIdentityMaxErr'] = max(
        abs(R['rms'][k] - R['l2'][k]/ne_sqrt) / max(R['rms'][k], 1e-300)
        for k in range(R['n']))

    # --- EXACT derived statistics ------------------------------------------
    R['maxU'] = [R['maxAbs'][k]/R['move'][k] for k in range(R['n'])]
    R['rmsU'] = [R['rms'][k]/R['move'][k]    for k in range(R['n'])]
    # participation number: move cancels, so this is move-invariant by construction
    R['Neff'] = [(R['l2'][k]/R['maxAbs'][k])**2 if R['maxAbs'][k] > 0 else float('nan')
                 for k in range(R['n'])]
    R['phiEff'] = [R['Neff'][k]/NE for k in range(R['n'])]

    # --- exact one-sided Markov caps on the active fraction ----------------
    # frac(u >= tau) <= rmsU^2/tau^2, capped at 1 (and at maxU: zero above max).
    R['capFrac'] = {}
    for t in TAUS:
        R['capFrac'][t] = [
            0.0 if R['maxU'][k] < t else min(1.0, (R['rmsU'][k]**2)/(t*t))
            for k in range(R['n'])]

    # --- exact active counts, where the study recorded them ----------------
    if run['nactive']:
        R['nActive'] = {}
        for name, key in [('epsRMS','nActive_epsRMS'), ('1e-4','nActive_1e4'),
                          ('1e-3','nActive_1e3'), ('1e-2','nActive_1e2')]:
            R['nActive'][name] = [_f(r[key]) for r in rows]
        R['epsRMS'] = col('epsRMS')[0]
    return R


def descents(R):
    """Iterations at which the move actually descended, with production context."""
    ev = []
    for k in range(R['n']):
        if R['descent'][k]:
            ev.append(dict(iter=R['outer'][k],
                           moveFrom=R['move'][k-1] if k > 0 else None,
                           moveTo=R['move'][k],
                           beyondProductionStop=(R['prodStop'] is not None
                                                 and R['outer'][k] > R['prodStop'])))
    return ev


def window(R, k_iter, back=10):
    """Indices of the `back` iterations strictly preceding outer index k_iter."""
    i = R['outer'].index(k_iter)
    lo = max(0, i - back)
    return list(range(lo, i))
