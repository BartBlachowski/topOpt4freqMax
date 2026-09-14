#!/usr/bin/env python3
"""ma4_scaling -- three-mesh active-count scaling analysis.

Implements PREREGISTRATION.md sections 7-9 exactly.  Nothing here was chosen
after seeing the 400x50 trajectory.

COMPARABILITY.  The active-set thresholds are on the RAW increment
|Delta rho_e| and are identical at all three meshes (epsRMS = 8.83883476483184e-4
is mesh-invariant because stop.toleranceRule='meshScaled' makes the tolerance
proportional to sqrt(NE)).  So the counts are directly comparable with no
rescaling -- which is exactly why this threshold convention was preserved rather
than switching to thresholds on u.
"""
import csv, json, math, os

REPO = '/Users/piotrek/Programming/topOpt4freqMax'
DIAG = os.path.join(REPO, 'analysis/OlhoffCurrent/diagnostics')
OUT  = os.path.join(DIAG, 'move_activity_400')
TAUS = ['1e-4', 'epsRMS', '1e-3', '1e-2']
COL  = {'1e-4': 'nActive_1e4', 'epsRMS': 'nActive_epsRMS',
        '1e-3': 'nActive_1e3', '1e-2': 'nActive_1e2'}
CGRID = [0.90, 0.95, 0.99]          # preregistered matched-maturity levels

# fixed-move (counterfactual) run per mesh, and the production run per mesh
MESHES = [
    dict(mesh='160x20', NE=3200,
         fixed=os.path.join(DIAG, 'move_stop/runs/fixedmove_160x20_iterations.csv'),
         prod =os.path.join(DIAG, 'move_stop/runs/baseline_160x20_iterations.csv')),
    dict(mesh='320x40', NE=12800,
         fixed=os.path.join(DIAG, 'move_stop/runs/fixedmove_320x40_iterations.csv'),
         prod =os.path.join(DIAG, 'move_stop/runs/baseline_320x40_iterations.csv')),
    dict(mesh='400x50', NE=20000,
         fixed=os.path.join(OUT, 'runs/F400_400x50_iterations.csv'),
         prod =os.path.join(OUT, 'runs/P400_400x50_iterations.csv')),
]


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


def completion(R):
    """Preregistered: c(k) = (Mnd(1)-Mnd(k)) / (Mnd(1)-Mnd(end))."""
    M = R['Mnd_pct']; M0, Mf = M[0], M[-1]
    return [(M0 - M[k]) / (M0 - Mf) for k in range(R['n'])]


def monotone(c):
    out, m = [], -1e9
    for x in c:
        m = max(m, x); out.append(m)
    return out


def first_at(cm, target):
    for i, x in enumerate(cm):
        if x >= target: return i
    return len(cm) - 1


def first_descent(Rp):
    """Iteration index (0-based) of the LAST iteration at move=0.04."""
    for k in range(1, Rp['n']):
        if Rp['move'][k] < Rp['move'][k-1]:
            return k - 1, Rp['outer'][k]
    return None, None


def pairwise_alpha(NEa, Na, NEb, Nb):
    if Na <= 0 or Nb <= 0: return float('nan')
    return math.log(Nb/Na) / math.log(NEb/NEa)


def loglog_fit(NEs, Ns):
    """Least squares log N = log C + alpha log NE.  Returns C, alpha, resid."""
    pts = [(math.log(ne), math.log(n)) for ne, n in zip(NEs, Ns) if n > 0]
    if len(pts) < 2: return float('nan'), float('nan'), float('nan')
    m = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0]**2 for p in pts); sxy = sum(p[0]*p[1] for p in pts)
    den = m*sxx - sx*sx
    alpha = (m*sxy - sx*sy)/den
    lc = (sy - alpha*sx)/m
    resid = math.sqrt(sum((y - (lc + alpha*x))**2 for x, y in pts)/m)
    return math.exp(lc), alpha, resid


def main():
    M = {}
    for spec in MESHES:
        if not os.path.exists(spec['fixed']):
            raise SystemExit(f"missing {spec['fixed']} -- run F400 first")
        F = load(spec['fixed']); P = load(spec['prod'])
        cm = monotone(completion(F))
        i_desc, desc_iter = first_descent(P)
        rec = dict(spec)
        rec.update(F=F, P=P, cmono=cm, descentIdx=i_desc, descentIter=desc_iter)
        M[spec['mesh']] = rec

    R = {'schema': 'olhoff_move_activity_400_scaling/1',
         'preregistration': 'PREREGISTRATION.md',
         'thresholdConvention': 'absolute on |drho_e|; epsRMS=8.83883476483184e-4 identical at all meshes',
         'matchedMaturityGrid': CGRID}

    # ---- remaining evolution at each mesh's production first descent ------
    rem = []
    for mesh in ['160x20', '320x40', '400x50']:
        m = M[mesh]; F = m['F']; i = m['descentIdx']
        Mi, Mf = F['Mnd_pct'][i], F['Mnd_pct'][-1]
        rem.append(dict(mesh=mesh, NE=m['NE'], descentIter=m['descentIter'],
                        lastIterAtMove004=F['outer'][i],
                        Mnd_at_descent=Mi, Mnd_fixedMoveFinal=Mf,
                        remaining_absolute=Mi - Mf,
                        remaining_relative=(Mi - Mf)/Mi,
                        completion_at_descent=m['cmono'][i],
                        omega1_at_descent=F['omega1'][i],
                        omega1_fixedMoveFinal=F['omega1'][-1],
                        omega1_relChange=abs(F['omega1'][-1]-F['omega1'][i])/F['omega1'][i],
                        gray_at_descent=F['gray_frac'][i], mid_at_descent=F['mid_frac'][i]))
    R['remaining_evolution'] = rem

    # ---- scaling at matched maturity (PRIMARY) ---------------------------
    scal = []
    for tau in TAUS:
        col = COL[tau]
        for c in CGRID:
            NEs, Ns, per = [], [], []
            for mesh in ['160x20', '320x40', '400x50']:
                m = M[mesh]; k = first_at(m['cmono'], c)
                n = m['F'][col][k]
                NEs.append(m['NE']); Ns.append(n)
                per.append(dict(mesh=mesh, NE=m['NE'], iter=m['F']['outer'][k],
                                c=m['cmono'][k], N_active=n, frac=n/m['NE']))
            a12 = pairwise_alpha(NEs[0], Ns[0], NEs[1], Ns[1])
            a23 = pairwise_alpha(NEs[1], Ns[1], NEs[2], Ns[2])
            a13 = pairwise_alpha(NEs[0], Ns[0], NEs[2], Ns[2])
            C, ag, res = loglog_fit(NEs, Ns)
            pw = [a for a in (a12, a23, a13) if a == a]
            scal.append(dict(threshold=tau, c=c, points=per,
                             alpha_160_320=a12, alpha_320_400=a23, alpha_160_400=a13,
                             alpha_global=ag, C_global=C, logResidual=res,
                             pairwise_spread=(max(pw)-min(pw)) if pw else float('nan')))
    R['scaling_matched_maturity'] = scal

    # ---- scaling at each mesh's own production first descent (SECONDARY) --
    sec = []
    for tau in TAUS:
        col = COL[tau]; NEs, Ns, per = [], [], []
        for mesh in ['160x20', '320x40', '400x50']:
            m = M[mesh]; i = m['descentIdx']; n = m['F'][col][i]
            NEs.append(m['NE']); Ns.append(n)
            per.append(dict(mesh=mesh, NE=m['NE'], iter=m['F']['outer'][i],
                            c=m['cmono'][i], N_active=n, frac=n/m['NE']))
        C, ag, res = loglog_fit(NEs, Ns)
        sec.append(dict(threshold=tau, points=per,
                        alpha_160_320=pairwise_alpha(NEs[0], Ns[0], NEs[1], Ns[1]),
                        alpha_320_400=pairwise_alpha(NEs[1], Ns[1], NEs[2], Ns[2]),
                        alpha_160_400=pairwise_alpha(NEs[0], Ns[0], NEs[2], Ns[2]),
                        alpha_global=ag, C_global=C, logResidual=res,
                        note='meshes are at DIFFERENT maturity here; descriptive only'))
    R['scaling_at_production_descent'] = sec

    # ---- preregistered verdict ------------------------------------------
    per_tau_spread = {}
    global_alphas = {}
    for tau in TAUS:
        rows = [s for s in scal if s['threshold'] == tau]
        sp = [s['pairwise_spread'] for s in rows if s['pairwise_spread'] == s['pairwise_spread']]
        ga = [s['alpha_global'] for s in rows if s['alpha_global'] == s['alpha_global']]
        per_tau_spread[tau] = max(sp) if sp else float('nan')
        global_alphas[tau] = (min(ga), max(ga)) if ga else (float('nan'),)*2
    allg = [v for t in TAUS for v in global_alphas[t] if v == v]
    spread_tau = (max(allg) - min(allg)) if allg else float('nan')
    worst_pair = max((v for v in per_tau_spread.values() if v == v), default=float('nan'))

    if worst_pair <= 0.10 and spread_tau <= 0.10:
        verdict = 'SINGLE_POWER_LAW_ACTIVITY_SCALING_SUPPORTED'
    elif worst_pair <= 0.25 and spread_tau <= 0.25:
        verdict = 'APPROXIMATE_POWER_LAW_FAMILY_SUPPORTED_BUT_ALPHA_UNCERTAIN'
    else:
        verdict = 'SINGLE_POWER_LAW_NOT_SUPPORTED'

    R['verdict_inputs'] = dict(worst_pairwise_spread=worst_pair,
                               spread_across_thresholds=spread_tau,
                               per_threshold_worst_spread=per_tau_spread,
                               global_alpha_range_per_threshold=
                                 {t: list(global_alphas[t]) for t in TAUS})
    R['scaling_verdict'] = verdict

    with open(os.path.join(OUT, 'SCALING_ANALYSIS.json'), 'w') as f:
        json.dump(R, f, indent=2)
    print('wrote SCALING_ANALYSIS.json')
    print(f"\n  worst pairwise spread   = {worst_pair:.4f}")
    print(f"  spread across thresholds = {spread_tau:.4f}")
    print(f"  VERDICT: {verdict}\n")
    for r in rem:
        print(f"  {r['mesh']}: descent it {r['descentIter']}, remaining "
              f"{r['remaining_absolute']:.3f} pts = {100*r['remaining_relative']:.1f}%  "
              f"(c={r['completion_at_descent']:.3f})")
    return R


if __name__ == '__main__':
    main()
