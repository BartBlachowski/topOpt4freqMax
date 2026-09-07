#!/usr/bin/env python3
"""ma4_spatial -- 400x50 spatial and persistence analysis.

These questions were UNANSWERABLE in the prior offline study, because the
per-element trajectories of all three earlier runs had been deleted.  They are
answerable here only because the Phase-A retention mechanism kept the raw
history.  Brief sec. C13.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ma4_traj

OUT = os.path.join(ma4_traj.REPO, 'analysis/OlhoffCurrent/diagnostics/move_activity_400')
EPS = 8.83883476483184e-4
TAUS = {'1e-4': 1e-4, 'epsRMS': EPS, '1e-3': 1e-3, '1e-2': 1e-2}


def interface_adjacency(mask, nely, nelx):
    """Fraction of active elements having at least one INACTIVE 4-neighbour.

    A compact blob has a low value (only its rim qualifies); a thin front or a
    scattered dust has a high one.  Distinguishes 'localised moving front' from
    'diffuse activity' without assuming either.
    """
    m = mask.reshape(nelx, nely).T
    if m.sum() == 0:
        return float('nan')
    nb = np.zeros_like(m, dtype=bool)
    nb[:-1, :] |= ~m[1:, :];  nb[1:, :] |= ~m[:-1, :]
    nb[:, :-1] |= ~m[:, 1:];  nb[:, 1:] |= ~m[:, :-1]
    # domain edge counts as inactive neighbour
    edge = np.zeros_like(m, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    return float((m & (nb | edge)).sum() / m.sum())


def jaccard(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float('nan')


def main(descent_iter, mature_iter, mid_iter):
    T = ma4_traj.Traj('F')
    nely, nelx, NE = T.nely, T.nelx, T.NE
    R = {'schema': 'olhoff_move_activity_400_spatial/1',
         'NE': NE, 'mesh': [nelx, nely], 'nOuter': int(T.nOuter),
         'states': {'beforeProductionFirstDescent': descent_iter,
                    'representativeLaterFixedMove': mid_iter,
                    'matureOrCapEndpoint': mature_iter}}

    # ---- per-state spatial description ----------------------------------
    states = []
    for label, k in [('beforeProductionFirstDescent', descent_iter),
                     ('representativeLaterFixedMove', mid_iter),
                     ('matureOrCapEndpoint', mature_iter)]:
        d = np.abs(T.drho(k)); u = T.u(k); rho = T.rho(k)
        s = dict(label=label, iter=int(k), move=float(T.move[k-1]),
                 Mnd=float(100*np.mean(4*rho*(1-rho))),
                 u_max=float(u.max()), u_RMS=float(np.sqrt(np.mean(u**2))),
                 Neff=float((np.linalg.norm(d)/d.max())**2) if d.max() > 0 else None)
        for name, tau in TAUS.items():
            mask = d > tau
            n = int(mask.sum())
            e = dict(count=n, frac=n/NE,
                     interfaceAdjacency=interface_adjacency(mask, nely, nelx))
            if n:
                idx = np.flatnonzero(mask)
                col = idx // nely; row = idx % nely
                e['colSpanFrac'] = float((col.max()-col.min()+1)/nelx)
                e['rowSpanFrac'] = float((row.max()-row.min()+1)/nely)
                # how gray are the active elements vs the design as a whole?
                e['meanRhoActive'] = float(rho[mask].mean())
                e['grayFracActive'] = float(np.mean((rho[mask] > 0.1) & (rho[mask] < 0.9)))
                e['grayFracAll'] = float(np.mean((rho > 0.1) & (rho < 0.9)))
            s[f'active_{name}'] = e
        states.append(s)
    R['spatial_states'] = states

    # ---- persistence: the question the lost data could not answer --------
    # Element IDENTITY is available now, so "same elements or rotating?" is
    # measurable rather than speculative.
    pers = []
    for name, tau in [('1e-3', 1e-3), ('1e-2', 1e-2)]:
        for lo, hi, tag in [(descent_iter-9, descent_iter, 'window_before_descent'),
                            (mature_iter-9, mature_iter, 'window_at_endpoint')]:
            lo = max(2, lo)
            masks = [np.abs(T.drho(k)) > tau for k in range(lo, hi+1)]
            if len(masks) < 2: continue
            jac = [jaccard(masks[i], masks[i+1]) for i in range(len(masks)-1)]
            stack = np.vstack(masks)
            everActive = stack.any(axis=0).sum()
            alwaysActive = stack.all(axis=0).sum()
            occ = stack.sum(axis=0)
            births = int(np.sum(stack[1:] & ~stack[:-1]))
            deaths = int(np.sum(~stack[1:] & stack[:-1]))
            pers.append(dict(threshold=name, window=tag, iters=[int(lo), int(hi)],
                             meanPopulation=float(stack.sum(axis=1).mean()),
                             consecutiveJaccard_mean=float(np.mean(jac)),
                             consecutiveJaccard_min=float(np.min(jac)),
                             everActive=int(everActive), alwaysActive=int(alwaysActive),
                             alwaysOverEver=float(alwaysActive/everActive) if everActive else None,
                             meanOccupancyOfEverActive=float(occ[occ > 0].mean()) if everActive else None,
                             birthsPerIter=births/(len(masks)-1),
                             deathsPerIter=deaths/(len(masks)-1)))
    R['persistence'] = pers

    # ---- high-utilisation elements: persistent or rotating? -------------
    hi = []
    for thr in [0.5, 0.9]:
        lo = max(2, descent_iter-9)
        masks = [T.u(k) >= thr for k in range(lo, descent_iter+1)]
        stack = np.vstack(masks)
        ever = stack.any(axis=0).sum(); always = stack.all(axis=0).sum()
        jac = [jaccard(masks[i], masks[i+1]) for i in range(len(masks)-1)]
        hi.append(dict(uThreshold=thr, window=[int(lo), int(descent_iter)],
                       meanPopulation=float(stack.sum(axis=1).mean()),
                       everActive=int(ever), alwaysActive=int(always),
                       consecutiveJaccard_mean=float(np.mean(jac)) if jac else None))
    R['high_utilisation_persistence'] = hi

    with open(os.path.join(OUT, 'SPATIAL_ANALYSIS.json'), 'w') as f:
        json.dump(R, f, indent=2)
    print('wrote SPATIAL_ANALYSIS.json')
    for s in states:
        a = s['active_1e-3']
        print(f"  {s['label']:32s} it={s['iter']:4d} Mnd={s['Mnd']:6.2f} "
              f"n>1e-3={a['count']:6d} ({100*a['frac']:5.2f}%) "
              f"ifaceAdj={a['interfaceAdjacency']:.3f}")
    T.close()
    return R


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
