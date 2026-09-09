#!/usr/bin/env python3
"""ml_analyze -- Phases 8-15.  Zero scientific runs; offline only.

Extracts the single-stage endpoint S at the first frozen A OR B exhaustion event,
decomposes the value bought by the lower rungs, and evaluates the preregistered
materiality thresholds.
"""
import os, sys, json, hashlib
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_frozen as F

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(STUDY))
CV = os.path.join(ROOT, 'diagnostics', 'two_branch_controller_validation')
EV = os.path.join(ROOT, 'evidence', 'two_branch_controller_validation')
LEVELS = [0.04, 0.02, 0.01, 0.005]
MESH = {'m160': ('C160x20', 160, 20), 'm320': ('C320x40', 320, 40), 'm400': ('C400x50', 400, 50)}

# preregistered materiality thresholds (PREREGISTRATION.md section 6)
TH = dict(Mnd_rel_pct=2.0, omega1_rel_pct=0.10, topo_frac=0.01, rho_mean_abs=0.01,
          volume_worsen=1e-5, cost_domination_mult=2.0)


def vechash(v):
    return hashlib.sha256(np.ascontiguousarray(v, dtype='<f8').tobytes()).hexdigest()


def load(tag):
    with h5py.File(os.path.join(EV, f'{tag}_trajectory.mat'), 'r') as h:
        RHO = np.array(h['RHO']).T
        hist = {k: np.array(h['hist'][k]) for k in
                ('dxNorm2', 'move', 'stage', 'omega', 'beta', 'nInner', 'cumInner',
                 'vol', 'gap12', 'dxOuter', 'tOuter', 'N', 'innerConv')}
        rho0 = float(np.array(h['cfg']['design']['initial']).ravel()[0])
    return RHO, hist, rho0


def state_at(T, RHO, hist, k, NE, label):
    """Every Phase-8 field at outer iteration k (1-indexed)."""
    i = k - 1
    om = hist['omega']                                  # (n, 5) in HDF5
    rho = RHO[:, i]
    return dict(
        label=label, iteration=int(k), move=float(T['move'][i]), stage=int(T['stage'][i]),
        omega1=float(om[i, 0]), omega2=float(om[i, 1]),
        gap12=float(T['gap12'][i]), volume=float(T['volume'][i]),
        Mnd=float(T['Mnd'][i]), gray=float(T['gray'][i]), mid=float(T['mid'][i]),
        maxAbsDrho=float(T['maxAbs'][i]), maxAbs_over_move=float(T['ratio'][i]),
        l2Drho=float(T['l2'][i]), rmsDrho=float(T['rms'][i]),
        cosTheta=float(T['exCos'][i]), netPath=float(T['exNet'][i]),
        medcos=float(T['exMedcos'][i]), mednet=float(T['exMednet'][i]),
        boundFrac=float(T['boundFrac'][i]),
        betaStallFires=bool(T['betaStallFires'][i]),
        nativeStopHolds=bool(T['prodStopRaw'][i]),
        nativeStopAdmitted=bool(T['prodStopAdmit'][i]),
        subspaceN=int(T['multN'][i]),
        innerCumulative=int(T['cumInner'][i]),
        wall_s_cumulative=float(np.sum(hist['tOuter'].ravel()[:k])),
        rho_sha256=vechash(rho))


def main():
    B = json.load(open(os.path.join(CV, 'evidence', 'baselines.json')))
    EVT = json.load(open(os.path.join(STUDY, 'evidence', 'event_verification.json')))
    out = {'thresholds': TH, 'mesh': {}}

    for key, (tag, nx, ny) in MESH.items():
        NE = nx * ny
        RHO, hist, rho0 = load(tag)
        T = np.genfromtxt(os.path.join(CV, 'runs', f'{tag}_iterations.csv'),
                          delimiter=',', names=True)
        rec = json.load(open(os.path.join(CV, 'runs', f'{tag}_record.json')))
        n = int(rec['nOuter'])
        kS = EVT[tag]['offline_declaration']            # verified, not assumed
        tOuter = hist['tOuter'].ravel()

        S = state_at(T, RHO, hist, kS, NE, 'S single-stage endpoint')
        Fst = state_at(T, RHO, hist, n, NE, 'F four-rung final')
        S['branch'] = EVT[tag]['offline_branch']
        S['window'] = EVT[tag]['offline_window']
        Fst['status'] = rec['status']
        Fst['terminalBranch'] = rec['terminalBranch'] or None

        # ---- Phase 9: what the lower rungs bought -----------------------
        drho = RHO[:, n - 1] - RHO[:, kS - 1]
        d = dict(
            dMnd=Fst['Mnd'] - S['Mnd'],
            dMnd_rel_pct=100 * (Fst['Mnd'] - S['Mnd']) / S['Mnd'],
            domega1=Fst['omega1'] - S['omega1'],
            domega1_rel_pct=100 * (Fst['omega1'] - S['omega1']) / S['omega1'],
            dgray=Fst['gray'] - S['gray'], dmid=Fst['mid'] - S['mid'],
            dgap12=Fst['gap12'] - S['gap12'],
            dvolume_abs=abs(Fst['volume'] - 0.5) - abs(S['volume'] - 0.5),
            rho_mean_abs=float(np.mean(np.abs(drho))),
            rho_rms=float(np.linalg.norm(drho) / np.sqrt(NE)),
            rho_max_abs=float(np.max(np.abs(drho))),
            extra_outer=n - kS,
            extra_inner=int(T['cumInner'][n - 1] - T['cumInner'][kS - 1]),
            extra_wall_s=float(np.sum(tOuter[kS:n])),
            frac_outer_after=(n - kS) / n,
            frac_inner_after=float((T['cumInner'][n - 1] - T['cumInner'][kS - 1]) / T['cumInner'][n - 1]),
            frac_wall_after=float(np.sum(tOuter[kS:n]) / np.sum(tOuter[:n])),
            cost_mult_vs_S=(n - kS) / kS)

        # ---- Phase 10: rung by rung -------------------------------------
        move = hist['move'].ravel()
        starts = [1] + [i + 1 for i in range(1, n) if move[i] != move[i - 1]]
        rungs = []
        for j, s0 in enumerate(starts):
            e0 = starts[j + 1] - 1 if j + 1 < len(starts) else n
            a, z = s0 - 1, e0 - 1
            prev = starts[j] - 2 if j > 0 else None      # state entering this rung
            base = prev if prev is not None and prev >= 0 else a
            rungs.append(dict(
                rung=j + 1, move=float(move[a]), iterFrom=int(s0), iterTo=int(e0),
                iterations=int(e0 - s0 + 1),
                inner=int(T['cumInner'][z] - (T['cumInner'][base] if j > 0 else 0)),
                wall_s=float(np.sum(tOuter[a:e0])),
                Mnd_from=float(T['Mnd'][base]), Mnd_to=float(T['Mnd'][z]),
                dMnd=float(T['Mnd'][z] - T['Mnd'][base]),
                dMnd_rel_pct=float(100 * (T['Mnd'][z] - T['Mnd'][base]) / T['Mnd'][base]),
                omega1_from=float(T['omega1'][base]), omega1_to=float(T['omega1'][z]),
                domega1_rel_pct=float(100 * (T['omega1'][z] - T['omega1'][base]) / T['omega1'][base]),
                dgray=float(T['gray'][z] - T['gray'][base]),
                rho_mean_abs=float(np.mean(np.abs(RHO[:, z] - RHO[:, base]))),
                completed=bool(j + 1 < len(starts) or rec['status'] == 'CONVERGED'),
                terminal_status=(rec['status'] if j + 1 == len(starts) else 'DESCENDED')))
            for r in rungs[-1:]:
                r['dMnd_per_1000_outer'] = r['dMnd'] / max(r['iterations'], 1) * 1000
                r['dMnd_per_1000_inner'] = r['dMnd'] / max(r['inner'], 1) * 1000
                r['dMnd_per_wall_min'] = r['dMnd'] / max(r['wall_s'] / 60.0, 1e-9)

        # ---- Phase 13/14: fraction of P->F benefit banked at S ----------
        b = B[key]
        tot_Mnd = b['Mnd'] - Fst['Mnd']
        at_S_Mnd = b['Mnd'] - S['Mnd']
        tot_om = Fst['omega1'] - b['omega1']
        at_S_om = S['omega1'] - b['omega1']
        banked = dict(
            P_Mnd=b['Mnd'], S_Mnd=S['Mnd'], F_Mnd=Fst['Mnd'],
            total_Mnd_gain=tot_Mnd, gain_at_S=at_S_Mnd,
            pct_Mnd_banked_at_S=100 * at_S_Mnd / tot_Mnd if tot_Mnd else float('nan'),
            P_omega1=b['omega1'], S_omega1=S['omega1'], F_omega1=Fst['omega1'],
            total_omega1_gain=tot_om, omega1_gain_at_S=at_S_om,
            pct_omega1_banked_at_S=100 * at_S_om / tot_om if tot_om else float('nan'))

        # ---- Phase 11: materiality verdict per criterion ---------------
        mat = dict(
            Mnd=bool(-d['dMnd_rel_pct'] >= TH['Mnd_rel_pct']),        # improvement = M_nd falls
            omega1=bool(d['domega1_rel_pct'] >= TH['omega1_rel_pct']),
            topology=bool(abs(d['dgray']) >= TH['topo_frac'] or abs(d['dmid']) >= TH['topo_frac']
                          or d['rho_mean_abs'] >= TH['rho_mean_abs']),
            volume=bool(d['dvolume_abs'] <= -TH['volume_worsen']),     # S worse than F by >= 1e-5
            multiplicity=bool(Fst['subspaceN'] != S['subspaceN'] or Fst['omega2'] <= Fst['omega1']
                              or not np.isfinite(Fst['omega1'])))
        mat['any'] = bool(any(mat.values()))
        mat['cost_dominated'] = bool(d['cost_mult_vs_S'] >= TH['cost_domination_mult'] and not mat['any'])
        mat['failure_risk'] = bool(rec['status'] == 'CAP_HIT')

        out['mesh'][key] = dict(mesh=[nx, ny], NE=NE, tag=tag, S=S, F=Fst,
                                production=b, lower_rung_delta=d, rungs=rungs,
                                banked=banked, materiality=mat)
        report(key, S, Fst, b, d, rungs, banked, mat)

    out['verdict_counts'] = dict(
        material_meshes=[k for k in out['mesh'] if out['mesh'][k]['materiality']['any']],
        n_material=sum(1 for k in out['mesh'] if out['mesh'][k]['materiality']['any']))
    json.dump(out, open(os.path.join(STUDY, 'evidence', 'ladder_analysis.json'), 'w'),
              indent=1, default=str)
    print('\nwrote evidence/ladder_analysis.json')
    return out


def report(key, S, Fst, b, d, rungs, banked, mat):
    print(f"\n{'='*78}\n{key}   S = single stage @ {S['iteration']} (branch {S['branch']}) "
          f"|  F = four rungs @ {Fst['iteration']} ({Fst['status']})\n{'='*78}")
    print(f"  {'':22} {'P production':>14} {'S single-stage':>16} {'F four-rung':>14}")
    for lbl, pv, sv, fv in [
            ('M_nd [%]', b['Mnd'], S['Mnd'], Fst['Mnd']),
            ('omega1', b['omega1'], S['omega1'], Fst['omega1']),
            ('omega2', b['omega2'], S['omega2'], Fst['omega2']),
            ('gap12', b['gap12'], S['gap12'], Fst['gap12']),
            ('gray', b['gray'], S['gray'], Fst['gray']),
            ('mid', b['mid'], S['mid'], Fst['mid']),
            ('volume', b['volume'], S['volume'], Fst['volume'])]:
        print(f"  {lbl:22} {pv:>14.6f} {sv:>16.6f} {fv:>14.6f}")
    print(f"  {'outer':22} {b['nOuter']:>14d} {S['iteration']:>16d} {Fst['iteration']:>14d}")
    print(f"  {'inner MMA':22} {b['innerTotal']:>14d} {S['innerCumulative']:>16d} {Fst['innerCumulative']:>14d}")
    print(f"  {'wall [s]':22} {b['wall_s']:>14.1f} {S['wall_s_cumulative']:>16.1f} {Fst['wall_s_cumulative']:>14.1f}")
    print(f"\n  lower rungs (0.02+0.01+0.005) bought:")
    print(f"    d M_nd    {d['dMnd']:+.4f} ({d['dMnd_rel_pct']:+.3f} % rel)   "
          f"[material if <= -{TH['Mnd_rel_pct']} %]  -> {'MATERIAL' if mat['Mnd'] else 'not material'}")
    print(f"    d omega1  {d['domega1']:+.6f} ({d['domega1_rel_pct']:+.4f} % rel) "
          f"[material if >= +{TH['omega1_rel_pct']} %] -> {'MATERIAL' if mat['omega1'] else 'not material'}")
    print(f"    d gray {d['dgray']:+.5f}  d mid {d['dmid']:+.5f}  mean|drho| {d['rho_mean_abs']:.5f} "
          f"-> {'MATERIAL' if mat['topology'] else 'not material'}")
    print(f"    volume feasibility change {d['dvolume_abs']:+.2e} -> "
          f"{'MATERIAL' if mat['volume'] else 'not material'}")
    print(f"    multiplicity S N={S['subspaceN']} F N={Fst['subspaceN']} -> "
          f"{'MATERIAL' if mat['multiplicity'] else 'not material'}")
    print(f"    cost: +{d['extra_outer']} outer (x{d['cost_mult_vs_S']:.2f} of S), "
          f"+{d['extra_inner']} inner, +{d['extra_wall_s']:.0f} s "
          f"({100*d['frac_wall_after']:.1f} % of run wall time)")
    print(f"    ANY material benefit: {mat['any']}   cost-dominated: {mat['cost_dominated']}   "
          f"failure risk: {mat['failure_risk']}")
    print(f"\n  P->F benefit already banked at S:  M_nd {banked['pct_Mnd_banked_at_S']:.2f} %   "
          f"omega1 {banked['pct_omega1_banked_at_S']:.2f} %")
    print(f"\n  rung decomposition:")
    print(f"    {'rung':<5}{'move':>7}{'iters':>7}{'inner':>8}{'wall s':>9}"
          f"{'dM_nd':>10}{'dM_nd %':>10}{'domega1 %':>11}  status")
    for r in rungs:
        print(f"    {r['rung']:<5}{r['move']:>7g}{r['iterations']:>7d}{r['inner']:>8d}"
              f"{r['wall_s']:>9.0f}{r['dMnd']:>10.4f}{r['dMnd_rel_pct']:>10.3f}"
              f"{r['domega1_rel_pct']:>11.4f}  {r['terminal_status']}")


if __name__ == '__main__':
    main()
