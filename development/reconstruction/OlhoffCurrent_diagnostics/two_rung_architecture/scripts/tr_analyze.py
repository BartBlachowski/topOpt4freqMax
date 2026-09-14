#!/usr/bin/env python3
"""tr_analyze -- Phases 5-16.  Zero scientific runs; offline only.

Extracts P / S1 / S2 / F, decomposes rung 2 (S1->S2) against rungs 3+4 (S2->F),
evaluates the preregistered materiality thresholds and production-relative
acceptance gates, and computes the cost savings of terminating at S2.
"""
import os, sys, json, hashlib
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tr_frozen as F

HERE  = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT  = os.path.dirname(os.path.dirname(STUDY))
REPO  = os.path.dirname(os.path.dirname(ROOT))
CV    = os.path.join(ROOT, 'diagnostics', 'two_branch_controller_validation')
EV    = os.path.join(ROOT, 'evidence', 'two_branch_controller_validation')
MESH  = {'m160': ('C160x20', 160, 20), 'm320': ('C320x40', 320, 40), 'm400': ('C400x50', 400, 50)}

# ---- PREREGISTRATION S9, inherited verbatim from move_ladder_necessity S6 ----
TH = dict(Mnd_rel_pct=2.0, omega1_rel_pct=0.10, topo_frac=0.01, rho_mean_abs=0.01,
          volume_worsen=1e-5, cost_domination_mult=2.0)

# ---- PREREGISTRATION S10, inherited verbatim from the controller study S11 ---
GATES = {
    'm160': dict(Mnd_max_mult=1.10, omega1_min_mult=0.99),
    'm320': dict(Mnd_max_mult=0.80, omega1_min_mult=0.99),
    'm400': dict(Mnd_max_mult=0.80, omega1_min_mult=0.99)}
VOL_TOL = 1e-4
OUTER_MULT_MAX, WALL_MULT_MAX = 8.0, 10.0


def vechash(v):
    return hashlib.sha256(np.ascontiguousarray(v, dtype='<f8').tobytes()).hexdigest()


def load(tag):
    with h5py.File(os.path.join(EV, f'{tag}_trajectory.mat'), 'r') as h:
        RHO  = np.array(h['RHO']).T
        hist = {k: np.array(h['hist'][k]).ravel() for k in ('tOuter',)}
        rho0 = float(np.array(h['cfg']['design']['initial']).ravel()[0])
    return RHO, hist, rho0


def state_at(T, RHO, tOuter, k, NE, label):
    i = k - 1
    return dict(
        label=label, iteration=int(k), move=float(T['move'][i]), stage=int(T['stage'][i]),
        omega1=float(T['omega1'][i]), omega2=float(T['omega2'][i]),
        gap12=float(T['gap12'][i]), volume=float(T['volume'][i]), volErr=float(T['volErr'][i]),
        Mnd=float(T['Mnd'][i]), gray=float(T['gray'][i]), mid=float(T['mid'][i]),
        maxAbsDrho=float(T['maxAbs'][i]), maxAbs_over_move=float(T['ratio'][i]),
        l2Drho=float(T['l2'][i]), rmsDrho=float(T['rms'][i]),
        l2_over_tol=float(T['l2'][i]) / F.tol_for(NE),
        cosTheta=float(T['exCos'][i]), netPath=float(T['exNet'][i]),
        medcos=float(T['exMedcos'][i]), mednet=float(T['exMednet'][i]),
        A=bool(T['exA'][i]), B=bool(T['exB'][i]), E=bool(T['exE'][i]),
        nA=int(T['exNA'][i]), nB=int(T['exNB'][i]), declared=bool(T['exDecl'][i]),
        exStageStart=int(T['exStageStart'][i]),
        boundFrac=float(T['boundFrac'][i]),
        betaStallRel=(None if np.isnan(T['betaStallRel'][i]) else float(T['betaStallRel'][i])),
        betaStallFires=bool(T['betaStallFires'][i]),
        nativeStopHolds=bool(T['prodStopRaw'][i]),
        nativeStopAdmitted=bool(T['prodStopAdmit'][i]),
        subspaceN=int(T['multN'][i]), degen=int(T['degen'][i]),
        nInner=int(T['nInner'][i]), innerCumulative=int(T['cumInner'][i]),
        wall_s_cumulative=float(np.sum(tOuter[:k])),
        rho_sha256=vechash(RHO[:, i]))


def delta(a, b, RHO, NE, label):
    """b - a, with density-field distance."""
    d = RHO[:, b['iteration'] - 1] - RHO[:, a['iteration'] - 1]
    return dict(
        block=label, frm=a['label'], to=b['label'],
        dMnd=b['Mnd'] - a['Mnd'],
        dMnd_rel_pct=100 * (b['Mnd'] - a['Mnd']) / a['Mnd'],
        domega1=b['omega1'] - a['omega1'],
        domega1_rel_pct=100 * (b['omega1'] - a['omega1']) / a['omega1'],
        domega2=b['omega2'] - a['omega2'],
        dgap12=b['gap12'] - a['gap12'],
        dgray=b['gray'] - a['gray'], dmid=b['mid'] - a['mid'],
        dvolume_abs=abs(b['volume'] - 0.5) - abs(a['volume'] - 0.5),
        rho_mean_abs=float(np.mean(np.abs(d))),
        rho_rms=float(np.linalg.norm(d) / np.sqrt(NE)),
        rho_max_abs=float(np.max(np.abs(d))),
        d_outer=b['iteration'] - a['iteration'],
        d_inner=b['innerCumulative'] - a['innerCumulative'],
        d_wall_s=b['wall_s_cumulative'] - a['wall_s_cumulative'],
        dSubspaceN=b['subspaceN'] - a['subspaceN'])


def materiality(d):
    """Preregistered S9 bars.  Improvement direction: M_nd DOWN, omega1 UP."""
    return dict(
        Mnd=bool(-d['dMnd_rel_pct'] >= TH['Mnd_rel_pct']),
        omega1=bool(d['domega1_rel_pct'] >= TH['omega1_rel_pct']),
        topology=bool(abs(d['dgray']) >= TH['topo_frac'] or abs(d['dmid']) >= TH['topo_frac']
                      or d['rho_mean_abs'] >= TH['rho_mean_abs']),
        volume=bool(d['dvolume_abs'] <= -TH['volume_worsen']),
        multiplicity=bool(d['dSubspaceN'] != 0),
        any=None)


def main():
    B = json.load(open(os.path.join(CV, 'evidence', 'baselines.json')))
    EVT = json.load(open(os.path.join(STUDY, 'evidence', 'event_verification.json')))
    out = {'thresholds': TH, 'gates_spec': dict(GATES=GATES, VOL_TOL=VOL_TOL,
           OUTER_MULT_MAX=OUTER_MULT_MAX, WALL_MULT_MAX=WALL_MULT_MAX), 'mesh': {}}

    for key, (tag, nx, ny) in MESH.items():
        NE = nx * ny
        RHO, hist, rho0 = load(tag)
        tOuter = hist['tOuter']
        T = np.genfromtxt(os.path.join(CV, 'runs', f'{tag}_iterations.csv'),
                          delimiter=',', names=True)
        rec = json.load(open(os.path.join(CV, 'runs', f'{tag}_record.json')))
        n = int(rec['nOuter'])
        V = EVT[tag]
        kE1, kE2 = V['kE1'], V['kE2']

        S1 = state_at(T, RHO, tOuter, kE1, NE, 'S1 single-stage endpoint (move=0.04)')
        S2 = state_at(T, RHO, tOuter, kE2, NE, 'S2 two-rung endpoint (move=0.02)')
        Fs = state_at(T, RHO, tOuter, n,   NE, 'F four-rung final')
        S1['branch'] = V['kE1_branch']; S1['window'] = V['stages'][0]['offline_window']
        S2['branch'] = V['kE2_branch']; S2['window'] = V['stages'][1]['offline_window']
        Fs['status'] = rec['status']
        Fs['terminalBranch'] = rec['exhaustion'].get('terminalBranch') or None
        S1['policy_status'] = 'CONVERGED (single-stage policy would terminate here)'
        S2['policy_status'] = 'CONVERGED (two-rung policy would terminate here)'

        P = B[key]

        # ---- rung decomposition -----------------------------------------
        rung2  = delta(S1, S2, RHO, NE, 'RUNG 2  (move=0.02):  S1 -> S2')
        rung34 = delta(S2, Fs, RHO, NE, 'RUNGS 3+4 (0.01, 0.005):  S2 -> F')
        s1_to_F = delta(S1, Fs, RHO, NE, 'RUNGS 2+3+4:  S1 -> F')
        m2, m34 = materiality(rung2), materiality(rung34)
        m2['any']  = any(v for k, v in m2.items() if k != 'any')
        m34['any'] = any(v for k, v in m34.items() if k != 'any')

        # ---- S2 vs production -------------------------------------------
        vsP = {}
        for st, nm in ((S1, 'S1'), (S2, 'S2'), (Fs, 'F')):
            vsP[nm] = dict(
                dMnd=st['Mnd'] - P['Mnd'],
                dMnd_rel_pct=100 * (st['Mnd'] - P['Mnd']) / P['Mnd'],
                domega1=st['omega1'] - P['omega1'],
                domega1_rel_pct=100 * (st['omega1'] - P['omega1']) / P['omega1'],
                dgray=st['gray'] - P['gray'], dmid=st['mid'] - P['mid'],
                dgap12=st['gap12'] - P['gap12'],
                outer_mult=st['iteration'] / P['nOuter'],
                inner_mult=st['innerCumulative'] / P['innerTotal'],
                wall_mult=st['wall_s_cumulative'] / P['wall_s'])

        # ---- preregistered acceptance gates at S2 -----------------------
        g = GATES[key]
        A = {}
        A['A_Mnd_bound']    = bool(S2['Mnd'] <= g['Mnd_max_mult'] * P['Mnd'])
        A['A4_omega1_0.99'] = bool(S2['omega1'] >= g['omega1_min_mult'] * P['omega1'])
        A['A5_volume']      = bool(abs(S2['volume'] - 0.5) <= VOL_TOL)
        # A6 is the preregistered P10 verbatim: subspace size 2 throughout, omega2 >
        # omega1, all omega finite, no NaN/Inf, no non-converged inner solve.  It
        # says nothing about `degen`, which counts EXPECTED near-degeneracy hits in
        # the multiplicity-aware subspace and is reported descriptively only.
        # Evaluated over the two-rung PREFIX [1..kE2] -- the only iterations the
        # two-rung policy would execute.
        pre = slice(0, kE2)
        A['A6_physics']     = bool(
            np.all(T['multN'][pre] == 2) and np.all(T['omega2'][pre] > T['omega1'][pre])
            and np.all(np.isfinite(T['omega1'][pre])) and np.all(np.isfinite(T['omega2'][pre]))
            and np.all(np.isfinite(T['Mnd'][pre])) and np.all(T['innerConv'][pre] == 1))
        A['A7_outer_mult']  = bool(vsP['S2']['outer_mult'] <= OUTER_MULT_MAX)
        A['A7_wall_mult']   = bool(vsP['S2']['wall_mult'] <= WALL_MULT_MAX)
        A['A8_terminal_persistence'] = bool(S2['declared'] and max(S2['nA'], S2['nB']) >= 20
                                            and S2['move'] == 0.02)
        if key == 'm160':
            A['A9_no_regression'] = bool(S2['omega1'] >= P['omega1'])
        A['all'] = all(v for v in A.values() if isinstance(v, bool))

        # same gates evaluated at S1, for the comparison the brief demands
        A1 = {}
        A1['A_Mnd_bound']    = bool(S1['Mnd'] <= g['Mnd_max_mult'] * P['Mnd'])
        A1['A4_omega1_0.99'] = bool(S1['omega1'] >= g['omega1_min_mult'] * P['omega1'])
        if key == 'm160':
            A1['A9_no_regression'] = bool(S1['omega1'] >= P['omega1'])

        # ---- physics over the whole window (Phase 15) -------------------
        sl = slice(0, n)
        phys = dict(
            subspaceN_always2=bool(np.all(T['multN'][sl] == 2)),
            degenTotal_full=float(np.sum(T['degen'][sl])),
            degenTotal_prefix=float(np.sum(T['degen'][0:kE2])),
            degen_note='degen counts EXPECTED near-degeneracy hits in the '
                       'multiplicity-aware subspace; it is not a failure indicator '
                       'and is not part of the preregistered P10/A6 gate',
            subspaceN_prefix_always2=bool(np.all(T['multN'][0:kE2] == 2)),
            innerNonConv_prefix=int(np.sum(T['innerConv'][0:kE2] == 0)),
            gap12_min_full=float(np.min(T['gap12'][sl])),
            omega2_gt_omega1_always=bool(np.all(T['omega2'][sl] > T['omega1'][sl])),
            omega_finite=bool(np.all(np.isfinite(T['omega1'][sl])) and np.all(np.isfinite(T['omega2'][sl]))),
            min_gap12_S2_to_F=float(np.min(T['gap12'][kE2 - 1:n])),
            innerNonConv_total=int(np.sum(T['innerConv'][sl] == 0)))

        # ---- cost of terminating at S2 (Phase 13) -----------------------
        cost = dict(
            total_outer=n, total_inner=int(Fs['innerCumulative']),
            total_wall_s=float(np.sum(tOuter[:n])),
            S2_outer=kE2, S2_inner=int(S2['innerCumulative']),
            S2_wall_s=S2['wall_s_cumulative'],
            saved_outer=n - kE2, saved_inner=int(Fs['innerCumulative'] - S2['innerCumulative']),
            saved_wall_s=float(Fs['wall_s_cumulative'] - S2['wall_s_cumulative']),
            saved_outer_pct=100 * (n - kE2) / n,
            saved_inner_pct=100 * (Fs['innerCumulative'] - S2['innerCumulative']) / Fs['innerCumulative'],
            saved_wall_pct=100 * (Fs['wall_s_cumulative'] - S2['wall_s_cumulative']) / Fs['wall_s_cumulative'],
            rung2_outer=kE2 - kE1, rung2_inner=int(S2['innerCumulative'] - S1['innerCumulative']),
            rung2_wall_s=S2['wall_s_cumulative'] - S1['wall_s_cumulative'],
            rung34_outer=n - kE2, rung34_inner=int(Fs['innerCumulative'] - S2['innerCumulative']),
            rung34_wall_s=Fs['wall_s_cumulative'] - S2['wall_s_cumulative'])
        cost['rung2_cost_mult_vs_S1'] = cost['rung2_outer'] / kE1
        cost['rung34_cost_mult_vs_S2'] = cost['rung34_outer'] / kE2
        m2['cost_dominated']  = bool(cost['rung2_cost_mult_vs_S1'] >= TH['cost_domination_mult'] and not m2['any'])
        m34['cost_dominated'] = bool(cost['rung34_cost_mult_vs_S2'] >= TH['cost_domination_mult'] and not m34['any'])
        m34['failure_risk']   = bool(rec['status'] == 'CAP_HIT')
        m2['failure_risk']    = False

        # ---- Phase 14: does the two-rung policy terminate honestly? ------
        term = dict(
            E_satisfied_on_0p02=bool(S2['declared']),
            branch=S2['branch'], iteration=kE2, window=S2['window'],
            persistence=max(S2['nA'], S2['nB']),
            move_at_event=S2['move'],
            same_frozen_concept=True,
            capped=bool(rec['status'] == 'CAP_HIT'),
            four_rung_terminal_status=rec['status'],
            two_rung_terminal_status='CONVERGED',
            cap_avoided=bool(rec['status'] == 'CAP_HIT'))

        # ---- Phase 16: bound limitation (descriptive only) --------------
        bnd = dict(
            S1_l2_over_tol=S1['l2_over_tol'], S2_l2_over_tol=S2['l2_over_tol'],
            F_l2_over_tol=Fs['l2_over_tol'],
            S1_maxAbs_over_move=S1['maxAbs_over_move'], S2_maxAbs_over_move=S2['maxAbs_over_move'],
            S1_boundFrac=S1['boundFrac'], S2_boundFrac=S2['boundFrac'],
            S1_branch=S1['branch'], S2_branch=S2['branch'],
            rung2_Mnd_rel_pct=rung2['dMnd_rel_pct'], rung2_omega1_rel_pct=rung2['domega1_rel_pct'])

        # ---- wall-time reliability (Phase 13, PREREGISTRATION S11) -------
        # Seconds per INNER MMA iteration should be near-constant for a fixed mesh.
        # Where it is not, elapsed time is contaminated by machine conditions and
        # is down-weighted in favour of the durable inner/outer work metrics.
        spi = lambda a, b: float(np.sum(tOuter[a:b]) / max(np.sum(T['nInner'][a:b]), 1))
        blocks = {'first50': spi(0, 50), 'rung1': spi(0, kE1), 'rung2': spi(kE1, kE2),
                  'rungs34': spi(kE2, n)}
        wall = dict(s_per_inner=blocks,
                    drift_ratio=max(blocks.values()) / min(blocks.values()),
                    reliable=bool(max(blocks.values()) / min(blocks.values()) < 1.5),
                    note='seconds per inner MMA iteration should be near-constant at a '
                         'fixed mesh; drift indicates machine contention, not algorithm '
                         'cost.  Where drift_ratio >= 1.5 wall time is reported but the '
                         'durable metrics are outer iterations and inner MMA iterations.')

        # ---- density distance to production, where P's field survives -----
        topoP = dict(available=bool(P.get('rho_available', False)))
        if topoP['available']:
            pt = os.path.join(REPO, P['trajectory'])
            with h5py.File(pt, 'r') as h:
                rhoP = np.array(h['RHO']).T[:, int(P['nOuter']) - 1]
            for nm, st in (('S1', S1), ('S2', S2), ('F', Fs)):
                dd = RHO[:, st['iteration'] - 1] - rhoP
                topoP[nm] = dict(mean_abs=float(np.mean(np.abs(dd))),
                                 rms=float(np.linalg.norm(dd) / np.sqrt(NE)),
                                 max_abs=float(np.max(np.abs(dd))))
            topoP['rhoP_sha256'] = vechash(rhoP)
            topoP['rhoP_sha256_recorded'] = P.get('rho_sha256')
            topoP['rhoP_hash_matches_baseline'] = bool(
                topoP['rhoP_sha256'] == P.get('rho_sha256'))
        else:
            topoP['reason'] = ('production density field UNAVAILABLE -- raw .mat lost; '
                               'baselines.json records rho_available = false')

        out['mesh'][key] = dict(
            mesh=[nx, ny], NE=NE, tag=tag, tol=F.tol_for(NE),
            production=P, S1=S1, S2=S2, F=Fs,
            rung2=rung2, rung34=rung34, rungs234=s1_to_F,
            materiality_rung2=m2, materiality_rung34=m34,
            vs_production=vsP, gates_at_S2=A, gates_at_S1=A1,
            physics=phys, cost=cost, termination=term, bound_limitation=bnd,
            wall_reliability=wall, topology_vs_production=topoP,
            counterfactual_validity=V['counterfactual_validity'],
            replay_all_match=V['replay_all_match'])

    p = os.path.join(STUDY, 'evidence', 'analysis.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('written', p)


if __name__ == '__main__':
    main()
