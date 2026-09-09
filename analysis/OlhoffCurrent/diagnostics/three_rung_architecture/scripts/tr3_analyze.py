#!/usr/bin/env python3
"""tr3_analyze -- Phases 5-13, 16-17, 19.  Zero scientific runs; offline only.

Extracts P / S1 / S2 / S3 / F, decomposes the ladder rung by rung (rung 3 is
isolated from rung 4 for the first time), evaluates the INHERITED materiality
thresholds and production-relative acceptance gates, and computes cost.
"""
import os, sys, json, hashlib
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tr3_frozen as F

HERE  = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT  = os.path.dirname(os.path.dirname(STUDY))
REPO  = os.path.dirname(os.path.dirname(ROOT))
CV    = os.path.join(ROOT, 'diagnostics', 'two_branch_controller_validation')
EV    = os.path.join(ROOT, 'evidence', 'two_branch_controller_validation')
MESH  = {'m160': ('C160x20', 160, 20), 'm320': ('C320x40', 320, 40), 'm400': ('C400x50', 400, 50)}

# ---- PREREGISTRATION S9, inherited verbatim -----------------------------
TH = dict(Mnd_rel_pct=2.0, omega1_rel_pct=0.10, topo_frac=0.01, rho_mean_abs=0.01,
          volume_worsen=1e-5, cost_domination_mult=2.0)
# ---- PREREGISTRATION S10, inherited verbatim ----------------------------
GATES = {'m160': dict(Mnd_max_mult=1.10, omega1_min_mult=0.99),
         'm320': dict(Mnd_max_mult=0.80, omega1_min_mult=0.99),
         'm400': dict(Mnd_max_mult=0.80, omega1_min_mult=0.99)}
VOL_TOL = 1e-4
OUTER_MULT_MAX, WALL_MULT_MAX = 8.0, 10.0


def vechash(v):
    return hashlib.sha256(np.ascontiguousarray(v, dtype='<f8').tobytes()).hexdigest()


def load(tag):
    with h5py.File(os.path.join(EV, f'{tag}_trajectory.mat'), 'r') as h:
        RHO  = np.array(h['RHO']).T
        tOuter = np.array(h['hist']['tOuter']).ravel()
    return RHO, tOuter


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
        offset_from_stage_start=int(k - T['exStageStart'][i]),
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
    """b - a.  Relative changes normalised by the EARLIER state (inherited)."""
    d = RHO[:, b['iteration'] - 1] - RHO[:, a['iteration'] - 1]
    return dict(
        block=label, frm=a['label'], to=b['label'],
        dMnd=b['Mnd'] - a['Mnd'], dMnd_rel_pct=100 * (b['Mnd'] - a['Mnd']) / a['Mnd'],
        domega1=b['omega1'] - a['omega1'],
        domega1_rel_pct=100 * (b['omega1'] - a['omega1']) / a['omega1'],
        domega2=b['omega2'] - a['omega2'],
        domega2_rel_pct=100 * (b['omega2'] - a['omega2']) / a['omega2'],
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
    m = dict(
        Mnd=bool(-d['dMnd_rel_pct'] >= TH['Mnd_rel_pct']),
        omega1=bool(d['domega1_rel_pct'] >= TH['omega1_rel_pct']),
        topology=bool(abs(d['dgray']) >= TH['topo_frac'] or abs(d['dmid']) >= TH['topo_frac']
                      or d['rho_mean_abs'] >= TH['rho_mean_abs']),
        volume=bool(d['dvolume_abs'] <= -TH['volume_worsen']),
        multiplicity=bool(d['dSubspaceN'] != 0))
    m['any'] = any(m.values())
    return m


def main():
    B = json.load(open(os.path.join(CV, 'evidence', 'baselines.json')))
    EVT = json.load(open(os.path.join(STUDY, 'evidence', 'event_verification.json')))
    out = {'thresholds': TH,
           'gates_spec': dict(GATES=GATES, VOL_TOL=VOL_TOL,
                              OUTER_MULT_MAX=OUTER_MULT_MAX, WALL_MULT_MAX=WALL_MULT_MAX),
           'relative_convention': '100*(b-a)/a, normalised by the earlier state (inherited)',
           'mesh': {}}

    for key, (tag, nx, ny) in MESH.items():
        NE = nx * ny
        RHO, tOuter = load(tag)
        T = np.genfromtxt(os.path.join(CV, 'runs', f'{tag}_iterations.csv'),
                          delimiter=',', names=True)
        rec = json.load(open(os.path.join(CV, 'runs', f'{tag}_record.json')))
        n = int(rec['nOuter'])
        V = EVT[tag]
        k1, k2, k3 = V['kE1'], V['kE2'], V['kE3']
        P = B[key]

        S1 = state_at(T, RHO, tOuter, k1, NE, 'S1 single-stage endpoint (move=0.04)')
        S2 = state_at(T, RHO, tOuter, k2, NE, 'S2 two-rung endpoint (move=0.02)')
        S3 = state_at(T, RHO, tOuter, k3, NE, 'S3 three-rung endpoint (move=0.01)')
        Fs = state_at(T, RHO, tOuter, n,  NE, 'F four-rung final')
        for st, br in ((S1, V['kE1_branch']), (S2, V['kE2_branch']), (S3, V['kE3_branch'])):
            st['branch'] = br
        S1['window'] = V['stages'][0]['offline_window']
        S2['window'] = V['stages'][1]['offline_window']
        S3['window'] = V['stages'][2]['offline_window']
        S3['policy_status'] = 'CONVERGED (three-rung policy would terminate here)'
        Fs['status'] = rec['status']
        Fs['terminalBranch'] = rec['exhaustion'].get('terminalBranch') or None

        # ---- rung-by-rung, rung 3 ISOLATED from rung 4 -------------------
        P_as_state = dict(label='P production baseline', iteration=int(P['nOuter']),
                          Mnd=P['Mnd'], omega1=P['omega1'], omega2=P['omega2'],
                          gap12=P['gap12'], volume=P['volume'], gray=P['gray'],
                          mid=P['mid'], subspaceN=2,
                          innerCumulative=int(P['innerTotal']),
                          wall_s_cumulative=float(P['wall_s']))
        rungs = {
            'rung1': delta(dict(label='initial uniform design', iteration=1,
                                Mnd=float(T['Mnd'][0]), omega1=float(T['omega1'][0]),
                                omega2=float(T['omega2'][0]), gap12=float(T['gap12'][0]),
                                volume=float(T['volume'][0]), gray=float(T['gray'][0]),
                                mid=float(T['mid'][0]), subspaceN=int(T['multN'][0]),
                                innerCumulative=int(T['cumInner'][0]),
                                wall_s_cumulative=float(tOuter[0])),
                          S1, RHO, NE, 'RUNG 1 (move=0.04):  start -> S1'),
            'rung2': delta(S1, S2, RHO, NE, 'RUNG 2 (move=0.02):  S1 -> S2'),
            'rung3': delta(S2, S3, RHO, NE, 'RUNG 3 (move=0.01):  S2 -> S3'),
            'rung4': delta(S3, Fs, RHO, NE, 'RUNG 4 (move=0.005): S3 -> F'),
            'rungs34': delta(S2, Fs, RHO, NE, 'RUNGS 3+4 combined (the two-rung residual): S2 -> F')}
        mat = {r: materiality(rungs[r]) for r in ('rung2', 'rung3', 'rung4', 'rungs34')}

        # ---- vs production ----------------------------------------------
        vsP = {}
        for st, nm in ((S1, 'S1'), (S2, 'S2'), (S3, 'S3'), (Fs, 'F')):
            vsP[nm] = dict(
                dMnd=st['Mnd'] - P['Mnd'], dMnd_rel_pct=100 * (st['Mnd'] - P['Mnd']) / P['Mnd'],
                domega1=st['omega1'] - P['omega1'],
                domega1_rel_pct=100 * (st['omega1'] - P['omega1']) / P['omega1'],
                dgray=st['gray'] - P['gray'], dmid=st['mid'] - P['mid'],
                dgap12=st['gap12'] - P['gap12'],
                outer_mult=st['iteration'] / P['nOuter'],
                inner_mult=st['innerCumulative'] / P['innerTotal'],
                wall_mult=st['wall_s_cumulative'] / P['wall_s'])

        # ---- acceptance gates at S3 (and, for comparison, at S2/S1) ------
        g = GATES[key]
        pre = slice(0, k3)

        def gates(st, prefix_end):
            sl = slice(0, prefix_end)
            G = {}
            G['A_Mnd_bound']    = bool(st['Mnd'] <= g['Mnd_max_mult'] * P['Mnd'])
            G['A4_omega1_0.99'] = bool(st['omega1'] >= g['omega1_min_mult'] * P['omega1'])
            G['A5_volume']      = bool(abs(st['volume'] - 0.5) <= VOL_TOL)
            G['A6_physics']     = bool(
                np.all(T['multN'][sl] == 2) and np.all(T['omega2'][sl] > T['omega1'][sl])
                and np.all(np.isfinite(T['omega1'][sl])) and np.all(np.isfinite(T['omega2'][sl]))
                and np.all(np.isfinite(T['Mnd'][sl])) and np.all(T['innerConv'][sl] == 1))
            G['A7_outer_mult']  = bool(st['iteration'] / P['nOuter'] <= OUTER_MULT_MAX)
            G['A7_wall_mult']   = bool(st['wall_s_cumulative'] / P['wall_s'] <= WALL_MULT_MAX)
            G['A8_terminal_persistence'] = bool(
                st['declared'] and max(st['nA'], st['nB']) >= 20)
            if key == 'm160':
                G['A9_no_regression'] = bool(st['omega1'] >= P['omega1'])
            G['all'] = all(v for v in G.values() if isinstance(v, bool))
            return G

        A3 = gates(S3, k3)
        A2 = gates(S2, k2)
        A1 = gates(S1, k1)

        # ---- physics over the three-rung prefix (Phase 17) ---------------
        phys = dict(
            subspaceN_prefix_always2=bool(np.all(T['multN'][pre] == 2)),
            subspaceN_full_always2=bool(np.all(T['multN'][:n] == 2)),
            omega2_gt_omega1_prefix=bool(np.all(T['omega2'][pre] > T['omega1'][pre])),
            omega2_gt_omega1_full=bool(np.all(T['omega2'][:n] > T['omega1'][:n])),
            omega_finite=bool(np.all(np.isfinite(T['omega1'][:n]))
                              and np.all(np.isfinite(T['omega2'][:n]))),
            innerNonConv_prefix=int(np.sum(T['innerConv'][pre] == 0)),
            innerNonConv_total=int(np.sum(T['innerConv'][:n] == 0)),
            min_gap12_S3_to_F=float(np.min(T['gap12'][k3 - 1:n])),
            min_gap12_full=float(np.min(T['gap12'][:n])),
            degenTotal_prefix=float(np.sum(T['degen'][pre])),
            degenTotal_full=float(np.sum(T['degen'][:n])),
            degen_note=('degen counts EXPECTED near-degeneracy hits in the '
                        'multiplicity-aware subspace; it is not a failure indicator '
                        'and is not part of the preregistered P10/A6 gate'))

        # ---- cost (Phase 13) ---------------------------------------------
        stage_bounds = [(1, k1), (k1 + 1, k2), (k2 + 1, k3), (k3 + 1, n)]
        per_stage = []
        for si, (a, b) in enumerate(stage_bounds):
            if a > b:
                per_stage.append(None); continue
            inner = int(T['cumInner'][b - 1] - (T['cumInner'][a - 2] if a > 1 else 0))
            per_stage.append(dict(
                rung=si + 1, move=float(T['move'][a - 1]), iterFrom=a, iterTo=b,
                outer=b - a + 1, inner=inner, wall_s=float(np.sum(tOuter[a - 1:b])),
                pct_total_outer=100 * (b - a + 1) / n,
                pct_total_inner=100 * inner / int(Fs['innerCumulative']),
                terminal_status=('CAP_HIT' if (b == n and rec['status'] == 'CAP_HIT')
                                 else ('CONVERGED' if b == n else 'DESCENDED'))))
        cost = dict(
            total_outer=n, total_inner=int(Fs['innerCumulative']),
            total_wall_s=float(np.sum(tOuter[:n])),
            S3_outer=k3, S3_inner=int(S3['innerCumulative']),
            S3_wall_s=S3['wall_s_cumulative'],
            saved_outer=n - k3, saved_inner=int(Fs['innerCumulative'] - S3['innerCumulative']),
            saved_wall_s=float(Fs['wall_s_cumulative'] - S3['wall_s_cumulative']),
            saved_outer_pct=100 * (n - k3) / n,
            saved_inner_pct=100 * (Fs['innerCumulative'] - S3['innerCumulative'])
                            / int(Fs['innerCumulative']),
            saved_wall_pct=100 * (Fs['wall_s_cumulative'] - S3['wall_s_cumulative'])
                           / Fs['wall_s_cumulative'],
            per_stage=per_stage,
            rung3_cost_mult_vs_S2=(k3 - k2) / k2,
            rung4_cost_mult_vs_S3=(n - k3) / k3)
        for r, mult in (('rung3', cost['rung3_cost_mult_vs_S2']),
                        ('rung4', cost['rung4_cost_mult_vs_S3'])):
            mat[r]['cost_dominated'] = bool(mult >= TH['cost_domination_mult']
                                            and not mat[r]['any'])
        mat['rung4']['failure_risk'] = bool(rec['status'] == 'CAP_HIT')
        mat['rung3']['failure_risk'] = False

        # ---- marginal benefit per unit work (Phase 13) -------------------
        for r in ('rung2', 'rung3', 'rung4'):
            d = rungs[r]
            rungs[r]['domega1_per_1000_inner'] = (1000 * d['domega1'] / d['d_inner']
                                                  if d['d_inner'] else None)
            rungs[r]['dMnd_per_1000_inner'] = (1000 * d['dMnd'] / d['d_inner']
                                               if d['d_inner'] else None)
            rungs[r]['domega1_per_100_outer'] = (100 * d['domega1'] / d['d_outer']
                                                 if d['d_outer'] else None)

        # ---- termination (Phase 10/11) -----------------------------------
        term = dict(
            E_satisfied_on_0p01=bool(S3['declared']),
            branch=S3['branch'], iteration=k3, window=S3['window'],
            stage3_start=V['stage3_start'], offset=V['kE3_minus_stage3start'],
            persistence=max(S3['nA'], S3['nB']), move_at_event=S3['move'],
            four_rung_terminal_status=rec['status'],
            three_rung_terminal_status='CONVERGED',
            cap_avoided=bool(rec['status'] == 'CAP_HIT'),
            stage4_declares=bool(EVT[tag]['stages'][3]['offline_decl'] is not None)
                            if len(EVT[tag]['stages']) > 3 else None)

        # ---- bound limitation, descriptive (Phase 19) --------------------
        bnd = {f'{nm}_{f}': st[f] for nm, st in (('S1', S1), ('S2', S2), ('S3', S3), ('F', Fs))
               for f in ('l2_over_tol', 'maxAbs_over_move', 'boundFrac')}
        bnd.update(S1_branch=S1['branch'], S2_branch=S2['branch'], S3_branch=S3['branch'])

        # ---- wall-time reliability ---------------------------------------
        spi = lambda a, b: float(np.sum(tOuter[a:b]) / max(np.sum(T['nInner'][a:b]), 1))
        blocks = {'first50': spi(0, 50), 'rung1': spi(0, k1), 'rung2': spi(k1, k2),
                  'rung3': spi(k2, k3), 'rung4': spi(k3, n)}
        wall = dict(s_per_inner=blocks,
                    drift_ratio=max(blocks.values()) / min(blocks.values()),
                    reliable=bool(max(blocks.values()) / min(blocks.values()) < 1.5))

        # ---- density distance to production, where P's field survives ----
        topoP = dict(available=bool(P.get('rho_available', False)))
        if topoP['available']:
            with h5py.File(os.path.join(REPO, P['trajectory']), 'r') as h:
                rhoP = np.array(h['RHO']).T[:, int(P['nOuter']) - 1]
            for nm, st in (('S1', S1), ('S2', S2), ('S3', S3), ('F', Fs)):
                dd = RHO[:, st['iteration'] - 1] - rhoP
                topoP[nm] = dict(mean_abs=float(np.mean(np.abs(dd))),
                                 rms=float(np.linalg.norm(dd) / np.sqrt(NE)),
                                 max_abs=float(np.max(np.abs(dd))))
            topoP['rhoP_sha256'] = vechash(rhoP)
            topoP['rhoP_hash_matches_baseline'] = bool(vechash(rhoP) == P.get('rho_sha256'))
        else:
            topoP['reason'] = ('production density field UNAVAILABLE -- raw .mat lost; '
                               'baselines.json records rho_available = false')

        out['mesh'][key] = dict(
            mesh=[nx, ny], NE=NE, tag=tag, tol=F.tol_for(NE),
            production=P, S1=S1, S2=S2, S3=S3, F=Fs,
            rungs=rungs, materiality=mat,
            vs_production=vsP,
            gates_at_S1=A1, gates_at_S2=A2, gates_at_S3=A3,
            physics=phys, cost=cost, termination=term, bound_limitation=bnd,
            wall_reliability=wall, topology_vs_production=topoP,
            counterfactual_validity=V['counterfactual_validity'],
            replay_all_match=V['replay_all_match'],
            declaration_timing=[{k: s[k] for k in
                ('stage', 'move', 'stageStart', 'stageEnd', 'nIter', 'first_evaluable_iter',
                 'first_median_defined_iter', 'first_evaluable_matches_theory',
                 'earliest_possible_declaration', 'offline_decl', 'offline_branch',
                 'declaration_offset', 'declared_at_earliest_possible',
                 'E_true_from_first_evaluable', 'E_unbroken_to_declaration',
                 'E_true_count_after_first_evaluable', 'E_window_length')}
                for s in V['stages']])

    p = os.path.join(STUDY, 'evidence', 'analysis.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('written', p)


if __name__ == '__main__':
    main()
