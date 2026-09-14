#!/usr/bin/env python3
"""r240_verify -- Phases 10, 11, 12 for the new C240x30 causal trajectory.

(10) Extract S1/S2/S3/F with the inherited declaration-index convention.
(11) Re-audit the three-rung counterfactual against the REALIZED trajectory:
     the ten per-mesh checks, plus an element-wise replay of the frozen rule
     from raw RHO / hist.dxNorm2 against the trace the solver acted on.
(12) Declaration timing: stageStart, first mathematically evaluable iteration,
     first E-true iteration, first persistent declaration, and the offset.
"""
import os, sys, json
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r240_frozen as F

HERE  = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT  = os.path.dirname(os.path.dirname(STUDY))
EV    = os.path.join(ROOT, 'evidence', 'three_rung_resolution_240')
TAG   = 'C240x30'
NX, NY = 240, 30
NE = NX * NY
W, P, WNP = F.W, F.P, F.WNP


def main():
    with h5py.File(os.path.join(EV, f'{TAG}_trajectory.mat'), 'r') as h:
        RHO  = np.array(h['RHO']).T
        hist = {k: np.array(h['hist'][k]).ravel() for k in
                ('dxNorm2', 'move', 'stage', 'cumInner', 'tOuter')}
        rho0 = float(np.array(h['cfg']['design']['initial']).ravel()[0])
    T = np.genfromtxt(os.path.join(STUDY, 'runs', f'{TAG}_iterations.csv'),
                      delimiter=',', names=True)
    rec = json.load(open(os.path.join(STUDY, 'runs', f'{TAG}_record.json')))
    n = int(rec['nOuter'])
    amp   = hist['dxNorm2'][:n]
    move  = np.round(hist['move'][:n], 12)
    stage = hist['stage'][:n].astype(int)

    starts = [1] + [int(k + 1) for k in range(1, n) if stage[k] != stage[k - 1]]
    assert starts == [int(v) for v in rec['exhaustion']['stageStarts']], starts

    stages = []
    for si, s in enumerate(starts):
        e = (starts[si + 1] - 1) if si + 1 < len(starts) else n
        r = F.replay_stage(RHO, amp, rho0, NE, s, e)
        sl = slice(s - 1, e)
        dd = np.nonzero(T['exDecl'][sl].astype(int))[0]
        recDecl = int(s + dd[0]) if dd.size else None

        firstEval    = s + W - 1                 # median first defined here
        earliestDecl = s + W - 1 + P - 1         # = s + 38
        medDef = np.nonzero(~np.isnan(r['medcos'][sl]))[0]
        firstMed = int(s + medDef[0]) if medDef.size else None
        Eidx = np.nonzero(r['E'][sl])[0]
        firstEtrue = int(s + Eidx[0]) if Eidx.size else None
        Ewin = r['E'][firstEval - 1:e] if firstEval <= e else np.array([], bool)

        stages.append(dict(
            stage=si + 1, move=float(move[s - 1]), stageStart=s, stageEnd=e,
            nIter=e - s + 1,
            offline_decl=r['decl'], offline_branch=r['branch'],
            offline_window=([r['begin'], r['decl']] if r['decl'] else None),
            recorded_decl=recDecl, decl_match=(r['decl'] == recDecl),
            A_match=bool(np.array_equal(r['A'][sl], T['exA'][sl].astype(bool))),
            B_match=bool(np.array_equal(r['B'][sl], T['exB'][sl].astype(bool))),
            E_match=bool(np.array_equal(r['E'][sl], T['exE'][sl].astype(bool))),
            nA_match=bool(np.array_equal(r['nA'][sl], T['exNA'][sl].astype(int))),
            nB_match=bool(np.array_equal(r['nB'][sl], T['exNB'][sl].astype(int))),
            stageStart_trace_ok=bool(np.all(T['exStageStart'][sl].astype(int) == s)),
            move_constant=bool(np.allclose(move[sl], move[s - 1])),
            first_evaluable_iter=firstEval,
            first_median_defined_iter=firstMed,
            first_evaluable_matches_theory=bool(firstMed == firstEval),
            first_E_true_iter=firstEtrue,
            earliest_possible_declaration=earliestDecl,
            declaration_offset=(None if r['decl'] is None else int(r['decl'] - s)),
            declared_at_earliest_possible=(None if r['decl'] is None
                                           else bool(r['decl'] == earliestDecl)),
            E_true_from_first_evaluable=(None if Ewin.size == 0 else bool(Ewin[0])),
            E_unbroken_to_declaration=(None if r['decl'] is None
                                       else bool(np.all(r['E'][firstEval - 1:r['decl']]))),
            E_true_fraction_after_first_evaluable=(float(np.mean(Ewin)) if Ewin.size else None),
            terminal=(e == n)))

    k1, k2, k3 = (stages[0]['offline_decl'], stages[1]['offline_decl'],
                  stages[2]['offline_decl'])

    # ---- PREREGISTRATION S8 / three_rung S7: the ten validity checks ----
    c = {}
    c['01_same_initialization_rho0'] = bool(rho0 == 0.5)
    c['02_stage1_holds_0.04_to_kE1'] = bool(
        np.all(move[:k1] == 0.04) and np.all(stage[:k1] == 1))
    desc = [int(k + 1) for k in range(1, k3) if move[k] != move[k - 1]]
    c['03_exactly_two_descents_before_kE3'] = bool(desc == [k1 + 1, k2 + 1])
    c['04_first_descent_to_0.02_at_kE1+1'] = bool(move[k1] == 0.02)
    c['05_stage2_holds_0.02_to_kE2'] = bool(
        np.all(move[k1:k2] == 0.02) and np.all(stage[k1:k2] == 2))
    c['06_second_descent_to_0.01_at_kE2+1'] = bool(move[k2] == 0.01)
    c['07_stage3_holds_0.01_to_kE3'] = bool(
        np.all(move[k2:k3] == 0.01) and np.all(stage[k2:k3] == 3))
    c['08_no_0.005_at_or_before_kE3'] = bool(np.all(move[:k3] >= 0.01))
    c['09_reset_semantics_stage2_and_stage3'] = bool(
        np.all(T['exStageStart'][k1:k2].astype(int) == k1 + 1) and
        np.all(T['exStageStart'][k2:k3].astype(int) == k2 + 1))
    c['10_divergence_only_at_kE3'] = bool(
        stage[k3 - 1] == 3 and move[k3 - 1] == 0.01 and
        (k3 == n or move[k3] == 0.005))
    c['all'] = all(c.values())

    out = dict(
        tag=TAG, mesh=[NX, NY], NE=NE, tol=F.tol_for(NE), nOuter=n,
        status=rec['status'], rho0=rho0,
        recorded_stageStarts=[int(v) for v in rec['exhaustion']['stageStarts']],
        stages=stages,
        kE1=k1, kE1_branch=stages[0]['offline_branch'],
        kE2=k2, kE2_branch=stages[1]['offline_branch'],
        kE3=k3, kE3_branch=stages[2]['offline_branch'],
        stage3_start=stages[2]['stageStart'],
        kE3_offset=k3 - stages[2]['stageStart'],
        stage4_start=stages[3]['stageStart'],
        terminal_decl=stages[3]['offline_decl'],
        terminal_branch=stages[3]['offline_branch'],
        terminal_offset=stages[3]['declaration_offset'],
        counterfactual_validity=c,
        replay_all_match=all(s['A_match'] and s['B_match'] and s['E_match']
                             and s['nA_match'] and s['nB_match'] and s['decl_match']
                             for s in stages),
        # the preregistered S1 prediction, checked
        prediction_S1=206, prediction_window=10,
        S1_prediction_hit=bool(abs(k1 - 206) <= 10),
        S1_branch_predicted='B', S1_branch_hit=bool(stages[0]['offline_branch'] == 'B'),
        fixedmove_window_begin_recorded=187,
        S1_window_begin=stages[0]['offline_window'][0] if stages[0]['offline_window'] else None)

    p = os.path.join(STUDY, 'evidence', 'event_verification.json')
    json.dump(out, open(p, 'w'), indent=1)

    print(f"=== {TAG}  NE={NE}  tol={out['tol']:.4g}  n={n}  {out['status']}")
    for s in stages:
        print(f"  stage {s['stage']} move={s['move']:<6.4g} [{s['stageStart']}..{s['stageEnd']}]"
              f" n={s['nIter']:5d} decl={s['offline_decl']} ({s['offline_branch']})"
              f" rec={s['recorded_decl']} match={s['decl_match']}"
              f" firstEval={s['first_evaluable_iter']} firstEtrue={s['first_E_true_iter']}"
              f" earliest={s['earliest_possible_declaration']}"
              f" offset={s['declaration_offset']} atEarliest={s['declared_at_earliest_possible']}"
              f" E@firstEval={s['E_true_from_first_evaluable']}"
              f" Efrac={s['E_true_fraction_after_first_evaluable']}"
              f" trace={int(s['A_match'])}{int(s['B_match'])}{int(s['E_match'])}"
              f"{int(s['nA_match'])}{int(s['nB_match'])}")
    print(f"  S1={k1}({out['kE1_branch']}) S2={k2}({out['kE2_branch']}) S3={k3}({out['kE3_branch']})"
          f"  stage4Start={out['stage4_start']} terminal={out['terminal_decl']}"
          f"({out['terminal_branch']}) offset={out['terminal_offset']}")
    print(f"  S1 prediction 206+-10: hit={out['S1_prediction_hit']}  branch B: hit={out['S1_branch_hit']}"
          f"  window begins {out['S1_window_begin']} (fixed-move arm recorded 187)")
    for k, v in c.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  replay_all_match = {out['replay_all_match']}")
    print('\nwritten', p)


if __name__ == '__main__':
    main()
