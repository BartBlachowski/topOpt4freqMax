#!/usr/bin/env python3
"""tr3_verify -- Phases 1, 3, 4, 14.  Zero scientific runs.

(1)  Replays the FROZEN rule offline, stage by stage, from the RAW trajectory
     (RHO and hist.dxNorm2), requiring element-wise agreement with the controller
     trace the solver actually acted on, and exact agreement of every recorded
     declaration.
(3)  Checks the ten counterfactual-validity conditions of PREREGISTRATION.md S7
     for the THREE-rung policy -- not assumed from the two-rung proof.
(4)  Reproduces S1 and S2 from telemetry rather than accepting the recorded
     numbers, and pins the index convention.
(14) Records, for every stage, the first iteration at which the frozen predicate
     could mathematically be evaluated with complete required history, and
     compares it to the actual declaration.
"""
import os, sys, json
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tr3_frozen as F

HERE  = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT  = os.path.dirname(os.path.dirname(STUDY))
CV    = os.path.join(ROOT, 'diagnostics', 'two_branch_controller_validation')
EV    = os.path.join(ROOT, 'evidence', 'two_branch_controller_validation')
TWO   = os.path.join(ROOT, 'diagnostics', 'two_rung_architecture')
MESH  = {'m160': ('C160x20', 160, 20), 'm320': ('C320x40', 320, 40), 'm400': ('C400x50', 400, 50)}
W, P, WNP = F.W, F.P, F.WNP


def load(tag):
    with h5py.File(os.path.join(EV, f'{tag}_trajectory.mat'), 'r') as h:
        RHO  = np.array(h['RHO']).T
        hist = {k: np.array(h['hist'][k]).ravel() for k in
                ('dxNorm2', 'move', 'stage', 'cumInner', 'tOuter')}
        rho0 = float(np.array(h['cfg']['design']['initial']).ravel()[0])
    return RHO, hist, rho0


def main():
    prior = json.load(open(os.path.join(TWO, 'evidence', 'event_verification.json')))
    out = {}
    for key, (tag, nx, ny) in MESH.items():
        NE = nx * ny
        RHO, hist, rho0 = load(tag)
        T = np.genfromtxt(os.path.join(CV, 'runs', f'{tag}_iterations.csv'),
                          delimiter=',', names=True)
        rec = json.load(open(os.path.join(CV, 'runs', f'{tag}_record.json')))
        n = int(rec['nOuter'])
        amp = hist['dxNorm2'][:n]
        move = np.round(hist['move'][:n], 12)
        stage = hist['stage'][:n].astype(int)

        starts = [1] + [int(k + 1) for k in range(1, n) if stage[k] != stage[k - 1]]
        assert starts == [int(v) for v in rec['exhaustion']['stageStarts']]

        stages = []
        for si, s in enumerate(starts):
            e = (starts[si + 1] - 1) if si + 1 < len(starts) else n
            r = F.replay_stage(RHO, amp, rho0, NE, s, e)
            sl = slice(s - 1, e)
            recDecl = None
            dd = np.nonzero(T['exDecl'][sl].astype(int))[0]
            if dd.size:
                recDecl = int(s + dd[0])

            # ---- Phase 14: the earliest mathematically evaluable window ----
            # cos(j) needs j >= s+1; net(j) needs j >= s+9; the trailing 20-median
            # is defined only once the whole window lies inside the stage, i.e.
            # j >= s+19.  So the predicate is first EVALUABLE at s+19 (= s+W-1),
            # and the earliest possible DECLARATION is s+19+19 = s+38.
            firstEval = s + W - 1
            earliestDecl = s + W - 1 + P - 1
            medDefined = np.nonzero(~np.isnan(r['medcos'][sl]))[0]
            firstMedIdx = int(s + medDefined[0]) if medDefined.size else None
            # was E true at every iteration from the first evaluable one?
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
                # Phase 14 fields
                first_evaluable_iter=firstEval,
                first_median_defined_iter=firstMedIdx,
                first_evaluable_matches_theory=bool(firstMedIdx == firstEval),
                earliest_possible_declaration=earliestDecl,
                declaration_offset=(None if r['decl'] is None else int(r['decl'] - s)),
                declared_at_earliest_possible=(None if r['decl'] is None
                                               else bool(r['decl'] == earliestDecl)),
                E_true_from_first_evaluable=(None if Ewin.size == 0 else bool(Ewin[0])),
                E_unbroken_to_declaration=(None if r['decl'] is None
                                           else bool(np.all(r['E'][firstEval - 1:r['decl']]))),
                E_true_count_after_first_evaluable=(int(np.sum(Ewin)) if Ewin.size else 0),
                E_window_length=int(Ewin.size),
                terminal=(e == n)))

        kE1 = stages[0]['offline_decl']
        kE2 = stages[1]['offline_decl'] if len(stages) > 1 else None
        kE3 = stages[2]['offline_decl'] if len(stages) > 2 else None

        # ---- PREREGISTRATION S7: the ten three-rung validity checks -------
        c = {}
        c['01_same_initialization_rho0'] = bool(rho0 == 0.5)
        c['02_stage1_holds_0.04_to_kE1'] = bool(
            np.all(move[:kE1] == 0.04) and np.all(stage[:kE1] == 1))
        d1 = [int(k + 1) for k in range(1, kE3) if move[k] != move[k - 1]]
        c['03_exactly_two_descents_before_kE3'] = bool(
            len(d1) == 2 and d1 == [kE1 + 1, kE2 + 1])
        c['04_first_descent_to_0.02_at_kE1+1'] = bool(move[kE1] == 0.02)
        c['05_stage2_holds_0.02_to_kE2'] = bool(
            np.all(move[kE1:kE2] == 0.02) and np.all(stage[kE1:kE2] == 2))
        c['06_second_descent_to_0.01_at_kE2+1'] = bool(move[kE2] == 0.01)
        c['07_stage3_holds_0.01_to_kE3'] = bool(
            np.all(move[kE2:kE3] == 0.01) and np.all(stage[kE2:kE3] == 3))
        c['08_no_0.005_at_or_before_kE3'] = bool(np.all(move[:kE3] >= 0.01))
        c['09_reset_semantics_stage2_and_stage3'] = bool(
            np.all(T['exStageStart'][kE1:kE2].astype(int) == kE1 + 1) and
            np.all(T['exStageStart'][kE2:kE3].astype(int) == kE2 + 1))
        c['10_divergence_only_at_kE3'] = bool(
            stage[kE3 - 1] == 3 and move[kE3 - 1] == 0.01 and
            (kE3 == n or move[kE3] == 0.005))
        c['all'] = all(c.values())

        # ---- Phase 4 cross-check against the prior audit -----------------
        pv = prior[tag]
        cross = dict(prior_kE1=pv['kE1'], prior_kE1_branch=pv['kE1_branch'],
                     prior_kE2=pv['kE2'], prior_kE2_branch=pv['kE2_branch'],
                     kE1_reproduced=bool(kE1 == pv['kE1']),
                     kE2_reproduced=bool(kE2 == pv['kE2']),
                     kE1_branch_reproduced=bool(stages[0]['offline_branch'] == pv['kE1_branch']),
                     kE2_branch_reproduced=bool(stages[1]['offline_branch'] == pv['kE2_branch']))

        out[tag] = dict(
            mesh=[nx, ny], NE=NE, tol=F.tol_for(NE), nOuter=n, status=rec['status'],
            rho0=rho0,
            recorded_stageStarts=[int(v) for v in rec['exhaustion']['stageStarts']],
            stages=stages,
            kE1=kE1, kE1_branch=stages[0]['offline_branch'],
            kE2=kE2, kE2_branch=stages[1]['offline_branch'],
            kE3=kE3, kE3_branch=stages[2]['offline_branch'],
            stage3_start=stages[2]['stageStart'],
            kE3_minus_stage3start=(None if kE3 is None else kE3 - stages[2]['stageStart']),
            counterfactual_validity=c,
            cross_check_prior=cross,
            replay_all_match=all(st['A_match'] and st['B_match'] and st['E_match']
                                 and st['nA_match'] and st['nB_match'] and st['decl_match']
                                 for st in stages))

    p = os.path.join(STUDY, 'evidence', 'event_verification.json')
    json.dump(out, open(p, 'w'), indent=1)
    for tag, d in out.items():
        print(f"=== {tag}  NE={d['NE']} tol={d['tol']:.4g} n={d['nOuter']} {d['status']}")
        for st in d['stages']:
            print(f"   stage {st['stage']} move={st['move']:.4g} [{st['stageStart']}..{st['stageEnd']}]"
                  f" decl={st['offline_decl']} ({st['offline_branch']}) rec={st['recorded_decl']}"
                  f" match={st['decl_match']}  firstEval={st['first_evaluable_iter']}"
                  f" earliest={st['earliest_possible_declaration']}"
                  f" offset={st['declaration_offset']}"
                  f" atEarliest={st['declared_at_earliest_possible']}"
                  f" Etrue@firstEval={st['E_true_from_first_evaluable']}"
                  f" trace={int(st['A_match'])}{int(st['B_match'])}{int(st['E_match'])}"
                  f"{int(st['nA_match'])}{int(st['nB_match'])}")
        print(f"   kE1={d['kE1']}({d['kE1_branch']}) kE2={d['kE2']}({d['kE2_branch']}) "
              f"kE3={d['kE3']}({d['kE3_branch']})  stage3Start={d['stage3_start']} "
              f"offset={d['kE3_minus_stage3start']}")
        print(f"   cross-check prior: kE1 {d['cross_check_prior']['kE1_reproduced']} "
              f"kE2 {d['cross_check_prior']['kE2_reproduced']}")
        for k, v in d['counterfactual_validity'].items():
            print(f"   [{'PASS' if v else 'FAIL'}] {k}")
    print('\nwritten', p)


if __name__ == '__main__':
    main()
