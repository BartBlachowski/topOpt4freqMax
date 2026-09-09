#!/usr/bin/env python3
"""tr_verify -- Phases 1, 3, 4.  Zero scientific runs.

(1) Replays the FROZEN rule offline, stage by stage, from the raw trajectory
    (RHO and hist.dxNorm2), and requires element-wise agreement with the
    controller trace the solver actually acted on, plus exact agreement of every
    recorded declaration.
(3) Checks the five counterfactual-validity conditions of PREREGISTRATION.md S7.
(4) Verifies the stage-1 events against the previous study's recorded values
    without hard-coding them.
"""
import os, sys, json
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
LEVELS4 = [0.04, 0.02, 0.01, 0.005]


def load(tag):
    with h5py.File(os.path.join(EV, f'{tag}_trajectory.mat'), 'r') as h:
        RHO  = np.array(h['RHO']).T
        hist = {k: np.array(h['hist'][k]).ravel() for k in
                ('dxNorm2', 'move', 'stage', 'cumInner', 'tOuter')}
        rho0 = float(np.array(h['cfg']['design']['initial']).ravel()[0])
    return RHO, hist, rho0


def main():
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

        # ---- stage boundaries read from the trace, not assumed -----------
        starts = [1] + [int(k + 1) for k in range(1, n) if stage[k] != stage[k - 1]]
        assert starts == [int(v) for v in rec['exhaustion']['stageStarts']], (starts, rec['exhaustion']['stageStarts'])

        stages = []
        for si, s in enumerate(starts):
            e = (starts[si + 1] - 1) if si + 1 < len(starts) else n
            r = F.replay_stage(RHO, amp, rho0, NE, s, e)
            recA = T['exA'][s - 1:e].astype(bool)
            recB = T['exB'][s - 1:e].astype(bool)
            recE = T['exE'][s - 1:e].astype(bool)
            recNA = T['exNA'][s - 1:e].astype(int)
            recNB = T['exNB'][s - 1:e].astype(int)
            recSS = T['exStageStart'][s - 1:e].astype(int)
            recDecl = None
            dd = np.nonzero(T['exDecl'][s - 1:e].astype(int))[0]
            if dd.size:
                recDecl = int(s + dd[0])
            stages.append(dict(
                stage=si + 1, move=float(move[s - 1]), stageStart=s, stageEnd=e,
                nIter=e - s + 1,
                offline_decl=r['decl'], offline_branch=r['branch'], offline_window=
                    ([r['begin'], r['decl']] if r['decl'] else None),
                recorded_decl=recDecl,
                decl_match=(r['decl'] == recDecl),
                A_match=bool(np.array_equal(r['A'][s - 1:e], recA)),
                B_match=bool(np.array_equal(r['B'][s - 1:e], recB)),
                E_match=bool(np.array_equal(r['E'][s - 1:e], recE)),
                nA_match=bool(np.array_equal(r['nA'][s - 1:e], recNA)),
                nB_match=bool(np.array_equal(r['nB'][s - 1:e], recNB)),
                stageStart_trace_ok=bool(np.all(recSS == s)),
                min_latency=(None if r['decl'] is None else int(r['decl'] - s)),
                earliest_possible=38,
                declared_at_earliest=(None if r['decl'] is None else bool(r['decl'] - s == 38)),
                move_constant=bool(np.allclose(move[s - 1:e], move[s - 1])),
                terminal=(e == n)))

        kE1 = stages[0]['offline_decl']
        kE2 = stages[1]['offline_decl'] if len(stages) > 1 else None

        # ---- PREREGISTRATION S7: the five counterfactual-validity checks --
        c = {}
        c['1_stage1_holds_0.04_to_kE1'] = bool(
            np.all(move[:kE1] == 0.04) and np.all(stage[:kE1] == 1))
        descents = [int(k + 1) for k in range(1, kE2) if move[k] != move[k - 1]]
        c['2_exactly_one_descent_before_kE2'] = bool(
            len(descents) == 1 and descents[0] == kE1 + 1 and move[kE1] == 0.02)
        c['3_stage2_holds_0.02_to_kE2'] = bool(
            np.all(move[kE1:kE2] == 0.02) and np.all(stage[kE1:kE2] == 2))
        c['4_no_0.01_or_lower_at_or_before_kE2'] = bool(np.all(move[:kE2] >= 0.02))
        c['5_reset_semantics_stageStart_is_kE1_plus_1'] = bool(
            np.all(T['exStageStart'][kE1:kE2].astype(int) == kE1 + 1))
        c['all'] = all(c.values())

        out[tag] = dict(
            mesh=[nx, ny], NE=NE, tol=F.tol_for(NE), nOuter=n,
            status=rec['status'],
            recorded_stageStarts=[int(v) for v in rec['exhaustion']['stageStarts']],
            recorded_descents=[[int(x) for x in row] for row in rec['exhaustion']['descents']],
            stages=stages,
            kE1=kE1, kE1_branch=stages[0]['offline_branch'],
            kE2=kE2, kE2_branch=stages[1]['offline_branch'] if len(stages) > 1 else None,
            first_descent_applied=kE1 + 1,
            counterfactual_validity=c,
            replay_all_match=all(st['A_match'] and st['B_match'] and st['E_match']
                                 and st['nA_match'] and st['nB_match'] and st['decl_match']
                                 for st in stages))

    p = os.path.join(STUDY, 'evidence', 'event_verification.json')
    json.dump(out, open(p, 'w'), indent=1)
    for tag, d in out.items():
        print(f"=== {tag}  NE={d['NE']}  tol={d['tol']:.4g}  n={d['nOuter']}  {d['status']}")
        for st in d['stages']:
            print(f"   stage {st['stage']} move={st['move']:.4g} [{st['stageStart']}..{st['stageEnd']}]"
                  f" n={st['nIter']:5d} decl={st['offline_decl']} ({st['offline_branch']})"
                  f" rec={st['recorded_decl']} match={st['decl_match']}"
                  f" latency={st['min_latency']} earliest={st['declared_at_earliest']}"
                  f" A/B/E/nA/nB={int(st['A_match'])}{int(st['B_match'])}{int(st['E_match'])}"
                  f"{int(st['nA_match'])}{int(st['nB_match'])}")
        print(f"   kE1={d['kE1']} ({d['kE1_branch']})   kE2={d['kE2']} ({d['kE2_branch']})")
        for k, v in d['counterfactual_validity'].items():
            print(f"   [{'PASS' if v else 'FAIL'}] {k}")
    print('\nwritten', p)


if __name__ == '__main__':
    main()
