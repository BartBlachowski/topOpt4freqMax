#!/usr/bin/env python3
"""cv_analyze -- Phases 13-20.  Post hoc only; no solver is invoked.

Transition audit, termination audit, causal comparison, the fine-mesh causal
test, the coarse-mesh safety test, physics safety, the 400x50 prefix
equivalence check, and the fifteen preregistered promotion gates.

Implemented in Python rather than MATLAB so that the analysis of completed runs
is never blocked by MATLAB licence availability; it reads only the tracked
per-iteration CSVs, the run records, the frozen baselines, and (for the prefix
check) the declared raw trajectories.
"""
import json, os, hashlib, math
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(STUDY))
REPO = os.path.dirname(os.path.dirname(ROOT))

KEYS = ['m160', 'm320', 'm400']
TAGS = {'m160': 'C160x20', 'm320': 'C320x40', 'm400': 'C400x50'}
LEVELS = [0.04, 0.02, 0.01, 0.005]
P = 20


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_csv(path):
    return np.genfromtxt(path, delimiter=',', names=True)


def first(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return v[0] if v else None


def analyse():
    B = json.load(open(os.path.join(STUDY, 'evidence', 'baselines.json')))
    A = {'generated': __import__('datetime').datetime.now(
             __import__('datetime').timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
         'preregistration_sha256': sha(os.path.join(STUDY, 'PREREGISTRATION.md')),
         'mesh': {}, 'runsExecuted': [], 'runsMissing': []}

    print('=' * 72); print('CV_ANALYZE'); print('=' * 72)

    for k in KEYS:
        tag = TAGS[k]
        recF = os.path.join(STUDY, 'runs', tag + '_record.json')
        if not os.path.isfile(recF):
            A['runsMissing'].append(tag)
            print(f'  {tag}: NO RUN RECORD -- run not executed')
            continue
        A['runsExecuted'].append(tag)
        C = json.load(open(recF))
        T = load_csv(os.path.join(STUDY, 'runs', tag + '_iterations.csv'))
        b = B[k]
        n = int(C['nOuter'])
        m = {'mesh': b['mesh'], 'NE': b['NE'], 'tol': C['tol'], 'prod': b,
             'cand': {kk: vv for kk, vv in C.items() if kk != 'log'},
             'candLog': C['log']}

        # ---------------- Phase 13: transition audit --------------------
        d = np.atleast_2d(np.array(C['descents'], dtype=float))
        if d.size == 0:
            d = np.zeros((0, 4))
        br = C['eventBranch']
        if isinstance(br, str):
            br = [br]
        tr = []
        for j, row in enumerate(d):
            it, stageFrom, declIter, declBegin = (int(row[0]), int(row[1]),
                                                  int(row[2]), int(row[3]))
            i0 = declIter - 1
            tr.append(dict(
                index=j + 1, iter=it, moveBefore=float(T['move'][it - 2]),
                moveAfter=float(T['move'][it - 1]), stageFrom=stageFrom,
                branch=br[j], declIter=declIter, declBegin=declBegin,
                A=int(T['exA'][i0]), B=int(T['exB'][i0]), E=int(T['exE'][i0]),
                nA=int(T['exNA'][i0]), nB=int(T['exNB'][i0]),
                betaStall=int(T['betaStallFires'][i0]),
                prodStageShadow=int(T['prodStageShadow'][i0]),
                omega1=float(T['omega1'][i0]), Mnd=float(T['Mnd'][i0]),
                gray=float(T['gray'][i0]), mid=float(T['mid'][i0]),
                volume=float(T['volume'][i0]),
                cosTheta=float(T['exCos'][i0]), netPath=float(T['exNet'][i0]),
                medcos=float(T['exMedcos'][i0]), mednet=float(T['exMednet'][i0]),
                amplitude=float(T['exAmp'][i0]), boundFrac=float(T['boundFrac'][i0])))
        m['transitions'] = tr

        m['allTransitionsAttributable'] = all(
            (t['A'] == 1 or t['B'] == 1) and (t['nA'] >= P or t['nB'] >= P)
            and t['iter'] == t['declIter'] + 1 for t in tr)
        changes = [i + 1 for i in range(1, n) if T['move'][i] != T['move'][i - 1]]
        m['moveChangeIters'] = changes
        m['noUndeclaredTransition'] = changes == [t['iter'] for t in tr]
        m['oneRungPerTransition'] = all(
            LEVELS.index(round(t['moveAfter'], 6)) - LEVELS.index(round(t['moveBefore'], 6)) == 1
            for t in tr)

        # ---------------- Phase 14: termination audit -------------------
        i1 = n - 1
        m['term'] = dict(
            status=C['status'], iter=n, move=float(T['move'][i1]),
            stage=int(T['stage'][i1]), A=int(T['exA'][i1]), B=int(T['exB'][i1]),
            E=int(T['exE'][i1]), nA=int(T['exNA'][i1]), nB=int(T['exNB'][i1]),
            branch=C['terminalBranch'], declIter=C['terminalDeclIter'],
            declBegin=C['terminalDeclBegin'],
            omega1=C['omega1'], omega2=C['omega2'], Mnd=C['Mnd_final'],
            gray=C['gray_final'], mid=C['mid_final'], volume=C['volume_final'],
            maxAbsDrho=float(T['maxAbs'][i1]), l2Drho=float(T['l2'][i1]),
            betaStall=int(T['betaStallFires'][i1]),
            prodStopAdmit=int(T['prodStopAdmit'][i1]),
            nativeStopWouldHold=int(T['prodStopRaw'][i1]),
            innerConv=int(T['innerConv'][i1]),
            innerNonConvTotal=int(C['innerNonConv']))
        m['convergenceHonest'] = (C['status'] != 'CONVERGED') or (
            abs(T['move'][i1] - 0.005) < 1e-12 and int(T['stage'][i1]) == 4
            and (T['exNA'][i1] >= P or T['exNB'][i1] >= P))
        m['capHit'] = C['status'] == 'CAP_HIT'
        m['reachedMoveMin'] = bool(np.any(np.abs(T['move'] - 0.005) < 1e-12))
        m['levelsVisited'] = sorted(set(round(float(x), 6) for x in T['move']), reverse=True)
        m['genuineTerminalExhaustion'] = (C['status'] == 'CONVERGED'
                                          and m['convergenceHonest'])

        # ---------------- Phase 15: causal comparison -------------------
        kP = b['firstDescentIter']
        kC = tr[0]['iter'] if tr else None
        m['delta'] = dict(
            Mnd_abs=C['Mnd_final'] - b['Mnd'],
            Mnd_rel_pct=100 * (C['Mnd_final'] - b['Mnd']) / b['Mnd'],
            omega1_abs=C['omega1'] - b['omega1'],
            omega1_rel_pct=100 * (C['omega1'] - b['omega1']) / b['omega1'],
            gray_abs=C['gray_final'] - b['gray'],
            mid_abs=C['mid_final'] - b['mid'],
            gap12_abs=C['gap12'] - b['gap12'],
            volume_abs=C['volume_final'] - b['volume'],
            outer_mult=C['nOuter'] / b['nOuter'],
            inner_mult=C['innerTotal'] / b['innerTotal'],
            wall_mult=C['wall_s'] / b['wall_s'],
            firstDescent_prod=kP, firstDescent_cand=kC,
            descentDelay=(kC - kP) if kC else None)

        if kC and kP <= n:
            a, z = int(kP) - 1, int(kC) - 2
            m['gainDuringDelay'] = dict(
                fromIter=int(kP), toIter=int(kC) - 1, extraIterations=int(kC) - int(kP),
                Mnd_at_prodDescent=float(T['Mnd'][a]), Mnd_at_candDescent=float(T['Mnd'][z]),
                Mnd_change=float(T['Mnd'][z] - T['Mnd'][a]),
                Mnd_change_pct=float(100 * (T['Mnd'][z] - T['Mnd'][a]) / T['Mnd'][a]),
                omega1_at_prodDescent=float(T['omega1'][a]),
                omega1_at_candDescent=float(T['omega1'][z]),
                omega1_change_pct=float(100 * (T['omega1'][z] - T['omega1'][a]) / T['omega1'][a]),
                gray_change=float(T['gray'][z] - T['gray'][a]),
                mid_change=float(T['mid'][z] - T['mid'][a]))

        # ---------------- Phase 18: physics safety ----------------------
        m['physics'] = dict(
            omega1_finite=bool(np.isfinite(C['omega1'])),
            omega2_finite=bool(np.isfinite(C['omega2'])),
            omega2_gt_omega1=bool(C['omega2'] > C['omega1']),
            subspaceMax=int(np.max(T['multN'])), subspaceMin=int(np.min(T['multN'])),
            gap12_min=float(np.min(T['gap12'])), gap12_final=C['gap12'],
            degenTotal=float(np.sum(T['degen'])),
            volume_maxDeviation=float(np.max(np.abs(T['volume'] - 0.5))),
            anyNaN=bool(np.any(~np.isfinite(T['omega1'])) or np.any(~np.isfinite(T['omega2']))
                        or np.any(~np.isfinite(T['Mnd']))),
            innerNonConv=int(C['innerNonConv']))

        # ---------------- beta authority --------------------------------
        shadow = C['prodShadowDescents']
        if isinstance(shadow, (int, float)):
            shadow = [shadow]
        m['beta'] = dict(
            firstStall=C['betaStallFirst'], stallCount=int(np.sum(T['betaStallFires'])),
            prodShadowDescents=list(shadow) if shadow else [],
            prodStopWouldAdmitAt=C['prodStopFirst'],
            candDescents=[t['iter'] for t in tr],
            anyDescentAtBetaStallOnly=any(not (t['nA'] >= P or t['nB'] >= P) for t in tr))

        # ---------------- terminal-stage regime diagnosis ---------------
        s_last = int(T['exStageStart'][i1])
        sl = slice(s_last - 1, n)
        amp, mc = T['exAmp'][sl], T['exMedcos'][sl]
        mcv = mc[np.isfinite(mc)]
        m['terminalStage'] = dict(
            stageStart=s_last, nIterations=n - s_last + 1, move=float(T['move'][i1]),
            tol=float(C['tol']),
            amp_median=float(np.median(amp)), amp_max=float(np.max(amp)),
            amp_over_tol_median=float(np.median(amp) / C['tol']),
            fracAmpBelowTol=float(np.mean(amp < C['tol'])),
            medcos_median=float(np.median(mcv)) if mcv.size else None,
            fracMedcosPositive=float(np.mean(mcv > 0)) if mcv.size else None,
            A_everTrue=bool(np.any(T['exA'][sl] > 0)), B_everTrue=bool(np.any(T['exB'][sl] > 0)),
            maxNA=int(np.max(T['exNA'][sl])), maxNB=int(np.max(T['exNB'][sl])),
            Mnd_min=float(np.min(T['Mnd'][sl])), Mnd_max=float(np.max(T['Mnd'][sl])),
            Mnd_range_pct=float(100 * (np.max(T['Mnd'][sl]) - np.min(T['Mnd'][sl]))
                                / np.median(T['Mnd'][sl])),
            omega1_min=float(np.min(T['omega1'][sl])), omega1_max=float(np.max(T['omega1'][sl])))
        ts = m['terminalStage']
        ts['blockedBranchA_byAmplitude'] = ts['fracAmpBelowTol'] > 0.99 and not ts['A_everTrue']
        ts['blockedBranchB_byCoherence'] = (ts['fracMedcosPositive'] is not None
                                            and ts['fracMedcosPositive'] < 0.01
                                            and not ts['B_everTrue'])
        ts['lowAmplitudeCancellation'] = bool(ts['blockedBranchA_byAmplitude']
                                              and ts['blockedBranchB_byCoherence'])

        A['mesh'][k] = m
        report_mesh(m, b, C, tr)

    A['prefix'] = prefix_check()
    A['gates'] = gates(A)
    out = os.path.join(STUDY, 'evidence', 'analysis.json')
    json.dump(A, open(out, 'w'), indent=1, default=str)
    print(f'\n  wrote {out}')
    return A


def report_mesh(m, b, C, tr):
    print(f"\n  --- {b['mesh'][0]}x{b['mesh'][1]} ---")
    print(f"   prod: {b['status']:<17} outer={b['nOuter']:<5} inner={b['innerTotal']:<6} "
          f"wall={b['wall_s']:<9.1f} omega1={b['omega1']:.6f} Mnd={b['Mnd']:.4f}  descent@{b['firstDescentIter']}")
    print(f"   cand: {C['status']:<17} outer={C['nOuter']:<5} inner={C['innerTotal']:<6} "
          f"wall={C['wall_s']:<9.1f} omega1={C['omega1']:.6f} Mnd={C['Mnd_final']:.4f}  "
          f"descent@{tr[0]['iter'] if tr else None}")
    d = m['delta']
    print(f"   dMnd = {d['Mnd_abs']:+.4f} ({d['Mnd_rel_pct']:+.2f} %)   "
          f"domega1 = {d['omega1_abs']:+.6f} ({d['omega1_rel_pct']:+.3f} %)")
    print(f"   cost: outer x{d['outer_mult']:.2f}  inner x{d['inner_mult']:.2f}  wall x{d['wall_mult']:.2f}")
    for t in tr:
        print(f"   T{t['index']}: iter {t['iter']:<5} {t['moveBefore']:g} -> {t['moveAfter']:<6g} "
              f"branch {t['branch']}  decl {t['declIter']} (window {t['declBegin']}-{t['declIter']})  "
              f"nA={t['nA']} nB={t['nB']}  betaStall={t['betaStall']}")
    tm = m['term']
    print(f"   term: {tm['status']} @ {tm['iter']}  move={tm['move']:g} stage={tm['stage']}  "
          f"branch={tm['branch']!r}  honest={m['convergenceHonest']}  genuine={m['genuineTerminalExhaustion']}")
    ts = m['terminalStage']
    print(f"   terminal stage from {ts['stageStart']} ({ts['nIterations']} iters at move {ts['move']:g}): "
          f"amp/tol median {ts['amp_over_tol_median']:.4f}, med20cos median "
          f"{ts['medcos_median']:.4f}, A ever {ts['A_everTrue']}, B ever {ts['B_everTrue']}")
    if ts['lowAmplitudeCancellation']:
        print("   *** LOW-AMPLITUDE CANCELLATION: A blocked by amplitude, B blocked by coherence ***")


def prefix_check():
    """C400 stage-1 prefix must reproduce the fixed-move arm F400 bitwise."""
    Pd = {'checked': False, 'detail': 'not run'}
    f = os.path.join(ROOT, 'evidence', 'move_activity_400', 'F400_400x50_trajectory.mat')
    c = os.path.join(ROOT, 'evidence', 'two_branch_controller_validation', 'C400x50_trajectory.mat')
    if not (os.path.isfile(f) and os.path.isfile(c)):
        Pd['detail'] = 'trajectory missing (C400 not executed)' if not os.path.isfile(c) \
                       else 'F400 missing'
        print(f"\n  prefix equivalence (C400 vs F400): {Pd['detail']}")
        return Pd
    with h5py.File(f, 'r') as hf, h5py.File(c, 'r') as hc:
        RF, RC = hf['RHO'], hc['RHO']          # v7.3 stores transposed: (nOuter, NE)
        nF, nC = RF.shape[0], RC.shape[0]
        n = min(nF, nC)
        rho_eq = bool(np.array_equal(RF[:n, :], RC[:n, :]))
        def hg(h, name):
            return np.array(h['hist'][name])
        om_eq = bool(np.array_equal(hg(hf, 'omega')[:, :n] if hg(hf, 'omega').ndim == 2
                                    else hg(hf, 'omega')[:n],
                                    hg(hc, 'omega')[:, :n] if hg(hc, 'omega').ndim == 2
                                    else hg(hc, 'omega')[:n]))
        be_eq = bool(np.array_equal(hg(hf, 'beta').ravel()[:n], hg(hc, 'beta').ravel()[:n]))
        dx_eq = bool(np.array_equal(hg(hf, 'dxNorm2').ravel()[:n], hg(hc, 'dxNorm2').ravel()[:n]))
        in_eq = bool(np.array_equal(hg(hf, 'nInner').ravel()[:n], hg(hc, 'nInner').ravel()[:n]))
        mv_ok = bool(np.all(hg(hc, 'move').ravel()[:n] == 0.04))
    Pd = dict(checked=True, nCompared=int(n), nF=int(nF), nC=int(nC),
              rho_bitwise=rho_eq, omega_bitwise=om_eq, beta_bitwise=be_eq,
              dx_bitwise=dx_eq, inner_bitwise=in_eq, move_allEqual=mv_ok)
    Pd['pass'] = all([rho_eq, om_eq, be_eq, dx_eq, in_eq, mv_ok])
    Pd['detail'] = (f"n={n} rho={rho_eq} omega={om_eq} beta={be_eq} dx={dx_eq} "
                    f"inner={in_eq} move004={mv_ok}")
    print(f"\n  prefix equivalence (C400 vs F400): {Pd['detail']} -> "
          f"{'PASS' if Pd['pass'] else 'FAIL'}")
    return Pd


def gates(A):
    ks = list(A['mesh'].keys())
    M = A['mesh']
    def every(fn):
        return all(fn(M[k]) for k in ks) if ks else False
    def has(k):
        return k in M
    G = {}
    G['P2'] = every(lambda x: x['convergenceHonest'])
    G['P3'] = every(lambda x: x['cand']['status'] != 'SOLVER_FAILURE'
                    and x['physics']['innerNonConv'] == 0)
    G['P4'] = has('m160') and M['m160']['delta']['firstDescent_cand'] is not None \
        and M['m160']['delta']['firstDescent_cand'] <= 400
    G['P5'] = has('m160') and M['m160']['cand']['Mnd_final'] <= 1.10 * M['m160']['prod']['Mnd'] \
        and M['m160']['cand']['omega1'] >= 0.99 * M['m160']['prod']['omega1']
    G['P6'] = has('m320') and M['m320']['delta']['firstDescent_cand'] >= \
        M['m320']['prod']['firstDescentIter'] + 50 \
        and M['m320']['cand']['Mnd_final'] <= 0.80 * M['m320']['prod']['Mnd']
    G['P7'] = has('m400') and M['m400']['delta']['firstDescent_cand'] >= \
        M['m400']['prod']['firstDescentIter'] + 50 \
        and M['m400']['cand']['Mnd_final'] <= 0.80 * M['m400']['prod']['Mnd']
    G['P8'] = every(lambda x: x['cand']['omega1'] >= 0.99 * x['prod']['omega1'])
    G['P9'] = every(lambda x: abs(x['cand']['volume_final'] - 0.5) <= 1e-4)
    G['P10'] = every(lambda x: x['physics']['omega1_finite'] and x['physics']['omega2_finite']
                     and x['physics']['omega2_gt_omega1'] and x['physics']['subspaceMax'] <= 2
                     and not x['physics']['anyNaN'] and x['physics']['innerNonConv'] == 0)
    G['P11'] = every(lambda x: x['allTransitionsAttributable'] and x['noUndeclaredTransition']
                     and x['oneRungPerTransition'] and not x['beta']['anyDescentAtBetaStallOnly'])
    G['P12'] = every(lambda x: x['cand']['status'] != 'CONVERGED' or
                     (abs(x['term']['move'] - 0.005) < 1e-12 and x['term']['stage'] == 4
                      and (x['term']['nA'] >= P or x['term']['nB'] >= P)))
    G['P13'] = every(lambda x: x['delta']['outer_mult'] <= 8 and x['delta']['wall_mult'] <= 10)
    G['allThreeRunsExecuted'] = len(ks) == 3
    G['allTerminatedGenuinely'] = every(lambda x: x['genuineTerminalExhaustion'])
    print('\n  GATES:')
    for k, v in G.items():
        print(f'    {k:<24} {"PASS" if v else "FAIL"}')
    return G


if __name__ == '__main__':
    analyse()
