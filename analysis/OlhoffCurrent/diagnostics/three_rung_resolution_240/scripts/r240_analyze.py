#!/usr/bin/env python3
"""r240_analyze -- Phases 13-19.  The rung-4 materiality test at 240x30.

Extracts S1/S2/S3/F, computes every rung increment against the INHERITED frozen
bars, and runs the preregistered running-best tail analysis.
"""
import os, sys, json, hashlib
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r240_frozen as F

HERE  = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT  = os.path.dirname(os.path.dirname(STUDY))
EV    = os.path.join(ROOT, 'evidence', 'three_rung_resolution_240')
TAG, NX, NY = 'C240x30', 240, 30
NE = NX * NY

# ---- PREREGISTRATION S6, inherited verbatim ----------------------------
TH = dict(Mnd_rel_pct=2.0, omega1_rel_pct=0.10, topo_frac=0.01, rho_mean_abs=0.01,
          volume_worsen=1e-5, cost_domination_mult=2.0)
VOL_TOL = 1e-4


def vechash(v):
    return hashlib.sha256(np.ascontiguousarray(v, dtype='<f8').tobytes()).hexdigest()


def state_at(T, RHO, tOuter, k, label):
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
        betaStallFires=bool(T['betaStallFires'][i]),
        nativeStopHolds=bool(T['prodStopRaw'][i]),
        nativeStopAdmitted=bool(T['prodStopAdmit'][i]),
        subspaceN=int(T['multN'][i]), degen=int(T['degen'][i]),
        nInner=int(T['nInner'][i]), innerCumulative=int(T['cumInner'][i]),
        wall_s_cumulative=float(np.sum(tOuter[:k])),
        rho_sha256=vechash(RHO[:, i]))


def delta(a, b, RHO, label):
    d = RHO[:, b['iteration'] - 1] - RHO[:, a['iteration'] - 1]
    return dict(
        block=label, frm=a['label'], to=b['label'],
        dMnd=b['Mnd'] - a['Mnd'], dMnd_rel_pct=100 * (b['Mnd'] - a['Mnd']) / a['Mnd'],
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
    V = json.load(open(os.path.join(STUDY, 'evidence', 'event_verification.json')))
    rec = json.load(open(os.path.join(STUDY, 'runs', f'{TAG}_record.json')))
    with h5py.File(os.path.join(EV, f'{TAG}_trajectory.mat'), 'r') as h:
        RHO = np.array(h['RHO']).T
        tOuter = np.array(h['hist']['tOuter']).ravel()
    T = np.genfromtxt(os.path.join(STUDY, 'runs', f'{TAG}_iterations.csv'),
                      delimiter=',', names=True)
    n = int(rec['nOuter'])
    k1, k2, k3 = V['kE1'], V['kE2'], V['kE3']

    S1 = state_at(T, RHO, tOuter, k1, 'S1 single-stage endpoint (move=0.04)')
    S2 = state_at(T, RHO, tOuter, k2, 'S2 two-rung endpoint (move=0.02)')
    S3 = state_at(T, RHO, tOuter, k3, 'S3 three-rung endpoint (move=0.01)')
    Fs = state_at(T, RHO, tOuter, n,  'F four-rung final (move=0.005)')
    for st, br in ((S1, V['kE1_branch']), (S2, V['kE2_branch']), (S3, V['kE3_branch'])):
        st['branch'] = br
    for st, i in ((S1, 0), (S2, 1), (S3, 2)):
        st['window'] = V['stages'][i]['offline_window']
    Fs['status'] = rec['status']; Fs['branch'] = V['terminal_branch']
    Fs['window'] = [rec['terminalDeclBegin'], rec['terminalDeclIter']]
    S3['policy_status'] = 'CONVERGED (three-rung policy would terminate here)'

    rungs = {
        'rung1': delta(dict(label='initial uniform design', iteration=1,
                            Mnd=float(T['Mnd'][0]), omega1=float(T['omega1'][0]),
                            omega2=float(T['omega2'][0]), gap12=float(T['gap12'][0]),
                            volume=float(T['volume'][0]), gray=float(T['gray'][0]),
                            mid=float(T['mid'][0]), subspaceN=int(T['multN'][0]),
                            innerCumulative=int(T['cumInner'][0]),
                            wall_s_cumulative=float(tOuter[0])),
                      S1, RHO, 'RUNG 1 (0.04):  start -> S1'),
        'rung2': delta(S1, S2, RHO, 'RUNG 2 (0.02):  S1 -> S2'),
        'rung3': delta(S2, S3, RHO, 'RUNG 3 (0.01):  S2 -> S3'),
        'rung4': delta(S3, Fs, RHO, 'RUNG 4 (0.005): S3 -> F   <-- THE TEST'),
        'rungs34': delta(S2, Fs, RHO, 'RUNGS 3+4 combined: S2 -> F')}
    mat = {r: materiality(rungs[r]) for r in ('rung2', 'rung3', 'rung4', 'rungs34')}

    # ---- PREREGISTRATION S7: running-best over the rung-4 tail ----------
    tail = slice(k3, n)                       # iterations k3+1 .. n
    tw1, tMnd = T['omega1'][tail], T['Mnd'][tail]
    best_w1_i = int(k3 + 1 + np.argmax(tw1)); best_Mnd_i = int(k3 + 1 + np.argmin(tMnd))
    tailA = dict(
        tail_from=k3 + 1, tail_to=n, tail_len=n - k3,
        best_omega1=float(np.max(tw1)), best_omega1_iter=best_w1_i,
        best_omega1_rel_pct=100 * (float(np.max(tw1)) - S3['omega1']) / S3['omega1'],
        best_omega1_material=bool(100 * (float(np.max(tw1)) - S3['omega1']) / S3['omega1']
                                  >= TH['omega1_rel_pct']),
        best_Mnd=float(np.min(tMnd)), best_Mnd_iter=best_Mnd_i,
        best_Mnd_rel_pct=100 * (float(np.min(tMnd)) - S3['Mnd']) / S3['Mnd'],
        best_Mnd_material=bool(-100 * (float(np.min(tMnd)) - S3['Mnd']) / S3['Mnd']
                               >= TH['Mnd_rel_pct']),
        final_omega1=Fs['omega1'], final_Mnd=Fs['Mnd'],
        # is anything still trending at the end? last 100 vs preceding 100
        omega1_last100_mean=float(np.mean(T['omega1'][n - 100:n])),
        omega1_prev100_mean=float(np.mean(T['omega1'][n - 200:n - 100])),
        Mnd_last100_mean=float(np.mean(T['Mnd'][n - 100:n])),
        Mnd_prev100_mean=float(np.mean(T['Mnd'][n - 200:n - 100])))
    tailA['omega1_trend_last200_rel_pct'] = 100 * (
        tailA['omega1_last100_mean'] - tailA['omega1_prev100_mean']) / tailA['omega1_prev100_mean']
    tailA['Mnd_trend_last200_rel_pct'] = 100 * (
        tailA['Mnd_last100_mean'] - tailA['Mnd_prev100_mean']) / tailA['Mnd_prev100_mean']
    # stage-4 dynamics: is it the 320x40 low-amplitude-cancellation regime?
    st4 = slice(k3, n)
    tol = F.tol_for(NE)
    tailA['stage4_amp_below_tol_frac'] = float(np.mean(T['l2'][st4] < tol))
    tailA['stage4_medcos_negative_frac'] = float(np.nanmean(T['exMedcos'][st4] < 0))
    tailA['stage4_E_true_frac'] = float(np.mean(T['exE'][st4] != 0))
    tailA['stage4_lowamp_cancellation_frac'] = float(np.mean(
        (T['l2'][st4] < tol) & (np.nan_to_num(T['exMedcos'][st4], nan=1.0) < 0)))

    phys = dict(
        subspaceN_prefix_always2=bool(np.all(T['multN'][:k3] == 2)),
        subspaceN_full_always2=bool(np.all(T['multN'][:n] == 2)),
        omega2_gt_omega1_prefix=bool(np.all(T['omega2'][:k3] > T['omega1'][:k3])),
        omega2_gt_omega1_full=bool(np.all(T['omega2'][:n] > T['omega1'][:n])),
        omega_finite=bool(np.all(np.isfinite(T['omega1'][:n]))
                          and np.all(np.isfinite(T['omega2'][:n]))),
        innerNonConv_prefix=int(np.sum(T['innerConv'][:k3] == 0)),
        innerNonConv_total=int(np.sum(T['innerConv'][:n] == 0)),
        min_gap12_S3_to_F=float(np.min(T['gap12'][k3 - 1:n])),
        min_gap12_full=float(np.min(T['gap12'][:n])),
        degenTotal_prefix=float(np.sum(T['degen'][:k3])),
        degenTotal_full=float(np.sum(T['degen'][:n])),
        degen_note=('degen counts EXPECTED near-degeneracy hits in the multiplicity-aware '
                    'subspace; not a failure indicator and not part of the frozen gate'))

    bounds = [(1, k1), (k1 + 1, k2), (k2 + 1, k3), (k3 + 1, n)]
    per_stage = []
    for si, (a, b) in enumerate(bounds):
        inner = int(T['cumInner'][b - 1] - (T['cumInner'][a - 2] if a > 1 else 0))
        per_stage.append(dict(
            rung=si + 1, move=float(T['move'][a - 1]), iterFrom=a, iterTo=b,
            outer=b - a + 1, inner=inner, wall_s=float(np.sum(tOuter[a - 1:b])),
            pct_total_outer=100 * (b - a + 1) / n,
            pct_total_inner=100 * inner / int(Fs['innerCumulative']),
            terminal_status=('CONVERGED' if b == n else 'DESCENDED')))
    cost = dict(
        total_outer=n, total_inner=int(Fs['innerCumulative']),
        total_wall_s=float(np.sum(tOuter[:n])),
        S3_outer=k3, S3_inner=int(S3['innerCumulative']), S3_wall_s=S3['wall_s_cumulative'],
        saved_outer=n - k3, saved_inner=int(Fs['innerCumulative'] - S3['innerCumulative']),
        saved_wall_s=float(Fs['wall_s_cumulative'] - S3['wall_s_cumulative']),
        saved_outer_pct=100 * (n - k3) / n,
        saved_inner_pct=100 * (Fs['innerCumulative'] - S3['innerCumulative'])
                        / int(Fs['innerCumulative']),
        saved_wall_pct=100 * (Fs['wall_s_cumulative'] - S3['wall_s_cumulative'])
                       / Fs['wall_s_cumulative'],
        per_stage=per_stage,
        rung4_cost_mult_vs_S3=(n - k3) / k3)
    mat['rung4']['cost_dominated'] = bool(cost['rung4_cost_mult_vs_S3']
                                          >= TH['cost_domination_mult'] and not mat['rung4']['any'])
    mat['rung4']['failure_risk'] = bool(rec['status'] == 'CAP_HIT')
    mat['rung3']['cost_dominated'] = False; mat['rung3']['failure_risk'] = False

    # production baseline: none exists at 240x30
    prod = dict(available=False,
                reason=('no 240x30 production baseline exists; baselines.json covers only '
                        '160x20, 320x40 and 400x50.  Production-relative omega1/M_nd gates '
                        'are UNAVAILABLE and are NOT imputed from another mesh.'))

    spi = lambda a, b: float(np.sum(tOuter[a:b]) / max(np.sum(T['nInner'][a:b]), 1))
    blocks = {'first50': spi(0, 50), 'rung1': spi(0, k1), 'rung2': spi(k1, k2),
              'rung3': spi(k2, k3), 'rung4': spi(k3, n)}
    wall = dict(s_per_inner=blocks, drift_ratio=max(blocks.values()) / min(blocks.values()),
                reliable=bool(max(blocks.values()) / min(blocks.values()) < 1.5))

    out = dict(thresholds=TH, volume_gate=VOL_TOL,
               relative_convention='100*(b-a)/a, normalised by the earlier state (inherited)',
               mesh=[NX, NY], NE=NE, tol=F.tol_for(NE), status=rec['status'],
               nOuter=n, cap=rec['cap'],
               S1=S1, S2=S2, S3=S3, F=Fs, rungs=rungs, materiality=mat,
               tail_analysis=tailA, physics=phys, cost=cost,
               production=prod, wall_reliability=wall,
               counterfactual_validity=V['counterfactual_validity'],
               replay_all_match=V['replay_all_match'],
               declaration_timing=[{k: s[k] for k in
                   ('stage','move','stageStart','stageEnd','nIter','first_evaluable_iter',
                    'first_median_defined_iter','first_evaluable_matches_theory',
                    'first_E_true_iter','earliest_possible_declaration','offline_decl',
                    'offline_branch','declaration_offset','declared_at_earliest_possible',
                    'E_true_from_first_evaluable','E_unbroken_to_declaration',
                    'E_true_fraction_after_first_evaluable')} for s in V['stages']])

    p = os.path.join(STUDY, 'evidence', 'analysis.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('written', p)


if __name__ == '__main__':
    main()
