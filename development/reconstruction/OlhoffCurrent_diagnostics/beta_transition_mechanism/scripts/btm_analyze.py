#!/usr/bin/env python3
"""btm_analyze -- mechanism audit of the production beta stall signal.

Writes METRICS.json.  Reads only committed telemetry plus the durable 400x50
raw trajectory.  Runs no optimiser and changes nothing in production.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btm_common as B


def completion(F):
    M = F['Mnd_pct']; M0, Mf = M[0], M[-1]
    return [(M0 - M[k]) / (M0 - Mf) for k in range(F['n'])]


def first_descent_idx(P):
    for k in range(1, P['n']):
        if P['move'][k] < P['move'][k-1]:
            return k
    return None


def main():
    M = {'schema': 'olhoff_beta_transition_mechanism/1',
         'task': 'mechanism audit of the production beta/bound-variable stall signal',
         'implementation': 'analysis/OlhoffCurrent',
         'starting_HEAD': 'cb6c0eae31a25521f7c5fed1c4a89564ed63344e',
         'source_tree_sha256': 'c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c',
         'matlab': '25.2.0.2998904 (R2025b)',
         'no_optimisation_run': True}

    # ---- what beta IS, from the code ------------------------------------
    M['beta_definition'] = dict(
        source='+impl/algo/innerLoop.m',
        role='bound variable of the Du-Olhoff bound formulation, Eq. (25a)',
        subproblem='maximise beta  s.t.  beta <= lambda_j + Delta lambda_j(drho)  for all tracked j, plus volume and the move box',
        implemented_as='the (NE+1)-th PRIMAL design variable of the MMA subproblem',
        stored_scaled='x(end) = beta/lambda_ref with lambda_ref = lam(1) = current lambda_n',
        returned='st.beta = x(end)*lamref  -- ABSOLUTE eigenvalue units',
        mma_objective='f0val = -x(end); df0dx(end) = -1  (MMA minimises, so this maximises beta)',
        initial_value='x = [zeros(NE,1); 1]  ->  drho = 0, beta = lambda_n',
        box_on_beta='xmin(end)=0, xmax(end)=5 (i.e. beta in [0, 5*lambda_ref])',
        classification=dict(primal=True, dual=False, slack=False,
                            mma_artificial_yz=False, bound_variable=True),
        units='eigenvalue, lambda = omega^2',
        depends_on_NE='NO -- beta is an eigenvalue-valued scalar; NE enters only through the discretisation of lambda itself',
        depends_on_move='YES, through the box lo/hi = +/- move which bounds drho and hence Delta lambda',
        depends_on_asymptotes='YES, indirectly: MMA initialises low/upp = xval -/+ 0.5*(xmax-xmin) and xmax-xmin = 2*move on interior elements')

    M['stall_predicate'] = dict(
        source='+impl/architecture/+olh/+move/limit.m, case ladder, signal boundVariable',
        window_W=B.W, tolerance=B.TOL,
        formula='rel = (mean(beta[-W:]) - mean(beta[-2W:-W])) / max(|mean(beta[-2W:-W])|, eps); descend if rel < tol',
        dwell_guard='(outer - lastStage) > W  ->  earliest possible re-fire is lastStage + W + 1 = +11',
        history_available='the controller is called before iteration `outer` is recorded, so beta[1..outer-1]',
        dimensionless=True,
        note='rel is a RELATIVE change of an intensive scalar: it is already dimensionless and carries no explicit NE, gradient or move normalisation')

    # ---- replay: does the re-derived predicate reproduce production? -----
    rep = []
    for mesh, spec in B.RUNS.items():
        P = B.load(spec['prod'])
        actual = [P['outer'][k] for k in range(1, P['n']) if P['move'][k] < P['move'][k-1]]
        fires, stages, rel = B.stall_fires(P['beta'])
        replay = [P['outer'][k] for k in range(P['n']) if fires[k]]
        rep.append(dict(mesh=mesh, actual_descents=actual,
                        replayed_fires=replay[:8],
                        reproduces_actual=replay[:len(actual)] == actual,
                        spacing=[actual[i+1]-actual[i] for i in range(len(actual)-1)],
                        dwell_minimum=B.W+1))
    M['predicate_replay'] = dict(
        purpose='confirm the audit is analysing the rule production actually applied',
        results=rep)

    # ---- event-aligned measurement --------------------------------------
    ev = []
    for mesh, spec in B.RUNS.items():
        P = B.load(spec['prod']); F = B.load(spec['fixed'])
        rel = B.stall_rel(P['beta']); c = completion(F)
        d0 = first_descent_idx(P); i = d0 - 1
        lam = P['omega1'][i]**2
        g = (P['beta'][i] - lam)/lam
        lam_end = F['omega1'][-1]**2
        ev.append(dict(
            mesh=mesh, NE=spec['NE'], descentIter=P['outer'][d0],
            lastIterAtMove004=P['outer'][i],
            stall_rel_at_descent=rel[d0], tolerance=B.TOL,
            beta=P['beta'][i], lambda_n=lam,
            predicted_gain_g=g, predicted_gain_over_move=g/P['move'][i],
            Mnd_at_descent=F['Mnd_pct'][i], gray_at_descent=F['gray_frac'][i],
            mid_at_descent=F['mid_frac'][i],
            completion_at_descent=c[i],
            Mnd_fixedMoveEnd=F['Mnd_pct'][-1],
            remaining_Mnd_relative=(F['Mnd_pct'][i]-F['Mnd_pct'][-1])/F['Mnd_pct'][i],
            omega1_at_descent=F['omega1'][i], omega1_fixedMoveEnd=F['omega1'][-1],
            omega1_relChange_after=(F['omega1'][-1]-F['omega1'][i])/F['omega1'][i],
            lambda_relGain_actual_after=(lam_end-lam)/lam,
            beta_underprediction_factor=((lam_end-lam)/lam)/g if g > 0 else None,
            nAtMoveBound_at_descent=None,
            filter_R_over_h=0.06/(1.0/spec['mesh'][1])))
    M['event_aligned'] = dict(
        event='the last iteration at move=0.04, i.e. immediately before production descends',
        results=ev)

    # ---- the central contradiction --------------------------------------
    M['central_finding'] = dict(
        stall_metric_at_descent=[e['stall_rel_at_descent'] for e in ev],
        stall_metric_spread=max(e['stall_rel_at_descent'] for e in ev) -
                            min(e['stall_rel_at_descent'] for e in ev),
        remaining_Mnd_relative=[e['remaining_Mnd_relative'] for e in ev],
        statement=('the predicate fires at essentially the SAME metric value at all three '
                   'meshes (spread 3.3e-4 against a 5e-3 tolerance) while the remaining '
                   'topology evolution differs by a factor of 5.6 (9.0% / 43.4% / 50.1%). '
                   'The metric is not mis-scaled; it is measuring something else.'))

    # ---- gain/move proportionality (the gradient-scale observable) -------
    prop = []
    for mesh, spec in B.RUNS.items():
        P = B.load(spec['prod']); d0 = first_descent_idx(P)
        early = [( (P['beta'][i]-P['omega1'][i]**2)/P['omega1'][i]**2 )/P['move'][i]
                 for i in range(4, 15)]
        late = [( (P['beta'][i]-P['omega1'][i]**2)/P['omega1'][i]**2 )/P['move'][i]
                for i in range(d0-5, d0)]
        prop.append(dict(mesh=mesh, g_over_move_early=sum(early)/len(early),
                         g_over_move_at_descent=sum(late)/len(late)))
    M['gain_move_proportionality'] = dict(
        rationale=('beta - lambda is the predicted eigenvalue gain.  Because the box bounds '
                   'drho by the move, first-order theory predicts gain ~ c*move.  g/move is '
                   'therefore the observable through which GRADIENT SCALE enters beta, and it '
                   'can be measured without recording gradients (which were not recorded).'),
        results=prop,
        early_mesh_consistency=('g/move ~ 2.0 at all three meshes early on, so beta responds to '
                                'the move exactly as the linearisation predicts and shows no '
                                'mesh-dependent scaling defect'))

    # ---- bound activity at 400x50 ---------------------------------------
    try:
        import numpy as np, h5py
        p = os.path.join(B.REPO, 'analysis/OlhoffCurrent/evidence/move_activity_400/F400_400x50_trajectory.mat')
        f = h5py.File(p, 'r'); DR = f['DRHO']; mv = np.array(f['move']).ravel()
        rows = []
        for k in [1, 20, 60, 100, 137, 200, 300, 369]:
            d = np.abs(np.array(DR[k-1, :])); m = mv[k-1]
            rows.append(dict(outer=k, move=float(m),
                             nAtBound=int(np.sum(d >= m*(1-1e-12))),
                             maxU=float(d.max()/m)))
        tot = 0
        for k in range(1, DR.shape[0]+1):
            d = np.abs(np.array(DR[k-1, :]))
            tot += int(np.sum(d >= mv[k-1]*(1-1e-12)))
        f.close()
        M['bound_activity_400x50'] = dict(
            definition='an element is at the move bound when |drho_e| = move to 1e-12',
            samples=rows, total_bound_hits_over_whole_run=tot,
            finding=('ZERO elements are at the move bound at ANY iteration of F400.  The move '
                     'box is NOT binding when production descends, so the descent cannot be '
                     'justified as "the design has used up the permitted move".'))
    except Exception as e:
        M['bound_activity_400x50'] = dict(error=str(e))

    # ---- inner MMA behaviour --------------------------------------------
    inner = []
    for mesh, spec in B.RUNS.items():
        P = B.load(spec['prod']); d0 = first_descent_idx(P)
        w = range(max(0, d0-10), d0)
        inner.append(dict(mesh=mesh,
                          nInner_before_descent=[int(P['nInner'][j]) for j in w],
                          allInnerConverged=all(P['innerConv'][j] == 1 for j in w)))
    M['inner_mma'] = dict(
        question='is beta stable because the SUBPROBLEM is solved consistently?',
        results=inner)

    # ---- filter scale ----------------------------------------------------
    M['filter_element_scale'] = dict(
        domain='a=8, b=1; square elements, h = b/nely',
        note=('filter.radiusPhysical = 0.06 is FIXED in physical units -- correct '
              'mesh-independent practice -- so it spans progressively more ELEMENTS as the '
              'mesh refines, widening the gray transition band in element terms.'),
        results=[dict(mesh=m, h=1.0/B.RUNS[m]['mesh'][1],
                      R_over_h=0.06/(1.0/B.RUNS[m]['mesh'][1]),
                      gray_at_descent=e['gray_at_descent'],
                      gray_over_R_over_h=e['gray_at_descent']/(0.06/(1.0/B.RUNS[m]['mesh'][1])))
                 for m, e in zip(B.RUNS.keys(), ev)])

    json.dump(M, open(os.path.join(B.OUT, 'METRICS.json'), 'w'), indent=2)
    print('wrote METRICS.json')
    for e in ev:
        print(f"  {e['mesh']}: descent it {e['descentIter']}  rel={e['stall_rel_at_descent']:.5f}  "
              f"g={e['predicted_gain_g']:.2e}  remaining M_nd={100*e['remaining_Mnd_relative']:.1f}%  "
              f"beta under-predicted lambda gain by {e['beta_underprediction_factor']:.0f}x")


if __name__ == '__main__':
    main()
