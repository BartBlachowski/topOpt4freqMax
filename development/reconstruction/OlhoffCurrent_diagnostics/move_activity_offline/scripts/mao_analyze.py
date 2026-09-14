#!/usr/bin/env python3
"""mao_analyze -- offline characterisation of the design-activity signal.

Writes METRICS.json.  Runs no optimiser and reads only committed CSVs.

The two fixed-move runs are the analytical backbone: they are BITWISE IDENTICAL
to the production baselines up to the iteration before production's first move
descent (verified here, not assumed), so they are an exact counterfactual
continuation of production at move = 0.04.  That is what makes "how much
evolution remained when production descended" a measurement rather than a guess.
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mao_common as M

L = {r['key']: M.load(r) for r in M.RUNS}
A, B = L['ms_fixedmove_160x20'], L['ms_fixedmove_320x40']
NE_RATIO = B['NE'] / A['NE']
THRS = ['1e-4', 'epsRMS', '1e-3', '1e-2']
# u-equivalent of each absolute |drho| threshold at move = 0.04
UEQ = {t: (8.83883476483184e-4 if t == 'epsRMS' else float(t)) / 0.04 for t in THRS}


def completion(R):
    """Fraction of this run's total M_nd decrease that is complete at k."""
    M0, Mf = R['Mnd'][0], R['Mnd'][-1]
    return [(M0 - R['Mnd'][k]) / (M0 - Mf) for k in range(R['n'])]


def spearman(x, y):
    n = len(x)
    def rank(v):
        idx = sorted(range(n), key=lambda i: v[i]); r = [0.0]*n; i = 0
        while i < n:
            j = i
            while j+1 < n and v[idx[j+1]] == v[idx[i]]: j += 1
            a = (i+j)/2.0 + 1
            for k in range(i, j+1): r[idx[k]] = a
            i = j+1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    den = math.sqrt(sum((rx[i]-mx)**2 for i in range(n)) * sum((ry[i]-my)**2 for i in range(n)))
    return num/den if den else float('nan')


def fire(S, theta, D):
    """First index at which S has been strictly below theta for D consecutive iters."""
    run = 0
    for k in range(len(S)):
        run = run + 1 if S[k] < theta else 0
        if run >= D: return k
    return None


def theta_window(R, c, S, D, cmin):
    """Range of thresholds for which the D-persistent rule fires at completion >= cmin."""
    v = [x for x in S if x == x and x > 0]
    if not v: return None
    lo, hi = min(v), max(v)
    grid = [lo*(hi/lo)**(i/600.0) for i in range(601)]
    ok = [t for t in grid if (lambda k: k is not None and c[k] >= cmin)(fire(S, t, D))]
    return (min(ok), max(ok)) if ok else None



def monotone(c):
    """Running max of the completion series: a monotone maturity envelope.

    M_nd is not monotone in k (it fluctuates), so matching two runs "at equal
    maturity" needs a monotone index.  Taking the running max is the weakest
    such construction that uses no smoothing and invents no values.
    """
    out, m = [], -1e9
    for x in c:
        m = max(m, x); out.append(m)
    return out


def at_completion(cmono, v, target):
    """Value of v at the first iteration whose maturity envelope reaches target."""
    for i in range(len(cmono)):
        if cmono[i] >= target: return v[i]
    return v[-1]


def collapse_residual(vA, vB, cA, cB, alpha, grid):
    """RMS of log(S_320/S_160) at matched maturity -- 0 means the meshes collapse."""
    fA, fB = A['NE']**alpha, B['NE']**alpha
    d = []
    for g in grid:
        x = at_completion(cA, vA, g)/fA
        y = at_completion(cB, vB, g)/fB
        if x > 0 and y > 0: d.append(math.log(y/x))
    return math.sqrt(sum(z*z for z in d)/len(d)) if d else float('nan')


def main():
    cA, cB = completion(A), completion(B)
    Mx = {}

    # ---- 0. provenance-ish identity checks --------------------------------
    ident = {}
    for mesh, b, f, d in [('160x20', 'ms_baseline_160x20', 'ms_fixedmove_160x20', 79),
                          ('320x40', 'ms_baseline_320x40', 'ms_fixedmove_320x40', 130)]:
        Bl, F = L[b], L[f]; n = d-1
        ident[mesh] = dict(
            iterationsCompared=n,
            maxAbsDiff_Mnd_maxAbs_l2=max(abs(Bl['Mnd'][i]-F['Mnd'][i]) + abs(Bl['maxAbs'][i]-F['maxAbs'][i])
                                          + abs(Bl['l2'][i]-F['l2'][i]) for i in range(n)),
            note='fixed-move arm is an exact counterfactual continuation of the baseline')
    Mx['baseline_fixedmove_identity'] = ident
    Mx['rms_l2_identity_maxRelErr'] = {k: L[k]['rmsIdentityMaxErr'] for k in L}

    # ---- 1. per-run activity summary --------------------------------------
    runs = []
    for k, R in L.items():
        rec = dict(key=k, study=R['study'], mesh=list(R['mesh']), NE=R['NE'],
                   policy=R['policy'], nIter=R['n'], productionStopIter=R['prodStop'],
                   csv=os.path.relpath(R['path'], M.REPO),
                   descents=M.descents(R),
                   Mnd_first=R['Mnd'][0], Mnd_last=R['Mnd'][-1],
                   omega1_last=R['omega1'][-1],
                   maxU_min=min(R['maxU']), maxU_max=max(R['maxU']),
                   maxU_median=sorted(R['maxU'])[R['n']//2],
                   rmsU_min=min(R['rmsU']), rmsU_max=max(R['rmsU']),
                   Neff_min=min(R['Neff']), Neff_max=max(R['Neff']),
                   Neff_final=R['Neff'][-1],
                   hasExactActiveCounts=R['nactive'])
        runs.append(rec)
    Mx['runs'] = runs

    # ---- 2. what production descent actually cost --------------------------
    dsc = []
    for mesh, base, fm, i in [('160x20', 'ms_baseline_160x20', 'ms_fixedmove_160x20', 77),
                              ('320x40', 'ms_baseline_320x40', 'ms_fixedmove_320x40', 128)]:
        Bl, F = L[base], L[fm]
        rem = F['Mnd'][i] - F['Mnd'][-1]
        w = M.window(Bl, Bl['outer'][i]+1, 10)
        d = dict(mesh=mesh, descentIter=Bl['outer'][i]+1, lastIterAtMove004=Bl['outer'][i],
                 move_from=0.04, move_to=0.02,
                 Mnd_at=F['Mnd'][i], Mnd_fixedMoveFinal=F['Mnd'][-1],
                 Mnd_remaining=rem, Mnd_remaining_relative=rem/F['Mnd'][i],
                 omega1_at=F['omega1'][i], omega1_fixedMoveFinal=F['omega1'][-1],
                 omega1_relChange=abs(F['omega1'][-1]-F['omega1'][i])/F['omega1'][i],
                 activity_at_descent=dict(
                     maxU=Bl['maxU'][i], rmsU=Bl['rmsU'][i],
                     Neff=Bl['Neff'][i], phiEff=Bl['phiEff'][i],
                     exactActiveCount={t: Bl['nActive'][t][i] for t in THRS},
                     exactActiveFrac={t: Bl['nActive'][t][i]/Bl['NE'] for t in THRS}),
                 preceding10=dict(
                     maxU=[Bl['maxU'][j] for j in w], rmsU=[Bl['rmsU'][j] for j in w],
                     Neff=[Bl['Neff'][j] for j in w],
                     activeFrac_1e3=[Bl['nActive']['1e-3'][j]/Bl['NE'] for j in w]))
        dsc.append(d)
    Mx['production_first_descent'] = dsc

    # the cross-mesh ordering that any threshold rule would have to satisfy
    need = []
    iA, iB = 77, 128
    cand = {
        'maxU': (L['ms_baseline_160x20']['maxU'][iA], L['ms_baseline_320x40']['maxU'][iB]),
        'rmsU': (L['ms_baseline_160x20']['rmsU'][iA], L['ms_baseline_320x40']['rmsU'][iB]),
        'Neff': (L['ms_baseline_160x20']['Neff'][iA], L['ms_baseline_320x40']['Neff'][iB]),
        'phiEff': (L['ms_baseline_160x20']['phiEff'][iA], L['ms_baseline_320x40']['phiEff'][iB]),
    }
    for t in THRS:
        cand[f'activeFrac>{t}'] = (L['ms_baseline_160x20']['nActive'][t][iA]/3200,
                                   L['ms_baseline_320x40']['nActive'][t][iB]/12800)
        cand[f'activeCount>{t}'] = (L['ms_baseline_160x20']['nActive'][t][iA],
                                    L['ms_baseline_320x40']['nActive'][t][iB])
    for nm, (a, b) in cand.items():
        need.append(dict(statistic=nm, at_160x20_iter78=a, at_320x40_iter129=b,
                         requiredOrdering='S(160x20) < S(320x40)',
                         satisfied=bool(a < b)))
    Mx['cross_mesh_ordering_at_production_descent'] = dict(
        rationale=('160x20 had 9.0% of its M_nd evolution left, 320x40 had 43.4%. '
                   'A rule "descend when S < theta" must therefore permit at 160x20 '
                   'and block at 320x40, which requires S(160x20@78) < S(320x40@129).'),
        tests=need)

    # ---- 3. scaling exponent of the mature active set ----------------------
    exps = []
    for cmin in [0.95, 0.99, 0.999]:
        for D in [5, 10, 20]:
            for t in THRS + ['Neff']:
                SA = A['nActive'][t] if t != 'Neff' else A['Neff']
                SB = B['nActive'][t] if t != 'Neff' else B['Neff']
                wa, wb = theta_window(A, cA, SA, D, cmin), theta_window(B, cB, SB, D, cmin)
                rec = dict(quantity=t, c_min=cmin, persistence=D)
                if wa is None or wb is None:
                    rec.update(admissible=False,
                               reason='160x20 never admissible' if wa is None else '320x40 never admissible')
                else:
                    alo = math.log(wb[0]/wa[1])/math.log(NE_RATIO)
                    ahi = math.log(wb[1]/wa[0])/math.log(NE_RATIO)
                    rec.update(window_160x20=list(wa), window_320x40=list(wb),
                               alpha_lo=alo, alpha_hi=ahi,
                               admissible=bool(alo <= ahi),
                               includes_alpha_0_constantCount=bool(alo <= 0 <= ahi),
                               includes_alpha_0p5_interfaceLength=bool(alo <= 0.5 <= ahi),
                               includes_alpha_1_constantFraction=bool(alo <= 1 <= ahi))
                exps.append(rec)
    Mx['normalisation_exponent'] = dict(
        model='statistic = activeCount / NE^alpha',
        derivation=('Dividing a count series by the per-mesh constant NE^alpha rescales its '
                    'admissible-threshold window exactly, so a single threshold works at both '
                    'meshes iff the count windows overlap after that rescaling, i.e. '
                    'alpha in [log_R(loB/hiA), log_R(hiB/loA)] with R = NE_320/NE_160 = 4.'),
        NE_ratio=NE_RATIO, results=exps)

    # ---- 4. predictive value: does S track REMAINING evolution? ------------
    pred = []
    for name in ['maxU', 'rmsU', 'Neff', 'phiEff'] + [f'n>{t}' for t in THRS] + [f'frac>{t}' for t in THRS]:
        row = dict(statistic=name)
        for R, tag in [(A, '160x20'), (B, '320x40')]:
            if name in ('maxU', 'rmsU', 'Neff', 'phiEff'): S = R[name]
            elif name.startswith('n>'): S = R['nActive'][name[2:]]
            else: S = [x/R['NE'] for x in R['nActive'][name[5:]]]
            rem = [R['Mnd'][k]-R['Mnd'][-1] for k in range(R['n'])]
            row[f'spearman_{tag}'] = spearman(S, rem)
        pred.append(row)
    # omega1 trailing relative range, for comparison
    orow = dict(statistic='omega1_relRange_W10')
    for R, tag in [(A, '160x20'), (B, '320x40')]:
        s = []
        for k in range(R['n']):
            w = R['omega1'][max(0, k-9):k+1]
            s.append((max(w)-min(w))/abs(sum(w)/len(w)))
        rem = [R['Mnd'][k]-R['Mnd'][-1] for k in range(R['n'])]
        orow[f'spearman_{tag}'] = spearman(s, rem)
    pred.append(orow)
    Mx['predictive_correlation'] = dict(
        target='remaining M_nd evolution at move=0.04, rem(k) = Mnd(k) - Mnd(final)',
        method='Spearman rank correlation over the whole fixed-move trajectory',
        note='Positive and large = statistic falls as the topology matures, WITHIN that mesh.',
        results=pred)

    # ---- 5. omega1 as a maturity proxy ------------------------------------
    om = []
    for mesh, R, i in [('160x20', A, 77), ('320x40', B, 128)]:
        w0, w1 = R['omega1'][i], R['omega1'][-1]
        m0, m1 = R['Mnd'][i], R['Mnd'][-1]
        g0, g1 = R['gray'][i], R['gray'][-1]
        om.append(dict(mesh=mesh, fromIter=R['outer'][i], toIter=R['outer'][-1],
                       omega1_relChange=abs(w1-w0)/w0, Mnd_relChange=abs(m1-m0)/m0,
                       gray_relChange=abs(g1-g0)/g0,
                       ratio_Mnd_over_omega1=(abs(m1-m0)/m0)/(abs(w1-w0)/w0)))
    Mx['omega1_as_maturity_proxy'] = dict(
        question='does omega1 settling imply the topology has finished evolving?',
        results=om)

    # ---- 6. self-normalised variant (peak-relative) ------------------------
    selfn = []
    for t in THRS:
        pa, pb = max(A['nActive'][t]), max(B['nActive'][t])
        SA = [x/pa for x in A['nActive'][t]]; SB = [x/pb for x in B['nActive'][t]]
        for cmin in [0.95, 0.99]:
            wa, wb = theta_window(A, cA, SA, 10, cmin), theta_window(B, cB, SB, 10, cmin)
            rec = dict(threshold=t, c_min=cmin, peak_160x20=pa, peak_320x40=pb,
                       peakFrac_160x20=pa/A['NE'], peakFrac_320x40=pb/B['NE'])
            if wa is None or wb is None:
                rec.update(overlap=None, admissible=False)
            else:
                lo, hi = max(wa[0], wb[0]), min(wa[1], wb[1])
                rec.update(window_160x20=list(wa), window_320x40=list(wb),
                           overlap=[lo, hi] if lo < hi else None, admissible=bool(lo < hi))
            selfn.append(rec)
    Mx['self_normalised_active_set'] = dict(
        definition='s(k) = activeCount(k) / max_j activeCount(j)',
        finding=('the peak is reached at iteration 1 where essentially every element moves, '
                 'so the peak is ~NE and this variant degenerates to the area fraction'),
        results=selfn)

    # ---- 7. exact vs Markov cap, tightness ---------------------------------
    tight = []
    for R, tag in [(A, '160x20'), (B, '320x40')]:
        for i in range(R['n']):
            m = R['move'][i]; t = 1e-2/m
            ex = R['nActive']['1e-2'][i]/R['NE']
            cap = min(1.0, R['rmsU'][i]**2/(t*t))
            if ex > 0: tight.append(cap/ex)
    tight.sort()
    Mx['markov_cap_tightness'] = dict(
        statement='frac(u>=t) <= rmsU^2/t^2 is exact; this is how loose it is in practice',
        comparedAgainst='exact frac(|drho|>1e-2) from the move_stop runs',
        n=len(tight), median=tight[len(tight)//2], p10=tight[len(tight)//10],
        p90=tight[9*len(tight)//10], min=tight[0], max=tight[-1])


    # ---- 8. mesh collapse: an independent estimate of alpha ---------------
    cAm, cBm = monotone(cA), monotone(cB)
    grid = [0.50 + 0.499*i/199 for i in range(200)]
    coll = []
    for t in THRS + ['Neff']:
        vA = A['nActive'][t] if t != 'Neff' else A['Neff']
        vB = B['nActive'][t] if t != 'Neff' else B['Neff']
        scan = []
        for i in range(201):
            a = -0.2 + 1.6*i/200
            scan.append((collapse_residual(vA, vB, cAm, cBm, a, grid), a))
        r, a = min(scan)
        band = [aa for rr, aa in scan if rr <= 1.5*r]
        coll.append(dict(quantity=t, best_alpha=a, best_residual=r,
                         alpha_band_1p5x=[min(band), max(band)],
                         residual_at_alpha0_constantCount=collapse_residual(vA, vB, cAm, cBm, 0.0, grid),
                         residual_at_alpha0p5_interface=collapse_residual(vA, vB, cAm, cBm, 0.5, grid),
                         residual_at_alpha1_constantFraction=collapse_residual(vA, vB, cAm, cBm, 1.0, grid),
                         collapses=bool(r < 0.25)))
    Mx['mesh_collapse'] = dict(
        method=('match the two fixed-move runs at equal M_nd maturity, then find the alpha '
                'minimising the RMS log-discrepancy of activeCount/NE^alpha between meshes'),
        completionGrid=[grid[0], grid[-1]], nGridPoints=len(grid),
        caveat=('ONE exponent fitted to TWO meshes. The band width is fit sensitivity, NOT '
                'statistical confidence: two points cannot validate a power law. A third mesh '
                'is required and that needs an optimisation run, which this task forbids.'),
        results=coll)

    Mx['schema'] = 'olhoff_move_activity_offline/1'
    Mx['task'] = 'offline characterisation of the design-activity signal; NO optimisation run'
    with open(os.path.join(M.OUT, 'METRICS.json'), 'w') as f:
        json.dump(Mx, f, indent=2, sort_keys=False)
    print('wrote METRICS.json')
    for mesh, d in zip(['160x20', '320x40'], dsc):
        print(f"  {mesh}: production descent at {d['descentIter']}, "
              f"M_nd remaining {d['Mnd_remaining']:.3f} ({100*d['Mnd_remaining_relative']:.1f}%)")


if __name__ == '__main__':
    main()
