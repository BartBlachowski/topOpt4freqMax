#!/usr/bin/env python3
"""Stage-1 prefix equivalence at 160x20 and 320x40.

Under the candidate, stage 1 holds move = 0.04 and changes nothing else, so its
prefix must coincide with an independent fixed-move 0.04 arm.  PREREGISTRATION
section 6 states this check for 400x50 only, because no same-version fixed-move arm was
believed to survive for the coarser meshes.  Two do
(diagnostics/move_stop/runs/fixedmove_*.mat), so the check is performed there too.

This compares element-level DENSITY trajectories, which is the strongest available
form: it needs no convention agreement and no tolerance.
"""
import json, os
import numpy as np
import h5py

REPO = '/Users/piotrek/Programming/topOpt4freqMax'
STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(REPO, 'analysis/OlhoffCurrent/diagnostics/move_stop/runs')
EV = os.path.join(REPO, 'analysis/OlhoffCurrent/evidence/two_branch_controller_validation')

# mesh -> (candidate trajectory, fixed-move arm, last stage-1 iteration)
CASES = {
    '160x20': (os.path.join(EV, 'C160x20_trajectory.mat'),
               os.path.join(MS, 'fixedmove_160x20.mat'), 102),
    '320x40': (os.path.join(EV, 'C320x40_trajectory.mat'),
               os.path.join(MS, 'fixedmove_320x40.mat'), 274),
}


def main():
    out = {}
    for mesh, (cp, fp, s1end) in CASES.items():
        r = {'candidate': os.path.relpath(cp, REPO), 'fixedmove': os.path.relpath(fp, REPO),
             'stage1LastIter': s1end}
        if not (os.path.isfile(cp) and os.path.isfile(fp)):
            r['checked'] = False
            r['detail'] = ('candidate trajectory missing' if not os.path.isfile(cp)
                           else 'fixed-move arm missing')
            print(f'  {mesh}: NOT CHECKED -- {r["detail"]}')
            out[mesh] = r
            continue
        with h5py.File(cp, 'r') as hc, h5py.File(fp, 'r') as hf:
            RC, RF = np.array(hc['RHO']), np.array(hf['RHO'])
        n = min(s1end, RC.shape[0], RF.shape[0])
        eq = bool(np.array_equal(RC[:n, :], RF[:n, :]))
        d = np.abs(RC[:n, :] - RF[:n, :])
        rowdiff = [i + 1 for i in range(n) if not np.array_equal(RC[i, :], RF[i, :])]
        lim = min(RC.shape[0], RF.shape[0])
        firstAny = next((i + 1 for i in range(lim)
                         if not np.array_equal(RC[i, :], RF[i, :])), None)
        r.update(checked=True, nCompared=int(n),
                 candNOuter=int(RC.shape[0]), fmNOuter=int(RF.shape[0]),
                 coversWholeStage1=bool(n == s1end),
                 prefixBitwiseIdentical=eq, maxAbsDiffInPrefix=float(d.max()),
                 firstDifferingIteration=firstAny,
                 firstDifferenceIsFirstDescent=bool(firstAny == s1end + 1))
        print(f'  {mesh}: compared 1..{n} (stage 1 ends {s1end}, fixed-move arm has '
              f'{RF.shape[0]})  bitwise={eq}  maxAbsDiff={d.max():.3e}  '
              f'first difference at {firstAny} (descent at {s1end + 1})')
        if rowdiff[:3]:
            print(f'      differing rows in prefix: {rowdiff[:3]}')
        out[mesh] = r
    p = os.path.join(STUDY, 'evidence', 'prefix_stage1.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('wrote', p)


if __name__ == '__main__':
    main()
