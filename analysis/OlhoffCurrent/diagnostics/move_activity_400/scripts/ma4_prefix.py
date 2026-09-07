#!/usr/bin/env python3
"""ma4_prefix -- the counterfactual validity gate (brief sec. C4, prereg sec. 4).

F400 may be read as production's CONTINUATION only if the two arms are bitwise
identical up to the iteration immediately before P400's first move descent.
That property is what made the 160x20/320x40 fixed-move evidence strong (both
compared at exactly 0.000e+00), and it is asserted here rather than assumed.

Comparison is on raw float64 bit patterns -- not a tolerance.  A single differing
ULP anywhere in the 20000-element field, at any iteration of the prefix, fails
the gate.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ma4_traj

OUT = os.path.join(ma4_traj.REPO, 'analysis/OlhoffCurrent/diagnostics/move_activity_400')
FIELDS = ['omega', 'vol', 'dxOuter', 'dxNorm2', 'move', 'beta', 'nInner', 'gap12']


def bitwise_equal(a, b):
    """True iff every float64 has an identical bit pattern (NaN==NaN allowed)."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    return a.view(np.uint64).tobytes() == b.view(np.uint64).tobytes()


def main():
    P = ma4_traj.Traj('P'); F = ma4_traj.Traj('F')
    mv = P.move
    desc = None
    for k in range(1, len(mv)):
        if mv[k] < mv[k-1]:
            desc = k + 1          # 1-based iteration at which the move DROPPED
            break
    R = {'schema': 'olhoff_move_activity_400_prefix/1',
         'P400_nOuter': int(P.nOuter), 'F400_nOuter': int(F.nOuter),
         'P400_firstDescentIter': desc}

    if desc is None:
        R['gate'] = 'NO_DESCENT'
        R['detail'] = 'P400 never descended from move=0.04 within its cap'
        json.dump(R, open(os.path.join(OUT, 'PREFIX_GATE.json'), 'w'), indent=2)
        print('NO DESCENT in P400'); return R

    n = desc - 1                  # prefix = iterations 1..n, all at move 0.04
    R['prefixIterations'] = int(n)

    # ---- full density field, every element, every prefix iteration -------
    rhoP = np.array(P.f['RHO'][:n, :])
    rhoF = np.array(F.f['RHO'][:n, :])
    drP  = np.array(P.f['DRHO'][:n, :])
    drF  = np.array(F.f['DRHO'][:n, :])
    R['rho_bitwise_identical']  = bool(bitwise_equal(rhoP, rhoF))
    R['drho_bitwise_identical'] = bool(bitwise_equal(drP, drF))
    R['rho_maxAbsDiff']  = float(np.max(np.abs(rhoP - rhoF)))
    R['drho_maxAbsDiff'] = float(np.max(np.abs(drP - drF)))
    R['rho_elementsCompared'] = int(rhoP.size)

    # ---- scalar histories -------------------------------------------------
    hf = {}
    for name in FIELDS:
        try:
            a = np.array(P.f['hist'][name]); b = np.array(F.f['hist'][name])
        except KeyError:
            hf[name] = 'ABSENT'; continue
        a = a.reshape(a.shape[0], -1)[:n]; b = b.reshape(b.shape[0], -1)[:n]
        hf[name] = dict(bitwise=bool(bitwise_equal(a, b)),
                        maxAbsDiff=float(np.max(np.abs(a - b))))
    R['hist_fields'] = hf

    allhist = all(v['bitwise'] for v in hf.values() if isinstance(v, dict))
    ok = R['rho_bitwise_identical'] and R['drho_bitwise_identical'] and allhist
    R['gate'] = 'COUNTERFACTUAL_PREFIX_VERIFIED' if ok else 'COUNTERFACTUAL_PREFIX_MISMATCH'

    json.dump(R, open(os.path.join(OUT, 'PREFIX_GATE.json'), 'w'), indent=2)
    print(f"P400 first descent at iteration {desc}; prefix = iterations 1..{n}")
    print(f"  rho  bitwise identical: {R['rho_bitwise_identical']}  "
          f"(max |diff| = {R['rho_maxAbsDiff']:.3e} over {R['rho_elementsCompared']} values)")
    print(f"  drho bitwise identical: {R['drho_bitwise_identical']}")
    for k2, v in hf.items():
        if isinstance(v, dict):
            print(f"  hist.{k2:<9} bitwise={v['bitwise']}  max|diff|={v['maxAbsDiff']:.3e}")
    print(f"  GATE: {R['gate']}")
    P.close(); F.close()
    return R


if __name__ == '__main__':
    main()
