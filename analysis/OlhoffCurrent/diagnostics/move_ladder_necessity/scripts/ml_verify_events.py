#!/usr/bin/env python3
"""ml_verify_events -- Phases 3, 6, 7.

Independently recompute the frozen A OR B exhaustion event on the move=0.04
prefix of each candidate trajectory, and check it against (a) the controller's
own recorded predicate trace and (b) the descent the solver actually applied.
"""
import os, sys, json
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_frozen as F

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(STUDY))
CV = os.path.join(ROOT, 'diagnostics', 'two_branch_controller_validation')
EV = os.path.join(ROOT, 'evidence', 'two_branch_controller_validation')

MESH = {'C160x20': (160, 20), 'C320x40': (320, 40), 'C400x50': (400, 50)}


def load(tag):
    p = os.path.join(EV, f'{tag}_trajectory.mat')
    with h5py.File(p, 'r') as h:
        RHO = np.array(h['RHO']).T                      # v7.3 is transposed -> (NE, n)
        hist = {k: np.array(h['hist'][k]) for k in
                ('dxNorm2', 'move', 'stage', 'omega', 'beta', 'nInner', 'cumInner',
                 'vol', 'gap12', 'dxOuter', 'tOuter', 'N', 'innerConv', 'multJ', 'degen')}
        rho0 = float(np.array(h['cfg']['design']['initial']).ravel()[0])
    return RHO, hist, rho0


def main():
    out = {}
    for tag, (nx, ny) in MESH.items():
        NE = nx * ny
        RHO, hist, rho0 = load(tag)
        amp = hist['dxNorm2'].ravel()
        move = hist['move'].ravel()
        n = RHO.shape[1]

        # the move = 0.04 prefix, i.e. stage 1
        pre = int(np.argmax(move != move[0])) if np.any(move != move[0]) else n
        assert np.all(move[:pre] == 0.04), 'stage-1 prefix is not all move=0.04'

        q = F.quantities(RHO[:, :pre], amp[:pre], rho0, NE, stage_start=1)
        A, B, tol = F.branches(q, amp[:pre], NE)
        k, br, w0 = F.declare(A, B)

        # cross-check against the controller's own recorded trace
        T = np.genfromtxt(os.path.join(CV, 'runs', f'{tag}_iterations.csv'),
                          delimiter=',', names=True)
        recA = T['exA'][:pre].astype(bool)
        recB = T['exB'][:pre].astype(bool)
        rec_decl = int(np.argmax((T['exNA'] >= 20) | (T['exNB'] >= 20))) + 1

        out[tag] = dict(
            mesh=[nx, ny], NE=NE, tol=float(tol), nOuter=int(n),
            prefix_len=int(pre), first_descent_applied=int(pre + 1),
            offline_declaration=int(k), offline_branch=br, offline_window=[int(w0), int(k)],
            recorded_declaration=int(rec_decl),
            A_elementwise_match=bool(np.array_equal(A, recA)),
            B_elementwise_match=bool(np.array_equal(B, recB)),
            declaration_matches=bool(k == rec_decl),
            descent_is_declaration_plus_one=bool(pre + 1 == k + 1))
        r = out[tag]
        print(f"{tag}: NE={NE} tol={tol:.4f}")
        print(f"   offline frozen rule  -> declare at {k} (branch {br}, window {w0}-{k})")
        print(f"   controller recorded  -> declare at {rec_decl}")
        print(f"   A/B elementwise identical over the whole prefix: "
              f"A={r['A_elementwise_match']} B={r['B_elementwise_match']}")
        print(f"   solver applied first descent at iteration {pre+1} "
              f"(= declaration + 1: {r['descent_is_declaration_plus_one']})")
    json.dump(out, open(os.path.join(STUDY, 'evidence', 'event_verification.json'), 'w'), indent=1)
    print('\nwrote evidence/event_verification.json')
    return out


if __name__ == '__main__':
    main()
