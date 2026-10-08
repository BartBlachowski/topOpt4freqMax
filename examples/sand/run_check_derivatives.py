#!/usr/bin/env python
"""Finite-difference verification of every problem class on a tiny mesh.

NAND aggregation uses nullspace_optimizer's own ``check_derivatives`` (random
perturbation, componentwise factor criterion).

For the two SAND problems that criterion is unusable: it divides each constraint's
finite difference by its linearization, and for the state equality K(rho)T - F both
are round-off zero after the retraction (T is re-solved).  The authors' own
``ex13_heat_SAND.py --check`` returns False for the same reason.  Here the SAND
problems are checked by the first-order Taylor remainder per block,

    e(h) = || F(retract(x, h dx)) - F(x) - h dF dx ||_inf / || h dF dx ||_inf ,

with dx tangent to the state constraint (dT = -K^-1 D_rho(KT) drho, as in the
authors' check).  e(h) must fall below 1e-3 before round-off takes over.

    python run_check_derivatives.py            # N=10
    python run_check_derivatives.py --N 20
"""
import argparse
import sys

import numpy as np
import scipy.sparse.linalg as spla

import sand_env
from sand_problems import STRATEGIES, make_problem


def tangent_dx(problem, x):
    rng = np.random.default_rng(0)
    drho = rng.random(problem.nbRho) - 0.5
    dG = problem.dG(x)
    dT = -spla.factorized(dG[:, problem.nbRho:].tocsc())(dG[:, : problem.nbRho] @ drho)
    dx = np.concatenate((drho, dT))
    return dx / np.linalg.norm(dx)


def taylor_check(problem, x, dx, hs=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5), tol=1e-3):
    J0, G0, H0 = problem.J(x), np.asarray(problem.G(x)), np.asarray(problem.H(x))
    dJ, dG, dH = np.asarray(problem.dJ(x)), problem.dG(x), problem.dH(x)
    print(f"    {'h':>8}  {'e_J':>10}  {'e_H':>10}  {'||G(x+dx)||':>12}  {'||dG dx||':>10}")
    best = {"J": np.inf, "H": np.inf}
    for h in hs:
        x1 = problem.retract(x, h * dx)
        J1, G1, H1 = problem.J(x1), np.asarray(problem.G(x1)), np.asarray(problem.H(x1))
        lin_J, lin_H = h * (dJ @ dx), h * (dH @ dx)
        eJ = abs(J1 - J0 - lin_J) / max(abs(lin_J), 1e-300)
        eH = np.max(np.abs(H1 - H0 - lin_H)) / max(np.max(np.abs(lin_H)), 1e-300)
        best["J"], best["H"] = min(best["J"], eJ), min(best["H"], eH)
        print(f"    {h:8.0e}  {eJ:10.2e}  {eH:10.2e}  {np.linalg.norm(G1):12.2e}  {np.linalg.norm(h * (dG @ dx)):10.2e}")
    ok = best["J"] < tol and best["H"] < tol
    print(f"    best relative remainders: J {best['J']:.2e}, H {best['H']:.2e}  (state residual stays ~0 by construction)")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--pnorm", type=int, default=10)
    p.add_argument("--strategies", nargs="+", default=["sand_exact", "sand_aggregation", "nand_aggregation"],
                   choices=list(STRATEGIES))
    a = p.parse_args(argv)
    sand_env.setup()
    from nullspace_optimizer.utils import check_derivatives

    ok_all = True
    for s in a.strategies:
        problem = make_problem(s, a.N, pnorm=a.pnorm)
        x = problem.x0()
        print(f"\n=== {STRATEGIES[s]['paper_name']}  N={a.N}  len(x)={len(x)}  #G={len(problem.G(x))}  #H={len(problem.H(x))}")
        if hasattr(problem, "nbT"):
            # perturb the start so that no temperature bound is exactly active
            x = problem.retract(x, 0.05 * tangent_dx(problem, x))
            ok = taylor_check(problem, x, tangent_dx(problem, x))
        else:
            ok = check_derivatives(problem, plot=False, verbose=False)
        print(f"    -> {'PASS' if ok else 'FAIL'}")
        ok_all &= bool(ok)
    print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
