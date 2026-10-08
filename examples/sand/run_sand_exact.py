#!/usr/bin/env python
"""Paper Fig. 4/5 (column "SAND exact"), Table 2 row "SAND exact"; with
``--alpha eps`` the "SAND-eps exact" row (Fig. 6/7); with ``--N 200`` Table 5 /
Fig. 14-16.  One case per invocation.

    python run_sand_exact.py                       # 100x100, 500 it, QPALM (~2.5 h here)
    python run_sand_exact.py --qp-solver piqp      # ~1.2 h here
    python run_sand_exact.py --alpha eps           # SAND-eps exact
    python run_sand_exact.py --quick               # 30x30, 20 it smoke test
"""
import argparse
import sys

import sand_env
from sand_problems import make_problem
import sand_runner as R


def main(argv=None):
    p = R.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--alpha", default="balanced", help="'balanced' (Eq. 4.11), 'eps' (Eq. 4.12), or a float")
    p.add_argument("--strategy", default=None, choices=["sand_exact", "sand_eps_exact"],
                   help="alternative to --alpha")
    a = p.parse_args(argv)
    versions = sand_env.setup()
    N = R.mesh_size(a)
    alpha = {"sand_exact": "balanced", "sand_eps_exact": "eps"}.get(a.strategy, a.alpha)
    alpha = R.parse_init(alpha)
    strategy = "sand_eps_exact" if alpha == "eps" else "sand_exact"
    problem = make_problem(strategy, N, init=R.parse_init(a.init), alpha=alpha, maxT=a.maxT)
    params = R.params_from_args(a)
    out = a.out or R.RESULTS_ROOT + "/sand_exact"
    tag = a.tag or f"N{N}_alpha-{alpha}_qp-{a.qp_solver}_it{params['maxit']}_init-{a.init}"
    R.run_case(problem, params, out, tag, versions=versions)


if __name__ == "__main__":
    sys.exit(main())
