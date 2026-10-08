#!/usr/bin/env python
"""Paper Table 4, Fig. 10-13: influence of the QP solver on "SAND exact" (left half)
and on "NAND aggregation" (right half).

Open-source solvers installed here: OSQP, QPALM, PIQP.  MOSEK, Gurobi and CPLEX are
included automatically when their Python bindings import (licenses needed).

    python run_qp_solver_comparison.py                    # both halves, 100x100, 500 it
    python run_qp_solver_comparison.py --problems sand_exact --solvers piqp qpalm
    python run_qp_solver_comparison.py --quick
"""
import argparse
import os
import sys

import sand_env
from sand_problems import STRATEGIES, make_problem
import sand_runner as R


def main(argv=None):
    p = R.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--solvers", nargs="+", default=None, help="default: every importable one of Table 4")
    p.add_argument("--problems", nargs="+", default=["sand_exact", "nand_aggregation"],
                   choices=list(STRATEGIES))
    p.add_argument("--pnorm", type=int, default=10)
    a = p.parse_args(argv)
    versions = sand_env.setup()
    solvers = a.solvers or sand_env.available_qp_solvers()
    print("QP solvers:", solvers)
    N = R.mesh_size(a)
    out = a.out or R.RESULTS_ROOT + "/qp_solvers"
    group = a.tag or f"N{N}_it{R.params_from_args(a)['maxit']}_init-{a.init}"
    for s in a.problems:
        rows = []
        for qp in solvers:
            a.qp_solver = qp
            params = R.params_from_args(a)
            problem = make_problem(s, N, init=R.parse_init(a.init), pnorm=a.pnorm, maxT=a.maxT)
            summ = R.run_case(problem, params, os.path.join(out, group, s), qp, versions=versions)
            summ["qp"] = qp
            rows.append(summ)
        cols = [("QP solver", lambda r: r["qp"]),
                ("J", lambda r: R.fmt(r["J"])),
                ("max(T) [C]", lambda r: R.fmt(r["maxT_C"], ".3f")),
                ("h", lambda r: R.fmt(r["aggregation_h_minus_1"], ".2e")),
                ("avg [s]", lambda r: R.fmt(r["avg_s_per_iter"], ".2f")),
                ("iterations", lambda r: r["iterations"])]
        R.write_table(rows, os.path.join(out, group, s, "table4.csv"), os.path.join(out, group, s, "table4.md"), cols)


if __name__ == "__main__":
    sys.exit(main())
