#!/usr/bin/env python
"""Paper Table 2 (and Fig. 4-7): the optimization strategies on one mesh.

Runs, in order: SAND exact, SAND-eps exact, SAND aggregation, SAND-eps aggregation,
NAND aggregation.  "NAND exact" is excluded (dense Jacobians, 16 GB at 100x100 in the
paper; the paper substitutes "SAND-eps exact" for it at 200x200, Fig. 14).

    python run_strategies_comparison.py                 # Table 2 at 100x100 (several hours)
    python run_strategies_comparison.py --N 200         # Table 5
    python run_strategies_comparison.py --quick         # smoke test
    python run_strategies_comparison.py --strategies sand_exact nand_aggregation
"""
import argparse
import os
import sys

import sand_env
from sand_problems import STRATEGIES, make_problem
import sand_runner as R


def main(argv=None):
    p = R.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    p.add_argument("--pnorm", type=int, default=10, help="aggregation exponent (paper: 10)")
    a = p.parse_args(argv)
    versions = sand_env.setup()
    N = R.mesh_size(a)
    params = R.params_from_args(a)
    out = a.out or R.RESULTS_ROOT + "/strategies"
    group = a.tag or f"N{N}_qp-{a.qp_solver}_it{params['maxit']}_p{a.pnorm}_init-{a.init}"
    rows = []
    for s in a.strategies:
        problem = make_problem(s, N, init=R.parse_init(a.init), pnorm=a.pnorm, maxT=a.maxT)
        summ = R.run_case(problem, params, os.path.join(out, group), s, versions=versions)
        summ["paper_name"] = STRATEGIES[s]["paper_name"]
        rows.append(summ)
    cols = [("Name", lambda r: r["paper_name"]),
            ("J", lambda r: R.fmt(r["J"])),
            ("max(T) [C]", lambda r: R.fmt(r["maxT_C"], ".3f")),
            ("h", lambda r: R.fmt(r["aggregation_h_minus_1"], ".2e")),
            ("avg [s]", lambda r: R.fmt(r["avg_s_per_iter"], ".2f")),
            ("mem [GB] (process peak so far)", lambda r: R.fmt(r["peak_rss_GB"], ".2f")),
            ("iterations", lambda r: r["iterations"])]
    R.write_table(rows, os.path.join(out, group, "table2.csv"), os.path.join(out, group, "table2.md"), cols)


if __name__ == "__main__":
    sys.exit(main())
