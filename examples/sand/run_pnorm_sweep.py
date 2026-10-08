#!/usr/bin/env python
"""Paper Table 3: "NAND aggregation" for p-norm exponents 1, 10, 20, 30, 40, 50.

    python run_pnorm_sweep.py
    python run_pnorm_sweep.py --pnorms 10 50 --quick
"""
import argparse
import os
import sys

import sand_env
from sand_problems import make_problem
import sand_runner as R


def main(argv=None):
    p = R.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--pnorms", nargs="+", type=int, default=[1, 10, 20, 30, 40, 50])
    p.add_argument("--strategy", default="nand_aggregation", choices=["nand_aggregation", "sand_aggregation", "sand_eps_aggregation"])
    a = p.parse_args(argv)
    versions = sand_env.setup()
    N = R.mesh_size(a)
    params = R.params_from_args(a)
    out = a.out or R.RESULTS_ROOT + "/pnorm_sweep"
    group = a.tag or f"{a.strategy}_N{N}_qp-{a.qp_solver}_it{params['maxit']}_init-{a.init}"
    rows = []
    for pn in a.pnorms:
        problem = make_problem(a.strategy, N, init=R.parse_init(a.init), pnorm=pn, maxT=a.maxT)
        summ = R.run_case(problem, params, os.path.join(out, group), f"p{pn}", versions=versions)
        rows.append(summ)
    cols = [("p", lambda r: r["problem"]["pnorm"]),
            ("J", lambda r: R.fmt(r["J"])),
            ("max(T) [C]", lambda r: R.fmt(r["maxT_C"], ".3f")),
            ("h", lambda r: R.fmt(r["aggregation_h_minus_1"], ".2e")),
            ("avg [s]", lambda r: R.fmt(r["avg_s_per_iter"], ".2f")),
            ("iterations", lambda r: r["iterations"])]
    R.write_table(rows, os.path.join(out, group, "table3.csv"), os.path.join(out, group, "table3.md"), cols)


if __name__ == "__main__":
    sys.exit(main())
