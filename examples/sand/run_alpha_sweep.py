#!/usr/bin/env python
"""Paper Fig. 8/9 (and Sect. 4.2.2): "SAND exact" for different metric weights alpha
of Eq. (4.10).  The paper does not list the swept values; the defaults below bracket
the balanced value (Eq. 4.11, about 1.8e-3 at 100x100 and 4.5e-4 at 200x200) from the
degenerate 1e-8 up to 1, where the temperature field is nearly frozen (Fig. 9).

    python run_alpha_sweep.py
    python run_alpha_sweep.py --alphas 1e-8 balanced 1e-2 1
    python run_alpha_sweep.py --quick
"""
import argparse
import os
import sys

import sand_env
from sand_problems import make_problem
import sand_runner as R


def main(argv=None):
    p = R.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--alphas", nargs="+", default=["1e-8", "1e-4", "balanced", "1e-2", "1e-1", "1"])
    a = p.parse_args(argv)
    versions = sand_env.setup()
    N = R.mesh_size(a)
    params = R.params_from_args(a)
    out = a.out or R.RESULTS_ROOT + "/alpha_sweep"
    group = a.tag or f"N{N}_qp-{a.qp_solver}_it{params['maxit']}_init-{a.init}"
    rows = []
    for al in a.alphas:
        alpha = R.parse_init(al)
        problem = make_problem("sand_exact", N, init=R.parse_init(a.init), alpha=alpha, maxT=a.maxT)
        summ = R.run_case(problem, params, os.path.join(out, group), f"alpha-{al}", versions=versions)
        summ["alpha_label"] = al
        rows.append(summ)
    cols = [("alpha", lambda r: r["alpha_label"]),
            ("alpha value", lambda r: R.fmt(r["problem"]["alpha"], ".3e")),
            ("J", lambda r: R.fmt(r["J"])),
            ("max(T) [C]", lambda r: R.fmt(r["maxT_C"], ".3f")),
            ("avg [s]", lambda r: R.fmt(r["avg_s_per_iter"], ".2f")),
            ("iterations", lambda r: r["iterations"])]
    R.write_table(rows, os.path.join(out, group, "alpha_sweep.csv"), os.path.join(out, group, "alpha_sweep.md"), cols)


if __name__ == "__main__":
    sys.exit(main())
