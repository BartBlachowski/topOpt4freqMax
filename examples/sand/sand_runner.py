"""Common driver for the SAND heat-sink runners: CLI, solve, save, plot.

Every runner produces, under ``examples/sand/results/<runner>/<tag>/``:

    summary.json   final cost J, max T [C], aggregation value h, avg s/iter, memory,
                   problem description, optimizer params, component versions
    history.csv    per-iteration J, max(T)-Tmax, ||G||, aggregation h-1, wall time
    rho.npy, T.npy final density and physical temperature (nodal, P1)
    design.png     density + temperature (the paper's Fig. 4/6/9/10/14 panels)
    history.png    cost and constraint histories (Fig. 5/7/8/11/12/15/16)

Paper parameters (Sect. 4.2) are the defaults: dt = 0.05, alphaJ = alphaC = 1,
itnormalisation = 50, maxit = 500, K = 0.01, QPALM, Tmax = 300 C, 100 x 100 mesh.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time

import numpy as np

import sand_env

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(HERE, "results")

PAPER_PARAMS = dict(
    dt=0.05, alphaJ=1.0, alphaC=1.0, itnormalisation=50, maxit=500, K=0.01,
    qp_solver="qpalm", qp_solver_options=dict(max_iter=1000), tol_qp=1e-8,
    method_xiC="qp", qp_saturate_slack=True,
    save_only_N_iterations=1, save_only_Q_constraints=5,
)
QUICK = dict(N=30, maxit=20)   # smoke-test preset, ~10 s


def add_common_args(p: argparse.ArgumentParser, default_maxit=500):
    p.add_argument("--N", type=int, default=100, help="edges per side (paper: 100 or 200)")
    p.add_argument("--maxit", type=int, default=default_maxit, help="paper: 500")
    p.add_argument("--qp-solver", default="qpalm", help="osqp|qpalm|piqp|mosek|gurobi|cplex")
    p.add_argument("--init", default="script",
                   help="'script' = authors' rho0=0.4; 'paper' = uniform rho0 with max T = Tmax; or a float")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--tol-qp", type=float, default=1e-8)
    p.add_argument("--maxT", type=float, default=300.0)
    p.add_argument("--quick", action="store_true", help=f"smoke preset {QUICK}")
    p.add_argument("--out", default=None, help="results directory (default results/<runner>)")
    p.add_argument("--tag", default=None, help="sub-directory name (default built from the settings)")
    p.add_argument("--debug", type=int, default=0, help="nullspace_optimizer verbosity")
    return p


def params_from_args(args, **override):
    params = dict(PAPER_PARAMS)
    params.update(maxit=args.maxit, qp_solver=args.qp_solver, dt=args.dt, tol_qp=args.tol_qp, debug=args.debug)
    params.update(override)
    if args.quick:
        params["maxit"] = min(params["maxit"], QUICK["maxit"])
    return params


def mesh_size(args):
    return QUICK["N"] if args.quick else args.N


def parse_init(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def peak_memory_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 ** 3) if sys.platform == "darwin" else ru / (1024 ** 2)


# --------------------------------------------------------------------------- #
def run_case(problem, params, outdir, tag, versions=None, extra=None, make_plots=True):
    """Solve ``problem`` with nlspace_solve and persist everything.  Returns the summary."""
    from nullspace_optimizer import nlspace_solve

    case_dir = os.path.join(outdir, tag)
    os.makedirs(case_dir, exist_ok=True)
    desc = problem.describe()
    print(f"\n=== {desc['strategy']}  N={desc['N']}  n={desc['n_vertices']}  qp={params['qp_solver']}  "
          f"maxit={params['maxit']}  alpha={desc.get('alpha')}  -> {case_dir}", flush=True)

    x0 = problem.x0()
    T0 = problem.physical_T(x0)
    print(f"    initial uniform rho0 = {desc['vfrac0']:.5f}, initial max T = {np.max(T0):.3f} C "
          f"(Tmax = {desc['maxT']})", flush=True)

    t_start = time.time()
    results = nlspace_solve(problem, dict(params))
    wall = time.time() - t_start

    x = results["x"][-1]
    its = np.asarray(results["it"])
    J_hist = np.asarray(results["J"], dtype=float)
    tviol = np.asarray(results.get("maxT_minus_Tmax", []), dtype=float)
    resid = np.asarray(results.get("state_residual", np.zeros(len(its))), dtype=float)
    agg = results.get("aggregation", None)
    agg = np.asarray(agg, dtype=float) if agg is not None else np.full(len(its), np.nan)
    times = np.asarray(results.get("time", []), dtype=float)
    per_iter = times[1:] if len(times) > 1 else times

    def _pad(a, n):
        a = np.asarray(a, dtype=float)
        return np.concatenate((a, np.full(max(0, n - len(a)), np.nan)))[:n]

    n_rows = len(its)
    with open(os.path.join(case_dir, "history.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["it", "J", "maxT_minus_Tmax_C", "state_residual", "aggregation_h_minus_1", "iter_time_s"])
        for row in zip(its, _pad(J_hist, n_rows), _pad(tviol, n_rows), _pad(resid, n_rows),
                       _pad(agg, n_rows), _pad(times, n_rows)):
            w.writerow([int(row[0])] + [f"{v:.10g}" for v in row[1:]])

    rho, T = problem.split(x)
    np.save(os.path.join(case_dir, "rho.npy"), rho)
    np.save(os.path.join(case_dir, "T.npy"), T / problem.rescale)

    summary = dict(
        tag=tag, problem=desc, params={k: (v if not callable(v) else str(v)) for k, v in params.items()},
        iterations=int(its[-1]) if len(its) else 0,
        J=float(problem.J(x)), maxT_C=float(np.max(problem.physical_T(x))),
        maxT_minus_Tmax_C=float(problem.max_T_violation(x)),
        aggregation_h_minus_1=problem.aggregation_value(x),
        state_residual=float(problem.state_residual(x)),
        avg_s_per_iter=float(np.mean(per_iter)) if len(per_iter) else float("nan"),
        wall_s=wall, peak_rss_GB=peak_memory_gb(),
        initial_maxT_C=float(np.max(T0)),
        versions=versions or {}, extra=extra or {},
    )
    with open(os.path.join(case_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"    done: it={summary['iterations']}  J={summary['J']:.5f}  max T={summary['maxT_C']:.3f} C  "
          f"h-1={summary['aggregation_h_minus_1']}  avg {summary['avg_s_per_iter']:.2f} s/iter  "
          f"wall {wall/60:.1f} min  peak RSS {summary['peak_rss_GB']:.2f} GB", flush=True)

    if make_plots:
        try:
            plot_design(problem, x, os.path.join(case_dir, "design.png"), title=f"{desc['strategy']} ({tag})")
            plot_history(summary, os.path.join(case_dir, "history.csv"), os.path.join(case_dir, "history.png"))
        except Exception as exc:  # noqa: BLE001 - plots must never lose a finished run
            print(f"    [plot skipped: {type(exc).__name__}: {exc}]")
    return summary


# --------------------------------------------------------------------------- #
def plot_design(problem, x, path, title=""):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    rho, _ = problem.split(x)
    T = problem.physical_T(x)
    Th = problem.Th
    tri = mtri.Triangulation(Th.vertices[:, 0], Th.vertices[:, 1], np.asarray(Th.triangles[:, :3]) - 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    im0 = ax[0].tripcolor(tri, rho, shading="gouraud", cmap="gray_r", vmin=0, vmax=1)
    fig.colorbar(im0, ax=ax[0], fraction=0.046)
    ax[0].set_title(r"$\rho$"); ax[0].set_aspect("equal")
    im1 = ax[1].tricontourf(tri, T, levels=30, cmap="turbo")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, label="T [°C]")
    ax[1].set_title(f"T   (max {np.max(T):.1f} °C, Tmax {problem.maxT:g})"); ax[1].set_aspect("equal")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(summary, csv_path, png_path):
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(csv_path)))
    it = np.array([int(r["it"]) for r in rows])
    J = np.array([float(r["J"]) for r in rows])
    tv = np.array([float(r["maxT_minus_Tmax_C"]) for r in rows])
    agg = np.array([float(r["aggregation_h_minus_1"]) for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(it, J); ax[0].set_xlabel("iteration"); ax[0].set_ylabel("J (volume fraction)"); ax[0].grid(alpha=.3)
    ax[1].plot(it, tv, label=r"$\max(T-T_{max})$ [°C]")
    if np.isfinite(agg).any():
        ax[1].plot(it, agg * summary["problem"]["maxT"], label=r"aggregation $(h-1)\,T_{max}$ [°C]")
    ax[1].set_yscale("symlog", linthresh=1e-3); ax[1].axhline(0, color="k", lw=.5)
    ax[1].set_xlabel("iteration"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(f"{summary['problem']['strategy']}  N={summary['problem']['N']}  qp={summary['params']['qp_solver']}")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def write_table(rows, path_csv, path_md, columns):
    """Persist a list of summaries as CSV + Markdown in the paper's table layout."""
    with open(path_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([c[0] for c in columns])
        for r in rows:
            w.writerow([c[1](r) for c in columns])
    with open(path_md, "w") as fh:
        fh.write("| " + " | ".join(c[0] for c in columns) + " |\n")
        fh.write("|" + "---|" * len(columns) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(str(c[1](r)) for c in columns) + " |\n")
    print("\n" + open(path_md).read())


def fmt(v, spec=".5f"):
    if v is None:
        return "NA"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)
