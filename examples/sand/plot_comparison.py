#!/usr/bin/env python
"""Side-by-side figures for a group of finished cases, in the paper's layout.

Given a directory containing case sub-directories (each with summary.json,
history.csv, rho.npy, T.npy), produces next to them:

    fig_designs.png    densities (top) and temperatures (bottom), one column per
                       case  -- Fig. 4 / 6 / 9 / 10 / 14 style
    fig_histories.png  cost (left) and constraint (right) histories overlaid
                       -- Fig. 5 / 7 / 8 / 11 / 12 / 16 style

    python plot_comparison.py results/strategies/smoke
    python plot_comparison.py results/qp_solvers/<group>/sand_exact --cases osqp qpalm piqp
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

import sand_env


def load_cases(group_dir, names=None):
    cases = []
    paths = [os.path.join(group_dir, n) for n in names] if names else sorted(glob.glob(os.path.join(group_dir, "*")))
    for p in paths:
        sj = os.path.join(p, "summary.json")
        if not os.path.isfile(sj):
            continue
        s = json.load(open(sj))
        rows = list(csv.DictReader(open(os.path.join(p, "history.csv"))))
        hist = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
        cases.append(dict(name=os.path.basename(p), summary=s, hist=hist,
                          rho=np.load(os.path.join(p, "rho.npy")), T=np.load(os.path.join(p, "T.npy"))))
    if not cases:
        sys.exit(f"no finished cases under {group_dir}")
    return cases


def mesh_triangulation(N):
    """Same structured mesh the cases used: FreeFEM square(N, N, flags=1), rebuilt locally."""
    import matplotlib.tri as mtri
    from ex13_heat_SAND import init_mesh
    Th = init_mesh(N)
    return mtri.Triangulation(Th.vertices[:, 0], Th.vertices[:, 1], np.asarray(Th.triangles[:, :3]) - 1)


def label(c):
    s = c["summary"]
    return f"{c['name']}\nJ={s['J']:.4f}, max T={s['maxT_C']:.1f} °C"


def fig_designs(cases, path):
    import matplotlib.pyplot as plt
    tris = {}
    n = len(cases)
    fig, ax = plt.subplots(2, n, figsize=(3.6 * n, 7), squeeze=False)
    Tmax = cases[0]["summary"]["problem"]["maxT"]
    vmax = max(np.max(c["T"]) for c in cases)
    for j, c in enumerate(cases):
        N = c["summary"]["problem"]["N"]
        tri = tris.setdefault(N, mesh_triangulation(N))
        ax[0, j].tripcolor(tri, c["rho"], shading="gouraud", cmap="gray_r", vmin=0, vmax=1)
        im = ax[1, j].tricontourf(tri, c["T"], levels=np.linspace(0, vmax, 31), cmap="turbo")
        ax[1, j].tricontour(tri, c["T"], levels=[Tmax], colors="w", linewidths=0.8)
        ax[0, j].set_title(label(c), fontsize=9)
        for a in ax[:, j]:
            a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax[1, :].tolist(), fraction=0.02, pad=0.02, label=f"T [°C]  (white line: Tmax = {Tmax:g})")
    ax[0, 0].set_ylabel(r"$\rho$", fontsize=12); ax[1, 0].set_ylabel("T", fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_histories(cases, path):
    import matplotlib.pyplot as plt
    Tmax = cases[0]["summary"]["problem"]["maxT"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for c in cases:
        h = c["hist"]
        ax[0].plot(h["it"], h["J"], label=c["name"])
        agg = h["aggregation_h_minus_1"]
        if np.isfinite(agg).any():       # aggregation strategies: plot what the optimizer constrains
            ax[1].plot(h["it"], agg * Tmax, label=f"{c['name']} (h−1)·Tmax")
            ax[1].plot(h["it"], h["maxT_minus_Tmax_C"], ls=":", color=ax[1].lines[-1].get_color(),
                       label=f"{c['name']} max(T−Tmax)")
        else:
            ax[1].plot(h["it"], h["maxT_minus_Tmax_C"], label=f"{c['name']} max(T−Tmax)")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("J (volume fraction)"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].set_yscale("symlog", linthresh=1e-2); ax[1].axhline(0, color="k", lw=.5)
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("constraint [°C]"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("group_dir")
    p.add_argument("--cases", nargs="+", default=None, help="sub-directory names, in plotting order")
    a = p.parse_args(argv)
    sand_env.setup()
    cases = load_cases(a.group_dir, a.cases)
    fig_designs(cases, os.path.join(a.group_dir, "fig_designs.png"))
    fig_histories(cases, os.path.join(a.group_dir, "fig_histories.png"))
    print("wrote", os.path.join(a.group_dir, "fig_designs.png"), "and fig_histories.png for", [c["name"] for c in cases])


if __name__ == "__main__":
    sys.exit(main())
