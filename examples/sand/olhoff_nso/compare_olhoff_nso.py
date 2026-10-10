#!/usr/bin/env python
"""Merge the MATLAB production Olhoff reference with the Python NSO / MMA arms.

    python compare_olhoff_nso.py                      # all results/*.json + olhoff_ref_160x20.mat
    python compare_olhoff_nso.py --cases nso_160x20_dt0.02_itn50_euclid mma_160x20_move0.1

Writes results/comparison.md, results/comparison.csv, results/fig_histories.png,
results/fig_designs.png.
"""
import argparse
import csv
import glob
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load_ref(path):
    r = sio.loadmat(path, squeeze_me=True)
    om = np.atleast_2d(r["hist_omega"])
    if om.shape[0] < om.shape[1]:
        om = om.T
    rho = np.asarray(r["rho"], dtype=float).ravel()
    nO = int(r["nOuter"])
    tOuter = np.asarray(r["hist_tOuter"], dtype=float).ravel()
    return dict(tag="olhoff_matlab_production", label="Du-Olhoff production (MATLAB, nested MMA)",
                omega1=om[:, 0], omega2=om[:, 1], drho_l2=np.asarray(r["hist_dxNorm2"], dtype=float).ravel(),
                rho=rho, iterations=nO, omega_final=np.asarray(r["omega"], dtype=float).ravel(),
                Mnd=float(np.mean(4 * rho * (1 - rho))), vol=float(rho.mean()),
                wall_s=float(r["wall_s"]), s_per_iter=float(tOuter.sum() / max(nO, 1)),
                eig_time_s=float(np.sum(r["hist_tEig"])), inner_time_s=float(np.sum(r["hist_tInner"])),
                inner_iters=int(np.sum(r["hist_nInner"])), status=str(r["status"]),
                eps_l2=0.05 * np.sqrt(rho.size / 3200))


def load_case(path):
    d = json.load(open(path))
    s, h = d["summary"], d["history"]
    rho = np.load(path.replace(".json", "_rho.npy"))
    return dict(tag=s["tag"], label=s["tag"], omega1=np.asarray(h["omega1"]), omega2=np.asarray(h["omega2"]),
                drho_l2=np.asarray(h["drho_l2"]), rho=rho, iterations=s["iterations"],
                omega_final=np.asarray(s["omega"]), Mnd=s["Mnd"], vol=s["vol"], wall_s=s["wall_s"],
                s_per_iter=s["s_per_iter"], eig_time_s=s["eig_time_s"], inner_time_s=0.0, inner_iters=0,
                status="CAP_HIT" if s["iterations"] >= s["args"]["maxit"] else "STOPPED",
                eps_l2=s["olhoff_eps_l2"], native_stop=s["first_iter_below_olhoff_eps"],
                omega1_native_stop=s["omega1_at_native_stop"], nelx=s["args"]["nelx"], nely=s["args"]["nely"])


def period2_index(w, tail=40):
    """Mean |w_k - w_{k-1}| over |w_k - w_{k-2}| on the tail: ~1 = smooth, >>1 = period-2 flip-flop."""
    w = np.asarray(w, dtype=float)[-tail:]
    if len(w) < 6:
        return np.nan
    d1 = np.mean(np.abs(np.diff(w))); d2 = np.mean(np.abs(w[2:] - w[:-2]))
    return float(d1 / max(d2, 1e-12))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--ref", default=os.path.join(RES, "olhoff_ref_160x20.mat"))
    ap.add_argument("--nelx", type=int, default=160); ap.add_argument("--nely", type=int, default=20)
    ap.add_argument("--suffix", default="", help="appended to comparison/fig file names")
    a = ap.parse_args(argv)
    cases = []
    if os.path.isfile(a.ref):
        cases.append(load_ref(a.ref))
    paths = [os.path.join(RES, c + ".json") for c in a.cases] if a.cases else sorted(glob.glob(os.path.join(RES, f"*_{a.nelx}x{a.nely}_*.json")))
    for p in paths:
        if os.path.basename(p).startswith(("sanity", "pred_")):
            continue
        try:
            cases.append(load_case(p))
        except Exception as exc:  # noqa: BLE001
            print("skip", p, exc)
    if not cases:
        sys.exit("nothing to compare")

    # ---- table ---------------------------------------------------------------
    rows = []
    for c in cases:
        w = c["omega_final"]
        rows.append(dict(arm=c["tag"], iterations=c["iterations"], status=c["status"],
                         omega1=f"{w[0]:.3f}", omega2=f"{w[1]:.3f}", gap12_pct=f"{100*(w[1]-w[0])/w[0]:.2f}",
                         Mnd_pct=f"{100*c['Mnd']:.2f}", vol=f"{c['vol']:.4f}",
                         wall_s=f"{c['wall_s']:.0f}", s_per_iter=f"{c['s_per_iter']:.3f}",
                         eig_share_pct=f"{100*c['eig_time_s']/max(c['wall_s'],1e-9):.0f}",
                         period2_tail=f"{period2_index(c['omega1']):.2f}",
                         omega1_last40_minmax=f"{np.min(c['omega1'][-40:]):.2f}..{np.max(c['omega1'][-40:]):.2f}",
                         first_it_drho_below_eps=c.get("native_stop", "-"),
                         omega1_at_that_it=(f"{c['omega1_native_stop']:.3f}" if c.get("omega1_native_stop") else "-")))
    keys = list(rows[0].keys())
    with open(os.path.join(RES, "comparison" + a.suffix + ".csv"), "w", newline="") as fh:
        w_ = csv.DictWriter(fh, keys); w_.writeheader(); w_.writerows(rows)
    with open(os.path.join(RES, "compariso" + a.suffix + "n.md"), "w") as fh:
        fh.write("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(str(r[k]) for k in keys) + " |\n")
    print(open(os.path.join(RES, "compariso" + a.suffix + "n.md")).read())

    # ---- figures ---------------------------------------------------------------
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for c in cases:
        it = np.arange(len(c["omega1"]))
        ax[0].plot(it, c["omega1"], label=c["label"], lw=1.2)
        ax[1].plot(it, 100 * (c["omega2"] - c["omega1"]) / c["omega1"], lw=1.0)
        ax[2].semilogy(it, c["drho_l2"], lw=1.0)
    ax[0].set_ylabel(r"$\omega_1$"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].set_ylabel(r"$(\omega_2-\omega_1)/\omega_1$ [%]"); ax[1].set_ylim(-1, 30); ax[1].grid(alpha=.3)
    ax[2].axhline(cases[0]["eps_l2"], color="k", ls="--", lw=.8, label="Olhoff stop eps")
    ax[2].set_ylabel(r"$\|\Delta\rho\|_2$"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
    for x_ in ax: x_.set_xlabel("iteration")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_histories" + a.suffix + ".png"), dpi=150); plt.close(fig)

    n = len(cases)
    fig, ax = plt.subplots(n, 1, figsize=(12, 1.9 * n), squeeze=False)
    for i, c in enumerate(cases):
        img = c["rho"].reshape((a.nely, a.nelx), order="F")
        ax[i, 0].imshow(img, cmap="gray_r", vmin=0, vmax=1, aspect="equal", interpolation="nearest")
        ax[i, 0].set_title(f"{c['label']}: ω1={c['omega_final'][0]:.2f}, ω2={c['omega_final'][1]:.2f}, "
                           f"M_nd={100*c['Mnd']:.1f}%, it={c['iterations']}", fontsize=9)
        ax[i, 0].set_xticks([]); ax[i, 0].set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_designs" + a.suffix + ".png"), dpi=150); plt.close(fig)
    print("figures:", os.path.join(RES, "fig_histories" + a.suffix + ".png"), os.path.join(RES, "fig_designs" + a.suffix + ".png"))


if __name__ == "__main__":
    sys.exit(main())
