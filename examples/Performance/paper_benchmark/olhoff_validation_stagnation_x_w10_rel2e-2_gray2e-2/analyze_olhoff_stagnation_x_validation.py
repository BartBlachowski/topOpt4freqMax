"""Du-Olhoff validation of the frozen stagnation_x rule (W = 10, eps_x = 0.02, eps_M = 0.02).

Inputs (this folder):
  exports/olhoffval_export_<mesh>.mat   written on the remote by olhoffval_post.m: the continuation
      history H (400 outer, stop disabled, diagnostics on), the live run summary, the bitwise tie of
      the live stop design to the continuation, E1 every 10 outer, designs every 5 outer.
  ../campaign_remote_olhoff_validation_stagnation_x_w10_rel2e-2_gray2e-2/benchmark_results.json

Conventions: outer k analyses design rho_k (rho_1 = the initial design); r_x(k) = ||rho_k - rho_{k-1}||/
||rho_{k-1}||, R_x(k) = max r_x(k-W+1..k), R_M(k) = range M_nd(k-W..k), defined once k - W >= 2.  The run
stops at the first k with R_x < eps_x and R_M < eps_M; its final design is the design after the update of
outer k (= the design analysed at k + 1).  The continuation endpoint (after outer 400) is a continuation
reference, not an optimum.

    python3 analyze_olhoff_stagnation_x_validation.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
PB = os.path.dirname(HERE)
LIVE = os.path.join(PB, "campaign_remote_olhoff_validation_stagnation_x_w10_rel2e-2_gray2e-2")
MESHES = [(160, 20), (240, 30), (400, 50), (800, 100)]
W, EPS_X, EPS_M = 10, 0.02, 0.02
# 800x100 topology sequence; None entries are filled from the data (stop, reference)
SNAP_800 = None


def mesh_s(m):
    return f"{m[0]}x{m[1]}"


def load(m):
    E = sio.loadmat(os.path.join(HERE, "exports", f"olhoffval_export_{mesh_s(m)}.mat"), squeeze_me=True,
                    struct_as_record=False)
    H = E["H"]
    h = {f: np.atleast_1d(np.asarray(getattr(H, f), float)) for f in H._fieldnames}
    e1 = np.atleast_1d(E["E1"])
    e1k = np.array([int(r.k) for r in e1])
    e1w = np.array([float(r.omega1_E1) for r in e1])
    e1c = np.array([float(r.components) for r in e1])
    return dict(h=h, info=E["info"], live=E["live"], check=E["check"], e1k=e1k, e1w=e1w, e1c=e1c,
                kSnap=np.atleast_1d(E["kSnap"]).astype(int), Xsnap=np.asarray(E["Xsnap"], float),
                xStop=np.asarray(E["xStop"], float), xRef=np.asarray(E["xRef"], float))


def mnd(x):
    return 4 * np.mean(x * (1 - x))


def first_k(mask, k):
    idx = np.flatnonzero(mask)
    return int(k[idx[0]]) if idx.size else None


def recross(series, k, tol, after):
    """iterations > after at which series >= tol (series was below tol at 'after')."""
    if after is None:
        return []
    sel = (k > after) & np.isfinite(series) & (series >= tol)
    return list(k[sel].astype(int))


def settle_k(values, k, ref, tol_abs):
    """first k after which |values - ref| <= tol_abs for the rest of the trajectory."""
    bad = np.flatnonzero(np.abs(values - ref) > tol_abs)
    if bad.size == 0:
        return int(k[0])
    return int(k[bad[-1]]) + 1 if bad[-1] + 1 < len(k) else None


def main():
    D = {m: load(m) for m in MESHES}
    res = json.load(open(os.path.join(LIVE, "benchmark_results.json")))
    runs = {tuple(r["mesh"]): r for r in res["runs"] if not r.get("is_warmup") and r["method_key"] == "olhoff"}
    L, rows = [], []
    P = L.append
    P("# Du-Olhoff validation of the frozen stagnation_x rule (W = 10, eps_x = 0.02, eps_M = 0.02)\n")

    hist_rows = []
    for m in MESHES:
        d = D[m]
        h, live, chk, info = d["h"], d["live"], d["check"], d["info"]
        k = h["k"].astype(int)
        n = len(k)
        rx, Rx, RM, Mnd, w1 = h["rx"], h["Rx"], h["RM"], h["Mnd"], h["omega1"]
        k_stop = int(live.k_stop)
        k_rep = None if np.isnan(float(chk.k_replay)) else int(chk.k_replay)
        k_rx1 = first_k(np.isfinite(rx) & (rx < EPS_X), k)
        k_x = first_k(np.isfinite(Rx) & (Rx < EPS_X), k)
        k_M = first_k(np.isfinite(RM) & (RM < EPS_M), k)
        rx_up = recross(rx, k, EPS_X, k_rx1)
        Rx_up = recross(Rx, k, EPS_X, k_x)
        RM_up = recross(RM, k, EPS_M, k_M)
        # governing test (counterfactual): each test alone would stop at its first window pass
        if k_x is None or k_M is None:
            gov = "undetermined (a test never passes in the continuation)"
        elif k_stop == k_x and k_x > k_M:
            gov = "relative-change governed"
        elif k_stop == k_M and k_M > k_x:
            gov = "grayness governed"
        elif k_stop == k_x == k_M:
            gov = "both effectively simultaneous"
        else:
            gov = (f"interleaved: R_x first < tol at {k_x}, R_M first < tol at {k_M}, joint pass only at {k_stop}")
        # values at the stop (design after outer k_stop) and at the reference (after outer n)
        w_stop = float(live.omega1_native)
        w_ref = float(np.atleast_1d(info.omegaFinal)[0])
        e1_stop = float(live.omega1_E1)
        e1_ref = float(d["e1w"][d["e1k"] == n][0]) if np.any(d["e1k"] == n) else np.nan
        m_stop, m_ref = mnd(d["xStop"]), mnd(d["xRef"])
        relL2 = np.linalg.norm(d["xStop"] - d["xRef"]) / np.linalg.norm(d["xRef"])
        binmis = 100 * np.mean((d["xStop"] > 0.5) != (d["xRef"] > 0.5))
        comp_ref = float(d["e1c"][d["e1k"] == n][0]) if np.any(d["e1k"] == n) else np.nan
        # settling of the continuation (analysed designs) towards its endpoint
        k_set_w = settle_k(w1, k, w1[-1], 0.005 * w1[-1])
        k_set_m = settle_k(Mnd, k, Mnd[-1], 0.01)
        row = dict(mesh=mesh_s(m), k_stop_live=k_stop, k_stop_replay=k_rep,
                   stop_design_bit_exact=bool(chk.design_bit_exact), config_hash_match=bool(chk.hash_match),
                   k_first_rx_below=k_rx1, rx_back_above_after_first=len(rx_up),
                   rx_back_above_last=(rx_up[-1] if rx_up else ""),
                   k_x_first_window=k_x, Rx_back_above_after_first=len(Rx_up), Rx_back_above_last=(Rx_up[-1] if Rx_up else ""),
                   k_M_first_window=k_M, RM_back_above_after_first=len(RM_up), RM_back_above_last=(RM_up[-1] if RM_up else ""),
                   governing=gov, Rx_at_stop=float(Rx[k == k_stop][0]), RM_at_stop=float(RM[k == k_stop][0]),
                   live_final_rel_change=float(live.final_stagnation_relative_change),
                   live_final_gray_range=float(live.final_stagnation_grayness_range),
                   omega1_native_stop=w_stop, omega1_native_ref=w_ref, d_omega_native=(w_ref - w_stop) / w_ref,
                   omega1_E1_stop=e1_stop, omega1_E1_ref=e1_ref, d_omega_E1=(e1_ref - e1_stop) / e1_ref,
                   Mnd_stop=m_stop, Mnd_ref=m_ref, d_Mnd=m_stop - m_ref, relL2_stop_ref=relL2,
                   binary_mismatch_pct=binmis, components_stop=float(live.components), components_ref=comp_ref,
                   k_settle_omega_0p5pct=k_set_w, k_settle_Mnd_0p01=k_set_m, n_cont=n,
                   inner_total_live=int(live.inner_total), wall_live_s=float(live.total_wall_s),
                   status_live=str(live.status))
        rows.append(row)
        for i in range(n):
            hist_rows.append(dict(mesh=mesh_s(m), k=int(k[i]), omega1=w1[i], Mnd=Mnd[i], rx=rx[i], Rx=Rx[i], RM=RM[i],
                                  move=h["move"][i], moveMean=h["moveMean"][i], nInner=h["nInner"][i],
                                  cumInner=h["cumInner"][i], N=h["N"][i], tOuter=h["tOuter"][i]))

    with open(os.path.join(HERE, "olhoff_stagnation_x_validation_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(HERE, "olhoff_stagnation_x_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist_rows[0].keys()))
        w.writeheader()
        w.writerows(hist_rows)

    for r in rows:
        P(f"## {r['mesh']}")
        for key, val in r.items():
            if key == "mesh":
                continue
            P(f"  {key:32s} {val:.6g}" if isinstance(val, float) else f"  {key:32s} {val}")
        P("")
    P("Live run timing (benchmark): " + "; ".join(
        f"{mesh_s(m)} {runs[m]['counts']['count1']} outer / {runs[m]['counts']['count2']} inner, "
        f"{runs[m]['times']['total_wall_time_s']:.0f} s" for m in MESHES if m in runs))
    open(os.path.join(HERE, "olhoff_stagnation_x_validation_summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    plots(D, rows)


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors="#52514e")


C_X, C_M, C_SINGLE, C_STOP, C_W = "#2a78d6", "#eb6834", "#86b6ef", "#0b0b0b", "#1baf7a"


def plots(D, rows):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                         "text.color": "#0b0b0b", "axes.labelcolor": "#52514e"})
    # window quantities, all four meshes
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), facecolor="#fcfcfb")
    for ax, m, r in zip(axs.ravel(), MESHES, rows):
        h = D[m]["h"]
        k = h["k"]
        ax.semilogy(k, h["rx"], color=C_SINGLE, linewidth=1.0, label="r_x (single step)")
        ax.semilogy(k, h["Rx"], color=C_X, linewidth=2, label="R_x (window max)")
        ax.semilogy(k, h["RM"], color=C_M, linewidth=2, label="R_M (window range of M_nd)")
        ax.axhline(EPS_X, color="#52514e", linewidth=1, linestyle="--")
        ax.axvline(r["k_stop_live"], color=C_STOP, linewidth=1.2)
        ax.text(r["k_stop_live"], ax.get_ylim()[1], f" stop {r['k_stop_live']}", va="top", fontsize=9)
        ax.set_title(f"{r['mesh']}: stop metrics (tolerance 0.02 dashed)", loc="left")
        style(ax, "outer iteration k", "value")
    axs[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_olhoff_stop_metrics.png"), dpi=150)
    plt.close(fig)

    # 800x100 history
    m = (800, 100)
    d, r = D[m], rows[MESHES.index(m)]
    h, k = d["h"], d["h"]["k"]
    fig, axs = plt.subplots(5, 1, figsize=(11, 13), facecolor="#fcfcfb", sharex=True)
    axs[0].plot(k, h["omega1"], color=C_X, linewidth=2, label="native ω1 (analysed design)")
    axs[0].plot(d["e1k"], d["e1w"], "o", color=C_W, markersize=4, label="ω1 E1 (common evaluator)")
    axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title("800x100: ω1", loc="left")
    axs[1].plot(k, 100 * h["Mnd"], color=C_X, linewidth=2)
    axs[1].set_title("M_nd [%]", loc="left")
    axs[2].semilogy(k, h["rx"], color=C_X, linewidth=1.2)
    axs[2].set_title("r_x: single-step relative change", loc="left")
    axs[3].semilogy(k, h["Rx"], color=C_X, linewidth=2)
    axs[3].set_title("R_x: window max of r_x (W = 10)", loc="left")
    axs[4].semilogy(k, h["RM"], color=C_M, linewidth=2)
    axs[4].set_title("R_M: window range of M_nd (W = 10)", loc="left")
    for i, ax in enumerate(axs):
        if i >= 2:
            ax.axhline(EPS_X, color="#52514e", linewidth=1, linestyle="--")
        ax.axvline(r["k_stop_live"], color=C_STOP, linewidth=1.2)
        style(ax, "outer iteration k" if i == 4 else "", "")
    axs[0].text(r["k_stop_live"], axs[0].get_ylim()[0], f" stagnation_x stop, k = {r['k_stop_live']}", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_olhoff_800x100_history.png"), dpi=150)
    plt.close(fig)

    # 800x100 topology sequence
    ks = list(SNAP_800) if SNAP_800 else []
    ks = sorted(set([kk for kk in ks if kk in set(d["kSnap"])] + [r["k_stop_live"], int(d["kSnap"][-1])]))
    fig, axs = plt.subplots(len(ks), 1, figsize=(10, 1.55 * len(ks) + 0.4), facecolor="#fcfcfb")
    for ax, kk in zip(np.atleast_1d(axs), ks):
        j = int(np.flatnonzero(d["kSnap"] == kk)[0])
        x = d["Xsnap"][:, j].reshape((m[1], m[0]), order="F")
        ax.imshow(1 - x, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        lab = "stop" if kk == r["k_stop_live"] else ("continuation reference" if kk == ks[-1] else "")
        ax.set_title(f"after outer {kk}{' (' + lab + ')' if lab else ''}: M_nd = {100 * mnd(x):.1f} %", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_olhoff_800x100_topology_sequence.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
