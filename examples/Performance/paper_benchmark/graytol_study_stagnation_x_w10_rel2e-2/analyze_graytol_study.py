"""stagnation_x graynessTol study (W = 10, relativeChangeTol = 0.02), Proposed + Yuksel.

Campaigns of examples/Performance/performance_comparison.m for graynessTol in
{0.005, 0.010, 0.020}; 0.020 is the audited baseline of the window study (its bit-identical
repeat is campaign_..._gray2e-2_repeat).  Designs and objective histories come from exports
of benchmark_records.mat written by ../window_study_stagnation_x_rel2e-2_gray2e-2/wstudy_export.m.

Which of the two stagnation_x tests governs a stop is decided counterfactually where the
data allow it:
  * at eps_M < 0.02: the stop equals the baseline stop -> the relative-change test (the grayness
    test was already met); a later stop -> the grayness test;
  * at the baseline: the eps_x = 0.03 / eps_M = 0.02 run of the relativeChangeTol study stops at the
    same iteration -> not limited by eps_x, i.e. the grayness test; earlier -> the relative-change test.
Yuksel is classified per stage (stage 2 only when stage 1 ended at the same iteration).

    python3 analyze_graytol_study.py
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
TOLS = [0.005, 0.010, 0.020]
BASE = 0.020
TAG = {0.005: "gray5e-3", 0.010: "gray1e-2", 0.020: "gray2e-2"}
CAMPAIGNS = {t: os.path.join(PB, f"campaign_remote_stagnation_x_w10_rel2e-2_{TAG[t]}") for t in TOLS}
EXPORTS = {0.005: os.path.join(HERE, "exports", "export_gray5e-3.mat"),
           0.010: os.path.join(HERE, "exports", "export_gray1e-2.mat"),
           0.020: os.path.join(PB, "window_study_stagnation_x_rel2e-2_gray2e-2", "exports", "export_w10.mat")}
# references from the relativeChangeTol study (eps_M = 0.02)
REL03 = os.path.join(PB, "campaign_remote_stagnation_x_w10_rel3e-2_gray2e-2")
REL005 = os.path.join(PB, "campaign_remote_stagnation_x_w10_rel5e-3_gray2e-2")
REL005_EXPORT = os.path.join(PB, "reltol_study_stagnation_x_w10_gray2e-2", "exports", "export_rel5e-3.mat")

COLORS = {0.020: "#86b6ef", 0.010: "#2a78d6", 0.005: "#104281"}   # ordinal ramp, loose = light
MARKERS = {0.020: "o", 0.010: "s", 0.005: "^"}
CAP_REASONS = ("max_iter", "maxiter", "budget", "cap")
METHODS = ["proposed", "yuksel"]


def load_rows(cdir):
    res = json.load(open(os.path.join(cdir, "benchmark_results.json")))
    rows = {}
    for r in res["runs"]:
        if r.get("is_warmup"):
            continue
        st, c, t, ev = r["stopping"], r["counts"], r["times"], r.get("evaluator") or {}
        reason, s1 = str(st.get("stop_reason")), str(st.get("stage1_stop_reason"))
        rows[(r["method_key"], tuple(r["mesh"]))] = dict(
            method=r["method_key"], ne=r["n_elements"], status=r["status"],
            count1=c["count1"], count2=c["count2"], k1=c.get("stage1_iterations"), k2=c.get("stage2_iterations"),
            time1=t["time1"], time2=t["time2"], other=t["overhead_time_s"], total=t["total_wall_time_s"],
            omega1_native=r["omega1_native"], omega1_E1=r.get("omega1_common_E1"), native_flag=r.get("omega1_native_flag"),
            mnd=st.get("final_grayness"), final_rel_change=st.get("final_stagnation_relative_change"),
            final_gray_range=st.get("final_stagnation_grayness_range"), rel_tol=st.get("stagnation_relative_change_tol"),
            window=st.get("stagnation_window"), gray_tol=st.get("stagnation_grayness_tol"),
            stop_reason=reason, stage1_stop_reason=s1,
            cap_hit=any(s in (reason + s1).lower() for s in CAP_REASONS),
            timing_fail=r["accounting"]["timing_accounting_fail"] or r["accounting"]["independent_crosscheck_fail"],
            n_components=(ev.get("connectivity_binary") or {}).get("n_components"))
    return res, rows


def load_export(path):
    E = sio.loadmat(path, squeeze_me=True)["E"]
    return {(str(e["method"]), (int(e["nelx"]), int(e["nely"]))): dict(
        x=np.atleast_1d(e["x"]).astype(float), obj=np.atleast_1d(e["obj"]).astype(float)) for e in np.atleast_1d(E)}


def iters(r):
    return r["count2"] if r["method"] == "proposed" else r["k1"] + r["k2"]


def regularity(k):
    k = np.asarray(k, float)
    d = np.diff(k)
    return dict(mean=k.mean(), std=k.std(ddof=1), cv=k.std(ddof=1) / k.mean(), min=k.min(), max=k.max(),
                max_jump=np.abs(d).max(), mean_abs_jump=np.abs(d).mean(),
                sign_changes=int(np.sum(np.diff(np.sign(d[d != 0])) != 0)))


def fmt(vals, w="{:>9}"):
    return " ".join(w.format(v) for v in vals)


def mesh_s(m):
    return f"{m[0]}x{m[1]}"


def governing(t, m, mesh, data, rel03):
    """Counterfactual classification of the governing test; 'x' = relative change, 'M' = grayness."""
    b = data[BASE][(m, mesh)]
    r = data[t][(m, mesh)]
    if m == "proposed":
        stages = [("k", r["count2"], b["count2"], rel03[(m, mesh)]["count2"])]
    else:
        stages = [("k1", r["k1"], b["k1"], rel03[(m, mesh)]["k1"])]
        if r["k1"] == b["k1"]:
            stages.append(("k2", r["k2"], b["k2"], rel03[(m, mesh)]["k2"] if rel03[(m, mesh)]["k1"] == b["k1"] else None))
        else:
            stages.append(("k2", r["k2"], None, None))
    out = []
    for name, k, kb, k03 in stages:
        if t == BASE:
            g = "?" if k03 is None else ("M" if k03 == k else "x")
        else:
            g = "?" if kb is None else ("x" if k == kb else "M")
        out.append(f"{name}:{g}")
    return " ".join(out)


def main():
    results, data, exports = {}, {}, {}
    for t in TOLS:
        results[t], data[t] = load_rows(CAMPAIGNS[t])
        exports[t] = load_export(EXPORTS[t])
    _, rel03 = load_rows(REL03)
    _, rel005 = load_rows(REL005)
    rel005_exp = load_export(REL005_EXPORT)
    meshes = sorted({k[1] for k in data[BASE]}, key=lambda m: m[0] * m[1])
    L = []
    P = L.append

    P("# stagnation_x graynessTol study (W = 10, relativeChangeTol = 0.02) -- analysis output\n")
    for t in TOLS:
        cfgs = sorted({(r["window"], r["rel_tol"], r["gray_tol"]) for r in data[t].values()})
        P(f"eps_M {t:.3f}: (window, rel_tol, gray_tol) in rows = {cfgs}; label {results[t]['run_label']}")

    # ---- master table ------------------------------------------------------------------------------
    table = []
    for t in TOLS:
        for m in METHODS:
            for mesh in meshes:
                r = data[t][(m, mesh)]
                table.append(dict(
                    gray_tol=t, method=m, mesh=mesh_s(mesh), ne=r["ne"], status=r["status"], count1=r["count1"],
                    count2=r["count2"], k1=r["k1"] if m == "yuksel" else "", k2=r["k2"] if m == "yuksel" else "",
                    iters=iters(r), total_s=r["total"], time1_s=r["time1"], time2_s=r["time2"], other_s=r["other"],
                    omega1_native=r["omega1_native"], native_flag=r["native_flag"], omega1_E1=r["omega1_E1"],
                    Mnd=r["mnd"], final_rel_change=r["final_rel_change"], final_gray_range=r["final_gray_range"],
                    governing=governing(t, m, mesh, data, rel03),
                    stop_reason=r["stop_reason"], stage1_stop_reason=r["stage1_stop_reason"] if m == "yuksel" else "",
                    cap_hit=r["cap_hit"], timing_fail=r["timing_fail"], binary_components=r["n_components"]))
    with open(os.path.join(HERE, "graytol_study_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    # ---- counts --------------------------------------------------------------------------------------
    P("\n## Iteration counts (columns: graynessTol)\n")
    P("Proposed k_stop:")
    P(fmt(["mesh", "Ne"] + [f"{t:.3f}" for t in TOLS]))
    for mesh in meshes:
        P(fmt([mesh_s(mesh), mesh[0] * mesh[1]] + [data[t][("proposed", mesh)]["count2"] for t in TOLS]))
    P("\nYuksel k1 / k2 / k1+k2:")
    P(fmt(["mesh"] + [f"{t:.3f}" for t in TOLS], "{:>14}"))
    for mesh in meshes:
        P(fmt([mesh_s(mesh)] + [f"{data[t][('yuksel', mesh)]['k1']}/{data[t][('yuksel', mesh)]['k2']}/"
                                 f"{iters(data[t][('yuksel', mesh)])}" for t in TOLS], "{:>14}"))
    P("\nDelay relative to the 0.020 baseline (per mesh 160x20 ... 800x100):")
    series = {"Proposed k": ("proposed", lambda r: r["count2"]), "Yuksel k1": ("yuksel", lambda r: r["k1"]),
              "Yuksel k2": ("yuksel", lambda r: r["k2"]), "Yuksel k1+k2": ("yuksel", lambda r: r["k1"] + r["k2"])}
    for name, (m, fn) in series.items():
        for t in TOLS[:-1]:
            P(f"  {name:13s} {t:.3f}: " + " ".join(f"{fn(data[t][(m, mesh)]) - fn(data[BASE][(m, mesh)]):+d}" for mesh in meshes))
    P("\n## Regularity across the nine meshes\n")
    P(fmt(["series", "tol", "mean", "std", "CV", "min", "max", "maxJump", "meanJump", "signChg"]))
    for name, (m, fn) in series.items():
        for t in TOLS:
            g = regularity([fn(data[t][(m, mesh)]) for mesh in meshes])
            P(fmt([name, f"{t:.3f}", f"{g['mean']:.1f}", f"{g['std']:.1f}", f"{g['cv']:.3f}", int(g["min"]), int(g["max"]),
                   int(g["max_jump"]), f"{g['mean_abs_jump']:.1f}", g["sign_changes"]]))

    # ---- stop diagnostics + governing test -------------------------------------------------------------
    P("\n## At the stop: window max of relative change / grayness range / M_nd / governing test (x = rel. change, M = grayness)\n")
    for m in METHODS:
        P(f"{m}:")
        for mesh in meshes:
            P(f"  {mesh_s(mesh):8s} " + " | ".join(
                f"{t:.3f}: {data[t][(m, mesh)]['final_rel_change']:.4f} {data[t][(m, mesh)]['final_gray_range']:.4f} "
                f"{data[t][(m, mesh)]['mnd']:.4f} [{governing(t, m, mesh, data, rel03)}]" for t in TOLS))

    # ---- engineering vs baseline -------------------------------------------------------------------------
    P("\n## Final engineering result relative to the 0.020 baseline\n")
    eng = []
    for m in METHODS:
        for t in TOLS[:-1]:
            for mesh in meshes:
                a, b = data[BASE][(m, mesh)], data[t][(m, mesh)]
                xa, xb = exports[BASE][(m, mesh)]["x"], exports[t][(m, mesh)]["x"]
                eng.append(dict(method=m, gray_tol=t, mesh=mesh_s(mesh),
                                d_omega_E1_pct=100 * (b["omega1_E1"] / a["omega1_E1"] - 1),
                                d_omega_native_pct=100 * (b["omega1_native"] / a["omega1_native"] - 1),
                                d_Mnd=b["mnd"] - a["mnd"], rel_l2_design=np.linalg.norm(xb - xa) / np.linalg.norm(xa),
                                binary_mismatch_pct=100 * np.mean((xa > 0.5) != (xb > 0.5)),
                                components_base=a["n_components"], components=b["n_components"],
                                d_iters=iters(b) - iters(a), d_wall_pct=100 * (b["total"] / a["total"] - 1)))
    with open(os.path.join(HERE, "graytol_study_engineering_vs_0p02.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eng[0].keys()))
        w.writeheader()
        w.writerows(eng)
    P(fmt(["method", "tol", "mesh", "dIters", "dWall%", "dw1_E1%", "dw1_nat%", "dMnd", "relL2x", "bin%", "comp"], "{:>10}"))
    for e in eng:
        P(fmt([e["method"], f"{e['gray_tol']:.3f}", e["mesh"], f"{e['d_iters']:+d}", f"{e['d_wall_pct']:+.1f}",
               f"{e['d_omega_E1_pct']:+.3f}", f"{e['d_omega_native_pct']:+.2f}", f"{e['d_Mnd']:+.4f}",
               f"{e['rel_l2_design']:.4f}", f"{e['binary_mismatch_pct']:.2f}", f"{e['components_base']}->{e['components']}"], "{:>10}"))
    P("\nSummary per (method, tol):")
    for m in METHODS:
        for t in TOLS[:-1]:
            sel = [e for e in eng if e["method"] == m and e["gray_tol"] == t]
            P(f"  {m:8s} {t:.3f}: dw1_E1 {min(e['d_omega_E1_pct'] for e in sel):+.3f} .. {max(e['d_omega_E1_pct'] for e in sel):+.3f} %, "
              f"max|dMnd| = {max(abs(e['d_Mnd']) for e in sel):.4f}, max relL2x = {max(e['rel_l2_design'] for e in sel):.4f}, "
              f"max bin = {max(e['binary_mismatch_pct'] for e in sel):.2f} %, rows delayed = {sum(e['d_iters'] > 0 for e in sel)}/9")

    # ---- Yuksel 320x40 / 800x100 against the eps_x = 0.005 'later event' state ------------------------------
    P("\n## Yuksel key cases vs the eps_x = 0.005 (eps_M = 0.02) state of the relativeChangeTol study\n")
    for mesh in [(320, 40), (800, 100)]:
        ref = rel005[("yuksel", mesh)]
        xr = rel005_exp[("yuksel", mesh)]["x"]
        P(f"{mesh_s(mesh)}: eps_x=0.005 reference k1/k2 = {ref['k1']}/{ref['k2']}, omega1_E1 = {ref['omega1_E1']:.3f}, "
          f"M_nd = {ref['mnd']:.4f}, components = {ref['n_components']}")
        for t in TOLS:
            r = data[t][("yuksel", mesh)]
            x = exports[t][("yuksel", mesh)]["x"]
            P(f"  eps_M {t:.3f}: k1/k2 = {r['k1']}/{r['k2']} ({iters(r)}), omega1_E1 = {r['omega1_E1']:.3f} "
              f"({100 * (r['omega1_E1'] / ref['omega1_E1'] - 1):+.3f} % vs ref), M_nd = {r['mnd']:.4f}, "
              f"components = {r['n_components']}, relL2 to ref = {np.linalg.norm(x - xr) / np.linalg.norm(xr):.4f}, "
              f"bin mismatch to ref = {100 * np.mean((x > 0.5) != (xr > 0.5)):.2f} %")

    # ---- Proposed trajectory identity ------------------------------------------------------------------
    P("\n## Proposed: objective histories are prefixes of each other?")
    for t in TOLS[:-1]:
        out = []
        for mesh in meshes:
            ha, hb = exports[BASE][("proposed", mesh)]["obj"], exports[t][("proposed", mesh)]["obj"]
            n = min(len(ha), len(hb))
            out.append("IDENT" if np.array_equal(ha[:n], hb[:n]) else "DIFF")
        P(f"  {t:.3f} vs 0.020: " + " ".join(out))

    # ---- timing -------------------------------------------------------------------------------------------
    P("\n## Wall time [s]\n")
    P(fmt(["method", "mesh"] + [f"{t:.3f}" for t in TOLS]))
    for m in METHODS:
        for mesh in meshes:
            P(fmt([m, mesh_s(mesh)] + [f"{data[t][(m, mesh)]['total']:.2f}" for t in TOLS]))
    P("\nCampaign totals [s]:")
    for m in METHODS:
        tb = sum(data[BASE][(m, mesh)]["total"] for mesh in meshes)
        P(f"  {m:8s} " + "  ".join(f"{t:.3f}: {sum(data[t][(m, mesh)]['total'] for mesh in meshes):.1f} "
                                    f"({100 * (sum(data[t][(m, mesh)]['total'] for mesh in meshes) / tb - 1):+.0f} %)" for t in TOLS))
    P("\nScaling fits produced by the benchmark (secondary observation only):")
    for t in TOLS:
        P(f"  {t:.3f}: " + "; ".join(f"{d['method']} p={d['p']:.3f} R2={d['R2']:.3f}" for d in results[t]["scaling"]["methods"]))

    P("\n## Anomalies\n")
    anom = [t for t in table if t["status"] != "NATIVE_CONVERGED" or t["cap_hit"] or t["timing_fail"]
            or "stagnation_x" not in t["stop_reason"]
            or (t["method"] == "yuksel" and "stagnation_x" not in t["stage1_stop_reason"])]
    P("\n".join(f"  {a}" for a in anom) if anom else "  none: every row NATIVE_CONVERGED on stagnation_x_tolerance "
      "(Yuksel: both stages), no cap hit, no timing-accounting failure")

    open(os.path.join(HERE, "graytol_study_summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    plots(data, exports, meshes, rel005_exp)


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


def plots(data, exports, meshes, rel005_exp):
    ne = np.array([m[0] * m[1] for m in meshes])
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                         "text.color": "#0b0b0b", "axes.labelcolor": "#52514e"})

    def series(ax, fn, m):
        for t in TOLS:
            ax.plot(ne, [fn(data[t][(m, mesh)]) for mesh in meshes], color=COLORS[t], marker=MARKERS[t], markersize=6,
                    linewidth=2.4, label=f"εM = {t:.3f}" + (" (baseline)" if t == BASE else ""))

    def grid(specs, fname, logy=False):
        nrows = (len(specs) + 1) // 2
        fig, axs = plt.subplots(nrows, 2, figsize=(11, 3.9 * nrows), facecolor="#fcfcfb", squeeze=False)
        axs = axs.ravel()
        for ax, (fn, m, title, ylabel) in zip(axs, specs):
            series(ax, fn, m)
            ax.set_title(title, loc="left")
            style(ax, "Number of elements Ne", ylabel)
            if logy:
                ax.set_xscale("log")
                ax.set_yscale("log")
        axs[0].legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, fname), dpi=150)
        plt.close(fig)

    grid([(lambda r: r["count2"], "proposed", "Proposed: SIMP iterations at stop", "iterations"),
          (lambda r: r["k1"] + r["k2"], "yuksel", "Yuksel: k1 + k2", "iterations"),
          (lambda r: r["k1"], "yuksel", "Yuksel: stage-1 iterations k1", "iterations"),
          (lambda r: r["k2"], "yuksel", "Yuksel: stage-2 iterations k2", "iterations")], "fig_iterations_vs_ne.png")
    grid([(lambda r: r["omega1_E1"], "proposed", "Proposed: omega1 (common evaluator E1)", "omega1 E1 [rad/s]"),
          (lambda r: r["omega1_E1"], "yuksel", "Yuksel: omega1 (common evaluator E1)", "omega1 E1 [rad/s]"),
          (lambda r: 100 * r["mnd"], "proposed", "Proposed: final M_nd", "M_nd [%]"),
          (lambda r: 100 * r["mnd"], "yuksel", "Yuksel: final M_nd", "M_nd [%]")], "fig_engineering_vs_ne.png")
    grid([(lambda r: r["total"], "proposed", "Proposed: total wall time", "wall time [s]"),
          (lambda r: r["total"], "yuksel", "Yuksel: total wall time", "wall time [s]")], "fig_walltime_vs_ne.png", logy=True)

    for m in METHODS:
        fig, axs = plt.subplots(len(meshes), len(TOLS), figsize=(11, 0.62 * len(meshes) + 0.7), facecolor="#fcfcfb",
                                gridspec_kw=dict(hspace=0.12, wspace=0.04))
        for i, mesh in enumerate(meshes):
            for j, t in enumerate(TOLS):
                x = exports[t][(m, mesh)]["x"].reshape((mesh[1], mesh[0]), order="F")
                ax = axs[i, j]
                ax.imshow(1 - x, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"εM = {t:.3f}")
                if j == 0:
                    ax.set_ylabel(mesh_s(mesh), rotation=0, ha="right", va="center", fontsize=9)
        fig.suptitle(f"{'Proposed' if m == 'proposed' else 'Yuksel'}: final designs (εx = 0.02, W = 10)", x=0.02,
                     ha="left", fontweight="bold")
        fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.01)
        fig.savefig(os.path.join(HERE, f"fig_topologies_{m}.png"), dpi=130)
        plt.close(fig)

    # Yuksel close-ups, with the eps_x = 0.005 reference state in the last column
    crops = [((800, 100), slice(0, 160), "Yuksel 800x100: left end bay"),
             ((320, 40), slice(80, 240), "Yuksel 320x40: centre")]
    cols = [(f"εM = {t:.3f}", exports[t]) for t in TOLS] + [("ref: εx = 0.005, εM = 0.02", rel005_exp)]
    fig, axs = plt.subplots(len(crops), len(cols), figsize=(12, 4.2), facecolor="#fcfcfb", squeeze=False,
                            gridspec_kw=dict(hspace=0.35, wspace=0.05))
    for i, (mesh, sl, title) in enumerate(crops):
        for j, (lab, ex) in enumerate(cols):
            x = ex[("yuksel", mesh)]["x"].reshape((mesh[1], mesh[0]), order="F")[:, sl]
            ax = axs[i, j]
            ax.imshow(1 - x, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(lab, fontsize=9)
        fig.text(0.01, 0.97 - i * 0.49, title, fontweight="bold", fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.02)
    fig.savefig(os.path.join(HERE, "fig_yuksel_feature_closeups.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
