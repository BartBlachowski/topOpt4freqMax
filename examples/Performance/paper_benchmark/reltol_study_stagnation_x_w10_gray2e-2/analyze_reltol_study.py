"""stagnation_x relativeChangeTol study (W = 10, gray 2e-2), Proposed + Yuksel.

Reads the nine-mesh campaigns of examples/Performance/performance_comparison.m for
relativeChangeTol in {0.005, 0.010, 0.015, 0.020, 0.030} (benchmark_results.json plus an
export of benchmark_records.mat written by wstudy_export.m: final designs x and
objective histories) and writes tables and figures into this folder.  The baseline is
the W = 10 / 0.020 campaign of the window study; the 0.020 '_repeat' campaign is the
same configuration run again and is used as a determinism and timing-noise check.

    python3 analyze_reltol_study.py
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
TOLS = [0.005, 0.010, 0.015, 0.020, 0.030]
BASE = 0.020
TAG = {0.005: "rel5e-3", 0.010: "rel1e-2", 0.015: "rel1p5e-2", 0.020: "rel2e-2", 0.030: "rel3e-2"}
CAMPAIGNS = {t: os.path.join(PB, f"campaign_remote_stagnation_x_w10_{TAG[t]}_gray2e-2") for t in TOLS}
EXPORTS = {t: os.path.join(HERE, "exports", f"export_{TAG[t]}.mat") for t in TOLS}
# the baseline is the window study's W = 10 campaign and its export
EXPORTS[BASE] = os.path.join(PB, "window_study_stagnation_x_rel2e-2_gray2e-2", "exports", "export_w10.mat")
REPEAT = os.path.join(PB, "campaign_remote_stagnation_x_w10_rel2e-2_gray2e-2_repeat")
REPEAT_EXPORT = os.path.join(HERE, "exports", "export_rel2e-2_repeat.mat")

# ordinal one-hue ramp, loose = light, tight = dark (validated, light surface)
COLORS = {0.030: "#86b6ef", 0.020: "#5598e7", 0.015: "#2a78d6", 0.010: "#1c5cab", 0.005: "#104281"}
MARKERS = {0.030: "v", 0.020: "o", 0.015: "s", 0.010: "D", 0.005: "^"}
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
                max_jump=np.abs(d).max(), rel_max_jump=(np.abs(d) / k[:-1]).max(), mean_abs_jump=np.abs(d).mean(),
                sign_changes=int(np.sum(np.diff(np.sign(d[d != 0])) != 0)))


def fmt(vals, w="{:>9}"):
    return " ".join(w.format(v) for v in vals)


def mesh_s(m):
    return f"{m[0]}x{m[1]}"


def main():
    results, data, exports = {}, {}, {}
    for t in TOLS:
        results[t], data[t] = load_rows(CAMPAIGNS[t])
        exports[t] = load_export(EXPORTS[t])
    rep_res, rep = load_rows(REPEAT)
    rep_exp = load_export(REPEAT_EXPORT)
    meshes = sorted({k[1] for k in data[BASE]}, key=lambda m: m[0] * m[1])
    L = []
    P = L.append

    P("# stagnation_x relativeChangeTol study (W = 10, graynessTol 0.02) -- analysis output\n")
    for t in TOLS:
        cfgs = sorted({(r["window"], r["rel_tol"], r["gray_tol"]) for r in data[t].values()})
        P(f"tol {t:.3f}: (window, rel_tol, gray_tol) in rows = {cfgs}; label {results[t]['run_label']}")
    P("")

    # ---- master table -----------------------------------------------------------
    table = []
    for t in TOLS:
        for m in METHODS:
            for mesh in meshes:
                r = data[t][(m, mesh)]
                table.append(dict(
                    rel_tol=t, method=m, mesh=mesh_s(mesh), ne=r["ne"], status=r["status"], count1=r["count1"],
                    count2=r["count2"], k1=r["k1"] if m == "yuksel" else "", k2=r["k2"] if m == "yuksel" else "",
                    iters=iters(r), total_s=r["total"], time1_s=r["time1"], time2_s=r["time2"], other_s=r["other"],
                    omega1_native=r["omega1_native"], native_flag=r["native_flag"], omega1_E1=r["omega1_E1"],
                    Mnd=r["mnd"], final_rel_change=r["final_rel_change"], final_gray_range=r["final_gray_range"],
                    stop_reason=r["stop_reason"], stage1_stop_reason=r["stage1_stop_reason"] if m == "yuksel" else "",
                    cap_hit=r["cap_hit"], timing_fail=r["timing_fail"], binary_components=r["n_components"]))
    with open(os.path.join(HERE, "reltol_study_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    # ---- counts ---------------------------------------------------------------------
    P("## Iteration counts (columns: relativeChangeTol)\n")
    P("Proposed k_stop:")
    P(fmt(["mesh", "Ne"] + [f"{t:.3f}" for t in TOLS]))
    for mesh in meshes:
        P(fmt([mesh_s(mesh), mesh[0] * mesh[1]] + [data[t][("proposed", mesh)]["count2"] for t in TOLS]))
    P("\nYuksel k1 / k2 / k1+k2:")
    P(fmt(["mesh"] + [f"{t:.3f}" for t in TOLS], "{:>14}"))
    for mesh in meshes:
        P(fmt([mesh_s(mesh)] + [f"{data[t][('yuksel', mesh)]['k1']}/{data[t][('yuksel', mesh)]['k2']}/"
                                 f"{iters(data[t][('yuksel', mesh)])}" for t in TOLS], "{:>14}"))

    P("\n## Regularity across the nine meshes\n")
    P(fmt(["series", "tol", "mean", "std", "CV", "min", "max", "maxJump", "relMaxJ", "meanJump", "signChg"]))
    series = {"Proposed k": ("proposed", lambda r: r["count2"]), "Yuksel k1": ("yuksel", lambda r: r["k1"]),
              "Yuksel k2": ("yuksel", lambda r: r["k2"]), "Yuksel k1+k2": ("yuksel", lambda r: r["k1"] + r["k2"])}
    for name, (m, fn) in series.items():
        for t in TOLS:
            g = regularity([fn(data[t][(m, mesh)]) for mesh in meshes])
            P(fmt([name, f"{t:.3f}", f"{g['mean']:.1f}", f"{g['std']:.1f}", f"{g['cv']:.3f}", int(g["min"]),
                   int(g["max"]), int(g["max_jump"]), f"{g['rel_max_jump']:.2f}", f"{g['mean_abs_jump']:.1f}",
                   g["sign_changes"]]))
    P("\nDelay relative to the 0.020 baseline (iterations, per mesh 160x20 ... 800x100):")
    for name, (m, fn) in series.items():
        for t in TOLS:
            if t == BASE:
                continue
            P(f"  {name:13s} {t:.3f}: " + " ".join(f"{fn(data[t][(m, mesh)]) - fn(data[BASE][(m, mesh)]):+d}" for mesh in meshes))

    # ---- stop diagnostics -----------------------------------------------------------
    P("\n## Reported window maximum of the relative change / grayness range / M_nd at the stop\n")
    for m in METHODS:
        P(f"{m}:")
        for mesh in meshes:
            P(f"  {mesh_s(mesh):8s} " + " | ".join(
                f"{t:.3f}: {data[t][(m, mesh)]['final_rel_change']:.4f} {data[t][(m, mesh)]['final_gray_range']:.4f} "
                f"{data[t][(m, mesh)]['mnd']:.4f}" for t in TOLS))

    # ---- engineering vs baseline --------------------------------------------------------
    P("\n## Final engineering result relative to the 0.020 baseline\n")
    eng = []
    for m in METHODS:
        for t in TOLS:
            if t == BASE:
                continue
            for mesh in meshes:
                a, b = data[BASE][(m, mesh)], data[t][(m, mesh)]
                xa, xb = exports[BASE][(m, mesh)]["x"], exports[t][(m, mesh)]["x"]
                eng.append(dict(method=m, rel_tol=t, mesh=mesh_s(mesh),
                                d_omega_E1_pct=100 * (b["omega1_E1"] / a["omega1_E1"] - 1),
                                d_omega_native_pct=100 * (b["omega1_native"] / a["omega1_native"] - 1),
                                d_Mnd=b["mnd"] - a["mnd"], rel_l2_design=np.linalg.norm(xb - xa) / np.linalg.norm(xa),
                                binary_mismatch_pct=100 * np.mean((xa > 0.5) != (xb > 0.5)),
                                components_base=a["n_components"], components=b["n_components"]))
    with open(os.path.join(HERE, "reltol_study_engineering_vs_0p02.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eng[0].keys()))
        w.writeheader()
        w.writerows(eng)
    P(fmt(["method", "tol", "mesh", "dw1_E1%", "dw1_nat%", "dMnd", "relL2x", "bin%", "comp"], "{:>10}"))
    for e in eng:
        P(fmt([e["method"], f"{e['rel_tol']:.3f}", e["mesh"], f"{e['d_omega_E1_pct']:+.3f}", f"{e['d_omega_native_pct']:+.2f}",
               f"{e['d_Mnd']:+.4f}", f"{e['rel_l2_design']:.4f}", f"{e['binary_mismatch_pct']:.2f}",
               f"{e['components_base']}->{e['components']}"], "{:>10}"))
    P("\nSummary per (method, tol): max |dw1_E1| %, max |dMnd|, max relL2x, max binary mismatch %")
    for m in METHODS:
        for t in TOLS:
            if t == BASE:
                continue
            sel = [e for e in eng if e["method"] == m and e["rel_tol"] == t]
            P(f"  {m:8s} {t:.3f}: max|dw1_E1| = {max(abs(e['d_omega_E1_pct']) for e in sel):.3f} % "
              f"(signed range {min(e['d_omega_E1_pct'] for e in sel):+.3f} .. {max(e['d_omega_E1_pct'] for e in sel):+.3f}), "
              f"max|dMnd| = {max(abs(e['d_Mnd']) for e in sel):.4f}, max relL2x = {max(e['rel_l2_design'] for e in sel):.4f}, "
              f"max bin = {max(e['binary_mismatch_pct'] for e in sel):.2f} %")

    # ---- trajectory identity --------------------------------------------------------------
    P("\n## Trajectory identity (Proposed: objective histories are prefixes of each other?)\n")
    for t in TOLS:
        if t == BASE:
            continue
        out = []
        for mesh in meshes:
            ha, hb = exports[BASE][("proposed", mesh)]["obj"], exports[t][("proposed", mesh)]["obj"]
            n = min(len(ha), len(hb))
            out.append("IDENT" if np.array_equal(ha[:n], hb[:n]) else "DIFF")
        P(f"  {t:.3f} vs 0.020: " + " ".join(out))
    P("  Yuksel: stage-1 length differs between tolerances, so stage 2 starts from different designs (no prefix test).")

    # ---- repeat run -------------------------------------------------------------------------
    P("\n## Repeat of the 0.020 baseline (same tree except label, run later on the same host)\n")
    same = True
    tdev = []
    for m in METHODS:
        for mesh in meshes:
            a, b = data[BASE][(m, mesh)], rep[(m, mesh)]
            if (a["count1"], a["count2"]) != (b["count1"], b["count2"]) or a["omega1_E1"] != b["omega1_E1"] \
                    or not np.array_equal(exports[BASE][(m, mesh)]["x"], rep_exp[(m, mesh)]["x"]):
                same = False
                P(f"  DIFFERS: {m} {mesh_s(mesh)} counts {a['count1']}/{a['count2']} vs {b['count1']}/{b['count2']}")
            tdev.append((m, mesh, b["total"] / a["total"] - 1))
    P(f"  counts, omega1_E1 and final designs bit-identical: {same}")
    P("  wall-time repeat/baseline - 1 [%]: " + ", ".join(f"{m[0].upper()} {mesh_s(mesh)} {100 * d:+.1f}" for m, mesh, d in tdev))
    P(f"  => timing noise between identical runs: max |dev| = {100 * max(abs(d) for *_, d in tdev):.1f} %, "
      f"median |dev| = {100 * np.median([abs(d) for *_, d in tdev]):.1f} %")

    # ---- timing -------------------------------------------------------------------------------
    P("\n## Wall time [s] (total, timed solve only)\n")
    P(fmt(["method", "mesh"] + [f"{t:.3f}" for t in TOLS]))
    for m in METHODS:
        for mesh in meshes:
            P(fmt([m, mesh_s(mesh)] + [f"{data[t][(m, mesh)]['total']:.2f}" for t in TOLS]))
    P("\nCampaign totals (sum over nine meshes) [s]:")
    for m in METHODS:
        P(f"  {m:8s} " + "  ".join(f"{t:.3f}: {sum(data[t][(m, mesh)]['total'] for mesh in meshes):.1f}" for t in TOLS))
    P("\nScaling fits produced by the benchmark (secondary observation only):")
    for t in TOLS:
        sc = results[t]["scaling"]
        tot = {d["method"]: d for d in sc["methods"]}
        P(f"  {t:.3f}: " + "; ".join(f"{k} p={v['p']:.3f} R2={v['R2']:.3f}" for k, v in tot.items()))
    P("\nPer-iteration cost [ms/iter], 800x100 (Proposed SIMP loop; Yuksel stage 1 | stage 2):")
    for t in TOLS:
        p, y = data[t][("proposed", meshes[-1])], data[t][("yuksel", meshes[-1])]
        P(f"  {t:.3f}: P {1e3 * p['time2'] / p['count2']:.1f}   Y {1e3 * y['time1'] / y['k1']:.1f} | {1e3 * y['time2'] / y['k2']:.1f}")

    # ---- anomalies ----------------------------------------------------------------------------
    P("\n## Anomalies\n")
    anom = [t for t in table if t["status"] != "NATIVE_CONVERGED" or t["cap_hit"] or t["timing_fail"]
            or "stagnation_x" not in t["stop_reason"]
            or (t["method"] == "yuksel" and "stagnation_x" not in t["stage1_stop_reason"])]
    P("\n".join(f"  {a}" for a in anom) if anom else "  none: every row NATIVE_CONVERGED on stagnation_x_tolerance "
      "(Yuksel: both stages), no cap hit, no timing-accounting failure")

    open(os.path.join(HERE, "reltol_study_summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    plots(data, exports, meshes)


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


def plots(data, exports, meshes):
    ne = np.array([m[0] * m[1] for m in meshes])
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                         "text.color": "#0b0b0b", "axes.labelcolor": "#52514e"})

    def series(ax, fn, m):
        for t in TOLS:
            ax.plot(ne, [fn(data[t][(m, mesh)]) for mesh in meshes], color=COLORS[t], marker=MARKERS[t],
                    markersize=6, linewidth=2.6 if t == BASE else 1.8, label=f"εx = {t:.3f}" + (" (baseline)" if t == BASE else ""))

    def grid(specs, fname, ncols=2, logy=False, legend_ax=0):
        nrows = (len(specs) + ncols - 1) // ncols
        fig, axs = plt.subplots(nrows, ncols, figsize=(11, 3.9 * nrows), facecolor="#fcfcfb", squeeze=False)
        axs = axs.ravel()
        for ax, (fn, m, title, ylabel) in zip(axs, specs):
            series(ax, fn, m)
            ax.set_title(title, loc="left")
            style(ax, "Number of elements Ne", ylabel)
            if logy:
                ax.set_xscale("log")
                ax.set_yscale("log")
        for ax in axs[len(specs):]:
            ax.axis("off")
        axs[legend_ax].legend(frameon=False, fontsize=9)
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
    grid([(lambda r: r["final_rel_change"], "proposed", "Proposed: window max of relative change at stop", "relative L2 change"),
          (lambda r: r["final_rel_change"], "yuksel", "Yuksel (stage 2): window max of relative change at stop", "relative L2 change")],
         "fig_stop_relchange_vs_ne.png")

    # topology montages (rows: mesh, columns: tolerance)
    for m in METHODS:
        fig, axs = plt.subplots(len(meshes), len(TOLS), figsize=(14, 0.5 * len(meshes) + 0.7), facecolor="#fcfcfb",
                                gridspec_kw=dict(hspace=0.12, wspace=0.04))
        for i, mesh in enumerate(meshes):
            for j, t in enumerate(TOLS):
                x = exports[t][(m, mesh)]["x"].reshape((mesh[1], mesh[0]), order="F")
                ax = axs[i, j]
                ax.imshow(1 - x, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"εx = {t:.3f}")
                if j == 0:
                    ax.set_ylabel(mesh_s(mesh), rotation=0, ha="right", va="center", fontsize=9)
        fig.suptitle(f"{'Proposed' if m == 'proposed' else 'Yuksel'}: final designs", x=0.02, ha="left", fontweight="bold")
        fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.01)
        fig.savefig(os.path.join(HERE, f"fig_topologies_{m}.png"), dpi=130)
        plt.close(fig)

    # close-ups of the two features under watch (Yuksel): 800x100 left end bay, 320x40 centre
    crops = [((800, 100), slice(0, 160), "Yuksel 800x100: left end bay (first 1/5 of the span)"),
             ((320, 40), slice(80, 240), "Yuksel 320x40: centre (middle half of the span)")]
    fig, axs = plt.subplots(len(crops), len(TOLS), figsize=(14, 4.2), facecolor="#fcfcfb", squeeze=False,
                            gridspec_kw=dict(hspace=0.35, wspace=0.05))
    for i, (mesh, cols, title) in enumerate(crops):
        for j, t in enumerate(TOLS):
            x = exports[t][("yuksel", mesh)]["x"].reshape((mesh[1], mesh[0]), order="F")[:, cols]
            ax = axs[i, j]
            ax.imshow(1 - x, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"εx = {t:.3f}", fontsize=9)
        axs[i, 0].set_ylabel(title.split(":")[0], fontsize=9)
        fig.text(0.01, 0.97 - i * 0.49, title, fontweight="bold", fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.02)
    fig.savefig(os.path.join(HERE, "fig_yuksel_feature_closeups.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
