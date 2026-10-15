"""stagnation_x window-length study (W = 10/15/20, rel 2e-2, gray 2e-2), Proposed + Yuksel.

Reads three nine-mesh campaigns of examples/Performance/performance_comparison.m
(benchmark_results.json + an export of benchmark_records.mat written by
wstudy_export.m: final designs x and objective histories) and writes tables and
figures into this folder.  Iteration counts first; curve fits are only quoted.

    python3 analyze_window_study.py
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
WINDOWS = [10, 15, 20]
CAMPAIGNS = {W: os.path.join(PB, f"campaign_remote_stagnation_x_w{W}_rel2e-2_gray2e-2") for W in WINDOWS}
EXPORTS = {W: os.path.join(HERE, "exports", f"export_w{W}.mat") for W in WINDOWS}
# Mac reference of the W = 10 configuration (same tolerances, stale label)
MAC_W10 = os.path.join(PB, "campaign_mac_stagnation_x_no_olhoff_rel1e-2_gray1e-2_audited_4")
MAC_W10_EXPORT = os.path.join(HERE, "exports", "export_mac_w10.mat")

COLORS = {10: "#2a78d6", 15: "#eb6834", 20: "#1baf7a"}   # categorical slots 1-3
MARKERS = {10: "o", 15: "s", 20: "^"}
CAP_REASONS = ("max_iter", "maxiter", "budget", "cap")


def load_rows(cdir):
    res = json.load(open(os.path.join(cdir, "benchmark_results.json")))
    rows = {}
    for r in res["runs"]:
        if r.get("is_warmup"):
            continue
        st, c, t, ev = r["stopping"], r["counts"], r["times"], r.get("evaluator") or {}
        key = (r["method_key"], tuple(r["mesh"]))
        reason = str(st.get("stop_reason"))
        s1reason = str(st.get("stage1_stop_reason"))
        cap = any(s in reason.lower() for s in CAP_REASONS) or any(s in s1reason.lower() for s in CAP_REASONS)
        rows[key] = dict(
            method=r["method_key"], nelx=r["mesh"][0], nely=r["mesh"][1], ne=r["n_elements"],
            status=r["status"], ok=r["ok"], count1=c["count1"], count2=c["count2"],
            k1=c.get("stage1_iterations"), k2=c.get("stage2_iterations"),
            k_total=st.get("total_iterations"),
            time1=t["time1"], time2=t["time2"], other=t["overhead_time_s"], total=t["total_wall_time_s"],
            omega1_native=r["omega1_native"], omega1_E1=r.get("omega1_common_E1"),
            native_flag=r.get("omega1_native_flag"),
            mnd=st.get("final_grayness"), mnd_eval=ev.get("grayness"),
            final_rel_change=st.get("final_stagnation_relative_change"),
            final_gray_range=st.get("final_stagnation_grayness_range"),
            window=st.get("stagnation_window"), rel_tol=st.get("stagnation_relative_change_tol"),
            gray_tol=st.get("stagnation_grayness_tol"),
            stop_reason=reason, stage1_stop_reason=s1reason, cap_hit=cap,
            timing_fail=r["accounting"]["timing_accounting_fail"],
            crosscheck_fail=r["accounting"]["independent_crosscheck_fail"],
            connected=(ev.get("connectivity_binary") or {}).get("left_right_connected"),
            n_components=(ev.get("connectivity_binary") or {}).get("n_components"),
        )
    return res, rows


def load_export(path):
    E = sio.loadmat(path, squeeze_me=True)["E"]
    out = {}
    for e in np.atleast_1d(E):
        out[(str(e["method"]), (int(e["nelx"]), int(e["nely"])))] = dict(
            x=np.atleast_1d(e["x"]).astype(float), obj=np.atleast_1d(e["obj"]).astype(float))
    return out


def iters(row):
    """The method's optimization iteration count used for regularity."""
    return row["count2"] if row["method"] == "proposed" else row["k1"] + row["k2"]


def regularity(k):
    k = np.asarray(k, float)
    d = np.diff(k)
    return dict(mean=k.mean(), std=k.std(ddof=1), cv=k.std(ddof=1) / k.mean(), min=k.min(), max=k.max(),
                max_jump=np.abs(d).max(), mean_abs_jump=np.abs(d).mean(),
                rel_max_jump=(np.abs(d) / k[:-1]).max(), sign_changes=int(np.sum(np.diff(np.sign(d[d != 0])) != 0)))


def main():
    data, results, exports = {}, {}, {}
    for W in WINDOWS:
        results[W], data[W] = load_rows(CAMPAIGNS[W])
        exports[W] = load_export(EXPORTS[W])
    _, mac = load_rows(MAC_W10)
    mac_exp = load_export(MAC_W10_EXPORT) if os.path.exists(MAC_W10_EXPORT) else None
    meshes = sorted({k[1] for k in data[10]}, key=lambda m: m[0] * m[1])
    methods = ["proposed", "yuksel"]
    lines = []
    P = lines.append

    # ---- sanity: configuration actually differs only in W ------------------
    P("# stagnation_x window study -- analysis output\n")
    for W in WINDOWS:
        ws = {(r["window"], r["rel_tol"], r["gray_tol"]) for r in data[W].values()}
        P(f"W={W}: (window, rel_tol, gray_tol) recorded in rows = {sorted(ws)}; label {results[W]['run_label']}")
    P("")

    # ---- master table --------------------------------------------------------
    cols = ["W", "method", "mesh", "ne", "status", "count1", "count2", "k1", "k2", "iters", "total_s",
            "time1_s", "time2_s", "s_per_iter", "s_per_iter_stage1", "s_per_iter_stage2",
            "omega1_native", "native_flag", "omega1_E1", "Mnd", "final_rel_change", "final_gray_range",
            "stop_reason", "stage1_stop_reason", "cap_hit", "timing_fail", "binary_components"]
    table = []
    for W in WINDOWS:
        for m in methods:
            for mesh in meshes:
                r = data[W][(m, mesh)]
                it = iters(r)
                table.append(dict(
                    W=W, method=m, mesh=f"{mesh[0]}x{mesh[1]}", ne=r["ne"], status=r["status"],
                    count1=r["count1"], count2=r["count2"], k1=r["k1"] if m == "yuksel" else "",
                    k2=r["k2"] if m == "yuksel" else "", iters=it, total_s=r["total"],
                    time1_s=r["time1"], time2_s=r["time2"],
                    s_per_iter=(r["time2"] / it if m == "proposed" else (r["time1"] + r["time2"]) / it),
                    s_per_iter_stage1=(r["time1"] / r["k1"] if m == "yuksel" else ""),
                    s_per_iter_stage2=(r["time2"] / r["k2"] if m == "yuksel" else ""),
                    omega1_native=r["omega1_native"], native_flag=r["native_flag"], omega1_E1=r["omega1_E1"],
                    Mnd=r["mnd"], final_rel_change=r["final_rel_change"], final_gray_range=r["final_gray_range"],
                    stop_reason=r["stop_reason"], stage1_stop_reason=r["stage1_stop_reason"] if m == "yuksel" else "",
                    cap_hit=r["cap_hit"], timing_fail=r["timing_fail"] or r["crosscheck_fail"],
                    binary_components=r["n_components"]))
    with open(os.path.join(HERE, "window_study_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)

    # ---- iteration counts ----------------------------------------------------
    def fmt_row(vals, wfmt="{:>6}"):
        return " ".join(wfmt.format(v) for v in vals)

    P("## Iteration counts (rows: mesh; columns: W)\n")
    P("Proposed k_stop:")
    P(fmt_row(["mesh", "Ne"] + [f"W{W}" for W in WINDOWS] + ["MacW10"], "{:>9}"))
    for mesh in meshes:
        P(fmt_row([f"{mesh[0]}x{mesh[1]}", mesh[0] * mesh[1]] + [data[W][("proposed", mesh)]["count2"] for W in WINDOWS]
                  + [mac[("proposed", mesh)]["count2"]], "{:>9}"))
    P("\nYuksel k1 / k2 / k1+k2:")
    P(fmt_row(["mesh"] + [f"W{W}" for W in WINDOWS] + ["MacW10"], "{:>14}"))
    for mesh in meshes:
        cells = []
        for src in [data[W] for W in WINDOWS] + [mac]:
            r = src[("yuksel", mesh)]
            cells.append(f"{r['k1']}/{r['k2']}/{r['k1'] + r['k2']}")
        P(fmt_row([f"{mesh[0]}x{mesh[1]}"] + cells, "{:>14}"))

    P("\n## Regularity across the nine meshes (counts)\n")
    P(fmt_row(["series", "W", "mean", "std", "CV", "min", "max", "maxJump", "relMaxJ", "meanJump", "signChg"], "{:>9}"))
    series = {"Proposed k": lambda r: r["count2"], "Yuksel k1": lambda r: r["k1"],
              "Yuksel k2": lambda r: r["k2"], "Yuksel k1+k2": lambda r: r["k1"] + r["k2"]}
    reg = {}
    for name, fn in series.items():
        m = "proposed" if name.startswith("Proposed") else "yuksel"
        for W in WINDOWS:
            g = regularity([fn(data[W][(m, mesh)]) for mesh in meshes])
            reg[(name, W)] = g
            P(fmt_row([name, W, f"{g['mean']:.1f}", f"{g['std']:.1f}", f"{g['cv']:.3f}", int(g['min']), int(g['max']),
                       int(g['max_jump']), f"{g['rel_max_jump']:.2f}", f"{g['mean_abs_jump']:.1f}", g['sign_changes']], "{:>9}"))

    # ---- engineering result vs W = 10 ---------------------------------------
    P("\n## Final engineering result relative to W = 10\n")
    P("omega1_E1 relative change [%], |dMnd| [abs, Mnd as fraction], design ||x_W - x_10||/||x_10||, binary(x>0.5) mismatch [% of elements]")
    eng = []
    for m in methods:
        for W in WINDOWS[1:]:
            for mesh in meshes:
                a, b = data[10][(m, mesh)], data[W][(m, mesh)]
                xa, xb = exports[10][(m, mesh)]["x"], exports[W][(m, mesh)]["x"]
                eng.append(dict(method=m, W=W, mesh=f"{mesh[0]}x{mesh[1]}",
                                d_omega_E1_pct=100 * (b["omega1_E1"] - a["omega1_E1"]) / a["omega1_E1"],
                                d_omega_native_pct=100 * (b["omega1_native"] - a["omega1_native"]) / a["omega1_native"],
                                d_Mnd=b["mnd"] - a["mnd"],
                                rel_l2_design=np.linalg.norm(xb - xa) / np.linalg.norm(xa),
                                binary_mismatch_pct=100 * np.mean((xa > 0.5) != (xb > 0.5)),
                                components_10=a["n_components"], components_W=b["n_components"]))
    with open(os.path.join(HERE, "window_study_engineering_vs_w10.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eng[0].keys()))
        w.writeheader()
        w.writerows(eng)
    P(fmt_row(["method", "W", "mesh", "dw1_E1%", "dw1_nat%", "dMnd", "relL2x", "bin%", "comp"], "{:>10}"))
    for e in eng:
        P(fmt_row([e["method"], e["W"], e["mesh"], f"{e['d_omega_E1_pct']:+.3f}", f"{e['d_omega_native_pct']:+.2f}",
                   f"{e['d_Mnd']:+.4f}", f"{e['rel_l2_design']:.4f}", f"{e['binary_mismatch_pct']:.2f}",
                   f"{e['components_10']}->{e['components_W']}"], "{:>10}"))
    P("\nSummary per (method, W): max |dw1_E1| %, max |dMnd|, max relL2x, max binary mismatch %")
    for m in methods:
        for W in WINDOWS[1:]:
            sel = [e for e in eng if e["method"] == m and e["W"] == W]
            P(f"  {m:8s} W{W}: max|dw1_E1| = {max(abs(e['d_omega_E1_pct']) for e in sel):.3f} %, "
              f"max|dMnd| = {max(abs(e['d_Mnd']) for e in sel):.4f}, "
              f"max relL2x = {max(e['rel_l2_design'] for e in sel):.4f}, "
              f"max bin = {max(e['binary_mismatch_pct'] for e in sel):.2f} %")

    # ---- trajectory prefix identity (Proposed: full history; Yuksel: stage 2 only) --
    P("\n## Trajectory identity: does the W run pass bit-identically through the W=10 stop?\n")
    for m in methods:
        for W in WINDOWS[1:]:
            out = []
            for mesh in meshes:
                h10, hW = exports[10][(m, mesh)]["obj"], exports[W][(m, mesh)]["obj"]
                if m == "yuksel" and data[10][(m, mesh)]["k1"] != data[W][(m, mesh)]["k1"]:
                    out.append(f"{mesh[0]}x{mesh[1]}:k1 differs")
                    continue
                n = min(len(h10), len(hW))
                same = np.array_equal(h10[:n], hW[:n])
                mx = np.max(np.abs(h10[:n] - hW[:n]) / np.abs(h10[:n])) if n else np.nan
                out.append(f"{mesh[0]}x{mesh[1]}:{'IDENT' if same else f'maxrel {mx:.1e}'}")
            P(f"  {m} W{W}: " + ", ".join(out))
    P("  (Yuksel objective_history is the stage-2 history; when k1 differs stage 2 starts from a different design.)")

    # ---- Mac vs remote W = 10 ------------------------------------------------
    P("\n## Cross-host check: remote W=10 vs Mac W=10 (same configuration)\n")
    for m in methods:
        diffs = []
        for mesh in meshes:
            a, b = data[10][(m, mesh)], mac[(m, mesh)]
            if iters(a) != iters(b) or (m == "yuksel" and a["k1"] != b["k1"]):
                diffs.append(f"{mesh[0]}x{mesh[1]} remote {a['k1'] if m == 'yuksel' else ''}/{iters(a)} vs Mac "
                             f"{b['k1'] if m == 'yuksel' else ''}/{iters(b)}")
        dw = max(abs(data[10][(m, mesh)]["omega1_E1"] / mac[(m, mesh)]["omega1_E1"] - 1) for mesh in meshes)
        P(f"  {m}: count differences: {diffs if diffs else 'none'}; max |omega1_E1 rel diff| = {100 * dw:.4f} %")

    # ---- timing and scaling ----------------------------------------------------
    P("\n## Wall time [s] (total, timed solve only)\n")
    P(fmt_row(["method", "mesh"] + [f"W{W}" for W in WINDOWS] + ["W15/W10", "W20/W10"], "{:>9}"))
    for m in methods:
        for mesh in meshes:
            t = [data[W][(m, mesh)]["total"] for W in WINDOWS]
            P(fmt_row([m, f"{mesh[0]}x{mesh[1]}"] + [f"{v:.2f}" for v in t] + [f"{t[1] / t[0]:.2f}", f"{t[2] / t[0]:.2f}"], "{:>9}"))
    P("\nCampaign totals (sum over nine meshes) [s]:")
    for m in methods:
        P(f"  {m:8s} " + "  ".join(f"W{W} {sum(data[W][(m, mesh)]['total'] for mesh in meshes):.1f}" for W in WINDOWS))
    P("\nScaling fits produced by the benchmark (T = C*Ne^p, total wall; stage time):")
    for W in WINDOWS:
        sc = results[W]["scaling"]
        tot = {d["method"]: d for d in sc["methods"]}
        stg = {d["method"]: d for d in sc["stage_time"]["methods"]}
        P("  W%d: " % W + "; ".join(f"{k} p={tot[k]['p']:.3f} R2={tot[k]['R2']:.3f} (stage p={stg[k]['p']:.3f} R2={stg[k]['R2']:.3f})"
                                    for k in tot))
    P("\nPer-iteration cost [ms/iter] (Proposed: SIMP loop; Yuksel: stage 1 | stage 2):")
    for mesh in meshes:
        cells = []
        for W in WINDOWS:
            p, y = data[W][("proposed", mesh)], data[W][("yuksel", mesh)]
            cells.append(f"P {1e3 * p['time2'] / p['count2']:.1f} Y {1e3 * y['time1'] / y['k1']:.1f}|{1e3 * y['time2'] / y['k2']:.1f}")
        P(f"  {mesh[0]}x{mesh[1]:<4} " + "   ".join(f"W{W}: {c}" for W, c in zip(WINDOWS, cells)))

    # ---- anomalies -------------------------------------------------------------
    P("\n## Anomalies\n")
    anom = [t for t in table if t["status"] != "NATIVE_CONVERGED" or t["cap_hit"] or t["timing_fail"]
            or "stagnation_x" not in t["stop_reason"]
            or (t["method"] == "yuksel" and "stagnation_x" not in t["stage1_stop_reason"])]
    P("\n".join(f"  {a}" for a in anom) if anom else "  none: every row NATIVE_CONVERGED on stagnation_x_tolerance "
      "(Yuksel: both stages), no cap hit, no timing-accounting failure")
    flagged = [t for t in table if t["native_flag"]]
    P(f"\n  native-omega dagger rows (native deviates >5 % from E1; void modes): "
      f"{', '.join(f'W{t['W']} {t['method']} {t['mesh']}' for t in flagged) or 'none'}")

    open(os.path.join(HERE, "window_study_summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
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

    def series_plot(ax, fn, method):
        for W in WINDOWS:
            ax.plot(ne, [fn(data[W][(method, m)]) for m in meshes], color=COLORS[W], marker=MARKERS[W],
                    markersize=6, linewidth=2, label=f"W = {W}")

    # iteration counts
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), facecolor="#fcfcfb")
    specs = [(axs[0, 0], lambda r: r["count2"], "proposed", "Proposed: SIMP iterations at stop"),
             (axs[0, 1], lambda r: r["k1"] + r["k2"], "yuksel", "Yuksel: k1 + k2"),
             (axs[1, 0], lambda r: r["k1"], "yuksel", "Yuksel: stage-1 iterations k1"),
             (axs[1, 1], lambda r: r["k2"], "yuksel", "Yuksel: stage-2 iterations k2")]
    for ax, fn, m, title in specs:
        series_plot(ax, fn, m)
        ax.set_title(title, loc="left")
        style(ax, "Number of elements Ne", "iterations")
        ax.set_ylim(bottom=0)
    axs[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_iterations_vs_ne.png"), dpi=150)
    plt.close(fig)

    # wall time
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    for ax, m, title in [(axs[0], "proposed", "Proposed: total wall time"), (axs[1], "yuksel", "Yuksel: total wall time")]:
        series_plot(ax, lambda r: r["total"], m)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        style(ax, "Number of elements Ne", "wall time [s]")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_walltime_vs_ne.png"), dpi=150)
    plt.close(fig)

    # engineering result
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), facecolor="#fcfcfb")
    for j, m in enumerate(["proposed", "yuksel"]):
        name = "Proposed" if m == "proposed" else "Yuksel"
        series_plot(axs[0, j], lambda r: r["omega1_E1"], m)
        axs[0, j].set_title(f"{name}: omega1 (common evaluator E1)", loc="left")
        style(axs[0, j], "Number of elements Ne", "omega1 E1 [rad/s]")
        series_plot(axs[1, j], lambda r: 100 * r["mnd"], m)
        axs[1, j].set_title(f"{name}: final M_nd", loc="left")
        style(axs[1, j], "Number of elements Ne", "M_nd [%]")
    axs[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_engineering_vs_ne.png"), dpi=150)
    plt.close(fig)

    # topology montage from the final designs (same colour map for all)
    for m in ["proposed", "yuksel"]:
        fig, axs = plt.subplots(len(meshes), 3, figsize=(11, 0.62 * len(meshes) + 0.7), facecolor="#fcfcfb",
                                gridspec_kw=dict(hspace=0.12, wspace=0.04))
        for i, mesh in enumerate(meshes):
            for j, W in enumerate(WINDOWS):
                x = exports[W][(m, mesh)]["x"].reshape((mesh[1], mesh[0]), order="F")
                ax = axs[i, j]
                ax.imshow(1 - x, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"W = {W}", loc="center")
                if j == 0:
                    ax.set_ylabel(f"{mesh[0]}x{mesh[1]}", rotation=0, ha="right", va="center", fontsize=9)
        fig.suptitle(f"{'Proposed' if m == 'proposed' else 'Yuksel'}: final designs", x=0.02, ha="left", fontweight="bold")
        fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.01)
        fig.savefig(os.path.join(HERE, f"fig_topologies_{m}.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
