#!/usr/bin/env python3
"""nmp_report.py -- tables, metrics, verdicts (PREREGISTRATION.md section 10) and figures.

Post-processing only.  Inputs are evidence/EXTRACT.json (from nmp_extract.m), the
per-mesh history CSVs and topology grids it wrote, the locks, the launcher/hook
logs and the runner's own artifacts.  Nothing here re-solves anything.
"""
import csv, glob, hashlib, json, math, os, re, statistics, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/piotrek/Programming/topOpt4freqMax"
D = os.path.join(REPO, "analysis/OlhoffCurrent/diagnostics/nine_mesh_pedersen_production")
EV = os.path.join(D, "evidence")
FIG = os.path.join(D, "figures")
os.makedirs(FIG, exist_ok=True)

MESHES = ["160x20", "240x30", "320x40", "400x50", "480x60", "560x70", "640x80", "720x90", "800x100"]


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def jload(p):
    with open(p) as f:
        return json.load(f)


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


lock = jload(os.path.join(D, "CAMPAIGN_LOCK.json"))
X = jload(os.path.join(EV, "EXTRACT.json"))
runs = as_list(X["runs"])
assert [r["mesh"] for r in runs] == MESHES, [r["mesh"] for r in runs]
by = {r["mesh"]: r for r in runs}
out_root = lock["output_root_abs"]
logs = os.path.join(D, "logs")
launch = jload(os.path.join(logs, "campaign_LAUNCH.json"))
end = jload(os.path.join(logs, "campaign_END.json"))
armed = jload(os.path.join(logs, "campaign_HOOKS_ARMED.json"))
after_path = os.path.join(logs, "campaign_AFTER_RUN.json")
after = jload(after_path) if os.path.exists(after_path) else None
ident = jload(os.path.join(EV, "IDENTITY_CHECK.json"))
runner_ident = jload(os.path.join(EV, "RUNNER_CONFIG_IDENTITY.json"))
stdout_log = os.path.join(logs, "campaign_performance_comparison_stdout.log")
log_text = open(stdout_log, errors="replace").read()
OLH_NAME = "Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)"

NE = np.array([by[m]["NE"] for m in MESHES], float)
NDOF = np.array([by[m]["ndof"] for m in MESHES], float)
w1 = np.array([by[m]["omega1"] for m in MESHES])
w2 = np.array([by[m]["omega2"] for m in MESHES])
gap = np.array([by[m]["gap_terminal"] for m in MESHES])
mnd = np.array([by[m]["Mnd"] for m in MESHES])
gray = np.array([by[m]["gray_fraction"] for m in MESHES])
outer = np.array([by[m]["outer"] for m in MESHES], float)
wall = np.array([by[m]["wall_s"] for m in MESHES])
wpo = wall / outer
eigpo = np.array([by[m]["tEig_per_outer"] for m in MESHES])
innpo = np.array([by[m]["tInner_per_outer"] for m in MESHES])
tmean = np.array([by[m]["tOuter_mean"] for m in MESHES])
tmed = np.array([by[m]["tOuter_median"] for m in MESHES])

# =============================================================================
# Verdict 3: integrity
# =============================================================================
events = open(os.path.join(out_root, "runs", "HOOK_EVENTS.log")).read().splitlines()
pre_order = [re.search(r"PRECHECK (\S+)", e).group(1) for e in events if " PRECHECK " in e]
post_order = [re.search(r"POSTCHECK (\S+)", e).group(1) for e in events if " POSTCHECK " in e]
run_dirs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out_root, "runs", "*")) if os.path.isdir(p))
snapshots = [armed["identity"]] + [by[m]["precheck"]["identity"] for m in MESHES] + \
            [by[m]["postcheck"]["identity"] for m in MESHES] + ([after["identity"]] if after else [])
snap_keys = ("head", "impl_tree_sha256", "source_manifest_sha256", "runner_sha256", "olhoffcurrent_run_sha256")
same_identity = all(all(s[k] == snapshots[0][k] for k in snap_keys) for s in snapshots) and \
    snapshots[0]["head"] == lock["head"] and snapshots[0]["impl_tree_sha256"] == lock["impl_tree_sha256"] and \
    snapshots[0]["source_manifest_sha256"] == lock["source_manifest_sha256"]
failure_markers = glob.glob(os.path.join(out_root, "runs", "*", "*FAIL*")) + \
    glob.glob(os.path.join(out_root, "runs", "*", "*ERROR*"))
scripts_unchanged = all(sha(os.path.join(REPO, s["path"])) == s["sha256"] for s in lock["scripts"])
runner_x = X["runner"]
integrity = {
    "a_hooks_armed_and_9_pre_9_post_pass_in_order":
        pre_order == ["warmup_48x6"] + MESHES and post_order == ["warmup_48x6"] + MESHES and
        all(by[m]["precheck"]["pass"] and by[m]["postcheck"]["pass"] for m in MESHES),
    "b_runner_records_exactly_nine_olhoff_frozen_hashes":
        runner_x["n_records"] == 9 and as_list(runner_x["method_keys"]) == ["olhoff"] * 9 and
        all(by[m]["runner"]["effective_config_hash"] == lock["nine_config_hashes"][m] == by[m]["config_hash"]
            for m in MESHES) and all(by[m]["runner"]["method_key"] == "olhoff" for m in MESHES),
    "c_identity_constant_in_all_snapshots": same_identity and all(s["pass"] for s in snapshots),
    "d_no_repeated_or_replaced_run": run_dirs == sorted(MESHES + ["warmup_48x6"]) and
        not glob.glob(os.path.join(REPO, "examples/Performance/conference_benchmark/nine_mesh_pedersen_b21483b_attempt*")),
    "e_exit0_after_run_reached_breakpoints_armed": end["exit_code"] == 0 and after is not None and
        bool(after and after["breakpoints_still_armed"]),
    "f_tap_bit_consistent_with_runner_accounting": all(
        all(by[m]["postcheck"]["tap_crosscheck"].values()) and all(by[m]["runner_accounting_equal_to_tap"].values())
        and by[m]["runner"]["x_sha256"] == by[m]["rho_sha256"] for m in MESHES),
    "g_runner_preflight_pass": bool(runner_x["preflight_pass"]),
    "no_failure_marker_files": not failure_markers,
    "pinned_scripts_unchanged": scripts_unchanged,
    # Direct test of WHICH methods executed (run 1 used a looser regex that matched the
    # runner's header echo of the inert cfg.yukselMaxIters literal; see
    # logs/POSTPROCESS_report.run1_check_defect.txt):
    "only_olhoff_executed_per_log": (
        re.findall(r"^  methods\s+: (.*)$", log_text, re.M) == [OLH_NAME] and
        re.findall(r"^  (.+?) \.\.\. ", log_text, re.M) == [OLH_NAME] * 9 and
        re.findall(r"^  ([^:\n]+): \S+ in [0-9.]+ s$", log_text, re.M) == [OLH_NAME]),
}
V3 = all(integrity.values())

# =============================================================================
# Verdict 4: termination
# =============================================================================
term_rows = {}
for m in MESHES:
    r = by[m]
    t = {
        "runner_status_native_converged": r["runner"]["status"] == "NATIVE_CONVERGED",
        "solver_status_converged": r["solver_status"] == "CONVERGED",
        "log_line": bool(r["log_converged_line"]),
        "final_dx_below_eps": r["final_dxNorm2"] < r["eps"],
        "outer_below_cap": r["outer"] < r["maxOuter"],
        "inner_all_converged": r["n_inner_not_converged"] == 0,
        "box_not_collapsed_to_floor": not r["final_move_max_at_floor"],
    }
    term_rows[m] = t
V4 = all(all(t.values()) for t in term_rows.values())

# =============================================================================
# Verdict 5: mesh trend
# =============================================================================
adj_rel = np.abs(np.diff(w1)) / w1[:-1]
coarse_fine_rel = abs(w1[-1] - w1[0]) / w1[0]
coarse4 = adj_rel[:4].max()
fine4 = adj_rel[4:].max()
vol_err = np.array([by[m]["volume_error"] for m in MESHES])
eval_status = [by[m]["evaluator"].get("status", "ABSENT") for m in MESHES]
spikes_last10 = [by[m]["n_spikes_last10"] for m in MESHES]
flags = {
    "S1_any_not_native_converged": any(by[m]["runner"]["status"] != "NATIVE_CONVERGED" for m in MESHES),
    "S2_adjacent_omega1_change_gt_2pct": bool((adj_rel > 0.02).any()),
    "S3_coarse_to_fine_omega1_change_gt_5pct": bool(coarse_fine_rel > 0.05),
    "S4_fine_half_change_exceeds_05pct_and_coarse_half": bool(fine4 > 0.005 and fine4 > coarse4),
    "S5_volume_error_gt_1e-3": bool((np.abs(vol_err) > 1e-3).any()),
    "S6_fine_grayness_gt_2x_coarse": bool(mnd[-1] > 2 * mnd[0] or gray[-1] > 2 * gray[0]),
    "S7_terminal_localized_mode_indicator": any(s > 0 for s in spikes_last10) or any(s != "PASS" for s in eval_status),
}
V5 = not any(flags.values())

timing_ok = all(not by[m]["runner"]["accounting"]["timing_accounting_fail"] and
                not by[m]["runner"]["accounting"]["independent_crosscheck_fail"] for m in MESHES)
V1 = ident["verdict"] == "NINE_MESH_CAMPAIGN_IDENTITY_PASS"
V2 = runner_ident["verdict"] == "PERFORMANCE_RUNNER_CONFIG_IDENTITY_PASS"
V6 = V1 and V2 and V3 and V4 and V5 and timing_ok

verdicts = {
    "identity": "NINE_MESH_CAMPAIGN_IDENTITY_PASS" if V1 else "NINE_MESH_CAMPAIGN_IDENTITY_FAIL",
    "runner_config_identity": "PERFORMANCE_RUNNER_CONFIG_IDENTITY_PASS" if V2 else "PERFORMANCE_RUNNER_CONFIG_IDENTITY_FAIL",
    "integrity": "NINE_MESH_CAMPAIGN_INTEGRITY_PASS" if V3 else "NINE_MESH_CAMPAIGN_INTEGRITY_FAIL",
    "termination": "PEDERSEN_ADAPTIVE_CROSS_MESH_TERMINATION_PASS" if V4 else "PEDERSEN_ADAPTIVE_CROSS_MESH_TERMINATION_FAIL",
    "mesh_trend": "PEDERSEN_ADAPTIVE_MESH_TREND_ACCEPTABLE" if V5 else "PEDERSEN_ADAPTIVE_MESH_TREND_SUSPICIOUS",
    "baseline": "OLHOFF_PRODUCTION_BASELINE_VALIDATED" if V6 else "OLHOFF_PRODUCTION_BASELINE_NOT_VALIDATED",
}

# =============================================================================
# Scaling fits (log-log OLS, all nine points, leave-one-out range)
# =============================================================================
def fit(x, y):
    lx, ly = np.log(x), np.log(y)
    A = np.vstack([np.ones_like(lx), lx]).T
    beta, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ beta
    r2 = 1 - ((ly - pred) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    loo = []
    for k in range(len(x)):
        idx = [j for j in range(len(x)) if j != k]
        b, *_ = np.linalg.lstsq(A[idx], ly[idx], rcond=None)
        loo.append(float(b[1]))
    return {"C": float(math.exp(beta[0])), "p": float(beta[1]), "R2": float(r2), "n": len(x),
            "loo_p_min": min(loo), "loo_p_max": max(loo),
            "loo_p_by_dropped_mesh": dict(zip(MESHES, loo)),
            "residual_log": dict(zip(MESHES, [float(v) for v in (ly - pred)]))}


fits = {}
for name, y in [("total_wall_s", wall), ("wall_per_outer_s", wpo), ("tOuter_mean_s", tmean),
                ("tOuter_median_s", tmed), ("eig_time_per_outer_s", eigpo), ("inner_time_per_outer_s", innpo),
                ("inner_time_per_inner_iter_s", np.array([by[m]["tInner_per_inner_iter"] for m in MESHES])),
                ("outer_iterations", outer)]:
    fits[name] = {"vs_NE": fit(NE, y), "vs_ndof": fit(NDOF, y)}

# =============================================================================
# Tables
# =============================================================================
cols = ["mesh", "elements", "dofs_total", "dofs_free", "outer", "inner_total", "inner_max_per_outer",
        "omega1", "omega2", "omega3", "gap_pct", "min_gap_hist_pct", "volume", "volume_error", "M_nd",
        "gray_fraction", "terminal_status", "final_dxNorm2", "eps", "final_move_max", "total_wall_s",
        "wall_per_outer_s", "tOuter_mean_s", "tOuter_median_s", "eig_time_per_outer_s",
        "inner_time_per_outer_s", "eig_count", "rho_sha256", "config_hash"]
rows = []
for m in MESHES:
    r = by[m]
    rows.append({
        "mesh": m, "elements": r["NE"], "dofs_total": r["ndof"], "dofs_free": r["nfree"], "outer": r["outer"],
        "inner_total": r["inner_total"], "inner_max_per_outer": r["inner_max"],
        "omega1": f'{r["omega1"]:.10f}', "omega2": f'{r["omega2"]:.10f}', "omega3": f'{r["omega3"]:.10f}',
        "gap_pct": f'{100 * r["gap_terminal"]:.4f}', "min_gap_hist_pct": f'{100 * r["gap_min"]:.4f}',
        "volume": f'{r["volume"]:.10f}', "volume_error": f'{r["volume_error"]:.3e}',
        "M_nd": f'{r["Mnd"]:.6f}', "gray_fraction": f'{r["gray_fraction"]:.6f}',
        "terminal_status": r["runner"]["status"], "final_dxNorm2": f'{r["final_dxNorm2"]:.6e}',
        "eps": f'{r["eps"]:.6g}', "final_move_max": f'{r["final_move_max"]:.6g}',
        "total_wall_s": f'{r["wall_s"]:.3f}', "wall_per_outer_s": f'{r["wall_per_outer"]:.4f}',
        "tOuter_mean_s": f'{r["tOuter_mean"]:.4f}', "tOuter_median_s": f'{r["tOuter_median"]:.4f}',
        "eig_time_per_outer_s": f'{r["tEig_per_outer"]:.4f}', "inner_time_per_outer_s": f'{r["tInner_per_outer"]:.4f}',
        "eig_count": r["eig_count"], "rho_sha256": r["rho_sha256"], "config_hash": r["config_hash"]})
with open(os.path.join(D, "RESULTS_TABLE.csv"), "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=cols)
    wtr.writeheader()
    wtr.writerows(rows)

md = ["# RESULTS_TABLE — nine-mesh Pedersen/adaptive production campaign",
      "",
      "Nine rows, one per preregistered mesh, in run order; nothing is omitted. The source is `evidence/EXTRACT.json`: tapped solver results, cross-checked bit for bit against the runner records in `benchmark_records.mat`.",
      "",
      "Frequencies are the **native** Pedersen/linear-mass eigenfrequencies of the final design. The gap is (ω₂−ω₁)/ω₁ of the final analysis. Times come from the runner's own timers: total = caller-side `tic/toc` around `olhoffSolve`; eig = assembly + `eigs` per outer iteration; inner = nested MMA per outer iteration.",
      "",
      "| mesh | elements | DOFs | outer | inner | ω₁ | ω₂ | gap % | volume | M_nd | gray frac. | status | total wall [s] | wall/outer [s] | mean tOuter [s] | eig/outer [s] | ρ SHA-256 | config hash |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for m in MESHES:
    r = by[m]
    md.append(f'| {m} | {r["NE"]} | {r["ndof"]} ({r["nfree"]} free) | {r["outer"]} | {r["inner_total"]} | '
              f'{r["omega1"]:.4f} | {r["omega2"]:.4f} | {100 * r["gap_terminal"]:.2f} | {r["volume"]:.6f} | '
              f'{r["Mnd"]:.4f} | {r["gray_fraction"]:.4f} | {r["runner"]["status"]} | {r["wall_s"]:.1f} | '
              f'{r["wall_per_outer"]:.3f} | {r["tOuter_mean"]:.3f} | {r["tEig_per_outer"]:.4f} | '
              f'`{r["rho_sha256"][:16]}…` | `{r["config_hash"][:16]}…` |')
md += ["", "The full hashes and the remaining columns are in `RESULTS_TABLE.csv`: ω₃, minimum gap in the history, volume error, final ‖Δρ‖₂, ε, final largest box, median tOuter, inner time per outer, and eigensolve count.", ""]
open(os.path.join(D, "RESULTS_TABLE.md"), "w").write("\n".join(md))

# =============================================================================
# Figures
# =============================================================================
labels = MESHES
x = np.arange(9)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


def series_fig(name, y, ylabel, fmt="{:.2f}", old=None, oldlabel=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(x, y, "o-", color="#1f4e79", label="this campaign (native)")
    if old is not None:
        ax.plot(x, old, "s--", color="#999999", mfc="none", label=oldlabel)
    for xi, yi in zip(x, y):
        ax.annotate(fmt.format(yi), (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("mesh")
    if old is not None:
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(FIG, name)
    fig.savefig(p)
    plt.close(fig)
    return p


figs = {}
figs["omega1"] = series_fig("omega1_vs_mesh.png", w1, "ω₁ native [rad/s]", "{:.2f}")
figs["omega2"] = series_fig("omega2_vs_mesh.png", w2, "ω₂ native [rad/s]", "{:.2f}")
figs["gap"] = series_fig("gap_vs_mesh.png", 100 * gap, "terminal gap (ω₂−ω₁)/ω₁ [%]", "{:.2f}")
figs["mnd"] = series_fig("Mnd_vs_mesh.png", mnd, "M_nd = 4·mean ρ(1−ρ)", "{:.4f}")
figs["gray"] = series_fig("gray_fraction_vs_mesh.png", gray, "gray fraction  mean(0.1<ρ<0.9)", "{:.4f}")
figs["outer"] = series_fig("outer_iterations_vs_mesh.png", outer, "outer iterations", "{:.0f}")


def loglog_fig(name, ys, ylabel):
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, xv, xl in [(axs[0], NE, "elements NE"), (axs[1], NDOF, "total DOFs")]:
        for lab, y, mk in ys:
            ax.loglog(xv, y, mk, label=lab)
            f = fit(xv, y)
            xx = np.geomspace(xv.min(), xv.max(), 50)
            ax.loglog(xx, f["C"] * xx ** f["p"], ":", color="gray", lw=0.8)
            ax.annotate(f'p={f["p"]:.2f}', (xv[-1], y[-1]), textcoords="offset points", xytext=(4, 0), fontsize=7)
        ax.set_xlabel(xl)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
    fig.suptitle("log-log OLS over all nine points (dotted); no point excluded", fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIG, name)
    fig.savefig(p)
    plt.close(fig)
    return p


figs["wall"] = loglog_fig("total_wall_vs_size.png", [("total wall [s]", wall, "o-")], "seconds")
figs["per_outer"] = loglog_fig("time_per_outer_vs_size.png",
                               [("wall/outer", wpo, "o-"), ("median tOuter", tmed, "^-"),
                                ("eig/outer", eigpo, "s-"), ("inner/outer", innpo, "d-")], "seconds per outer iteration")

# nine final topologies, one rendering convention
fig, axs = plt.subplots(9, 1, figsize=(7.2, 9.6))
for ax, m in zip(axs, MESHES):
    r = by[m]
    G = np.fromfile(os.path.join(EV, "topology_grid", m + ".bin"), dtype="<f8").reshape((100, 800), order="F")
    ax.imshow(1 - G, cmap="gray", vmin=0, vmax=1, extent=[0, 8, 0, 1], aspect="equal", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{m}  {r["runner"]["status"]}  outer {r["outer"]}  ω₁ {r["omega1"]:.2f}  ω₂ {r["omega2"]:.2f}  '
                 f'M_nd {r["Mnd"]:.3f}  gray {r["gray_fraction"]:.3f}', fontsize=7, pad=2)
fig.suptitle("final densities, black = 1, white = 0, common [0,1] scale, 8×1 domain "
             "(nearest-neighbour on an 800×100 grid; no smoothing)", fontsize=7)
fig.tight_layout()
figs["topologies"] = os.path.join(FIG, "nine_final_topologies.png")
fig.savefig(figs["topologies"])
plt.close(fig)

# histories (spectral / box / cost)
H = {m: np.genfromtxt(os.path.join(EV, "history", m + ".csv"), delimiter=",", names=True) for m in MESHES}
cmap = plt.get_cmap("viridis")
fig, axs = plt.subplots(2, 2, figsize=(9, 6.2))
for i, m in enumerate(MESHES):
    h = H[m]
    c = cmap(i / 8)
    axs[0, 0].plot(h["outer"], h["omega1"], color=c, lw=0.9, label=m)
    axs[0, 0].plot(h["outer"], h["omega2"], color=c, lw=0.6, ls="--")
    axs[0, 1].semilogy(h["outer"], np.maximum(h["gap12"], 1e-6), color=c, lw=0.9)
    axs[1, 0].semilogy(h["outer"], h["moveMax"], color=c, lw=0.9)
    axs[1, 0].semilogy(h["outer"], h["moveMean"], color=c, lw=0.6, ls="--")
    axs[1, 1].plot(h["outer"], h["tOuter"], color=c, lw=0.7)
axs[0, 0].set_ylabel("ω₁ (solid), ω₂ (dashed)")
axs[0, 0].legend(fontsize=6, ncol=3)
axs[0, 1].axhline(0.05, color="r", lw=0.6, ls=":")
axs[0, 1].axhline(0.02, color="m", lw=0.6, ls=":")
axs[0, 1].set_ylabel("gap12 (dotted: 0.05 multiplicity.tolerance, 0.02 coalescence)")
axs[1, 0].axhline(0.002, color="r", lw=0.6, ls=":")
axs[1, 0].set_ylabel("box: largest (solid), mean (dashed); floor dotted")
axs[1, 1].set_ylabel("tOuter [s]")
for ax in axs.flat:
    ax.set_xlabel("outer iteration")
fig.tight_layout()
figs["histories"] = os.path.join(FIG, "histories_spectral_box_cost.png")
fig.savefig(figs["histories"])
plt.close(fig)

# design-change convergence
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for i, m in enumerate(MESHES):
    h = H[m]
    ax.semilogy(h["outer"], h["dxNorm2"] / by[m]["eps"], color=cmap(i / 8), lw=0.8, label=m)
ax.axhline(1, color="r", lw=0.7, ls=":")
ax.set_xlabel("outer iteration")
ax.set_ylabel("‖Δρ‖₂ / ε(mesh)")
ax.legend(fontsize=6, ncol=3)
fig.tight_layout()
figs["dx"] = os.path.join(FIG, "design_change_over_eps.png")
fig.savefig(figs["dx"])
plt.close(fig)

# =============================================================================
# Upstream sweep comparison (after freeze)
# =============================================================================
cmp_rows = []
for m in MESHES:
    r = by[m]
    o = r["upstream_sweep"]
    if not o.get("available"):
        cmp_rows.append({"mesh": m, "available": False})
        continue
    d_native = [abs(v) for v in o["omega_rel_diff"]]
    d_eq4 = [abs(v) for v in o["w_eq4_rel_diff"]]
    same_counts = o["outer"] == r["outer"] and o["inner"] == r["inner_total"] and o["status"] == r["solver_status"]
    if same_counts and (o["rho_identical"] or max(d_native) <= 1e-12):
        cls = "negligible/numerical"
    elif same_counts and max(d_native) <= 1e-6:
        cls = "plausibly environmental/reporting"
    else:
        cls = "scientifically material"
    cmp_rows.append({
        "mesh": m, "available": True, "class_frequencies_counts": cls, "rho_identical": o["rho_identical"],
        "rho_sha_old": o["rho_sha256"], "rho_sha_new": r["rho_sha256"], "rho_L1_mean": o["rho_L1_mean"],
        "rho_Linf": o["rho_Linf"], "outer_old": o["outer"], "outer_new": r["outer"], "inner_old": o["inner"],
        "inner_new": r["inner_total"], "status_old": o["status"], "status_new": r["solver_status"],
        "w1_native_old": o["omega"][0], "w1_native_new": r["omega1"], "w2_native_old": o["omega"][1],
        "w2_native_new": r["omega2"], "max_rel_diff_native_w123": max(d_native),
        "w1_eq4_old": o["w_eq4"][0], "w1_eq4_new": r["w_eq4"][0], "w2_eq4_old": o["w_eq4"][1],
        "w2_eq4_new": r["w_eq4"][1], "max_rel_diff_eq4_w123": max(d_eq4),
        "gap_eq4_pct_old": o["gap_eq4_pct"], "gap_eq4_pct_new": 100 * r["gap_eq4"],
        "gap_native_pct_old": o["gap_native_pct"], "gap_native_pct_new": 100 * r["gap_terminal"],
        "Mnd_old": o["Mnd"], "Mnd_new": r["Mnd"], "gray_old": o["gray_fraction"], "gray_new": r["gray_fraction"],
        "hist_omega_identical": o["hist_omega_identical"], "hist_nInner_identical": o["hist_nInner_identical"],
        "wall_old_s": o["wall_s"], "wall_new_s": r["wall_s"], "wall_ratio_new_over_old": r["wall_s"] / o["wall_s"]})

# =============================================================================
# METRICS / EVIDENCE
# =============================================================================
trend = {
    "omega1": w1.tolist(), "omega2": w2.tolist(), "gap_terminal": gap.tolist(),
    "omega1_adjacent_abs": np.diff(w1).tolist(), "omega1_adjacent_rel": (np.diff(w1) / w1[:-1]).tolist(),
    "omega1_coarse_to_fine_abs": float(w1[-1] - w1[0]), "omega1_coarse_to_fine_rel": float((w1[-1] - w1[0]) / w1[0]),
    "omega1_range_abs": float(w1.max() - w1.min()), "omega1_range_rel_to_median": float((w1.max() - w1.min()) / np.median(w1)),
    "omega1_fine5_range_abs": float(w1[4:].max() - w1[4:].min()),
    "omega1_fine5_range_rel": float((w1[4:].max() - w1[4:].min()) / np.median(w1[4:])),
    "omega1_largest_adjacent_rel_coarse4": float(coarse4), "omega1_largest_adjacent_rel_fine4": float(fine4),
    "omega2_adjacent_rel": (np.diff(w2) / w2[:-1]).tolist(),
    "Mnd": mnd.tolist(), "gray_fraction": gray.tolist(), "outer": outer.tolist(), "wall_s": wall.tolist(),
    "wall_per_outer_s": wpo.tolist(), "volume_error": vol_err.tolist(),
}
metrics = {"schema": "nmp_metrics/1", "meshes": MESHES, "per_mesh": {}, "trend": trend, "fits": fits,
           "timing_accounting_all_pass": timing_ok, "runner_scaling_fit": X["runner"].get("scaling"),
           "topology_comparisons": X.get("topology_comparisons")}
skip = {"log", "mode_table", "precheck", "postcheck", "tap", "runner", "upstream_sweep", "files", "spikes",
        "dxNorm2_below_eps_iters"}
for m in MESHES:
    r = by[m]
    pm = {k: v for k, v in r.items() if k not in skip}
    pm["status"] = r["runner"]["status"]
    pm["status_note"] = r["runner"]["status_note"]
    pm["spikes"] = r["spikes"]
    pm["mode_table_Ex_Ey_nzy"] = r["mode_table"]
    pm["files"] = r["files"]
    pm["runner_timing_accounting"] = r["runner"]["accounting"]
    metrics["per_mesh"][m] = pm
json.dump(metrics, open(os.path.join(D, "METRICS.json"), "w"), indent=1)

evidence = {
    "schema": "nmp_evidence/1",
    "verdicts": verdicts,
    "verdict_inputs": {"integrity": integrity, "termination": term_rows, "mesh_trend_flags": flags,
                       "timing_accounting_all_pass": timing_ok,
                       "identity_check": ident["checks"], "runner_config_identity": runner_ident["checks"]},
    "campaign": {"launch": launch, "end": end, "hooks_armed_when": armed["when"], "pid": armed["pid"],
                 "after_run": after and {k: after[k] for k in ("when", "breakpoints_still_armed")},
                 "hook_event_order_pre": pre_order, "hook_event_order_post": post_order,
                 "failure_marker_files": failure_markers},
    "upstream_sweep_comparison": cmp_rows,
    "figures": {k: os.path.relpath(v, D) for k, v in figs.items()},
    "lock_sha256": sha(os.path.join(D, "CAMPAIGN_LOCK.json")),
    "preregistration_sha256": sha(os.path.join(D, "PREREGISTRATION.md")),
}
json.dump(evidence, open(os.path.join(D, "EVIDENCE.json"), "w"), indent=1)

print(json.dumps(verdicts, indent=1))
print("integrity", integrity)
print("termination fails", {m: [k for k, v in t.items() if not v] for m, t in term_rows.items()})
print("trend flags", flags)
print("timing ok", timing_ok)
print("adjacent rel", np.round(adj_rel * 100, 4), "coarse->fine %", round(coarse_fine_rel * 100, 4))
for c in cmp_rows:
    print(c.get("mesh"), c.get("class_frequencies_counts"), c.get("rho_identical"), c.get("max_rel_diff_native_w123"),
          c.get("outer_old"), c.get("outer_new"))
for k, v in fits.items():
    print(k, "vs NE p=%.3f R2=%.3f loo[%.3f,%.3f]" % (v["vs_NE"]["p"], v["vs_NE"]["R2"], v["vs_NE"]["loo_p_min"], v["vs_NE"]["loo_p_max"]))
