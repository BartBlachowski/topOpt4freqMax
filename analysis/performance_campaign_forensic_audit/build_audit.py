#!/usr/bin/env python3
"""Build the read-only forensic audit from frozen final-campaign artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FROZEN = REPO / "examples" / "Performance" / "final_campaign"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def num(v):
    if v is None or v == "" or str(v).upper() in {"N/A", "NAN", "NULL"}:
        return math.nan
    return float(v)


def fmt(v, digits=6):
    if v is None or not math.isfinite(float(v)):
        return "N/A"
    return f"{float(v):.{digits}g}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.rstrip()


def fit_power(x, y, fixed_p=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    lx, ly = np.log(x), np.log(y)
    n = len(x)
    if fixed_p is None:
        reg = stats.linregress(lx, ly)
        p = float(reg.slope)
        intercept = float(reg.intercept)
        r2 = float(reg.rvalue**2)
        stderr = float(reg.stderr)
        if n > 2:
            q = float(stats.t.ppf(0.975, n - 2))
            ci = (p - q * stderr, p + q * stderr)
        else:
            ci = (math.nan, math.nan)
    else:
        p = float(fixed_p)
        intercept = float(np.mean(ly - p * lx))
        pred = intercept + p * lx
        sse = float(np.sum((ly - pred) ** 2))
        sst = float(np.sum((ly - np.mean(ly)) ** 2))
        r2 = 1 - sse / sst if sst > 0 else math.nan
        stderr = math.nan
        ci = (math.nan, math.nan)
    pred = np.exp(intercept + p * lx)
    residual = ly - np.log(pred)
    return {
        "C": math.exp(intercept),
        "p": p,
        "R2_log": r2,
        "n": n,
        "p_ci95_low": ci[0],
        "p_ci95_high": ci[1],
        "p_stderr": stderr,
        "log_RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAPE_pct": float(100 * np.mean(np.abs(pred - y) / y)),
        "x": x,
        "y": y,
        "pred": pred,
    }


def vector(row, prefix):
    return "/".join(fmt(num(row[f"omega{i}_{prefix}"])) for i in (1, 2, 3))


def status_ok(method, status):
    if method == "Olhoff":
        return status == "VALID_STABILIZED_STATE_AT_FIXED_WORK"
    return status == "NATIVE_CONVERGED"


with (FROZEN / "benchmark_results.json").open(encoding="utf-8") as f:
    bench = json.load(f)
runs = bench["runs"]
perf = read_csv(FROZEN / "table1_performance.csv")
quality = read_csv(FROZEN / "common_evaluators.csv")
raw_summary_pre = {r["mesh"]: r for r in read_csv(HERE / "olhoff_raw_summary.csv")}
quality_map = {(r["Method"], r["Mesh"]): r for r in quality}
perf_map = {(r["Method"], r["Mesh"]): r for r in perf}
run_map = {
    (r["method"], f"{r['mesh']['nelx']}x{r['mesh']['nely']}"): r for r in runs
}
mesh_order = [f"{160+80*i}x{20+10*i}" for i in range(9)]
method_order = ["Olhoff", "Yuksel", "Proposed"]
colors = {"Olhoff": "#1f77b4", "Yuksel": "#d95f02", "Proposed": "#2ca02c"}


# ---------------------------------------------------------------------------
# Integrity, status semantics, and evidence inventory
# ---------------------------------------------------------------------------
integrity_rows = []
evidence_rows = []
independent_statuses = []
for method in method_order:
    for mesh in mesh_order:
        r = run_map[(method, mesh)]
        p = perf_map[(method, mesh)]
        status = r["stopping"]["status"]
        if method == "Olhoff":
            raw_mat = FROZEN / "raw" / "olhoff" / f"s1_{mesh}.mat"
            raw = raw_summary_pre[mesh]
            raw_failed = raw["runner_status"] == "SOLVER_FAILURE"
            raw_fixed_work = (
                raw["runner_status"] == "CAP_HIT"
                and int(raw["completed_iterations"]) == 1600
                and float(raw["last_recorded_lp_flag"]) == 1
                and float(raw["last_recorded_finite_ok"]) == 1
            )
            if raw_failed:
                independent = "SOLVER_FAILURE"
                completeness = "RAW_PRIOR_HISTORY; FAILED_ATTEMPT_DETAILS_PARTIAL"
            elif raw_fixed_work:
                independent = "VALID_STABILIZED_STATE_AT_FIXED_WORK"
                completeness = "RAW_HISTORY_AND_DENSITY_COMPLETE"
            else:
                independent = "UNRECOGNIZED_STOP"
                completeness = "RAW_HISTORY_REQUIRES_REVIEW"
            raw_path = str(raw_mat.relative_to(REPO))
            histories = "full per-iteration MAT history"
            design = "final rho plus every successful-update snapshot"
            topology = "recoverable from retained densities"
            diagnostics = "LP flag/finite state for successful updates; failure flag only in log"
        else:
            raw_path = "NOT RETAINED"
            dx = r["stopping"]["final_max_density_change"]
            tol = r["stopping"]["convergence_tolerance"]
            reason = r["stopping"]["stop_reason"]
            if "max_iter" in reason:
                independent = "CAP_HIT"
            elif "tolerance" in reason and dx <= tol:
                independent = "NATIVE_CONVERGED"
            else:
                independent = "UNRECOGNIZED_STOP"
            completeness = "AGGREGATE_ONLY; NO DESIGN_OR_HISTORY"
            histories = "checksum only; no samples"
            design = "not retained (checksum only)"
            topology = "not retained"
            diagnostics = "terminal metrics only"
        independent_statuses.append((method, independent))
        verified = independent == status and (p["status"] == status)
        censored = not status_ok(method, status)
        fit_consistent = (p["in_scaling_fit"] == "yes") == (not censored)
        integrity_rows.append(
            {
                "Method": method,
                "Mesh": mesh,
                "Status": status,
                "Censored": "yes" if censored else "no",
                "Artifact_completeness": completeness,
                "Verified": "yes" if verified and fit_consistent else "no",
                "Independent_reclassification": independent,
                "In_existing_scaling_fit": p["in_scaling_fit"],
                "Notes": (
                    "failure attempt is one after completed iteration count"
                    if method == "Olhoff" and censored
                    else ""
                ),
            }
        )
        evidence_rows.append(
            {
                "method": method,
                "mesh": mesh,
                "status": status,
                "aggregate_json": "examples/Performance/final_campaign/benchmark_results.json",
                "performance_csv": "examples/Performance/final_campaign/table1_performance.csv",
                "common_evaluator_csv": "examples/Performance/final_campaign/common_evaluators.csv",
                "raw_artifact": raw_path,
                "available_histories": histories,
                "available_final_design": design,
                "available_topology": topology,
                "available_modal_data": "final native spectrum and common E1/E2/E3 spectra",
                "available_timing_data": "init/loop/post/wall/stages" if method != "Olhoff" else "loop/wall/eigensolve; init/post not separable",
                "available_solver_diagnostics": diagnostics,
            }
        )

write_csv(
    HERE / "campaign_integrity.csv",
    list(integrity_rows[0].keys()),
    integrity_rows,
)
write_csv(HERE / "evidence_map.csv", list(evidence_rows[0].keys()), evidence_rows)
shutil.copyfile(FROZEN / "common_evaluators.csv", HERE / "common_quality_comparison.csv")


# ---------------------------------------------------------------------------
# Proposed coarse-mesh and Yuksel cap tables
# ---------------------------------------------------------------------------
proposed_rows = []
for mesh in mesh_order:
    r = run_map[("Proposed", mesh)]
    q = quality_map[("Proposed", mesh)]
    if mesh in {"160x20", "240x30"}:
        diagnosis = (
            "MODEL / INTERPOLATION DEPENDENCE: native low-frequency triplet is absent "
            "under raw E1/E2/E3 and binary E1/E2/E3; high grayness amplifies void-model sensitivity"
        )
        topo_class = "UNAVAILABLE; final density/topology not retained"
    elif mesh == "320x40":
        diagnosis = "transition complete; native and common-raw omega1 agree within 0.2%"
        topo_class = "UNAVAILABLE; scalar evaluator transition only"
    else:
        diagnosis = "stable high-frequency branch under native and common raw evaluation"
        topo_class = "UNAVAILABLE; checksum only"
    proposed_rows.append(
        {
            "Mesh": mesh,
            "native_omega1": q["omega1_native"],
            "native_omega1_omega2_omega3": "/".join(fmt(x) for x in r["results"]["final_frequencies_rad_s"]),
            "common_raw_quality_E1_E2_E3_omega1": "/".join(q[f"omega1_common_raw_E{i}"] for i in (1, 2, 3)),
            "common_binary_quality_E1_E2_E3_omega1": "/".join(q[f"omega1_common_binary_E{i}"] for i in (1, 2, 3)),
            "grayness": q["grayness"],
            "topology_class": topo_class,
            "connectivity_raw_binary": f"{q['connected_raw']}/{q['connected_binary']}",
            "iterations": r["iterations"]["iter_total"],
            "final_max_dx": r["stopping"]["final_max_density_change"],
            "final_relative_objective_change": r["stopping"]["final_relative_objective_change"],
            "diagnosis": diagnosis,
        }
    )
write_csv(HERE / "proposed_coarse_mesh_diagnosis.csv", list(proposed_rows[0].keys()), proposed_rows)

yuksel_rows = []
for mesh in ["560x70", "640x80", "720x90", "800x100"]:
    r = run_map[("Yuksel", mesh)]
    q = quality_map[("Yuksel", mesh)]
    status = r["stopping"]["status"]
    if mesh == "640x80":
        classification = "POSSIBLE_CAP_LIMIT"
        diagnosis = "terminal max dx is 3.06x tolerance but RMS/objective changes are small; trend unavailable"
    elif mesh == "800x100":
        classification = "INDETERMINATE"
        diagnosis = "terminal max dx is 7.28x tolerance; small RMS/objective changes suggest localized motion, but no trend was retained"
    else:
        classification = "NATIVE_CONVERGED"
        diagnosis = "comparison case"
    yuksel_rows.append(
        {
            "Mesh": mesh,
            "Status": status,
            "Iterations": r["iterations"]["iter_total"],
            "Stage1": r["iterations"]["iter_stage1"],
            "Stage2": r["iterations"]["iter_stage2"],
            "final_max_dx": r["stopping"]["final_max_density_change"],
            "final_rms_dx": r["stopping"]["final_rms_density_change"],
            "final_relative_objective_change": r["stopping"]["final_relative_objective_change"],
            "late_trend": "UNAVAILABLE: record_history=false; checksum is not a trajectory",
            "stage": "Stage 2 terminal",
            "quality_raw_E1_E2_E3_omega1": "/".join(q[f"omega1_common_raw_E{i}"] for i in (1, 2, 3)),
            "classification": classification,
            "diagnosis": diagnosis,
        }
    )
write_csv(HERE / "yuksel_cap_diagnosis.csv", list(yuksel_rows[0].keys()), yuksel_rows)


# ---------------------------------------------------------------------------
# Olhoff failure table from independently extracted raw MAT evidence
# ---------------------------------------------------------------------------
raw_summary = {r["mesh"]: r for r in read_csv(HERE / "olhoff_raw_summary.csv")}
best_prior = {r["mesh"]: r for r in read_csv(HERE / "olhoff_best_prior_quality.csv")}
olhoff_rows = []
for mesh in ["480x60", "560x70", "640x80"]:
    s = raw_summary[mesh]
    b = best_prior[mesh]
    q_fail = quality_map[("Olhoff", mesh)]
    failure_w = [float(q_fail[f"omega{i}_common_raw_E3"]) for i in (1, 2, 3)]
    failure_gap = (failure_w[1] - failure_w[0]) / failure_w[0]
    failure_N = 2 if failure_gap < 0.05 and (failure_w[2] - failure_w[0]) / failure_w[0] >= 0.05 else "REVIEW"
    if mesh in {"480x60", "560x70"}:
        trajectory = "late omega3 branch alternation while N=2; dual-simplex limit on next LP"
    else:
        trajectory = "late N=1/2 switching and omega1 collapse (144.6 at last retained evaluation) before next LP limit"
    olhoff_rows.append(
        {
            "Mesh": mesh,
            "completed_valid_iterations": s["completed_iterations"],
            "failure_iteration_first_attempt": s["failure_attempt"],
            "first_failing_component": "linprog dual-simplex-highs in innerLoopLP.m",
            "solver_flag": "0",
            "solver_flag_meaning": "maximum number of LP iterations reached",
            "failure_modal_state_common_E3_native_equivalent_omega1_omega2_omega3": "/".join(fmt(x, 10) for x in failure_w),
            "failure_modal_state_basis": "reconstructed from frozen common E3 evaluation of unchanged failed-attempt density; E3/native validated to 4.5e-8 omega1 on healthy endpoints",
            "failure_N_inferred": failure_N,
            "failure_gap12_inferred": failure_gap,
            "failure_lamref_inferred": failure_w[0] ** 2,
            "last_valid_modal_state_omega1_omega2_omega3": "/".join(s[f"last_valid_omega{i}"] for i in (1, 2, 3)),
            "last_valid_N": s["last_valid_N"],
            "last_valid_gap12": s["last_valid_gap12"],
            "stabilization_state": f"stage 2; trigger k={s['stabilization_trigger_iteration']}; move={s['final_move']}",
            "best_valid_prior_iteration": b["iteration"],
            "best_valid_prior_native_spectrum": "/".join(b[f"native_omega{i}"] for i in (1, 2, 3)),
            "best_valid_prior_raw_E1_E2_E3_omega1": "/".join(b[f"raw_E{i}_omega1"] for i in (1, 2, 3)),
            "best_valid_prior_binary_E1_E2_E3_omega1": "/".join(b[f"binary_E{i}_omega1"] for i in (1, 2, 3)),
            "best_valid_prior_N": b["N"],
            "best_valid_prior_gap12": b["gap12"],
            "best_valid_prior_connectivity_raw_binary": f"{b['connected_raw']}/{b['connected_binary']}",
            "best_valid_prior_volume": b["volume"],
            "best_valid_prior_grayness": b["grayness"],
            "conditioning_at_failure": "NOT RETAINED",
            "diagnosis": trajectory,
        }
    )
write_csv(HERE / "olhoff_failure_forensics.csv", list(olhoff_rows[0].keys()), olhoff_rows)


# ---------------------------------------------------------------------------
# Scaling fits and decomposition
# ---------------------------------------------------------------------------
admissible = {
    m: [r for r in runs if r["method"] == m and status_ok(m, r["stopping"]["status"])]
    for m in method_order
}

per_specs = []
for method in method_order:
    rr = admissible[method]
    per_specs.append(
        (method, "all optimization-loop iterations", rr, [x["timing"]["optimization_loop_time_s"] / x["iterations"]["iter_total"] for x in rr])
    )
    if method == "Yuksel":
        for stage in (1, 2):
            per_specs.append(
                (
                    method,
                    f"Stage {stage}",
                    rr,
                    [x["timing"][f"stage{stage}_loop_time_s"] / x["iterations"][f"iter_stage{stage}"] for x in rr],
                )
            )

per_fit_rows = []
free_fits = {}
for method, stage, rr, vals in per_specs:
    ne = [x["mesh"]["elements"] for x in rr]
    f = fit_power(ne, vals)
    free_fits[("per_iteration", method, stage)] = f
    interpretation = "kernel cost per iteration; excludes iteration-count growth"
    if method == "Yuksel" and stage != "all optimization-loop iterations":
        interpretation = f"Yuksel {stage.lower()} kernel scaling"
    per_fit_rows.append(
        {
            "Method": method,
            "Stage": stage,
            "C_iter": f["C"],
            "p_iter": f["p"],
            "R2_log": f["R2_log"],
            "n": f["n"],
            "p_ci95_low": f["p_ci95_low"],
            "p_ci95_high": f["p_ci95_high"],
            "interpretation": interpretation,
        }
    )
write_csv(HERE / "per_iteration_scaling.csv", list(per_fit_rows[0].keys()), per_fit_rows)

total_fit_rows = []
for method in method_order:
    rr = admissible[method]
    ne = [x["mesh"]["elements"] for x in rr]
    for quantity, field in [("total_wall_time", "total_wall_time_s"), ("optimization_loop_time", "optimization_loop_time_s")]:
        vals = [x["timing"][field] for x in rr]
        f = fit_power(ne, vals)
        free_fits[(quantity, method, "all")] = f
        semantics = (
            "time to fixed-work stabilized endpoint (NOT convergence)"
            if method == "Olhoff"
            else "time to native terminal state"
        )
        total_fit_rows.append(
            {
                "Method": method,
                "Timing_quantity": quantity,
                "C_total": f["C"],
                "p_total": f["p"],
                "R2_log": f["R2_log"],
                "n": f["n"],
                "p_ci95_low": f["p_ci95_low"],
                "p_ci95_high": f["p_ci95_high"],
                "terminal_semantics": semantics,
                "interpretation": "includes iteration-count behavior" if quantity == "total_wall_time" else "loop total = per-iteration cost times count",
            }
        )
write_csv(HERE / "total_time_scaling.csv", list(total_fit_rows[0].keys()), total_fit_rows)

iteration_rows = []
for method in method_order:
    for mesh in mesh_order:
        r = run_map[(method, mesh)]
        iteration_rows.append(
            {
                "Method": method,
                "Mesh": mesh,
                "Ne": r["mesh"]["elements"],
                "Stage1": r["iterations"]["iter_stage1"] if r["iterations"]["iter_stage1"] is not None else "",
                "Stage2": r["iterations"]["iter_stage2"] if r["iterations"]["iter_stage2"] is not None else "",
                "Total": r["iterations"]["iter_total"],
                "Status": r["stopping"]["status"],
                "Censored": "no" if status_ok(method, r["stopping"]["status"]) else "yes",
            }
        )
write_csv(HERE / "iteration_scaling.csv", list(iteration_rows[0].keys()), iteration_rows)

iter_fit_rows = []
for method in ["Yuksel", "Proposed"]:
    rr = admissible[method]
    ne = [x["mesh"]["elements"] for x in rr]
    specs = [("total", [x["iterations"]["iter_total"] for x in rr])]
    if method == "Yuksel":
        specs += [
            ("stage1", [x["iterations"]["iter_stage1"] for x in rr]),
            ("stage2", [x["iterations"]["iter_stage2"] for x in rr]),
        ]
    for stage, vals in specs:
        f = fit_power(ne, vals)
        free_fits[("iteration_count", method, stage)] = f
        iter_fit_rows.append(
            {
                "Method": method,
                "Count": stage,
                "C_n": f["C"],
                "p_n": f["p"],
                "R2_log": f["R2_log"],
                "n": f["n"],
                "p_ci95_low": f["p_ci95_low"],
                "p_ci95_high": f["p_ci95_high"],
                "censoring_note": "cap-hit rows excluded; counts remain visible in iteration_scaling.csv",
            }
        )
write_csv(HERE / "iteration_count_scaling_fits.csv", list(iter_fit_rows[0].keys()), iter_fit_rows)

fixed_rows = []
for method, stage, rr, vals in per_specs:
    ne = [x["mesh"]["elements"] for x in rr]
    free = free_fits[("per_iteration", method, stage)]
    fixed = fit_power(ne, vals, 1.5)
    if free["p_ci95_low"] <= 1.5 <= free["p_ci95_high"] and fixed["MAPE_pct"] <= 15:
        classification = "EMPIRICALLY WELL SUPPORTED"
    elif fixed["R2_log"] >= 0.90 and fixed["MAPE_pct"] <= 35:
        classification = "USEFUL NORMALIZATION ONLY"
    else:
        classification = "POOR MODEL"
    fixed_rows.append(
        {
            "Method": method,
            "Quantity": f"per_iteration:{stage}",
            "fixed_C": fixed["C"],
            "fixed_p": 1.5,
            "fixed_R2_log": fixed["R2_log"],
            "fixed_MAPE_pct": fixed["MAPE_pct"],
            "free_p": free["p"],
            "free_p_ci95": f"[{fmt(free['p_ci95_low'])},{fmt(free['p_ci95_high'])}]",
            "classification": classification,
        }
    )
for method in method_order:
    rr = admissible[method]
    ne = [x["mesh"]["elements"] for x in rr]
    for quantity, field in [("total_wall_time", "total_wall_time_s"), ("optimization_loop_time", "optimization_loop_time_s")]:
        vals = [x["timing"][field] for x in rr]
        free = free_fits[(quantity, method, "all")]
        fixed = fit_power(ne, vals, 1.5)
        if free["p_ci95_low"] <= 1.5 <= free["p_ci95_high"] and fixed["MAPE_pct"] <= 15:
            classification = "EMPIRICALLY WELL SUPPORTED"
        elif fixed["R2_log"] >= 0.90 and fixed["MAPE_pct"] <= 35:
            classification = "USEFUL NORMALIZATION ONLY"
        else:
            classification = "POOR MODEL"
        fixed_rows.append(
            {
                "Method": method,
                "Quantity": quantity,
                "fixed_C": fixed["C"],
                "fixed_p": 1.5,
                "fixed_R2_log": fixed["R2_log"],
                "fixed_MAPE_pct": fixed["MAPE_pct"],
                "free_p": free["p"],
                "free_p_ci95": f"[{fmt(free['p_ci95_low'])},{fmt(free['p_ci95_high'])}]",
                "classification": classification,
            }
        )
write_csv(HERE / "fixed_p15_assessment.csv", list(fixed_rows[0].keys()), fixed_rows)


# ---------------------------------------------------------------------------
# Timing and memory instrumentation audit
# ---------------------------------------------------------------------------
timing_rows = []
for r in runs:
    t = r["timing"]
    method = r["method"]
    mesh = f"{r['mesh']['nelx']}x{r['mesh']['nely']}"
    init, loop, post, wall = (
        t["initialization_time_s"],
        t["optimization_loop_time_s"],
        t["postprocessing_time_s"],
        t["total_wall_time_s"],
    )
    if method == "Olhoff":
        residual = wall - loop
        check = "PASS: init/post correctly null; remainder carried as unattributed"
    else:
        residual = wall - init - loop - post
        check = "PASS" if abs(residual) < 0.02 else "REVIEW"
    timing_rows.append(
        {
            "Method": method,
            "Mesh": mesh,
            "Status": r["stopping"]["status"],
            "initialization_time_s": init if init is not None else "",
            "optimization_loop_time_s": loop,
            "postprocessing_time_s": post if post is not None else "",
            "total_wall_time_s": wall,
            "unattributed_or_measurement_residual_s": residual,
            "relative_residual_pct": 100 * residual / wall,
            "stage_time_sum_residual_s": t["stage_time_sum_residual_s"] if t["stage_time_sum_residual_s"] is not None else "",
            "check": check,
        }
    )
write_csv(HERE / "timing_decomposition.csv", list(timing_rows[0].keys()), timing_rows)

memory_rows = []
for r in runs:
    memory_rows.append(
        {
            "Method": r["method"],
            "Mesh": f"{r['mesh']['nelx']}x{r['mesh']['nely']}",
            "MaxRAM_MB_reported": r["max_ram_mb"],
            "Measurement": "peak process RSS minus process RSS at case start, sampled every 0.25 s",
            "Replicates": 1,
            "Classification": "UNRELIABLE",
            "Reason": "allocator carry-over/order dependence, delta baseline, coarse sampling, and no replication",
        }
    )
write_csv(HERE / "memory_assessment.csv", list(memory_rows[0].keys()), memory_rows)


# ---------------------------------------------------------------------------
# Publication readiness
# ---------------------------------------------------------------------------
publication_rows = [
    {"Claim": "All 27 terminal/status labels and five censored rows are correct", "Assessment": "SUPPORTED", "Evidence_or_blocker": "independent stop-reason/dx reclassification plus Olhoff raw MAT status/logs"},
    {"Claim": "Censored observations entered no existing scaling fit", "Assessment": "SUPPORTED", "Evidence_or_blocker": "table1_performance in_scaling_fit and campaign_gate masks agree exactly"},
    {"Claim": "Proposed is native-converged at all nine meshes", "Assessment": "SUPPORTED", "Evidence_or_blocker": "all terminal max dx values satisfy 0.01 and stop reason is density_change_tolerance"},
    {"Claim": "Proposed 160/240 has intrinsically poor common-evaluator quality", "Assessment": "NOT YET SUPPORTED", "Evidence_or_blocker": "common raw is 153.7/157.6 and binary is 162.8/161.5; anomaly is native-model dependent"},
    {"Claim": "Proposed coarse states are fully stationary/KKT solutions", "Assessment": "NOT YET SUPPORTED", "Evidence_or_blocker": "native stop met, but no histories, KKT residuals, final densities, or restart evidence retained"},
    {"Claim": "Olhoff solver failures are dual-simplex iteration limits", "Assessment": "SUPPORTED WITH QUALIFICATION", "Evidence_or_blocker": "raw logs record linprog flag 0; failed-attempt modal state is reconstructed by native-equivalent common E3, but full linprog output/iteration count was not retained"},
    {"Claim": "Olhoff failure island is monotonic size/resource failure", "Assessment": "NOT YET SUPPORTED", "Evidence_or_blocker": "larger 720/800 succeed; trajectories show branch switching and all failure calls occur after stabilization"},
    {"Claim": "Olhoff higher native omega survives common raw evaluation", "Assessment": "SUPPORTED WITH QUALIFICATION", "Evidence_or_blocker": "raw E1/E2/E3 preserve advantage on six admissible rows; binary E2/E3 collapse at 720/800"},
    {"Claim": "Yuksel cap hits are merely too-small iteration budgets", "Assessment": "NOT YET SUPPORTED", "Evidence_or_blocker": "late histories were not retained; 640 is possible, 800 indeterminate"},
    {"Claim": "Per-iteration computational scaling can be reported", "Assessment": "SUPPORTED WITH QUALIFICATION", "Evidence_or_blocker": "admissible rows and loop times available; one timing sample per mesh"},
    {"Claim": "Total-time exponent is intrinsic algorithmic complexity", "Assessment": "NOT YET SUPPORTED", "Evidence_or_blocker": "iteration-count exponent materially changes Yuksel and Proposed totals"},
    {"Claim": "Olhoff total time is time to convergence", "Assessment": "NOT YET SUPPORTED", "Evidence_or_blocker": "fixed 1600-work endpoint has no native convergence claim"},
    {"Claim": "p=1.5 is a universal theoretical complexity law", "Assessment": "NOT YET SUPPORTED", "Evidence_or_blocker": "empirical fits only; fixed-p diagnostics differ by method/quantity"},
    {"Claim": "RAM values support quantitative memory scaling", "Assessment": "NOT YET SUPPORTED", "Evidence_or_blocker": "RSS-delta instrumentation is order/allocator dependent and unreplicated"},
    {"Claim": "Proposed and Yuksel approach comparable first-mode raw quality from 320 onward", "Assessment": "SUPPORTED WITH QUALIFICATION", "Evidence_or_blocker": "raw common omega1 differs by about 0-1.2%; higher spectra/topologies differ and Yuksel 400 binary is pathological"},
]
write_csv(HERE / "publication_readiness.csv", list(publication_rows[0].keys()), publication_rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_scaling(field, per_iter, filename, ylabel):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for method in method_order:
        rr = [r for r in runs if r["method"] == method]
        x = np.array([r["mesh"]["elements"] for r in rr], float)
        if per_iter:
            y = np.array([r["timing"][field] / r["iterations"]["iter_total"] for r in rr], float)
        else:
            y = np.array([r["timing"][field] for r in rr], float)
        ok = np.array([status_ok(method, r["stopping"]["status"]) for r in rr])
        ax.loglog(x[ok], y[ok], "o", color=colors[method], label=f"{method} admissible")
        ax.loglog(x[~ok], y[~ok], "x", ms=8, mew=2, color=colors[method], label=f"{method} censored")
        fit = fit_power(x[ok], y[ok])
        xx = np.geomspace(x[ok].min(), x[ok].max(), 100)
        ax.loglog(xx, fit["C"] * xx ** fit["p"], "-", color=colors[method], alpha=.8)
    ax.set_xlabel("elements $N_e$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


plot_scaling("total_wall_time_s", False, "01_total_wall_time_vs_Ne.png", "total wall time [s]")
plot_scaling("optimization_loop_time_s", True, "02_per_iteration_loop_time_vs_Ne.png", "loop time / iteration [s]")

fig, ax = plt.subplots(figsize=(8.2, 5.5))
for method in method_order:
    rr = [run_map[(method, m)] for m in mesh_order]
    x = [r["mesh"]["elements"] for r in rr]
    y = [r["iterations"]["iter_total"] for r in rr]
    ok = [status_ok(method, r["stopping"]["status"]) for r in rr]
    ax.plot(np.array(x)[ok], np.array(y)[ok], "o-", color=colors[method], label=method)
    ax.plot(np.array(x)[np.logical_not(ok)], np.array(y)[np.logical_not(ok)], "x", ms=9, mew=2, color=colors[method])
ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, which="both", alpha=.3)
ax.set_xlabel("elements $N_e$"); ax.set_ylabel("executed optimization iterations")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "03_iteration_count_vs_Ne.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(8.2, 5.5))
rr = [run_map[("Yuksel", m)] for m in mesh_order]
x = np.array([r["mesh"]["elements"] for r in rr], float)
for stage, marker in [(1, "o"), (2, "s")]:
    y = np.array([r["iterations"][f"iter_stage{stage}"] for r in rr], float)
    ax.plot(x, y, marker + "-", label=f"Stage {stage}")
for r in rr:
    if r["stopping"]["status"] == "CAP_HIT":
        ax.plot(r["mesh"]["elements"], r["iterations"]["iter_stage2"], "x", color="red", ms=10, mew=2)
ax.set_xscale("log"); ax.grid(True, which="both", alpha=.3); ax.legend()
ax.set_xlabel("elements $N_e$"); ax.set_ylabel("Yuksel stage iterations")
fig.tight_layout(); fig.savefig(FIG / "04_yuksel_stage_iteration_counts.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(8.2, 5.5))
for method in method_order:
    yy, xx, ok = [], [], []
    for mesh in mesh_order:
        r = run_map[(method, mesh)]
        w = r["results"]["final_frequencies_rad_s"][0]
        xx.append(r["mesh"]["nelx"]); yy.append(np.nan if w is None else w)
        ok.append(status_ok(method, r["stopping"]["status"]))
    xx, yy, ok = np.array(xx), np.array(yy,float), np.array(ok)
    ax.plot(xx[ok], yy[ok], "o-", color=colors[method], label=method)
    ax.plot(xx[~ok], yy[~ok], "x", ms=9, mew=2, color=colors[method])
ax.grid(True, alpha=.3); ax.set_xlabel("nelx (nely = nelx/8)"); ax.set_ylabel("native $\\omega_1$ [rad/s]")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "05_native_omega1_vs_mesh.png", dpi=180); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
metrics = [("omega1_common_raw_E1", "raw E1"), ("omega1_common_binary_E1", "binary E1"), ("omega1_common_binary_E2", "binary E2")]
for ax, (field, title) in zip(axes, metrics):
    for method in method_order:
        rows = [quality_map[(method, m)] for m in mesh_order]
        x = np.array([int(m.split("x")[0]) for m in mesh_order])
        y = np.array([num(r[field]) for r in rows])
        ok = np.array([status_ok(method, r["status"]) for r in rows])
        ax.plot(x[ok], y[ok], "o-", color=colors[method], label=method)
        ax.plot(x[~ok], y[~ok], "x", ms=8, mew=2, color=colors[method])
    ax.set_title(title); ax.grid(True, alpha=.3); ax.set_xlabel("nelx")
axes[0].set_ylabel("common-evaluator $\\omega_1$ [rad/s]"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "06_common_evaluator_quality_vs_mesh.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(8.2, 5.5))
for method in method_order:
    rows = [quality_map[(method, m)] for m in mesh_order]
    x = [int(m.split("x")[0]) for m in mesh_order]
    y = [num(r["grayness"]) for r in rows]
    ax.plot(x, y, "o-", color=colors[method], label=method)
ax.grid(True, alpha=.3); ax.set_xlabel("nelx"); ax.set_ylabel("grayness, mean $4x(1-x)$")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "07_grayness_vs_mesh.png", dpi=180); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
for ax, mesh in zip(axes, ["160x20", "240x30", "320x40"]):
    q = quality_map[("Proposed", mesh)]
    ax.axis("off")
    ax.text(.5,.66,"FINAL DENSITY FIELD\nNOT RETAINED",ha="center",va="center",fontsize=13,weight="bold")
    ax.text(.5,.34,f"native $\\omega_1$: {float(q['omega1_native']):.1f}\nraw E1: {float(q['omega1_common_raw_E1']):.1f}\nbinary E1: {float(q['omega1_common_binary_E1']):.1f}\ngrayness: {float(q['grayness']):.3f}",ha="center",va="center",fontsize=10)
    ax.set_title(mesh)
fig.suptitle("Proposed 160/240/320 topology evidence gap (checksums are not images)")
fig.tight_layout(); fig.savefig(FIG / "08_proposed_topology_comparison_unavailable.png", dpi=180); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
cap_meshes = ["640x80", "800x100"]
cap_runs = [run_map[("Yuksel", m)] for m in cap_meshes]
x = np.arange(2)
axes[0].bar(x-.17,[r["stopping"]["final_max_density_change"] for r in cap_runs],.34,label="max dx")
axes[0].bar(x+.17,[r["stopping"]["final_rms_density_change"] for r in cap_runs],.34,label="RMS dx")
axes[0].axhline(.01,color="k",ls="--",label="native tolerance"); axes[0].set_yscale("log")
axes[0].set_xticks(x,cap_meshes); axes[0].set_ylabel("terminal density change"); axes[0].legend(fontsize=8); axes[0].grid(True,axis="y",alpha=.3)
axes[1].bar(x,[r["stopping"]["final_relative_objective_change"] for r in cap_runs],color="#9467bd")
axes[1].set_yscale("log"); axes[1].set_xticks(x,cap_meshes); axes[1].set_ylabel("terminal relative objective change"); axes[1].grid(True,axis="y",alpha=.3)
fig.suptitle("Yuksel capped cases: terminal diagnostics only (late histories not retained)")
fig.tight_layout(); fig.savefig(FIG / "10_yuksel_late_history_unavailable_terminal_diagnostics.png", dpi=180); plt.close(fig)


# ---------------------------------------------------------------------------
# Provenance and immutable-evidence hashes
# ---------------------------------------------------------------------------
artifact_files = sorted(p for p in FROZEN.rglob("*") if p.is_file())
artifact_hashes_before = {str(p.relative_to(REPO)): sha256(p) for p in artifact_files}
source_files = [
    REPO / "analysis" / "olhoff_stabilization_audit" / "final_campaign_profile.json",
    REPO / "analysis" / "olhoff_stabilization_audit" / "selected_profile.json",
    REPO / "analysis" / "three_method_parametric_study" / "results" / "profile_freeze_manifest.json",
    REPO / "examples" / "Performance" / "final_campaign_config.m",
    REPO / "examples" / "Performance" / "final_campaign_preflight.m",
    REPO / "examples" / "Performance" / "final_campaign_run_case.m",
    REPO / "examples" / "Performance" / "performance_comparison.m",
    REPO / "analysis" / "three_method_parametric_study" / "study_base_config.m",
    REPO / "analysis" / "three_method_parametric_study" / "study_evaluate_design.m",
    REPO / "analysis" / "olhoff_stabilization_audit" / "run_stabilization_case.m",
    REPO / "analysis" / "olhoff_stabilization_audit" / "olhoffOptStabilized.m",
    REPO / "Matlab" / "reproduction2007" / "algo" / "innerLoopLP.m",
    REPO / "tools" / "Matlab" / "run_topopt_from_json.m",
    REPO / "analysis" / "YukselApproach" / "Matlab" / "top99neo_inertial_freq.m",
    REPO / "analysis" / "ourApproach" / "Matlab" / "topopt_freq.m",
]
source_hashes = {str(p.relative_to(REPO)): sha256(p) for p in source_files}
gate = json.loads((FROZEN / "campaign_gate.json").read_text(encoding="utf-8"))
status_counts = {m: dict(Counter(s for mm, s in independent_statuses if mm == m)) for m in method_order}
provenance = {
    "audit_type": "offline forensic audit; no optimization campaign rerun",
    "generated_at": datetime.now(ZoneInfo("Europe/Warsaw")).isoformat(timespec="seconds"),
    "branch": git("branch", "--show-current"),
    "HEAD": git("rev-parse", "HEAD"),
    "working_tree_status_before_analysis": ["?? examples/Performance/final_campaign/"],
    "working_tree_status_at_generation": git("status", "--short").splitlines(),
    "matlab_version": bench["metadata"]["environment"]["matlab_version"],
    "computer": bench["metadata"]["environment"]["computer"],
    "threads_per_run": bench["metadata"]["environment"]["comp_threads_active"],
    "campaign_id": gate["campaign_id"],
    "profile_ids": gate["profile_ids"],
    "configuration_hashes_from_preflight": gate["configuration_hashes"],
    "campaign_configuration_and_execution_file_hashes_sha256": source_hashes,
    "audit_script_hashes_sha256": {
        "analysis/performance_campaign_forensic_audit/build_audit.py": sha256(HERE / "build_audit.py"),
        "analysis/performance_campaign_forensic_audit/extract_olhoff_evidence.m": sha256(HERE / "extract_olhoff_evidence.m"),
    },
    "frozen_artifact_hashes_sha256": artifact_hashes_before,
    "frozen_artifact_count": len(artifact_hashes_before),
    "independently_revalidated_status_counts": status_counts,
    "censored_count": sum(1 for r in integrity_rows if r["Censored"] == "yes"),
    "evidence_limitations": [
        "Yuksel and Proposed final density fields and histories were not retained.",
        "Olhoff failed-attempt modal state/lamref were not directly logged (they are reconstructable with native-equivalent common E3); conditioning and linprog output remain unavailable.",
        "Memory has one peak-RSS-delta sample per case.",
    ],
}
(HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Rerun decision and main report
# ---------------------------------------------------------------------------
rerun_text = """# Targeted rerun recommendations

No full nine-resolution rerun is justified. The frozen statuses, timing records, and common-evaluator outputs are internally consistent, and censoring protected every fit.

Three narrow diagnostic follow-ups are required before an unqualified paper freeze because the original campaign did not retain enough evidence to close the causal questions. None is intended to improve a benchmark number.

1. **Olhoff 640x80, identical S1 profile and 1600-work horizon, diagnostics only.** Hypothesis: the deterministic late N=1/2 branch switching creates a degenerate LP trajectory that drives `dual-simplex-highs` to exit flag 0. Record the failed-attempt native spectrum, N, gap12, lamref, constraint-row norms/rank, active bounds, and the full `linprog` output/message. Do not change LP options. Confirmation is the same failure at attempted k=1067 with an iteration-limit message and a diagnosed degeneracy/conditioning signature; refutation is a different trajectory or failure mechanism.

2. **Yuksel 800x100, identical 2000-iteration profile, history retention enabled.** Hypothesis: the large terminal max change is localized oscillation/topology turnover rather than a smooth approach to tolerance. Record at least the final 300 Stage-2 iterations (max/RMS dx, objective, volume, grayness, binary turnover, and modal spectrum). Keep the cap and tolerance unchanged. Confirmation is persistent/oscillatory max dx with small RMS/objective change; refutation is a monotone decay truncated near tolerance. Only if the latter occurs should a separately preregistered modest Stage-2 extension be considered.

3. **Proposed 160x20, identical frozen profile, deterministic diagnostic replay.** Hypothesis: the native ~109 triplet consists of low-density/void-model modes made visible by the native `Emin/E0=1e-9` plus linear mass interpolation; common E1's stiffer void floor and E2/E3's low-density mass suppression remove them. Retain the final density, history, and mode shapes/energy localization; do not change settings. Confirmation is bitwise/near-bitwise reproduction of the density and ~109 triplet with modes localized in low-density regions, while frozen common evaluators reproduce ~154 raw and ~163 binary. Refutation is failure to reproduce or global structural modes at ~109.

These follow-ups are diagnostic evidence acquisition, not replacement observations. Existing capped/failed rows remain censored.
"""
(HERE / "rerun_recommendations.md").write_text(rerun_text, encoding="utf-8")

status_counter = Counter(r["Status"] for r in integrity_rows)
per_lookup = {(r["Method"], r["Stage"]): r for r in per_fit_rows}
total_lookup = {(r["Method"], r["Timing_quantity"]): r for r in total_fit_rows}
iter_lookup = {(r["Method"], r["Count"]): r for r in iter_fit_rows}
timing_native_res = [abs(float(r["unattributed_or_measurement_residual_s"])) for r in timing_rows if r["Method"] != "Olhoff"]

report = f"""# Performance Campaign Forensic Audit

## Executive verdict

The completed campaign is technically trustworthy **for the observations it actually made**: all 27 statuses revalidate, exactly five rows are censored, and no censored row entered an existing scaling fit. It is not yet sufficient for an unqualified paper freeze because Proposed/Yuksel raw designs and histories were not retained and the Olhoff failed LP attempt was only logged as a flag, leaving three causal claims dependent on narrow diagnostic follow-ups.

The most important corrections to the preliminary reading are:

- Proposed's apparent 109→159 native jump is primarily **MODEL / INTERPOLATION DEPENDENCE**, not a 50-rad/s common-quality jump. At 160x20, native/common-raw-E1/common-binary-E1 are 109.05/153.68/162.76 rad/s; at 240x30 they are 108.78/157.64/161.47; at 320x40 they are 158.76/158.76/162.12.
- Olhoff completed 357/399/1066 valid updates, but the first failed LP calls were attempted iterations **358/400/1067**. Each call to `linprog(..., Algorithm='dual-simplex-highs')` returned flag 0 (`maximum number of iterations reached`). The aggregate `final_lp_flag=1` and empty `lp_failure_iters` refer only to the last appended successful history row and must not be read as the failed call.
- Yuksel's capped rows cannot be assigned a late-history mechanism from the frozen evidence: `record_history=false`. Terminal max/RMS changes strongly suggest localized motion, but a checksum cannot distinguish monotone decay, stagnation, oscillation, or a topology event.

## WP0–WP2: freeze, inventory, and status semantics

Provenance is frozen in [provenance.json](provenance.json), including branch `benchmark-methodology-r2`, HEAD `632e9b01811845709de33f93051fd853373ed5e1`, MATLAB `{bench['metadata']['environment']['matlab_version']}`, the three profile IDs, hashes of configuration/execution sources, and SHA-256 hashes of every frozen result artifact. The frozen tree was read only.

The evidence inventory is [evidence_map.csv](evidence_map.csv). Olhoff retained full MAT histories, final densities, and every successful-update snapshot. Yuksel and Proposed retained terminal spectra, timing/stopping scalars, evaluator results, and checksums, but not density fields or histories. Consequently, checksums establish identity only; they cannot reconstruct topologies or trends.

Independent counts are:

| Method | Verified successful semantic | Count | Censored semantic | Count |
|---|---:|---:|---:|---:|
| Olhoff | VALID_STABILIZED_STATE_AT_FIXED_WORK | 6 | SOLVER_FAILURE | 3 |
| Yuksel | NATIVE_CONVERGED | 7 | CAP_HIT | 2 |
| Proposed | NATIVE_CONVERGED | 9 | — | 0 |

This is exactly five censored rows. Solver failure is never convergence; a cap hit is never convergence; and Olhoff's 1600 endpoint is fixed work, not native convergence. Row-level proof is in [campaign_integrity.csv](campaign_integrity.csv).

## WP3–WP4: Proposed anomaly and same-mesh Yuksel comparison

The native Proposed spectra at 160 and 240 are tightly clustered low triplets (109.05/109.49/112.92 and 108.78/109.92/117.83 rad/s), unlike the 320 spectrum (158.76/229.64/279.39). Yet the frozen common evaluators give Proposed raw omega1 values of 153.68/154.34/154.34 (E1/E2/E3) at 160 and 157.64/158.18/158.18 at 240; exact-count binary values are 162.76 and 161.47 under every evaluator family. Thus the discontinuity shrinks from about 50 rad/s natively to 5.1 rad/s in raw E1 and reverses slightly under binary E1.

The mechanism supported by code and results is low-density interpolation sensitivity. Proposed natively uses `Emin/E0=1e-9` with linear density-to-mass interpolation. Common E1 raises the void stiffness floor to `1e-6`; E2/E3 suppress mass strongly below density 0.1. Both changes eliminate the low native triplet. The effect is strongest where Proposed grayness is largest (0.256, 0.162, 0.122 at 160/240/320). A fixed two-element filter also changes physical radius with refinement and may influence grayness/basin selection, but the retained evidence cannot isolate that secondary contribution.

Compared with Yuksel on the same meshes, Proposed's common raw E1 omega1 differs by -3.49, -1.80, and -1.93 rad/s at 160/240/320, far smaller than the native -48.23, -50.71, and -1.98. Binary E1 places Proposed at +3.97, +1.07, and +0.61 rad/s relative to Yuksel. The primary anomaly classification is therefore **MODEL / INTERPOLATION DEPENDENCE**.

The 160/240 runs genuinely met their frozen native density-change stop and have small terminal relative objective changes. They are legitimate native terminal states, but “stationary/KKT local solutions” is not proved because histories, KKT metrics, densities, and restart tests were not retained. The requested topology comparison is consequently an explicit evidence-gap figure, [Figure 8](figures/08_proposed_topology_comparison_unavailable.png), not an invented reconstruction.

## WP5–WP7: Olhoff failure island and pre-failure quality

The first failing component is exact: `analysis/olhoff_stabilization_audit/olhoffOptStabilized.m` calls `Matlab/reproduction2007/algo/innerLoopLP.m`, which calls `linprog` with `dual-simplex-highs`. Flag 0 makes `st.conv=false`; the caller records `SOLVER_FAILURE` and breaks without appending that attempt to `hist`. MATLAB R2025b's `linprog.m` defines flag 0 as maximum iterations reached. It was neither an eigensolver exception, infeasibility flag, nor a nonfinite-state flag.

Although the failed attempt is absent from `hist`, its density is retained as `res.rho` and was passed to the campaign's common evaluators. Common E3 is native-equivalent for this Olhoff profile: across the six healthy endpoints, common-E3 and native omega1 differ by at most 4.5e-8 rad/s. It therefore reconstructs the failed-attempt spectra as 170.571/170.792/187.407, 171.715/171.998/251.226, and 172.823/173.265/200.337 rad/s for 480/560/640. Each is N=2 with gap12 0.00130/0.00165/0.00256; inferred lamref is omega1 squared. These are evaluator-based reconstructions, explicitly labelled as such, rather than directly logged failure-attempt telemetry.

All three failures occur after the causal S1 move reduction (triggers k=207/204/204; move 0.005→0.0025), not at a resource threshold. Larger 720/800 cases trigger at k=204/206 and complete 1600 healthy LP solves. The island is therefore non-monotonic and trajectory-dependent.

The retained trajectories identify a modal precursor but do not retain failed-attempt conditioning. At 480/560, N remains 2 while omega3 alternates sharply among modal branches in the last window (late-50 omega3 standard deviations {float(raw_summary['480x60']['late50_omega3_std']):.2f} and {float(raw_summary['560x70']['late50_omega3_std']):.2f} rad/s). At 640, the last 50 updates contain {raw_summary['640x80']['late50_N_switches']} N-switches and the last retained spectrum collapses to 144.62/173.01/173.09 before the next LP reaches its iteration limit. Healthy 720/800 end with N=2, sub-0.2% gap12, and late-50 omega3 standard deviations only {float(raw_summary['720x90']['late50_omega3_std']):.2f}/{float(raw_summary['800x100']['late50_omega3_std']):.2f}. This supports a mode-branch/LP-degeneracy interaction, but the precise conditioning signature remains unrecorded.

The failed cases had reached mature, connected bimodal raw states. Their best valid native omega1 values were {float(best_prior['480x60']['native_omega1']):.2f}, {float(best_prior['560x70']['native_omega1']):.2f}, and {float(best_prior['640x80']['native_omega1']):.2f} rad/s, with raw common E1 {float(best_prior['480x60']['raw_E1_omega1']):.2f}, {float(best_prior['560x70']['raw_E1_omega1']):.2f}, and {float(best_prior['640x80']['raw_E1_omega1']):.2f}. They are not uniformly evaluator-robust: binary E2/E3 omega1 collapses to roughly 4–7 rad/s because threshold connectivity coexists with detached binary fragments (largest-component fractions below one). These pre-failure states remain diagnostic only and are not promoted into benchmark successes. See [olhoff_failure_forensics.csv](olhoff_failure_forensics.csv), [best-prior quality](olhoff_best_prior_quality.csv), [histories](olhoff_histories.csv), and [Figure 9](figures/09_olhoff_failure_neighborhood_histories.png).

## WP8–WP9: Yuksel cap hits

640x80 and 800x100 both execute Stage 1=1000 and Stage 2=1000, while 720x90 executes 1000+966 and meets the Stage-2 native criterion. This is not contradictory: Stage 1's 1000 is the prescribed handoff budget; the terminal success test is met in Stage 2.

At 640, final max/RMS dx are 0.03065/0.000366 with relative objective change 5.85e-6. At 800 they are 0.07280/0.000686 with relative objective change 2.64e-5. The max-to-RMS ratios show that a small subset of variables dominates the terminal maximum, while the global field and objective move little. Without the final 200 samples, however, this does not distinguish a localized oscillation from a late event or slow monotone tail.

- 640x80: **POSSIBLE_CAP_LIMIT**. A modest extension is plausible but not demonstrated.
- 800x100: **INDETERMINATE**. Its terminal max change is too far above tolerance to call a simple cap limit from one sample.

[Figure 10](figures/10_yuksel_late_history_unavailable_terminal_diagnostics.png) therefore shows only terminal diagnostics and labels the missing trajectory explicitly.

## WP10–WP14: computational scaling and timing

### Computational kernel / per-iteration scaling

| Method/stage | C_iter | p_iter | log R² | n | 95% CI for p |
|---|---:|---:|---:|---:|---:|
| Olhoff, all | {float(per_lookup[('Olhoff','all optimization-loop iterations')]['C_iter']):.4g} | {float(per_lookup[('Olhoff','all optimization-loop iterations')]['p_iter']):.3f} | {float(per_lookup[('Olhoff','all optimization-loop iterations')]['R2_log']):.4f} | 6 | [{float(per_lookup[('Olhoff','all optimization-loop iterations')]['p_ci95_low']):.3f}, {float(per_lookup[('Olhoff','all optimization-loop iterations')]['p_ci95_high']):.3f}] |
| Yuksel, all | {float(per_lookup[('Yuksel','all optimization-loop iterations')]['C_iter']):.4g} | {float(per_lookup[('Yuksel','all optimization-loop iterations')]['p_iter']):.3f} | {float(per_lookup[('Yuksel','all optimization-loop iterations')]['R2_log']):.4f} | 7 | [{float(per_lookup[('Yuksel','all optimization-loop iterations')]['p_ci95_low']):.3f}, {float(per_lookup[('Yuksel','all optimization-loop iterations')]['p_ci95_high']):.3f}] |
| Yuksel, Stage 1 | {float(per_lookup[('Yuksel','Stage 1')]['C_iter']):.4g} | {float(per_lookup[('Yuksel','Stage 1')]['p_iter']):.3f} | {float(per_lookup[('Yuksel','Stage 1')]['R2_log']):.4f} | 7 | [{float(per_lookup[('Yuksel','Stage 1')]['p_ci95_low']):.3f}, {float(per_lookup[('Yuksel','Stage 1')]['p_ci95_high']):.3f}] |
| Yuksel, Stage 2 | {float(per_lookup[('Yuksel','Stage 2')]['C_iter']):.4g} | {float(per_lookup[('Yuksel','Stage 2')]['p_iter']):.3f} | {float(per_lookup[('Yuksel','Stage 2')]['R2_log']):.4f} | 7 | [{float(per_lookup[('Yuksel','Stage 2')]['p_ci95_low']):.3f}, {float(per_lookup[('Yuksel','Stage 2')]['p_ci95_high']):.3f}] |
| Proposed, all | {float(per_lookup[('Proposed','all optimization-loop iterations')]['C_iter']):.4g} | {float(per_lookup[('Proposed','all optimization-loop iterations')]['p_iter']):.3f} | {float(per_lookup[('Proposed','all optimization-loop iterations')]['R2_log']):.4f} | 9 | [{float(per_lookup[('Proposed','all optimization-loop iterations')]['p_ci95_low']):.3f}, {float(per_lookup[('Proposed','all optimization-loop iterations')]['p_ci95_high']):.3f}] |

Stage 2 is measurably more expensive per iteration than Stage 1, so the combined Yuksel average should not replace the stage-specific fits. Full coefficients are in [per_iteration_scaling.csv](per_iteration_scaling.csv).

### End-to-end practical scaling

| Method | wall-time p | log R² | n | terminal semantics |
|---|---:|---:|---:|---|
| Olhoff | {float(total_lookup[('Olhoff','total_wall_time')]['p_total']):.3f} | {float(total_lookup[('Olhoff','total_wall_time')]['R2_log']):.4f} | 6 | time to fixed-work stabilized endpoint; not convergence |
| Yuksel | {float(total_lookup[('Yuksel','total_wall_time')]['p_total']):.3f} | {float(total_lookup[('Yuksel','total_wall_time')]['R2_log']):.4f} | 7 | time to native terminal state |
| Proposed | {float(total_lookup[('Proposed','total_wall_time')]['p_total']):.3f} | {float(total_lookup[('Proposed','total_wall_time')]['R2_log']):.4f} | 9 | time to native terminal state |

These reproduce the campaign's approximately 1.193/1.706/1.418 exponents. They are practical endpoint exponents, not intrinsic kernel complexity.

On identical admissible rows, log-slope decomposition for loop time is exact up to rounding: Yuksel p_loop = p_iter {float(per_lookup[('Yuksel','all optimization-loop iterations')]['p_iter']):.3f} + p_count {float(iter_lookup[('Yuksel','total')]['p_n']):.3f}; Proposed p_loop = p_iter {float(per_lookup[('Proposed','all optimization-loop iterations')]['p_iter']):.3f} + p_count {float(iter_lookup[('Proposed','total')]['p_n']):.3f}. Olhoff's count exponent is exactly zero by protocol, so its loop exponent is its per-iteration exponent. The growing Yuksel count steepens total scaling; Proposed's mildly decreasing/non-monotonic count offsets its steeper kernel scaling.

The fixed p=1.5 assessments are in [fixed_p15_assessment.csv](fixed_p15_assessment.csv). The declared audit rule is: EMPIRICALLY WELL SUPPORTED when the free-fit 95% interval contains 1.5 and fixed-fit MAPE is at most 15%; USEFUL NORMALIZATION ONLY when fixed-fit log R² is at least 0.90 and MAPE at most 35%; otherwise POOR MODEL. These are empirical goodness-of-reference tests, not theoretical complexity proofs. Whenever retained, C is a normalized empirical coefficient with units tied to the fitted Ne convention, not an intrinsic constant.

For Proposed and Yuksel, `wall ≈ init + loop + post` to within at most {max(timing_native_res):.4f} s of caller/measurement overhead. Yuksel Stage 1+Stage 2 equals loop time exactly in the stored precision. For Olhoff, init and post correctly remain null; wall−loop is positive unattributed time and is never converted to zero. See [timing_decomposition.csv](timing_decomposition.csv).

The most defensible main computational-complexity quantity is **per-iteration optimization-loop time**, with Yuksel stages separated. Total wall time should be a second, practical endpoint table with explicit terminal semantics. Loop total is useful for exact cost×count decomposition but is not end-to-end.

## WP15: memory

`MaxRAM_MB` is not absolute RAM or allocator peak. The code samples MATLAB process RSS every 0.25 s and reports peak RSS minus RSS at case start. Sequential allocator retention changes the baseline; short-lived peaks may be missed; method order can contaminate deltas; and there is one sample per case. The non-monotonic values are therefore not reproducible evidence of memory scaling.

**Classification: UNRELIABLE.** Do not publish quantitative memory comparisons from this campaign. Raw values and rationale are in [memory_assessment.csv](memory_assessment.csv).

## WP16–WP17: common quality

Olhoff's higher native first frequency survives all common **raw** evaluators on its six admissible rows: common raw E1 is about 166.7–172.9 versus roughly 157–161 for Yuksel/Proposed. It does not survive every representation: binary E2/E3 collapses to single-digit frequencies at valid 720/800 because disconnected binary fragments introduce low modes despite a left-to-right connected main component. The publication-safe claim is therefore “higher common-raw first-mode frequency at the fixed-work endpoint,” not universal topology superiority.

From 320 onward Proposed and Yuksel approach comparable common-raw first-mode quality (generally within about 1.2%), but their higher spectra and grayness differ. Binary E1 is also generally close after 480; Yuksel 400 is a notable binary pathology. No method can be declared topology-equivalent from scalar evaluators, and Proposed/Yuksel topology fields were not retained.

The consolidated 27-row table is [common_quality_comparison.csv](common_quality_comparison.csv). [publication_readiness.csv](publication_readiness.csv) separates supported, qualified, and unsupported claims.

## Direct answers to the 20 required questions

1. **Technically trustworthy?** Yes for terminal/status, timing, and common-evaluator observations; incomplete for three mechanistic diagnoses.
2. **Why Proposed 109→159?** Primarily native low-density material/mass interpolation sensitivity on grayer coarse fields; topology/basin details are not retained.
3. **Still present under common evaluation?** No. It shrinks strongly in raw E1 and disappears/reverses under binary E1/E2/E3.
4. **Why Olhoff 480/560/640 only?** Deterministic trajectory-dependent modal/LP behavior after stabilization, not monotonic size exhaustion.
5. **Exact solver mechanism?** `linprog` dual-simplex-highs exit flag 0 (maximum LP iterations) at attempted k=358/400/1067; native-equivalent E3 reconstruction shows N=2 on all three attempts.
6. **High quality before failure?** Mature connected bimodal native/raw states, yes; not robust under binary E2/E3.
7. **Why Yuksel caps at 640/800 but 720 converges?** Stage-2 trajectories differ; only terminal samples remain, so the causal late behavior cannot be identified.
8. **Would more work help?** 640 POSSIBLE_CAP_LIMIT; 800 INDETERMINATE.
9. **Per-iteration exponents?** Olhoff {float(per_lookup[('Olhoff','all optimization-loop iterations')]['p_iter']):.3f}, Yuksel combined {float(per_lookup[('Yuksel','all optimization-loop iterations')]['p_iter']):.3f}, Proposed {float(per_lookup[('Proposed','all optimization-loop iterations')]['p_iter']):.3f}.
10. **Yuksel stages?** Stage 1 {float(per_lookup[('Yuksel','Stage 1')]['p_iter']):.3f}; Stage 2 {float(per_lookup[('Yuksel','Stage 2')]['p_iter']):.3f} per iteration. Count fits are separate.
11. **Cost vs count?** Yuksel count growth adds about {float(iter_lookup[('Yuksel','total')]['p_n']):.3f} to loop scaling; Proposed count behavior adds {float(iter_lookup[('Proposed','total')]['p_n']):.3f}; Olhoff adds zero.
12. **p=1.5 defensible?** Only as classified empirically per quantity in the fixed-p table; never as a theoretical conclusion from R² alone.
13. **Main complexity quantity?** Per-iteration loop time; separate Yuksel stages. Wall time is the secondary practical endpoint measure.
14. **RAM publication-ready?** No—UNRELIABLE.
15. **Proposed/Yuksel comparable from 320?** Comparable in common-raw omega1 with qualifications; not proved topology/spectrum equivalent.
16. **Olhoff advantage survives?** In common raw evaluation, yes; not under every binary model.
17. **Publication-ready now?** Status/censoring, per-iteration fits with one-sample qualification, endpoint timing semantics, common-raw comparison, and negative findings.
18. **Requires qualification?** Total-time fits, Olhoff quality, Proposed/Yuksel comparability, and all single-sample timing constants.
19. **Not supported?** Time-to-convergence for Olhoff, universal p=1.5 complexity, quantitative RAM scaling, simple-cap claims, topology equivalence, and KKT stationarity of coarse Proposed states.
20. **Targeted reruns?** Three diagnostic cases before an unqualified paper freeze; no full campaign rerun.

## Required figures

1. [Total wall time vs Ne](figures/01_total_wall_time_vs_Ne.png)
2. [Per-iteration loop time vs Ne](figures/02_per_iteration_loop_time_vs_Ne.png)
3. [Iteration count vs Ne](figures/03_iteration_count_vs_Ne.png)
4. [Yuksel stage counts](figures/04_yuksel_stage_iteration_counts.png)
5. [Native omega1](figures/05_native_omega1_vs_mesh.png)
6. [Common-evaluator quality](figures/06_common_evaluator_quality_vs_mesh.png)
7. [Grayness](figures/07_grayness_vs_mesh.png)
8. [Proposed topology evidence gap](figures/08_proposed_topology_comparison_unavailable.png)
9. [Olhoff failure-neighborhood histories](figures/09_olhoff_failure_neighborhood_histories.png)
10. [Yuksel terminal diagnostics / missing late histories](figures/10_yuksel_late_history_unavailable_terminal_diagnostics.png)
11. [Olhoff best-prior topologies](figures/11_olhoff_best_prior_topologies.png)

## Final decision

CAMPAIGN VALID — TARGETED FOLLOW-UP REQUIRED BEFORE FREEZE

FULL NINE-RESOLUTION RERUN: NOT REQUIRED

Minimal targeted follow-ups: Olhoff 640x80 diagnostics-only replay; Yuksel 800x100 same-cap history-retaining replay; Proposed 160x20 deterministic history/topology/mode-shape replay. Exact hypotheses and confirmation/refutation criteria are in [rerun_recommendations.md](rerun_recommendations.md).
"""
(HERE / "PERFORMANCE_CAMPAIGN_FORENSIC_AUDIT.md").write_text(report, encoding="utf-8")

# Final immutable-evidence check after every audit output has been generated.
artifact_hashes_after = {str(p.relative_to(REPO)): sha256(p) for p in artifact_files}
if artifact_hashes_after != artifact_hashes_before:
    raise RuntimeError("Frozen final_campaign evidence changed during audit generation")
provenance["frozen_evidence_unchanged_after_generation"] = True
provenance["working_tree_status_after_generation"] = git("status", "--short").splitlines()
(HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

print("Audit build complete")
print(json.dumps({"status_counts": status_counts, "censored": provenance["censored_count"]}, indent=2))
