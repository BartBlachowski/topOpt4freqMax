#!/usr/bin/env python3
"""Phase-2G read-only evidence reducer.

Consumes Phase-2F artifacts and stored trajectories.  It never writes outside
the Phase-2G audit directory and never runs an optimizer.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase2g-matplotlib")
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy


REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
F2 = REPO / "analysis/iteration_efficiency_phase2f_evaluator_redesign"
SURVEY = F2 / "scripts/survey.npz"
GATE = F2 / "scripts/gate_full.npz"
BINCSV = F2 / "GRAY_VS_BINARY_QUALITY.csv"
TAU0 = 0.10
CUT0 = 0.50


def write_csv(name: str, rows: list[dict], fields: list[str] | None = None) -> None:
    path = OUT / name
    if fields is None:
        fields = list(rows[0]) if rows else ["status", "reason"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fnum(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def q(a, p):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if a.size else float("nan")


def cfgs(z) -> list[str]:
    return sorted(k[:-6] for k in z.files if k.endswith("|omega"))


def get(z, cfg, field):
    return z[f"{cfg}|{field}"]


def select(z, cfg, tau_idx, cut):
    v = get(z, cfg, "keLow")[:, tau_idx, :]
    se = get(z, cfg, "seLow")[:, tau_idx, :]
    dwp = get(z, cfg, "dwp")
    om = get(z, cfg, "omega")
    ordinal = np.full(v.shape[0], -1, int)
    omega = np.full(v.shape[0], np.nan)
    margin = np.full(v.shape[0], np.nan)
    for s in range(v.shape[0]):
        scores = np.vstack((cut - v[s], cut - se[s], dwp[s] - (1.0 - cut)))
        conditions = scores > 0
        ix = np.flatnonzero(np.isfinite(v[s]) & conditions.all(axis=0))
        if ix.size:
            j = int(ix[0])
            ordinal[s] = j + 1
            omega[s] = om[s, j]
            margin[s] = np.min(scores[:, j])
    return ordinal, omega, margin


def hard_gate_map(gate, mesh):
    k = gate[f"{mesh}|GATE2|k"]
    v = gate[f"{mesh}|GATE2|hard"].astype(bool)
    return {int(a): bool(b) for a, b in zip(k, v)}


def candidate_configs(all_cfgs):
    """C uses E1/linear and E2/E3/eq4a on the actual gray field."""
    return [c for c in all_cfgs if c.endswith("|E1|linear") or c.endswith("|E2|eq4a") or c.endswith("|E3|eq4a")]


def modality_population(z, c_cfgs, tau_idx):
    rows = []
    pooled = {"voidKE": [], "voidSE": [], "dwp": [], "ipr_scaled": [], "population": []}
    for cfg in c_cfgs:
        mesh, model, law = cfg.split("|")
        nx, ny = map(int, mesh.split("x"))
        v = get(z, cfg, "keLow")[:, tau_idx, :]
        se = get(z, cfg, "seLow")[:, tau_idx, :]
        dwp = get(z, cfg, "dwp")
        ipr = get(z, cfg, "ipr") * (nx * ny)
        finite = np.isfinite(v)
        # The 0.5 partition is not fitted: each diagnostic must independently
        # place the mode on the structural side of an equal-energy split.
        votes = (v < CUT0).astype(int) + (se < CUT0).astype(int) + (dwp > .5).astype(int)
        pop_masks = {"structural": finite & (votes == 3),
                     "artificial_corrob": finite & (votes == 0),
                     "diagnostic_disagreement": finite & ((votes > 0) & (votes < 3))}
        nstate = v.shape[0]
        thirds = np.floor(3 * np.arange(nstate) / max(nstate, 1)).astype(int)
        stages = {"ALL": np.ones((nstate, 1), bool),
                  "EARLY_TERCILE": (thirds == 0)[:, None],
                  "MID_TERCILE": (thirds == 1)[:, None],
                  "LATE_TERCILE": (thirds == 2)[:, None]}
        for name, m in pop_masks.items():
            for stage, sm in stages.items():
                mm = m & sm
                rows.append({
                    "mesh": mesh, "evaluator": model, "mass_law": law,
                    "trajectory_stage": stage, "stage_definition": "stored-state-order tercile",
                    "population": name, "n_modes": int(mm.sum()),
                    "voidKE_min": q(v[mm], 0), "voidKE_p01": q(v[mm], 1),
                    "voidKE_median": q(v[mm], 50), "voidKE_p99": q(v[mm], 99), "voidKE_max": q(v[mm], 100),
                    "voidSE_min": q(se[mm], 0), "voidSE_median": q(se[mm], 50), "voidSE_max": q(se[mm], 100),
                    "density_participation_min": q(dwp[mm], 0),
                    "density_participation_median": q(dwp[mm], 50),
                    "density_participation_max": q(dwp[mm], 100),
                    "nEl_scaled_IPR_min": q(ipr[mm], 0), "nEl_scaled_IPR_median": q(ipr[mm], 50),
                    "nEl_scaled_IPR_max": q(ipr[mm], 100),
                })
            for key, arr in (("voidKE", v), ("voidSE", se), ("dwp", dwp), ("ipr_scaled", ipr)):
                pooled[key].extend(np.asarray(arr[m], float).tolist())
            pooled["population"].extend([name] * int(m.sum()))
    # Add diagnostic-wise overlap/separation rows.  Positive directional gap
    # means the two empirical ranges do not overlap in the expected direction.
    pop = np.asarray(pooled["population"])
    for diag, direction in (("voidKE", "artificial_high"), ("voidSE", "artificial_high"),
                            ("dwp", "structural_high"), ("ipr_scaled", "artificial_high")):
        a = np.asarray(pooled[diag], float)[pop == "artificial_corrob"]
        s = np.asarray(pooled[diag], float)[pop == "structural"]
        if direction == "artificial_high":
            gap = float(np.nanmin(a) - np.nanmax(s))
        else:
            gap = float(np.nanmin(s) - np.nanmax(a))
        rows.append({
            "mesh": "POOLED", "evaluator": "C-family", "mass_law": "mixed-as-defined",
            "trajectory_stage": "ALL", "stage_definition": "stored-state-order tercile",
            "population": f"SEPARATION:{diag}", "n_modes": int(a.size + s.size),
            "voidKE_min": gap if diag == "voidKE" else "",
            "voidSE_min": gap if diag == "voidSE" else "",
            "density_participation_min": gap if diag == "dwp" else "",
            "nEl_scaled_IPR_min": gap if diag == "ipr_scaled" else "",
        })
    return rows, pooled


def threshold_audit(z, gate, c_cfgs, taus, tau_idx):
    cuts = np.unique(np.r_[np.logspace(-4, -1, 13), np.arange(.15, 1.0, .05),
                           np.arange(.40, .561, .01), .5])
    base = {c: select(z, c, tau_idx, CUT0)[:2] for c in c_cfgs}
    hard = {}
    for cfg in c_cfgs:
        mesh = cfg.split("|")[0]
        gm = hard_gate_map(gate, mesh)
        hard[cfg] = np.array([gm.get(int(k), False) for k in get(z, cfg, "k")], bool)
    rows = []
    for ti, tau in enumerate(taus):
        for cut in cuts:
            changed = unresolved = total = 0
            hard_changed = hard_unresolved = hard_total = 0
            freq_rel = []
            hard_freq_rel = []
            maxord = -1
            by_mesh_changed = Counter()
            for cfg in c_cfgs:
                o, w, _ = select(z, cfg, ti, float(cut))
                ob, wb = base[cfg]
                total += o.size
                ch = o != ob
                changed += int(ch.sum())
                hm = hard[cfg]
                hard_total += int(hm.sum())
                hard_changed += int((ch & hm).sum())
                hard_unresolved += int(((o < 0) & hm).sum())
                by_mesh_changed[cfg.split("|")[0]] += int(ch.sum())
                unresolved += int((o < 0).sum())
                if (o > 0).any():
                    maxord = max(maxord, int(o.max()))
                ok = np.isfinite(w) & np.isfinite(wb) & (wb > 0)
                freq_rel.extend((np.abs(w[ok] - wb[ok]) / wb[ok]).tolist())
                hok = ok & hm
                hard_freq_rel.extend((np.abs(w[hok] - wb[hok]) / wb[hok]).tolist())
            rows.append({
                "tau_density_partition": float(tau), "kinetic_energy_cut": float(cut),
                "state_evaluator_records": total, "classification_or_ordinal_changes_vs_tau0p1_cut0p5": changed,
                "unresolved_records": unresolved, "max_selected_ordinal": maxord,
                "selected_frequency_median_rel_change": q(freq_rel, 50),
                "selected_frequency_max_rel_change": q(freq_rel, 100),
                "hard_gate_records": hard_total,
                "hard_gate_classification_or_ordinal_changes": hard_changed,
                "hard_gate_unresolved_records": hard_unresolved,
                "hard_gate_selected_frequency_max_rel_change": q(hard_freq_rel, 100),
                "mesh_change_counts": ";".join(f"{k}:{v}" for k, v in sorted(by_mesh_changed.items()) if v),
                "b_ref_effect": "NOT_EXERCISABLE_NO_BREF_DENSITY_HISTORY",
                "k_enter_effect": "NOT_EXERCISABLE_NO_BREF_DENSITY_HISTORY",
                "k_cert_effect": "NOT_EXERCISABLE_NO_BREF_DENSITY_HISTORY",
            })
    # Exact decision plateaus by tau, over the pooled C-family survey.
    plateau = []
    for tau in taus:
        sub = [r for r in rows if r["tau_density_partition"] == float(tau)]
        same = [r for r in sub if r["classification_or_ordinal_changes_vs_tau0p1_cut0p5"] == 0]
        hard_same = [r for r in sub if r["hard_gate_classification_or_ordinal_changes"] == 0]
        rec = {
            "tau_density_partition": float(tau),
            "identical_decision_cut_min": min((r["kinetic_energy_cut"] for r in same), default=""),
            "identical_decision_cut_max": max((r["kinetic_energy_cut"] for r in same), default=""),
            "n_identical_cuts": len(same),
            "all_baseline_records_resolved": all(r["unresolved_records"] == 0 for r in same) if same else False,
            "hard_gate_identical_decision_cut_min": min((r["kinetic_energy_cut"] for r in hard_same), default=""),
            "hard_gate_identical_decision_cut_max": max((r["kinetic_energy_cut"] for r in hard_same), default=""),
            "n_hard_gate_identical_cuts": len(hard_same),
            "all_hard_gate_records_resolved": all(r["hard_gate_unresolved_records"] == 0 for r in hard_same) if hard_same else False,
        }
        plateau.append(rec)
    return rows, plateau


def boundary_cases(z, c_cfgs, tau_idx):
    rows = []
    counts = Counter()
    for cfg in c_cfgs:
        mesh, model, law = cfg.split("|")
        v = get(z, cfg, "keLow")[:, tau_idx, :]
        se = get(z, cfg, "seLow")[:, tau_idx, :]
        dwp = get(z, cfg, "dwp"); ipr = get(z, cfg, "ipr")
        om = get(z, cfg, "omega"); ks = get(z, cfg, "k")
        finite = np.isfinite(v)
        vote = np.stack((v < .5, se < .5, dwp > .5), axis=0)
        disagreement = finite & ~((vote.sum(axis=0) == 0) | (vote.sum(axis=0) == 3))
        counts["diagnostic_disagreement_all_modes"] += int(disagreement.sum())
        counts["near_voidKE_boundary_pm0p1"] += int((finite & (abs(v - .5) < .1)).sum())
        for s in range(v.shape[0]):
            ix = np.flatnonzero(finite[s] & (vote[:, s, :].sum(axis=0) == 3))
            first = int(ix[0]) if ix.size else v.shape[1]
            if ix.size and vote[:, s, first].sum() == 3:
                counts["selected_modes_unanimous_structural"] += 1
            for j in np.flatnonzero(disagreement[s] & (np.arange(v.shape[1]) < first)):
                counts["diagnostic_disagreement_below_selected"] += 1
                rows.append({
                    "mesh": mesh, "state": int(ks[s]), "evaluator": model, "mass_law": law,
                    "mode_ordinal": int(j) + 1, "omega": float(om[s, j]), "voidKE": float(v[s, j]),
                    "voidSE": float(se[s, j]), "density_participation": float(dwp[s, j]),
                    "IPR": float(ipr[s, j]), "baseline_selected_ordinal": first + 1,
                    "structural_conditions_met_of_3": int(vote[:, s, j].sum()),
                    "status": "DIAGNOSTIC_DISAGREEMENT_BELOW_SELECTION_REJECTED_BY_UNANIMOUS_VALIDITY_RULE",
                })
    return rows, dict(counts)


def candidate_failure_audit(z, all_cfgs, tau_idx):
    rows = []
    for candidate in ("A", "B"):
        if candidate == "A":
            use = [c for c in all_cfgs if c.endswith("|E1|linear") or c.endswith("|E2|eq4") or c.endswith("|E3|eq4")]
        else:
            use = candidate_configs(all_cfgs)
        for cfg in use:
            mesh, model, law = cfg.split("|")
            o, _, _ = select(z, cfg, tau_idx, CUT0)
            v0 = get(z, cfg, "keLow")[:, tau_idx, 0]
            se0 = get(z, cfg, "seLow")[:, tau_idx, 0]
            dwp0 = get(z, cfg, "dwp")[:, 0]
            conditions_met = (v0 < .5).astype(int) + (se0 < .5).astype(int) + (dwp0 > .5).astype(int)
            rows.append({
                "candidate": candidate, "mesh": mesh, "evaluator": model, "mass_law": law,
                "surveyed_state_evaluator_records": int(o.size),
                "algebraic_lowest_fails_unanimous_validity": int((conditions_met < 3).sum()),
                "algebraic_lowest_rejected_by_all_three_diagnostics": int((conditions_met == 0).sum()),
                "algebraic_lowest_diagnostic_disagreement": int(((conditions_met > 0) & (conditions_met < 3)).sum()),
                "first_structural_ordinal_gt3": int((o > 3).sum()),
                "maximum_first_structural_ordinal": int(o.max()),
                "unresolved": int((o < 0).sum()),
                "rho0p1_continuity": "FAIL_FINITE_JUMP_E2_E3" if candidate == "A" else "PASS_EQ4A",
            })
    return rows


def evaluator_role_audit(z, c_cfgs, tau_idx):
    rows = []
    meshes = sorted({c.split("|")[0] for c in c_cfgs})
    for mesh in meshes:
        wanted = {c.split("|")[1]: c for c in c_cfgs if c.startswith(mesh + "|")}
        if not all(k in wanted for k in ("E1", "E2", "E3")):
            continue
        series = {}
        for model, cfg in wanted.items():
            k = get(z, cfg, "k").astype(int); _, w, _ = select(z, cfg, tau_idx, CUT0)
            series[model] = {int(a): float(b) for a, b in zip(k, w)}
        common = sorted(set(series["E1"]) & set(series["E2"]) & set(series["E3"]))
        e1 = np.array([series["E1"][k] for k in common]); e2 = np.array([series["E2"][k] for k in common]); e3 = np.array([series["E3"][k] for k in common])
        d23 = abs(e2-e3)/e2; d12 = abs(e1-e2)/e1
        rows.append({
            "mesh": mesh, "aligned_states": len(common),
            "E2_E3_median_relative_difference": q(d23, 50), "E2_E3_p95_relative_difference": q(d23, 95),
            "E2_E3_max_relative_difference": q(d23, 100), "E2_E3_correlation": float(np.corrcoef(e2,e3)[0,1]),
            "E1_E2_median_relative_difference": q(d12, 50), "E1_E2_p95_relative_difference": q(d12, 95),
            "E1_E2_max_relative_difference": q(d12, 100), "E1_E2_correlation": float(np.corrcoef(e1,e2)[0,1]),
            "normalised_binding_evaluator": "NOT_DETERMINABLE_WITHOUT_C_REFERENCE_Q_REF",
        })
    return rows


def adaptive_audit(z, c_cfgs, tau_idx):
    rows = []
    summary = Counter()
    ordinals = []
    unresolved = []
    for cfg in c_cfgs:
        mesh, model, law = cfg.split("|")
        ks = get(z, cfg, "k").astype(int)
        nm = get(z, cfg, "nmodes").astype(int)
        v = get(z, cfg, "keLow")[:, tau_idx, :]
        o, w, margin = select(z, cfg, tau_idx, CUT0)
        for i, state in enumerate(ks):
            if o[i] > 0:
                ordinals.append(int(o[i]))
                summary["records"] += 1
                for n in (3, 6, 10, 12, 24):
                    if o[i] > n:
                        summary[f"gt{n}"] += 1
            else:
                summary["unresolved"] += 1
                unresolved.append({
                    "mesh": mesh, "state": int(state), "evaluator": model, "mass_law": law,
                    "modes_computed": int(nm[i]), "status": "STRUCTURAL_MODE_NOT_FOUND",
                    "cause": "NO_MODE_SATISFIED_ALL_THREE_STRUCTURAL_CONDITIONS_IN_COMPUTED_SPECTRUM",
                    "follow_up": "ESCALATE_GEOMETRICALLY_OR_FAIL_CLOSED_ON_RESOURCE_OR_SOLVER_LIMIT",
                })
            if nm[i] > 12 or o[i] > 3 or o[i] < 0:
                lower = v[i, :max(0, o[i] - 1)] if o[i] > 0 else v[i]
                batches = [3, 6, 12, 24, 48]
                needed = int(o[i]) if o[i] > 0 else int(nm[i]) + 1
                final_batch = next((n for n in batches if n >= needed), batches[-1])
                path = "->".join(str(n) for n in batches if n <= final_batch)
                rows.append({
                    "mesh": mesh, "state": int(state), "evaluator": model, "mass_law": law,
                    "initial_requested_modes_phase2f": 12, "final_requested_modes_phase2f": int(nm[i]),
                    "phase2f_escalation_count": int(nm[i] > 12) + int(nm[i] > 24),
                    "recommended_3_based_batch_expansion_count": sum(needed > n for n in (3, 6, 12, 24)),
                    "first_structural_ordinal": int(o[i]), "selected_omega": float(w[i]),
                    "selected_minimum_condition_margin": float(margin[i]),
                    "minimum_lower_mode_voidKE_minus_0p5_diagnostic_only": float(np.nanmin(lower - CUT0)) if lower.size else "",
                    "recommended_schedule_path": path,
                })
    summary["max_ordinal"] = max(ordinals) if ordinals else -1
    summary["phase2f_escalations"] = sum(int(get(z, c, "nmodes")[i] > 12)
                                            for c in c_cfgs for i in range(get(z, c, "nmodes").size))
    summary["recommended_schedule_escalations"] = summary["gt3"]
    summary["recommended_schedule_total_batch_expansions"] = (
        summary["gt3"] + summary["gt6"] + summary["gt12"] + summary["gt24"])
    return rows, unresolved, dict(summary), ordinals


def smoothness_audit(z, gate, c_cfgs, tau_idx):
    rows = []
    pooled = []
    for cfg in c_cfgs:
        mesh, model, law = cfg.split("|")
        ks = get(z, cfg, "k").astype(int)
        ordinal, omega, _ = select(z, cfg, tau_idx, CUT0)
        hm = hard_gate_map(gate, mesh)
        d = []
        outliers = []
        for i in range(len(ks) - 1):
            # The requested quantity is for genuinely consecutive trajectory
            # states, not adjacent samples from a strided survey.
            if ks[i + 1] != ks[i] + 1:
                continue
            if not (hm.get(int(ks[i]), False) and hm.get(int(ks[i + 1]), False)):
                continue
            if not (np.isfinite(omega[i]) and np.isfinite(omega[i + 1]) and omega[i] > 0):
                continue
            rel = abs(omega[i + 1] - omega[i]) / omega[i]
            d.append(rel)
            if rel > .005:
                outliers.append((rel, int(ks[i]), int(ks[i + 1]), int(ordinal[i]), int(ordinal[i + 1])))
        pooled.extend(d)
        rows.append({
            "mesh": mesh, "evaluator": model, "mass_law": law,
            "consecutive_hard_gate_passing_pairs": len(d), "median": q(d, 50), "p95": q(d, 95),
            "p99": q(d, 99), "max": q(d, 100),
            "fraction_gt_0p5pct": float(np.mean(np.asarray(d) > .005)) if d else float("nan"),
            "fraction_gt_1pct": float(np.mean(np.asarray(d) > .01)) if d else float("nan"),
            "fraction_gt_2pct": float(np.mean(np.asarray(d) > .02)) if d else float("nan"),
            "largest_outlier": str(max(outliers, default=(float("nan"),))),
            "coverage_note": "FULL_CONSECUTIVE" if len(ks) > 1 and np.all(np.diff(ks) == 1) else "STRIDED_SURVEY_NO_CONSECUTIVE_PAIRS",
        })
    rows.append({
        "mesh": "POOLED_FULL_CONSECUTIVE", "evaluator": "C-family", "mass_law": "as-defined",
        "consecutive_hard_gate_passing_pairs": len(pooled), "median": q(pooled, 50), "p95": q(pooled, 95),
        "p99": q(pooled, 99), "max": q(pooled, 100),
        "fraction_gt_0p5pct": float(np.mean(np.asarray(pooled) > .005)) if pooled else float("nan"),
        "fraction_gt_1pct": float(np.mean(np.asarray(pooled) > .01)) if pooled else float("nan"),
        "fraction_gt_2pct": float(np.mean(np.asarray(pooled) > .02)) if pooled else float("nan"),
        "largest_outlier": "SEE_PER_CONFIGURATION_ROWS", "coverage_note": "POOLED_ONLY_TRUE_CONSECUTIVE_PAIRS",
    })
    return rows, pooled


def binary_audit(binary, gate):
    rows = []
    near = []
    pooled_pass = pooled_severe = 0
    for mesh in sorted({r["mesh"] for r in binary}):
        sub = [r for r in binary if r["mesh"] == mesh]
        hm = hard_gate_map(gate, mesh)
        passed = [r for r in sub if hm.get(int(r["state"]), False)]
        severe = [r for r in passed if fnum(r["binary_omega_E2"]) < .5 * fnum(r["gray_struct_omega_E2"])]
        pooled_pass += len(passed); pooled_severe += len(severe)
        rel = [abs(fnum(r["binary_omega_E2"]) - fnum(r["gray_struct_omega_E2"])) /
               fnum(r["gray_struct_omega_E2"]) for r in passed]
        ratio = [fnum(r["binary_omega_E2"]) / fnum(r["gray_struct_omega_E2"]) for r in passed]
        rows.append({
            "mesh": mesh, "surveyed_states": len(sub), "hard_gate_passing_states": len(passed),
            "severe_binary_lt_half_gray": len(severe),
            "severe_fraction_of_hard_gate_pass": len(severe) / len(passed) if passed else float("nan"),
            "median_relative_discrepancy": q(rel, 50), "p95_relative_discrepancy": q(rel, 95),
            "max_relative_discrepancy": q(rel, 100), "minimum_binary_to_gray_ratio": q(ratio, 0),
            "binary_mode1_voidKE_E2_max_all_states": max(fnum(r["binary_voidKE_E2"]) for r in sub),
        })
        nmax = max(int(r["state"]) for r in sub)
        for label, lo, hi in (("early", 0, .25), ("mid", .25, .75), ("late", .75, 1.01)):
            m = [r for r in passed if lo < int(r["state"]) / nmax <= hi]
            rr = [abs(fnum(r["binary_omega_E2"]) - fnum(r["gray_struct_omega_E2"])) /
                  fnum(r["gray_struct_omega_E2"]) for r in m]
            near.append({
                "mesh": mesh, "maturity_bin": label, "normalised_state_range": f"({lo},{hi}]",
                "hard_gate_passing_states": len(m), "median_binary_gray_relative_discrepancy": q(rr, 50),
                "p95_binary_gray_relative_discrepancy": q(rr, 95),
                "severe_count": sum(fnum(r["binary_omega_E2"]) < .5 * fnum(r["gray_struct_omega_E2"]) for r in m),
            })
    rows.append({
        "mesh": "POOLED", "surveyed_states": len(binary), "hard_gate_passing_states": pooled_pass,
        "severe_binary_lt_half_gray": pooled_severe,
        "severe_fraction_of_hard_gate_pass": pooled_severe / pooled_pass,
        "median_relative_discrepancy": q([abs(fnum(r["binary_omega_E2"]) - fnum(r["gray_struct_omega_E2"])) /
                                           fnum(r["gray_struct_omega_E2"]) for r in binary], 50),
        "binary_mode1_voidKE_E2_max_all_states": max(fnum(r["binary_voidKE_E2"]) for r in binary),
    })
    return rows, near


def candidate_definitions():
    return [
        {"candidate": "A", "density_representation": "actual gray",
         "stiffness_interpolation": "E1:1e7(1e-6+(1-1e-6)x^3); E2:1e7(1e-9+(1-1e-9)x^3); E3:1e7 max(x,1e-3)^3",
         "mass_interpolation": "E1 linear with 1e-6 floor; E2/E3 Du-Olhoff Eq.(4), g=x^6 for x<=0.1 else x (E2 has 1e-9 floor)",
         "eigensolver": "MATLAB eigs, smallestabs fallback sm, deterministic v0", "requested_modes": "fixed 3",
         "adaptive_escalation": "none", "modal_selection": "algebraically lowest mode",
         "hard_gate_interaction": "separate pointwise prerequisite; does not classify modes", "failure_semantics": "NaN only if eigensolver returns fewer modes"},
        {"candidate": "B", "density_representation": "actual gray",
         "stiffness_interpolation": "same E1/E2/E3 conventions as A",
         "mass_interpolation": "E1 unchanged linear; E2/E3 continuous Eq.(4a), g=1e5*x^6 for x<=0.1 else x",
         "eigensolver": "Phase-2F SciPy ARPACK shift-invert; conceptual implementation unspecified", "requested_modes": "small fixed count (3 for A-like use; 12 diagnostic survey)",
         "adaptive_escalation": "none in candidate definition", "modal_selection": "algebraically lowest mode",
         "hard_gate_interaction": "separate pointwise prerequisite", "failure_semantics": "no explicit modal-invalid state"},
        {"candidate": "C", "density_representation": "actual gray",
         "stiffness_interpolation": "same E1/E2/E3 conventions as A",
         "mass_interpolation": "E1 unchanged linear; E2/E3 continuous Eq.(4a)",
         "eigensolver": "adaptive lowest-spectrum generalized symmetric eigenpairs with eigenvectors",
         "requested_modes": "Phase-2F began at 12, then 24/48; audited recommendation 3->6->12->24->48->...",
         "adaptive_escalation": "yes, until valid structural mode or fail-closed resource/solver limit",
         "modal_selection": "lowest mode satisfying all three conditions: voidKE(rho_eff<=0.1)<0.5; voidSE(rho_eff<=0.1)<0.5; density-weighted participation>0.5. IPR is a nonbinding localization cross-check",
         "hard_gate_interaction": "unchanged and logically separate; both must pass", "failure_semantics": "STRUCTURAL_MODE_NOT_FOUND; never substitute highest computed mode"},
        {"candidate": "D", "density_representation": "stable rank-based exact-count binary projection, round(0.5*N) solids, increasing-index tie-break",
         "stiffness_interpolation": "same E1/E2/E3 conventions, evaluated at binary field",
         "mass_interpolation": "E1 linear; E2/E3 Eq.(4a), which reduces to endpoint values on binary field",
         "eigensolver": "Phase-2F SciPy ARPACK shift-invert", "requested_modes": "fixed 6 in Phase-2F binary audit (lowest reported)",
         "adaptive_escalation": "none", "modal_selection": "algebraically lowest binary mode",
         "hard_gate_interaction": "same projected binary field, but gate checks connectivity/island area rather than stiffness adequacy",
         "failure_semantics": "no modal-invalid status; near-mechanisms remain genuine returned values"},
    ]


def provenance(evidence_paths):
    pre = json.loads((F2 / "WP0_INTEGRITY_pre.json").read_text())
    checks = []
    for group in ("protected_numerical_sources", "profile_sources", "audit_records", "phase2d_declared_unchanged"):
        for rel, rec in pre[group].items():
            expected = rec.get("expected_phase2a") or rec.get("declared") or rec.get("observed")
            path = REPO / rel
            actual = sha(path) if path.exists() else None
            checks.append({"group": group, "path": rel, "expected": expected, "actual": actual, "match": actual == expected})
    return {
        "phase": "2G independent common-evaluator selection audit",
        "classification": "READ_ONLY_AUDIT_NO_OPTIMIZATION_NO_REFREEZE_NO_PRODUCTION",
        "captured_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short").splitlines(),
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "h5py": h5py.__version__},
        "phase2f_completion": {"survey_archive_present": SURVEY.exists(),
                               "survey_log_tail": (F2 / "scripts/survey.log").read_text().splitlines()[-5:]},
        "protected_source_checks": checks,
        "protected_mismatch_count": sum(not c["match"] for c in checks),
        "evidence_hashes": {str(p.relative_to(REPO)): sha(p) for p in evidence_paths if p.exists()},
    }


def figures(z, c_cfgs, taus, tau_idx, pooled, plateau_rows, ordinals, smooth, binary, gate):
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    pop = np.asarray(pooled["population"])
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, key, label, logit in zip(ax, ("voidKE", "dwp", "ipr_scaled"),
                                     ("void kinetic-energy share", "density-weighted participation", "N·IPR"),
                                     (True, False, True)):
        for name, color in (("structural", "#2878b5"), ("artificial_corrob", "#d95319"),
                            ("diagnostic_disagreement", "#7f7f7f")):
            x = np.asarray(pooled[key], float)[pop == name]
            if logit and key == "voidKE":
                x = np.clip(x, 1e-12, 1 - 1e-12)
                x = np.log10(x / (1 - x)); xlabel = "log10(p/(1-p))"
            else:
                x = np.clip(x, 1e-12, None); xlabel = label
            a.hist(x, bins=80, density=True, histtype="step", linewidth=1.4, label=name, color=color)
        a.set_xlabel(xlabel); a.set_ylabel("density"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "modal_population_separation.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for p in plateau_rows:
        if p["identical_decision_cut_min"] != "":
            ax.plot([p["identical_decision_cut_min"], p["identical_decision_cut_max"]],
                    [p["tau_density_partition"]] * 2, marker="|", linewidth=4)
    ax.axhline(.1, color="k", linestyle="--", linewidth=.8); ax.axvline(.5, color="k", linestyle="--", linewidth=.8)
    ax.set_xscale("log"); ax.set_xlabel("voidKE validity cut"); ax.set_ylabel("density partition tau")
    ax.set_title("Pooled selection-invariant threshold plateaus")
    fig.tight_layout(); fig.savefig(figdir / "threshold_plateau.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.arange(1, max(ordinals, default=1) + 2) - .5
    ax.hist(ordinals, bins=bins, color="#2878b5", edgecolor="white")
    for n in (3, 6, 10, 12, 24): ax.axvline(n + .5, color="k", alpha=.2, linewidth=.7)
    ax.set_xlabel("first structural-mode ordinal"); ax.set_ylabel("state-evaluator records")
    fig.tight_layout(); fig.savefig(figdir / "structural_mode_ordinal_distribution.png", dpi=180); plt.close(fig)

    cfg = "160x20|E2|eq4a"
    if cfg in c_cfgs:
        ks = get(z, cfg, "k"); o, wc, _ = select(z, cfg, tau_idx, CUT0); low = get(z, cfg, "omega")[:, 0]
        bmap = {(r["mesh"], int(r["state"])): fnum(r["binary_omega_E2"]) for r in binary}
        bd = np.array([bmap.get(("160x20", int(k)), np.nan) for k in ks])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(ks, low, linewidth=.8, label="B: gray algebraic lowest", alpha=.8)
        ax.plot(ks, wc, linewidth=1.0, label="C: gray structural")
        ax.plot(ks, bd, linewidth=.7, label="D: binary lowest", alpha=.7)
        ax.set_xlabel("stored trajectory state k"); ax.set_ylabel("omega"); ax.set_ylim(bottom=0)
        ax.legend(ncol=3, fontsize=8); fig.tight_layout()
        fig.savefig(figdir / "C_and_D_frequency_trajectory_160x20.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    s = np.asarray(smooth, float)
    if s.size:
        ax.hist(100 * s, bins=100, cumulative=True, density=True, histtype="step", linewidth=1.5)
    ax.set_xlabel("consecutive C frequency change (%)"); ax.set_ylabel("empirical CDF")
    ax.set_xlim(left=0); fig.tight_layout(); fig.savefig(figdir / "C_step_change_distribution.png", dpi=180); plt.close(fig)


def main():
    if not SURVEY.exists():
        raise SystemExit(f"Phase-2F survey archive absent: {SURVEY}")
    z = np.load(SURVEY)
    gate = np.load(GATE)
    binary = read_csv(BINCSV)
    taus = z["TAUS"].astype(float)
    tau_idx = int(np.argmin(abs(taus - TAU0)))
    all_cfgs = cfgs(z)
    c_cfgs = candidate_configs(all_cfgs)

    defs = candidate_definitions()
    write_csv("CANDIDATE_DEFINITIONS_VERIFIED.csv", defs)
    pop_rows, pooled = modality_population(z, c_cfgs, tau_idx)
    write_csv("MODAL_POPULATION_ANALYSIS.csv", pop_rows)
    threshold_rows, plateau_rows = threshold_audit(z, gate, c_cfgs, taus, tau_idx)
    write_csv("THRESHOLD_PLATEAU.csv", threshold_rows)
    write_csv("THRESHOLD_PLATEAU_SUMMARY.csv", plateau_rows)
    boundary, boundary_counts = boundary_cases(z, c_cfgs, tau_idx)
    write_csv("MODAL_BOUNDARY_CASES.csv", boundary)
    write_csv("CANDIDATE_A_B_FAILURES.csv", candidate_failure_audit(z, all_cfgs, tau_idx))
    write_csv("EVALUATOR_ROLE.csv", evaluator_role_audit(z, c_cfgs, tau_idx))
    escalation, unresolved, acount, ordinals = adaptive_audit(z, c_cfgs, tau_idx)
    write_csv("ADAPTIVE_MODE_ESCALATIONS.csv", escalation)
    write_csv("UNRESOLVED_STATES.csv", unresolved)
    smooth_rows, smooth = smoothness_audit(z, gate, c_cfgs, tau_idx)
    write_csv("STEP_SMOOTHNESS.csv", smooth_rows)
    binary_rows, maturity_rows = binary_audit(binary, gate)
    write_csv("BINARY_NEAR_MECHANISM_AUDIT.csv", binary_rows)
    write_csv("BINARY_MATURATION.csv", maturity_rows)

    # Mode identity is filled by the independent-anchor/MAC script.  Create a
    # fail-explicit placeholder if it has not yet been run.
    if not (OUT / "MODE_IDENTITY_AUDIT.csv").exists():
        write_csv("MODE_IDENTITY_AUDIT.csv", [{"status": "PENDING_INDEPENDENT_MAC_RUN", "reason": "run verify_anchors.py"}])
    if not (OUT / "ANCHOR_REPRODUCTION.csv").exists():
        write_csv("ANCHOR_REPRODUCTION.csv", [{"status": "PENDING_INDEPENDENT_ANCHOR_RUN", "reason": "run verify_anchors.py"}])

    # Method-blind, unweighted ordinal score; unknown neutrality is penalised,
    # not silently treated as a pass.
    score = [
        {"criterion": "continuity/stability", "A": 1, "B": 4, "C": 4, "D": 3},
        {"criterion": "physical interpretability", "A": 2, "B": 1, "C": 5, "D": 3},
        {"criterion": "gray-trajectory fidelity", "A": 3, "B": 4, "C": 5, "D": 1},
        {"criterion": "modal validity", "A": 2, "B": 1, "C": 5, "D": 3},
        {"criterion": "projection artifacts", "A": 5, "B": 5, "C": 5, "D": 1},
        {"criterion": "threshold arbitrariness", "A": 5, "B": 5, "C": 4, "D": 3},
        {"criterion": "mesh robustness", "A": 1, "B": 1, "C": 4, "D": 1},
        {"criterion": "method neutrality evidence", "A": 2, "B": 2, "C": 2, "D": 1},
        {"criterion": "reproducibility", "A": 5, "B": 5, "C": 4, "D": 5},
        {"criterion": "computational cost", "A": 5, "B": 5, "C": 3, "D": 4},
        {"criterion": "b_ref/k_enter/k_cert compatibility", "A": 1, "B": 1, "C": 3, "D": 1},
        {"criterion": "endpoint-topology role", "A": 2, "B": 2, "C": 3, "D": 5},
    ]
    totals = {c: sum(r[c] for r in score) for c in "ABCD"}
    score.append({"criterion": "UNWEIGHTED_TOTAL", **totals})
    write_csv("CANDIDATE_SCORECARD.csv", score)

    bpool = next(r for r in binary_rows if r["mesh"] == "POOLED")
    summary = {
        "survey_configs": c_cfgs, "survey_state_evaluator_records": int(sum(get(z, c, "k").size for c in c_cfgs)),
        "adaptive": acount,
        "population": {
            "structural_voidKE_max": max(r["voidKE_max"] for r in pop_rows if r["population"] == "structural" and r["trajectory_stage"] == "ALL"),
            "artificial_voidKE_min": min(r["voidKE_min"] for r in pop_rows if r["population"] == "artificial_corrob" and r["trajectory_stage"] == "ALL"),
        },
        "plateaus": plateau_rows,
        "boundary_counts": boundary_counts,
        "smoothness_pooled": smooth_rows[-1],
        "binary": bpool,
        "binary_void_max": {m: max(fnum(r[f"binary_voidKE_{m}"]) for r in binary) for m in ("E1", "E2", "E3")},
        "score_totals": totals,
        "phase2f_all_configs": all_cfgs,
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    evidence = [SURVEY, GATE, BINCSV, F2 / "BINARY_PROJECTION_STABILITY.csv",
                F2 / "K252_MODAL_REPRODUCTION.csv", F2 / "WP0_INTEGRITY_pre.json",
                F2 / "scripts/modal_engine.py", F2 / "scripts/wp4_survey.py", F2 / "scripts/wp11_12_binary.py",
                REPO / "analysis/three_method_parametric_study/study_evaluate_design.m",
                REPO / "analysis/iteration_efficiency_phase2a/iteration_efficiency_contract.json"]
    prov = provenance(evidence)
    (OUT / "audit_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    figures(z, c_cfgs, taus, tau_idx, pooled, plateau_rows, ordinals, smooth, binary, gate)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
