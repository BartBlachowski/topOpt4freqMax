#!/usr/bin/env python3
"""Read-only Phase-2F/2G regressions for the frozen Candidate-C definition."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
G = REPO / "analysis/iteration_efficiency_phase2g_evaluator_selection_audit"
F = REPO / "analysis/iteration_efficiency_phase2f_evaluator_redesign"

z = np.load(F / "scripts/survey.npz")
taus = z["TAUS"].astype(float)
ti = int(np.argmin(abs(taus - 0.1)))
cfgs = sorted(k[:-6] for k in z.files if k.endswith("|omega"))
cfgs = [c for c in cfgs if c.endswith("|E1|linear") or c.endswith("|E2|eq4a") or c.endswith("|E3|eq4a")]

ordinals = []
unresolved = []
selected = {}
for cfg in cfgs:
    v = z[cfg + "|keLow"][:, ti, :]
    se = z[cfg + "|seLow"][:, ti, :]
    dwp = z[cfg + "|dwp"]
    omega = z[cfg + "|omega"]
    ks = z[cfg + "|k"].astype(int)
    valid = np.isfinite(v) & (v < .5) & (se < .5) & (dwp > .5)
    for i, k in enumerate(ks):
        ix = np.flatnonzero(valid[i])
        if ix.size:
            j = int(ix[0]); ordinals.append(j + 1); selected[(cfg, int(k))] = (j, omega[i], v[i], se[i], dwp[i])
        else:
            unresolved.append((cfg, int(k)))

o = np.asarray(ordinals)
counts = {"records": int(o.size), "max_ordinal": int(o.max()),
          "gt3": int((o > 3).sum()), "gt6": int((o > 6).sum()),
          "gt10": int((o > 10).sum()), "gt12": int((o > 12).sum()),
          "unresolved": len(unresolved)}

plateau = list(csv.DictReader((G / "THRESHOLD_PLATEAU.csv").open(newline="")))
binding = [r for r in plateau if abs(float(r["tau_density_partition"]) - .1) < 1e-12
           and .48 - 1e-12 <= float(r["kinetic_energy_cut"]) <= .56 + 1e-12]
plateau_pass = len(binding) == 10 and all(int(r["hard_gate_classification_or_ordinal_changes"]) == 0
                                         and int(r["hard_gate_unresolved_records"]) == 0 for r in binding)

neg = {}
for model in ("E2", "E3"):
    cfg = f"240x30|{model}|eq4a"; j, om, v, se, dwp = selected[(cfg, 594)]
    neg[model] = {"voidke_only_ordinal": int(np.flatnonzero(v < .5)[0] + 1),
                  "unanimous_ordinal": j + 1,
                  "voidke_only_omega": float(om[np.flatnonzero(v < .5)[0]]),
                  "unanimous_omega": float(om[j]),
                  "rejected_mode_voidKE": float(v[np.flatnonzero(v < .5)[0]]),
                  "rejected_mode_voidSE": float(se[np.flatnonzero(v < .5)[0]]),
                  "rejected_mode_densityParticipation": float(dwp[np.flatnonzero(v < .5)[0]])}

binary = list(csv.DictReader((G / "BINARY_NEAR_MECHANISM_MODES.csv").open(newline="")))
d833 = next(r for r in binary if r["mesh"] == "400x50" and r["state"] == "833"
            and r["mode_ordinal"] == "1")

expected = {"records":16536,"max_ordinal":18,"gt3":244,"gt6":69,"gt10":6,"gt12":5,"unresolved":0}
checks = {
    "adaptive_counts": counts == expected,
    "hard_gate_threshold_plateau_0p48_to_0p56": plateau_pass,
    "k594_voidke_only_differs_E2": neg["E2"]["voidke_only_ordinal"] != neg["E2"]["unanimous_ordinal"],
    "k594_voidke_only_differs_E3": neg["E3"]["voidke_only_ordinal"] != neg["E3"]["unanimous_ordinal"],
    "k594_disagreement_rejected": all(x["rejected_mode_voidKE"] < .5 and
                                         x["rejected_mode_voidSE"] > .5 and
                                         x["rejected_mode_densityParticipation"] < .5 for x in neg.values()),
    "D_k833_severe_anchor": abs(float(d833["binary_omega"]) - 4.672712513392596) < 1e-7 and
                            abs(float(d833["gray_structural_omega"]) - 170.93147031839717) < 1e-7,
}
result = {"schema_version":"phase2h_offline_regression_v1","source":"stored Phase-2F/2G evidence only",
          "pass":all(checks.values()),"checks":checks,"adaptive":counts,
          "k594_negative_control":neg,"D_k833":d833,
          "production_results":False,"optimizer_run":False}
(HERE / "offline_regression_results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps(result,indent=2))
if not result["pass"]: raise SystemExit(1)
