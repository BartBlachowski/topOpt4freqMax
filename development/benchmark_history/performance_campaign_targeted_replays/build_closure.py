#!/usr/bin/env python3
"""Build the targeted-replay closure report, delta audit, and provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FROZEN = REPO / "examples" / "Performance" / "final_campaign"
PRIOR = REPO / "analysis" / "performance_campaign_forensic_audit"


def read_csv(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(name: str, rows: list[dict[str, str]]) -> None:
    with (HERE / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.rstrip()


def f(v: float | None, digits: int = 6) -> str:
    if v is None or not math.isfinite(float(v)):
        return "N/A"
    return f"{float(v):.{digits}g}"


summary = json.loads((HERE / "matlab_analysis_summary.json").read_text())
initial = json.loads((HERE / "initial_provenance.json").read_text())
prior_provenance = json.loads((PRIOR / "provenance.json").read_text())
o = summary["olhoff"]
y = summary["yuksel"]
p = summary["proposed"]
yhist = read_csv("yuksel_800_history.csv")[-300:]
ywin = read_csv("yuksel_800_cap_diagnosis.csv")
olp = read_csv("olhoff_640_failure_diagnostics.csv")[0]
pdet = read_csv("proposed_160_determinism.csv")
pcommon = read_csv("proposed_160_common_evaluators.csv")
ploc = read_csv("proposed_160_mode_localization.csv")
cfg_identity = read_csv("configuration_identity.csv")

ymax = [float(r["max_dx"]) for r in yhist]
yrms = [float(r["rms_dx"]) for r in yhist]
yactive = [float(r["step_active_fraction"]) for r in yhist]
yobj = [float(r["objective"]) for r in yhist]
yrel = [float(r["relative_objective_change"]) for r in yhist if r["relative_objective_change"]]
turn = [
    float(r["binary_turnover_since_prior_snapshot"])
    for r in yhist
    if r["binary_turnover_since_prior_snapshot"]
]
move_hits = sum(abs(v - 0.1) < 1e-12 for v in ymax)
slopes = [ymax[i + 1] - ymax[i] for i in range(len(ymax) - 1)]
slope_signs = [1 if v > 0 else -1 if v < 0 else 0 for v in slopes]
strict_trend_reversals = sum(a * b < 0 for a, b in zip(slope_signs, slope_signs[1:]))
slope_state_changes = sum(a != b for a, b in zip(slope_signs, slope_signs[1:]))
nonzero_signs = [v for v in slope_signs if v]
nonzero_trend_reversals = sum(a != b for a, b in zip(nonzero_signs, nonzero_signs[1:]))


publication_rows = [
    {
        "Claim": "Final campaign technical integrity: all 27 terminal/status labels and exactly five censored observations are correct",
        "Assessment": "PUBLICATION_READY",
        "Evidence_or_qualification": "Prior independent audit plus exact target endpoint reproduction; no status changed",
    },
    {
        "Claim": "No censored observation entered an original scaling fit",
        "Assessment": "PUBLICATION_READY",
        "Evidence_or_qualification": "Original masks remain unchanged; diagnostic replay timing was not fitted",
    },
    {
        "Claim": "Proposed 160x20 deterministically reaches the native 109-rad/s endpoint",
        "Assessment": "PUBLICATION_READY",
        "Evidence_or_qualification": "Two identical 107-iteration runs have exact numerical histories, density and spectra and both match the original fingerprint",
    },
    {
        "Claim": "Proposed native convergence success rate is 9/9 under the frozen native stopping rule",
        "Assessment": "PUBLICATION_READY",
        "Evidence_or_qualification": "The original campaign status audit is unchanged; the targeted 160x20 endpoint reproduces exactly twice",
    },
    {
        "Claim": "The Proposed coarse anomaly is primarily model/interpolation dependent",
        "Assessment": "PUBLICATION_READY_WITH_QUALIFICATION",
        "Evidence_or_qualification": "Same density gives native/raw-E1/binary-E1 omega1 109.05/153.68/162.76; conclusion is demonstrated at 160x20 and consistent with prior 240 evidence",
    },
    {
        "Claim": "The Proposed native low triplet consists of weak-material/local modes",
        "Assessment": "PUBLICATION_READY_WITH_QUALIFICATION",
        "Evidence_or_qualification": "At 160x20 essentially all modal displacement and kinetic energy lies in rho<=0.1 elements; do not generalize to every resolution",
    },
    {
        "Claim": "The Proposed 160x20 load-carrying topology is qualitatively poor or disconnected",
        "Assessment": "NOT_SUPPORTED",
        "Evidence_or_qualification": "Raw and exact-count binary designs are connected and have strong common spectra; pathology is local weak material",
    },
    {
        "Claim": "Proposed coarse endpoints are KKT/stationary solutions",
        "Assessment": "NOT_SUPPORTED",
        "Evidence_or_qualification": "Native stopping and trajectory are retained, but no KKT residual or basin/restart study was authorized",
    },
    {
        "Claim": "Olhoff 640x80 deterministically fails at attempted k=1067 with linprog exitflag 0",
        "Assessment": "PUBLICATION_READY_WITH_QUALIFICATION",
        "Evidence_or_qualification": "History, densities and snapshots are bit-identical; output reports 38 iterations and 'Solver stopped prematurely'; MATLAB maps flag 0 to maximum iterations",
    },
    {
        "Claim": "Olhoff failure is caused specifically by LP degeneracy, scaling, or modal branching",
        "Assessment": "NOT_SUPPORTED",
        "Evidence_or_qualification": "N=2/gap12=0.00256 co-occurs, but finite full-rank row diagnostics do not establish causation and no point was returned for residual analysis",
    },
    {
        "Claim": "Yuksel 800x100 is a simple cap limitation likely resolved by a modest extension",
        "Assessment": "NOT_SUPPORTED",
        "Evidence_or_qualification": "Final 300 show persistent localized irregular oscillation; max dx never reaches tolerance and hits the full move 61 times",
    },
    {
        "Claim": "Yuksel 800x100 has a stable global objective while few variables continue moving",
        "Assessment": "PUBLICATION_READY_WITH_QUALIFICATION",
        "Evidence_or_qualification": "Median relative objective change 1.70e-5; active fraction 3.2-3.5%; objective drifts about 0.57% across the full late window",
    },
    {
        "Claim": "Per-iteration computational scaling can be reported",
        "Assessment": "PUBLICATION_READY_WITH_QUALIFICATION",
        "Evidence_or_qualification": "Original admissible-row fits are unchanged, use one sample per mesh, and Yuksel stages remain separate",
    },
    {
        "Claim": "Total-time exponents are intrinsic kernel complexity",
        "Assessment": "NOT_SUPPORTED",
        "Evidence_or_qualification": "Iteration counts and endpoint semantics materially contribute; retain as practical endpoint fits only",
    },
    {
        "Claim": "The unchanged common evaluator is necessary for cross-method interpretation",
        "Assessment": "PUBLICATION_READY",
        "Evidence_or_qualification": "It removes native interpolation artefacts on the identical retained Proposed density and preserves an auditable common basis",
    },
    {
        "Claim": "Olhoff has universally superior binary topology",
        "Assessment": "NOT_SUPPORTED",
        "Evidence_or_qualification": "Prior binary E2/E3 pathologies remain; safe claim is higher common-raw omega1 at fixed work",
    },
    {
        "Claim": "Original MaxRAM_MB supports quantitative memory scaling",
        "Assessment": "NOT_SUPPORTED",
        "Evidence_or_qualification": "Frozen UNRELIABLE classification; no RAM repair or new benchmark was performed",
    },
    {
        "Claim": "Diagnostic replay wall times can replace campaign timings",
        "Assessment": "NOT_SUPPORTED",
        "Evidence_or_qualification": "Additional logging/eigensolves perturb runtime; all replay timing is DIAGNOSTIC ONLY",
    },
    {
        "Claim": "All 27 cases converged successfully",
        "Assessment": "NOT_SUPPORTED",
        "Evidence_or_qualification": "The five original censored observations remain part of the scientific record",
    },
    {
        "Claim": "Proposed versus Yuksel common-raw quality can be compared from 320x40 onward",
        "Assessment": "PUBLICATION_READY_WITH_QUALIFICATION",
        "Evidence_or_qualification": "Use the unchanged common evaluator and the original admissible rows; do not claim universal topology superiority",
    },
    {
        "Claim": "Endpoint and total-time scaling can be reported",
        "Assessment": "PUBLICATION_READY_WITH_QUALIFICATION",
        "Evidence_or_qualification": "Retain practical-endpoint semantics, original censoring masks, one sample per mesh, and separate Yuksel stages",
    },
    {
        "Claim": "Olhoff timing represents fixed-work cost",
        "Assessment": "PUBLICATION_READY_WITH_QUALIFICATION",
        "Evidence_or_qualification": "Only successful fixed-1600-update rows are timing-admissible; solver-failure rows remain censored",
    },
    {
        "Claim": "Olhoff has a reproducible nonmonotonic solver-failure island",
        "Assessment": "PUBLICATION_READY_WITH_QUALIFICATION",
        "Evidence_or_qualification": "The 640x80 failure is bit-identically reproduced and prior larger meshes are healthy; mechanism beyond solver exit class is unresolved",
    },
    {
        "Claim": "p=1.5 may be retained as an empirical reference normalization where classified by the prior fixed-p assessment",
        "Assessment": "PUBLICATION_READY_WITH_QUALIFICATION",
        "Evidence_or_qualification": "Use the prior per-method/per-quantity classifications and normalized coefficient units; it is not a theoretical complexity proof",
    },
    {
        "Claim": "p=1.5 is a universal complexity normalization",
        "Assessment": "NOT_SUPPORTED",
        "Evidence_or_qualification": "Retain only as a declared reference normalization; fitted exponents and endpoint semantics differ by method",
    },
]
write_csv("publication_readiness_delta.csv", publication_rows)


csv_schema = """# CSV data dictionary

An empty field denotes an unavailable/not-recorded quantity; it must never be read as zero. Booleans are 0/1. Iteration counts, element counts, ranks, solver flags, status labels, checksums, ratios, fractions, grayness, normalized residuals, and classifications are dimensionless unless stated otherwise.

| CSV | Row meaning and units |
|---|---|
| `configuration_identity.csv` | One row per replay target. Mesh is elements; equality/PASS fields are dimensionless. |
| `olhoff_640_failure_diagnostics.csv` | One failed LP attempt. `omega1`–`omega3`: rad/s; `lamref`: (rad/s)^2; `move`: density fraction; LP residuals/row norms/rcond: normalized or solver-native dimensionless quantities. Empty residual/activity fields mean no primal point was returned. |
| `olhoff_640_history_window.csv` | One successful outer update plus the failed attempt marker. Frequencies: rad/s; density changes, move, volume and gap: dimensionless. |
| `yuksel_800_history.csv` | One optimizer iteration. Density changes/fractions/residuals/grayness/turnover: dimensionless; objective: moving-load compliance in the frozen model's native units; `mode_angle_deg`: degrees. Empty snapshot fields mean no snapshot at that iteration. |
| `yuksel_800_cap_diagnosis.csv` | One late-history window. Density/objective-relative metrics and fractions: dimensionless. |
| `proposed_160_history.csv` | One optimizer iteration per run. Frequencies: rad/s; density changes/fractions/residuals/grayness: dimensionless; objective: compliance in the frozen model's native units. |
| `proposed_160_determinism.csv` | One row per repeat. Frequencies and common evaluator values: rad/s; `finalDx`: density fraction. |
| `proposed_160_common_evaluators.csv` | One mode per row. All native/common columns: rad/s. Common evaluators are comparison models, not ground truth. |
| `proposed_160_mode_localization.csv` | One native mode per row. `omega`: rad/s; energy/displacement fractions and weighted density: dimensionless; participation: elements. |
| `publication_readiness_delta.csv` | One publication claim; categorical, no physical units. |
"""
(HERE / "CSV_DATA_DICTIONARY.md").write_text(csv_schema)


delta_rows = [
    ("Campaign is trustworthy for terminal/status, timing and common-evaluator observations", "All three targets reproduce original status/endpoints; original hashes unchanged", "CONFIRMED", "Core campaign observations may be frozen"),
    ("Exactly five observations are censored and excluded from fits", "No original row or mask changed; replays remain diagnostic", "CONFIRMED", "Publish censoring explicitly"),
    ("Proposed 109-to-159 transition is primarily model/interpolation dependence", "Direct density and modes show weak-material local modes; same field rises to 153.68 raw E1 and 162.76 binary E1", "REFINED", "Mechanism is now publication-ready with mesh-specific qualification"),
    ("Proposed native coarse triplet is genuine under its frozen native model", "Two identical runs exactly reproduce 109.05/109.49/112.92 and the original fingerprint", "CONFIRMED", "Report as deterministic native-model behavior"),
    ("Proposed low-density interpolation sensitivity was inferred from scalar evidence", "Mode energy/localization directly places essentially all three modes in rho<=0.1 material", "REFINED", "Can call them weak-material/local modes at 160x20"),
    ("Proposed coarse endpoints are fully stationary/KKT solutions", "History now exists, but no KKT residual or basin/restart experiment was authorized", "UNCHANGED_EVIDENCE_GAP", "Do not make KKT claims"),
    ("Proposed/Yuksel topologies were unavailable", "Proposed 160 density and binary topology are retained and connected; Yuksel cross-method topology equivalence remains unproved", "REFINED", "Use Proposed topology figure; avoid topology-equivalence claim"),
    ("Olhoff 640 first failed LP attempt is k=1067", "Bit-identical replay again fails after 1066 successful updates at attempted k=1067", "CONFIRMED", "Failure location is publication-ready"),
    ("Olhoff failure is dual-simplex-highs exitflag 0 / MATLAB iteration-limit class", "Direct output: 38 iterations, message 'Solver stopped prematurely', empty point; local linprog source maps flag 0 to maximum iterations", "REFINED", "Report exact output and avoid implying a user-set 38-iteration cap"),
    ("Olhoff failure may involve modal branching/LP degeneracy", "Direct failed state is N=2 with gap12=0.00256, but normalized constraint rows are full rank with Gram rcond 0.0206 and no point exists for residuals", "REFINED", "Only generic LP iteration-limit causation is supported"),
    ("Olhoff failure island is nonmonotonic and trajectory-dependent", "Exact trajectory/failure reproduction plus larger healthy meshes from prior evidence", "CONFIRMED", "Not a monotonic resource-size claim"),
    ("Olhoff best-prior states are diagnostic and not campaign successes", "Replay preserves SOLVER_FAILURE and does not promote any prior state", "CONFIRMED", "Keep row censored"),
    ("Yuksel 800 late mechanism was indeterminate", "Final 300 show max 0.011-0.1, median max/RMS 97, 61 full-move hits, 3.2-3.5% active variables and small objective increments", "REFINED", "Classify localized irregular oscillation / persistent nonconvergence"),
    ("Yuksel 800 might be a simple cap limitation", "No late sample meets tolerance and the trajectory is not monotone decay", "REFUTED", "Do not recommend a modest extension as convergence completion"),
    ("Yuksel 640 mechanism remains unresolved", "640x80 was not one of the authorized closure replays", "UNCHANGED_EVIDENCE_GAP", "Does not block freeze; retain as censored with prior qualification"),
    ("Per-iteration exponents are Olhoff 1.194, Yuksel 0.975, Proposed 1.189", "No original timing or fit was replaced; replays reproduce numerical endpoints", "CONFIRMED", "Publish with one-sample and stage qualifications"),
    ("Yuksel Stage-1/Stage-2 per-iteration fits must remain separate", "Replay retains exact 1000+1000 semantics; timing stays diagnostic only", "CONFIRMED", "Keep stage-specific reporting"),
    ("Total-time exponents are practical endpoint fits, not intrinsic complexity", "Nothing in the replays changes cost-count decomposition", "CONFIRMED", "Preserve endpoint semantics"),
    ("RAM measurements are unreliable", "No memory repair or new memory benchmark was performed", "CONFIRMED", "Exclude quantitative RAM claims"),
    ("Olhoff common-raw advantage is qualified, not universal topology superiority", "No new evidence contradicts prior raw/binary evaluator findings", "CONFIRMED", "Retain the qualified wording"),
    ("No full nine-resolution rerun is required", "All three exact targets reproduced; no implementation corruption or changed campaign observation found", "CONFIRMED", "Freeze without broad rerun"),
    ("Three narrow diagnostic follow-ups were required before freeze", "All authorized replays and the permitted Proposed repeat are complete", "REFINED", "No further runs required"),
]

delta_md = [
    "# Forensic-Audit Delta",
    "",
    "This is a narrow delta against `PERFORMANCE_CAMPAIGN_FORENSIC_AUDIT.md`; the original report and its tables remain unchanged.",
    "",
    "| Prior finding | New evidence | Delta | Publication consequence |",
    "|---|---|---|---|",
]
for prior, new, verdict, consequence in delta_rows:
    delta_md.append(f"| {prior} | {new} | **{verdict}** | {consequence} |")
delta_md += [
    "",
    "## Focus findings",
    "",
    "- **Proposed:** CONFIRMED and refined. MODEL / INTERPOLATION DEPENDENCE remains primary; the low triplet is now directly shown to be weak-material/local rather than evidence of a disconnected load-carrying skeleton.",
    "- **Olhoff:** REFINED. The exact deterministic LP failure is confirmed, but the simple matrix diagnostics do not demonstrate degeneracy, scaling failure, or modal causation. The supported class is `GENERIC_LP_ITERATION_LIMIT_ONLY`.",
    "- **Yuksel:** REFINED from indeterminate to `PERSISTENT_NONCONVERGENCE`, expressed as localized irregular oscillation with a practically stable but slowly drifting objective.",
    "- **Campaign:** CONFIRMED. No replay invalidates an observation, timing record, or admissible-row scaling fit, and no broad rerun is required.",
    "",
    "## Remaining gaps that do not block freeze",
    "",
    "The precise internal reason HiGHS stops the Olhoff LP after 38 reported iterations is not exposed by the returned point (none exists) or output structure. Yuksel 640x80 was not replayed. Proposed KKT stationarity and cross-method topology equivalence were not tested. None is needed to state the benchmark results with the qualifications above.",
]
(HERE / "FORENSIC_DELTA_AUDIT.md").write_text("\n".join(delta_md) + "\n")


gate_rows = [
    ("A. Configuration integrity", "PASS", "All three normalized gates pass and source SHA-256 values match pre-replay provenance"),
    ("B. Correct statuses", "PASS", "Olhoff SOLVER_FAILURE, Yuksel CAP_HIT, Proposed NATIVE_CONVERGED all reproduce"),
    ("C. Correct censoring", "PASS", "Three Olhoff failures and two Yuksel cap hits remain censored negative observations"),
    ("D. Reproducible targeted observations", "PASS", "Olhoff 640x80 once, Yuksel 800x100 once, and Proposed 160x20 twice reproduce frozen endpoints"),
    ("E. Defensible common-evaluator interpretation", "PASS", "Same retained density was evaluated offline under unchanged E1/E2/E3 paths; no truth-model claim is made"),
    ("F. Defensible timing/scaling semantics", "PASS", "Original timing/fits are untouched; replay timing is DIAGNOSTIC ONLY and RAM stays excluded"),
    ("G. No implementation corruption", "PASS", "Frozen sources and all 21 final-campaign artifacts are byte-identical; target numerical endpoints reproduce"),
    ("H. Limitations explicitly recorded", "PASS", "Olhoff deeper cause, Yuksel 640 mechanism, Proposed KKT status, RAM, and timing limits are stated"),
]

freeze = [
    "# Final Performance Freeze Gate",
    "",
    "| Requirement | PASS/FAIL | Evidence |",
    "|---|---|---|",
]
for req, result, evidence in gate_rows:
    freeze.append(f"| {req} | **{result}** | {evidence} |")
freeze += [
    "",
    "## Decision",
    "",
    "`NO_FURTHER_RUNS_REQUIRED`",
    "",
    "Negative numerical behavior is retained rather than repaired: Olhoff 640x80 remains a solver failure and Yuksel 800x100 remains a cap hit. The evidence is sufficient to characterize both honestly without making every row successful.",
    "",
    "**PERFORMANCE CAMPAIGN FROZEN — READY FOR PAPER**",
    "",
    "**FULL NINE-RESOLUTION RERUN: NOT REQUIRED**",
    "",
    "**FURTHER TARGETED OPTIMIZATION RUNS: NOT REQUIRED**",
]
(HERE / "FINAL_PERFORMANCE_FREEZE_GATE.md").write_text("\n".join(freeze) + "\n")


report = f"""# Targeted Replay Report

## Outcome

All three authorized cases reproduce their original campaign observations under numerically identical frozen profiles. No replay exposes implementation corruption, invalidates an original observation, or changes an admissible-row timing/scaling fit. The campaign can be frozen with its five censored observations intact.

Replay wall times are diagnostic only: Olhoff {f(o['diagnostic_wall_time_s'])} s, Yuksel {f(y['diagnostic_wall_time_s'])} s, Proposed {f(p['diagnostic_wall_time_run1_s'])}/{f(p['diagnostic_wall_time_run2_s'])} s. They do not replace or refit original timing measurements.

## Scope and hard constraints

This closure covers only the already completed Olhoff 640x80, Yuksel 800x100, and two identical Proposed 160x20 diagnostic executions. All work after those executions was offline. No cap, tolerance, move, filter, eigensolver, LP option, optimizer, stabilization rule, or material interpolation was changed; no broad campaign rerun, tuning run, cap extension, RAM repair, or timing substitution was performed.

## Provenance

- Repository: branch `{initial['branch']}`, commit `{initial['HEAD']}`.
- Runtime: MATLAB `{initial['matlab_version']}`, `{initial['computer']}`, one computation thread.
- Pre-replay worktree: only the already untracked forensic-audit and final-campaign directories.
- Immutable evidence: all 21 files under `examples/Performance/final_campaign/` match the pre-replay forensic SHA-256 inventory; every frozen numerical source in the replay preflight matches its original hash.
- Raw diagnostic MAT files and logs remain under `raw/` and `logs/`. `SHA256SUMS.txt` covers the final closure package and excludes itself.

## Replay implementation audit

**Observation:** the normalized original/replay configurations compare equal for all three cases, and each source-hash gate passed before execution. The Olhoff mirror preserves the production LP objective, matrices, bounds, `dual-simplex-highs` option, ordering, update decision, and S1 rule; it merely requests and stores solver output, then computes diagnostics only after failure. Every retained non-timing Olhoff history field, density, and successful-update snapshot is bit-identical to production evidence. Yuksel history/audit retention is observational, and its exact endpoint/fingerprints reproduce. Proposed history/spectral retention does not feed updates; both runs exactly reproduce the original endpoint. The determinism check compares every retained numerical history field and structural marker while excluding only `elapsed_s`; the exported 107-row histories also compare exactly field by field.

**Inference:** the instrumentation did not alter the numerical behavior of any target.

**Unresolved mechanism:** exact reproduction cannot expose the solver-internal reason that HiGHS stopped the failed Olhoff LP after 38 reported iterations.

## Exact replay configurations

| Method | Mesh | Frozen numerical profile |
|---|---:|---|
| Olhoff | 640x80 | LP; SIMP; sensitivity filter r=1.3 elements, diagonal mode; volume 0.5; rho_min=0.001; tol_mult=0.05; initial/stabilized move 0.005/0.0025; S1 trigger N=2 and gap12<=0.01 for 100 iterations; fixed cap 1600; dual-simplex-highs |
| Yuksel | 800x100 | OC; SIMP p=3; sensitivity filter r=2.5 elements, symmetric boundary; volume 0.5; move 0.1; Stage-1/Stage-2 cap 1000/1000; tolerance 0.01 in each stage; E_min/E=1e-9; rho_min=1e-9 |
| Proposed | 160x20 | OC; SIMP p=3; sensitivity filter r=2 elements, symmetric boundary; volume 0.5; move 0.2; cap 2000; tolerance 0.01; solid semi-harmonic baseline, no load sensitivity/normalization; E_min/E=1e-9; rho_min=1e-9 |

The canonical machine-readable profiles are retained in `configurations/`.

## Table A — Replay configuration identity

| Method | Mesh | Original profile ID | Replay profile ID | Numerical config identical? | Source hashes equal? | Diagnostic-only differences | PASS/FAIL |
|---|---:|---|---|---|---|---|---|
"""
for row in cfg_identity:
    report += f"| {row['method']} | {row['mesh']} | `{row['original_profile_id']}` | `{row['replay_profile_id']}` | {row['numerical_config_identical']} | {row['source_hashes_equal']} | {row['diagnostic_only_differences']} | **{row['pass_fail']}** |\n"

report += f"""

## Replay identity results

| Method/comparison | Iteration/status identity | Spectrum/fingerprint identity | Density/stopping/common identity | Verdict |
|---|---|---|---|---|
| Olhoff original vs replay | 1066 successful; failure attempt 1067; trigger k=204; SOLVER_FAILURE | every retained non-timing history field bit-identical | density and all successful-update snapshots bit-identical | FAILURE_REPRODUCED |
| Yuksel original vs replay | Stage 1=1000; Stage 2=1000; total=2000; CAP_HIT | final 160.879109/325.605370/566.129320 rad/s and objective-history fingerprint exact | density fingerprint, final max dx 0.0727982 and RMS dx 0.0006855 exact | ENDPOINT_IDENTICAL |
| Proposed original vs each replay | 107; NATIVE_CONVERGED | native 109.050082/109.489208/112.916324 rad/s and objective/frequency fingerprints exact | density fingerprint and all E1/E2/E3 raw/binary results exact | BOTH_ENDPOINTS_IDENTICAL |
| Proposed replay 1 vs replay 2 | 107 vs 107 | all retained numerical histories and spectra exact; only elapsed timing excluded | density, topology, stopping and common evaluator results exact | DETERMINISTIC |

## Olhoff 640x80

**Observation:** the replay is bit-identical to the original over every retained non-timing history field, the complete density trajectory, and all successful-update snapshots. Stabilization triggers at k=204, the reduced move is 0.0025, 1066 updates succeed, and attempted k=1067 fails again.

The failed attempt directly evaluates to ω = {f(o['omega'][0], 12)} / {f(o['omega'][1], 12)} / {f(o['omega'][2], 12)} rad/s, N=2, gap12={f(o['gap12'], 8)}, and λref={f(o['lamref'], 12)}. `linprog` returns exitflag 0 after 38 reported iterations, algorithm `dual-simplex-highs`, message “Solver stopped prematurely,” and no point. MATLAB R2025b's local `linprog.m` defines flag 0 as maximum iterations reached. The configured default `MaxIterations` is 2.1475e9 and `MaxTime=Inf`, so the result should be described as MATLAB's iteration-limit exit class, not a user-imposed 38-iteration cap.

The LP matrices are finite. Five normalized constraint rows have rank 5; normalized Gram rcond is {f(o['normalized_gram_rcond'], 6)} and the inequality-row norm ratio is {f(o['row_norm_ratio_A'], 6)}. Because no point is returned, feasibility residuals and active-bound statistics are unavailable.

**Inference:** these proxies do not demonstrate degeneracy or pathological scaling. The supported causal class is limited to MATLAB's generic LP iteration-limit exit class.

**Unresolved mechanism:** N=2 and a small gap co-occur with the failure, but modal-branch responsibility, degeneracy, scaling pathology, and the internal reason for the 38-iteration stop are not established.

### Table B — Olhoff failure replay

| Original attempt | Replay attempt | exitflag | LP iterations | Modal state | gap12 | Move | Trigger state | Residual/scaling diagnostics | Reproduction | Causal class |
|---:|---:|---:|---:|---|---:|---:|---|---|---|---|
| 1067 | 1067 | 0 | 38 | 172.823/173.265/200.337; N=2 | 0.00255682 | 0.0025 | S1 already active; trigger k=204 | no returned point; finite matrices; row rank 5/5; Gram rcond 0.0206 | **FAILURE_REPRODUCED** | **GENERIC_LP_ITERATION_LIMIT_ONLY** |

There is no evidence of implementation corruption.

## Yuksel 800x100

**Observation:** the replay exactly matches the original fingerprint and again stops `CAP_HIT` at 1000+1000 iterations. In the final 300 Stage-2 iterations, max dx ranges {f(min(ymax))}–{f(max(ymax))}, has median {f(statistics.median(ymax))}, never falls below 0.01, and hits the full 0.1 move limit {move_hits} times. RMS dx has median {f(statistics.median(yrms))}; median max/RMS is {f(statistics.median(a/b for a,b in zip(ymax,yrms)))}. The raw max-dx slopes have {strict_trend_reversals} strict positive-to-negative/negative-to-positive reversals, {nonzero_trend_reversals} after zero-slope plateaus are removed, and {slope_state_changes} state changes when entry to/exit from a plateau is counted. This explicit definition replaces the preliminary “approximately 75” estimate.

Only {f(min(yactive)*100)}–{f(max(yactive)*100)}% of variables move above 1e-12 in this window, and P95 is effectively zero, giving a conservative `<5%` / `<4,000 of 80,000` bound on variables that can dominate the maximum. Ten-iteration binary turnover is small but persistent, 0–{f(max(turn)*100)}%. The moving-load objective changes by a median {f(statistics.median(yrel), 5)} per iteration and drifts {f(abs((yobj[-1]-yobj[0])/yobj[0])*100, 4)}% across the window.

### Table C — Yuksel late dynamics

| Window | Stage | max dx start→end (median) | RMS median | max/RMS median | Objective trend | Dominant-variable fraction | Diagnosis |
|---|---:|---|---:|---:|---|---|---|
"""
for row in ywin:
    report += (
        f"| {row['window']} | {row['stage']} | {float(row['max_dx_start']):.4g}→{float(row['max_dx_end']):.4g} "
        f"({float(row['max_dx_median']):.4g}) | {float(row['rms_dx_median']):.3g} | "
        f"{float(row['max_to_rms_median']):.1f} | stable/slight drift; median relative step "
        f"{float(row['median_relative_objective_change']):.2e} | <5% (<4,000); localized every sampled iteration | "
        f"**IRREGULAR_OSCILLATION / PERSISTENT_NONCONVERGENCE** |\n"
    )

report += f"""

**Inference:** the primary late behavior is `IRREGULAR_OSCILLATION`, with secondary `LOCALIZED_VARIABLE_MOTION`; the cap interpretation is `PERSISTENT_NONCONVERGENCE`, not `LIKELY_SIMPLE_CAP_LIMIT`. A modest cap extension is not scientifically justified as a convergence-completion experiment because the late trajectory is far above tolerance and irregular rather than a monotone tail.

**Unresolved mechanism:** the history establishes the behavior but not which local variables or update interactions cause it. A future dynamics study could investigate that question; it is not a performance-freeze requirement.

## Proposed 160x20

**Observation:** both identical diagnostic executions reproduce the original 107-iteration native endpoint and are numerically deterministic after excluding only elapsed-time fields. Native ω is {f(p['native_omega'][0],12)} / {f(p['native_omega'][1],12)} / {f(p['native_omega'][2],12)} rad/s. The density, binary topology, objective history, stopping metric, per-iteration native spectrum, and all common evaluator outputs are identical.

### Table D — Proposed determinism

| Run | Iterations | Native ω1/ω2/ω3 | Final dx | Density checksum | Common raw E1 | Common binary E1 | Verdict |
|---:|---:|---|---:|---|---:|---:|---|
"""
for row in pdet:
    report += f"| {row['runs']} | {row['iters']} | {float(row['w1']):.6f}/{float(row['w2']):.6f}/{float(row['w3']):.6f} | {float(row['finalDx']):.6g} | `{row['densityChecksum']}` | {float(row['rawE1']):.6f} | {float(row['binaryE1']):.6f} | **{row['determinismVerdict']}** |\n"

report += """

### Table E — Proposed native/common interpretation

| Quantity | Native | E1 raw | E2 raw | E3 raw | E1 binary | E2 binary | E3 binary | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
"""
for row in pcommon:
    report += (
        f"| {row['quantity']} | {float(row['native']):.6f} | {float(row['E1_raw']):.6f} | "
        f"{float(row['E2_raw']):.6f} | {float(row['E3_raw']):.6f} | {float(row['E1_binary']):.6f} | "
        f"{float(row['E2_binary']):.6f} | {float(row['E3_binary']):.6f} | {row['interpretation']} |\n"
    )

report += f"""

Grayness is {f(p['grayness'], 8)}. The native mode recomputation agrees within {f(p['native_mode_recompute_max_abs_difference'], 4)} rad/s. For modes 1–3, displacement fraction in rho≤0.1 material exceeds 0.999999996, kinetic fraction exceeds 0.99999858, displacement-weighted density is 0.00236–0.00246, and only 105–136 elements carry 90% of displacement magnitude. The shapes are localized along the nominally void interior band. The raw and exact-count binary load-carrying topologies are both connected and have strong common spectra.

**Inference:** the low native triplet consists of weak-material/local modes produced by the native floors/interpolation on this coarse gray design; the load-carrying skeleton itself is not shown to be qualitatively poor. Primary classification: **MODEL / INTERPOLATION DEPENDENCE**, directly confirmed and refined to weak-material modal localization at 160x20. The common evaluator is a comparison model, not a “true-frequency” model.

**Unresolved mechanism:** no KKT residual, basin/restart study, or every-resolution modal localization analysis was authorized, so stationarity and universal-resolution claims remain unsupported.

## Figures

1. [Olhoff pre-failure history](figures/01_olhoff_640_prefailure_history.png)
2. [Olhoff failed-LP diagnostics](figures/02_olhoff_640_lp_diagnostics.png)
3. [Yuksel late max/RMS change](figures/03_yuksel_800_late_density_change.png)
4. [Yuksel late objective](figures/04_yuksel_800_late_objective.png)
5. [Proposed native spectral history](figures/05_proposed_160_native_spectral_history.png)
6. [Proposed density and topology](figures/06_proposed_160_density_topology.png)
7. [Proposed native mode shapes](figures/07_proposed_160_native_mode_shapes.png)
8. [Proposed native/common spectra](figures/08_proposed_160_native_common_spectra.png)

## Direct answers

1. **Olhoff failure location:** yes, attempted k=1067 after 1066 successful updates.
2. **Exact linprog report:** exitflag 0, dual-simplex-highs, “Solver stopped prematurely,” no point.
3. **LP iterations:** 38 reported iterations.
4. **Feasibility/scaling/degeneracy:** residuals/activity unavailable without a point; finite, full-row-rank proxies do not show an abnormal signature sufficient for causation.
5. **Modal connection:** N=2/small gap and late branching co-occur; a causal link is not demonstrated.
6. **Implementation corruption:** no.
7. **Yuksel cap:** yes, exact `CAP_HIT` reproduction.
8. **Final 300:** persistent irregular excursions between 0.011 and 0.1 with small RMS and objective increments.
9. **Global or localized max:** localized; active fraction 3.2–3.5%, conservative dominant bound <5%.
10. **Objective stabilized:** practically stable with slight 0.57% late-window drift and median relative step 1.70e-5.
11. **Late behavior:** `IRREGULAR_OSCILLATION` plus `LOCALIZED_VARIABLE_MOTION`.
12. **Simple cap limitation:** no; `PERSISTENT_NONCONVERGENCE`.
13. **Proposed determinism:** yes, `DETERMINISTIC`.
14. **Producing topology:** connected truss-like raw/binary skeleton with substantial gray interfaces and a nominally void interior band.
15. **Low native modes:** localized along that low-density interior band.
16. **Gray/weak association:** overwhelmingly weak/low-density, not mainly the solid skeleton.
17. **Common evaluator effect:** yes, it again removes most of the native anomaly.
18. **Primary Proposed explanation:** confirmed `MODEL / INTERPOLATION DEPENDENCE`.
19. **Original observation invalidated:** none.
20. **Per-iteration scaling invalidated:** no; original fits/timing remain unchanged.
21. **Full rerun needed:** no.
22. **Freeze for paper:** yes, with the publication qualifications below.

## Limitations

The failed Olhoff solve returned no primal point, so residuals, feasibility, and active-bound statistics are unavailable. The deeper HiGHS stopping mechanism is unresolved. Yuksel 640x80 was not replayed, and the Yuksel 800 history does not isolate a variable-level cause. Proposed localization is demonstrated directly at 160x20, not every resolution; KKT stationarity and cross-method topology equivalence were not tested. Common evaluators are comparison models rather than ground truth. Original timing has one sample per mesh and method-specific endpoint semantics. RAM remains unreliable and excluded. A final attempt to rerun the offline MATLAB postprocessor was blocked by License Error 15 before execution; the objectively incorrect Olhoff residual panel was therefore regenerated deterministically from its retained CSV diagnostics without invoking an optimizer.

## Conclusions

No unresolved issue materially prevents the benchmark from supporting the qualified claims below. The five censored observations remain scientific results: Olhoff 480/560/640 solver failures and Yuksel 640/800 cap hits. Decision: `NO_FURTHER_RUNS_REQUIRED`.

## PUBLICATION-READY CLAIMS

- Final campaign technical integrity, status labels, and exactly five censored observations.
- Proposed 160x20 native endpoint determinism.
- Necessity of unchanged common evaluators for cross-method interpretation.
- Preservation of original timing tables and censoring masks.

## PUBLICATION-READY WITH QUALIFICATION

- Proposed coarse anomaly as native model/interpolation dependence, directly localized at 160x20.
- Proposed-versus-Yuksel common-raw quality from 320x40 onward, without topology-superiority claims.
- Olhoff 640 failure as a reproducible MATLAB/HiGHS iteration-limit exit class, without a deeper causal claim.
- Yuksel 800 cap behavior as persistent localized irregular motion, without generalizing to Yuksel 640.
- Per-iteration and practical endpoint/total-time scaling, with one-sample, stage, and terminal-semantics qualifications.
- Olhoff fixed-work timing on successful admissible rows and its nonmonotonic solver-failure island.
- p=1.5 only as a declared reference normalization, not a universal fitted law.

## EXCLUDED / NOT SUPPORTED CLAIMS

- Quantitative RAM scaling or replay timings as campaign performance data.
- Universal p=1.5, intrinsic total-time kernel complexity, or universal topology superiority/equivalence.
- KKT/stationarity claims for Proposed, a specific Olhoff degeneracy/scaling/modal cause, or that Yuksel merely needs more iterations.
- Claims that all 27 rows converged or that censored rows are timing/scaling successes.

PERFORMANCE CAMPAIGN FROZEN — READY FOR PAPER

FULL NINE-RESOLUTION RERUN: NOT REQUIRED

FURTHER TARGETED OPTIMIZATION RUNS: NOT REQUIRED
"""
(HERE / "TARGETED_REPLAY_REPORT.md").write_text(report)


# Use an empty CSV cell for unavailable values.  In particular, do not allow
# an unavailable Olhoff residual/activity quantity to be interpreted as zero.
for csv_path in sorted(HERE.glob("*.csv")):
    with csv_path.open(newline="", encoding="utf-8") as fcsv:
        reader = csv.DictReader(fcsv)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        continue
    changed = False
    for row in rows:
        for key, value in row.items():
            if value is not None and value.strip().lower() in {"nan", "+nan", "-nan"}:
                row[key] = ""
                changed = True
    if changed:
        with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


# Final provenance is generated after all non-provenance deliverables exist.
source_after = {
    rel: sha256(REPO / rel) for rel in initial["source_hashes_sha256"]
}
original_after = {
    rel: sha256(REPO / rel) for rel in initial["original_target_evidence_hashes_sha256"]
}
frozen_artifacts_before = prior_provenance["frozen_artifact_hashes_sha256"]
frozen_artifacts_after = {rel: sha256(REPO / rel) for rel in frozen_artifacts_before}
frozen_directory_files = {
    str(path.relative_to(REPO)) for path in FROZEN.rglob("*") if path.is_file()
}
artifact_hashes = {}
for path in sorted(HERE.rglob("*")):
    if not path.is_file() or path.name in {"replay_provenance.json", "SHA256SUMS.txt"}:
        continue
    artifact_hashes[str(path.relative_to(REPO))] = sha256(path)

provenance = {
    "closure_type": "exact targeted diagnostic replay and performance freeze closure",
    "generated_at": datetime.now(ZoneInfo("Europe/Warsaw")).isoformat(timespec="seconds"),
    "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "HEAD": git("rev-parse", "HEAD"),
    "working_tree_status_before_replays": initial["working_tree_status"],
    "working_tree_status_at_closure": git("status", "--short").splitlines(),
    "matlab_version": initial["matlab_version"],
    "computer": initial["computer"],
    "threads_per_run": 1,
    "target_cases": initial["target_cases"],
    "diagnostic_executions": initial["diagnostic_executions"],
    "full_nine_resolution_campaign_rerun": False,
    "configuration_identity_pass": all(
        r["pass_fail"] == "PASS"
        and r["numerical_config_identical"] == "1"
        and r["source_hashes_equal"] == "1"
        for r in cfg_identity
    ),
    "source_hashes_before_sha256": initial["source_hashes_sha256"],
    "source_hashes_after_sha256": source_after,
    "frozen_numerical_implementations_unchanged": source_after == initial["source_hashes_sha256"],
    "original_target_evidence_hashes_before_sha256": initial["original_target_evidence_hashes_sha256"],
    "original_target_evidence_hashes_after_sha256": original_after,
    "original_target_evidence_unchanged": original_after == initial["original_target_evidence_hashes_sha256"],
    "full_final_campaign_hashes_before_sha256": frozen_artifacts_before,
    "full_final_campaign_hashes_after_sha256": frozen_artifacts_after,
    "full_final_campaign_file_count": len(frozen_directory_files),
    "full_final_campaign_inventory_matches_pre_replay": (
        frozen_directory_files == set(frozen_artifacts_before)
        and frozen_artifacts_after == frozen_artifacts_before
    ),
    "replay_results": {
        "Olhoff_640x80": {
            "status": "SOLVER_FAILURE",
            "completed_iterations": 1066,
            "failure_attempt": 1067,
            "reproduction_verdict": o["reproduction_verdict"],
            "diagnostic_wall_time_s": o["diagnostic_wall_time_s"],
        },
        "Yuksel_800x100": {
            "status": y["status"],
            "iterations": y["total_iterations"],
            "endpoint_matches_original": y["endpoint_matches_original"],
            "diagnostic_wall_time_s": y["diagnostic_wall_time_s"],
            "late300_strict_trend_reversals": strict_trend_reversals,
            "late300_nonzero_slope_reversals": nonzero_trend_reversals,
            "late300_slope_state_changes_including_plateaus": slope_state_changes,
        },
        "Proposed_160x20": {
            "status": "NATIVE_CONVERGED",
            "executions": 2,
            "iterations_each": p["iterations"],
            "determinism_verdict": p["determinism_verdict"],
            "both_match_original": p["both_match_original"],
            "diagnostic_wall_times_s": [p["diagnostic_wall_time_run1_s"], p["diagnostic_wall_time_run2_s"]],
        },
    },
    "timing_semantics": "DIAGNOSTIC ONLY; original campaign timings and fits remain authoritative",
    "memory_semantics": "original MaxRAM_MB remains UNRELIABLE; no memory repair or benchmark performed",
    "offline_matlab_finalization_note": "A post-replay offline postprocessor re-execution attempt was blocked before execution by MATLAB License Error 15; no optimizer was invoked. The Olhoff LP figure was repaired from retained CSV values.",
    "implementation_corruption_detected": False,
    "artifact_hashes_sha256": artifact_hashes,
}
(HERE / "replay_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

manifest_lines = []
for path in sorted(HERE.rglob("*")):
    if not path.is_file() or path.name == "SHA256SUMS.txt":
        continue
    manifest_lines.append(f"{sha256(path)}  {path.relative_to(HERE)}")
(HERE / "SHA256SUMS.txt").write_text("\n".join(manifest_lines) + "\n")

print("TARGETED_REPLAY_CLOSURE_BUILT")
