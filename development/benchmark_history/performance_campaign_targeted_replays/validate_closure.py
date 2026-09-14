#!/usr/bin/env python3
"""Independent, read-only validation of the targeted-replay closure package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import struct
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
errors: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def read_csv(name: str) -> list[dict[str, str]]:
    try:
        with (HERE / name).open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            check(reader.fieldnames is not None, f"{name}: missing header")
            return list(reader)
    except Exception as exc:  # noqa: BLE001 - validation must report every defect
        errors.append(f"{name}: parse failure: {exc}")
        return []


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


required = [
    "TARGETED_REPLAY_REPORT.md",
    "replay_provenance.json",
    "configuration_identity.csv",
    "olhoff_640_failure_diagnostics.csv",
    "yuksel_800_history.csv",
    "yuksel_800_cap_diagnosis.csv",
    "proposed_160_history.csv",
    "proposed_160_determinism.csv",
    "proposed_160_common_evaluators.csv",
    "FORENSIC_DELTA_AUDIT.md",
    "publication_readiness_delta.csv",
    "FINAL_PERFORMANCE_FREEZE_GATE.md",
    "CSV_DATA_DICTIONARY.md",
    "SHA256SUMS.txt",
]
for name in required:
    check((HERE / name).is_file(), f"missing required artifact: {name}")

expected_rows = {
    "configuration_identity.csv": 3,
    "olhoff_640_failure_diagnostics.csv": 1,
    "olhoff_640_history_window.csv": 101,
    "yuksel_800_history.csv": 2000,
    "yuksel_800_cap_diagnosis.csv": 4,
    "proposed_160_history.csv": 214,
    "proposed_160_determinism.csv": 2,
    "proposed_160_common_evaluators.csv": 3,
    "proposed_160_mode_localization.csv": 3,
    "publication_readiness_delta.csv": 25,
}
tables: dict[str, list[dict[str, str]]] = {}
for name, expected in expected_rows.items():
    rows = read_csv(name)
    tables[name] = rows
    check(len(rows) == expected, f"{name}: expected {expected} data rows, found {len(rows)}")
    for row_number, row in enumerate(rows, start=2):
        for column, value in row.items():
            check(value is not None, f"{name}:{row_number}:{column}: malformed CSV row")
            if value is not None:
                check(value.strip().lower() not in {"nan", "+nan", "-nan"},
                      f"{name}:{row_number}:{column}: literal NaN must be an empty unavailable field")

cfg = tables["configuration_identity.csv"]
check(all(r.get("pass_fail") == "PASS" for r in cfg), "configuration gate did not PASS")
check(all(r.get("numerical_config_identical") == "1" for r in cfg), "numerical configuration mismatch")
check(all(r.get("source_hashes_equal") == "1" for r in cfg), "source hash mismatch")

olp = tables["olhoff_640_failure_diagnostics.csv"]
if olp:
    row = olp[0]
    check(row["original_failure_attempt"] == row["replay_failure_attempt"] == "1067", "Olhoff failure attempt mismatch")
    check(row["linprog_exitflag"] == "0" and row["lp_iterations"] == "38", "Olhoff solver output mismatch")
    check(row["reproduction_verdict"] == "FAILURE_REPRODUCED", "Olhoff reproduction mismatch")
    check(row["causal_classification"] == "GENERIC_LP_ITERATION_LIMIT_ONLY", "Olhoff causal class mismatch")
    unavailable = [
        "max_inequality_violation",
        "max_equality_residual",
        "max_lower_bound_violation",
        "max_upper_bound_violation",
        "active_bound_fraction",
    ]
    check(all(row[name] == "" for name in unavailable), "Olhoff unavailable residual/activity fields are not empty")
    check(row["returned_point_finite"] == "0", "Olhoff unexpectedly reports a returned primal point")

yhist = tables["yuksel_800_history.csv"]
if len(yhist) == 2000:
    late = yhist[-300:]
    max_dx = [float(r["max_dx"]) for r in late]
    rms_dx = [float(r["rms_dx"]) for r in late]
    signs = []
    for a, b in zip(max_dx, max_dx[1:]):
        d = b - a
        signs.append(1 if d > 0 else -1 if d < 0 else 0)
    strict = sum(a * b < 0 for a, b in zip(signs, signs[1:]))
    state = sum(a != b for a, b in zip(signs, signs[1:]))
    nonzero = [s for s in signs if s]
    nz_rev = sum(a != b for a, b in zip(nonzero, nonzero[1:]))
    check((strict, nz_rev, state) == (42, 54, 67), "Yuksel trend-reversal definitions changed")
    check(min(max_dx) > 0.01 and max(max_dx) == 0.1, "Yuksel late max-dx bounds mismatch")
    check(math.isclose(max_dx[-1], 0.072798219357221, rel_tol=0, abs_tol=1e-15), "Yuksel final max dx mismatch")
    check(math.isclose(rms_dx[-1], 0.000685500334059052, rel_tol=0, abs_tol=1e-15), "Yuksel final RMS dx mismatch")

ydiag = tables["yuksel_800_cap_diagnosis.csv"]
check(all(r.get("cap_classification") == "PERSISTENT_NONCONVERGENCE" for r in ydiag), "Yuksel cap class mismatch")
check(all("IRREGULAR_OSCILLATION" in r.get("motion_pattern", "") for r in ydiag), "Yuksel primary motion class mismatch")

phist = tables["proposed_160_history.csv"]
if len(phist) == 214:
    run1 = [{k: v for k, v in r.items() if k != "run"} for r in phist if r["run"] == "1"]
    run2 = [{k: v for k, v in r.items() if k != "run"} for r in phist if r["run"] == "2"]
    check(len(run1) == len(run2) == 107, "Proposed per-run history length mismatch")
    check(run1 == run2, "Proposed retained numerical histories differ")

pdet = tables["proposed_160_determinism.csv"]
check(all(r.get("determinismVerdict") == "DETERMINISTIC" for r in pdet), "Proposed determinism verdict mismatch")
check(len({r.get("densityChecksum") for r in pdet}) == 1, "Proposed density fingerprints differ")

pcommon = tables["proposed_160_common_evaluators.csv"]
if len(pcommon) == 3:
    check(float(pcommon[0]["native"]) < float(pcommon[0]["E1_raw"]) < float(pcommon[0]["E1_binary"]),
          "Proposed mode-1 native/common ordering mismatch")

summary = json.loads((HERE / "matlab_analysis_summary.json").read_text())
check(summary["olhoff"]["reproduction_verdict"] == "FAILURE_REPRODUCED", "summary Olhoff verdict mismatch")
check(summary["yuksel"]["status"] == "CAP_HIT", "summary Yuksel status mismatch")
check(summary["proposed"]["determinism_verdict"] == "DETERMINISTIC", "summary Proposed verdict mismatch")

provenance = json.loads((HERE / "replay_provenance.json").read_text())
check(provenance["configuration_identity_pass"] is True, "provenance configuration gate failed")
check(provenance["frozen_numerical_implementations_unchanged"] is True, "frozen sources changed")
check(provenance["full_final_campaign_inventory_matches_pre_replay"] is True, "final campaign artifacts changed")
check(provenance["full_final_campaign_file_count"] == 21, "final campaign file count mismatch")
check(provenance["implementation_corruption_detected"] is False, "implementation corruption flag set")

figures = sorted((HERE / "figures").glob("*.png"))
check(len(figures) == 8, f"expected 8 figures, found {len(figures)}")
for path in figures:
    try:
        data = path.read_bytes()[:24]
        check(data[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name}: invalid PNG signature")
        width, height = struct.unpack(">II", data[16:24])
        check(width >= 1000 and height >= 500, f"{path.name}: unexpectedly small image {width}x{height}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{path.name}: image read failure: {exc}")

schema = (HERE / "CSV_DATA_DICTIONARY.md").read_text()
for name in expected_rows:
    check(f"`{name}`" in schema, f"CSV data dictionary omits {name}")
check("comparison models, not ground truth" in schema, "common-evaluator qualification missing from schema")

report_lines = [line.strip() for line in (HERE / "TARGETED_REPLAY_REPORT.md").read_text().splitlines() if line.strip()]
check(report_lines[-3:] == [
    "PERFORMANCE CAMPAIGN FROZEN — READY FOR PAPER",
    "FULL NINE-RESOLUTION RERUN: NOT REQUIRED",
    "FURTHER TARGETED OPTIMIZATION RUNS: NOT REQUIRED",
], "targeted replay report does not end with the required three decision lines")

manifest_rows = []
for line in (HERE / "SHA256SUMS.txt").read_text().splitlines():
    checksum, relative = line.split("  ", 1)
    manifest_rows.append((checksum, relative))
manifest_paths = {relative for _, relative in manifest_rows}
actual_paths = {
    str(path.relative_to(HERE))
    for path in HERE.rglob("*")
    if path.is_file() and path.name != "SHA256SUMS.txt"
}
check("SHA256SUMS.txt" not in manifest_paths, "manifest hashes itself")
check(manifest_paths == actual_paths, "manifest file inventory does not match closure directory")
for checksum, relative in manifest_rows:
    path = HERE / relative
    check(path.is_file() and digest(path) == checksum, f"manifest hash mismatch: {relative}")

if errors:
    print("CLOSURE_VALIDATION_FAIL")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)

print(
    "CLOSURE_VALIDATION_PASS: required artifacts, 10 CSV schemas/row counts, "
    "8 PNGs, replay identities, frozen hashes, null preservation, and SHA256 manifest verified"
)
