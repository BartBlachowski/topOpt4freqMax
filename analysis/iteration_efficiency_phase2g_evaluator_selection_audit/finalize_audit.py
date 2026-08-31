#!/usr/bin/env python3
"""Finalize figures, provenance, and package hashes for Phase-2G."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase2g-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
F2 = REPO / "analysis/iteration_efficiency_phase2f_evaluator_redesign"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(name):
    with (OUT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


figdir = OUT / "figures"; figdir.mkdir(exist_ok=True)

# Method-blind scorecard heatmap.
score = read_csv("CANDIDATE_SCORECARD.csv")
body = [r for r in score if r["criterion"] != "UNWEIGHTED_TOTAL"]
mat = np.array([[float(r[c]) for c in "ABCD"] for r in body])
fig, ax = plt.subplots(figsize=(6.5, 6.5))
im = ax.imshow(mat, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")
ax.set_xticks(range(4), list("ABCD")); ax.set_yticks(range(len(body)), [r["criterion"] for r in body], fontsize=8)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]): ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=8)
ax.set_title("Method-blind candidate scorecard (unweighted)")
fig.colorbar(im, ax=ax, label="score (1 poor, 5 strong)"); fig.tight_layout()
fig.savefig(figdir / "candidate_scorecard.png", dpi=180); plt.close(fig)

# Escalation-event figure.
esc = read_csv("ADAPTIVE_MODE_ESCALATIONS.csv")
fig, ax = plt.subplots(figsize=(8, 3.8))
if esc:
    x = np.arange(len(esc)); ords = [int(r["first_structural_ordinal"]) for r in esc]
    labels = [f"{r['mesh']} k={r['state']} {r['evaluator']}" for r in esc]
    ax.bar(x, ords, color="#2878b5"); ax.axhline(12, color="#d95319", linestyle="--", label="initial Phase-2F batch")
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8); ax.set_ylabel("first structural ordinal"); ax.legend()
ax.set_title("Every observed 12-to-24 escalation")
fig.tight_layout(); fig.savefig(figdir / "adaptive_escalation_events.png", dpi=180); plt.close(fig)

# Hash every Phase-2F artifact, whether directly numerical or supporting narrative.
f2files = sorted(p for p in F2.rglob("*") if p.is_file())
with (OUT / "PHASE2F_EVIDENCE_HASHES.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["path", "bytes", "sha256"])
    for p in f2files: w.writerow([str(p.relative_to(REPO)), p.stat().st_size, digest(p)])

prov_path = OUT / "audit_provenance.json"
prov = json.loads(prov_path.read_text())
prov["captured_at_final"] = datetime.now(timezone.utc).astimezone().isoformat()
prov["branch_final"] = git("branch", "--show-current")
prov["head_final"] = git("rev-parse", "HEAD")
prov["git_status_final"] = git("status", "--short").splitlines()
prov["environment"]["matlab"] = "25.2.0.2998904 (R2025b), release 2025b, maca64"
prov["phase2f_all_evidence_hashes_file"] = "PHASE2F_EVIDENCE_HASHES.csv"
prov["phase2f_all_evidence_file_count"] = len(f2files)
prov["phase2f_completed_survey"] = {
    "archive_bytes": (F2 / "scripts/survey.npz").stat().st_size,
    "archive_sha256": digest(F2 / "scripts/survey.npz"),
    "log_last_line": (F2 / "scripts/survey.log").read_text().splitlines()[-1],
}
extra = [
    REPO / "analysis/iteration_efficiency_phase2d_delta_audit/OLD_DEFECT_INDEPENDENT_REPRODUCTION.csv",
    REPO / "analysis/iteration_efficiency_phase2d_delta_audit/EQ4A_INDEPENDENT_STABILITY.csv",
    REPO / "analysis/iteration_efficiency_phase2d_delta_audit/BINDING_EVALUATOR_RECHECK.csv",
    REPO / "analysis/iteration_efficiency_study_design/QUALITY_EFFORT_SPEC.md",
    REPO / "analysis/iteration_efficiency_study_design/ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md",
    REPO / "analysis/olhoff_nested_mma_route_audit/OLHOFF_NESTED_MMA_ROUTE_AUDIT.md",
    REPO / "analysis/iteration_efficiency_phase2b_precision/qualification_runs/gray_full_24x4_h200_paired_states.mat",
    REPO / "analysis/iteration_efficiency_phase2b_precision/qualification_runs/s1_transition_96x12_h320_paired_states.mat",
]
extra += sorted((REPO / "examples/Performance/final_campaign/raw/olhoff").glob("s1_*.mat"))
prov["additional_evidence_hashes"] = {str(p.relative_to(REPO)): digest(p) for p in extra if p.exists()}
prov["write_boundary"] = {
    "allowed": "analysis/iteration_efficiency_phase2g_evaluator_selection_audit/ only",
    "audit_created_paths_outside_allowed": [],
    "note": "Phase-2F survey completion occurred independently while this audit waited; Phase-2G did not write it.",
}
prov_path.write_text(json.dumps(prov, indent=2) + "\n")

# Package hash list, excluding itself so it is self-consistent.
files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
with (OUT / "SHA256SUMS.txt").open("w", encoding="utf-8") as f:
    for p in files: f.write(f"{digest(p)}  {p.relative_to(OUT)}\n")

print(f"finalized {len(files)} audit files; hashed {len(f2files)} Phase-2F evidence files")
