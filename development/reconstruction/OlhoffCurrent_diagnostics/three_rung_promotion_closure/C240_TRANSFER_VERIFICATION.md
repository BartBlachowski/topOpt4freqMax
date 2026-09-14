# C240_TRANSFER_VERIFICATION — Phase 2

# `ORIGINAL_C240_EVIDENCE_TRANSFER_FAIL`

The original `C240x30_trajectory.mat` **has not been transferred to this host.**
The task brief's Phase 2 asks this document to verify that the artifact "has now
been copied into the exact evidence location expected by repository policy." It
has not, and this is the finding that stops the task.

## 1. The authoritative expectation

Recovered from the authoritative C240 study manifest —
`diagnostics/three_rung_resolution_240/EVIDENCE.json`, artifact class
`required` — **not** from the previously reported prefix:

| Field | Value |
|---|---|
| policy path | `analysis/OlhoffCurrent/evidence/three_rung_resolution_240/C240x30_trajectory.mat` |
| **expected SHA-256 (full)** | **`183d7ce60d512fc2c045c3cb575404b00c8f0e223adaf97db86c9eef1fd50b0d`** |
| expected bytes | `131 203 128` |

The full digest is recorded here so no future check can pass on the `183d7ce6…`
prefix alone. Exact full-digest equality is required.

## 2. What is actually on this host

| Check | Result |
|---|---|
| parent directory `evidence/three_rung_resolution_240/` exists | **no** |
| artifact present at the policy path | **no** |
| actual SHA-256 | *n/a — no file* |
| actual bytes | *n/a — no file* |
| **exact match** | **no** |

`analysis/OlhoffCurrent/evidence/` on this host contains only
`move_activity_400/`, `two_branch_controller_validation/` and
`three_rung_promotion_validation_retry1/`.

## 3. The search performed before concluding

The absence was not inferred from one `ls`. Four independent methods:

1. **Direct path check** — parent directory does not exist.
2. **Spotlight** (`mdfind`) for `C240x30` and for every `*trajectory.mat` on the
   machine. Only the C240 **CSV and JSON** are present; the eight
   `*trajectory.mat` files found are C160/C320/C400, F400/P400, the retry's own
   C320 three-rung trajectory, and two unrelated `iteration_efficiency_final`
   files.
3. **Exhaustive `find`** over all of `/Users/piotrek` and `/Volumes` for
   `*C240x30*` and `*240x30*`. It returned only the C240 CSV/JSON in this repo
   plus unrelated artifacts in the **upstream** `/Matlab/Olhoff` development
   tree.
4. **Digest elimination** of every upstream 240×30 `.mat`, in case one had been
   renamed:

| Candidate | bytes | sha256 | match |
|---|---|---|:--:|
| `results/FINAL_lp_240x30.mat` | 602 367 | `696f8cfb98f90f59…` | no |
| `audit_termination_mesh_admission/runs/TMA_240x30.mat` | 5 495 967 | `2f5dffca018143ae…` | no |
| `audit_stepcontrol/runs/OOS_240x30_S2_mv004.mat` | 514 327 | `201ff981a7e44334…` | no |
| `audit_s2_design_continuation/runs/SDC_240x30.mat` | 3 055 408 | `e141815ebbc8353a…` | no |
| `audit_m4_topology_restoration/…/FINAL_lp_240x30.mat` | 602 367 | `696f8cfb98f90f59…` | no |

None matches, and all are one to two orders of magnitude smaller than the
131 MB expected. They belong to different upstream studies (`TMA`, `OOS`,
`SDC`, `FINAL_lp`) and are not the `three_rung_resolution_240` candidate
trajectory under another name.

Also checked: `~/.Trash` holds no matching file, no external volumes are
mounted (only `Macintosh HD`), and `tmutil` reports **no Time Machine backup
for this host**.

## 4. Classification — unchanged

```
REMOTE_REQUIRED_EVIDENCE_NOT_LOCAL
```

This is **not** evidence loss and **not** scientific damage. The artifact was
produced by the prior `three_rung_resolution_240` task on a different machine,
and `analysis/OlhoffCurrent/evidence/` is git-ignored wholesale, so it never
travelled with the repository. `evidence/.gitignore` states the expected
consequence verbatim:

> "On a fresh clone this directory is empty and the gate reports
> REQUIRED_MISSING. That is the correct, honest answer."

## 5. What was NOT done

- **Not regenerated.** Re-running C240 is forbidden by the brief and would be
  wrong on the merits regardless: a re-run produces a different container and
  could never reproduce the declared digest.
- **Not re-declared.** Rewriting `three_rung_resolution_240/EVIDENCE.json` to
  match an absent-or-substituted file would falsify the study's evidence
  declaration.
- **Source machine identity** — not recoverable from this host. No transfer
  occurred, so there is no destination path, size, actual digest or copy date
  to record.

## 6. Consequence

Per the frozen criteria (`PREREGISTRATION.md` §5, criterion 2) and the brief:

```
PROMOTION_PROVENANCE_BLOCKED
PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED
```

STOP. No scientific run was executed to work around this, and none may be.

**To close:** copy the original file from the machine that produced it into
`analysis/OlhoffCurrent/evidence/three_rung_resolution_240/` and verify it
against the full digest in §1. That is a file transfer, not compute.

**But note carefully:** C240 is necessary and **not sufficient**. Five further
container artifacts are in the same transfer class — see
`PROMOTION_PROVENANCE.md` §3.
