# FINALIZATION_GATE_REVIEW — repaired rule (commit `b21483b`)

```
FINALIZATION_GATE_PROVENANCE_LOGIC_PASS
```

**Reviewed:** `analysis/OlhoffCurrent/olhoffcurrent_finalization_gate.m` at `b21483b158f58e05e7b56957f2fbe8e1d2891395`.
**Evidence:** `evidence/step1_amendment_evidence.json` and `logs/STEP1_amendment.log.txt`, produced from a **fresh clone of the committed tip using that clone's own code**, not a working copy.

## Requirements (owner brief, Part C)

| requirement | met | how (probe ids refer to `provenance_gate_hardening/PROBE_TABLE.md`) |
|---|---|---|
| **C1** every hash-file line validates on its own | yes | per-line verdicts against each line's own digest; duplicates are validated individually and reported. P5 and P6 FAIL, P11 PASS (identical duplicates, reported). |
| **C2** historical existence proved before hashing | yes | `git cat-file -e` → `-t blob` → `cat-file blob > tmp`, each exit status checked; no pipeline. P9 and P18 FAIL (absent at freeze); P19 PASS (genuinely empty file); P7 and P20 FAIL. |
| **C3** exact digest + exact path + one specific commit | yes | the only admissible commit is the study's freeze commit, and it needs a committed, unmodified hash file and EVIDENCE.json whose declared tree equals the tree at freeze. There is **no ancestry search**. P3, P12, P13, P22, P23 and P24 FAIL. |
| **C4** current source tied to committed HEAD | yes | raw SHA-256 of every HEAD `+impl` blob against the working tree; index never read; `--no-replace-objects`; `GIT_*` environment scrubbed. P10, P28, P36 and P37 FAIL. |
| **C5** dirty `+impl` fails | yes | G6 applies to every study. P8, P10, P26, P27, P28, P36 and P37 FAIL. |
| **C6** manifest is only a consistency check | yes | manifest must equal its HEAD blob, HEAD rows and HEAD tree; PROVENANCE.md tree row = HEAD tree. P25 and P29 FAIL. |
| **C7** separate statuses | yes | `HISTORICAL_SOURCE_HASH_VERIFIED` vs `CURRENT_SOURCE_HASH_VERIFIED`, each with per-line records. P8, P10, P36 and P37 show historical VERIFIED and current NOT_VERIFIED. R1 reports both, with freeze tree `edbfe47e…` ≠ HEAD tree `4ba9a3ae…`. |
| fabricated digest fails | yes | P2, P5, P7, P13 and P35 (a replace-ref forgery) FAIL |
| unrelated ancestor or another file cannot satisfy | yes | P3, P4 and P12 FAIL |
| malformed / non-canonical / shadowed / linked references | yes | P14–P17, P21, P30–P34 and P38 FAIL |

## Committed-code results

- **A. Probe suite from the committed tip:** 39/39 as expected. The repository was untouched.
- **B. R1 in the clean clone.** The 7 superseded lines are `HISTORICAL_VERIFIED` at freeze `bba45e7`, last changed in `1438aa3`, with declared tree = freeze tree. Current status is `CURRENT_SOURCE_HASH_VERIFIED`.
  - In the clean clone the study's overall `ok = 0`, because its **git-ignored raw `.mat` evidence is absent** from any clone: G2/G4/G5 report MISSING for non-source files. That is EVIDENCE_POLICY item 6, not provenance.
  - In the merged normal checkout, where the evidence is present, R1 passes in full: `evidence/POST_studies.json`, and `test_finalization_gate` H and H2.
- **C. Attempt 1's original attack scripts**, re-run unmodified except for their output path:

  | probe | result now |
  |---|---|
  | P5 | FAIL |
  | P9 | FAIL |
  | P10 | FAIL |
  | every other negative | FAIL |
  | P0 | PASS |
  | P1 (an uncommitted sandbox study) | FAIL, by the stricter S3.1 rule |

## Discrimination (the probes are not vacuous)

| gate | fail-open probes |
|---|---|
| original `9b30ec4` rule | 24 of 39 |
| first repair `adf86a3` | 6 of 39 |
| `b21483b` | **0** |

## Residual notes (not defects)

- **Explicit `RepoRoot`.** A caller passing `RepoRoot` is trusted to name the repository under evaluation. The default derives it from the dispatched OlhoffCurrent root.
- **Non-source lines.** They keep the pre-existing study-local-first resolution. Production source can no longer be reached that way.
