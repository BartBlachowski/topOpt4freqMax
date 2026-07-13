# Revision_v1 current-state report

Consistency audit and controlled migration following the `OlhoffApproachExact` policy
change. No experiment was regenerated, no MATLAB was run, no numerical algorithm was
modified, and no accepted scientific conclusion was changed. All moves preserve
provenance (`git mv` for tracked files).

Governing rule applied throughout: **a result exists only if an artifact proves it**, and
**a run that reaches the iteration cap is a failure**.

---

## 1. Inventory summary

Counted by artifact *family* (a script plus the configs and outputs it owns).

| Status | Count | Families |
|---|---:|---|
| **ACTIVE_VALID** | **6** | master runner `run_all_revision_experiments.m`; Gate I1 smoke (`exp_smoke_fail.m` + `output/smoke/`); `exp2_authoritative_sweep.m`; `exp3_authoritative_mesh_convergence.m`; `exp2_pilot_authoritative.m` + output; authoritative `clamped_beam_*.json` mesh configs |
| **ACTIVE_NEEDS_RERUN** | **7** | EXP2; EXP2b; EXP3; CR2; **A4 (not implemented)**; S1; and the two legacy stage scripts still bound in the unpatched runner (`exp2_clamped_beam.m`, `exp3_mesh_convergence.m`) |
| **DIAGNOSTIC_ARCHIVE** | **8** | Olhoff-Exact reconstruction (6 scripts + 7 output dirs); Eq. 4b hypothesis test; S1 mode diagnostic; localized-mode onset study; α=1 discrepancy note; pre-authoritative superseded results; failed campaign registry; **retired EXP1/EXP5 evidence (`archive/obsolete_evidence/`)** |
| **OBSOLETE_DELETE_CANDIDATE** | **4** | four `.DS_Store` files (OS cache, no scientific provenance) — **deleted** |

Only **one** artifact in the entire directory is currently publishable evidence: the
Gate I1 fail-loud infrastructure result. Every remaining reviewer-facing experiment is
ACTIVE_NEEDS_RERUN.

> **Updated 2026-07-13.** EXP1 and EXP5 are now **retired from the reviewer evidence
> chain** — see the migration report at the end of this document and
> [SCIENTIFIC_DECISION_EXP1_EXP5.md](SCIENTIFIC_DECISION_EXP1_EXP5.md). The counts above
> reflect that decision; the retired EXP4 is folded into CR2.

---

## 2. Migrations performed

All moves were provenance-preserving (`git mv` where tracked; plain `mv` for the
untracked PNGs).

| # | From | To | Reason |
|---|---|---|---|
| 1 | 6 × `*olhoff_exact*.m` (top level) | `archive/olhoff_exact_reconstruction/scripts/` | Policy: Exact is a closed diagnostic campaign, not reviewer evidence. These are the only files that `addpath` OlhoffApproachExact. |
| 2 | 7 × `output/{pilot,phase}_olhoff_exact_*/` | `archive/olhoff_exact_reconstruction/output/` | Same. |
| 3 | `output/OLHOFF_EXACT_ARCHIVE.md` | `archive/olhoff_exact_reconstruction/` | Index travels with the material it governs. |
| 4 | `eq4b_exp3_400x50_hypothesis_test.m` + `output/eq4b_exp3_400x50/` | `archive/diagnostics/eq4b_hypothesis_test/` | Hypothesis **refuted** (all four questions "no"); run capped 2000/2000 and failed A5. |
| 5 | `s1_exp3_400x50_mode_diagnostic.m` + output | `archive/diagnostics/s1_mode_diagnostic/` | Closed diagnostic. |
| 6 | `exp_scaling_study.m` + output | `archive/diagnostics/localized_mode_onset/` | Explicitly "NOT a mesh-convergence study"; 3/6 meshes mode-invalid. |
| 7 | `exp2_alpha1_discrepancy_diagnosis.md` | `archive/diagnostics/` | Closed diagnostic note. |
| 8 | 19 legacy result files (`all_revision_results.mat`, legacy Exp1/Exp2b/Exp4/Exp5 `.mat`, Exp4 diaries, Exp2b correlation CSVs, `.fig`s) | `archive/superseded_runs/pre_authoritative/` | Produced before the authoritative load `F = ω₀²M(x)Φ₀`; they describe a different method and must not be mixed with authoritative results. |
| 9 | 200 × `topopt_config_*mode_*.png` | `archive/superseded_runs/pre_authoritative/building_mode_plots_ambiguous/` | Written under a generic basename and **overwritten across alpha values** — which alpha each plot belongs to is unrecoverable. Archived, not deleted (real output, ambiguous provenance). |
| 10 | `output/exp1/`, `output/exp2/`, `campaign_progress.json`, `campaign_summary.md` | `archive/superseded_runs/campaign_r1_full_20260701/` | Failed campaign registry + its stage outputs. Quarantined so the next campaign starts clean and EXP5 cannot silently consume the superseded EXP1 timing `.mat`. |
| 11 | *(new)* `archive/README.md` | — | Governance: archive is preserved provenance, never reviewer evidence; historical wording ("exact", "reference", "production", "freeze") is superseded. |
| 12 | *(updated)* `README.md` | — | Corrected stale paths (the six Exact scripts and the archive index have moved), added directory layout and pointers to this report and the execution plan. |

### Incidental finding while migrating

The campaign registry had already been **overwritten**. It no longer describes the
2026-07-03 campaign (`r1_full_20260701…`, EXP1 accepted / EXP2 `eigs:AminusBSingular`) but
a second attempt, `r1_full_20260706T100544049`, which **died after 0.15 s** with
`run_all:OutputConflict` because `output/exp1` was non-empty — it ran nothing. The Jul-3
provenance survives only inside `exp1/exp1_stage_result.json` (`elapsed 55 931 s`,
`status accepted`). Migration item 10 clears this blocker: stage directories are now empty
and the next launch will not abort.

---

## 3. Archived directories

```
archive/
├── README.md                                  governance for the whole tree
├── olhoff_exact_reconstruction/
│   ├── OLHOFF_EXACT_ARCHIVE.md
│   ├── scripts/                               6 reconstruction scripts
│   └── output/                                7 output directories
├── diagnostics/
│   ├── eq4b_hypothesis_test/                  refuted hypothesis
│   ├── s1_mode_diagnostic/                    mode energy/localization study
│   ├── localized_mode_onset/                  onset-of-localization study
│   └── exp2_alpha1_discrepancy_diagnosis.md
└── superseded_runs/
    ├── pre_authoritative/                     19 legacy artifacts
    │   └── building_mode_plots_ambiguous/     200 PNGs, provenance unrecoverable
    └── campaign_r1_full_20260701/             failed campaign + exp1/, exp2/
```

Nothing in `archive/` is scheduled for deletion.

---

## 4. Deletion candidates

| File | Justification |
|---|---|
| `./.DS_Store` | macOS Finder cache. No scientific provenance. |
| `./cr2/.DS_Store` | idem |
| `./output/.DS_Store` | idem |
| `./cr2/mma_diagnostic/.DS_Store` | idem |

These four were the **only** OBSOLETE_DELETE_CANDIDATE artifacts and have been deleted.
Every other questionable artifact was archived instead, per the "if uncertain, archive"
rule — including the 200 ambiguous-provenance PNGs and the failed CR2 outputs.

---

## 5. Remaining mandatory computations

See `REVISION_EXECUTION_PLAN.md` for the full table. Summary:

| Stage | Why it must run | Est. |
|---|---|---:|
| **A4** | **No script, no config, no artifact exists.** Must be implemented, then run. **New critical path.** | ~16 h |
| S1 mitigation | `pmass=6` failed: 9/10 modes still localized. Gates EXP2b and EXP3. | ~3 h |
| CR2 | All three attempts capped (OC **and** MMA) → cause is not the optimizer. Claim withheld. | ~3 h |
| EXP2 | 0/5 alphas accepted; two "accepted" rows are degenerate iteration-1 runs. | ~3 h |
| EXP3 | 400×50 mode-invalid; Δω = 0.55 vs 0.05 threshold. | ~6 h |
| EXP2b | α=1.00 and α=0.75 capped. | ~2 h |
| ~~EXP1~~ | **RETIRED** — construct-invalid benchmark; supports zero surviving claims. | — |
| ~~EXP5~~ | **RETIRED** — depended only on EXP1; scaling claim withdrawn. | — |

**Total ≈ 33 h** (was ~48 h before EXP1/EXP5 were retired).

---

## 6. Remaining reviewer blockers

1. **CR1 — α = 0.75 non-monotonicity.** Unresolved. The run intended to settle it
   "converged" at iteration 1 with grayness 1.0 (degenerate). The manuscript's
   monotonicity claim still omits the 1.73× counterexample.
2. **Omitted load sensitivity (Eq. 6).** No converged A/B comparison exists. Both CR2
   variants capped under OC **and** under MMA. Negligibility **cannot** be claimed.
3. **Frozen-eigenpair reliability (A4).** Zero evidence. The N-sweep was never implemented.
4. **Mesh convergence.** Current evidence **contradicts** it (Δω = 0.55, MAC 0.786,
   topology correlation −0.09).
5. **"No spurious low-density modes".** Refuted by the localized-mode evidence; the only
   attempted mitigation failed. Claim must be retracted or narrowed.
6. ~~**8.6 % gap / 7.1× speedup / O(n_e^1.3).**~~ **RESOLVED by retraction (2026-07-13).**
   All three claims are removed from the manuscript, and the experiments that produced
   them (EXP1/EXP5) are retired as construct-invalid: the local comparators are not
   faithful reference implementations, so no cross-code performance comparison is made.
   See [SCIENTIFIC_DECISION_EXP1_EXP5.md](SCIENTIFIC_DECISION_EXP1_EXP5.md).
7. **Master-runner stage bindings are obsolete** (EXP2/EXP3 still point at
   pre-authoritative scripts in the *unpatched* runner). Resolved by
   `proposed/stage_rewiring.patch` + `proposed/acceptance_gates.patch`, which are prepared
   and validated but **not yet applied**. **The campaign must not be launched until both
   are applied, in that order.**
8. **SS-beam figure captions** carry `ω₁ = 174.3 / 160.5 / 159.3` rad/s as saved local
   endpoints. With EXP1 retired these have no accepted artifact — drop the values or back
   each with one accepted converged run. Not a performance claim; **decision required.**

---

## 7. Confirmation: no active experiment depends on OlhoffApproachExact

Verified by exhaustive grep over the post-migration tree:

- **Executable dependencies:** every `addpath(... 'OlhoffApproachExact' ...)` occurrence —
  all six of them — now resides inside `archive/olhoff_exact_reconstruction/scripts/`.
  **Zero** occurrences remain in any active script, configuration, or output.
- **Approach dispatch:** `exp1_perf_table.m` selects `approaches = {'Olhoff','Yuksel','OurApproach'}`.
  The `Olhoff` key dispatches to `analysis/OlhoffApproach` only. No configuration anywhere
  selects an `OlhoffExact` key.
- **Path allowlist:** `run_all_revision_experiments.m::localAddActiveAnalysisPaths` admits
  only `{ourApproach, OlhoffApproach, YukselApproach, elastic2D, LabandaApproach}`. The
  archived reconstruction tree cannot enter the MATLAB path during a production run.
- **Residual mentions** in `exp1_perf_table.m`, `exp5_scaling.m` and
  `run_all_revision_experiments.m` are **exclusionary comments only** (e.g. `exp5_scaling.m:3`:
  "The Olhoff series is OlhoffApproach only. OlhoffApproachExact is excluded."), not imports.

**Confirmed: no active Revision_v1 experiment depends on `OlhoffApproachExact`.**
`analysis/OlhoffApproach` is the sole active local comparison implementation.

---

# Migration report — EXP1/EXP5 obsolescence (2026-07-13)

Applied per [SCIENTIFIC_DECISION_EXP1_EXP5.md](SCIENTIFIC_DECISION_EXP1_EXP5.md).
No numerical algorithm was modified, no experiment regenerated, A4 not implemented,
CR2 untouched. Nothing deleted.

## 1. Every file changed

| File | Change |
|---|---|
| `SCIENTIFIC_DECISION_EXP1_EXP5.md` | **NEW** — formal decision memo (decision, rationale, reviewer/manuscript/implementation/computational impact, risks, confidence 85%). |
| `paper/main.tex` | **5 reviewer-facing edits** (M1–M5, see §2). |
| `REVISION_EXECUTION_PLAN.md` | EXP1/EXP5 removed from the mandatory campaign and listed as retired; runtime recomputed; new critical path identified. |
| `REVISION_CURRENT_STATE_REPORT.md` | This migration report appended. |
| `README.md` | Active-stage list and decision-memo pointer added. |
| `archive/README.md` | New `obsolete_evidence/` section. |
| `archive/obsolete_evidence/README.md` | **NEW** — governs the retired EXP1/EXP5 artifacts. |
| `archive/obsolete_evidence/exp1_exp5/exp1_perf_table.m` | **MOVED** (git mv) from `examples/Revision_v1/`. |
| `archive/obsolete_evidence/exp1_exp5/exp5_scaling.m` | **MOVED** (git mv) from `examples/Revision_v1/`. |
| `proposed/stage_rewiring.patch` | **REGENERATED** — EXP1/EXP5 stages removed. |
| `proposed/acceptance_gates.patch` | **REGENERATED** — all EXP1 work dropped; `exp1_perf_table.m` no longer touched. |

## 2. Every reviewer-facing claim modified

| # | Location | Modification |
|---|---|---|
| M1 | §`sec:discussion` | Promissory "pending accepted instrumented measurements" → permanent: no cross-code comparison; comparators not faithful (three deviations named, incl. the doubled eigensolve); faithful benchmarking is future work. |
| M2 | §"The proposed approach" | Deleted "…quantitative runtime, memory, speedup, and scaling statements **require the controlled local-comparison evidence**". Operation-count argument retained and sharpened. |
| M3 | §"The proposed approach" | **Removed** the claim of "close agreement with those produced by the Yuksel code". Replaced by the Rayleigh rationale + explicit statement that the frozen-eigenpair accuracy cost is not assessed by cross-code comparison. |
| M4 | Conclusions | "Quantitative performance conclusions are **withheld until**…" → permanent statement + future work. |
| M5 | Conclusions | **Deleted** the unsupported comparative-memory sentence about `OlhoffApproach` MMA history arrays. |

Verified: `grep` finds **no** remaining instance of "pending accepted", "withheld until",
"require the controlled local-comparison", "without accepted instrumentation", or
"close agreement with those produced by the Yuksel".

**Open manuscript item (flagged, not changed):** the SS-beam figure captions carry
`omega_1 = 174.3 / 160.5 / 159.3` rad/s as saved local endpoints. With EXP1 retired these
have no accepted artifact — either drop the values or back each with one accepted
converged run. Not a performance claim; decision required.

## 3. Every stage removed

| Stage | Disposition |
|---|---|
| **EXP1** (performance table) | Removed from the registry; script archived; `localAccept_Exp1` deleted; added to the preflight P2 denylist. |
| **EXP5** (scaling fit) | Removed from the registry; script archived; `localAccept_Exp5` deleted; the EXP5→EXP1 dispatch rebinding and `localLoadExp1Result` helper deleted; added to the preflight P2 denylist. |

## 4. Remaining active stages

`S1 → EXP2 → EXP2b → EXP3 → A4`, plus the **I1** smoke gate and **CR2** (governed
standalone, not a runner stage). S1 remains scheduled before EXP2b and EXP3.
**Untouched: S1, EXP2, EXP2b, EXP3, A4, CR2.**

Preserved and re-verified in MATLAB R2025b: **smoke, dry_run, resume, force, stage mode,
progress tracking**. Runner parses; 18/18 acceptance-gate tests pass; `full` aborts with
`run_all:PreflightFailed` (A4) before any computation.

## 5. Revised runtime estimate

| | Before | After |
|---|---:|---:|
| Mandatory campaign | ~48 h | **~33 h** |

Removed: EXP1 15.5 h (measured) + EXP5 20 s. **New critical path: A4** (~16 h, currently
not implemented) — now both the longest stage and the only one with zero artifacts.

## 6. Confirmation: no active claim depends on EXP1 or EXP5

- **Manuscript:** only four tables remain (`clampedBeamFreq`, `clampedBeamMAC`,
  `buildingFreq`, `buildingMAC`); none is a performance table. No `\ref`, table, figure, or
  number depends on EXP1 or EXP5. The abstract's only quantitative figure (`4.61x`) is a
  MAC-tracked frequency gain from **EXP2b**. The `O(n_e^1.3)` scaling claim, the 8.6% gap,
  the 7.1x speedup and the memory headline were already withdrawn.
- **Code:** no active stage, config, or acceptance gate references `exp1_perf_table` or
  `exp5_scaling`; both are denied by preflight P2 and resolve under `archive/`.
- **Telemetry:** the comparator-telemetry proposal is **withdrawn**; no active experiment
  requires convergence metadata from the Olhoff or Yuksel comparators.

**Confirmed: no active reviewer claim depends on EXP1 or EXP5.**
