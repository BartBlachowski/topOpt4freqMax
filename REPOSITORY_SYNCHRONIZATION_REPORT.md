# Repository Synchronization Report

**Date:** 2026-07-14
**Task:** repository-consistency only. No architectural refactor, no campaign redesign, no
new experiments, no algorithm changes, no regenerated numerical results.
**Governing decisions (FINAL, not reconsidered):**

1. `OlhoffApproachExact` is diagnostic only; not reviewer evidence.
2. **EXP1** is removed from the reviewer evidence chain.
3. **EXP5** is removed from the reviewer evidence chain.
4. The active campaign is **S1 → EXP2 → EXP2b → EXP3 → A4**.
5. **No quantitative cross-code performance benchmark remains in the manuscript.**

---

## 0. Headline finding

The repository's central inconsistency was that
[`SCIENTIFIC_DECISION_EXP1_EXP5.md`](examples/Revision_v1/SCIENTIFIC_DECISION_EXP1_EXP5.md)
declared itself **"DECIDED and APPLIED"** and described a runner with EXP1/EXP5 removed,
S1/A4 registered, and a four-check preflight — while the **actual runner on disk was still
runner v3**: it registered `EXP1, EXP2, EXP2b, EXP3, EXP4, EXP5`, retained
`localAccept_Exp1` / `localAccept_Exp5`, retained the EXP5→EXP1 dispatch rebinding, had
**no S1 stage, no A4 stage, and no preflight at all**.

Worse, the runner was **not merely out of date — it was inoperable.** Its EXP1 and EXP5
stages call `exp1_perf_table()` and `exp5_scaling()`, but those two scripts had already
been `git mv`'d to `archive/obsolete_evidence/exp1_exp5/`, which the path allowlist
deliberately excludes. Any `full` or `fast` campaign would have died at stage 1 with an
undefined-function error.

The runner changes were real but lived **only inside the two unapplied patches**. Applying
them (Phase 3) was therefore the single act that made the largest class of inconsistencies
disappear.

---

## 1. Every modified file

**20 modified, 2 added (by patch), 2 verification artifacts removed.** No numerical result,
solver, or algorithm was touched.

### Runner and campaign infrastructure

| File | Change |
|---|---|
| `examples/Revision_v1/run_all_revision_experiments.m` | **Both patches applied.** Registry rewritten to S1/EXP2/EXP2b/EXP3/A4; EXP1, EXP4, EXP5 stages and their gates removed; EXP5→EXP1 dispatch rebinding and `localLoadExp1Result` deleted; preflight P1–P4 added; EXP2/EXP3 rewired to the authoritative scripts. |
| `examples/Revision_v1/exp2b_building.m` | From `acceptance_gates.patch`: records termination/convergence metadata so the capped-run rule can be enforced. **Recorded only — the computation is unchanged.** |
| `scripts/revision_v1/check_revision_run.m` | **NEW** (from patch): single implementation of the declared acceptance rule. |
| `scripts/revision_v1/test_acceptance_gates.m` | **NEW** (from patch): 18 synthetic-fixture gate tests. Stale `EXP1` labels in its comments corrected. |

### Registry / README / plans

| File | Change |
|---|---|
| `examples/Revision_v1/README.md` | Was **self-contradictory**: declared active stages "Exp1--Exp5" *and* "S1 → EXP2 → EXP2b → EXP3 → A4" in the same file, and published a production chain `run_all_revision_experiments → exp1_perf_table → Olhoff`. Now states the accepted campaign only. |
| `examples/Revision_v1/REVISION_EXECUTION_PLAN.md` | Blocker 1 ("the two patches must be applied — the runner will not work until it is patched") **cleared**, with the verification evidence recorded. |
| `examples/Revision_v1/REVISION_CURRENT_STATE_REPORT.md` | §7 no longer describes `exp1_perf_table.m` as the live approach-dispatch path; now records preflight P1/P2 as the enforcement mechanism. |
| `examples/Revision_v1/SCIENTIFIC_DECISION_EXP1_EXP5.md` | §5 carried the false "applied" claim. Now carries an explicit delivery note (specified 07-13, delivered by patch 07-14) and a verification line against the **real** repository rather than "a disposable sandbox". |
| `examples/Revision_v1/proposed/STAGE_REWIRING_PROPOSAL.md` | Status `PROPOSED, NOT APPLIED` → `APPLIED 2026-07-14`. |
| `examples/Revision_v1/proposed/ACCEPTANCE_GATES_PROPOSAL.md` | Status `PROPOSED, NOT APPLIED` → `APPLIED 2026-07-14`; title de-scoped from "EXP1 / EXP2b" to "EXP2b". |

### Implementation maps and source plans

| File | Change |
|---|---|
| `scripts/revision_v1/IMPLEMENTATION_MAP.md` | **Workstream 5 "Rebuild Performance Evidence" (P1-1 … P1-7, all Tier-1 mandatory) retired.** These were live work orders to rebuild exactly the evidence that decisions 2/3/5 retired. Phase-4 execution steps 26–31 voided; Gate P1 removed from the hard-ordering constraints; MS-18 and Gate-V1 scopes rewritten from "Exp1–Exp5" to the accepted campaign. |
| `examples/Revision_v1/revision_v1_update1.md` | Source doc of the map. Workstream 5 replaced with a retirement record; Gate P1 removed; Tier-1 item "corrected Exp1 comparator/timing evidence" and Tier-2 item "accepted local `OlhoffApproach` comparison evidence" removed; compute strategy no longer sequences a timing study or "Launch Exp1". |
| `examples/Revision_v1/revision_implementation_audit.md` | Historical audit **preserved**, with a superseding banner. Its forward-looking work orders — "Correct Exp1 timing", "retain the external comparison", and the Exp1/Exp5 entries under "Positive Evidence That Can Be Retained" — marked **VOID**. |
| `REVISION_R1_STATUS.md` | Gate P1 `NOT_STARTED` → **RETIRED (gate removed)**; gate table updated; "must be removed or marked *pending regeneration* until P1 produces accepted timing data" → removed permanently; the "performance claims" limitation text replaced with the withdrawal statement; stale "Table 3 in `algorithms_comparison.tex`" reference corrected (that table no longer exists). |

### Manuscript and reviewer planning

| File | Change |
|---|---|
| `paper/main.tex` | **The one surviving manuscript contradiction of decision 5**: a promissory clause, *"If quantitative performance reporting is retained after the revision evidence gates, it will compare only the named local implementations."* This kept the door open for a cross-code benchmark. Replaced with a permanent statement that no such comparison is reported **and none is deferred**. |
| `paper/reviews/revision_plan.tex` | Added a superseding performance-evidence policy box; rewrote "Empirical comparison policy", which still authorized a retained performance table "once the existing A2 timing campaign has been accepted". C5 hardware note re-anchored (it pointed at the removed Table 1 and at the legacy perf driver). |
| `paper/reviews/REVISION_AUDIT.md` | Historical audit **preserved**; override banner extended to EXP1/EXP5; "not accepted performance evidence **until P1 passes**" and "mn8/M4 → delivered by exp5" neutralized (P1 will never pass — the gate is gone). |
| `paper/reviews/algorithms_comparison.tex` | Closed a residual door: "any separate reviewer-facing table is admissible after provenance is verified" → must be non-comparative. |

### Archive references and legacy code

| File | Change |
|---|---|
| `NUMERICAL_BEHAVIOR_FREEZE.md` | **Stale archive paths.** Its artifact table pointed at `examples/Revision_v1/output/phase1…4_*`, but those directories were moved to `archive/olhoff_exact_reconstruction/output/`. Paths corrected. |
| `OLHOFF_EXACT_MIGRATION_REPORT.md` | Historical migration record **preserved**, with a banner noting that two files it lists as modified (`exp1_perf_table.m`, `exp5_scaling.m`) were **subsequently archived**. Its "no active experiment depends on Exact" evidence bullets no longer assert a stage set (Exp1…Exp5) that no longer exists. |
| `docs/olhoff_implementation_analysis.tex` | "runtime, convergence, memory, and scaling **require accepted instrumented evidence**" — a promissory clause implying the measurements are still coming. Replaced with the withdrawal. |
| `examples/Performance/performance_comparison.m` | **A live, ungoverned cross-code performance benchmark driver** (Olhoff/Yuksel/Proposed, printing a table that "mirrors Table 1 from Yuksel et al."). Not deleted — consistent with the archive policy that *nothing is deleted*. Given a governance header: not a campaign stage, not reviewer evidence, may not populate any table, figure, or speedup/scaling claim. |

### Removed

`examples/Revision_v1/output/campaign_progress.json` and `campaign_summary.md` — dry-run
registry artifacts created by my own Phase-4 verification (`status: completed`,
`dry_run: true`, `completed_stages: []`). Deleted so the next campaign starts from a clean
registry. **No result file was touched.**

---

## 2. Every resolved inconsistency

| # | Inconsistency | Resolution |
|---|---|---|
| 1 | Decision memo said the runner had EXP1/EXP5 removed; runner still registered them | Patches applied; verified |
| 2 | Runner had no S1 and no A4 stage, so the accepted campaign was unrepresented | Registry is now exactly S1/EXP2/EXP2b/EXP3/A4 |
| 3 | Runner registered EXP4, retired earlier as pre-authoritative | EXP4 unregistered; script denied by name in preflight P2 |
| 4 | Runner called `exp1_perf_table` / `exp5_scaling`, which are archived and off-path → **campaign inoperable** | Stages removed; both names on the P2 denylist |
| 5 | `localAccept_Exp1`, `localAccept_Exp5`, `localLoadExp1Result`, EXP5→EXP1 rebinding all dead code | Removed (0 occurrences remain) |
| 6 | Runner had no preflight despite docs describing "preflight P2 denies exp1_perf_table by name" | P1–P4 preflight present and firing |
| 7 | `Revision_v1/README.md` internally self-contradictory (Exp1–Exp5 *and* S1→A4) | Single accepted campaign stated |
| 8 | `IMPLEMENTATION_MAP.md` Workstream 5 = 7 live Tier-1 orders to rebuild the retired evidence | Retired |
| 9 | `revision_v1_update1.md` Workstream 5 + Gate P1 + "Launch Exp1" ordering | Retired |
| 10 | `REVISION_R1_STATUS.md` Gate P1 `NOT_STARTED` (i.e. still pending) | `RETIRED — gate removed` |
| 11 | Manuscript promissory clause reopening a cross-code performance comparison | Replaced with permanent withdrawal |
| 12 | `revision_plan.tex` still authorized a retained performance table pending "the A2 timing campaign" | Superseded |
| 13 | `REVISION_AUDIT.md` deferred exp5 evidence "until P1 passes" | Neutralized (P1 cannot pass; it no longer exists) |
| 14 | `algorithms_comparison.tex` left a door open for a future reviewer-facing comparative table | Closed |
| 15 | `docs/olhoff_implementation_analysis.tex` promissory "require accepted instrumented evidence" | Withdrawn |
| 16 | `NUMERICAL_BEHAVIOR_FREEZE.md` archive paths pointed at pre-archive locations | Corrected |
| 17 | `OLHOFF_EXACT_MIGRATION_REPORT.md` cited `exp1_perf_table.m`/`exp5_scaling.m` at live paths and asserted a stale dependency chain | Banner + evidence bullets corrected |
| 18 | Both patch proposals still said "PROPOSED, NOT APPLIED" | Marked APPLIED |
| 19 | `test_acceptance_gates.m` (created by the patch) referenced `localAccept_Exp1` | Comments corrected |
| 20 | `examples/Performance/performance_comparison.m` = ungoverned live cross-code perf benchmark | Governance header; retained, not deleted |

---

## 3. Every remaining inconsistency

**None that contradicts an accepted decision.** Two deliberate, documented retentions:

1. **`examples/Performance/performance_comparison.m` still exists and still runs.** It is
   not deleted (the governing archive policy is explicit that *nothing is deleted*), it is
   not a campaign stage, and it is now labelled as non-evidence. If policy is that no
   cross-code timing driver may exist at all, that is a **new decision** and was not taken
   here.
2. **`exp2_clamped_beam.m`, `exp3_mesh_convergence.m`, `exp4_sensitivity_ablation.m`
   remain in the active directory.** They are pre-authoritative, are *not* registered as
   stages, and are denied by name in preflight P2. Retained as provenance.

Historical documents (`revision_implementation_audit.md`, `REVISION_AUDIT.md`,
`OLHOFF_EXACT_MIGRATION_REPORT.md`, archive READMEs) still *describe* EXP1/EXP5 findings.
That is intended: they are the record of **why** the evidence was retired. Each now carries
a superseding banner, and none issues a live work order.

---

## 4. Remaining scientific blockers

These are **not** synchronization defects. They are open science, unchanged by this task.

| # | Blocker | Status |
|---|---|---|
| S-1 | **A4 does not exist.** No script, no config, no artifact. It is the critical path (~16 h) *and* the sole evidence for the frozen-eigenpair reliability claim, and it now also carries the accuracy question formerly proxied by EXP1. | **NOT IMPLEMENTED** |
| S-2 | **S1 mitigation failed its scientific goal.** The run passed its gates (1579/2000, MAC 0.973) but 9 of 10 modes remain localized low-density modes; `pmass=6` is not a working mitigation. Gates EXP2b and EXP3. | Unresolved |
| S-3 | **EXP2: 0 of 5 alphas accepted.** α=0.75 and α=0.00 "converged" at iteration 1 with grayness 1.0 (degenerate); α=0.50 mode-invalid (MAC 0.748); α=0.25 capped 2000/2000. The CR1 α=0.75 non-monotonicity demand is unresolved. | Unresolved |
| S-4 | **EXP3 contradicts mesh convergence.** 400×50 is mode-invalid (MAC 0.786 < 0.8); relative tracked-ω change between meshes 0.546 against a declared threshold of 0.05; topology correlation −0.088. | Unresolved |
| S-5 | **EXP2b:** α=1.00 and α=0.75 capped 2000/2000; many of the first ten topology modes have max MAC < 0.01, so the "no spurious low-density modes" claim is unsupported. **The abstract's only quantitative figure (4.61×) depends on EXP2b.** | Unresolved |
| S-6 | **CR2:** every attempt capped. Not a runner stage (`run_cr2_production` takes no `outDir`); must be run and cited manually. | Unresolved |
| S-7 | **SS-beam figure captions** carry ω₁ = 174.3 / 160.5 / 159.3 rad/s as "saved local endpoints". With EXP1 retired these have **no accepted artifact**. Either drop the values or back each with one accepted converged run. The decision memo explicitly flags this as *"Decision required"* — it is a scientific decision and I did not take it. | **DECISION REQUIRED** |

---

## 5. Remaining computational blockers

| # | Blocker |
|---|---|
| C-1 | **A4 must be written before it can be run** (~16 h estimated). Preflight P3 blocks every `full`/`fast` campaign until it exists — verified: `full` aborts in ~5 s. |
| C-2 | Remaining mandatory campaign ≈ **33 h** (was ~48 h; EXP1's 15.5 h and EXP5's 20 s are gone). Critical path is A4, which is independent of the S1 → EXP2 → EXP3 chain (~12 h) and should start first. |
| C-3 | S1, EXP2, EXP2b, EXP3 all need reruns; none currently holds an accepted artifact under the authoritative load. |

## 6. Remaining infrastructure blockers

| # | Blocker |
|---|---|
| I-1 | **CR2 is not a runner stage.** `run_cr2_production()` takes no `outDir` and writes to a hardcoded path; registering it would change the experiment's signature — out of scope for synchronization. It must be run and cited manually, under archive governance. |
| I-2 | `run_all:OutputConflict`: stage directories must be empty before launch. Preflight P4 now reports conflicts for all stages up-front rather than after hours of compute. |
| I-3 | Cosmetic: the runner banner still prints "fail-loud runner v3". Harmless; not touched (it would be a code change with no consistency effect). |

---

## 7. Current active campaign graph

Read directly from the registry in `run_all_revision_experiments.m` (not from any plan):

```
S1  [ACTIVE_NEEDS_RERUN]  ─┬─→ EXP2  [ACTIVE_NEEDS_RERUN] ──→ EXP3  [ACTIVE_NEEDS_RERUN]
   (gates EXP2b/EXP3)      │        (dependsOn: S1)              (dependsOn: S1, EXP2)
                          └─→ EXP2b [ACTIVE_NEEDS_RERUN]
                                    (dependsOn: S1)

A4  [NOT IMPLEMENTED — placeholder; dependsOn: none]   ← CRITICAL PATH (~16 h)

I1  smoke gate            [PASSED — fail-loud verified]
CR2                       [governed standalone; NOT a runner stage]
```

Stage dependencies as actually declared: `S1 {}` · `EXP2 {S1}` · `EXP2b {S1}` ·
`EXP3 {S1, EXP2}` · `A4 {}`.

**Retired and not in the graph:** EXP1, EXP4, EXP5 — all three denied *by name* in preflight
P2, alongside the pre-authoritative and archived-Exact runners.

---

## 8. Confirmation against the accepted decisions

Verified against the **actual** repository (MATLAB R2025b, real working tree — not a sandbox):

| Decision | Evidence |
|---|---|
| **1.** `OlhoffApproachExact` is diagnostic only | No active stage, config, or path selects it. Preflight **P1** fails the campaign if any active MATLAB path entry references it; the allowlist admits only `{ourApproach, OlhoffApproach, YukselApproach, elastic2D, LabandaApproach}`. `tools/Matlab/run_topopt_from_json.m` retains an `OlhoffExact` branch for archive compatibility **behind a warning**, excluded from the production-choice list. Every remaining reference is inside the archive, inside the Exact trees themselves, or is an exclusionary governance statement. |
| **2.** EXP1 removed from the evidence chain | Not in the registry. `localAccept_Exp1` and `localLoadExp1Result`: **0 occurrences**. `exp1_perf_table` on the P2 denylist. Script preserved unmodified at `archive/obsolete_evidence/exp1_exp5/exp1_perf_table.m`. |
| **3.** EXP5 removed from the evidence chain | Not in the registry. `localAccept_Exp5`: **0 occurrences**. EXP5→EXP1 dispatch rebinding deleted. `exp5_scaling` on the P2 denylist. Script preserved at `archive/obsolete_evidence/exp1_exp5/exp5_scaling.m`. |
| **4.** Active campaign = S1, EXP2, EXP2b, EXP3, A4 | Registry emits exactly `'s1','S1'` `'exp2','EXP2'` `'exp2b','EXP2b'` `'exp3','EXP3'` `'a4','A4'`. Dry run prints that graph and nothing else. |
| **5.** No cross-code performance benchmark in the manuscript | `paper/main.tex` has **4 tables** — `clampedBeamFreq`, `clampedBeamMAC`, `buildingFreq`, `buildingMAC`. None is a performance table. Zero promissory/pending performance language. The two surviving `speedup` mentions (lines 123, 191) are **literature background attributed to Yuksel & Yilmaz's own published claim**, not claims about the proposed method. |

### Execution evidence (actual runs, this session)

```
test_acceptance_gates            →  passed: 18   failed: 0   ALL TESTS PASSED
run_all(...,'dry_run',true)      →  S1 · EXP2 · EXP2b · EXP3 · A4   (no other stage)
run_all('full')                  →  run_all:PreflightFailed in 4.7 s
                                    "P3 mandatory stage not implemented: A4"
                                    — aborts BEFORE any computation
run_all('smoke')                 →  run_all:GateI1Confirmed
                                    "Gate I1 PASSED — runner correctly identified failure"
```

**The repository now reflects the accepted scientific decisions.**

---

## FINAL VERIFICATION

> ## **B — Repository is synchronized, but scientific work remains.**

**Synchronized (objective evidence above):** the registry, the stage graph, the stage
dependencies, the preflight denylists, the README, the revision plans, the implementation
maps, the migration reports, the reviewer planning, the manuscript, and the archive
manifests all now express the same campaign — S1 → EXP2 → EXP2b → EXP3 → A4 — with EXP1 and
EXP5 retired, `OlhoffApproachExact` diagnostic-only, and no cross-code performance benchmark
anywhere in the manuscript. Both patches applied cleanly, in order, and are verified by
execution rather than by inspection.

**Not A**, because "ready for the remaining computation campaign" is false on the
repository's own evidence: **A4 is not implemented**, and preflight P3 therefore refuses to
start a `full` campaign at all. A4 is simultaneously the critical path, the sole evidence
for the frozen-eigenpair claim, and the inheritor of the accuracy question EXP1 used to
proxy. Beyond it, S1's mitigation has failed its scientific goal, EXP2 has 0 of 5 alphas
accepted, EXP3 currently *contradicts* mesh convergence, EXP2b is capped — and the
abstract's only quantitative figure (4.61×) depends on EXP2b. One manuscript item
(the SS-beam caption frequencies, now artifact-less) explicitly awaits an author decision.

**Not C:** no file contradicts an accepted decision. The remaining EXP1/EXP5 text is
confined to historical records that carry superseding banners and issue no work orders, and
to exclusionary denylists whose purpose is to keep the retired evidence out.

Synchronization is complete. The science is not.
