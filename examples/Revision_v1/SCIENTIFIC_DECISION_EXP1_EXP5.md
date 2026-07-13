# Scientific decision: EXP1 and EXP5 are obsolete as reviewer evidence

**Status: DECIDED and APPLIED.** Date: 2026-07-13.
Classification: **D — obsolete after the current revision strategy.**
Confidence: **85%.**

---

## 1. Decision

**EXP1** (local-implementation performance table: timing, memory, iteration counts,
`omega_1`, across 3 methods x 4 meshes x 10 samples) and **EXP5** (log-log scaling fit,
which consumes EXP1) are **removed from the reviewer evidence chain**.

Both are preserved, unmodified, as governed historical artifacts in
`archive/obsolete_evidence/exp1_exp5/`. Nothing is deleted.

**Comparator telemetry is NOT implemented.** The proposal to export convergence metadata
from `topFreqOptimization_MMA` (Olhoff) and the Yuksel branch is **withdrawn** as
scientifically unjustified.

---

## 2. Scientific rationale

### 2.1 EXP1 supports zero surviving manuscript claims

Verified against `paper/main.tex`: only four tables remain (`clampedBeamFreq`,
`clampedBeamMAC`, `buildingFreq`, `buildingMAC`). **There is no performance table.** The
8.6% frequency gap, the 7.1x speedup, the comparative memory headline, and the
`O(n_e^1.3)` scaling claim were all already withdrawn during the OlhoffApproachExact
migration. No `\ref`, table, or number in the manuscript depended on EXP1. The abstract's
only quantitative figure — `4.61x` — is a MAC-tracked frequency gain produced by **EXP2b**.

What remained were three *promissory* clauses ("pending accepted instrumented
measurements", "withheld until…", "…require the controlled local-comparison evidence").
EXP1 existed only to redeem IOUs the manuscript had written to itself.

### 2.2 EXP1 is construct-invalid, not under-instrumented

This is the decisive point. The manuscript **itself** documents, in
Section~`sec:discussion`, that the local `OlhoffApproach` departs from the published
formulation in three ways — and the third is fatal:

> *a **trial eigensolve** is performed immediately after every MMA update… This
> effectively **doubles the number of eigensolves per outer iteration** relative to the
> original scheme.*

plus an added Heaviside projection, a seven-level continuation schedule, and a grayness
penalty that the published method does not have. The local Yuksel-inspired implementation
is likewise unfaithful: its iteration counts do not reproduce the 180–200 reported by the
original authors.

Therefore any speedup EXP1 could produce would measure the proposed method against a
comparator **the authors themselves certify performs roughly twice the eigensolves it
needs to**. That is a strawman benchmark. Instrumenting it to ten-sample precision does
not make it meaningful — it yields a *precise* measurement of a quantity that licenses no
inference about either published method.

**The missing comparator telemetry is a symptom, not the disease. A validity defect
cannot be repaired by a measurement fix.**

### 2.3 A4 is scientifically superior for the one question worth asking

The only substantive claim bundled into EXP1 was that the proposed method's frequencies
are "in close agreement with those produced by the Yuksel code" — i.e. *does freezing the
eigenpair cost accuracy?*

**A4** (eigenpair-refresh sweep `N = {1, 5, 10, 50, inf}`) answers exactly this question
*within a single implementation*, isolating the frozen-eigenpair approximation error
without confounding it with cross-code implementation differences. It strictly dominates
the EXP1 proxy. A4 can also yield a *legitimate* efficiency number (eigensolves per
iteration, frozen vs. refreshed) with **no unfaithful comparator involved**.

### 2.4 Why not merely reduce EXP1's scope

A stripped-down EXP1 reporting only `omega_1` would **still** have to prove its comparator
runs were not capped — which requires exactly the comparator telemetry in question. Scope
reduction does not escape the blocker; it only shrinks the compute bill while retaining
the invalid comparator. Rejected.

---

## 3. Reviewer impact

Seven demands move from *"answer with evidence"* to *"resolved by retraction"*. This is
explicitly permitted by the revision plan's own Definition of Complete, item 8: *"every
reviewer demand is FULFILLED with evidence **or explicitly resolved by correction,
qualification, or retraction**."*

| Reviewer demand | Resolution |
|---|---|
| Separate initialization / per-iteration / total runtime (R1, MR1) | Retracted — no cross-code performance claim is made. |
| Timing and memory standard deviations | Retracted — same. |
| Clarify the source of the speedup | Retracted — no speedup is claimed. |
| Bound the speedup versus canonical Olhoff | Retracted — no speedup is claimed. |
| Report `omega_1` for every method and mesh | Retracted as a benchmark table; the SS-beam figure endpoints remain, labelled as saved local endpoints. |
| Explain the Yuksel 180–200 vs 1000+ iteration discrepancy | Moot — the comparator is no longer used for any quantitative claim; its unfaithfulness is now stated as the *reason* for withdrawal. |
| Correct the `O(n_e^1.3)` scaling claim with a log-log fit (M4) | Resolved by removal of the claim (EXP5 retired). |

**Not affected:** the hardware/software specification (R2/MR4) is printed and manifested by
the runner for every stage, independently of EXP1.

The response letter must state the reason plainly: *we do not present a speedup, because
our own Section~`sec:discussion` documents that the comparator performs roughly twice the
required eigensolves; a benchmark against it would be a strawman, and we decline to
publish one.* This is a stronger position than the previous data supported — the
regenerated EXP1 numbers gave `Δω ≈ ±0.5%`, contradicting the 8.6% claim outright.

---

## 4. Manuscript impact (applied)

Five edits to `paper/main.tex`. No claim was strengthened; every change removes or
narrows.

| # | Location | Change |
|---|---|---|
| M1 | §`sec:discussion` | "No quantitative performance claim is made here **pending accepted instrumented measurements**…" → permanent statement: no cross-code comparison is made; the comparators are not faithful; the three specific deviations are named; faithful benchmarking is future work. |
| M2 | §"The proposed approach" | Deleted "…**but quantitative runtime, memory, speedup, and scaling statements require the controlled local-comparison evidence described above**." Kept and sharpened the operation-count argument (one eigensolve at initialization vs. one per design iteration). |
| M3 | §"The proposed approach" | Deleted "…in **close agreement with those produced by the Yuksel code**…". Replaced with the Rayleigh-principle rationale plus an explicit statement that the accuracy cost of freezing is a property of the approximation, not assessed by cross-code comparison. |
| M4 | Conclusions | "Quantitative performance conclusions are **withheld until** the controlled local-comparison evidence is accepted." → permanent statement + operation-count framing + future work. |
| M5 | Conclusions | **Deleted** "The proposed code avoids the MMA history arrays used by the local `OlhoffApproach'; no comparative memory headline is asserted without accepted instrumentation." (unsupported memory comparison). |

**Abstract: unchanged** — it makes no measured-performance claim.

### Deliberate deviation from the brief

The brief said to *prefer re-sourcing* the frozen-eigenpair accuracy claim to A4. **This
was not done, on purpose.** A4 has zero results today; asserting "confirmed by the refresh
study" would create a brand-new IOU — precisely the promissory pattern this decision
eliminates. M3 therefore **deletes** the claim. The sentence becomes eligible for
reinstatement, sourced to A4, **once A4 is implemented and accepted**.

### Outstanding manuscript item (flagged, not changed)

The simply-supported-beam figure captions carry `omega_1 = 174.3 / 160.5 / 159.3` rad/s as
"saved endpoints" of the local implementations. With EXP1 retired these have no accepted
artifact. Under *a result exists only if an artifact proves it*, either drop the values or
back each with one accepted converged run. This is **not** a performance claim and is cheap
either way. **Decision required.**

---

## 5. Implementation impact

- Master runner: EXP1 and EXP5 stages removed; the EXP5→EXP1 dispatch rebinding and the
  `localLoadExp1Result` helper are gone; the now-dead `localAccept_Exp1` and
  `localAccept_Exp5` gates are removed. **No scaling stage remains active.**
- Preflight **P2** now denies `exp1_perf_table` and `exp5_scaling` by name, in addition to
  the pre-authoritative and archived runners.
- The acceptance-gates safety patch **no longer touches `exp1_perf_table.m` at all**; the
  EXP1 convergence-metadata blocker **is dissolved**, not worked around.
- Preserved unchanged: **smoke, dry_run, resume, force, stage mode, progress tracking**.
- **Untouched, as required: S1, EXP2, EXP2b, EXP3, A4, CR2.**

Validated in a disposable sandbox with MATLAB R2025b: runner parses, 18/18 acceptance-gate
tests pass, dry-run reports **S1 → EXP2 → EXP2b → EXP3 → A4**, `full` aborts with
`run_all:PreflightFailed` (A4 not implemented) before any computation, `smoke` still yields
`run_all:GateI1Confirmed`.

---

## 6. Computational impact

| | Before | After |
|---|---:|---:|
| EXP1 | 15.5 h (measured) | **retired** |
| EXP5 | 20 s (measured) | **retired** |
| Remaining mandatory campaign | ~48 h | **~33 h** |

The campaign's single longest pole is removed. The new critical path is **A4** (~16 h
estimated, and currently *not implemented*).

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Perceived as evasion — reviewers explicitly asked for the timing decomposition. | Moderate | State the construct-validity reason and cite the manuscript's own documentation of the comparator deviations. A strawman benchmark is worse than none. |
| Contribution looks thin without any empirical efficiency evidence. | Moderate | The efficiency argument survives as an operation-count claim (standard in the topology-optimization literature). The empirical contribution is mode targeting, the α-weighting rule, and honest limitations. |
| An editor insists on *some* empirical efficiency number. | Low–moderate | **Fallback already planned:** A4's `N=1` (refresh every iteration) vs `N=inf` (frozen) is a within-codebase cost comparison with no unfaithful comparator — the only valid way to quantify what freezing saves. |
| Loss of independent cross-validation of `omega_1`. | Low | The comparators share the same FE core; they were never independent. |

---

## 8. Verdict on comparator telemetry

**Not scientifically justified. Withdrawn.**

It is trivially cheap — `topFreqOptimization_MMA` already computes `change_x` and keeps
`dx_hist`. That cheapness is the trap. Telemetry would deliver a *precise* measurement of
a comparison that is invalid by construction, and having it on hand would create standing
pressure to reinstate the strawman benchmark it enables.

If it is ever wanted for internal diagnostics, it must be added as **diagnostic-only**,
explicitly outside the reviewer evidence chain — the same governance already applied to
`OlhoffApproachExact`.
