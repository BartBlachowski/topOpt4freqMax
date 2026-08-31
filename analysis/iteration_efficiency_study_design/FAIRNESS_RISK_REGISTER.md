# Fairness and bias risk register

The protocol accepts any ordering, including a failure to reach the gate. Risks are
reviewed against all three methods rather than against an expected Proposed advantage.

Phase-1C delta: rows F01, F03, F04, F07 and F10 are revised where the audit repairs changed
the underlying control, rows F25–F27 are revised for the repaired scaling discipline, and
rows F28–F31 are new. The superseded control wording is kept in
each cell so the change is traceable.

| ID | Choice / risk | Could favor Proposed | Could favor Yuksel | Could favor Olhoff | Control / required wording |
|---|---|---|---|---|---|
| F01 | Common evaluator choice (M4) | E1 *is* Proposed's interpolation up to floor values, verified to 9–11 digits | E2 is Yuksel's piecewise mass law | E3 is Olhoff's `rho_min=1e-3` model | **REVISED.** Superseded control: "E1 raw as primary common model, E2/E3 as rescans." No evaluator is primary. Acceptance requires the threshold under all three via `min_e [Q_e/Q_ref_e] >= q`; per-evaluator endpoints are co-equal; disagreement is `MODEL_DEPENDENT`. E2/E3 share the piecewise `x^6` mass law and differ only in stiffness floor. Mitigating fact: agreement within 0.429%, ordering preserved wherever all evaluator values are available |
| F02 | Native frequency as quality | Coarse Proposed is severely penalized by weak-material local modes | Native formulation is method-specific | Native multiplicity model differs | Native frequency cannot pass the universal spectral gate |
| F03 | Primary relative-to-own reference (M2, M3) | A low-quality Proposed endpoint becomes easy to match; its native stop makes it plateau by construction | Same for Yuksel, which also terminates on a change test | Same for a low-quality/failed Olhoff path, though the selected profile never becomes stationary and is penalised by steady late improvement | **REVISED.** Superseded control did not name the plateau/noise/late-improvement preferences. R is labelled self-referenced maturation work; all three structural preferences are stated in the estimand definition; sustained-floor trajectories are mandatory; achieved E1/E2/E3 and reference sit beside every count; the best-observed benchmark is now mandatory; A appears only if independently instantiated |
| F04 | Best-sustained rather than last endpoint | Can ignore a later deterioration | Can ignore late irregular oscillation | Can recognize mature quality before later LP failure | Reference is the max of a full `P`-window floor, never a single peak; later failures disclosed. **Phase-1C addition:** the floor is now evaluated on a separate reference trajectory and frozen by a causal first-passage rule, so a later improvement cannot revise it either |
| F05 | Common absolute target | A target near Proposed quality could favor it | A target near Yuksel quality could favor it | A target near Olhoff quality can censor both other methods | A is omitted unless target provenance is independent; optional best-observed is never called absolute or A |
| F06 | Exact-count binary topology | Gray Proposed designs can be materially reinterpreted | Crisper fields may be less affected | Thresholding can expose Olhoff detached fragments / thin joints | Raw spectral baseline plus strict common binary graph; raw/binary metrics both disclosed |
| F07 | Topology strictness (C1, Mo6, Mo7) | A single projection-tie pixel must not veto a structurally correct design | Same | The superseded aggregate clause nearly excluded Olhoff at 640x80 (0.6% pass; longest run 5); 800x100 is unavailable/RUN_ERROR/N/A and not inferred | **REVISED.** Superseded control: "T1 filter-derived footprint is baseline; T0 strict sensitivity." The aggregate-area veto is deleted; `a_res=5`/`r_common` are retired because they were derived from one method's `rmin`; the scale is the FE-geometric `A_sig=4·A_e(160x20)=0.01`. T0 is demoted to a known strict diagnostic with its outcome stated up front; the sensitivity is method-neutral 1×1/3×3 FE patch scales, able to discriminate in the permissive direction. Fixed-fraction LCC thresholds remain rejected |
| F08 | Support-to-support connectivity | Natural for this support-only eigenproblem | Same | Same | Use exact element footprints of both prescribed supports, not arbitrary end columns |
| F09 | Volume tolerance | OC equality may make Proposed pass easily | OC equality may make Yuksel pass easily, though physical volume drifts | LP volume equality typically passes tightly | Use pre-existing common `rV<=1e-3`, raw field, 5e-4/2e-3 sensitivity; no method-specific tolerance |
| F10 | Same `P=100` iterations (Mo1, Mo3) | `P-1` is ~30–93% of Proposed's native run, and certification may extend past its native stop | Mixes two stages if not guarded | `P-1` is ~6% of Olhoff's fixed horizon | **REVISED.** Only Stage 2 eligible for Yuksel; paired headline `k_enter/k_cert`; `k_native` printed beside `k_cert`; P described accurately as a convention inherited from Olhoff evidence and applied uniformly, not derived from all three methods; `k_enter` scaling primary and `k_cert` fit demoted to descriptive with an identity check, because the additive constant cut the frozen exponent by 32%; P=50/200 rescan at every q |
| F11 | Olhoff `N=2`, gap<=1% | No burden on Proposed | No burden on Yuksel | Additional burden on Olhoff | It is part of the intended reproduced Olhoff result and prior frozen audit; do not impose artificial symmetry |
| F12 | No modal gap for Proposed/Yuksel | Could let either pass with a wide gap | Same | Olhoff uniquely burdened | Their formulations do not require multiplicity; adding a gap would redefine them |
| F13 | Requiring Olhoff S1 policy stage 2 | — | — | Adds the frozen 100-state stabilization floor | Report trigger count; the frozen selected method includes this causal continuation; independent auditor challenges asymmetry before production |
| F14 | Counting Yuksel stages in one total | Proposed has one homogeneous loop | Hides that Stage 1 prepares a mode and has different cost | — | Always retain Stage 1, Stage 2, total, and stage times; state that the units are chronological, not homogeneous |
| F15 | Olhoff outer count only | Makes expensive eigen/LP work look like cheap OC iterations | Same | Can make Olhoff appear iteration-efficient despite high inner effort | Do not claim equal work per count; report time and LP solver effort; main quantity is explicitly method-level iteration effort |
| F16 | `nInner=1` in selected Olhoff code | — | — | Could falsely suggest negligible inner work | State it means one `linprog` call; capture solver-reported iterations separately; omit redundant code count from main table |
| F17 | Later solver failure after certification | — | — | Can allow Olhoff an earlier pass despite a failed long observer run | Valid for minimum-work estimand; label `PASS_WITH_LATER_FAILURE` and show failure attempt/status |
| F18 | Measurement horizon | A discretionary extension could improve a preferred row | Same | Same | No discretionary review or tranche: after reference freeze every cell uses the same equation `B_meas=min(max(B0,b_ref+P-1),B_ref)`, with only already-frozen inputs; the horizon and inputs are published |
| F19 | Full density instrumentation | Memory/I/O burden differs with iteration count | Same | Olhoff already stores large snapshots | Exclude observer I/O from timing; timing replays are lightweight and trajectory-checked |
| F20 | Return-equivalent state indexing | Proposed is post-update | Yuksel returned field is not indexed like Proposed | Olhoff history spectra are pre-update while snapshots are post-update | Explicit checkpoint identity tests; no raw loop-index comparison |
| F21 | Fixed filter radii in element units | Different radii and physical scales | Different | Different | Treat as frozen method identity, not common regularization; do not infer continuum-equivalent topology effort |
| F22 | Deterministic single run for counts | Can hide basin variability | Same | Same | Existing paths are deterministic; verify two prefix identities at one coarse and one fine mesh; if not, expand to seed distribution before production |
| F23 | Timing replay at known endpoints | Could optimize code path after seeing target | Same | Same | Same executable/profile; only diagnostic writes and enter/cert stop horizons differ; source/config hashes and prefix identity required |
| F24 | Images | A visually cleaner Proposed result can bias readers | Same | Same | Standard raw/binary grids, fixed orientation/supports; images cannot set PASS/FAIL |
| F25 | Free power-law fits on nine meshes (M5, Mo2, Mo4) | Selective censoring/range can manufacture favorable slopes; a non-monotone 3x-range count series yields a weakly identified exponent | Same | Same | **REVISED.** Fit only certified positive points; publish all status markers, `C/p/R2_log/n_valid/range`; at least 3 valid meshes; no extrapolation. Added: a **common-support companion fit** is mandatory and cross-method exponent comparison outside common support is prohibited; leave-one-out `p` ranges accompany every `p`; preregistered `WEAKLY_IDENTIFIED` labels (`R2_log<0.80`, LOO range spans zero, LOO width `> |p|`) prevent quoting a poorly determined exponent as a small one |
| F26 | Additive certification window in scaling (Mo1) | `+99` flattens a short-count method's fitted slope most, i.e. Proposed at 107–330 native iterations | Same | Same | **REVISED.** Superseded control fitted enter and cert as co-equal layers. `k_enter` scaling is now primary and the `k_cert` power fit is secondary/descriptive with a mandatory caption caveat; the pipeline verifies that fitting `k_cert-(P-1)` reproduces the `k_enter` fit exactly. Frozen evidence: the constant cut the Olhoff exponent from +0.1451 to +0.0991, a 32% reduction from bookkeeping. P=50/200 rescan retained |
| F27 | Mean iteration cost (Mi5) | A Proposed OC update contains **no eigensolve** under the frozen solid-reference profile, so it can look directly comparable to a rich outer update | Yuksel combined mean hides stage heterogeneity | Outer mean hides LP variability; the frozen 800x100 eigensolve alone was ~75% of outer cost (median `tEig` 1.368 s of 1.817 s) | **REVISED.** State native units; show Yuksel stages and Olhoff decomposition; pair counts with platform time. Added: the no-eigensolve fact and the 75% figure are stated numerically in the accounting spec, the Main Table 1 footnote, and the F2 in-axes note, with F3 placed adjacent to F2 |

## Quantities that are not equivalent algorithmic work

- one Proposed OC update;
- one Yuksel Stage-1 OC update;
- one Yuksel Stage-2 moving-load OC update;
- one Olhoff eigenanalysis/generalized-gradient/LP outer update;
- one LP call;
- one MATLAB/HiGHS-reported solver iteration or pivot;
- one second on the reference platform.

They may appear in adjacent columns but must never be described as interchangeable units.

## Phase-1C additional risks

| ID | Choice / risk | Could favor Proposed | Could favor Yuksel | Could favor Olhoff | Control / required wording |
|---|---|---|---|---|---|
| F28 | Reference-phase stabilization rule (C2) | A method that plateaus early stabilizes early and freezes a reachable reference | Same | A method still improving may never stabilize and lose R entirely at that mesh | `L_ref=500`, `epsilon_ref=0.001` and `B_ref=3200` are fixed before Phase 2 and applied identically; the rule is causal first-passage; **no cap fallback**, so no horizon can lower a bar; `REFERENCE_NOT_ESTABLISHED` is a publishable outcome; mandatory diagnostic reports what the superseded horizon rule would have given at 900/1600/2000/3200 |
| F29 | `A_sig` from the coarsest-mesh 2×2 patch (C1) | — | — | More permissive at fine meshes (100 elements at 800x100); current Olhoff 800x100 topology evidence is unavailable/RUN_ERROR/N/A | Derived from FE geometry alone, before any cross-method scan; constant physical area is the mesh-invariant statement while a constant element count silently tightens 25×; available frozen evidence shows pathological states still fail; 1×1/3×3 OAT rescan published |
| F30 | Three co-primary q levels (M1) | A method with an early plateau looks best at loose q | Same | A method with a better endpoint may only lead at tight q | The three levels are exactly the Phase-1A baseline and its two declared sensitivities, elevated together before any ranking was consulted; all three are reported; a ranking that changes with q, or curves that cross, is a reportable result and not an anomaly |
| F31 | All-evaluator minimum acceptance (M4) | Conservative for all: can only delay entry | Same | Same | Minimum is taken over dimensionless attainment ratios, not absolute frequencies — the absolute-units minimum was rejected precisely because level offsets would let one evaluator dominate everywhere; per-evaluator endpoints published beside the robust pair |

## Expected-result firewall

No rule may be changed because the observed order is surprising or unattractive. The
protocol remains valid if Olhoff has the smallest outer count, Yuksel the smallest total
count or the lowest time, Proposed the largest count but cheapest iterations, Olhoff the
best absolute quality, the ranking changes with the requested quality level, the
quality–effort curves cross, or any method is `NOT_REACHED` or
`REFERENCE_NOT_ESTABLISHED`. Acceptance constants (`A_sig`, `P`, `L_ref`, `epsilon_ref`,
`B_ref`, the q levels, volume tolerance, Olhoff gap), evaluator definitions, the repaired
topology gate, status precedence, fit eligibility, mesh range, and table/figure inclusion
rules are frozen before production identities are unblinded. The completed campaign may
motivate the design and safety budget, but its outcome cannot choose a new-study gate,
suppress a censored point, select a fit range, or be relabelled as an R/A trajectory
result.

The full method-blind re-audit — the protocol re-read with the methods renamed A, B and C,
asking whether any remaining choice systematically rewards cheaper endpoint quality, a
shorter reference horizon, native interpolation, smoother topology, or fewer algorithmic
layers — is recorded in `PHASE1C_AUDIT_RESPONSE.md` WP14.

## Phase 2H additions

- **F32 — modal cherry-picking:** controlled by the frozen unanimous classifier and lowest
  valid ordinal, with disagreement states retained.
- **F33 — finite-mode censoring:** controlled by adaptive doubling without a scientific
  ceiling and fail-closed technical exhaustion.
- **F34 — projection contamination:** controlled by excluding exact-count binary D from Q,
  reference, persistence, and status.
- **F35 — Olhoff route collapse:** controlled by principal LP and separately labelled MMA
  rows with non-interchangeable accounting.
- **F36 — stale qualification reuse:** controlled by exact Candidate C classifier,
  evaluator, contract, scope, provenance, and route hash binding in preflight.
