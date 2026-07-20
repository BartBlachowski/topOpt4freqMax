# A4 Recovery Phase 3 — Production Execution & Validation Report

**Role of this document:** scientific execution and validation record only. No algorithm,
configuration, protocol constant, or implementation was modified. Two blocking implementation
defects were discovered and are reported, not repaired, per the Phase 3 mandate.

**Executed:** 2026-07-20. MATLAB R2025b (25.2.0.3042426 Update 1), macOS arm64.
**Driver:** `a4_eigenpair_refresh(examples/Revision_v1/output/a4)`, all options at default.
**Base config:** `examples/Revision_v1/a4_ss_400x50_base.json`, hash `fnv1a32_c141e407`.
**Commit at execution:** `03729b4`.

---

## 1. Production summary

| Item | Value |
|---|---|
| Arms required (§9.1) | `{∞, 50, 10, 5, 1}` |
| Arms executed | **1 of 5** (`N = ∞` only) |
| Arms not executed | `N = 50`, `N = 10`, `N = 5`, `N = 1` |
| Driver-emitted run verdict | **HALTED** |
| Cause of halt | `N = ∞` bit-identity stop gate returned `pass = false` |
| Is the halt scientifically justified? | **No — it is a false positive (Defect P3-1)** |
| Secondary failure | Halt path crashed before writing reports/manifests (Defect P3-2) |
| Section 8.5 campaign verdict | **INCOMPLETE** |

The campaign terminated after arm 1. The termination was caused by a defective validator, not by
any deviation in the optimization. Execution was halted at this point on instruction and **not**
resumed by working around the gate.

### Measured cost model (provenance only, §3.8/§4.5)

| Quantity | Measured |
|---|---|
| Bare optimization iteration, 400×50 | 0.489 s |
| Screening event, full ladder `[20,40,80,160,320]` | 15.8–20.3 s |
| Screening event, `[20,40]` confirm-and-stop | ≈ 1.5–2 s |
| `N = ∞` arm, 2000 iterations + 25 events | **363.7 s** |

Projected complete five-arm sweep incl. both replays: ≈ 3–4 h. The registry estimate
(`run_all_revision_experiments.m:525`) carries `estRuntimeSeconds = 43200`, updated from the 57600
the audit flagged as m-10; it is within one order of magnitude of that projection, but R-5 cannot be
closed against an actual, because no complete sweep exists.

---

## 2. Validator summary

Each validator is confirmed or rejected individually. "Fixture" means the assertion was discharged
on synthetic or reduced-scale inputs only; "production" means at 400×50 in this run.

| Validator | Verdict | Evidence |
|---|---|---|
| **V-P2-1** — non-perturbation | **INCONCLUSIVE** | Declared test (full arm diagnostics-on vs diagnostics-off) is invoked *after* the identity gate and never ran. Supporting evidence is strong but not the declared test: all 25 production events carry `read_only_proof` with `design_bit_identical`, `reference_bit_identical`, `matrices_bit_identical`, `rng_bit_identical` **all true**. |
| **V-P2-2** — `N = ∞` bit identity | **CONFIRMED on the specification's criteria; the gate implementation is REJECTED** | All six §9.2 scalars bit-equal; topology CSV SHA-256 `9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806` — exactly the value §9.2 declares. The gate nonetheless returned false. See Defect P3-1. |
| **V-P2-3** — window recovery | **CONFIRMED (production)** | Iteration 25 → index **49**, `mac_prev = 0.9775288450111248`, `m_final = 160`, class **E-1**. Iteration 30 → index **37**, `mac_prev = 0.9663501395105896`, `m_final = 80`, class **E-1**. Reproduces the audit's independent measurements. Driver's own `window_recovery.pass = true`. |
| **V-P2-4** — screening symmetry | **CONFIRMED (fixture only)** | Fixture suite pass. Production cross-arm comparison is impossible with one executed arm. |
| **V-P2-5** — ladder determinism | **CONFIRMED (fixture only)** | Fixture suite pass. No production repeat performed. |
| **V-P2-6** — determinism replay, finite `N` | **REJECTED — not executed** | Requires a completed finite arm. None exists. `localRunFiniteReplay` never reached. |
| **V-P2-7** — classifier fixtures | **CONFIRMED** | 14/14 in-run fixture suite, covering E-0, E-1 (above index 20, asserting E-1 not E-2/B3), E-2a, E-2b, E-3→E-5 escalation, E-4 dual-condition, E-5 solver. |
| **V-P2-8** — factor drift | **CONFIRMED for the executed arm** | `base_config_hash = fnv1a32_c141e407` recorded on the arm and in Table A4-1; negative hash fixture passes. "All five arms share the hash" is unverifiable with one arm. |
| **V-P2-9** — Phase 1 regressions | **CONFIRMED** | Fixture suite reports Phase-1 regressions remain 10/10. |

**Production-only validators outstanding: V-P2-1 (production instance), V-P2-6.**
**Validator whose implementation is defective: V-P2-2.**

---

## 3. `N = ∞` verification — every declared invariant

Spec §9.2 declares seven invariants. All seven hold.

| Invariant | Expected (§9.2) | Reproduced | Bit-equal |
|---|---|---|:--:|
| `ω₁ᵗʳᵃᶜᵏ` | 159.56562699328325 | 159.56562699328325 | ✅ |
| final `max\|Δx_e\|` | 3.034903639330122e-03 | 3.034903639330122e-03 | ✅ |
| iterations | 2000 | 2000 | ✅ |
| `j*` | 1 | 1 | ✅ |
| MAC to `Φ₀` | 0.9996284251363903 | 0.9996284251363903 | ✅ |
| `ω₁/ω₂` gap | 67.37267502573462 | 67.37267502573462 | ✅ |
| topology CSV SHA-256 | `9c3d961b…f269806` | `9c3d961b…f269806` | ✅ |

Objective, frequencies, references, hashes, stopping iteration and topology are therefore all
verified. **No deviation exists.** The stopping iteration is the cap (2000), not convergence
(`change = 3.03e-03 > 1e-03`), consistent with the preserved run.

Independent adjudication of the gate's contrary claim:

```
isequal(in_memory_topology, readmatrix(preserved.csv))      = 0   <- what the gate tests
isequal(readmatrix(new.csv), readmatrix(preserved.csv))     = 1   <- CSV-level identity
isequal(roundtrip_15_sigdig(in_memory), readmatrix(pres.))  = 1   <- cause confirmed
max |in_memory - preserved|  = 5.551e-16   (max 6 ulps, 3719/20000 entries)
```

The topology CSV is written at `%.15g`; a double requires 17 significant digits to round-trip.
The gate compares a full-precision in-memory array against a truncated decimal read.

---

## 4. Arm summary and Section 8 assessment

| Arm | Executed | §8 status | Warnings | Phase 3 verdict |
|---|:--:|---|---|---|
| `N = ∞` | yes | **ACCEPTED WITH WARNING** | W-2, W-5 | **PASS** |
| `N = 50` | **no** | — | — | **INCONCLUSIVE** (no data) |
| `N = 10` | **no** | — | — | **INCONCLUSIVE** (no data) |
| `N = 5` | **no** | — | — | **INCONCLUSIVE** (no data) |
| `N = 1` | **no** | — | — | **INCONCLUSIVE** (no data) |

`N = ∞` satisfies §8.2 conditions 1–6 (no E-5; terminal state at cap with true iteration count;
no scheduled refreshes to attempt; all 25 grid points screened; per-iteration histories complete at
length 2000; all endpoints finite). Condition 7 holds (0 deferrals). Condition 8 fails — two §8.3
warnings apply — so the arm is **ACCEPTED WITH WARNING**, a valid scientific observation:

- **W-2 — ceiling reached:** `m_final = M_max = 320` at iterations 1, 2, 3.
- **W-5 — contamination observed:** E-4 at iterations 1, 2, 3.

W-1 (deferral), W-3 (unconfirmed), W-4 (deep truncation, index > 160) do **not** apply.

### `N = ∞` endpoint record

| Quantity | Value |
|---|---|
| `ω₁ᵗʳᵃᶜᵏ` | 159.56562699328325 |
| `ω₁ᵐⁱⁿ` | 159.56562699328325 |
| `ω₁ᵗʰʳᵉˢʰ` | 162.46769148252696 |
| MAC to `Φ₀` | 0.9996284251363903 |
| `j*` | 1 |
| `ω₁/ω₂` gap | 67.37267502573462 |
| grayness | 0.09691701139556157 |
| feasibility | 9.186229522450962e-05 |
| solid components | 1 |
| `limit_cycle` | `null` (§7.6 M-3 — correct) |
| serialized `N` | `"inf"` (not `null` — audit m-8 closed) |
| legacy companion class | ACCEPTED_WITH_BREAKDOWN / B4 (expected per §7.6 M-2) |

---

## 5. Event statistics (`N = ∞`, 25 events)

| Iteration | `m_final` | Rungs solved | Outcome | Selected | Stability | Admissible | Class |
|---:|---:|---|---|---:|---|---:|---|
| 1 | 320 | 20,40,80,160,320 | REFERENCE_UNAVAILABLE | — | n/a | 0/320 | E-2a, E-4 |
| 2 | 320 | 20,40,80,160,320 | REFERENCE_UNAVAILABLE | — | n/a | 0/320 | E-2a, E-4 |
| 3 | 320 | 20,40,80,160,320 | REFERENCE_UNAVAILABLE | — | n/a | 0/320 | E-2a, E-4 |
| 5 | 40 | 20,40 | SELECTED | 20 | confirmed | 1/40 | E-0 |
| 8 | 40 | 20,40 | SELECTED | 8 | confirmed | 1/40 | E-0 |
| 10 | 40 | 20,40 | SELECTED | 13 | confirmed | 1/40 | E-0 |
| 15 | 40 | 20,40 | SELECTED | 19 | confirmed | 1/40 | E-0 |
| **20** | **160** | 20,40,80,160 | SELECTED | **43** | confirmed | 1/160 | **E-1** |
| **25** | **160** | 20,40,80,160 | SELECTED | **49** | confirmed | 1/160 | **E-1** |
| **30** | **80** | 20,40,80 | SELECTED | **37** | confirmed | 1/80 | **E-1** |
| 40 | 40 | 20,40 | SELECTED | 18 | confirmed | 1/40 | E-0 |
| 50 | 40 | 20,40 | SELECTED | 5 | confirmed | 1/40 | E-0 |
| 75…2000 (13 events) | 40 | 20,40 | SELECTED | 1 | confirmed | 1/40 | E-0 |

Screening exposure: `G ∩ [1, 2000]` = all 25 grid points; the final iteration (2000) is already in
`G`. **25 events recorded, 25 expected — S-4 satisfied for this arm.**

---

## 6. Refresh statistics

| Quantity | `N = ∞` |
|---|---:|
| Scheduled refreshes | 0 |
| Effective refreshes | 0 |
| Deferred refreshes | 0 |
| Deferral fraction | 0.0000 |
| Longest consecutive deferral run | 0 |
| Degenerate (§5.4) | false |

The refresh code path was inert throughout, as §5.2 requires, and its inertness is corroborated by
the bit-identical reproduction of the preserved trajectory. Scheduled and effective counts are
recorded separately (I-8). The `1 + ⌊(n_iter−1)/N⌋` formula is not exercised by `N = ∞` and remains
untested in production — it can only be tested on a finite arm (**not executed**).

---

## 7. Candidate statistics

| Quantity | Value |
|---|---:|
| Candidate rows written | **2120** |
| Rows predicted from event table | 3·320 + 19·40 + 2·160 + 1·80 = **2120** ✅ |
| §6.1 mandatory fields present | **23 of 23** (no omissions, no extras) |
| Iteration-history rows | **2000** (= realized iteration count) ✅ |
| Ties broken by index (§3.6) | 0 |
| Events with `n_admissible = 0` | 3 |
| Events with `n_admissible = 1` | 22 |
| Events with `n_admissible ≥ 2` | **0** |

**Finding — selection is never competitive.** In all 22 successful events the admissible set
contained exactly one candidate. Spec §9.3 raised this concern for `N = 50`'s first refresh ("the
selection was forced, not competitive"); the frozen arm shows the same property at *every* event.
The four-condition screen is effectively a uniqueness filter at this mesh and formulation, so the
`argmax mac_prev` rule of §3.4 never actually discriminates between candidates. This is a
reportable property of the apparatus and bears on how §3.6's tie-break and §3.4's stability test
should be interpreted.

**R-1 hand reconstruction (iteration 25, index 49), from the CSV alone:**

```
mode_index 49   omega 160.770004231836   mac_prev 0.977528845011125
support_kinetic_fraction 0.941181050427319  >= 0.50  -> cond_kinetic_pass  = 1
support_connectivity     1                           -> cond_supports_pass = 1
low_density_strain_frac  0.0224669696707284 <= 0.50  -> cond_strain_pass   = 1
mac_prev                 0.977528845011125  >= 0.80  -> cond_mac_pass      = 1
admissible 1   selected 1   tie_flag 0
eigensolver_status: converged; residual=6.877463632157139e-10
```

Reconstruction succeeded without re-execution, for this arm. The §6.3 standard is met for
`N = ∞`; it is untested for the four unexecuted arms.

---

## 8. Failure taxonomy summary

| Class | Count | Iterations | Notes |
|---|---:|---|---|
| E-0 clean selection | 19 | 5–2000 | — |
| **E-1 window truncation** | **3** | 20, 25, 30 | Indices 43, 49, 37 — all above `m₀ = 20` |
| **E-2a no connected candidate** | **3** | 1, 2, 3 | Ceiling reached, solver healthy, admissible set empty |
| E-2b no continuous candidate | 0 | — | — |
| E-3 inadmissible selection | **0** | — | Tripwire never fired — **S-8 satisfied** |
| **E-4 disconnected-mode contamination** | **3** | 1, 2, 3 | Both conditions measured |
| E-5 implementation failure | 0 | — | No solver failure, no NaN/Inf, no telemetry mismatch |
| B3 | **0** | — | Retired code never emitted — **S-5 satisfied** |

Every E-2 event is sub-classified (all E-2a) — **S-7 satisfied**. Every screening event carries an
E-class.

### E-4 is genuinely measured for the first time

| Iteration | Condition 1: `n_solid_components ≥ 2` | Condition 2: best-`mac_prev` candidate support kinetic fraction `< τ_kin` |
|---:|---|---|
| 1 | **4** components | **0.000** < 0.50 |
| 2 | **4** components | **0.000** < 0.50 |
| 3 | **4** components | **0.000** < 0.50 |

Both the topological and the modal condition are recorded with measured values — **S-6 satisfied**.
This is the mechanism the completed run *named* (as B3) but never demonstrated.

**Scientifically the most important observation in this run:** these E-4 events occur in the
**frozen** arm, which never refreshes. The audit had noted `Solid components = 4` at iteration 2 of
the completed `N = 1` arm and the completed run attributed it to refreshing. It is now measured on
an arm where refreshing is impossible. Early-iteration disconnection and void-mode domination are
therefore properties of the **formulation at early iterations**, not consequences of the treatment.
This is direct, independent confirmation of the audit's C-2 thesis, obtained from the one arm whose
trajectory is provably unchanged.

Equally: the E-1 events at iterations 20/25/30 (indices 43/49/37) occur in the frozen arm, and all
three lie outside the retired 20-mode window. Under the completed implementation this arm would
have failed its screen at those iterations too. **C-1 and C-2 are both confirmed corrected, and
both are confirmed to have been arm-independent defects.**

---

## 9. Artifact verification

### Present and consistent with this run (15:12)

`a4_result.json`, `a4_screening_events.json`, `a4_pre_screen.json`,
`a4_eigenpair_refresh_results.mat`, `a4_candidate_telemetry.csv`, `a4_iteration_histories.csv`,
`a4_topology_inf.csv`, `a4_table.md`, `a4_table2.md`, and nine figures
`a4_fig1…a4_fig9` (including the two Phase-2-specific figures, `fig8_required_window` and
`fig9_selected_index`).

### Missing or defective

| Artifact | Status |
|---|---|
| `a4_topology_{50,10,5,1}.csv` | **MISSING** — arms not executed |
| `A4_RECOVERY_PHASE2_REPORT.md` | **NOT WRITTEN** — writer crashed (Defect P3-2); file on disk is the stale implementation-pass version dated 13:15 |
| `A4_RECOVERY_PHASE2_VALIDATION.md` | **NOT WRITTEN** — same cause, same staleness |
| `a4_manifest.json` | **STALE (11:42)** — lists the pre-run artifact set |
| `a4_stage_manifest.json` | **STALE (11:42)** — same |
| `a4_stage_result.json` | **STALE (11:42)** |
| Manifest agreement (§10.2, audit m-5) | **FAILS** — both manifests list `a4_topology_Ninf.csv`, `a4_fig2_tracked_vs_min.png`, `a4_fig3_refresh_events.png`, `a4_fig4_topologies.png`, none of which this run produces |
| Orphaned artifacts | `a4_fig2_tracked_vs_min.png`, `a4_fig3_refresh_events.png`, `a4_fig4_topologies.png`, `a4_topology_N50.csv`, `a4_topology_Ninf.csv` (00:34) coexist with the new set under different names |
| Git tracking (§10.9, R-2) | **FAILS** — only `a4_result.json` is tracked. `.gitignore` correctly un-ignores `output/a4/*.{mat,png,csv}`, so the artifacts are trackable; they were simply never added. Attributable to the halt. |

**Preserved baseline integrity:** the driver reads the baseline from `a4_topology_Ninf.csv` and
writes to `a4_topology_inf.csv`, so the preserved Phase-1 artifacts were **not** overwritten. A full
backup of the pre-run `output/a4` was taken before execution regardless. Baseline hashes verified
unchanged: `N=inf` `9c3d961b…`, `N=50` `5d9bdd21…`.

---

## 10. Defects discovered

### Defect P3-1 — `V-P2-2` topology check is unsatisfiable (BLOCKING, false stop condition)

**Location:** `examples/Revision_v1/a4_eigenpair_refresh.m:849-850`, `localFrozenIdentity`.

```matlab
gate.topology_bit_identical = ~isempty(preservedTopology) && ...
    isequal(arm.topology(:), preservedTopology(:));
```

`preservedTopology` comes from `readmatrix` of a CSV written at `%.15g`. `arm.topology` is a
full-precision in-memory double. A double needs 17 significant digits to round-trip, so the two can
differ by up to a few ulps on any value that is not exactly representable in 15 digits — here,
3719 of 20000 entries, max 6 ulps, max absolute deviation 5.551e-16.

**The comparison can therefore never succeed, for any run, including a perfect one.**

Spec §9.2 states the criterion as *"topology CSV SHA-256 unchanged"*. That criterion **passes**.
The implementation substituted a stricter and unsatisfiable test that the specification does not
ask for.

**Consequence:** every Phase 2 campaign halts after arm 1 with verdict HALTED, reporting a
diagnostic-perturbation stop condition that did not occur. Per §8.5 this asserts that "the
diagnostic instrumentation perturbed the optimization, invalidating every arm" — a claim the run's
own artifacts contradict.

### Defect P3-2 — the HALTED path destroys its own record (BLOCKING)

**Location:** `examples/Revision_v1/a4_eigenpair_refresh.m:622`, `localSection11Lines`, reached from
the halt branch at line 194.

```matlab
v=res.validation; ac=res.acceptance_checks;   % line 622
```

`res.acceptance_checks` is assigned only at line 220, *after* the arm loop. On the halt path at
line 194 the field does not exist:

```
Unrecognized field name "acceptance_checks".
  in localSection11Lines (line 622)
  in localWriteRecoveryReport (line 616)
  in localWriteArtifacts (line 452)
  in a4_eigenpair_refresh (line 194)
```

**Consequences:**
1. `A4_RECOVERY_PHASE2_REPORT.md` and `A4_RECOVERY_PHASE2_VALIDATION.md` are never written.
2. Manifests and `a4_stage_result.json` are never refreshed, so the artifact set on disk
   contradicts both manifests.
3. The intended `a4:FrozenBitIdentityFailed` error is replaced by an unrelated field-access error,
   so **the actual reason for the halt is never reported to the operator.**

This violates §4.5 (telemetry shall survive a halt) and §8.5 (HALTED must be a reportable state).
It is the same class of defect as audit finding **M-5** — the completed run's driver discarding
accumulated telemetry on the failure path — which Phase 2 exists in part to correct. The defect is
present specifically on the failure path, i.e. exactly where §4.5 applies.

### Defect P3-3 — artifact-set contamination and manifest divergence

New figures adopt different filenames from the completed run (`a4_fig2_mac_vs_iteration.png` vs
`a4_fig2_tracked_vs_min.png`, etc.), and topologies change from `a4_topology_Ninf.csv` to
`a4_topology_inf.csv`. Old files are neither removed nor superseded in place, so the directory holds
two overlapping generations, and both manifests describe the older one. §10.2 requires the two
manifests to list an identical artifact set (audit m-5); they currently agree with each other but
not with reality.

---

## 11. Section 11 checklist status

Assessed against what this execution actually evidences.

| Item | Status |
|---|---|
| S-1 question/δ/levels/outcomes unchanged | **SATISFIED** |
| S-2 single-factor guarantee | **PARTIAL** — holds for the executed arm; cross-arm claim untestable with one arm |
| S-3 window as response variable | **SATISFIED** — Table A4-2 and Figure 8 both present |
| S-4 identical screening exposure | **PARTIAL** — 25/25 for `N = ∞`; unverifiable across arms |
| S-5 no B3 emitted | **SATISFIED** |
| S-6 E-4 dual-condition evidence | **SATISFIED** |
| S-7 every E-2 sub-classified | **SATISFIED** |
| S-8 no E-3 | **SATISFIED** |
| S-9 out-of-scope items untouched and stated | **SATISFIED** — `limit_cycle: null`, B4 retained, no decision emitted |
| I-1 constants in one block, matching §3.2/§4.1 | **SATISFIED** — `a4_phase2_constants.m` verified value-by-value |
| I-2 `m₀`-then-confirm search | **SATISFIED** — every successful event solved ≥ 2 rungs |
| I-3 widest-window reporting | **SATISFIED** |
| I-4 no short-circuit admissibility | **SATISFIED** — all four conditions recorded on all 2120 rows |
| I-5 diagnostics read-only, cannot terminate | **SATISFIED** — `read_only_proof` all-true on all 25 events |
| I-6 §5.2 refresh / §5.4 deferral | **UNTESTED** — requires a finite arm |
| I-7 telemetry persists before halt | **VIOLATED** — Defect P3-2 |
| I-8 scheduled vs effective counts, V-A4-3 formula | **PARTIAL** — counts separated; formula unexercised by `N = ∞` |
| I-9 `limit_cycle: null`, `N` as `"inf"` | **SATISFIED** |
| V-P2-1 … V-P2-9 | See §2 above |
| R-1 reconstructability | **PARTIAL** — demonstrated for `N = ∞`; one E-1 and one E-2 reconstructed; no finite-arm data |
| R-2 artifacts exist, tracked, in both manifests | **VIOLATED** |
| R-3 reachable commit SHA + config hash | **PARTIAL** — `03729b4` and `fnv1a32_c141e407` recorded in written artifacts; stale manifests carry neither |
| R-4 no protocol constant altered | **SATISFIED** — nothing was changed at any point |
| R-5 runtime estimate within one order | **UNEVALUABLE** — estimate is 43200 s; no complete sweep to compare |
| D-1 both reports exist and are complete | **VIOLATED** — Defect P3-2 |
| D-2 `A4_SPECIFICATION_V3.md` carries amendment pointers | **SATISFIED** — lines 330, 372, 459, 534 |
| D-3 Tables A4-1 and A4-2 issued | **PARTIAL** — both issued, one row each |
| D-4 nine figures, δ band, `Interpreter','none'` | **PARTIAL** — nine figures exist; single-arm content |
| D-5 `N = 50` pre-/post-Phase-2 endpoints both reported | **NOT SATISFIED** — `N = 50` not executed |
| D-6 open findings stated | **SATISFIED** in this document |

---

## 12. Remaining issues

1. **Defect P3-1** — `V-P2-2` topology comparison is unsatisfiable; halts every campaign at arm 1.
2. **Defect P3-2** — the halt path crashes, suppressing the true halt reason and the two mandated
   reports; violates §4.5 and I-7.
3. **Defect P3-3** — orphaned artifacts and manifests that describe a superseded artifact set.
4. **Four arms carry no data.** `N = 50, 10, 5, 1` are the arms that carry the entire scientific
   question (§9.4). `N*` remains unlocated. A4 remains a one-point design — weaker than the
   two-point design the completed run produced.
5. **`n_admissible = 1` at every successful event** (§7) — the selection rule never discriminates.
   Not a defect, but it means the `argmax mac_prev` rule, the §3.6 tie-break, and arguably the §3.4
   stability test are untested by production data as *selection* mechanisms.
6. **V-P2-1's production instance and V-P2-6 remain undischarged**, exactly the two validators the
   Phase 2 implementation report itself listed as pending. Phase 2's own stated gap is unclosed.
7. **R-5 unevaluable**; **I-6 and the V-A4-3 formula untested** in production.

---

## 13. Final recommendation

1. **Do not report Phase 2 as COMPLETE.** The §8.5 verdict is INCOMPLETE, and the driver's own
   emitted verdict is HALTED.
2. **Fix Defect P3-1 to match the specification**, not more strictly: compare the topology CSV
   SHA-256 as §9.2 declares, or compare `readmatrix(new)` against `readmatrix(preserved)`. Do not
   loosen it to a tolerance — exact CSV identity is achievable and is what the spec asks for.
3. **Fix Defect P3-2 before any re-run**, because it is the defect that makes every *other* failure
   unreportable. Initialize `acceptance_checks` at struct construction, or guard the halt path.
   Note that a specification whose §4.5 mandates telemetry survival was implemented with a crash on
   the failure path — that pairing deserves a regression test of its own.
4. **Add a regression test that runs the halt path**, asserting reports and manifests are written
   and the halt reason surfaces. No existing fixture covers it: the suite is 14/14 green while both
   blocking defects are present, which is itself a coverage finding.
5. **Clear or version the output directory** before re-running, so orphaned artifacts cannot be
   mistaken for current ones.
6. **Then re-execute all five arms** — approximately 3–4 h on this hardware, well inside the 43200 s
   registry budget — and discharge V-P2-1 (production), V-P2-6, S-4 across arms, I-6, I-8, R-1,
   R-2, R-3, R-5, D-1, D-3, D-4, D-5.
7. **Preserve the `N = ∞` result from this run.** It is a fully valid, bit-verified arm and the
   E-1/E-2a/E-4 findings on the frozen trajectory are scientifically load-bearing independently of
   the campaign's completion.

---

## 14. Final conclusion

> **Does A4 Recovery Phase 2 successfully satisfy its scientific specification under production
> execution?**
>
> **No.**

The failure is one of **completeness and of validator implementation**, not of scientific method.
What was executed was executed correctly, and the two corrections Phase 2 exists to deliver are
both confirmed at production scale:

- **C-1 is fixed.** The adaptive ladder located the physical mode at index 49 (iteration 25) and
  index 37 (iteration 30), reproducing the audit's independent measurements exactly and classifying
  both as E-1 rather than as a terminal failure.
- **C-2 is fixed, and is confirmed to have been arm-independent.** The frozen arm — which cannot
  refresh — exhibits the same early-iteration screen failures (E-2a at iterations 1–3) and the same
  disconnected-mode contamination (E-4, 4 components, best-candidate support kinetic fraction 0.000)
  that the completed run attributed to small `N`. The confound is not merely removed by
  construction; it is now measured to have been a confound.
- **`N = ∞` reproduces bit-identically** on all seven §9.2 invariants, discharging the scientific
  content of V-P2-2, V-A4-5 and V-A4-6 at production scale.

**Specification items that remain unsatisfied:**

| Item | Why |
|---|---|
| **§8.5 COMPLETE** | Only 1 of 5 arms executed; verdict INCOMPLETE |
| **V-P2-1** (production instance) | Never ran — gated behind the halt |
| **V-P2-2** (as implemented) | Gate is unsatisfiable by construction — Defect P3-1 |
| **V-P2-6** | No finite arm exists to replay |
| **I-7** | Telemetry did not survive the halt — Defect P3-2 |
| **S-2, S-4** | Cross-arm guarantees untestable with one arm |
| **I-6, I-8** (formula) | Refresh/deferral paths unexercised |
| **R-1** (finite arms), **R-2**, **R-3**, **R-5** | Artifacts untracked, manifests stale, no actual runtime |
| **D-1, D-3, D-4, D-5** | Reports unwritten; tables and figures single-arm; `N = 50` comparison impossible |

Two blocking implementation defects stand between the current state and a COMPLETE campaign.
Neither is scientific; both are mechanical; both are in the *validation* apparatus rather than in
the experiment. Consistent with the Phase 3 mandate, neither was repaired.
