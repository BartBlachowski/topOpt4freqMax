# Final pre-production audit — iteration-efficiency paper campaign

**Verdict: NOT AUTHORIZED.** Three blockers, all narrowly scoped and correctable without a new methodology phase.

The question this audit answers: *can the nine-mesh paper-facing production campaign now be run without changing the scientific meaning of the experiment?* Not yet — but the gap is small, and the scientific core of the harness is sound.

## What is right

The parts most likely to be wrong are, in fact, right, and I verified them by execution rather than by reading the integration report:

- **The native optimizer modifications are bit-neutral.** Pre-integration versions of `olhoffOptStabilized.m` and `innerLoopLP.m` were restored from HEAD and run against the instrumented ones on the same configuration. Final density, eigenvalues, `dxOuter`, `beta`, `lpFlag`, `nInner`, `N`, `policyStage` and `moveLimit` are all bit-identical, with capture ON and OFF alike. The historical `single` snapshot never touched the optimizer's own `rho`, so promoting storage to `double` could not perturb anything — and did not.
- **Candidate C is implemented exactly as frozen.** Actual gray field, E1 linear-with-floor, E2 and E3 on Eq. (4a) with the E3 effective-density clamp, strict unanimous classifier, IPR non-binding, adaptive 3→6→12→24→48 with no scientific ceiling, fail-closed on unresolved modes. No fallback to the lowest eigenmode, first three modes, Candidate D, Eq. (4), or a native E2/E3 interpretation. The ordinal-13 anchor reproduces exactly: `[1 7 13]`, schedule `[3 6 12 24]`, three escalations, margins `[0.4995 0.4999 0.4458]`.
- **Every reference and endpoint anchor reproduces.** `b_ref = 2100`, `Q_ref = [162.6601, 162.9978, 162.9978]`, `B_meas = 3200`, and all nine (q, P) endpoint pairs — 229/328, 309/408, 453/552 at P=100, plus the P=50 and P=200 sensitivities. No horizon-relative max, no cap fallback, no look-ahead.
- **The hard topology gate is the repaired one.** Exact-count projection with index tie-break, support-footprint connectivity, per-component significance. The aggregate detached-area veto is genuinely absent — aggregate quantities are computed and stamped `DIAGNOSTIC_ONLY`.
- **No authoritative float32 path is reachable.** Enforced at four independent layers.
- **All 19 executed negative controls pass**, including stale Candidate-C hash, wrong contract, float32 policy, unresolved structural mode, missing reference, and invalid result schema.
- **Both interpretations survive.** The harness delivers iteration counts *and* wall-clock time, per-iteration cost, and method-specific inner-work — it did not decay into an endpoint-count study.

## What blocks

### B-1 (CRITICAL) — the Yuksel timing replay times a different computation

`run_timing_firewall.localOnce` sets `max_iters = k_enter` for Yuksel, but `run_topopt_from_json.m:535` clamps `stage1MaxIter = min(stage1MaxIter, maxiter)`. Measured at 96×12: the reference run gives Stage 1 = 151 / Stage 2 = 3200; the timing replay at horizon 150 gives Stage 1 = **150** / Stage 2 = 150.

A truncated Stage 1 hands a different design to Stage 2, so the timed run is not the run whose endpoints are reported, and Stage-1 work is under-counted. Nothing detects it. The frozen campaign records Yuksel Stage 1 hitting its cap at 640×80, 720×90 and 800×100 — exactly where timing matters most. This corrupts the computational-performance half of the paper.

### B-2 (MAJOR) — "common support" is asserted but never enforced

`fit_scaling_table.m` hardcodes `support = "common"` while fitting each method over its own meshes. Measured: Proposed at 9 meshes and Olhoff-lp at 7 both emit `support = "common"` with `n_valid = [9 7]`. The frozen contract requires `common_support_companion_required: true` and forbids `cross_method_comparison_outside_common_support`. Unequal support is the *expected* case here, given the documented Olhoff 800×100 `RUN_ERROR` and the LP iteration-limit failure at 400×50. The smoke test cannot catch this because its synthetic input gives every method every mesh.

### B-3 (MAJOR) — a single cell failure aborts the entire campaign

`localProduction` has no `try`/`catch`, and rows are written only after the full mesh × method loop. A `SOLVER_FAILURE` — documented as an actually-occurring LP iteration-limit failure — throws and destroys every completed cell. It also means a failing 800×100 cannot *record* a genuine `RUN_ERROR`; the contract defines that status and the harness never emits it. Failing closed is right; failing the whole campaign is not.

## Cost

The **principal** nine-mesh campaign is practical: ~702 optimization runs, ~67 h native optimization, ~54 h offline Candidate-C and topology evaluation, **~120 h ≈ 5 days** serial single-threaded, and **~80–110 GB** storage.

The **optional MMA** nine-mesh campaign is not. Measured at 160×20, nested MMA costs **12.58 s/outer against LP's 0.184 s — 68×**, with ~100 inner MMA iterations per outer. Nine meshes would be an order-10³-hour campaign. Recommended scope: three coarse meshes (~150–200 h) if an MMA scaling exponent is wanted, or 160×20 alone (~25 h) for inner-work characterisation. It must not gate the principal campaign, and the frozen protocol does not require it to.

## Non-blocking findings

F-04 `ie2a.account_iterations` is hash-pinned but never executed (accounting is reimplemented inline in `build_rows`, correctly — a provenance defect). F-05 the contract's Olhoff source hashes are stale and `validate_contract` is called with `VerifyFiles=false`, silencing it. F-06 `tail_truncated` is a P=100 quantity written onto P=50/200 rows. F-07 MMA cost and an undocumented LP-vs-MMA filter-radius asymmetry (1.3 elements vs 0.06 physical). F-08 `provenance_hash` always empty. F-09 unused `olhoff_variant_plan` runner/role strings disagree with what executes. F-10 timing components and range/MAD not fully realised.

## Severity tally

| severity | count | blocking |
|---|---:|---:|
| CRITICAL | 1 | 1 |
| MAJOR | 2 | 2 |
| MODERATE | 4 | 0 |
| MINOR | 3 | 0 |

## Deliverables

`SOURCE_IDENTITY_AUDIT.md` · `NUMERICAL_BEHAVIOR_NEUTRALITY.md` · `METHOD_ROUTING_AUDIT.md` · `CANDIDATE_C_VERIFICATION.md` · `ACCOUNTING_AND_TIMING_AUDIT.md` · `PERFORMANCE_OUTPUT_AUDIT.md` · `PRODUCTION_MANIFEST_AUDIT.md` · `PREFLIGHT_NEGATIVE_CONTROLS.md` · `PRODUCTION_COST_ESTIMATE.md` · `PRODUCTION_LAUNCH_CHECKLIST.md` · `BLOCKERS.md` · `FINDINGS.csv` · `audit_provenance.json` · `SHA256SUMS.txt`

No `PRODUCTION_AUTHORIZATION.md` is issued.

---

**FINAL PRE-PRODUCTION AUDIT FAILED — NINE-MESH CAMPAIGN NOT AUTHORIZED**
