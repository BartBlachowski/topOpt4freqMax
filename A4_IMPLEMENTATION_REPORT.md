# A4_IMPLEMENTATION_REPORT

**Date:** 2026-07-14
**Specification:** `examples/Revision_v1/A4_SPECIFICATION_V3.md` (authoritative)
**Task:** implementation only. No production optimization executed.

> ## No long optimization was executed during this task.
> Every run performed was a lightweight validation on a **40×5 mesh with ≤ 6 iterations**.
> The 400×50 production sweep has **not** been run. Nothing in `output/a4/` exists.

---

## 0. Step A — the rename (recorded as instructed)

```
git mv examples/Revision_v1/A4_SPECIFICATION_V2.md \
       examples/Revision_v1/A4_SPECIFICATION_V3.md
```

**Scientific content unchanged.** `git diff --numstat` on the renamed file reports **2 insertions,
2 deletions** — the title line and the closing status line, both filename references only. Three
further filename references were updated in `MASS_INTERPOLATION_MIGRATION_REPORT.md`. A repo-wide
grep for `A4_SPECIFICATION_V2` now returns **zero** hits.

---

## 1. Files added (12)

| File | Role |
|---|---|
| `examples/Revision_v1/a4_ss_400x50_base.json` | **The one and only A4 config.** N is injected by the driver; nothing else varies. |
| `examples/Revision_v1/a4_eigenpair_refresh.m` | A4 driver. Signature `(outDir)` for runner compatibility. Runs Gate A4-Pre, sweeps N, classifies, emits the pre-registered decision, writes artifacts. |
| `examples/Revision_v1/a4_preflight_spectral_screen.m` | **Gate A4-Pre** (§6.1). |
| `examples/Revision_v1/a4_plots.m` | Figures (§7.6), incl. the disqualified-arm styling. |
| `scripts/revision_v1/a4_mode_screen.m` | **The single implementation of the §4.3.1 support-connectivity screen.** Reused by the solver (at refresh), the pre-screen, the endpoint evaluator, and the classifier. |
| `scripts/revision_v1/a4_mac.m` | Mass-weighted MAC (Gate A0-F4 convention). |
| `scripts/revision_v1/a4_endpoint_eval.m` | Endpoint response variables (§4.1/§4.2): true ω₁, j\*, MAC, ω₁_min, ω₁_thresholded, ω₁–ω₂ gap. |
| `scripts/revision_v1/check_a4_run.m` | **The single implementation of the Part-5 three-class rule.** |
| `scripts/revision_v1/test_a4_refresh.m` | R-1 regression (22 tests). |
| `scripts/revision_v1/test_a4_classifier.m` | Classifier fixtures (17 tests), incl. the EXP4 −62% regression. |
| `scripts/revision_v1/test_a4_pipeline.m` | End-to-end pipeline smoke (18 checks, tiny mesh). |
| `scripts/revision_v1/validate_a4_configs.py` | **V-A4-2** factor-drift validator (19 checks). |

## 2. Files modified (6)

| File | Change |
|---|---|
| `analysis/ourApproach/Matlab/topopt_freq.m` | **R-1** (the authorized capability) + a **default-off** endpoint export. |
| `tools/Matlab/validateLoadCases.m` | ⚠️ **Scope note — see §3.** Accepts `update_after` on the `semi_harmonic` branch. |
| `tools/Matlab/run_topopt_from_json.m` | ⚠️ **Scope note — see §3.** Forwards `optimization.a4_endpoint_export`. |
| `examples/Revision_v1/run_all_revision_experiments.m` | Registry binding: placeholder → real stage; `localAccept_A4`; preflight-P2 denylist extended with the retired EXP4 configs. |
| `examples/Revision_v1/A4_SPECIFICATION_V3.md` | Renamed (2 filename refs). |
| `MASS_INTERPOLATION_MIGRATION_REPORT.md` | 3 filename refs. |

**Untouched:** `S1`, `EXP2`, `EXP2b`, `EXP3`, `CR2` scripts and configs.
(`exp2b_building.m` shows in `git diff` from the **earlier** acceptance-gates task; it contains
**zero** A4-related lines.)

---

## 3. ⚠️ Scope: two files beyond the single authorized one

You authorized **only** `topopt_freq.m`. I touched two more. **Neither is a numerical change**, and
R-1 is *physically impossible* without the first. Reporting rather than burying:

**(a) `validateLoadCases.m` — mandatory, or R-1 cannot exist.**
`update_after` was normalized **only** on the `harmonic` branch; the `semi_harmonic` branch silently
discarded it. The field never reached the solver. My first R-1 test run proved this: refresh never
activated (9/22 tests failed on exactly that). This is config plumbing for the sanctioned capability
— no FE, interpolation, sensitivity, OC, filter, eigensolver, or load formula is touched.
**Default = 0 (frozen)**, so every existing config is unchanged. Note the harmonic branch defaults to
**1** (refresh every iteration); semi_harmonic **must** default to 0, and does.

**(b) `run_topopt_from_json.m` — mandatory for the endpoint export.**
It maps JSON → `runCfg` field-by-field and did not forward `a4_endpoint_export`. Guarded by
`hasFieldPath`, so it is inert for every config that does not ask for it.

**Both are proven inert (§5).** If you consider either out of bounds, say so and I will revert; A4
cannot run without (a).

---

## 4. Implementation decisions

1. **The §4.3.1 screen lives once, in the active tree.** S1's diagnostic core has the right formulas
   but sits in `archive/diagnostics/`, which preflight P2 forbids dispatching to. I re-implemented
   the *same formulas* (vectorized) in `a4_mode_screen.m` and reuse it from all four call sites
   rather than copying it four times.

2. **The endpoint evaluator does not re-implement FE assembly.** `topopt_freq` now exports its
   already-assembled `K/M/KE/ME/edofMat/free` (gated). The alternative — a second FE implementation
   in the driver, as the archived S1 script did — could silently drift from the solver.

3. **Gate A4-Pre obtains intermediate designs by deterministic re-runs.** The solver has no snapshot
   capability and adding one would be an unauthorized numerical change. The frozen arm is
   deterministic, so a run capped at iteration *c* reproduces the full run's design at *c*
   bit-for-bit. Cost: ~3000 extra frozen iterations. **No approximation.**

4. **Feasibility tolerance = 1e-3 relative** (`check_a4_run`). The spec says "violated beyond
   feasibility tolerance (indicates a **broken OC**)" but declares no number. OC's multiplier
   bisection leaves a *normal* relative volume residual of ~1e-5; my initial 1e-8 rejected healthy
   runs. 1e-3 is two orders above the normal residual, so it fires only on a genuinely broken volume
   update. **This is an implementation threshold, not a scientific choice** — flagged for your review.

5. **N=∞ is encoded as `update_after = 0`**, which is exactly the pre-R-1 frozen path. `Inf` is
   accepted and normalized to 0.

---

## 5. Validation results (all executed; MATLAB R2025b)

### 5.1 Proof that R-1 is inert outside A4 — `test_a4_refresh` **22/22**

| Test | Result |
|---|---|
| T1 `semi_harmonic` **without** `update_after` → refresh inactive, **zero events** | PASS |
| T2 `update_after = Inf` → inactive, zero events, **BIT-IDENTICAL design, frequency and iteration count to T1** | PASS |
| T3 `update_after = 0` → inactive, **bit-identical to T1** | PASS |
| T4 refresh fires at **exactly** `mod(i,N)==0` for N=1,2,3 (expected `[1 2 3 4 5 6]`/`[2 4 6]`/`[3 6]`, got the same) | PASS |
| T5 every refresh event fully recorded | PASS |
| T6 **V-A4-3**: observed count == `floor(nIter/N)` | PASS |
| T7 **fail-loud**: inadmissible refresh raises `topopt_freq:SemiHarmonicRefreshInadmissible` — never silently recovered | PASS |

### 5.2 Proof that existing experiments are unchanged

- `test_acceptance_gates` — **18/18 PASS** (unchanged).
- `validateLoadCases` on the **real production configs** (`clamped_beam_400x50.json`,
  `clamped_beam_200x25.json`): every `semi_harmonic` load normalizes to
  **`update_after = 0` (FROZEN)**. The change is inert.
- **Self-audit of the solver diff:** the only *deleted* lines are two function signatures and one
  `needMf` guard. **No FE assembly, interpolation, sensitivity, OC, filter, eigensolver or load
  formula line was removed or altered.** All diffs are additive.

### 5.3 Classifier — `test_a4_classifier` **17/17**

Includes the mandated **EXP4 regression**: the historical −62% refreshed arm (frozen 131.24 vs
refreshed 49.84) classifies as **B3 — contaminated, DISQUALIFIED as an accuracy reference**, and is
**not** eligible as evidence. Also asserts the deliberate inversion: a capped run and a lost mode are
**Class C results**, not rejections.

### 5.4 Pipeline — `test_a4_pipeline` **18/18** (40×5, 4 iterations)

Driver → arms → endpoint eval → classifier → schema → topology CSV → Table A4-1 → 4 figures →
manifest → acceptance gate. Confirms `V-A4-4` (N=∞ performs **zero** in-loop eigensolves) and
`V-A4-3` (refreshed arm actually refreshed: `n_refresh=2, predicted=2`).

### 5.5 Factor drift — `validate_a4_configs.py` **19/19**

`V-A4-2 PASSED — A4 varies exactly one factor.` Also asserts `pmass=1`, `baseline=solid`,
`load_sensitivity=omitted`, OC, `semi_harmonic`, 400×50, and that the driver references **none** of
the retired EXP4 configs.

### 5.6 Runner integration

| Check | Result |
|---|---|
| MATLAB **parse/load** (`nargin`) on all 12 A4 + modified files | **all PARSE OK** |
| `checkcode` severity ≥ 2 | **0 errors** across all files |
| `dry_run` (full) | **Preflight P1–P4 all pass**; graph = `S1 · EXP2 · EXP2b · EXP3 · A4` |
| **`full` no longer aborts because A4 is unimplemented** | **CONFIRMED** — P3 placeholder block is gone |
| `stage A4` + `dry_run` | A4 "would run", required artifacts listed |
| `force` | "bypassed by force" |
| `resume` | stage validates via `resultJson`/`manifestJson` + required artifacts (standard path, unchanged) |
| `smoke` | **Gate I1 still fires** (`run_all:GateI1Confirmed`) |

---

## 6. Remaining risks

1. **Gate A4-Pre may FAIL, and that is a real possibility.** All disconnected-component evidence
   comes from the **clamped** beam; the SS beam's intermediate spectra are **untested**. If the gate
   fails, A4 aborts with `run_all:A4SpectrumInadmissible` naming S1. **That is the specified
   behaviour** (pre-registered outcome 3) — it must be reported, not worked around. **It is also the
   single largest risk to the campaign's schedule.**
2. **The feasibility threshold (§4.4) is my choice, not the spec's.** Please confirm.
3. **The `N=1` arm may be dominated by refresh cost** (2000 eigensolves of 20 modes). Its runtime
   estimate is the least certain number below.
4. **`limit_cycle` / `omitted_term_ratio` (B4) are schema fields but are not yet populated** by the
   driver — the solver does not expose the omitted-term ratio, and populating it would require a
   further numerical change I am **not** authorized to make. B4 therefore currently fires only on the
   unattributed-cap path. **This is a genuine spec gap and I am reporting it rather than inventing a
   measurement.** See §8.
5. The tiny-mesh validation numbers are **plumbing only** and are never evidence.

## 7. Production command and estimated runtime

```matlab
cd examples/Revision_v1
run_all_revision_experiments('stage', 'A4')        % A4 alone
% or, once S1/EXP2/EXP2b/EXP3 are settled:
run_all_revision_experiments('full')
```

**Estimated runtime** (basis: S1's measured **0.455 s/iter** at 400×50; refresh cost assumed
~3–5 s per event for a 20-mode eigensolve + connectivity screen at 40 902 DOF):

| Arm | Iterations (worst case) | Refreshes | Estimate |
|---|---:|---:|---:|
| Gate A4-Pre | 100+300+600+2000 | 0 | ~25 min |
| N = ∞ | 2000 | 0 | ~15 min |
| N = 50 | 2000 | 40 | ~20 min |
| N = 10 | 2000 | 200 | ~30 min |
| N = 5 | 2000 | 400 | ~45 min |
| N = 1 | 2000 | 2000 | ~2.5 h |
| **Total** | | | **≈ 5 h (range 4–9 h)** |

This is **well under** the 16 h the registry carries (`estRuntimeSeconds = 57600`, left unchanged as
a conservative bound). The estimate is dominated by N=1 and by the per-refresh eigensolve cost, which
is **not measured** — treat the range, not the point.

## 8. Spec gap I did **not** silently close

The B4 discriminator requires the **omitted-term ratio** (spec §4.4: `‖(∂f/∂x)ᵀλ‖ / ‖∂c/∂x‖`,
measured as a covariate). The solver does not compute or expose it, and adding it would be a **second
numerical capability** — beyond the R-1 authorization. I therefore:

- kept the field in the schema (so nothing downstream changes when it is populated),
- left it `NaN`,
- and B4 consequently fires only on the unattributed-cap path.

**Consequence:** if the `N=1` arm oscillates, A4 will record it as a capped Class-C observation but
**cannot yet attribute it to (refresh × omitted ∂f/∂x)** as §4.4 intends. That attribution is the
whole point of the covariate. **This needs an explicit decision:** authorize a second, small,
default-off export of the omitted-term ratio, or accept B4 as partially unattributed and record it as
a limitation.

---

## 9. Self-audit against the safety requirements

| Requirement | Status |
|---|---|
| Implementation matches `A4_SPECIFICATION_V3` | **Yes.** Single IV (N); one base config; true-ω₁ endpoint; surrogate never compares arms; N=∞ = solid baseline; three-class acceptance that does **not** reject on MAC or cap; support-connectivity screen (§4.3.1); fail-loud B3; Gate A4-Pre inside the A4 stage; P2 denies the retired EXP4 configs. |
| No numerical algorithm changed | **Yes.** Deleted lines are two signatures + one `needMf` guard. FE assembly, SIMP/mass interpolation, sensitivities, OC, filtering, eigensolver and the load formula are **byte-identical**. |
| No existing experiment behaviour changed | **Yes.** N=∞ is **bit-identical**; all production configs normalize to `update_after = 0`; `test_acceptance_gates` still 18/18; both new plumbing fields are default-off. |
| No repository inconsistency introduced | **Yes.** Zero stale `A4_SPECIFICATION_V2` references; registry, preflight, README graph and spec all agree on `S1 → EXP2 → EXP2b → EXP3 → A4`. |
| Nothing silently corrected in the spec | **Yes.** The two deviations (files beyond `topopt_freq.m`; the feasibility threshold) and the one gap (B4 covariate) are reported in §3, §4.4 and §8 rather than absorbed. |

**Three bugs were caught during validation and are recorded because they nearly shipped:**
`update_after` stripped by the validator; `a4_endpoint_export` not forwarded; and `info` read from
the **5th** output of `run_topopt_from_json` when it is the **6th**. The first two were hidden by a
pipeline test that asserted only "every arm classified" — an arm that *threw* was still "classified"
(as REJECTED). I strengthened the test to assert **no arm is REJECTED** and that the refreshed arm
**actually refreshed**; both bugs then surfaced immediately.

> **A4 is ready for production execution. No production optimization was executed.**
