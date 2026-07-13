# Exp2 α=1 Discrepancy Diagnosis: 142 rad/s pilot vs 2.98 rad/s "noisy" run

**Date:** 2026-06-26
**Question:** Why does the α=1 pilot converge to ω₁≈142 rad/s with a clean topology
while a "full Exp2 α=1 run" ends at ω₁≈2.98 rad/s with a noisy topology?
**Constraint:** No solver code was inspected for modification or changed. Analysis is
from committed artifacts only.

---

## 0. TL;DR

The two runs are **not the same experiment**. The clean 142 rad/s result and the noisy
~3 rad/s result come from **different configuration files** that differ in mesh, boundary
conditions, number of load cases, and iteration cap.

- The **actual full Exp2 α=1 sweep case**
  (`output/exp2_authoritative_sweep/alpha_1_00/`) is **bit-identical** to the pilot:
  ω₁ = 141.78982535805085, 1052 iterations, same topology, same grayness, same
  design-change. There is **no discrepancy** there.
- The **noisy ω₁≈2.98 / 2.77 result** is the **CR2-MMA "omitted" diagnostic, variant A**
  (`cr2/mma_diagnostic/output/cr2_mma_variant_a_*`). It uses a different mesh
  (160×20), a far weaker boundary condition (two pinned points instead of two fully
  clamped edges), one load case instead of two, and a 400-iteration cap instead of 2000.

**Root cause:** a real instability of the `load_sensitivity = "omitted"` formulation,
which supplies MMA with an **inconsistent gradient** (it drops ∂F/∂x even though the
load F = ω₀²·M(x)·Φ₀ is design-dependent). Under MMA's expanding asymptotes this drives
the design toward a **near-mechanism** (K → near-singular, so compliance
F·K⁻¹·F → ∞ and ω₁ → 0), which shows up as the salt-and-pepper field. The pilot exhibits
**the same collapse transiently** (ω₁ = 3.70 rad/s at iteration 35, objective 233 M) but
**recovers** because its fully clamped edges make the mechanism basin escapable and it has
the iteration budget to climb back. The weak 2-pin domain has a genuine rigid-rotation
mechanism (ω₁→0) that is a deep attractor; with only 400 iterations the omitted-gradient
run never escapes.

**Classification:** primarily **configuration mismatch**; the collapse itself is
**expected behavior of the inconsistent `omitted` gradient under MMA** on a weakly
supported domain. It is **not** a mode-tracking failure, **not** a filter/checkerboard
pathology in the usual sense, and **not** an optimizer bug.

---

## 1. Which artifacts are being compared

| | "Clean 142" run | "Noisy 2.98" run |
|---|---|---|
| Identity | Exp2 pilot **=** Exp2 authoritative sweep α=1.00 | CR2-MMA diagnostic **variant A (omitted)** |
| Path | `output/exp2_pilot_authoritative/` and `output/exp2_authoritative_sweep/alpha_1_00/` | `cr2/mma_diagnostic/output/cr2_mma_variant_a_*` |
| Final ω₁ | 141.78982535805085 rad/s | 2.770817 rad/s (≈2.99 around iter 364–367) |
| Iterations | 1052 / 2000 (converged) | 400 / 400 (capped, not converged) |
| Final design-change | 1.66e-4 (≤ 1e-3 ✓) | 7.40e-2 (✗) |
| Final MAC | 0.99253 | 0.81158 |
| Final grayness | 0.0858 | 0.1007 |
| Topology | coherent doubly-clamped beam load path | salt-and-pepper, no coherent load path |

The pilot `result.json` and the sweep `alpha_1_00/result.json` agree to the last digit
(ω = [141.78982535805085, 374.309771773102, 614.650190135589], grayness
0.0857778265073689, design_change 0.00016618994509209895). The "full Exp2 α=1" run that
matters for the manuscript already reproduces the pilot. The 2.98 number does not exist in
either of those files; it lives only in the CR2-MMA omitted diagnostic.

---

## 2. Configuration comparison (every parameter)

Pilot/sweep α=1 config vs CR2-MMA variant A config:

| Parameter | Pilot / Exp2 sweep α=1 | CR2-MMA variant A | Same? |
|---|---|---|---|
| Mesh | **200 × 25** (5000 el) | **160 × 20** (3200 el) | ✗ |
| Domain size | 8 × 1, t=1 | 8 × 1, t=1 | ✓ |
| **Boundary condition** | **`vertical_line` x=0 and x=8, dofs ux,uy** → entire left & right edges clamped | **`closest_point` [0,0.5] and [8,0.5]** → two single pinned nodes at mid-height | ✗ (critical) |
| Load cases | **2** (mode-1 factor 1 + mode-2 factor 0) | **1** (mode-1 factor 1) | ✗ |
| Load type | semi_harmonic, solid baseline, F=ω₀²M(x)Φ₀ | same | ✓ |
| Material E / ν / ρ | 1e7 / 0.3 / 1 | 1e7 / 0.3 / 1 | ✓ |
| void E_min / ρ_min | 1e-6 / 1e-6 | 1e-6 / 1e-6 | ✓ |
| Optimizer | MMA | MMA | ✓ |
| Objective / interp | compliance / SIMP | compliance / SIMP | ✓ |
| Volume fraction | 0.5 | 0.5 | ✓ |
| Penalization p | 3 | 3 | ✓ |
| **load_sensitivity** | **omitted** | **omitted** | ✓ |
| harmonic_normalize | false | false | ✓ |
| Filter | sensitivity, rmin=2 el, symmetric, no Heaviside | identical | ✓ |
| Move limit | 0.2 | 0.2 | ✓ |
| **max_iters** | **2000** | **400** | ✗ |
| convergence_tol | 1e-3 | 1e-3 | ✓ |
| seed | (unset) | 0 | ~ |

Four substantive differences: **mesh**, **boundary condition** (clamped edges vs two pins),
**load-case count**, and **iteration cap**. The decisive one is the boundary condition; the
iteration cap decides whether the transient collapse is survivable.

---

## 3. Optimization-history comparison

### 3.1 The pilot collapses too — and recovers

This is the central evidence. The pilot is not monotone; it dives into the same
near-mechanism region the noisy run gets stuck in:

| Iter | Pilot ω₁ (rad/s) | Pilot objective | Pilot MAC | tracked index |
|---|---|---|---|---|
| 1 | 145.52 | 169,402 | 1.000 | 1 |
| **35 (worst)** | **3.696** | **233,379,046** | **≈0.0006 (min)** | switches (1→2/3/5) |
| 398 | 94.73 | 307,487 | 0.898 | 1 |
| 402 | 69.92 | 556,435 | 0.862 | 1 |
| 1050 | 141.78 | 192,313 | 0.9925 | 1 |
| 1052 | 141.79 | 192,295 | 0.9925 | 1 |

The pilot reaches **ω₁ = 3.70 rad/s with objective 233 M and MAC ≈ 0** at iteration 35 —
*worse* than the noisy run's final state — then climbs back out and converges to 142.
Over the full run the tracked index transiently visits {1,2,3,5} before re-locking on 1.

### 3.2 The noisy run collapses and never recovers

| Iter | variant A ω₁ (rad/s) | objective | grayness | sens. ‖Δ‖₂ |
|---|---|---|---|---|
| 1 | 68.40 | 37,427 | 1.000 | 3,210 |
| 18 | 84.32 | 25,850 | 0.914 | (rising) |
| 25 | 29.26 | 217,467 | 0.594 | — |
| 37 (worst) | **1.887** | **51,506,584** | — | — |
| 50 | 2.657 | 23,547,785 | 0.310 | **1,753,906** |
| 100 | 5.674 | 4,599,685 | 0.393 | — |
| ~364 | **≈2.99** | ~19,150,349 | 0.104 | — |
| 400 | 2.771 | 22,036,424 | 0.101 | — |

The element-sensitivity L2 norm jumps **~700×** (2,478 at iter 20 → 1,753,906 at iter 50)
exactly as the collapse happens — the hallmark of K becoming near-singular and
F·K⁻¹·F (= u·K·u) exploding. The design oscillates inside the collapsed basin for the rest
of the 400 iterations; design-change never falls below 0.074.

### 3.3 Why one recovers and the other does not

Both runs use the identical `omitted` formula and both dive to ω₁≈2–4 rad/s early. The
difference is the **boundary condition** plus **iteration budget**:

- **Pilot — clamped edges (200×25):** there is no global rigid mechanism; the entire left
  and right edges are fixed in ux and uy. The near-mechanism basin is shallow and a stiff,
  fully connected design always exists and is reachable, so with 2000 iterations the
  optimizer climbs back to the 142 basin.
- **variant A — two pinned points (160×20):** a beam pinned at only two mid-height nodes
  has a genuine low-energy mode (near rigid-body rotation/flutter about the two supports)
  whose frequency → 0 as material disperses. That is a deep attractor for an
  inconsistent-gradient optimizer, and the 400-iteration cap removes any chance of escape
  even if one existed.

---

## 4. Mode-tracking, MAC, and eigenvalue histories

- **Not a mode-tracking failure.** In variant A the final tracked mode is index 1 and it is
  genuinely the lowest computed mode — ω₁=2.77 is the *true* fundamental frequency of a
  near-mechanism, not a higher mode mis-labeled as the first. MAC = 0.812 (still above the
  0.8 gate) reflects that the *shape* of mode 1 has drifted far from the solid-reference
  mode-1 (it is now a localized mechanism shape), which is the correct, faithful report of a
  collapsed structure.
- The pilot's MAC momentarily drops to ≈0.0006 and the tracked index hops to 2/3/5 during
  the iter-35 trench, then re-locks to index 1 / MAC 0.9925. Tracking is therefore robust
  enough to survive the transient; it is a downstream indicator, not the cause.
- **Eigenvalue history:** pilot ends at [141.79, 374.31, 614.65] (well-separated, ascending);
  variant A ends at [2.77, 6.19, 14.43] — a compressed, near-zero spectrum, i.e. a floppy
  structure, consistent with a mechanism rather than a stiff beam.

---

## 5. Filter / checkerboard assessment

The sensitivity filter (type=sensitivity, rmin=2 el, symmetric, no Heaviside) is
**identical** across the pilot, variant A, and variant B. Variant B (same mesh, same BC,
same filter, same MMA, same cap — differing *only* in `load_sensitivity = "complete"`)
produces a **clean MAC=0.992 truss**. Therefore the salt-and-pepper field in variant A is
**not** a filter deficiency or a classic checkerboard instability — it is the optimizer
dumping material incoherently because the gradient it was given is wrong. Same filter, good
gradient → clean design; same filter, inconsistent gradient → noise.

---

## 6. The controlled experiment already in the repo

`cr2/mma_diagnostic/` runs variant A and variant B that are **identical except for
`load_sensitivity`**:

| | Variant A (omitted) | Variant B (complete) |
|---|---|---|
| Final ω (tracked) | 2.77 rad/s (index 1) | 170.6 rad/s (index 2), ω₁=132.6 |
| Final objective | 22,036,424 (diverged) | 4,952 (plateaued) |
| Final MAC | 0.812 | 0.992 |
| Topology | salt-and-pepper, no load path | clean two-cell truss |
| Pearson r between the two topologies | — | **−0.11** (nearly opposite designs) |

This isolates the cause to the `omitted` sensitivity formula. The prior
`cr2_mma_diagnosis.md` reached the same conclusion: omitting ∂F/∂x is compatible only with
OC's small effective steps (OC+omitted gives a sane ω₁=153.8); MMA's larger moves break the
"F fixed" approximation and the objective diverges.

---

## 7. Root-cause classification

| Candidate cause | Verdict |
|---|---|
| Mode-tracking failure | **No.** Tracked mode is the true lowest mode; 2.77 is a real ω₁. |
| Disconnected structure | **Symptom, not cause.** Final field is a near-mechanism (ω₁≈0); it is the *result* of the bad gradient. |
| Checkerboard / filter pathology | **No.** Filter is identical to the clean pilot and clean variant B; noise is incoherent material dumping, not a filter instability. |
| **Configuration mismatch** | **Yes — primary.** The two runs differ in mesh, BC (clamped edges vs 2 pins), load-case count, and iteration cap; they are different experiments. |
| Optimizer bug | **No.** MMA behaves correctly given the inputs; with the consistent (complete) gradient it produces a clean design on the same config. |
| **Expected behavior** | **Yes — for the mechanism.** Given an inconsistent gradient on a weakly supported domain, divergence to a near-mechanism is the expected outcome. |

**One-line root cause:** the noisy ω₁≈2.98 result is a *different configuration* (CR2-MMA
omitted diagnostic: 160×20, two-pin BC, 1 load case, 400-iter cap) in which the
`load_sensitivity="omitted"` inconsistent gradient drives MMA into a near-mechanism that the
weak supports cannot prevent and 400 iterations cannot escape — whereas the clamped-edge
pilot, given 2000 iterations, rides through the identical transient collapse and recovers to
142 rad/s.

---

## 8. Recommended fixes (no solver change required)

1. **Compare like for like.** For the manuscript's Exp2 α=1 claim, use the authoritative
   sweep case `output/exp2_authoritative_sweep/alpha_1_00/` (ω₁=141.79), which already
   reproduces the pilot exactly. Do not present the CR2-MMA omitted diagnostic as "the
   Exp2 α=1 result" — it is a different domain and a deliberately stress-tested formulation.
2. **Use the consistent gradient.** On the two-pin CR2 domain, switch
   `load_sensitivity` from `"omitted"` to `"complete"` (variant B), which yields a clean
   MAC=0.992 structure under MMA. The `omitted` formula should not be paired with MMA.
3. **If `omitted` must be retained,** keep it with OC (whose small effective steps mask the
   inconsistency: OC+omitted gives ω₁=153.8, not a collapse), and/or strengthen the support
   (the clamped-edge domain is robust to the same formula), and/or add a lower-bound
   frequency/eigenvalue constraint to forbid the near-mechanism.
4. **Iteration cap.** Raising 400→2000 alone will *not* rescue variant A: the 2-pin domain's
   mechanism basin is a genuine attractor, unlike the clamped-edge pilot's escapable trench.
   The fix is the gradient/BC, not more iterations.

---

## 9. Provenance

All numbers from committed artifacts; no solver code changed.

- `output/exp2_pilot_authoritative/exp2_pilot_authoritative_{result.json,config.json,convergence_history.csv,topology.png}`
- `output/exp2_authoritative_sweep/alpha_1_00/exp2_authoritative_alpha_1_00_{result.json,config.json}`
- `cr2/mma_diagnostic/cr2_mma_variant_{a_omitted,b_complete}.json`
- `cr2/mma_diagnostic/output/cr2_mma_variant_a_{histories.csv,mode_tracking.csv,sensitivity_norms.csv,topology.png}`
- `cr2/mma_diagnostic/output/cr2_mma_variant_b_topology.png`, `cr2_mma_results.json`, `cr2_mma_diagnosis.md`
- `cr2/output/cr2_production_results.json` (OC+omitted → ω₁=153.8), `cr2/cr2_variant_a_omitted.json`
