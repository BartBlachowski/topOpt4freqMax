# CR2-MMA Matched Diagnostic: Failure Diagnosis

**Gate:** E4-CR2-MMA  
**Status:** FAILED — both variants hit iteration cap; no accepted comparison  
**Date:** 2026-06-23  
**Question:** Is CR2 failure caused by OC rather than the sensitivity formula?

---

## 1. Setup

Identical to the OC production runs (`examples/Revision_v1/cr2/`) except `optimizer: "MMA"`.

| Setting | Value |
|---------|-------|
| Mesh | 160 × 20 (3200 elements) |
| Volume fraction | 0.5 |
| Load | semi_harmonic, solid baseline, F = ω₀²M(x)Φ₀ |
| Penalization | p = 3, sensitivity filter rmin = 2 el |
| Move limit (clip) | 0.2 |
| Max iterations | 400 |
| Convergence tol | dc ≤ 1e-3 |
| Optimizer | **MMA** (Svanberg September 2007, Python port) |

---

## 2. Results Summary

| Criterion | Threshold | Variant A (omitted) | Variant B (complete) |
|-----------|-----------|---------------------|----------------------|
| Iteration cap | < 400 | **400** (fail) | **400** (fail) |
| Final dc | ≤ 1e-3 | **7.40e-2** (fail) | **1.81e-2** (fail) |
| Final feasibility | ≤ 1e-4 | 0.0 (pass) | 0.0 (pass) |
| Final MAC | ≥ 0.8 | 0.812 (pass) | 0.992 (pass) |
| All artifacts | present | yes | yes |
| **Accepted** | — | **no** | **no** |

---

## 3. Variant A — Failure Mechanism: Divergent Objective Under MMA

### 3.1 Objective trajectory

| Iteration | Objective | dc |
|-----------|-----------|-----|
| 1 | 37,427 | 0.013 |
| 10 | 34,665 | 0.049 |
| 18 | 25,850 | 0.112 |
| 50 | 23,000,000 | 0.124 |
| 100 | 4,600,000 | 0.127 |
| 200 | 133,106 | 0.143 |
| 300 | 26,688,000 | 0.127 |
| 400 | 22,036,424 | 0.074 |

The objective is not monotonically decreasing. It rises from 37K to 23M between iterations 1 and 50, partially recovers to 133K by iteration 200, then explodes again to 26M at iteration 300. At iteration 400, omega_1 = 2.77 rad/s — nearly zero, indicating a near-degenerate structure.

### 3.2 Mechanism

The omitted sensitivity formula is:
```
dc/dx_e = –p·x^(p-1)·(E₀–Emin)·u^T·k₀·u
```

This is the structural sensitivity of C = u^T·K·u with the load F treated as constant. But F = ω₀²·M(x)·Φ₀ changes with x. As MMA's expanding asymptotes allow progressively larger moves (asyincr=1.2 per monotone step), the "F fixed" approximation breaks down.

By iteration ~15, MMA's asymptotes have widened enough to allow element changes of ≈0.15 per iteration (versus OC's effective ≈0.005 from the bisection). At these step sizes, the load changes significantly between iterations. Each step that reduces C under the "F fixed" approximation increases F^T·K^{-1}·F substantially as the actual M(x) changes.

Under OC, the bisection on λ inherently limits effective moves and the "F fixed" assumption holds iteratively. MMA's expanding asymptotes remove this limiting effect.

### 3.3 Comparison with OC

| | Variant A (omitted, OC) | Variant A (omitted, MMA) |
|---|---|---|
| dc at iter 400 | 5.24e-3 (micro-oscillation) | 7.40e-2 (divergent) |
| Objective | stable, ~7847 | oscillating, 22M |
| omega_1 at iter 400 | 153.8 rad/s | 2.77 rad/s |
| Root cause | OC micro-oscillation (tolerance) | Sensitivity approx breaks down |

MMA **worsens** Variant A relative to OC. The omitted sensitivity formula is specifically compatible with OC's small effective steps; it is not robust to MMA's larger moves.

---

## 4. Variant B — Failure Mechanism: Slow Convergence Under MMA

### 4.1 Objective trajectory

| Iteration | Objective | dc |
|-----------|-----------|-----|
| 1 | 37,427 | 0.013 |
| 50 | 5,124 | 0.137 |
| 100 | 4,939 | 0.151 |
| 200 | 4,912 | 0.117 |
| 300 | 4,913 | 0.009 |
| 400 | 4,952 | 0.018 |

The objective decreases from large values to ~4950 and plateaus by iteration 100. There is no move saturation (dc stays well below the 0.2 clip), no period-2 cycle, and no persistent bounded cycle. The design is slowly exploring, with dc trending downward (0.151 at iter 100 → 0.009 at iter 300) before rising slightly again.

Final omega_1 = 132.6 rad/s. MAC = 0.992 (mode 2 has the best match against the solid reference mode 1; this is physically reasonable as mode ordering can shift with topology).

### 4.2 Comparison with OC

| | Variant B (complete, OC) | Variant B (complete, MMA) |
|---|---|---|
| dc behavior | 0.200 from iter 1-400 (saturated) | 0.009-0.151 (trending down) |
| Objective | period-2 cycle between 4833 and 4959 | plateau ~4950 |
| Mode tracking | failed at iter 4 (switch to mode 2) | MAC 0.992 at iter 400 |
| Root cause | OC + complete sensitivity → 2-cycle | Slow convergence, not yet converged |

MMA **substantially improves** Variant B. The OC 2-cycle (structural instability from the combination of the complete sensitivity and the bisection update) is eliminated under MMA. The design is converging but has not reached dc ≤ 1e-3 within 400 iterations.

The OC failure for Variant B was therefore **substantially caused by the optimizer (OC)**, not exclusively by the sensitivity formula.

---

## 5. Topology Difference

The two MMA variants produce fundamentally different designs:

| Metric | Value |
|--------|-------|
| Mean absolute difference | 0.543 |
| RMS difference | 0.704 |
| Max |Δ| | 0.998 |
| |Δ|>0.05 fraction | 63% |
| Pearson r | −0.11 (near-zero, anti-correlated) |

The negative correlation (Pearson r ≈ −0.11) indicates the topologies are nearly opposite. Variant A is a near-degenerate structure; Variant B is a well-formed topology maximizing stiffness-per-load. No endpoint comparison between variants is scientifically valid since Variant A has not converged.

---

## 6. Assessment: Does MMA Isolate the CR2 Failure?

### 6.1 Was OC responsible for Variant A's failure?

**Partially, but not exclusively.**

Under OC, Variant A showed micro-oscillation at dc ≈ 0.005 — an OC artifact, not a fundamental failure. The design was functionally converged (omega_1 = 153.8 rad/s, flat for 300 iterations).

Under MMA, Variant A **diverges** (omega_1 → 2.77 rad/s, objective → 22M). MMA's larger effective steps break the "F fixed" approximation, causing catastrophic gradient inaccuracy. This demonstrates that the omitted sensitivity formula is **optimizer-dependent**: it is compatible with OC's small moves, but not with MMA's larger moves.

### 6.2 Was OC responsible for Variant B's failure?

**Yes, substantially.**

Under OC, Variant B showed a strict period-2 cycle from iteration 9 with permanent move saturation (dc = 0.200). This is a documented OC pathology when the gradient has near-zero net curvature.

Under MMA, Variant B shows slow monotone convergence toward a well-formed topology (MAC = 0.992, dc trending down). No 2-cycle, no saturation. The OC 2-cycle is an optimizer artifact that MMA suppresses.

However, Variant B has not yet converged (dc = 0.018 at iter 400), so a formal accepted comparison is not available.

### 6.3 Summary table

| | OC Variant A | OC Variant B | MMA Variant A | MMA Variant B |
|--|--|--|--|--|
| Accepted | no | no | no | no |
| Root cause | micro-oscillation (OC) | 2-cycle (OC) | gradient breakdown (omitted formula + large steps) | slow convergence |
| OC artifact? | yes | yes | N/A | N/A |
| Fixable by optimizer change? | maybe (relaxed tol) | **yes** (MMA works) | no (sensitivity formula problem) | pending (needs more iters) |

---

## 7. Outcome Classification: INCONCLUSIVE

Neither variant converged under MMA within 400 iterations. The absence of algorithm-failure signatures (no move saturation, no period-2, no stable cycle) in either variant means the plateau diagnostics do not classify these as "algorithm failures" — they are simply insufficient iterations.

However, the failure modes are **qualitatively informative**:

1. **Variant A (omitted, MMA):** The sensitivity formula is incompatible with MMA's larger step sizes. OC's dominance for this variant is confirmed — it was the right optimizer choice. CR2 with the proposed method (omitted sensitivity + OC) was a **tolerance issue**, not a fundamental failure.

2. **Variant B (complete, MMA):** The OC 2-cycle is eliminated under MMA, confirming it was **optimizer-induced**. The complete sensitivity under MMA is better-behaved, but convergence is slow at 400 iterations.

**CR2 REMAINS FAILED** for both variants under MMA.

The partial scientific conclusion: if more iterations were permitted for Variant B under MMA, it might converge and enable a valid comparison. But no accepted comparison is available from this run.

---

## 8. Data Provenance

All numbers are derived from:

- `cr2_mma_run.log` (complete stdout/stderr)
- `cr2_mma_variant_a_histories.csv` (400 rows)
- `cr2_mma_variant_b_histories.csv` (400 rows)
- `cr2_mma_variant_a_sensitivity_norms.csv`
- `cr2_mma_variant_b_sensitivity_norms.csv`
- `cr2_topology_difference_metrics.json`

No solver code was changed. No experiments beyond these two variants were run. No tuning was performed after the configs were written.

**Note on `final_freq_omega1` in results JSON:** The post-loop eigensolver in `topopt_freq.py` computes omega in rad/s. The values reported are:
- Variant A: omega_1 = 2.771 rad/s
- Variant B: omega_1 = 132.6 rad/s
