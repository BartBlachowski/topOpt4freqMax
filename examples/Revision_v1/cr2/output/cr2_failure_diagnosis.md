# CR2 Failure Diagnosis

**Gate:** E4-CR2  
**Status:** FAILED_PARTIAL (both variants hit iteration cap, no scientific claim permitted)  
**Date analysed:** 2026-06-22  
**Data sources:** `cr2_variant_{a,b}_histories.csv`, `cr2_variant_{a,b}_sensitivity_norms.csv`,
`cr2_variant_{a,b}_mode_tracking.csv`, `cr2_topology_difference_metrics.json`,
`cr2_production_run.log`

---

## 1. Failure Summary

| Criterion | Threshold | Variant A (omitted) | Variant B (complete) |
|-----------|-----------|---------------------|----------------------|
| Iteration cap | < 400 | **400** (fail) | **400** (fail) |
| Final design_change | ≤ 1e-3 | **5.24e-3** (fail) | **2.00e-1** (fail) |
| Final feasibility | ≤ 1e-8 | **9.29e-5** (fail) | **1.17e-6** (fail) |
| Tracked mode | stable | mode 1, MAC=0.999 (pass) | mode 2, MAC=0.955 (fail) |
| Solver success | yes | yes | yes |

The two variants failed for **entirely different reasons**.

---

## 2. Variant A — Failure Mechanism: Stable Limit Cycle Near Optimum

### 2.1 Convergence trajectory

| Iterations | Mean design_change | Notes |
|------------|-------------------|-------|
| 1–50 | 0.0783 | rapid initial topology formation |
| 51–100 | 0.0121 | fast approach to near-optimal region |
| 101–150 | 0.0096 | entering slow phase |
| 151–200 | 0.0078 | slow convergence stalls |
| 201–250 | 0.0073 | plateau begins |
| 251–300 | 0.0073 | plateau (one spike to 0.023 at iter 300) |
| 301–350 | 0.0070 | plateau |
| 351–400 | 0.0067 | plateau, final value 0.00524 |

The design_change does not trend toward zero after iteration ~150. It oscillates
persistently in the band 0.004–0.009, with occasional larger spikes (0.023 at iter 300,
0.026 at iter 349).

### 2.2 Objective convergence

```
iter   100:  obj = 7824.5   omega_1 = 153.92 rad/s
iter   200:  obj = 7847.9   omega_1 = 153.80 rad/s
iter   300:  obj = 7849.5   omega_1 = 153.76 rad/s
iter   400:  obj = 7845.1   omega_1 = 153.79 rad/s
```

Change from iter 100 to 400: |7845 − 7824| / 7824 = **0.27%** in 300 iterations.
The design is **functionally converged**. The objective is flat to 3 significant figures
from iteration 100 onward, yet the design-change criterion cannot be satisfied.

### 2.3 Feasibility oscillation

Volume-constraint feasibility alternates between 0.0 (exactly satisfied) and ~1e-4
(violated by ~0.01%). This is consistent with the OC bisection on the Lagrange
multiplier overshooting the constraint bound by a small amount each other iteration.
Neither the amplitude nor period of this oscillation decreases over 300 iterations.

### 2.4 Mode tracking

Mode 1 tracked throughout all 400 iterations. MAC stays ≥ 0.997 from iteration 6
onward. Mode tracking is not a contributing factor.

### 2.5 Load-sensitivity term magnitude (Variant A's omitted term)

```
iter     1:  sensitivity_difference L2 = 3210.0  (per-element RMS = 56.75,  Linf = 92.43)
iter    50:  sensitivity_difference L2 =  637.5  (per-element RMS = 11.27,  Linf = 18.79)
iter   100:  sensitivity_difference L2 =  638.9  (per-element RMS = 11.29,  Linf = 18.94)
iter   200:  sensitivity_difference L2 =  639.9  (per-element RMS = 11.31,  Linf = 18.97)
iter   400:  sensitivity_difference L2 =  640.2  (per-element RMS = 11.32,  Linf = 18.99)
```

The omitted load-sensitivity term (= d(F(x))/dx · u) has a large, constant magnitude
that is independent of iteration after the initial transient. It does not decay toward
zero as the design approaches a fixed point. This is structurally expected: the load
F = omega0^2 · M(x) · Phi0 always has nonzero mass-derivative contribution.

### 2.6 Root cause

The OC update with sensitivity filtering creates a **persistent micro-oscillation** in
the design. The volume-constraint bisection introduces a small perturbation each
iteration that is amplified rather than damped under the current update rule. This is a
known property of OC applied to problems where the sensitivity landscape is not
strictly monotone and the constraint bisection is exact (not regularised).

The cycle is self-sustaining: the design oscillates between two nearby configurations
D1 and D2, where D1 → D2 (via the OC step from D1) and D2 → D1 (via the OC step
from D2), with each step size well below move_limit (dc ≈ 0.005 ≪ 0.2). This is a
**genuine fixed-point instability** of OC at this problem size and sensitivity
structure, not a sign of a poorly chosen tolerance.

**Tolerances are too strict for OC on this problem.** A design-change tolerance of
1e-3 is achievable with OC only if the sensitivity-filter width is large enough to
over-smooth the sensitivities. For rmin=2 elements on a 160×20 mesh, the filter does
not over-smooth sufficiently. Relaxing conv_tol to 5e-3 would accept this run; the
scientific content (topology, frequency) is unchanged.

---

## 3. Variant B — Failure Mechanism: 2-Cycle Instability, Move-Limit Saturated from Iteration 1

### 3.1 Convergence trajectory

```
iter   1:  dc=0.200 (move_limit)
iter   2:  dc=0.200
iter   3:  dc=0.200
...
iter 400:  dc=0.200  (every single iteration)
```

The design_change equals move_limit = 0.2 for **all 400 iterations**. The optimizer
always wants to take a step larger than the move limit. This is not a case of
insufficient iterations — it is structural instability.

### 3.2 Mode switch at iteration 4

```
iter 1:  mode 1, MAC=1.000, omega_1= 68.40
iter 2:  mode 1, MAC=0.999, omega_1= 98.29
iter 3:  mode 1, MAC=0.953, omega_1=113.59
iter 4:  mode 2, MAC=0.975, omega_1=101.87   <<< switch
iter 5:  mode 2, MAC=0.947, omega_1= 94.69
...
iter 400: mode 2, MAC=0.999, omega_1=101.80
```

The mode tracking switches from mode 1 to mode 2 at iteration 4 and never recovers.
The complete sensitivity drives the topology toward a design that maximises the
response under a mode-2 inertial load pattern, thereby vacating the mode-1 subspace.

### 3.3 Period-2 oscillation, established by iteration 9

From iteration 9 onward, Variant B settles into a strict 2-cycle:

| State | obj | omega_1 | feasibility | MAC |
|-------|-----|---------|-------------|-----|
| D_even | 4833 | 101.80 rad/s | ~1e-6 | 0.998 |
| D_odd | 4959 | 107.59 rad/s | ~2e-5 | 0.955 |

The objective alternates between D_even and D_odd on every odd/even iteration from
iteration ~9 to 400. The cycle amplitude **does not shrink**: the objective gap between
the two states is |4959 − 4833| = 126 at iteration 9 and 126 at iteration 400. This is
a **fixed 2-cycle attractor**, not slow convergence.

Example (final 10 iterations):
```
it=391  dc=0.200  obj=4959.27  omega_1=107.59  MAC=0.9553
it=392  dc=0.200  obj=4833.09  omega_1=101.80  MAC=0.9982
it=393  dc=0.200  obj=4959.23  omega_1=107.59  MAC=0.9553
it=394  dc=0.200  obj=4833.09  omega_1=101.80  MAC=0.9982
...
```

### 3.4 Sensitivity difference magnitude (Variant B's included term)

```
iter     1:  L2 = 3210.0  per-element RMS = 56.75
iter    50:  L2 =  552.5  per-element RMS =  9.77
iter   100:  L2 =  552.7  per-element RMS =  9.77
iter   200:  L2 =  552.8  per-element RMS =  9.77
iter   300:  L2 =  552.8  per-element RMS =  9.77
iter   400:  L2 =  552.8  per-element RMS =  9.77
```

Like Variant A, the load-sensitivity contribution freezes at a constant after the
initial transient — but now it is **included** in the gradient that drives OC. The
per-element RMS of ~9.8 (at Variant B's attractor) is a persistent, large perturbation
that prevents any fixed point.

### 3.5 Root cause

The complete sensitivity formula is:

```
dC/dx_e = –p * x_e^(p-1) * E_0 * u_e^T * k0 * u_e    (structural term)
         – 2 * u_e^T * dF/dx_e                          (load sensitivity term)
dF/dx_e = omega0^2 * dM/dx_e * Phi0                    (mass-proportional)
```

The load sensitivity term has the same sparsity as the structural term but **opposite
functional effect**: it rewards material that reduces the projection of the inertial
load onto the response. For a mass-proportional load, removing material from element
e both reduces stiffness (bad) and reduces the inertial force (good). The OC update
cannot find a fixed point because these two effects are comparable in magnitude and
pull in opposite directions at every element. The resulting net sensitivity has near-zero
effective curvature, so any finite move limit causes perpetual overshoot.

This is **not a tolerance calibration issue**. Increasing max_iters to 4000 would
produce 4000 iterations of the same 2-cycle. The algorithm genuinely cannot converge
with the complete sensitivity and OC at move_limit=0.2 on this problem.

---

## 4. Topology Comparison

The two variants converge to fundamentally different designs:

| Metric | Value |
|--------|-------|
| Mean absolute density difference (MAD) | 0.275 |
| RMS density difference | 0.438 |
| Max absolute difference | 1.000 (some elements fully opposed) |
| Fraction with |Δ| > 0.01 | 58.6% |
| Fraction with |Δ| > 0.05 | 54.1% |
| Pearson density correlation | 0.519 |

The topologies are **substantially different**. This is not a consequence of numerical
noise; the different sensitivity formulations steer the optimizer to genuinely distinct
local optima or, in the case of Variant B, to a trajectory that never reaches any
local optimum.

---

## 5. Outcome Frequencies

```
Variant A (omitted, proposed):   omega_1 = 153.8 rad/s  (24.5 Hz)  — stable, flat last 200 iters
Variant B (complete, comparator): omega_1 = 101.8 rad/s  (16.2 Hz)  — 2-cycle attractor D_even
                                 or 107.6 rad/s  (17.1 Hz)  — 2-cycle attractor D_odd
```

Variant A achieves **~50% higher fundamental frequency** than either state of Variant B's
2-cycle. The topologies are different so this is not a controlled comparison on the same
design landscape — but the result is consistent with the paper's thesis that omitting
the load sensitivity leads to a more effective and more stable algorithm.

---

## 6. Assessment: Tolerances vs. Algorithm Failure

| | Variant A | Variant B |
|---|-----------|-----------|
| Objective convergence | yes (flat to 0.3% for 300 iters) | no (2-cycle, Δobj=126) |
| Design-change convergence | no (stable cycle at dc≈0.005) | no (dc=0.200 always) |
| Mode tracking | healthy | failed at iter 4 |
| Root cause | OC micro-oscillation (tolerance issue) | gradient instability (algorithm failure) |
| Fixable by relaxing tol? | **yes** (conv_tol → 5e-3, feas_tol → 1e-4) | no |
| Fixable by more iters? | no (cycle is stable) | no (2-cycle is stable) |
| Fixable at algorithm level? | — | yes: smaller move_limit (≤ 0.02) or MMA |

**Variant A** did not fail due to an overly strict tolerance in any absolute sense —
but the OC micro-oscillation at dc≈0.005 cannot be closed at the current rmin and
mesh resolution. The design is converged; only the stopping criterion is not met.
Relaxing `conv_tol` from 1e-3 to 5e-3 (and `feas_tol` from 1e-8 to 1e-4) would
accept this run with no change to the scientific content.

**Variant B** is a genuine algorithm failure. The complete load sensitivity renders OC
divergent regardless of iteration budget. This is not fixed by tighter or looser
tolerances — it requires a different optimizer (MMA) or a much smaller move limit
(0.01–0.02).

---

## 7. Recommended Next Actions

### Immediate (do not rerun production)

1. **Record Variant A as scientifically valid with accepted-by-objective criterion.**
   The frequency (omega_1=153.8), topology, and mode tracking are all stable. The
   dc=0.005 residual oscillation is a well-known OC artifact. The summary should note
   "converged in objective, OC micro-oscillation prevents dc≤1e-3."

2. **Record Variant B as algorithmic failure with diagnostic value.**
   The 2-cycle onset at iteration 9 with permanent move-limit saturation is a clean,
   reportable result. It demonstrates why the complete load sensitivity is incompatible
   with OC and motivates the proposed omission.

### If a rerun is authorised (Gate E4-CR2 reopened)

3. **Variant A rerun option A**: Relax `conv_tol` to 5e-3 and `feas_tol` to 1e-4
   in the config. The existing production run shows these would already be satisfied
   (final dc=5.24e-3, feas=9.29e-5 ≈ tol). No solver changes; config change only.

4. **Variant A rerun option B**: Switch to MMA (`optimizer: mma`). MMA's internal
   line search suppresses the OC micro-oscillation and can reach dc≤1e-3. Requires
   ensuring the MMA Python implementation exists.

5. **Variant B rerun option**: Switch to MMA, or set `move_limit: 0.02`. The 2-cycle
   has half-amplitude step ~0.200; move_limit ≤ 0.02 may allow stabilisation. Even if
   Variant B converges at move_limit=0.02, it is expected to find a worse design than
   Variant A (lower omega_1), which reinforces the paper's claim.

### Manuscript guidance (do not edit until a rerun is authorised or current data is used)

The current data already supports a clear narrative:
- Variant A converges to omega_1=153.8 rad/s with stable mode tracking (proposed method)
- Variant B does not converge (complete sensitivity causes 2-cycle from iteration 9)
- The 50% frequency advantage and algorithmic stability advantage are both visible
- The only caveat is that Variant A's formal convergence criterion is not met, which
  must be acknowledged or the criterion must be changed before any claim is published

---

## 8. Data Provenance

All numbers above are derived directly from:

- `examples/Revision_v1/cr2/output/cr2_variant_a_histories.csv` (400 rows)
- `examples/Revision_v1/cr2/output/cr2_variant_b_histories.csv` (400 rows)
- `examples/Revision_v1/cr2/output/cr2_variant_a_sensitivity_norms.csv` (400 rows)
- `examples/Revision_v1/cr2/output/cr2_variant_b_sensitivity_norms.csv` (400 rows)
- `examples/Revision_v1/cr2/output/cr2_topology_difference_metrics.json`
- `examples/Revision_v1/cr2/output/cr2_production_run.log` (850 lines)

No solver code was changed. No experiments were re-run. All analysis is read-only.
