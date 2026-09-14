# SOCP_CERTIFICATION — Parts 4 and 14, and the termination

```
C480_FULL_RUN_SOCP_COVERAGE_FAIL
```

## Outcome

| | |
|---|---|
| outer iterations executed | 15 |
| accepted (certified) SOCP updates | **14** (outer 1–14) |
| rejected | **1** (outer 15), no update applied |
| termination | **`SOCP_CERTIFICATE_FAILURE`** at outer 15, fail-closed as preregistered |
| N = 1 / N = 2 / N > 2 | 0 / **15** / 0 |
| eligible for LP / SOCP / unsupported | 0 / **15** / 0 (E1–E8 passed at all 15) |
| apex (predicted e₁ = e₂) iterations | 0 of the 14 accepted; **both attempts at outer 15** |

Every accepted update came from a certified solve. The run did not reach the
controller's first exhaustion declaration, so coverage FAILS.

## Certificate quality of the 14 accepted updates

Source: `run/C480x60_socp_socp_iterations.csv`, `evaluations/analysis.json → treatment.socp`.

| bar | preregistered | worst accepted | median |
|---|---|---|---|
| C1 max production row | ≤ 1e-8 | 2.5e-11 | 5.1e-13 |
| C2 raw box violation | ≤ 1e-9 | 2.5e-12 | 0 |
| C3 weak-duality gap | \|gap\| ≤ 1e-8 | **6.2e-9** (outer 3) | 1.9e-11 |
| C5 row complementarity | ≤ 1e-6 | 3.0e-11 | 5.4e-13 |
| C6 box complementarity / (sRow0·move) | ≤ 1e-4 | 4.8e-8 | 1.7e-10 |
| C7 \|bs stationarity\| | ≤ 1e-6 | 9.3e-15 | 2.2e-16 |
| C8 stationarity RMS / max | ≤ 1e-6 / 1e-5 | 0 / 0 | 0 |
| E9 affine rows vs production | ≤ 1e-12 | 1.7e-14 | 2.1e-15 |
| E10 cone vs production row 1 | ≤ 1e-10 | 1.5e-14 | 2.1e-15 |
| E11 row 2 − row 1 | ≤ 1e-12 | −7.3e-3 | −5.6 |
| E12 gradient relative error | ≤ 1e-9 | 2.3e-17 | 4.1e-18 |

- Accepted from attempt 1 (`schur`): 13. Accepted from attempt 2 (`augmented`): **1**, at outer 5.
  There, the schur point's best dual gap was 6.9e-8 > 1e-8, while the augmented point
  certified at gap 5.3e-11. This is the preregistered cascade working as written.
- coneprog exit flag of the accepted schur solves was −7 in all 14 cases. Acceptance
  never used the exit flag.
- Certifying dual candidate: complementary slackness 5, `fp_dualbound` general 9.
- Interior-point iterations per accepted solve: median 23, max 32.

**Bound saturation.** Median 99.98% of Δρ entries sat on a bound. 12 of 14 steps
had ≥ 99.9% of entries on a bound, and **13 of 14 steps moved ≥ 99% of all elements**.
Median gray full-move fraction was 99.97% (minimum 94.2% at outer 3). In outer 1–12
the move bound was active for essentially every element; outer 13–14 had 61% at
±move and 38% at a density bound.

**Solution degeneracy** (Amendment 1 cross-solver telemetry, non-gating, 14 accepted
iterations). Median d2 was 1.7e-7, 90th percentile 5.6e-5, maximum 2.7e-3 (outer 13).
dinf exceeded 0.1 at 1 of 14 iterations (0.214 at outer 13, 4 elements). The cross
point certified in 13 of 14 cases, and max |Δbs| was 6.9e-8. The flat-face
non-uniqueness first seen in preflight P4 recurs, but it is small.

## The termination (outer 15)

`evaluations/termination_record.json`. At ρ₁₄, ω₁ = 148.029, ω₂ = 150.148
(relative λ gap 0.0288, ω gap12 0.0143), λ_J = 107 140, move 0.04.

| | schur (attempt 1) | augmented (attempt 2) |
|---|---|---|
| exit flag / IPM iterations | **1** / 28 | **1** / 28 |
| solve time | 58.2 s | 4.1 s |
| bs − 1 (predicted gain 668.8 in λ) | 3.0523e-2 | 3.0523e-2 (Δ = 8.4e-13) |
| max production row | 3.2e-12 | 3.2e-12 |
| ‖A_c x − b_c‖ (cone apex test 2‖s‖ ≤ 1e-9) | **2.98e-12 → apex** | **2.98e-12 → apex** |
| predicted separation e₂ − e₁ | 1.3e-7 (λ units) | 1.3e-7 |
| best dual gap (`fp_dualbound` general / aligned) | 3.59e-4 / 3.71e-4 | 3.60e-4 / 3.71e-4 |
| failed bars | C3 (gap), C6 (box complementarity 0.29) | C3, C6 |

The exact sub-problem optimum at outer 15 lies **at the apex of the second-order
cone**. There the predicted cluster eigenvalues coincide, which is the double
eigenvalue that max–min eigenvalue optimization is known to seek. The predicted
separation shrank along the run: 66 936 (outer 1), 31 423 (9), 3 196 (11), 133 (12),
1 075 (13), 193 (14), then 1.3e-7 (15). At the apex, the direction w = s/‖s‖ is
undefined. The preregistered candidates (i) and (ii) are therefore unavailable there
by design, and the remaining derivative-free dual search (`fp_dualbound`) did not
find a witness within 1e-8. Both backends returned the same feasible point with
exit flag 1. Under the preregistration this is `SOCP_CERTIFICATE_FAILURE`: no
update, no fallback, no continuation, no rerun.

## POST-HOC DIAGNOSTIC — excluded from every verdict

`scripts/cs_posthoc_apex.m` → `evaluations/posthoc_apex_diagnostic.json`. The
rejected outer-15 problem was re-posed from the saved treatment state (λ, λ_J, dOff
bitwise equal to the record). No design was updated and nothing was continued.

| question | result |
|---|---|
| does schur reproduce the rejected point? | yes: exit flag 1, 28 iterations, bs equal to the record within 1e-14, at the apex |
| structure of the rejected point | 236 interior elements; 99.2% on bounds; 98.5% of gray elements at ±move |
| exact Lagrangian dual solved as its own conic program (28 806 variables) | its interior-point method stalled (exit −7, 16 iterations, 467 s); dual-feasible witness with **gap 8.6e-6**, box complementarity 1.4e-3 |
| generalized complementary slackness with free p | gap 1.5e-2 (poor) |
| cone multiplier | μ = 1.000, ‖p‖/μ = 0.924 < 1: a strictly interior conic multiplier, i.e. a mixed-eigenvector (nonsmooth, multiple-eigenvalue) optimum |

Interpretation, bounded by the evidence:

- **Supported.** The rejected point is feasible and provably within **8.6e-6** of the
  sub-problem optimum in bs units, which is ≤ 0.03% of its predicted gain. Two
  independent backends agree on its objective to 8e-13.
- **Not supported.** No construction tried, including the exact dual solve, certified
  it to the preregistered 1e-8. So "exact to 1e-8" is unproven at this state.
- **Therefore.** The stop is best characterized as **certificate unattainability at
  the cone apex**, not as a wrong or infeasible primal step. That is a limitation of
  the preregistered certification procedure in exactly the multiple-eigenvalue regime
  that problem (25) is designed for. It does not change any verdict.
