# STOPPING_COMPARISON — Part 16 (what "natural convergence" is)

## 1. Source criterion (executed)

| | |
|---|---|
| formula | stop at the first outer k with ‖Δρ_k‖₂ < ε (`stop.norm = l2`, `stop.field = designVariable`) |
| threshold | ε = 0.05·√(NE/3200) = 0.15 at 480×60 (meshScaled) → RMS per element 8.84e−4 |
| monitored quantity | the increment returned by problem (25), i.e. bounded by the per-element box |
| persistence | none (single iteration) |
| guards | none (`settledMove = false`, `boxInactiveFraction = 0`) |
| KKT? | no |
| design change? | yes (absolute, L2) |
| eigenvalue-based? | no |
| relative / absolute | absolute, mesh-scaled |

Because |Δρ_e| ≤ d_e ≥ 0.002, the test can fire only when the "box-floor-equivalent" moving set is
below (ε/0.002)²/NE = **19.5 % of elements** — a mesh-independent fraction. Oscillating elements are
contracted to the floor by ×0.7 per reversal (0.10 → 0.002 after 11 reversals), so the criterion
fires once oscillation is confined and monotone evolution has ended.

## 2. Target criteria

- **Three-rung (C480):** convergence only at the last rung (0.01) and only after the stage-exhaustion
  rule E = A ∨ B has held for 20 iterations; the §3.5.1 ε-test is computed but not used. Terminal
  declaration at 386, branch B (‖Δρ‖₂ < ε and positive median directional coherence).
- **Canonical production:** ‖Δρ‖₂ < ε with the settled-move guard; β-stall ladder.
- On C480 the raw ε-test was first satisfied at outer 289, still at move 0.04, 20 iterations
  before the first descent.

## 3. Endpoint stationarity (frozen definition, `gray_kkt_forensic_audit/scripts/stationarity.py`)

Validation: this audit's re-implementation gives **0.3340391** at the C480 endpoint vs the prior
**0.334039** (|Δ| = 7e−8). Gray-fit physical residual (RMS over 0.1 < ρ < 0.9, normalized by the
interior raw RMS), simple-eigenvalue branch valid at both endpoints (gap > 5 %):

| endpoint | evaluated law | physical gray-fit RMS | filtered-model RMS (raw scale) | gray elements |
|---|---|---|---|---|
| C480 386 | SIMP + eq.4b (native) | **0.334** | 0.049 | 8 274 |
| C480 386 | Pedersen + eq.2 | 0.306 | 0.046 | 8 274 |
| S480 112 | Pedersen + eq.2 (native) | **0.334** | 0.011 | 4 346 |
| S480 112 | SIMP + eq.4b | 0.364 | 0.014 | 4 346 |
| M1 64 | Pedersen + eq.2 | 0.268 | 0.109 | 9 158 |
| M1 64 | SIMP + eq.4b (native) | not valid (gap 0.14 %, localized modes) | — | 9 158 |

Preregistered rule (§11): under a common formulation, S480's residual is ≥ C480's (0.364 ≥ 0.334
under SIMP/4b; 0.334 ≥ 0.306 under Pedersen/eq.2) → **HEURISTIC_STOP**. The source endpoint is not
a better physical KKT point; it is a much less gray design at a similar residual, and it is ~4× closer
to stationarity of the filtered local model.

## 4. Classification of natural convergence

**A different heuristic stop on a fundamentally different trajectory, not genuinely better local
optimality.**

- The criterion is a design-change test with no persistence; it fired at 112 after the design had
  already stopped changing (M_nd range 0.0021 and ω₁ range 0.11 % over the last 30 iterations; M_nd
  0.134 at 80, 0.131 at 100). It preserved an already-sharp design; it did not create it.
- The same criterion fires falsely when the formulation misbehaves: **M1 stopped at 64 with
  ‖Δρ‖₂ = 0.1165 < 0.15 in a localized-mode state** (final native ω₁ = 34.36), because repeated spike
  reversals had contracted 80 % of the boxes to the floor. Under SIMP/eq.(4b) the "natural"
  termination is not credible; under Pedersen it was clean at all 18 committed runs.
- The target's three-rung stop is also a design-dynamics heuristic, with persistence and coherence
  guards; its endpoint is equally non-stationary physically (0.334).

Migration consequence: the new preset's caveat must state that the ε-test is a design-change
heuristic whose credibility at this radius rests on the Pedersen law (spike-free), and a spike
guard or post-hoc spectral check should be reported with every benchmark row.
