# ENDPOINT_METRICS — Parts 8 and 9

**The treatment has no scientific endpoint.** It stopped fail-closed at outer 15
(no update applied), inside stage 1 at move 0.04. It never reached a controller
declaration. "Treatment final" below is the last accepted design ρ₁₄. It is compared
with two control states:

- the preregistered comparator, the control endpoint ρ₃₈₆ (not like-for-like: 14 vs 386 iterations);
- a descriptive matched-iteration comparator, control ρ₁₄.

No causal claim is drawn from either comparison (`CAUSAL_VERDICT.md`). Definitions and
code are the prior audit's, by proof (`CONTROL_IDENTITY.md`).

## Topology-quality metrics

| metric | control final (386) | control at outer 14 | **treatment at outer 14** |
|---|---|---|---|
| M_nd | 26.342 % | 71.844 % | **48.127 %** |
| gray fraction (0.1 < ρ < 0.9) | 0.2873 | 0.7990 | **0.5451** |
| mid fraction (0.4 ≤ ρ ≤ 0.6) | 0.1186 | 0.3497 | **0.3073** |
| gray area | 2.298 | 6.392 | **4.361** |
| broad-core fraction (gray ∧ depth > R) | 0.1301 | 0.7017 | **0.3625** |
| broad-core physical area | 1.041 | 5.613 | **2.900** |
| max gray depth | 0.3667 | 0.6719 | **0.4000** |
| max gray depth / R (R = 0.06) | 6.11 | 11.20 | **6.67** |
| gray depth p50 / p90 / p99 | 0.047 / 0.239 / 0.350 | 0.250 / 0.502 / 0.626 | 0.100 / 0.306 / 0.383 |
| gray components (4- / 8-connected) | 7 / 7 | 1 / 1 | **1 / 1** |
| largest gray component area | 0.794 | 6.392 | **4.361** |
| ρ < 0.01 / ρ > 0.99 | 32.7 % / 33.7 % | 0 % / 8.5 % | 15.8 % / 22.5 % |
| ρ quantiles 10 / 25 / 50 / 75 / 90 % | 0.0022 / 0.0022 / 0.494 / 0.9986 / 0.9994 | 0.171 / 0.337 / 0.451 / 0.643 / 0.977 | **0.0010 / 0.180 / 0.420 / 0.915 / 1.000** |

At matched iteration 14 the exact-SOCP trajectory was much further along than MMA:
M_nd 48 % vs 72 %, broad core 0.36 vs 0.70. The treatment was already exactly at the
bounds for 38 % of elements (ρ = 10⁻³ or 1 to machine precision), while no control
element was. It was still far grayer than the control endpoint, as any design 14
iterations from uniform must be. The per-iteration trajectories (FIG_06–08) show
treatment M_nd falling roughly 2× faster per iteration than the control over outer
1–14. Where it would have ended is **unknown**.

## Objective and spectrum

| | control final (386) | control at outer 14 (ρ₁₄) | **treatment at outer 14 (ρ₁₄)** |
|---|---|---|---|
| ω₁ | 163.93226 | 133.43279 | **148.02874** |
| ω₂ | 185.21032 | 174.61633 | **150.14759** |
| λ₁ | 26 873.79 | 17 804.3 | **21 912.51** |
| λ₂ | 34 302.86 | 30 490.9 | **22 544.30** |
| gap12 (ω) | 0.1298 | 0.3086 | **0.0143** |
| next-mode gap (ω₃ − ω₂)/ω₂ | 1.167 | 1.799 | **1.180** |
| volume | 0.4999985 | 0.4998495 | **0.5000000** |
| controller terminal state | stage 3, terminal declaration (B) | stage 1 | **stage 1, no declaration, fail-closed** |
| total outer iterations | 386 | — | **14 accepted + 1 rejected** |
| ω₁ gain from the uniform design (68.216) | +95.72 | +65.22 | **+79.81** |

Matched-iteration ω₁ was 10.9 % higher under exact SOCP. The treatment had reached
90.3 % of the control's final ω₁ after 14 iterations, where the control needed about
25. The two lowest modes were nearly coalescent (gap12 0.014). That is the state in
which the next exact sub-problem optimum sat at the cone apex, and its certificate
failed.

Figures: `FIG_01`–`FIG_05`, `FIG_06`–`FIG_09`, `FIG_17_gray_component_maps`.
