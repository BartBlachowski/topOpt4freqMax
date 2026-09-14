# WP8 — Candidate B: Eq. (4a), lowest algebraic mode
PHASE 2F — EVIDENCE ONLY

Candidate B is the Phase-2D amendment as proposed, and Phase 2E rejected it. This phase
quantifies its failure systematically across all eight meshes rather than at a handful of
states.

## Numerical continuity and stability: excellent

Independently reproduced in Phase 2E and not re-litigated here. Eq. (4a) removes the
discontinuity entirely: branch-side sensitivity on 1600 production states falls from
2.6496e-02 to **2.6560e-10**, matching the branch-free E1 control at 2.6252e-10 (ratio
1.012); float32 sensitivity falls from 2.6736e-02 to **5.5960e-08** against E1's 5.5949e-08.
On every perturbation-response measure candidate B is as good as an evaluator can be.

## Artificial-mode incidence: the failure

| mesh | model | states | invalid lowest mode | % | hard-gate PASS & invalid | % of PASS | worst ω₁ / ω_structural |
|---|---|---|---|---|---|---|---|
| 160x20 | E2 | 1600 | 30 | 1.88% | 7 / 1065 | 0.66% | **0.1888** |
| 160x20 | E3 | 1600 | 34 | 2.12% | 7 / 1065 | 0.66% | **0.1346** |
| 240x30 | E2 | 1600 | 53 | 3.31% | 53 / 1490 | 3.56% | 0.1894 |
| 240x30 | E3 | 1600 | 59 | 3.69% | 59 / 1490 | 3.96% | **0.1341** |
| 320x40 | E2 | 1600 | 47 | 2.94% | 47 / 1564 | 3.01% | 0.2688 |
| 320x40 | E3 | 1600 | 56 | 3.50% | 56 / 1564 | 3.58% | 0.1917 |
| 400x50 | E2 | 400 | 24 | 6.00% | 24 / 386 | 6.22% | 0.3082 |
| 400x50 | E3 | 400 | 31 | 7.75% | 31 / 386 | 8.03% | 0.2214 |
| 480x60 | E2 | 357 | 14 | 3.92% | 14 / 325 | 4.31% | 0.4440 |
| 480x60 | E3 | 357 | 23 | 6.44% | 23 / 325 | 7.08% | 0.3145 |
| **560x70** | E2 | 200 | 47 | **23.50%** | 47 / 176 | **26.70%** | 0.4039 |
| **560x70** | E3 | 200 | 67 | **33.50%** | 67 / 176 | **38.07%** | 0.2867 |
| 640x80 | E2/E3 | 134 | 32 / 36 | 23.9 / 26.9% | — | — | — |
| 720x90 | E2/E3 | 160 | 30 / 40 | 18.8 / 25.0% | — | — | — |

**The lowest algebraic eigenvalue is a void mode at between 1.9% and 33.5% of states,
depending on mesh and evaluator, and the affected states are overwhelmingly hard-gate-passing.**
At the worst state the reported ω₁ is **13.4%** of the structural value — an 87% error, in a
study whose tightest acceptance band is 0.5%.

The incidence is **not** confined to early gray states: it occurs at 560x70 in a third of the
sampled trajectory, and Phase 2E found an affected state at k = 1385 of 1600 at 320x40, close
to convergence.

## Quality-sequence behaviour

On the 160x20 trajectory, candidate B's scalar sequence has a **maximum single-iteration step
of 1.222e+00 — a 122% jump between consecutive states** — and 131 steps above 0.5%. There are
no undefined states, but the sequence is not a continuous measurement of anything: it silently
alternates between reporting a structural frequency and reporting a void-mode frequency.

## Assessment

Candidate B is **numerically impeccable and physically invalid**. It is the clearest
demonstration in this whole investigation that perturbation stability and measurement validity
are independent properties, and that testing only the former can certify an evaluator that
does not measure the estimand at all.

**Candidate B is refuted.** No threshold, tolerance or parameter choice rescues it, because
the defect is in *which mode is reported*, not in how accurately it is computed.
