# COST_ANALYSIS — Part 15

Recorded, not benchmarked. Host: shared macOS machine, MATLAB R2025b, one numerical
thread for the solve. Control timings come from its retained record and `hist`. The
treatment has only 14 accepted iterations, so cost is compared per outer iteration
and over the matched first 14.

| | control, all 386 | control, first 14 | **treatment, 14 accepted** |
|---|---|---|---|
| Σ tOuter | 2 798.9 s | 24.06 s | **600.5 s** |
| wall (process solve) | 2 799.3 s | — | 667.3 s (includes the rejected outer 15: 64 s of attempts and the final analysis) |
| Σ tEig (assembly + eigensolve) | 104.8 s | 4.32 s | 4.73 s |
| Σ tGrad | 6.51 s | — | 0.19 s |
| Σ tInner | 2 686.9 s | 19.54 s | **595.6 s** |
| of which: eligibility | — | — | 0.014 s |
| assembly (conic data) | — | — | 0.27 s |
| accepted SOCP solves | — | — | 506.2 s |
| certificates | — | — | 11.7 s |
| cross-solver diagnostic (excluded, Amendment 1) | — | — | 67.4 s solve + 8.9 s certificate |
| inner iterations | 7 300 MMA sub-iterations | 254 | 360 interior-point iterations (incl. 1 second attempt) |
| **inner-solver cost per outer iteration** | 6.96 s | 1.40 s | **≈ 37.1 s** excluding cross-solver (median solve 32.4 s + certificate 0.64 s) |
| inner share of outer time | 96.0 % | 81.2 % | 99.2 % (≈ 86 % excluding cross-solver) |

- One exact `schur` solve costs about **23×** the MMA inner loop per outer iteration
  in early stage 1 (37.1 s vs 1.40 s), and about 5× the control's run-average MMA cost
  (6.96 s, which rises at the smaller move levels).
- `augmented` solves the same problem in 2–5 s: 4.1 s at the rejected outer 15, and
  2.3–2.8 s in preflight. It was demoted to second attempt because it selects a
  different point on the flat optimal face (P4 design bars failed); speed played no
  part in that decision.
- Assembly and eligibility cost nothing measurable, and certificates cost about 2 % of
  solve time. The `schur` factorization dominates.
- The run's evidence cannot project a full-run cost: the treatment never reached the
  regime where the control's per-iteration MMA cost triples.

No code was optimized.
