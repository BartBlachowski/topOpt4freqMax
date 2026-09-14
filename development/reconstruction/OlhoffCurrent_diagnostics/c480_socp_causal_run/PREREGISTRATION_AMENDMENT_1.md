# PREREGISTRATION AMENDMENT 1 — pre-launch, before any C480 treatment iteration

Written 2026-09-13 ~11:40 +0200. At this time NO C480 treatment iteration had
been executed; no treatment trajectory, state, or endpoint existed. The original
preregistration (`AUDIT_PREREGISTRATION.md`, SHA-256
`5b6186f2f2b438f73ce4cfeae8e4565326dc91d99b19e207aa47604c32286ddd`) is unchanged
and remains in force except for the two items below. Its SHA-256 is re-verified
at launch and at finalization. This amendment's SHA-256 is recorded in
`evaluations/amendment1_sha256.txt` immediately after writing.

## What triggered it

Preflight P4 (known-answer test on the hashed frozen C480 problem) FAILED for the
preregistered primary backend `LinearSolver='augmented'`
(`evaluations/preflight_P4_original_augmented_FAIL.json`):

| | augmented (attempt 1 as preregistered) | bar |
|---|---|---|
| certified | yes (gap 6.59e-12, candidate complementary slackness) | — |
| \|bs − bsO\| | 4.91e-12 | ≤ 1e-8 PASS |
| gain recovery | 1.0000 | ≥ 0.999 PASS |
| d2 | 0.0107 | ≤ 0.01 **FAIL** |
| dinf | 0.745 | ≤ 0.1 **FAIL** |

The read-only pre-launch diagnostic `scripts/cs_preflight_degeneracy.m`
(`evaluations/preflight_degeneracy.json`) established the cause:

- The two points differ materially in exactly 4 elements (two mirror-symmetric
  pairs, ρ = 0.5731 and ρ = 0.0144), all interior in both solutions.
- Their certified reduced costs are \|q\|/sRow0 ≈ 1.7e-6.
- The segment between the oracle and the augmented point is feasible everywhere
  (max production row ≤ 1.0e-13), and bs varies linearly along it by 4.9e-12 in total.

So problem (25) at the frozen state has a **degenerate, numerically flat optimal
face**. Different exact interior-point backends select different points on it.
This is a scientific property of the sub-problem, not a software defect. It also
shows that the prior study's design-space fidelity bars were only ever met by the
schur oracle against itself.

`LinearSolver='schur'` reproduces the certified oracle **bitwise** at 1, 4 and 8
BLAS threads (27 iterations, ~38 s), and is independently certified (gap 6.45e-12).

## The amendment (two items only)

1. **§4.2 attempt order is swapped.** Attempt 1 = `'schur'` (the certified-oracle
   configuration validated by `frozen_inner_solver_study`); attempt 2, only if
   attempt 1 is not certified, = `'augmented'`. Formulation, tolerances,
   certificate, every bar C1–C8 and E1–E12, and every fail-closed rule are
   unchanged.
2. **§4.4 cross-solver diagnostic runs at every outer iteration** (not every
   50th). The non-accepted backend re-solves the identical problem. d2, dinf,
   |Δbs|, sign/bound agreement and certification of that point are recorded as
   per-iteration **solution-degeneracy telemetry**. It never gates acceptance, and
   its time is excluded from inner-solver cost.

No threshold of any kind is changed. P4 is re-run with the amended primary and
must pass its unchanged bars. The design-space bars of P4 are **not** waived.

## Consequence recorded before launch

Per-iteration SOCP wall time rises from ~3 s to ~40 s. If the treatment needs
~400 outer iterations the run takes ~4.5 h; at the 1600 cap it takes ~18 h. This
is accepted: the causal question requires the certified oracle method. Because of
the flat optimal face, "exact solution of (25)" means certified ε-optimality
(gap ≤ 1e-8), with the selection on degenerate faces made by the validated
backend. The size of that non-uniqueness is measured at every iteration by item 2
and reported alongside the causal verdict.

## Non-scientific launch detail (disclosed)

The run is launched under `caffeinate -i` so macOS idle sleep does not suspend it.
This has no effect on arithmetic.
