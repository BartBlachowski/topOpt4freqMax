# PROVENANCE — filtered_subproblem_integrability_audit

## 1. Locks honoured

| lock | status |
|---|---|
| ZERO optimization runs | **0** — no `olhoffSolve` call was made by this audit |
| ZERO accepted density updates | **0** — no ρ was written anywhere, at any point |
| ZERO continuation | none |
| ZERO controller transitions | none |
| production files modified | **none** |
| filter / p / q / mass / multiplicity / MMA formulation altered | **none** |
| projection run | no |
| p-continuation run | no |
| another mesh run as a primary case | no — 480×60 only; **no 400 or 800 evaluation was performed** |

Every frozen-subproblem solve is labelled **FROZEN-SUBPROBLEM CERTIFICATION**
and its `drho` is discarded in code (`clear drho`) without ever being added to a
density.

## 2. Preregistration

`AUDIT_PREREGISTRATION.md`, SHA-256
`53d6623a9f0807217c515ed66e158ce617ab5d4a834bc9e9c07058e533644f81`, frozen
before any evaluation beyond file-identity checks. No tolerance, bar, direction,
δ, amplitude or verdict rule was changed after seeing a result.

**One preregistered statistic was found to be mis-specified, and is reported
anyway.** §4 defines the Level-2 projected stationarity residual with a bound
classification tolerance of 1e−12 of the box width. `mmasub` uses an
interior-point solver that never places a variable exactly on a bound, so at
that tolerance **zero** variables are classified active and the statistic omits
the box multipliers `−ξ + η` from the Lagrangian gradient. The preregistered
number (1.006) is reported unchanged; the mathematically correct residual
including `ξ, η` (0.360) is reported alongside it, together with a
bound-tolerance sweep. **Both cross the preregistered FAIL bar of 0.1**, so the
verdict does not depend on the error. This is disclosed rather than quietly
corrected.

## 3. State audited

The 480×60 three-rung canary endpoint, ρ SHA-256
`0a498a7d6ab0565b29c15ff9364060d937d4df10aa661fc90d02c038ce6e4a60`, config hash
`03097a28…782e`, `+impl` tree `edbfe47e…52cb` (verified), outer 386 — byte-identical
to the state `gray_kkt_forensic_audit` analysed.

## 4. Compute performed — all read-only

| activity | evaluations | note |
|---|---|---|
| state identity + spectrum at ρ₃₈₆ | 1 | Part 1 |
| final subproblem reproduction (19 inner) | 2 eigensolves | bitwise exact |
| frozen-subproblem certification, `tolInner = 1e−10` | 500 inner | drho discarded |
| extended certification, `maxInner = 5000`, `tolInner = 1e−12` | 5000 inner | drho discarded |
| Jacobian symmetry (Part 5) | 88 | 31.2 s |
| closed-loop integrals (Part 6) | 1 024 | 390.3 s |
| mixed partials (Part 7) | 61 | 21.3 s |
| analytic column verification | 13 | |
| asymmetry decomposition | 22 | |
| counterfactual operators | 22 | analysis only, no run uses them |

No FE evaluation advanced any design. Each perturbed ρ is a temporary
evaluation point constructed in memory and discarded.

## 5. Audit-only code, and how it relates to production

`scripts/fi_innerloop_audit.m` is a **mirror** of `+impl/algo/innerLoop.m`.
Every numeric expression is copied character-for-character. The only differences
are that the `mmasub` duals production discards with `~` are captured, an
optional `tolInner` override is accepted for the certification re-solve, and a
`lean` flag limits what is retained for the 5000-iterate run. No constraint,
bound, scaling, asymptote or MMA constant differs. Its fidelity is proved by the
bitwise reproduction of `DRHO(:,386)`.

`scripts/fi_counterfactual_operators.m` evaluates the antisymmetry of `M·Hess`
for three operators other than the production filter. It is **diagnostic
arithmetic only**: it changes no file, and no optimization uses its output.

Everything else (`fi_setup`, `fi_eval`, `fi_state`, `fi_subproblem`,
`fi_kkt_refine`, `fi_inner_converge`, `fi_directions`, `fi_symmetry`,
`fi_loops`, `fi_mixed`, `fi_analytic_verify`, `fi_decompose`, `fi_filter`,
`fi_admissibility`, `fi_export_fields`, `fi_figures.py`) is new to this audit.

## 6. Disclosed imperfections

1. **The preregistered bound tolerance** (§2 above).
2. **One loop corner leaves the design box.** `D4a,D4b` at `a = 1e−3` takes one
   element to 2.20e−06 below ρmin (0.22 % of ρmin). Disclosed in
   `CLOSED_LOOP_INTEGRALS.md`; that pair has the smallest asymmetry of the ten
   and excluding it strengthens the result.
3. **The submatrix test of the analytic decomposition is inconclusive by
   construction** — it cannot separate the two skew terms. Two other tests do,
   and both succeed; recorded in `MIXED_PARTIAL_AUDIT.md` rather than presented
   as a success.
4. **A first extended-certification attempt was aborted.** Its recorder retained
   full per-iterate state (~2.4 MB × 5 000 ≈ 12 GB) and was killed after 24
   minutes without completing. It produced no result and none is reported; the
   recorder was made lean and the run repeated. No scientific quantity came from
   the aborted attempt.

## 7. Host and toolchain

Apple M1 Max, 10 cores, 64 GiB; macOS 26.6.2; MATLAB **25.2.0.2998904 (R2025b)**,
`maxNumCompThreads(1)`. The validated C320 run used build 25.2.0.3042426; the
difference is recorded, and no floating-point-exact comparison against that run
is made here.

**The host was shared.** Three unrelated single-threaded MATLAB jobs
(`run_repro`, 800×100) belonging to other work were running concurrently for
part of this audit, which is why the 5 000-iterate certification took 7 334 s
rather than the estimated ~4 000 s. **No result in this audit is a timing
measurement**, and every quantity reported is deterministic arithmetic on
frozen inputs, so concurrency affects nothing but wall-clock. It is recorded
for completeness. Those jobs were not interfered with.

## 8. Epistemic class

Unchanged: a **reconstruction (class C)** of Du & Olhoff (2007). Nothing in this
audit strengthens or weakens that classification, and nothing here is a claim
about the authors' implementation.
