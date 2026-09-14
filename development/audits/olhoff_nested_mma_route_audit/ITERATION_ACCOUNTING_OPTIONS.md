# WP16 / WP17 — Work metrics and timing
NESTED-MMA ROUTE AUDIT — CANDIDATE MEASURES, NOTHING FROZEN

## The commensurability problem, stated first

One LP call and one MMA sub-iterate are **not** the same unit of work and must never be
added. Measured on this machine at 160x20:

| unit | count per outer | wallclock per outer | source |
|---|---|---|---|
| LP call (`linprog`, HiGHS) | exactly 1 (verified for all 1600 outers of `lprmin1.2`) | 0.012 s inner + 0.031 s eigensolve | `res.hist` |
| MMA sub-iterate (`mmasub`+`subsolv`) | mean 142.8, median 107 | 17.775 s inner + 0.029 s eigensolve | `res.hist` |

The eigensolve cost is **identical** between routes (0.031 s vs 0.029 s). The entire
difference is the inner solver. Any combined scalar of the form
`N_outer + N_inner_total` would be meaningless, and this audit does not compute one.

Also note: HiGHS/simplex iterations inside `linprog` are **not** counted as inner iterations
anywhere here. The LP inner count is the number of `linprog` *calls*, which is 1 per outer
by construction (`innerLoopLP` performs a single solve).

## Candidate paper-facing work vector for the MMA route

Reported as a decomposition, never collapsed:

| measure | BASE_mma (752 outer) | fm_mma_diag (400 outer) |
|---|---|---|
| `N_outer` | 752 | 400 |
| `N_inner_total` (MMA sub-iterates) | **107 373** | 55 278 |
| mean `N_inner` per outer | 142.8 | 138.2 |
| median `N_inner` | 107 | 105 |
| p95 `N_inner` | 300 | 300 |
| max `N_inner` | 300 | 300 |
| cap-hit count (`= maxInner`) | 114 (15.16%) | 46 (11.5%) |
| converged-inner count | 638 (84.84%) | 354 (88.5%) |
| `N_inner_total / N_outer` | 142.8 | 138.2 |

Split by multiplicity regime — this is the result with genuine scientific content:

| regime | outers | mean | median | min | p90 | p95 | max | total | cap-hit | converged |
|---|---|---|---|---|---|---|---|---|---|---|
| `N = 1` (simple) | 69 | **93.4** | 93 | 83 | 100 | 102 | 108 | 6 448 | **0.00%** | **100.00%** |
| `N = 2` (multiple) | 683 | **147.8** | 111 | 26 | 300 | 300 | 300 | 100 925 | **16.69%** | 83.31% |
| all | 752 | 142.8 | 107 | 26 | 300 | 300 | 300 | 107 373 | 15.16% | 84.84% |

Mann–Whitney U test, one-sided (`N=1` inner effort < `N=2` inner effort):
**U = 12 677, p = 1.1e-10**. Ratio of means **1.581**.

**Multiplicity materially increases inner subproblem effort, by about 58% in the mean, and
introduces a heavy upper tail that the simple-mode phase does not have at all** (p90 and p95
both saturate the 300 cap under multiplicity; the simple-mode maximum over 69 outers is 108).
This is a clean, quantified statement about the cost of the multiple-eigenvalue formulation
and it is the most publishable single result in this audit.

The corresponding LP work vector is degenerate by construction: `N_outer = 1600`,
`N_inner_total = 1600`, mean/median/p95/max all exactly 1, cap-hits 0, converged 1600.
The LP route therefore **cannot** exhibit this phenomenon — which is simultaneously its
practical advantage and the reason it carries no information about multiplicity cost.

## Timing (WP17)

Timing **was** instrumented in `res.hist` (`tEig`, `tGrad`, `tInner`) for the saved runs, so
no reconstruction from timestamps was needed. BASE_mma itself has no `.mat`, so its timing is
taken from its bit-identical prefix `fm_mma_diag`.

| | LP (`lprmin1.2`, 1600 outer) | MMA (`fm_mma_diag`, 400 outer) |
|---|---|---|
| wallclock | **70.7 s** | **7 121.7 s** |
| mean `tEig` per outer | 0.031 s | 0.029 s |
| mean `tGrad` per outer | 0.001 s | 0.001 s |
| mean `tInner` per outer | **0.012 s** | **17.775 s** |
| inner share of optimizer time | 27.4% | **99.8%** |
| wallclock per outer | 0.044 s | **17.8 s** |

**Ratio per outer iteration: ~400×.** Extrapolating BASE_mma's 752 outers gives ≈ 3.7 h at
the *smallest* production mesh.

Scaling to the campaign is the decisive practical fact. The MMA inner cost is dominated by
`mmasub`/`subsolv`, which are O(number of design variables) per sub-iterate. From 160x20
(3 200 elements) to 720x90 (64 800 elements) is a 20× increase in design variables, so a
conservative estimate of ~350 s per outer at 720x90 and ~1 600 outers gives **on the order of
150 hours for a single mesh, single method**. The nine-mesh iteration-efficiency campaign is
not feasible on this route without a fundamentally faster inner solve.

This timing evidence is single-run and uninstrumented for repetition, so it is indicative,
not a controlled benchmark. A controlled timing replay would be required before any of these
numbers entered a paper table. Iteration accounting remains primary.
