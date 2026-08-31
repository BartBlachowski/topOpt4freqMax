# WP19 — Computational cost of the candidates
PHASE 2F — EVIDENCE ONLY

All timings on the audit machine (Apple silicon, single process, `scipy.sparse` +
ARPACK shift-invert at σ = 0 over a sparse LU). Sparse throughout; the only dense solve in
this phase is the deliberate WP1 cross-check.

## Measured cost of one modal solve

| mesh | elements | free DOF | nnz(K) | assemble | symmetrise+slice | sparse LU alone | eigsh k=12 (total) |
|---|---|---|---|---|---|---|---|
| 320x40 | 12 800 | 26 318 | 413 110 | 0.027 s | 0.009 s | 0.130 s | **0.35 s** |
| 720x90 | 64 800 | 131 218 | 2 340 616 | 0.129 s | 0.041 s | 1.224 s | **2.97 s** |

The eigensolve dominates: 94% at 720×90. Assembly optimisation would buy under 5%, so it
was not pursued. Neither `sksparse.cholmod` nor `pypardiso` is installed; with CHOLMOD the
symbolic factorisation could be analysed once per mesh and reused across all states, which
is the single largest available speedup and is worth installing before any production-scale
evaluator sweep.

Scaling of the full k=12 solve across the mesh sequence (measured, one state each):

| mesh | 160x20 | 240x30 | 320x40 | 400x50 | 480x60 | 560x70 | 640x80 | 720x90 |
|---|---|---|---|---|---|---|---|---|
| seconds | 0.07 | 0.17 | 0.35 | 0.69 | 1.17 | 1.47 | 2.17 | 2.97 |

Least-squares fit over these eight points: **T ∝ N_el^1.264**. Mildly super-linear, as
expected for a 2-D sparse LU on a long thin domain.

## Cost of each candidate, per state

| candidate | eigenpairs needed | eigenvectors needed | extra work vs the frozen evaluator |
|---|---|---|---|
| **A** Eq. (4), lowest | 3 (frozen) | no | baseline |
| **B** Eq. (4a), lowest | 3 | no | none |
| **C** Eq. (4a), lowest structural | ≥ the first structural ordinal | **yes** — the diagnostic needs φ | see below |
| **D** exact-count binary | 3 | no | one extra projection (O(N log N) sort), negligible |

Candidate C is the only one that costs more, on two counts.

**More modes.** Requesting `k` modes instead of 3. Measured at 160×20, per state:

| k | 3 | 6 | 12 | 24 | 48 |
|---|---|---|---|---|---|
| seconds (median of 5) | 0.0465 | 0.0550 | 0.0663 | 0.0903 | 0.1557 |
| factor vs k = 3 | 1.00× | 1.18× | **1.43×** | **1.94×** | 3.35× |

Strongly sub-linear in `k`, because the sparse LU is paid once and each additional Arnoldi
vector costs only a back-substitution. Going from 3 to 12 modes costs **1.43×**; to 24
modes, **1.94×**.

**Eigenvectors.** The frozen evaluator calls `eigs` discarding eigenvectors. Candidate C
needs them, plus one element-wise energy reduction per mode:

    ke = m_e · Σ (u_e^T ME u_e)        8x8 contraction per element per mode

Measured, k = 12:

| mesh | eigsh without eigenvectors | with eigenvectors | energy reduction | reduction as % of solve | eigenvector memory |
|---|---|---|---|---|---|
| 160x20 | 0.041 s | 0.041 s | 0.004 s | 9.3% | 0.6 MB |
| 720x90 | 2.624 s | 2.664 s | 0.079 s | 2.9% | **12.6 MB** |

Returning eigenvectors is essentially free in ARPACK (they are already formed internally);
the added cost is the energy reduction, which *falls* as a share of the solve with mesh size
— 9.3% at 160×20, 2.9% at 720×90.

## Full-trajectory burden

Cost to evaluate one full 1600-state trajectory, one evaluator, k = 12 with eigenvectors:

| mesh | 160x20 | 320x40 | 480x60 | 640x80 | 720x90 |
|---|---|---|---|---|---|
| minutes | 1.9 | 9.3 | 31 | 58 | 79 |

For a nine-mesh, three-method, three-evaluator campaign the dominant term is the largest
meshes. Summing the measured per-state costs over the eight available meshes at 1600 states
each gives ≈ **3.3 h per (method, evaluator)** single-threaded, hence ≈ 30 h for
3 methods × 3 evaluators. That is embarrassingly parallel across states and meshes; on 8
cores it is roughly 4 h.

Candidate D on the same basis is ≈ 0.70× that (k = 3 rather than 12, no energy reduction),
i.e. ≈ 21 h single-threaded.

## Assessment

Neither candidate is impractical. Candidate C costs roughly **1.4–1.9×** candidate D in
post-processing (k = 12–24 vs k = 3, plus a 3–9% energy reduction), plus 12.6 MB of
transient memory at the largest mesh, and none of it enters any optimizer's
timed loop — `contract.timing` records `observer_inside_common_evaluator: false`, and
timing replays exclude common evaluator calls by construction, so this cost cannot
contaminate the iteration-efficiency measurement it is meant to support.

**Cost is therefore not a discriminator between C and D.** It should not be used as one.
The single practical recommendation is to install CHOLMOD before any full campaign sweep,
which would cut the dominant factorisation term substantially.
