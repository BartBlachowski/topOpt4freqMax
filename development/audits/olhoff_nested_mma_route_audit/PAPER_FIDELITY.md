# WP13 — Paper fidelity of the nested-MMA route
NESTED-MMA ROUTE AUDIT

Classification scheme:

    PAPER-LITERAL      stated explicitly by Du & Olhoff (2007)
    PAPER-SANCTIONED   offered by the paper as a permissible alternative
    INFERRED           not recoverable from the paper; chosen by reconstruction
    TUNED              selected because it reproduces the benchmark

| element | nested MMA | Eq. (22) LP | classification |
|---|---|---|---|
| use of MMA as the sub-optimizer | yes | no | **PAPER-LITERAL for MMA.** Section 3.5.3 states MMA was used. This is the single fidelity argument in MMA's favour and it is real. |
| full off-diagonal coupling, Eq. (25d) | yes (`offDiag=1`, `deltaLambda`) | no — Eq. (22) forces `f_sk' drho = 0` | **PAPER-LITERAL for MMA; PAPER-SANCTIONED for LP.** The paper prints Eq. (22) and attributes the route to Krog & Olhoff (1999). Both are in the paper. |
| erratum form of the subeigenvalue problem | yes | n/a | PAPER-LITERAL (with erratum) |
| filter radius | 1.2 elements | 1.2 elements (matched run) | **INFERRED + TUNED.** The paper states a sensitivity filter was used and gives no radius, and no statement of whether it is fixed in element or physical units. Identified by sweep. |
| move limit | 0.01 fixed | 0.005 fixed | **INFERRED + TUNED.** The paper gives no move limit. Fig. 4's smooth history is incompatible with an unrestricted step, so *some* restriction is implied, but not this one. |
| inner convergence tolerance | `dx/max|xmma| < 0.01` | n/a (one LP solve) | **INFERRED.** The paper says only "Increments drho_e converged?" and gives no criterion. The source declares this a reconstruction. |
| inner iteration cap | 300 | n/a | **INFERRED.** Not in the paper. |
| minimum inner iterations | 5 | n/a | **INFERRED.** Not in the paper. |
| multiplicity tolerance | 5% relative frequency, no hysteresis | 5% relative frequency, no hysteresis | **INFERRED + TUNED.** The paper says only "very small tolerance". 5% is not small; it is chosen operationally. Neither route can claim fidelity here. |
| what is filtered | diagonal `f_jj` and `f_JJ` only; off-diagonal `f_sk` unfiltered | diagonal `f_jj` and `f_JJ` | **INFERRED.** The paper does not state the filtering semantics for the generalized tensor. For MMA this means the Eq. (25d) subeigenvalue problem mixes filtered and unfiltered entries — a genuine internal inconsistency of the reconstruction. |
| MMA asymptote handling | fresh per inner solve | n/a | **INFERRED.** Not in the paper. |
| `beta` scaling by `lam_ref` | yes | yes | **INFERRED.** Numerical conditioning choice. |
| outer stopping | `max|drho| < 1e-3` | same | **INFERRED.** Not in the paper. |
| number of eigenpairs `Jcalc = n + Nmax` | 5 | 5 | PAPER-SANCTIONED (`J = n + N` is stated; `Nmax` is not) |

## What the fidelity comparison actually shows

**Nested MMA is more paper-literal than the LP route on exactly two counts** — the choice of
MMA as the sub-optimizer, and the retention of the full Eq. (25d) coupling. Those are not
trivial: they are the two places where the paper is explicit and the LP route deliberately
departs.

**On every other undocumented control the two routes are equally inferred**, and both carry
the same tuned filter radius and the same operationally-chosen 5% multiplicity tolerance.
Nine of the thirteen rows above are INFERRED for both.

**Neither route establishes historical fidelity to the 2007 code.** The paper omits the
filter radius, the mesh, the move limit, the stopping rules, the multiplicity tolerance, the
tensor-filtering semantics and the asymptote handling. BASE_mma demonstrates that *a* nested
MMA realisation can traverse the benchmark basin; it does not demonstrate that it is *the*
realisation the authors used. That distinction is the central lesson of the existing
post-mortem and it applies unchanged here.

**Is the LP route still paper-sanctioned?** Yes, unambiguously. Eq. (22) is printed in the
paper, and the final paragraph of section 3.5.3 attributes the linearising route to
Krog & Olhoff (1999). Adopting LP is a choice between two routes the paper itself offers, not
a departure from the paper.
