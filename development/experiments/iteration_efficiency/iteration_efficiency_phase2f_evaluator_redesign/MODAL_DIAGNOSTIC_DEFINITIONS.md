# WP2 — Modal diagnostic definitions
PHASE 2F — EVIDENCE GENERATION ONLY — NO METHODOLOGY AMENDMENT

Everything defined here is **diagnostic**. No threshold below is proposed, frozen, or
used by any frozen rule. Where a partition threshold is unavoidable it is **swept**, and
two threshold-free measures are carried alongside so that conclusions can be checked
without any cutoff at all.

## 1. What is being partitioned

For a density field `x` the evaluator forms, per element `e`, a stiffness `E_e` and a mass
`m_e` through the model's own interpolation (`modal_engine.interp`). The **effective
density** `z_eff` is the quantity the mass law is actually evaluated at:

| model | `z_eff` | reason |
|---|---|---|
| E1 | `x` | no clamp |
| E2 | `x` | no clamp |
| E3 | `max(x, 1e-3)` | frozen E3 clamps before both the stiffness cube and the mass law |

Using `z_eff` rather than `x` matters for E3 only, and keeps the partition consistent with
what the evaluator itself sees.

## 2. Modal energy densities

For eigenpair `(λ_j, φ_j)` of `K φ = λ M φ` on the free DOFs, expanded to full DOFs as
`u_j`, the element-wise kinetic and strain energies are

    ke_{e,j} = m_e · u_{e,j}^T ME u_{e,j}          (ME = the unit Q4 consistent mass matrix)
    se_{e,j} = E_e · u_{e,j}^T KE u_{e,j}          (KE = the unit Q4 stiffness matrix)

Both are non-negative by construction (ME and KE are positive semi-definite, `m_e, E_e > 0`);
they are clipped at zero only to absorb round-off. Each is normalised by its modal total:

    ke_n(e,j) = ke_{e,j} / Σ_e ke_{e,j}            Σ_e ke_n(e,j) = 1
    se_n(e,j) = se_{e,j} / Σ_e se_{e,j}

`ke_n` is the modal kinetic-energy distribution — the natural measure here, because the
pathology under investigation is mass placed in weak material.

## 3. Threshold-based localisation (swept, never frozen)

For a diagnostic partition threshold `τ`:

    voidKE(j; τ) = Σ_{e : z_eff(e) ≤ τ} ke_n(e,j)      ∈ [0,1]
    voidSE(j; τ) = Σ_{e : z_eff(e) ≤ τ} se_n(e,j)      ∈ [0,1]
    solidKE(j; τ) = 1 − voidKE(j; τ)

τ is swept over

    τ ∈ {0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}

`τ = 0.1` is included because it is where the Du–Olhoff mass branch lives, **not** because
it is privileged. Sensitivity across the whole sweep is reported in
`STRUCTURAL_MODE_THRESHOLD_SWEEP.csv` and `MODAL_LOCALIZATION_DISTRIBUTIONS.csv`.

### A degenerate region of the sweep, and why it must be excluded

**Small τ produces a vacuous plateau, and this audit initially mistook it for robustness.**

The artificial modes are localised at ρ ≈ 0.05–0.10 — their density-weighted participation
is 0.04–0.09, not near the 1e-3 floor. A partition at `τ = 0.01` or `0.02` therefore
*excludes the very elements carrying them*: their `voidKE(τ)` collapses to ~0, every mode
is classified structural, the selection degenerates to ordinal 1 at every state, and any cut
from 1e-4 to 0.95 gives the same (wrong) answer. Measured as a "plateau" this looks like
**3.98 decades of threshold invariance**; it is nothing of the kind.

`STRUCTURAL_MODE_THRESHOLD_PLATEAUS.csv` therefore carries a `degenerate` column, set when
the selection is identically ordinal 1 across the plateau. **42 of 238 plateaus are
degenerate, and all of them lie at τ = 0.01 or τ = 0.02.** They are excluded from every
robustness statement in this phase.

The lesson generalises: a threshold-invariance claim must be accompanied by evidence that
the criterion is still *discriminating* over the invariant range. Invariance achieved by
accepting everything is not robustness.

## 4. Threshold-free measures

Two measures require no partition at all. They exist so that every threshold-based
conclusion can be corroborated without a cutoff.

**Density-weighted modal participation.** The kinetic-energy-weighted mean effective
density a mode rides on:

    P(j) = Σ_e ke_n(e,j) · z_eff(e)          ∈ [min z_eff, max z_eff]

A mode carried by solid material scores near 1; a mode trapped in void scores near the void
density. Continuous in the density field, no cutoff, and directly interpretable as
"what material is this mode actually moving?".

**Inverse participation ratio.** Spatial concentration of modal kinetic energy:

    IPR(j) = Σ_e ke_n(e,j)^2          ∈ [1/N_el, 1]

`1/N_el` for a mode spread uniformly over the mesh, `1` for a mode confined to one element.
It measures localisation without reference to density at all, so it is independent evidence
that a mode is a local artefact rather than a global structural mode.

## 5. Worked values at the decisive state (160x20, k = 252)

Reproduced in `K252_MODAL_REPRODUCTION.csv`. E2 under Eq. (4a):

| mode | ω | voidKE(τ=0.1) | voidSE(τ=0.1) | P (density-weighted) | IPR |
|---|---|---|---|---|---|
| 1 | 31.404 | 1.000000 | 0.999997 | 0.080 | 0.654 |
| 2 | 104.993 | 0.999990 | 0.999902 | 0.080 | 0.502 |
| 3 | 107.724 | 0.999985 | 0.999956 | 0.086 | 0.552 |
| **4** | **166.367** | **0.001522** | **0.000120** | **0.926** | **0.000947** |
| 5 | 166.505 | 0.001614 | 0.000631 | 0.918 | 0.000909 |
| 6 | 218.146 | 0.999992 | 0.999994 | 0.050 | 0.986 |
| 8 | 311.705 | 0.001008 | 0.002342 | 0.942 | 0.003334 |

All three measures agree on the same partition of the spectrum, and each separates the two
populations by two to three orders of magnitude. That agreement — between a threshold-based
measure and two threshold-free ones — is the reason the classification can be called
physical rather than conventional.

## 6. What these diagnostics are not

- They are **not** part of any frozen rule, and computing them changes no frozen output.
- They are **not** a modal-validity criterion. Whether a defensible criterion exists, and
  over how wide a threshold interval it would give identical decisions, is the question
  `CANDIDATE_C_STRUCTURAL_MODE_ANALYSIS.md` investigates — it is not assumed here.
- The word "artificial" is used below only for modes independently shown to carry
  essentially all of their kinetic **and** strain energy in weak material while being
  spatially concentrated. Where a mode does not clearly satisfy that, it is reported as
  ambiguous rather than classified.
