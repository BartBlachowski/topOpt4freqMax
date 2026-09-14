# SCIENTIFIC_CONFIG_PROVENANCE

Field-level provenance for every configurable quantity in the Du–Olhoff
eigenfrequency solver.

## Classes

| Class | Meaning |
|---|---|
| **A** | Explicitly specified by a Du–Olhoff source. Quoted below. |
| **B** | Directly implied by, or reconstructable from, a source — including the authors' own lineage (Svanberg 1987, Sigmund 1997, Krog & Olhoff 1999, Seyranian/Lund/Olhoff 1994). |
| **C** | Under-specified reconstruction choice. The source poses the question and does not answer it. |
| **D** | Later experimental modification or new method. Absent from every source. |

A field may carry two classes — typically **A** for the *measure* and **C** for
its *value*. Those are listed as `A/C` and the split is stated.

## Textual basis

Verified by extraction from the PDFs in `docs/` at baseline `2029baa`:

* **Du & Olhoff (2007)**, SMO 34:91–110, plus the Publisher's Erratum (SMO 34:545).
* **Olhoff & Du (2014)**, *Structural Topology Optimization with Respect to
  Eigenfrequencies*. Word counts over the extracted text: `filter` **0**,
  `tolerance` **0**, `move limit` **0**. **The 2014 paper closes none of the 2007
  gaps.** It does print eq. (19d) already corrected, double-sourcing the erratum.
* Du & Olhoff (2007) word counts: `move limit` **0**, `trust region` **0**,
  `step size` **0**, `continuation` **0**, `projection` **0**, `Heaviside` **0**,
  `density filter` **0**, `filter` **1** (the sensitivity-filter sentence quoted below).

The four sentences that carry most of the class-A weight:

> **(P1)** "The power p in (1) … is normally assigned values increasing from 1 to 3
> during the optimization process." §2.1

> **(P2)** "the mesh-independent filter developed by Sigmund (1997) … has been
> applied to the sensitivities of the objective functions in the computational
> models in the paper." §1

> **(P3)** "the term 'multiplicity' is used if the numerical value of the relative
> difference between eigenfrequencies in question is within a predefined, very
> small tolerance." §3.5.1

> **(P4)** "a check for convergence of ρ_e is performed by investigating whether the
> norm of the vector Δρ = (Δρ_1, …, Δρ_NE) of design increments is less than a
> small, predefined value ε." §3.5.1

---

## 1. Material — stiffness

| Field | Value(s) used | Class | Basis |
|---|---|---|---|
| stiffness law `ρ^p Kₑ` | fixed | **A** | Eq. (1), Eq. (3). |
| `p` | 3 | **A/C** | **A**: p is the penalization power of Eq. (1), p≥1, and §2.2 states p "is kept unchanged at a value about p=3". **C**: *holding it fixed* contradicts (P1); the reconstruction fixed p=3 because the reported initial eigenfrequencies (68.7/104.1/146.1) are only consistent with p=3, and p=1 is off by ≈2×. Recorded in `CLAUDE.md` §2. |
| p continuation, existence | `[1 2 3]` | **A** | (P1) states increasing p from 1 to 3 is the *normal* practice. Running p-continuation is therefore closer to the paper's stated practice than fixed p=3 — a point the audit programme reached independently. |
| p schedule *values* | `[1 2 3]` | **B** | The endpoints 1 and 3 are printed; the integer step is the obvious reading. No schedule is given. |
| p transition *rule* | S2 stall event | **C** | The paper gives no trigger. Reusing the existing ladder stall event was chosen to avoid introducing a new constant. |
| p continuation *driver* | ladder stage (P1) / own counter (PD1) | **C** | Both are reconstruction. See `ARCHITECTURE_VARIANT_INVENTORY.md` §2.1. |
| stop blocked while `p < p_end` | on | **B** | Forced by the definition of the experiment: a p=3 problem cannot be declared converged at p<3. Not a tuning choice. |

## 2. Material — mass

| Field | Value(s) | Class | Basis |
|---|---|---|---|
| mass model `eq2` | `ρ^q Mₑ` | **A** | Eq. (2). |
| mass model `eq4` | `ρ`, `ρ^r` for ρ≤0.1 | **A** | Eq. (4), after Tcherniak (2002). |
| mass model `eq4a` | `c₀ρ⁶`, C⁰ | **A** | Eq. (4a), `c₀=10⁵` printed. |
| mass model `eq4b` | `c₁ρ⁶+c₂ρ⁷`, C¹ | **A** | Eq. (4b), `c₁=6×10⁵`, `c₂=−5×10⁶` printed. |
| *which* model | `4b` frozen | **C** | §2.2 says all three were applied and gave "only negligible differences". Choosing `4b` for the frozen realization is a reconstruction choice; the paper expresses no preference. |
| `q` | 1 | **A** | Eq. (2), "normally, q=1 is chosen". |
| `r` | 6 | **A** | §2.2, "r is chosen to be about r=6". Note *about*: the value is soft in the source, exact in the code. |
| cutoff density | 0.1 | **A** | Printed in (4), (4a), (4b) as ρₑ ≤ 0.1. |
| `c₀`,`c₁`,`c₂` | 1e5, 6e5, −5e6 | **A** | Printed; and algebraically verified to give C⁰/C¹ at ρ=0.1. |
| mass-model continuation (`massLowP`) | `lin` while p<p_end | **D** | No source switches mass model during a run. The *models* are class A; *scheduling between them* is not. |

## 3. Filtering

| Field | Value(s) | Class | Basis |
|---|---|---|---|
| filter exists, mesh-independent, Sigmund (1997) | yes | **A** | (P2). |
| filter applied to **sensitivities** | yes | **A** | (P2), explicitly "applied to the sensitivities". |
| top88 `ft=1` algebraic form | `H(ρ·df)/(Hs·max(1e-3,ρ))` | **B** | Sigmund (1997) as published in Andreassen et al. (2011); the `max(1e-3,ρ)` guard is published behaviour and is retained deliberately. |
| filter radius | R=0.06 physical | **C** | Never stated for any example, in either paper. The single strongest determinant of member thickness. |
| radius expressed **physically** | `rminEl = R/(b/nely)` | **C** | A reconstruction *policy* — mesh-independence of the filter under refinement. Sound, but not the authors'. |
| application scope (`diag`/`all`/`none`) | `all` frozen | **C** | The paper has one sensitivity vector; the multiple case has N(N+1)/2 vectors f_sk and the paper never says which are filtered. Named as an open question in `CLAUDE.md` §5. |
| **density** filter | used under projection | **D** | Not published. (P2) says sensitivities were filtered. Substituting a density filter is a departure from a printed choice. |

## 4. Projection

| Field | Value(s) | Class | Basis |
|---|---|---|---|
| projection enabled | off frozen / on in T-runs | **D** | `projection`, `Heaviside`: **0** occurrences in either paper. Absent from the Krog & Olhoff lineage. |
| tanh Heaviside operator | — | **D** | Post-2007 literature (Wang, Lazarov & Sigmund 2011). Outside the reproduction's own source rule. |
| `β` schedule `[1 2 4 8]` | — | **D** | — |
| `η` | 0.5 | **D** | — |
| continuation trigger = outer convergence | — | **D** | — |
| affine floor `ρmin+(1−ρmin)P` | — | **D** | Chosen to reproduce the frozen density range [ρmin,1] smoothly. |

## 5. Multiplicity

| Field | Value(s) | Class | Basis |
|---|---|---|---|
| multiplicity is re-detected in step 1 of every outer iteration | yes | **A** | §3.5.1, Fig. 1. |
| measure = relative frequency difference | yes | **A** | (P3). |
| tolerance **value** | 0.05 frozen | **C** | (P3) says "predefined, very small" and never gives a number. Krog & Olhoff §5.3 report 1e-4 for their own examples — class **B** if adopted, but it is *not* what the frozen realization uses, and their detector does not gate the ascent. |
| detector shape (diameter vs chain test) | diameter from ω_n | **C** | The paper does not say which. |
| `latch` / `hyst` persistence | not used | **C** | Persistence and hysteresis are mentioned in **no** source. |
| `subspace` (fixed N=`subN`, no classifier) | frozen | **C** | An *interpolation* assembled only from the paper's own (19), (24), (25c), (25d) — but the interpolation itself is not written anywhere. |
| `subN` = 2 | frozen | **C** | — |
| diagonal offsets `diag(λ_j−λ_n)` in (25d) | on with `subspace` | **C** | (25d) as printed assumes exact degeneracy (p.96, eq. 17). Retaining the actual separation is reconstruction — sound, verified against finite differences, but not printed. |
| off-diagonal terms retained | yes | **A** | Eq. (25d) is a determinant over s,k; the erratum form is mandatory. Dropping them is the *alternative* (see below). |
| forcing `f_sk'Δρ = 0`, s≠k | not used | **B** | Eq. (22) / Krog & Olhoff (1999). Presented in the final paragraph of §3.5.3 as something one *may* additionally do — grammatically an option, not a report of what was run. |
| λ̃ = ω_n² | first eigenvalue of cluster | **A** | §3.5.1 verbatim: "we set λ̃ = ω_n²". Using the cluster *mean* would be the reconstruction; the code follows the text. |
| `Nmax` / J computed | 4 | **C** | The paper computes J = n+N; how many modes to extract in advance is unstated. |
| Δλ_j gradient `Σ_sk v_js v_jk (f_sk)_e` | — | **C** | The paper never states how the constraint gradients are formed. Flagged as *the* central algorithmic gap in `CLAUDE.md` §5. |

## 6. Inner optimizer

| Field | Value(s) | Class | Basis |
|---|---|---|---|
| MMA used | yes | **A** | §3.5.3, "the MMA method (Svanberg 1987) has been used". |
| `mmasub`, not GCMMA | yes | **B** | Svanberg 1987 is the 1987 method; GCMMA is the 1995/2002 variant and would change iteration counts. |
| MMA variant (`published` Sept-2007 constants) | `published` | **B** | Svanberg's own published constants; `asfound` is a local lineage modification (move 1.0, asyinit 0.01). |
| MMA hyper-constants `a₀=1,a=0,c=1000,d=0` | — | **C** | Not stated by Du & Olhoff; conventional MMA usage. |
| independent variables = β and Δρ_e | yes | **A** | §3.5.2 explicitly; Δ(ω_j²) are *dependent*. |
| `innerVar='drho'` (MMA state reset per outer iteration) | frozen | **C** | The paper's inner loop is per-outer-iteration; whether MMA asymptote history persists is unstated. |
| inner convergence criterion | `max|dx|/max|Δρ| < tolInner` | **C** | Fig. 1 says only "Increments Δρ_e converged?". The relative form was chosen because an absolute test degenerates as the move limit shrinks. |
| `tolInner`, `minInner`, `maxInner` | 0.05, 5, 500 | **C** | — |
| β scaled by λ_n; `xmax(β)=5` | — | **C** | Numerical conditioning, not in the source. |

## 7. Move-limit policy

| Field | Value(s) | Class | Basis |
|---|---|---|---|
| existence of any move limit | S2 ladder | **B** | **The only bound printed on Δρ is the box (25f).** "move limit", "trust region", "step size" occur **0 times** in either paper. The *justification* is class B via Krog & Olhoff (CISM Eq. 103): the multiple-eigenvalue model is a first-order directional expansion valid only for increments of restricted magnitude. No numeric value exists anywhere in the lineage. |
| policy family (`S0…S3`) | `S2` frozen, `S0` no-descent | **C** | Every functional form is reconstruction. |
| `move` initial, `s2Levels`, `moveMin` | 0.04, `[0.04 0.02 0.01 0.005]`, 0.002 | **C** | — |
| stall window `W=10`, `tol=5e-3` | — | **C** | — |
| stall detector shape (mean-of-window vs mean-of-previous-window) | — | **C** | Chosen because a max−min test cannot fire while the signal oscillates. |
| re-arm dwell `(k − lastStage) > W` | — | **C** | — |
| stall **signal** `beta` | frozen | **C** | — |
| stall signal `drms` | tested, **not adopted** | **D** | `audit_s2_design_continuation` verdict `S2_LADDER_ITSELF_DEFECTIVE`; the drms trigger was **not** adopted. |
| S3 gain-ratio band | not run | **D** | Lineage-*derived*, not lineage-specified. |

## 8. Stopping

| Field | Value(s) | Class | Basis |
|---|---|---|---|
| criterion: norm of Δρ below ε | yes | **A** | (P4). |
| the monitored object is the **design increment Δρ** | yes | **A** | (P4) names Δρ = (Δρ_1,…,Δρ_NE). |
| which norm | `l2` | **B** | (P4) writes "the norm" unqualified; the Euclidean norm is the natural reading of the unqualified symbol. `max` is kept as a labelled alternative. |
| ε value | `0.05·√(NE/3200)` | **C** | Never given. |
| ε mesh-scaling law | — | **C** | A reconstruction policy so that ε means the same RMS density change at every mesh. Not the authors'. |
| guard `settledMove` | frozen | **C** | Necessary *because of* the reconstruction's own move ladder, which the paper does not have. Under a ladder, an iteration that lowers the move limit mechanically lowers ‖Δρ‖, so (P4) is uninterpretable there. |
| guard `ladderExhausted` (R1) | R1 run | **D** | Preregistered stopping safeguard, `audit_m4_topology_restoration`. |
| guard `maxDesignChange` (R2) | all mature runs | **D** | As above. `ε_RMS = ε/√NE`. |
| iteration cap `maxOuter` | 200/400/600/1200 | **C** | Safety cap; reaching it is a distinct status, not convergence. |

## 9. Domain, discretization, runtime

| Field | Value(s) | Class | Basis |
|---|---|---|---|
| `a=8, b=1, E=1e7, ν=0.3, ρ_m=1`, plane stress | — | **A** | §4. |
| `ρ_min = 1e-3` | — | **A** | Eq. (7e). |
| `ρ₀ = 0.5`, `volfrac = 0.5` | — | **A** | §4, uniform initial design, α = 50%. |
| mesh `nelx×nely` | 160×20 … 800×100 | **C** | NE is never reported anywhere in the paper. |
| element type `Q4`, consistent mass | — | **C** | "plane stress elements", nothing more. Confounded with support placement — a 5% effect either way. |
| support idealization `mid`, `axial='both'` | — | **C** | The paper *draws* corner supports; its *numbers* fit mid-height supports with axial restraint at both ends. The drawing and the numbers disagree and the numbers won. This is a reconstruction ruling and must not be presented as the authors' choice. |
| `bc='a'` simply supported | — | **A** | Fig. 2(a). |
| eigensolver `eigs` + fixed start vector | — | **C** | Not stated. The fixed start vector is *required* for determinism near the degeneracies this study is about; ARPACK's default start is random. |
| `eigs` tol 1e-12, maxit 5000, p=max(20,4J) | — | **C** | — |
| `threads = 1` | — | **C** | Required for meaningful complexity measurement, not a scientific choice. |
| `diag` | — | **C** | Diagnostics only; provably inert. |

---

## 10. Summary counts

| Class | Fields | Comment |
|---|---|---|
| **A** | 21 | Geometry, material, both interpolation families with all their printed constants, the bound formulation, MMA, the multiplicity *measure*, the convergence *measure*. |
| **B** | 8 | Norm choice, top88 filter form, MMA variant, LP route, existence of a move bound, p schedule endpoints, stop-block-while-p<p_end. |
| **C** | 38 | Everything numeric the paper leaves open, plus every move-control and stopping-guard form, plus the subspace/offset formulation and the constraint-gradient reconstruction. |
| **D** | 9 | Projection (5 fields), density filter, mass-model continuation, `drms` signal, S3 family. |

**The reconstruction is larger than the paper.** Two thirds of the fields that
determine a trajectory are class C or D. That is the strongest single argument
for the canonical-configuration work: without it, "what am I running?" cannot be
answered without reading audit reports.

## 11. Things that must not be blurred

1. **Fixed p=3 is a reconstruction ruling, not the paper's procedure.** (P1) says
   p normally increases from 1 to 3. The reconstruction fixed p=3 on numerical
   evidence. Both are defensible; only one is printed, and it is not the frozen one.
2. **The paper filtered sensitivities.** Any density-filter or projection run is a
   departure from a printed choice, whatever its merits.
3. **Mass models (4)/(4a)/(4b) are all published and all reported as used.**
   Selecting `4b` is ours. `massLowP` scheduling between them is ours.
4. **The move ladder has no source at all.** Its stopping guard `settledMove`
   exists only to repair a defect the ladder itself introduces.
5. **"M4" is not a mass model.** See `TERMINOLOGY.md`.
6. **Support placement contradicts the paper's own figure.** Recorded, not reconciled.
