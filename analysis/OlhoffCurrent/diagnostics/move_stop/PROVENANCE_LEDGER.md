# PROVENANCE_LEDGER — move/stopping diagnostic

Which ingredients of the canonical production realization are **published** and
which are **our reconstruction**, classified *before* any diagnostic run.

Sources: Du & Olhoff (2007); Olhoff & Du (2014); Krog & Olhoff (1999) / CISM;
Bendsøe & Sigmund, *Topology Optimization* (2003); Sigmund (1997).
Repository evidence:
`analysis/OlhoffCurrent/+impl/architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md`
(field-level, with the quoted sentences), and the schema's own class column.

**Class key.** **A** explicitly specified by a Du–Olhoff source · **B** directly
implied by one · **C** under-specified reconstruction choice · **D** later
experimental modification.

> **Rule applied throughout:** an ingredient is *not* Du–Olhoff merely because it
> exists in OlhoffCurrent. Where the papers are silent, the row says so.

---

## The four sentences carrying most of the class-A weight

> **(P1)** "The power p in (1) … is normally assigned values increasing from 1 to 3
> during the optimization process." — §2.1

> **(P2)** "the mesh-independent filter developed by Sigmund (1997) … has been
> **applied to the sensitivities** of the objective functions in the computational
> models in the paper." — §1

> **(P3)** "the term 'multiplicity' is used if the numerical value of the relative
> difference between eigenfrequencies in question is within a predefined, very
> small tolerance." — §3.5.1

> **(P4)** "a check for convergence of ρ_e is performed by investigating whether
> **the norm of the vector Δρ** = (Δρ_1, …, Δρ_NE) of design increments is less
> than a small, predefined value ε." — §3.5.1

Word counts over the extracted text of Du & Olhoff (2007): `move limit` **0**,
`trust region` **0**, `step size` **0**, `continuation` **0**, `projection` **0**,
`Heaviside` **0**, `density filter` **0**. Over Olhoff & Du (2014): `filter` **0**,
`tolerance` **0**, `move limit` **0** — **the 2014 paper closes none of the 2007
gaps.**

---

## Ledger

### PUBLISHED / SOURCE-SUPPORTED (A, B)

| Ingredient | Value in production | Class | Basis |
|---|---|---|---|
| **Sensitivity filtering** — that a filter exists, is Sigmund (1997), mesh-independent | `filter.type = sensitivity` | **A** | (P2) |
| **Sensitivity filtering — applied to the sensitivities, not densities** | `sensitivity` | **A** | (P2), explicitly |
| **Mass model eq. (4b)** — the C¹ law and its printed constants `c₁=6×10⁵`, `c₂=−5×10⁶` | `eq4b` | **A** | Eq. (4b) |
| **q = 1** | 1 | **A** | Eq. (2), "normally, q=1 is chosen" |
| Mass cut-off ρ ≤ 0.1 | 0.1 | **A** | printed in (4), (4a), (4b) |
| `ρ_min = 1e-3` | 1e-3 | **A** | Eq. (7e) |
| **p is the penalization power, p ≈ 3** | 3 | **A** (value) | §2.2 states p "is kept unchanged at a value about p=3" |
| **Multiplicity measure** = relative frequency difference | — | **A** | (P3) |
| Multiplicity re-detected every outer iteration | yes | **A** | §3.5.1, Fig. 1 |
| Off-diagonal terms retained in (25d) | true | **A** | (25d) is a determinant over s,k; the erratum form is mandatory |
| λ̃ = ω_n² | first eigenvalue of cluster | **A** | §3.5.1 verbatim |
| **The convergence *measure*** — "the norm of the vector Δρ of design increments < ε" | `stop.field = designVariable` | **A** | (P4) |
| **MMA** as the optimizer | `mma` | **A** | named in the paper |
| MMA variant = Svanberg's *published* Sept-2007 constants | `published` | **B** | Svanberg's own published constants; `asfound` is a local lineage modification |
| **Which norm** = L2 | `l2` | **B** | (P4) writes "the norm" unqualified; the Euclidean norm is the natural reading |
| **That *some* bound on Δρ exists** | — | **B** | Justification via Krog & Olhoff (CISM Eq. 103): the multiple-eigenvalue model is a first-order directional expansion valid only for increments of restricted magnitude |

### RECONSTRUCTION CHOICES (C) — **the machinery under test**

| Ingredient | Value in production | Class | Basis / what the paper actually says |
|---|---|---|---|
| **Holding p fixed at 3** (rather than 1→3) | fixed | **C** | *Contradicts (P1).* Fixed because the reported initial eigenfrequencies (68.7/104.1/146.1) fit p=3 and p=1 is off by ≈2× |
| **Choosing eq4b** among (4)/(4a)/(4b) | `eq4b` | **C** | §2.2 says all three were applied and gave "only negligible differences"; the paper expresses no preference |
| **Filter radius R = 0.06·b** | 0.06 | **C** | **Never stated, for any example, in either paper.** The single strongest determinant of member thickness |
| Radius expressed *physically* (`rminEl = R/(b/nely)`) | physical | **C** | A reconstruction policy for mesh-independence. Sound, not the authors' |
| Filter scope = every `f_sk` | `all` | **C** | The paper has one sensitivity vector; the multiple case has N(N+1)/2 and the paper never says which are filtered |
| Multiplicity **tolerance value** 0.05 | 0.05 | **C** | (P3) says "predefined, very small" and never gives a number |
| **Fixed subspace, no classifier**, `subN = 2` | `subspace`, 2 | **C** | An interpolation assembled from the paper's own (19),(24),(25c),(25d) — but the interpolation itself is written nowhere |
| Diagonal offsets `diag(λ_j−λ_n)` in (25d) | true | **C** | (25d) as printed assumes exact degeneracy; retaining the actual separation is reconstruction |
| Inner convergence criterion, `tolInner = 0.05` | 0.05 | **C** | Fig. 1 says only "Increments Δρ_e converged?" |
| **THE MOVE-LIMIT LADDER — its form** | `policy = ladder` | **C** | **"move limit", "trust region", "step size" occur 0 times in either paper.** Only the box (25f) bounds Δρ |
| **Ladder levels `[0.04 0.02 0.01 0.005]`** | — | **C** | **No numeric value exists anywhere in the lineage** |
| Initial move 0.04, `moveMin` 0.002 | — | **C** | — |
| **Bound-variable (β) stall trigger** | `boundVariable` | **C** | The paper has no continuation of any kind |
| **Stall window W = 10** | 10 | **C** | — |
| **Stall tolerance 5e-3** | 5e-3 | **C** | — |
| Stall detector *shape* (mean-of-window) | — | **C** | Chosen because a max−min test cannot fire while the signal oscillates |
| **ε mesh-scaling law** `ε = 0.05·√(NE/3200)` | `meshScaled` | **C** | A reconstruction policy so ε means the same RMS density change at every mesh. **Not the authors'.** The paper's ε is "a small, predefined value" and NE is never reported |
| ε value 0.05 at 160×20 | 0.05 | **C** | The paper gives no number |
| **`settledMove` guard** | true | **C** | **Exists only because of the reconstruction's own move ladder**, which the paper does not have. Under a ladder, an iteration that lowers the move limit mechanically lowers ‖Δρ‖, so (P4) is uninterpretable there |
| Iteration cap 400 | 400 | **C** | — |
| Mesh 160×20 / 320×40 | — | **C** | NE is never reported anywhere in the paper |

### LATER EXPERIMENTAL MODIFICATIONS (D) — **all OFF in production, and stay off**

| Ingredient | State | Class |
|---|---|---|
| Heaviside projection (`enabled`, `eta`, `beta.levels`, continuation) | **off** | **D** |
| Density filter | **off** (`filter.type = sensitivity`) | **D** |
| p continuation | **off** | **D** (as *implemented*; note (P1) makes a schedule closer to the paper's stated practice) |
| Mass-model continuation | **off** | **D** |
| `designRms` stall signal | not used | **D** |
| Stop guards `ladderExhausted`, `maxDesignChange` | **off** | **D** |

---

## What this ledger establishes for the diagnostic

The three ingredients whose interaction this diagnostic tests —

1. the **move-limit ladder** and its descent events,
2. the **L2 convergence measure evaluated on Δρ**, and
3. the **`settledMove` admission guard**

— are **class B/C at best and class C in every numeric and structural detail**:

* the *existence* of a step bound is **B** (Krog & Olhoff's validity argument);
  its **ladder form, its four levels, its trigger, its window and its tolerance
  are all C**, with *no numeric value anywhere in the lineage*;
* the convergence **measure** is **A** (P4), but the **threshold value**, the
  **mesh-scaling law** and the **`settledMove` guard** are **C**;
* `settledMove` is the sharpest case: it exists *only* to suppress a symptom
  that the ladder itself creates, and the ladder is not in the paper.

**Consequently, if the diagnostic finds that grayness is driven by the
ladder/L2/settledMove interaction, that is a finding about our reconstruction —
not about the Du–Olhoff formulation.** Conversely, the sensitivity filter,
p ≈ 3, eq4b, q = 1, the multiplicity measure and MMA are all published, so a
grayness floor attributable to *those* would be a property of the printed
method as reconstructed at an unstated filter radius.

**Caution recorded in advance:** the filter *radius* (C) is described in the
repository evidence as "the single strongest determinant of member thickness",
and it is unstated in both papers. Any residual grayness floor is therefore
confounded with the radius choice and **must not** be attributed to the printed
formulation without separating that factor — which this diagnostic does **not**
do.
