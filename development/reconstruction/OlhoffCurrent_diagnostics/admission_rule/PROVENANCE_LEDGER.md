# PROVENANCE_LEDGER — convergence admission rule

What the sources actually support about stopping, design-change norms and move
limits — separated from our reconstruction, and from the **new policy design**
this study introduces.

Verified against the local PDFs, not inherited from earlier reports:

* `references/Du2007_Topological.pdf` (Du & Olhoff 2007)
* `references/Olhoff and Du - 2014 - Structural Topology Optimization with Respect to E.pdf`
* `docs/Bendsøe and Sigmund - 2004 - Topology Optimization.pdf`
* `analysis/OlhoffCurrent/+impl/filter/top88_reference.m` (Andreassen et al. 2011),
  which ships **inside the production tree**

---

## 1. SOURCE-SUPPORTED FACTS

### Du & Olhoff (2007) — what it says about convergence

Extracted verbatim from the local PDF (§3.5.1):

> "a check for convergence of ρ_e is performed by investigating whether **the
> norm of the vector Δρ = (Δρ_1, …, Δρ_NE) of design increments is less than a
> small, predefined value ε**. If the design variables ρ_e have converged, the
> optimum topology design has been obtained — otherwise, the updated …"

This establishes, and *only* establishes:

| Fact | Class |
|---|---|
| convergence is judged on the **design increment vector Δρ** | **A** |
| the test is **a norm of Δρ below a threshold** | **A** |
| *which* norm | **not stated** — "the norm", unqualified |
| the value of ε | **not stated** — "a small, predefined value" |
| N_E | **never reported**, in any example |

### Du & Olhoff (2007) — what it says about the filter

> "the mesh-independent filter developed by Sigmund (1997) … has been **applied
> to the sensitivities** of the objective functions in the computational models
> in the paper."

### Du & Olhoff (2007) — what is ABSENT

Word counts over the extracted text of the local PDF, independently recomputed
for this study:

| Term | Occurrences |
|---|---|
| `move limit` | **0** |
| `trust region` | **0** |
| `step size` | **0** |
| `continuation` | **0** |
| `projection` | **0** |
| `Heaviside` | **0** |
| `density filter` | **0** |

**The paper contains no move limit at all.** The only bound it places on Δρ is
the box (25f). Every question this study is about — when a step bound may be
reduced, and whether a reduction may license convergence — is therefore
*outside* the paper.

### Olhoff & Du (2014)

Closes none of these gaps.

### Bendsøe & Sigmund (2004) — the design-change convergence criterion

The 99-line code (Sigmund 2001), reproduced in the book, uses:

```matlab
 6  change = 1.;
 8  while change > 0.01
30  change = max(max(abs(x-xold)));
```

i.e. **a max-norm of the design change with threshold 0.01**, and the book
footnotes it:

> "This is a rather 'sloppy' convergence criterion and **could be decreased if
> needed**."

The same criterion appears in `top88_reference.m` (Andreassen et al. 2011)
inside the production tree:

```matlab
83  change = max(abs(xnew(:)-x(:)));
53  while change > 0.01
```

| Fact | Class |
|---|---|
| max-norm of the design change is a standard convergence measure | **A** (textbook) |
| the value **0.01** for it | **A** (textbook; present twice in this repository) |
| tightening it is sanctioned | **A** (the footnote says so explicitly) |

### Bendsøe & Sigmund (2004) — move limits

Move limits are discussed as an *algorithmic* device: "a move limit. Both η and
ζ control the changes that can happen at each iteration step and **they can be
made adjustable for efficiency of the method**." — an efficiency device, with no
convergence semantics attached, and no statement that reducing one licenses
termination.

---

## 2. RECONSTRUCTION CHOICES (ours, not the sources')

| Ingredient | Value | Class |
|---|---|---|
| L2 as the norm in (P4) | `stop.norm = l2` | **B** (natural reading of "the norm") |
| ε = 0.05 at 160×20 | — | **C** |
| **ε mesh-scaling law** `0.05·√(N_E/3200)` | `meshScaled` | **C** |
| the move ladder itself | `[0.04 0.02 0.01 0.005]` | **C** (its *existence* is B via Krog & Olhoff; its form and every number are C) |
| stall signal / window 10 / tolerance 5e-3 | — | **C** |
| **`settledMove` guard** | true | **C** — exists only to suppress a symptom the ladder itself creates |

## 3. NEW POLICY DESIGN (this study — neither source-supported nor historical)

The admission rule proposed here is **new engineering**, and is labelled as such.
It is *compatible* with (P4) — it still judges the design increment — but the
combination, the dwell requirement and the dimensionless ratio are ours.

**Nothing in this study may be described as "the Du–Olhoff stopping criterion".**
The paper specifies a norm of Δρ below an unstated ε and nothing else.

### Where each new constant comes from — none is topology-tuned

| Constant | Value | Origin |
|---|---|---|
| local absolute threshold `τ_abs` | **0.01** | Bendsøe & Sigmund 99-line / `top88_reference.m` in this tree. Literature. |
| local **dimensionless** threshold `τ_rel` | **0.5** | The ladder's own halving factor: at `max|Δρ|/move < 0.5` the realized step would fit inside the **next** ladder level, so the next descent could not further restrict it. Derived from the move schedule, not chosen for output. |
| move-settling dwell `D` | **10** | `move.continuation.window` — the solver's own existing judgement of the timescale on which a response to a move level can be assessed. |
| objective window | **10** | same field, kept consistent |
| objective tolerance `τ_obj` | **5e-3** | `move.continuation.tolerance` — the solver's own existing definition of "relative progress has stalled", applied to the objective instead of the bound variable. |

### A trap that was checked and avoided

`stop.norm = 'max'` already exists in the schema. **It is not a usable max-norm
criterion**: `olhoffSolve` line 402 compares `max|Δρ|` against the *same*
`tolOuter`, which is the L2-magnitude, mesh-scaled threshold (0.05 at 160×20,
0.10 at 320×40). Since `max|Δρ| ≤ move ≤ 0.04` always, that test would be
satisfied at **every** iteration from the first. Turning on the existing flag
would not implement this study's rule; it would disable stopping control
entirely. The candidate therefore uses explicit new semantic quantities and does
not reuse `tolOuter`.

### A second trap: an absolute threshold alone is still manufacturable

`max|Δρ| ≤ move` always. With the ladder able to descend to 0.005, **any**
absolute threshold τ_abs ≥ 0.005 becomes automatically satisfied once the ladder
reaches its finest level — reproducing exactly the defect under study.

This is why the candidate **requires the dimensionless ratio as well**. The
ratio `max|Δρ|/move` is invariant under a move-limit change by construction: if
the design is still pressing against its bound the ratio stays near 1 whatever
the bound is, and a descent cannot reduce it. That property, not the absolute
value, is what enforces the central invariant.
