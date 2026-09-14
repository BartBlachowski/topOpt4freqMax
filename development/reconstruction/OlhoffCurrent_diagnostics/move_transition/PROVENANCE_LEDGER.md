# PROVENANCE_LEDGER — the move ladder and its transition signal

What the sources actually support about **step bounds, their reduction, and
when a reduction may occur** — separated from our reconstruction and from the
new policy this study tests.

Verified **for this study**, by re-extracting the local PDFs, not inherited from
the earlier reports:

* `references/Du2007_Topological.pdf` (Du & Olhoff 2007)
* `references/Olhoff and Du - 2014 - Structural Topology Optimization with Respect to E.pdf`
* `docs/Bendsøe and Sigmund - 2004 - Topology Optimization.pdf`
* `analysis/OlhoffCurrent/+impl/architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md`
  and the schema's own class column (`+impl/architecture/+olh/+config/schema.m`)

**Class key.** **A** explicitly specified by a Du–Olhoff source · **B** directly
implied by one, or by the authors' own methodological lineage · **C**
under-specified reconstruction choice · **EXPERIMENTAL** new policy introduced
by a diagnostic study, default-off.

---

## 1. What the papers contain — recounted for this study

Occurrence counts over the freshly extracted text:

| term | Du & Olhoff 2007 | Olhoff & Du 2014 |
|---|---|---|
| `move limit` | **0** | **0** |
| `move-limit` | **0** | **0** |
| `trust region` | **0** | **0** |
| `step size` | **0** | **0** |
| `continuation` | **0** | **0** |
| `projection` | **0** | **0** |
| `Heaviside` | **0** | **0** |
| `density filter` | **0** | **0** |
| `stall` | **0** | — |
| `window` | **0** | — |
| `move` (any use at all) | **2** | **2** |

Both occurrences of the bare word `move` in each paper are ordinary English and
have nothing to do with step control:

> "Note that if in (16a–d), we **remove** the bound variable β₁ …" — Du & Olhoff §3
>
> "… the techniques enable us … to **move** structural resonance frequencies far
> away from external excitation frequencies …" — Du & Olhoff, conclusions

**There is no move limit in either paper.** The only bound placed on the design
increment in the printed formulation is the box (25f),
`0 < ρ_min ≤ ρ_e + Δρ_e ≤ 1`.

The one sentence that governs stopping is §3.5.1, verbatim from the local PDF:

> "… a check for convergence of ρ_e is performed by investigating whether **the
> norm of the vector Δρ = (Δρ₁, …, Δρ_NE) of design increments is less than a
> small, predefined value ε**."

It specifies a *measure*, not a threshold, not a norm, and says nothing whatever
about step bounds or their reduction.

## 2. What Bendsøe & Sigmund (2004) contain — and the caution it carries

Move limits do exist in the textbook, as an **algorithmic efficiency device**:

> "The variable η in (1.12) is a tuning parameter and ζ a **move limit**. Both η
> and ζ control the changes that can happen at each iteration step and **they can
> be made adjustable for efficiency of the method**."

and, discussing adaptive move-limit strategies (Zhou, Shyy & Thomas 2001), the
book records a caution that bears directly on this study:

> "… it is **unclear whether 'playing' with the move-limits will jeopardize
> convergence of the algorithm**."

So: the *existence* of a move limit and the *idea* of making it adjustable are
textbook. **Nothing in any source attaches convergence semantics to a move
reduction, and the one textbook remark on adaptive move limits is a warning, not
a licence.**

## 3. The classification the brief asks for (§3)

| Ingredient | Class | Basis |
|---|---|---|
| **existence of a move limit** | **B** | Not in Du & Olhoff. Justified through the authors' own lineage: Krog & Olhoff (CISM, Eq. 103) make the multiple-eigenvalue sensitivity model a *first-order directional* expansion `a + Δa = a + ε·e`, valid only for increments of restricted magnitude. Textbook-supported as a device (B&S above). **No numeric value exists anywhere in the lineage.** |
| **the values `[0.04 0.02 0.01 0.005]`** | **C** | `schema.m:100`, class C in the schema itself. No number for a step bound appears in Du & Olhoff (2007), Olhoff & Du (2014) or the Krog & Olhoff lineage. |
| **move continuation** (that the bound is reduced during the run at all) | **C** | `move.policy = ladder`, `schema.m:97`, class C. `continuation` occurs **0** times in either paper. |
| **the bound-variable stall trigger** | **C** | `move.continuation.signal = boundVariable`, `schema.m:107`, class C. The paper has no continuation of any kind, hence no trigger for one. |
| **window = 10** | **C** | `move.continuation.window`, `schema.m:108`, class C. `window` occurs **0** times. |
| **tolerance = 5e-3** | **C** | `move.continuation.tolerance`, `schema.m:109`, class C. |
| **the stall detector's *shape*** (mean-of-window vs mean-of-previous-window) | **C** | Chosen in `olh.move.limit` because a max−min test cannot fire while the signal oscillates. Reconstruction. |
| **`settledMove` admission guard** | **C** | Exists *only* because the ladder exists. Not in the paper. |
| **utilization-gated transition (`r_rho < 0.5` × 10)** | **EXPERIMENTAL** | Introduced by *this* study. Default-off. Its two constants are inherited from the preceding admission-rule preregistration and from `move.continuation.window`, but the **rule** is new engineering. |

### The one-line conclusion the brief asks for

> **The move ladder must not be called "Du–Olhoff".** Its existence is class B
> (lineage, not text); **its form, its four levels, its continuation, its
> trigger, its window and its tolerance are all class C**, and no numeric value
> for any of them exists anywhere in the lineage. Nothing in this study, and
> nothing in the production realization's step control, may be attributed to the
> printed method.

## 4. What this study's candidate is, epistemically

The utilization-gated transition is **new engineering**, and is labelled as such
throughout. It is *compatible* with the paper — it still judges the design
increment vector Δρ, which is the only quantity §3.5.1 monitors — but the ratio,
the threshold and the dwell are ours.

| Constant | Value | Origin |
|---|---|---|
| `r_rho` threshold | **0.5** | The ladder's own halving factor, adopted as `τ_rel` in the preceding admission-rule preregistration: below it the realized step already fits inside the *next* ladder level, so the next descent could not further restrict it. Derived from the move schedule, **not** from output. |
| persistence | **10** | `move.continuation.window` — the solver's own existing judgement of the timescale on which a response to a move level can be assessed. Reused, not invented. |

Neither is fitted in this task, under any outcome (preregistration §11).

## 5. Why the ratio, verified from the implementation

`+impl/algo/innerLoop.m:45-46` builds the sub-problem box as

```matlab
lo = max(ctx.rhomin - ctx.rho, -ctx.move);
hi = min(1          - ctx.rho,  ctx.move);
```

so `|Δρ_e| ≤ move` for every element, identically, and therefore

    r_rho = max_e|Δρ_e| / move  ∈  [0, 1] .

**The analytic property under test.** Let the sub-problem's optimum, ignoring
the move box, place the largest element increment at magnitude `s`. Then
`max|Δρ| = min(s, move)` and

    r_rho = min(s/move, 1).

* If the design is still **pressing against its bound** (`s ≥ move`), then
  `r_rho = 1` **for every value of move**. Halving the move halves numerator and
  denominator together and the statistic does not move. A descent therefore
  **cannot** manufacture a low `r_rho`.
* Only when the design's own preferred step becomes small relative to the bound
  (`s < move`) does `r_rho` fall below 1, and it then measures exactly how much
  of the permitted step the design is declining to use.

An **absolute** threshold on `max|Δρ|` has the opposite property: since
`max|Δρ| ≤ move`, any absolute threshold ≥ 0.005 becomes automatically satisfied
once the ladder reaches its finest level — which is the defect the preceding
admission-rule study measured directly.

This establishes only that `r_rho` is **invariant under a change of the move
bound *for a fixed sub-problem***. It does **not** establish that `r_rho` is the
correct transition statistic; that is the hypothesis this study tests, and §12
of the brief names the specific way it may fail.

**A limit on the argument, found empirically and recorded here.** The
sub-problem is *not* fixed across a descent: `innerLoop` initialises MMA's
moving asymptotes at the move box itself (`low = xmin`, `upp = xmax`), so
halving the move halves the asymptote spread and shrinks `s` as well. Measured
in this study, the invariance is **exact in the saturated regime** (160×20,
descent 0.04→0.02: `r_rho` 1.000→0.999, `max|Δρ|` 0.03999→0.01998) but only
**approximate when partially saturated** (320×40, descent 0.04→0.02: `r_rho`
0.65–0.77 → ~0.39, because `max|Δρ|` fell 3.3× while the move fell 2×). See
`REPORT.md`, "An unplanned finding".

## 6. Numerical confirmation of the invariance property

From the archived evidence, at 320×40, `r_rho` in the ten iterations preceding
each production descent:

| descent | move before → after | `r_rho` over the preceding 10 |
|---|---|---|
| iter 130 | 0.04 → 0.02 | 0.746 0.798 0.749 0.702 0.759 0.806 0.729 0.726 0.770 0.647 |
| iter 141 | 0.02 → 0.01 | 0.409 0.409 0.343 0.355 0.390 0.416 0.380 0.401 0.395 0.393 |
| iter 152 | 0.01 → 0.005 | 0.277 0.245 0.199 0.199 0.198 0.199 0.169 0.199 0.190 0.201 |

and at 160×20:

| descent | move before → after | `r_rho` over the preceding 10 |
|---|---|---|
| iter 79 | 0.04 → 0.02 | 1.000 0.999 0.999 0.999 0.999 0.999 0.999 1.000 0.999 1.000 |
| iter 90 | 0.02 → 0.01 | 0.997 0.997 0.992 0.996 0.995 0.996 0.995 0.997 0.995 0.997 |
| iter 101 | 0.01 → 0.005 | 0.581 0.446 0.778 0.472 0.445 0.669 0.520 0.763 0.550 0.378 |

At 160×20 the design is **fully bound-saturated** (`r_rho ≈ 1.000`) through the
first two descents: the production trigger descends while the design is using
100 % of the step it is allowed. That is the phenomenon this study was
commissioned to test, and it is visible in the archive before any new run.

*(Source: `diagnostics/admission_rule/runs/unstopped_{160x20,320x40}_iterations.csv`,
column `ratio_maxAbs_over_move`. Reused as design input under §2 of the brief;
no threshold in this study was chosen after seeing it, and both are inherited.)*
