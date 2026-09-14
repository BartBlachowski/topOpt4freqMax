# LITERATURE_INTERPRETATION — Part 13

Three kinds of statement are kept separate below, as the task requires:
what a **source says**, what **we derived**, and what **we measured**.

## 1. What the source says

Du & Olhoff (2007), §1, extracted verbatim from
`references/Du2007_Topological.pdf` in this repository:

> "With a view to prevent checkerboard formation and dependency of the optimum
> solutions on finite element refinement, the mesh-independent filter developed
> by Sigmund (1997), see also Sigmund and Petersson (1998), has been applied to
> the sensitivities of the objective functions in the computational models in
> the paper."

Two things follow directly:

* the filter is **prescribed by the source**, not invented by this
  reconstruction;
* it is applied **to the sensitivities**, explicitly — not to the density.

The stated *purposes* are checkerboard suppression and mesh-independence. The
source does **not** claim the filtered sensitivity is the gradient of a modified
objective, and does not discuss the question at all.

## 2. What this implementation does with that instruction

`+impl/architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md` already classifies the
choices, and this audit confirms the code matches:

| field | class | basis |
|---|---|---|
| filter exists, mesh-independent, Sigmund (1997) | **A** | the quote above |
| applied to **sensitivities** | **A** | the quote above, explicitly |
| top88 `ft=1` algebra `H(ρ·df)/(Hs·max(1e−3,ρ))` | **B** | Andreassen et al. (2011); the `max(1e−3,ρ)` guard is published behaviour, retained deliberately |
| radius R = 0.06 physical | **C** | never stated in either paper |
| applied to **every** `f_sk` rather than the diagonal | **C** | the paper has one sensitivity vector and does not say |

The implementation is **faithful to the printed choice**. This audit found no
discrepancy between `applyFilter.m` and the published top88 `ft = 1` form, and
the `max(1e−3, ρ)` guard is never active at the frozen state (0 of 28 800
elements).

## 3. What we could NOT establish from the literature available here

Whether the wider topology-optimization literature characterizes the sensitivity
filter as a heuristic without an associated modified objective.

All 30 PDFs in `references/` were searched for discussion of the sensitivity
filter's mathematical status (its relation to an objective function, a
potential, or a functional). **No such discussion was found.** Sigmund (1997),
Sigmund & Petersson (1998) and Bendsøe & Sigmund's monograph are cited by the
source but are **not present in this repository**, so their position cannot be
quoted or verified here.

Accordingly this audit makes **no claim** about what the literature says on the
point. Any statement of the form "the filter is known to be heuristic" would be
unverifiable from the material in hand and is not made.

## 4. What we derived and measured, which stands on its own

Independently of any literature position:

* `g_filt = A(ρ)·∇λ₁(ρ)` with `A = diag(1/(Hs∘ρ))·H·diag(ρ)` — exact, from the
  code;
* `‖A − Aᵀ‖_F/‖A‖_F = 1.4135`, against a maximum of √2 — A is essentially
  maximally asymmetric;
* `J_filt = A·D_{g/ρ} + A·Hess − diag(g_filt/ρ)` — verified to 1e−08;
* measured Jacobian antisymmetry 0.287, independent of FD step over a 33× range;
* closed-loop integrals with exponent 1.999 and exact sign reversal;
* the physical gradient passes the same tests at the 1e−06 level.

## 5. The scientifically precise statement

Not: *"the filter is wrong."* The filter is what the source prescribes, and the
code implements it faithfully.

The defensible statement is the one the task itself anticipates:

> **The sensitivity-filtered update field is not the gradient of the physical
> objective, and is not the gradient of any scalar function in a neighbourhood
> of the frozen 480×60 endpoint. The algorithm can therefore converge — in the
> sense that its increments become small and its controller declares
> exhaustion — without converging to a stationary point of the underlying FE
> optimization problem.**

That is a statement about a *mathematical property of a prescribed method*, not
about an implementation defect, and not about the authors' intent. Du & Olhoff
introduce the filter for checkerboard suppression and mesh-independence; whether
it also preserves stationarity is a question their text does not raise, and this
audit answers it in the negative for this reconstruction at this state.
