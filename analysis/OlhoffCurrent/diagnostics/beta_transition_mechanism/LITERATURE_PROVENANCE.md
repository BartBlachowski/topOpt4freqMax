# LITERATURE PROVENANCE — what is published and what is ours

Sources consulted are the ones this implementation already documents, chiefly
`+impl/architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md`, the provenance column
of `+impl/architecture/+olh/+config/schema.m`, and the MMA source header.
Classification uses this tree's existing scheme:

**A** published · **B** source-supported (lineage, not stated) ·
**C** under-specified reconstruction choice · **D** later experimental modification

## A. Is beta itself part of the published mathematics?

**Yes — but of Du & Olhoff, not of MMA.**

`schema.m` / `SCIENTIFIC_CONFIG_PROVENANCE.md` §6 records:

> `independent variables = β and Δρ_e` | **A** | *"§3.5.2 explicitly; Δ(ω_j²) are
> dependent."*

So beta is the bound variable of the published bound formulation (25a). Its use
as the **objective of the sub-problem** is class A.

Two adjacent choices are *not* published:

| ingredient | class | basis |
|---|---|---|
| beta scaled by `λ_n`; `xmax(β) = 5` | **C** | "Numerical conditioning, not in the source." |
| MMA hyper-constants `a₀=1, a=0, c=1000, d=0` | **C** | "Not stated by Du & Olhoff; conventional MMA usage." |

**beta is NOT part of published MMA.** MMA's own variables are `x` (design),
`y`, `z` (artificial), and duals `lam, xsi, eta, mu, zet, s`. beta enters MMA
only because this implementation appends it to `x` as the `(NE+1)`-th design
variable. Confusingly, `mmasub.m:108` defines an *unrelated* internal quantity
also called `beta` — the upper trust-bound vector. See `CODE_TRACE.md`.

The MMA implementation itself is class **A** at the level of "MMA was used"
(`optimizer.inner.type = 'mma'`, *"sec. 3.5.3: the MMA method (Svanberg 1987) has
been used"*), with the `published` Sept-2007 constants class **B**.

## B. Is beta-stall as a MOVE-CONTINUATION signal publication-supported?

**No. Entirely reconstruction.**

`SCIENTIFIC_CONFIG_PROVENANCE.md` §7 records the whole move-limit policy, and the
stall signal specifically:

| ingredient | class |
|---|---|
| stall **signal** = `beta` | **C** |
| stall window `W = 10`, tolerance `5e-3` | **C** |
| stall detector shape (mean-of-window vs mean-of-previous-window) | **C** |
| re-arm dwell `(k − lastStage) > W` | **C** |
| policy family, `move` initial, ladder levels, `moveMin` | **C** |
| existence of *any* move limit | **B** |

The word counts recorded in that document are decisive:

> Du & Olhoff (2007): `move limit` **0**, `trust region` **0**, `step size` **0**,
> **`continuation` 0**.
> Olhoff & Du (2014): `move limit` **0**, `tolerance` **0**, `filter` **0**.

And the document's own summary of findings states plainly:

> **"The move ladder has no source at all."** Its stopping guard `settledMove`
> *"exists only to repair a defect the ladder itself introduces."*

The only class-**B** element is the *existence* of a move limit, justified via
Krog & Olhoff (CISM Eq. 103): the multiple-eigenvalue sensitivity model is a
first-order directional expansion valid only for increments of restricted
magnitude. **No numeric value appears anywhere in the lineage.**

## C. Classification of the four ingredients the brief asks about

| ingredient | class | note |
|---|---|---|
| beta tolerance `5e-3` | **RECONSTRUCTION CHOICE (C)** | no source; no sensitivity study on record |
| stall window `W = 10` | **RECONSTRUCTION CHOICE (C)** | no source |
| move ladder `[0.04 0.02 0.01 0.005]` | **RECONSTRUCTION CHOICE (C)** | levels have no source; existence is B |
| move-descent logic (stall ⇒ one rung, dwell-guarded) | **RECONSTRUCTION CHOICE (C)** | no source |

Nothing in this group is published. That matters for the verdict: rejecting the
beta-stall transition rule discards **no published content whatsoever**. The
published bound formulation (25), including beta as the subproblem objective, is
untouched by such a decision — beta continues to do its published job inside the
subproblem regardless of whether it also drives the move ladder.

## D. Does any source state that beta stationarity implies topology stationarity?

**No.** No such statement exists in Du & Olhoff (2007), Olhoff & Du (2014), or
the Krog & Olhoff lineage material this tree cites. The bound formulation
introduces beta purely as the device that converts a max–min eigenvalue objective
into a smooth constrained programme; the papers make no claim about beta's
behaviour across *outer* iterations, and indeed print no outer-loop convergence
criterion at all (`SCIENTIFIC_CONFIG_PROVENANCE.md` §8 records the stopping
tolerance `ε` as class **C**: *"Never given."*).

This is the crucial separation the brief demands:

> **"beta exists in MMA"** — false as stated; beta exists in *Du & Olhoff*, and is
> published there as a subproblem variable.
>
> **"beta is a valid topology-maturity criterion"** — asserted nowhere, by anyone.
> It is an undocumented inference made by this reconstruction.

## E. Summary

| claim | status |
|---|---|
| beta is the bound variable of Eq. (25a) | **PUBLISHED (A)** |
| beta is a primal subproblem variable | **PUBLISHED (A)**, confirmed in code |
| beta scaling by `λ_n`, `β ≤ 5λ_n` | **RECONSTRUCTION (C)** |
| a move limit exists at all | **SOURCE-SUPPORTED (B)** via Krog & Olhoff |
| move ladder and its levels | **RECONSTRUCTION (C)** |
| beta stall as the continuation trigger | **RECONSTRUCTION (C)** |
| `W = 10`, `tol = 5e-3`, dwell guard | **RECONSTRUCTION (C)** |
| beta stationarity ⇒ topology maturity | **ASSERTED BY NO SOURCE** |
