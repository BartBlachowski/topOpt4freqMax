# MATHEMATICAL ANALYSIS — what "beta has stalled" does and does not imply

## 1. The subproblem beta optimises

At outer iteration `k`, with design `rho_k` fixed, `innerLoop` solves the
Du–Olhoff bound formulation (25):

```
    maximise_{drho, beta}   beta
    subject to
        (25b)   beta <= lambda_J + f_JJ' drho                    (simple mode J)
        (25c)   beta <= lambda_j + Delta lambda_j(drho),  j=1..N (multiple modes)
        (25e)   sum(rho + drho) = V*NE                           (volume)
        (25f)   max(rho_min - rho, -m) <= drho <= min(1 - rho, m) (box + move m)
```

Its optimal value is what the code returns as `st.beta`. Writing
`lambda_n(k) = omega_1(k)^2` for the current smallest tracked eigenvalue:

```
    beta_k  =  max_{drho feasible}  min_j [ lambda_j(k) + Delta lambda_j(drho) ]
```

so beta is the **largest lower bound on the smallest eigenvalue that ONE
linearised step can promise**, given the move box. Define the **predicted gain**

```
    g_k  =  ( beta_k - lambda_n(k) ) / lambda_n(k)     >= 0
```

`g_k = 0` exactly when the linearisation predicts no further improvement.

## 2. What the stall test measures

The predicate is a relative increase of a 10-iteration mean of beta below
`5e-3`. Since `beta_k = lambda_n(k)·(1 + g_k)`:

```
    rel_k  =  [ <lambda(1+g)>_recent - <lambda(1+g)>_prev ] / <lambda(1+g)>_prev
```

**Measured fact (`METRICS.json`, `event_aligned`):** at the first production
descent `g = 9.1e-3` (160x20), `4.3e-4` (320x40), `4.4e-4` (400x50). At the two
finer meshes `g` is four orders of magnitude below 1, so `beta ≈ lambda_n` to
within 0.04% and

> **beta stall ⟺ omega_1^2 is increasing by less than 0.5% per 10 outer iterations.**

The stall test is, to excellent approximation, a test on the **rate of objective
progress**. Nothing else.

## 3. Dependence of beta on the listed quantities

| quantity | dependence | evidence |
|---|---|---|
| number of variables **NE** | **none, explicitly.** beta is eigenvalue-valued; NE changes the discretisation of `lambda`, not beta's meaning or magnitude | beta at descent = 28849, 27541, 26532 for NE = 3200, 12800, 20000 — it *falls slightly*, tracking `omega_1^2`, and shows no NE trend |
| **move m** | `g ≈ c·m` while the linearisation is informative | `g/m = 2.03, 1.97, 1.95` over iterations 5–15 at the three meshes — the same constant |
| **asymptote distances** | proportional to `m` at MMA iterations 1–2 (`low/upp = x ∓ 0.5(xmax-xmin)`, `xmax-xmin = 2m`) | code trace |
| **objective scaling** | divided out: beta is formed as `bs·lamref` with `lamref = lambda_n` | `innerLoop.m:153` |
| **constraint scaling** | volume row is normalised by `Vtot`; eigenvalue rows by `lamref` | `innerLoop.m:104-112` |
| **gradient magnitudes** | enter only through `Delta lambda(drho)`, hence through `g` | `g/m ≈ 2.0` at all meshes ⇒ no mesh-dependent gradient-scale defect |
| **bound-active variables** | would truncate the achievable `drho` | **measured: ZERO elements are ever at the move bound at 400x50**, so this channel is inactive at the descent |
| **multiplicity** | sets the number of `(25c)` rows | `N = 2` throughout all runs examined |

The one scaling law that *can* be derived cleanly is the early one. For small
`m`, with the box active and the volume constraint fixing the mean, the optimum
of a linear program in `drho` scales linearly with the box half-width, so
`g = c·m`. That is confirmed at `c ≈ 2.0`, identically at all three meshes.

**Why no exact law holds later:** once the box stops binding (zero elements at
the bound), the optimum is interior and set by the min-max structure over `N`
eigenvalue rows and the volume equality. `Delta lambda_j` is the *nonlinear*
multiple-eigenvalue expansion `deltaLambda`, and the active set of `(25b)/(25c)`
changes between iterations. No closed-form `g(m, NE, ∇λ)` survives that, and
none is asserted here.

## 4. The central mathematical statement

**What beta stall implies.** The linearised subproblem at the current design
predicts an eigenvalue improvement whose 10-iteration running mean has grown by
less than 0.5%. Equivalently: *locally, first-order eigenvalue progress has
become slow.*

**What beta stall does NOT imply.** Three separate things, each measured:

1. **It does not imply that little total eigenvalue gain remains.** beta is a
   *one-step* prediction; remaining gain accumulates over many steps. Measured
   under-prediction of the actual remaining `lambda` gain to the fixed-move
   endpoint:

   | mesh | predicted `g` at stall | actual remaining `lambda` gain | actual / predicted |
   |---|---:|---:|---:|
   | 160x20 | 9.06e-03 | 4.67e-03 | **0.5x** — beta *over*-predicted, i.e. conservative |
   | 320x40 | 4.28e-04 | 7.50e-03 | **17.6x** under-predicted |
   | 400x50 | 4.40e-04 | 4.36e-02 | **99.2x** under-predicted |

   The progression is monotone in refinement and crosses from over- to
   under-prediction: at the coarse mesh beta's one-step promise was pessimistic
   relative to what remained, and by 400x50 it was optimistic by two orders of
   magnitude in the wrong direction.

   At 400x50 `omega_1` rose monotonically by a further 2.159% after beta declared
   stall (no mode crossing: multiplicity stayed `N = 2`, largest single-iteration
   jump 0.044).

2. **It does not imply that rho has stopped moving.** `M_nd` fell from 32.40% to
   16.16% after the 400x50 stall verdict.

3. **Most fundamentally, it says nothing whatever about the DISTRIBUTION of rho.**
   This is the structural point, and it is a statement about the functional form
   of beta, not about any dataset:

   > beta depends on the design only through `lambda_j(rho)` and the gradients
   > `f_JJ`, `F` entering `Delta lambda_j`. Any change `d rho` lying in the null
   > space of those gradients leaves the entire subproblem — and therefore beta —
   > unchanged to first order, no matter how large `|d rho|` is or how much it
   > changes `M_nd`.

   Redistributing material within a gray transition band while holding the
   eigenvalues and their gradients fixed is exactly such a change. `M_nd` is a
   pure distribution functional, `100·mean(4 rho (1-rho))`; it is not a function
   of `lambda` at all. **beta and `M_nd` are functionally independent**, and no
   rescaling of beta can create a dependence that the formula does not contain.

## 5. Inner versus outer stationarity

The distinction the brief requires is sharp here and is settled by the code, not
by analogy.

beta is the **optimal value of the inner subproblem**. It is recomputed from
scratch at every outer iteration (`x = [zeros(NE,1); 1]` at `innerLoop.m:52`, and
`optimizer.inner.variable = 'increment'`, so MMA state does not persist across
outer iterations). It is therefore an **inner-problem diagnostic** in the precise
sense that it is the answer to *"how much can one linearised step promise from
here?"*.

The move controller consumes it as an **outer-problem maturity signal** — *"has
the topology finished exploiting this move level?"* Those are different
questions, and beta answers only the first.

The mechanism by which they coexist is not mysterious. A sequence of subproblems
can each promise a negligible gain while their composition moves the design a
long way: `g ≈ 4.4e-4` per step at 400x50, sustained over 232 further iterations,
delivered 4.36% in `lambda` and halved `M_nd`. Small rate, large distance. The
inner solves are meanwhile *healthy* — 17–21 iterations, always converged — so
beta is stable because each subproblem is being solved consistently and each
honestly reports a small local promise.

This is the same **rate-versus-distance confusion** that the preceding activity
study found in `max(u)` and `RMS(u)`: those measure how fast the design is
moving, beta measures how fast the objective is improving, and neither measures
how far there is left to go.

## 6. Why refinement makes it worse

Two measured effects compound, and neither is a defect of beta's scale.

**(a) The eigenvalue-neutral subspace grows with refinement.** `filter.radiusPhysical
= 0.06·b` is fixed in *physical* units — correct mesh-independent practice — so
in element units `R/h = 1.2, 2.4, 3.0` at the three meshes. The gray transition
band is that many elements wide. Gray fraction at the descent is `0.1525, 0.2642,
0.3479`; dividing by `R/h` gives `0.127, 0.110, 0.116` — **near-constant**, i.e.
grayness at the descent is essentially proportional to the filter radius measured
in elements. Every one of those extra gray elements is a design degree of freedom
that can be redistributed at nearly constant `lambda`, i.e. invisibly to beta.

**(b) beta's one-step prediction degrades monotonically with refinement**
(0.5x, 17.6x, 99.2x above).

So at 160x20 beta stall genuinely did coincide with near-maturity (9.0% left, and
its one-step promise was if anything conservative). At 400x50 the same predicate
value corresponds to 50.1% remaining. **The predicate did not change; the meaning
of its output did.**

## 7. Is there a normalisation that repairs this?

For a normalisation to help, it would have to make the stall metric fire at
different points at different meshes. But the metric already fires at
**essentially the same value at all three**: `rel = 0.00455, 0.00484, 0.00488`,
a spread of `3.3e-4` against a `5e-3` tolerance. There is no mis-scaling to
correct — the quantity is a *relative* change of an *intensive* scalar and is
already dimensionless and mesh-consistent.

Consider each normalisation the brief lists:

| candidate | motivation | verdict |
|---|---|---|
| relative beta change | already what is used | no change |
| by variable count `NE` | beta has no NE dependence to remove | multiplies a mesh-consistent metric by a mesh-dependent constant; would *introduce* mesh dependence, not remove it |
| by gradient scale | gradient scale enters via `g`, measured mesh-consistent (`g/m ≈ 2.0`) | nothing to correct |
| by MMA asymptote / bound scale | equivalent to dividing by `m`; `g/m` is mesh-consistent early and collapses late at every mesh alike | nothing to correct |
| by a known MMA scale | beta is not an MMA quantity | ill-posed |

Every one of these rescales a quantity that is *already* comparable across
meshes. None of them can introduce a dependence on the distribution of `rho`,
because — by §4 — beta does not contain one. **A monotone transformation of a
functional cannot make it sensitive to a variable it does not depend on.** That
is the decisive argument, and it is structural rather than empirical.

## 8. On circularity: claimed by hypothesis, refuted by measurement

It is natural to suspect a feedback loop — descending the move shrinks the box,
which mechanically depresses beta, which triggers the next descent. **The data
refute this.** Across all seven recorded descents the change in beta from the
iteration before to the iteration after is at most **0.115%**, and at 400x50
**zero elements are ever at the move bound**, so the box is not even active when
production descends. The `g ≈ 2m` coupling is real *early*, but by the descent
`g` has collapsed to `~4e-4` and beta is no longer move-limited.

What the data *do* show is a **one-way ratchet**, which is a different defect.
Once `rel` falls below tolerance it stays below it, so every later rung fires at
the earliest instant the dwell guard permits: descents at `79, 90, 101` and
`130, 141, 152` — spacing **exactly 11 = W+1** at both meshes that reached the
ladder floor. After the first descent the ladder is not detecting anything; it is
free-running to the bottom at the minimum legal interval. That is a consequence
of testing a *rate* that has permanently decayed, not of a circular dependence.
