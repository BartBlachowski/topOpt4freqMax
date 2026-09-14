# REPORT — mechanism audit of the production beta stall signal

**Mechanism audit. No optimisation was run; nothing in production was changed.**
The question is not "what should replace beta" but "what does beta actually
measure, and is that the quantity the move controller needs?"

| | |
|---|---|
| Starting HEAD | `cb6c0eae31a25521f7c5fed1c4a89564ed63344e`, branch `benchmark-methodology-r2`, clean |
| Integrity | currentness **CURRENT**, manifest **74/74**, `+impl` tree `c1455374…`, dispatch **PASS**, 0 forbidden paths, evidence gate **PASS** |
| Method | code trace → mathematics → existing telemetry (160x20, 320x40, 400x50) + the durable 400x50 raw trajectory |
| Companion files | [`CODE_TRACE.md`](CODE_TRACE.md) · [`MATHEMATICAL_ANALYSIS.md`](MATHEMATICAL_ANALYSIS.md) · [`LITERATURE_PROVENANCE.md`](LITERATURE_PROVENANCE.md) · [`DATA_INVENTORY.md`](DATA_INVENTORY.md) · [`PROVENANCE.md`](PROVENANCE.md) · [`METRICS.json`](METRICS.json) |

---

## Headline

`beta` is the **bound variable of Du–Olhoff Eq. (25a)** — a primal, scalar,
eigenvalue-valued design variable appended to MMA's vector. At the optimum it is
the largest lower bound on the smallest eigenvalue that **one linearised step**
can promise inside the move box.

By the time production descends, `beta` has collapsed onto `omega_1^2` (predicted
gain `4.4e-4` at 400x50), so **"beta has stalled" means "omega_1^2 is improving
by less than 0.5% per 10 iterations"** — a statement about the *rate of objective
progress*, and about nothing else.

The stall metric is **not mis-scaled**. It fires at `rel = 0.00455, 0.00484,
0.00488` at the three meshes — a spread of `3.3e-4` against a `5e-3` tolerance —
while the topology evolution still remaining is **9.0%, 43.4%, 50.1%**. The same
metric value means wildly different things. There is no scale error to repair,
because the metric is a *relative* change of an *intensive* scalar and is already
dimensionless and mesh-consistent.

The structural reason is a property of the formula, not of any dataset: **beta
depends on the design only through `lambda_j(rho)` and the eigenvalue gradients.
`M_nd` is a pure distribution functional and is not a function of `lambda` at
all.** Any redistribution of gray material in the null space of those gradients
changes `M_nd` arbitrarily while leaving beta fixed. No normalisation can make a
functional sensitive to a variable it does not contain.

---

## The thirty questions

### 1. What exactly is beta in the implemented algorithm?

The **bound variable of the Du–Olhoff bound formulation (25a)**. The subproblem
is `maximise beta` subject to `beta <= lambda_j + Delta lambda_j(drho)` for every
tracked mode, plus volume and the move box. At the optimum, beta is the largest
lower bound on the smallest eigenvalue achievable by one linearised step within
the box. It is **not** an MMA quantity.

### 2. Where is it computed?

`+impl/algo/innerLoop.m`. It is the `(NE+1)`-th entry of MMA's design vector
(`nvar = NE+1`, line 46), initialised to `lambda_n` (line 52, `x = [zeros(NE,1); 1]`),
driven by the objective `f0val = -x(end)` (line 133 — MMA minimises, so this
maximises beta), and returned at line 153 as `st.beta = x(end)*lamref` with
`lamref = lambda_n`. `olhoffSolve.m:368` records it as `hist.beta(outer)`.

### 3. Is it primal, dual, artificial, slack, bound, or other?

**Primal, and specifically a bound variable.** It is literally a design variable
of the MMA subproblem. It is *not* a dual (`lam, xsi, eta, mu, zet, s`), *not* an
MMA artificial variable (`y`, `z`), and *not* a slack. Units: eigenvalue,
`lambda = omega^2`; scalar.

**A trap worth recording:** `mmasub.m:108` defines an unrelated internal quantity
*also* named `beta` — the upper trust-bound **vector** on the design variables.
Reading `mmasub` alone would give exactly the wrong answer, which is why the
brief's instruction not to infer meaning from the name was well placed.

### 4. What mathematical condition does beta stall represent?

Exactly this, transcribed from `olh.move.limit`:

> the mean of beta over the last 10 completed iterations exceeds the mean over
> the preceding 10 by less than `5e-3` (0.5%), **and** at least 11 iterations
> have passed since the last rung change.

Since `beta = lambda_n·(1+g)` with measured `g = 4.3e-4` (320x40) and `4.4e-4`
(400x50), this reduces to: **`omega_1^2` is increasing by less than 0.5% per 10
outer iterations.** It is a test on the *rate of objective progress*.

### 5. What does beta stall NOT represent?

Three things, each measured rather than argued:

1. **Not that little total eigenvalue gain remains.** beta is a one-step
   prediction; gain accumulates over many steps. Ratio of actual remaining
   `lambda` gain to beta's prediction at the stall: **0.5x, 17.6x, 99.2x** at
   160x20, 320x40, 400x50. At 400x50 `omega_1` rose a further **2.159%** *after*
   beta declared stall — monotonically, with multiplicity fixed at `N=2` and a
   largest single-iteration jump of 0.044, so this is genuine improvement, not a
   mode crossing.
2. **Not that `rho` has stopped moving.** `M_nd` fell 32.40% → 16.16% afterwards.
3. **Not anything at all about the *distribution* of `rho`** — the structural
   point, see Q12.

### 6. Is beta part of published MMA?

**No.** MMA's variables are `x`, `y`, `z` and its duals; beta enters MMA only
because this implementation appends it to `x`. beta *is* published — but in
**Du & Olhoff §3.5.2**, which this tree already classifies **A**: *"independent
variables = β and Δρ_e … Δ(ω_j²) are dependent."* Its `lambda_n` scaling and the
cap `beta <= 5·lambda_ref` are class **C**, ours.

### 7. Is beta-stall-driven move continuation part of Du–Olhoff?

**No — none of it.** The recorded word counts are decisive: in Du & Olhoff (2007)
`move limit` **0**, `trust region` **0**, `step size` **0**, `continuation` **0**;
in Olhoff & Du (2014) `move limit` **0**. This tree's own provenance document
states it plainly: *"The move ladder has no source at all."* Only the *existence*
of some move limit is class **B**, justified via Krog & Olhoff's first-order
directional expansion — and no numeric value appears anywhere in the lineage.

### 8. Is the current beta tolerance publication-supported?  ### 9. The stall window?  ### 10. The move ladder?

**No, no, and no** — all three are class **C** reconstruction choices with no
source, recorded as such in `SCIENTIFIC_CONFIG_PROVENANCE.md` §7: stall signal
`beta` **C**, `W = 10` and `tol = 5e-3` **C**, detector shape **C**, dwell guard
**C**, ladder levels `[0.04 0.02 0.01 0.005]` **C**.

This matters for the verdict: **rejecting the beta-stall transition rule discards
no published content.** The published bound formulation is untouched — beta
continues to do its published job *inside* the subproblem either way.

### 11. Is beta primarily an inner-subproblem or an outer-topology quantity?

**Unambiguously inner.** It is the optimal *value* of the inner subproblem,
recomputed from scratch each outer iteration (`x = [zeros(NE,1); 1]`, and
`optimizer.inner.variable = 'increment'` so MMA state does not persist). It
answers *"how much can one linearised step promise from here?"*

The move controller consumes it as an answer to *"has the topology finished
exploiting this move level?"* **The production controller is using an
inner-subproblem diagnostic as an outer-topology maturity signal.** The code and
the mathematics both support this statement; it is not an inference from the
empirical mismatch alone.

### 12. Why can beta stall while rho continues meaningful evolution?

Two independent reasons.

**Rate versus distance.** Each subproblem can honestly promise almost nothing
while their composition travels a long way. At 400x50, `g ≈ 4.4e-4` per step,
sustained over 232 further iterations, delivered 4.36% in `lambda` and halved
`M_nd`. The inner solves were meanwhile perfectly healthy — 17–21 iterations,
always converged — so beta was stable *because each subproblem was being solved
consistently and each correctly reported a small local promise*.

**Functional independence.** beta depends on the design only through
`lambda_j(rho)` and the gradients entering `Delta lambda_j`. `M_nd =
100·mean(4 rho(1-rho))` is not a function of `lambda`. Any `d rho` in the null
space of the eigenvalue gradients changes `M_nd` while leaving beta fixed to
first order. Redistributing material inside a gray transition band is exactly
such a change. **beta and `M_nd` are functionally independent quantities.**

### 13. Does beta or its stall metric scale with NE?

**No.** beta is eigenvalue-valued: at the descent it is 28849, 27541, 26532 for
NE = 3200, 12800, 20000 — tracking `omega_1^2` and *falling slightly* with
refinement, with no NE trend. The stall metric is a relative change and is
dimensionless. Empirically it fires at 0.00455 / 0.00484 / 0.00488 — spread
`3.3e-4`. **This is the finding that rules out a scaling defect.**

### 14. Does it scale with gradient magnitude?

Gradient norms were never recorded (see `DATA_INVENTORY.md`), but gradient scale
enters beta *only* through the predicted gain `g`, which **is** observable.
Measured `g/move = 2.03, 1.97, 1.95` over iterations 5–15 at the three meshes —
the same constant. So beta responds to gradient scale exactly as first-order
theory predicts, **with no mesh-dependent defect**. This is why no new
instrumentation run was authorised under §14.

### 15. Does it scale with active bounds?

It could in principle — bound-active variables truncate the achievable `drho` —
but **that channel is inactive**. Across the entire 400x50 fixed-move run,
**zero** elements are at the move bound at **any** iteration (total bound hits =
0 over 20000 × 369). The move box is not binding when production descends.

### 16. Does it scale with move/asymptote geometry?

**Yes, early; no, by the time it matters.** The coupling is real and traced in
code: `move → box (±m) → xmax−xmin = 2m → MMA asymptotes `x ∓ 0.5(xmax−xmin)` →
trust bounds`. Measured, `g ≈ 2.0·m` at all three meshes early. But by the
descent `g` has collapsed to `~4e-4`, the box is not binding, and beta is no
longer move-limited.

### 17. Does move descent mechanically alter beta?

**No — measured, and this refutes the natural hypothesis.** Across all seven
recorded descents the change in beta from the iteration before to the iteration
after is at most **0.115%** (160x20 it 90); the others are `+0.016%, +0.002%,
+0.007%, −0.019%, −0.000%, +0.010%`. Because the box is not binding, shrinking it
barely moves the subproblem optimum.

### 18. Is there evidence of a feedback loop between move and beta stall?

**Not of circularity — and the brief is right to demand demonstration.** The
"descent depresses beta, which triggers the next descent" hypothesis is
**refuted** by Q17.

What the data *do* show is a **one-way ratchet**, a different and real defect.
Once `rel` falls below tolerance it stays below it, so every subsequent rung
fires at the earliest instant the dwell guard allows: descents at **79, 90, 101**
and **130, 141, 152** — spacing **exactly 11 = W+1** at both meshes that reached
the ladder floor. After the first descent the ladder is not detecting anything;
it is free-running to the bottom at the minimum legal interval. That follows from
testing a *rate* that has permanently decayed, not from a circular dependence.

### 19. Why is the mismatch mild at 160x20 but severe at 320x40 and 400x50?

Two measured effects compound, neither of them a scale error.

**The eigenvalue-neutral subspace grows with refinement.**
`filter.radiusPhysical = 0.06·b` is fixed in *physical* units — correct
mesh-independent practice — so in element units `R/h = 1.2, 2.4, 3.0`. The gray
transition band is that many elements wide. Gray fraction at the descent is
`0.1525, 0.2642, 0.3479`; divided by `R/h` these give `0.127, 0.110, 0.116` —
**near-constant**. Grayness at the descent is essentially proportional to the
filter radius measured in elements, and every extra gray element is a degree of
freedom that can be redistributed at nearly constant `lambda`, i.e. invisibly to
beta.

**beta's one-step prediction degrades monotonically:** 0.5x (over-predicting,
i.e. conservative), 17.6x, 99.2x.

So at 160x20 beta stall genuinely did coincide with near-maturity. **The
predicate did not change; the meaning of its output did.**

### 20. Can the mismatch be explained by a theoretically justified normalisation?

**No, and this is a structural argument rather than a failed search.** For a
normalisation to help it would have to make the metric fire at different points
at different meshes — but it already fires at essentially the *same* value at all
three. There is no mis-scaling to correct.

Taking each candidate the brief lists: normalising by `NE` would multiply a
mesh-consistent metric by a mesh-dependent constant and thereby *introduce* mesh
dependence; normalising by gradient scale or by the asymptote/bound scale divides
by quantities already shown mesh-consistent (`g/m ≈ 2.0`); "normalise by a known
MMA scale" is ill-posed because beta is not an MMA quantity. And decisively:
**every candidate is a monotone transformation of beta, and no such transformation
can create sensitivity to a variable beta's formula does not contain.**

Per §23, this negative result is reported as a result. No normalisation was
adopted merely because it aligned three data points, and none was invented to
rescue the existing code.

### 21. Or is beta structurally unsuitable for deciding move-stage maturity?

**Yes.** beta answers "how much does the current linearisation promise?"; the
controller needs "has the topology finished exploiting this move level?" These
are different questions, and beta is functionally independent of the distribution
that defines the second. It also fails at a weaker version of its own job —
certifying that *eigenvalue* progress is exhausted — by a factor of 99 at 400x50.

### 22. Does the evidence support changing the beta tolerance alone?

**No.** The tolerance is not the problem: the metric fires at the same value at
all three meshes, and remaining evolution still differs 5.6-fold. Lowering it
would delay every descent by an unprincipled amount and would not create the
missing dependence on `rho`'s distribution. (It would also be a pure fit to three
points, which §23 forbids.)

### 23. Does the evidence support changing the stall window alone?

**No.** `W` affects smoothing and, through the dwell guard `W+1`, the ratchet
spacing. A larger `W` would slow the free-run down the ladder — a symptom — while
leaving the signal measuring the same quantity. Neither `W` nor `tol` was tuned
here, per the scope lock.

### 24. Does the evidence support keeping beta as one component of a future rule?

**Not as a maturity component.** beta is a legitimate and published measure of
*objective progress*, and it remains the subproblem objective regardless. But a
maturity rule needs a quantity that responds to the design distribution, and beta
provably does not. Whether a future rule should *also* consult eigenvalue
progress is a design question this audit deliberately leaves open (brief §18).

### 25. What information about topology maturity is beta missing?

Everything that lives in the **null space of the eigenvalue gradients** —
principally the distribution of intermediate densities. beta sees `rho` only
through `lambda_j(rho)` and `∇lambda_j`. It cannot see how much gray material
exists, where it is, whether it is still moving, or how far it is from 0/1. Those
are precisely what "has the topology finished exploiting this move level?" asks
about.

### 26. Does the persistent low-amplitude spatial activity say what that missing information is?

It is consistent with it, and is reported as a mechanistic observation only. The
400x50 evidence showed the low-amplitude active set is persistent (Jaccard 0.960,
77.9% always active), structured along the members, and strongly gray-enriched
(gray fraction 0.861 among active elements versus 0.348 over the design) — i.e.
exactly the eigenvalue-neutral gray redistribution beta is blind to, and it is
*coherent* rather than noise. Per brief §18 no criterion is defined from this.

### 27. Is another optimisation run needed before deciding beta's fate?

**No.** The decisive facts are (i) beta's functional form, from the code; (ii) the
metric firing at the same value across meshes against 5.6-fold different remaining
evolution; (iii) `g/move ≈ 2.0` showing no gradient-scale defect; (iv) zero
bound-active elements; (v) ≤0.115% beta change across descents. All come from
code, mathematics and existing telemetry. §14's authorisation criterion was
tested and does not apply.

### 28. Does anything justify projection?  ### 29. Density filtering?  ### 30. Changing R = 0.06·b?

**No, no, and no.** One finding must specifically not be misread: grayness at the
descent is proportional to `R/h`, the filter radius in *elements*. That is not an
argument to shrink `R` — a fixed *physical* radius is correct mesh-independent
practice, and shrinking it would reintroduce mesh dependence in the filter. The
finding is that **beta cannot see the gray band the filter legitimately creates**,
which is a beta problem. Decisively, 400x50 showed that band *does* resolve
(`M_nd` 32.4% → 16.2%, with `omega_1` improving 2.14%) with **no regularisation
change at all** — simply by not descending the move. Nothing here indicts the
filter, its radius, `p`, or the mass law.

---

## Figures

| # | file | note |
|---|---|---|
| 1 | [`fig1_beta_vs_iteration.png`](figures/fig1_beta_vs_iteration.png) | beta and `omega_1^2`, three meshes |
| 2 | [`fig2_stall_metric_vs_iteration.png`](figures/fig2_stall_metric_vs_iteration.png) | the stall metric and the ratchet |
| 3 | [`fig3_move_with_stall_events.png`](figures/fig3_move_with_stall_events.png) | move ladder with descent events |
| 4 | [`fig4_aligned_at_first_descent.png`](figures/fig4_aligned_at_first_descent.png) | aligned at first descent |
| 5 | [`fig5_remaining_evolution_vs_mesh.png`](figures/fig5_remaining_evolution_vs_mesh.png) | **the central contradiction** |
| 6 | [`fig6_beta_vs_Mnd.png`](figures/fig6_beta_vs_Mnd.png) | beta against topology maturity |
| 7 | [`fig7_beta_vs_rho_change.png`](figures/fig7_beta_vs_rho_change.png) | design motion after stall |
| 8 | [`fig8_beta_vs_inner_iterations.png`](figures/fig8_beta_vs_inner_iterations.png) | inner-solve effort |
| 9 | [`fig9_bound_active.png`](figures/fig9_bound_active.png) | zero bound-active elements |
| 10 | [`fig10_gradient_scale_substitute.png`](figures/fig10_gradient_scale_substitute.png) | **substitute** — gradient norms were never recorded; `g/move` is the observable they enter through |
| 11 | [`fig11_move_asymptote_vs_beta.png`](figures/fig11_move_asymptote_vs_beta.png) | move/asymptote coupling |
| 12 | [`fig12_dependency_diagram.png`](figures/fig12_dependency_diagram.png) | move → MMA geometry → beta → stall → move |

Figure 10 is explicitly labelled as a substitute rather than a reconstruction of
data that does not exist.

---

## Verdicts

**Mechanism**

    BETA_TRANSITION_SIGNAL_STRUCTURALLY_UNSUITABLE

beta is the optimal value of the inner linearised subproblem — the eigenvalue
improvement one step can promise. It depends on the design only through
`lambda_j(rho)` and `∇lambda_j`, and is therefore functionally independent of the
density distribution that defines topology maturity. This is not a scale error:
the stall metric is already dimensionless and fires at the same value
(`3.3e-4` spread) across meshes whose remaining evolution differs 5.6-fold, and
no monotone renormalisation can create a dependence the formula does not contain.
beta additionally fails a weaker form of its own job, under-predicting the
remaining eigenvalue gain by 99x at 400x50.

**Next step**

    NEW_TOPOLOGY_MATURITY_SIGNAL_REQUIRED

Per brief §18, this audit deliberately does **not** define that signal.

**Production**

    KEEP_CURRENT_MOVE_POLICY_PENDING_REVIEW
