# Investigation Plan — Separating "Wrong Mathematics" from "Insufficient Globalization" in the N = 2 Reconstruction

**Status: plan only. No experiment in this document has been run. No production
file, no campaign file, and no optimizer is modified by this document.**

Follows: `faithful_reconstruction_report_v2.md` (Revision V2.1),
`independent_scientific_audit_phase_b.md`, `independent_scientific_audit_round2.md`.
Inherits their terminology, evidence grades ([A]/[B]/[C]) and markers
([OBS]/[EV]/[INF]/[DEC]/[HYP]/[CONC]) verbatim; §0.0 of the report is normative
here too.

Date: 2026-07-29 · target platform MATLAB R2025b (Optimization Toolbox present:
`linprog` and `fmincon` both available — verified).

---

## 0. Framing

### 0.1 The question, stated precisely

The report established, at grade [A], that across 19 runs a clustered lowest
eigenvalue is reached at most once per run and held for at most one outer
iteration, and that forcing `N = 2` at the one testable near-coalescent design
does not retain it (§6.3, §6.5, C4). It explicitly left open (§6.3, "Open
alternatives") whether the N = 2 subproblem is correctly reconstructed at all.

The brief poses two hypotheses:

| | Hypothesis | Falsified by |
|---|---|---|
| **H1** | The reconstruction of the published N ≥ 2 mathematics is incorrect or incomplete. | Independent verification of every clustered quantity against derivatives, against an independent exact solution of Eq. (25), and against basis rotation — all passing. |
| **H2** | The mathematics is correct; the failure is insufficient globalization (step control). | A step-length sweep that reaches zero step length without recovering cluster retention. |

### 0.2 The hypothesis space is larger than two, and the plan must say so

Treating H1 and H2 symmetrically, as required, exposes that they are not
exhaustive. Two further hypotheses are live and must be carried through the
decision tree, or its terminal conclusions will be over-claimed:

| | Hypothesis | Why it cannot be dismissed a priori |
|---|---|---|
| **H3** | The **forward model** (element formulation, filter type and radius, mass interpolation Eq. (4b), support geometry, ρ_min) differs from the original in a way that changes the optimization path. | The report names this as "the principal residual caveat on the campaign's strongest finding" (§12). It is verified only against one scalar at one design (initial ω₁ to 0.4 %, Fig. 2). Nothing on the path is verified. |
| **H4** | The published formulation, correctly reconstructed and correctly solved, **does not preserve a cluster on this problem** — i.e. the observed behaviour is the true behaviour of the printed mathematics. | This is the null hypothesis. It is the outcome if H1, H2 and H3 are all falsified, and it must be reachable by the tree, otherwise the investigation cannot conclude "the paper is incomplete" without begging the question. |

**Constraint honoured throughout:** the plan never assumes the paper is wrong and
never assumes the implementation is wrong. Every node of the decision tree in
§5 has at least one branch that assigns the defect to the reconstruction and at
least one that assigns it elsewhere.

### 0.3 The fidelity ladder — normative for all reporting below

Every result produced under this plan must be tagged with the **lowest** ladder
level at which it holds. This is the mechanism that keeps H1 and H2 separable in
the written record, and it is the mechanism that prevents Phase 2 and Phase 4
results from being silently folded back into a fidelity claim.

| Level | Contents | Status |
|:--:|---|---|
| **L0** | Eq. (25a–f) exactly as printed; Fig. 1 loop; ρ := ρ + Δρ; box (25f) the only restriction on Δρ. | The paper. Already shown [A] to collapse at outer iteration 1. |
| **L1** | L0 + the 18 declared reconstructions of quantities the paper does not specify (§12), **none of which restricts the step**. | Still a reconstruction of the paper. All of Phase 1 lives here. |
| **L2** | L1 + a **fixed** restriction on ‖Δρ‖ (`outer_move`, and/or damping `alpha`). | **Not in the paper.** This is Regime B. Already outside L1 today. |
| **L3** | L2 + an **adaptive** restriction or a **step-acceptance test** (line search, trust region, pred/ared). | Beyond the paper. Phase 4 lives here. |
| **L4** | L3 + cluster-aware control (hysteresis on N, cluster-triggered contraction). | Beyond the paper. Phase 3's remedies live here. |

[EV] The paper contains no move limit, no damping factor, no line search and no
step-acceptance test; Fig. 1 box 4 is an unconditional update. The **only**
convergence gate the paper draws is the inner-loop diamond ("Increments Δρ_e
converged?"). Enforcing that gate is L1; everything else in L2–L4 is an addition.

### 0.4 What the paper actually prints (transcribed from `references/Du2007_Topological.pdf`, pp. 98–99)

Reproduced here because four Phase 1 verifications compare directly against it,
and the report paraphrases rather than quotes it.

```
(25a)  max  β                              over  β, Δρ_1 … Δρ_NE
(25b)  β − [ ω_J² + f_JJᵀ Δρ ]  ≤ 0        for  j = J = n + N
(25c)  β − [ ω_j² + Δ(ω_j²)  ]  ≤ 0        j = n, …, n+N−1
(25d)  det( f_skᵀ Δρ − δ_sk Δ(ω²) ) = 0    s,k = n, …, n+N−1
(25e)  Σ_e (ρ_e + Δρ_e) V_e − V* ≤ 0 ,     V* = α V₀
(25f)  0 < ρ ≤ ρ_e + Δρ_e ≤ 1 ,            e = 1, …, NE
```

with, from §3.4:

```
(18)   det( f_skᵀ Δρ − δ_sk Δλ ) = 0       (Seyranian et al. 1994)
(19)   f_sk = { φ_sᵀ(K'_{ρ1} − λ̃ M'_{ρ1})φ_k , … }ᵀ ,  λ̃ = the N-fold eigenvalue
(22)   f_skᵀ Δρ = 0 ,  s ≠ k               (vanishing off-diagonal case)
(23)   Δλ_j = f_jjᵀ Δρ                     (⇒ the simple-eigenvalue formulas)
```

Four textual facts from §3.4/§3.5.3 that bear directly on this investigation:

1. [EV] "the increments Δλ_j … are generally **nonlinear** functions of the
   direction of the design increment vector Δρ. Thus, unlike simple eigenvalues,
   multiple eigenvalues **do not admit a usual linearization** in terms of the
   design variables." — The paper itself states that Eq. (25) for N ≥ 2 is *not*
   a linear program, and that its model is a **directional derivative**, i.e.
   first-order and locally valid over an **unspecified** radius.
2. [EV] "if we introduce the additional constraints f_skᵀΔρ = 0 for s ≠ k … the
   suboptimization problems … reduce to linear programming problems (see Krog and
   Olhoff 1999)." — A **paper-sanctioned alternative reconstruction** of the
   N ≥ 2 subproblem. Testing it is L1, not a modification.
3. [EV] (25c) is written against the **individual** ω_j², j = n…n+N−1, under the
   stated premise that they are exactly equal (Eq. 17). The reconstruction
   substitutes λ̄ = mean(λ_cluster) and applies it to a *near*-cluster, where the
   premise does not hold. The pairing of the N roots Δ(ω_j²) of (25d) with the
   N indices j is **not specified** by the paper and is vacuous only under exact
   multiplicity.
4. [EV] J is chosen as n+N and ω_J is **assumed simple**. `detect_multiplicity.m`
   does not verify this (its own docstring says so).

### 0.5 What the implementation actually solves

From `inner_loop_mma.m:182–191`, the cluster rows are

  β̂ − 1 − μ_i(F(Δρ))/λ̄ ≤ 0,  i = 1…N,  F(Δρ)_sk = f_skᵀΔρ,

with μ_i the ascending eigenvalues of F and the MMA gradient
∂μ_i/∂Δρ_e = q_iᵀ F_e q_i, q_i the i-th eigenvector of F **at the current inner
iterate**. Two structural observations, both testable, neither yet tested:

* [INF] The constraint set of the true Eq. (25) is **convex**: requiring
  β ≤ λ̄ + μ_i for all i is equivalent to β ≤ λ̄ + μ_min(F(Δρ)), and μ_min of a
  symmetric matrix affine in Δρ is a **concave** function of Δρ. Equivalently,
  Eq. (25c–d) is the linear matrix inequality λ̄ I + F(Δρ) ⪰ β I, and — since
  qᵀF(Δρ)q = (Σ_{s,k} q_s q_k f_sk)ᵀ Δρ is linear in Δρ for each fixed unit q —
  **Eq. (25) for N ≥ 2 is a semi-infinite linear program**, indexed by q ∈ S^{N−1}.
  For N = 1 it degenerates to exactly the LP the campaign already solved exactly
  in §2.4. *This gives an independent, exact, certifiable solution method for the
  clustered subproblem, structurally identical to the one already trusted for
  N = 1.*
* [INF] MMA linearizes each μ_i at the current iterate. The linearization of a
  **concave** function is an **over-estimate**. The MMA-approximated feasible set
  is therefore a **relaxation** of the true Eq. (25) feasible set, so the returned
  Δρ* may be infeasible for the printed problem and β* may exceed the printed
  problem's optimum. Whether this is material at the tested designs is measurable
  and is P1.2 below. This does not arise for N = 1 (the constraint is linear and
  MMA's model is exact there), which is precisely why the campaign's N = 1
  results are safe and its N = 2 results are not.
* [INF] At the inner-loop start Δρ = 0, hence F = 0, whose spectrum is
  **N-fold degenerate**: μ_i is non-differentiable exactly at the starting point,
  and the q_i returned by `eig` are an arbitrary tie-break (the identity), which
  makes the first linearization pick out f_11 and f_22 **in whatever basis the
  eigensolver happened to return** and discard f_12 entirely. This is a concrete,
  named candidate mechanism for a basis-dependent solve — see P1.4.

None of these three observations is a claim that the implementation is wrong.
They are the reasons the verifications below are the right ones to run first.

---

## Phase 0 — Preconditions (must precede Phases 1–4)

Three items. They are cheap, two of them require **no new solver run at all**,
and without them no later experiment is interpretable.

### P0.1 — Frozen fixtures

**Why required.** The report states the decisive limitation plainly: "the
effective sample size is one" (§6.3 item 6). Every Phase 1–4 experiment below is
a *controlled* experiment at a *fixed* design; each therefore needs a versioned,
byte-stable design vector, not a re-derivation from a trajectory.

**What to do.** `out.rho_snapshots` in `results/<tag>/run.mat` is the full
nEl × n_iter post-update design history (`recon_solve.m:351`), so the required
designs already exist on disk and need no rerun. Freeze, hash and archive as a
fixture set:

| Fixture | Source | ω (rad/s) | g₁₂ | Role |
|---|---|---|---|---|
| **F-A** | V4 CC 160×20 i2000, post-update it 21 | 284.4755, 284.8956, 302.0267, 391.4099 | 1.4766e-03 | The near-coalescent iterate. Primary N = 2 test bed. |
| **F-B** | V4 CC 240×30 i2000, post-update it 23 | 90.86, 90.90, … | 4.90e-04 | The only solver-detected N = 2 state. Degraded design; secondary. |
| **F-C** | Uniform ρ = volfrac, both meshes | 145.5692, 363.0493, … | 1.494 | N = 1 control; reproduces §2.4. |
| **F-D** | 240×30 V4, iterations 20–26 inclusive | — | — | Approach/departure window for Phase 3 profiling. |
| **F-E** | **New**: square domain, 4-corner pins, uniform ρ — the geometry already used by `verify_multiplicity.m` case D | — | ~0 (exact, by symmetry) | **Exact-multiplicity control problem.** |

**F-E is not optional.** The Seyranian theory reconstructed in Eq. (18)–(19) is
exact only for an exactly multiple eigenvalue. Verifying it on a beam near-cluster
with g₁₂ = 1.5e-3 conflates "is the formula implemented correctly" with "is the
formula applicable at this gap". F-E separates them: on a symmetry-degenerate
domain the theory holds exactly, so any failure there is unambiguously an
implementation defect. Every Phase 1 derivative test runs on F-E **first** and on
F-A only after F-E passes.

**Confirms.** Fixtures reload to a design whose recomputed spectrum matches the
archived `eigen_history.csv` to ≤ 1e-10 relative.
**Falsifies.** Any mismatch ⇒ the archive and the solver disagree; stop and
resolve before anything else, because §6 of the report rests on this archive.

### P0.2 — Zero-cost re-analysis of the existing archive

**Why required.** Eight quantities this investigation needs are already recorded
across all 19 runs and can be extracted by re-analysis alone, with no MATLAB
solver run. Doing this first prevents designing experiments to measure things
already measured, and it converts the "sample size one" problem into a measured
distribution wherever the data allow.

**What to extract**, from `eigen_history.csv`, `multiplicity_history.csv`,
`outer_history.csv`, `inner_history.csv`, `run.mat`:

| # | Quantity | Question it answers |
|---|---|---|
| 1 | Full distribution of g₁₂ per iteration, all 19 runs, both meshes | How often is the solver anywhere near a cluster? |
| 2 | Detection-flip counts N: 1→2→1 on a grid of tolerances 1e-4 … 5e-2 | Phase 3: is flapping the problem, at any tolerance? |
| 3 | **λ_J/λ̄ headroom** per iteration | See §3.M8 — the single most under-used number in the archive. |
| 4 | Count of iterations with `J_idx = 0` (⇒ (25b) silently **dropped**, β then bounded only by the cluster rows) | A structural failure mode of `n_modes = 4`. |
| 5 | Count of iterations with N ≥ 3 | Same. |
| 6 | g₁₂ trajectory shape in the ±5 iterations around every local minimum of g₁₂ | Phase 3: is a cluster *approached* or *jumped into*? |
| 7 | Joint distribution (pred_ratio, ‖Δρ‖∞/box, g₁₂) | Phase 4: offline evaluation of acceptance rules. |
| 8 | Per-iteration inner `converged` flag vs. g₁₂ | Are the near-cluster steps also the non-converged ones? |

**Cost.** Hours of Python on existing CSVs. **Expected information gain: high.**
This is the highest information-per-unit-cost item in the entire plan and must be
done first.

### P0.3 — Determinism

**Why required.** Every A/B comparison in Phases 2–4 assumes that re-running an
unchanged configuration reproduces the trajectory bitwise. `eigs` with a Krylov
restart is not guaranteed deterministic across runs unless the starting vector is
fixed, and the report's own equivalence proof (T2) tests bit-identity of two code
paths within one session, not across sessions.

**Experiment.** Re-run one 300-iteration V4 160×20 trajectory twice in separate
MATLAB sessions; compare `rho_snapshots` bitwise.
**Confirms.** Bit-identical ⇒ all later A/B differences are attributable to the
manipulated variable.
**Falsifies.** Any divergence ⇒ fix the eigensolver seeding **before** Phase 2,
and treat every single-run comparison in the existing report as having an
unquantified run-to-run component.

---

## Phase 1 — Mathematical verification

All of Phase 1 is at **L1**: no step control is introduced anywhere, and all
experiments are single-subproblem or single-step probes at frozen designs. Every
item states why it is required, the experiment, the confirming result, and the
falsifying result. Items are numbered in **descending expected information gain**,
not in the order the brief lists them; §6 gives the mapping and the schedule.

---

### P1.1 — Cluster-splitting rate along the step actually taken *(run this first)*

**Covers the brief's:** off-diagonal sensitivity terms, Eq. (18) reconstruction,
and — decisively — the H1/H2 separation itself.

**Why required.** This is the experiment that separates the two hypotheses at the
lowest possible cost, and it has never been run. The report measured the eigengap
**before** and **after** the accepted step and found it re-opened by a factor of
236 (§6.2). It never measured the gap *in between*. Those two hypotheses make
opposite predictions about the interior:

* If the Eq. (25) direction is right and the step is too long (**H2**), then
  g₁₂(ρ + tΔρ*) stays O(1e-3) for small t and blows up only as t → 1.
* If the Eq. (25) direction itself splits the cluster (**H1** or **H4**), then
  g₁₂ rises at **first order in t**, from t → 0⁺, and **no step-length control of
  any kind can retain the cluster**, because every globalization strategy in
  existence shortens the step along the model direction. H2 would be dead.

There is an exact, free arithmetic predictor of which case obtains. For N = 2 the
roots of Eq. (25d) are

  μ_{1,2}(d) = ½(F₁₁+F₂₂) ∓ √( ¼(F₁₁−F₂₂)² + F₁₂² ),  F_sk = f_skᵀd,

so the first-order splitting rate is

  **μ₂ − μ₁ = 2 √( ¼(F₁₁−F₂₂)² + F₁₂² ) ≥ 0**,

which vanishes **only** if the direction satisfies both f₁₁ᵀd = f₂₂ᵀd and
f₁₂ᵀd = 0 — a codimension-2 condition. A generic direction splits a cluster at
first order; retention requires the subproblem to *select* a non-generic
direction. Whether Eq. (25) does so is the whole question, and (μ₂−μ₁)/λ̄ at the
returned Δρ* answers it with no eigensolve at all.

**Experiment.**
1. At fixture F-A, recompute both accepted increments already reported in §6.3 —
   the N = 1 one and the forced-N = 2 one — to full inner convergence.
2. *Arithmetic, no eigensolve:* evaluate F₁₁, F₂₂, F₁₂ and hence μ₁, μ₂ and the
   splitting rate (μ₂−μ₁)/λ̄ at each returned Δρ*.
3. *Path profile:* for t on a log-spaced grid t ∈ {1e-6 … 1} ∪ {0.05 k}, solve the
   full eigenproblem at ρ + tΔρ* and record λ₁, λ₂, g₁₂, and the first-order
   predictions λ_i + t μ_i.
4. Repeat for the paper's own off-diagonal-suppressed variant (Eq. 22/23, §0.4
   fact 2) and for a random direction of equal norm as a null control.
5. Repeat on F-B and F-E.

**Independent variable:** t, and the direction family. **Controlled:** design,
spectrum, gradients, all step controls (none applied — the profile *is* the step
sweep), mesh, p, filter.

**Confirms H2** (globalization): g₁₂(t) ≈ g₁₂(0)·(1 + O(t²)) for t ≲ t*, with a
measurable t* > 0; realised λ_i track λ_i + tμ_i within a few per cent up to t*;
(μ₂−μ₁)/λ̄ ≪ g₁₂. Then a step restriction below t*‖Δρ*‖ **must** retain the
cluster, and Phase 2 is the right next step.

**Falsifies H2 outright**: (μ₂−μ₁)/λ̄ ≳ g₁₂ and g₁₂(t) rises linearly from t = 0⁺.
Then the increment returned by the reconstructed Eq. (25) destroys the cluster
infinitesimally, no trust region can help, and the investigation must go to
P1.2/P1.3 to decide whether the *solve* or the *formulation* is responsible. In
that event Phase 2 should not be run at all until Phase 1 completes.

**Cost:** ≈ 60 eigensolves + arithmetic. Hours. **Information gain: maximal.**

---

### P1.2 — Independent exact solution of Eq. (25) for N = 2

**Covers the brief's:** reconstruction of Eq. (25), reconstruction of the
constraint system, generalized gradients.

**Why required.** The report's single strongest result (§2.4) was obtained by
solving the N = 1 subproblem **twice, independently** — once with MMA and once
exactly as an LP — and finding agreement to 0.02 %. No such independent solution
exists for N = 2. The N = 2 subproblem was tested only *behaviourally*: "it runs,
converges and produces a different step" (§6.3 item 1). Behavioural plausibility
is not correctness, and §0.5 gives a specific reason to expect a discrepancy:
MMA's linearization of the concave μ_min is a **relaxation** of the printed
constraint, so `inner_loop_mma` may be solving a strictly larger problem than
Eq. (25).

**Experiment.** Exploit the semi-infinite LP structure of §0.5. Solve, at F-A and
F-E, by a **supporting-hyperplane (Kelley cutting-plane) method** whose master
problem is a finite LP in (β, Δρ):

  maximize β subject to
   β ≤ λ̄ + g(q_r)ᵀΔρ  for a finite cut set {q_r} ⊂ S¹, g(q) = Σ_{s,k} q_s q_k f_sk,
   β ≤ λ_J + f_JJᵀΔρ,  Σ(ρ+Δρ)V_e ≤ V*,  ρ_min ≤ ρ+Δρ ≤ 1,

adding at each round the cut q = the eigenvector of the smallest eigenvalue of
F(Δρ) at the incumbent. Each master problem is an LP with ~3200 variables and a
handful of coupling rows — `linprog` handles it directly (availability verified).
Terminate on the certified gap between the master's β (an upper bound, since the
cut set is a relaxation) and λ̄ + μ_min(F(Δρ)) at the incumbent (a lower bound).
Cross-check with `fmincon` on the equivalent closed-form second-order-cone
statement of the N = 2 constraint, and — as a third, wholly independent check —
against the paper's own Eq. (22)-constrained LP reduction.

**Confirms.** β*_MMA and β*_exact agree to ≲ 0.1 %; cos∠(Δρ*_MMA, Δρ*_exact) ≳
0.99; and, critically, the **true** constraint λ̄ + μ_min(F(Δρ*_MMA)) ≥ β*_MMA
holds. Then `inner_loop_mma` does solve Eq. (25) for N = 2, and this branch of H1
is closed.

**Falsifies.** β*_MMA > β*_exact beyond the certified gap, and/or
μ_min(F(Δρ*_MMA)) < β*_MMA − λ̄ by a material margin. Then the reconstruction
solves a **relaxation** of the printed subproblem for N ≥ 2, the report's §6.3
forced-N = 2 result is void as evidence about the paper, and the correct action is
to replace the inner solve with the certified one — **which is a correctness fix
at L1, not an algorithmic modification.** Retention must then be re-tested with
the exact solver before any globalization work begins.

**Secondary output of independent value.** The exact solve reports whether
μ₁ = μ₂ at the optimum. Maximizing a min-eigenvalue frequently drives the
optimum to a repeated eigenvalue — the very phenomenon that makes optimal designs
bimodal — but this is *not* guaranteed inside a small box, and it has never been
checked here. Whether the printed Eq. (25), solved exactly, equalizes the cluster
increments at F-A is the sharpest available statement about the mathematics
itself, independent of any solver.

**Cost:** implementation of a cutting-plane driver + a few dozen LP solves per
design. Days. **Information gain: maximal.**

---

### P1.3 — Well-posedness: is Δρ* a property of the problem or of the inner solver's path?

**Covers the brief's:** reconstruction of Eq. (25), generalized gradients.

**Why required.** A correctly posed convex subproblem has a solution independent
of how it is reached. §0.5 identifies three reasons the implemented N = 2
subproblem might not: the concave-constraint relaxation, the degenerate F(0) at
the starting point, and the `lambda_ref = lambda_bar` scaling that enters MMA's
conditioning. There is also a clean prediction to test: **at full inner
convergence, `move_lim` cannot change Δρ\* for N = 1** (the LP is convex, MMA's
model is exact, the clamp is inactive at a fixed point), **but it may change it
for N = 2** (the linearization point, and therefore the model, depends on the
path). If that asymmetry is observed, the N = 2 subproblem is not well posed as
implemented — and, incidentally, the confound the report flags in Regime B
(`move_lim`, `outer_move`, `alpha` changed together) partially dissolves for
N = 1.

**Experiment.** At F-A and F-C, with the inner budget large enough that the
declared stopping test is met (≥ 2000, as in the i2000 series), solve the same
subproblem under a matrix of solver-path perturbations, one at a time:

| Perturbation | Values | Must not change Δρ* if well posed |
|---|---|---|
| `move_lim` (inner per-iteration clamp) | 0.02, 0.2, Inf | ✔ |
| Inner start point x⁰ | Δρ = 0; Δρ = small random; Δρ = ½·(previous solution) | ✔ |
| `lambda_ref` normalisation | λ̄, 1, λ_J | ✔ |
| MMA asymptote initialisation | `low/upp` = box; = box scaled ×2 | ✔ |
| MMA `c` | 1e3, 1e2, 1e4 | ✔ (constraints must be satisfiable) |
| β̂ upper bound | 1e6, 1e3, 1e9 | ✔ (§2.6 shows it is not inert) |

**Confirms.** max ‖Δρ*_variant − Δρ*_base‖∞ below the inner tolerance for **all**
perturbations, at both N = 1 and N = 2.
**Falsifies.** Material dependence at N = 2 while N = 1 is invariant ⇒ the
implemented clustered subproblem is path-dependent, hence not the printed convex
problem. Falsification at N = 1 as well ⇒ a conditioning defect affecting the
whole campaign, and the report's §2.6 finding on the β box would be promoted from
"recorded but not causal" to "causal".

**Cost:** ~20 inner solves at frozen designs. Days. **Information gain: high.**

---

### P1.4 — Invariance under eigenvector-basis rotation inside the clustered eigenspace

**Covers the brief's:** invariance with respect to eigenvector basis rotation.

**Why required.** [EV] The paper states it outright: "any linear combination of
the eigenvectors φ_j … will satisfy the generalized eigenvalue problem …, which
implies that the eigenvectors are not unique." Under Φ → Φ R with R ∈ O(N), the
generalized gradients transform as f'_sk = Σ_{a,b} R_{as} R_{bk} f_ab, so
F → RᵀFR, so the **spectrum of F, the feasible set of Eq. (25), and therefore
Δρ\* are mathematically invariant**. Any observed dependence is an implementation
defect and nothing else. The report notes that individual-mode MAC is not
basis-invariant and correctly refuses to rest conclusions on it (§6.2) — but it
never checked whether the *solver* is invariant, and §0.5 names a concrete
mechanism by which it might not be (the arbitrary `eig` tie-break at F = 0, which
picks out the diagonal f₁₁, f₂₂ in whatever basis `eigs` returned and discards
f₁₂ at the first linearization).

**Experiment.** Three nested levels, cheapest first, at F-E (exact multiplicity)
then F-A:

* **(a) Formulation level.** For 100 random R ∈ O(2) (plus the discrete subgroup:
  swaps, sign flips), rebuild f'_sk from Φ R and check that the spectrum of
  F'(d) = f'ᵀd equals that of F(d) for many random d.
  *Confirms:* agreement at ≤ 1e-12 relative. *Falsifies:* any O(1) dependence ⇒
  `compute_generalized_gradients.m` or the λ̄ handling is basis-dependent.
* **(b) Filter level.** Same, but with the Sigmund filter applied per component as
  the outer loop does (`topopt_freq_exact.m:322–330`). The filter is linear in the
  sensitivity for fixed ρ, so it must commute with the constant rotation.
  *Confirms:* ≤ 1e-12. *Falsifies:* a defect in the per-component filtering of the
  f_sk array — which would be invisible to every existing test, since
  `verify_multiplicity.m` tests the **unfiltered** array.
* **(c) Pipeline level — the strong test.** Solve Eq. (25) end-to-end from Φ R for
  each R and compare Δρ*, β*, and the post-step spectrum.
  *Confirms:* ‖Δρ*_R − Δρ*_I‖∞ below the inner tolerance for all R.
  *Falsifies:* O(1) dependence ⇒ the implemented subproblem is not the printed
  one, **and this alone would explain non-retention with no globalization
  argument required.** Under falsification, cross-reference P1.3: if the exact
  solver of P1.2 *is* rotation-invariant while MMA is not, the defect is localized
  to the inner solve.

**Cost:** hours (a, b); ~200 inner solves (c). **Information gain: high** — this
is the cleanest possible discriminator, because the correct answer is known
analytically in advance.

---

### P1.5 — Generalized eigenvalue sensitivities and generalized gradients against finite differences

**Covers the brief's:** generalized eigenvalue sensitivities, generalized
gradients, finite-difference verification.

**Why required.** Existing coverage is real but narrower than the report's §6.3
implies, and its gaps are exactly where the failure lives:

| Already verified | Where | Gap |
|---|---|---|
| ∂λ_j/∂ρ_e vs central FD, simple eigenvalues, j = 1,2 | `verify_sensitivity_filter.m` A | 40×5 mesh only; **well-separated modes only**; not at production meshes; not at a near-cluster |
| f_sk vs FD **of K and M** (fixed φ) | `verify_multiplicity.m` B | Verifies Eq. (19)'s **algebra** only. Does **not** verify that the μ_i of F predict actual eigenvalue changes — Eq. (18) is never tested |
| f_sk(:,1,1) ≡ `compute_elem_sensitivity` | `verify_multiplicity.m` C | Machine precision. Adequate |
| f_sk = f_ks | `verify_multiplicity.m` D | Adequate; extend per P1.6 |

**Experiment — three tiers.**

* **(a) Simple-eigenvalue FD, extended.** Central differences on λ_j, j = 1…4, at
  F-C, F-A and a mid-trajectory design, on both production meshes, with elements
  stratified by density regime (solid, void at ρ_min, grey, adjacent to supports).
  Sweep h ∈ [1e-9, 1e-3] to exhibit the FD plateau.
  *Confirms:* a clean plateau; relative error ≤ 1e-6 on it; O(h²) approach; no
  systematic degradation at ρ → ρ_min (which would indict the Eq. (4b) mass
  derivative in `mass_interp.m`).
  *Falsifies:* no plateau; error not O(h²); regime-dependent bias.
  *Expected and diagnostically valuable:* the error **must** grow as g₁₂ → 0,
  because the simple-eigenvalue formula is invalid at a cluster. **Record the gap
  at which it exceeds a declared model tolerance — this number is the
  non-arbitrary basis for `mult_tol` in Phase 3** (§3.M1).
* **(b) Directional-derivative test of Eq. (18) — the missing test.** At F-E
  (exact multiplicity) and F-A, for ≥ 30 random directions d and for
  d = the accepted Δρ*: compare the sorted actual increments
  [λ_i(ρ+td) − λ_i(ρ)]/t against the sorted μ_i(F(d)), over t ∈ [1e-8, 1e-2].
  *Confirms:* max_i relative error → 0 as t → 0 with **O(t)** convergence, at F-E
  to the FD noise floor. This is the end-to-end validation of the entire cluster
  model — f_sk, λ̄, Φ orthonormalisation, cluster assembly — and it is
  basis-invariant, so it is immune to the ambiguity that makes MAC useless here.
  *Falsifies:* no convergence, convergence to wrong values, or convergence only
  for directions with small f₁₂ᵀd (⇒ the off-diagonal terms are wrong).
* **(c) Filtered-gradient consistency.** The subproblem is built from **filtered**
  f_sk, so the object whose derivative property matters is the filtered one. For
  the `sensitivity` filter no exact chain rule exists (it is a heuristic), which
  is a paper-faithful choice but means (b) will *not* pass on filtered gradients.
  Run (b) on both raw and filtered arrays and **report the discrepancy as a
  measured quantity**: it bounds how far the subproblem model can be from the true
  directional derivative for reasons that are the paper's, not the
  reconstruction's. For the `density` filter the chain rule is exact
  (`physical_to_design_sensitivity.m`) and (b) **must** pass — that is the control
  proving the discrepancy is attributable to the filter heuristic and not to a bug.

**Cost:** hours to days. **Information gain: high for (b), moderate for (a) and (c).**

---

### P1.6 — Symmetry properties

**Covers the brief's:** symmetry properties.

**Why required.** Symmetry of f_sk follows from symmetry of K and M and is the
cheapest available detector of indexing and reshape errors — and `inner_loop_mma`
performs a non-obvious `reshape(fsk, nEl, N*N)` with a column-major convention
(`inner_loop_mma.m:80–81`) and a `kron(q_i, q_i)` contraction (line 189) whose
index order is asserted in a comment but never tested.

**Experiment.** At F-E and F-A: (i) max|f_sk − f_ks| ≤ 1e-12·max|f| (already
covered; extend to the production meshes and to filtered arrays); (ii) F(d) is
symmetric for random d and `eig` returns real eigenvalues — record the discarded
imaginary parts from the `real()` calls at `inner_loop_mma.m:165,167` and assert
they are ≤ 1e-12 relative, since a silently discarded non-negligible imaginary
part is an undetectable failure today; (iii) verify the reshape/kron contraction
directly: q_iᵀF_e q_i computed by the `kron` route must equal the explicit double
sum Σ_{s,k} q_i(s) f_sk(e) q_i(k) to machine precision; (iv) M-orthonormality
‖Φ_clᵀMΦ_cl − I‖ at every fixture, not only at the initial design where the report
records 2.9e-15.

**Confirms.** All four at machine precision.
**Falsifies.** (iii) failing ⇒ the constraint gradients are transposed or
mis-strided, which would corrupt exactly and only the N ≥ 2 path — invisible to
every N = 1 result in the campaign. This is a low-probability, catastrophic-impact
check, which is why it is cheap and mandatory rather than optional.

**Cost:** hours. **Information gain: low expected, but the check is nearly free
and its failure mode is fatal.**

---

### P1.7 — Reconstruction of λ̄ and of the (25c)/(25d) pairing

**Covers the brief's:** reconstruction of λ̄.

**Why required.** §0.4 fact 3: the paper writes (25c) against the individual ω_j²
under the premise of exact multiplicity; the reconstruction substitutes the
cluster mean and applies it to a near-cluster. The report lists this as an
untested open alternative (§6.3 item 2, §10 item 7, §12 A15) and asserts the
difference is O(mult_tol) — plausible for the *magnitudes*, but the assertion has
never been tested, and it says nothing about the **discrete** effect: which of the
N constraint rows binds, and which root of (25d) pairs with which mode.

**Experiment.** At F-A and F-B, solve Eq. (25) under four reconstructions,
changing nothing else:

| Variant | (25c) form | f_sk built with | Reduces to the printed equations when |
|---|---|---|---|
| **λ̄-mean** (current) | β ≤ λ̄ + μ_i(F) | λ̄ = mean | exact multiplicity |
| **λ̄-min** | β ≤ λ_n + μ_i(F) | λ̄ = λ_n | exact multiplicity |
| **shifted-diagonal** | β ≤ μ_i(diag(λ_n…λ_{n+N−1}) + F) | λ̄ = mean | exact multiplicity **and** N = 1 |
| **individual-paired** | β ≤ λ_j + μ_j(F), ascending pairing | λ̄ = mean | exact multiplicity |

Record β*, Δρ*, cos∠ between increments, the splitting rate (μ₂−μ₁)/λ̄ of P1.1,
and the realised post-step g₁₂.

**Confirms.** All four agree to O(mult_tol·λ̄) in β* and to cos ≥ 0.999 in
direction, and all four give the same retention outcome ⇒ the λ̄ substitution is
immaterial and this branch of H1 closes.
**Falsifies.** Materially different Δρ*, or different retention outcomes ⇒ the
choice is a live reconstruction ambiguity. Note the asymmetry to report honestly:
the **shifted-diagonal** variant is the only one that reduces exactly to the
printed equations in *both* limiting cases (exact multiplicity **and** N = 1), so
if it retains where the mean does not, the correct conclusion is "the
reconstruction of λ̄ was under-determined and one admissible reading works" —
**a Phase 1 result, not a globalization result, and still L1.**

**Cost:** ~8 inner solves. Days. **Information gain: moderate-high.**

---

### P1.8 — Reconstruction of the constraint system

**Covers the brief's:** reconstruction of the constraint system.

**Why required.** Five specific items in (25b)–(25f) are either unverified or
known-divergent, and one of them is a structural trap that fires exactly at a
near-cluster.

**Experiment — five independent audits, each at F-A, F-B, F-C:**

1. **(25b) with J = n+N — the headroom trap.** When N = 1 at a *near*-cluster,
   J = 2, so λ_J is the cluster's own twin and (25b) caps β at essentially λ̄
   itself. The archive already shows this: λ_J/λ̄ = **1.003** at the critical
   iteration 22 (§6.2 table). Under N = 2, J = 3 and the ratio is 1.127. Measure
   λ_J/λ̄ as a function of g₁₂ over the whole archive (P0.2 item 3) and at each
   fixture, and measure the resulting β* headroom directly.
   *Confirms benign:* headroom stays bounded away from 1 whenever the solver is
   near a cluster. *Falsifies:* headroom → 1 whenever g₁₂ → 0 under N = 1 ⇒ the
   under-detected configuration has a **structurally degenerate objective**, and
   "N = 2 activated too late" (Phase 3 hypothesis 1) acquires a concrete
   mechanism rather than remaining a label.
2. **ω_J assumed simple.** [EV] The paper assumes it; `detect_multiplicity.m` does
   not check it. Measure the gap |ω_{J+1} − ω_J|/ω_J at every iteration of the
   archive. *Falsifies:* if ω_J is itself frequently near-double, (25b) is the
   wrong model for the J-mode too, and the constraint system is misreconstructed
   independently of anything about the cluster.
3. **J_idx = 0 ⇒ (25b) silently dropped.** When the cluster reaches the top of the
   computed set, `lambda_J = Inf` and the J-row is omitted (`inner_loop_mma.m:77`),
   leaving β bounded only by the cluster rows. Count occurrences in the archive
   (P0.2 item 4). *Falsifies:* any non-zero count on a non-collapsed run ⇒ some
   recorded steps solved a **different constraint system** from the printed one.
4. **(25e) volume and (25f) box.** Confirm V_e-weighting is correct for uniform
   elements, the residual is the recorded −2.6e-4 order, and that the implemented
   box is (25f) **intersected with ±`outer_move`** — the latter being an L2
   addition that must never be present in an L1 experiment. *Confirms:* with
   `outer_move = Inf`, the MMA box equals (25f) exactly.
5. **The β̂ ≤ 1e6 cap and the λ_ref scaling.** Neither is in the paper. §2.6 shows
   the cap is not numerically inert. Verify the cap never binds at N = 2 (where
   an extra constraint row changes the conditioning) and that the solution is
   invariant to λ_ref (covered by P1.3).

**Cost:** mostly re-analysis (P0.2) plus a handful of solves. **Information gain:
high for items 1–3, low for 4–5.**

---

### P1.9 — Eigensolver accuracy inside the cluster

**Covers the brief's:** influence of the number of extracted eigenmodes *(jointly
with P1.10)*.

**Why required.** Everything in Phase 1 consumes Φ_cluster from `eigs(...,'SM')`
with `n_modes = 4`. At g₁₂ ~ 1e-3 the *individual* eigenvectors of a cluster are
ill-conditioned — their condition number scales like 1/gap — while the *invariant
subspace* remains well conditioned. If ARPACK's tolerance is not tight enough at
the fixtures, the pair handed to `compute_generalized_gradients` may not even span
the true invariant subspace, in which case every N = 2 result in the campaign is
about the wrong subspace. This has never been checked, and it is a prerequisite
for interpreting P1.4: rotation invariance matters *because* the returned basis is
arbitrary, so the pipeline must be invariant **and** the subspace must be right.

**Experiment.** At F-A, F-B, F-E recompute the lowest 4 eigenpairs by four
independent routes: (i) `eigs` as configured; (ii) `eigs` with tol 1e-14 and a
large Krylov dimension; (iii) shift-invert about a shift inside the cluster;
(iv) a dense reference on a coarsened but still near-degenerate model where a
dense solve is affordable. Compare (a) eigenvalues, (b) **principal angles between
the invariant subspaces**, (c) per-mode residuals ‖Kφ − λMφ‖/‖λMφ‖, (d) the
resulting f_sk arrays and Δρ*.

**Confirms.** Subspace principal angles ≤ 1e-8 across all routes; residuals ≤
1e-10; individual eigenvectors **may** differ by an O(1) rotation, which is
expected and harmless *if and only if* P1.4(c) passes.
**Falsifies.** Subspace angles O(1) ⇒ the cluster basis is numerically
meaningless at the tested gaps, all N = 2 conclusions in the report are void, and
the fix (tighter tolerance, shift-invert, subspace refinement) is an L1 correctness
fix. Note the interaction: a failure here would also invalidate the report's g₁₂
measurements themselves at small gaps, which are the sole basis of the retention
conclusions (§6.2) — making this the highest-leverage *hidden* risk in the
campaign.

**Cost:** hours. **Information gain: high** (low probability of failure, very high
impact, very low cost).

---

### P1.10 — Influence of the number of extracted eigenmodes

**Covers the brief's:** influence of the number of extracted eigenmodes.

**Why required.** `n_modes = 4` is reconstruction assumption A6, never varied
(§12). With n = 1 and N ≤ 2 it is *structurally* sufficient (J ≤ 3), so the naive
expectation is no effect — but three mechanisms make that expectation testable
rather than obvious: the Krylov subspace dimension affects `eigs` accuracy
(P1.9); N ≥ 3 makes J = 4 the last computed mode with zero margin; and N ≥ 4
triggers the dropped-J-constraint path of P1.8 item 3.

**Experiment.** n_modes ∈ {4, 6, 8, 12, 20} at F-A, F-B, F-C, F-E. Measure: ω₁…ω₄
and their residuals; the detected N and J; λ_J; the assembled f_sk; β*; Δρ*; and
the realised post-step spectrum. Then, separately, a **path-level** check: two
full V4 trajectories at n_modes = 4 and 12, all else identical.

**Confirms.** Frozen-design quantities invariant to ≤ 1e-10 for n_modes ≥ 4; any
trajectory difference attributable solely to eigensolver accuracy and vanishing as
tolerances tighten.
**Falsifies.** Material dependence at a frozen design ⇒ either an eigensolver
accuracy effect (⇒ P1.9) or a cluster-window effect (⇒ the modal window is part of
the formulation and A6 must be promoted from "untested" to "material"). A
path-level difference with invariant frozen-design quantities is **not** evidence
of a defect — it is evidence of trajectory sensitivity, which is a Phase 2 finding
and must be reported as such.

**Cost:** low at frozen designs; two trajectories otherwise. **Information gain:
low-moderate**, but it closes a named open alternative (§6.3 item 3) cheaply.

---

### Phase 1 exit criterion

Phase 1 **passes** only if: P1.5(b) converges at O(t) on F-E; P1.4(c) is invariant;
P1.9 subspace angles are ≤ 1e-8; P1.2 certifies MMA against the exact solve; P1.3
shows path-independence; P1.6 passes at machine precision; and P1.7, P1.8, P1.10
show no material dependence. Anything less and the correct next action is a
correctness fix at L1 followed by re-entry — **not** Phase 2.

---

## Phase 2 — Isolation experiments (assume Phase 1 passes)

**Precondition.** If P1.1 falsified H2 (the cluster splits at first order along
the Eq. (25) direction), Phase 2 answers a question already settled and must not
be run. The gate is explicit.

### 2.0 The four distinct mechanisms currently conflated

The report's central methodological complaint about its own Regime-B evidence is
that "the Regime-B comparison changes `move_lim`, `outer_move` and `alpha`
together" (§0, [B] bullet). The code exposes **four** separable mechanisms, not
three, and they act at different points of the algorithm:

| Mech. | Symbol | Where it acts | What it changes |
|---|---|---|---|
| **A** | `outer_move` | MMA variable bounds `xmin/xmax` (`inner_loop_mma.m:118–122`) | The **feasible set of the subproblem** ⇒ changes Δρ* **direction and length** |
| **B** | `alpha` | Outer update ρ := ρ + α·Δρ (`topopt_freq_exact.m:373–375`) | **Scalar length only**, along a **fixed** direction |
| **C** | `post_coalescence_trust_*` | Contracts A and the inner clamp when the gap is small (`topopt_freq_exact.m:307–316`) | Adaptive radius, gap-triggered. Default-disabled; never enabled in the campaign |
| **D** | `move_lim` | Per-**inner**-iteration clamp on the MMA iterate (`inner_loop_mma.m:217–220`) | The **inner solver's path**. Predicted inert at inner convergence for N = 1 (P1.3) |

**A and B are not the same experiment.** A changes the direction; B does not. This
is why the brief's "move limits" and "line search / step scaling" are genuinely
orthogonal here and must not be merged. **D is not a trust region at all** and
must be neutralised (`move_lim = Inf`, inner budget ≥ 2000 so the declared inner
stopping test is met) in every Phase 2 run, or it silently confounds A and B —
this is the same confound that §2.3 of the report identified in the inner-budget
sweep, where changing the budget changed the increment's direction and component
distribution as well as its magnitude.

### 2.1 Two experiment classes, with different inferential status

| Class | Setup | What it supports |
|---|---|---|
| **Local (controlled)** | Start from a frozen fixture (F-A/F-B), take 1 and then 8 steps | **Causal** claims about retention. Initial condition identical across arms; no path confounding |
| **Path-level (descriptive)** | Full trajectory from the uniform design | **Descriptive** only. Changing any step control changes the whole path, so "did it retain a cluster" is confounded with "did it ever reach one" |

Both are needed; only the first supports causal language. The report's existing
retention evidence is entirely of the second kind, which is precisely why its
conclusions had to be withdrawn to [B].

### 2.2 Response variables (fixed in advance, identical across all arms)

Primary: **retention length** R = number of consecutive outer iterations with
g₁₂ ≤ g_ref, where g_ref is *derived in Phase 3 (§3.M3)*, not chosen now.
Secondary: post-step g₁₂; realised Δλ₁; `pred_ratio`; ‖Δρ‖∞ / box width
(saturation); (μ₂−μ₁)/λ̄; terminal ω₁; tail design-change statistics; number of
distinct near-cluster events per run.

### 2.3 Experiment A — subproblem move limit only

* **Independent variable:** `outer_move` ∈ {Inf, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01,
  0.005}. Geometric, spanning two decades below the recorded 0.2.
* **Controlled (identical in every arm):** fixture/initial design; mesh; p and
  continuation schedule; filter type and radius; ρ_min; `mult_tol`; `n_modes`;
  `inner_tol`; inner budget ≥ 2000; `move_lim = Inf`; **`alpha = 1`**; mechanism
  C disabled; acceptance gates disabled; outer budget.
* **Expected observations.** If the model has a finite validity radius (which
  §2.5 of the report measured for ω₁ at the initial design, and P1.1 measures for
  g₁₂ at F-A), retention length R should be **monotone non-decreasing** as the
  radius shrinks, up to a threshold, then flat. Saturation fraction should stay at
  100 % (the LP structure predicts vertex solutions for N = 1) while ‖Δρ‖∞ tracks
  the box.
* **Interpretation.**
  * Monotone improvement with a clear knee ⇒ step length is causal for retention;
    the knee locates the required radius; **conclusion is L2 and must be reported
    as an addition to the paper.**
  * R = 1 at every radius including 0.005 ⇒ the subproblem box cannot retain the
    cluster. Combined with P1.1, this is strong evidence against H2.
  * Non-monotone / knife-edge ⇒ retention is not a smooth function of the radius;
    report as such and do not tune. A non-monotone response is itself a finding:
    it indicates the trajectory, not the step, controls the outcome.

### 2.4 Experiment B — step scaling only

* **Independent variable:** `alpha` ∈ {1, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01,
  0.005, 0.001}.
* **Controlled:** everything in 2.3, with **`outer_move = Inf`** (so the
  subproblem is exactly (25f) — this arm is the closest to L0/L1 of any in Phase 2)
  and `move_lim = Inf`.
* **Why this is the decisive arm.** B alone holds the **direction** fixed at the
  Eq. (25) solution and varies only the distance travelled. It is therefore the
  discrete-trajectory counterpart of the continuous profile in P1.1, and the two
  must agree. Disagreement between B and P1.1 would itself indicate that the
  re-solve at the new design, not the step length, drives the outcome.
* **Expected observations.** If P1.1 found a finite t*, then B should retain the
  cluster for α ≲ t*, and R should rise sharply at that α.
* **Interpretation.**
  * Retention appears below a threshold α ⇒ pure step length; H2 supported;
    conclusion L2.
  * **No retention as α → 0** ⇒ **H2 is falsified in its strongest form.** No
    globalization built on shortening the Eq. (25) step can work, because in the
    limit α → 0 every such method reduces to this arm. Return to Phase 1/H3/H4.
  * Retention at small α but the trajectory stalls (ω₁ stops rising) ⇒ retention
    and progress are in conflict under a fixed scaling, which is the specific
    condition that motivates — but does not by itself justify — an adaptive
    mechanism, i.e. Experiment C.

### 2.5 Experiment C — trust-region adaptation only

* **Independent variable:** the adaptation **rule**, with its fixed base radius
  held at the value identified in A and its scaling factor held fixed:
  C0 = none (control, = the A arm at that radius); C1 = contract on `pred_ratio`
  below a declared value, expand above; C2 = contract on the *gap* trigger, i.e.
  the existing default-disabled `post_coalescence_trust_*` mechanism; C3 =
  contract on realised Δλ₁ < 0.
* **Controlled:** everything in 2.3–2.4, **including the base radius and the
  contraction factor**, so that only the *trigger* varies. Any arm that also
  changes the base radius is invalid and must be discarded.
* **Expected observations.** C1 and C3 respond to *model quality*, C2 to
  *proximity to a cluster*. If retention improves under C2 but not C1/C3, the
  mechanism is cluster-specific, not model-quality-specific — which points at
  Phase 3 (the model switch), not at globalization.
* **Interpretation.** Any positive result here is **L3** and is an algorithmic
  modification beyond the paper. It answers "what would make this work", never
  "what did the paper do".

### 2.6 Orthogonality, stated as a constraint and as an assumption

The brief forbids changing multiple mechanisms simultaneously, and every arm above
obeys that: one independent variable per arm, all four mechanisms of §2.0 pinned
in every other arm, and D neutralised throughout. One methodological caveat must
be recorded rather than silently assumed: **one-factor-at-a-time designs are valid
only if the mechanisms do not interact.** That assumption is itself testable with
four runs at the corners (A ∈ {0.2, 0.02}) × (B ∈ {1, 0.1}). It is recommended as
a *confirmatory* experiment after the OFAT sweeps complete, not as a substitute
for them; until it is run, all Phase 2 conclusions are local to the baseline
values of the un-manipulated controls, and must be stated that way.

---

## Phase 3 — Cluster preservation

The brief lists five candidate causes. **None of them can be tested before the
measurement programme below, and no threshold may be chosen before M1–M8 are in
hand.** That is the whole point of this phase: the report's withdrawn V1 claim
("the smallest step the procedure can take is larger than the basin of the bimodal
state") failed precisely because it compared a move limit to a basin width **that
was never measured** (§6.3). Phase 3 measures it.

### 3.1 Measurement programme — what must be known before any threshold is chosen

| # | Measurement | Method | Threshold it determines |
|---|---|---|---|
| **M1** | **N = 1 model-error curve** ε₁(g, t): relative error of the simple-eigenvalue prediction vs. the true Δλ₁, as a joint function of the eigengap g and the step length t | P1.5(a) at fixtures spanning g ∈ [1e-4, 1] | **`mult_tol`**: the gap below which the N = 1 model is inadmissible for the step lengths actually taken |
| **M2** | **Coupling curve** ‖f₁₂‖/‖f₁₁‖ as a function of g | Arithmetic at fixtures + archive | Whether Eq. (22) ever holds; the archive already gives 1.0154 at F-A, i.e. emphatically not |
| **M3** | **Cluster basin width** in design space: g₁₂(ρ + t d) for d = Δρ*(N=1), Δρ*(N=2), the exact-solve direction, and random d | P1.1 profiles, extended to several fixtures | **g_ref** (the retention criterion of §2.2) and the **maximum admissible step** for retention |
| **M4** | **Subspace stability**: principal angles between the cluster invariant subspace at ρ and at ρ + t d | Along the same profiles | A **basis-invariant** replacement for MAC, whose unreliability near multiplicity the report correctly flags (§6.2) |
| **M5** | **Model-switch magnitude**: ∠(Δρ*_{N=1}, Δρ*_{N=2}) and β*_{N=1} − β*_{N=2}, as a function of g | Paired solves at fixtures spanning g | The size of the discontinuity a hysteresis band would have to absorb |
| **M6** | **J-mode discontinuity**: change in the subproblem optimum induced solely by J flipping from n+1 to n+2 | Solve with N = 2 and J forced to each value | How much of M5 is the cluster model and how much is the J-mode swap |
| **M7** | **Detection-flip statistics** on a tolerance grid | P0.2 item 2 — **free, from the archive** | Whether flapping occurs at *any* tolerance; if it never does, "missing hysteresis" is falsified before any experiment is designed |
| **M8** | **Headroom curve** λ_J/λ̄ vs g₁₂ | P0.2 item 3 + P1.8 item 1 — **free, from the archive** | Whether the N = 1-at-a-near-cluster configuration is structurally degenerate |

### 3.2 Mapping the five candidate causes onto measurements

| Candidate cause | Testable only after | Falsified by |
|---|---|---|
| **N = 2 activated too late** | M1, M8 | ε₁(g, t) small at the gaps actually reached ⇒ N = 1 was an adequate model ⇒ activation timing is irrelevant. Conversely, headroom λ_J/λ̄ → 1 under N = 1 near a cluster (M8) would give this cause a concrete mechanism |
| **Exits N = 2 too early** | M3, M7 | If the gap re-opens by more than the detection tolerance in a *single admissible* step (M3), exiting is **forced by the dynamics**, not by an early exit rule, and this cause is mis-stated |
| **Hysteresis missing** | M5, M6, M7 | If M7 shows the solver never enters N = 2 twice at any tolerance, there is nothing to flap between, and hysteresis is a solution to a non-problem |
| **Accepted step too large** | M3 + Phase 2 A/B | Phase 2 B reaching α → 0 without retention |
| **Detection too sensitive** | M1, M7 | If `mult_tol` derived from M1 differs from 1e-3 by less than the measured spread of g₁₂, sensitivity is immaterial |

### 3.3 The prior question these measurements settle

M3 and the archive re-analysis (P0.2 item 6) answer something logically upstream
of all five candidates: **is a cluster ever approached continuously, or only
jumped into?** Both observed events arrived on steps with *negative* realised Δλ₁
(pred_ratio = −0.131 at 160×20; −1.573 at 240×30) — the solver fell into the
cluster rather than converging onto it. If the approach profile shows no monotone
narrowing of g₁₂ before either event, then "activated too late" and "exits too
early" are **category errors**: the algorithm never had a clustered *regime* to
enter or leave, and the right question becomes why the ascent never approaches
coalescence at all. That reframing would be a Phase 3 result of the first
importance, and it is obtainable from data already on disk.

### 3.4 Rule for choosing thresholds

Once M1–M8 exist, thresholds are **derived**, and the derivation is published with
the value:

* `mult_tol` := the largest g at which ε₁(g, t) exceeds a **declared** model
  tolerance at the step lengths actually taken. Derived from M1. If the resulting
  value coincides with the current 1e-3, that is a confirmation, not a
  coincidence to be tuned away.
* g_ref (retention criterion) := derived from M3 as the gap below which the
  invariant subspace is stable (M4) over an admissible step.
* Hysteresis band [g_on, g_off], **if Phase 3 shows one is needed**: g_off must
  exceed g_on by at least the measured single-step re-opening from M3, since a
  narrower band guarantees flapping by construction. This is a derivation, not a
  tuning.
* Any threshold that cannot be derived from M1–M8 is **not selected**; the
  corresponding candidate cause is reported as untestable with present evidence.

---

## Phase 4 — Acceptance mechanism

### 4.1 The question, disambiguated

"Should the optimizer reject steps after solving the MMA subproblem?" has two
distinct answers depending on which question is being asked, and conflating them
is how a reconstruction becomes an invention:

* **As a matter of fidelity:** **No.** [EV] Fig. 1 box 4 is an unconditional
  update; the paper has no acceptance test. Adding one leaves L1 for L3.
* **As a matter of what would make the printed formulation work on this forward
  model:** an open empirical question, answerable only at L3 and reportable only
  as "the printed formulation **plus** X".

### 4.2 The move that is not a modification — do this first

Computing an acceptance criterion and **recording** it while accepting
unconditionally is **instrumentation, not modification**. It changes no behaviour
(verifiable by bit-identity against the current runs, exactly as test T2 already
does for the two code paths), and it yields the entire acceptance dataset. Every
acceptance rule can then be evaluated **counterfactually offline**: "how many
steps would rule X have rejected, at which iterations, with what g₁₂?" — with no
solver modification at all.

Much of this already exists: `pred_ratio` is recorded (median 0.2 % of predicted
improvement, wrong sign in 144/300 iterations, §7.3), as are trial spectra
(`omega_trial`) and the fail-closed predicates. **The correct first action in
Phase 4 is P0.2 item 7 — an offline counterfactual evaluation of all four
acceptance rules on data already on disk.** It costs nothing and it bounds how
much any of them could possibly help before a single line is changed.

### 4.3 Comparison of the four mechanisms

| | Mechanism | Cost per outer iteration | Merit function required | Behaviour at a cluster | Ladder |
|---|---|---|---|---|---|
| **(a)** | **Current: unconditional accept** | 0 | none | Accepts the step that re-opens the gap ×236 | **L1** (paper) |
| **(b)** | **Classical line search** (backtracking/Armijo along Δρ*) | 1 eigensolve per trial (typically 1–4) | Yes — and this is the difficulty | Reduces to Phase 2 arm B with an adaptive α; retains iff B retains | **L3** |
| **(c)** | **Trust region** (radius adapted on the outcome; step **re-solved** at the new radius) | 1 eigensolve + 1 **re-solve** per rejection | Yes | Differs from (b) in that the direction is recomputed inside the smaller box — the only mechanism that can produce a direction the full-box subproblem never offers | **L3** |
| **(d)** | **Predicted/actual reduction test** (accept iff ared/pred ≥ η) | 1 eigensolve | Implicitly λ_n | Pure gate; on the archive it would have rejected the majority of steps (`pred_ratio` median 0.002, negative in 48 % of iterations) | **L3** |

**The merit-function problem must be settled before (b), (c) or (d) is run.** The
outer objective is max λ_n, but a step is computed by maximizing a *lower bound* β
under constraints. Three candidates, in decreasing defensibility:
λ_n(ρ) itself (continuous in ρ, non-smooth at crossings — acceptable for
backtracking, and it is the quantity the paper maximizes); λ_n plus a volume
penalty (needed only if volume residuals grow beyond the recorded −2.6e-4); β
itself (**rejected** — β is the model's prediction, so using it as the merit
function makes the test self-referential and vacuous). Declare the choice, and
report the sensitivity of every acceptance result to it.

**Note the structural difference between (b) and (c)**, which matters more here
than the usual textbook distinction: at F-A both the N = 1 and the forced-N = 2
increments **saturated** the box (‖Δρ‖∞ = 0.2000 exactly). A line search only
scales a saturated direction. A trust region re-solves inside a smaller box, and
if the subproblem's optimum is a vertex (as §2.4 established for N = 1), shrinking
the box moves the vertex — a *different* direction, not a shorter one. If P1.1
falsifies H2 (the direction splits the cluster at first order), then **(b) and (d)
are dead and only (c) could possibly work**, because only (c) changes the
direction. This is the sharpest practical consequence of the P1.1 result and
should determine which of these is implemented at all.

### 4.4 Verdict on scientific justification

| Mechanism | Justified for this reconstruction? |
|---|---|
| Inner-loop convergence gate (fail-closed) | **Yes — L1.** It enforces the paper's own Fig. 1 inner diamond. Already implemented and regression-tested (T3) |
| (25e), (25f) enforcement | **Yes — L1.** Printed constraints |
| Eq. (22)-constrained LP reduction | **Yes — L1.** [EV] Explicitly offered by the paper, §3.5.3 |
| Recording pred/ared without acting | **Yes — instrumentation, not a mechanism.** Do this first |
| A fixed step restriction (`outer_move`, `alpha`) | **Declared addition — L2.** Currently supported at [B] as necessary to produce any iteration history; it is **not** in the paper and must never be reported as part of the reconstruction |
| Line search, trust region, pred/ared acceptance | **Additions — L3.** Legitimate as answers to "what would make it work"; illegitimate as evidence about what the paper contained |
| Multiplicity hysteresis, cluster-triggered contraction | **Additions — L4.** Same |

**The symmetric statement the evidence will support**, whichever way Phase 2 and 4
come out, is this: the paper supplies a **first-order directional-derivative
model** and explicitly warns that multiple eigenvalues "do not admit a usual
linearization", yet specifies **no radius of validity** for it. Two readings are
equally consistent with the printed text — (R1) the original computation had a
restriction that went unprinted, (R2) it had none and something else in the
forward model made the full-box step benign. The plan discriminates between them
only weakly, and the discriminating experiment is stated in §5 node D14: if **no**
fixed or adaptive radius reproduces the paper's Fig. 4 (a smooth ~100-iteration
ascent to 456.4), R1 is weakened and H3 or H4 rises. This is the strongest
available inference, and it is still [HYP], not [CONC] — documentary evidence
about the 2007 implementation would be required to close it, and this campaign
does not have it.

---

## Phase 5 — Decision tree

Every node has a falsifying branch that assigns the defect to the reconstruction
and one that assigns it elsewhere. `⇒` marks a conclusion; `→` an action.

```
P0.3  Determinism
  ├─ FAIL → fix eigensolver seeding; re-baseline. Every existing single-run
  │         comparison carries an unquantified run-to-run component.
  └─ PASS ↓

P0.1/P0.2  Fixtures reload; archive re-analysis
  ├─ Archive disagrees with recomputed spectra
  │     ⇒ §6 of the report rests on an archive that cannot be reproduced.
  │       → resolve before anything else.
  └─ PASS ↓

P1.1  Cluster-splitting rate along the accepted direction
  ├─ (μ₂−μ₁)/λ̄ ≪ g₁₂  AND  g₁₂(t) flat for t ≲ t* > 0
  │     ⇒ the Eq. (25) direction preserves the cluster locally; the
  │       accepted step overshoots a MEASURED radius.
  │     → H2 alive. Continue Phase 1 (correctness still unproven), then Phase 2.
  │
  ├─ (μ₂−μ₁)/λ̄ ≳ g₁₂  AND  g₁₂ rises linearly from t → 0⁺
  │     ⇒ H2 FALSIFIED. No step-shortening globalization can retain the cluster.
  │     → skip Phase 2 arms A/B; go to P1.2 to decide SOLVE vs FORMULATION:
  │         ├─ P1.2: exact solve of Eq. (25) gives μ₁ ≈ μ₂ (cluster preserved)
  │         │     ⇒ the FORMULATION preserves clusters; the SOLVER does not.
  │         │     ⇒ H1 CONFIRMED, localized to the inner solve.
  │         │     → replace inner solve with the certified one (L1 fix); re-test.
  │         └─ P1.2: exact solve also splits (μ₁ ≪ μ₂ at the optimum)
  │               ⇒ the printed Eq. (25), correctly solved, does not preserve
  │                 this cluster on this forward model.
  │               ⇒ H1 and H2 both falsified for this design → go to D13.
  │
  └─ Mixed (flat for random d, splits for Δρ*)
        ⇒ the subproblem SELECTS a splitting direction — the strongest possible
          pointer at the objective/constraint structure rather than at step
          length. → P1.2 and P1.8 item 1 (headroom) before anything else.

P1.9  Eigensolver subspace accuracy
  ├─ subspace angles O(1) at g ~ 1e-3
  │     ⇒ the cluster basis is numerically meaningless; ALL N = 2 results in the
  │       report are void, and the g₁₂ measurements underlying the retention
  │       conclusions are suspect. → L1 fix (tolerance/shift-invert/refinement),
  │       re-run P1.1, re-enter the tree.
  └─ PASS ↓

P1.5(b)  Directional derivatives of Eq. (18)
  ├─ FAIL on F-E (EXACT multiplicity)
  │     ⇒ implementation defect in f_sk / λ̄ / cluster assembly. H1 CONFIRMED.
  │     → fix, re-verify, restart Phase 1.
  ├─ PASS on F-E, FAIL on F-A (near-cluster only)
  │     ⇒ NOT a defect: the Seyranian model is inapplicable at that gap.
  │     ⇒ feeds M1 directly; the correct response is a DERIVED mult_tol
  │       (Phase 3), not a code change.
  └─ PASS on both ↓

P1.6  Symmetry / reshape / kron contraction
  ├─ FAIL ⇒ constraint gradients mis-strided on the N ≥ 2 path only.
  │         H1 CONFIRMED. → fix; every N = 2 result to date is void.
  └─ PASS ↓

P1.4  Rotation invariance
  ├─ (a) or (b) FAIL ⇒ basis-dependent gradients or filtering. H1 CONFIRMED.
  ├─ (c) FAIL only  ⇒ the SOLVE is basis-dependent (candidate mechanism:
  │                   the degenerate F(0) tie-break). H1 CONFIRMED, localized.
  │                   → this alone explains non-retention; globalization is a
  │                     red herring until it is fixed.
  └─ PASS ↓

P1.3  Well-posedness (path-independence of Δρ*)
  ├─ N = 2 path-dependent, N = 1 invariant
  │     ⇒ the clustered subproblem as implemented is not the printed convex
  │       problem. H1 CONFIRMED. → adopt the certified solver (L1 fix).
  ├─ BOTH path-dependent ⇒ conditioning defect affecting the whole campaign;
  │     §2.6 (β box) promoted from "recorded" to "causal". → fix, re-baseline.
  └─ PASS ↓

P1.2  MMA vs exact solution of Eq. (25)
  ├─ MMA solves a RELAXATION (β*_MMA > β*_exact, true constraint violated)
  │     ⇒ H1 CONFIRMED. → certified solver (L1 fix); re-test retention:
  │         ├─ cluster now retained ⇒ published mathematics SUFFICIENT;
  │         │        the earlier failure was an inner-solver defect. [A]
  │         └─ still not retained ⇒ continue with a now-verified subproblem.
  └─ AGREES ↓

P1.7 / P1.8 / P1.10  λ̄, constraint system, n_modes
  ├─ shifted-diagonal λ̄ retains where the mean does not
  │     ⇒ the λ̄ reconstruction was under-determined; one admissible reading of
  │       the printed (25c) works. ⇒ Phase 1 result at L1, NOT globalization.
  ├─ headroom λ_J/λ̄ → 1 under N = 1 near clusters (M8)
  │     ⇒ the under-detected configuration is structurally degenerate:
  │       (25b) caps β at the cluster's own twin. ⇒ "activated too late" gains a
  │       mechanism → Phase 3 with a DERIVED mult_tol.
  ├─ ω_J itself near-double, or J_idx = 0 observed on non-collapsed runs
  │     ⇒ some recorded steps solved a DIFFERENT constraint system than printed.
  │       H1 CONFIRMED for those iterations. → fix; re-run affected series.
  └─ ALL PASS ⇒ **PHASE 1 PASSES: the reconstruction of the published
                 mathematics is verified.** → Phase 2.

Phase 2 (only if P1.1 left H2 alive)
  A (outer_move only)
   ├─ monotone improvement with a knee ⇒ step length causal; radius located.
   │        ⇒ conclusion is L2 and must be labelled an addition to the paper.
   └─ R = 1 at every radius ↓
  B (alpha only, outer_move = Inf — closest arm to L1)
   ├─ retention below a threshold α ⇒ pure step length; H2 SUPPORTED.
   │     └─ but trajectory stalls (ω₁ stops rising)
   │           ⇒ retention and progress conflict under a FIXED scaling
   │           → C is motivated (L3), and must be reported as such.
   ├─ NO retention as α → 0
   │     ⇒ H2 FALSIFIED (this arm is the limit of every step-shortening method).
   │     → D13.
   └─ B disagrees with the P1.1 continuous profile
         ⇒ the re-solve at the new design, not the step length, drives the
           outcome → back to Phase 1 with the intermediate designs as fixtures.
  C (adaptation rule only)
   ├─ C2 (gap trigger) works, C1/C3 (model-quality triggers) do not
   │     ⇒ the mechanism is cluster-specific, not globalization-generic
   │     → Phase 3 (model switch), not Phase 4.
   └─ any positive result ⇒ L3. Answers "what would make it work", never
        "what the paper did".

D13  Phase 1 passed AND Phase 2 exhausted without retention
  → Test H3 before concluding anything about the paper:
     vary the FORWARD model one factor at a time (filter type and radius, mass
     interpolation Eq. (4b) vs alternatives, ρ_min, support geometry, element
     formulation), holding the verified L1 algorithm fixed.
   ├─ some forward-model variant retains the cluster and ascends
   │     ⇒ H3 CONFIRMED: the discrepancy is in the reconstruction of the MODEL,
   │       not of the OPTIMIZER and not in the paper's algorithm. This is the
   │       branch the report flags as its principal residual caveat (§12).
   └─ no variant retains ↓

D14  Discriminating R1 from R2 (the only test bearing on the historical claim)
  → Does ANY fixed or adaptive radius reproduce the paper's Fig. 4 —
    a smooth ~100-iteration ascent to ω₁ ≈ 456.4 with a retained bimodal state?
   ├─ YES ⇒ the printed formulation plus an unprinted step restriction
   │        reproduces the published computation.
   │        ⇒ [B] Evidence supports R1: the published formulation is INCOMPLETE
   │          in the specific, identified sense that it omits a step restriction.
   │          Still [HYP] as to what the 2007 code contained; documentary
   │          evidence would be required to close it.
   └─ NO  ⇒ no restriction reproduces Fig. 4.
          ⇒ H4 is the surviving hypothesis for the tested forward model:
            the printed mathematics, verified and correctly solved, does not
            reproduce the published result here. The residual disjunction —
            forward model vs. printed formulation — is exactly the one §11 of
            the report already states it cannot close, and this campaign
            would not close it either. Report as [HYP], never as [CONC].

TERMINAL SUMMARY OF ADMISSIBLE CONCLUSIONS
  T1  Phase 1 fails at any node
        ⇒ "The reconstruction of the published N ≥ 2 mathematics was defective
           at <node>." Fix at L1. NO conclusion about the paper is licensed.
  T2  Phase 1 passes, Phase 2 A or B retains
        ⇒ "Published mathematics is sufficient; the reconstruction requires a
           step restriction the paper does not print." [B] pending D14.
  T3  Phase 1 passes, only Phase 2 C retains
        ⇒ "A FIXED restriction is insufficient; an ADAPTIVE one is required."
           L3. The contraction hypothesis of §11 would be tested — and this is
           the ONLY route by which it can be promoted above [C].
  T4  Phase 1 passes, Phase 2 exhausted, D13 retains
        ⇒ "The obstacle is in the forward model, not the optimizer." H3.
  T5  Phase 1 passes, Phase 2 and D13 exhausted, D14 = NO
        ⇒ "On this verified forward model, the printed formulation does not
           reproduce the published bimodal result by any tested means." H4.
           This is the strongest NEGATIVE conclusion available and it remains a
           statement about the tested reconstruction, not about the 2007 code.
  T6  Any branch where a fix at L1 restores retention
        ⇒ every prior conclusion drawn from N = 2 runs must be re-derived, and
           §6, §11 and C4 of the report amended accordingly.
```

---

## 6. Priority ordering by expected information gain

"Discriminates" = the hypotheses the item can separate in a single run.
Cost is order-of-magnitude effort, not wall-clock.

| Rank | Item | Cost | Discriminates | Why here |
|:--:|---|:--:|---|---|
| 1 | **P0.2** archive re-analysis | hours | M7, M8, all Phase 3 priors | Free. Answers eight questions from data already on disk. Nothing else should start first |
| 2 | **P1.1** splitting rate along the accepted direction | hours | **H1 vs H2, decisively** | The whole brief in one experiment; can kill Phase 2 before it is designed |
| 3 | **P1.9** eigensolver subspace accuracy | hours | H1 (hidden) | Low probability, catastrophic impact, trivial cost. Its failure would void §6 of the report |
| 4 | **P1.6** symmetry / kron contraction | hours | H1 (fatal, N ≥ 2 only) | Same profile: nearly free, invisible-if-wrong, corrupts only the clustered path |
| 5 | **P0.1** fixtures, incl. the exact-multiplicity control F-E | hours | prerequisite | Nothing downstream is interpretable without it |
| 6 | **P1.4** rotation invariance | hours→days | H1, localized to solve vs gradients | Correct answer known analytically in advance; a failure explains non-retention with no globalization argument |
| 7 | **P1.2** exact solution of Eq. (25) at N = 2 | days | **H1 vs H2/H4** | The N = 2 counterpart of the campaign's strongest existing method (§2.4). Also reports whether the printed formulation equalizes cluster increments at all |
| 8 | **P1.5(b)** directional derivatives of Eq. (18) | days | H1, and supplies M1 | End-to-end, basis-invariant validation of the cluster model; supplies the non-arbitrary basis for `mult_tol` |
| 9 | **P1.3** well-posedness | days | H1 | The N = 1/N = 2 asymmetry prediction is a clean, falsifiable discriminator |
| 10 | **P1.8** constraint-system audit (esp. headroom, ω_J simplicity, dropped J) | hours→days | H1 | Item 1 gives "activated too late" a mechanism; items 2–3 are cheap and would void specific iterations |
| 11 | **P1.7** λ̄ variants | days | H1 (reconstruction ambiguity) | Closes a named open alternative; shifted-diagonal is the only reading exact in both limits |
| 12 | **P1.5(a)/(c)** FD extension + filter consistency | days | H1/H3 boundary | Quantifies how much model error is the paper's filter heuristic rather than the reconstruction's |
| 13 | **P1.10** n_modes | days | H1 (weak) | Closes §6.3 item 3 cheaply; likely null at frozen designs |
| 14 | **Phase 2 B** (alpha only) | days | **H2, decisively** | The α → 0 limit of every step-shortening method; run before A because it is closer to L1 and its null result is stronger |
| 15 | **Phase 2 A** (outer_move only) | days | H2 | Locates the radius if B succeeds; changes direction as well as length, so it is the weaker isolation |
| 16 | **Phase 3** M1–M6 (M7, M8 already done at rank 1) | days | Phase 3 candidates | Cannot start before P1.1/P1.5; supplies every threshold |
| 17 | **Phase 4** offline counterfactual on the archive | hours | acceptance-rule bounds | Free; bounds the benefit of (b)/(c)/(d) before any code changes |
| 18 | **Phase 2 C** / **Phase 4 (b)(c)(d)** active | weeks | "what would work" (L3) | Last: answers a different question and cannot bear on fidelity |
| 19 | **D13** forward-model variation (H3) | weeks | H3 vs H4 | Only after the optimizer is exonerated, or the result is uninterpretable |

**Sequencing rule.** Ranks 1–6 are hours-to-days and jointly capable of killing
either hypothesis. **None of ranks 14 onward may begin before rank 2 (P1.1)
reports**, because P1.1 determines whether Phase 2 is a meaningful experiment or
a search along a direction already known to be wrong.

---

## 7. Reporting rules

Binding on every result produced under this plan, and inherited from the report's
§0.0:

1. Every claim carries an evidence grade [A]/[B]/[C] and names its object — the
   paper-explicit formulation, the tested reconstruction, or the historical
   implementation — which are never conflated.
2. Every result carries its **ladder level** (§0.3). A retention obtained at L2 is
   never reported as a property of the paper.
3. **Falsification is reported as prominently as confirmation.** A Phase 1 item
   that passes closes a named open alternative from §6.3 and must be recorded as
   closing it.
4. No threshold is reported without its derivation (§3.4). A value that was tuned
   is labelled tuned, and any conclusion resting on it is [C].
5. Sample size is stated with every retention claim. The campaign currently has
   **two** near-cluster events across 19 runs; frozen-fixture experiments raise
   the number of *observations* but not the number of *independent events*, and
   conclusions must be scoped accordingly.
6. Individual-mode MAC is not used as evidence anywhere (§6.2). Basis-invariant
   diagnostics only: the relative eigengap g₁₂ and the subspace principal angles
   of M4.
7. If any Phase 1 item forces an L1 correctness fix, every affected conclusion in
   `faithful_reconstruction_report_v2.md` — §6, §11 sub-verdicts, C4 — is
   re-derived, not patched.

---

## Appendix — quantities and their definitions

| Symbol | Definition | Where measured |
|---|---|---|
| g₁₂ | \|ω₂ − ω₁\| / max(ω₁, ε) — relative eigengap, **basis-invariant** | `multiplicity_history.csv` |
| F(d) | N × N matrix, F_sk = f_skᵀd | arithmetic from `fsk` |
| μ_i(d) | ascending eigenvalues of F(d) = the roots of Eq. (25d) | arithmetic |
| (μ₂−μ₁)/λ̄ | first-order cluster splitting rate; = 2√(¼(F₁₁−F₂₂)²+F₁₂²)/λ̄ | P1.1 |
| ‖f₁₂‖/‖f₁₁‖ | off-diagonal coupling; Eq. (22) holds iff ≈ 0. **1.0154 at F-A** | M2 |
| λ_J/λ̄ | headroom of the (25b) cap. **1.003 at F-A under N = 1**; 1.127 under N = 2 | M8 |
| pred_ratio | realised Δλ₁ / predicted Δλ₁ | `outer_history.csv` |
| t* | largest t with g₁₂(ρ+tΔρ*) within a declared factor of g₁₂(ρ) | P1.1 / M3 |
| R | consecutive outer iterations with g₁₂ ≤ g_ref | Phase 2 primary response |
| θ_sub | principal angles between cluster invariant subspaces | M4 |
| ε₁(g,t) | relative error of the N = 1 model vs. true Δλ₁ | M1 |
