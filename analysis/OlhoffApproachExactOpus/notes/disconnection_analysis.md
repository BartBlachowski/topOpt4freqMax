# Phase 2 — Disconnection, localized modes, disconnected optima, path-dependence

Analysis of what the **paper formulation** (Du & Olhoff 2007) does and does not permit. Derived from the spec ([du_olhoff_2007_spec.md](../specification/du_olhoff_2007_spec.md)); **no implementation code consulted**. The question being answered: *does the mathematics of the paper, on its own, admit disconnected topologies, localized modes, disconnected optima, and path-dependent solutions?*

---

## TL;DR

| Phenomenon | Permitted by the paper formulation? | Paper's own safeguard |
|---|---|---|
| **Disconnected topologies** | **YES** — nothing in the formulation forbids them | none (no connectivity constraint) |
| **Localized (spurious) modes** | **YES** in principle (low-density SIMP) | *explicitly suppressed* via modified mass model (eq. 4/4a/4b), `r≈6` |
| **Disconnected optima** | **YES** — admissible and, for some objectives, *favoured* | none structural; only the filter and the physics of the objective |
| **Path-dependence** | **YES** — built in (MMA + p-continuation + symmetric start + mode tracking) | the `J=n+N` cap and bound formulation reduce but do not remove it |

The formulation is a **pure eigenfrequency objective with a single global volume constraint and box constraints**. It contains *no* connectivity, perimeter, symmetry, or minimum-member constraint. Therefore every one of the four phenomena is *mathematically admissible*. Only one of them (localized modes) is given a dedicated countermeasure.

---

## 1. Disconnected topologies

### 1.1 Are they feasible?
The feasible set is defined entirely by:
- volume: `Σ_e ρ_e V_e ≤ α V_0` (eq. 7d) — a single scalar, geometry-blind,
- box: `ρ̲ ≤ ρ_e ≤ 1` (eq. 7e),
- the eigenproblem `Kφ=ω²Mφ` (which always has a solution as long as `K,M` are SPD — guaranteed by `ρ̲>0`).

None of these references *adjacency* of solid elements. A density field that forms two (or more) separated solid blocks, each independently meeting the box/volume limits, is **feasible**. ⇒ **The paper permits disconnected topologies.**

### 1.2 Do they produce a well-posed eigenproblem?
Yes, because `ρ̲ = 10⁻³ > 0`. The "void" is not truly void; every element retains tiny stiffness and mass. So even a "disconnected" black-and-white design is, numerically, a single connected SPD system with very soft connecting material. There are therefore **no exact rigid-body (ω=0) modes**; instead, a nearly-disconnected block produces **very low but nonzero** eigenfrequencies — which is exactly the mechanism of localized modes (Section 2 below).

### 1.3 Is disconnection *penalised*?
No. There is no perimeter term, no connectivity constraint, no length-scale. The Sigmund (1997) **sensitivity filter** (Section 1.4 of spec) imposes a *minimum feature size on the density gradient* and thereby discourages single-element checkerboards, but it does **not** enforce global connectivity — two solid regions separated by more than `r_min` of void are entirely compatible with the filter.

> **Consequence for the benchmark:** for the clamped–clamped beam (Fig. 2c / 3c), the supports exist at *both* ends. A topology that splits into two separate cantilever-like blocks (one anchored to each support) is feasible. Whether such a split is *optimal* depends on the eigenfrequency physics (Section 3), and the paper's published Fig. 3c looks connected — so a reproduction that converges to a *disconnected* CC design is exploring a feasible-but-different basin (this is the central Phase-3/Phase-4 question; see the project memory note "disconnected two-block CC topology").

---

## 2. Localized (spurious) modes

This is the one phenomenon the paper treats head-on (Section 2.2 of the paper → §1.2 of spec).

### 2.1 Mechanism
With `p≈3` (stiffness `∝ρ³`) and `q=1` (mass `∝ρ`), in a low-density region (`ρ_e ≤ 0.1`) the **stiffness-to-mass ratio `∝ ρ^{p-q} = ρ²` collapses**. A patch of soft material then behaves like a heavy mass on a very weak spring ⇒ a **local eigenmode with a very low eigenfrequency**, decoupled from the global structure. Because the objective *maximizes a low-order eigenfrequency*, such spurious low modes can hijack `ω_n` and corrupt the optimization (the optimizer would chase / be blocked by a physically meaningless mode living in the void).

### 2.2 The paper's countermeasure
Make **mass decay faster than stiffness** in low-density regions: use `r≈6 ≫ p≈3` for the mass exponent below `ρ=0.1` (eq. 4), optionally smoothed to `C⁰` (4a) or `C¹` (4b). Then in the void `mass ∝ ρ⁶` while `stiffness ∝ ρ³`, so the stiffness-to-mass ratio `∝ ρ^{3-6} = ρ⁻³ → ∞` as `ρ→0`: local modes are pushed to **high** frequencies, out of the way of the low-order objective.

### 2.3 Residual risk
The fix is heuristic and threshold-based (`0.1`). It suppresses void-localized modes but does **not** address *near-disconnection at intermediate density* nor genuine structural localization (a real thin neck that legitimately localizes a mode). The paper explicitly notes it relies on the assumption that "low-density regions contribute negligibly to the first several eigenfrequencies" — an assumption that **breaks down precisely when the optimum wants to disconnect**, because then the connecting region sits at intermediate-to-low density during the transition. ⇒ The localized-mode fix and the disconnection tendency interact, and the paper does not analyze that interaction.

---

## 3. Disconnected optima

### 3.1 Why an eigenfrequency objective can *favour* disconnection
Maximizing `ω_n² = (Rayleigh quotient)` rewards **high stiffness per unit mass** in the deforming mode. Two independent considerations push toward splitting:

1. **Mass economy.** With a fixed material budget `α V_0`, concentrating material into a few stiff load paths anchored directly to supports can raise the fundamental frequency more than spreading it into a single slender continuous member. If two separate stub-like blocks each anchored to a support achieve a higher `min ω` than one long connected span, the optimizer has no formulation-level reason to prefer the connected design.

2. **Mode decoupling.** Splitting into symmetric halves can create a *bimodal* fundamental frequency (two identical half-structures ⇒ two degenerate modes). The paper *celebrates* bimodality as the expected optimum ("creation of structures with multiple optimum eigenfrequencies is the rule rather than the exception"). A disconnected, symmetric two-block design is a natural way to manufacture exactly such a degenerate pair. **The objective therefore actively rewards the very symmetry that disconnection produces.**

### 3.2 The clamped–clamped case specifically
For CC boundary conditions, both ends are anchored. A design that disconnects into two mirror blocks (left block clamped at the left wall, right block clamped at the right wall, gap in the middle) yields two *identical, decoupled* cantilever-type sub-structures ⇒ an exactly **bimodal** fundamental frequency, with each block shorter (hence stiffer in its first mode) than a single clamped–clamped span of the full length. This is a textbook recipe for a *higher* maximized fundamental frequency than a connected design — **the formulation can prefer the disconnected optimum**.

The paper's published Fig. 3c is connected, so either (a) for their mesh/filter/continuation the connected basin won, or (b) the bimodal connected design they found has higher `ω_1` than the split design at their resolution. The formulation alone does **not** guarantee (a); it is a path/numerics outcome, not a formulation guarantee.

### 3.3 Gap maximization makes it worse
For `max (ω_n − ω_{n-1})` (eq. 16/26), the optimizer is free to *lower* `ω_{n-1}` as well as raise `ω_n`. Driving `ω_{n-1}` down is most easily achieved by creating a **soft, nearly-detached sub-region** whose local mode sits just below the gap — i.e. a controlled, near-disconnected feature. The Fig. 9 result (mass on a clamped beam, gap `ω_3−ω_2`) and Fig. 18 (bimaterial, where the gap goes from 0 to "infinitely large relative increase") both exploit this. **Gap problems structurally incentivize near-disconnection.**

### 3.4 Verdict
Disconnected optima are **admissible and, for several of the paper's own objectives, plausibly optimal or near-optimal**. The paper provides no mechanism to exclude them; it relies on the chosen objective + numerics landing in a connected basin. This is a genuine non-uniqueness/well-posedness gap in the formulation, not a bug.

---

## 4. Path-dependence

The procedure is path-dependent for at least four compounding reasons:

1. **MMA is history-dependent.** Moving-asymptote bounds adapt from the iterate history; different paths ⇒ different convex sub-approximations ⇒ different fixed points. (Spec §7.4.)

2. **Penalization continuation `p: 1→3`.** The objective landscape *changes shape* across outer iterations. The design tracked from `p=1` (near-convex, gray) to `p=3` (strongly non-convex, 0–1) is a homotopy whose endpoint depends on the schedule (which is **unspecified**, spec §11.4). Different schedules ⇒ different optima.

3. **Symmetric uniform start `ρ=0.5`.** A symmetric start on a symmetric domain sits on a symmetry-invariant manifold. Whether the iterate *stays* symmetric (yielding bimodal connected designs) or *breaks* symmetry (potentially toward a disconnected/asymmetric design) depends on numerical asymmetries (round-off, eigenvector sign/basis choice in the multiple-eigenvalue subspace, filter discretization). The paper does **not** specify any deliberate symmetry-breaking perturbation. ⇒ The branch taken at the first bifurcation is path-dependent.

4. **Multiple-eigenvalue handling + mode tracking.** When `ω_n` becomes multiple, the eigenvectors within the invariant subspace are non-unique (spec §11.10). The generalized gradients `f_sk` (eq. 19) depend on the basis chosen each iteration. The subeigenvalue problem (18) is basis-invariant *in theory*, but a concrete solver picks a basis and tracks modes across iterations — and *how* it tracks (by overlap? by order?) is unspecified (spec §11.8). Different tracking ⇒ different `N` detection ⇒ different `J=n+N` ⇒ different active constraints ⇒ different path.

### 4.1 Consequence
Two faithful implementations of the *same* equations can converge to *different* topologies (connected vs disconnected; bimodal vs trimodal) purely from differences in: continuation schedule, MMA parameters, multiplicity tolerance, eigen-basis/tracking, and round-off-driven symmetry breaking. **The published figures are one path's outcome, not a formulation-unique answer.** This is the key caveat for judging any reproduction mismatch in Phase 3.

---

## 5. Implications for Phases 3–4

1. A reproduction that yields a **disconnected CC topology** is **not violating the paper's formulation** — disconnection is feasible and the objective can favour it. The mismatch (if any) lives in *which basin* the path selects, governed by the **[N]**-classified items (continuation, filter radius, multiplicity tolerance, symmetry handling, mode tracking).

2. Therefore the discrepancy hunt in Phase 3 should focus on the **path-determining unspecified choices**, not on the equations (which are unambiguous after the erratum). The most leveraged suspects, in order:
   - continuation schedule for `p` (and whether `p` starts at 1 vs 3),
   - filter radius `r_min` (controls min feature size ⇒ whether a thin connecting neck survives),
   - multiplicity tolerance and mode tracking (controls `N`, `J`, and the bound constraints),
   - whether the nonlinear MMA sub-problem or the LP reduction (eq. 22) is used,
   - symmetry-breaking / initial perturbation.

3. The paper's own safeguards are limited to: `ρ̲>0` (well-posed eigenproblem), modified mass model (localized-mode suppression), and the `J=n+N` linearized cap (mode-switch prevention). **None of these enforce connectivity.** If a connected topology is a *requirement* (it is not stated as one in the paper), it would have to be added as an extra constraint — which would be a *deviation from* the paper, to be flagged as such in Phase 3/4.

---

## 6. Open questions to resolve against the implementation (Phase 3)

- Does the existing `OlhoffApproachExact` add any connectivity/symmetry handling not in the paper? (If yes → deviation.)
- Which mass model (4/4a/4b) and what `r` does it use? Does that change which modes are "localized"?
- What `p`-continuation and filter radius does it use, and does changing them move the result between connected/disconnected basins?
- Does it use MMA on the nonlinear det-problem, or the LP reduction? (The paper makes only the LP fully concrete.)
- How does it detect multiplicity and track modes across iterations?

These five questions are the bridge from "the formulation permits disconnection" (this note) to "why the specific reproduction differs from the published figures" (Phase 3).
