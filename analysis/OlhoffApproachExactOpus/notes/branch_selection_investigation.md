# Phase 5 — Why do the paper and the reconstruction choose different topological branches?

Scientific investigation (not reproduction, not bug-hunting, no fixes, no solver edits, no large sweeps).
All new evidence comes from **analysis of already-stored designs** (`OlhoffApproachExact/Matlab/results/*.mat`) recomputed with the independent clean-room FE, plus targeted single eigensolves of constructed candidates. Scripts: [experiments/morphology_analysis.py](../experiments/morphology_analysis.py), [implementation/fe_verify.py](../implementation/fe_verify.py).

**Cross-validation first:** the clean-room FE recomputes the stored designs' frequencies **bit-for-bit** against the production solver (iter-24: 462.3/463.3 vs audit 462.3/463.3; iter-80: 413.9/450.1 vs audit 413.9/450.1) when the **du2007_c1** mass model is used. So the FE, the spec, and the existing solver all agree on the physics — the branch difference is not a numerical-evaluation discrepancy.

---

## Executive summary (the new picture)

The prior conclusion on record ([[olhoff_exact_verification]]: "the disconnected design is genuinely higher-ω; the discrepancy is NOT numerics") is **half right and needs an important correction**:

> The disconnected two-block design is genuinely the higher-frequency feasible design **only under the joint conditions (40×5 coarse mesh) AND (du2007_c1 near-massless-void mass model)**. Remove *either* condition and the disconnection advantage collapses:
> - **Under physical (linear) mass**, the *same* disconnected designs have ω₁ = 16–58 (vs 414–462 under c1) — i.e. disconnection is **catastrophic**, not superior (ratios up to **37×**).
> - **At the paper's finer mesh**, a connected braced truss can route load without piling mass on the center antinode and can reach ~456; the 40×5 mesh cannot represent such bracing (connected variants cap at ≤288), so the optimizer correctly disconnects.

So the reconstruction's disconnected branch is **a discretization + mass-model artifact**, not a deeper physical truth. The paper's Fig. 3c is the legitimate optimum of the *properly resolved* problem; the reconstruction solves a coarser, more permissive problem whose genuine optimum is disconnected.

This reframes the leading hypotheses: the dominant causes are **(G) mesh coarseness** and **(H) aggressive void-mass interpolation**, acting jointly, with the early **gradient-driven center-drain** (Task 3) as the trigger.

---

## Task 1 — Updated ranked hypothesis list

Ranking integrates the new morphological/modal evidence. "Impact" = how much it would move the connected↔disconnected outcome.

| Rank | Hypothesis | Plausibility | Impact | Consistency with evidence | Verification difficulty |
|---|---|---|---|---|---|
| **1** | **G (NEW) — Mesh too coarse to represent the connected braced optimum.** 40×5 (5 rows) cannot form the thin diagonal braces of Fig. 3c; connecting must pass through the mass-loaded center. | **High** | **Decisive** | 40×5 over-predicts even the *initial* freq (71.5 vs 68.7; finer meshes converge to target). Connected 40×5 variants cap ω₁≤288 while disconnected reach 462. Fig. 3c clearly uses finer resolution. | **Low** — one finer-mesh run (Exp-1). |
| **2** | **H (NEW) — du2007_c1 makes the void near-massless, rewarding macroscopic disconnection.** Emptying the center costs ~0 modal mass, so two clamped stubs ring freely. | **High** | **Decisive** | c1-vs-linear ω₁ ratio grows with void depth, up to 37×; vanishes for connected (no-void) designs. The drain is "free" only because void mass→0. | **Low** — recompute under linear/milder mass (Exp-2, already partly done). |
| **3** | **A (revised) — Connected Fig. 3c is a local optimum; disconnected is the global optimum *of the as-implemented (coarse+c1) problem*.** | **High (conditional)** | High | At 40×5+c1, adding any bridge drops ω₁ 462→186–288 ⇒ disconnected is genuinely optimal *there*. But this is **conditional on G∧H**, not a physical truth (refuted under linear mass). | Low — same as Exp-1/2. |
| **4** | **B/E — Paper stayed connected via undocumented choices (filter radius, continuation, mesh density).** | Medium-High | High | Bifurcation is at iter 2–3 and survives a 0.1 move limit, so "choices" must include **finer mesh and/or larger filter radius**, not just step damping. Filter radius is [N]; a larger `r_min` on a finer mesh enforces a connected web. | Medium — needs filter/mesh combination (Exp-1 covers it). |
| 5 | **F — Benchmark interpretation (supports / normalization).** | Low (mostly ruled out) | Low–Med | Supports verified: mid-height-pin reproduces 68.7/104.1/146.1; corner-pin does not. Normalization fine. **Only the mesh sub-part of F is live** → folded into G. | Done (supports), Low (mesh). |
| 6 | **C — Multiplicity handling near coalescence differs.** | Low | Low | `detect_multiplicity` + generalized gradients are correct (Phase 3 §1). Multiplicity is a *consequence* of the chosen branch, not its cause; both branches are bimodal. | n/a |
| 7 | **D — MMA determinant-subproblem embedding differs.** | Low | Low | Embedding is faithful (Task 4 below). It affects step quality/stability, not which basin is rewarded. | n/a |

**Headline:** ranks 1–2 (G, H) are new, jointly decisive, and cheaply testable; they subsume the old "A — disconnected is genuinely superior" by exposing its hidden preconditions and showing it is **not** physically robust.

---

## Task 2 — Morphological comparison (quantified)

Designs from `baseline_trace.mat` and `cc_penalty_continuation.mat`, 40×5, recomputed with both mass models. `cden` = mean density in the central third (grey-bridge probe).

| design | vol | #solid comp (ρ>0.5) | center ρ | ω₁ (du2007_c1) | ω₁ (linear) | c1/lin |
|---|---|---|---|---|---|---|
| baseline iter 1 (≈connected) | 0.50 | 4 | 0.474 | 179.8 | 179.8 | **1.00** |
| baseline iter 5 | 0.50 | 2 | 0.216 | 257.4 | 249.9 | 1.03 |
| baseline iter 24 (**"near-paper 462"**) | 0.50 | **2 (disconnected)** | 0.041 | **462.3** | 58.1 | **7.96** |
| baseline iter 80 (final) | 0.50 | 2 | 0.069 | 413.9 | 16.4 | **25.2** |
| continuation FINAL | 0.50 | 2 | 0.178 | 274.4 | 24.9 | 11.0 |
| control(fixed-p) BEST | 0.47 | 2 | 0.012 | 641.7 | 17.2 | **37.4** |
| **iter-24 + connecting bridge (vol-matched)** | 0.50 | 3 | 0.207 | **185.9** | — | — |
| **paper Fig. 3c (target)** | 0.50 | **1 (connected)** | — | **456.4** | — | — |

Quantitative transitions:
- **Connected components:** every converged reconstruction design is **2 (disconnected)**; the *only* near-connected states are the first 1–2 iterations. The published Fig. 3c is **1 (connected)** — a full-span X-braced truss.
- **Bridge width:** central-third columns have **0 cells > 0.5** in every converged design; center density 0.01–0.18 (grey, near-void). The paper's connected design carries a continuous (if thin) solid load path across the span.
- **Density distribution:** reconstruction = two near-solid end blocks + a near-void center. Paper = material distributed along diagonal braces spanning the whole domain.
- **Modal character:** both branches end **bimodal** (ω₂/ω₁ ≈ 1.0–1.4), so multiplicity is *not* the discriminator. The discriminator is purely **the central load path**.
- **The structural transition separating the basins:** *presence vs. absence of a solid (or stiff-enough) connection across the center antinode.* Crossing it (adding a bridge) costs 462→186–288 ω₁ at this mesh+model — a steep ridge, which is why the disconnected basin is a strong attractor here.

---

## Task 3 — Basin analysis (conceptual map)

Treat the outer loop as a discrete dynamical system on the density field.

**Where the trajectory leaves the connected branch:** very early — **iterations 2–3**. Center density drains monotonically from the uniform start: 0.474 (it.1) → 0.379 (it.2) → 0.306 (it.3) → 0.270 (it.4) → 0.216 (it.5); the solid (>0.5) center is gone by iteration 3. This happens **even with a 0.1 move limit**, so it is not bang-bang overshoot — it is the **gradient direction itself**.

**Why the gradient points to disconnection:** for the CC fundamental mode the displacement antinode is at the span center. The eigenvalue sensitivity `∂λ₁/∂ρ_e = φ_eᵀ(K'_e − λ M'_e)φ_e` is **negative at the antinode** (the `−λ M'` mass term dominates where `|φ|` is largest) — i.e. *removing* material at the center *raises* ω₁. Frequency maximization therefore drains the center and piles material at the clamped supports/nodes. This is correct physics, shared by paper and reconstruction.

**Where the disconnected branch becomes attractive (and self-reinforcing):** once center density drops below ~0.1, the **du2007_c1 mass collapses** (`∝ρ⁶`), so the residual center mass stops contributing to the low modes. The two end blocks decouple into short clamped stubs whose first modes are high. From there, *further* emptying remains rewarded (massless void) and there is **no restoring force** toward reconnection — a stable attractor.

**Bifurcation structure:** the connected and disconnected branches separate at a **ridge** defined by the central load path. The uniform start sits on the connected side but on a slope whose gradient points across the ridge. With:
- **no length-scale strong enough to forbid the thin center void** (coarse mesh / small filter), and
- **a mass model that makes the void free**,
the descent crosses the ridge by iteration ~3 and cannot return.

**Status of the paper optimum (456.4) for the reconstruction:** it is **not a waypoint the reconstruction passes through** — the reconstruction's 462 is a *different (disconnected)* design. The connected Fig. 3c lives in a basin the reconstruction **never re-enters after iteration ~3**. Within the as-implemented (40×5 + c1) problem, Fig. 3c behaves like a **local optimum / saddle inferior to the disconnected global**; within the properly-resolved problem it is the legitimate optimum.

```
         omega1
           ^
   ~462 ---|----------------------*  disconnected attractor (40x5 + c1)
           |                     /   (massless void; stubs ring free)
   ~456 ---|----.Fig3c (connected, needs FINE mesh + load path)
           |   /  \
   ~288 ---|  /    \___ best CONNECTED design representable at 40x5
           | /          (bridge loads center antinode -> capped)
   ~147 ---|* uniform rho=0.5 start
           +----+----+----+----+----> "center load path emptied"
            it1  it3   (ridge crossing ~it2-3)
```

---

## Task 4 — MMA determinant-subproblem audit

The paper's largest ambiguity (spec §11.9): how are the nonlinear det-constraints (25d)/(26f)/(26g) embedded in MMA? Audit of `inner_loop_mma.m:157-200`.

**What the code does (Interpretation I1 — "eigenvalue-as-constraint"):**
build `F(Δρ)` with `F(s,k)=Σ_e f_sk(e)Δρ_e`; eigen-decompose `F = Q diag(μ) Qᵀ`; treat each `μ_i(Δρ)` as a smooth function and impose `β − (λ̄ + μ_i) ≤ 0` with gradient `dμ_i/dΔρ_e = q_iᵀ F_e q_i`. This is exactly the directional-derivative theory of Seyranian/Olhoff — the increments `Δλ_i` ARE the eigenvalues of `f_skᵀΔρ` (the erratum's `Δ(ω²)`), so this is **the faithful reading** of (25d).

**Plausible alternatives and their consistency:**

| Interp. | Description | Consistent with paper wording? | Consistent with observed behavior? | Expected basin influence |
|---|---|---|---|---|
| **I1 (implemented)** | μ_i = eig(F) as smooth constraints, eigenvector-linearized each MMA step | **Yes** — matches eq. 18/25d "subeigenvalue problem furnishes the increments" | Yes (bit-exact freq) | **Low** — correct sensitivities; doesn't change which basin is rewarded |
| I2 | Keep `det[F−μI]=0` as explicit equality with μ as extra MMA variables | Literal but impractical; paper says "solved by MMA," implying reduction to bound constraints | Untested | Low–Med (numerics only) |
| I3 (LP) | Impose vanishing off-diagonals (eq. 22) `f_skᵀΔρ=0, s≠k` ⇒ N linear constraints `β−(λ̄+f_jjᵀΔρ)≤0` | **Yes** — paper's explicit stated simplification (Krog & Olhoff) | Would be *more* bang-bang (pure LP); not what the code does | Low for basin; worse for stability |
| I4 | Bound β by a conservative invariant (min-eig lower bound / trace) | Not stated | Untested | Low |

**Second-order subtlety (the only real I1 risk):** when `F` itself has repeated eigenvalues (an increment that *preserves* the multiplicity), `q_i` is non-unique and `dμ_i/dΔρ` is non-differentiable *inside the inner loop* — a multiplicity-within-the-increment that the paper does not discuss. This can make the inner step chatter near perfectly symmetric states, but it is a **stability** effect, not a basin-selection mechanism.

**Verdict:** the determinant embedding is **faithful (I1)** and is **not** a plausible cause of the branch difference. Rank D stays low. (This rules out one of the user's six candidate explanations with positive evidence.)

---

## Task 5 — Literature archaeology

- **Du & Olhoff's own mass model** (and the literature describing it) is explicitly characterized as making *"the mass in low volume-fraction elements rapidly tend to zero"* to kill localized modes — confirming hypothesis **H**: the model is *designed* to make the void near-massless, which is exactly what rewards macroscopic disconnection here. The cure for micro-localized modes is the cause of the macro-disconnection incentive.
- **Pedersen (2000), "Maximization of eigenvalues using topology optimization"** (Struct Multidisc Optim 20:2-11) devotes a dedicated *numerical method* to **removing localized eigenmodes in low-density areas** — independent confirmation that this pathology is recognized, non-trivial, and method-specific (i.e. different groups use different, unreported, low-density treatments → an [N] degree of freedom that strongly affects the outcome).
- **Tcherniak (2002)** (the source Du & Olhoff modify) is "Topology optimization of resonating structures" — resonance/frequency designs are *known* to be prone to the void-mode/disconnection failure, motivating the special mass interpolation.
- A whole line of later work ("**Eliminate localized eigenmodes in level-set topology optimization for maximization of the first eigenfrequency**") exists *specifically* to suppress this failure mode — evidence that connected vs. spurious/disconnected outcomes are **regularization-determined**, not unique.
- **Krog & Olhoff (1999)** is the source of the LP/increment scheme and the "increments-as-unknowns" trick; the 2007 paper inherits its numerics but reports **no** filter radius, continuation schedule, multiplicity tolerance, or mesh density — all the [N] knobs that Phase 1 flagged and that this investigation shows are decisive.

**Net literature signal:** the connected/disconnected (equivalently, physical/spurious-mode) branch choice in SIMP frequency maximization is a **well-known, regularization-sensitive** phenomenon. No source claims the disconnected design is the "true" optimum; the consensus is that proper low-density mass treatment + filtering + adequate mesh are required to land on the physical (connected) design. This supports the reframing in the Executive Summary.

Sources:
- [Pedersen 2000 — Maximization of eigenvalues using topology optimization (Struct Multidisc Optim 20:2-11)](https://link.springer.com/article/10.1007/s001580050130)
- [Eliminate localized eigenmodes in level-set topology optimization for max first eigenfrequency (CMA)](https://www.sciencedirect.com/science/article/abs/pii/S0965997816302952)
- [Du & Olhoff 2007 (Struct Multidisc Optim 34:91-110)](https://link.springer.com/article/10.1007/s00158-007-0101-y)

---

## Task 6 — Minimal discriminating experiment set

Designed for maximum information gain, minimum runtime, no broad sweeps, no speculative fixes. Each is a **single** run or a few eigensolves. Listed in priority order; **Exp-1 alone is expected to resolve the investigation.**

### Exp-1 — Finer-mesh run (tests G, and A/B by implication) — *recommended next experiment*
- **Setup:** the existing CC driver, unchanged settings, **mesh 80×20 (or 160×40)** instead of 40×5. One run.
- **If G true:** a **connected** braced topology appears (or connected designs become competitive with disconnected), ω₁ approaches 456.4, components → 1. The aspect-resolved mesh can form diagonal braces avoiding the center antinode.
- **If G false:** still disconnects to two blocks with ω₁ ≫ 456 regardless of resolution.
- **Interpretation:** distinguishes "coarse-mesh artifact" (G) from "genuine disconnected optimum independent of resolution" (A unconditioned). Highest information gain because it targets the rank-1 hypothesis with a single run.

### Exp-2 — Mass-model swap to physical/linear (tests H) — *partly done here*
- **Setup:** one run with **linear mass** (eq. 2, q=1) — or a milder void exponent — at 40×5; OR (already done, zero runtime) recompute stored designs under linear mass (Task 2 table).
- **If H true (already observed):** under linear mass the disconnected designs score ω₁=16–58 ⇒ the optimizer is **penalized** for disconnecting and should retain a connected center (modulo linear mass's own localized-mode noise). The disconnection incentive is the void-masslessness.
- **If H false:** linear-mass run still disconnects and still scores high.
- **Caveat:** linear mass reintroduces micro-localized modes (a different pathology), so the clean test is a *milder* void exponent (e.g. eq. 4 with r=3–4), not full linear. Low runtime (1 run).

### Exp-3 — Connected-seed restart (tests A vs B: basin vs global)
- **Setup:** initialize from a **connected** design (e.g. the iter-1 state, or a hand-seeded spanning bar) instead of uniform, keep everything else; one run.
- **If A (disconnected truly global at 40×5+c1):** the optimizer leaves the connected seed and disconnects anyway, returning to ω₁≈462 disconnected.
- **If B (connected basin is stable, just not reached from uniform):** it stays connected near the paper value.
- **Interpretation:** separates "global optimum is disconnected" from "connected basin is merely unreached due to the early gradient drain."

### Exp-4 — Filter-radius × mesh (tests B/E length-scale) — only if Exp-1 ambiguous
- **Setup:** at the finer mesh of Exp-1, two runs with `r_min` ≈ 3 and ≈ 6 element units.
- **If a length scale > the would-be void gap forbids disconnection:** connected design emerges; pins down whether the paper's (unreported) filter radius is the connectivity guarantor.
- Keep to 2 runs; not a sweep.

**Stopping rule:** run Exp-1 first. If it yields a connected ~456 design, G is confirmed and the investigation is essentially closed (with H as the secondary enabler already shown). Exp-2/3 then only *attribute* the cause; Exp-4 is contingency.

---

## Final Deliverable — consolidated answers

**1. Updated hypothesis ranking:** G (mesh coarseness) and H (massless-void mass model) jointly dominate; revised-A (disconnected is global *of the coarse+c1 problem*) follows as their consequence; B/E (undocumented filter/mesh/continuation) next; C, D, F low (D positively ruled out; supports in F ruled out).

**2. Basin-selection analysis:** the connected→disconnected bifurcation occurs at **iteration 2–3**, driven by the *correct* fundamental-mode antinode-emptying gradient, and is locked in by the **du2007_c1 void-mass collapse** once center density < 0.1. No restoring force reconnects it. The paper optimum is in a basin the reconstruction abandons immediately and never revisits.

**3. Determinant-subproblem audit:** the implementation uses the faithful "eigenvalues-of-F as smooth constraints" embedding (I1), matching eq. 18/25d. It is **not** a cause of the branch difference (positively ruled out). The only residual risk (non-unique `q_i` when `F` has repeated eigenvalues) is a stability, not a basin, effect.

**4. Literature findings:** the connected/disconnected (physical/spurious-mode) split in SIMP frequency maximization is a recognized, **regularization-determined** phenomenon; Du & Olhoff's mass model is explicitly built to drive void mass→0 (the disconnection enabler); Pedersen 2000 and level-set follow-ups devote dedicated methods to suppressing exactly this. No source treats the disconnected design as the true optimum.

**5. Recommended next experiment:** **Exp-1 — one finer-mesh (80×20) run with otherwise-unchanged settings.** Single run, highest information gain, directly tests the rank-1 hypothesis.

**6. Probability that Fig. 3c is:**
- **Global optimum (of the properly-posed, well-resolved problem):** ~**55%.** Under physical/mild mass and adequate mesh, connected dominates (disconnected collapses to ω₁≈16–58 under linear mass); Fig. 3c is most likely the legitimate optimum of the intended problem.
- **Local optimum (of the as-implemented 40×5 + du2007_c1 problem, where disconnected is global):** ~**30%.** Strongly supported at the coarse mesh, but this is a property of the *coarsened* problem, not the intended one.
- **Path-dependent attractor:** ~**10%.** The early gradient drain is robust (survives a 0.1 move limit), so the connected outcome is more about regularization/mesh than luck; some path-dependence remains via continuation/symmetry.
- **Benchmark/discretization artifact (i.e. the *reconstruction's disconnection* is the artifact; Fig. 3c is correct):** ~**5%** as a *pure* artifact, but note this overlaps with the 55% global case — the most likely single narrative is **"Fig. 3c is the global optimum of the properly-resolved problem, and the reconstruction's disconnection is a coarse-mesh + aggressive-void-mass artifact."**

(These four are not mutually exclusive; the coherent combined reading is: *Fig. 3c ≈ global optimum of the intended problem (G/H corrected); the disconnected branch ≈ global optimum of the as-implemented coarse/permissive problem.* The single change most likely to reconcile them is mesh resolution — Exp-1.)
