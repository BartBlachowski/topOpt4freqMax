# A4_SPECIFICATION_V3 — Eigenpair-Refresh Study

**Status:** authoritative definition of A4. Supersedes A4-1 … A4-4 in
`scripts/revision_v1/IMPLEMENTATION_MAP.md` and the A4 section of
`examples/Revision_v1/revision_v1_update1.md`.
**Date:** 2026-07-14
**Type:** experiment design only. No code, no patches, no runs.

**Supersedes because:** the previous plan reused the retired EXP4 configurations
(`ss_beam_harmonic_frozen.json` / `ss_beam_harmonic_periodic.json`), varied the load
model, optimizer, sensitivity treatment and mesh alongside `N`, judged the surrogate by
the surrogate, left `N=∞` undefined, and used an acceptance gate that would have rejected
its own primary finding. Those conclusions are accepted and are not re-argued here.

---

## 0. Two facts that determine this design

Both were established by reading the code, and both must be understood before Part 1.

### 0.1 The proposed method has no refresh capability

`update_after` — the refresh interval — is parsed **only** under `case 'harmonic'`
([`topopt_freq.m:817-822`](../../analysis/ourApproach/Matlab/topopt_freq.m#L817-L822)). The
`case 'semi_harmonic'` branch records the mode number and nothing else
([`:830`](../../analysis/ourApproach/Matlab/topopt_freq.m#L830)).

**The proposed method — `semi_harmonic`, the one the manuscript describes — cannot refresh
its eigenpair at all.** The refresh knob exists only on the `harmonic` path, which is a
different load model carrying a partial `∂f/∂x` term and driven by MMA.

This is the root cause of the previous plan's central error. It did not choose the harmonic
configs by oversight; it chose them because **they are the only place `update_after` does
anything.** A4 therefore requires exactly one new solver capability (§7, R-1), and that
capability is the *whole* implementation burden.

### 0.2 The mass model — SUPERSEDED AND CORRECTED (2026-07-14)

> **This section previously argued that the manuscript's `p_M = 3` was correct, that A4 must
> pin it, and that the `p_M = 1` default "is the S1 pathology" and "explains EXP4's −62%".
> All of that is WITHDRAWN.** An independent audit and
> [`MASS_INTERPOLATION_DECISION.md`](../../MASS_INTERPOLATION_DECISION.md) established the
> opposite: **the implementation was right and the manuscript was wrong**, and the void-mass
> mechanism I proposed was **empirically refuted**. The corrected position follows.

**The declared mass model is LINEAR** ([`MASS_INTERPOLATION_DECISION.md`](../../MASS_INTERPOLATION_DECISION.md)):

```
m_e(x_e) = [ ρ_min + x_e (ρ₀ − ρ_min) ] · m_e⁰        (p_M = 1)
```

SIMP `p_K = 3` penalizes **stiffness only**; there is **no low-density correction branch**. The
manuscript's global `m = x^d, d = p = 3` was an error introduced by a drafting pass, and has been
corrected. A global `d = p` would make frequency independent of uniform density and would miss
Du & Olhoff's own published initial frequencies by a factor of two.

**A4 therefore pins `pmass = 1`, not 3.** This is the method as actually implemented and now as
actually published.

**Why the mass exponent cannot be A4's shield.** The void-mass mechanism (`ω² ~ x^{p_K−p_M} → 0`,
void vibrates, kinetic energy trapped in low-density material) is *mathematically real* but is
**not what these runs exhibit**. The repository's own diagnostics show, for **every** mode and at
**every** `pmass` tested:

```
low_density_kinetic_fraction = 0.0000
```

The void carries essentially **no kinetic energy**. The flagged modes are instead **solid islands
not connected to the supports**, resonating through weak `E_min` material — 126 disconnected
components at `pmass=1`, 198 at `pmass=6`. Three mass models (linear, `x⁶`, Du & Olhoff Eq. 4b)
were each tested on the same 400×50 case and **none removes the family** (8, 9, 9 flagged modes).

**Consequences for A4 — these are load-bearing, not editorial:**

1. **There is no mass setting that buys A4 a clean spectrum.** The earlier plan to "pin `p_M = 3`
   and the pathology collapses" is void. **Gate A4-Pre (§6.1) is therefore not a formality — it is
   the only thing standing between A4 and a repeat of EXP4.**
2. **The B3 contamination detector must be rebuilt.** It was specified to fire on *kinetic energy
   in low-density elements* — a quantity that is **0.0000 in every observed case**. As originally
   written it **would never have fired**, and A4 would have published contaminated refresh arms as
   accuracy evidence. The corrected discriminator is **support-connectivity based** (§4.3.1).
3. **EXP4's −62% remains unexplained in mechanism**, and must be treated as such. It is a warning,
   not a solved case.

---

# PART 1 — Scientific objective

## 1.1 Research question

> **Does optimizing the quasi-static surrogate with a permanently frozen reference eigenpair
> `(ω₀, Φ₀)` produce a design whose *true* fundamental frequency is materially worse than one
> obtained by periodically refreshing that eigenpair on the evolving design — and if so, at
> what refresh interval, and by what mechanism, does the frozen mode cease to be a valid
> proxy for the optimized mode?**

The experiment varies exactly one thing: the refresh interval `N`. `N = ∞` is the published
method. `N` finite recomputes `(ω₀, Φ₀)` on the current design every `N` iterations.

## 1.2 Hypotheses

Optimization here is **deterministic** (the repository has already established that repeated
runs give identical results; `std ≈ 0` was reported as an honest artifact of determinism).
There is therefore **no sampling distribution and no statistical test.** The hypotheses are
stated against a **pre-declared equivalence margin `δ`**, fixed before any run.

**`δ = 5%` relative on the tracked true fundamental frequency.**

*Justification of `δ`:* the manuscript's headline gains are `2.53×` and `4.61×` (i.e. +153%
and +361%). A penalty of ≤5% from freezing is between one and two orders of magnitude
smaller than the effect being claimed and would not alter any qualitative conclusion, the
choice of method, or any design recommendation. A penalty >5% would. `δ` is a scientific
threshold, not a numerical tolerance, and it is pre-registered so that it cannot be chosen
after seeing the data.

- **H₀ (freezing is benign):** for every finite `N`,
  `[ω₁ᵗʳᵃᶜᵏ(N) − ω₁ᵗʳᵃᶜᵏ(∞)] / ω₁ᵗʳᵃᶜᵏ(∞) ≤ δ`.
  Refreshing buys nothing beyond the margin; the frozen mode is an adequate proxy on the
  tested class.
- **H₁ (freezing costs accuracy):** some finite `N` yields
  `ω₁ᵗʳᵃᶜᵏ(N) > ω₁ᵗʳᵃᶜᵏ(∞) · (1 + δ)`, from a run that is **clean** — i.e. not classified as
  spurious-mode contaminated (B3) or sensitivity-omission unstable (B4).

The `clean` qualifier is load-bearing. Without it, H₁ can be "confirmed" by an artifact, which
is exactly what EXP4 did in reverse.

**A third outcome is pre-registered as legitimate** (see Part 5): *the refresh reference is
unavailable* — every refreshed arm is contaminated or unstable, so no accuracy reference can
be constructed. This is a real result about the method's diagnosability, and declaring it in
advance is what stops the experiment from being retrofitted into whichever story the data
happen to permit.

## 1.3 Scientific contribution

A4 delimits the **validity domain of the frozen quasi-static approximation** — the single
assumption that distinguishes the proposed method from a conventional eigenvalue
formulation. Without it, the method is an unbounded heuristic: it works on three examples for
undocumented reasons. With it, the method has a stated regime of applicability and a
quantified cost of its central simplification.

It is also, after the retirement of EXP1 and EXP5, **the only remaining quantitative claim the
paper can make about its own approximation**, and it makes it *within a single implementation*
— so it survives the reasoning that killed the cross-code benchmarks. No comparator is
involved; `N` is an internal knob.

## 1.4 Reviewer question addressed

**Reviewer 1, Required item** ([`final_review_V1.tex:100-107`](../../paper/reviews/final_review_V1.tex#L100-L107)):

> *"**Frozen-mode reliability not demonstrated.** The conclusions acknowledge an 'accuracy
> ceiling' from the frozen initial mode shape, but the paper does not demonstrate conditions
> under which the approximation fails. A reliability diagnostic — for example, a benchmark
> where the initial and optimized mode shapes differ substantially, a MAC-based threshold
> analysis, or **a comparison against a periodically updated-mode variant** — is needed to
> bound the practical scope of the method."*

A4 is the third of the three offered forms. **Reviewer 2's** unanalyzed-coalescence concern
(the SS-beam Du–Olhoff optimum is bimodal; the frozen mode may degrade as `ω₁` and `ω₂`
approach) is addressed as a secondary observable (§4.3), because the SS beam is precisely
where coalescence is expected.

## 1.5 Manuscript claim supported

Two sentences currently stand without any artifact:

1. [`main.tex:661`](../../paper/main.tex#L661) — *"…confirming the **robustness of the
   quasi-static approximation across structurally different problem classes.**"*
2. [`main.tex:704`](../../paper/main.tex#L704) — *"…the frequency gain will be suboptimal
   **relative to formulations that update the eigenpair**."*

Sentence 2 asserts a **direction**: refresh beats frozen. The only data ever produced (EXP4,
retired) pointed the other way by 62%. A4 either evidences this sentence, reverses it, or
forces its retraction. Sentence 1 is a scope claim that A4 alone **cannot** support (see Part
8) and must be narrowed regardless of outcome.

---

# PART 2 — Experimental design

## 2.1 The single independent variable

| Factor | Levels |
|---|---|
| **Refresh interval `N`** | `{∞, 50, 10, 5, 1}` — 5 arms |

`N = ∞` — compute `(ω₀, Φ₀)` once on the reference design; never recompute. This is the
published method, unmodified.
`N = k` — additionally recompute `(ω₀, Φ₀)` on the **current design** `x⁽ⁱ⁾` at every
iteration `i` with `i mod k = 0`, and use that pair until the next refresh.
`N = 1` — recompute every iteration (a fully design-dependent inertial load).

**Why five levels and not two.** A two-point design (`∞` vs `1`) cannot distinguish *"freezing
is harmless"* from *"refreshing is unstable"* — both produce "the frozen arm wins" and the two
are opposite conclusions. A graded sweep locates the **breakdown threshold `N*`**, which is
what the reviewer actually asked for ("conditions under which the approximation fails"), and
it exposes non-monotonicity in `N`, which is the fingerprint of the instability mechanism
rather than of an accuracy trend.

## 2.2 Everything else is held constant

All values are the manuscript's SS-beam benchmark, §4.1.

| Held fixed | Value | Why it must not move |
|---|---|---|
| **Mesh** | SS beam, `400 × 50`, `L=8 m`, `H=1 m`, `t=1` | Mesh changes the spectrum and the localized-mode population. EXP3 already shows mesh-dependent mode validity; varying it here would confound `N` with the very pathology under control. |
| **Optimizer** | **OC**, move limit `0.2` | The proposed method's optimizer. MMA has different iterate paths, asymptote history, and scaling behaviour. EXP4 varied this alongside `N` — a primary confound. |
| **Stiffness interpolation** | SIMP, `p_K = 3`, `E₀ = 10⁷ Pa`, `E_min = 10⁻⁹E₀`, `ν = 0.3` | Manuscript §4.1. |
| **Mass interpolation** | **LINEAR: `pmass = 1`, `ρ₀ = 1`, `ρ_min = 10⁻⁹ρ₀`; no low-density branch** | **[`MASS_INTERPOLATION_DECISION.md`](../../MASS_INTERPOLATION_DECISION.md).** This is the declared method and what every production run already uses. It is a **fixed factor, not a free choice**: A4 must test the method *as implemented and as published*. It must be **stated explicitly in the base config** rather than left to the solver default, so that the arms cannot drift. *(Superseded: this row formerly pinned `p_M = 3` on the strength of the manuscript's erroneous equation.)* |
| **Filter** | Sensitivity filter, `r_min = 2` elements, symmetric BC, **no Heaviside** | The method claims no projection. Heaviside would alter the gray-material distribution and hence the spectrum the refresh arms see. |
| **Load model** | `semi_harmonic`, authoritative `f(x) = ω₀² M(x) Φ₀`, mode 1, `α = 1` | The proposed method. The `harmonic` path is a different model with a partial `∂f/∂x` and must not be used. |
| **Sensitivity model** | **Load sensitivity omitted** (`∂f/∂x ≡ 0`), stiffness-only `∂c/∂x` | The proposed method's defining approximation. It is *measured as a covariate* (§4.4), never varied — varying it would make this CR2, not A4. |
| **Continuation** | **None** | The manuscript's stated advantage is that no continuation schedule is required. Introducing one would test a different method. |
| **Stopping criteria** | `max|Δx_e| < 10⁻³`; iteration cap `2000` | Manuscript §4.1. Identical cap for all arms so that "capped" is comparable across `N`. |
| **Volume** | `V_f = 0.5`, no passive elements | Manuscript §4.1. The absence of passive elements is load-bearing for Part 3. |
| **Baseline (`N=∞` reference)** | Fully solid domain — see Part 3 | |
| **Initial design** | Uniform `x_e = V_f` | The optimizer's starting iterate. |
| **Load normalization** | `harmonic_normalize = false` | Gate A0 requires it. |

**The governing principle:** every arm must differ from `N = ∞` in *exactly one respect* — how
often the reference eigenpair is recomputed. Any other difference makes the arm
uninterpretable, and the previous plan's failure was that it differed in four.

This is enforced structurally, not by discipline: **one base configuration file, with `N`
injected by the driver as the sole override** (§7). Five separate JSONs invite exactly the
drift that produced `ss_beam_harmonic_*.json`.

---

# PART 3 — Baseline definition (`N = ∞`)

## 3.1 The contradiction

Four sources, three answers:

| Source | Reference design for `Φ₀` |
|---|---|
| `paper/main.tex` §3.2 ([:645](../../paper/main.tex#L645)) | **uniform** field `ρ_e⁽⁰⁾ = V_f` |
| `paper/main.tex` §3.3, §4.3 | **fully solid** domain |
| `examples/Revision_v1/ss_beam.json` | `semi_harmonic_baseline: "initial"` |
| Accepted decision **A0-F1** + solver guard | **fully solid** |

Reviewer 2 filed this as Required revision **C2** — *"The method is underspecified."*

## 3.2 The decision

> ### `N = ∞` is defined by the **fully solid domain**, `x_e = 1 ∀e`.
> `(ω₀, Φ₀)` is the fundamental eigenpair of `K(1)Φ = ω²M(1)Φ`, mass-normalized per A0-F4.

**Justification.**

1. **It is already the accepted decision (A0-F1)** and the authoritative formulation
   `F(x) = ω₀²M(x)Φ₀, reference = solid, frozen` on which EXP2 / EXP2b / EXP3 are being
   re-run. Choosing "uniform" would desynchronize A4 from the rest of the campaign and
   contradict a final decision.
2. **The solver already enforces it.** `topopt_freq.m` **hard-errors** on any other value:
   `topopt_freq:AuthoritativeBaselineRequired — "Authoritative semi_harmonic loads require
   semi_harmonic_baseline='solid'."`
   ([`:136`](../../analysis/ourApproach/Matlab/topopt_freq.m#L136)). The competing definition is
   not merely disfavoured; it is unrunnable. `ss_beam.json` — which sets `"initial"` *and*
   carries `semi_harmonic_rho_source`, forbidden by Gate A0 — **would throw today.**
3. **It is well-defined in the presence of passive elements**, where "the initial design" and
   "the solid domain" genuinely differ. The uniform reading does not generalize to the
   building; the solid reading does.

## 3.3 Why this choice does not perturb A4's endpoint (an invariance result)

A pleasant and important consequence, which must be **verified numerically, not assumed**
(validator V-A4-1):

For a **spatially uniform** density field `x_e = c ∀e` with **no passive elements**, every
elementwise interpolation gives a *scalar multiple* of the solid matrices:

```
K(c) = [E_min + c^{p_K}(E₀ − E_min)] · K̂   =  a(c) · K̂
M(c) = [ρ_min + c^{p_M}(ρ₀ − ρ_min)] · M̂   =  b(c) · M̂
```

Substituting into `K Φ = ω² M Φ` gives `K̂ Φ = (ω² b/a) M̂ Φ`. Therefore:

> **The eigenvectors of a uniform design are *identical* to those of the solid domain.**
> Only the eigenvalues rescale: `ω²(c) = [a(c)/b(c)] · ω²(1)`.

So `Φ₀(uniform V_f) = Φ₀(solid)` **exactly**, and the two candidate baselines differ *only* in
the scalar `ω₀²`. Since `f = ω₀² M(x) Φ₀` enters a linear static solve, scaling `f` by `κ`
scales `u` by `κ` and `∂c/∂x` by `κ²`; the OC Lagrange multiplier found by bisection absorbs
`κ²` exactly, leaving the update ratio — and hence the entire design trajectory — unchanged.

**Consequence:** for the SS beam (uniform start, no passive elements) the uniform-vs-solid
question **cannot change A4's topology or its true `ω₁`.** It changes only the *reported* `ω₁⁽⁰⁾`
and therefore every gain ratio `ω̃₁/ω₁⁽⁰⁾`. The decision is safe for A4's primary endpoint and
consequential for the manuscript's reported gains.

**Two caveats, both to be checked rather than trusted:**
- The invariance relies on OC's multiplier bisection having a wide enough bracket. If the
  bracket is hardcoded (e.g. `[0, 10⁹]`) and `κ²` pushes `λ*` outside it, invariance breaks as
  an *implementation* artifact. V-A4-1 must confirm it empirically.
- It does **not** extend to the **building**, which has passive solid regions: its initial
  field is not uniform, so `Φ₀(initial) ≠ Φ₀(solid)` there, and its reported `ω₁⁽⁰⁾ = 19.84`,
  `ω₂⁽⁰⁾ = 90.93` and its `4.61×` gain **do** depend on this choice. That is Reviewer 2's C2 and
  it is outside A4's scope — but it must be fixed, because A4 is about to make the reference
  design a load-bearing quantity.

## 3.4 Locations that must adopt the single definition

| # | Location | Required change |
|---|---|---|
| 1 | `paper/main.tex` §3.2 ([:645](../../paper/main.tex#L645)) | "solved once on the uniform density field `ρ_e⁽⁰⁾ = V_f`" → the **solid** reference domain. |
| 2 | `paper/main.tex` [:661](../../paper/main.tex#L661) | The Rayleigh argument ("the initial **uniform** design is a reasonable starting estimate") must be restated for the solid reference. By §3.3 the mode *shape* is the same for uniform-start domains, so the argument survives — but the wording must be made true. |
| 3 | `paper/main.tex` §4.3 | State the density at which `ω₁⁽⁰⁾ = 19.84`, `ω₂⁽⁰⁾ = 90.93` were computed (Reviewer 2, C2). If the building's baseline changes, **the `4.61×` and `2.53×` gains must be recomputed** — they are ratios to `ω₁⁽⁰⁾`. |
| 4 | `examples/Revision_v1/ss_beam.json` | `semi_harmonic_baseline: "initial"` → `"solid"`; **remove** `semi_harmonic_rho_source` (Gate A0 forbids it). **This config would error today.** |
| 5 | `examples/Revision_v1/ss_beam_harmonic_{frozen,periodic}.json` | Retired EXP4 variants. Archive under `archive/obsolete_evidence/`; they must never be A4 inputs. |
| 6 | `examples/Revision_v1/ablation_*.json` | EXP4 configs; retired. Same treatment. |
| 7 | `scripts/revision_v1/IMPLEMENTATION_MAP.md` A4-1…A4-4 | Replace with a pointer to this document. |
| 8 | `examples/Revision_v1/revision_v1_update1.md` §A4 | Same. |
| 9 | `analysis/ourApproach/Python/topopt_freq.py` | Confirm the same guard exists (A0-F4 requires MATLAB/Python parity on the reference definition). |

---

# PART 4 — Response variables

**The surrogate may not judge itself.** The optimizer minimizes compliance under `f = ω₀²M(x)Φ₀`.
That objective is a *monotone function of its own assumption*: a frozen `Φ₀` will always look
excellent when scored against a compliance functional built from that same `Φ₀`. Every accuracy
claim must therefore be read from an **independent exact eigensolve of the final design.**

## 4.1 Primary endpoint

> **Phase-2 amendment:** mode-window and screening provisions in this section are
> superseded by `A4_RECOVERY_PHASE2_SPECIFICATION.md` §§3–4. Endpoint definitions
> not explicitly amended remain binding.

> **`ω₁ᵗʳᵃᶜᵏ`** — the true frequency of the **Φ₁-type mode** in the converged design,
> from an exact generalized eigensolve `K(x*)φ = ω²M(x*)φ` on the final density field.
>
> The tracked mode is the eigenvector `j*` with **maximum mass-weighted MAC against `Φ₀`**
> (A0-F4 definition, consistent with manuscript Eq. 9), taken over the first 20 modes.

This — not the lowest eigenvalue, and not the surrogate objective — is the quantity `H₀`/`H₁`
are stated on. The manuscript's own reported gains are MAC-tracked, so this is the
apples-to-apples measure.

## 4.2 Companion frequency measures (the contamination detector)

| Symbol | Definition | Purpose |
|---|---|---|
| `ω₁ᵐⁱⁿ` | Lowest eigenvalue of the final design, whatever mode it is | The *conventional* `ω₁`. |
| `j*` | Index of the tracked mode in the sorted spectrum | Detects mode-ordering shift (the `j*=3` phenomenon already seen in the clamped/building cases). |
| **`ω₁ᵗʳᵃᶜᵏ − ω₁ᵐⁱⁿ` gap** | Divergence between the two | **Primary spurious-mode signature.** A large gap means a non-physical mode has descended below the design mode. |
| `ω₁ᵗʰʳᵉˢʰ` | `ω₁ᵗʳᵃᶜᵏ` recomputed on a **volume-preserving thresholded** design | Gray material is where localized modes live. If `ω₁ᵗʳᵃᶜᵏ ≈ ω₁ᵗʰʳᵉˢʰ`, the result is not a gray-material artifact. If they diverge, it is. |

Reporting `ω₁ᵐⁱⁿ` alongside `ω₁ᵗʳᵃᶜᵏ` is a matter of integrity: it is the number a reader would
naively compute, and any divergence must be shown, not hidden behind mode tracking.

## 4.3 Per-iteration observables (recorded for every arm)

| Observable | Notes |
|---|---|
| **MAC history** | Mass-weighted MAC of the tracked mode against `Φ₀` (the *original solid* reference) **and** against the *previously used* `Φ` — the second detects refresh-to-refresh drift. |
| **Mode identity** | Tracked index `j*` per refresh; a full record of every reindexing event. |
| **Refresh count** | Number of eigensolves actually performed. Predicted analytically: `1` for `N=∞`; `1 + ⌊(n_iter−1)/N⌋` for finite `N`. Discrepancy = implementation bug (V-A4-3). |
| **Objective history** | Surrogate compliance. **Diagnostic only.** It may describe the optimizer's behaviour; it may never be used to compare arms, because the objective is *not the same functional across arms* (each refresh redefines it). This point is subtle and essential: comparing surrogate objectives across `N` is meaningless. |
| **Optimization history** | `max|Δx_e|` per iteration (log scale). Distinguishes decay from a **limit cycle** — the period-2 signature CR2 already observed. |
| **Feasibility** | `|V(x) − V_f|/V_f` per iteration. |
| **Convergence** | Final `max|Δx_e|`, iteration count, `capped` flag. |
| **ω₁/ω₂ separation** | Tracked at each refresh (and at the endpoint for `N=∞`). Addresses Reviewer 2's coalescence concern: the frozen mode is predicted to degrade as `ω₁ → ω₂`. |
| **Mode-admissibility screen** | Per refresh, for every candidate eigenvector — see §4.3.1. **This is the B3 discriminator.** |
| **Grayness** | `ḡ = mean(4x(1−x))` of the final design. |

### 4.3.1 Mode-admissibility screen (support-connectivity based)

> **Phase-2 amendment:** superseded by `A4_RECOVERY_PHASE2_SPECIFICATION.md` §§3–6
> (adaptive ladder, common diagnostic grid, complete candidate telemetry, and
> deferred operational refresh).

**Corrected 2026-07-14.** The screen was originally specified as *"fraction of kinetic energy in
elements with `x_e < 0.1`"*. **That quantity is `0.0000` for every mode in every diagnosed run, at
every `pmass` tested.** A screen built on it would never fire, and A4 would have accepted
contaminated refresh arms as accuracy evidence — the precise failure it exists to prevent.

The spurious modes in this repository are **not** void-mass modes. They are **solid components not
connected to the supports**, whose kinetic energy sits in the island and whose strain energy sits
in the surrounding weak `E_min` material. The screen must therefore test **connectivity**, using
quantities the S1 post-processing already computes:

A candidate eigenvector `φ` is **admissible** iff all of:

| Quantity | Admissibility condition | Observed: physical mode | Observed: spurious mode |
|---|---|---|---|
| `largest_support_component_kinetic_fraction` | **≥ 0.5** — most kinetic energy on the support-connected component | 0.937 | 0.0006 |
| `dominant_component_touches_both_supports` | **true** | true | false |
| `low_density_strain_fraction` | **≤ 0.5** — strain energy not parked in void | 0.022 | 0.998 |
| mass-weighted MAC to the previously used `Φ` | **≥ 0.8** (continuity) | — | — |

*(Reference values from `s1_mitigation_400x50_mode_summary.json`, modes 1 and 2.)*

The separation is roughly three orders of magnitude on the first quantity, so the screen is not
threshold-sensitive. If **no** candidate among the first 20 modes is admissible, the refresh does
**not** silently pick one: it records a **B3 event** and terminates the arm as Class C.

**Also recorded per refresh (diagnostic, not screening):** the number of disconnected solid
components in the current design, and the kinetic-energy fraction in `x < 0.1` elements — the
latter retained precisely so that its being `≈ 0` is documented rather than assumed.

## 4.4 Measured covariate — not a factor

| Covariate | Definition |
|---|---|
| **Omitted-term ratio** | `‖(∂f/∂x)ᵀ·λ‖ / ‖∂c/∂x‖` — the magnitude of the load-sensitivity term the method discards, relative to the term it keeps, evaluated at the current design. CR2 measured this at **71.3%** at the initial design. |

This is **measured, never varied.** It is what makes the CR2 confound *detectable* rather than
merely *acknowledged*: as `N → 1` the load becomes progressively more design-dependent, so the
omitted term should grow. If an arm destabilizes while this ratio is large, the instability is
attributable to *(refresh × omitted sensitivity)* — **not** to freezing. Varying it would turn
A4 into CR2 and violate the single-factor rule.

## 4.5 Computation cost

Eigensolve count per arm, and wall-clock time.

**Governance:** the eigensolve count is `1` versus `1 + ⌊(n_iter−1)/N⌋` — it is **true by
construction**, and it is reported as an *analytic* operation count, not as a measurement.
Wall-clock time is recorded for provenance only and **may not appear in any table, figure, or
claim.** A4 must not be allowed to regrow into a performance benchmark; that is precisely the
narrative EXP1 and EXP5 were retired to prevent.

## 4.6 The separation that matters

| **Scientific outcomes** (the findings) | **Algorithm failures** (bugs) |
|---|---|
| MAC decay; loss of the tracked mode | Solver exception; stack trace |
| Mode-ordering shift (`j* ≠ 1`) | `NaN`/`Inf` in `K`, `M`, `u`, or `ω` |
| `ω₁ᵗʳᵃᶜᵏ` differs across `N` | Eigensolver (ARPACK) non-convergence |
| Divergence of `ω₁ᵐⁱⁿ` from `ω₁ᵗʳᵃᶜᵏ` | Volume constraint violated beyond tolerance |
| Limit-cycle oscillation in `Δx` | Missing required artifact |
| **Iteration cap reached** | Config-hash mismatch; factor drift |
| Refresh selecting a localized mode | Non-deterministic replay |

**The left column must never trigger a rejection.** Every entry in it is a measurement of how
the approximation behaves. The previous plan's gate would have discarded most of them.

---

# PART 5 — Acceptance criteria

## 5.1 The category error being corrected

The campaign's standing rule — *"a capped run is a failure, not a result"* — is correct for
**evidence runs**, which must not report an unconverged endpoint as a result. It is **wrong for
A4**, whose purpose is to characterize failure. A gate that rejects mode loss and capping would
reject exactly the runs that carry the finding, and would accept only the arms that were never
in doubt. That is how an experiment guarantees a null result.

A4 is therefore gated on **integrity of measurement**, not on **success of optimization**.

## 5.2 Three outcome classes

> **Phase-2 amendment:** code B3 is retired and decomposed by
> `A4_RECOVERY_PHASE2_SPECIFICATION.md` §7. Other provisions remain binding.

### Class A — REJECTED (experiment failure; fix and re-run)

The run tells us nothing because the machinery was broken:

- exception, stack trace, or solver abort;
- `NaN`/`Inf` in any of `K`, `M`, `u`, `ω`, `Φ`, or the sensitivities;
- final eigensolve failed to converge (ARPACK), so no primary endpoint exists;
- volume constraint violated beyond feasibility tolerance (indicates a broken OC update);
- a required artifact is missing;
- **factor drift** — the arm's configuration differs from the base in *any* respect other than
  `N` (V-A4-2). This is the guard against the failure that voided the previous plan;
- **non-deterministic replay** — identical config yields a different result.

### Class B — ACCEPTED (clean run)

- converged (`max|Δx_e| < 10⁻³` **before** the cap);
- feasible;
- tracked mode retained (`MAC ≥ 0.8` against `Φ₀`);
- `ω₁ᵗʳᵃᶜᵏ ≈ ω₁ᵐⁱⁿ` and `≈ ω₁ᵗʰʳᵉˢʰ` (no contamination signature);
- all artifacts present.

Only Class B arms may serve as an **accuracy reference** in the `H₀`/`H₁` decision.

### Class C — ACCEPTED WITH APPROXIMATION BREAKDOWN (a valid scientific observation)

Artifacts complete, numerics sound, feasible — but the *approximation* misbehaved. **This is a
result, and it is reported, not discarded.** Every Class C run must be assigned a mechanism:

| Code | Signature | Interpretation |
|---|---|---|
| **B1 — mode migration** | `MAC ≥ 0.8` but `j* ≠ 1` | The frozen mode is still *found*, but it is no longer the lowest. The approximation is intact; the *ordering* shifted. **Not a failure of freezing.** (Already observed at `j*=3` in the clamped/building cases.) |
| **B2 — frozen-mode breakdown** | `MAC < 0.8` against `Φ₀`, in a run that is otherwise clean, **and** a clean refreshed arm attains higher `ω₁ᵗʳᵃᶜᵏ` | **The finding Reviewer 1 asked for.** The initial mode is a poor proxy for the optimal mode. Supports H₁ and bounds the method's scope. |
| **B3 — spurious-mode contamination** *(discriminator corrected 2026-07-14 — see §0.2)* | At a refresh, the selected eigenvector fails the **mode-admissibility screen of §4.3.1** — its kinetic energy sits on a solid component not connected to the supports, and/or its low-density **strain** fraction is high — and/or `ω₁ᵐⁱⁿ ≪ ω₁ᵗʳᵃᶜᵏ`. | The refresh locked onto a disconnected-island mode. **The arm is disqualified as an accuracy reference** and reported as contaminated. *This is EXP4's −62%, now detected instead of published.* **The former discriminator (kinetic energy in `x<0.1` elements) is void — that quantity is `0.0000` for every observed mode and would never have fired.** |
| **B4 — sensitivity-omission instability** | `Δx` history is a bounded limit cycle (period-2 signature) rather than decaying, **and** the omitted-term ratio (§4.4) exceeds its declared threshold | Non-convergence is attributable to *(refresh × omitted `∂f/∂x`)*. **Says nothing about freezing.** Expected most strongly at `N=1`, the regime CR2 already destabilized. |

`N = ∞` is not exempt. If the *published method itself* caps or loses its mode, that is a Class C
observation with a manuscript consequence — the convergence claim must be corrected. It is not
swept up as a "failed run."

## 5.3 Pre-registered decision rule

Fixed before execution:

1. **All five arms Class B, and `Δω₁ᵗʳᵃᶜᵏ(N) ≤ δ = 5%` for every finite `N`**
   → **H₀ retained.** Freezing costs nothing material on this benchmark. `main.tex:704`'s
   directional claim must be softened: refreshing confers no measurable benefit here.

2. **Some clean (Class B, or Class C/B1 or C/B2 only) arm exceeds `N=∞` by `> δ`**
   → **H₀ rejected, H₁ supported.** Freezing costs accuracy. Report the penalty, the breakdown
   threshold `N*`, and bound the scope. `main.tex:704` is evidenced.

3. **`N=∞` exceeds every finite arm by `> δ`, and those arms are Class C/B3 or C/B4**
   → **Neither hypothesis is supported.** The refresh reference is *unavailable*: the refreshed
   arms are contaminated or unstable, so they cannot referee the frozen arm. **This is EXP4's
   outcome.** It must be reported as such — *"a periodically-updated reference could not be
   constructed on this problem, for the following stated mechanism"* — and the frozen-mode
   reliability claim must fall back to the MAC-threshold route and be explicitly scoped.

4. **`N=∞` exceeds every finite arm by `> δ` with those arms *clean*** → a genuine and
   surprising result: refreshing *hurts*. Report it. Do not explain it away.

Declaring outcome 3 in advance is what prevents A4 from being retrofitted. It is the outcome
the previous design could not distinguish from outcome 4 — and it published the wrong one.

---

# PART 6 — Dependencies

Reasoned, not assumed.

## 6.1 Does A4 depend on S1? — **YES, and the dependency is now HARDER than the previous version claimed.**

> **Phase-2 amendment:** the four-point checkpoint set is superseded by the common
> grid `G` in `A4_RECOVERY_PHASE2_SPECIFICATION.md` §4.1.

The structural argument is unchanged and remains correct:

> The refreshed arms — and **only** the refreshed arms — must solve an eigenproblem on
> **intermediate gray designs**. The frozen arm never does. The pathology is therefore
> **asymmetric across the levels of the independent variable**, which is the textbook definition
> of a confound.

**What has changed is the escape route — there isn't one.**

The previous version of this section argued that the pathology is *generated by the mass exponent*,
and that pinning `p_M = 3` would collapse it and free A4 from S1. **That argument is withdrawn**
([`MASS_INTERPOLATION_DECISION.md`](../../MASS_INTERPOLATION_DECISION.md), §0.2). The evidence:

- Three mass models — linear (the declared method), `pmass = 6`, and Du & Olhoff Eq. 4b — were
  each tested on the same 400×50 case. **None removes the localized-mode family** (8, 9, 9 of the
  first 10 modes flagged).
- The flagged modes carry `low_density_kinetic_fraction = 0.0000` in every case. They are **not**
  void-mass modes; they are **disconnected solid components**.

> **A4 therefore cannot configure its way to a clean spectrum.** The contamination that destroyed
> EXP4 is real, is unexplained in mechanism, and is **not** removable by any mass setting available
> to us. A4 must instead (a) *detect* it, via the corrected support-connectivity screen (§4.3.1),
> and (b) *decide empirically* whether the SS beam is affected at all, via Gate A4-Pre.

Two consequences:

1. **A4 pins `pmass = 1`** (Part 2) — the declared method — and gains **no protection** from doing
   so. The pin is for correctness, not for safety.
2. **The instruction "A4 is independent of S1, start it first" remains withdrawn.** It was wrong in
   the previous version for the wrong reason; it is still wrong, for a stronger one.

**The one genuine unknown, and it is decidable.** All of the disconnected-component evidence comes
from the **clamped-beam** 400×50 designs (EXP3 / S1). The **SS beam** — A4's benchmark — is the
cleanest tracker in the suite (MAC 0.9998 for the frozen method's endpoint). Whether its
*intermediate* designs carry the same pathology is **not known**, and no artifact answers it. That
is exactly what Gate A4-Pre is for.

### Gate A4-Pre — spectral admissibility screen (mandatory, cheap, and now decisive)

Run the `N=∞` arm **first** (it is needed anyway). At iterations `{100, 300, 600, final}` take its
design and apply the **§4.3.1 mode-admissibility screen** to the first 10 modes, and record the
disconnected-component count. Then:

- **PASS** — an admissible `Φ₁`-type mode (support-connected, `MAC ≥ 0.8` to `Φ₀`) is identifiable
  at every checkpoint. → The SS beam's intermediate spectra are usable; **the refreshed arms have
  something clean to refresh into, and A4 proceeds.** A4 is then genuinely independent of S1's
  mitigation work — not because the mass model protects it, but because this benchmark is
  empirically unaffected.
- **FAIL** — the low spectrum is dominated by disconnected-island modes at any checkpoint. →
  **A4 is blocked on S1.** No mass setting will rescue it, and refreshing cannot be made
  meaningful. **This must be reported, not worked around** — it is pre-registered decision-rule
  outcome 3 (§5.3).

Cost: one frozen-arm run (needed regardless) plus four eigensolves. **It converts an assumed
dependency into a measured one for minutes of compute, and it is now the only thing preventing a
16-hour repeat of EXP4.**

Cost: one frozen-arm run (needed regardless) plus four eigensolves. **This converts an assumed
dependency into a measured one, for minutes of compute, and it prevents a 16-hour repeat of EXP4.**

## 6.2 Does A4 depend on CR2? — **NO blocking dependency. A shared instrument, and a declared interaction.**

The omitted `∂f/∂x` is a **fixed factor** in A4, not a varied one. A4 does not need CR2's verdict
to execute.

But the two are **not independent in interpretation**, and the plan's claim that "CR2 and A4 are
independent and can run in parallel" is wrong in one direction:

> As `N → 1`, the load becomes fully design-dependent, so the term the method omits grows in
> relative magnitude. `N = 1` therefore sits in precisely the regime where the omitted-sensitivity
> approximation is least defensible — the regime where CR2 already observed a **period-2
> instability** and measured the omitted term at **71.3%** of the retained one.

A4 handles this **without** violating the single-factor rule, by *measuring* the omitted-term ratio
as a covariate (§4.4) and using it **only for classification** (B4). If `N=1` oscillates, A4 reports
*"non-convergence at N=1, with omitted-term ratio X% — attributable to refresh × omitted load
sensitivity, not to the frozen approximation."*

**Governance in both directions:** A4 must not be used to settle CR2, and CR2 must not be used to
settle A4. They share a diagnostic, not a conclusion. If CR2 completes first, its accepted
omitted-term threshold should be adopted as A4's B4 discriminator; if not, A4 declares its own
threshold in advance.

## 6.3 Summary

| Dependency | Verdict |
|---|---|
| **S1** | **Real, and decidable only by measurement.** A4 pins `pmass = 1` (the declared method) and gains **no protection** from it: no mass setting removes the contamination. Gate A4-Pre determines empirically whether the SS beam's *intermediate* spectra are affected at all — the disconnected-component evidence comes from the clamped beam, and the SS beam is untested. **A4 must not be "started first because it is independent"** — that instruction stays withdrawn. |
| **CR2** | **No blocking dependency.** Shared covariate (omitted-term ratio); declared interaction at small `N`; conclusions kept separate. |
| **EXP2 / EXP2b / EXP3** | **None for execution.** But A4 alone cannot support the manuscript's *cross-class* robustness claim (Part 8). |

---

# PART 7 — Implementation roadmap

Specification only. No code is written here.

## 7.1 Required new capability (the entire implementation burden)

**R-1 — refresh interval on the `semi_harmonic` load path.**

Today `update_after` is honoured only under `case 'harmonic'`. A4 requires the equivalent on
`semi_harmonic`, with these exact semantics:

- `update_after = 0` (or `Inf`) → **frozen**: `(ω₀, Φ₀)` computed once on the solid reference.
  **Must be bit-identical to today's behaviour.**
- `update_after = k > 0` → at every iteration `i` with `i mod k = 0`, recompute the eigenpair on
  the **current design** `x⁽ⁱ⁾` and replace the reference used to build `f = ω₀²M(x)Φ₀`.
- At each refresh the mode is selected by **MAC continuity against the previously used `Φ`**,
  among modes passing the localization screen — never by raw index. Every refresh event records
  the chosen index, MAC, localization metric, and `ω`.
- If **no** mode passes the screen, the run does not silently pick one: it records a **B3 event**
  and terminates that arm as Class C.

**Hard regression constraint:** with `N = ∞`, results must be **bit-identical** to the unmodified
solver. The new code path must be additive and inert when unused (V-A4-6). This is what allows a
solver change without disturbing a frozen campaign.

## 7.2 Scripts

| Path | Role |
|---|---|
| `examples/Revision_v1/a4_eigenpair_refresh.m` | A4 driver. Loops the five `N` levels over **one** base config, runs Gate A4-Pre first, writes per-arm artifacts. Signature `(outDir)` for runner compatibility. |
| `examples/Revision_v1/a4_preflight_spectral_screen.m` | Gate A4-Pre (§6.1). Callable standalone. |
| `scripts/revision_v1/check_a4_run.m` | The Part-5 classifier: `REJECTED` / `ACCEPTED` / `ACCEPTED_WITH_BREAKDOWN{B1..B4}`. **Single implementation of the rule**, mirroring `check_revision_run.m`. |
| `scripts/revision_v1/test_a4_classifier.m` | Synthetic-fixture tests for the classifier — including a fixture reproducing EXP4's −62% and asserting it classifies as **B3**, not as evidence. |
| `scripts/revision_v1/validate_a4_configs.py` | Factor-drift validator (V-A4-2). |

## 7.3 Configuration

**One base config, `N` injected by the driver.**

`examples/Revision_v1/a4_ss_400x50_base.json` — SS beam 400×50, OC, `semi_harmonic`,
`baseline: "solid"`, load sensitivity omitted, `p_K = 3`, **`pmass: 1` (linear — stated
explicitly, not left to the solver default, so the arms cannot drift)**, `E_min = ρ_min = 1e-9`,
`r_min = 2`, `V_f = 0.5`, `move 0.2`, `max_iters 2000`, `tol 1e-3`, `harmonic_normalize: false`,
no Heaviside, no continuation.

Five sibling JSONs are **rejected as a design**: they are how `ss_beam_harmonic_*.json` drifted
into four simultaneous factor changes. The base config's hash is recorded in every arm's result;
V-A4-2 asserts it is identical across arms.

`ss_beam.json` is **not** the base: it sets `semi_harmonic_baseline: "initial"` and carries
`semi_harmonic_rho_source`, and would throw under Gate A0 today.

## 7.4 Runner integration

- Replace the A4 placeholder (`localMakePlaceholderStage`) with a real stage dispatching to
  `a4_eigenpair_refresh`.
- **Do not add a stage.** The accepted campaign graph is `S1 → EXP2 → EXP2b → EXP3 → A4`; Gate
  A4-Pre runs *inside* the A4 stage and aborts with a specific identifier
  (`run_all:A4SpectrumInadmissible`) naming S1 as the blocker. The graph is unchanged.
- `dependsOn`: `{}` for execution, **but** the stage must fail loud if `pmass ≠ 1` or if the
  baseline is not `solid` — the two preconditions that make A4 meaningful.
- `localAccept_A4` must implement Part 5 — and in particular must **not** reject on `MAC < 0.8` or
  on iteration cap. Both are results.
- Preflight **P2** must additionally deny `ss_beam_harmonic_frozen`, `ss_beam_harmonic_periodic`,
  and `ablation_*` by name, so the retired EXP4 configs can never re-enter A4.

## 7.5 Result schema (per arm)

```
a4_result[N]:
  config:      N, base_config_hash, commit_sha, baseline="solid", pmass, load_sensitivity
  primary:     omega1_tracked, mode_index_jstar, mac_to_phi0
  companion:   omega1_min, omega1_thresholded, omega1_omega2_gap
  refresh:     n_eigensolves, events[{iter, index, mac_prev, mac_phi0, localization, omega}]
  histories:   design_change[], objective[], feasibility[], mac[], jstar[]
  covariate:   omitted_term_ratio[]
  termination: converged, capped, n_iter, final_design_change
  quality:     grayness, feasibility_final
  cost:        eigensolve_count (analytic + observed), wall_clock  # provenance only
  class:       ACCEPTED | ACCEPTED_WITH_BREAKDOWN | REJECTED
  breakdown:   [] | B1 | B2 | B3 | B4   (+ evidence fields)
```

## 7.6 Plots and tables

**Figures**
1. **`ω₁ᵗʳᵃᶜᵏ` vs `N`**, with the `±δ` equivalence band around `N=∞`. *The primary figure.* Arms
   classified B3/B4 are plotted in a visually distinct, explicitly disqualified style — shown, but
   never allowed to read as accuracy evidence.
2. MAC vs iteration, per arm, with refresh events marked.
3. `max|Δx_e|` vs iteration (log). Exposes limit cycles (B4).
4. Tracked mode index `j*` vs iteration.
5. Spectrum + localization metric at each refresh event (the B3 diagnostic).
6. Final topologies, five panels.
7. `ω₁`/`ω₂` separation vs iteration (Reviewer 2's coalescence concern).

**Table A4-1** — one row per `N`: `ω₁ᵗʳᵃᶜᵏ`, `ω₁ᵐⁱⁿ`, `ω₁ᵗʰʳᵉˢʰ`, MAC, `j*`, iterations, converged,
eigensolves, grayness, omitted-term ratio, **class**, and `Δω₁ vs N=∞` — the last column populated
**only** for clean arms, and left explicitly blank (not zero, not a dash) for B3/B4.

## 7.7 Provenance

Per-arm manifest; base-config hash; commit SHA; hardware/software block (runner already emits it);
full artifact list; determinism replay record. Standard campaign governance.

## 7.8 Validators

| ID | Assertion |
|---|---|
| **V-A4-1** | **Baseline invariance.** `MAC(Φ₀ᵘⁿⁱᶠᵒʳᵐ, Φ₀ˢᵒˡⁱᵈ) ≥ 0.9999`; `ω₀²` ratio matches the predicted `a/b` scalar; the `N=∞` design is invariant to the choice. *Confirms §3.3 empirically — including that OC's multiplier bisection bracket does not break scale-invariance.* |
| **V-A4-2** | **Single factor.** All arms share an identical base-config hash; `N` is the only differing key. |
| **V-A4-3** | **Refresh accounting.** Observed eigensolve count `= 1 + ⌊(n_iter−1)/N⌋`. |
| **V-A4-4** | **Operation count.** The `N=∞` arm performs exactly **one** eigensolve at initialization and **one** at final verification, and **zero inside the loop**. *This directly evidences the manuscript's surviving efficiency claim.* |
| **V-A4-5** | **Determinism.** Replay of any arm reproduces its result. |
| **V-A4-6** | **Inertness.** With `N=∞`, the refresh code path produces results **bit-identical** to the pre-R-1 solver. |
| **V-A4-7** | **Classifier.** `check_a4_run.m` unit tests pass, including the EXP4 −62% fixture → **B3**. |

---

# PART 8 — Reviewer 1 perspective

*Reading as Reviewer 1, who wrote: "the paper does not demonstrate conditions under which the
approximation fails… needed to bound the practical scope of the method."*

## Would this revised A4 completely answer my concern? — **No. It answers the core of it, and it is necessary, but it is not sufficient.**

**What it does answer, and answers well.** It gives me a controlled, single-factor comparison
against a periodically-updated-mode variant — the third diagnostic I offered — judged by the *true*
frequency rather than by the surrogate that is under suspicion. It reports the accuracy penalty of
freezing against a margin declared in advance. It distinguishes a genuine breakdown of the frozen
mode (B2) from mode reordering (B1), from spurious-mode contamination (B3), and from an instability
caused by the omitted load sensitivity (B4). That last distinction is important to me: it means the
authors cannot present an artifact as a finding, and — reading the EXP4 history — they very nearly
did.

**What remains missing.** Four things, and I would say so in a second-round report:

1. **It runs on the easiest problem in the paper.** The SS beam has the cleanest mode tracking in
   the suite (MAC 0.9998 at the frozen endpoint), and Gate A4-Pre exists precisely because we do
   not know whether its *intermediate* spectra are clean. A refresh sweep on
   the *most favourable* case is at real risk of finding "no penalty" and then generalizing that to
   the method. **The stress cases are the clamped beam and the building** — the very ones where the
   targeted mode migrates to `topo-mode 3` and where topo-modes 1 and 2 show `MAC < 0.10`. A4 tells
   me the penalty is small *where I already believed it would be small.*
2. **Therefore it cannot support the manuscript's cross-class claim.** [`main.tex:661`](../../paper/main.tex#L661)
   asserts *"robustness of the quasi-static approximation across structurally different problem
   classes."* A4 on one class cannot establish that. That claim must either be narrowed to the
   benchmark A4 actually covers, or supported by the accepted MAC-threshold evidence from EXP2 /
   EXP2b / EXP3 — **none of which is currently accepted** (EXP2: 0 of 5 alphas; EXP3: mode-invalid;
   EXP2b: capped). Until those land, the sentence is unsupported whatever A4 returns.
3. **It covers `α = 1` only.** The paper's most interesting claim is the α-weighting rule for
   multi-mode targeting. The frozen-eigenpair question applies to `Φ₂` exactly as it does to `Φ₁`,
   and a frozen `Φ₂` under a weighted load is *a priori* the more fragile case. A4 says nothing
   about it. I would accept an explicit scoping statement here rather than a second sweep — but it
   must be stated, not left silent.
4. **My first suggestion is still unaddressed.** I offered *"a benchmark where the initial and
   optimized mode shapes differ substantially."* Nobody has constructed one. A4 measures the penalty
   where the shapes happen to agree; it does not probe the regime where they do not. The honest
   version of the paper's limitation section needs at least one case — even a small, cheap,
   deliberately adversarial one — where `Φ₀` is a known-poor proxy, to show the failure mode is
   understood rather than merely avoided.

**My verdict as reviewer:** A4-v2 is a genuine, well-controlled experiment and I would credit it. It
converts the frozen-eigenpair assumption from an unexamined premise into a measured one **on the SS
beam.** But the manuscript's scope claim must be narrowed to what A4 actually covers, and points 3
and 4 must be met by explicit, honest limitation text — or by evidence.

---

# PART 9 — Final decision

## Is A4 now scientifically well-defined? — **YES.**

Every element that made the previous plan invalid is now closed:

| Previous defect | Resolution |
|---|---|
| Reused retired EXP4 configs | New single base config; retired configs denied by name in preflight P2 |
| Varied 4 factors besides `N` | Exactly one IV; all other factors enumerated, justified, and enforced structurally (one base config) and mechanically (V-A4-2) |
| Surrogate judged by the surrogate | Primary endpoint is true `ω₁` from an independent exact eigensolve; the surrogate objective is explicitly barred from cross-arm comparison |
| `N=∞` undefined | Fixed as the **solid domain** — already the accepted decision, already enforced by the solver, and shown to be *immaterial* to A4's endpoint by the invariance result of §3.3 (to be confirmed by V-A4-1) |
| Gate rejected approximation failure | Three-class framework; approximation failure is a **valid observation (Class C)** with four discriminated mechanisms |
| Confounded with S1 and CR2 | S1: the confound is **real and not configurable away** — no mass model removes it. A4 therefore *detects* it (support-connectivity screen, §4.3.1) and *measures* whether the SS beam is affected at all (Gate A4-Pre). CR2: no blocking dependency; interaction handled by a measured covariate and the B4 classifier |

**A4 is well-defined as a protocol.** Note the distinction that matters: *well-defined* is not the
same as *unconditionally executable*. A4 carries one declared, decidable precondition — Gate A4-Pre —
whose failure is itself a reportable scientific outcome (pre-registered as decision-rule outcome 3).
A protocol with a stated admissibility gate and pre-registered null outcomes is well-defined; one
that would have silently published its own contamination was not.

**Two preconditions must be met before A4 runs.** Neither is a research question:

1. **`pmass = 1` (linear)** must be stated explicitly in the base config — the declared method per
   [`MASS_INTERPOLATION_DECISION.md`](../../MASS_INTERPOLATION_DECISION.md). It is already the
   solver default and what every production run uses, so this pins existing behaviour against drift;
   it does **not** change any numerics. *(Superseded: this precondition formerly demanded
   `pmass = 3`, on the strength of the manuscript's erroneous equation.)*
2. **`semi_harmonic_baseline = "solid"`** must be used, matching Gate A0 and the solver's own guard
   (`topopt_freq:AuthoritativeBaselineRequired`); `ss_beam.json`, which sets `"initial"`, would
   throw today.

Plus one new capability: **R-1**, the refresh interval on the `semi_harmonic` load path — which does
not currently exist, and whose absence is the reason the previous plan went wrong.

> ## **A4_SPECIFICATION_V3 is the authoritative experiment definition.**

---

## Appendix — the mass-model correction, and what it cost this specification

**The v1 appendix claimed that the spurious-mode pathology was "generated by a configuration
default (`pmass = 1`)" that contradicted the manuscript, and that pinning `pmass = 3` would
dissolve it. That claim was wrong, and it is retracted.**

The independent audit and [`MASS_INTERPOLATION_DECISION.md`](../../MASS_INTERPOLATION_DECISION.md)
established that **the implementation was right and the manuscript was wrong**: the declared mass
law is linear, a global `d = p = 3` is unphysical (it makes frequency independent of uniform
density and misses Du & Olhoff's published initial frequencies by 2×), and — decisively — the
void-mass mechanism I invoked **is not what these runs exhibit**:
`low_density_kinetic_fraction = 0.0000` for every mode, at every `pmass` tested. Three mass models
were tried; none removes the family. The modes are **disconnected solid components**.

**Two things in this specification changed as a direct result, and one of them was a latent defect
of the first order:**

1. **The B3 contamination detector was inoperative.** It was specified to fire on kinetic energy in
   low-density elements — a quantity that is **identically zero** in every observed case. As
   written, **it would never have fired.** A4 would have run its refresh arms into a polluted
   spectrum, failed to detect the pollution, and published it as an accuracy result — which is
   precisely, and ironically, the EXP4 failure this specification was written to prevent. The
   detector is now **support-connectivity based** (§4.3.1), keyed on quantities that separate the
   physical from the spurious modes by three orders of magnitude.

2. **A4 lost its escape route from S1.** The v1 dependency argument was "pin the right mass
   exponent and the confound disappears." No mass setting removes the confound. A4 can now only
   *detect* the contamination and *measure*, via Gate A4-Pre, whether the SS beam suffers it at all
   — which remains genuinely unknown, since all the disconnected-component evidence comes from the
   clamped beam.

**What survives untouched:** the single-factor design, the true-`ω₁` endpoint, the prohibition on
the surrogate judging itself, the unique `N=∞` baseline and its invariance result, the three-class
acceptance framework, the pre-registered null outcome, and the finding that `update_after` does not
exist on the `semi_harmonic` path. The mass-model error did not touch the experimental logic — it
corrupted one detector and one dependency argument, and both are now corrected.

**Standing conclusion:** the campaign is testing a linear mass model, the manuscript now says so,
and the spurious modes remain **unexplained in mechanism**. A4 must be run with that uncertainty
declared, not designed away.
