# Plan — exact implementation of Olhoff & Du (2014) in `analysis/OlhoffApproachExact`

Target paper: `references/Olhoff2014_Structural.pdf` — N. Olhoff, J. Du,
*Structural Topology Optimization with Respect to Eigenfrequencies of Vibration*,
CISM Courses and Lectures, Springer 2014, pp. 275–297. DOI 10.1007/978-3-7091-1643-2_11.

Scope of this plan: the 2D beam problems of §3.1 (max ω₁), §3.2 (max ω₂) and
§3.3 (max gap ω₃−ω₂), implemented per Fig. 1 (main loop + inner loop) with a
correct multiple-eigenfrequency (bimodality) path. 3D plate examples §3.4–3.6
are explicitly out of scope.

Status of this document: implementation plan, written 2026-07-30. It supersedes
`OlhoffApproachExact.txt` (which targets Du & Olhoff 2007 and carries an
ARCHIVE STATUS block). See §9.3 for what happens to that file.

---

## 0. Findings from the investigation

### F1 — There is no source code left in this directory

Working tree of `analysis/OlhoffApproachExact/` currently contains only
`experiments/` result artifacts, `Matlab/results/`, and `OlhoffApproachExact.txt`.
All 24 `.m` files were deleted in commit `6f2e04c` ("clean main", 2026-07-01)
and are not on disk.

* Last recoverable version: `git show 6f2e04c^:analysis/OlhoffApproachExact/Matlab/<file>`
  (= `b98cc96`, 2026-06-18). 24 files, ~4000 lines.
* **Unrecoverable**: the version actually used by the July campaigns
  (`mesh_resolution_campaign`, `faithful_reconstruction`, both 2026-07-29 —
  recorded as HEAD + ~672 lines of default-off diagnostics) was never committed.
  Its results in `experiments/` cannot be regenerated from any commit.
* Side observation, not in scope: `analysis/OlhoffApproach/Python/mma.py` is
  also gone while `solver.py:11` still does `from .mma import mmasub` — that
  Python solver is currently unrunnable.

**Consequence for this plan:** Phase 0 is a recovery phase, and every phase ends
with a commit. Losing the solver again is the single cheapest failure to avoid.

### F2 — Olhoff2014 is the optimizer authority, not the complete specification

What the chapter contains in full:

| Item | Eq. |
|---|---|
| max–min problem, volume + box constraints | (1a)–(1e) |
| simple-eigenvalue sensitivity | (3), (4), (5) |
| bound formulation, n-th eigenfrequency | (9a)–(9c) |
| bound formulation, gap ω_n − ω_{n−1} | (10a)–(10d) |
| multiple eigenvalue: sub-eigenvalue problem | (12) |
| generalized gradient vectors **f**_sk | (13) |
| N=1 reduction; vanishing off-diagonal case | (14)–(18) |
| **computational procedure flow chart** | Fig. 1 |
| increment subproblem, n-th eigenfrequency | (19a)–(19f) |
| increment subproblem, gap | (20a)–(20i) |
| beam benchmarks | §3.1–3.3, Figs. 2–7 |

What the chapter does **not** contain, and must be imported or reconstructed:

* **Material interpolation.** Deferred to "the preceding paper Olhoff and Du
  (2013A)", §2 Eq. (3) and §2.3 — that paper is **not in `references/`**. The
  only interpolation statement in Olhoff2014 is Eq. (5),
  `(λ_j)'_ρe = φ_jᵀ(p ρ_e^{p−1}K_e* − λ_j q ρ_e^{q−1}M_e*)φ_j`,
  i.e. `K_e = ρ_e^p K_e*`, `M_e = ρ_e^q M_e*` with p and q unspecified. There is
  **no** modified low-density mass law anywhere in this chapter.
* No filter or regularization of any kind is mentioned.
* No mesh resolution for the 2D examples.
* No move limit, multiplicity tolerance, MMA parameters, or stopping tolerances.

`references/Du2007_Topological.pdf` (Du & Olhoff, SMO 34:91–110, cited in the
chapter's reference list) supplies all of these. Equation mapping:

| Olhoff2014 | Du2007 |
|---|---|
| (9) | (15) |
| (10) | (16) |
| (12) | (18) |
| (13) | (19) |
| (19) | (25) |
| (20) | (26) |
| Fig. 1 | Fig. 1 |

The Du2007 erratum (DOI 10.1007/s00158-007-0167-6) inserts Δ between δ_sk and
(ω²) in (25d)/(26f)/(26g). Olhoff2014 (19d)/(20f)/(20g) already print the
corrected form — no erratum handling needed for this target.

**Rule for this implementation:** every ingredient is tagged **[E]** explicit in
Olhoff2014, **[D]** imported from Du2007, or **[R]** reconstructed. §1 is that
table. Nothing goes in untagged.

### F3 — The forward FE model is already validated; reuse it verbatim

Q4 bilinear plane stress, 2×2 Gauss, consistent mass, `K_e = ρ^p K_e*` with no
additive `E_min`. Verified 2026-06-11: uniform ρ = 0.5 reproduces the Fig. 2
initial frequencies to <0.5 % (SS 68.4 / 104.1→103.7 / 146.1→145.6), all three
unimodal as the paper states, mass-interpolation derivatives FD-clean to <1e-5
on both sides of the ρ = 0.1 kink.

Two non-obvious facts to preserve:

* **The published initial frequencies are p = 3 penalized values.** ω² ∝ 0.5^{p−1},
  so p = 1 gives ≈143/211/295 — a clean factor of 2 too high. Evaluate the
  initial design at p = 3.
* **The SS/CS support is a pin at MID-HEIGHT of the end edge, not at the bottom
  corner.** Verified directly against Fig. 2(a) rendered at 300 dpi: the support
  triangle's apex sits on the left edge at ≈0.45–0.50 b. A bottom-corner pin
  gives ≈100 rad/s (arch action), not 68.7. Consequence: **`nely` must be even.**
  The recovered `build_supports_exact.m` uses `mid_idx = round(nely/2)+1`, and
  MATLAB rounds 2.5 away from zero, so at `nely = 5` the pin lands at y = 0.600 H
  and the SS/CS problems are not mirror-symmetric — a different structural
  problem, not a coarse approximation.

### F4 — Why every previous attempt failed is known, and it is mathematical

For N = 1, subproblem (19) is a **linear program** in (β, Δρ). The paper says so
itself: §2.5 final paragraph ("the sub-optimization problems (19) and (20) both
reduce to linear programming problems"), and after (20i), "(19) and (20) can be
solved using the MMA method (Svanberg 1987) or a linear programming algorithm".

Its exact optimum is a vertex of box ∩ volume. With one linking constraint, all
but one Δρ_e sits at a bound. Measured in `experiments/faithful_reconstruction`:
converged MMA reproduces the exact LP vertex to 0.02 % in objective, cos = 0.9976,
fraction of variables at a bound = 0.9997; that one accepted, feasible, in-bounds
step takes CC ω₁ from 145.57 to 0.0638. Reproduced at all four meshes
(492 → 14 942 DOF) in `experiments/mesh_resolution_campaign`.

**Therefore Fig. 1 as literally drawn — inner loop iterated to convergence, then
ρ := ρ + Δρ — cannot work, at any mesh, for any MMA setting.** This is not a bug
to find. Prior campaigns burned months treating it as one.

Corollary: `inner_max_iter = 30` in the old code never met its declared inner
stopping test (0/300 outer iterations, any mesh; 312–324 iterations were needed).
The truncation *was* the step control — undeclared, unreported, and the only
reason those runs produced non-zero frequencies at all.

### F5 — Fig. 4 is a high-information acceptance target that no prior campaign used

Fig. 4 (SS beam, 80 iterations) is a complete iteration history. Read off at
300 dpi (values ±3 %, iteration indices ±2):

| feature | value |
|---|---|
| ω₁ at iteration 1 | 68.7 |
| ω₁ trajectory | **monotone non-decreasing over all 80 iterations, no drop** |
| ω₂ at iteration 1 | ≈255 |
| ω₂ peak | ≈327 at iteration ≈7 |
| ω₃ at iteration 1 | ≈432 |
| ω₃ peak | ≈528 at iteration ≈9 |
| ω₁/ω₂ coalescence (bimodality onset) | iteration ≈20, at ω ≈160 |
| ω₁ final | 174.7 |
| ω₃ final | ≈288 (Fig. 5c gives 284.9) |

The monotone ω₁ is the important one: it is direct evidence that the step
actually taken in the paper was small and never overshot. Any reconstruction
whose ω₁ drops is wrong regardless of where it ends up. This 6-point fingerprint
becomes the primary acceptance test for step control (Phase 5), replacing
endpoint-only comparison.

### F6 — The missing ingredient is a step restriction, and adding one is reconstruction, not deviation

The paper's own LP reduction cites Krog & Olhoff (1999). Sequential *linear*
programming is not defined without a move limit; an LP subproblem over a box has
no interior optimum. Fig. 1's step 3 box is (19f) only, but a working SLP/SCP
implementation of it must have had a trust region. Its absence from the chapter
is a documentation gap.

So: add `|Δρ_e| ≤ m` to (19f), tag it **[R]**, calibrate it against the Fig. 4
fingerprint, and report m in every result. Do not hide it, and do not add five
interacting safeguards to disguise it. The old code had six (`alpha`, `move_lim`,
`outer_move`, `acceptance_check`, `max_freq_drop`, `min_alpha`); this
implementation has one.

**Honest framing of the deliverable:** what is achievable is *"Olhoff2014 (9)+(12)+(13)+(19)
and Fig. 1, implemented exactly, with one declared reconstructed parameter m and
the Du2007 interpolation/filter imports listed in §1"*. It is not
*"paper-literal with zero additions"* — F4 proves that object does not exist.

### F7 — Mesh and mass-model constraints from prior campaigns

* Use **≥160×20**. At 40×5 and 80×10 the optimizer produces disconnected
  two-block topologies, ω₁ collapse events, and mode-tracking failures; at
  160×20↔240×30 the consecutive-mesh topology correlation is 0.92–0.95 for all
  three BCs and 160×20 is already the connected full-span X-braced Fig. 3c class.
* SIMP penalty continuation p:1→3 is **refuted** as a connectivity fix (tested
  under three schedules; disconnection preferred even at p = 1). Use fixed p = 3.
* `du2007_c1` mass interpolation suppresses spurious localized low-density modes
  by ~5000× (mode-1 low-density strain fraction 0.000196 vs 0.992 for linear
  mass) — but it also makes void nearly massless, which is what rewards
  disconnection on coarse meshes. Both effects are real; §7.3 ablates it.
* `eigs` is deterministic here (seeds 0–5 bit-identical) — single runs are valid
  comparisons, no replicate averaging needed.
* The recorded `413.869` CC 40×5 baseline is **not** reproducible from
  `topopt_freq_exact` and must not be used as a reference.

---

## 1. Exactness contract

Every entry is fixed before coding and reproduced verbatim in the final report.

### [E] — explicit in Olhoff2014

| # | Item | Source |
|---|---|---|
| E1 | Objective `max β`, bound constraints `β − ω_j² ≤ 0`, j = n…J | (9a)(9b) |
| E2 | Volume `Σ ρ_e V_e − V* ≤ 0`, `V* = αV₀`, α = 0.5 | (1d), §3.1 |
| E3 | Box `0 < ρ ≤ ρ_e ≤ 1`, `ρ = 1e−3` | (1e), §2.1 |
| E4 | M-orthonormalization `φ_jᵀMφ_k = δ_jk` | (1c) |
| E5 | Simple sensitivity `(λ_j)' = φ_jᵀ(K'_ρe − λ_j M'_ρe)φ_j` | (4) |
| E6 | Interpolated form `φ_jᵀ(pρ^{p−1}K_e* − λ_j qρ^{q−1}M_e*)φ_j` | (5) |
| E7 | Generalized gradients `f_sk = {φ_sᵀ(K'_ρe − λ̃M'_ρe)φ_k}_e`, `f_sk = f_ks` | (13) |
| E8 | Sub-eigenvalue problem `det[f_skᵀΔρ − δ_sk Δ(ω²)] = 0` | (12), (19d) |
| E9 | Increment subproblem (19a)–(19f); J = n + N guard constraint (19b) | (19) |
| E10 | Gap subproblem (20a)–(20i) with β₁, β₂ and R-fold lower cluster | (20) |
| E11 | Fig. 1 loop structure: eigensolve → detect N → f_sk → inner loop → ρ += Δρ | Fig. 1 |
| E12 | Outer stop `‖Δρ‖ < ε` | Fig. 1 |
| E13 | Subproblem solver is MMA *or* an LP algorithm | after (20i) |
| E14 | Geometry a = 8, b = 1; E = 1e7, ν = 0.3, ρ_m = 1; plane stress; α = 50 %; uniform ρ = 0.5 start | §3.1, Fig. 2 |
| E15 | BCs: (a) SS both ends, (b) clamped–simply, (c) clamped–clamped | Fig. 2 |
| E16 | Targets ω₁⁰ = 68.7/104.1/146.1; ω₁^opt = 174.7/288.7/456.4, all bimodal | Figs. 2, 3 |
| E17 | ω₂^opt = 598.3/732.8/849.0, all bimodal (n = 2) | Fig. 6 |
| E18 | Gap example: CC beam, m_c = ½m_b at mid-point of lower edge, gap → 810 (+548 %), ω₃ω₄ω₅ trimodal | §3.3, Fig. 7 |

### [D] — imported from Du2007 (documented dependency, Olhoff2014 silent)

| # | Item | Value | Why needed |
|---|---|---|---|
| D1 | Stiffness exponent p | 3 | E6 leaves p free; p = 3 reproduces E16's initial values |
| D2 | Mass exponent q | 1 | Du2007 Eq. (2) "q = 1 normally" |
| D3 | Low-density mass law, C¹ variant `m = 6e5ρ⁶ − 5e6ρ⁷` for ρ ≤ 0.1 | `du2007_c1` | Du2007 Eq. (4b); suppresses spurious localized modes |
| D4 | Sigmund sensitivity filter `ŝ_e = Σ H_ei ρ_i s_i / (ρ_e Σ H_ei)` | r_min = 2.5 elements | Du2007 §3; Olhoff2014 mentions no filter |
| D5 | Filter applies to f_sk and to the J-mode gradient; **not** to the volume gradient; K and M are assembled from **raw** ρ | — | Du2007 states sensitivity filtering, not density filtering |

D3 is the one import that materially changes the physics. §7.3 runs it against
the Olhoff2014-literal `M_e = ρ^q M_e*` and reports both.

### [R] — reconstructed (unspecified everywhere; declared and calibrated)

| # | Item | Default | Calibration / arbiter |
|---|---|---|---|
| R1 | **Move limit** `\|Δρ_e\| ≤ m` added to (19f) | calibrated, expect 0.02–0.1 | Fig. 4 fingerprint (F5); Phase 5 |
| R2 | Mesh | 160×20 primary, 240×30 confirmation; nely even | F7; mesh-convergence check |
| R3 | Multiplicity tolerance, with hysteresis | join ≤0.5 %, leave >1.5 % on λ | Phase 3.2 sweep; FD audit R6 |
| R4 | Cluster reference model (see Phase 4.1) | C-A `λ̃ = λ_n` | FD audit R6 decides C-A vs C-C |
| R5 | Number of modes computed | `n_modes = n + N_max + 3`, ≥6 | J = n+N must always exist |
| R6 | Per-iteration finite-difference audit of predicted vs realised Δλ | always on | is itself the arbiter for R3/R4 |
| R7 | Inner stopping test | KKT residual of (19) < 1e-6, **reported** | Phase 4.4 |
| R8 | Outer stopping test | `‖Δρ‖_∞ < 1e-4` **and** full-problem KKT residual < tol | Phase 6 |
| R9 | Inner subproblem solver | `lp` exact (reference) / `mma` (paper-literal) — both implemented | cross-validated, Phase 4.4 |

### Explicitly excluded (present in `analysis/OlhoffApproach`, absent here)

Heaviside projection; β continuation; grayness penalty; density filtering;
additive `E_min`; additive `ρ_min` in the mass law; two-sided volume equality;
best-seen-design return; trial-eigensolve acceptance/backtracking; α-damping of
the outer update; MMA asymptote persistence across outer iterations (Fig. 1 poses
a *fresh* subproblem with Δρ re-zeroed each outer iteration — resetting MMA state
is the paper-consistent choice).

---

## 2. Module layout

```
analysis/OlhoffApproachExact/
  PLAN_Olhoff2014_exact.md      this document
  README.md                     supersedes OlhoffApproachExact.txt (§9.3)
  Matlab/
    fe_q4_exact.m               RESTORE verbatim from 6f2e04c^
    assemble_KM_exact.m         RESTORE verbatim
    mass_interp.m               RESTORE verbatim (+ 'olhoff2014_pow' mode, D3 ablation)
    build_filter.m              RESTORE verbatim
    apply_sensitivity_filter.m  RESTORE verbatim
    compute_elem_sensitivity.m  RESTORE verbatim
    build_supports_exact.m      RESTORE + even-nely assertion (F3)
    detect_multiplicity.m       REWRITE  — hysteresis, λ-based, R/downward cluster
    generalized_gradients.m     REWRITE  — cluster reference model R4, basis handling
    subproblem_lp.m             NEW — exact solver for (19)/(20), cutting-plane LP
    subproblem_mma.m            NEW — MMA solver for (19)/(20) (replaces inner_loop_mma.m)
    subproblem_kkt.m            NEW — KKT residual of (19)/(20)
    lumped_mass.m               NEW — non-structural concentrated mass (§3.3)
    topopt_freq_exact.m         REWRITE — Fig. 1 main loop, one step parameter
    run_ss_n1.m  run_cs_n1.m  run_cc_n1.m
    run_ss_n2.m  run_cs_n2.m  run_cc_n2.m
    run_cc_gap23.m
    verify/
      v_forward_model.m         Phase 2
      v_sensitivities_fd.m      Phase 2
      v_multiplicity.m          Phase 3
      v_basis_invariance.m      Phase 3
      v_subproblem_lp_vs_mma.m  Phase 4
      v_inner_kkt.m             Phase 4
      v_n1_reduction.m          Phase 4
  experiments/                  existing artifacts, untouched
```

Primary language **MATLAB**: `tools/Matlab/mmasub.m` + `subsolv.m` already exist,
all prior campaigns are MATLAB, and `linprog` is available (Optimization Toolbox
confirmed installed in R2025b) which the exact reference solver needs. A Python
port is Phase 10, after §3.1 reproduces — as an independent check, not in parallel.

---

## 3. Phases

### Phase 0 — Recover, re-baseline, commit

1. Restore all 24 files from `6f2e04c^` into `Matlab/` (working copy, then commit
   immediately as the recovery point, before any edit).
2. Record SHA-256 of every solver file in `Matlab/SHA256.txt`. Re-record before
   and after every campaign; a campaign that changes a solver file mid-run is void.
3. Regression: run `verify_initial_frequencies.m` at 160×20, even nely. Expect
   SS/CS/CC within 0.5 % of 68.7/104.1/146.1 at p = 3. If this fails, stop —
   something in the restore is wrong.
4. Add `analysis/OlhoffApproachExact/**/*.m` to version control explicitly and
   confirm `git status` is clean afterwards. **Commit at the end of every phase.**

Exit: recovery commit exists; SHA manifest written; initial frequencies reproduce.

### Phase 1 — Freeze the exactness contract

Write `README.md` §1 as the E/D/R table above, with the Olhoff2014 equation number
or the Du2007 citation next to every row. Nothing is implemented that does not
appear in that table; adding a row later is a documented amendment with a date.

Exit: table complete; every [R] row has a named arbiter.

### Phase 2 — Forward model and Fig. 2 verification

Deliverable `v_forward_model.m`, `v_sensitivities_fd.m`.

* Assert `mod(nely,2) == 0` in `build_supports_exact.m` for SS and CS; error
  otherwise with the 0.600 H explanation (F3).
* Mesh convergence of ω₁⁰ for the three BCs across 80×10, 160×20, 240×30, 320×40.
* Add `mass_interp` mode `olhoff2014_pow` = pure `ρ^q`, q configurable (E6/D2 literal).
* Finite-difference check of `dK_e/dρ_e`, `dM_e/dρ_e` for every mass mode, on both
  sides of ρ = 0.1.
* Finite-difference check of Eq. (4)/(5) simple sensitivity: central differences
  on λ_j for 20 random elements, relative error < 1e-6.
* Finite-difference check of the **off-diagonal** f_sk (s≠k) via the projected
  matrix: perturb ρ_e, recompute the cluster subspace, and compare
  `Φ_cᵀ(K−λ̃M)Φ_c` increments against `F_e`. This is the check the prior campaign
  never had.

Acceptance: ω₁⁰ within 0.5 % of 68.7/104.1/146.1 at ≥160×20 and p = 3; all FD
errors < 1e-5; all three initial designs unimodal (ω₂/ω₁ ≈ 2.5–3.7, per Fig. 2's
"all unimodal").

### Phase 3 — Bimodality: detection, generalized gradients, invariance

This is Fig. 1 steps 1–2 and Eqs. (11)–(13).

**3.1 Enough modes.** `n_modes = n_target + N_max + 3`, minimum 6 (old default was
4). J = n + N must exist or (19b) cannot be formed; the old code silently set
`lambda_J = Inf` and dropped the guard constraint.

**3.2 Detection with hysteresis.** Cluster on λ, relative: join the cluster when
`(λ_j − λ_n)/λ_n ≤ tol_join`, leave only when `> tol_leave`, with
`tol_join < tol_leave` (defaults 0.005 / 0.015 on λ, i.e. ≈0.25 % / 0.75 % on ω).
Rationale: natural coalescence in this problem is 0.3–1.3 % wide on ω, so the old
single `mult_tol = 1e-3` never fired N = 2; a single loose tolerance instead makes
N chatter between 1 and 2 across outer iterations, which re-poses a different
subproblem every step. Hysteresis fixes both. Log N, the cluster λ spread, and the
eigengap `(λ_{n+N} − λ_{n+N−1})/λ_n` every iteration.

For the gap problem (20), the same routine must also detect the **R-fold cluster
below** ω_{n−1}, scanning downward (Olhoff2014 footnote *1 on p. 281).

**3.3 Cluster reference λ̃ and the constraint model.** See Phase 4.1 — the choice
is made there and validated here by 3.4.

**3.4 Per-iteration finite-difference audit (R6).** After each accepted step, the
next outer eigensolve is computed anyway. Compare:

```
predicted  Δλ_j = eig( F(Δρ) )                   (or eig(diag(λ)+F) − λ, model C-C)
realised   Δλ_j = λ_j(ρ+Δρ) − λ_j(ρ)
```

Log `max_j |predicted − realised| / λ_n`. If the multiple-eigenvalue path is
wired wrong, this diverges on the first N ≥ 2 iteration. The memory record on the
previous campaign says plainly: *"N=2 implementation never verified against
derivatives"*. This closes that.

**3.5 Mode tracking.** Report MAC, but make **no decisions** on individual MAC —
it is fragile near multiplicity. Decisions use the subspace angle between
consecutive cluster subspaces (basis-invariant). Report **minimum** MAC and break
count, never median MAC (median falls with refinement while minimum rises).

**3.6 Basis invariance test** (`v_basis_invariance.m`) — the acid test:
replace `Φ_c ← Φ_c Q` for a random N×N orthogonal Q, recompute f_sk, re-solve the
subproblem. Δρ must be identical to 1e-10. If it is not, the cluster machinery is
wrong. (It should hold: F ↦ QᵀFQ leaves eig(F) unchanged, and the constraint
depends only on eig(F).)

Exit: FD audit error < 1 % on a case that reaches N = 2; basis invariance passes;
N = 1 path bit-identical to the general path when N = 1.

### Phase 4 — The increment subproblem, done exactly

**4.1 What (19b)–(19d) actually means.** F is symmetric (f_sk = f_ks) and *linear*
in Δρ. Requiring (19c) for all j = n…n+N−1 with (19d) defining Δ(ω²) as the
eigenvalues of F is therefore equivalent to

```
β  ≤  λ̃ + μ_min( F(Δρ) )        ⟺        λ̃·I_N + Σ_e Δρ_e F_e  ⪰  β·I_N
```

with `(F_e)_{sk} = f_sk[e]`. **Subproblem (19) is a semidefinite program**: linear
objective, one N×N linear matrix inequality, one linear volume inequality, box
bounds. It is convex. For N = 1 the LMI collapses to a scalar linear constraint
and (19) is an LP — which is exactly what §2.5 of the paper says.

Stating it this way settles the questions that stalled every previous attempt:
which eigenvalue branch to constrain (all of them ⟺ the smallest ⟺ the LMI); why
individual μ_i are non-differentiable but the feasible set is still convex; and
what "bimodal optimum" means as a KKT condition.

Three cluster-reference models, all identical at exact multiplicity:

| | constraint | convex | note |
|---|---|---|---|
| **C-A** (default, paper) | `λ̃·I + ΣΔρ_e F_e ⪰ β·I`, λ̃ = λ_n | yes | (19c) read with ω_j² = λ̃ |
| C-B (literal (19c), sorted pairing) | `β ≤ min_j (λ_j + μ_j(F))` | no | ω_j² read individually |
| C-C (degenerate perturbation) | `diag(λ_n…λ_{n+N−1}) + ΣΔρ_e F_e ⪰ β·I` | yes | correct for *near*-multiple clusters |

Default C-A. C-C is the declared near-degeneracy variant and is the likelier fix
if bimodality forms but will not hold; the Phase 3.4 FD audit decides between them
on evidence, not preference. The old code used λ̃ = mean(cluster λ) with no
justification and no test — that is C-A with a different λ̃; A/B it.

**4.2 Exact reference solver — cutting-plane LP (`subproblem_lp.m`).**

Since `μ_min(F) = min_{‖q‖=1} qᵀFq`, the LMI is equivalent to infinitely many
linear constraints `β ≤ λ̃ + Σ_e Δρ_e (qᵀF_e q)`. Algorithm:

```
Q ← eigenvectors of F(0) (= {e_1..e_N})
repeat
    solve LP over (β, Δρ):
        max β
        s.t.  β − λ̃ − Σ_e Δρ_e (qᵀF_e q) ≤ 0            for each q ∈ Q      (19c/19d)
              β − λ_J − f_JJᵀΔρ ≤ 0                                          (19b)
              Σ_e (ρ_e + Δρ_e)V_e − V* ≤ 0                                   (19e)
              max(ρ−ρ_e, −m) ≤ Δρ_e ≤ min(1−ρ_e, +m)                         (19f)+R1
    build F(Δρ*), take μ_min and its eigenvector q*
    if β* − λ̃ − μ_min ≤ tol_cut: stop
    Q ← Q ∪ {q*}
```

The cut coefficient `Σ_{s,k} q_s q_k f_sk` is the same `fsk2D * kron(q,q)`
already used as the μ_i gradient — one shared routine. This terminates in a few
cuts for N ≤ 3, needs only `linprog`, and returns the **exact** optimum of (19).
It is the ground truth against which the MMA path is measured. For N = 1 it is a
single LP with no cuts.

**4.3 MMA solver (`subproblem_mma.m`), paper-literal option E13.**

Keep the N smooth constraints `β − λ̃ − μ_i(F) ≤ 0` with
`∂μ_i/∂Δρ_e = q_iᵀF_e q_i`, and fix what the old `inner_loop_mma.m` got wrong:

* Symmetrize before decomposing: `F = (F+F')/2; [Q,mu] = eig(F,'vector')` —
  guarantees real μ and orthonormal Q. The old code used general `eig` and took
  `real(...)`.
* Degenerate μ_i: when `|μ_i − μ_j| < tol`, individual gradients are meaningless.
  Fall back to the single cut on μ_min.
* **β bound.** `beta_max_hat = 1e6` is *not* inert — it drives a β asymptote span
  of 9e3 and a P/Q dynamic range of 5.7e13 inside `mmasub`. Set
  `β̂ ∈ [0, 1 + m·Σ_e|f_nn[e]|/λ̃]`, or use the C-A LP relaxation value as the cap.
* Optional `eliminate_beta` mode: at the optimum β = λ̃ + min(μ_min, λ_J−λ̃+f_JJᵀΔρ),
  so the subproblem is a concave maximization in Δρ alone — removes the
  ill-conditioned mixed-scale variable entirely.
* **Real stopping test** (`subproblem_kkt.m`): KKT residual of (19), not iterate
  change. Every outer iteration logs `inner_iters`, `kkt_residual`, and
  `stop_reason ∈ {kkt_met, max_iter, stalled}`. The old code's
  `norm(Δρ_new − Δρ_old) < inner_tol·√nEl` was never met in 300 outer iterations
  and the truncation was silently doing the step control.

**4.4 Inner-loop validation** (this is "proper function of the internal loop"):

| id | test | criterion |
|---|---|---|
| V-I1 | N = 1, random f and ρ: MMA vs `linprog` | objective within 1e-6 rel.; cos(Δρ) > 0.999 (prior measurement: 0.02 %, 0.9976 — regression) |
| V-I2 | N = 2,3, random symmetric F_e: cutting-plane vs bisection-on-β feasibility | β within 1e-8 |
| V-I3 | declared-vs-achieved | `stop_reason` logged every outer iteration; a run whose inner loop never meets its test is reported as truncated, not as converged |
| V-I4 | N = 1 reduction | general N path with N = 1 bit-identical to the N = 1 path (E14/E15) |
| V-I5 | basis invariance | Phase 3.6 |
| V-I6 | unrestricted-step reproduction | with `m = Inf`, N = 1: reproduce the known LP vertex collapse (CC ω₁ 145.57 → ≈0.06, ≥99.9 % of variables at a bound). **This is a required PASS**: it proves the subproblem is solved exactly and that F4 is real, not an artifact |

Exit: all six pass, both solvers agree.

### Phase 5 — Step restriction, calibrated against Fig. 4

One parameter, `m` (R1), entering (19f). No α-damping, no acceptance test, no
backtracking, no best-seen return.

1. Sweep `m ∈ {0.01, 0.02, 0.05, 0.1, 0.2, Inf}` on SS 160×20, 80 outer iterations,
   `solver = 'lp'` (exact subproblem, so m is the only variable).
2. Score each against the F5 fingerprint:
   * ω₁ monotone non-decreasing (hard requirement),
   * ω₂ peak 327 ± 10 % at iteration 7 ± 3,
   * ω₃ peak 528 ± 10 % at iteration 9 ± 3,
   * coalescence by iteration 20 ± 5,
   * ω₁ within 3 % of 174.7 by iteration 80,
   * ω₃ final within 5 % of 284.9.
3. Choose the **largest** m satisfying the hard requirement and scoring best on
   the rest. Expected 0.02–0.1 from the trajectory shape; report the actual sweep.
4. Repeat the sweep with `solver = 'mma'` and report whether the chosen m
   transfers. If MMA needs a different m, that is a property of MMA, not of the
   paper — report it as such.
5. If the result is strongly m-dependent, **report the sweep, not a single m.**

Deliverable: `experiments/step_calibration/REPORT.md` with the Fig. 4 overlay plot.

Exit: an m is selected with a stated criterion, or the sweep is reported as
inconclusive with the evidence.

### Phase 6 — Main loop and convergence semantics

Fig. 1 steps 0–4 exactly:

```
0.  ρ ← 0.5 uniform;  choose n
1.  assemble K(ρ), M(ρ) from RAW ρ;  eigensolve (1b)(1c), M-orthonormalize
    detect N of ω_n  (and R of ω_{n−1} for problem (20))
2.  f_sk (Eq. 13) if N > 1, else the usual gradient (Eq. 4/5)
    sensitivity-filter f_sk and f_JJ  [D4/D5]
3.  inner loop: solve (19) (or (20)) for Δρ  ← subproblem_lp / subproblem_mma
4.  ρ := ρ + Δρ                                    (no damping: α = 1)
    stop when ‖Δρ‖_∞ < ε AND full-problem KKT residual < tol
```

* **Stop reason is always recorded.** `{kkt_converged, increment_small, max_iter,
  eigensolver_failure}`. A run that hits `max_iter` is never reported as a
  converged optimum. Every previous campaign's headline frequencies (327.14,
  369.43, 300.90, 312.28, 328.55, 371.54) are samples from non-converged
  oscillations 4–5 orders above tolerance — do not repeat that.
* **Full-problem KKT residual**: at (ρ*, β*), take the multipliers from the last
  subproblem and evaluate stationarity for (9a)–(9c). This is the definition of
  "converged" used in the report.
* Return the **final** ρ (paper semantics, E12). Best-seen ρ may be logged as a
  diagnostic and must be labelled as such.
* History logged per outer iteration: ω₁…ω_{n_modes}, β, volume, N (and R), cluster
  spread, eigengap, inner iterations, inner KKT residual, inner stop reason,
  ‖Δρ‖₂ and ‖Δρ‖_∞, FD audit error (3.4), min MAC, subspace angle, number of
  8-connected structural components.

Exit: a CC 160×20 run reaches `kkt_converged` or the failure is characterized
with the above telemetry.

### Phase 7 — Reproduce §3.1 (n = 1, three BCs)

Configuration: p = 3 fixed (no continuation); `du2007_c1` [D3]; sensitivity filter
r_min = 2.5 [D4]; 160×20 primary and 240×30 confirmation; nely even; α = 0.5
volume; uniform ρ = 0.5 start; m from Phase 5; `solver = 'lp'` primary,
`solver = 'mma'` as the E13 alternative.

**7.1 Decision rule, declared before the runs** (this is the piece the A4 campaign
was missing — no verdict is possible without it):

| verdict | condition |
|---|---|
| **PASS** | for all three BCs, on both meshes: \|ω₁ − target\|/target ≤ 3 %; N = 2 sustained over the final ≥20 iterations with (ω₂−ω₁)/ω₁ < 0.5 %; single 8-connected structural component spanning support to support; `stop_reason = kkt_converged` |
| **PARTIAL** | any strict subset of the above, itemized per BC and per mesh |
| **FAIL** | ω₁ collapse (>50 % drop from the initial value at any accepted iterate), or no BC converges |

Whichever occurs is what gets reported.

**7.2 Secondary targets:** the F5 Fig. 4 fingerprint for SS; ω₃^opt ≈ 284.9 for SS;
the Fig. 3 topology class (connected, X-braced, solid end blocks for CS/CC).

**7.3 Required ablations** (one factor at a time, everything else at the Phase 7
configuration):

| id | factor | arms |
|---|---|---|
| A1 | mass model | `du2007_c1` [D3] vs `olhoff2014_pow` q = 1 [E6 literal] |
| A2 | filter | sensitivity filter r_min = 2.5 [D4] vs no filter [Olhoff2014 literal] |
| A3 | cluster model | C-A vs C-C (Phase 4.1) |
| A4 | subproblem solver | `lp` vs `mma` |

A1 and A2 measure exactly how much of any success is owed to the Du2007 imports
rather than to Olhoff2014. That number belongs in the report.

### Phase 8 — §3.2 (n = 2) and §3.3 (gap, Eq. 20)

**8.1 n = 2** (`run_*_n2.m`). Targets 598.3 / 732.8 / 849.0, all bimodal (E17).
This is the first case where (19b)'s J = n + N guard genuinely binds and where the
"exchange order with ω_n" language after (9c) matters. `n_modes ≥ 7`.

**8.2 Gap problem** (`run_cc_gap23.m`), Eq. (20). Requires:
* β₁ and β₂ as two bound variables, objective `max(β₂ − β₁)`;
* constraints (20b)–(20i), including the R-fold **lower** cluster (20d)/(20g) and
  its own sub-eigenvalue problem — the downward detection from Phase 3.2;
* a design-independent lumped mass `m_c = ½ m_b` at the mid-point of the lower
  edge (`lumped_mass.m`; ∂M_c/∂ρ = 0, so it contributes to M but not to f_sk);
* upper bound is the **max** of the lower cluster and lower bound the **min** of
  the upper cluster ⇒ two LMIs of opposite sense; the cutting-plane solver
  generalizes directly (cuts on μ_min of the upper block and μ_max of the lower).

Targets: gap → 810, +548 % over the initial design (⇒ initial gap ≈ 125);
ω₃ = ω₄ = ω₅ **trimodal** at the optimum. N = 3 is the real stress test of the
N×N machinery — Phase 3.6 and 4.4 must pass at N = 3 before this is attempted.

**8.3 Out of scope:** §3.4–3.6 3D plates and bi-material. Record as future work.

### Phase 9 — Report and provenance

1. `REPORT.md`: the §1 contract table as executed; the Phase 5 m calibration with
   the Fig. 4 overlay; the Phase 7 verdict against the declared rule; the A1–A4
   ablation table; every [R] value used.
2. Frequency table with paper value, computed value, relative error, N, mesh, and
   `stop_reason` per cell. No cell reports a number without its stop reason.
3. SHA-256 manifest before/after each campaign.
4. Language for any comparison the paper reproduction feeds into: state the
   contract, i.e. "Olhoff2014 (9), (12), (13), (19) and Fig. 1 as specified, with
   the Du2007 interpolation and filter imports D1–D5 and the reconstructed move
   limit R1 = <value>". Never "paper-exact" unqualified.

### Phase 10 — Python port (optional, after Phase 7 passes)

Port to `Python/` as an independent implementation check, not in parallel with
MATLAB development. `scipy.optimize.linprog` (HiGHS) for the exact solver; a
vendored MMA for the E13 path (note `analysis/OlhoffApproach/Python/mma.py` is
currently missing from the repo and would need restoring from `c2f4ba0` first).
Cross-check: identical ρ trajectory to 1e-8 for the first 10 outer iterations on
CC 80×10.

---

## 4. Risks and kill criteria

| id | risk | mitigation / kill criterion |
|---|---|---|
| K1 | With a calibrated m, ω₁ still plateaus below target and N stays 1 | Run the A1–A4 ablations. If none moves it, report a negative result against the Phase 7.1 rule and **stop**. Do not keep tuning — that is what produced 19 runs and no verdict last time. |
| K2 | The result is strongly m-dependent | Report the sweep, not a point value. m is a declared [R] parameter; its sensitivity is a finding, not a defect to hide. |
| K3 | `du2007_c1` makes void massless ⇒ disconnection is rewarded | ≥160×20 (F7) plus ablation A1. If A1 shows the connected optimum only exists under D3, say so explicitly. |
| K4 | Near-degenerate cluster ⇒ ill-conditioned eigenvectors ⇒ garbage f_sk | Phase 3.4 FD audit catches it in one iteration; Phase 3.6 basis invariance proves the formulation is basis-free. |
| K5 | N chattering between outer iterations | Hysteresis (R3); logged. If chatter persists, the FD audit will show which N is right. |
| K6 | Scope creep into 3D plates / bi-material | Out of scope, stated in §8.3. |
| K7 | Solver lost again | Commit at the end of every phase; SHA manifest; §0/F1. |

## 5. Answering the two specific asks

**Bimodality.** It is not primarily a detection-tolerance problem. The three
things that were actually missing are: (i) the recognition that (19b)–(19d) is the
LMI `λ̃I + ΣΔρ_e F_e ⪰ βI`, which makes the multiple-eigenvalue constraint convex
and basis-free and removes all the branch-selection ambiguity (Phase 4.1);
(ii) a per-iteration finite-difference audit that proves the f_sk / sub-eigenvalue
path is right instead of assuming it (Phase 3.4); (iii) hysteresis so N stops
flip-flopping and the same subproblem is posed twice in a row (Phase 3.2). The
tolerance value matters, but it was never the binding issue.

**Internal loop.** Two solvers, one exact and one paper-literal, cross-validated
(Phase 4.2–4.4). The inner loop reports a real KKT residual and a stop reason
every outer iteration, so truncation can never again act as undeclared step
control (F4). And the step control that Fig. 1 omits is added as exactly one
declared parameter, calibrated against Fig. 4's iteration history rather than
against the endpoint (Phase 5, F5/F6).

## 6. Sequencing

Phases are strictly ordered; each ends with a commit and its exit criterion.
Phases 0–2 are recovery and verification of ground already covered. Phase 3–5 is
the new work and the only part where the outcome is genuinely uncertain. Phase 6–7
is the reproduction attempt under a rule declared in advance. Phase 8 extends to
the two harder examples only if Phase 7 is PASS or a well-characterized PARTIAL.
