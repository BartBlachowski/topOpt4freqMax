# Phase 3 — Paper-derived spec vs. `analysis/OlhoffApproachExact` (MATLAB)

Clean-room comparison of the **paper-derived specification** ([du_olhoff_2007_spec.md](../specification/du_olhoff_2007_spec.md), built without reading code) against the existing **`OlhoffApproachExact`** MATLAB implementation. Read-only inspection; no files in `OlhoffApproachExact/` were modified.

**Scope note:** the pre-existing `OlhoffApproachExact/OlhoffApproachExact.txt` compares the paper against the *older* `analysis/OlhoffApproach` solver — a different artifact. This document instead compares the paper against `OlhoffApproachExact` *itself* (the "exact" rewrite), which is the relevant question for Phase 4.

Files inspected (all under `analysis/OlhoffApproachExact/Matlab/`):
`topopt_freq_exact.m`, `inner_loop_mma.m`, `compute_generalized_gradients.m`, `compute_elem_sensitivity.m`, `detect_multiplicity.m`, `mass_interp.m`, `assemble_KM_exact.m`, `fe_q4_exact.m`, `build_filter.m`, `apply_sensitivity_filter.m`, `build_supports_exact.m`, `run_clamped_clamped_exact.m`, plus audit artifacts in `Matlab/results/`.

---

## 0. Verdict up front

The implementation is an **algorithmically faithful** reconstruction of the paper — including the hardest part (the multiple-eigenvalue subeigenvalue problem, which my spec flagged as the single biggest reproduction ambiguity, §11.9). Every numbered equation I derived has a correct counterpart in the code.

**Yet it does not reproduce the benchmark** (CC target `ω₁ = 456.4`, bimodal). The audit data show it limit-cycles and terminates at `ω₁ ≈ 414`, multiplicity `N = 1` (simple), never locking onto the bimodal optimum. The mismatch is therefore **not in the equations** — it is in the **`[N]`-classified numerical choices** the paper left unspecified, exactly as Phase 2 predicted. The two dominant causes are (a) **no step damping / trust region on the full-step increment update**, and (b) **no penalization continuation** — compounded by (c) the **disconnection tendency** the formulation permits.

---

## 1. Exact matches (implementation correctly realizes the spec)

| # | Spec item | Code location | Match |
|---|---|---|---|
| 1 | Q4 bilinear plane-stress, 2×2 Gauss, consistent mass (spec §12 [I]) | `fe_q4_exact.m` | ✓ exact; DOF order LL,LR,UR,UL matches spec memory note |
| 2 | Stiffness `K_e = ρ_e^p K*_e`, **no** additive E_min (eq. 1) | `assemble_KM_exact.m:32`, `:35` | ✓ exact (`Ee = rho.^penal`) |
| 3 | Mass models (eq. 2/4/4a/4b) incl. coefficients `c0=1e5, c1=6e5, c2=−5e6, r=6, thr=0.1` | `mass_interp.m` | ✓ exact; all four modes present; C0/C1 continuity checks correct |
| 4 | Global assembly `K=Σρ^p K*_e`, `M=Σ m(ρ) M*_e` (eq. 3) | `assemble_KM_exact.m:35-38` | ✓ exact |
| 5 | M-orthonormalization `φᵀMφ=δ` (eq. 7c) | `topopt_freq_exact.m:170-174` | ✓ exact |
| 6 | Generalized gradients `f_sk = φ_sᵀ(K'_e − λ̄ M'_e)φ_k` (eq. 19) | `compute_generalized_gradients.m:80` | ✓ exact; uses `λ̄ = mean(cluster λ)` |
| 7 | Subeigenvalue problem (eq. 18): `F(s,k)=Σ_e f_sk(e)Δρ_e`, eigenvalues `μ_i = Δλ_i`, gradient `dμ_i/dΔρ_e = q_iᵀ F_e q_i` | `inner_loop_mma.m:158-188` | ✓ exact — faithful realization of the directional-derivative theory (spec §6.2). Resolves spec §11.9. |
| 8 | Cluster constraints `β − (λ̄ + μ_i) ≤ 0` (eq. 25c) | `inner_loop_mma.m:182-188` | ✓ exact |
| 9 | J-mode cap `β − (λ_J + ∇λ_Jᵀ Δρ) ≤ 0`, `J=n+N` (eq. 25b) | `inner_loop_mma.m:194-200`, `detect_multiplicity.m:56` | ✓ exact |
| 10 | Volume `Σ(ρ+Δρ)/nEl − volfrac ≤ 0`, one-sided (eq. 25e) | `inner_loop_mma.m:204-206` | ✓ exact (single inequality, not two-sided) |
| 11 | Box `ρ̲−ρ ≤ Δρ ≤ 1−ρ` (eq. 25f) | `inner_loop_mma.m:118-122` | ✓ exact (when `outer_move=Inf`) |
| 12 | Sensitivity filter (Sigmund) on **objective sensitivities only**, not on densities, not on volume sensitivity (spec §1.4) | `apply_sensitivity_filter.m`, `topopt_freq_exact.m:189-214` | ✓ matches paper's literal statement |
| 13 | Nested outer/inner loop; update `ρ:=ρ+Δρ`; stop `‖Δρ‖<ε` (Fig. 1) | `topopt_freq_exact.m:144-291` | ✓ structure exact |
| 14 | `ρ̲ = 10⁻³` (eq. 7e) | `topopt_freq_exact.m:336` | ✓ exact |
| 15 | Multiplicity by relative freq. tolerance (spec §7.2) | `detect_multiplicity.m:46` | ✓ matches rule |
| 16 | N=1 reduction to simple eigenvalue (eqs. 20-21) | `inner_loop_mma.m:35-39`, `compute_elem_sensitivity.m` | ✓ exact; `compute_generalized_gradients` reduces to `compute_elem_sensitivity` for N=1 |

**This is a genuinely careful "exact" implementation.** It even chose the paper's *default* inner solver (MMA on the nonlinear det-coupled subproblem), not the easier LP reduction (eq. 22) — i.e. the harder, more faithful route.

---

## 2. Ambiguities — paper `[N]` items resolved by an implementation choice

These are not errors; they are the implementation filling gaps the paper left open. They are listed because **they are exactly where reproduction is sensitive** (Phase 2 §5).

| Spec `[N]` item | Implementation choice | Comment / risk |
|---|---|---|
| Mesh resolution (§11.1) | `40 × 5` for the 8×1 beam (`run_*` + defaults `topopt_freq_exact.m:329-330`) | Plausible; **must be checked against initial-freq targets** 68.7/104.1/146.1 (memory says FE was verified vs Fig. 2 — so mesh/units are OK). |
| Filter radius (§11.2) | `rmin_elem = 2.5` elements (`:339`) | **Highly leveraged.** Audit `rmin` sweep (1.5/2.0/2.5/3.0) gives wildly different finals (289/253/414/217) — none reach 456.4 bimodal. Confirms filter radius is a dominant `[N]` lever. |
| Which mass model (§11.3) | `du2007_c1` (eq. 4b) default (`:338`) | Audit `mass_mode` sweep: linear/step/c0/c1 give 240/239/271/414 — strong dependence; paper claims "negligible differences," which the audit **contradicts** for the optimizer (though plausibly true for a *converged* design). |
| Multiplicity tolerance (§11.6) | `mult_tol = 1e-3` relative (`:345`) | Audit sweep 1e-4…5e-2 changes whether `N=2` is ever detected; at `1e-2`/`5e-2` some `best` snapshots reach `N=2`. The tolerance gates the entire multiple-eigenvalue machinery. |
| Outer tol `ε` (§11.6) | `‖Δρ‖/√nEl < 1e-3` (default) / `1e-4` (run script) | Reasonable. |
| Inner tol / max-iter | `1e-4`, `30` (`:348-349`) | Reasonable. |
| Modes computed | `n_modes = n+3` (`:344`) | ≥ `n+N` for `N≤3`; fine for these examples. |
| MMA params | `a0=1, a=0, c=1e3, d=1` (`inner_loop_mma.m:135-138`) | `d=1` chosen for conditioning (documented). Acceptable, paper `[N]`. |
| Bound-variable scaling | `β̂ = β/λ̄` (`inner_loop_mma.m:96`, `:112`) | Mathematically equivalent to using `β=ω²` directly; flagged-OK. |
| Eigensolver | `eigs(Kf,Mf,n_modes,'SM')` shift-invert-ish, with fallback (`topopt_freq_exact.m:152-165`) | Paper `[N]`; acceptable. Note: `'SM'` can be fragile; a `sigma` shift might be more robust (cf. project memory's `eigsh sigma=1e-6` note for the Python side). |

---

## 3. Deviations from the paper (genuine differences, not just `[N]` choices)

| # | Paper says | Implementation does | Severity |
|---|---|---|---|
| **D1** | `p` is "normally assigned values **increasing from 1 to 3** during the optimization" (eq. 1 text) — continuation existence is **[E]**, schedule is [N] | `topopt_freq_exact.m` uses **fixed `penal=3`** throughout the outer loop (`:337`, read once, never advanced). Continuation lives only in a *separate experiment* (`results/penalty_continuation/`, `*_persist_mma_experiment.m`). | **MEDIUM** (downgraded after Phase-4 verification). Fixed `p=3` is a departure from the paper's *stated* method, but the independent FE check ([experiments/initial_frequency_verification.md](../experiments/initial_frequency_verification.md)) shows the paper's **published initial frequencies are themselves the `p=3` penalized values** (p=1 would give ~2× higher). So `p=3` reproduces the reported numbers exactly; continuation may aid path *stability* but is **not** required to match the benchmark frequencies, and is not the primary cause of divergence. |
| **D2** | Inner sub-problem is a *local linearization* (eqs. 23-25) valid only for small `Δρ`; the paper's "MMA" supplies an implicit trust region via moving asymptotes | Defaults disable **all** extra damping: `move_lim=Inf`, `outer_move=Inf`, `alpha=1.0` (full step), `acceptance_check=false` (`:353-357`). The only bound on `Δρ` is the full box `[ρ̲−ρ, 1−ρ]`. | **HIGH (root cause).** "Paper-exact" was interpreted as "no trust region," but the paper's increment formulation is only locally valid. With the full box range and a *linear* (N=1) constraint set, the inner MMA walks `Δρ` toward bang-bang corners ⇒ huge steps ⇒ `ω` oscillation (see §4). |
| **D3** | Reported design is the **final converged** `ρ` (Fig. 1 stop) | Returns final `ρ` (correct, `:306`) — but the audit also tracks a **`best_iter` / best-seen** snapshot, and the *final* never matches the paper while a *transient* best (iter 24, `ω₁=462`, **N=1**) briefly exceeds it. | LOW (semantics correct) but **diagnostic**: the run does not converge, so "final" is a limit-cycle point, not an optimum. |
| **D4** | Single global volume **inequality** `ΣρV − V* ≤ 0` (eq. 7d) | One-sided inequality ✓ (no deviation) | none |
| **D5** | No connectivity/symmetry constraint (spec §4) | none added ✓ (faithful — but see §4/Phase 2) | none (faithful), but consequential |

> Note on D2 nuance: `inner_loop_mma.m:104` sets a *fallback* `outer_move=0.2` only if the arg is empty, but `topopt_freq_exact.m:217-219` always passes `outer_move` (default `Inf`), so the 0.2 fallback never fires in the default path. The box bounds are therefore the full density range.

---

## 4. Possible causes of the benchmark mismatch (independent analysis)

Audit evidence (`results/optimizer_audit/baseline_diagnostics.txt`, `sweep_summary.csv`, `penalty_continuation/continuation_trajectory.csv`):
- CC base run ends iter 80 at `ω₁=413.9, ω₂=450.1, N=1`, volume active. Target is `456.4, N=2`.
- A *transient* best at iter 24 reached `ω₁=462.3` but **`N=1`** (simple) — i.e. it overshot a non-bimodal design and then fell away.
- `ω₁` history oscillates violently (continuation CSV: 330 → 536 → 948 → 516 → 694 → 1050 → 1241 → 377 …) — a **limit cycle**, not convergence.
- Every parameter sweep (mass mode, mult_tol, rmin, n_modes) fails to land the connected bimodal `456.4`.

Ranked root-cause hypotheses (independent of the existing notes, though consistent with the project memory "reproduction gap is optimizer-side; paper-literal MMA diverges; N=1 LP bang-bang"):

### C1 — Full-step increment with no trust region ⇒ bang-bang LP limit cycle *(primary)*
When `N=1`, the cluster constraint `β − (λ̄ + f_nnᵀΔρ) ≤ 0` is **linear in `Δρ`**, the volume constraint is linear, and the objective (maximize `β`) is linear. The inner sub-problem is then effectively a **linear program over a box**, whose solution sits at a **vertex** (bang-bang `Δρ`). MMA's asymptotes convexify only mildly; with the full box range available and no `move_lim`, the accumulated inner `Δρ` is large. The frozen-gradient linearization (eqs. 23-25) is valid only locally, so a large step lands somewhere the prediction did not anticipate ⇒ the next outer eigensolve sees a very different `ω` ⇒ oscillation. **This is the dominant failure mode and it is a direct consequence of deviation D2.** It manifests precisely while `N=1`, which is most of the run.

### C2 — The path never locks onto the bimodal manifold *(mechanism of non-convergence)*
The paper's stabilizer for the step is the multiple-eigenvalue machinery: when `ω_n` becomes `N`-fold, the constraints (25c) become genuinely **nonlinear** (the `μ_i(Δρ)` are eigenvalues of `F(Δρ)`), which curbs the step and pins the design to the coalescence surface. But to *reach* that surface you must approach it smoothly. The bang-bang steps of C1 jump across it instead of settling onto it, so the design stays in the `N=1` regime where there is no such stabilization — a self-reinforcing failure. The paper's smooth `p`-continuation (absent, D1) and an implicit/explicit trust region (absent, D2) are exactly what would allow the gentle approach.

### C3 — Disconnection / wrong basin *(compounding, formulation-level)*
Per Phase 2 (`notes/disconnection_analysis.md` §3.2), the CC case admits a *feasible and possibly preferred* **disconnected two-block** design (each block clamped to one wall, gap in the middle) that yields a bimodal pair *different* from the paper's connected Fig. 3c. The project memory records that this disconnected topology is in fact preferred even at `p=1`. If the optimizer is drawn to that basin, it will never match the connected `456.4` design no matter how the optimizer is stabilized — the targets live in different basins. The localized-mode mass fix (eq. 4b) does **not** prevent this, because near-disconnection passes through *intermediate* density (`ρ>0.1`), outside the void regime the fix addresses (spec §1.2 / Phase 2 §2.3).

### C4 — Mass-model / filter sensitivity contradicts the paper's "negligible differences" claim *(secondary)*
The audit shows the *optimizer trajectory* depends strongly on mass model and `rmin` (§2). The paper asserts the three mass models differ negligibly — true perhaps for a *converged* 0–1 design, but **not** for the transient optimizer path, which is what determines which basin/limit-cycle is reached. So a paper-literal reading ("any mass model is fine") is misleading for reproduction.

### C5 — Eigensolver robustness *(minor)*
`eigs(...,'SM')` without an explicit shift can mis-order or miss closely-spaced modes near coalescence — precisely the regime that matters here. Unlikely to be primary (fallback path exists), but worth ruling out with a `sigma`-shift solve.

---

## 5. What this implies for Phase 4

The equations are right; the **path control is wrong/missing**. A new implementation is **not** needed to fix the *physics or sensitivities* — those are correct and reusable (matches in §1). What is needed is to supply the paper's missing-but-necessary **path-determining choices** and to test the **basin** question. Concretely, the leverage points (in order) are:

1. **Add penalization continuation `p:1→3`** (restore D1 to the default path), starting near-convex.
2. **Add a trust region / step damping** — either an explicit `move_lim`/`outer_move` per outer iteration, or `alpha<1` damping, or rely on a *tighter MMA asymptote init* — to make the increment linearization locally valid (fix D2). The implementation already exposes all three knobs; they are merely defaulted off.
3. **Diagnose the basin**: with stabilized steps, check whether the result is the connected Fig. 3c (`456.4`, bimodal) or the disconnected two-block design. If disconnected, that is a *formulation* outcome (Phase 2), and matching the paper would require either (a) a different/luckier path, or (b) an explicit connectivity/symmetry handling that is *not in the paper* (and must be flagged as a deviation if added).
4. Re-examine whether the paper's bimodal coalescence is reachable at all for this mesh/filter with paper-faithful settings, or whether the published figure used an unreported stabilization (very likely — the paper omits move limits, continuation schedule, MMA params, and multiplicity tolerance, all `[N]`).

**Bottom line for Phase 4:** the decision is *not* "reimplement from scratch." It is "the existing `OlhoffApproachExact` is a correct paper transcription whose default `[N]`-settings make it diverge; a reconstruction should keep its FE/sensitivity/subeigenproblem core (confirmed by the spec) and differ only by supplying the path-control the paper omitted, while explicitly testing the connected-vs-disconnected basin question." Any reconstruction in `implementation/` should reuse the §1-matched components (they are paper-confirmed) and treat continuation + trust-region + basin diagnosis as the new contribution.

Any faithful reconstruction must explicitly declare whether it implements the nonlinear MMA subproblem or the optional LP reduction. The paper states that MMA was used, but does not operationalize how the nonlinear subeigenvalue constraints were supplied to MMA. The LP reduction is mathematically explicit, but it is presented as an additional simplification rather than the stated default.