# BOTTOM LINE

**The algorithm is not minimizing anything, and it is not solving the problem it
poses itself either.**

Two independent findings, both measured at the frozen 480×60 endpoint with zero
optimization runs and zero density updates.

**1. The sensitivity-filtered field has a curl.** Its Jacobian antisymmetry is
0.287 and does not move across a 33× range of finite-difference step; its
closed-loop integrals scale as a^1.999 and reverse sign to 3.9e−13; the physical
gradient passes the identical tests at 2.6e−06 and behaves as pure FD noise. By
the converse of the Poincaré lemma **no scalar function has this field as its
gradient** near this state. The mechanism is exact, not inferred: the
decomposition `J_filt = A·D_{g/ρ} + A·Hess − diag(g_filt/ρ)` is verified to
1e−08, and **99 %** of the obstruction is the `A·Hess` non-commutation term, not
the ρ-weighting — so symmetrizing the operator would not repair it.

**2. The final inner MMA subproblem never reaches its own KKT point.** Not at
production's 19 sub-iterates, not at 500, not at 5000. The relative step stalls
and oscillates at ~2e−03, the KKT residual *degrades* (0.360 → 0.586 → 0.597),
and the increment the subproblem actually wants is **15.7× larger** than the one
production returns — **97.9 %** of the move limit against 6.2 %.

The second finding is logically prior and it decides the next action. The
endpoint is currently set by **where the inner loop is truncated**, so the
filter's non-conservativity cannot yet be assigned responsibility for it. That
is the task's Case B.

What this does explain is the paradox the previous audit left open: a design can
sit still, with a flat terminal window, carrying a large physical KKT residual,
because "convergence" here means a fixed point of a heuristic update field —
not a stationary point of any optimization problem. What it does **not** explain
is why grayness grows with mesh; this audit examined one mesh and performed no
400 or 800 evaluation.

## Final verdicts

```
FROZEN_480_STATE_IDENTITY_PASS
FINAL_INNER_MMA_KKT_FAIL
FILTERED_FIELD_LOCALLY_NONCONSERVATIVE
EFFECTIVE_SCALAR_OBJECTIVE_NOT_IDENTIFIED
SURROGATE_PHYSICAL_KKT_MISMATCH
SURROGATE_MISMATCH_PARTIALLY_EXPLAINS_NONSTATIONARY_GRAYNESS
INNER_MMA_CERTIFICATION_FAILURE_REQUIRES_RESOLUTION
PERFORMANCE_CAMPAIGN_STILL_BLOCKED
```

## The evidence in one table

| test | filtered field | physical control |
|---|---|---|
| median Jacobian asymmetry, δ = 1e−3 … 3e−5 | **0.28675, 0.28675, 0.28681, 0.28701** | 2.6e−06 → 1.4e−04 (∝1/δ) |
| median normalized closed-loop integral | **0.272 … 0.285** | 2.9e−06 → 4.7e−05 |
| loop amplitude exponent | **1.999** | 0.978 |
| Frobenius skew ratio, mixed partials | 5.593e−04 | 2.073e−06 |
| orientation reversal, relative sum | 3.9e−13 | — |
| multiplicity exclusions of 88 samples | **0** (min gap12 = 0.12979) | — |

---

## Direct answers

**1. Is the authoritative 480 state identified exactly?** Yes.
ρ₃₈₆ SHA-256 `0a498a7d6ab0565b29c15ff9364060d937d4df10aa661fc90d02c038ce6e4a60`,
config hash `03097a28…782e`, `+impl` tree `edbfe47e…52cb` verified, outer 386,
stage 3, move 0.01, N = 2, multJ = 0 — identical to the state
`gray_kkt_forensic_audit` analysed. `FROZEN_480_STATE_IDENTITY_PASS`.

**2. Was ρ ever updated in this task?** No. Zero density updates. Every
perturbed ρ is a temporary in-memory evaluation point; every `drho` computed is
discarded in code without being added to any density.

**3. Was any optimization run executed?** No. Zero `olhoffSolve` calls. The only
solves are frozen-subproblem certifications, labelled as such.

**4. Was the exact final inner MMA subproblem recovered?** Yes, and
**bitwise**: the reconstruction returns `drho` identical to `DRHO(:,386)` (max
abs difference 0), `nInner` 19 = 19, β identical to the last digit.

**5. Are exact retained dual variables available?** **No.** `innerLoop` discards
`mmasub`'s `lam, xsi, eta, mu, zet, s` with `~`, and the asymptotes are local to
it. Nothing dual survives in the trajectory.

**6. If not, how were duals reconstructed?** By re-running the identical
`mmasub` calls on identical inputs through an audit-only mirror
(`fi_innerloop_audit.m`) whose numeric expressions are copied
character-for-character from `innerLoop.m`, and capturing what production throws
away. They are **reconstructed**, exact for the MMA convex subproblem. No
best-fit dual was needed. The mirror's fidelity is proved by the bitwise
reproduction in Q4.

**7. Is the final inner MMA subproblem primal feasible?** Yes. All four
constraints ≤ 0 (max = −3.09e−06), zero box violations, move limit respected at
6.2 % utilization.

**8. Is it dual feasible?** Yes. `λ = [0.99726, 3.62e−07, 2.00e−08, 0.63243]`,
all ≥ 0; `min ξ = min η = 1.58e−06 > 0`.

**9. Does complementarity hold?** Yes, at `mmasub`'s interior-point floor:
max\|λᵢfᵢ\| = 4.70e−06, max ξ·gap = 7.40e−06, max η·gap = 1.04e−05. Active set
{spectral mode 1, volume}; the mode-2 and next-mode rows are slack by 0.28 and
5.0 with duals at 1e−07 and 1e−08.

**10. Does inner Lagrangian stationarity hold?** **No.** Exact MMA residual
(including box multipliers) normalized RMS = **0.3597**, max 2.94. The
preregistered projected statistic gives 1.0060; it is reported unchanged
together with a disclosed error in its bound tolerance (Q34 note). Both cross
the preregistered FAIL bar of 0.1.

**11. Does the final inner MMA KKT pass?** **No — `FINAL_INNER_MMA_KKT_FAIL`.**
And it fails structurally, not by under-convergence: at 500 and 5000 sub-iterates
the residual is 0.586 and 0.597, *worse* than production's 0.360, while
max\|Δρ\| climbs 6.2 % → 87.6 % → 97.9 % of the move limit and the relative step
oscillates non-monotonically at ~2e−03.

**12. What exact physical gradient is being approximated?**
`g_phys = F(:,1,1) = φ₁ᵀ(∂K/∂ρ − λ₁∂M/∂ρ)φ₁ = ∂λ₁/∂ρ`, with M-orthonormal modes,
from `genGrad` before filtering. Sign-invariant in φ₁, so well defined
regardless of what `eigs` returns.

**13. What exact filtered vector reaches MMA?**
`g_filt = (H·(ρ∘g_phys)) ./ (Hs∘max(1e−3,ρ)) = A(ρ)·g_phys` with
`A = diag(1/(Hs∘ρ))·H·diag(ρ)`. The `max(1e−3,ρ)` guard is inactive — 0 of
28 800 elements. That this is what MMA sees is proved, not assumed: at
`drho = 0` the `deltaLambda` subeigenvalue matrix is already diagonal, so the
basis is the identity and the active row's gradient is exactly `−g_filt/lamref`.

**14. Is the filtered field locally conservative?** **No.**
`FILTERED_FIELD_LOCALLY_NONCONSERVATIVE`, on all four preregistered conditions.

**15. How large is Jacobian antisymmetry?** Median relative 0.287 over ten
direction pairs; per-pair 4.3e−03 to 1.97. Values near 2 mean `uᵀJv` and `vᵀJu`
have **opposite signs**.

**16. Does asymmetry converge away with FD refinement?** **No** — it is
constant to 3–4 significant figures across δ = 1e−3 → 3e−5 (0.28675 → 0.28701).
The control does the opposite, rising as 1/δ, which is the FD-noise signature.

**17. Are closed-loop integrals nonzero?** Yes. Normalized 0.272–0.285, matching
the independently measured Jacobian asymmetry as Stokes requires, with amplitude
exponent **1.999** — a genuine curl — and exact sign reversal (relative 3.9e−13).

**18. Does the physical positive control give approximately zero loop?**
Yes: 2.9e−06 normalized, with exponent 0.978 — linear in path length, i.e.
accumulated FD noise, not a curl. The control passes decisively; had it failed,
the preregistration made that a stop condition.

**19. Could eigenvalue mode switching explain the observed asymmetry?** **No.**
Zero of 88 perturbed samples fell below `gap12 = 0.05`; the minimum observed was
0.12979 and there were zero ordering changes. The control uses the *same*
eigenpairs at the *same* perturbed densities and shows 1e−06. The asymmetry is
δ-independent, and it is predicted analytically from the operator structure
alone, with no eigenvalue derivative involved.

**20. Is the filter operator symmetric?** **No.**
‖A − Aᵀ‖_F/‖A‖_F = **1.4135**, against a theoretical maximum of √2 ≈ 1.41421 —
essentially maximally asymmetric. `H` itself *is* exactly symmetric; the
asymmetry is introduced by the ρ-weighting and the `Hs` normalization. The exact
condition, `ρ²Hs` constant across each stencil, has a median relative violation
of 0.0503 and a p90 of 0.9508.

**21. Is it ρ-dependent?** Yes, through both `diag(ρ)` and `diag(1/ρ)`. Its row
sums are `ρ̃/ρ`, measured range **[0.668, 245.09]** — it is not an averaging
operator, and it amplifies void-region sensitivities by 30.7× in RMS.

**22. Under what conditions would `H·g_phys` be integrable?** Three, jointly
sufficient: A constant in ρ, A symmetric, and **A commuting with the Hessian**.
The third is binding and cannot be arranged by construction, since the Hessian
changes with the design.

**23. Are those conditions satisfied here?** **None of the three.** And this was
tested, not just argued: symmetrizing the ρ-free operator reduces the
antisymmetry from 1.227 to **0.031** — a 40× improvement that still does not
reach zero, exactly as the commutation requirement predicts.

**24. Does the volume constraint change the integrability conclusion?** **No,
provably.** Its gradient is `1/Vtot` on every element and independent of ρ, so it
adds a constant vector field whose Jacobian is zero; therefore
`skew(J_reduced) = skew(J_filt)` exactly, for any multiplier value. Confirmed
numerically: the volume-neutral direction pairs reproduce their parents' values
to three digits.

**25. Can an effective scalar regularized objective be identified?** **No —
`EFFECTIVE_SCALAR_OBJECTIVE_NOT_IDENTIFIED`**, and not for want of searching:
a nonzero curl **excludes** the existence of a potential. A conservative
alternative does exist in principle — the density filter, where
`∇[λ₁(Wρ)] = Wᵀ∇λ₁(Wρ)` has the symmetric Jacobian `WᵀHess W` for *any* W — but
that is a different formulation, not an identification of this one.

**26. Does MMA appear to solve the surrogate it is given?** **No.** See Q11.
Its own convex subproblem is solved cleanly at every call, but the outer
fixed-point iteration on the surrogate does not converge, at any budget tested.

**27. Does the surrogate correspond to physical KKT stationarity?** **No —
`SURROGATE_PHYSICAL_KKT_MISMATCH`.** The filtered subproblem is nearly
stationary inside the gray regions (residual 0.013 core, 0.016 shell) where the
physical problem is not (0.334 best gray-only fit, previous audit). The two
agree in solid and disagree most in void, where the filter amplifies most.

**28. Does this explain the broad gray patches?** Partially. It explains why the
algorithm can **stop** there while the physical residual stays substantial — the
surrogate is satisfied in gray and there is no potential being minimized. It
does **not** establish that the filter *creates* the patches; coexistence is not
causation and a non-conservative field can perfectly well have discrete fixed
points.

**29. Does it explain the mesh-growing trend?** **No.** One mesh was examined;
no 400 or 800 evaluation was performed. The obstruction is dominated by
`A·Hess`, and both factors change under refinement — a hypothesis for a future
measurement, not a result.

**30. What is the single highest-information next action?** Resolve the
inner-solve failure at the frozen 480 state: characterize the oscillation with
full primal/dual/asymptote state retained, and test the same frozen subproblem
under a conservative-approximation (GCMMA-style) safeguard to learn whether it
has a reachable KKT point at all. Frozen-state certification only, no density
update. See `NEXT_ACTION.md`.

**31. Is projection still premature?** **Yes.** Nothing here bears on it, it
would change filter and variable representation together on top of an
unexplained inner-solve failure, and it is CLASS D — absent from every Du &
Olhoff source.

**32. Is p-continuation still unjustified?** **Yes.** This audit produced no
evidence about the penalization schedule.

**33. Is the performance campaign still blocked?** **Yes —
`PERFORMANCE_CAMPAIGN_STILL_BLOCKED`.** The endpoint depends on an inner
truncation tolerance, and the campaign's purpose (mesh convergence) needs the
diagnosis rather than nine more meshes.

**34. Were zero accepted density updates executed?** **Yes — zero.** Also zero
optimization runs, zero continuation, zero controller transitions, zero
production files modified, and no 400 or 800 evaluation.

*Disclosed error:* `AUDIT_PREREGISTRATION.md` §4 specified a 1e−12 bound
classification tolerance for the projected stationarity residual. `mmasub` is an
interior-point method that never places a variable exactly on a bound, so at
that tolerance zero variables are classified active and the statistic omits the
box multipliers. The preregistered number (1.006) is reported unchanged beside
the correct one (0.360) and a tolerance sweep. Both cross the FAIL bar, so no
verdict depends on the error.

---

## Figures

All twelve required figures, plus one supplementary.

| # | figure |
|---|---|
| 1 | `FIG_01_gradient_maps` — physical vs filtered vs density |
| 2 | `FIG_02_filtered_minus_physical` — difference and log amplification ratio |
| 3 | `FIG_03_asymmetry_distribution` — filtered vs control |
| 4 | `FIG_04_uJv_vs_vJu` — conservative fields lie on the diagonal |
| 5 | `FIG_05_symmetry_vs_step` — flat for filtered, 1/δ for control |
| 6–7 | `FIG_06_07_loop_integrals` — a² vs a¹ scaling, both fields |
| 8 | `FIG_08_mixed_partial_asymmetry` — 30×30 submatrices |
| 9 | `FIG_09_inner_kkt_breakdown` — residual map and class bars |
| 10 | `FIG_10_complementarity` — active set and box complementarity |
| 11 | `FIG_11_residual_by_class` — residual and amplification by class |
| 12 | `FIG_12_core_and_mechanism` — core geometry and the S1/S2 split |
| 13 | `FIG_13_inner_convergence` — supplementary: the inner loop never converges |

## Artifact guide

Preregistration: `AUDIT_PREREGISTRATION.md` (SHA-256 `53d6623a…`).
Identity and recovery: `STATE_IDENTITY.md`, `FINAL_SUBPROBLEM_RECOVERY.md`.
Inner solve: `INNER_MMA_KKT.md`.
Field mathematics: `GRADIENT_OBJECTS.md`, `JACOBIAN_SYMMETRY.md`,
`CLOSED_LOOP_INTEGRALS.md`, `MIXED_PARTIAL_AUDIT.md`,
`FILTER_OPERATOR_ANALYSIS.md`, `MULTIPLICITY_CONTROL.md`,
`EFFECTIVE_OBJECTIVE.md`.
Interpretation: `LITERATURE_INTERPRETATION.md`, `GRAYNESS_IMPLICATION.md`.
Decisions: `NEXT_ACTION.md`, `PERFORMANCE_STATUS.md`.
Provenance and integrity: `PROVENANCE.md`, `METRICS.json`, `DATA_MANIFEST.json`,
`EVIDENCE.json`, `FINAL_SHA256.txt`, `evaluations/`.

# WHAT WE LEARNED

The sensitivity-filtered update field is not a gradient. That is now a
measurement with a passing positive control, an exact analytic decomposition
verified to 1e−08, and a mechanism attributed 99 % to non-commutation between
the filter operator and the physical Hessian rather than to the ρ-weighting
everyone would suspect first. Symmetrizing the operator would not fix it; only
changing the composition structure — as a density filter does — would, and that
is a different formulation, not a repair of this one.

The final inner subproblem is never solved. Production stops it at 6 % of the
move limit; the subproblem wants 98 %. The design's trajectory is therefore
shaped by a stopping tolerance that `innerLoop`'s own header describes as a
reconstruction, the paper having given no criterion.

Together these say the endpoint is a fixed point of a truncated iteration on a
field with a curl — neither a physical stationary point nor the surrogate's own.

# WHAT WE DID NOT LEARN

Why the inner iteration oscillates rather than converging. Whether a convergent
method would find a KKT point of the same subproblem at all. Whether the filter's
non-conservativity causes the gray patches, as opposed to merely coexisting with
them. Why grayness grows with mesh — one mesh was studied, by instruction.
Whether the wider literature characterizes the sensitivity filter's mathematical
status: all 30 reference PDFs were searched and none discusses it, and Sigmund
(1997), Sigmund & Petersson (1998) and Bendsøe & Sigmund are not in this
repository, so no literature position is asserted.

# WHAT I WOULD DO NEXT

Resolve the inner solve before touching the formulation. Three frozen-state
certifications, none of which updates a density: characterize the oscillation
with full state retained; re-solve the same subproblem under a
conservative-approximation safeguard to see whether a reachable KKT point
exists; and localize the failure against the void region, where the residual is
0.578 and the filter amplifies 30×.

Only then is the filter question answerable. And when it is asked, the target
should be the composition structure — evaluating the objective at a filtered
density, which is conservative for any operator — not a symmetrized or retuned
sensitivity filter, which the counterfactual measurement here shows would not
restore integrability.

Do not start projection or p-continuation. Do not run the nine-mesh campaign.
Neither is supported by anything measured here.
