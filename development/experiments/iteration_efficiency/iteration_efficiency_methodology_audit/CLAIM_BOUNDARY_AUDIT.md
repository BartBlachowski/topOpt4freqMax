# Claim boundary audit

Audit-only. Classifies prospective paper claims as **A** (supported if the experiment
succeeds, once the mandatory corrections are applied), **B** (supported only with an
explicit qualification in the same sentence), or **C** (not supported by this experiment).

Where a claim is already decidable from frozen evidence, that is noted — several are.

---

## A — Supported if the experiment succeeds

**A1. "The proposed method requires fewer method-level design updates to enter and certify a
state within 1 % of the best common-E1 quality it itself sustains."**
Conditional on the data showing it. Must appear with the achieved quality in the same
sentence or the adjacent column, and with `k_enter` or `k_cert` named.

**A2. "The Du–Olhoff method requires more optimization machinery per update."**
**Already supported, before the study, from source.** Per outer iteration it performs one
generalized eigensolve with multiplicity detection, generalized-gradient construction for
the clustered modes plus mode `J` with filtering, and one LP subproblem with `N_e + 1`
variables and `N(N−1)/2` equality constraints. The proposed method performs one FE solve and
one OC update, with **no eigensolve inside the loop** (its reference eigenpair is computed
once from the fully solid design and never refreshed). This is a structural claim about the
algorithms, not an empirical one.

**A3. "The Yuksel method is intermediate in chronological method-level update count."**
If observed, and only with the Stage-1/Stage-2 decomposition shown, since `N_total` includes
a preparatory stage that optimizes a different objective.

**A4. "The proposed method reaches an accepted solution with a plain OC optimizer, no LP
subproblem, no multiplicity handling, and no per-iteration eigensolve."**
**Already supported from source.** This is the cleanest claim in the paper and does not
depend on the new runs at all.

**A5. Per-method, per-mesh statements of the form:** "method M entered the accepted regime
at `k_enter` = X and certified at `k_cert` = Y, reaching common E1-raw ω₁ = Z, which is
W % of its own sustained reference and V % of the best observed across the three methods."
This is the full, honest unit of result and every headline number should reduce to it.

---

## B — Supported only with qualification

**B1. "The proposed method is computationally faster."**
Qualification: on the named reference platform, single-threaded, serial, at the named
endpoints, for these nine meshes, with the platform key adjacent. The protocol already
forbids hardware-independent timing claims — keep that prohibition verbatim.

**B2. "The proposed method scales better."**
Qualification: as *empirical scaling of the named quantity over `N_e` ∈ [3200, 80 000]*
only; on common mesh support only; with the leave-one-out `p` range shown; and never for
`k_cert` without the certification-convention caveat. Note the frozen evidence that the
coarsest mesh alone moves `p` by 3× on a four-point series, and that the completed
campaign's own exponents (1.19 / 1.71 / 1.42) were fitted on 6 / 7 / 9 points over different
ranges.

**B3. "The proposed method is more efficient."**
Qualification: "efficient" must be defined in the same sentence as *self-referenced
maturation work*. Unqualified, it reads as work per unit of achieved quality, which is not
measured and which the frozen 6–7 % quality gap contradicts.

**B4. "The Du–Olhoff method fails at some meshes."**
Qualification, mandatory: "the LP subproblem, solved by `dual-simplex-highs` in the recorded
MATLAB version, returned exit flag 0 at outer iterations 358, 400 and 1067 at three meshes."
The forensic closure classifies this `GENERIC_LP_ITERATION_LIMIT_ONLY`, with `linprog`'s
`MaxIterations` at its 2.1475e9 default and `MaxTime = Inf` — so it is a reproducible
failure of this solver on this LP, not a property of the Du–Olhoff formulation. The status
name `GENUINE_SOLVER_FAILURE` is accurate but reads as a method indictment in a table; the
classification must travel with it (Finding Mo5).

**B5. "The proposed method requires fewer iterations than Du–Olhoff."**
Qualification: iteration units are not equal work. At 800x100 the frozen per-update times
are 1.844 s (Olhoff), 0.814 s (Proposed), 0.489 s (Yuksel) — a 3.8× spread — and the Olhoff
eigensolve alone is 75 % of its outer-iteration cost. The per-iteration cost figure must be
adjacent, not merely cross-referenced.

**B6. "The Du–Olhoff method is iteration-efficient" (should the data show low outer counts).**
Same qualification, in the opposite direction. Recorded here because the firewall requires
the constraint to bind symmetrically: a low Olhoff outer count would also not mean low
computational work.

**B7. "Certification requires ~100 additional iterations for every method."**
Qualification: equal in iterations, unequal in seconds and unequal in proportion. It is 30–93 %
of Proposed's native run and 6 % of Olhoff's fixed horizon; at 800x100 it costs 183 s for
Olhoff against 81 s for Proposed.

---

## C — Not supported by this experiment

**C1. "The proposed method has lower algorithmic complexity."**
Complexity requires an operation-count model. This study has 1.398 decades of empirical
points, one deterministic run per cell, and method-dependent censoring.
`SCALING_AND_FIGURE_SPEC.md` §6 already prohibits this language, and that prohibition is
correct and should not be relaxed.

**C2. "The proposed method produces equally good solutions."**
**Contradicted by frozen evidence.** Du–Olhoff leads common E1-raw ω₁ at every one of the
nine meshes by 6.1–7.2 % over Proposed, with the identical ordering under E2-raw and
E3-raw. That is seven times the R equivalence margin. R is designed not to test this claim,
and the existing endpoint evidence points against it. This claim must not appear in any
form, including softened forms ("comparable quality", "similar performance", "without loss
of quality").

**C3. "The proposed method is less optimizer-dependent."**
Nothing in the design varies the optimizer. `topopt_freq.m` does contain an MMA branch, but
the frozen profile is OC and `IMPLEMENTATION_REQUIREMENTS.md` §2 forbids changing it. The
study holds the optimizer fixed rather than varying it, so it cannot speak to optimizer
dependence.

**C4. "Fewer iterations means less computation."**
Not without the per-iteration cost figure. See B5.

**C5. "These results transfer to other problems."**
One benchmark: an 8×1 beam, mid-height pinned at both ends, `V_f` = 0.5, `p` = 3,
first-eigenfrequency maximization, plane-stress Q4 with a consistent mass matrix, uniform
initial design. No claim about other boundary conditions, load cases, volume fractions,
objectives, element types, or 3D.

**C6. "The reported counts are converged solutions."**
The gate is *persistent acceptability*, not stationarity — `ACCEPTANCE_GATE_SPEC.md` §5 is
explicit about this and correct. The Olhoff profile in particular is documented as never
becoming stationary (`selected_profile.json`:
`simple_native_stopping_rule_identified: false`). "Converged" must not be used for any
accepted state.

**C7. "The topology gate is a neutral common resolution standard."**
Not as specified. `a_res` derives from the smallest of three different frozen filter
radii (5 / 9 / 21 cells), and the aggregate clause is not resolution-derived at all.
Whatever survives the C1 correction should be described as "the strictest of the three
frozen filter footprints", never as a common standard.

**C8. "`k_enter` is the minimum work required, full stop."**
`k_enter` is a functional of the observation budget through `Q_ref`, and the budgets differ
by 3.5× across methods. Until Finding C2 is corrected, `k_enter` is "minimum work relative
to the quality this budget revealed", not "minimum work".

---

## Claims the study is uniquely well placed to make, and should

The corrections in this audit are mostly restrictive. Three claims are *strengthened* by the
evidence and deserve to be made explicitly:

1. **The three methods reach materially different quality on the same benchmark under a
   common evaluator, and the ordering is evaluator-independent.** A 6–7 % ω₁ separation,
   stable across E1, E2 and E3 and across all nine meshes, is a substantive result in its
   own right and is currently buried in a supplementary companion figure.
2. **Per-update computational content differs qualitatively, not just quantitatively.** One
   method performs a generalized eigensolve and an LP per update; the other two perform a
   single linear solve. Naming that structure is more informative than any exponent this
   study can fit.
3. **`nInner` in the selected Du–Olhoff LP path is one `linprog` call, not a simplex
   iteration** — and the 38-iteration failed call proves the levels differ. Publishing that
   distinction corrects a genuine hazard for anyone reading this family of codes, and the
   package already documents it better than the literature does.
