# Phase 2E — Independent delta audit of the Phase-2D common-evaluator amendment

READ_ONLY_INDEPENDENT_DELTA_AUDIT — NO OPTIMIZATION — NO METHODOLOGY EDITING —
NO REFREEZE — NO PRODUCTION

Date 2026-08-30 · branch `benchmark-methodology-r2` · HEAD `632e9b01811845709de33f93051fd853373ed5e1`
No optimizer was run. No file outside `analysis/iteration_efficiency_phase2d_delta_audit/`
was modified. No authorization token was set.

---

## Verdict in one paragraph

Phase 2D did narrowly scoped, honest, well-provenanced work, and **every numerical claim it
makes is independently reproduced here**. Its scope discipline is exemplary: two functional
lines, E1 untouched, every protected source byte-identical, the frozen evaluator and contract
deliberately left alone, and the loss of native identity disclosed rather than concealed. The
amendment does exactly what it claims: it removes the Du–Olhoff Eq. (4) discontinuity and
drives the common evaluator's response to storage precision down to the branch-free E1
control. Nevertheless the amendment **must not be refrozen**, because Phase-2D's validation
measured only the evaluator's response to *perturbations* of a state and never its *value* at
a state. When this audit measured the value, it found that Eq. (4a) reintroduces the spurious
localized eigenmodes that Du & Olhoff introduce Eq. (4) to eliminate — on the gray
intermediate trajectory states this study exists to score, at a magnitude three orders larger
than the instability being cured, in data Phase 2D itself published.

---

## 1. What this audit did independently

Nothing below relies on a Phase-2D artifact except as a comparison target.

- The primary source `references/Du2007_Topological.pdf` §2.2 was extracted and read
  directly (`scripts/du2007_section22.txt`), not through any paraphrase.
- The branch-point mathematics was recomputed in **exact rational arithmetic** and in
  IEEE-754 double (`scripts/wp2_math.py`).
- The evaluator's spectral core was **re-implemented from scratch in Python/NumPy/SciPy**
  (`scripts/audit_evaluator.py`) — different language, different sparse eigensolver (ARPACK
  shift-invert at sigma = 0 with a fixed start vector and `tol = 0`), and independently
  validated against the frozen MATLAB evaluator to 5e-12 relative.
- The frozen decision engines `reference_phase`, `scan_persistence` and `measurement_budget`
  were **re-implemented in Python** (`scripts/frozen_engines.py`) and validated by
  reproducing the stored Phase-2B outputs exactly.
- Selected results were confirmed by a **third, non-iterative solver**: dense LAPACK
  `scipy.linalg.eigh` on the full 6758-DOF generalized problem.
- Every `.mat` artifact in the repository was enumerated and every numeric dataset inside it
  classified, to test Phase-2D's evidence-availability claim rather than accept it.

## 2. What was confirmed

| Phase-2D claim | Phase-2D | this audit | agreement |
|---|---|---|---|
| Eq. (4) branch straddle, max rel E2 | 4.0021e-03 | **4.0021e-03** | exact to 5 s.f. |
| Eq. (4) float32, 236 paired states, max rel E2 | 2.6736e-02 | **2.6736e-02** | exact to 5 s.f. |
| paired states with a branch crossing | 70 of 236 | **70 of 236** | exact |
| Eq. (4) branch side, 160x20, max rel E2 | 2.6496e-02 | **2.6496e-02** | exact to 5 s.f. |
| Eq. (4a) float32, max rel E2 | 5.5955e-08 | **5.5960e-08** | 4 s.f. |
| Eq. (4a) branch side, 160x20, max rel E2 | 2.6533e-10 | **2.6560e-10** | 3 s.f. |
| Eq. (4a) branch straddle, max rel E2 | 2.1551e-13 | 8.2820e-13 | both at the solver floor (see P2) |
| binding evaluator, 160x20 surrogate | 150/1600 → 0/1600 | **150/1600 → 0/1600** | exact, shares (319,33,1248) and (582,517,501) exact |
| hard gate identical | 1600/1600 | **1600/1600**, 1065 passing | exact |
| Phase-2C figure 751/3200 | (cited) | **751/3200**, shares (2695,20,485) | exact |
| Phase-2B `b_ref` 2200/2100, `k_enter` 233/315/609 vs 232/309/524 | (cited) | **reproduced exactly** | exact |
| amended trajectory values, all 1600 states | — | agree with Phase-2D's CSV to **1.07e-11** | — |

The amended E2 residual tracks the branch-free E1 control to a ratio of **1.012** on the
trajectory and **1.0002** on the paired states. Phase-2D's central stability claim — that the
mechanism is gone rather than merely reduced — is correct.

## 3. The finding that governs the verdict

### D1 — Eq. (4a) reintroduces spurious localized eigenmodes

Du & Olhoff §2.2 states plainly what the low-density mass branch is for:

> "application of the SIMP model … may lead to the occurrence of spurious, localized
> eigenmodes associated with very low values of corresponding eigenfrequencies … **To
> eliminate these spurious eigenmodes** …"

and justifies indifference between (4), (4a) and (4b) on one ground:

> "…only found negligible differences in the **final results**. The reason is that the region
> with lower density … has a very small contribution to the first several eigenfrequencies …
> Furthermore, **all intermediate values of the material density will approach 0 or 1 during
> the design process**, which implies that the changes of the interpolation model … must have
> very limited influence on the **final 0–1 design**."

Phase 2C used the failure of that premise to argue against Eq. (4) for this study. **The
identical failure applies to Eq. (4a), and Phase 2D did not test it.**

Measured at state k = 252 of the stored 160x20 Olhoff production trajectory (1251 of 3200
elements at ρ ≤ 0.1), by dense LAPACK, reporting the fraction of each mode's kinetic energy
carried by elements with ρ ≤ 0.1:

| | ω₁ | ω₂ | ω₃ | ω₄ | ω₅ | lowest structural mode |
|---|---|---|---|---|---|---|
| **E2, Eq. (4)** | 166.487 | 166.636 | 311.862 | 514.537 | 708.197 | **index 1** |
| void KE share | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | |
| **E2, Eq. (4a)** | **31.404** | **104.993** | **107.724** | 166.367 | 166.505 | **index 4** |
| void KE share | **1.0000** | **1.0000** | **1.0000** | 0.0015 | 0.0016 | |
| **E3, Eq. (4a)** | **22.390** | **100.311** | **100.624** | **154.400** | 166.367 | **index 5** |
| void KE share | **1.0000** | **1.0000** | **1.0000** | **0.9999** | 0.0015 | |
| **E1** (both laws, unchanged) | 165.869 | 166.019 | 291.537 | … | … | **index 1** |
| void KE share | 0.0077 | 0.0077 | 0.9999 | | | |

Under Eq. (4a) the evaluator's reported ω₁ for E2 is a **void mode carrying 100% of its
kinetic energy in the low-density region**, at 18.9% of the structural value. Three spurious
modes sit below the structure for E2 and four for E3 — and the frozen evaluator computes only
**three** modes and takes the lowest, with no modal-validity check.

Extent, on stored evidence:

| mesh | states screened | states with amended E2 or E3 mode-1 spurious | worst ω₁ ratio to E1 |
|---|---|---|---|
| 160x20 | 1600 (all) | **34** (k = 237…272, near-contiguous) | **0.1893** |
| 240x30 | 800 (stride 2) | **27** | 0.1976 |
| 320x40 | 400 (stride 4) | **11** (incl. k = 1385, near convergence) | 0.2944 |

Under Eq. (4) the minimum `E2/E1` ratio over all 1600 states of the 160x20 trajectory is
**1.0029** — Eq. (4)'s suppression works exactly as the source intends. The defect is created
by the amendment.

**It is in Phase-2D's own data.** `AMENDED_OLHOFF_TRAJECTORY_EVALUATION.csv` records
`new_E2 = 31.4041` and `new_E3 = 22.3898` at k = 252 against `new_E1 = 165.8691`. This audit's
independent implementation agrees with that file to 1.07e-11. Nothing in
`PHASE2D_AMENDMENT_REPORT.md` mentions it.

**Why the validation missed it (finding D3).** Every Phase-2D experiment measured the
evaluator's *response to a perturbation* — one ULP, float32 rounding, branch side — and each
of those is genuinely excellent under Eq. (4a). None measured the *value*. Computing
`max |new − old| / old` from Phase-2D's own columns gives **8.11e-01 for E2** and **8.66e-01
for E3**, with 32 states above 1%. That figure appears nowhere in the report.

**Why it reaches the frozen decision path.** Seven of the 34 affected 160x20 states
(k = 249, 250, 255, 256, 257, 258, 259) pass the hard gate and therefore enter the acceptance
scan, where the robust ratio would read ≈ 0.19 against q levels of 0.98–0.995. Any P = 100
persistence window overlapping the block fails. The block sits where the estimand lives —
Olhoff's own `trigger_iterations` at 160x20 is 245, and the 96x12 `k_enter` values are
233/315/609.

**The mechanism, and why it matters for the fix.** E1 is untouched by any of this, and not
by accident. E1 carries a stiffness floor of `1e-6`; E2's is `1e-9`, a thousand times softer,
and E3 has no additive mass floor at all — only a `1e-3` density clamp, giving a void
stiffness of `1e-2` against E1's `10`. **Eq. (4)'s `x^6` suppression is what compensated for
those weak floors.** Probing the option space at the six worst affected states plus the
converged final state (`REMEDY_FEASIBILITY_PROBE.csv`, 105 evaluations) makes the ordering
plain — the more void mass a law admits, the worse E2/E3 become, while E1 stays structural
throughout:

| mass law for E2/E3 | E2 ω₁ at k = 256 | E3 ω₁ | void participation | E1 under the same law |
|---|---|---|---|---|
| Eq. (4) | 166.511 | 166.511 | 0.0000 | structural |
| Eq. (4a) | 33.724 | 23.847 | **1.0000** | structural (166.375) |
| Eq. (4b) | 21.329 | 15.082 | **1.0000** | structural (166.284) |
| linear (Eq. 2, q = 1) | 13.498 | 9.545 | **1.0000** | structural (165.929) |
| Eq. (4a) on the **exact-count binary field** | 169.321 | 169.321 | **0.0000** | structural |

A linear mass law for E2/E3 is therefore refuted outright — spurious at **all seven** probed
states including the converged k = 1600 — even though E1 uses one safely. Only two candidates
survive the probe: Eq. (4a) with a declared modal-validity rule, or evaluating E2/E3 on the
exact-count binary field the evaluator already computes. Both are genuine methodology changes,
not two-line amendments.

**Why it is also a neutrality failure.** The artefact's incidence depends on how long a
method carries a large low-density gray region, which is a property of the update law, not of
design quality. No Proposed or Yuksel density trajectory exists in the repository, so the
bias cannot be measured for two of the three methods.

**Eq. (4b) is not the way out (finding D2).** It places 2× to 6× more mass in the void than
Eq. (4a) at every density in (0, 0.1). At k = 252 it gives E2 ω₁ = **22.206** and E3
ω₁ = **15.832** — worse than Eq. (4a). Phase-2D's decision not to adopt it was right, for a
better reason than the one given.

## 4. What passes

| WP | subject | ruling |
|---|---|---|
| WP0 | integrity, protected sources | **PASS** — 0 mismatches across 24 hashed entries plus Phase-2D's own 28-file manifest |
| WP1 | source provenance | **PASS** — every claim about Eq. (4)/(4a)/(4b) verified against the primary PDF |
| WP2 | mathematics | **PASS** — confirmed in exact rational arithmetic; Eq. (4a) C0, not C1, global Lipschitz constant 6 |
| WP3 | minimum correction | **FAIL** — minimal in *scope*, insufficient in *effect*; see §5 |
| WP4 | implementation scope | **PASS** — exactly two functional lines, verified by call path, not diff alone |
| WP5 | native-identity cleanup | **FAIL** — incomplete; N1, N3, N4 |
| WP6/7 | old defect and cure | **PASS** — reproduced independently, both directions |
| WP8 | float32 mechanism | **PASS** — branch crossing still occurs under Eq. (4a); the finite jump does not |
| WP9 | binding-evaluator instability | **PASS with qualification** — reproduced exactly; surrogate-normalisation caveat (B1); 751/3200 vs 150/1600 reconciled (B2) |
| WP10 | hard-gate invariance | **PASS** — structural, not merely observational |
| WP11 | numerical scale | **PASS** — 922× margin against the binding decision margin |
| WP12/13 | reference/persistence gap | **B — non-blocking for refreeze, mandatory post-refreeze** |
| WP14 | Phase-2B classification | **PASS** — correct in both directions |
| WP15 | new precision qualification | **YES, required** |
| WP16 | neutrality | **FAIL** — see §6 |
| WP17 | C1 kink | **ACCEPTABLE** — irrelevant to this use, and Eq. (4b) would be worse |
| WP18 | collateral drift | **NONE** |
| WP19 | refreeze obligations | audited; five items added |
| WP20 | retrospective classification | revised; see `RETROSPECTIVE_RESULT_CLASSIFICATION.csv` |
| WP21 | new optimizer run | **not needed to justify a mass law; needed to qualify storage** |

## 5. WP3 — is Eq. (4a) the minimum evidence-supported correction?

    EQ4A_MINIMUM_CORRECTION = FAIL

The demonstrated defect was a **finite value jump**, not a derivative defect, and C0
continuity is exactly the property whose absence caused it — so the *logic* of choosing
Eq. (4a) over Eq. (4b), over arbitrary smoothing, over an E1-only evaluator, and over a
tolerance/snap around 0.1 is sound, and this audit reaches the same conclusion on each of
those comparisons independently:

| option | assessment |
|---|---|
| keep Eq. (4) | refuted: 2.67e-2 float32 error is **439×** the binding decision margin, exceeded on 48.7% of states |
| Eq. (4b) (C1) | unnecessary — the amended residual is already at the branch-free E1 floor, so a lower Lipschitz constant buys nothing — **and** worse on D1 |
| arbitrary smoothing | introduces a free parameter absent from the source: a new methodological degree of freedom requiring its own neutrality defence |
| tolerance / snap near 0.1 | still discontinuous, merely relocated to 0.1 ± τ, and adds a free parameter |
| E1-only evaluation | destroys the multi-perspective design and reinstates the M4 objection that E1 is one competitor's own constitutive model |

But minimality of scope is not the whole test. The correction must also *produce a fit-for-
purpose evaluator*, and D1 shows it does not. Phase-2D's minimum-correction analysis
considered only the stability axis and never asked whether the replacement still measures the
structural first eigenfrequency on the states being scored.

## 6. WP16 — neutrality

    AMENDED_NEUTRALITY_ARGUMENT = FAIL

The *structure* of Phase-2D's replacement argument is right: identical treatment and
producer-independence are the correct substitutes for native identity, and both genuinely
hold. But a method-independent mapping is necessary, not sufficient. Detail and the measured
E2–E3 degeneracy (median **5.2e-09** — six orders tighter than the frozen wording implies) are
in `NEUTRALITY_AUDIT.md`.

## 7. WP12 — the central open question, ruled

Phase-2D's claim that no reference-length density evidence exists was **not** taken on trust.
Every `.mat` file in the repository was enumerated; the longest density trajectory anywhere is
**1601 snapshots**, and the four 3200-length artifacts
(`probe_96x12_H3200`, `decide_96x12`, `final_96x12`, `resolve_96x12`) carry quality arrays,
pass matrices and reference structures but **no density field**. The claim is confirmed.

This audit then measured what Phase 2D asserted. Re-implementing the frozen engines and
propagating a bounded relative perturbation through them exactly:

| frozen decision | critical relative Q perturbation | amended float32 (5.60e-08) | amended double-ULP |
|---|---|---|---|
| `b_ref` | 8.756e-05 | **1560×** | 1.05e+08× |
| `k_enter`/`k_cert` q = 0.98 | 9.001e-05 | 1610× | 1.08e+08× |
| **q = 0.99 (binding)** | **5.162e-05** | **922×** | 6.22e+07× |
| q = 0.995 | 6.491e-05 | 1160× | 7.82e+07× |

Zero of 3200 states sit within twice the amended perturbation of an acceptance threshold.
The same analysis explains the Phase-2B failure exactly: the Eq. (4) error was 439× the
critical perturbation, on 48.7% of states.

**Ruling: B.** The gap is non-blocking for refreeze and mandatory in the new post-refreeze
precision qualification. It is *not* the reason refreeze is blocked.

---

## Required final summary

1. **Is Du–Olhoff Eq. (4) intentionally discontinuous?** Yes. The source prints it
   coefficient-free and states "It is noted that (4) is discontinuous at the low value
   ρe = 0.1 … Numerically, this is not a serious problem, as the discontinuity only occurs at
   a single point."
2. **Is Eq. (4a) source-defined?** Yes — §2.2, "the following revised form of (4)", separately
   numbered, alongside (4b).
3. **Is c0 = 1e5 correct?** Yes — "the coefficient c0 = 10^5 enforces the C0 continuity at the
   value ρe = 0.1". Matches `massScale.m` case `'4a'` and `mass_interp.m` `du2007_c0`.
4. **Is Eq. (4a) C0 continuous at x = 0.1?** Yes — residual **exactly 0** in ℚ; 6.939e-17 in
   IEEE double, one ULP.
5. **Is Eq. (4a) C1 continuous?** **No** — one-sided derivatives 6 and 1. Phase 2D does not
   claim otherwise.
6. **Is the C1 kink relevant to this post-hoc use?** No. The evaluator is never differentiated
   and the amended residual already equals the branch-free E1 control, so a lower Lipschitz
   constant cannot reduce it. `EQ4A_C1_KINK = ACCEPTABLE`.
7. **Is Eq. (4a) the minimum evidence-supported correction?** **No** —
   `EQ4A_MINIMUM_CORRECTION = FAIL`. Minimal in scope, insufficient in effect (D1).
8. **Did Phase 2D change only common E2/E3 mass interpolation?** Yes. Two functional lines,
   verified by call-path inspection and by bit-identical E1 output at all 1600 states.
9. **Was E1 unchanged?** Yes — level shift 0.0000e+00 at every state.
10. **Were all native optimizers unchanged?** Yes — all six `protected_numerical_sources`,
    three `profile_sources` and `massScale.m` / `defaultCfg.m` / `mass_interp.m` re-verify.
11. **Were protected hashes preserved?** Yes — 0 mismatches, before and after this audit.
12. **Were stale native-identity claims removed?** **No — `NATIVE_IDENTITY_CLEANUP = FAIL`.**
    `FAIRNESS_RISK_REGISTER.md` F01 (a contract-listed normative document) still asserts them;
    `PHASE2_FINAL_READINESS.md` item 4 still mandates the false `x^6` description; the
    contract's `E2_E3_shared_mass_law` string and two `normative_documents` digests are stale.
13. **Old max double-ULP instability, independently reproduced.** **4.0021e-03** (E2 and E3),
    against an E1 control of 8.1090e-13. Single-element variant: 2.5213e-06.
14. **New max double-ULP instability.** **8.2820e-13** (E2), 5.3133e-13 (E3), E1 control
    8.1090e-13 — at or below the eigensolver differencing floor, i.e. indistinguishable from
    E1 and from zero.
15. **Old float32 instability.** **2.6736e-02** (E2 and E3) over 236 genuine paired states,
    70 with branch crossings; E1 5.5949e-08.
16. **New float32 instability.** **5.5960e-08** (E2), 5.5945e-08 (E3) — E1 is 5.5949e-08. The
    amended figure *is* the generic quantisation floor.
17. **Old 160x20 branch effect.** **2.6496e-02** (E2 and E3) over 1600 states, 761 with at
    least one at-risk element, up to 620 in a single state.
18. **New 160x20 branch effect.** **2.6560e-10** (E2), 2.6557e-10 (E3), E1 2.6252e-10 —
    ratio to the branch-free control 1.012.
19. **Old binding-evaluator instability on the audited dataset.** **150 of 1600 (9.38%)** on
    the 160x20 trajectory under the surrogate maximum normalisation; shares (319, 33, 1248).
    Under a different arbitrary surrogate (final-state value) the same data give 255/1600.
20. **New binding-evaluator instability.** **0 of 1600 (0.00%)**; shares (582, 517, 501).
    Under the alternative surrogate, 1/1600.
21. **150/1600 versus 751/3200.** Both valid, different experiments. **751/3200 = 23.47%** is
    the 96x12 horizon-3200 trajectory under the **frozen `Q_ref` normalisation** recomputed
    per branch, perturbation = branch side; independently reproduced here to the state, with
    binding shares (2695, 20, 485) matching Phase-2C exactly. **150/1600 = 9.38%** is the
    160x20 production trajectory, horizon 1600, under a **surrogate maximum normalisation**
    because `Q_ref` is unobtainable there (the reference does not establish on that artifact).
    Different dataset, horizon and normalisation; no contradiction.
22. **Hard-gate invariance.** `HARD_GATE_INVARIANCE = PASS`, structurally.
    `topology_metrics(x, nelx, nely, opts)` takes only the density field — no evaluator
    argument, no global, no call into the evaluator. Independently re-implemented and
    reproduced **1600/1600 identical**, 1065 passing.
23. **Maximum amended perturbation / 0.5% band.** float32 **1.119e-05**; double-ULP
    **1.657e-10**. Against the *binding decision margin* rather than the band: **922×** and
    **6.2e+07×** of headroom.
24. **Any remaining branch-side classification flip?** No. Branch-side robust perturbation is
    2.66e-10 and the binding evaluator never changes (0/1600). **But** 34 states of the same
    trajectory carry a spurious mode-1 under Eq. (4a) regardless of branch side — a level
    defect, not a branch-side flip.
25. **Does adequate B_ref = 3200 density evidence exist?** **No.** Exhaustively verified: the
    longest density trajectory in the repository is 1601 snapshots; the four 3200-length
    artifacts retain quality arrays only.
26. **Can `b_ref` be directly recomputed under Eq. (4a)?** No — it needs the amended Q over
    3200 updates, which needs densities that were not retained.
27. **Can `B_meas` be directly recomputed?** No, for the same reason. (Note it was already
    insensitive on the existing artifact: `B0 = B_ref = 3200` saturates the formula.)
28. **Can `k_enter` be directly recomputed?** No.
29. **Can `k_cert` be directly recomputed?** No.
30. **Is the missing end-to-end reference/persistence evidence blocking refreeze?** **No** —
    ruling B. The rules are provably unchanged, the pointwise perturbation is bounded by
    measurement, and it sits 922× below the binding decision margin on the one
    reference-length trajectory that exists.
31. **Is it blocking production?** **Yes.** `production_preflight` fails closed on
    `olhoff_lossless_trajectory`; the required artifact does not exist.
32. **Is the amended neutrality argument adequate?** **No** —
    `AMENDED_NEUTRALITY_ARGUMENT = FAIL`. Identical treatment and producer-independence hold,
    but the instrument stops measuring the estimand on a class of states whose frequency is
    method-dependent and unmeasurable for two of three methods.
33. **Is E2/E3 dependence disclosed adequately?** No. The frozen wording says the minimum is
    "closer to two-way in evidential terms"; measured, E2 and E3 agree to a median of
    **5.2e-09** — at the 0.5% measurement scale it *is* two-way.
34. **Does Phase 2B remain historically valid?** **Yes** —
    `PHASE2B_HISTORICAL_CLASSIFICATION = PASS`. Reproduced exactly from stored arrays; files
    unmodified.
35. **Is Phase 2B valid as a qualification under the NEW evaluator?** **No.** Its failure
    mechanism — a factor-1e5 jump on branch crossing — is absent under any continuous law.
    Verified: the decisive pair still crosses branches under Eq. (4a), but `g` changes by
    1.49e-08 instead of 9.99990e+04.
36. **Is a new precision qualification required?** **Yes.**
37. **What exact invariants must it prove?** Eleven, listed in
    `PRECISION_REQUALIFICATION_REQUIREMENTS.md`: pointwise E1/E2/E3 bounds; identical
    `volume_pass`/`topology_pass`/`hard_gate_pass`; identical `b_ref`; `Q_ref` bounds;
    identical `B_meas` and truncation flag; identical per-state acceptance at all three q;
    identical `k_enter`; identical `k_cert`; identical final status; identical `k_enter` under
    OAT P = 50 and 200; and — added by this audit — the minimum observed decision margin with
    its ratio to the measured perturbation, plus the mode-1 void-participation diagnostic.
38. **Is a new optimizer run required to justify Eq. (4a)?** **No.** The entire assessment,
    including the finding that blocks it, was made from stored densities.
39. **Is a limited reference-length run required for qualification?** **Yes.** Specified:
    96x12, horizon 3200, reference run separate from measurement, float64 density snapshots at
    every update (~29 MB), frozen P = 100 / L_ref = 500 / ε_ref = 1e-3 / q ∈ {.98,.99,.995}.
    Not to be run before a mass law is settled.
40. **Were topology semantics unchanged?** Yes — definition, `A_sig`, `a_sig_by_mesh`,
    aggregate-island diagnostic role, volume gate, exact-count projection with index
    tie-break: all verified unchanged and independently reproduced.
41. **Were persistence semantics unchanged?** Yes — `P`, OAT levels, `k_enter`/`k_cert`
    definitions, the scan, and the reference rule are byte-identical and were reproduced
    exactly.
42. **Were timing/accounting/scaling methodologies unchanged?** Yes. Raw timing and iteration
    accounting are unaffected; only scaling fits *over `k_enter`/`k_cert`* inherit the
    evaluator dependency through their endpoints.
43. **Which prior results remain valid?** E1 columns everywhere; topology, exact-count and
    hard-gate results; raw timing and per-update cost; iteration accounting; native endpoints
    and native frequencies; the Phase-2A engines; Phase 2C in full; Phase 2B as a historical
    negative result; and all of Phase-2D's stability measurements as raw data.
44. **Which results must be recomputed?** E2/E3 columns of `common_evaluators.csv`; any
    E2/E3-based absolute-quality comparison; the 0.429% agreement figure; scaling fits over
    `k_enter`/`k_cert`. Full dependency chains in
    `RETROSPECTIVE_RESULT_CLASSIFICATION.csv`.
45. **Is the amendment approved for refreeze?** **No.**
46. **What is the exact next action?** Run the mode-1 void-participation screening diagnostic
    (`scripts/wp_spurious*.py`, no optimizer, ~1 hour) over all eight stored Olhoff density
    trajectories, and settle the low-density mass treatment on that evidence before anything
    else. This audit already probed the option space at the worst affected states
    (`REMEDY_FEASIBILITY_PROBE.csv`): Eq. (4) is refuted by Phase 2C, Eq. (4a) by D1,
    Eq. (4b) by D2, and a linear mass law for E2/E3 is refuted outright — spurious at **all
    seven** probed states including the converged one. Only two candidates survive: Eq. (4a)
    with a declared modal-validity rule, or evaluation of E2/E3 on the exact-count binary
    field, which was structural with **0.0000** void participation in every probe. Both are
    real methodology changes, not two-line amendments. Full sequence in
    `POST_AUDIT_ACTION_PLAN.md`.

---

# FINAL VERDICT

    DO NOT REFREEZE — AMENDMENT SCIENTIFICALLY OR TECHNICALLY UNSOUND

Tied to findings: **D1** (CRITICAL, blocks refreeze and production) and **D3** (MAJOR, blocks
refreeze). The reference-length gap is explicitly **not** the reason — it is ruled
non-blocking, and this audit produced the quantitative margin evidence that supports that
ruling. Findings N1–N4, B1, R1 and P1 are corrections required at whatever refreeze
eventually occurs and do not, on their own, block it.

    PRODUCTION STATUS: BLOCKED
