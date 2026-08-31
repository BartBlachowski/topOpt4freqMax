# Delta audit — MAJOR findings recheck

All **8 of 8** original MAJOR findings are **CLOSED**. Verification below goes beyond the
author's ledger to the repaired specification text and, where a factual claim is load-bearing,
to source code or frozen evidence.

| # | Finding | Delta disposition |
|---|---|---|
| M1 | δ_R treated as a minor tolerance | CLOSED |
| M2 | R's structural preferences understated | CLOSED |
| M3 | absolute quality optional / absent from Table 1 | CLOSED (numeric error → N4) |
| M4 | E1 is Proposed's own interpolation | CLOSED (qualification below) |
| M5 | exponents compared on disjoint mesh support | CLOSED |
| M6 | Yuksel Stage-1 cap change undisclosed | CLOSED |
| M7 | false claim that Stage 1 is not carried into Stage 2 | CLOSED (verified in source) |
| M8 | asymmetric method gates not exposed | CLOSED |

---

## M1 — δ_R (D4)

`q ∈ {0.980, 0.990, 0.995}` is declared **co-primary** in `QUALITY_EFFORT_SPEC.md` Sec 1 and
`ACCEPTANCE_GATE_SPEC.md` Sec 4/9, and — importantly — Sec 9 lists the quality levels under
"mandatory disclosures" with the explicit wording that they are *co-primary, not baseline plus
hidden sensitivity*. `SCALING_AND_FIGURE_SPEC.md` Sec 3 requires the three levels reported
together and states that material dependence of `p` on `q` is a result rather than a nuisance.

- **Declared levels / preregistration:** fixed before Phase 2; they are exactly the Phase-1A
  baseline (1%) and its two already-declared sensitivities (0.5%, 2%). They were therefore not
  selected against any observed ranking. Three is defensible as the smallest useful set: two
  levels cannot distinguish a curve from a line.
- **Is one level privileged?** The 99% landmark may anchor prose and compact plots but "may
  never stand alone as the primary conclusion." Acceptable — it is a presentation convenience,
  not an analysis default.
- **Curves primary?** Yes. Figure F1 (quality vs native updates) is designated the primary
  scientific figure; the tabulated landmarks are co-primary companions.
- **Threshold-dependent exponents visible?** Yes — `p(q)` is reported per level and per
  evaluator, with LOO ranges and weak-identification labels.
- **Censored levels visible?** Yes — status precedence and censoring discipline apply
  per (method, mesh, q) cell, and censored points are shown at the observed boundary and
  excluded from fits.
- **Post-hoc selection prevented?** Yes: the levels are frozen and protocol-hashed, all three
  must be reported, and the result firewall forbids a production result from triggering a
  threshold change.

**CLOSED.** The original concern — that the threshold moved the answer more than the method
did, while being presented as a tolerance — is structurally addressed by making the threshold
an axis of the result.

---

## M2 — structural preferences of R

`QUALITY_EFFORT_SPEC.md` Sec 6 now states all three preferences in the estimand definition:
early plateau yields an early crossing; oscillation depresses a window floor relative to
transient peaks; steady late improvement delays crossing and may prevent stabilisation. Sec 5
and figure F1 make the sustained-floor and reference-stability trajectories mandatory, so a
reader can see which regime a method was in.

I verified this is in the normative spec, not only in the response document. The disclosure is
the correct remedy: the preference is intrinsic to a self-referenced estimand and cannot be
removed without abandoning R. The counterweights (mandatory best-observed benchmark, F4
absolute quality) are the same ones M3 installs. **CLOSED.**

---

## M3 — absolute quality (D6)

The mechanism is repaired and inseparable in the paper-facing layer:

- best-observed benchmark **mandatory** (`REFERENCE_QUALITY_SPEC.md` Sec 8);
- Main Table 1 carries E1/E2/E3 quality at entry, ratio to own reference, **and** ratio to
  best-observed, in the same row as the counts;
- Main Table 2 exposes absolute reference quality with its caption stating the frozen gap;
- figure F4 plots absolute reference/endpoint quality vs mesh; F2 carries an in-axes note
  pointing to F3;
- `QUALITY_EFFORT_SPEC.md` Sec 4 prohibits any single-number statement that omits `q`,
  evaluator semantics, enter-vs-cert, and absolute achieved quality.

Applying the test in D6: a headline of the form *"Method X reaches 99% of its endpoint
fastest"* cannot be produced without `q`, evaluator semantics, and the absolute quality in the
same sentence or adjacent panel. **CLOSED as a mechanism.**

**But the disclosure's number is wrong.** Recomputing from
`examples/Performance/final_campaign/common_evaluators.csv`:

| mesh | Olhoff over Proposed | Olhoff over Yuksel |
|---|---:|---:|
| 160x20 | 8.50% | 6.09% |
| 240x30 | 7.60% | 6.39% |
| 320x40 | 7.22% | 5.94% |
| 400x50 | 6.90% | 6.59% |
| 480x60 | 6.23% | 6.04% |
| 560x70 | 6.56% | 6.82% |
| 640x80 | 7.28% | 7.38% |
| 720x90 | 7.45% | 7.71% |
| 800x100 | — | — |

Actual: **6.2%–8.5%** over Proposed and **5.9%–7.7%** over Yuksel, across **eight** meshes —
the Olhoff 800x100 row is `RUN_ERROR` with `omega1_common_raw_E1 = N/A`. The package repeats
"6.1–7.2% ... at every one of the nine frozen meshes" in at least four normative documents.
Neither normalisation nor any status filter reproduces that range (`in_scaling_fit=yes` gives
6.90%–8.50% on five meshes).

The error understates the maximum gap by ~1.3 percentage points — i.e. it weakens the very
safeguard M3 exists to install. It originates in **my own original M3 wording**
(`FINDINGS.csv`) and the author propagated it in good faith. Raised as **N4 (MODERATE,
non-blocking)** with the corrected values.

Also confirmed while recomputing: the ordering Olhoff > {Yuksel, Proposed} holds under E1, E2
and E3 at every mesh, and Proposed and Yuksel swap places at 560x70 and above.

---

## M4 — evaluator neutrality (D5)

I re-verified the evaluator identities directly in
`analysis/three_method_parametric_study/study_evaluate_design.m`:

```
E1: Ee = 1e7*(1e-6 + (1-1e-6)*z^3);  rr = 1e-6 + (1-1e-6)*z         <- LINEAR mass  (Proposed)
E2: Ee = 1e7*(1e-9 + (1-1e-9)*z^3);  rr = 1e-9 + (1-1e-9)*g, g=z^6 for z<=0.1  (Yuksel)
E3: z3 = max(z,1e-3); Ee = 1e7*z3^3; rr = g(z3), g=z3^6 for z3<=0.1            (Olhoff)
```

- **No native interpolation treated as universal truth?** Correct. `QUALITY_EFFORT_SPEC.md`
  Sec 2 and `ACCEPTANCE_GATE_SPEC.md` Sec 3 state that E1 *is* Proposed's model and withdraw
  the neutrality claim rather than defending it with provenance.
- **E1/E2/E3 individually visible?** Yes — co-equal primary decompositions in Main Table 1 /
  Table 1B and Supplement S1, with the explicit rule that no evaluator may be called a
  sensitivity.
- **Robust quantity well defined?** Yes: `r_e(k) = Q_e(k)/Q_ref_e`, `r_all(k) = min_e r_e(k)`,
  `S_q = [r_all >= q]`. Dimensionless, and exactly equivalent to requiring all three
  evaluator-specific thresholds.
- **Symmetric?** Yes. Critically, the author considered and **rejected** the absolute-units
  minimum `min(E1,E2,E3)` that my original finding floated as option B, on the correct ground
  that differing level offsets would let one model dominate at every state — reintroducing a
  privileged evaluator through the back door. The normalised minimum has no such degeneracy.
- **New pessimism or favouritism?** The gate is uniformly conservative — it can only delay
  entry relative to any single evaluator — and applies identically to all methods, so it
  shifts no ranking. Reference freezing likewise waits for the slowest-stabilising evaluator,
  which is symmetric.
- **Mitigating fact verified:** maximum E1/E2/E3 spread on a frozen endpoint is **0.429%**
  with ordering preserved at every mesh — matching my original ~0.43%.

**Non-blocking qualification (no rule change requested):** E2 and E3 share the *same*
piecewise `x^6` mass law and differ only in the stiffness floor (1e-9 vs clipping at 1e-3).
E1, with linear mass, is the structural outlier. The three-way minimum is therefore closer to
two-way in practice, so "three independent models agree" overstates the evidential
independence. This is a property of the frozen evaluators, not of the repair, and every
evaluator is reported individually. It should be stated once where the 0.429% agreement is
quoted. **CLOSED.**

---

## M5 — scaling on disjoint support

`SCALING_AND_FIGURE_SPEC.md` Sec 4 requires a **common-support companion fit** for every
per-method full-range fit, restricted to meshes valid for every compared method at the same
`q`/evaluator/quantity; prohibits cross-method exponent comparison outside common support; and
requires an explicit statement that no comparative exponent exists when common support has
fewer than three meshes. Supplement S2 places full-range and common-support fits in adjacent
rows with a support label.

This is the exact correction requested. Note the practical consequence, correctly accepted by
the author: with censoring at the tighter `q` levels, common support may be small or empty and
the study then reports **no** comparative exponent. That is the right outcome. **CLOSED.**

---

## M6 — Yuksel Stage-1 cap change

Disclosed at both points of use — `ITERATION_ACCOUNTING_SPEC.md` (Stage 2 section) and
`IMPLEMENTATION_REQUIREMENTS.md` Sec 3.3, which is titled as a disclosed non-neutral budget
change. Both name 640x80, 720x90, 800x100 and state that `N_stage1`, the Stage-2 trajectory
and the chronological total are **not comparable** to frozen campaign values there. Figure F6
marks the three meshes. The author also records why retaining the binding 1000 cap was
rejected (it would make Stage-1 reference stabilisation unreachable by construction), which is
a sound reason I had not supplied. **CLOSED.**

---

## M7 — Yuksel factual error (D7)

Verified in source, `analysis/YukselApproach/Matlab/top99neo_inertial_freq.m`:

```
% Use stage-1 outputs as initial guesses for stage 2
x = xPhys;                                    <- line 237
U_est = U;
[xPhys_stage2,U_stage2] = deal(xPhys, U);
```

The Stage-1 filtered physical field **is** carried into Stage 2 as its design variable, and
the Stage-1 displacement becomes the initial mode estimate. `ITERATION_ACCOUNTING_SPEC.md`
now states exactly this, notes the one-time re-filtering shift in raw/physical identification,
and **deletes** the false rationale rather than softening it.

Stage-2-only eligibility now rests solely on objective mismatch (Stage 1 optimises point-load
compliance, not the inertial/eigenfrequency objective) — a valid reason, retained in
`ACCEPTANCE_GATE_SPEC.md` Sec 2 and `QUALITY_EFFORT_SPEC.md` Sec 8. Stage 1 is separately
reported for distinct algorithmic role, objective/update regime, timing, and transparency:
all valid, none dependent on the withdrawn claim.

Definitions verified: `N_stage1`, `N_stage2_to_e = s_e`, `N_total_to_e = N_stage1 + s_e`, with
all three retained in every table and the sum explicitly qualified as *chronological* work,
not homogeneous units. As the original finding predicted, the correction **strengthens** the
chronological-sum justification, since the density trajectory really is continuous across the
handoff. **CLOSED.**

---

## M8 — asymmetric method-specific gates

`ACCEPTANCE_GATE_SPEC.md` Sec 2 requires the first method-condition satisfaction index
`k_gate` beside every endpoint; `QUALITY_EFFORT_SPEC.md` Sec 8 and
`ITERATION_ACCOUNTING_SPEC.md` repeat it per method and cite the frozen 160x20 Olhoff policy
trigger at outer iteration 245 as the worked example; Main Table 1 and Supplement S3 carry the
column; F1 marks gate satisfaction.

The asymmetry itself is retained, which is what I recommended — imposing symmetry would
redefine the methods. The requirement was to make its magnitude visible, and a per-row
`k_gate` does exactly that. **CLOSED.**
