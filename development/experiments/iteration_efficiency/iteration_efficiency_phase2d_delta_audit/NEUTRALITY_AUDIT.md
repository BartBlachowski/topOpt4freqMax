# WP16 — Evaluator neutrality under the amendment
READ_ONLY_INDEPENDENT_DELTA_AUDIT

## The replacement argument Phase 2D offers

Native identity is genuinely lost, and Phase 2D says so plainly rather than concealing it.
Neutrality is re-based on four properties. Assessed one by one:

| # | claimed property | verdict | evidence |
|---|---|---|---|
| 1 | **Identical treatment** — same evaluator family, same code path, every density field | **HOLDS** | one file, one `solve_modes`, no method argument anywhere in the call path |
| 2 | **Producer-independence** — no input depends on method identity, run order or provenance | **HOLDS** | `study_evaluate_design(x, nelx, nely, volfrac)`; there is no other input |
| 3 | **Stability at the decision scale** | **HOLDS for perturbation; FAILS for level validity** | see below |
| 4 | **Preserved multi-model perspective** | **QUANTITATIVELY MUCH WEAKER THAN DISCLOSED** | see below |

Properties 1 and 2 are the right substitutes for native identity and they genuinely hold.
The trade Phase 2D articulates — that a common *measuring instrument* should prize stable,
method-independent scoring over literal reproduction of each contestant's internal numerical
device — is sound in principle.

## Property 3, split into its two halves

**Perturbation stability: achieved.** Nearly identical density fields now receive nearly
identical scores. Independently reproduced: branch-side sensitivity on 1600 production states
falls from 2.6496e-02 to 2.6560e-10, matching the branch-free E1 control at 2.6252e-10; the
binding evaluator no longer changes with branch side (0/1600 against 150/1600).

**Level validity: not achieved.** Producer-independence guarantees that the *same* field gets
the *same* score. It does not guarantee that the score is the quantity the study claims to
measure. At 34 states of the 160x20 production trajectory (k = 237…272), 27 states of the
240x30 trajectory and 11 of 320x40, the amended E2/E3 `omega_raw(1)` is a **spurious
localized void mode**, not the structural first eigenfrequency — 100% of its modal kinetic
energy sits in elements with ρ ≤ 0.1, and its value falls to as little as **18.9%** of the
structural value. Under Eq. (4) the same states were clean. See finding **D1**.

This is a neutrality problem as well as a validity problem. The incidence of the artefact
depends on **how much low-density gray area a design carries at a given iteration**, and that
is a property of the optimizer's update law, not of the design's quality. Olhoff's
fixed-horizon LP updates, Proposed's OC updates and Yuksel's two-stage scheme traverse that
regime differently. The amended evaluator therefore penalises methods in proportion to how
long they linger in a regime the instrument mishandles. No Proposed or Yuksel density
trajectory exists in the repository, so this bias **cannot be measured** — the same
limitation Phase 2D records, now with a concrete reason to care about it.

## Property 4 — how close is "closer to two-way"?

The frozen specifications disclose that E2 and E3 "share the same piecewise mass law and
differ only in stiffness floor, so the three-evaluator minimum is closer to two-way in
evidential terms". Measured over 1600 production states:

| pair | law | median relative difference | p95 | max |
|---|---|---|---|---|
| E2 vs E3 | Eq. (4) | **5.184e-09** | 4.995e-08 | 2.533e-07 |
| E2 vs E3 | Eq. (4a) | **8.621e-09** | 7.763e-08 | 2.929e-01 † |
| E1 vs E2 | Eq. (4) | 3.099e-03 | 4.133e-03 | 2.520e-02 |
| E1 vs E2 | Eq. (4a) | 1.975e-03 | 2.628e-03 | 8.107e-01 † |

† the maxima are driven entirely by the spurious-mode block; outside it the p95 figures are
representative.

E2 and E3 agree to **5 parts in 10⁹** — six orders of magnitude tighter than the tightest
0.5% acceptance band. At the measurement scale the three-evaluator minimum is not "closer to
two-way": it **is** two-way. That E2 and E3 nevertheless swap which of them binds the minimum
(E2 binds 20 of 3200 states, E3 485, in the 96x12 case) is a consequence of the degeneracy,
not evidence against it — the argmin between two quantities differing by 5e-09 is arbitrary.

This is a **pre-existing** condition. The amendment neither creates it (median 5.2e-09 →
8.6e-09) nor cures it. It is recorded here because WP16 requires the limitation to remain
visible, and the current qualitative wording understates it by six orders of magnitude.

## Ruling

    AMENDED_NEUTRALITY_ARGUMENT = FAIL

Not because the *structure* of the replacement argument is wrong — properties 1 and 2 are
correctly chosen and hold — but because:

1. **A method-independent mapping is necessary, not sufficient.** Applying the same
   instrument identically to everyone is not neutral if the instrument stops measuring the
   estimand on a class of states, and the frequency of those states is method-dependent.
2. The artefact's method-dependence cannot be quantified on available evidence, and Phase 2D
   did not identify the artefact at all, so its own neutrality section does not address it.
3. Property 4 as stated is quantitatively misleading and should be replaced with the measured
   figure at refreeze.

**Required disclosure regardless of which mass law is finally adopted:** the measured E2–E3
degeneracy (median 5e-09) belongs in `QUALITY_EFFORT_SPEC.md` §2 and in the F01 row of
`FAIRNESS_RISK_REGISTER.md`, in place of the present qualitative wording.
