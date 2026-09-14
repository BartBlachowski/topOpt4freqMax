# REPORT — topology-maturity move transition

**Is `OlhoffCurrent` now ready for the final nine-mesh performance campaign?**

**No.** The task stopped at Phase A, at the brief's own §A10 stop condition, and
it stopped for two independent reasons — one scientific, one evidential.

| | |
|---|---|
| Repository HEAD at task start | `7154d8201e9defb06d0d758da866c3769c07179a` (branch `benchmark-methodology-r2`, clean) |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` — **unchanged by this task** |
| MATLAB | 25.2.0.3042426 (R2025b) Update 1 |
| Phase 0 | **all gates PASS** (see [`PROVENANCE.md`](PROVENANCE.md) §2) |
| Solver runs executed | **0** |
| Production code changed | **none** |

---

## Headline

**1 — The signal does not exist in the family we were told to search.**
At the two states of known and opposite maturity, every mesh-normalised,
outlier-robust candidate returns the *same number*:

| | 160×20 at its descent | 320×40 at its descent |
|---|---|---|
| topology evolution still remaining | **2.20 %** (mature) | **11.61 %** (premature) |
| `RW_end` = ‖ρ(k)−ρ(k−10)‖₁ / (NE·10·move) | 0.026068 | 0.025057 → **ratio 1.04** |
| `DW_unsat` (saturated elements deleted) | 0.010615 | 0.010023 → **ratio 1.06** |
| `med_DW` | 2.82e-5 | 2.47e-5 → **ratio 1.14** |

Every candidate that *does* discriminate reads **larger at the more mature
mesh**. The cause is structural: at `move = 0.04` the two meshes are in different
dynamical regimes — 160×20 sits in a **period-2 limit cycle** (`d₂/d₁ ≈ 0.35`,
`max|Δρ| = 0.0400` at all 600 iterations, 56–250 elements permanently at the
bound), while 320×40 is in **coherent convergent descent** (`d₂/d₁ ≈ 2.0`, zero
elements at the bound after iteration ~100). Residual motion at the coarse mesh
is *large but unproductive*; at the fine mesh it is *small but productive*. A
motion-magnitude scalar must therefore get the ordering backwards, and a
directedness ratio must divide by that magnitude and so inherits the saturated
minority that produces it — failing §A9 and §A10 in turn.

**2 — The mesh the brief calls decisive has no data at all.**
The brief's §A5 states 400×50 has retained raw element trajectories and that
160×20 and 320×40 were lost. The repository states the reverse. 160×20 and
320×40 raw densities are present, manifested and hash-valid (`RHO` 3200×600 and
12800×600). **400×50 has no trajectory, no diagnostic run, and no OlhoffCurrent
production run under the move ladder at any point in this repository's history.**
Consequently brief facts 8, 9, 10 and 13 cannot be sourced, and fact 8 is
contradicted by measurement. Details and the search performed:
[`PROVENANCE.md`](PROVENANCE.md) §3.

Nothing was fabricated to close that gap, and no rule was preregistered on a
statistic already shown to be blind.

---

## The thirty required answers

**1. Was existing evidence sufficient, or were 160/320 trajectories regenerated?**
Sufficient; **nothing was regenerated, and nothing needed to be.** The 160×20 and
320×40 raw trajectories the brief believed lost are present and hash-valid. The
real gap is 400×50, which §A6 does not authorise regenerating and which has never
been run.

**2. What topology-maturity statistic was selected?** **None.**
`TOPOLOGY_MATURITY_SIGNAL_NOT_IDENTIFIED`.

**3. Why is it physically/algorithmically meaningful?** N/A — none selected.

**4. Why is it not merely an objective-progress proxy?** N/A. (All candidates
tested were density-based and none was an objective proxy; that criterion was not
the one that failed.)

**5. Why is it robust to isolated moving elements?** N/A. This was tested
directly: `DW_unsat` deletes the saturated population before averaging and is
robust — but it is also **blind** (ratio 1.06 across opposite maturity states),
so robustness was achieved only by discarding the discrimination.

**6. Why is it mesh-consistent?** N/A — and mesh-consistency is precisely what
failed. At matched ground-truth maturity the mesh ratios swing **37–81×** and
cross 1.0 non-monotonically, so no threshold is even consistently conservative.

**7. Why is it not automatically satisfied by move reduction?** N/A. `RW_end`
passes this test by construction; `coher`, the only candidate that ordered the
states correctly, **fails** it — it collapses from 0.99 to 0.24–0.44 the moment
the ladder descends.

**8. Was W = 10 retained?** **Yes**, throughout, and never tuned. It is the
window already in `olh.move.limit` and in both preceding preregistered studies.

**9. How was the threshold chosen without performance fitting?** It was **not
chosen.** Any threshold on a blind statistic could only have been defended by the
M_nd it produced — the fitting §B1 forbids — so §B1's `THRESHOLD_NOT_PREREGISTRABLE`
condition applies on top of §A10's.

**10. What exact transition rule was preregistered?** **None.** See
[`PREREGISTRATION.md`](PREREGISTRATION.md), which records the absence so it
cannot be back-filled.

**11. Did 160×20 pass?** Not run. Phase C was not entered.
**12. Did 320×40 pass?** Not run.
**13. Did 400×50 pass?** Not run — and could not have been: it has no production
baseline to compare against.

**14. How did first-descent maturity change at each mesh?** Unchanged; no
controller was deployed. Measured *production* first-descent maturity, re-derived
from retained data: 160×20 descends at iteration 79 with **2.20 %** of topology
evolution remaining; 320×40 descends at 130 with **11.61 %** remaining.

**15. How did M_nd change?** Unchanged. The M_nd production *forgoes* by
descending: 160×20 13.390 → 11.458 % (1.93 pts); 320×40 23.322 → 13.304 %
(**10.02 pts**).

**16. How did ω₁ change?** Unchanged. Forgone by descending: **+0.176 %** at
160×20, **+0.370 %** at 320×40.

**17. Was volume feasible?** Yes throughout the retained trajectories analysed
(0.5 to ~1e-6). No new runs were made.

**18. Did any mesh CAP_HIT?** No new runs. In the retained evidence the prior
`max|Δρ|/move` candidate produced `CAP_HIT` at 160×20, and this study identifies
the mechanism: the period-2 limit cycle holds `max|Δρ|` at 0.0400 for all 600
iterations.

**19. Did the new rule avoid isolated-element veto?** N/A — no rule. The finding
is that avoiding the veto (`DW_unsat`, `med_DW`, `RW_end`) costs all the
discrimination.

**20. Did it avoid premature fine-mesh descent?** N/A — no rule. The premature
fine-mesh descent is confirmed and quantified: 10.02 M_nd points at 320×40.

**21. Was the controller VALIDATED?** No. `TOPOLOGY_MATURITY_CONTROLLER_INCONCLUSIVE`
— there was no controller to validate.

**22. If validated, was it promoted cleanly?** N/A — not validated, so Phase D
was not executed and nothing was promoted.

**23. Is historical β-continuation behaviour still reproducible explicitly?**
**Yes, and it is still the production default** — nothing was changed. The
production preset `duOlhoffFixedPenaltySensitivityFiltered` still selects
`move.continuation.signal = 'boundVariable'`, and `test_preset_equivalence`
reproduces the frozen conference realisation **bitwise** (ω₁ = 169.49522702153845,
outer 91, inner 2241).

**24. Does the final performance driver use canonical OlhoffCurrent?**
**Not audited.** §D4 is gated behind Phase D, which is gated behind validation.
Auditing it now would have meant reporting a readiness finding for a
configuration that does not exist.

**25. Is all benchmark raw evidence now durable/manifested?** **Unresolved** —
same gate. Note the pre-existing hazard is real and unaddressed: the diagnostic
`.mat` trajectories are `.gitignore`d, and the *only* thing making them durable is
their appearance in `FINAL_SHA256.txt`. `move_transition`'s own `armP/armU` `.mat`
files are **not** in its manifest (only its CSVs are), so the 160×20/320×40
densities this study depended on are currently protected by nothing but their
presence on this disk.

**26. Are timing semantics still correct?** Not audited — §D4 gate.

**27. Is the nine-mesh campaign authorized?** **No.**

**28. If not, what exact blocker remains?** Four, listed in the next section.

**29. Does anything in this task justify projection?** **No.** The failure is in
the *observability* of stage maturity. Projection changes the formulation and
would not make a blind statistic discriminating; it is also excluded by the hard
scientific lock.

**30. Does anything justify changing R = 0.06·b?** **No.** Nothing in this
analysis bears on filter radius, and it too is under the lock.

---

## Blockers

**B1 — No topology-maturity observable exists in the mandated family.**
Every candidate is blind at the known states (ratios 1.04–1.14), or orders them
backwards, or fails §A9/§A10. Root cause is the regime split in
[`OFFLINE_SIGNAL_ANALYSIS.md`](OFFLINE_SIGNAL_ANALYSIS.md) §5, which is a property
of the dynamics, not of the statistics. *Clearing this requires a different class
of observable. Limit-cycle onset (persistent `d₂/d₁ < 1`) is the natural
hypothesis and was measured: it fires at iteration **80** at 160×20 against a
production descent at **79**, which explains why the coarse-mesh anchor survives
a trigger that is wrong at 320×40 (`d₂/d₁ = 2.005` there when production
descends). It is **untestable on existing data** — the 320×40 fixed-move arm
holds `move = 0.04` only to iteration 214 and is still coherent there. Recorded
as a hypothesis only; not implemented and not preregistered, per §A4 and the
one-candidate-per-task rule.*

**B2 — 400×50 has no evidence base.** No trajectory, no scalar history under the
production ladder, no production baseline. §C6 makes it the decisive mesh; §A6
authorises regenerating only 160×20 and 320×40. *Clearing this requires explicit
authorisation to run 400×50 ARM P under the production preset (~2–3 h wall on
this machine, extrapolating from 320×40's 4831 s / 6460 s at 600 iterations).*

**B3 — Brief facts 8, 9, 10, 13 are unsourced and fact 8 is contradicted.** The
three studies they cite (`move_activity_offline`, `move_activity_400`,
`beta_transition_mechanism`) have never existed in this repository.
*Clearing this requires either producing those studies or withdrawing the facts.*

**B4 — §D4/§D5 performance-driver audit not performed**, being gated behind a
validation that did not occur.

---

## Verdicts

Phase A terminal condition (§A10):

> ### `TOPOLOGY_MATURITY_SIGNAL_NOT_IDENTIFIED`

Phase C verdict — no controller was defined, preregistered or run, so no
evidence bears on one either way:

> ### `TOPOLOGY_MATURITY_CONTROLLER_INCONCLUSIVE`

Promotion verdict: **not applicable** — Phase D is entered only on
`TOPOLOGY_MATURITY_CONTROLLER_VALIDATED`. Nothing was promoted.

> ### `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`

Blocked by **B1** (no validated maturity observable), **B2** (400×50 has no
evidence base), **B3** (unsourced premises), **B4** (driver audit not performed).

---

## Standing state, unchanged

* `BETA_TRANSITION_SIGNAL_STRUCTURALLY_UNSUITABLE` — reaffirmed, and now with a
  measured cost at 320×40: 10.02 M_nd points and 0.370 % ω₁.
* `NEW_TOPOLOGY_MATURITY_SIGNAL_REQUIRED` — reaffirmed, and narrowed: it is
  **not** obtainable from windowed density-motion magnitude.
* `KEEP_CURRENT_MOVE_POLICY_PENDING_REVIEW` — **in force**, unchanged. The β-stall
  trigger remains production, and remains a Class-C reconstruction choice that
  should not be attributed to Du–Olhoff.

The negative result is preserved. No Candidate 2 was invented.
