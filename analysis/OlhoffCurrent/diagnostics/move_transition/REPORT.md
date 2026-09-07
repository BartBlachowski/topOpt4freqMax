# REPORT — move-stage transition signal

Does the move ladder descend because the **design** has finished exploiting its
current move level, or merely because a bound-variable signal has stalled while
the topology is still reorganizing?

| | |
|---|---|
| Implementation | `analysis/OlhoffCurrent` (sole production Olhoff) |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` (74 source files) — **unchanged by this study** |
| Repository HEAD at freeze | `a1f2c6c4dfae8d48ee76ba30cd735bf7a776d3c1` (branch `benchmark-methodology-r2`) |
| Promoted upstream commit | `695f03bdac20c423a4e1d389cf9db9187597bcc3` · currentness `CURRENT` |
| Production preset | `duOlhoffFixedPenaltySensitivityFiltered` |
| cfg hash 160×20 / 320×40 (both arms) | `0c9482af58007919…` / `d1953405fc75c0d3…` |
| MATLAB | R2025b Update 1, **1 computational thread** |
| Preregistration | [`PREREGISTRATION.md`](PREREGISTRATION.md), SHA-256 `25b0e04db7f6e17a509ad3f5bcdb301c0b6a9d0feb4c4017724907a98d5b3102`, frozen 2026-09-07T11:39:22Z **before the first candidate run** |
| Provenance | [`PROVENANCE_LEDGER.md`](PROVENANCE_LEDGER.md) |

Path gate passed before every run: `CURRENT`, 74/74 source files, 32 owned
symbols resolving inside `+impl/`, no competing Olhoff tree on the path,
published Svanberg MMA and the Sigmund sensitivity filter both winning
resolution. Re-verified **after** the runs: the manifest still hashes unchanged.

---

## Headline

**The production move transition is premature — demonstrably, at both meshes.**
When production descends from `move = 0.04`, the design is still consuming
**99.9 %** of its permitted step at 160×20 and **65–81 %** at 320×40. It is not
descending because the design has finished with the step; it is descending
because a *different* signal — the bound variable β — has stopped improving.

**Blocking that descent removes essentially all of the fine-mesh grayness.** At
320×40 the utilization-gated candidate terminates naturally at iteration 210
with `M_nd = 13.37 %` against production's `23.36 %` — a **42.8 % relative
reduction**, which is **99.1 % of the entire excess** over the historical
fixed-move reference — while ω₁ *improves* by 0.38 % and the mid-density
fraction falls by 67 %.

**And the candidate statistic is nevertheless not usable.** At 160×20 it never
fires: over 600 iterations `max|Δρ|/move` never once falls below 0.5 — its
*minimum* is 0.9976 — so the ladder never descends and the run is `CAP_HIT`.
The preregistered §12 failure mode occurred, and the spatial measurement
confirms its anticipated cause exactly: **66 elements out of 3200 (2.1 %) sit at
the move bound while 90 % of the design moves less than 3.8 % of it.** The
maximum is being pinned by a small saturated minority long after the bulk
topology has matured.

Per §12 the candidate was **not rescued**: 0.5 was not raised, 10 was not
shortened, and no percentile, RMS or robust statistic was substituted.

---

## §8 Baseline regression gate — PASS at both meshes, bitwise

ARM P was run through the candidate solver copy with the candidate **off**. It
must reproduce the archive; it does, exactly.

| | 160×20 | 320×40 |
|---|---|---|
| **B-1** 12 history fields × 600 iterations vs archived unstopped | **bitwise** (0 mismatched fields) | **bitwise** (0 mismatched fields) |
| **B-2** design at the archived production stop vs archived final | **bitwise** (max\|Δ\| = **0**) | **bitwise** (max\|Δ\| = **0**) |
| **B-3** post-loop ω₁ recomputed at that design | **169.49522702153845** | **165.95078925220545** |
| archived ω₁ | 169.49522702153845 | 165.95078925220545 |
| **B-4** production move transitions | **{79, 90, 101}** ✓ | **{130, 141, 152}** ✓ |
| production admission replay | iteration **91** ✓ | iteration **131** ✓ |
| `settledLocalObjective` replay | iteration **100** ✓ | iteration **140** ✓ |

The last row independently reproduces the preceding admission-rule study's C1
result at both meshes. **`BASELINE_REPRODUCTION_FAIL` did not occur.**

### A second, independent verification the study did not have to run

ARM U at stage 1 holds `move ≡ 0.04`, which is exactly the archived fixed-move
diagnostic arm. This was used as a **pre-registered prediction**: from the
archive alone, ARM U at 320×40 had to descend at iteration **214** with
`M_nd = 13.3182` immediately before, and ARM U at 160×20 had to make no descent
at all. Both predictions were then confirmed by the live runs to the digit, and
ARM U 160×20 is **bitwise identical to the archived fixed-move arm over its full
400 iterations**. The candidate controller does what it is specified to do.

---

## §9 Was ARM U a genuine single-factor experiment?

**Yes, and in the strongest available sense: the field-level configuration diff
between ARM P and ARM U is empty.**

Both arms resolve to the *same* scientific configuration — identical cfg hashes
(`0c9482af…` at 160×20, `d1953405…` at 320×40) over all 80 schema fields, with
only the free-text `runtime.name` differing. The experimental factor is not a
configuration field at all: it is the move-transition controller handed to the
solver.

Versus production, both arms differ in exactly the two declared stopping-policy
fields that unstop the run (`stop.tolerance 0.05→0`, `stop.toleranceRule
meshScaled→explicit`) plus the run name. Nothing else — asserted field-by-field
over all 80 fields at run time, plus an explicit re-assertion of the scope lock
on `p = 3`, `eq4b`, `q = 1`, sensitivity filter, `R = 0.06·b`, projection off,
ladder policy, published MMA, subspace size 2, off-diagonal terms, and the move
levels `[0.04 0.02 0.01 0.005]`.

**No production file was modified.** The candidate is default-off:
`mt_moveLimit` with `metric = 'boundVariableStall'` delegates verbatim to
`olh.move.limit`. The solver copy `mt_olhoffSolveT.m` differs from
`+impl/architecture/olhoffSolve.m` in **exactly two lines** — the function
signature and the one call to the move controller — and that claim is not
asserted but *proved* by the §8 bitwise gate.

---

## The key measurement (§14): M_nd immediately before the first descent

| | production first descent | ARM U first descent | historical fixed-move mature |
|---|---|---|---|
| **160×20** | iteration **79**, `M_nd = 13.494` | **never** (600 iterations) | 12.275 @ 400 (`CAP_HIT`) |
| **320×40** | iteration **130**, `M_nd = 23.451` | iteration **214**, `M_nd = 13.318` | 13.282 @ 216 |

At 320×40 the candidate delays the first descent by **84 iterations** and, in
that time, M_nd falls from 23.45 to 13.32 — arriving essentially *at* the
historical fixed-move mature value (13.28) before the ladder is allowed to move.

`r_rho` over the ten iterations preceding each first descent:

| | preceding `r_rho` | all < 0.5? |
|---|---|---|
| 160×20 production, iter 79 | 1.000 0.999 0.999 0.999 0.999 0.999 0.999 1.000 0.999 1.000 | **no** |
| 320×40 production, iter 130 | 0.746 0.798 0.749 0.702 0.759 0.806 0.729 0.726 0.770 0.647 | **no** |
| 320×40 ARM U, iter 214 | 0.383 0.361 0.306 0.355 0.279 0.368 0.245 0.232 0.341 0.238 | **yes** (count = 10) |

---

## §15 Stage-by-stage analysis

**160×20**

| arm | stage | move | first–last | n | M_nd start→end | Δ | ω₁ start→end | `r_rho` med / min / max | mature at descent? |
|---|---|---|---|---|---|---|---|---|---|
| P | 1 | 0.04 | 1–78 | 78 | 99.459 → 13.494 | −85.97 | 68.40 → 169.09 | 0.9993 / 0.9976 / 0.9998 | **no** |
| P | 2 | 0.02 | 79–89 | 11 | 13.444 → 13.361 | −0.083 | 169.28 → 169.37 | 0.9963 / 0.9924 / 0.9988 | **no** |
| P | 3 | 0.01 | 90–91 | 2 | 13.386 → 13.403 | +0.017 | 169.52 → 169.48 | 0.7534 / 0.5810 / 0.9258 | **no** |
| U | 1 | 0.04 | 1–600 | 600 | 99.459 → 11.458 | −88.00 | 68.40 → 169.57 | 0.9993 / 0.9976 / 1.0000 | never descends |

**320×40**

| arm | stage | move | first–last | n | M_nd start→end | Δ | ω₁ start→end | `r_rho` med / min / max | mature at descent? |
|---|---|---|---|---|---|---|---|---|---|
| P | 1 | 0.04 | 1–129 | 129 | 99.614 → 23.451 | −76.16 | 68.28 → 165.92 | 0.9870 / 0.4141 / 0.9994 | **no** |
| P | 2 | 0.02 | 130–131 | 2 | 23.401 → 23.360 | −0.041 | 165.94 → 165.95 | 0.3986 / 0.3877 / 0.4094 | — (run ends) |
| U | 1 | 0.04 | 1–210 | 210 | 99.614 → 13.371 | −86.24 | 68.28 → 166.59 | 0.9429 / 0.2455 / 0.9994 | **yes** — terminates here |

ARM U 320×40, **unstopped continuation past its own admission point** (evidence
that the ladder does execute cleanly once gated):

| stage | move | first–last | n | M_nd start→end | Δ | `r_rho` med / min / max |
|---|---|---|---|---|---|---|
| 1 | 0.04 | 1–213 | 213 | 99.614 → 13.318 | −86.30 | 0.9399 / 0.2321 / 0.9994 |
| 2 | 0.02 | 214–223 | 10 | 13.305 → 13.257 | −0.047 | 0.1761 / 0.1230 / 0.3248 |
| 3 | 0.01 | 224–233 | 10 | 13.253 → 13.216 | −0.037 | 0.0614 / 0.0475 / 0.0640 |
| 4 | 0.005 | 234–600 | 367 | 13.222 → 13.122 | −0.100 | 0.0256 / 0.0194 / 0.0343 |

**Answer to "was the stage actually mature when descent occurred?"** For
production: **no**, at every descent, at both meshes — `r_rho` medians of
0.999 / 0.996 / 0.753 (160×20) and 0.987 / 0.399 (320×40) at the stages it left.
For ARM U at 320×40: **yes**, by construction, and the design confirms it — the
three post-admission stages between them move M_nd by only 0.18 points, i.e.
once utilization has genuinely fallen there is almost nothing left for the finer
levels to do.

---

## §16 Hard primary gates

**160×20 ARM U** — G1 ✓ (no solver failure; all inner solves converged; only the
pre-existing `omega_J (J=3) is itself multiple` notices at iterations 8 and 12,
identical to the archived baseline) · G2 ✓ (`mean ρ = 0.499991`) · G3 ✓
(ω₁ **+0.0757 %**, an improvement) · G4 ✓ (see `fig1`) · G5 ✓ **vacuously — no
descent occurred** · G6 ✓ · G7 ✓ (**`CAP_HIT`, reported as `CAP_HIT`**).

**320×40 ARM U** — G8 ✓ · G9 ✓ (`mean ρ = 0.499999`) · G10 ✓ (ω₁ **+0.379 %**) ·
G11 ✓ (every descent had `count = 10` and all ten preceding `r_rho < 0.5`) ·
G12 ✓ (no descent precedes the admission point at all) · **G13 ✓
(−42.76 %, against the ≥ 25 % preregistered bar)** · **G14 ✓ (mid-density
−67.48 %)** · G15 ✓.

Every gate passes. **G5 passing vacuously at 160×20 is the whole problem**, and
is reported as such rather than as a success.

---

## The twenty-two questions

**1. Did the production baseline reproduce?** Yes — bitwise at both meshes on
all four B-gates, including a max design difference of exactly 0 and post-loop
ω₁ agreeing to all 17 digits.

**2. Which parts of the move ladder are publication-supported?** Only that
*some* bound on Δρ is defensible, and that is **class B via lineage, not text**
(Krog & Olhoff's first-order directional expansion; Bendsøe & Sigmund's move
limit as an efficiency device). `move limit`, `trust region`, `step size` and
`continuation` occur **0 times** in Du & Olhoff (2007) and **0 times** in Olhoff
& Du (2014) — recounted from the PDFs for this study. The only printed bound on
Δρ is the box (25f).

**3. Which parts are reconstruction choices?** The ladder's existence as a
*schedule*, the four levels `[0.04 0.02 0.01 0.005]`, the continuation itself,
the bound-variable stall trigger, the window 10 and the tolerance 5e-3 — **all
class C**, all with no numeric counterpart anywhere in the lineage. The
`settledMove` guard is class C and exists only because the ladder does. The
candidate rule is **EXPERIMENTAL**. Bendsøe & Sigmund's own caution is worth
recording: *"it is unclear whether 'playing' with the move-limits will
jeopardize convergence of the algorithm."*

**4. Was ARM U a genuine single-factor transition-policy experiment?** Yes —
identical cfg hashes; the empty field-level diff is documented in
`CONFIG_DIFF.json`. See §9 above.

**5. At 160×20, did `r_rho < 0.5` persist for 10 iterations?** **No — not once,
in 600 iterations.** `r_rho` min 0.997645, median 0.999319, max 0.999995;
fraction of iterations below 0.5 = **0.0000**; the persistence counter never
reached even 1.

**6. At 320×40, did it?** **Yes, three times** — at iterations 214, 224 and 234,
each with the counter at exactly 10 and all ten preceding values below 0.5.

**7. Did isolated bound-saturated elements prevent transition?** **Yes, at
160×20, and the measurement is unambiguous.** At iteration 600 the per-element
utilization `|Δρ_e|/move` has median **0.0008**, 90th percentile **0.0379**,
99th percentile **0.9960**, maximum **0.9993**. **66 elements of 3200 (2.06 %)**
sit at the bound (`≥ 0.99·move`); 85 (2.66 %) exceed half of it. The median
across the whole run is 2.44 % of elements at the bound. So the bulk topology
has matured — 90 % of the design moves under 4 % of its allowed step — while a
~2 % minority holds the maximum at ~1.0 indefinitely. This is precisely the
failure mode §12 anticipated. **No robust statistic was designed in response**;
that is a future task.

**8. When did the first descent occur, production vs ARM U?** 160×20:
production at **79**, ARM U **never**. 320×40: production at **130**, ARM U at
**214** — a delay of 84 iterations.

**9. M_nd immediately before each first descent?** 160×20: production
**13.494**; ARM U n/a. 320×40: production **23.451**; ARM U **13.318**.

**10. How much M_nd reduction occurred while move remained 0.04?** 320×40:
99.614 → 13.318 over 213 iterations (**−86.30 points**), versus production's
99.614 → 23.451 over 129 iterations (−76.16). The extra 84 iterations at
`move = 0.04` are worth **10.13 points of M_nd**. 160×20: 99.459 → 11.458 over
600 iterations (−88.00).

**11. What happened to mid-density fraction?** It improved substantially.
320×40: 0.09563 → **0.03109** (**−67.5 %**). 160×20: 0.02500 → **0.01188**
(−52.5 %). G14 passes with a wide margin at both.

**12. What happened to ω₁?** It improved at both meshes. Post-loop, 320×40:
165.95079 → **166.57995** (**+0.379 %**); 160×20: 169.49523 → **169.62359**
(+0.0757 %). Nothing was traded away for the discreteness.

**13. Was volume feasible?** Yes. `mean ρ` = 0.499991 (160×20) and 0.499999
(320×40); both inside the `|mean − 0.5| ≤ 1e-3` gate by two to three orders of
magnitude.

**14. Did the complete ladder execute?** 320×40: **yes in the unstopped
continuation** — all four stages, descents at 214/224/234, reaching
`move = 0.005`. But **not before ARM U's own termination at 210**. 160×20:
**no** — the ladder never left stage 1.

**15. Did ARM U terminate naturally?** 320×40: **yes**, at iteration 210 under
`settledLocalObjective`. 160×20: **no — `CAP_HIT` at 600**, and it stays
`CAP_HIT`. Its M_nd of 11.46 is *lower* than production's 13.40, but the run
never satisfied a convergence criterion and that number must not be read as a
converged result.

**16. Was termination credible under `settledLocalObjective`?** At 320×40, yes,
on all four components at iteration 210: A dwell = 209 iterations since the last
move change (≫ 10); B1 `max|Δρ| = 0.00982 < 0.01`; B2 `r_rho = 0.2455 < 0.50`;
C ω₁ relative range over 10 iterations well inside 5e-3. There is **no preceding
descent at all**, so the old immediate-post-descent artifact is structurally
impossible here.

**17. Did the candidate materially reduce 320×40 grayness?** **Yes,
decisively.** `M_nd` 23.3596 → 13.3707, a **42.76 %** relative reduction against
the preregistered 25 % bar. Measured against the historical fixed-move reference
of 13.2824, the candidate removes **99.1 % of the entire production excess**.
For contrast, the preceding admission-rule study achieved 1.5 % on the same
quantity.

**18. Does the evidence show the ladder descended prematurely?** **Yes, and this
is the study's firmest finding.** Production leaves `move = 0.04` while the
design is consuming 99.9 % (160×20) and 65–81 % (320×40) of its permitted step.
The bound variable β had stalled; the design had not. Allowing the design to
finish at 0.04 is worth 10.13 M_nd points at 320×40.

**19. Or is max-utilization too sensitive as a transition statistic?** **Also
yes, at 160×20** — and the two findings are not in conflict. The trigger being
*wrong* and the proposed replacement being *unusable* are independent facts.
The replacement fails for two distinct reasons. First, it is a **maximum** over
elements, and a ~2 % saturated minority pins it near 1.0 forever. Second — an
unplanned finding, see the section above — its move-invariance is exact only
while the design is fully saturated; once partially released, a descent itself
depresses the statistic, so the rule becomes partially self-fulfilling and ARM
U's later descents fall at the minimum spacing the persistence rule permits.

**20. Is 400×50 authorized?** **No.** §18 authorizes it only if ARM U passes at
**both** primary meshes. At 160×20 the candidate never fires and terminates
`CAP_HIT`, so the confirmatory mesh was **not run**.

**21. Should anything be promoted to production now?** **No.** The candidate
degenerates to a fixed move at 160×20 and inherits exactly the coarse-mesh
`CAP_HIT` the brief warned about in §7. No production file was changed.

**22. Is projection / filter-radius work justified yet?** **No, and it is now
clearly premature.** The candidate reduced 320×40 `M_nd` to 13.37 — within
0.09 points of the fixed-move reference — using nothing but a transition-timing
change. The refinement-dependent excess was **not** a discreteness-mechanism
deficiency; it was step-schedule timing. Whatever floor remains at ~13 % is a
separate question, and it stays confounded with the class-C, never-published
`R = 0.06·b`. The move/termination machinery is **not yet settled** — question 19
is open — so that work comes later.

---

## An unplanned finding: `r_rho`'s move-invariance is exact only when saturated

The preregistration argued analytically (and `PROVENANCE_LEDGER.md` §5 derives)
that `r_rho = min(s/move, 1)` is invariant under a change of the move bound,
where `s` is the step the sub-problem would take if unbounded. That argument is
correct **for a fixed sub-problem**. The trajectories show that the sub-problem
is not fixed: `innerLoop` initialises MMA's moving asymptotes *at the move box*
(`low = xmin`, `upp = xmax`), so halving the move also halves the asymptote
spread, making the convex approximation more conservative and shrinking `s`
itself.

Measured across the production descents, three iterations either side:

| descent | move | `r_rho` before | `r_rho` after | `max|Δρ|` before → after | invariance |
|---|---|---|---|---|---|
| 160×20 @ 79 | 0.04→0.02 | 1.000 0.999 1.000 | 0.999 0.997 0.997 | 0.03999 → 0.01998 | **exact** |
| 160×20 @ 90 | 0.02→0.01 | 0.997 0.995 0.997 | 0.926 0.581 0.446 | 0.01994 → 0.00926 | broken |
| 320×40 @ 130 | 0.04→0.02 | 0.726 0.770 0.647 | 0.388 0.409 0.409 | 0.02588 → 0.00775 | broken |
| 320×40 @ 141 | 0.02→0.01 | 0.401 0.395 0.393 | 0.317 0.277 0.245 | 0.00786 → 0.00317 | partial |

So:

* **In the saturated regime (`r_rho ≈ 1`) the invariance holds exactly**, and
  that is the regime it was introduced to protect. At the 160×20 descent from
  0.04 to 0.02, `max|Δρ|` halves precisely with the move and `r_rho` does not
  move. A bound-saturated design cannot be made to look settled by shrinking its
  bound. The preceding admission-rule study's use of the ratio as an *admission*
  guard is therefore sound.
* **In the partially-saturated regime it is only approximate.** At the 320×40
  descent from 0.04 to 0.02, `max|Δρ|` fell by 3.3× while the move fell by 2×,
  so `r_rho` dropped from ~0.65–0.77 to ~0.39 as a *consequence of the descent*.

**Why this matters for a transition rule.** Used as a *transition* statistic
rather than an admission guard, `r_rho` is partially self-fulfilling once
descents begin: each descent depresses the statistic and so makes the next
descent easier to justify. ARM U's own gated descents at 320×40 show the
signature — 214, 224, 234, spaced exactly 10 apart, i.e. **the minimum the
persistence rule allows**, with stage `r_rho` medians collapsing 0.176 → 0.061 →
0.026. After the first genuinely-earned descent, the remaining three were
effectively automatic.

This does not weaken the study's central finding, which rests entirely on the
`move = 0.04` regime *before* any descent, where the arms are exactly
comparable. It does mean the ratio is a weaker measure of "design utilization"
than the analytic argument alone suggests, and any future transition statistic
must be checked against this effect rather than assumed immune to it.

## What ARM U's 320×40 result is, and what it is not

This must be stated plainly, because the headline number invites over-reading.

ARM U's converged design at 320×40 was reached at iteration **210**, and its
first descent would not have come until **214**. **The reported ARM U design is
therefore a `move = 0.04` design that never descended at all.** It is — bitwise,
up to iteration 213 — the archived fixed-move trajectory, terminated by
`settledLocalObjective` instead of by the production rule.

So the 320×40 result does **not** demonstrate that a utilization-gated *ladder*
works. It demonstrates that **not descending** works, which the preceding
move/stop diagnostic had already established, and that a principled,
non-fitted, preregistered rule reproduces that benefit and terminates
credibly — where the raw fixed-move arm at 160×20 could not terminate at all,
and neither can ARM U.

The unstopped continuation does show the gated ladder executing cleanly once
utilization has fallen (all four stages, every descent legal, no pathology), but
those descents lie past the point where the candidate says the run is finished.

---

## Figures

`fig1_topology_comparison` — 160×20 and 320×40, production vs candidate. The
320×40 pair is the visual statement of the whole study: production carries large
gray blobs at both supports; the candidate is crisp, with the same member
layout. `fig2_Mnd_vs_iteration` · `fig3_omega1_vs_iteration` ·
`fig4_rrho_vs_iteration` (threshold 0.5 drawn; the 160×20 panel shows the
candidate pinned at ~1.0 for 600 iterations) · `fig5_persistence_vs_iteration`
(persistence 10 drawn) · `fig6_move_vs_iteration` ·
`fig7_mid_vs_iteration` · `fig8_stage_Mnd_reduction`. Move descents are marked
on every per-iteration panel, admission points are drawn as dashed verticals,
and the historical fixed-move arm is overlaid as evidence where meaningful.

## Data

`METRICS.json` (runs, baseline gate, per-descent records, stage tables, gates,
spatial utilization distributions); `CONFIG_DIFF.json`;
`runs/arm{P,U}_{160x20,320x40}_iterations.csv` — 27 columns each, covering every
quantity §13 requires, including `r_rho`, the persistence counter, the existing
bound-variable stall signal and its relative measure, ω₁ stability, inner
iteration counts and status, and multiplicity state. `runs/*.mat` is not tracked
(large, reproducible from `code/`).

## Deviations from the preregistration

1. **`mt_spatial` was added after the runs.** It only *measures* the
   distribution of per-element utilization; it defines no alternative statistic
   and none is proposed. It was needed to answer report question 7 honestly
   rather than by assertion.
1b. **The invariance limit was not anticipated.** The preregistration asserted
   `r_rho`'s move-invariance analytically; the trajectories show it is exact only
   in the saturated regime. This is reported as a finding, not corrected away,
   and no threshold was changed in response.
2. **No 400×50** — correctly withheld under §18, since ARM U does not pass at
   160×20.
3. No threshold was changed at any point. `0.5` and `10` are exactly as frozen.
4. ω₁ gates use the post-loop value (the solver's own terminal `assemble2D` +
   `eigSolve` at the admission design) rather than `hist.omega`, which is
   recorded before the iteration's update. This is stricter, not looser, and it
   reproduces the archived baseline ω₁ bitwise.

## What was not done

No production file modified; the `+impl/` manifest hashes unchanged after the
study. No preset changed. No projection, no β, no density filter, no
p-continuation, no mass continuation, no filter-radius change, no move-value
change, no threshold tuning, no second transition rule, no percentile or robust
statistic, no 800×100, no nine-mesh campaign, no `performance_comparison` rerun,
no candidate promoted.

---

# PREMATURE_MOVE_TRANSITION_PARTIAL

The mechanism is **confirmed**: the production ladder descends while the design
is still saturating its move bound (99.9 % utilization at 160×20, 65–81 % at
320×40), and preventing that descent removes 99.1 % of the refinement-driven
grayness excess at 320×40 with an ω₁ improvement and a 67 % better mid-density
fraction.

It is **partial**, not confirmed outright, for two reasons that must not be
glossed:

1. **At 160×20 the frozen statistic never fires** — `r_rho` never once drops
   below 0.5 in 600 iterations, because ~2 % of elements stay pinned at the move
   bound while 90 % of the design has stopped moving. This is the preregistered
   §12 outcome, and it is recorded here as its own experimental finding:
   **`MAX_UTILIZATION_TRANSITION_TOO_SENSITIVE`**. The candidate was not
   rescued.
2. **At 320×40 the credible termination occurred *before* the first descent**,
   so a matured *ladder descent* was never the operative mechanism at the
   reported design. The gain is the fixed-move gain, obtained by a principled
   rule rather than by fiat — not evidence that a gated ladder works.

# KEEP_CURRENT_MOVE_POLICY_PENDING_REVIEW

No production file was changed by this task, and no promotion is proposed. The
next question is not projection, β or the filter radius: it is whether a
transition statistic exists that measures bulk design maturity without being
held hostage by a saturated 2 % — and that is a separate, future task.
