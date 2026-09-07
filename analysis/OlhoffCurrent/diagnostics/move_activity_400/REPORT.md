# REPORT — 400x50 third-mesh measurement of design-activity scaling

**Not a controller experiment.** No transition rule was implemented, run, tuned or
promoted. Production move policy, admission and preset are untouched. The purpose
was to obtain an independent third-mesh measurement before any controller is designed.

| | |
|---|---|
| Implementation | `analysis/OlhoffCurrent` — `+impl` tree `c1455374…`, **74/74 verified**, currentness **CURRENT** |
| Preregistration | [`PREREGISTRATION.md`](PREREGISTRATION.md), SHA-256 `706b8865…`, frozen 2026-09-07T21:09:40Z **before the first 400x50 run** |
| Starting commit | `4ef315b` (Phase-A retention repair; clean tree) |
| Arms | **P400** production (139 outer, CONVERGED) · **F400** fixed move 0.04 (369 outer, CONVERGED) |
| Raw evidence | 138.6 MB of element-level trajectory, declared and **hash-gated** |
| Companion files | [`PROVENANCE.md`](PROVENANCE.md) · [`METRICS.json`](METRICS.json) · [`SCALING_ANALYSIS.json`](SCALING_ANALYSIS.json) · [`SPATIAL_ANALYSIS.json`](SPATIAL_ANALYSIS.json) · [`DATA_MANIFEST.json`](DATA_MANIFEST.json) |

---

## Headline

**The third mesh breaks the two-mesh power law.** The exponent depends
*materially* on which activity threshold is used — from `alpha ~ +0.90` at
`|drho|>1e-4` to `alpha ~ -2.0` at `|drho|>1e-2` — so by the preregistered rule
the result is **`SINGLE_POWER_LAW_NOT_SUPPORTED`**.

**Premature descent is confirmed and is monotonically worse with refinement:**
9.0% -> 43.4% -> **50.1%** of the topology's remaining evolution is truncated at
160x20, 320x40, 400x50. At 400x50, holding `move = 0.04` halves `M_nd`
(32.33% -> **16.16%**) while `omega1` *improves* by 2.14%.

**And the retention repair paid for itself immediately.** Questions the previous
study had to record as permanently unanswerable — is the saturated population
persistent or rotating, where is it, is the max statistic a moving front or
isolated dust — are answered here from the retained trajectory. The answers
matter: the low-threshold active set is a **persistent, structured, gray-tracking
population** (Jaccard 0.960), while the high-utilisation set is **transient
scattered dust** (Jaccard 0.386, interface adjacency 1.000).

---

## The twenty-eight questions

### 1. What raw trajectory evidence was lost from the prior studies?

The complete `NE x nOuter` per-element density history of **all three** earlier
diagnostics. `move_stop` wrote four (`baseline`/`fixedmove` at 160x20 and
320x40) and had hashed them in `FINAL_SHA256.txt`, so their loss is **provable**;
`admission_rule` and `move_transition` wrote theirs (`ar_run.m:81`,
`mt_run.m:83`) but never manifested them, so their loss is only inferable from
the code. No copy survives anywhere in the repository.

Irrecoverable consequences, absent rerunning those optimisations (forbidden, and
pointless now): per-element `rho`/`Delta rho` at 160x20 and 320x40, all
percentiles of their utilisation distributions, element identity (hence
persistence, Jaccard, birth/death), and all spatial information.

### 2. Why was it lost?

`*.mat` is ignored repo-wide and again in `diagnostics/.gitignore`, whose comment
reasoned that raw solver state is "fully reproducible from `code/`". Nothing ever
checked the files still existed, and all three studies declared themselves frozen
regardless. **The defect was not that the files were untracked** — 100 MB
trajectories do not belong in git — **it was that they were undeclared**: nothing
named them, nothing hashed them, and nothing failed when they vanished.

A repo-wide sweep found the exposure is broader: **193 of 220** `.mat` files on
disk are untracked *and* unmanifested, including
`campaign_9mesh_r2/benchmark_records.mat`, the archive `move_stop` verified its
production baseline against bitwise. That is recorded in
[`EVIDENCE_POLICY.md`](../../EVIDENCE_POLICY.md) as residual exposure rather than
silently fixed; it is outside this study's scope.

### 3. What prevents recurrence now?

A four-tier policy (source / metadata / evidence / scratch) and a gate:

- `olhoffcurrent_evidence_declare` **measures** size, SHA-256 and variable
  dimensions from the file and **refuses** to declare a `required` artifact that
  does not exist — a study whose evidence was never produced cannot be frozen;
- `olhoffcurrent_evidence_gate` classifies every declared artifact
  (`REQUIRED_PRESENT_MATCH` / `REQUIRED_MISSING` / `REQUIRED_HASH_MISMATCH` /
  `OPTIONAL_*` / `SCRATCH`) and **fails closed**, including on an unrecognised class;
- `tests/test_evidence_retention.m` covers R1–R10. R5/R6 are load-bearing: they
  assert the gate works on exactly the arrangement that failed before — a large
  file git does not track — and verify with `git check-ignore` that the path
  really is ignored, so the test cannot degrade into a tracked-file test.

All of it lives **outside `+impl/`**, so the canonical tree hash is unchanged
(re-verified: `c1455374…`, 74/74). The full pre-existing suite still passes with
**0 failures**.

For this study the gate reports **PASS**: 2 required, 2 present and matching.

### 4. What is the scientific provenance of the 400x50 run?

Production entry point, config hash `044d50a4…`, preset
`duOlhoffFixedPenaltySensitivityFiltered`, single-threaded, diagnostics on. Only
`runtime.name` differs from production for P400; F400 adds exactly
`move.policy='fixed'`, `move.initial=0.04` — the same override pair `move_stop`
used at 160x20 and 320x40 — plus its preregistered cap. Both asserted field-by-
field over the whole schema, with an explicit scope lock on `p`, mass model,
filter, radius, projection, multiplicity and MMA variant.

One provenance issue no previous check caught: **this machine runs MATLAB
25.2.0.2998904, while every prior study recorded 25.2.0.3042426 (Update 1)**.
Combining meshes across builds would be unsound, so the preregistration declared
a control in advance — rerun the 160x20 production baseline and compare against
the committed CSV. It reproduces: **91 vs 91 outer, descents [79,90] vs [79,90],
max relative error 4.136e-15**. Stated precisely, that is equivalence at the
archived CSV's precision (~15 significant digits), **not** literal bitwise
identity — the per-element state that would license a bitwise claim is exactly
what was lost.

### 5. Did P400 and F400 share a bitwise-identical prefix?

**Yes — `COUNTERFACTUAL_PREFIX_VERIFIED`.** Over iterations 1–137 (everything
before P400's first descent):

| compared | result |
|---|---|
| full density field `rho`, every element, every iteration | **bitwise identical**, max abs diff `0.000e+00` over **2,740,000** values |
| `drho` | **bitwise identical** |
| `hist.omega`, `vol`, `dxOuter`, `dxNorm2`, `move`, `beta`, `nInner`, `gap12` | **all bitwise identical**, `0.000e+00` |

Comparison was on raw float64 bit patterns, not a tolerance. F400 is therefore a
genuine continuation of production, not a similar run, and the causal reading of
§8 below is licensed.

### 6. At what iteration did production first descend from move=0.04?

**Iteration 138** (0.04 -> 0.02), and P400 then **converged at 139**. Production
descends once and immediately stops — the same pattern as 320x40 (descend 130,
stop 131), and unlike 160x20 (descend 79 and 90, stop 91).

### 7. How mature was the topology at that point?

Barely. At iteration 137, `M_nd = 32.396%`, gray fraction 0.348, mid-density
fraction 0.188 — against a fixed-move endpoint of `M_nd = 16.159%`. In the
preregistered completion measure the design was at **c = 0.806**, i.e. only ~81%
of the way through its `M_nd` evolution, versus 0.882 at 320x40 and 0.986 at 160x20.

### 8. What fraction of subsequent fixed-move evolution remained?

**50.1%** — 16.237 percentage points of `M_nd` still to come. Across the three
meshes:

| mesh | NE | first descent | `M_nd` there | fixed-move endpoint | **remaining** |
|---|---:|---:|---:|---:|---:|
| 160x20 | 3200 | it 79 | 13.494% | 12.275% | **1.218 pts = 9.0%** |
| 320x40 | 12800 | it 130 | 23.451% | 13.282% | **10.168 pts = 43.4%** |
| **400x50** | **20000** | **it 138** | **32.396%** | **16.159%** | **16.237 pts = 50.1%** |

Monotone in refinement. The prior study's correction stands and strengthens:
production's descent is roughly on time at 160x20 and progressively more
premature as the mesh refines.

### 9. What was M_nd at first descent?  ### 10. What was mature F400 M_nd?

`32.396%` at the descent (iteration 137); `32.328%` at production's own
termination (139). F400's converged endpoint is **`16.159%`** — a **50.0%
relative reduction**. F400 reached `CONVERGED` at 369 outer, well inside its 600
cap, so this is genuine convergence under the inherited rule, not a cap artefact.

### 11. What happened to omega1?

It **improved**: 162.8826 (P400) -> 166.3649 (F400), **+2.138%**. Holding the
move costs nothing in the objective; it buys half the grayness and a better
first eigenfrequency. From the descent point to the F400 endpoint `omega1` moves
2.16% while `M_nd` moves 50.1% — `omega1` remains a poor maturity proxy, as at
the coarser meshes.

### 12. What did the element-activity distribution look like?

At iteration 137 (`move = 0.04`), the full distribution — recoverable **only**
because the trajectory was retained:

| statistic | value |
|---|---|
| `max(u)` | 0.5299 |
| `P99` | 0.1990 |
| `P97.5` | 0.1535 |
| `P95` | 0.1134 |
| `P90` | 0.0708 |
| `P75` | 0.0214 |
| `P50` | 0.000263 |
| `mean` / `RMS` | 0.0209 / 0.0482 |
| `N_eff` (participation) | 165.6 of 20000 |

Note `max(u) = 0.530` at 400x50 versus 0.647 at 320x40 and 0.9997 at 160x20 —
**`max(u)` at the descent falls monotonically as the mesh refines while remaining
evolution rises**, extending the prior study's inversion to a third mesh. At
400x50 the population with `u >= 0.9` is **empty** for the whole run: the
"saturated elements veto descent" pathology that dominated 160x20 simply does not
occur here.

### 13. How many low-threshold active elements existed?

At iteration 137, on the raw `|drho|` thresholds:

| threshold | count | fraction |
|---|---:|---:|
| `>1e-4` | 7670 | 38.35% |
| `>epsRMS` (8.839e-4) | 4930 | 24.65% |
| `>1e-3` | 4648 | 23.24% |
| `>1e-2` | 86 | 0.43% |

### 14. How were they distributed spatially?

This is the question the lost data could not answer, and the answer is sharp.

**The low-threshold set is a persistent, structured, gray-tracking population.**
Figure [10](figures/fig10_map_active_mask.png) shows it forming coherent bands
along the structural members and the diagonal bracing — not diffuse dust. It is
strongly enriched in gray material: among `|drho|>1e-3` elements the gray
fraction is **0.861** against **0.348** for the design as a whole, with mean
`rho = 0.538`. Over the 10 iterations before the descent its consecutive Jaccard
overlap is **0.960**, and **77.9%** of ever-active elements are active in *every*
one of those iterations.

**The high-utilisation set is transient scattered dust.** For `|drho|>1e-2`
(86 elements) the interface adjacency is **1.000** — *every* member has an
inactive neighbour, at all three states examined — and only 6.7% persist across
the window. For `u >= 0.5` the mean population is 7.6, with **0** elements active
in all ten iterations and Jaccard 0.386.

So the max statistic is dominated by **isolated, rotating, local activity**, not
by a physically meaningful moving front — the prior study's open question,
answered directly.

As the topology matures the active set thins onto its own boundary: interface
adjacency for `|drho|>1e-3` rises 0.558 -> 0.692 -> **0.875** from the descent to
the mature endpoint, while the count falls 4648 -> 3224 -> 2256.

### 15. What are alpha_160_320, alpha_320_400 and alpha_160_400?

At matched `M_nd` maturity (the preregistered primary comparison):

| threshold | c | `a_160_320` | `a_320_400` | `a_160_400` | global | spread |
|---|---:|---:|---:|---:|---:|---:|
| `1e-4` | 0.90 | 0.865 | 0.837 | 0.858 | 0.860 | 0.028 |
| `1e-4` | 0.95 | 0.859 | 1.085 | 0.914 | 0.901 | 0.225 |
| `1e-4` | 0.99 | 0.794 | 0.931 | 0.827 | 0.820 | 0.137 |
| `epsRMS` | 0.90 | 0.815 | **0.209** | 0.667 | 0.703 | 0.606 |
| `epsRMS` | 0.95 | 0.801 | 0.925 | 0.831 | 0.824 | 0.123 |
| `epsRMS` | 0.99 | 0.722 | 0.922 | 0.770 | 0.759 | 0.200 |
| `1e-3` | 0.90 | 0.805 | **0.065** | 0.625 | 0.667 | 0.740 |
| `1e-3` | 0.95 | 0.787 | 0.891 | 0.812 | 0.806 | 0.104 |
| `1e-3` | 0.99 | 0.695 | 0.929 | 0.752 | 0.738 | 0.235 |
| `1e-2` | 0.90 | **-1.342** | **-4.923** | -2.214 | -2.007 | **3.581** |
| `1e-2` | 0.95 | -0.209 | -0.330 | -0.239 | -0.232 | 0.120 |
| `1e-2` | 0.99 | -0.453 | -1.228 | -0.642 | -0.597 | 0.774 |

### 16. Does a global power law survive pairwise scrutiny?

**No, and this is exactly the failure mode the requirement was designed to
catch.** At `1e-3, c=0.90` the global three-point fit is a respectable 0.667
while the pairwise exponents are 0.805 and **0.065** — a twelve-fold disagreement
the global number completely conceals. Averaging that away would have produced a
confident and false "alpha ~ 0.67".

The failure is *structured*, not random, and the structure is worth stating:

- the `1e-2` rows are degenerate — counts fall to 12 and 2 elements, and every
  exponent is **negative**;
- the `c=0.90` rows for `epsRMS`/`1e-3` are unstable because the 320x40 active
  count jitters ±5% iteration to iteration there (3166, 3134, 3370, 3332, 3060,
  3252, 3034), so a single-iteration sample is unreliable. The 400x50 trajectory
  is smooth at the same point (3380…3488), so the instability is on the 320x40 side;
- excluding both — **a post-hoc restriction that was NOT preregistered** — the
  remaining six rows (low thresholds, `c >= 0.95`) give global alphas
  `[0.738, 0.901]`, spread **0.163**, worst pairwise spread **0.235**, which the
  preregistered rule *would* have called `APPROXIMATE_…_ALPHA_UNCERTAIN`.

That restriction is reported as a **hypothesis for a future preregistered test,
not as a result**. Choosing the admissible thresholds and maturity window after
seeing which combination behaves is precisely the move that would make this
analysis unfalsifiable.

### 17. Does alpha depend materially on diagnostic activity threshold?

**Yes — decisively, and this is the single clearest finding.** Global alpha runs
from **+0.90** (`1e-4`) through **+0.82** (`epsRMS`), **+0.74…0.81** (`1e-3`) to
**-0.23…-2.01** (`1e-2`). The spread across thresholds is **2.91**, far outside
the preregistered 0.25 tolerance. "The active count scales as `NE^alpha`" is not
a property of the design; it is a property of the *threshold you pick*.

### 18. Is constant-count scaling refuted?

**Yes, for every low threshold.** All pairwise and global exponents at `1e-4`,
`epsRMS` and `1e-3` are `>= 0.065` and cluster near 0.7–0.9; a constant count
(`alpha = 0`) would require the active population to be mesh-independent, and it
grows from ~700 to ~3300 to ~4600 elements at the descent. Refuted.

### 19. Is area-fraction scaling supported or refuted?

**Supported at the descent points, not at matched maturity.** At each mesh's own
first descent the active *fraction* is nearly invariant — 22.12%, 25.58%, 23.24%
at `|drho|>1e-3` — giving pairwise exponents 1.105, 0.785, 1.027, i.e. `alpha ~ 1`.
But at matched maturity the same threshold gives 0.74–0.81, materially below 1.
The two views disagree, which is itself evidence against a single clean law.

### 20. Is interface-like sqrt(NE) scaling supported or refuted?

**Refuted for the low thresholds.** Every low-threshold estimate at `c >= 0.95`
lies in 0.74–0.93, well above 0.5, and no pairwise low-threshold exponent at
`c >= 0.95` falls near 0.5. This is despite the *spatial* evidence that the
active set becomes increasingly interface-like as it matures (adjacency 0.558 ->
0.875) — the geometry looks like a front, but its size does not scale like one.

### 21. Is an NE^alpha normalisation still scientifically defensible?

**As a fitted convenience within a fixed threshold and maturity window, yes;
as a law, no.** Two things must be separated. A normalisation exists that
preserves the ordering property a controller would need — permitting descent at
160x20 (9.0% remaining) while blocking at 320x40 and 400x50 — and it holds for
any `alpha < 1.027` at `|drho|>1e-3`, which comfortably contains the fitted
0.74–0.90. So the family is not dead. But the exponent itself is not a stable
quantity: it moves with threshold by more than 3.0 and with maturity sampling by
0.6, so writing `N/NE^0.83 < const` into a controller would be encoding an
artefact of two arbitrary choices.

### 22. Is active count still the best transition-statistic family?

**Yes — it remains the best of the families examined, and it is now better
motivated, but it is not yet calibratable.** In its favour, from this mesh: it is
the only family whose active set is a *persistent, physically coherent*
population (Jaccard 0.960, gray-enriched 0.861 vs 0.348, structured along
members); it preserves the required cross-mesh ordering at three meshes; and it
falls monotonically with maturity (4648 -> 3224 -> 2256).

Against the alternatives, this mesh is damning. `max(u)` reaches only 0.530 at
the descent and never attains 0.9 at all, so the statistic that vetoed descent
forever at 160x20 carries almost no signal at 400x50 — the same statistic behaves
oppositely at the two ends of the mesh range. As a purely descriptive note on the
recorded trajectory: the previously published `max(u)<0.5`-for-10 condition would
first be satisfied at iteration **216**, where **39.3%** of evolution still
remained — badly premature again. (Per brief §14 that says what the rule would
have *requested* on this trajectory; it says nothing about where a changed
trajectory would end up.)

### 23. Is there enough evidence to design ONE controller candidate next?

**No.** By the preregistered rule the scaling verdict is
`SINGLE_POWER_LAW_NOT_SUPPORTED`, which forecloses
`ACTIVE_COUNT_CONTROLLER_READY_FOR_DESIGN`. Designing one now would mean fixing
a threshold and an exponent that this experiment has just shown to be
interdependent and unstable.

What *would* settle it is narrow and stated for the next preregistration: the
open question is no longer "does the active count work" but **"is there a
threshold at which the exponent is stable across meshes and maturity?"** The
`1e-4`/`epsRMS`/`1e-3` band at `c >= 0.95` is the pre-declared place to look, and
it needs a fourth mesh to test rather than to fit.

### 24. Does anything justify projection now?  ### 25. The sensitivity filter?  ### 26. Changing R = 0.06*b?

**No, no, and no.** Nothing in this study bears on any of them; all three remain
scope-locked and were asserted unchanged before each run. One result must
specifically *not* be misread as support: the finding that low-threshold activity
is concentrated in gray material (0.861 vs 0.348) describes where the *design
increment* lives, not a deficiency of the filter or an argument for projection.
The 50% grayness reduction reported here was obtained **with no regularisation
change whatsoever** — only by not descending the move — which if anything weakens
the case that a regularisation change is what the grayness needs.

### 27. Did any production scientific source change?

**No.** `git diff-tree` on `analysis/OlhoffCurrent/+impl/` for the mixed commit
`7154d82` is empty, and the Phase-A additions live outside `+impl/`. The tree
hash `c1455374…` is unchanged and was re-verified 74/74 before both runs, with
`currentness = CURRENT` and the dispatch gate passing (sensitivity filter and
published MMA both resolving inside `+impl/`).

### 28. Are all required raw evidence artifacts present and hash-verified?

**Yes — `olhoffcurrent_evidence_gate` reports PASS:** 2 required, 2
`REQUIRED_PRESENT_MATCH`, 0 missing, 0 mismatched.

| artifact | class | bytes | dims | precision |
|---|---|---:|---|---|
| `P400_400x50_trajectory.mat` | required | 36,576,273 | `RHO`,`DRHO` 20000x139 | double |
| `F400_400x50_trajectory.mat` | required | 102,057,786 | `RHO`,`DRHO` 20000x369 | double |

Both carry `RHO`, `DRHO`, `move`, full `hist` and resolved `cfg`, so `rho(k)`,
`rho(k-1)`, `Delta rho_e(k)` and `move(k)` are exactly reconstructible.
Reconstruction was validated at run time: volume agreement `< 1e-12`, final
design **bitwise** equal to `res.rho`, and the clamp measured inert —
**zero** elements out of 20000x139 ever had `rho+drho` outside `[rho_min, 1]`,
with a residual of 5.551e-17, below `eps/2 = 1.11e-16`, i.e. round-off of the
subtraction. Prior studies asserted that inertness in a comment; here it is
measured.

---

## Figures

All in [`figures/`](figures/), production first descent marked on every
iteration-axis plot.

| # | file | |
|---|---|---|
| 1 | [`fig1_Mnd_vs_iteration.png`](figures/fig1_Mnd_vs_iteration.png) | P400 vs F400 `M_nd`, with the truncated evolution shaded |
| 2 | [`fig2_omega1_vs_iteration.png`](figures/fig2_omega1_vs_iteration.png) | `omega1` |
| 3 | [`fig3_move_vs_iteration.png`](figures/fig3_move_vs_iteration.png) | move limit |
| 4 | [`fig4_activity_percentiles.png`](figures/fig4_activity_percentiles.png) | full utilisation distribution — new at this mesh |
| 5–6 | [`fig5_active_counts.png`](figures/fig5_active_counts.png) · [`fig6_active_fractions.png`](figures/fig6_active_fractions.png) | active counts and fractions |
| 7–10 | [`rho`](figures/fig7_map_rho.png) · [`abs drho`](figures/fig8_map_absdrho.png) · [`u`](figures/fig9_map_u.png) · [`active mask`](figures/fig10_map_active_mask.png) | spatial maps at iteration 137 |
| 11 | [`fig11_Nactive_vs_NE_loglog.png`](figures/fig11_Nactive_vs_NE_loglog.png) | `N_active` vs `NE`, three meshes |
| 12 | [`fig12_pairwise_vs_global_alpha.png`](figures/fig12_pairwise_vs_global_alpha.png) | pairwise vs global exponents |
| 13 | [`fig13_remaining_evolution_vs_mesh.png`](figures/fig13_remaining_evolution_vs_mesh.png) | remaining evolution vs mesh |

---

## Verdicts

**Phase A**

    TRAJECTORY_RETENTION_GATE_PASS

**Phase B**

    PROVENANCE_STATE_ACCEPTABLE

`7154d82` is 143 pure additions with no `+impl/` change; its unrelated trees are
already on the forbidden-paths list. Log hygiene, not executable ambiguity.
History was not rewritten.

**Phase C — scaling**

    SINGLE_POWER_LAW_NOT_SUPPORTED

Worst pairwise spread **3.581**, spread across thresholds **2.908**, both far
outside the preregistered 0.25 tolerance. The exponent depends materially on the
diagnostic threshold, from +0.90 to -2.01.

**Phase C — controller readiness**

    MOVE_ACTIVITY_MODEL_REQUIRES_REVISION

Follows from the preregistered rule, which permits
`ACTIVE_COUNT_CONTROLLER_READY_FOR_DESIGN` only when the scaling verdict is one
of the first two options.

**Production**

    KEEP_CURRENT_MOVE_POLICY_PENDING_REVIEW
