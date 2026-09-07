# REPORT — canonical move/stopping diagnostic

Was a substantial part of the fine-mesh grayness caused by our reconstruction's
move/stopping machinery rather than by the printed Du–Olhoff formulation?

| | |
|---|---|
| Implementation | `analysis/OlhoffCurrent` (sole production Olhoff) |
| `+impl/` tree SHA-256 | `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c` (74 source files) |
| Production preset | `duOlhoffFixedPenaltySensitivityFiltered` |
| Effective config hash (160×20) | `ca9a5c90d5c412759b43ca5c44f8d6b3d3e13ca70bfde6d8f6e9cead0da7cde6` |
| Repository HEAD at freeze | `a19d13eccc8dfe6487145f6c10ad6f8e024be525` |
| MATLAB | R2025b Update 1 (25.2.0.3042426), **1 computational thread** |
| Preregistration | [`PREREGISTRATION.md`](PREREGISTRATION.md), frozen before the first run; hashes in [`FINAL_SHA256.txt`](FINAL_SHA256.txt) |
| Provenance | [`PROVENANCE_LEDGER.md`](PROVENANCE_LEDGER.md) |

The path gate passed before every run; `olhoffcurrent_assert_dispatch` verified
all 32 owned symbols resolve inside `+impl/`, with no competing Olhoff tree on
the MATLAB path.

---

## The four arms

| Arm | Mesh | Status | outer | inner | ω₁ | **M_nd** | gray | mid |
|---|---|---|---|---|---|---|---|---|
| Production: move ladder | 160×20 | NATIVE_CONVERGED | 91 | 2241 | 169.495227 | **13.4025 %** | 0.1494 | 0.0250 |
| Production: move ladder | 320×40 | NATIVE_CONVERGED | 131 | 2614 | 165.950789 | **23.3596 %** | 0.2637 | 0.0956 |
| Diagnostic: fixed move 0.04 | 160×20 | **CAP_HIT** | 400 | 11429 | 169.560396 | **12.2754 %** | 0.1350 | 0.0181 |
| Diagnostic: fixed move 0.04 | 320×40 | NATIVE_CONVERGED | 216 | 4070 | 166.531071 | **13.2824 %** | 0.1542 | 0.0314 |

`M_nd = 100·mean(4ρ(1−ρ))`; gray = fraction with 0.1<ρ<0.9; mid = fraction with
0.4≤ρ≤0.6. All on the physical density.

---

## The sixteen questions

### 1. Did the canonical production baseline reproduce?

**Yes, bitwise, at both meshes.** Against the archived nine-mesh conference
record (`campaign_9mesh_r2/benchmark_records.mat`):

| | 160×20 | 320×40 |
|---|---|---|
| final design ρ | **bitwise** | **bitwise** |
| ω₁ | **bitwise** | **bitwise** |
| outer / inner | 91/91, 2241/2241 | 131/131, 2614/2614 |

The baselines were run with `runtime.diagnostics = true` while the archive was
produced with it off, so this **simultaneously proves the per-iteration recorder
is bitwise inert** — the property the diagnostic depends on.

The baseline arm was additionally asserted, field by field over all 80 schema
fields, to be identical to what the production entry point
`olhoffcurrent_config` produces (only `runtime.name` differs).

### 2. Does the problem still exist after the OlhoffCurrent cleanup?

**Yes.** The refinement-dependent grayness is not an artifact of the old
implementation lineage. Under the clean, path-isolated, integrity-verified
production implementation, M_nd rises from **13.40 % at 160×20 to 23.36 % at
320×40** — it nearly doubles under one mesh refinement.

### 3. Does every/most termination occur shortly after a move descent?

**Every one of them, at a gap of exactly one iteration.**

| Arm | descents | at iteration | terminates | gap |
|---|---|---|---|---|
| Production 160×20 | 2 | 79, 90 | **91** | **1** |
| Production 320×40 | 1 | 130 | **131** | **1** |

The gap is 1 rather than 0 because `settledMove` refuses to read the test on the
iteration where the limit changed. The solver says so itself:

```
iter 90: ||drho|| below eps but the move limit just changed (0.02 -> 0.01);
         convergence NOT asserted
converged at outer iteration 91 (||drho||_2 = 2.358e-02, max|drho| = 5.810e-03)
```

The guard delays admission by exactly one iteration — nowhere near enough for
the design to re-equilibrate.

### 4. How much does a move descent reduce the convergence statistic?

| Mesh | iter | move | ‖Δρ‖₂ before → after | **factor** | max\|Δρ\| before → after |
|---|---|---|---|---|---|
| 160×20 | 79 | 0.04→0.02 | 0.5235 → 0.2848 | ×1.84 | 0.0400 → 0.0200 |
| 160×20 | 90 | 0.02→0.01 | 0.1124 → 0.04094 | **×2.75** | 0.0199 → 0.00926 |
| 320×40 | 130 | 0.04→0.02 | 0.2691 → 0.09283 | **×2.90** | 0.0259 → 0.00775 |

In one iteration a halving of the move limit cuts the statistic by 2.75–2.90×.
The final descent carries it **across ε** in both cases.

### 5. How much does topology actually change at that instant?

**Essentially nothing.**

| Mesh | iter | M_nd before → after | change | ω₁ change |
|---|---|---|---|---|
| 160×20 | 90 | 13.361 → 13.386 | **+0.19 %** | +0.002 % |
| 320×40 | 130 | 23.451 → 23.401 | **−0.21 %** | +0.015 % |

So the convergence measure falls by a factor of ~2.9 while the discreteness
measure moves by ~0.2 %. **The measure drops because the allowed move was
reduced, not because the design settled.**

Three further pieces of evidence for the same conclusion:

* **M_nd was still falling steeply at termination** — −0.102 %/iter (160×20) and
  −0.160 %/iter (320×40) over the last 30 iterations.
* **Bound saturation.** `max|Δρ|/move` averaged over the last 25 iterations is
  **0.978** (160×20) and **0.780** (320×40): the optimizer was still pressing
  against its own move limit when it was declared converged. For comparison the
  fixed-move 320×40 arm ends at **0.201** — a genuinely small step inside its
  bound.
* **The arms are the same trajectory.** Baseline and fixed-move are numerically
  identical until the first descent (at 320×40 both give M_nd = 46.108 % at
  iteration 50, 30.843 % at 91, 28.483 % at 100). They separate only when the
  ladder acts.

### 6–7. Does fixed move reduce grayness at 320×40, and by how much?

**Yes — by 43 %.**

| Metric | Production: move ladder | Diagnostic: fixed move 0.04 | change |
|---|---|---|---|
| **M_nd** | 23.3596 % | **13.2824 %** | **−43.14 %** |
| gray fraction | 0.2637 | 0.1542 | −41.53 % |
| **mid-density fraction** | 0.0956 | **0.0314** | **−67.16 %** |
| volume | 0.499999 | 0.500000 | — |

This clears the preregistered "substantial" threshold of ≥ 25 % decisively.

At 160×20 the same change gives only **−8.41 %** (13.4025 → 12.2754). **That
asymmetry is the hypothesis, not a contradiction of it:** H1 predicts the effect
*grows with refinement*.

### 8. What happens to ω₁?

**It improves.** +0.350 % at 320×40 (165.9508 → 166.5311) and +0.038 % at
160×20. There is no objective/discreteness trade-off here: the fixed-move arm is
both less gray and better on the objective, because it is simply further along
the same optimization.

### 9. Does fixed move converge under the inherited stopping rule?

**Mesh-dependent, and this matters.**

* **320×40: yes** — NATIVE_CONVERGED at 216 outer iterations, ‖Δρ‖₂/ε = 0.971,
  reached at the *full* move limit with no descent at all.
* **160×20: no** — **CAP_HIT** at the preregistered 400-iteration cap, with
  ‖Δρ‖₂/ε = **7.88**. This is exactly the outcome §B7 registered in advance.

### 10. If not, does its physical trajectory nevertheless stabilize?

**Yes, unambiguously.** For fixed-move 160×20 at the cap:

| Quantity | Value |
|---|---|
| M_nd slope, last 30 iterations | **−0.0013 %/iter** (flat) |
| ω₁ slope, last 30 iterations | −2.7e-03 /iter |
| ω₁ relative range, last 25 | 3.4e-04 |
| **‖Δρ‖₂ / ε** | **7.88 — never approaches the tolerance** |
| bound saturation `max|Δρ|/move`, final | **1.000** |

The optimizer state has settled; the stopping *rule* simply cannot admit it. The
reason is visible in the saturation figure of 1.000: with a fixed move, a small
set of elements keeps stepping at the full bound indefinitely, which puts a
**floor under ‖Δρ‖₂ that no amount of settling can clear**. At 160×20 that floor
is ≈ 0.4, about 8× ε.

**This is stated plainly: CAP_HIT is not convergence and is not relabelled as
such.** It is also the reason fixed move is *not* a drop-in production remedy —
which is why §B6 defined it as a diagnostic instrument.

### 11. Does active-set scaling explain increasing refinement sensitivity?

**No — the proposed model is not supported, and the data are reported as
measured.**

`N_active` was measured, not assumed. At the matched operating point (last
iteration at move = 0.04):

| Mesh | N_E | N_active (\|Δρ\|>ε_RMS) | fraction |
|---|---|---|---|
| 160×20 | 3200 | 720 | 22.5 % |
| 320×40 | 12800 | 3400 | 26.6 % |

`N_active` is **not** approximately constant under refinement: the count scales
≈ 4.7× for a 4× increase in N_E, i.e. roughly proportionally. The proposed
relation `RMS(Δρ) ≈ move·√(N_active/N_E)` over-predicts the measured RMS by
2.5–15× and its ratio is not constant across arms or meshes (0.064 – 0.407). The
data are not forced to this model.

**The mechanism that the data do support is simpler and is the mesh-scaled
tolerance itself.** ε is scaled as `0.05·√(N_E/3200)`, so ε **doubles** from
0.05 to 0.10 while ‖Δρ‖₂ does not. The consequence is the number of halvings
needed to cross the threshold:

| Mesh | ‖Δρ‖₂/ε at the last move=0.04 iteration | log₂ → halvings needed | descents actually taken |
|---|---|---|---|
| 160×20 | **10.47** | 3.39 | **2** |
| 320×40 | **2.69** | 1.43 | **1** |

The finer mesh sits **3.9× closer to its own tolerance** at the same move limit,
so a **single** descent suffices to terminate it — and it therefore terminates
much earlier in its de-graying trajectory (M_nd 23.4 % rather than 13.4 %).
Refinement does not dilute a constant active set; it *raises the tolerance*
toward the statistic, so fewer bound reductions are needed to trip it.

### 12–13. Decomposition: how much is refinement-driven excess, how much remains?

Taking each arm's own settled value as the floor:

| Mesh | production M_nd | fixed-move M_nd (floor) | **refinement-driven excess** | share of observed |
|---|---|---|---|---|
| 160×20 | 13.4025 % | 12.2754 % | **1.13 points** | 8.4 % |
| 320×40 | 23.3596 % | 13.2824 % | **10.08 points** | **43.1 %** |

Two things follow:

1. **The excess grows ~9× from 160×20 to 320×40.** That is the refinement
   dependence H1 predicts.
2. **The residual floor is essentially mesh-independent**: 12.28 % (160×20) and
   13.28 % (320×40). Note that the 320×40 fixed-move result (13.28 %) lands at
   the **160×20 production value (13.40 %)** — removing the descent removes
   almost exactly the refinement-driven excess and nothing else.

### 14. Is the residual compatible with ordinary p=3 SIMP + sensitivity filtering?

**Not established, and deliberately not claimed.**

A mesh-independent floor of ~12–13 % M_nd is *consistent* with a gray transition
band of fixed physical width: the filter radius is specified physically
(`R = 0.06·b`), so the band occupies a roughly constant fraction of the domain
under refinement, which is what the two floors show. Sensitivity filtering is
also well known not to produce black-and-white designs by itself, which is why
the preregistration explicitly declined to predict a binary topology.

**But this cannot be attributed to the printed Du–Olhoff formulation**, for a
reason recorded in `PROVENANCE_LEDGER.md` before any run: the filter radius is
**class C — never stated for any example, in either paper** — and the repository
evidence calls it "the single strongest determinant of member thickness." The
residual floor is therefore confounded with an unstated reconstruction choice,
and **this experiment did not vary it**. Separating filter radius from
formulation would need a radius study, which is out of scope here.

### 15. Is any evidence now invalidated by the old implementation-lineage mess?

**No.** The two production baselines reproduce the archived conference anchors
**bitwise** under the clean implementation, so the archived Olhoff rows remain
valid as descriptions of what that realization does. What this diagnostic
changes is the *interpretation*: those runs were terminated by a move descent
rather than by the design settling, so their grayness at fine meshes reflects
the reconstruction's stopping machinery, not a converged property of the
formulation.

### 16. Should the production preset be reconsidered in a future task?

**Yes — but not by adopting fixed move, and not in this task.**

The evidence says the production realization terminates on a bound reduction
rather than on settlement, and that at 320×40 this costs ~10 points of M_nd and
0.35 % of ω₁. That is a real defect worth addressing. But:

* fixed move is **not** a remedy — it fails to converge at all at 160×20
  (‖Δρ‖₂/ε = 7.88 at the cap) because elements churning at the bound floor the
  statistic;
* the sound direction the evidence points to is the **admission rule**, not the
  move policy: a criterion that cannot be satisfied by shrinking the bound
  (e.g. measuring the *physical* change, or requiring settlement over a window
  at an unchanged bound, or a bound-saturation condition) would remove the
  mechanism without discarding the step control the multiple-eigenvalue
  expansion needs;
* all three ingredients involved — ladder, ε scaling law, `settledMove` — are
  **class C reconstruction choices with no numeric value anywhere in the
  Du–Olhoff lineage**, so revising them does not contradict the papers.

That is a design decision requiring review, and no production file was changed.

---

## Figures

| File | Content |
|---|---|
| `fig1_topology_comparison.png` | final topologies, all four arms |
| `fig2_Mnd_vs_iteration.png` | M_nd vs iteration |
| `fig3_omega1_vs_iteration.png` | ω₁ vs iteration |
| `fig4_l2_vs_iteration.png` | ‖Δρ‖₂ and ε, with descents marked |
| `fig5_maxabs_vs_iteration.png` | max\|Δρ\| vs iteration |
| `fig6_move_vs_iteration.png` | move limit vs iteration |
| `fig7_active_set.png` | active-element count and fraction |

`fig4` carries the argument: at 320×40 the two curves are **indistinguishable
until iteration 130**, then production drops vertically to ε while the fixed arm
decays to the same ε on its own by iteration 216. At 160×20 the fixed arm
plateaus near 8ε and never converges.

## Data

`METRICS.json` (all metrics, transitions, active set, stability),
`runs/*_iterations.csv` (complete per-iteration recorders: 28 columns),
`runs/*.mat` (full results plus reconstructed density history).

`runs/*.mat` is **not tracked in git** — 37 MB of raw per-iteration density
history, fully reproducible from `code/` and already summarised in the committed
CSVs and `METRICS.json`. Regenerate with
`ms_run('baseline'|'fixedmove', nelx, nely, runsDir)`. The figures, CSVs and
`FINAL_SHA256.txt` **are** tracked: `analysis/OlhoffCurrent/diagnostics/.gitignore`
re-includes `*.png`, `*.csv` and `*.txt` inside this tree only, because here they
are the evidence the report cites rather than scratch output.

The density history was reconstructed as the solver forms it,
`ρ_k = min(1, max(ρ_min, ρ_{k−1}+Δρ_k))`, and **validated against `hist.vol`
with maximum error 0.00e+00** in every run. Because projection and density
filtering are off, the design variable **is** the physical FE density — asserted
per run, not assumed.

## Deviations from the preregistration

1. **The fixed-move 320×40 arm converged**, where §B7 anticipated it might not.
   The registered handling (report CAP_HIT honestly if reached) was applied
   as written to 160×20, which did hit the cap.
2. **A `bound saturation` diagnostic (`max|Δρ|/move`) was added during analysis.**
   It is derived from already-recorded quantities, adds no run and no
   configuration change, and is reported as a post-hoc addition rather than a
   preregistered metric.
3. **No 400×50 confirmatory mesh was run.** The primary result was not
   ambiguous, so the §B4 authorization did not trigger.
4. One process note: the first attempt to run fixed-move 160×20 appeared to have
   exited early; a duplicate was briefly launched and killed within a minute
   once the original was seen to be still running. The original completed
   normally. No result is affected — the solver is deterministic and
   single-threaded — and only one artifact was written.

## What was NOT done

No projection, no density filtering, no β sharpening, no p continuation, no mass
continuation, no tolerance tuning, no change to the filter, p, mass
interpolation or multiplicity, no 800×100, no nine-mesh campaign, no production
file modified, no historical directory reorganized.

---

# MOVE_STOP_INTERACTION_CONFIRMED

Against the decision rule frozen in `PREREGISTRATION.md` §9, all three
conditions for `CONFIRMED` are met:

1. **Baseline termination follows closely after a move descent at both meshes** —
   gap of exactly 1 iteration at 160×20 and 320×40.
2. **The descent reduces ‖Δρ‖₂ by a factor materially larger than the
   simultaneous topology change** — ×2.75 and ×2.90 against M_nd changes of
   +0.19 % and −0.21 %.
3. **The fixed-move arm substantially reduces M_nd at 320×40** — −43.14 %,
   against the preregistered ≥ 25 % threshold.

Recorded qualification: the effect is **small at 160×20 (−8.41 %) and large at
320×40 (−43.14 %)**. That is the refinement dependence the primary hypothesis
predicts, not a weakening of it — but it does mean the conclusion is established
at two meshes only, with the strong effect resting on the finer of the two.

Answering the central question directly: **yes — at 320×40, 43 % of the observed
grayness is attributable to our reconstruction's move/stopping machinery rather
than to the printed Du–Olhoff formulation.** The remaining ~13 % floor is *not*
attributed to the printed method, because it is confounded with the unstated
class-C filter radius.

# KEEP_PRODUCTION_PRESET_PENDING_REVIEW

No production file was changed by this task.
