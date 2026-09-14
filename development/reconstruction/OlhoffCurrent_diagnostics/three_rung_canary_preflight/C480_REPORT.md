# C480_REPORT — canary 1, 480×60

## Verdict

```
C480_THREE_RUNG_CANARY_PASS
```

All ten preregistered gates hold. Evaluated mechanically by
`scripts/cp_analyze.py` against `PREREGISTRATION.md` §6; full record in
`evidence/analysis_480x60.json`.

## The run

| | |
|---|---|
| mesh / NE / free DOF | 480×60 / 28 800 / 58 678 |
| config hash | `03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e` (= frozen) |
| `+impl` tree | `edbfe47e…52cb` |
| status | **CONVERGED** |
| outer iterations | 386 (cap 1600, headroom 1214) |
| inner MMA total | 7 300; max 35 per outer; **0 non-converged** |
| final move / stage | 0.01 / 3 |
| ω₁ / ω₂ / ω₃ | 163.93225938567002 / 185.21032464590309 / 401.39946949684344 |
| gap12 / gap23 | 0.12979791372346 / 1.1672629226489644 |
| M_nd / grayness | 26.34156302529312 % / 0.28729166666666667 |
| volume / error | 0.4999984544949872 / −1.55e−06 |
| terminal max\|dρ\| / ‖dρ‖₂ | 6.243012e−04 / 7.857752e−03 (ε = 0.15) |
| multiplicity N final | 2 |
| wall | 2799.278047375 s (46.7 min) |
| ρ SHA-256 | `0a498a7d6ab0565b29c15ff9364060d937d4df10aa661fc90d02c038ce6e4a60` |

Trajectory rebuilt from `res.diag.drho` and **proved exact**: the rebuild ends
bitwise at `res.rho`, and the clamp-displacement residual is 5.551e−17.

## Controller events

| event | iteration | window | branch | move | amp | amp/ε | med₂₀cos | med₂₀net | persistence |
|---|---|---|---|---|---|---|---|---|---|
| S1 declaration | 308 | 289–308 | **B** | 0.04 | 0.146125 | 0.97417 | 0.9985 | 0.9944 | nB = 20 |
| S2 declaration | 347 | 328–347 | **B** | 0.02 | 0.039801 | 0.26534 | 0.9999 | 0.9993 | nB = 20 |
| S3 declaration (terminal) | 386 | 367–386 | **B** | 0.01 | 0.007858 | 0.05239 | 0.8462 | 0.9365 | nB = 20 |

Stage starts 1 / 309 / 348; two descents, both consumed by the exhaustion rule
at the iteration after declaration, exactly as `olh.move.limit`'s
`stageExhaustion` branch specifies. Every declaration reached the full
persistence P = 20 on a single branch; counter A never fired.

**Structure matches the validated in-range runs exactly.** Total = S1 + 78,
with S2 and S3 each at the minimum possible dwell of 39 = (W−1) + P — the same
pattern all four validated meshes show. Nothing new appeared in the controller.

| mesh | S1 declaration | S2 | S3 | total | pattern |
|---|---|---|---|---|---|
| 160×20 | 102 | 39 | 39 | 180 | S1 + 78 |
| 240×30 | 206 | 39 | 39 | 284 | S1 + 78 |
| 320×40 | 274 | 39 | 39 | 352 | S1 + 78 |
| 400×50 | 388 | 39 | 39 | 466 | S1 + 78 |
| **480×60** | **308** | **39** | **39** | **386** | **S1 + 78** |

One thing the canary *did* show that the in-range set could not: **S1 length is
not monotone in mesh.** It rose 102 → 206 → 274 → 388 and then *fell* to 308 at
480×60. The budget projection in `PREREGISTRATION.md` §4 assumed monotonicity
and over-predicted by 65 % (508 projected vs 308 observed). That projection was
explicitly not an acceptance criterion and no gate reads it — but it means the
extrapolated 800×100 figure should not be trusted either.

## The ten gates

| # | gate | result | evidence |
|---|---|---|---|
| 1 | three-rung policy actually used | **PASS** | runtime hash = frozen hash; levels `[0.04 0.02 0.01]`; `signalDrivesMove` and `ruleAdmitsStop` both true |
| 2 | valid stage sequence | **PASS** | observed 1 → 2 → 3, monotone, no skips |
| 3 | terminal persistent E at move = 0.01 | **PASS** | declared iteration 386, stage 3, move 0.01, status CONVERGED |
| 4 | no hidden legacy/beta transition | **PASS** | beta holds neither authority; exactly 2 descents, both exhaustion events |
| 5 | no numerical failure | **PASS** | status CONVERGED; no NaN in ω or ρ; eigensolver never failed |
| 6 | inner MMA solves acceptable | **PASS** | 0 of 386 non-converged; max 35 inner, cap 500 |
| 7 | no substantial unresolved terminal evolution | **PASS** | see below |
| 8 | no config drift | **PASS** | hash identical before and after |
| 9 | telemetry / evidence complete | **PASS** | 386 CSV rows, no missing `hist` field, trajectory rebuild exact |
| 10 | no outcome-driven intervention | **PASS** | one run, no repeat, no parameter touched; preregistration frozen at `25a2500b…` before any run |

### Gate 7 in detail

Over the final 20 iterations: ω₁ moved 163.925639 → 163.931727, i.e. **+0.0037 %**;
M_nd moved 26.3776 → 26.3416, i.e. **−0.036 points**; volume error −1.5e−06.
Terminal amp/ε = 0.0524, inside the range the four validated in-range runs
produced at their own terminal declarations, [0.0327, 0.1036]. Stage 3 lasted
39 iterations — the minimum dwell, identical to all four.

So the design had genuinely stopped evolving at the resolution the terminal rung
can express. Gate 7 is about *unresolved evolution*, and there is none.

## Beta was recorded and had no authority

`beta` reached 26 873.74 and is retained as a diagnostic. It entered no
decision: the continuation signal is `stageExhaustion` and the stop rule is
`stageExhaustion`, both verified at run time, and `olh.config.validate` will not
resolve a half-applied policy.

Two counterfactual replays on the realized path (predicate replays, **not**
counterfactual trajectories — production under its own rule would have taken a
different path after its first descent):

* the **legacy `designChange` stop** would first have admitted convergence at
  **iteration 289** — 97 iterations and one full ladder descent before the
  three-rung run actually stopped, and before the first descent at 309;
* the **beta-stall shadow ladder** would also have reached stage 3.

The first is the substantive one: on this trajectory the legacy rule would have
stopped at a point the frozen rule does not consider exhausted.

## Next-mode warnings

4 of 386 iterations (1.04 %), first at iteration 12, **all four inside stage 1**,
none in stage 2, stage 3, or the terminal declaration window. The legacy 480×60
run recorded 4 of 164 (2.44 %). Detail in `MULTIPLICITY_WARNING_AUDIT.md`.

## Timing

| | value |
|---|---|
| total wall | 2799.278 s |
| Σ tOuter | 2798.859 s |
| mean wall / outer | 7.2509 s |
| mean assembly+eigensolve / outer | 0.27157 s (3.75 %) |
| mean gradients / outer | 0.016865 s (0.23 %) |
| mean inner MMA / outer | 6.96098 s (**96.00 %**) |
| other | 0.5854 s total (0.02 %) |
| mean inner steps / outer | 18.91 |
| mean s per MMA step | 0.36807 s |

Host: load 11.11 before (decaying post-boot artefact), 2.87 after; calibration
kernels 0.013 → 0.012 s (dgemm) and 0.003 → 0.003 s (sparse solve), i.e. the
host was materially the same at both ends of the run.

### A real finding: MMA sub-problem cost rises sharply as the move limit tightens

| stage | move | iterations | tOuter/it | tEig/it | tGrad/it | tInner/it | inner steps/it | **s per MMA step** |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.04 | 308 | 5.9597 | 0.2721 | 0.0167 | 5.6693 | 19.45 | **0.29151** |
| 2 | 0.02 | 39 | 11.0366 | 0.2662 | 0.0176 | 10.7515 | 17.74 | **0.60594** |
| 3 | 0.01 | 39 | 13.6627 | 0.2728 | 0.0174 | 13.3713 | 15.85 | **0.84382** |

Assembly+eigensolve and gradient time are **flat** across stages (0.272 and
0.017 s), as they must be — neither depends on the move limit. Inner *step
counts fall* (19.45 → 15.85). Yet time per MMA step nearly **triples**
(0.292 → 0.844 s, ×2.90), and per-outer cost more than doubles.

Consequence for cost accounting: stages 2 and 3 are 78 of 386 iterations
(20 %) but 963 of 2799 s (**34 %**) of wall time. Any projection that prices the
terminal rungs at stage-1 rates underestimates them by roughly a factor of two.

The mechanism is not resolved here. The cost is inside `innerLoop`/`mmasub`,
whose internal dual iterations are not instrumented, so whether a tighter box
costs more dual work per sub-iterate cannot be separated from the retained
telemetry. That is an **instrumentation gap, recorded, not a claim**; no
parameter was changed in response to it.

## Science, against the comparators

| | legacy 480×60 | **three-rung 480×60** | change |
|---|---|---|---|
| policy | beta / four-rung / designChange | stageExhaustion, three rungs | — |
| status | NATIVE_CONVERGED | CONVERGED | — |
| outer | 164 | 386 | +222 |
| inner MMA | 3 463 | 7 300 | +3 837 |
| final move / stage | 0.02 / 2 | **0.01 / 3** | reaches the last rung |
| ω₁ | 161.9055204201859 | **163.93225938567002** | **+2.027 (+1.25 %)** |
| M_nd | 34.67169512013076 % | **26.34156302529312 %** | **−8.33 points** |
| gap12 | 0.06966 | 0.12980 | wider |
| wall | 898.26 s | 2799.28 s | ×3.12 |

The direction matches the four in-range meshes: more work, a better ω₁, a less
gray design. **The wall-time ratio is not a like-for-like measurement** — the
canary runs with `runtime.diagnostics = true` and the legacy campaign did not
(`PREREGISTRATION.md` §8).

But the level is the thing to watch:

| mesh | three-rung M_nd % | three-rung ω₁ | legacy M_nd % | improvement |
|---|---|---|---|---|
| 160×20 | 12.756 | 169.975 | 13.402 | −0.65 pts |
| 240×30 | 12.917 | 167.039 | 15.604 | −2.69 pts |
| 320×40 | 12.940 | 166.426 | 23.360 | −10.42 pts |
| 400×50 | 15.373 | 166.452 | 32.328 | −16.96 pts |
| **480×60** | **26.342** | **163.932** | 34.672 | **−8.33 pts** |

Three-rung M_nd held near 13 % through 320, rose to 15.4 % at 400, and **jumps
to 26.3 % at 480** — while the advantage over legacy *narrows* from 17.0 to 8.3
points. ω₁ also falls (166.45 → 163.93) where it had been flat across 240–400.

This is the first quantitative sign that fine-mesh degradation is **not purely a
stopping artefact**: the controller did everything right here — correct stages,
full persistence, a terminal window with no measurable evolution — and the
design is still markedly grayer than the same controller produced one mesh
coarser. One mesh is not a trend, and 800×100 is the test.

## Files

`runs/C480x60_three_rung_iterations.csv` (386 × 55, frozen `cv_export` schema) ·
`runs/C480x60_three_rung_supplement.csv` (386 × 21, Part-B columns) ·
`runs/C480x60_three_rung_record.json` ·
`evidence/analysis_480x60.json` · `evidence/stage_timing_480x60.json` ·
`evidence/preflight_480x60.json` · `evidence/hostprobe_C480x60_three_rung_{pre,post}.json` ·
raw trajectory `evidence/three_rung_canary_preflight/C480x60_three_rung_trajectory.mat` (155.5 MB) ·
figures 1, 3, 4, 5, 6, 7, 8, 9, 10 for this mesh.
