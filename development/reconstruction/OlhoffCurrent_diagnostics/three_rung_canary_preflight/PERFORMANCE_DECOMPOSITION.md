# PERFORMANCE_DECOMPOSITION — Part E

All per-outer timing below is **nondeterministic performance telemetry**:
written by the solver, never read back, excluded from every hash and every
scientific-state comparison.

## 1. Canary 480×60

| quantity | value |
|---|---|
| total wall | 2799.278047375 s |
| Σ tOuter | 2798.858787 s (0.015 % below total; the residue is setup + final analysis) |
| N_outer | 386 |
| N_inner | 7 300 |
| mean inner / outer | 18.912 |
| mean wall / outer | 7.25093 s |
| mean assembly+eigensolve / outer | 0.27157 s |
| mean gradient / outer | 0.016865 s |
| mean inner-MMA / outer | 6.96098 s |
| mean s per MMA step | 0.368074 s |
| T_other (total) | 0.5854 s |

Decomposition `T_outer = T_eig + T_grad + T_inner + T_other`:

| component | share |
|---|---|
| inner MMA | **96.00 %** |
| assembly + eigensolve | 3.75 % |
| gradients | 0.23 % |
| other | 0.02 % |

`T_total = N_outer × T_outer_avg` = 386 × 7.25093 = 2798.86 s. ✓

Assembly and eigensolve are **not separable** in this instrumentation, at any
mesh — `tEig` brackets both.

## 2. Canary 800×100

| quantity | value |
|---|---|
| total wall | 7554.096479416667 s |
| Σ tOuter | 7552.474774 s |
| N_outer | 468 |
| N_inner | 10 404 |
| mean inner / outer | 22.231 |
| mean wall / outer | 16.13777 s |
| mean assembly+eigensolve / outer | 1.18725 s |
| mean gradient / outer | 0.048926 s |
| mean inner-MMA / outer | 14.89901 s |
| mean s per MMA step | 0.670198 s |
| T_other (total) | 1.2077 s |

| component | share |
|---|---|
| inner MMA | **92.32 %** |
| assembly + eigensolve | 7.36 % |
| gradients | 0.30 % |
| other | 0.02 % |

`T_total = N_outer × T_outer_avg` = 468 × 16.13777 = 7552.5 s ✓

### 480 → 800 scaling, stage 1 (NE ratio 2.7778)

| component | 480×60 | 800×100 | ratio | exponent in NE |
|---|---|---|---|---|
| wall / outer | 5.9597 | 13.9432 | 2.340 | 0.832 |
| assembly + eigensolve | 0.2721 | 1.1636 | 4.276 | **1.422** |
| gradients | 0.0167 | 0.0487 | 2.915 | 1.047 |
| inner MMA / outer | 5.6693 | 12.7284 | 2.245 | 0.792 |
| s per MMA step | 0.2915 | 0.5863 | 2.011 | 0.684 |

Consistent with the legacy nine-mesh fits (§4): assembly+eigensolve 1.142,
gradients 0.895, s per MMA step 0.724. Stage 1 is used for the comparison
because it is the only stage with enough samples at both meshes (308 and 390
iterations, against 39 for each lower rung).

The eigensolve share rises from 3.75 % at 480×60 to 7.36 % at 800×100 — the
fastest-growing component, and still a minority. The nested MMA dominates at
both meshes, as it does at all nine legacy meshes.

## 3. The finding: MMA sub-problem cost is a strong function of the move limit

**480×60**

| stage | move | iterations | tOuter/it | tEig/it | tGrad/it | tInner/it | inner steps/it | **s / MMA step** |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.04 | 308 | 5.9597 | 0.2721 | 0.0167 | 5.6693 | 19.45 | **0.29151** |
| 2 | 0.02 | 39 | 11.0366 | 0.2662 | 0.0176 | 10.7515 | 17.74 | **0.60594** |
| 3 | 0.01 | 39 | 13.6627 | 0.2728 | 0.0174 | 13.3713 | 15.85 | **0.84382** |

**800×100** — the same effect, independently

| stage | move | iterations | tOuter/it | tEig/it | tGrad/it | tInner/it | inner steps/it | **s / MMA step** |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.04 | 390 | 13.9432 | 1.1636 | 0.0487 | 12.7284 | 21.71 | **0.58628** |
| 2 | 0.02 | 39 | 20.9952 | 1.2994 | 0.0499 | 19.6436 | 22.59 | **0.86958** |
| 3 | 0.01 | 39 | 33.2257 | 1.3120 | 0.0502 | 31.8607 | 27.08 | **1.17667** |

Three things at once, at both meshes:

* assembly+eigensolve and gradients are **flat** across stages at both meshes
  (480: 0.272 / 0.017 s; 800: 1.16–1.31 / 0.049 s) — neither depends on the move
  limit, and neither moves materially;
* inner **step counts do not explain it** — they *fall* at 480 (19.45 → 15.85)
  and rise only mildly at 800 (21.71 → 27.08);
* yet time per MMA **step** rises sharply — ×2.895 at 480 (0.29151 → 0.84382 s)
  and ×2.007 at 800 (0.58628 → 1.17667 s) — and per-outer cost more than doubles
  at both meshes.

Consequence: the two terminal rungs are **20.2 % of iterations but 34.4 % of
wall time** at 480×60, and **16.7 % of iterations but 28.0 % of wall time** at
800×100. Pricing them at stage-1 rates underestimates them roughly twofold.

The mechanism is **not resolved**. The cost sits inside `innerLoop`/`mmasub`,
whose internal dual iterations are not instrumented, so "a tighter box costs
more dual work per sub-iterate" is a plausible reading that the retained
telemetry cannot confirm. Recorded as an instrumentation gap. Nothing was
changed in response.

The effect is reproducible across two meshes and is not an artefact of the
fixed-work benchmark either: `cp_fixedwork`, evaluating the terminal state at
move 0.01 independently, returns 0.85123 s and 1.22894 s per MMA step — within
0.9 % and 4.4 % of the in-run stage-3 figures.

### Verdict

```
C800_RUNTIME_BEHAVIOR_EXPLAINED
```

The 800×100 canary's 7554 s decomposes completely: 468 outer iterations at
16.14 s each; 92.3 % of that in the nested MMA, 7.4 % in assembly+eigensolve,
0.3 % in gradients; per-outer cost 2.34× the 480×60 canary for a 2.78× mesh
(exponent 0.83), with every kernel scaling as the legacy fits and the fixed-work
benchmark predict. No unexplained residue: Σ tOuter accounts for 99.98 % of
total wall.

## 4. The legacy 720 → 800 inversion, explained

This is the inversion that motivated choosing 800×100 as the stress canary. It
is settled by the **legacy** record alone, arithmetically.

| | 720×90 | 800×100 | ratio |
|---|---|---|---|
| NE | 64 800 | 80 000 | 1.2346 |
| outer iterations | 223 | 170 | **0.7623** |
| wall per outer [s] | 10.7514 | 11.6886 | **1.0872** |
| total wall [s] | 2397.57 | 1987.06 | **0.8288** |

```
0.8288  =  1.0872  ×  0.7623      (identity reproduced to 16 digits)
total      per-outer   iteration
wall       cost        count
```

The finer mesh is **8.7 % more expensive per outer iteration** and ran **23.8 %
fewer iterations**. A stopping-regime effect, not anomalous scaling. Total
runtime is not required to be monotonic in NE, and here it is not.

It did not buy anything: the legacy 800×100 run stopped at `move = 0.02`, one
iteration after a move halving, at M_nd 50.7 % — the worst grayness in the
series. Fewer iterations, worse design, cheaper total.

Per-outer components, 720 → 800 (all legacy):

| component | 720 | 800 | ratio | local exponent in NE |
|---|---|---|---|---|
| assembly + eigensolve | 0.9178 | 1.1142 | 1.214 | 0.920 |
| gradients | 0.0299 | 0.0357 | 1.194 | 0.842 |
| inner MMA | 9.7983 | 10.5299 | 1.075 | 0.342 |
| — MMA steps/outer | 21.664 | 21.841 | 1.008 | 0.038 |
| — s per MMA step | 0.4523 | 0.4821 | 1.066 | 0.303 |

Every component rises; nothing got cheaper.

### Legacy nine-mesh log-log fits

| quantity | C | p | R²log |
|---|---|---|---|
| total wall | 0.03702 | 0.9825 | 0.980 |
| wall per outer | 0.003516 | 0.7174 | 0.985 |
| assembly+eigensolve per outer | 2.377e−6 | 1.1423 | 0.988 |
| gradient per outer | 1.374e−6 | 0.8949 | 0.996 |
| inner MMA per outer | 0.004129 | 0.6956 | 0.984 |
| s per MMA step | 1.437e−4 | 0.7239 | 0.995 |
| outer count | 10.53 | 0.2651 | 0.893 |

Assembly+eigensolve is the only component whose exponent exceeds 1; its share
rises from 2.1 % at 160×20 to 9.5 % at 800×100 — still a minority, because the
nested MMA dominates everywhere. The 480×60 canary agrees: 96.0 % inner,
3.75 % assembly+eigensolve.

## 5. Why legacy and canary wall times are not directly comparable

1. **Different recorder state.** Canaries need `runtime.diagnostics = true`;
   the legacy campaign ran `false`. Equal-mesh upper bound at 400×50: ≈ 1.79×
   per outer, itself confounded.
2. **Different controller, hence different stage mix.** The 480×60 canary spent
   20 % of its iterations in the two stages that cost 2–2.3× a stage-1
   iteration. The legacy run never reached stage 3 at all.
3. **Different session.** Same machine, different day. `cp_hostprobe` closes
   this gap *between the two canaries* — it cannot close it retroactively
   against the legacy campaign.

Sizes of the effect at both canary meshes:

| mesh | canary s/outer | legacy s/outer | ratio | canary total | legacy total | total ratio |
|---|---|---|---|---|---|---|
| 480×60 | 7.2509 | 5.4772 | 1.32 | 2799.3 s | 898.3 s | 3.12 |
| 800×100 | 16.1378 | 11.6886 | 1.38 | 7554.1 s | 1987.1 s | 3.80 |

In both cases the per-outer ratio is ~1.3–1.4 and the *total* ratio is ~3–4,
because the canaries run 2.4–2.8× more iterations. An unknown part of the
per-outer ratio is the recorder, part is the stage mix, and part is the session.
None of it is a controller inefficiency: the extra total cost is overwhelmingly
extra iterations, which is what the frozen rule requires before it will stop.

## 6. Host-load control (Part G)

| probe | load avg | dgemm 1200³ | sparse solve 200k |
|---|---|---|---|
| C480 pre | 11.11 | 0.013 s | 0.003 s |
| C480 post | 2.87 | 0.012 s | 0.003 s |
| C800 pre | 3.19 | 0.013 s | 0.003 s |
| C800 post | 2.85 | 0.012 s | 0.003 s |
| 480 fixed-work | 2.28 | 0.013 s | 0.003 s |
| 800 fixed-work | 2.21 | 0.011 s | 0.003 s |

The C480 pre-run load figure is a decaying post-boot artefact (the machine had
booted minutes earlier; measured CPU consumption at that moment was ≈ 0.56 of
10 cores). The calibration kernels are the reliable reading, and they are
**effectively identical across all six probes** — dgemm 0.011–0.013 s, sparse
solve 0.003 s throughout — so both canaries and both fixed-work benchmarks ran
on a host in materially the same state, despite load-average readings differing
by 5×. That is exactly what the calibration exists to distinguish, and it is why
the 480-vs-800 comparisons in this study are admissible while the
canary-vs-legacy ones carry caveats.

No thermal or cache control was attempted and none is claimed. Small wall-time
differences are not evidence of exact reproducibility.
