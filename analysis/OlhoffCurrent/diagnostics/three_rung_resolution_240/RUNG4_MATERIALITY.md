# RUNG-4 MATERIALITY — the decisive test at 240×30

Phases 13–19. Bars are those of `PREREGISTRATION.md` §6, **inherited verbatim**
from three earlier frozen preregistrations. Nothing is loosened, tightened,
replaced, padded or made mesh-specific.

```
omega1        material if the block improves omega1 by >= 0.10 % relative
M_nd          material if the block improves M_nd  by >= 2 % relative
topology      material if |d gray| >= 0.01 or |d mid| >= 0.01 or mean|d rho_e| >= 0.01
volume        material if |volume - 0.5| worsens by >= 1e-5
multiplicity  material if subspace size leaves 2, mode order changes, omega2 <= omega1,
              or a NaN/Inf appears.  GAP MAGNITUDE ALONE IS NOT MATERIAL.
cost          cost-dominated if the block costs >= 2x the outer iterations used to reach
              the state it starts from while sub-material on every metric above
relative      100*(b - a)/a, normalised by the EARLIER state
```

---

## 1. The primary test — objective retention (Phase 14)

This is an eigenfrequency-**maximization** reconstruction, so `ω₁` is the primary
gate.

```
omega1(S3) = 167.03846293238027
omega1(F)  = 167.04969317770568

rung4_omega1_pct = 100 * (167.04969317770568 - 167.03846293238027) / 167.03846293238027
                 = +0.0067231493443241 %

frozen bar       =  0.10 %
```

**Classification: `IMMATERIAL`.** The measured benefit is **14.9× below** the
bar. No uncertainty padding was applied, and none was preregistered.

## 2. Every other frozen bar

| quantity | `S3` | `F` | change | relative | bar | verdict |
|---|---|---|---|---|---|---|
| `ω₁` | 167.038463 | 167.049693 | +0.011230 | **+0.00672 %** | 0.10 % | **IMMATERIAL** |
| `M_nd` | 12.91652 | 12.94249 | **+0.02596** | **+0.20100 %** (worse) | 2 % | **IMMATERIAL** |
| gray fraction | 0.150000 | 0.149444 | −0.000556 | — | 0.01 | **IMMATERIAL** |
| mid fraction | 0.030833 | 0.031111 | +0.000278 | — | 0.01 | **IMMATERIAL** |
| mean \|Δρ_e\| | — | — | 0.002120 | — | 0.01 | **IMMATERIAL** |
| max \|Δρ_e\| | — | — | 0.06693 | — | — | (reported) |
| \|volume − 0.5\| | 4.48e-07 | 7.99e-07 | +3.51e-07 | — | 1e-5 | **IMMATERIAL** |
| gap₁₂ | 0.180772 | 0.179466 | −0.001306 | — | not a bar | **IMMATERIAL** |
| subspace size | 2 | 2 | 0 | — | any change | **IMMATERIAL** |

**Rung 4 is below every frozen materiality bar at 240×30.**

Note that `M_nd` moves the **wrong way**: the final `move = 0.005` rung leaves
the design *less* discrete than `S3` did, by 0.201 % relative. It is immaterial
either way, but it is not a benefit.

## 3. The running-best tail analysis (preregistered §7)

The endpoint comparison alone could understate rung 4 if the tail peaked mid-way
and fell back. `PREREGISTRATION.md` §7 therefore required the **running best**
over the entire rung-4 tail, and it was computed even though the run converged:

| | `S3` | best anywhere in the 1074-iteration tail | at iteration | relative to `S3` | bar | verdict |
|---|---|---|---|---|---|---|
| `ω₁` (max) | 167.038463 | **167.059612** | 1175 | **+0.01266 %** | 0.10 % | **IMMATERIAL** |
| `M_nd` (min) | 12.91652 | **12.88639** | 609 | **−0.23330 %** | 2 % | **IMMATERIAL** |

Even the most generous possible reading of rung 4 — crediting it with the best
value it ever reached, at whichever iteration, on either metric — leaves it
**7.9× below** the objective bar and **8.6× below** the `M_nd` bar.

**There is no point in the tail at which rung 4 was materially ahead of `S3`.**

## 4. Is anything still trending at the end?

Last 100 iterations versus the preceding 100:

| | mean over 1159–1258 | mean over 1259–1358 | change |
|---|---|---|---|
| `ω₁` | 167.050710 | 167.050694 | **−0.000010 %** |
| `M_nd` | 12.93228 | 12.93785 | **+0.04309 %** (worsening) |

`ω₁` is flat to seven significant figures and `M_nd` is drifting slightly worse.
Nothing material is still improving.

## 5. Cost (Phase 17)

| | value |
|---|---|
| outer iterations after `S3` | **1074** (79.1 % of the whole run) |
| inner MMA iterations after `S3` | **38 675** (87.5 % of the whole run) |
| rung-4 cost multiplier vs `S3` | **3.78×** |
| cost-dominated (≥2× while sub-material) | **yes** |
| wall after `S3` | 13 365 s — reported, **not relied upon** (5.63× s/inner drift within the run) |

Rung 4 consumes seven eighths of the entire computational budget of this run and
returns nothing that clears any frozen bar.

## 6. Physics and multiplicity (Phase 16)

| | `S3` prefix `[1, 284]` | full run |
|---|---|---|
| subspace size `N = 2` at every iteration | ✅ | ✅ |
| `ω₂ > ω₁` at every iteration | ✅ | ✅ |
| all `ω`, `M_nd` finite; no NaN/Inf | ✅ | ✅ |
| non-converged inner solves | **0** | **0** |
| minimum gap over `[S3, F]` | — | 0.179425 |

No mode crossing, no multiplicity change, no loss of separation. The gap narrows
by 0.0013 across rung 4, which is explicitly **not** a materiality criterion
because the objective is `ω₁`.

`degen` (expected near-degeneracy hits inside the multiplicity-aware subspace) is
non-zero at essentially every iteration — 284 over the prefix, 1358 over the full
run. That is the normal behaviour of this formulation, is not a failure
indicator, and is not part of the frozen gate. Reported descriptively only.

## 7. Volume feasibility

`|volume − 0.5|` is **4.48e-07** at `S3` and **7.99e-07** at `F` — both more than
two orders of magnitude inside the 1e-4 acceptance gate. Rung 4 worsens
feasibility by 3.51e-07, which is 28× below the 1e-5 materiality bar.

## 8. Because the run CONVERGED, Phase 19 does not gate the verdict

The brief's Phase 19 applies only if rung 4 `CAP_HIT`s. It did not: the run
terminated at 1358 under the frozen terminal rule, Branch B, full 20-iteration
persistence, with 242 iterations of headroom below the cap. So `F` is a genuine
terminal state and the `S3 → F` comparison is a straightforward materiality test
rather than a comparison against an arbitrary point in a churning tail.

The tail analysis of §3 was nevertheless performed and reported, because it was
preregistered and because it strengthens the conclusion rather than weakening it.

**However**, one Phase-19-adjacent fact is recorded because it is scientifically
important: stage 4 spent **92.4 %** of its 1074 iterations in the documented
low-amplitude-cancellation hole (`amp < tol` **and** `med₂₀ cos < 0`), with `E`
true only 5.9 % of the time. That is mechanistically the **same** regime that
caps 320×40. This mesh escaped it; 320×40 did not. See `C240_ANALYSIS.md` §4.

## 9. Conclusion

> On the previously unavailable 240×30 causal mesh, does the final `move = 0.005`
> rung provide any scientifically material benefit beyond the exact three-rung
> endpoint?

**No — on every frozen bar, at the endpoint and at its running best, by margins
of one order of magnitude or more.** It costs 87.5 % of the run's inner work to
deliver it.
