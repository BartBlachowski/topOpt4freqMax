# THREE-RUNG ANALYSIS — the five states, and what S3 is worth

All values from `evidence/analysis.json`; index convention per
`COUNTERFACTUAL_VALIDITY.md` §6. Relative changes are normalised by the **earlier**
state, the convention inherited from `move_ladder_necessity` and
`two_rung_architecture`.

---

## 1. The five states

```
P   production baseline (beta-stall ladder), from the frozen baselines.json
S1  single-stage endpoint    = first frozen E declaration at move = 0.04
S2  two-rung endpoint        = first frozen E declaration at move = 0.02
S3  THREE-RUNG endpoint      = first frozen E declaration at move = 0.01
F   four-rung final state
```

### 160×20  (NE = 3200, tol = 0.05)

| state | outer | move | `M_nd` | `ω₁` | `ω₂` | gap₁₂ | gray | mid | volume |
|---|---|---|---|---|---|---|---|---|---|
| P | 91 | 0.01 | 13.4025 | 169.4952 | 171.9600 | 0.014766 | 0.14938 | 0.02500 | 0.49999901 |
| S1 | 102 | 0.04 | 13.0364 | **168.9804** | 173.4631 | 0.026527 | 0.14625 | 0.02688 | 0.49999851 |
| S2 | 141 | 0.02 | 12.7884 | 169.8175 | 171.9416 | 0.012513 | 0.14375 | 0.02625 | 0.49999699 |
| **S3** | **180** | **0.01** | **12.7561** | **169.9766** | 171.5258 | 0.009120 | 0.14500 | 0.02688 | 0.49999677 |
| F | 219 | 0.005 | 12.7041 | 170.0110 | 171.4302 | 0.008347 | 0.14438 | 0.02688 | 0.49999913 |

### 320×40  (NE = 12800, tol = 0.1)

| state | outer | move | `M_nd` | `ω₁` | `ω₂` | gap₁₂ | gray | mid | volume |
|---|---|---|---|---|---|---|---|---|---|
| P | 131 | 0.02 | 23.3596 | 165.9508 | 183.7697 | 0.106742 | 0.26375 | 0.09563 | 0.49999914 |
| S1 | 274 | 0.04 | 13.0121 | 166.4216 | 203.4232 | 0.222337 | 0.15234 | 0.02969 | 0.49999954 |
| S2 | 313 | 0.02 | 12.9797 | 166.4163 | 203.5528 | 0.223143 | 0.15281 | 0.03016 | 0.49999990 |
| **S3** | **352** | **0.01** | **12.9401** | **166.4273** | 203.5810 | 0.223232 | 0.15297 | 0.03016 | 0.49999874 |
| F | **1600 CAP_HIT** | 0.005 | 12.9233 | 166.4189 | 203.6763 | 0.223878 | 0.15219 | 0.03031 | 0.49999962 |

### 400×50  (NE = 20000, tol = 0.125)

| state | outer | move | `M_nd` | `ω₁` | `ω₂` | gap₁₂ | gray | mid | volume |
|---|---|---|---|---|---|---|---|---|---|
| P | 139 | 0.02 | 32.3283 | 162.8826 | 175.4039 | 0.076873 | 0.34760 | 0.18820 | 0.49999915 |
| S1 | 388 | 0.04 | 15.6649 | 166.4176 | 201.0646 | 0.208190 | 0.18220 | 0.03980 | 0.49999973 |
| S2 | 427 | 0.02 | 15.4413 | 166.4355 | 201.4595 | 0.210436 | 0.18000 | 0.03930 | 0.49999988 |
| **S3** | **466** | **0.01** | **15.3732** | **166.4427** | 201.5026 | 0.210643 | 0.17960 | 0.03920 | 0.49999911 |
| F | 505 | 0.005 | 15.3311 | 166.4562 | 201.5433 | 0.210789 | 0.17960 | 0.03920 | 0.49999943 |

## 2. The S3 event record

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| stage-3 start | 142 | 314 | 428 |
| **S3 iteration `kE(3)`** | **180** | **352** | **466** |
| offset from stage start | **38** | **38** | **38** |
| **triggering branch** | **B** | **B** | **B** |
| sustained window | [161, 180] | [333, 352] | [447, 466] |
| `A` / `B` / `E` at the event | 0 / 1 / 1 | 0 / 1 / 1 | 0 / 1 / 1 |
| `nA` / `nB` | 0 / 20 | 0 / 20 | 0 / 20 |
| `exStageStart` | 142 | 314 | 428 |
| move | 0.01 | 0.01 | 0.01 |
| `med₂₀ cosθ` | +0.56811 | +0.59417 | +0.19070 |
| `med₂₀ net/path` | +0.87031 | +0.75272 | +0.56666 |
| `‖Δρ‖₂` | 0.0051776 | 0.0032737 | 0.0079110 |
| `‖Δρ‖₂ / tol` | 0.1036 | 0.0327 | 0.0633 |
| RMS `Δρ` | 9.1529e-05 | 2.8935e-05 | 5.5940e-05 |
| `max|Δρ|` | 1.5548e-03 | 2.3280e-04 | 2.7277e-04 |
| `max|Δρ| / move` | 0.15548 | 0.02328 | 0.02728 |
| bound fraction | **0.0** | **0.0** | **0.0** |
| β-stall fired by then | yes | yes | yes |
| native stop holds | yes | yes | yes |
| subspace size `N` | 2 | 2 | 2 |
| cumulative outer | 180 | 352 | 466 |
| cumulative inner MMA | 4 283 | 6 498 | 8 848 |
| cumulative wall (s, unreliable) | 301.5 | 1 399.1 | 2 612.9 |
| `ρ` SHA-256 | `b779f0361545…` | `0c928dc01f56…` | `b3d388a576fe…` |

**All three meshes satisfy the frozen `E = A OR B` on `move = 0.01`, all via
Branch B, all with full 20-iteration persistence.** The three-rung policy
therefore terminates honestly on every primary mesh, under the same concept that
governs its descents — no second terminal rule is introduced.

Note that by S3 the design is no longer bound-limited anywhere: the bound
fraction is exactly zero on all three meshes and `max|Δρ|/move` has fallen to
0.16 / 0.023 / 0.027.

Two further observations, recorded because they are not what one might assume.
First, `med₂₀ cosθ` at S3 is only **+0.57 / +0.59 / +0.19** — far less coherent
than at S2 (+0.86 to +0.9999). Branch B requires only `med cos > 0` alongside
`amp < tol`, so it fires comfortably, but the motion at S3 is closer to
directionless than to smoothly convergent. Second, the native stop predicate
(`‖Δρ‖₂ < tol`) already **holds** at S3 on all three meshes, as it does at S2 on
the two fine meshes; Branch B at these rungs is the inherited native criterion
plus a weak coherence guard plus persistence, exactly as its own preregistration
says it is.

## 3. S3 against production

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| `ΔM_nd` vs P | −0.6464 (**−4.82 %**) | −10.4195 (**−44.60 %**) | −16.9551 (**−52.45 %**) |
| `Δω₁` vs P | **+0.4814 (+0.284 %)** | +0.4765 (+0.287 %) | +3.5601 (+2.186 %) |
| Δ gray | −0.00438 | −0.11078 | −0.16800 |
| Δ mid | +0.00187 | −0.06547 | −0.14900 |
| outer multiplier | 1.98× | 2.69× | 3.35× |
| inner MMA multiplier | 1.91× | 2.49× | 3.03× |

`S3` improves `M_nd` and `ω₁` against production on every mesh. It is strictly
better than `S2` on both headline quantities at every mesh, and strictly better
than `S1` — which regressed `ω₁` at 160×20 by 0.304 %.

## 4. S3 against F — the value of the final `move = 0.005` rung

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| `Δω₁` (F − S3) | +0.0344 (**+0.02025 %**) | −0.0084 (**−0.00504 %**) | +0.0136 (**+0.00815 %**) |
| `ΔM_nd` | −0.0520 (−0.4078 %) | −0.0168 (−0.1297 %) | −0.0422 (−0.2742 %) |
| `Δω₂` | −0.0956 | +0.0953 | +0.0407 |
| Δ gap₁₂ | −0.000767 | +0.000634 | +0.000146 |
| Δ gray | −0.000625 | −0.000781 | +0.000000 |
| Δ mid | +0.000000 | +0.000156 | +0.000000 |
| Δ\|volume−0.5\| | −2.36e-06 | −8.68e-07 | −3.19e-07 |
| mean \|Δρ_e\| | 0.000424 | 0.001723 | 0.000208 |
| RMS Δρ | 0.001695 | 0.005056 | 0.000387 |
| max \|Δρ_e\| | 0.03697 | 0.04167 | 0.00393 |
| Δ subspace size | 0 | 0 | 0 |
| outer / inner cost | 39 / 791 | **1 248 / 70 034** | 39 / 1 453 |
| **material on any bar?** | **no** | **no** | **no** |

**The final `move = 0.005` rung is below every preregistered materiality bar on
all three meshes.** At 320×40 it makes `ω₁` slightly *worse* while consuming
91.5 % of that mesh's entire inner budget and never terminating.

The decisive figure — the one that blocked the two-rung policy — is the 160×20
residual: **+0.02025 %**, five times *below* the frozen 0.10 % bar. Compare the
two-rung residual `S2 → F` at the same mesh: **+0.11393 %**, above it.

## 5. Termination semantics

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| four-rung terminal status | CONVERGED @219 | **CAP_HIT @1600** | CONVERGED @505 |
| three-rung terminal status | CONVERGED @180 | **CONVERGED @352** | CONVERGED @466 |
| terminal move | 0.01 | 0.01 | 0.01 |
| terminal branch / persistence | B / 20 | B / 20 | B / 20 |
| stage 4 declares at all? | yes (@219) | **no — never** | yes (@505) |
| CAP_HIT avoided | n/a | **yes** | n/a |

The 320×40 `CAP_HIT` label is **retained** for `F` throughout this study. `F` at
320×40 is not a converged endpoint and is nowhere described as one.

## 6. The finding that decides the verdict

At 160×20 the two-rung residual of **+0.11393 %** is material. It splits as:

| block | `Δω₁` | relative | vs the 0.10 % bar |
|---|---|---|---|
| rung 3 (`S2 → S3`, move 0.01) | +0.1591 | **+0.09367 %** | **below** |
| rung 4 (`S3 → F`, move 0.005) | +0.0344 | **+0.02025 %** | **below** |
| combined (`S2 → F`) | +0.1935 | **+0.11393 %** | **above** |

Both halves are individually sub-material; their sum is material. The
preregistered `THRESHOLD_SPLITTING` guard (`PREREGISTRATION.md` §11) therefore
fires, and caps the architecture verdict at `PARTIALLY_SUPPORTED`.

This is examined in full in `RUNG_VALUE_DECOMPOSITION.md` §4. The short statement
is that the 160×20 lower-ladder `ω₁` benefit beyond rung 2 is a **continuum of
small gains, not a discrete effect that one rung captures**: where you stop
determines how much you leave behind, and the materiality bar identifies no
natural stopping point. The three-rung policy satisfies the residual gate by
subdivision rather than by capturing a step.
