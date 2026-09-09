# TWO-RUNG ANALYSIS — the four states, and what S2 is worth

All values from `evidence/analysis.json`; index convention per
`COUNTERFACTUAL_VALIDITY.md` §4.

---

## 1. The four states

```
P   production baseline (beta-stall ladder), from the frozen baselines.json
S1  single-stage endpoint   = first frozen E declaration at move = 0.04
S2  TWO-RUNG endpoint       = first frozen E declaration at move = 0.02
F   four-rung final state
```

### 160×20  (NE = 3200, tol = 0.05)

| state | outer | move | `M_nd` | `ω₁` | `ω₂` | gap₁₂ | gray | mid | volume |
|---|---|---|---|---|---|---|---|---|---|
| P | 91 | 0.01 | 13.4025 | 169.4952 | 171.9600 | 0.014766 | 0.14938 | 0.02500 | 0.49999901 |
| S1 | 102 | 0.04 | 13.0364 | **168.9804** | 173.4631 | 0.026527 | 0.14625 | 0.02688 | 0.49999851 |
| **S2** | **141** | **0.02** | **12.7884** | **169.8175** | 171.9416 | 0.012513 | 0.14375 | 0.02625 | 0.49999699 |
| F | 219 | 0.005 | 12.7041 | 170.0110 | 171.4302 | 0.008347 | 0.14438 | 0.02688 | 0.49999913 |

### 320×40  (NE = 12800, tol = 0.1)

| state | outer | move | `M_nd` | `ω₁` | `ω₂` | gap₁₂ | gray | mid | volume |
|---|---|---|---|---|---|---|---|---|---|
| P | 131 | 0.02 | 23.3596 | 165.9508 | 183.7697 | 0.106742 | 0.26375 | 0.09563 | 0.49999914 |
| S1 | 274 | 0.04 | 13.0121 | 166.4216 | 203.4232 | 0.222337 | 0.15234 | 0.02969 | 0.49999954 |
| **S2** | **313** | **0.02** | **12.9797** | **166.4163** | 203.5528 | 0.223143 | 0.15281 | 0.03016 | 0.49999990 |
| F | **1600 CAP_HIT** | 0.005 | 12.9233 | 166.4189 | 203.6763 | 0.223878 | 0.15219 | 0.03031 | 0.49999962 |

### 400×50  (NE = 20000, tol = 0.125)

| state | outer | move | `M_nd` | `ω₁` | `ω₂` | gap₁₂ | gray | mid | volume |
|---|---|---|---|---|---|---|---|---|---|
| P | 139 | 0.02 | 32.3283 | 162.8826 | 175.4039 | 0.076873 | 0.34760 | 0.18820 | 0.49999915 |
| S1 | 388 | 0.04 | 15.6649 | 166.4176 | 201.0646 | 0.208190 | 0.18220 | 0.03980 | 0.49999973 |
| **S2** | **427** | **0.02** | **15.4413** | **166.4355** | 201.4595 | 0.210436 | 0.18000 | 0.03930 | 0.49999988 |
| F | 505 | 0.005 | 15.3311 | 166.4562 | 201.5433 | 0.210789 | 0.17960 | 0.03920 | 0.49999943 |

## 2. The S2 event record

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| iteration `kE(2)` | 141 | 313 | 427 |
| **triggering branch** | **A** (cancellation) | **B** (amplitude) | **B** (amplitude) |
| sustained window | [122, 141] | [294, 313] | [408, 427] |
| `A` / `B` / `E` at the event | 1 / 0 / 1 | 0 / 1 / 1 | 0 / 1 / 1 |
| `nA` / `nB` | 20 / 0 | 0 / 20 | 0 / 20 |
| `exStageStart` | 103 | 275 | 389 |
| stage-2 declaration latency | 38 | 38 | 38 |
| move | 0.02 | 0.02 | 0.02 |
| `med₂₀ cosθ` | −0.85673 | +0.99996 | +0.99964 |
| `med₂₀ net/path` | 0.22610 | 0.99975 | 0.99879 |
| `cosθ` | −0.87328 | +0.99995 | +0.99978 |
| net/path | 0.22700 | 0.99974 | 0.99856 |
| `‖Δρ‖₂` | 0.109469 | 0.010803 | 0.018062 |
| `‖Δρ‖₂ / tol` | **2.1894** | 0.1080 | 0.1445 |
| RMS `Δρ` | 1.9352e-03 | 9.5489e-05 | 1.2772e-04 |
| `max|Δρ|` | 0.019941 | 7.884e-04 | 1.194e-03 |
| `max|Δρ| / move` | **0.99707** | 0.03942 | 0.05971 |
| bound fraction | 0.006875 | 0.0 | 0.0 |
| β-stall fired by then | yes | yes | yes |
| native stop holds (`‖Δρ‖₂ < tol`) | **no** | yes | yes |
| native stop admitted | no | yes | yes |
| subspace size `N` | 2 | 2 | 2 |
| cumulative inner MMA | 3 647 | 5 752 | 8 037 |
| cumulative wall (s, unreliable) | 216.7 | 1 012.0 | 2 065.5 |
| `ρ` SHA-256 | `3931d0559912a398…` | `e4b4617d97eddb34…` | `3bcdf4e7e64566a9…` |

**All three meshes satisfy the frozen `E = A OR B` on `move = 0.02`.** The
two-rung policy therefore terminates on every primary mesh, under the same
concept that governs its descent — no second terminal rule is introduced.

## 3. A structural fact that must be stated, not buried

The stage-2 declaration latency is **38 on every mesh**, which is the *earliest
arithmetically possible* value: the trailing 20-median is first defined at
`stageStart + 19`, and the persistence counter then needs 20 consecutive hits,
giving `stageStart + 38`. The same is true of stage 3 on all three meshes and
stage 4 at 160×20 and 400×50.

That means: **in every post-first stage on which the rule fires, the predicate is
already true at the first iteration at which it can be evaluated.** The detector
is not observing a transition inside those stages; it is reporting a condition
inherited across the descent and confirming it for the minimum admissible time.
Every lower rung is consequently exactly 39 iterations long, except 320×40's
stage 4, where the predicate is *never* true and the run reaches the cap.

This is a property of the frozen rule and is recorded here as an observation.
**It is not used to modify any rule in this task** (PREREGISTRATION §14), and it
does not weaken the counterfactual: `S2` is still exactly the state the two-rung
policy would return. It does mean the two-rung policy's stage 2 is, in practice,
"take 39 more iterations at half the move limit and stop" rather than "run at
0.02 until dynamics say stop" — a fact the next task should carry forward.

## 4. S2 against production

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| `ΔM_nd` vs P | −0.6141 (**−4.58 %**) | −10.3799 (**−44.44 %**) | −16.8870 (**−52.24 %**) |
| `Δω₁` vs P | **+0.3223 (+0.190 %)** | +0.4655 (+0.281 %) | +3.5529 (+2.181 %) |
| Δ gray | −0.00563 | −0.11094 | −0.16760 |
| Δ mid | +0.00125 | −0.06547 | −0.14890 |
| outer multiplier | 1.55× | 2.39× | 3.07× |
| inner MMA multiplier | 1.63× | 2.20× | 2.75× |

`S2` improves `M_nd` against production on every mesh and improves `ω₁` on every
mesh. Compare `S1`, which improved `M_nd` everywhere but **lost** `ω₁` at
160×20 (−0.5148, −0.304 %).

## 5. S2 against F

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| `ΔM_nd` (F − S2) | −0.0843 (−0.659 %) | −0.0564 (−0.435 %) | −0.1102 (−0.714 %) |
| `Δω₁` (F − S2) | **+0.1935 (+0.1139 %)** | +0.0026 (+0.0016 %) | +0.0207 (+0.0124 %) |
| `Δω₂` | −0.5114 | +0.1236 | +0.0838 |
| Δ gap₁₂ | −0.004160 | +0.000723 | +0.000353 |
| Δ gray | +0.000625 | −0.000625 | −0.000400 |
| Δ mid | +0.000625 | +0.000156 | −0.000100 |
| Δ\|volume−0.5\| | −2.14e-06 | +2.87e-07 | +4.47e-07 |
| mean \|Δρ_e\| | 0.001432 | 0.002036 | 0.000598 |
| RMS Δρ | 0.005960 | 0.005803 | 0.001236 |
| max \|Δρ_e\| | 0.1124 | 0.0497 | 0.0145 |
| Δ subspace size | 0 | 0 | 0 |

Topologically S2 and F are the same design on all three meshes: mean per-element
density change of 0.0006–0.0020 against a 0.01 materiality bar, and the
difference maps in `figures/F9_topology_S2_vs_F.png` are near-empty.

## 6. Termination semantics (Phase 14)

Tested under the same frozen concept, with no separate terminal rule:

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| four-rung terminal status | CONVERGED @219 | **CAP_HIT @1600** | CONVERGED @505 |
| two-rung terminal status | CONVERGED @141 | **CONVERGED @313** | CONVERGED @427 |
| terminal move | 0.02 | 0.02 | 0.02 |
| terminal persistence | 20 (A) | 20 (B) | 20 (B) |
| CAP_HIT avoided | n/a | **yes** | n/a |

The 320×40 `CAP_HIT` label is **retained** for `F` throughout this study. `F` at
320×40 is not a converged endpoint and is never described as one.
