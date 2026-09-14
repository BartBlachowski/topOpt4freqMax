# THE SINGLE-STAGE COUNTERFACTUAL — why no simulation is needed

---

## 1. The policy

```
SINGLE-STAGE (S)
    move = 0.04, held
    terminate at the first persistently satisfied frozen E = A OR B
    no descent to 0.02, 0.01 or 0.005
```

## 2. The prefix argument, stated in full

Let `F` be the four-rung candidate already run, and `S` the single-stage policy.
Both start from the same uniform initial design, the same formulation, the same
`move = 0.04`, and the same frozen exhaustion rule.

The two policies are **identical functions of the trajectory** up to and
including the first declaration, because:

1. the scientific formulation is the same (`p = 3`, Eq. 4b, `q = 1`, sensitivity
   filter, `R = 0.06·b`, projection off, subspace size 2, published MMA, same FE,
   same eigensolver, same objective, same volume constraint);
2. the move limit in force is `0.04` for both at every iteration `k ≤` the first
   declaration, since `F` descends only *after* it;
3. the exhaustion predicate reads only `ρ`, `Δρ` and `‖drho‖₂`, all of which are
   therefore identical;
4. the controllers differ **only** in what they do once `E` is declared — `F`
   descends a rung, `S` stops.

A difference in behaviour at step `t` cannot affect the trajectory at any step
`< t`. So `S`'s terminal state **is** `F`'s recorded state at the first
declaration. No simulation is required, and running one would produce the same
numbers at a cost of hours.

## 3. This is not merely asserted — it is checked

### 3.1 The move really was 0.04 throughout the prefix

Verified directly on each trajectory: `all(move[1..k_S] == 0.04)`, asserted in
`scripts/ml_verify_events.py`. The first value of `move` differing from 0.04
occurs at 103 / 275 / 389, one iteration after the declarations at 102 / 274 /
388.

### 3.2 No earlier controller intervention touched the prefix

`hist.stage` is 1 throughout each prefix; the controller's `descents` record
contains no entry before the first declaration; and the previous study's
transition audit established that the set of iterations at which `move` changed
equals exactly {declaration + 1} on every mesh, each advancing one rung.

### 3.3 The exhaustion event is recomputed, not trusted

An independent implementation of the frozen rule was run over `RHO` and
`hist.dxNorm2`:

| mesh | offline recomputation | controller's own trace | `A`,`B` element-wise |
|---|---|---|---|
| 160×20 | declare **102**, branch **A**, window 83–102 | declare 102 | identical |
| 320×40 | declare **274**, branch **A**, window 255–274 | declare 274 | identical |
| 400×50 | declare **388**, branch **B**, window 369–388 | declare 388 | identical |

The expected event indices 103 / 275 / 389 from the previous study are confirmed
as *first descents*; the *declarations* are one iteration earlier, which is the
frozen semantics (`transition = declaration + 1`).

### 3.4 The strongest evidence: bitwise identity with an independent arm

C400 iterations **1–369** are bitwise identical to `F400_400x50_trajectory.mat`
— a fixed-move `0.04` arm produced two days earlier under the *pre-controller*
source tree — across `RHO` (20000 × 369), all five `ω`, `β`, `‖Δρ‖₂`, inner
iteration counts, volume and relative gap.

That is a direct empirical demonstration of the prefix argument: an independent
run that only ever held `move = 0.04` produced, iteration for iteration and bit
for bit, exactly what the four-rung candidate produced before its first descent.
The single-stage policy's prefix is that same object.

No comparable same-build fixed-move arm survives for 160×20 or 320×40
(`RETENTION_AUDIT.md`), so for those two meshes the prefix argument rests on
§3.1–3.3 — the move history, the controller trace and the independent
recomputation — and **bitwise equivalence is not claimed**.

### 3.5 Telemetry is inert

Established in the previous study (`cv_tests` test 15): diagnostics on and off
give identical `ρ`, `ω`, `dxNorm2` and controller trace. The recorder that
produced these trajectories did not perturb them.

## 4. The single-stage endpoints

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| iteration | **102** | **274** | **388** |
| triggering branch | **A** | **A** | **B** |
| window | 83–102 | 255–274 | 369–388 |
| move | 0.04 | 0.04 | 0.04 |
| ω₁ | 168.980391 | 166.421616 | 166.417621 |
| ω₂ | 173.463059 | 203.423191 | 201.064626 |
| relative gap | 0.026528 | 0.222336 | 0.208193 |
| volume | 0.499998507 | 0.499999541 | 0.499999726 |
| `M_nd` [%] | 13.036370 | 13.012132 | 15.664940 |
| gray fraction | 0.146250 | 0.152344 | 0.182200 |
| mid-density fraction | 0.026875 | 0.029687 | 0.039800 |
| `max\|Δρ\|` | 0.03998441 | 0.02689667 | 0.00570252 |
| `max\|Δρ\|/move` | 0.99961026 | 0.67241674 | 0.14256299 |
| `‖Δρ‖₂` | 0.62964970 | 0.10312614 | 0.07997757 |
| RMS `Δρ` | 0.01113074 | 0.00091151 | 0.00056553 |
| `cosθ` | −0.91968357 | −0.67533627 | +0.99856049 |
| net/path | 0.12027094 | 0.39523928 | 0.99367370 |
| bound fraction | 0.071250 | 0.000000 | 0.000000 |
| β stalled? | yes | yes | yes |
| native stop holds? | no | no | **yes** |
| subspace size | 2 | 2 | 2 |
| cumulative inner MMA | 2 716 | 5 066 | 7 337 |
| cumulative wall [s] | 134.98 | 788.84 | 1 722.41 |
| ρ SHA-256 | `cf00d80b8be994bf…` | `14fc924dca1ae523…` | `9905d773dda05b6f…` |

Full-precision values and density hashes: `evidence/ladder_analysis.json`.

## 5. What S avoids

At 320×40 the four-rung run continued past this point for a further **1 326
outer iterations** and terminated `CAP_HIT` at 1600 without ever satisfying the
terminal exhaustion condition. The single-stage policy terminates at 274, so
**that pathology cannot arise**: there is no terminal rung to fail to exhaust.
