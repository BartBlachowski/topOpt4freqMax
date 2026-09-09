# CAUSAL_ANALYSIS — production vs the frozen two-branch controller

Generated from `evidence/analysis.json`; every number traces to a tracked
per-iteration CSV or the frozen `evidence/baselines.json`.

Preregistration SHA-256 `8e323f837f7bbdaa5176d92621b4a45e4b377af131af5b3172429c629da27fbf`.

Runs executed: **C160x20, C320x40, C400x50**


## 1. Headline causal comparison

| quantity | 160x20 prod → cand | 320x40 prod → cand | 400x50 prod → cand |
|---|---|---|---|
| terminal status | NATIVE_CONVERGED → **CONVERGED** | NATIVE_CONVERGED → **CAP_HIT** | CONVERGED → **CONVERGED** |
| outer iterations | 91 → 219  (×2.41) | 131 → 1600  (×12.21) | 139 → 505  (×3.63) |
| inner MMA total | 2241 → 5074  (×2.26) | 2614 → 76532  (×29.28) | 2918 → 10301  (×3.53) |
| wall s | 125.5 → 425.3  (×3.39) | 388.0 → 34373.8  (×88.60) | 541.0 → 4689.3  (×8.67) |
| first descent iter | 79 → 103  (+24 later) | 130 → 275  (+145 later) | 138 → 389  (+251 later) |
| final move | 0.01 → 0.005 | 0.02 → 0.005 | 0.02 → 0.005 |
| M_nd % | 13.4025 → **12.7041**  (-5.21 %) | 23.3596 → **12.9233**  (-44.68 %) | 32.3283 → **15.3311**  (-52.58 %) |
| omega1 | 169.495227 → **170.011316**  (+0.304 %) | 165.950789 → **166.426697**  (+0.287 %) | 162.882616 → **166.456276**  (+2.194 %) |
| omega2 | 171.9600 → 171.4283 | 183.7697 → 203.6849 | 175.4039 → 201.5439 |
| gap12 | 0.01477 → 0.00833 | 0.10674 → 0.22387 | 0.07687 → 0.21079 |
| gray frac | 0.149375 → 0.144375 | 0.263750 → 0.152188 | 0.347600 → 0.179600 |
| mid frac | 0.025000 → 0.026875 | 0.095625 → 0.030312 | 0.188200 → 0.039200 |
| volume | 0.499999009 → 0.499999134 | 0.499999139 → 0.499999617 | 0.499999150 → 0.499999432 |

## 2. Transition audit (Phase 13)

No transition may occur without the frozen `E = A OR B`; none may occur
because β stalls. Every row below carries its declaring counter at 20.

| mesh | iter | move | branch | decl | window | nA | nB | β stalled? | prod would be at | ω₁ | M_nd | gray | vol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 160x20 | 103 | 0.04 → 0.02 | **A** | 102 | 83–102 | 20 | 0 | yes | stage 4 | 168.9804 | 13.0364 | 0.1462 | 0.499999 |
| 160x20 | 142 | 0.02 → 0.01 | **A** | 141 | 122–141 | 20 | 0 | yes | stage 4 | 169.8175 | 12.7884 | 0.1437 | 0.499997 |
| 160x20 | 181 | 0.01 → 0.005 | **B** | 180 | 161–180 | 0 | 20 | yes | stage 4 | 169.9766 | 12.7561 | 0.1450 | 0.499997 |
| 320x40 | 275 | 0.04 → 0.02 | **A** | 274 | 255–274 | 20 | 0 | yes | stage 4 | 166.4216 | 13.0121 | 0.1523 | 0.500000 |
| 320x40 | 314 | 0.02 → 0.01 | **B** | 313 | 294–313 | 0 | 20 | yes | stage 4 | 166.4163 | 12.9797 | 0.1528 | 0.500000 |
| 320x40 | 353 | 0.01 → 0.005 | **B** | 352 | 333–352 | 0 | 20 | yes | stage 4 | 166.4273 | 12.9401 | 0.1530 | 0.499999 |
| 400x50 | 389 | 0.04 → 0.02 | **B** | 388 | 369–388 | 0 | 20 | yes | stage 4 | 166.4176 | 15.6649 | 0.1822 | 0.500000 |
| 400x50 | 428 | 0.02 → 0.01 | **B** | 427 | 408–427 | 0 | 20 | yes | stage 4 | 166.4355 | 15.4413 | 0.1800 | 0.500000 |
| 400x50 | 467 | 0.01 → 0.005 | **B** | 466 | 447–466 | 0 | 20 | yes | stage 4 | 166.4427 | 15.3732 | 0.1796 | 0.499999 |

## 3. Termination audit (Phase 14)

A candidate may report `CONVERGED` only at `move = 0.005` after the frozen
terminal persistence.

| mesh | status | iter | move | A | B | E | branch | nB | amp | tol | med₂₀cos | ‖Δρ‖₂ | honest | genuine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 160x20 | **CONVERGED** | 219 | 0.005 | 0 | 1 | 1 | B | 20 | 0.00288 | 0.0500 | 0.9206 | 0.00234 | yes | yes |
| 320x40 | **CAP_HIT** | 1600 | 0.005 | 0 | 0 | 0 | — | 0 | 0.00487 | 0.1000 | -0.8854 | 0.00419 | yes | no |
| 400x50 | **CONVERGED** | 505 | 0.005 | 0 | 1 | 1 | B | 20 | 0.00161 | 0.1250 | 0.5108 | 0.00135 | yes | yes |

## 4. Preregistered promotion gates (Phase 20)

| gate | requirement (abbreviated) | result |
|---|---|---|
| **P2** |  | **PASS** |
| **P3** |  | **PASS** |
| **P4** |  | **PASS** |
| **P5** |  | **PASS** |
| **P6** |  | **PASS** |
| **P7** |  | **PASS** |
| **P8** |  | **PASS** |
| **P8_m400_likeForLike** |  | **PASS** |
| **P9** |  | **PASS** |
| **P10** |  | **PASS** |
| **P11** |  | **PASS** |
| **P12** |  | **PASS** |
| **P13** |  | **FAIL** |
| **P1** |  | **PASS** |
| **P14** |  | **PASS** |
| **P15** |  | **FAIL** |
| **allThreeRunsExecuted** |  | **PASS** |
| **allTerminatedGenuinely** |  | **FAIL** |
