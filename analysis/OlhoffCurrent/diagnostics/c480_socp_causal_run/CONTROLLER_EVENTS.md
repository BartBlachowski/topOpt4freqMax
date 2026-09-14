# CONTROLLER_EVENTS — Part 7

The frozen A OR B controller (W = P = 20, Wnp = 10, ε = 0.15, levels
[0.04 0.02 0.01]) was used unchanged. Its code is untouched production
(`olh.move.limit`, `olh.move.exhaustion`). The iteration-3 control replay (P3)
reproduced exAmp and exCos bitwise.

## Control (retained canary)

| event | iteration | branch | move | amp | amp/ε | med₂₀cos | med₂₀net | persistence | ω₁ (pre) | terminal |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 308 | B | 0.04 | 0.14612 | 0.974 | 0.99848 | 0.99442 | nB = 20 | 163.867 | no → descent at 309 |
| S2 | 347 | B | 0.02 | 0.03980 | 0.265 | 0.99991 | 0.99927 | nB = 20 | 163.910 | no → descent at 348 |
| S3 | 386 | B | 0.01 | 0.00786 | 0.052 | 0.84621 | 0.93654 | nB = 20 | 163.932 | **yes** (CONVERGED) |

## Treatment

**No controller event occurred.** The run stopped fail-closed at outer 15, inside
stage 1 (move 0.04). No S1, S2 or S3 declaration was reached, and no descent
happened. The detector could not have declared before outer 20 in any case: the
trailing 20-iteration medians are undefined until k ≥ stage start + 19, so the
earliest possible declaration is at outer 39.

| outer | stage | move | exAmp = ‖Δρ‖₂ | amp/ε | exCos | A | B | nA | nB |
|---|---|---|---|---|---|---|---|---|---|
| 1–12 | 1 | 0.04 | 6.788 (= 0.04·√28800) | 45.3 | 0.84 → 0.47 | 0 | 0 | 0 | 0 |
| 13 | 1 | 0.04 | 5.705 | 38.0 | 0.34 | 0 | 0 | 0 | 0 |
| 14 | 1 | 0.04 | 5.330 | 35.5 | 0.22 | 0 | 0 | 0 | 0 |

Values: `evaluations/traj_treatment.csv` (exAmp, exCos, exA, exB, exNA, exNB).
The full-saturation amplitude is 45× ε. Branch B (amp < ε) was therefore
structurally out of reach while exact steps moved every element. Branch A
additionally requires a negative median coherence; coherence was still positive
(0.22) but falling monotonically at termination. What the controller would have
done after outer 15 is unknown and is not extrapolated.
