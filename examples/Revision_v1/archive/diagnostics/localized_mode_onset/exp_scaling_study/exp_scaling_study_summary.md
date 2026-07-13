# Extended Exp3/S1 Scaling Study: Onset of Localised Modes

Purpose: characterise onset of mesh-dependent localised low-density modes.
This is NOT a mesh-convergence study. MATLAB only. alpha=1.00 only.
Physical filter radius: 0.08 m (= 2 elements at 200x25 reference mesh).
Created: 2026-06-29T10:36:13Z

**Onset mesh (first MAC<0.8 or majority localised): 80x10**

## Results Table

| s | mesh | N_e | classification | iters | omega_1 (rad/s) | MAC | grayness | s1_overall | phys | loc | amb |
|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| 0.5 | 80x10 | 800 | accepted | 1309 | 3.977 | 0.9041 | 0.0521 | likely localized/spurious low-density mode influence | 0 | 6 | 4 |
| 1.0 | 160x20 | 3200 | accepted | 873 | 4.561 | 0.9134 | 0.0815 | likely localized/spurious low-density mode influence | 0 | 10 | 0 |
| 1.5 | 240x30 | 7200 | mode invalid | 2000 | 15.72 | 0.0772 | 0.0784 | likely localized/spurious low-density mode influence | 0 | 10 | 0 |
| 2.0 | 320x40 | 12800 | mode invalid | 1428 | 94.34 | 0.6148 | 0.0664 | likely localized/spurious low-density mode influence | 1 | 9 | 0 |
| 2.5 | 400x50 | 20000 | mode invalid | 1750 | 64.39 | 0.7861 | 0.0608 | likely localized/spurious low-density mode influence | 0 | 8 | 2 |
| 3.0 | 480x60 | 28800 | accepted | 1083 | 89.12 | 0.9324 | 0.0938 | likely localized/spurious low-density mode influence | 1 | 9 | 0 |

## S1 Mode 1 Detail

| s | mesh | mode1 class | mode1 low-density strain frac |
|---|---|---|---:|
| 0.5 | 80x10 | localized low-density mode | 0.9996 |
| 1.0 | 160x20 | localized low-density mode | 0.9921 |
| 1.5 | 240x30 | localized low-density mode | 0.9983 |
| 2.0 | 320x40 | localized low-density mode | 0.9997 |
| 2.5 | 400x50 | ambiguous | 0.002223 |
| 3.0 | 480x60 | physical global mode | 0.03959 |

## Notes

- 400x50 (s=2.5) result reused from Exp3 if previously accepted.
- Onset defined as: first mesh where MAC<0.8 OR majority of first 10 modes are localised.
- No mesh-convergence claim is made. No Exp2 alpha sweep, CR2, A4, P1, Python, or manuscript edits.
