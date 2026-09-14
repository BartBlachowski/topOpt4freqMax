# REFERENCE_KKT — Part 9

## Verdict

```
REFERENCE_PROBLEM25_KKT_PASS
GLOBAL_PROBLEM25_REFERENCE_CERTIFIED
```

Evaluated by `scripts/fp_kkt.m` on the **exact production problem** (all four
rows via `deltaLambda`/`ddlam`, production scaling, production box) at the
conic reference `xRef`. Two multiplier sets were used; the preregistration
(§5) requires actual multipliers or a mathematically correct KKT treatment and
forbids the 1e−12 bound-classification shortcut.

## Multiplier set A — `coneprog`'s returned duals

`lambda.soc = 0.5808`, `lambda.ineqlin = [3.0e−17, 0.72912]`, `lambda.lower`,
`lambda.upper` as returned. Stationarity normalized RMS **0.419**, max 4.89 ⇒
`REFERENCE_PROBLEM25_KKT_FAIL` *for this multiplier set*. Complementarity and
primal feasibility pass. The solver's dual iterate stalled (see
`CONIC_REFERENCE.md`); these duals are not the problem's multipliers, and the
FAIL is a statement about them, not about the primal point.

## Multiplier set B — the certificate multipliers (decisive)

`μ = [1.0000000000, 0, 4.1e−17, 0.7291136447]` for rows 1–4 (row 2 carries 0
because it is redundant and slack by 0.278), box multipliers
`ξ = max(q, 0)`, `η = max(−q, 0)` with `q = ∇f0 + Σ_i μ_i ∇c_i` formed from
the cone direction `s/‖s‖` at `xRef`. These are the maximizers of the
weak-duality bound, verified — not assumed — by the gap of 9.2e−12.

| condition | preregistered PASS bar | measured |
|---|---|---|
| primal: max production row value | ≤ 1e−8 | **4.0e−14** (rows: 1.5e−14, −0.27813, −4.94958, 4.0e−14) |
| primal: box violation | ≤ 1e−10 | **0** |
| dual: min multiplier (μ, ξ, η) | ≥ −1e−10 | **0** |
| complementarity: max\|μ_i c_i\| | ≤ 1e−6 | **2.9e−14** |
| complementarity: max(ξ·gap, η·gap)/(sRow0·move) | ≤ 1e−6 | **5.0e−7** |
| stationarity: normalized RMS (drho rows) | ≤ 1e−6 | **1.9e−15** |
| stationarity: normalized max | ≤ 1e−5 | **7.6e−14** |
| stationarity: bs row | — | −4.1e−17 |
| cone dual: `‖p‖ ≤ μ`, cone complementarity | ≤ 1e−8 | 1.0000 ≤ 1.0000, **0** |

Normalizer `sRow0 = RMS(F11/lamref) = 6.4346e−4` (per-point normalizer
6.4347e−4; identical to four digits).

All PASS ⇒ `REFERENCE_PROBLEM25_KKT_PASS`. Together with
`FROZEN_PROBLEM25_CONVEXITY_CERTIFIED` and the duality gap 9.2e−12 ≤ 1e−8 ⇒
`GLOBAL_PROBLEM25_REFERENCE_CERTIFIED`.

## The one number near its bar

Box complementarity 5.0e−7 against a bar of 1e−6. It is the interior-point
character of `coneprog`: variables sit 1e−12…1e−9 inside their bounds and the
product with the multiplier is ~3e−12 raw (`max_eta_gap = 3.2e−12`); the
normalization by `sRow0·move = 6.4e−6` inflates it. The same quantity is
bounded by the duality gap (its sum over all variables is the 9.2e−12 box
term of the gap decomposition), which is the more precise statement.

## Bound activity — reported by sweep, never by a single tolerance

| tolerance (× box width) | at −move | at floor | at +move | at ceiling | interior |
|---|---|---|---|---|---|
| 1e−8 | 4 710 | 9 398 | 4 570 | 9 684 | 438 |
| 1e−7 | 4 888 | 9 404 | 4 772 | 9 686 | 50 |
| **1e−6** | **4 906** | **9 404** | **4 788** | **9 686** | **16** |
| 1e−5 | 4 908 | 9 404 | 4 796 | 9 686 | 6 |
| 1e−4 | 4 910 | 9 404 | 4 796 | 9 686 | 4 |
| 1e−3 | 4 910 | 9 404 | 4 797 | 9 686 | 3 |

The count is stable from 1e−6 upward: the reference is a vertex-like point
with 3–16 genuinely interior coordinates. Reduced-gradient magnitudes agree:
0 coordinates have `|q_e| ≤ 1e−8·max|q|`, 22 have `|q_e| ≤ 1e−6·max|q|`.
Those few near-zero reduced gradients are the coordinates along which the
optimal face may have positive dimension (`REFERENCE_SOLUTION.md`).
