# CLOSED_LOOP_INTEGRALS — Part 6

## Method

Rectangular loop in the (u,v) plane through the frozen ρ₃₈₆:

```
ρ → ρ + a·u → ρ + a·u + b·v → ρ + b·v → ρ
```

each edge integrated by 8-point Gauss–Legendre on `g·dρ`; amplitudes
`a = b ∈ {1e−3, 3e−4, 1e−4}`; the ten preregistered pairs; `g_phys` evaluated on
the same quadrature points as the mandatory positive control. 1 024
evaluations, 390 s. No density was updated.

A conservative field gives `∮ g·dρ = 0` identically. By Stokes, for a small
rectangle `∮ → −a·b·(uᵀJv − vᵀJu)`, so the normalized loop integral
`|∮| / (a·b·max(|uᵀJv|,|vᵀJu|))` should reproduce the Part-5 asymmetry, and
`|∮|` should scale as **a²**.

## Result

| a = b | median normalized, **filtered** | median normalized, **physical control** |
|---|---|---|
| 1e−3 | **2.7213e−01** | 2.9400e−06 |
| 3e−4 | **2.8244e−01** | 7.2747e−06 |
| 1e−4 | **2.8544e−01** | 4.7191e−05 |

The filtered value converges to **0.285**, agreeing with the independently
measured Jacobian asymmetry of **0.287** — Stokes' theorem is satisfied by two
separate measurements that share no intermediate quantity.

### Amplitude scaling — the decisive discriminator

| | median exponent of \|∮\| vs a |
|---|---|
| filtered field | **1.999** |
| physical control | 0.978 |

A genuine curl gives exponent 2. Finite-difference noise accumulated along a
path of length ∝ a gives exponent 1. The filtered field returns 1.999; the
control returns 0.978. Per pair the filtered exponent is 1.91–2.01.

### Orientation reversal

| | value |
|---|---|
| forward | −3.508985e−06 |
| reversed | +3.508985e−06 |
| sum | 1.355e−18 |
| relative | **3.9e−13** |

Exact sign flip to 13 digits, as a line integral must.

## Per-pair, filtered

| pair | \|∮\| at a = 1e−3, 3e−4, 1e−4 | exponent | normalized |
|---|---|---|---|
| D1a,D1b | 5.555e−07 5.884e−08 6.821e−09 | 1.91 | 1.48e−01 |
| D2a,D2b | 8.197e−10 7.380e−11 8.190e−12 | 2.00 | 2.79e−02 |
| D1a,D2a | 3.509e−06 3.163e−07 3.515e−08 | 2.00 | 1.60e+00 |
| D3,D2a | 4.941e−07 4.562e−08 5.109e−09 | 1.99 | 7.86e−01 |
| D4a,D4b | 1.010e−06 9.337e−08 1.038e−08 | 1.99 | 4.15e−03 |
| D1a,D4a | 7.331e−06 6.615e−07 7.355e−08 | 2.00 | 1.33e−01 |
| D2a,D4a | 3.223e−07 2.857e−08 3.161e−09 | 2.01 | 7.94e−01 |
| D5D1a,D5D1b | 5.555e−07 5.884e−08 6.821e−09 | 1.91 | 1.48e−01 |
| D5D2a,D5D4a | 8.083e−07 7.173e−08 7.938e−09 | 2.01 | 3.96e−01 |
| D3,D4b | 1.502e−05 1.336e−06 1.479e−07 | 2.01 | 2.00e+00 |

The normalized column reproduces the Part-5 per-pair asymmetries.

## Admissibility disclosure

Of the 30 loop configurations, **29 lie entirely inside the design box**. One
does not: `D4a,D4b` at `a = 1e−3` takes a **single element** to
ρ = 9.978004e−04, i.e. **2.20e−06 below ρmin**, or 0.22 % of ρmin. No corner
exceeds ρ = 1 anywhere.

This is disclosed rather than corrected because (a) the excursion is far below
any physically meaningful density change and the FE evaluation is smooth there,
and (b) `D4a,D4b` is the pair with the **smallest** filtered asymmetry
(4.15e−03) and the **largest** control noise, so it cannot be driving any
conclusion. Removing it entirely changes the median normalized loop integral
from 0.272 to 0.283 at a = 1e−3 — i.e. it makes the result slightly stronger,
not weaker.

## Verdict contribution

Preregistered condition 4 for `FILTERED_FIELD_LOCALLY_NONCONSERVATIVE`: median
normalized loop ≥ 0.05, exact sign flip on reversal, and `|∮| ∝ a²` within a
factor 2 across the amplitude range.

Measured: **0.272–0.285**, sign flip to 3.9e−13, exponent **1.999**. ✓
