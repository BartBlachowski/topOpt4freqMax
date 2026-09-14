# STATE_IDENTITY — Part 1

## Verdict

```
FROZEN_480_STATE_IDENTITY_PASS
```

All preregistered identity checks pass; zero blockers.

## The authoritative state

| | |
|---|---|
| study of origin | `diagnostics/three_rung_canary_preflight` (the three-rung canary) |
| trajectory | `evidence/three_rung_canary_preflight/C480x60_three_rung_trajectory.mat` |
| final outer iteration | **386** |
| status | CONVERGED |
| ρ₃₈₆ SHA-256 | `0a498a7d6ab0565b29c15ff9364060d937d4df10aa661fc90d02c038ce6e4a60` |
| config hash | `03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e` |
| `+impl` tree | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`, verified |

The ρ and config hashes are identical to those `gray_kkt_forensic_audit`
recorded for its 480 endpoint, so this audit continues on exactly the state that
audit analysed — **not** legacy beta data, and not the four-rung parent.

## Two densities, never conflated

`olhoffSolve` evaluates the FE problem at the density **before** the update, so
the final MMA subproblem was posed at ρ₃₈₅, and ρ₃₈₆ is what that subproblem
produced.

| | SHA-256 | role |
|---|---|---|
| ρ₃₈₅ = `RHO(:,385)` | `9b1443e508a6e4ecb5288d30167c7258d837be127aa3a9a42d8cc092240f3ded` | the density the last subproblem was **built at** |
| Δρ₃₈₆ = `DRHO(:,386)` | `0b12cd7f9ae32decc6e95bf63e89fa7ca4530d13ec6dc7d815b618d46af8a9fd` | the increment that run produced |
| ρ₃₈₆ = `RHO(:,386)` = `res.rho` | `0a498a7d…e4a60` | the authoritative frozen endpoint |

Part 3 certifies the subproblem at ρ₃₈₅. Parts 4–9 probe the field at ρ₃₈₆.

## Recorded state

| quantity | at ρ₃₈₅ (pre-update) | at ρ₃₈₆ (endpoint) |
|---|---|---|
| ω₁ | 163.93172747916933 | 163.93225938567002 |
| ω₂ | 185.2086604517607 | 185.21032464590309 |
| ω₃ | 401.39842250345157 | 401.39946949684344 |
| ω₄, ω₅ | 559.4330251, 755.2308322 | 559.4329992, 755.2424166 |
| `gap12` | 0.129791427808232 | 0.12979791372346 |
| `gap23` | — | 1.1672629226489644 |
| multiplicity N | 2 | 2 (detected) |
| `dOff` | — | [0, 7429.078688150214] |

| endpoint scalar | value |
|---|---|
| β (recorded) | 26873.74466704797 |
| volume | 0.4999984544949872 (error −1.5455e−06) |
| M_nd | 26.34156302529312 % |
| gray fraction | 0.28729166666666667 |
| ρ range | [0.001015039971515184, 0.9998979225360609] |
| stage / move | 3 / 0.01 |
| `nInner` at 386 | 19, `innerConv` = 1 |
| `multJ` at 386 | 0 |

Note ρ never reaches either box bound exactly: the minimum is 1.015e−03 against
ρmin = 1e−03, the maximum 0.99990 against 1. The `max(1e−3, ρ)` guard in
`applyFilter` is therefore inactive everywhere, which matters for
`FILTER_OPERATOR_ANALYSIS.md` §1.

## Configuration in force

**Filter** — `sensitivity`, applied to `all` `f_sk`, physical radius 0.06,
`rminEl = 3.6`; `H` symmetric with 1 037 580 nonzeros; `Hs ∈ [19.76, 49.04]`.

**Multiplicity** — `subspace`, fixed size 2, tolerance 0.05, diagonal offsets
on, off-diagonal terms on.

**MMA** — `published` Svanberg Sept-2007, variable `increment`,
`tolInner = 0.05`, `minInner = 5`, `maxInner = 500`, `a0 = 1`, `a = 0`,
`c = 1000`, `d = 0`.

**Controller** — `stageExhaustion` for both continuation and stopping, ladder
`[0.04 0.02 0.01]`, terminal declaration on branch **B** at iteration 386.

## Confirmation it is the three-rung canary

The move ladder is three-rung, both controller switches are `stageExhaustion`,
the terminal declaration is branch B at 386, and the config hash matches the
three-rung canary's frozen hash. Legacy data would show
`[0.04 0.02 0.01 0.005]`, `boundVariable` and `designChange`. It does not.
