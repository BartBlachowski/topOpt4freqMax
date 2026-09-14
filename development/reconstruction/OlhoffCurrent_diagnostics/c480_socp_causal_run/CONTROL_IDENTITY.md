# CONTROL_IDENTITY — Part 1

```
C480_CONTROL_EVIDENCE_PASS
```

The control is the authoritative retained **three-rung C480 canary** from
`diagnostics/three_rung_canary_preflight`. It was **not rerun**. Every identity
item was checked mechanically by `scripts/cs_control_identity.m` (MATLAB) and
`scripts/cs_control_identity.py` (Python). Records:
`evaluations/control_identity_matlab.json`, `evaluations/control_identity.json`.

## Identity

| item | value | check |
|---|---|---|
| trajectory | `evidence/three_rung_canary_preflight/C480x60_three_rung_trajectory.mat` | SHA-256 `a87546bc…ab9b` = EVIDENCE.json ✔ |
| terminal state file | `C480x60_three_rung_state.mat` | SHA-256 = EVIDENCE.json ✔ |
| mesh | 480 × 60, NE = 28 800 | ✔ |
| initial ρ₀ | 0.5·ones, SHA-256 `8b5a00af…3b07` | ✔ |
| final ρ₃₈₆ | SHA-256 `0a498a7d…6a60` (MATLAB, Python, record) | ✔ |
| config hash (stored cfg) | `03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e` | ✔ |
| config hash (fresh `cp_config(480,60)`) | identical | ✔ |
| `+impl` tree (meta, and fresh manifest verify) | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` | ✔ |
| status / terminal iteration | CONVERGED / 386 | ✔ |
| trajectory rebuild from DRHO | max abs difference 0 | ✔ |

## Policy (not legacy, not beta, not old stopping)

| field | value |
|---|---|
| move.levels | [0.04 0.02 0.01] (three rungs, not the legacy four) |
| move.continuation.signal | `stageExhaustion` (not `boundVariable`, so beta has no authority) |
| stop.rule | `stageExhaustion` (not the old `designChange` production stop) |
| optimizer.inner | MMA, published variant, variable `increment` → `innerLoop` |
| multiplicity | `subspace`, size 2, diagonal offsets, off-diagonals |
| filter | sensitivity, R = 0.06 physical, applied to all f_sk |
| p / mass | 3 fixed / eq. 4b, q = 1 |
| projection | off |

## Trajectory and controller

| event | iteration | branch | move | amp | amp/ε | med₂₀cos | med₂₀net | nB |
|---|---|---|---|---|---|---|---|---|
| S1 declaration | 308 | B | 0.04 | 0.14612 | 0.9742 | 0.99848 | 0.99442 | 20 |
| S2 declaration | 347 | B | 0.02 | 0.03980 | 0.2653 | 0.99991 | 0.99927 | 20 |
| S3 declaration (terminal) | 386 | B | 0.01 | 0.00786 | 0.0524 | 0.84621 | 0.93654 | 20 |

Stage starts 1 / 309 / 348. Stage lengths 308 / 39 / 39. Final move 0.01.

## Endpoint

| metric | value |
|---|---|
| ω₁ / ω₂ / ω₃ | 163.93225938567002 / 185.21032464590309 / 401.39946949684344 |
| gap12 / gap23 | 0.12979791372346 / 1.1672629226489644 |
| volume | 0.4999984544949872 |
| M_nd | 26.341563025293 % |
| gray / mid fraction | 0.287292 / 0.118611 |
| broad-core fraction / area | 0.130139 / 1.041111 |
| max gray depth (depth/R) | 0.36667 (6.11) |
| gray components (4-conn.) / largest area | 7 / 0.79444 |
| total outer / Σ inner MMA sub-iterations | 386 / 7 300 (max 35, 0 non-converged) |
| wall / Σ tInner / Σ tEig / Σ tGrad | 2 799.28 s / 2 686.94 s / 104.83 s / 6.51 s |

The geometry metrics produced by this study's **copied** routine reproduce the
retained `gray_kkt_forensic_audit/evaluations/geometry.json["480"]` **exactly**
on every field, including the component list. The per-iteration grayness
reproduces that audit's `trajectory_480.csv` to 0 absolute difference. The
endpoint spectral recomputation (`scripts/cs_endpoint_spectral.m`) reproduces
the retained `spectral_480.mat` bitwise. The stationarity recomputation
(`scripts/cs_stationarity.py`) reproduces that audit's `stationarity.json["480"]`
exactly. The control comparison therefore uses the same definitions as the prior
audit, by proof rather than assertion.

## Non-blocking provenance defect found

The canary's `FINAL_SHA256.txt` **header** still reads
`scientific runs 0 (deployment preflight failed; canaries NOT_REACHED)`, while its
`EVIDENCE.json` records `scientific_runs: 2` and its reports describe two completed
canaries. Every file hash listed in the body of that manifest that this study
checked matches the file on disk: run CSVs, the record, C480_REPORT,
PREREGISTRATION, EFFECTIVE_CONFIG, EVIDENCE and analysis_480x60. The trajectory
hash matches EVIDENCE.json. The header is a stale template line, not evidence of
a missing run. It is recorded here and not repaired: that study is frozen.
