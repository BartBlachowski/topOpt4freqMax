# WP12 — Reference / persistence evidence audit (independent search)
READ_ONLY_INDEPENDENT_DELTA_AUDIT

Phase 2D claims that reference and persistence could not be exercised end-to-end under
Eq. (4a) because no density trajectory of reference length exists. **This audit did not
assume that search was complete.** Every `.mat` artifact in the repository was enumerated
and every numeric dataset inside it classified (`scripts/wp12_evidence_scan.py`,
results in `WP12_DENSITY_EVIDENCE_SCAN.csv`).

## What exists

| artifact | density field | snapshots retained | adequate for B_ref = 3200? |
|---|---|---|---|
| `final_campaign/raw/olhoff/s1_160x20.mat` | `res/rho_snapshots` float32 | **1601** (1600 updates) | no |
| `…/s1_240x30.mat`, `s1_320x40.mat`, `s1_400x50.mat`, `s1_720x90.mat` | float32 | 1601 | no |
| `…/s1_640x80.mat` | float32 | 1067 | no |
| `…/s1_560x70.mat` | float32 | 400 | no |
| `…/s1_480x60.mat` | float32 | 358 | no |
| `…/s1_800x100.mat` | — | zero-byte file | no |
| `targeted_replays/raw/olhoff/s1_640x80_diagnostic.mat` | float32 | 1067 | no |
| `targeted_replays/raw/yuksel/yuksel_800x100_diagnostic.mat` | `xPhysSnapshots` | 100 | no |
| `iteration_count_audit/results/*.mat` | `xPhysSnapshots` | 20–82 | no |
| `phase2b_precision/…/gray_full_24x4_h200_paired_states.mat` | double + float32 | 201 | no |
| `phase2b_precision/…/s1_transition_96x12_h320_paired_states.mat` | double + float32 | 321 | no |
| `phase2b_precision/…/prefix_validated_24x4_h5_paired_states.mat` | double + float32 | 6 | no |

**The longest density trajectory anywhere in the repository is 1601 snapshots.** The frozen
reference requires `B_ref = 3200` from a trajectory *separate from the measurement run*
(`reference.trajectory_separate_from_measurement: true`).

## The reference-length artifacts that do exist

The Phase-2B 96x12 horizon-3200 experiment produced four files. Their **complete** dataset
inventories were dumped:

| file | contents | density field present? |
|---|---|---|
| `phase2b_recheck/qualification_runs/probe_96x12_H3200.mat` | `Qhi` (3×3200), `Qlo` (3×3200), `H0`, `atrisk`, `atrisk3`, `cutTie`, `refLo/*` | **no** |
| `…/decide_96x12.mat` | `passHi`, `passLo`, `refHi/*`, `refLo/*`, `pHi/*`, `pLo/*`, `mbHi/*`, `mbLo/*`, `robHi`, `robLo` | **no** |
| `…/final_96x12.mat` | `passD`, `passS`, `refD/*`, `refS/*`, `perD/*`, `perS/*`, `mbD/*`, `mbS/*`, `rel`, `robD`, `robS` | **no** |
| `…/resolve_96x12.mat` | `Qd`, `Qh`, `Qs`, `hgD`, `hgS`, `nAt`, `nBelow`, `prefixBit`, `binDiff`, `castOk`, `sample` (45 states) | **no** |

**Phase-2D's claim is independently CONFIRMED.** The reference-length experiment retained
quality arrays, pass matrices and the reference/persistence structures, but not the density
fields. Re-evaluating it under Eq. (4a) is impossible offline.

## What can nevertheless be established from those artifacts

The frozen engines were re-implemented independently in Python from
`+ie2a/reference_phase.m`, `scan_persistence.m` and `measurement_budget.m`
(`scripts/frozen_engines.py`) and validated against the stored Phase-2B outputs:

| quantity | stored (MATLAB) | this audit (Python) | identical |
|---|---|---|---|
| `b_ref` double / single | 2200 / 2100 | **2200 / 2100** | yes |
| `Q_ref` double | 162.66009036, 163.33446321, 163.33446291 | identical to 11 digits | yes |
| `Q_ref` single | 162.66009036, 163.15340809, 163.15340743 | identical | yes |
| `k_enter` double @ .98/.99/.995 | 233 / 315 / 609 | **233 / 315 / 609** | yes |
| `k_enter` single | 232 / 309 / 524 | **232 / 309 / 524** | yes |
| `k_cert` double / single | 332,414,708 / 331,408,623 | identical | yes |
| `B_meas` double / single | 3200 / 3200 | identical | yes |

The reimplementation is therefore trustworthy, and it can be used to measure something
Phase 2D did not measure: **how much pointwise relative error in Q the frozen decisions
tolerate on a real reference-length trajectory.**

## Decision margins on the only reference-length trajectory that exists

`scripts/wp12_margins.py`; outputs `WP12_BREF_BLOCK_MARGINS.csv`,
`WP12_ACCEPTANCE_MARGINS.csv`, `WP12_CRITICAL_PERTURBATION.csv`.

**b_ref.** The freeze rule fires at the first block endpoint `b` (multiple of P = 100,
`b ≥ 600`) where `max_e gain_e(b) ≤ ε_ref = 1e-3`:

| b | max gain | candidate |
|---|---|---|
| 1900 | 2.352e-03 | no |
| 2000 | 2.352e-03 | no |
| **2100** | **1.174926e-03** | **no — misses by 1.749e-04** |
| **2200** | **2.338087e-04** | **yes — clears by 7.662e-04** |
| 2300+ | 0.000e+00 | yes |

The binding margin is the 2100 near-miss: **1.749e-04 in gain units**.

**Worst-case interval propagation.** `F` is a max of mins of `Q`, so a bounded relative
perturbation δ on every Q propagates exactly: `F → F·(1±δ)` and
`gain = 1 − F(b−L)/F(b) → 1 − r·(1∓δ)/(1±δ)`. Acceptance is `rob = min_e Q_e/Q_ref_e ≥ q`
with both numerator and reference perturbed, so `rob → rob·(1±δ)/(1∓δ)`. Bisecting for the
smallest δ that can change the outcome:

| frozen decision | critical δ | amended float32 δ = 5.596e-08 | amended double-ULP δ ≈ 8.3e-13 |
|---|---|---|---|
| `b_ref` | **8.756e-05** | safety factor **1.56e+03×** | 1.05e+08× |
| `k_enter`/`k_cert` @ q = 0.98 | **9.001e-05** | **1.61e+03×** | 1.08e+08× |
| `k_enter`/`k_cert` @ q = 0.99 | **5.162e-05** | **9.22e+02×** | 6.22e+07× |
| `k_enter`/`k_cert` @ q = 0.995 | **6.491e-05** | **1.16e+03×** | 7.82e+07× |

**Binding safety factor: 922×** (q = 0.99). This is a *worst-case adversarial* bound: it
assumes the perturbation takes its extreme value with the most damaging sign at every state
and every evaluator simultaneously. The realised safety factor is larger.

The same analysis explains the Phase-2B failure exactly. Under Eq. (4), the double-vs-single
error on this trajectory reached **2.2652e-02**, which is **439× larger** than the critical
δ of 5.162e-05, and **48.7% of all 3200 states** exceeded it. `b_ref` moved 2200 → 2100 and
`k_enter` moved at all three q levels. Nothing about that outcome was marginal.

Pointwise acceptance margins on the robust ratio, for context:

| q | min over all hard-gate states | min inside the certification window | states within 1.12e-07 of the threshold |
|---|---|---|---|
| 0.98 | 1.1467e-04 | 6.5061e-04 | **0** |
| 0.99 | 9.8308e-06 | 1.0323e-04 | **0** |
| 0.995 | 5.8020e-06 | 2.0755e-04 | **0** |

Not one of 3200 states sits within twice the amended float32 perturbation of its acceptance
threshold.

## The limitation this evidence cannot remove

These margins were measured on the **pre-amendment (Eq. 4) Q sequence**. Eq. (4a) shifts
E2/E3 levels, so the amended trajectory's margins are not these margins. What the analysis
establishes is the *order of magnitude* at which the frozen decisions operate — 1e-4 to
1e-5 — set by the ε_ref slack and by the per-iteration rate of change of Q near convergence,
neither of which is a property of the mass law. A pointwise perturbation of 5.6e-08 is three
to four orders below that scale.

**This is a bound on the amendment's numerical perturbation, not a demonstration of
end-to-end equivalence.** It is offered as what the available evidence can support, and it
is materially more than Phase 2D established.
