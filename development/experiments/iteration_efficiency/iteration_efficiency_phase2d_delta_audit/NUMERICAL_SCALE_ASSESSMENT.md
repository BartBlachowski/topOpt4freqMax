# WP11 — Numerical scale adequacy at the frozen quality scale
READ_ONLY_INDEPENDENT_DELTA_AUDIT

All figures below are this audit's own recomputation from stored densities, not Phase-2D's.
Ratios, not adjectives.

## Perturbation magnitudes, independently measured

| mechanism | evidence | Eq. (4) | Eq. (4a) | branch-free E1 control |
|---|---|---|---|---|
| branch straddle, nextbelow(0.1) → nextabove(0.1), all at-branch elements | 9 states, 96x12 | **4.0021e-03** | **8.2820e-13** | 8.1090e-13 |
| the same, a **single** at-branch element | 96x12 k = 204 | **2.5213e-06** | 3.5714e-16 | 1.1165e-13 |
| float32 storage, genuine paired states | 236 states (24x4 + 96x12) | **2.6736e-02** | **5.5960e-08** | 5.5949e-08 |
| branch side, production mesh | 1600 states, 160x20 | **2.6496e-02** | **2.6560e-10** | 2.6252e-10 |

The amended E2 figures track the E1 control to within a factor of 1.012 on the trajectory
and 1.0002 on the paired states. **The amended residual is the generic float32 / eigensolver
floor, not a branch effect.** The mechanism is gone, not merely reduced.

*Precision of claim.* The "new" double-ULP figures (Phase-2D 2.1551e-13, this audit
8.2820e-13) sit at the **eigensolver differencing floor** — both implementations report the
amended E2 value as indistinguishable from their own E1 value. Neither number should be
quoted as a measured physical sensitivity. The defensible statement is: *the amended
double-ULP sensitivity is at or below the eigensolver reproducibility floor of ~1e-12.* This
does not weaken the conclusion; it is still nine orders below the tightest band.

## Against the frozen acceptance bands

| q | band | Eq. (4) ULP / band | Eq. (4a) ULP / band | Eq. (4) f32 / band | Eq. (4a) f32 / band |
|---|---|---|---|---|---|
| 0.98 | 0.02 | 0.2001 | 4.14e-11 | **1.3368** | 2.798e-06 |
| 0.99 | 0.01 | 0.4002 | 8.28e-11 | **2.6736** | 5.596e-06 |
| **0.995** | **0.005** | **0.8004** | **1.657e-10** | **5.3473** | **1.119e-05** |

Under Eq. (4) a two-ULP branch straddle consumed **80% of the tightest band** and float32
storage **exceeded it 5.35-fold**. Under Eq. (4a) the same quantities are
**1.66e-10** and **1.12e-05** of the band.

Maximum amended perturbation as a fraction of the 0.5% band: **1.119e-05**, i.e.
**89 400× smaller than the band**.

## Against actual decision margins — the number that matters

A band width is not a margin. What decides `k_enter` is how close the robust ratio lies to
`q`, and what decides `b_ref` is how close `max_e gain_e(b)` lies to `ε_ref`. Measured on the
only reference-length trajectory in the repository (96x12, 3200 updates):

| frozen decision | critical relative Q perturbation δ* | amended f32 5.596e-08 | amended ULP ≈8.3e-13 |
|---|---|---|---|
| `b_ref` | 8.756e-05 | **1560×** margin | 1.05e+08× |
| `k_enter`/`k_cert` q = 0.98 | 9.001e-05 | **1610×** | 1.08e+08× |
| **`k_enter`/`k_cert` q = 0.99** | **5.162e-05** | **922× (binding)** | 6.22e+07× |
| `k_enter`/`k_cert` q = 0.995 | 6.491e-05 | **1160×** | 7.82e+07× |

- Tightest pointwise acceptance margin over 3200 states: **5.802e-06** (q = 0.995).
- States within twice the amended float32 perturbation of an acceptance threshold: **0 of
  3200**, at every q.
- Under Eq. (4) the double-vs-single error reached **2.2652e-02** = **439× the binding
  critical δ**, exceeded on **48.7%** of states. `b_ref` moved 2200 → 2100 and `k_enter`
  moved at all three q levels. The failure was not marginal, and neither is the cure.

## Ruling

    NUMERICAL_STABILITY_AT_FROZEN_SCALE = PASS

Eq. (4a) reduces the storage- and branch-induced perturbation of the common evaluator to the
level of the branch-free E1 control, three to four orders of magnitude below the smallest
decision margin observed on real reference-length evidence.

**This ruling concerns perturbation stability only.** It says nothing about whether the
amended evaluator returns the correct quantity. It does not, at a demonstrable set of
states — see finding **D1**, which is a *level-validity* defect, not a stability defect, and
is three orders of magnitude larger than anything in this table.
