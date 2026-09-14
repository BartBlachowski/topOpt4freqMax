# STATIONARITY_COMPARISON — Part 10

Frozen-state evaluation only (`scripts/cs_endpoint_spectral.m`, `scripts/cs_stationarity.py`).
The spectral part of `gray_kkt_forensic_audit/scripts/frozen_evaluate.m` and the
verbatim `kkt()` algebra were used, with bound tolerance 1e-7 and each design
normalized by its own raw interior RMS. The control recomputation reproduces that
audit's `spectral_480.mat` bitwise and its `stationarity.json["480"]` exactly.

**Interpretive limits (read first).**

1. The treatment state is ρ₁₄, a mid-trajectory design 15 % of the way into stage 1,
   not an endpoint. A mid-trajectory design is expected to be nonstationary. Its
   residuals say nothing about where exact SOCP would have converged.
2. At ρ₁₄ the two lowest eigenvalues are nearly double (ω gap12 = 0.014, λ gap
   2.9 %). The audit's residual uses the simple-branch derivative (Q = diag(1,0)),
   which the audit itself flagged as inappropriate near a crossing (its 800×100 note).
   At a near-double eigenvalue, first-order optimality involves a PSD mixture of both
   branches.
3. The scales differ: raw interior RMS is 8.45e-5 (treatment) vs 1.49e-4 (control 386)
   and 1.65e-4 (control 14).

The preregistered ratio rule (≤ 0.5 improved, ≥ 2 worsened) is still applied
mechanically. Given (1)–(3), **neither the physical nor the filtered result can be
attributed to the inner solver.**

## Gray-class RMS residuals

| residual | control final (386) | control at 14 | treatment at 14 | T / C386 | T / C14 |
|---|---|---|---|---|---|
| physical, all-free dual | 0.4446 | 0.4830 | 0.9261 | 2.08 | 1.92 |
| **physical, best gray-fit dual** | 0.3340 | 0.4528 | 0.9260 | **2.77 → WORSENED** | **2.05 → WORSENED** |
| **filtered, gray-fit dual** (common raw scale) | 0.0490 | 0.4547 | 0.8379 | **17.1 → WORSENED** | **1.84 → SIMILAR** |
| filtered, all-free dual | 0.3430 | 0.4795 | 0.8416 | 2.45 | 1.76 |

## By class, physical residual with gray-fit dual (RMS; n)

| class | control 386 | control 14 | treatment 14 |
|---|---|---|---|
| gray | 0.334 (8 274) | 0.453 (23 012) | 0.926 (15 700) |
| mid | 0.210 (3 416) | 0.356 (10 072) | 0.751 (8 851) |
| solid ρ > 0.9 | 1.496 (10 262) | 1.922 (3 960) | 3.106 (7 384) |
| void ρ < 0.1 | 0.311 (10 264) | 1.229 (1 828) | 0.419 (5 716) |
| broad core | 0.172 (3 748) | 0.397 (20 208) | 0.796 (10 440) |

## Global projected sign residual (all elements)

| | control 386 | control 14 | treatment 14 |
|---|---|---|---|
| raw: RMS / best μ / lower,upper bound counts | 0.882 / 1.010 / 0, 0 | 0.860 / 1.210 / 0, 0 | 0.708 / 0.525 / 4 556, 6 488 |
| filtered: RMS / μ | 4.316 / 0 | 0.805 / 1.197 | 0.817 / 0.531 |

The treatment is the only design with elements exactly on the box (4 556 at ρ_min,
6 488 at 1). Its sign-projected residual is therefore the only one in which bound
complementarity is active. The robustness sweep over bound tolerance 1e-7 → 1e-3
leaves its values unchanged (17 756 free elements at every tolerance), because its
bound elements sit exactly at the bounds.

## Answer to the key question

*Did exact solution of problem (25) materially improve physical stationarity,
filtered stationarity, neither, or both?* **Not determinable from this run.** The
mechanical ratios say worsened (physical) and worsened/similar (filtered), but they
compare an unconverged iteration-14 design at a near-double eigenvalue with a
converged design, or with an MMA design at a different stage of evolution. No
treatment endpoint exists. Figures: `FIG_15_physical_KKT_maps`, `FIG_16_filtered_residual_maps`.
