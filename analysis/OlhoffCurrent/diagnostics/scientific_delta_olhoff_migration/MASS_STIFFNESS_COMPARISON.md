# MASS_STIFFNESS_COMPARISON — Part 18 (low-density mode mechanism)

Scope: Olhoff only. Nothing here is extended to the Proposed method.

## 1. Measurements at frozen states (`evaluations/lowdensity_kkt.json`, `band_occupancy.json`)

Kinetic-energy fraction of the lowest three modes in elements with ρ < 0.3; *localized* = ≥ 50 %.

| state | law | ω₁, ω₂, ω₃ [rad/s] | KE in ρ<0.3 (modes 1/2/3) | localized? |
|---|---|---|---|---|
| C480 final | SIMP + eq.4b | 163.93, 185.21, 401.40 | 0.016 / 0.017 / 0.013 | no |
| C480 final | Pedersen + eq.2 | 163.65, 185.36, 405.57 | 0.020 / 0.020 / 0.015 | no |
| S480 final | SIMP + eq.4b | 166.22, 203.46, 367.85 | 0.014 / 0.015 / 0.012 | no |
| S480 final | Pedersen + eq.2 | 166.01, 203.44, 370.84 | 0.016 / 0.017 / 0.013 | no |
| **M1 final (64)** | **SIMP + eq.4b** | **34.36, 34.41, 42.69** | **1.000 / 1.000 / 1.000** | **yes, all three** |
| M1 final (64) | Pedersen + eq.2 | 163.55, 175.97, 373.49 | 0.015 / 0.015 / 0.009 | no |
| C480 iter 100 | SIMP / Pedersen | 156.16 / 155.95 | ≤ 0.05 | no |
| M1 iter 11 | SIMP / Pedersen | 150.21 / 150.02 | ≤ 0.03 | no |

Where the localized SIMP modes live (M1 final, mode 1): **99.9999 %** of kinetic energy in
0.1 < ρ ≤ 0.3; < 1e−6 in ρ ≤ 0.1. They are not in the eq.(4b) polynomial band.

## 2. Trajectory evidence

| run | law | controller | spike events ω₁(k) < 0.7 ω₁(k−1) |
|---|---|---|---|
| S480 (committed) | Pedersen + eq.2 | adaptive box | 0 |
| all 18 committed S / Rel sweep runs | Pedersen + eq.2 | adaptive box | 0 |
| **M1 (this audit)** | SIMP + eq.4b | adaptive box | **11** (iterations 20, 23, 38, 48, 50, 51, 53, 56, 58, 61, 63) |
| A2_240_d010 (committed) | SIMP + eq.4b | adaptive box | **13** |
| C3_800_d010_R06 (committed log) | SIMP + eq.4b | adaptive box | spikes to ω₁ 56–85; killed at 67 |
| C480 canary | SIMP + eq.4b | three-rung ladder | 0 |
| P1 / P2 (240, R = 0.0433) | Pedersen + eq.2 / eq.4 | adaptive box | 0 / 0 |

## 3. Mechanism (what is measured vs what is inferred)

Measured:
1. The low-gray band 0.1 < ρ ≤ 0.3 is comparably populated: C480 3.4–3.8 % and M1 3.4–6.7 % of
   elements over iterations 40–64 (both 3.4 % at M1's spikes 61 and 63, where C480 never spikes at
   any occupancy), so band occupancy alone does not predict spikes.
2. Exploratory spatial measure (not a verdict input, `island_detachment_exploratory.json`): the
   fraction of 0.1 < ρ ≤ 0.3 elements with no ρ > 0.5 element in their 3×3 neighbourhood is median
   **0.70** in M1 (0.52–0.82 at spike starts) vs **0.38** in C480 (0.18 at the end).
3. Pedersen evaluation of the very same M1 final design removes all three localized modes.

Inferred from 1–3 and the element factors: under SIMP + eq.(4b), gray islands (mass linear, stiffness
ρ³, M/K 11–100) surrounded by void of stiffness 1e−9…1e−6 are nearly unrestrained and produce
low-frequency local modes. Pedersen's void stiffness ρ/100 (1e−5…1e−4) anchors them. The adaptive
box, which moves monotone elements by up to 0.10 per iteration and contracts oscillating ones,
creates such detached islands more readily than the 0.04 ladder (M1 vs C480). The linear-mass part
of the package is not what suppresses the spikes: P1 (eq.2) and P2 (eq.4) under Pedersen are both
spike-free (MODERATE, other mesh/radius).

## 4. Conclusion for Part 18

The source's clean spectrum **is causally linked to its low-density stiffness treatment**, with
STRONG evidence at 480×60 (single-factor M1 vs S480; same-state modal energy on the M1 endpoint)
and consistent retained evidence at 240 and 800. The link is an interaction: the SIMP/4b pathology
is observed under the adaptive box and not under the three-rung ladder at 480. Whether Pedersen
would also change the three-rung ladder's gray endpoint is **not isolated**.
