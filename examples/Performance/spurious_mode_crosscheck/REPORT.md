# Cross-method verification of spurious localized eigenmodes

Offline modal re-evaluation of the 27 saved final designs of the nine-mesh benchmark.
No optimizer was run and no benchmark result was modified.
Study date: 2026-09-23.
Script: `run_spurious_mode_crosscheck.m` (modes `campaign`, `figures`, `probe`).

## Summary

The hypothesis holds for the designs examined, with one refinement. The low native frequencies of Proposed at 160×20 and 240×30 come from its material interpolation (SIMP stiffness with E_min = 10⁻⁹E₀ and linear mass) acting on elements of small but nonzero density. They do not come from a defect in the structural topology. Three results support this:

- **The structural frequency is unchanged.** On every one of the 27 saved designs, Proposed's material model still contains the structural mode. It lies within −0.11 % to +0.09 % of the E1 structural frequency and has the same shape (mass-weighted MAC ≥ 0.995 where it is found within the search).
- **The other methods' designs are affected too.** Evaluated with Proposed's model, **all nine Du–Olhoff designs** show low-density modes from about 16–17 rad/s upward. Between 572 and 756 eigenvalues lie below the structural mode. The **Yuksel–Yilmaz design at 160×20** shows 4 such modes (142–146 rad/s) below its structural mode. No other Yuksel–Yilmaz design is affected.
- **The low modes are a material-law artifact.** They fail all three tests of the unchanged E1 classifier. The lowest of them carry ≥ 99.99 % of their kinetic and ≥ 99.99 % of their strain energy in elements with x ≤ 0.1, and every rejected mode computed carries ≥ 99.5 % of its kinetic energy there. For Du–Olhoff, the same local shapes appear in the E1 spectrum (MACw 0.996–1.000). There they sit 17.7–21.2× higher in frequency, which matches the ratio predicted from the element stiffness-to-mass ratios of the two laws at the near-floor densities (10.6–22.4 for x = 2·10⁻³ … 10⁻³).

The refinement: how severe the artifact is does not follow from the material law alone. It depends on how many elements, and in how large a connected region, fall in the density band where Proposed's law has its smallest stiffness-to-mass ratio (around x ≈ 10⁻³). Du–Olhoff's design lower bound of 10⁻³ lies at exactly that minimum.

## 1. Sources and provenance (Phase 1)

| Methods | Source campaign | Records file SHA-256 | Campaign generated | Repository HEAD at run |
|---|---|---|---|---|
| Proposed, Yuksel–Yilmaz | `conference_benchmark/campaign_9mesh_r2` | `873125858df0…4921fae` | 2026-09-11T23:51:04+02:00 | `bba45e7` (dirty) |
| Du–Olhoff | `conference_benchmark/nine_mesh_pedersen_b21483b` | `261bf8fc94a1…2abe82` | 2026-09-14T05:34:28+02:00 | `b21483b` (dirty) |

- **These are the published table's sources.** Both SHA-256 hashes equal those recorded in `nine_mesh_comparison_pedersen_b21483b/benchmark_manifest.json`, the composed campaign behind the published Table 1. The script asserts this before reading any design.
- **All 27 designs are present.** Each is exactly one non-warm-up record per method and mesh, status `NATIVE_CONVERGED`.
- **The Du–Olhoff designs are the current formulation.** Every Du–Olhoff record carries `production_preset = duOlhoffPedersenAdaptiveBoxSensitivityFiltered`, which the script asserts. The Olhoff rows of `campaign_9mesh_r2` belong to the historical SIMP + Eq. (4b) implementation and are **not used**.
- **Density fields are used unchanged.** They are the saved `x` vectors, with per-design SHA-256 hashes in `results/per_case.csv`. No thresholding, filtering, clipping or rescaling is applied. The evaluator's own clamp to [0, 1] does nothing on these fields.
- **The recorded native frequencies come from these same fields.** Recomputing each design with its own native model reproduces the recorded native modes 1–3 to ≤ 5·10⁻¹⁰ relative (table C). This also covers Yuksel–Yilmaz: its saved field is `xPhys_stage2`, the field its final eigensolve used. Its final optimizer update is one step later and is not saved, but that has no bearing on any comparison here.
- **Environment.** Original campaigns: MATLAB R2025b (25.2.0.2998904), host `PMS.local`, one thread, Apple Accelerate BLAS. This study: MATLAB R2025b Update 1 (25.2.0.3042426), host `Lap.local`, one thread, repository HEAD `3402b06` (dirty). The update-level difference has no measurable effect, as the reproduction above shows.
- **Proposed/Yuksel–Yilmaz solver settings are not stored in their records.** Their `effective_config` is an empty struct. The material constants were therefore taken from the frozen profile (`benchmark_profile/study_base_config.m`: E_min ratio 10⁻⁹, ρ_min 10⁻⁹, p = 3) and **confirmed numerically** by the native-spectrum reproduction. The Du–Olhoff record stores its full configuration. Hash: `b1a5744d…d4f4`.

## 2. Material models verified in code (Phase 2)

| Model | Stiffness E(x)/E₀ | Mass ρ(x)/ρ₀ | Design lower bound | Code |
|---|---|---|---|---|
| P, Proposed native | 10⁻⁹ + (1−10⁻⁹)x³ | 10⁻⁹ + (1−10⁻⁹)x | 0 | `analysis/Proposed/Matlab/topopt_freq.m` lines 446, 439, 725–730 |
| Y, Yuksel–Yilmaz native | 10⁻⁹ + (1−10⁻⁹)x³ | 10⁻⁹ + (1−10⁻⁹)g; g = x for x > 0.1, x⁶ for x ≤ 0.1 | 0 | `analysis/Yuksel/Matlab/top99neo_inertial_freq.m` lines 976–980 |
| D, Du–Olhoff native | x³ for x ≥ 0.1; 0.01x for x < 0.1 | x | 10⁻³ | `+olh/+material/stiffnessInterpolation.m` lines 34–39; mass `eq2`, q = 1; `schema.m` line 63 |
| E1, common evaluator | 10⁻⁶ + (1−10⁻⁶)x³ | 10⁻⁶ + (1−10⁻⁶)x | — | `benchmark_profile/study_evaluate_design.m` line 186 |

**E1 classifier**, copied verbatim. A mode is structural if and only if all of the following hold:
- the eigenpair is valid: λ > 0 and relative residual ≤ 10⁻⁶;
- its diagnostics are finite;
- voidKE < 0.5;
- voidSE < 0.5;
- its kinetic-energy-weighted density is > 0.5.

Here "void" means x ≤ 0.1. The reported frequency is the lowest mode that passes.

**Discrepancies between the declared descriptions and the code** (none changes this study):

1. **Yuksel–Yilmaz mass law.** It has no continuity factor: the mass jumps by about 10⁵ at x = 0.1, and the cut-off is inclusive (x ≤ 0.1). In the benchmark's descriptions this is only called "sixth power below 0.1".
2. **Du–Olhoff has no E_min.** Its effective stiffness floor comes from the design bound: 0.01 × 10⁻³ = 10⁻⁵E₀. In the saved designs the smallest density is 1.00–1.04·10⁻³; no element sits exactly at 10⁻³.
3. **E1 is described as "linear mass" with a void-KE test only.** In code it also has a mass floor ρ_min = 10⁻⁶ and three energy tests.
4. **Proposed's solver default is ρ_min = 10⁻⁶.** The benchmark sets 10⁻⁹ through the run configuration, and the reproduction confirms that 10⁻⁹ was in effect.

## 3. Runtime probe (Phase 3)

Du–Olhoff 800×100 saved design, model P, 12 eigenpairs, 161,798 free DOFs: **assembly 0.68 s, `eigs` 1.68 s**. Peak resident memory of the whole MATLAB process was 1.84 GB, measured with `/usr/bin/time -l` rather than inside MATLAB.

The probe also showed the central result early. All 12 lowest P modes of that design are low-density modes (ω = 17.0–31.6 rad/s, voidKE ≈ 1).

## 4. Full campaign (Phase 4)

**Procedure for each of the 27 designs:**
- **Model P, adaptive:** 12 eigenpairs first, raised to 24 if no structural mode is found.
- **E1:** recomputed as a control.
- **The method's own native model:** recomputed as a control.
- **Supplementary checks, used only where the declared 24-mode search is exhausted:**
  1. A shift-invert search for 16 modes around the E1 structural eigenvalue. The unchanged classifier is applied, and the passing mode with the highest mass-weighted MAC against the E1 structural mode is taken.
  2. The exact number of eigenvalues below that mode, from the inertia of K − σM (Sylvester's law) via a sparse LDLᵀ factorization.

On every design where the structural mode was found within the search, the exact count equals the number of rejected modes, which independently confirms the count.

**Measured runtime:** 124.8 s for all 27 designs and all models; 153 s including figures and MATLAB start-up. Peak resident memory: 3.56 GB.

### Table A: frequencies (rad/s) under Proposed's material model

"P" = Proposed's native material model applied to each saved design. "Structural ω" is the first classifier-accepted mode. For Du–Olhoff it lies beyond the 24-mode search, so it comes from the targeted search, and its "index" is the exact count + 1. "Δ vs E1" compares that structural ω with the recorded E1 ω₁.

| Method | Mesh | ω₁ native | ω₁ E1 | P: lowest ω | P: lowest-mode class | P: structural ω | index | rejected below | reference | Δ vs E1 |
|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| Proposed | 160x20 | 109.05 | 153.68 | 109.05 | low-density (all 3 tests fail) | 153.50 | 11 | 10 | first within 12 | -0.11% |
| Yuksel–Yilmaz | 160x20 | 157.28 | 157.17 | 142.21 | low-density (all 3 tests fail) | 157.17 | 5 | 4 | first within 12 | 0.00% |
| Du–Olhoff | 160x20 | 169.21 | 169.20 | 16.17 | low-density (all 3 tests fail) | 169.25 | 573 | 572 | targeted (>24) | +0.03% |
| Proposed | 240x30 | 108.78 | 157.64 | 108.78 | low-density (all 3 tests fail) | 157.55 | 11 | 10 | first within 12 | -0.06% |
| Yuksel–Yilmaz | 240x30 | 159.49 | 159.44 | 159.44 | structural | 159.44 | 1 | 0 | first within 12 | 0.00% |
| Du–Olhoff | 240x30 | 167.34 | 167.33 | 16.26 | low-density (all 3 tests fail) | 167.35 | 719 | 718 | targeted (>24) | +0.01% |
| Proposed | 320x40 | 158.76 | 158.76 | 158.76 | structural | 158.76 | 1 | 0 | first within 12 | 0.00% |
| Yuksel–Yilmaz | 320x40 | 160.75 | 160.69 | 160.69 | structural | 160.69 | 1 | 0 | first within 12 | 0.00% |
| Du–Olhoff | 320x40 | 165.86 | 165.85 | 16.26 | low-density (all 3 tests fail) | 165.92 | 712 | 711 | targeted (>24) | +0.04% |
| Proposed | 400x50 | 159.52 | 159.52 | 159.52 | structural | 159.52 | 1 | 0 | first within 12 | 0.00% |
| Yuksel–Yilmaz | 400x50 | 160.06 | 159.97 | 159.97 | structural | 159.97 | 1 | 0 | first within 12 | 0.00% |
| Du–Olhoff | 400x50 | 166.46 | 166.45 | 16.51 | low-density (all 3 tests fail) | 166.59 | 757 | 756 | targeted (>24) | +0.09% |
| Proposed | 480x60 | 160.25 | 160.25 | 160.25 | structural | 160.25 | 1 | 0 | first within 12 | 0.00% |
| Yuksel–Yilmaz | 480x60 | 160.60 | 160.55 | 160.55 | structural | 160.55 | 1 | 0 | first within 12 | 0.00% |
| Du–Olhoff | 480x60 | 166.01 | 166.00 | 16.64 | low-density (all 3 tests fail) | 166.12 | 733 | 732 | targeted (>24) | +0.08% |
| Proposed | 560x70 | 160.72 | 160.72 | 160.72 | structural | 160.72 | 1 | 0 | first within 12 | 0.00% |
| Yuksel–Yilmaz | 560x70 | 160.39 | 160.34 | 160.34 | structural | 160.34 | 1 | 0 | first within 12 | 0.00% |
| Du–Olhoff | 560x70 | 165.81 | 165.80 | 16.83 | low-density (all 3 tests fail) | 165.92 | 700 | 699 | targeted (>24) | +0.07% |
| Proposed | 640x80 | 160.85 | 160.85 | 160.85 | structural | 160.85 | 1 | 0 | first within 12 | 0.00% |
| Yuksel–Yilmaz | 640x80 | 160.88 | 160.85 | 160.85 | structural | 160.85 | 1 | 0 | first within 12 | 0.00% |
| Du–Olhoff | 640x80 | 165.65 | 165.64 | 16.99 | low-density (all 3 tests fail) | 165.75 | 667 | 666 | targeted (>24) | +0.07% |
| Proposed | 720x90 | 161.07 | 161.07 | 161.07 | structural | 161.07 | 1 | 0 | first within 12 | 0.00% |
| Yuksel–Yilmaz | 720x90 | 160.62 | 160.60 | 160.60 | structural | 160.60 | 1 | 0 | first within 12 | 0.00% |
| Du–Olhoff | 720x90 | 165.42 | 165.41 | 17.10 | low-density (all 3 tests fail) | 165.52 | 620 | 619 | targeted (>24) | +0.07% |
| Proposed | 800x100 | 161.36 | 161.36 | 161.36 | structural | 161.36 | 1 | 0 | first within 12 | 0.00% |
| Yuksel–Yilmaz | 800x100 | 160.89 | 160.86 | 160.86 | structural | 160.86 | 1 | 0 | first within 12 | 0.00% |
| Du–Olhoff | 800x100 | 165.43 | 165.42 | 17.02 | low-density (all 3 tests fail) | 165.52 | 589 | 588 | targeted (>24) | +0.06% |

### Table B: low-density content of the saved designs (share of elements)

The band 3·10⁻⁴ ≤ x ≤ 3·10⁻³ spans half a decade either side of the minimum of Proposed's element stiffness-to-mass ratio, at x ≈ 8·10⁻⁴ (Fig. F2).

| Method | Mesh | x = 0 | 0 < x < 0.01 | 0.01 ≤ x < 0.1 | 3·10⁻⁴ ≤ x ≤ 3·10⁻³ | 10⁻³ ≤ x < 1.1·10⁻³ | min x | grayness |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed | 160x20 | 3.5% | 22.6% | 9.0% | 4.5% | 0.1% | 0.00e+00 | 0.256 |
| Yuksel–Yilmaz | 160x20 | 14.9% | 28.1% | 2.6% | 1.9% | 0.0% | 0.00e+00 | 0.081 |
| Du–Olhoff | 160x20 | 0.0% | 35.9% | 6.1% | 34.8% | 26.1% | 1.01e-03 | 0.115 |
| Proposed | 240x30 | 14.4% | 18.9% | 7.2% | 3.1% | 0.1% | 0.00e+00 | 0.162 |
| Yuksel–Yilmaz | 240x30 | 25.0% | 21.4% | 1.1% | 0.8% | 0.0% | 0.00e+00 | 0.043 |
| Du–Olhoff | 240x30 | 0.0% | 38.8% | 3.6% | 38.3% | 7.3% | 1.00e-03 | 0.123 |
| Proposed | 320x40 | 27.1% | 10.8% | 4.7% | 0.9% | 0.0% | 0.00e+00 | 0.122 |
| Yuksel–Yilmaz | 320x40 | 32.4% | 15.0% | 0.8% | 0.6% | 0.0% | 0.00e+00 | 0.034 |
| Du–Olhoff | 320x40 | 0.0% | 37.5% | 4.0% | 36.7% | 8.0% | 1.01e-03 | 0.141 |
| Proposed | 400x50 | 30.8% | 9.1% | 4.0% | 0.7% | 0.0% | 0.00e+00 | 0.099 |
| Yuksel–Yilmaz | 400x50 | 32.9% | 13.8% | 1.0% | 0.4% | 0.0% | 0.00e+00 | 0.037 |
| Du–Olhoff | 400x50 | 0.0% | 39.2% | 3.6% | 38.5% | 6.4% | 1.01e-03 | 0.122 |
| Proposed | 480x60 | 32.7% | 8.1% | 3.5% | 0.7% | 0.0% | 0.00e+00 | 0.092 |
| Yuksel–Yilmaz | 480x60 | 36.1% | 11.7% | 0.6% | 0.4% | 0.0% | 0.00e+00 | 0.029 |
| Du–Olhoff | 480x60 | 0.0% | 38.8% | 3.8% | 38.0% | 5.6% | 1.01e-03 | 0.131 |
| Proposed | 560x70 | 35.5% | 6.8% | 2.9% | 0.5% | 0.0% | 0.00e+00 | 0.078 |
| Yuksel–Yilmaz | 560x70 | 39.0% | 9.1% | 0.6% | 0.3% | 0.0% | 0.00e+00 | 0.023 |
| Du–Olhoff | 560x70 | 0.0% | 38.7% | 3.7% | 37.9% | 4.9% | 1.02e-03 | 0.133 |
| Proposed | 640x80 | 37.9% | 5.5% | 2.4% | 0.5% | 0.0% | 0.00e+00 | 0.069 |
| Yuksel–Yilmaz | 640x80 | 41.7% | 6.8% | 0.3% | 0.1% | 0.0% | 0.00e+00 | 0.020 |
| Du–Olhoff | 640x80 | 0.0% | 38.8% | 3.6% | 37.2% | 4.2% | 1.02e-03 | 0.133 |
| Proposed | 720x90 | 39.1% | 4.8% | 2.4% | 0.4% | 0.0% | 0.00e+00 | 0.063 |
| Yuksel–Yilmaz | 720x90 | 41.5% | 7.3% | 0.3% | 0.2% | 0.0% | 0.00e+00 | 0.015 |
| Du–Olhoff | 720x90 | 0.0% | 36.5% | 4.3% | 34.7% | 3.4% | 1.03e-03 | 0.162 |
| Proposed | 800x100 | 40.6% | 4.0% | 2.1% | 0.4% | 0.0% | 0.00e+00 | 0.055 |
| Yuksel–Yilmaz | 800x100 | 43.0% | 5.9% | 0.3% | 0.1% | 0.0% | 0.00e+00 | 0.014 |
| Du–Olhoff | 800x100 | 0.0% | 36.4% | 4.3% | 33.8% | 2.4% | 1.04e-03 | 0.165 |

### Table C: controls and modal correspondence

"MACw" is MAC weighted by the E1 mass matrix of the same field. Unweighted MAC is also stored in `per_case.csv`. It understates correspondence here, because the near-massless nodes of model P move with very large amplitude (e.g. 0.685 vs 0.995 for Proposed 160×20). The stringent re-solve used tol 10⁻¹⁴ and an independent start vector, and was run on every case whose lowest mode is rejected or differs from E1 by more than 5 %. No classification changed in any stringent re-solve.

| Method | Mesh | native modes 1–3 max rel. diff | E1 rel. diff | MACw(P structural, E1) | MACw(P mode 1, best E1) | matched E1 ω | E1 ω / P ω₁ | stringent re-solve Δω |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed | 160x20 | 1.4e-15 | 1.8e-16 | 0.995 | 0.343 | 274.97 | 2.52 | 2.2e-15 |
| Yuksel–Yilmaz | 160x20 | 8.2e-13 | 1.8e-16 | 1.000 | 0.000 | 592.76 | 4.17 | 1.8e-15 |
| Du–Olhoff | 160x20 | 3.1e-11 | 1.7e-16 | 0.862 | 0.996 | 342.86 | 21.20 | 1.3e-15 |
| Proposed | 240x30 | 1.9e-15 | 1.8e-16 | 0.997 | 0.477 | 284.08 | 2.61 | 2.5e-15 |
| Yuksel–Yilmaz | 240x30 | 3.0e-11 | 1.8e-16 | 1.000 | 1.000 | 159.44 | 1.00 | not needed |
| Du–Olhoff | 240x30 | 3.0e-11 | 5.1e-16 | 0.756 | 1.000 | 333.62 | 20.52 | 2.0e-15 |
| Proposed | 320x40 | 6.3e-13 | 0.0e+00 | 1.000 | 1.000 | 158.76 | 1.00 | not needed |
| Yuksel–Yilmaz | 320x40 | 3.4e-12 | 1.8e-16 | 1.000 | 1.000 | 160.69 | 1.00 | not needed |
| Du–Olhoff | 320x40 | 1.3e-10 | 1.7e-16 | 0.960 | 0.999 | 333.32 | 20.50 | 1.1e-15 |
| Proposed | 400x50 | 8.4e-13 | 3.6e-16 | 1.000 | 1.000 | 159.52 | 1.00 | not needed |
| Yuksel–Yilmaz | 400x50 | 1.0e-11 | 0.0e+00 | 1.000 | 1.000 | 159.97 | 1.00 | not needed |
| Du–Olhoff | 400x50 | 1.2e-10 | 1.7e-16 | 0.627 | 0.999 | 325.92 | 19.74 | 1.5e-15 |
| Proposed | 480x60 | 4.1e-13 | 3.5e-16 | 1.000 | 1.000 | 160.25 | 1.00 | not needed |
| Yuksel–Yilmaz | 480x60 | 1.5e-10 | 1.8e-16 | 1.000 | 1.000 | 160.55 | 1.00 | not needed |
| Du–Olhoff | 480x60 | 1.1e-10 | 3.4e-16 | 0.755 | 0.998 | 321.10 | 19.30 | 1.8e-15 |
| Proposed | 560x70 | 2.4e-13 | 0.0e+00 | 1.000 | 1.000 | 160.72 | 1.00 | not needed |
| Yuksel–Yilmaz | 560x70 | 9.9e-11 | 1.8e-16 | 1.000 | 1.000 | 160.34 | 1.00 | not needed |
| Du–Olhoff | 560x70 | 2.6e-10 | 1.7e-16 | 0.945 | 0.998 | 316.62 | 18.81 | 2.4e-15 |
| Proposed | 640x80 | 4.1e-13 | 1.8e-16 | 1.000 | 1.000 | 160.85 | 1.00 | not needed |
| Yuksel–Yilmaz | 640x80 | 1.8e-11 | 1.8e-16 | 1.000 | 1.000 | 160.85 | 1.00 | not needed |
| Du–Olhoff | 640x80 | 5.2e-10 | 3.4e-16 | 0.885 | 0.998 | 312.42 | 18.39 | 1.5e-15 |
| Proposed | 720x90 | 1.0e-13 | 1.8e-16 | 1.000 | 1.000 | 161.07 | 1.00 | not needed |
| Yuksel–Yilmaz | 720x90 | 2.0e-10 | 1.8e-16 | 1.000 | 1.000 | 160.60 | 1.00 | not needed |
| Du–Olhoff | 720x90 | 2.4e-10 | 0.0e+00 | 0.988 | 0.997 | 306.71 | 17.93 | 1.5e-15 |
| Proposed | 800x100 | 4.0e-13 | 0.0e+00 | 1.000 | 1.000 | 161.36 | 1.00 | not needed |
| Yuksel–Yilmaz | 800x100 | 4.7e-11 | 1.8e-16 | 1.000 | 1.000 | 160.86 | 1.00 | not needed |
| Du–Olhoff | 800x100 | 4.5e-10 | 0.0e+00 | 0.912 | 0.997 | 300.45 | 17.65 | 1.4e-15 |

## 5. Controls (Phase 5)

1. **Native reproduction.** Every design's own native model reproduces its recorded native modes 1–3 to ≤ 5.2·10⁻¹⁰ relative. This includes Proposed 160×20 and 240×30, whose recorded ω₁ = 109.05 and 108.78 are reproduced to about 10⁻¹⁵. The copied E1 reproduces all 27 recorded E1 values to ≤ 5.1·10⁻¹⁶, and puts the structural mode first on all 27.
2. **x = 0 is not the problem.** In model P the element stiffness-to-mass ratio relative to solid is exactly **1 at x = 0**, the same as solid material. It then falls to **2.0·10⁻⁶ at x = 10⁻³**, rises to 10⁻⁴ at x = 10⁻², and to 10⁻² at x = 0.1 (`results/stiffness_to_mass_ratio.csv`, Fig. F2). The artifact therefore comes from small positive densities, not from empty elements.
3. **Same field everywhere.** Each design's analyses (P, E1, native, targeted, figures) all read one saved vector, whose hash is recorded.
4. **Eigensolver.** Every computed eigenpair has a relative residual ≤ 9.4·10⁻⁸. The stringent, independently started re-solves change no frequency by more than 2.5·10⁻¹⁵ relative and no classification.
5. **Mode shapes and energy localization (Fig. F3).** The rejected lowest modes put all of their kinetic energy into the central low-density region. For Proposed that region is the gray ellipse (0 < x ≲ 0.1), for Yuksel–Yilmaz 160×20 small near-void pockets, and for Du–Olhoff the whole floor-density region. The accepted modes carry their energy in the solid members.

## 6. Answers to the scientific questions (Phase 6)

**Do the saved Du–Olhoff designs develop low-frequency localized modes under Proposed's material model? At which meshes?**

*Measured:* yes, at **all nine meshes**. The lowest P mode is at 16.2–17.1 rad/s, about 1/10 of the structural frequency. Between 572 and 756 eigenvalues lie below the structural mode, which itself is unchanged (+0.01 % to +0.09 % vs E1). About 34–39 % of their elements lie in 3·10⁻⁴…3·10⁻³, right at the minimum of model P's stiffness-to-mass ratio.

**Does the same happen to the Yuksel–Yilmaz designs, and is it related to their density distribution?**

*Measured:* only at **160×20**, with 4 rejected modes at 142.2–146.2 rad/s below the structural 157.17 rad/s. No Yuksel–Yilmaz design at 240×30 or finer shows any.

*Interpretation, plausible but not established:* this tracks the fraction of elements in the critical band, 1.9 % at 160×20 against ≤ 0.8 % from 240×30 on. Proposed shows the same pattern: 4.5 % and 3.1 % (affected) against ≤ 0.9 % (unaffected). The share of elements with 0 < x < 0.01 does **not** separate the cases on its own: Yuksel–Yilmaz 240×30 has 21.4 % yet is unaffected. What matters is the size and connectedness of a region at the critical density, not only how many such elements there are. That spatial measure was not quantified here.

**Does replacing the native interpolation substantially change the ordering or interpretation of the lowest frequencies?**

*Measured:* the **ordering** changes drastically for Du–Olhoff, where the structural mode drops from 1st to between 573rd and 757th, and for Yuksel–Yilmaz 160×20, from 1st to 5th. For Proposed nothing changes, because P is its native model. The **structural frequency itself** changes by at most 0.11 % in any design. So the lowest eigenvalue is not a measure of the structure under model P unless it passes the classifier. Under E1 all 27 designs have the structural mode as mode 1.

**Are the low modes supported by the energy-based classifier and visual inspection?**

*Measured:* yes. Every rejected lowest mode fails all three tests at once (voidKE ≥ 0.99999, voidSE ≥ 0.99997, KE-weighted density 0.0008–0.003), and Fig. F3 shows its energy confined to the low-density region. The frequency alone was never used as a criterion.

**Does the evidence establish an interpolation-dependent artifact, or are topology-dependent effects also needed?**

*Established:* the modes depend on the interpolation. The same saved field has them under P and not as its lowest mode under E1 or under its own native model. For Du–Olhoff, the same local shapes move up by the factor the element stiffness-to-mass ratios predict. The structural mode and its frequency do not depend on the interpolation to within 0.11 %.

*Also required:* a design-dependent factor. Whether the artifact appears, and how strongly, depends on the design containing a sufficiently large region at densities near 10⁻³. That region exists in every Du–Olhoff design (its floor), in Proposed only at the two coarsest meshes, and in Yuksel–Yilmaz only at 160×20. This concerns the low-density content of the design, not its load-carrying topology.

**Unresolved:**
- Why the targeted structural modes of Du–Olhoff have MACw only 0.63–0.99 against E1 at some meshes, although the frequency agrees to < 0.1 %. Mixing with nearby low-density modes at almost the same frequency is plausible but was not tested.
- No spatial (cluster-size) statistic of the critical band was computed.
- The Proposed 160×20 low modes have no clear one-to-one E1 counterpart (MACw ≤ 0.48), so the frequency-scaling test was only applied to Du–Olhoff.

## 7. Publication-ready interpretation

> The low native first eigenfrequency of the proposed method at the two coarsest meshes (109.05 and 108.78 rad/s) is not a property of the optimized structure. When each saved design is re-evaluated with a common material model, the structural first frequency is 153.68 and 157.64 rad/s, and it is the lowest mode. The low native values are localized modes of elements with small positive density, which the SIMP interpolation with linear mass makes too compliant for their mass: at x = 10⁻³ the element stiffness-to-mass ratio is 2·10⁻⁶ of that of solid material. The effect is not specific to our designs. With the same material model, every Du–Olhoff design, whose void elements lie at its lower density bound of 10⁻³, shows hundreds of such modes below the structural one, starting at about 16 rad/s, and so does the coarsest Yuksel–Yilmaz design. In all cases these modes carry more than 99 % of their kinetic energy in elements with x ≤ 0.1, while the structural mode and its frequency are unchanged to within 0.11 %. Methods that evaluate eigenmodes during optimization avoid the artifact through their low-density interpolations. The proposed method performs a single reference eigenanalysis on the solid design and optimizes against a load derived from it, so its optimization never depends on the eigenmodes of intermediate low-density designs. The artifact only appears when the final design is evaluated with the plain interpolation, which is why Table 1 reports the common-evaluator frequency.

## 8. Files

| File | Content |
|---|---|
| `run_spurious_mode_crosscheck.m` | Analysis script (`probe`, `campaign`, `figures`). |
| `results/per_case.csv`, `per_case.json` | One row per design: all frequencies, classes, MACs, counts, density bands, control results, field hash. |
| `results/modes_all.csv` | Every computed mode of every model: ω, residual, voidKE, voidSE, KE-weighted density, IPR, classification. |
| `results/stiffness_to_mass_ratio.csv` | Control 2: element stiffness-to-mass ratio of P, Y, D and E1 against x. |
| `results/timing.csv`, `probe_800x100_olhoff_P.json` | Assembly and `eigs` times. |
| `results/provenance.json` | Sources, hashes, campaign environments, method configurations, study environment and runtime. |
| `results/results.mat` | Everything above plus MAC matrices (unweighted and E1-mass-weighted). |
| `results/campaign_log.txt` | Console log with the `/usr/bin/time -l` output. |
| `figures/F1–F4` (.png + .fig) | F1: frequencies by model. F2: stiffness-to-mass ratio. F3: density and modal kinetic-energy maps. F4: low-density fractions. |

`provenance.json` records the script hash of the campaign run. The `figures` step was re-run afterwards with changes to figure layout only, so the current script hash differs from the recorded one, but no computed value changed. `*.png`, `*.fig` and `*.mat` are git-ignored in this repository, so they exist on this machine only.
