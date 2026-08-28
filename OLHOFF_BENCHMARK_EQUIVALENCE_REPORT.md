# Olhoff benchmark-path equivalence

**Generated:** 2026-08-27 10:43:41
**Verdict:** PASS
**Profile:** `perf_r3_olhoff_du2007repro_fig3a_best_rmin2el`
**Source commit:** `cb6353feb941f12b2aaa927e622649e1ccc926f7`
**Reproduction tree hash (frozen algo/fem/filter/mma):** `d0fcc873310aeea504a84bc6b93f484b073ceecc06685cf304e1df73f82a8747`
**Benchmark path code hash:** `322816b5054234f4153458f6110658def26d5e98e1f02ed3d11edd3c75eb4d97`
**MATLAB:** 25.2.0.3042426 (R2025b) Update 1 (MACA64)

Proves, per mesh, that

```text
A  repro2007_config -> olhoffOpt                              (direct clean-room oracle)
B  run_topopt_from_json -> OlhoffDu2007Repro -> run_repro2007 -> olhoffOpt  (benchmark)
```

execute the same normalized configuration and produce the same trajectory, bit for bit. The clean-room implementation is the oracle. Tolerance is exactly zero on every compared quantity; wall-clock columns (`t_eig`, `t_grad`, `t_inner`, `elapsed_s`) are the only exclusions.

## Results

| Mesh | Config identity | History identity | Density identity | Stop identity | Verdict |
|---|---|---|---|---|---|
| 160x20 | PASS | PASS | PASS | PASS | **PASS** |
| 240x30 | PASS | PASS | PASS | PASS | **PASS** |
| 320x40 | PASS | PASS | PASS | PASS | **PASS** |
| 400x50 | PASS | PASS | PASS | PASS | **PASS** |

## Trajectories

| Mesh | Outer A | Outer B | Status | Stop reason | final max&#124;dρ&#124; | LP failures | ω₁ final | Timing admissible |
|---|---|---|---|---|---|---|---|---|
| 160x20 | 1600 | 1600 | `CAP_HIT` | `max_outer_iterations` | 0.0050000000000000001 | 0 | 163.770704 | yes |
| 240x30 | 1600 | 1600 | `CAP_HIT` | `max_outer_iterations` | 0.0050000000000000001 | 0 | 167.040892 | yes |
| 320x40 | 1600 | 1600 | `CAP_HIT` | `max_outer_iterations` | 0.0050000000000000001 | 0 | 168.225327 | yes |
| 400x50 | 627 | 627 | `SOLVER_FAILURE` | `solver_failure_subproblem` | 0 | 1 | 168.050543 | **no** |

## Identity hashes

`direct` is path A, `benchmark` is path B. Equal hashes mean bit-identical arrays, not close ones.

| Mesh | Normalized config | Trajectory (direct) | Trajectory (benchmark) | Density (direct) | Density (benchmark) |
|---|---|---|---|---|---|
| 160x20 | `f94867c485ae2f18` | `1361868a9d4c6992` | `1361868a9d4c6992` | `3d76914158f03da7` | `3d76914158f03da7` |
| 240x30 | `dd96664614112290` | `6686e0a76ba05df0` | `6686e0a76ba05df0` | `3fda9f5cd6e834da` | `3fda9f5cd6e834da` |
| 320x40 | `c6b7bfb5516df3e3` | `7b6956433cee5183` | `7b6956433cee5183` | `a09085853689dc12` | `a09085853689dc12` |
| 400x50 | `dc9e274f2b9a4ce7` | `b8f4495a6afed6ef` | `b8f4495a6afed6ef` | `09d99a46eb26ec97` | `09d99a46eb26ec97` |

## Acceptance criteria (WP10)

| Criterion | 160x20 | 240x30 | 320x40 | 400x50 |
|---|---|---|---|---|
| 1. identical effective normalized configuration | PASS | PASS | PASS | PASS |
| 2. same initial spectrum | PASS | PASS | PASS | PASS |
| 3. same numerical trajectory (zero tolerance) | PASS | PASS | PASS | PASS |
| 4. same multiplicity / eigengap logic | PASS | PASS | PASS | PASS |
| 5. same volume history | PASS | PASS | PASS | PASS |
| 6. same LP/subproblem status sequence | PASS | PASS | PASS | PASS |
| 7. same stop classification | PASS | PASS | PASS | PASS |
| 8. same final density field / checksum | PASS | PASS | PASS | PASS |
| 9. no LP failure misclassified as convergence | PASS | PASS | PASS | PASS |

## Checkpoints

Values are from path A; `match` is exact equality against path B at that outer iteration. A checkpoint past the end of the shorter run is listed as a non-match rather than skipped: both paths read NaN there, and `NaN == NaN` would otherwise be scored as agreement when in fact one path stopped and the other did not.

### 160x20

| Iter | ω₁ | ω₂ | ω₃ | N | gap_rel | objective | max&#124;dρ&#124; | vol | lp_flag | inner_conv | match | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 68.398592 | 253.385067 | 420.767152 | 1 | 2.70454 | 69.141262 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 10 | 75.325280 | 273.248305 | 426.212692 | 1 | 2.62758 | 76.066989 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 25 | 86.720244 | 301.299253 | 452.791663 | 1 | 2.47438 | 87.438436 | 0.0050000000009094514 | 0.500000 | 1 | 1 | yes | -- |
| 50 | 104.908052 | 302.888469 | 522.324674 | 1 | 1.88718 | 105.567018 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 100 | 147.763298 | 159.271670 | 327.411934 | 1 | 0.0778838 | 148.827798 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 200 | 163.649740 | 171.908728 | 297.018186 | 1 | 0.0504675 | 163.939928 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 400 | 163.680643 | 171.920559 | 296.994385 | 1 | 0.0503414 | 163.988921 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 1600 (final, both) | 163.680643 | 171.920559 | 296.994385 | 1 | 0.0503414 | 163.988921 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |

### 240x30

| Iter | ω₁ | ω₂ | ω₃ | N | gap_rel | objective | max&#124;dρ&#124; | vol | lp_flag | inner_conv | match | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 68.320854 | 252.591600 | 406.663066 | 1 | 2.69714 | 69.072904 | 0.0050000000000000678 | 0.500000 | 1 | 1 | yes | -- |
| 10 | 75.251564 | 272.598484 | 414.224034 | 1 | 2.6225 | 76.006297 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 25 | 86.758546 | 300.975477 | 442.815603 | 1 | 2.46912 | 87.495967 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 50 | 105.137222 | 302.014894 | 517.995651 | 1 | 1.87258 | 105.825784 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 100 | 149.009005 | 152.157751 | 310.124426 | 2 | 0.0211312 | 149.977256 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 200 | 165.923695 | 170.635213 | 300.296292 | 2 | 0.0283957 | 165.995687 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 400 | 167.038649 | 176.811901 | 291.910858 | 1 | 0.0585089 | 167.113461 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 1600 (final, both) | 167.042239 | 176.816443 | 292.369548 | 1 | 0.0585134 | 167.130182 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |

### 320x40

| Iter | ω₁ | ω₂ | ω₃ | N | gap_rel | objective | max&#124;dρ&#124; | vol | lp_flag | inner_conv | match | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 68.274648 | 252.068315 | 397.433936 | 1 | 2.69198 | 69.030003 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 10 | 75.207564 | 272.173035 | 406.243233 | 1 | 2.61896 | 75.967861 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 25 | 86.739015 | 300.603608 | 435.650102 | 1 | 2.46561 | 87.483768 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 50 | 105.198283 | 302.181505 | 513.082791 | 1 | 1.87249 | 105.902256 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 100 | 151.006035 | 152.172875 | 299.004664 | 2 | 0.00772711 | 151.945818 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 200 | 166.918757 | 167.353233 | 278.236585 | 2 | 0.00260292 | 167.055700 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 400 | 167.999635 | 168.512162 | 239.482187 | 2 | 0.00305077 | 168.121905 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 1600 (final, both) | 168.293196 | 169.061872 | 242.791866 | 2 | 0.00456749 | 168.413074 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |

### 400x50

| Iter | ω₁ | ω₂ | ω₃ | N | gap_rel | objective | max&#124;dρ&#124; | vol | lp_flag | inner_conv | match | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 68.241543 | 251.674531 | 390.674491 | 1 | 2.688 | 68.998369 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 10 | 75.175658 | 271.844864 | 399.692258 | 1 | 2.61613 | 75.939149 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 25 | 86.717927 | 300.303241 | 429.487845 | 1 | 2.46299 | 87.467619 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 50 | 105.186920 | 302.110382 | 508.189165 | 1 | 1.87213 | 105.895946 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 100 | 151.191461 | 152.514493 | 300.282396 | 2 | 0.0087507 | 152.059618 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 200 | 167.483051 | 167.542502 | 274.277874 | 2 | 0.000354969 | 167.738341 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 400 | 168.589733 | 168.924072 | 260.474512 | 2 | 0.00198315 | 168.708822 | 0.0050000000000000001 | 0.500000 | 1 | 1 | yes | -- |
| 627 (final, both) | 168.050543 | 168.692560 | 187.205192 | 2 | 0.00382039 | 168.050543 | 0 | 0.500000 | 0 | 0 | yes | -- |

## Divergences and subproblem failures

### 400x50

**Subproblem failures:** 1 on path A, 1 on path B; sequence identity PASS. First failure at outer iteration 627 with linprog exit flag 0.

**`LP failure -> drho = 0 -> outer stop` chain:** CONFIRMED. The run does end on a failed subproblem.

| Iter | Path | ω₁ | ω₂ | ω₃ | N | gap_rel | λ_ref | lp_flag | inner_conv | β | max&#124;dρ&#124; | vol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 621 | A | 107.284608 | 168.220216 | 168.698784 | 1 | 0.567981 | 11510 | 1 | 1 | 28345.4 | 0.0050000000000000001 | 0.500000 |
| 621 | B | 107.284608 | 168.220216 | 168.698784 | 1 | 0.567981 | 11510 | 1 | 1 | 28345.4 | 0.0050000000000000001 | 0.500000 |
| 622 | A | 168.197161 | 168.631722 | 211.956544 | 2 | 0.00258364 | 28290.3 | 1 | 1 | 28369.9 | 0.0050000000000000001 | 0.500000 |
| 622 | B | 168.197161 | 168.631722 | 211.956544 | 2 | 0.00258364 | 28290.3 | 1 | 1 | 28369.9 | 0.0050000000000000001 | 0.500000 |
| 623 | A | 105.992424 | 154.692984 | 168.238423 | 1 | 0.459472 | 11234.4 | 1 | 1 | 31783.3 | 0.0050000000000000001 | 0.500000 |
| 623 | B | 105.992424 | 154.692984 | 168.238423 | 1 | 0.459472 | 11234.4 | 1 | 1 | 31783.3 | 0.0050000000000000001 | 0.500000 |
| 624 | A | 168.005115 | 168.709123 | 185.434908 | 2 | 0.0041904 | 28225.7 | 1 | 1 | 28367.3 | 0.0050000000000000001 | 0.500000 |
| 624 | B | 168.005115 | 168.709123 | 185.434908 | 2 | 0.0041904 | 28225.7 | 1 | 1 | 28367.3 | 0.0050000000000000001 | 0.500000 |
| 625 | A | 168.158355 | 168.809697 | 215.060195 | 2 | 0.00387338 | 28277.2 | 1 | 1 | 28327.7 | 0.0050000000000000001 | 0.500000 |
| 625 | B | 168.158355 | 168.809697 | 215.060195 | 2 | 0.00387338 | 28277.2 | 1 | 1 | 28327.7 | 0.0050000000000000001 | 0.500000 |
| 626 | A | 134.976465 | 168.189442 | 168.615788 | 1 | 0.246065 | 18218.6 | 1 | 1 | 28346.9 | 0.0050000000000000001 | 0.500000 |
| 626 | B | 134.976465 | 168.189442 | 168.615788 | 1 | 0.246065 | 18218.6 | 1 | 1 | 28346.9 | 0.0050000000000000001 | 0.500000 |
| 627 | A | 168.050543 | 168.692560 | 187.205192 | 2 | 0.00382039 | 28241 | 0 | 0 | 28241 | 0 | 0.500000 |
| 627 | B | 168.050543 | 168.692560 | 187.205192 | 2 | 0.00382039 | 28241 | 0 | 0 | 28241 | 0 | 0.500000 |

## Harness validation

A check that only ever returns PASS proves nothing. These control runs perturb the benchmark path alone, on a small mesh, and ask whether the harness notices. The `rho_min` control re-injects the exact configuration-mapping defect of `DIAGNOSTIC_REPRO2007_BENCHMARK.md`.

| Control | Expected | Verdict | Config identity | First divergence | Fields |
|---|---|---|---|---|---|
| unperturbed (positive control) | PASS | **PASS** | PASS | -- | -- |
| path B move = 0.0051 | FAIL | **FAIL** | FAIL | 1 | beta, objective, vol, rV, d_inf |
| path B rho_min = 1e-06 | FAIL | **FAIL** | FAIL | 100 | beta, objective |

The `rho_min` control diverges at outer iteration 100, which is where the historical defect first became observable: the initial design is uniform rho = 0.5 and the move limit is 0.005, so no element can reach the void floor until iteration 0.5 / 0.005 = 100. The harness locates the divergence at the iteration the mechanism predicts, not merely somewhere.

## Benchmark admission

`olhoff_equivalence_gate(nelx, nely)` is the precondition for an Olhoff timing or scaling row. It re-derives the normalized configuration hash on every call and refuses the row if the benchmark path code, the frozen reproduction bytes, the profile, the task JSON, that config hash, the mesh or the MATLAB release has moved since the proof was made.

The binding is a content hash of the code that defines the path (`olhoff_benchmark_path_hash`), not the repository HEAD. HEAD would be the obvious choice and is the wrong one: committing these artifacts moves HEAD, so a HEAD binding would invalidate every proof at the moment it was archived. The commit is still recorded above as provenance.

The harness itself is inside that hash, so editing `verify_repro2007_benchmark_equivalence.m` invalidates every existing proof and forces a re-run. That is deliberate and it is the expensive direction on purpose: a weakened comparison is exactly the defect a self-certifying check cannot otherwise catch. Two of this harness's own defects were found that way -- a checkpoint past the end of the shorter run compared `NaN` against `NaN` and read as agreement, and the code hash was being overwritten by the trajectory hash because both were briefly called `benchmark_path_hash`. Presentation-only code (`olhoff_equivalence_report`, `olhoff_equivalence_gate`, `olhoff_preflight`) is outside the hash and can be edited without re-proving.

| Mesh | Verdict | Row class | Admissible | Reason if refused |
|---|---|---|---|---|
| 160x20 | PASS | `ADMISSIBLE` | yes | -- |
| 240x30 | PASS | `ADMISSIBLE` | yes | -- |
| 320x40 | PASS | `ADMISSIBLE` | yes | -- |
| 400x50 | PASS | `INVALID_SOLVER_STATUS` | **no** | SOLVER_FAILURE: the two paths agree, but the run ended on a failed subproblem, so its iteration count and wall time do not measure the method converging. |

## Provenance

- Task JSON: `examples/Performance/performance_comparison.json`, SHA-256 `204126d33ffab704`
- Protocol profile: `olhoff_du_2007_repro_fig3a_best` in `BENCHMARK_PROTOCOL_R3.md`
- Deviation from the protocol profile:
  - filter radius: executed r_min = 2.0 elements (shared benchmark cross-resolution setting); BENCHMARK_PROTOCOL_R3.md section 3.4 records r_min = 1.3 elements for profile_id olhoff_du_2007_repro_fig3a_best.  r_min = 1.3 is the radius that reproduces paper Fig. 3a; r_min = 2.0 is a valid operating point of the same method but is NOT the paper-reproduction figure.
- Overrides applied to the task JSON: `postprocessing.visualize_live = false`, `postprocessing.save_final_image = false`, `postprocessing.save_snapshot_image = false`, `optimization.filter.radius = 2`, `optimization.filter.radius_units = 'element'`, `optimization.repro2007.support_type = 'SS'`, `optimization.repro2007.move = 0.005`, `optimization.repro2007.max_outer = 1600`, `optimization.repro2007.rho_min = 1e-3`, `optimization.repro2007.tol_outer = 1e-3`
- Per-mesh records: `examples/Performance/equivalence/olhoff_equivalence_<mesh>.{json,mat}`

No file under `Matlab/reproduction2007/algo`, `fem`, `filter` or `mma` was modified. The `runner/` directory is integration code written for this repository and is excluded from the clean-room SHA-256 manifest by construction (`PROVENANCE.md`, "Import integrity").
