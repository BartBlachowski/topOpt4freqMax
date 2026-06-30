# Exp3 Fundamental Forensic Audit

Scope: saved Exp3 alpha=1.00 mesh-convergence artifacts and source/config inspection only. No experiment reruns, no solver-code edits, and no manuscript edits were performed.

Primary inspected artifacts:

- `examples/Revision_v1/output/exp3_authoritative_mesh_convergence/mesh_200x25/exp3_authoritative_200x25_*`
- `examples/Revision_v1/output/exp3_authoritative_mesh_convergence/mesh_400x50/exp3_authoritative_400x50_*`
- `examples/Revision_v1/output/exp3_authoritative_mesh_convergence/exp3_authoritative_mesh_convergence_result.json`
- `examples/Revision_v1/output/exp3_authoritative_mesh_convergence/exp3_authoritative_mesh_convergence_manifest.json`
- `examples/Revision_v1/exp3_authoritative_mesh_convergence.m`
- `tools/Matlab/run_topopt_from_json.m`
- `analysis/ourApproach/Matlab/topopt_freq.m`
- `tools/Matlab/supportsToFixedDofs.m`

## Executive Finding

The 200x25 vs 400x50 discrepancy is **not explained by a basic setup inconsistency** in geometry, supports, material, optimizer settings, reference-load definition, or physical filter radius. The 400x50 case is an exact 2x refinement in both directions and uses the same physical problem definition.

The current Exp3 result should be classified as **inconclusive**, not as an invalid setup and not as proven genuine mesh non-convergence. It is invalid as mesh-convergence evidence because the fine case fails the declared final MAC gate and the two final topologies/frequencies diverge strongly, but the saved artifacts do not prove a single definitive physical cause.

## Pass/Fail Table

| # | Category | Status | Forensic conclusion |
|---:|---|---|---|
| 1 | Geometry and mesh scaling | PASS | Same `L=8`, `H=1`, thickness `1`; 400x50 is exactly 2x refinement of 200x25. |
| 2 | Boundary conditions | PASS | Same two vertical clamp lines at `x=0` and `x=8`, same `ux,uy` DOFs; support nodes refine physically as expected. |
| 3 | Loads/reference modes | PASS | Same active alpha=1 semi-harmonic solid-reference mode 1; reference frequencies agree within normal discretization error. |
| 4 | Filter | PASS with warning | Same physical radius `0.08`; element radius changes from 2 to 4 as intended. Warning: `filter.boundary_condition="symmetric"` is not used by `ourApproach`; actual edge treatment is truncated-neighborhood normalization by `Hs`. |
| 5 | Material/interpolation | PASS | Same SIMP, `penal=3`, `E_min_ratio=1e-6`, `rho_min=1e-6`, linear mass interpolation, no Heaviside. |
| 6 | Optimizer/settings | PASS | Same MMA, move limit, cap, tolerance, uniform initialization, no continuation. Main path is deterministic. |
| 7 | Assembly/eigensolve sanity | PASS with limitation | Initial uniform-design frequencies and solid-reference frequencies are close across meshes; eigensolve settings match. Only 6 final tracked modes are saved, not 10. |
| 8 | Result validity | FAIL | Fine mesh final MAC is `0.7861266876 < 0.8`; final frequencies and topologies differ substantially. |
| 9 | Artifact integrity | PASS with warning | Exp3 paths, topology CSV/JSON consistency, timestamps, and hashes are coherent; no CR2 topology hash match. Warning: manifests do not store config hashes, so hashes were computed during this audit. |

## 1. Geometry and Mesh Scaling

| field | 200x25 | 400x50 | verdict |
|---|---:|---:|---|
| `L` | 8 | 8 | match |
| `H` | 1 | 1 | match |
| thickness | 1 | 1 | match |
| `nelx` | 200 | 400 | 2x |
| `nely` | 25 | 50 | 2x |
| `hx=L/nelx` | 0.04 | 0.02 | 2x refinement |
| `hy=H/nely` | 0.04 | 0.02 | 2x refinement |
| element aspect `hx/hy` | 1.0 | 1.0 | match |

Coordinate convention is column-major node numbering:

- node `n = i*(nely+1) + j + 1`
- `x = i*(L/nelx)`
- `y = j*(H/nely)`
- `ux = 2*n - 1`, `uy = 2*n`

Source confirmation: `supportsToFixedDofs.m` documents and implements this convention, and `topopt_freq.m` computes `hx=L/nelx`, `hy=H/nely`.

## 2. Boundary Conditions

Both generated Exp3 configs contain the same supports:

- vertical line at `x=0`, DOFs `[ux, uy]`, tolerance `1e-6`
- vertical line at `x=8`, DOFs `[ux, uy]`, tolerance `1e-6`

The run log confirms:

| mesh | left-line nodes | right-line nodes | total support nodes | fixed DOFs |
|---|---:|---:|---:|---:|
| 200x25 | 26 | 26 | 52 | 104 |
| 400x50 | 51 | 51 | 102 | 204 |

Support-node coordinates:

| mesh | left nodes | right nodes | y coordinates |
|---|---|---|---|
| 200x25 | `1:26` | `5201:5226` | `0:0.04:1` |
| 400x50 | `1:51` | `20401:20451` | `0:0.02:1` |

DOF ranges:

| mesh | left first/last DOFs | right first/last DOFs |
|---|---|---|
| 200x25 | `(1,2)` to `(51,52)` | `(10401,10402)` to `(10451,10452)` |
| 400x50 | `(1,2)` to `(101,102)` | `(40801,40802)` to `(40901,40902)` |

Verdict: the supports are physically identical after refinement.

## 3. Loads and Reference Modes

Generated Exp3 load cases are identical except for mesh:

| load case | factor | load type | mode | load factor |
|---|---:|---|---:|---:|
| `alpha1.00_solid_reference_mode_1` | 1 | `semi_harmonic` | 1 | 1 |
| `alpha0.00_solid_reference_mode_2` | 0 | `semi_harmonic` | 2 | 1 |

The active load is `F(x) = omega0^2 * M(x) * Phi0`, with `semi_harmonic_baseline="solid"`, `harmonic_normalize=false`, and `load_sensitivity="omitted"`.

Saved MAT diagnostics:

| quantity | 200x25 | 400x50 | relative difference |
|---|---:|---:|---:|
| `omega0_1` | 291.035039725037 | 290.887159536947 | 0.000508 |
| `omega0_1^2` | 84701.3943477539 | 84615.3395834731 | 0.001016 |
| `omega0_2` | 725.785322972172 | 725.341706730345 | 0.000611 |
| `omega0_2^2` | 526764.335041821 | 526120.59152249 | 0.001222 |
| reference modal masses | `[1, 1]` | `[1, 1]` | match |

`Phi0` normalization and phase convention:

- Reference modes are mass-normalized by `mass_normalize_modes(phi, Mf)`.
- Phase/sign is made deterministic by `orient_modes_deterministic(phi)`.
- Saved reference modal masses are exactly `1` to printed precision.

Verdict: same physical reference mode is used. No reference-mode mismatch was found.

## 4. Filter

| quantity | 200x25 | 400x50 | verdict |
|---|---:|---:|---|
| config radius | 0.08 physical | 0.08 physical | match |
| radius in elements | 2 | 4 | correct 2x refinement |
| filter type | sensitivity | sensitivity | match |
| Heaviside | false | false | match |
| density/projection filter | no | no | match |

Source path:

- `run_topopt_from_json.m` converts physical radius by `rmin_elem = filterRadius / dx`, `rmin_phys = filterRadius`.
- `topopt_freq.m` builds weights as `rmin - sqrt(dx^2 + dy^2)` and normalizes by `Hs=sum(Hf,2)`.
- With sensitivity filtering, `xPhys=x`; only sensitivities are filtered.

Exact warning: the generated configs say `filter.boundary_condition="symmetric"`, but for `ourApproach` only `filter.type` is mapped to `ft`; `filterBC` is not passed into `topopt_freq`. The actual edge behavior is truncated-neighborhood weighting with local denominator normalization. This is a config/source mismatch in wording, but it is the same for both meshes and is not a 200x25-vs-400x50 inconsistency.

## 5. Material and Interpolation

| field | value |
|---|---:|
| material model | `plane_stress_isotropic` |
| `E` | `1e7` |
| `nu` | `0.3` |
| `rho` | `1` |
| `E_min_ratio` | `1e-6` |
| `Emin` | `10` |
| `rho_min` | `1e-6` |
| stiffness interpolation | SIMP |
| penalization | `3` |
| mass interpolation | linear, `pmass=1` default |
| Heaviside/projection | off |
| filter mode | sensitivity filter, not density filter |

Verdict: no material/interpolation mismatch.

## 6. Optimizer and Settings

| setting | value |
|---|---:|
| optimizer | MMA |
| move limit | 0.2 |
| max iterations | 2000 |
| convergence tolerance | 0.001 |
| volume fraction | 0.5 |
| initialization | uniform `x=0.5` for all active elements |
| continuation | none found |
| random seed | not used in main path; debug finite-difference helpers contain fixed RNG calls but are not invoked here |
| deterministic status | deterministic for inspected Exp3 path |

Final case states:

| mesh | classification | iterations | design change | feasibility |
|---|---|---:|---:|---:|
| 200x25 | accepted | 1052/2000 | 0.000166189945092 | 0 |
| 400x50 | mode invalid | 1750/2000 | 0.000903109639332 | 0 |

Both runs stopped before the iteration cap and satisfy the scalar design-change and feasibility gates. However, the last-window design changes remain non-negligible:

| metric | 200x25 | 400x50 |
|---|---:|---:|
| last-10 mean design change | 0.0125422569905 | 0.00755467376873 |
| last-10 max design change | 0.019854613705 | 0.0144387053195 |
| last-100 max design change | 0.0237162017603 | 0.0154032323831 |

This weakens convergence quality, but it does not identify a coarse/fine setup mismatch.

## 7. Assembly and Eigensolve Sanity

Saved initial uniform-design tracked frequencies, from iteration 1:

| mode | 200x25 omega | 400x50 omega | relative difference |
|---:|---:|---:|---:|
| 1 | 145.517956412906 | 145.444016092353 | 0.000508 |
| 2 | 362.893750160913 | 362.671941372697 | 0.000611 |
| 3 | 622.540123782176 | 622.507631502438 | 0.000052 |
| 4 | 640.900927085464 | 640.440245876627 | 0.000719 |
| 5 | 955.204988959105 | 954.407819787621 | 0.000834 |
| 6 | 1243.03906741613 | 1242.94018093752 | 0.000080 |

Saved solid-reference frequencies are listed in section 3 and also match closely.

Eigensolve settings in the inspected path:

- `localCurrentModesFromSubmatrices` requests `eigs(Kf, Mf, nReq, 'smallestabs')`.
- Primary options: `disp=0`, `maxit=2000`, `tol=1e-12`.
- On catch, it falls back to `eigs(..., 'smallestabs')`.
- Modes are sorted by eigenvalue, filtered to positive finite values, mass-normalized, and deterministically oriented.

Number of computed/saved modes:

- Gate A0 tracking computes 6 modes during history and final tracking.
- Postprocessing config requests only `compute_modes=1`.
- The requested first 10 final eigenfrequencies are **not available** in saved artifacts. Computing them now would be a new post-hoc eigensolve, so it was not done.

Saved final six-mode spectra:

| mode | 200x25 omega | 400x50 omega |
|---:|---:|---:|
| 1 | 141.789825358051 | 64.393651634491 |
| 2 | 374.309771773102 | 88.17909500334 |
| 3 | 614.650190135589 | 101.804704818001 |
| 4 | 652.934986404403 | 121.96434865125 |
| 5 | 867.891894282098 | 125.887924272032 |
| 6 | 1306.80198494537 | 133.145880142305 |

## 8. Result Validity

Declared Exp3 criteria require both cases accepted, same tracked mode index, final tracked MAC >= 0.8, relative tracked omega change <= 0.05, topology correlation >= 0.8, and topology MAD <= 0.15.

Final result table:

| mesh | classification | omega_1 | omega_2 | omega_3 | tracked mode | MAC | A5 |
|---|---|---:|---:|---:|---:|---:|---|
| 200x25 | accepted | 141.789825358051 | 374.309771773102 | 614.650190135589 | 1 | 0.992533132702 | pass |
| 400x50 | mode invalid | 64.393651634491 | 88.17909500334 | 101.804704818001 | 1 | 0.786126687583 | pass |

Exact validity mismatches:

- 400x50 final tracked MAC is `0.786126687583`, below the declared `0.8` threshold.
- Relative tracked omega change is `0.545851393273`, above the declared `0.05` threshold.
- Topology correlation is `-0.087969251187`, below the declared `0.8` threshold.
- Topology mean absolute difference is `0.518759676734`, above the declared `0.15` threshold.

Mode-tracking histories:

| metric | 200x25 | 400x50 |
|---|---:|---:|
| unique tracked indices | `[1,2,3,5]` | `[1,2,3,4,5,6]` |
| switch count | 36 | 44 |
| first MAC below 0.8 | iter 130 | iter 156 |
| minimum MAC | 0.000563099312 at iter 166 | 0.057459475920 at iter 208 |
| final MAC | 0.992533132702 | 0.786126687583 |

A5 lowest-mode check:

| mesh | pass | tracked mode omega | lowest mode omega | modes below tracked |
|---|---|---:|---:|---:|
| 200x25 | true | 141.789825358051 | 141.789825358051 | 0 |
| 400x50 | true | 64.393651634491 | 64.393651634491 | 0 |

Final first-mode shapes are **not available** as saved artifacts. The final tracking modes are not saved, and topology-mode visualization was disabled.

Density/connectivity/localization indicators from saved topology CSVs:

| metric | 200x25 | 400x50 |
|---|---:|---:|
| volume | 0.5 | 0.5 |
| min/max density | 0 / 0.957818690751 | 0 / 0.969650430002 |
| density `<0.05` | 47.72% | 48.43% |
| density `>0.95` | 52.12% | 51.56% |
| density in `[0.1,0.9]` | 0.16% | 0.01% |
| solid components at `rho>=0.5` | 23 | 126 |
| largest solid component size | 2514 | 10064 |
| largest solid fraction of solid elements | 0.962481 | 0.975950 |
| void components | 267 | 535 |
| largest void component size | 210 | 2646 |

The fine topology is much more fragmented by component count and has a much lower first frequency. This is consistent with mesh-sensitive/localized low-frequency behavior, but not by itself a proof of genuine asymptotic mesh non-convergence.

## 9. Artifact Integrity

Artifact consistency checks:

- Root manifest classification: `inconclusive/capped/mode invalid`, created `2026-06-26T10:36:50Z`.
- Per-case manifests point to the expected Exp3 config/result/topology files.
- Topology CSV and topology JSON match to max absolute difference `5.55e-16` for both meshes.
- Source `localWriteTopology` writes CSV, JSON, and PNG from the same `topology = reshape(xFinal(:), nely, nelx)` variable.
- File timestamps are coherent: config written first, topology/result/manifest after each case, root result/manifest last.
- No Exp3 topology CSV hash matches any inspected CR2 topology CSV hash.

Computed hashes for traceability:

| file | sha256 |
|---|---|
| Exp3 200 config | `820e5a29e4b7c1904f0ae0ed7f0022492c626e54d9332baac008847fb34ebae5` |
| Exp3 400 config | `0d6f28e865c7e7a27c1aa45fd4f1e5f48aff58922e0071002c69e5646f5c1868` |
| source 200 template config | `8a9939bef2d04d8a739f3ac38a64a706d8c52c152236b0236b64cfd9844e9850` |
| source 400 template config | `b4b7047cc21d372dbf6ffcdc3877c4e541cdba5b279fa071db0dc88d8857a878` |
| Exp3 200 topology CSV | `d2e1fca9b1880e2c082608eeb8b24fa75575663b586cd010dce88c1889bde74c` |
| Exp3 400 topology CSV | `1b26a50d17265c048674d63f2ccd292cb426452d4a84cb07dfc45e9d122e3a11` |

Warning: the Exp3 manifests do not themselves store config hashes. Therefore this audit can confirm computed hashes of the inspected files, but cannot confirm a manifest-embedded hash match.

CR2 contamination check:

- Exp3 result and manifest strings include `cr2_history` / `cr2_final_tracking` diagnostic field names, but their study/scope/artifact paths are Exp3-specific.
- CR2 topology hashes inspected under `examples/Revision_v1/cr2/**/output` do not match either Exp3 topology CSV.
- The Exp3 generator explicitly declares scope excluding CR2 and writes per-mesh artifacts under `output/exp3_authoritative_mesh_convergence`.

Verdict: no stale-file or CR2-artifact substitution was found.

## Exact Mismatches Found

1. 400x50 final tracked MAC fails the declared threshold: `0.786126687583 < 0.8`.
2. Relative tracked omega change fails the declared mesh criterion: `0.545851393273 > 0.05`.
3. Topology correlation fails the declared criterion: `-0.087969251187 < 0.8`.
4. Topology mean absolute difference fails the declared criterion: `0.518759676734 > 0.15`.
5. Saved final spectra contain only 6 tracking modes, not the requested first 10 final eigenfrequencies.
6. Config says `filter.boundary_condition="symmetric"`, but `ourApproach` does not consume that field; actual edge treatment is truncated-neighborhood denominator normalization.
7. Exp3 manifests do not store config hashes; hashes had to be computed externally during this audit.

None of these are a basic 200x25-vs-400x50 physical setup mismatch.

## Likely Root Cause Ranking

1. **Mesh-sensitive/local-minimum or localized low-frequency behavior on the fine mesh**. Best supported by the large drop in final `omega_1`, failed MAC, negative topology correlation, high topology MAD, and increased fine-mesh component count.
2. **Mode-family drift / tracking loss as a symptom of the changed design**. The fine case ends with mode 1 as the lowest mode, but MAC to the solid reference is below the declared gate.
3. **Fragile convergence evidence from a single final design-change criterion**. Both cases pass the final scalar tolerance, yet last-window topology changes remain larger than the final stopping value.
4. **Filter edge-treatment wording mismatch**. The `symmetric` field is ignored by `ourApproach`; this is real, but common to both meshes and therefore unlikely to explain the discrepancy by itself.
5. **Stale artifact, CR2 substitution, or generated-config mismatch**. Ruled out by paths, manifests, topology hash checks, topology CSV/JSON agreement, and source inspection.

## Classification

Current Exp3 should be classified as **inconclusive**.

It should not be classified as an **invalid setup**, because the fundamental setup is consistent across meshes. It also should not be classified as proven **genuine non-convergence**, because the fine case is mode-invalid under the declared gate and the saved artifacts do not isolate whether the divergence is asymptotic mesh non-convergence, optimizer basin dependence, localized-mode pathology, or length-scale insufficiency.

## Recommended Next Action

Do not use current Exp3 as mesh-convergence evidence. Mark it as an inconclusive failed validation case in the experiment record, with the specific reason: the 400x50 run used the same physical setup but failed the final MAC gate and produced a substantially different low-frequency topology.
