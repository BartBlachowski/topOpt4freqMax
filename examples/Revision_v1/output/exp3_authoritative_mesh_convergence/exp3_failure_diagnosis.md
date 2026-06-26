# Exp3 Failure Diagnosis

Scope: saved Exp3 alpha=1.00 artifacts only. No experiment reruns, solver-code edits, manuscript edits, or Python diagnostics were used.

Artifact set inspected:
- `exp3_authoritative_mesh_convergence_result.json`
- per-mesh result MAT/JSON files
- per-mesh generated configs
- convergence, frequency, mode-tracking, feasibility, and grayness CSV histories
- topology CSV/JSON/PNG files
- topology difference metrics JSON
- A5 lowest-mode check JSON/CSV files
- root and per-mesh manifests

## Conclusion

The Exp3 failure is primarily **mesh-dependence/local-minimum behavior with a likely localized low-frequency fine-mesh mode**, not an implementation/configuration mismatch and not a physical-filter-radius inconsistency.

The 400x50 case is invalid for accepted Exp3 evidence because its final tracked MAC is below the declared gate:

- 400x50 final tracked MAC: `0.7861266876`
- declared MAC threshold: `0.8`
- classification: `mode invalid`

However, the 400x50 result should not be dismissed as a simple tracking bookkeeping failure. It exposes that the 200x25 accepted design was not representative of a mesh-converged topology: the fine mesh reaches a different low-frequency topology and a different physical mode family despite using the same authoritative load and same physical filter radius.

## Field-By-Field Config Comparison

The generated Exp3 configs differ only in:

| field | 200x25 | 400x50 |
|---|---:|---:|
| `meta.name` | `... 200x25` | `... 400x50` |
| `domain.mesh.nelx` | 200 | 400 |
| `domain.mesh.nely` | 25 | 50 |

All other inspected fields match:

- geometry: rectangular, `length=8`, `height=1`, `thickness=1`
- supports: both vertical clamp lines at `x=0` and `x=8`, `ux/uy`
- material and void material
- load cases: alpha=1 mode-1 semi-harmonic active, mode-2 case factor 0
- authoritative load setting: `F(x) = omega0^2 * M(x) * Phi0`
- `semi_harmonic_baseline = solid`
- `harmonic_normalize = false`
- `load_sensitivity = omitted`
- `gate_a0_diagnostics = true`
- optimizer/objective/interpolation/volume/penalization/move limit
- `convergence_tol = 0.001`
- postprocessing diagnostics

This rules out a direct implementation/config mismatch between the two saved Exp3 cases.

## Physical Filter Radius

The filter was consistent in physical units:

| mesh | element size | filter radius | radius in elements |
|---|---:|---:|---:|
| 200x25 | `hx=hy=0.04` | `0.08` | 2 |
| 400x50 | `hx=hy=0.02` | `0.08` | 4 |

So the failure is not due to accidentally using the same element-count radius on both meshes. The fine mesh correctly used twice as many elements across the same physical filter radius.

## Reference-Mode Check

The solid reference modes agree to normal discretization accuracy:

| mesh | reference omega_1 | reference omega_2 |
|---|---:|---:|
| 200x25 | 291.0350397 | 725.785323 |
| 400x50 | 290.8871595 | 725.3417067 |

Diagnostics also show:

- `load_normalization_enabled = false`
- `obsolete_rho_source_used = false`

This rules out reference-mode mismatch as the primary cause.

## Final Results

| mesh | classification | iterations | omega_1 | omega_2 | omega_3 | tracked mode | MAC | design change | feasibility | grayness | A5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 200x25 | accepted | 1052/2000 | 141.789825 | 374.309772 | 614.650190 | 1 | 0.992533 | 0.00016619 | 0 | 0.0857778 | pass |
| 400x50 | mode invalid | 1750/2000 | 64.393652 | 88.179095 | 101.804705 | 1 | 0.786127 | 0.00090311 | 0 | 0.0607932 | pass |

Both designs satisfy the volume constraint and A5 lowest-mode check. A5 only says the tracked mode is the lowest computed mode within each final design. It does not prove the two meshes converged to the same physical mode.

## Frequency And MAC Histories

Mode tracking is not clean on either mesh during the run:

| metric | 200x25 | 400x50 |
|---|---:|---:|
| unique tracked mode indices during run | `[1 2 3 5]` | `[1 2 3 4 5 6]` |
| mode-index switch count | 36 | 44 |
| first MAC below 0.8 | iter 130, MAC 0.762980 | iter 156, MAC 0.757824 |
| minimum MAC | 0.000563 at iter 166 | 0.057459 at iter 208 |
| final MAC | 0.992533 | 0.786127 |

The important difference is recovery:

- 200x25 loses mode correspondence during the transient but recovers to MAC `0.992533`.
- 400x50 loses mode correspondence and never recovers above the `0.8` acceptance gate by the final design.

Frequency histories also diverge substantially:

| metric | 200x25 omega_1 | 400x50 omega_1 |
|---|---:|---:|
| first iteration | 145.517956 | 145.444016 |
| minimum during run | 3.696359 | 2.229175 |
| maximum during run | 196.646345 | 168.796518 |
| final | 141.789861 | 64.393633 |

The first-iteration frequencies are close, but the final frequencies are not. The relative final tracked omega change is `0.545851`, far above the declared `0.05` mesh-convergence tolerance.

## Convergence Tolerance

Both final design-change values satisfy the declared scalar tolerance:

- 200x25 final design change: `0.00016619`
- 400x50 final design change: `0.00090311`

But the histories show the single-final-iteration stopping criterion is fragile:

| metric | 200x25 | 400x50 |
|---|---:|---:|
| max design change in last 100 iterations | 0.0237162 | 0.0154032 |
| last-10 mean design change | 0.0125423 | 0.00755467 |
| last-10 max design change | 0.0198546 | 0.0144387 |
| last-10 omega_1 range | [141.759, 141.790] | [64.3922, 64.3936] |

This is a secondary issue. The final frequencies were stable over the last window, but the topology updates were not monotonically settled. Tightening or windowing the convergence test might improve evidence quality, but it would not explain the large frequency/topology mismatch by itself.

## Final Topology Statistics

The final topologies are both nearly binary, but they are not similar.

| metric | value |
|---|---:|
| topology correlation, fine mapped to coarse | -0.0879693 |
| topology mean absolute difference | 0.518760 |
| topology RMS difference | 0.664722 |
| max absolute difference | 0.969650 |
| binary Jaccard at 0.5 threshold | 0.300104 |
| mapped volume difference | -6.77e-15 |

Per-mesh density statistics:

| metric | 200x25 | 400x50 |
|---|---:|---:|
| volume | 0.5 | 0.5 |
| grayness | 0.0857778 | 0.0607932 |
| density `< 0.05` | 47.72% | 48.43% |
| density `> 0.95` | 52.12% | 51.56% |
| density in `[0.1, 0.9]` | 0.16% | 0.01% |
| solid connected components at `rho > 0.5` | 23 | 126 |
| largest solid component fraction | 0.962481 | 0.975950 |
| void connected components | 267 | 535 |

The 400x50 topology is more fragmented at element scale even though the same physical filter radius was used. The topology image also shows a different lower-band structural layout rather than a refined version of the 200x25 layout.

## Cause Classification

1. **Mode-tracking failure:** secondary, not primary. The 400x50 case fails the final MAC gate, but the topology and frequency mismatch indicate the final design itself differs, not just the label assigned to a mode.

2. **Mesh-dependence/checkerboard/local minima:** primary. The final topology correlation is negative, topology MAD is `0.518760`, and the fine mesh converges to a much lower omega family.

3. **Filter-radius inconsistency in physical units:** ruled out. Both use `radius = 0.08` physical; this is 2 elements on 200x25 and 4 elements on 400x50.

4. **Insufficient convergence tolerance:** secondary. Final scalar tolerance passes for both, but last-window design changes are not consistently small. This weakens confidence but does not explain the large topology/frequency split alone.

5. **Low-density/localized-mode effect:** likely contributing. The fine mesh final omega_1 is much lower (`64.39` vs `141.79`) and the topology is highly fragmented with many small components. This is consistent with a fine-mesh localized or weak-region mode becoming the lowest mode.

6. **Reference-mode mismatch:** ruled out. Solid reference modes agree closely and diagnostics show the intended solid reference, omitted load sensitivity, and no obsolete rho source.

7. **Implementation/config mismatch:** ruled out by generated config diff and manifest/result consistency.

## 400x50 Invalid Or 200x25 Nonrepresentative?

For the declared Exp3 gates, the 400x50 result is **invalid** because final MAC is below `0.8`.

For scientific interpretation, the 200x25 result should be treated as **nonrepresentative of a mesh-converged alpha=1 topology**. The fine mesh did not reproduce the coarse topology or frequency response under the same physical filter radius and same authoritative formulation.

Therefore, Exp3 should remain classified as:

`inconclusive/capped/mode invalid`

No mesh-convergence claim should be made from this evidence set.

## Recommendation

Recommended action: **manuscript qualification / exclusion from claims**, not a claim of mesh convergence.

State that the accepted 200x25 alpha=1 result did not pass the 400x50 mesh-convergence check: the fine mesh produced a different low-frequency topology and failed the final MAC gate. The result should be reported as mesh-specific evidence or excluded from mesh-converged manuscript claims until a controlled length-scale/localization mitigation study is run.

If a future controlled mitigation is allowed, the cleanest single mitigation would be to repeat this Exp3 check with an explicit mesh-independent length-scale control stronger than the current sensitivity filter alone, while holding all other settings fixed.
