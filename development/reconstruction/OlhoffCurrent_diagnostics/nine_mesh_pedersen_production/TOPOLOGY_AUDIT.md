# TOPOLOGY_AUDIT (Part 15)

**Figures.**

- `figures/nine_final_topologies.png`: all nine final densities on one convention. Black = 1, white = 0, common [0,1] gray scale, 8 × 1 extent, nearest-neighbour display on an 800×100 grid, no smoothing, gray not suppressed.
- The runner's own per-mesh images: `examples/Performance/conference_benchmark/nine_mesh_pedersen_b21483b/topologies/topology_olhoff_<mesh>.png`, from the shared `renderTopologyDensity`.

Visual attractiveness is **not** an acceptance criterion.

## 1. Final-design metrics

| mesh | M_nd = 4·mean ρ(1−ρ) | gray fraction 0.1<ρ<0.9 | void ρ≤0.1 | solid ρ≥0.9 | min ρ / max ρ | components at 0.5 (largest share) | left–right mirror IoU at 0.5 (native mesh) | left–right mirror mean \|Δρ\| |
|---|---|---|---|---|---|---|---|---|
| 160x20 | 0.1146 | 0.1419 | 0.421 | 0.438 | 0.00100 / 1.0000 | **16** (98.6 %) | 0.846 | 0.083 |
| 240x30 | 0.1227 | 0.1447 | 0.425 | 0.431 | 0.00101 / 0.9999 | 1 | 0.948 | 0.025 |
| 320x40 | 0.1406 | 0.1656 | 0.416 | 0.419 | 0.00101 / 0.9999 | 1 | **0.605** | **0.228** |
| 400x50 | 0.1216 | 0.1397 | 0.428 | 0.432 | 0.00101 / 0.9999 | 1 | 0.976 | 0.011 |
| 480x60 | 0.1307 | 0.1509 | 0.425 | 0.424 | 0.00101 / 0.9999 | 1 | 0.908 | 0.048 |
| 560x70 | 0.1329 | 0.1534 | 0.425 | 0.422 | 0.00102 / 0.9999 | 1 | 0.969 | 0.015 |
| 640x80 | 0.1330 | 0.1529 | 0.424 | 0.423 | 0.00102 / 0.9999 | 1 | 0.983 | 0.008 |
| 720x90 | **0.1617** | **0.1863** | 0.408 | 0.406 | 0.00103 / 0.9999 | 1 | 0.961 | 0.019 |
| 800x100 | **0.1645** | **0.1878** | 0.407 | 0.405 | 0.00104 / 0.9998 | 1 | **0.837** | **0.085** |

**Sources.**

- M_nd and gray fraction come from the tapped ρ. They equal the runner evaluator's `grayness` and `gray_fraction_01_09` bit for bit.
- Connectivity is the runner evaluator's `connectivity_raw_05`.
- Mirror metrics are computed on each native mesh (`evidence/TOPOLOGY_SYMMETRY.json`).
- All designs are **up–down symmetric** to about 1e-10, consistent with the mid-height supports. Every design is left–right connected.

**Neighbour and reference comparisons** use an 800×100 nearest-neighbour grid (`METRICS.json` `topology_comparisons`). They are descriptive only and have no threshold.

| pair | mean \|Δρ\| | IoU at 0.5 |
|---|---|---|
| 160x20 → 240x30 | 0.131 | 0.756 |
| 240x30 → 320x40 | 0.164 | 0.705 |
| 320x40 → 400x50 | 0.166 | 0.702 |
| 400x50 → 480x60 | 0.067 | 0.865 |
| 480x60 → 560x70 | 0.048 | 0.904 |
| 560x70 → 640x80 | 0.042 | 0.917 |
| 640x80 → 720x90 | 0.076 | 0.849 |
| 720x90 → 800x100 | 0.112 | 0.782 |
| each mesh vs 800x100 | 0.209 → 0.112 | 0.634 → 0.782 |

## 2. What the topologies show

1. **160x20.** Many thin members and several X-crossings in the end bays. At a 0.5 threshold the design has 16 disconnected fragments, though 98.6 % of the solid is one component. The resolution is too coarse for R = 0.06·b, which is 1.2 elements.
2. **240x30 to 640x80: one topology family.** Two chords, a large central opening, one X-crossing per side bay, and diamond-shaped end bays. 320x40 is the exception inside this range (item 3). Neighbour IoU rises to 0.86–0.92 from 400x50 to 640x80, the most mesh-consistent part of the series.
3. **320x40 breaks left–right symmetry strongly.** The left half carries crossing members that are absent from the right half: mirror IoU 0.605. The load case is symmetric. The design is also the one that sits **below** the ω₁ trend (165.86 against 166.46 at 400x50).
4. **720x90.** Additional thin, **gray** X-braces appear inside both end diamonds, in a mirror-symmetric pattern (IoU 0.961). M_nd rises from 0.133 to 0.162 and the gray fraction from 0.153 to 0.186.
5. **800x100 is left–right asymmetric.** The left end bay has more members than the right (mirror IoU 0.837). It is the grayest design (M_nd 0.165, gray 0.188).
6. **Solution space.** The finest two meshes differ visibly (IoU 0.78) yet agree in ω₁ to 0.009 rad/s. 640x80 and 800x100 differ more (IoU 0.70 on the common grid) with ω₁ only 0.22 rad/s (0.13 %) apart. The final ω₁ is therefore insensitive to these topological differences: several distinct layouts give nearly the same fundamental frequency.

## 3. Grayness

- **Controlled but not mesh-invariant.** M_nd lies between 0.115 and 0.141 from 160x20 to 640x80, then steps up by about 22 % to 0.162 and 0.165 at the two finest meshes. The gray fraction follows the same pattern (0.140–0.166, then 0.186–0.188).
- **Preregistered bound S6.** M_nd(800x100) / M_nd(160x20) = 1.44 and gray(800x100) / gray(160x20) = 1.32, both below the 2× bound, so S6 did not fire.
- **Not converged at the stop.** At every mesh the final M_nd is the minimum of its own history (TERMINATION_AUDIT.md §2): the designs were still becoming less gray when the heuristic stop fired.
- **Why the finest meshes are grayer is not identified.** Possible contributors are the stop timing and the physical filter radius at 5.4–6 elements. Neither was tested.

## 4. Answers

- **Consistent rendering and metrics for all nine meshes:** yes.
- **Gray or unattractive results suppressed:** none.
- **Topology mesh-converged:** **no.** The layout differs at 160x20, at 320x40 (asymmetric), at 720x90 (extra gray braces) and at 800x100 (asymmetric), while ω₁ stays nearly constant.
- **Symmetry breaking:** present at 320x40, 800x100 and 160x20. It is reported as it is and was not repaired.
