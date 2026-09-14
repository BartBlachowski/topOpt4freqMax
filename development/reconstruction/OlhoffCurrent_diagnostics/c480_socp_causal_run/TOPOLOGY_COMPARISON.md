# TOPOLOGY_COMPARISON — Part 11

Identical physical coordinates (8 × 1, 480 × 60, column-major). Code:
`scripts/cs_analyze.py::topology`. The treatment design is ρ₁₄, the last accepted
iterate; the run has no endpoint.

| metric | treatment 14 vs control final 386 | treatment 14 vs control 14 (matched) |
|---|---|---|
| ‖ρ_T − ρ_C‖₂ | 38.90 | 30.89 |
| ‖ρ_T − ρ_C‖∞ | 0.779 | 0.524 |
| RMS difference | 0.229 | 0.182 |
| ‖Δρ‖₂ / ‖ρ_C‖₂ | 0.348 | 0.322 |
| Pearson correlation | 0.845 | 0.874 |
| threshold-0.5 solid Jaccard | 0.694 | 0.670 |
| threshold-0.5 agreement | 83.2 % | 83.5 % |
| threshold-0.5 void Jaccard | 0.727 | 0.752 |
| gray-mask Jaccard | 0.481 | 0.682 |
| broad-core Jaccard | 0.357 | 0.517 |
| material relocation Σ\|Δρ\| / 2Σρ_C | 14.4 % | 13.3 % |
| elements with \|Δρ\| > 0.1 / > 0.5 | 41.7 % / 4.1 % | 48.3 % / 0.07 % |

## Reading

- **Matched iteration.** Both designs share the same global layout: a doubly-symmetric
  beam with top and bottom flanges converging to supports near x = 0 and x = 8, and a
  central void around x = 3–5 (`FIG_01`, `FIG_02`). Their correlation is 0.87 and
  they agree 83 % at threshold 0.5. The treatment differs in a coherent,
  symmetric pattern (`FIG_03` top). Regional mean Δρ (treatment − control, outer 14):
  flanges x ∈ [1,7] at y > 0.85 or y < 0.15: **+0.068**; web regions x ∈ [1,2.6] and
  [5.4,7], y ∈ [0.25,0.75]: **+0.066**; central region x ∈ [2.8,5.2]: **−0.097**;
  support ends x < 0.7 and x > 7.3: **−0.103**. Along the flange arcs and ligament
  edges the local differences reach +0.3 to +0.5. So exact steps moved material out of
  the center and the support ends into the flanges and web faster. The global layout
  did not change.
- **Against the control endpoint.** The control's final design contains X-shaped
  diagonal members at x ≈ 1.5–2.6 and 5.4–6.5. At outer 14 the treatment shows only
  their faint precursors in a still-gray web (`FIG_03` bottom). This difference is
  mainly one of *progress*, not evidence of a different final topology. Material
  relocation (14.4 %) and threshold disagreement (16.8 %) mostly reflect unfinished
  web members.
- Whether the exact-SOCP trajectory would have produced the same X-braced topology,
  a different one, or a crisper version of it **cannot be determined**.

Images are illustrations only. Every claim above is tied to the numbers in the table.
