# Disconnected/Localized Mode Audit

Generated: `2026-07-08 23:37:01`

## Scope

Stabilized OlhoffApproachExact CC 40x5 runs were audited against support-connectedness, isolated islands, modal localization, component energy concentration, filter radius, p-continuation, and symmetry. The connectedness-pruning step is post-run diagnostic only and is not claimed as an optimizer or reproduction method.

Paper target guard: `omega1 = 456.4 +/- 2.0%`, coalescence `gap12 <= 0.005`. A mode is labeled component-local when one non-support component carries at least 70% mean kinetic/strain energy.

## Summary

| variant | omega1 | omega2 | gap | N | rmin | penalty | symmetry | components | support-connected | isolated frac | mode1 | mode2 | mode1 top/support | mode2 top/support | support-pruned omega1 |
|---|---:|---:|---:|---:|---:|---|---|---:|---|---:|---|---|---:|---:|---:|
| `fixed_p3_rmin_2p5` | 489.188 | 489.198 | 2.037e-05 | 2 | 2.5 | fixed_p3 | none | 2 | no | 1.000 | disconnected_structure_mode | disconnected_structure_mode | 0.548 / 0.000 | 0.548 / 0.000 | NaN |
| `fixed_p3_rmin_3p5` | 646.595 | 646.618 | 3.5e-05 | 2 | 3.5 | fixed_p3 | none | 2 | no | 1.000 | disconnected_structure_mode | disconnected_structure_mode | 0.643 / 0.000 | 0.643 / 0.000 | NaN |
| `fixed_p3_rmin_5p0` | 807.756 | 808.030 | 0.0003387 | 2 | 5.0 | fixed_p3 | none | 2 | no | 1.000 | island_or_component_local_mode | island_or_component_local_mode | 0.839 / 0.000 | 0.836 / 0.000 | NaN |
| `cont_p123_rmin_2p5` | 334.436 | 363.972 | 0.08832 | 1 | 2.5 | p123 | none | 2 | no | 1.000 | disconnected_structure_mode | disconnected_structure_mode | 0.495 / 0.000 | 0.284 / 0.000 | NaN |
| `cont_p123_rmin_3p5` | 229.763 | 293.950 | 0.2794 | 1 | 3.5 | p123 | none | 4 | no | 1.000 | disconnected_structure_mode | disconnected_structure_mode | 0.156 / 0.000 | 0.409 / 0.000 | NaN |
| `cont_p123_rmin_5p0` | 254.070 | 491.987 | 0.9364 | 1 | 5.0 | p123 | none | 5 | no | 1.000 | disconnected_structure_mode | disconnected_structure_mode | 0.316 / 0.000 | 0.589 / 0.000 | NaN |
| `fixed_p3_rmin_3p5_sym_midspan` | 646.496 | 646.496 | 8.669e-08 | 2 | 3.5 | fixed_p3 | midspan | 2 | no | 1.000 | disconnected_structure_mode | disconnected_structure_mode | 0.321 / 0.000 | 0.321 / 0.000 | NaN |
| `fixed_p3_rmin_3p5_sym_midheight` | 646.774 | 646.843 | 0.0001065 | 2 | 3.5 | fixed_p3 | midheight | 2 | no | 1.000 | disconnected_structure_mode | disconnected_structure_mode | 0.644 / 0.000 | 0.644 / 0.000 | NaN |
| `fixed_p3_rmin_3p5_sym_both` | 646.563 | 646.563 | 9.145e-08 | 2 | 3.5 | fixed_p3 | both | 2 | no | 1.000 | disconnected_structure_mode | disconnected_structure_mode | 0.321 / 0.000 | 0.321 / 0.000 | NaN |
| `cont_p123_rmin_3p5_sym_both` | 735.972 | 735.972 | 6.952e-08 | 2 | 3.5 | p123 | both | 2 | no | 1.000 | disconnected_structure_mode | disconnected_structure_mode | 0.274 / 0.000 | 0.274 / 0.000 | NaN |

## Findings

- None of the audited stabilized final topologies are support-connected under the thresholded solid-component test.
- No variant satisfies the combined paper frequency/gap and support-connectedness guards.
- `1/10` variants have omega1 or omega2 classified as component-local by the energy-concentration test.
- Individual modes in a coalesced disconnected pair can be arbitrary mixtures over equivalent islands, so the 70% single-mode concentration test is conservative. The stronger invariant here is that the support component energy fraction is zero for omega1/omega2 in every variant.
- The support-connected pruning diagnostic has no candidate in these runs: no thresholded solid component touches both supports. Largest-component pruning leaves isolated island spectra, not a supported beam.
- Fixed p=3 with larger rmin raises the disconnected coalesced pair (rmin 2.5 -> 3.5 -> 5.0), reaching 807.8/808.0 at rmin 5.0, but the topology remains disconnected and the first pair becomes component-local.
- The p=1->2->3 continuation runs do not recover the paper basin. Without symmetry they lose coalescence or frequency, and with both-axis symmetry they recover a high coalesced pair that is still disconnected.
- Mirror symmetry about midspan, midheight, or both axes stabilizes symmetric-looking high-frequency coalesced states, but it does not create support-connected structural modes.

## Answer

The non-paper high frequencies in this audit are caused by disconnected and sometimes component-local modal behavior rather than a legitimate alternative support-connected CC beam optimum. The evidence is the lack of support-connected final topologies, isolated-material fraction of 1.0 in every thresholded design, zero support-component modal energy for omega1/omega2, and the inability of filter-radius changes, p-continuation, or mirror symmetry to produce support-connected beam modes.

## Evidence Files

- `disconnected_local_mode_audit_results/disconnected_local_mode_audit_summary.csv`
- `disconnected_local_mode_audit_results/<variant>/<variant>_components.csv`
- `disconnected_local_mode_audit_results/<variant>/<variant>_mode_component_energy.csv`
- `disconnected_local_mode_audit_results/<variant>/<variant>_topology.png`
- `disconnected_local_mode_audit_results/<variant>/<variant>_mode1_shape.png`
- `disconnected_local_mode_audit_results/<variant>/<variant>_mode2_shape.png`
- `disconnected_local_mode_audit_results/<variant>/<variant>_audit.mat`
