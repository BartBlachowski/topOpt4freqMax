# S1 Exp3 400x50 Mode Diagnosis

Scope: postprocessing-only diagnostic of the saved failed Exp3 400x50 alpha=1.00 case. No optimization rerun, solver-code edit, manuscript edit, or Exp2/Exp3/CR2/P1/A4 rerun was performed.

## Conclusion

**likely localized/spurious low-density mode influence.** At least one of the first three modes is classified as localized low-density.

Saved Exp3 fine-case status: final tracked MAC `0.786126687583`, tracked omega `64.3936516345` rad/s.

## Setup

| item | value |
|---|---:|
| mesh | 400x50 |
| domain | 8 x 1 |
| element size | 0.02 x 0.02 |
| fixed DOFs | 204 |
| low-density threshold | 0.05 |
| solid-component threshold | 0.5 |
| solid components | 126 |
| support-connected components | 1 |
| largest support-connected component id | 1 |

## Mode Summary

| mode | omega rad/s | Hz | MAC(ref mode 1) | mass residual | eig residual | low-density K frac | low-density S frac | eff elem frac K | eff elem frac S | support-comp K frac | support-comp S frac | class |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 64.3936516345 | 10.2485679614 | 0.786126687583 | 1.110e-16 | 3.267e-09 | 7.54476e-07 | 0.00222282 | 0.30913 | 0.013038 | 0.972395 | 0.996917 | ambiguous |
| 2 | 88.1790950033 | 14.0341388472 | 5.36530618168e-05 | 2.220e-16 | 1.700e-09 | 5.31578e-06 | 0.999899 | 0.000801581 | 0.00201102 | 3.31588e-05 | 9.93826e-05 | localized low-density mode |
| 3 | 101.804704818 | 16.2027220018 | 1.77624620142e-06 | 2.220e-16 | 1.314e-09 | 5.22681e-06 | 0.999586 | 0.000504428 | 0.00184336 | 0.000233233 | 0.000412084 | localized low-density mode |
| 4 | 121.964348651 | 19.4112289688 | 0.00125768820183 | 2.220e-16 | 1.418e-09 | 1.12191e-05 | 0.991826 | 0.000764778 | 0.00126024 | 0.00673177 | 0.00815458 | localized low-density mode |
| 5 | 125.887924272 | 20.0356854235 | 0.144717523546 | 2.220e-16 | 1.214e-09 | 1.25432e-06 | 0.0549846 | 0.153039 | 0.0192101 | 0.916309 | 0.944793 | ambiguous |
| 6 | 133.145880142 | 21.190824977 | 0.00652904051305 | 0.000e+00 | 1.201e-09 | 5.59212e-06 | 0.963746 | 0.000532244 | 0.00095744 | 0.0389773 | 0.0362422 | localized low-density mode |
| 7 | 133.660352364 | 21.2727057741 | 0.000240522467136 | 0.000e+00 | 8.671e-10 | 2.48392e-06 | 0.998366 | 0.000565654 | 0.00133842 | 0.00130373 | 0.0016289 | localized low-density mode |
| 8 | 140.96746828 | 22.435669392 | 5.82708303344e-05 | 2.220e-16 | 9.948e-10 | 6.85434e-06 | 0.999748 | 0.000203114 | 0.00109254 | 0.00024819 | 0.000251921 | localized low-density mode |
| 9 | 141.563060603 | 22.5304608542 | 5.21091099184e-06 | 0.000e+00 | 1.110e-09 | 9.62732e-06 | 0.999771 | 0.000147779 | 0.00227594 | 0.000112787 | 0.00022828 | localized low-density mode |
| 10 | 160.583263783 | 25.5576202088 | 3.9209408245e-08 | 0.000e+00 | 1.174e-09 | 5.80771e-06 | 0.999911 | 0.000431851 | 0.0016978 | 4.20616e-05 | 7.67046e-05 | localized low-density mode |

## Classification Reasons

- Mode 1: `ambiguous` - localized by participation metric but not clearly low-density or disconnected: effK 0.309, effS 0.013
- Mode 2: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 1.000
- Mode 3: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 1.000
- Mode 4: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.992
- Mode 5: `ambiguous` - localized by participation metric but not clearly low-density or disconnected: effK 0.153, effS 0.019
- Mode 6: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.964
- Mode 7: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.998
- Mode 8: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 1.000
- Mode 9: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 1.000
- Mode 10: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 1.000

## Artifacts

- JSON summary: `examples/Revision_v1/output/s1_exp3_400x50_mode_diagnostic/s1_exp3_400x50_mode_summary.json`
- mode summary CSV: `examples/Revision_v1/output/s1_exp3_400x50_mode_diagnostic/s1_exp3_400x50_modes_summary.csv`
- manifest: `examples/Revision_v1/output/s1_exp3_400x50_mode_diagnostic/s1_exp3_400x50_manifest.json`
- per-mode energy CSVs: `s1_exp3_400x50_mode_##_energy.csv`
- first six mode-shape figures: `s1_exp3_400x50_mode_##_shape.png`

## Interpretation

The classification is based on final-topology K/M reassembly, the first 10 eigenmodes, MAC to the saved solid-reference target mode, elementwise kinetic and strain energy fractions in `rho < 0.05` regions, energy participation/localization metrics, and association with solid components touching the support lines.
