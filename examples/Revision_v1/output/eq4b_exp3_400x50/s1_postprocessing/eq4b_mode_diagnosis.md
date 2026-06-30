# S1 Exp3 400x50 Mode Diagnosis

Scope: postprocessing-only diagnostic of the saved failed Exp3 400x50 alpha=1.00 case. No optimization rerun, solver-code edit, manuscript edit, or Exp2/Exp3/CR2/P1/A4 rerun was performed.

## Conclusion

**likely localized/spurious low-density mode influence.** At least one of the first three modes is classified as localized low-density.

Saved Exp3 fine-case status: final tracked MAC `0.924439952738`, tracked omega `112.642033651` rad/s.

## Setup

| item | value |
|---|---:|
| mesh | 400x50 |
| domain | 8 x 1 |
| element size | 0.02 x 0.02 |
| fixed DOFs | 204 |
| low-density threshold | 0.05 |
| solid-component threshold | 0.5 |
| solid components | 458 |
| support-connected components | 1 |
| largest support-connected component id | 1 |

## Mode Summary

| mode | omega rad/s | Hz | MAC(ref mode 1) | mass residual | eig residual | low-density K frac | low-density S frac | eff elem frac K | eff elem frac S | support-comp K frac | support-comp S frac | class |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 77.0138805747 | 12.2571397802 | 0.0524590828034 | 0.000e+00 | 2.013e-09 | 8.57666e-06 | 0.902057 | 0.0104425 | 0.00306868 | 0.0106488 | 0.0277383 | localized low-density mode |
| 2 | 112.642033651 | 17.9275364555 | 0.924439952738 | 1.110e-16 | 1.075e-09 | 6.74901e-05 | 0.0363958 | 0.303763 | 0.0715724 | 0.790941 | 0.874664 | physical global mode |
| 3 | 167.58522164 | 26.6720164132 | 0.000439186655853 | 0.000e+00 | 8.956e-10 | 1.76115e-05 | 0.992438 | 0.000395671 | 0.00198181 | 0.00280153 | 0.0033946 | localized low-density mode |
| 4 | 173.745953116 | 27.6525272806 | 1.57535282835e-05 | 1.110e-16 | 6.199e-10 | 0.000121127 | 0.904611 | 0.000647059 | 0.0017791 | 0.00154933 | 0.00356033 | localized low-density mode |
| 5 | 173.920405918 | 27.6802923064 | 1.92182436293e-06 | 0.000e+00 | 1.260e-09 | 0.000820513 | 0.947564 | 0.000438055 | 0.0017319 | 3.46088e-05 | 0.000593111 | localized low-density mode |
| 6 | 176.6748006 | 28.1186678353 | 0.000557749010096 | 2.220e-16 | 8.448e-10 | 3.66775e-06 | 0.994812 | 0.000390041 | 0.00158954 | 0.0022137 | 0.00188627 | localized low-density mode |
| 7 | 178.761962012 | 28.450849891 | 5.94339247726e-07 | 0.000e+00 | 7.622e-10 | 1.96184e-05 | 0.860171 | 0.000642387 | 0.00128178 | 0.000149505 | 0.000963282 | localized low-density mode |
| 8 | 186.845656662 | 29.737409853 | 8.52883238752e-08 | 1.110e-16 | 8.114e-10 | 3.59494e-06 | 0.991758 | 0.00025416 | 0.000773736 | 0.000477244 | 0.00116924 | localized low-density mode |
| 9 | 187.78411445 | 29.8867700489 | 4.59324451986e-05 | 2.220e-16 | 5.276e-10 | 1.03977e-05 | 0.99206 | 0.000312043 | 0.000951569 | 0.000157141 | 0.000635486 | localized low-density mode |
| 10 | 196.027466489 | 31.1987402735 | 1.2278163966e-05 | 0.000e+00 | 7.468e-10 | 8.23187e-06 | 0.991494 | 0.000368994 | 0.00162332 | 0.000969308 | 0.00229079 | localized low-density mode |

## Classification Reasons

- Mode 1: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.902
- Mode 2: `physical global mode` - energy is mostly on support-connected component: kinetic 0.791, strain 0.875
- Mode 3: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.992
- Mode 4: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.905
- Mode 5: `localized low-density mode` - low-density energy fractions are high: kinetic 0.001, strain 0.948
- Mode 6: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.995
- Mode 7: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.860
- Mode 8: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.992
- Mode 9: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.992
- Mode 10: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.991

## Artifacts

- JSON summary: `examples/Revision_v1/output/eq4b_exp3_400x50/s1_postprocessing/eq4b_mode_summary.json`
- mode summary CSV: `examples/Revision_v1/output/eq4b_exp3_400x50/s1_postprocessing/eq4b_modes_summary.csv`
- manifest: `examples/Revision_v1/output/eq4b_exp3_400x50/s1_postprocessing/eq4b_manifest.json`
- per-mode energy CSVs: `s1_exp3_400x50_mode_##_energy.csv`
- first six mode-shape figures: `s1_exp3_400x50_mode_##_shape.png`

## Interpretation

The classification is based on final-topology K/M reassembly, the first 10 eigenmodes, MAC to the saved solid-reference target mode, elementwise kinetic and strain energy fractions in `rho < 0.05` regions, energy participation/localization metrics, and association with solid components touching the support lines.
