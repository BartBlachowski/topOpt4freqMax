# S1 Exp3 400x50 Mode Diagnosis

Scope: postprocessing-only diagnostic of the saved failed Exp3 400x50 alpha=1.00 case. No optimization rerun, solver-code edit, manuscript edit, or Exp2/Exp3/CR2/P1/A4 rerun was performed.

## Conclusion

**likely localized/spurious low-density mode influence.** At least one of the first three modes is classified as localized low-density.

Saved Exp3 fine-case status: final tracked MAC `0.973514472548`, tracked omega `131.934195703` rad/s.

## Setup

| item | value |
|---|---:|
| mesh | 400x50 |
| domain | 8 x 1 |
| element size | 0.02 x 0.02 |
| fixed DOFs | 204 |
| low-density threshold | 0.05 |
| solid-component threshold | 0.5 |
| solid components | 198 |
| support-connected components | 1 |
| largest support-connected component id | 1 |

## Mode Summary

| mode | omega rad/s | Hz | MAC(ref mode 1) | mass residual | eig residual | low-density K frac | low-density S frac | eff elem frac K | eff elem frac S | support-comp K frac | support-comp S frac | class |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 131.934195703 | 20.997979409 | 0.973514472548 | 2.220e-16 | 1.101e-09 | 1.56053e-06 | 0.0224214 | 0.243377 | 0.0763718 | 0.936557 | 0.974055 | physical global mode |
| 2 | 145.885500317 | 23.2183985009 | 2.92037328222e-05 | 2.220e-16 | 1.400e-09 | 3.11507e-06 | 0.998217 | 0.000496597 | 0.00188838 | 0.000573693 | 0.00177408 | localized low-density mode |
| 3 | 147.107859545 | 23.4129430143 | 2.00967126481e-07 | 4.441e-16 | 1.173e-09 | 3.04497e-06 | 0.998229 | 0.000499419 | 0.00189178 | 0.000546619 | 0.00176178 | localized low-density mode |
| 4 | 160.119282016 | 25.4837752172 | 3.22610505952e-05 | 0.000e+00 | 9.672e-10 | 2.21048e-06 | 0.547024 | 0.0031332 | 0.00901961 | 0.39786 | 0.452019 | localized low-density mode |
| 5 | 165.330361075 | 26.3131442082 | 0.00769552074172 | 0.000e+00 | 1.053e-09 | 2.94967e-06 | 0.986475 | 0.00113833 | 0.00320796 | 0.0188761 | 0.0134763 | localized low-density mode |
| 6 | 165.509826927 | 26.3417070857 | 0.00157573473321 | 2.220e-16 | 8.449e-10 | 1.18035e-05 | 0.996084 | 0.000235939 | 0.00133241 | 0.00463542 | 0.00390318 | localized low-density mode |
| 7 | 169.847861878 | 27.0321267915 | 1.68361949666e-06 | 1.110e-16 | 9.175e-10 | 2.09212e-06 | 0.568959 | 0.00383135 | 0.0117621 | 0.425981 | 0.430149 | localized low-density mode |
| 8 | 177.652471837 | 28.2742690454 | 0.00127759826955 | 2.220e-16 | 7.731e-10 | 2.15919e-06 | 0.991289 | 0.000871366 | 0.00325009 | 0.0104999 | 0.00868528 | localized low-density mode |
| 9 | 178.521930687 | 28.4126477191 | 2.01271432229e-05 | 0.000e+00 | 8.126e-10 | 2.15515e-06 | 0.937072 | 0.000983302 | 0.00354821 | 0.0683113 | 0.0627911 | localized low-density mode |
| 10 | 186.463982101 | 29.6766644599 | 0.00018570288136 | 2.220e-16 | 7.543e-10 | 1.69165e-06 | 0.994246 | 0.000344474 | 0.000824848 | 0.00660403 | 0.00572954 | localized low-density mode |

## Classification Reasons

- Mode 1: `physical global mode` - energy is mostly on support-connected component: kinetic 0.937, strain 0.974
- Mode 2: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.998
- Mode 3: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.998
- Mode 4: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.547
- Mode 5: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.986
- Mode 6: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.996
- Mode 7: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.569
- Mode 8: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.991
- Mode 9: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.937
- Mode 10: `localized low-density mode` - low-density energy fractions are high: kinetic 0.000, strain 0.994

## Artifacts

- JSON summary: `examples/Revision_v1/output/s1_mitigation_400x50/s1_postprocessing/s1_mitigation_400x50_mode_summary.json`
- mode summary CSV: `examples/Revision_v1/output/s1_mitigation_400x50/s1_postprocessing/s1_mitigation_400x50_modes_summary.csv`
- manifest: `examples/Revision_v1/output/s1_mitigation_400x50/s1_postprocessing/s1_mitigation_400x50_manifest.json`
- per-mode energy CSVs: `s1_exp3_400x50_mode_##_energy.csv`
- first six mode-shape figures: `s1_exp3_400x50_mode_##_shape.png`

## Interpretation

The classification is based on final-topology K/M reassembly, the first 10 eigenmodes, MAC to the saved solid-reference target mode, elementwise kinetic and strain energy fractions in `rho < 0.05` regions, energy participation/localization metrics, and association with solid components touching the support lines.
