# Comparison: OlhoffApproachExact vs ourApproach (CC 80×10)

Generated: 2026-06-30T07:44:06Z

| metric | OlhoffApproachExact | ourApproach (scaling study) |
|---|---|---|
| solver | OlhoffApproachExact | ourApproach (semi_harmonic) |
| objective | maximize omega_1 (bound form.) | minimize semi-harmonic compliance |
| mass interpolation | du2007_c1 (Eq. 4b) | power (pmass=1, linear) |
| stiffness | SIMP no Emin (rho^p) | SIMP with Emin |
| penal | continuation 1->2->3 | fixed 3 |
| omega_1 (rad/s) | **249.7** | 3.977 |
| total outer iterations | 400 | 1309 |
| final multiplicity N | 1 | N/A (different formulation) |
| S1 overall | ambiguous | likely localized/spurious low-density mode influence |
| S1 mode 1 class | ambiguous (ld_strain=0.0001958) | localized (ld_strain=0.992) |
| S1 localized/10 | 0 | 6 |
| S1 physical/10 | 0 | 0 |

## Mode frequency table (OlhoffExact)

| mode | omega (rad/s) | Hz | S1 class |
|---:|---:|---:|---|
| 1 | 249.7 | 39.73 | ambiguous |
| 2 | 315.4 | 50.2 | ambiguous |
| 3 | 345.3 | 54.95 | ambiguous |
| 4 | 459 | 73.05 | ambiguous |
| 5 | 499.3 | 79.47 | ambiguous |
| 6 | 627.2 | 99.83 | ambiguous |
