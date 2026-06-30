# Stabilized OlhoffApproachExact Pilot

Generated: 2026-06-30T08:22:05Z

## Scope

Single CC 80x10 OlhoffApproachExact pilot with alpha-damping. No ourApproach changes, no manuscript edits, no full revision experiments.

## Settings Verification

- only changed parameter: `alpha` 1 -> 0.5
- verification pass: `true`
- mesh: 80x10, volfrac=0.5, mass=du2007_c1, p=3, rmin_elem=2.5
- inner MMA: max_iter=30, tol=0.0001, move_lim=Inf, outer_move=0.2

## Acceptance

| metric | value |
|---|---:|
| classification | still 2-cycle |
| outer iterations | 400/400 |
| final design change | 0.0763754 |
| outer tolerance | 0.001 |
| final volume | 0.499814 |
| final multiplicity N | 1 |

## First 6 Eigenfrequencies

| mode | omega rad/s | Hz |
|---:|---:|---:|
| 1 | 278.84183 | 44.379055 |
| 2 | 391.86951 | 62.36797 |
| 3 | 565.10705 | 89.939581 |
| 4 | 654.76743 | 104.20947 |
| 5 | 708.24898 | 112.72133 |
| 6 | 904.42155 | 143.94316 |

## S1 Diagnosis

- overall: ambiguous
- localized low-density modes: 0/10
- physical global modes: 0/10
- mode 1 low-density strain fraction: 0.0029267069
- support-connected solid components: 0

## Comparison Against Undamped Pilot

| metric | undamped alpha=1 | damped alpha=0.5 |
|---|---:|---:|
| omega parity gap, last window | 27.0265 | 6.96163 |
| beta parity gap, last window | 270.267 | 7.5174 |
| design change final | 0.14993 | 0.0763754 |
| localized low-density modes | 0 | 0 |
| mode 1 ld_strain_frac | 0.000195844 | 0.00292671 |
| support-connected components | 0 | 0 |
| 2-cycle removed |  | false |

## Conclusion

Classification: **still 2-cycle**. Last-window omega parity gap 6.962 rad/s with design-change mean 0.07674.
