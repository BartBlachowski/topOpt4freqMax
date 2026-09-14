# MULTIPLICITY_CONTROL — Part 10

## Why 480×60 is the clean case

| | ρ₃₈₅ (subproblem built here) | ρ₃₈₆ (frozen endpoint) |
|---|---|---|
| ω₁ | 163.93172748 | 163.93225939 |
| ω₂ | 185.20866045 | 185.21032465 |
| ω₃ | 401.3984225 | 401.39946950 |
| `gap12` | 0.129791 | **0.129798** |
| `gap23` | — | 1.167263 |
| `multJ` warnings at 386 | 0 | 0 |
| `N` (fixed subspace) | 2 | 2 |

`gap12 = 0.1298` is **2.6×** the multiplicity tolerance of 0.05. The first mode
is simple by a wide margin, so `∂λ₁/∂ρ` is a smooth, single-valued function of ρ
in a neighbourhood of the frozen state, and the positive control is meaningful.

This is why the preregistration names 480 and not 800. At 800×100 the
three-rung endpoint has `gap12 = 3.49e−05` — the first pair is numerically
degenerate, `∂λ₁/∂ρ` is not differentiable in the ordinary sense there, and a
measured Jacobian asymmetry could not be separated from mode interaction. No
800 evaluation was performed in this audit.

## Screening, applied to every perturbed evaluation

Every evaluation in Parts 5–7 recorded ω₁…ω₅ and `gap12`. The preregistered rule
excludes a sample if `gap12 < 0.05` or the mode ordering changes.

| test | samples | excluded | min `gap12` | ordering changes |
|---|---|---|---|---|
| Jacobian symmetry (Part 5) | 88 | **0** | 0.12979 | **0** |
| mixed partials (Part 7) | 61 | **0** | — | **0** |

No sample came within a factor of 2.5 of the exclusion threshold. The
preregistered "drop the δ if > 5 % excluded" rule was never triggered, and no δ
was dropped.

## Can mode switching explain the measured asymmetry?

No, on four independent grounds:

1. **No sample crossed a boundary.** `gap12` stayed within 0.12979 ± 1e−5 across
   all 88 perturbed evaluations.
2. **The positive control uses the same evaluations.** `g_phys` is computed from
   the identical eigenpairs at the identical perturbed densities and shows
   asymmetry at the 1e−06 level. If mode switching were corrupting the
   eigenpairs, it would corrupt both fields; it corrupts neither.
3. **δ-independence.** Mode-interaction artefacts scale with the perturbation;
   the filtered asymmetry is constant to 4 significant figures over a 33× range
   of δ.
4. **The asymmetry is predicted analytically.** `FILTER_OPERATOR_ANALYSIS.md` §5
   derives it from the operator structure alone — no eigenvalue derivative is
   needed to know that `A ≠ Aᵀ` — and the closed-form prediction closes the
   measured budget to ≤1e−3 on 9 of 10 pairs.

## Scope

This audit makes no claim about 800×100 or about near-degenerate states. The
`gray_kkt_forensic_audit` already recorded that near-cluster treatment is
material at 800; nothing here revisits it.
