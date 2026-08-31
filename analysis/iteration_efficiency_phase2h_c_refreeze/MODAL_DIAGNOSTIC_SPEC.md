# Modal diagnostic specification

For every computed mode and each of E1/E2/E3, archive frequency, normalized residual,
`voidKE`, `voidSE`, `densityParticipation`, IPR, the three signed classifier margins, and
the unanimous-valid flag. Also archive selected ordinal/frequency, modes requested,
escalation count, evaluator status, and precision class.

`voidKE` and `voidSE` are the fractions of element kinetic and strain energy in the
`rho_eff<=0.1` region. `densityParticipation` is the density-weighted normalized kinetic
participation `sum(KE_n*rho_eff)`. Fractions must be finite and normalized. An eigenpair with nonfinite values,
nonpositive eigenvalue, or residual above `1e-6` is numerically invalid and cannot be
selected. Diagnostic disagreement below the selected ordinal must remain visible.
