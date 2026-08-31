# WP1 — Source provenance verification (independent)
READ_ONLY_INDEPENDENT_DELTA_AUDIT — NO_OPTIMIZATION — NO_REFREEZE

The primary source was read directly, not through Phase-2D or Phase-2C paraphrase.

    references/Du2007_Topological.pdf
    sha256 (recorded in WP0_INTEGRITY_pre.json)
    Du J., Olhoff N., Struct Multidisc Optim (2007), section 2.2 "Localized eigenmodes",
    journal pages 93–94.

Text extracted with `pdftotext -layout`; transcript retained at
`scripts/du2007_section22.txt`.

## 1. Eq. (4) is the discontinuous, coefficient-free formulation — CONFIRMED

The source prints, verbatim:

>                  ρe Me ;  ρe > 0.1
>     Me(ρe) =                            (4)
>                  ρe^r Me ; ρe ≤ 0.1

with "the mass is set very low via a high value of the penalization power r … r is chosen
to be about r = 6". **No coefficient appears in Eq. (4).** The repository's `massScale.m`
case `'4'` and `mass_interp.m` mode `du2007_step` both implement exactly `rho.^6` on the
low branch, with no coefficient.

## 2. The source explicitly acknowledges the discontinuity — CONFIRMED

Verbatim:

> "It is noted that (4) is discontinuous at the low value ρe = 0.1 of the material density.
> Numerically, this is not a serious problem, as the discontinuity only occurs at a single
> point. However, we can always improve (4) by generating a continuous interpolation model
> for the mass with respect to any value of the material density between 0 and 1."

The discontinuity is therefore **intentional and disclosed by the authors**, not a
transcription defect in this repository.

## 3. Eq. (4a) is a distinct, source-defined alternative — CONFIRMED

The source introduces it as "the following revised form of (4)", numbered **(4a)**, and
separately numbers **(4b)** for the C1 variant. All three are presented as alternatives that
"we have applied … in the numerical solution scheme", not as corrections to one another.

## 4. c0 = 1e5 — CONFIRMED

Verbatim: "where the coefficient c0 = 10^5 enforces the C0 continuity at the value
ρe = 0.1 of the material density." `massScale.m` case `'4a'` uses `c0 = 1e5`;
`mass_interp.m` mode `du2007_c0` uses `c0 = 1e5`. Both match.

For (4b) the source gives "c1 = 6×10^5 and c2 = −5×10^6 ensure the C1 continuity";
`massScale.m` case `'4b'` uses `c1 = 6e5; c2 = -5e6`, and `mass_interp.m` `du2007_c1`
the same. Both match.

## 5. Eq. (4a) is continuous at ρe = 0.1 — CONFIRMED

Verified in exact rational arithmetic (see `MATHEMATICAL_VERIFICATION.md`):
`1e5·(1/10)^6 − 1/10 = 0` exactly in ℚ.

## 6. Eq. (4a) is NOT a typo correction of Eq. (4) — CONFIRMED

The source states Eq. (4) with its discontinuity, *discusses* that discontinuity, and only
then offers (4a) and (4b) as optional improvements. The hypothesis that a normalisation
coefficient had been dropped from Eq. (4) was raised and refuted by Phase 2C; this audit
re-read the source and reaches the same conclusion independently. The repository is more
complete than the frozen evaluator: it implements `'lin'`, `'4'`, `'4a'`, `'4b'`, and the
evaluator's use of `'4'` is a deliberate selection of the native variant.

## 7. Repository implementations correspond to the source — CONFIRMED

| source | `massScale.m` | `mass_interp.m` | agrees |
|---|---|---|---|
| Eq. (2), q = 1 | `'lin'` → `rho` | `linear` / `olhoff2014_pow` | yes |
| Eq. (4), r = 6 | `'4'` → `rho.^6` | `du2007_step` | yes |
| Eq. (4a), c0 = 1e5 | `'4a'` → `1e5*rho.^6` | `du2007_c0` | yes |
| Eq. (4b), c1 = 6e5, c2 = −5e6 | `'4b'` | `du2007_c1` | yes |

Both files verified byte-unchanged against the Phase-2A record (WP0).

## 8. Native Olhoff legitimately remains on its native frozen interpolation — CONFIRMED

`Matlab/reproduction2007/algo/defaultCfg.m` sets `massInterp = '4'`; the file is unmodified
and hash-matched. All six `protected_numerical_sources` re-verify. Phase 2D did not touch
any optimizer.

## What the source does NOT license — the decisive omission

The authors justify their indifference between (4), (4a) and (4b) on one stated ground:

> "In several of the examples … we have applied each of the three different interpolation
> models … and only found negligible differences in the final results. **The reason is that
> the region with lower density in all the three models has a very small contribution to the
> first several eigenfrequencies of the structure. Furthermore, all intermediate values of
> the material density will approach 0 or 1 during the design process**, which implies that
> the changes of the interpolation model in regions with lower density as shown in (4a) or
> (4b) must have very limited influence on the **final 0–1 design**."

Both premises are about **converged 0–1 designs**. Phase 2C established that the second
premise fails for this study, which scores **gray intermediate trajectory states**. Phase 2C
used that failure to argue against Eq. (4). **The same failure applies to Eq. (4a) and
Eq. (4b), and Phase 2D did not test it.** Section 2.2 states plainly what the low-density
branch is for:

> "application of the SIMP model … may lead to the occurrence of spurious, localized
> eigenmodes associated with very low values of corresponding eigenfrequencies … To
> eliminate these spurious eigenmodes …"

Whether Eq. (4a) still eliminates them **on gray intermediate states** is an empirical
question the source does not answer for this use. This audit answered it: it does not. See
`PHASE2D_DELTA_AUDIT.md` finding **D1** and `SPURIOUS_MODE_CHECK.csv`.

**WP1 ruling: SOURCE_PROVENANCE = PASS.** Every provenance claim Phase 2D makes about the
source is accurate. The defect found by this audit is not a provenance error; it is an
untested transfer of the source's own justification to a use the source did not contemplate.
