# WP2/WP3/WP4 — Provenance of the x = 0.1 / x^6 law
READ_ONLY_AUDIT — NOT_NEW_OPTIMIZATION_EVIDENCE

## Provenance chain

    PAPER   Du & Olhoff (2007), Struct Multidisc Optim, "Topological design of freely
            vibrating continuum structures...", section 2.2 "Localized eigenmodes",
            Eq. (4)  [references/Du2007_Topological.pdf]
              |
              |  adopted verbatim, cited, by
              v
    PAPER   Yuksel (2025), Eq. (10)  [references/Yuksel2025_Efficient.pdf]
              |
              v
    METHOD  Olhoff:  Matlab/reproduction2007/fem/massScale.m case '4'
                     default set at algo/defaultCfg.m:12  massInterp='4'
            Yuksel:  analysis/YukselApproach/Matlab/top99neo_inertial_freq.m:423-424
                     dMass=6.0 (line 84), xMassCut=0.1 (line 85)
              |
              v
    SPEC    iteration_efficiency_contract.json quality.evaluators:
            E2 "mass": "1e-9+(1-1e-9)*g(x), g=x^6 at x<=0.1 else x"
              |
              v
    CODE    analysis/three_method_parametric_study/study_evaluate_design.m:43-55

**The discontinuity is present at the very first point in the chain.** It originates in
the source literature, not in this repository.

## What the source actually says

Du & Olhoff (2007) print Eq. (4) in exactly the coefficient-free form:

> Me(ρe) = ρe Me,  ρe > 0.1 ;   ρe^r Me,  ρe ≤ 0.1

and then state, in the immediately following paragraph:

> "It is noted that (4) is discontinuous at the low value ρe = 0.1 of the material
> density. Numerically, this is not a serious problem, as the discontinuity only occurs at
> a single point. However, we can always improve (4) by generating a continuous
> interpolation model for the mass with respect to any value of the material density
> between 0 and 1."

They then give (4a) with "the coefficient c0 = 10^5 [which] enforces the C0 continuity at
the value ρe = 0.1", and (4b) with "c1 = 6×10^5 and c2 = −5×10^6 [which] ensure the C1
continuity". They add:

> "In several of the examples presented later in this paper, for comparison, we have
> applied each of the three different interpolation models, i.e., (4), (4a), and (4b) in
> the numerical solution scheme and only found negligible differences in the final
> results. The reason is that the region with lower density in all the three models has a
> very small contribution to the first several eigenfrequencies..."

Yuksel (2025) Eq. (10) restates the same discontinuous two-branch form with `d >> 1`,
citing Pedersen (2000), Tcherniak (2002), Du & Olhoff (2007) and Yuksel & Yilmaz.

## WP3 — is a coefficient missing?

**No.** This audit specifically tested the hypothesis that a normalization coefficient had
been dropped, because `1e5 · 0.1^6 = 0.1` exactly restores continuity and would be the
natural candidate. The hypothesis is **refuted**: Eq. (4) in the source is genuinely
coefficient-free, and c0 = 1e5 belongs to Eq. (4a), which the source presents as a
*separate, optional improvement*, not as part of (4).

The repository is in fact more complete than the evaluator: both `massScale.m` and
`mass_interp.m` implement all of `'4'`, `'4a'`, `'4b'` (and `'lin'`), with docstrings that
correctly state the continuity properties. The evaluator uses `'4'`. This is a deliberate
selection of the native variant, not a transcription error.

One documentation inaccuracy is noted, without methodological consequence:
`mass_interp.m:25` describes `du2007_step` as having a "Derivative discontinuous at 0.1",
which understates it — the *function itself* is discontinuous there, not merely its
derivative.

## WP4 — what is the branch FOR, and does that purpose survive re-use?

The source is explicit about purpose. Section 2.2 states that low-density subregions have
a very small stiffness-to-mass ratio (p = 3 against q = 1), which produces "spurious,
localized eigenmodes associated with very low values of corresponding eigenfrequencies".
Eq. (4) exists "to eliminate these spurious eigenmodes", following Pedersen (2000)
(linearizing the stiffness) and Tcherniak (2002) (setting low-density element mass to 0),
"with a slight modification to avoid numerical singularity".

So the branch is:

- **NOT a physical material model.** It is a numerical device.
- **An artificial-mode suppression mechanism** internal to the optimization scheme.
- Justified by the authors on the ground that the suppressed region "has a very small
  contribution to the first several eigenfrequencies".

Two premises underwrite the authors' tolerance of the discontinuity:

  P1. "the discontinuity only occurs at a single point" — a measure-zero argument;
  P2. "negligible differences in the final results" — an argument about converged 0–1
      designs.

Both premises hold for the use the authors made of it. Neither holds for the use the
frozen iteration-efficiency methodology makes of it:

- **P1 fails** because the Olhoff optimizer does not sample densities at random. WP8 shows
  it *parks* elements on a value 26 double-ULPs below 0.1. A measure-zero set that is also
  a deterministic attractor of the update law is reached with probability 1, not 0.
- **P2 fails** because the study does not compare final designs. It evaluates *gray
  intermediate states along a trajectory* and extracts the *iteration index* at which a
  quality band is first persistently attained. That estimand is sensitive to per-state
  evaluator values at the 0.5–2% level, which is exactly the scale of the observed effect.

There is a third mismatch the source never contemplated: E2/E3 are applied **post hoc to
another method's trajectory**. A device designed to keep a *particular optimizer* away
from spurious modes carries no guarantee of neutrality when used to score a design that a
different optimizer produced.
