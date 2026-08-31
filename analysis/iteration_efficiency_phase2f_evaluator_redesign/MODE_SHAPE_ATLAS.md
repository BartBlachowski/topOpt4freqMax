# WP6 — Mode-shape atlas
PHASE 2F — SUPPORTING PHYSICAL EVIDENCE — CLASSIFICATION IS NOT MADE BY APPEARANCE

A scalar energy ratio is not sufficient evidence that a mode is an artefact. This atlas
renders modal kinetic-energy densities so that the energy-based classification can be
checked against the physical mode shapes. Plotting conventions are identical in every
panel: element-wise `ke_n` on a shared logarithmic colour scale from 1e-8 to 1, `inferno`,
origin lower-left, no interpolation. The density field is shown above each set on a
grey scale from 0 to 1.

Files in `figures/`:

| file | content |
|---|---|
| `atlas_k252_E2_eq4a.png` | 160x20 k=252, E2 under Eq. (4a) — the decisive case |
| `atlas_k252_E2_eq4.png` | the same state under Eq. (4) — the control |
| `atlas_k252_E3_eq4a.png` | E3 under Eq. (4a) |
| `atlas_k252_E1_linear.png` | frozen E1 — a second control, different floor convention |
| `atlas_k1600_E2_eq4a.png` | converged state, E2 under Eq. (4a) |

## What the images show

**E2 under Eq. (4a), k = 252.** Modes 1, 2, 3, 6 and 7 are pinpoint concentrations of
kinetic energy in a handful of elements sitting inside the gray transition regions between
the truss members — visually a few bright pixels on an otherwise black field. Modes 4, 5 and
8 are smooth global patterns following the entire load-bearing topology, with energy
distributed along every member. There is no intermediate appearance in this state: a mode is
either a pinpoint or a global structural pattern. This matches the energy measures exactly
(voidKE 0.999985–1.0 versus 0.001008–0.001614) and the IPR (0.50–0.99 versus 0.0009–0.0033).

**E2 under Eq. (4) at the same state — the control.** All eight lowest modes are global
structural patterns, with `voidKE = 0.000000` to six decimals and density-participation
0.919–0.943. The `ρ^6` suppression removes void mass so completely that no void mode appears
anywhere in the lowest eight. **This is the property candidate A has and the others do not**,
and it is visible as well as measurable.

**Frozen E1 — the second control, and the more informative one.** E1's modes 1 and 2 are
structural (165.869, 166.019) but modes 3, 4, 6, 7 and 8 are void-localised, with
voidKE 0.9996–0.99995. So **void modes are not created by Eq. (4a)**: they are present under
the frozen E1 evaluator too, and they were present all along. What Eq. (4a) changes is
where they sit — E1 places them at 291–424, i.e. 1.76–2.56× above its structural pair,
whereas Eq. (4a) places three of them *below* the structure. E1's mode 6 (ω = 377.75,
density-participation 0.001) is a broad membrane-like motion of the entire large central
void region, not a pinpoint — the same physics at a different length scale.

This last observation matters for the whole assessment: the phenomenon is intrinsic to SIMP
evaluation of gray states with substantial low-density regions, and the mass law determines
only whether the artefacts fall above or below the structural spectrum. E1's margin at this
state is a factor of 1.76, not a comfortable one.

**Converged state, k = 1600.** No void mode appears below the structure under Eq. (4a); the
lowest mode is structural with void participation 0.0032. The pathology is a property of
gray intermediate states, not of converged designs — exactly as Du & Olhoff's own
justification for offering Eq. (4a) presumes, and exactly why that justification does not
transfer to this study.

## How the atlas was used

The images corroborate the classification; they do not make it. Every classification in
this phase is made from the three quantitative measures defined in
`MODAL_DIAGNOSTIC_DEFINITIONS.md`, and a mode is called artificial only where all three
agree. The atlas exists so that a reader can confirm that "voidKE ≈ 1" corresponds to what
it claims to correspond to — a mode that is physically confined to weak material — rather
than to some numerical artefact of the diagnostic itself.

## Boundary cases

Where the survey found modes on which the three measures disagree, they are listed in
`AMBIGUOUS_MODES.csv` and their count and position relative to the selected structural mode
are reported in `CANDIDATE_C_STRUCTURAL_MODE_ANALYSIS.md`. Ambiguous modes lying *below* the
selected structural mode are the ones that would matter; their count is reported there and
is the quantity on which candidate C's robustness turns.
