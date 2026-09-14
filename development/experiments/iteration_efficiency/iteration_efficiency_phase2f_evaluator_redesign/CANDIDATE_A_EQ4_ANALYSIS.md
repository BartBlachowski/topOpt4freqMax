# WP7 — Candidate A: original Du–Olhoff Eq. (4), lowest algebraic mode
PHASE 2F — EVIDENCE ONLY

## Its strength, now quantified across all eight meshes

**Eq. (4) suppresses artificial modes completely.** Over the whole survey, in every `eq4`
configuration at every mesh:

| configuration | modes classified artificial | first-structural ordinal (max) | structural voidKE max |
|---|---|---|---|
| `E2/eq4`, all meshes | **0** | **1** | 0.000000 – 0.000001 |
| `E3/eq4`, all meshes | **0** | **1** | 0.000000 – 0.000001 |

Not one mode in any surveyed `eq4` state carries appreciable kinetic energy in the
low-density region, and the lowest algebraic mode is the structural mode at **every single
state examined**. At the decisive state (160x20, k = 252) all fourteen lowest modes have
`voidKE = 0.000000` and density-participation 0.919–0.943; the mode-shape atlas
(`figures/atlas_k252_E2_eq4.png`) shows fourteen global structural patterns and no
localisation anywhere.

This is the property no other candidate has for free. The `ρ^6` suppression drives void mass
so low that void modes are pushed entirely out of the computed spectrum — which is precisely
what Du & Olhoff §2.2 introduces Eq. (4) to achieve, and it works exactly as advertised for
gray intermediate states.

## Its failure, independently confirmed

The Phase-2C/2D/2E evidence stands unchanged and was re-verified in Phase 2E:

| mechanism | Eq. (4) | branch-free E1 control |
|---|---|---|
| branch straddle, nextbelow(0.1) → nextabove(0.1) | **4.0021e-03** | 8.1090e-13 |
| float32 storage, 236 genuine paired states | **2.6736e-02** | 5.5949e-08 |
| branch side, 1600 production states at 160x20 | **2.6496e-02** | 2.6252e-10 |

Against the frozen decision machinery, the binding critical relative perturbation on the only
reference-length trajectory available is **5.16e-05**. The Eq. (4) float32 error is
**439× larger**, and exceeded it on **48.7%** of states. That is why `b_ref` moved 2200 → 2100
and `k_enter` moved at all three q levels in Phase 2B.

## Frequency continuity and mode identity along the trajectory

Measured on the 160x20 trajectory (`QUALITY_SEQUENCE_SUITABILITY.csv`):

| candidate | median step | max step | steps > 0.5% |
|---|---|---|---|
| **A** Eq. (4) lowest | 2.084e-04 | **3.254e-02** | 99 |
| B Eq. (4a) lowest | 1.152e-04 | 1.222e+00 | 131 |
| C Eq. (4a) structural | 1.100e-04 | 1.275e-02 | 99 |

Candidate A's sequence is well behaved — no discrete jumps, no undefined states, and mode
identity is trivially stable because the structural mode is always ordinal 1. Its 3.25e-02
maximum step is the branch-crossing instability itself, not a mode-selection artifact.

## Assessment

Candidate A is **physically valid and modally clean at every state surveyed**, and
**numerically unfit at the frozen decision scale**. Those two facts are independent and both
are firmly established. The instability is a property of the discontinuity, not of the
low-density suppression; the suppression is a property this study needs and A is the only
candidate that provides it without a selection rule.

Retaining A is not an option — Phase 2C settled that and Phase 2E re-confirmed it — but the
reason A must be replaced is *solely* its discontinuity, and any replacement must be shown to
preserve the artificial-mode suppression that A delivers for free. Candidates B and D fail
and pass that test respectively; candidate C recovers it by explicit selection.
