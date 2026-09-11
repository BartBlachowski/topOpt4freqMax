# POLICY_VALIDATION — Part G

## 1. Every required condition

| Condition | Result |
|---|:--:|
| dependency-specific scientific provenance PASS | ✅ `PROVENANCE.md` |
| single-factor PASS | ✅ `SINGLE_FACTOR_AUDIT.md` — 1 computational difference in 81 schema leaves |
| software tests PASS | ✅ `SOFTWARE_VALIDATION.md` — 29/29, plus 5 of 6 repository test files clean |
| **exactly one** scientific run | ✅ `SCIENTIFIC_SAFETY.md` — 1 run, 320×40, enforced in code |
| prefix equivalence PASS | ✅ `C320_PREFIX_EQUIVALENCE.md` |
| S1 / S2 / S3 exact | ✅ 274/A, 313/B, 352/B |
| termination PASS | ✅ `CONVERGED @352`, never entered 0.005 |
| terminal state oracle PASS | ✅ 13 of 13 exact, tested with `==` |
| no inner failure | ✅ `innerNonConv = 0` |
| no cap | ✅ `CONVERGED`, cap 1600 unreached |
| no scientific retuning | ✅ `SCIENTIFIC_SAFETY.md` §4 |

# `THREE_RUNG_PRODUCTION_POLICY_VALIDATED`

## 2. What has, and has not, been established

**Established at 320×40, to the bit:**

- The three-rung ladder `[0.04, 0.02, 0.01]` under frozen `E = A OR B` stage
  exhaustion reproduces the four-rung arm **exactly** through the terminal
  declaration at iteration 352 — `RHO` and `DRHO` bitwise identical over all
  4 505 600 entries, 52 telemetry columns and 36 `hist` fields with zero
  differing elements, and all three declaration events at their frozen
  iterations and branches.
- At that declaration it **terminates** instead of descending, at a terminal
  state equal to the oracle's in all 13 recorded quantities.
- Doing so eliminates 1248 outer iterations (78.00 %) and 70 034 inner MMA
  iterations (91.51 %), matching the prior prediction to the digit.
- `beta` had no authority: it drove no transition, and both descents carry a
  recorded branch and persistence window.

**Not established here, and not claimed:**

- **Any mesh law.** One mesh was run. Cross-mesh support for the three-rung
  architecture rests on the pre-existing `three_rung_architecture` and
  `three_rung_resolution_240` studies, consumed as inputs.
- **That the fourth rung is scientifically harmful.** The evidence is that it is
  *immaterial* — it changed nothing about the design reached at 352 — and
  *operationally pathological* at this mesh, where it failed to terminate within
  the cap at all. That is the case for removing it, and it is narrower than a
  claim of harm.
- **Anything about production as promoted.** Production still resolves to
  `[0.04 0.02 0.01 0.005]`. Nothing was promoted; see §3.

## 3. This verdict does not promote anything

`PROMOTION_PROVENANCE.md` returns **`PROMOTION_PROVENANCE_BLOCKED`**, on
grounds fixed in `PREREGISTRATION.md` §1 **before** the run and therefore
independent of what it showed. The binding constraint is H1: the original
`C240x30_trajectory.mat` is `REMOTE_REQUIRED_EVIDENCE_NOT_LOCAL` and cannot be
transferred from this host. H4 and H5 fail for the same three provenance classes.

Accordingly:

```
THREE_RUNG_POLICY_VALIDATED_BUT_PROMOTION_PROVENANCE_BLOCKED
PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED
NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED
```

**The validation is kept.** It rests on its own evidence — the raw trajectory,
the telemetry, the digests and the frozen preregistration — none of which
depends on the blocked provenance items. C320 must not be re-run when promotion
is later attempted. What stands between this result and promotion is a **file
transfer and two hash-file edits**: no compute, no optimization, no
re-derivation.
