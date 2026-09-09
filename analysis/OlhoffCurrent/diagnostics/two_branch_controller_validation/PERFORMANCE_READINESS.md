# PERFORMANCE_READINESS — is the nine-mesh campaign authorized?

**Decision: `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`.**

No part of that campaign was run in this task, and none is authorized by it.

---

## 1. The authorization rule

Preregistration §12, and Phase 24 of the brief, make authorization conjunctive:

| requirement | status |
|---|---|
| `TWO_BRANCH_CONTROLLER_VALIDATED` | **not met** — the verdict is `PARTIALLY_VALIDATED` |
| `CONTROLLER_PROMOTION_EQUIVALENCE_PASS`, if promotion occurred | not applicable — no promotion occurred |
| production source / config frozen | met |
| tests pass | met |
| evidence complete | **not met at the time of writing** — one declared raw trajectory still regenerating (§3) |
| no residual controller blocker | **not met** — see §2 |

The first and last fail, so the campaign stays blocked. Because promotion did not
occur, `CONTROLLER_PROMOTION_EQUIVALENCE_PASS` / `_FAIL` is **not issued** at all;
the question does not arise.

## 2. The residual blocker, stated precisely

The frozen union `E = A OR B` cannot detect a **low-amplitude cancelling** regime:
`amp < tol` satisfies Branch B's amplitude clause but `med₂₀cosθ < 0` fails its
coherence guard, while Branch A is simultaneously blocked by its own `amp ≥ tol`
clause. Nothing in the union covers that quadrant. This was preregistered as a known
hole and an accepted risk (§10); at 320×40 it was realized.

Its consequence is not cosmetic. At 320×40 the candidate reached `move = 0.005` at
iteration 353 and then ran 1 248 further iterations in a stationary limit cycle —
M_nd spanning 0.4 % of its value with a drift of −0.00055, the move limit inactive at
`max|Δρ|/move ≈ 0.02` — consuming 91.5 % of the run's inner MMA work and 95.9 % of its
wall time before hitting the cap.

**Why this specifically blocks a nine-mesh scaling campaign**, rather than being a
tolerable blemish:

* The campaign's purpose is *performance and scaling*. A controller whose termination
  behaviour is regime-dependent produces run lengths set by whether a mesh happens to
  land in the cancelling quadrant — 219 outer at 160×20, 1 600 (capped) at 320×40.
  Iteration counts and wall times measured under it would characterize the hole, not
  the method.
* The blind quadrant is entered *more* readily at the bottom of the ladder, because
  `amp` falls mechanically with each halving of the move while `tol` is fixed per
  mesh. Every mesh in the campaign traverses all four rungs by design.
* Six of the nine campaign meshes (480×60 … 800×100) are finer than any mesh tested
  here, and the two fine meshes tested are precisely where the terminal-stage regime
  diverged. There is no evidence base for predicting which of them cap.
* A campaign in which an unknown subset of meshes ends in `CAP_HIT` at a preregistered
  cap yields uninterpretable scaling curves, and the natural repair under time
  pressure — raising the cap — is exactly what the preregistration forbids.

A second, milder finding also bears on cost measurement: after the first descent
every stage declares exhaustion at its arithmetic floor `s + 38` on both meshes
(`REPORT.md` §7), so stages 2–4 contribute a fixed ~39 iterations each that measure
nothing. At 160×20 that is 53 % of the run. Any performance figure reported under the
current controller would be dominated by these constants rather than by the
optimizer's behaviour.

## 3. Evidence completeness

**Gate P15 currently reads FAIL**, and is reported that way rather than as a
near-pass: `DATA_MANIFEST.json` records `nRawMissing = 1`, because the candidate
320×40 trajectory lost with the untracked evidence root was still being regenerated
when this was written. Everything else is complete and hash-valid, with the recovery
history disclosed rather than smoothed over. `DATA_MANIFEST.json` and `FINAL_SHA256.txt` enumerate every tracked artifact and
every untracked raw trajectory with sizes and SHA-256s, marking each present or
missing. The `move_activity_400` evidence gate, which reported
`ok=0, required=2, missing=2` at resumption, now reports **PASS** (2 required, 2
present/match) after both artifacts were recomputed and re-declared through the
sanctioned `ma4_declare` route, with the original declaration preserved at
`evidence/rerun_20260909/move_activity_400_EVIDENCE.ORIG.json`.

The one qualification: `.mat` **container** hashes for the recomputed 400×50 pair
necessarily differ from the originally declared ones, because a v7.3 MAT-file is HDF5
and embeds creation metadata. The *scientific content* is bitwise — P400's telemetry
is identical across all 35 columns and 139 rows, and its final density SHA-256 equals
the value frozen in `baselines.json` before the file was lost. The distinction is
recorded in `PROVENANCE.md` §A4 and is not presented as an unbroken chain of file
hashes, because it is not one.

## 4. What would unblock it

Not a decision for this task, and deliberately not acted on here. Stated only so the
blocker is not mistaken for a dead end: the controller's causal claim survived this
study intact — delaying continuation past the β stall is what produces the grayness
improvement, confirmed at both fine meshes with ω₁ improving rather than degrading.
What failed is the separate question of how a stage is declared finished at the bottom
of the ladder. Those are independent, and the second is well posed.
