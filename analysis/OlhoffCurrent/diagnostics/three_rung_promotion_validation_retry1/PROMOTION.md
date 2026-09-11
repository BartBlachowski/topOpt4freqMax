# PROMOTION — Part I: **NOT PERFORMED**

# `PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED`

Part I is conditional on **both** `THREE_RUNG_PRODUCTION_POLICY_VALIDATED`
**and** promotion-level provenance H1–H5. The first holds
(`POLICY_VALIDATION.md`). The second does not
(`PROMOTION_PROVENANCE.md` → `PROMOTION_PROVENANCE_BLOCKED`). The condition is
a conjunction, so nothing was promoted.

## What was NOT changed

No canonical or production path was modified. Verified at task end:

| Path | State |
|---|---|
| `+impl/` tree hash | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` — unchanged, 75 files, manifest-verified |
| `olhoffcurrent_config` / `olhoffcurrent_preset` | untouched |
| `+olh/+presets/duOlhoffFrozenM4.m` | untouched — `move.levels` still `[0.04 0.02 0.01 0.005]` |
| production `cfgHash` at 320×40 | `2a5b500991ff5931eef4919e768503d41fd558fe56ffbffe40c382c74d676cad` — unchanged |

Production therefore still resolves to the four-rung ladder with
`move.continuation.signal = boundVariable` and `stop.rule = designChange`.

## The blocking item

**H1.** `analysis/OlhoffCurrent/evidence/three_rung_resolution_240/C240x30_trajectory.mat`
is `REMOTE_REQUIRED_EVIDENCE_NOT_LOCAL` (sha256 `183d7ce60d512fc2…`,
131 203 128 bytes). The brief is explicit: *"If not local: promotion remains
blocked."* It is not local, it must not be re-run, and it cannot be transferred
from this host.

H4 and H5 fail for the same three provenance classes; H2's repair is prepared
and verified but deliberately not applied while H1 blocks
(`PROVENANCE_REPAIR_STATUS.md`).

## What a future promotion would consist of — recorded, not applied

Prepared so the promotion is mechanical when provenance closes. Evidence:
`evidence/dispatch.json`.

**Configuration-only, three fields**, at 320×40 (the scope-relevant comparison;
`runtime.*` differences below are study harness settings, not policy):

| Field | production today | promoted target — the exact tested candidate |
|---|---|---|
| `move.levels` | `[0.04 0.02 0.01 0.005]` | **`[0.04 0.02 0.01]`** |
| `move.continuation.signal` | `boundVariable` | **`stageExhaustion`** |
| `stop.rule` | `designChange` | **`stageExhaustion`** |

The last two are not new to this study — they are the already-validated
`two_branch_controller_validation` arm `C` switches. **`move.levels` is the only
field this study contributes**, and `olh.config.validate` couples the other two,
so they must move together or not at all.

No source change, no duplicate implementation, no new controller. Per the
brief's preference, promotion would be **configuration-only**, using the exact
tested controller code — see `PROMOTION_EQUIVALENCE.md` §1, which shows that
code is already what production dispatches to.
