# PROMOTION — Phase 9: **NOT PERFORMED**

# `PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED`

Promotion is gated on `PROMOTION_PROVENANCE_PASS`
(`PREREGISTRATION.md` §5). The gate returned
**`PROMOTION_PROVENANCE_BLOCKED`**, so nothing was promoted and the frozen stop
condition applies.

## What was NOT changed

| Path | State at task end |
|---|---|
| `+impl/` tree hash | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` — **unchanged**, 75 files, manifest-verified, `CURRENT` |
| `+olh/+presets/duOlhoffFrozenM4.m` | untouched — `move.levels` still `[0.04 0.02 0.01 0.005]` |
| `olhoffcurrent_config.m`, `olhoffcurrent_preset.m` | untouched |
| `exhaustion.m`, `limit.m`, `olhoffSolve.m` | untouched |
| production `cfgHash` at 320×40 | `2a5b500991ff5931eef4919e768503d41fd558fe56ffbffe40c382c74d676cad` — **unchanged** |
| canonical production policy | `[0.04 0.02 0.01 0.005]`, `signal = boundVariable`, `stop.rule = designChange` |

Canonical production therefore still runs the **four-rung beta-continuation**
policy. That is the pre-existing state, not a regression introduced here.

## The blocking reason, stated once

The original `C240x30_trajectory.mat` is not on this host, and five further
git-ignored container artifacts (`C160`, `C320`, `C400`, `F400`, `P400`) have
digests that no action available here can reconcile. Every remaining blocker is
a **file transfer or an explicit owner decision**; none is scientific, and none
is resolvable by compute. `PROMOTION_PROVENANCE.md` §3.

## What is ready, so the next attempt is mechanical

- the exact validated policy, **re-resolved and hash-matched** to the run that
  produced it (`VALIDATED_POLICY_RECOVERY.md`);
- the complete promotion delta — **3 computational fields, 0 unexpected** — with
  the recommended implementation route (`PROMOTION_DIFF_PLAN.md`);
- proof that production already dispatches to the exact validated controller
  code (`DISPATCH_EQUIVALENCE.md`);
- two of the three provenance repairs already applied and verified
  (`HISTORICAL_HASH_REPAIR.md`, `TIMING_TELEMETRY_AUDIT.md`).

No scientific run stands between the current state and promotion. C320 must not
be re-run.
