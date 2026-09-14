# TARGET_IDENTITY — Part 2

```
OLHOFFCURRENT_TARGET_IDENTITY_PASS
```

Nothing listed here was modified. Evidence `evaluations/target_identity.json`
(`scripts/sd_target_identity.m`, using the target's own read-only tools).

| | |
|---|---|
| repository / branch | `/Users/piotrek/Programming/topOpt4freqMax`, `benchmark-methodology-r2` |
| HEAD | `013cc48451d33bed61c5c4eea174bbd898d548a2` ("Nine test passed") |
| dirty state at start | untracked only: `diagnostics/{c480_socp_causal_run, filtered_subproblem_integrability_audit, frozen_inner_solver_study, frozen_problem25_reference, gray_kkt_forensic_audit, nine_mesh_campaign_audit, three_rung_canary_preflight}/` — all pre-existing, untouched |
| audit branch | not created (no tracked changes to isolate; the audit is uncommitted for review) |
| `+impl` manifest | `olhoffcurrent_source_manifest()` → ok = 1, **75 files**, tree `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` = recorded; 0 mismatches / missing / extra (1 `.DS_Store` ignored) |
| `+impl` git state | clean at HEAD; last changed by commit `1438aa3` ("A & B tests", stage-exhaustion controller) |
| currentness | `UPSTREAM_AHEAD` (1 commit, informational). The tool reads the upstream branch as `repro/natural-convergence`, HEAD `6b08708`, and reports the upstream working tree dirty (see SOURCE_IDENTITY §1) |
| canonical production preset | `duOlhoffFixedPenaltySensitivityFiltered` → upstream `duOlhoffFrozenM4` |
| canonical production at 480×60 | `move.levels [0.04 0.02 0.01 0.005]`, `continuation.signal = boundVariable` (β stall), `stop.rule = designChange` + `settledMove`, cap 400; config hash `a49417d0571d3c2406d030cbab1fda58a1dad8d334991c5aaf22fa17f1d7f601` |
| frozen historical preset | `duOlhoffFrozenM4.m` SHA-256 `6ed3624cd19b3569f55b23ac…` (full digest in JSON); the conference realization, bitwise-anchored upstream (A1_frozen160) |
| validated but unpromoted candidate | three-rung `[0.04 0.02 0.01]` + `stageExhaustion` signal and stop (three_rung_promotion_closure: `PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED`) |
| authoritative C480 control used here | `three_rung_canary_preflight`, 386 outer, CONVERGED; stored cfg re-hashes to recorded `03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e`; trajectory container SHA-256 in JSON |
| production files hashed | `olhoffcurrent_{preset,config,run}.m`, `PROVENANCE.{json,md}`, `SOURCE_MANIFEST.json`, `README.md` (digests in JSON) |

## Provenance inconsistency found (recorded, not repaired)

`analysis/OlhoffCurrent/PROVENANCE.md` §3–§7 and `PROVENANCE.json` state that `+impl` is 74
files, byte-identical to `695f03b` except **one** adaptation (`hist.tOuter`). The verified state
is 75 files, and **seven** files differ from `695f03b`: `olhoffSolve.m` (tOuter + exhaustion),
`+olh/+move/limit.m`, `+olh/+move/exhaustion.m` (new), `+olh/+config/{schema,validate,fromLegacy,toLegacy}.m`.
`SOURCE_MANIFEST.json` was regenerated in `1438aa3`; the prose provenance was not. This is
documentation debt, not a production-integrity failure (the manifest is authoritative and
passes). The source plan's Phase 2 notes the same inconsistency.

## Pre-existing untracked study that contradicts the brief's premise

The brief describes a C480 exact-SOCP one-factor run as "considered next". The target already
contains `diagnostics/c480_socp_causal_run/` (untracked, 2026-09-13 11:22–12:19) with an
executed treatment that stopped fail-closed at outer 15 (`SOCP_CERTIFICATE_FAILURE`,
`C480_SOCP_CAUSAL_RESULT_INCONCLUSIVE`). This audit did **not** run it, did not modify it, and
uses it only as retained context (bang-bang exact steps; MMA attenuation).

## Tools used, read-only

`olhoffcurrent_source_manifest()` (Verify, never Write), `olhoffcurrent_currentness()` (git
`rev-parse`, `status --porcelain`, `merge-base --is-ancestor`, `rev-list --count`),
`olhoffcurrent_config(480,60)`, `olhoffcurrent_config_hash`, `olhoffcurrent_paths()`.
