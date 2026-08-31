# WP2B — Historical Mo9 scope, separated from new-trajectory authorization

## A. HISTORICAL_Mo9_SCOPE

`IMPLEMENTATION_REQUIREMENTS.md` section 4 permits the historical single-precision
Olhoff snapshot path for offline rescans at the eight available meshes, on a recorded
verified equivalence that is explicitly limited to:

> "at those final states the `float32` snapshot and the `float64` field give identical
> component counts and detached areas under the exact-count projection, and about
> 7-digit precision is comfortable against a 1% spectral band."

and scopes the permission:

> "any *new* offline quantity derived from the historical snapshots must repeat the
> equivalence check and record it."

### Quantities the frozen iteration-efficiency methodology derives from Olhoff snapshots

| Offline quantity | Covered by the recorded Mo9 equivalence? | Repeat check performable on historical data? |
|---|---|---|
| exact-count binary projection | YES, at final states only | at final states only (`res.rho` is double there) |
| component counts / detached areas | YES, at final states only | at final states only |
| topology PASS/FAIL (hard gate incl. volume_pass) | NO — Mo9 covers counts and areas, not the gate conjunction | at final states only |
| E1 | NO | NO — double unrecoverable at intermediate states |
| E2 | NO | NO |
| E3 | NO | NO |
| robust/common quality ratio | NO | NO |
| reference phase, `b_ref`, `Q_ref` | NO | NO |
| `B_meas`, tail truncation | NO | NO |
| persistence scan, `k_enter`, `k_cert` | NO | NO |
| final status classification | NO | NO |

**DOUBLE ORIGINAL NOT RECOVERABLE FROM FROZEN SNAPSHOT** at every intermediate state.
`double(single(snapshot))` is not an independent double reference and was not used as one.

### Result

The recorded Mo9 equivalence covers **topology component counts and detached areas at
final states only**. Every spectral quantity the iteration-efficiency study needs from
historical Olhoff snapshots (E1/E2/E3, robust quality, reference, `B_meas`, `k_enter`,
`k_cert`, status) is a **NEW offline quantity** in the Mo9 sense, and for all of them the
required repeat equivalence check is **NOT PERFORMABLE** on historical data.

This recheck additionally shows the 1%-spectral-band premise behind Mo9 does not hold for
E2/E3: measured E2/E3 single-storage error reaches 2.27e-2 on genuine paired evidence
(section WP8 of the main report), i.e. above a 1% band, because the E2/E3 mass law is
discontinuous at x = 0.1 rather than merely finitely precise.

**HISTORICAL_Mo9_SCOPE outcome: NOT EXTENDABLE.** The historical permission remains valid
only for what it recorded — final-state component counts and detached areas. It does not
extend to the spectral pipeline.

## B. NEW_TRAJECTORY_STORAGE_SCOPE

Determined independently in the main report from genuine paired double/single evidence
produced by checkpoint-limited reruns of the unmodified protected source.

The contract is explicit that A does not imply B:
`trajectory_storage.historical_olhoff_single_precision` =
"historical_evidence_only_not_automatic_authorization_for_new_trajectories".
No part of this qualification cites historical acceptance as evidence for new-trajectory
storage.

**NEW_TRAJECTORY_STORAGE_SCOPE outcome: NOT QUALIFIED.** See the main report.
