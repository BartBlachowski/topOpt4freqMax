# Campaign integrity

**NINE_MESH_CAMPAIGN_INTEGRITY_FAIL** — for the requested intended three-rung campaign. Every case is INVALID for that identity, starting at 160×20. They remain usable legacy endpoint observations with caveats. “Invalid campaign” here does not mean numerical file corruption was discovered.

| Mesh | Intended campaign | Legacy endpoint | Recorded status | Outer | Inner | rminEl | Last stage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 160x20 | INVALID | VALID_WITH_CAVEAT | NATIVE_CONVERGED | 91 | 2241 | 1.2 | 3 |
| 240x30 | INVALID | VALID_WITH_CAVEAT | NATIVE_CONVERGED | 104 | 2334 | 1.8 | 3 |
| 320x40 | INVALID | VALID_WITH_CAVEAT | NATIVE_CONVERGED | 131 | 2614 | 2.4 | 2 |
| 400x50 | INVALID | VALID_WITH_CAVEAT | NATIVE_CONVERGED | 139 | 2918 | 3 | 2 |
| 480x60 | INVALID | VALID_WITH_CAVEAT | NATIVE_CONVERGED | 164 | 3463 | 3.6 | 2 |
| 560x70 | INVALID | VALID_WITH_CAVEAT | NATIVE_CONVERGED | 190 | 3922 | 4.2 | 2 |
| 640x80 | INVALID | VALID_WITH_CAVEAT | NATIVE_CONVERGED | 199 | 4324 | 4.8 | 2 |
| 720x90 | INVALID | VALID_WITH_CAVEAT | NATIVE_CONVERGED | 223 | 4831 | 5.4 | 2 |
| 800x100 | INVALID | VALID_WITH_CAVEAT | NATIVE_CONVERGED | 170 | 3713 | 6 | 2 |


All nine density vectors have exactly NE finite values within [0.001,1], finite ordered positive final frequencies, matching volume/grayness, empty error strings, and zero reported unconverged inner solves. The raw MAT agrees with JSON on all compared counts, timings, terminal metrics, logs, frequency arrays, statuses and config hashes. All effective configs and physical radii are verified. The 400-outer cap was not hit by any Olhoff record.

However, **none retains its full Olhoff trajectory**. `olhoffcurrent_run` reduces `res.hist` to aggregate accounting and final stopping fields without returning the history; `confbench_run_case` preserves those fields but leaves Olhoff telemetry empty. `benchmark_records.mat` therefore contains no Olhoff `hist`, `RHO`, `DRHO`, beta path or E path. JSON additionally removes effective_config, x and telemetry. A final vector of the right length cannot prove complete iteration history, transient finiteness or absence of truncated trajectory data.

Expected campaign export files (MAT, results JSON, manifest, tables, timing schema, notes) exist. Raw data have no contemporaneous scientific-evidence declaration or checksum seal; current audit hashes establish today's bytes only. Tracked JSON matches committed HEAD. Stale ancillary graphics are not used. [INTEGRITY_TABLE.csv](INTEGRITY_TABLE.csv) records every per-case check.

The source path guard passed, but it guards implementation identity, not promotion of the intended policy. The software contained E-controller code while the selected preset remained beta. No evidence supports a silent exception-driven fallback. This is a configuration/promotion failure and an evidence-retention failure, not a falsification of the unexecuted controller.
