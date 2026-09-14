# RUNNER_INVOCATION (Part 6)

The agent launched the campaign. The operator did not start MATLAB.

## Shell command (verbatim from `logs/campaign_LAUNCH.json`)

```sh
analysis/OlhoffCurrent/diagnostics/nine_mesh_pedersen_production/scripts/nmp_launch.sh campaign
```

After its refusal checks, the launcher executes the following:

```sh
cd "/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffCurrent/diagnostics/nine_mesh_pedersen_production/logs/matlab_cwd_campaign" && \
NMP_LOCK="/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffCurrent/diagnostics/nine_mesh_pedersen_production/CAMPAIGN_LOCK.json" \
NMP_LOCK_SHA256=202c1f272c8387e8f3c5ac69203898b859d4798cd1833452ffbbdbd74921f9b5 \
/usr/bin/caffeinate -i -s /Applications/MATLAB_R2025b.app/bin/matlab -batch \
"addpath('/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffCurrent/diagnostics/nine_mesh_pedersen_production/scripts'); addpath('/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffCurrent'); nmp_arm_hooks(); run('/Users/piotrek/Programming/topOpt4freqMax/examples/Performance/performance_comparison.m'); nmp_after_run();" \
> "/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffCurrent/diagnostics/nine_mesh_pedersen_production/logs/campaign_performance_comparison_stdout.log" 2>&1
```

## What each part does

| part | role |
|---|---|
| `matlab -batch` | A non-interactive MATLAB process, the same convention as `postmerge_campaign_gate/scripts/pmg_job.sh`. |
| `cd …/matlab_cwd_campaign` | An empty working folder, so nothing in the current folder can shadow a function. The launcher refuses a non-empty folder. |
| `caffeinate -i -s` | Prevents idle and system sleep for the life of the process. It does not change priority or affinity. |
| `NMP_LOCK`, `NMP_LOCK_SHA256` | Tell the observation hooks which lock to verify against. Environment variables survive the runner's `clear`. |
| `addpath(.../scripts)` | Makes the `nmp_*` hook functions visible. They own no production symbol names, and the runner's dispatch gate still proves that `+impl` is the only Olhoff implementation. |
| `addpath(analysis/OlhoffCurrent)` | Makes `olhoffcurrent_run.m` resolvable so its breakpoints can be armed. The runner adds the same folder again itself (`performance_comparison.m:129`). |
| `nmp_arm_hooks()` | Verifies the lock, identity, the committed `olhoffcurrent_run.m` and the hook line text. It then arms the three conditional breakpoints and writes `logs/campaign_HOOKS_ARMED.json`. It errors, so the runner never starts, on any mismatch. |
| `run('.../performance_comparison.m')` | **The production campaign**: the authoritative runner with its committed logic and the two locked run-selection literals. |
| `nmp_after_run()` | Reached only if the runner returns normally. It records a final identity snapshot and whether the breakpoints were still armed. |

## Launcher refusal checks (`scripts/nmp_launch.sh`)

- The lock SHA-256 must equal `CAMPAIGN_LOCK.json.sha256`, and the lock mode must be `campaign`.
- `git rev-parse HEAD` must equal the locked HEAD `b21483b158f58e05e7b56957f2fbe8e1d2891395`.
- The runner file must hash to the locked edit `05d043c3…`.
- `git diff --name-only HEAD` must be exactly `examples/Performance/performance_comparison.m`.
- The output root `examples/Performance/conference_benchmark/nine_mesh_pedersen_b21483b` must not exist.
- The stdout log must not exist.
- The MATLAB working folder must be empty.

## Timestamps

The start time, end time and exit code are in `logs/campaign_LAUNCH.json` and `logs/campaign_END.json`, and they are copied into `RUN_INDEX.json`.
