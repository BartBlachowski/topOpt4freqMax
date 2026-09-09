# Provenance checkpoint — evidence gate pending

This is inventory material for a POST-HOC OFFLINE ARCHITECTURE AUDIT, not a
completed scientific audit. No substantive trajectory replay has been performed.

| Field | Recorded value |
|---|---|
| Branch | `benchmark-methodology-r2` |
| Starting HEAD | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` |
| Checkpoint HEAD (no commit made) | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` |
| Starting dirty state | 32 preexisting entries; exact status in `evidence/start.json` |
| Checkpoint dirty state | Same 32 entries plus this new audit directory |
| MATLAB installed | 25.2.0.3042426 (R2025b) Update 1 |
| MATLAB invoked by this audit | No |
| Scientific optimization runs launched by this audit | **0** |
| Canonical source tree SHA-256 | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` |
| Canonical source manifest SHA-256 | `431e7b3084507cd8b53862847b828cfd9f8948172e349bbae0c72749e8134aeb` |
| Original controller SHA-256 (`+impl/architecture/+olh/+move/exhaustion.m`) | `17b37a384b1aa5d987d9c861e16071d1140af130f92406ccc11cec4518bcae0c` |
| Causal preregistration SHA-256 | `8e323f837f7bbdaa5176d92621b4a45e4b377af131af5b3172429c629da27fbf` |
| Mechanism preregistration SHA-256 | `62748225253f85f6a2fbc1bad35489a2c201cd45ee64ff279601003354b73abd` |
| Offline preregistration SHA-256 | `b60dc9ae65d1a9f42f14bdea677fc0395550a59c8e250242bb125982a39e786a` |
| Inventory tests | Source integrity PASS; frozen hashes PASS; all present manifested inputs PASS |
| Retained prior tests | 17 controller checks and 5 suite checks recorded PASS, not rerun here |
| New rule/replay/floor tests | Pending evidence gate; not executed |
| Manifest status | **110 PASS, 1 MISSING, 0 hash mismatches; incomplete** |
| Prior P15 | **FAIL / pending C320**, unchanged |

## Preexisting C320 process

PID 63654 was already running when this task began. Its script is recorded in
`evidence/start.json`. At the latest observation it had elapsed 01:36:14, consumed
94:14.22 CPU time and was using 99.2% CPU (state RN). This demonstrates progress
between observations; it does not establish completion or eventual success.
The startup log reports MATLAB Update 1 and the frozen candidate configuration.
The raw file is still absent. The process was neither launched nor interrupted
by this audit. Its preexisting work is not counted as a new launch by this task.

Inspection of `cv_run.m` shows the raw RHO/DRHO export occurs only after the solve
returns. Starting this function again would run the scientific optimizer and is
not allowed here. No substitute trajectory was generated. If an export fails,
recovery may use retained numerical data; do not rerun the optimizer.

## Input verification

All 75 canonical source files match SOURCE_MANIFEST.json, with no source extras,
missing files or mismatches under the authoritative artifact policy. Both prior
preregistrations and every frozen definition in the retained resume provenance
match recorded hashes. The source manifest was copied as a small starting snapshot.
Large raw histories are referenced in place, not copied.

Both prior DATA_MANIFEST.json files are hashed in this audit's DATA_MANIFEST.json.
Every entry was checked against its recorded hash. The prior causal manifest
correctly states eight of nine raw inputs present. This audit does not rewrite it
or turn its missing C320 record into a pass. The fixed-move extended histories
and associated analysis objects all match their retained mechanism manifest.
A recovered artifact will still need completion, schema, numerical equivalence,
configuration/source and new retention checks before P15 can close.

| Retained raw path | Check | Observed SHA-256 |
|---|---|---|
| `analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C160x20_trajectory.mat` | PASS | `81244cf570e1f5382f3840f54bc5dec3cbd51fd803ed6f518b73b498fc39c4e1` |
| `analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C320x40_trajectory.mat` | MISSING | `None` |
| `analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C400x50_trajectory.mat` | PASS | `673be8c7e2ba868b56e18777f8edbdbb528b4c4872d56cf827594af8e6a27bb8` |
| `analysis/OlhoffCurrent/evidence/move_activity_400/P400_400x50_trajectory.mat` | PASS | `0f4d2dacb2fc40d14399cba603f158a160b4b2b0849a52e679f068cbc939c517` |
| `analysis/OlhoffCurrent/evidence/move_activity_400/F400_400x50_trajectory.mat` | PASS | `f03cec07b45f71633d5fe125dcec84df5a64d5d55c7e5b90305e5c5144fe165c` |
| `analysis/OlhoffCurrent/diagnostics/move_stop/runs/baseline_160x20.mat` | PASS | `aad98bb1842ce43f47f1fe88b6cf943d421f10d9e98219af41bfcacecc226c6c` |
| `analysis/OlhoffCurrent/diagnostics/move_stop/runs/baseline_320x40.mat` | PASS | `21900ae78741cfb8cf65d041fd64641a0ecae3babe94eeeb0ad917e4352f3fe9` |
| `analysis/OlhoffCurrent/diagnostics/move_stop/runs/fixedmove_160x20.mat` | PASS | `55ea2de29b730ab14350f018bdb22795c494e9bfc22a91ee65c214011e7b3ef9` |
| `analysis/OlhoffCurrent/diagnostics/move_stop/runs/fixedmove_320x40.mat` | PASS | `7175740945e81f6fe2e39fe1f4e5972d95b056f9d8cef7038c25b087ce7b76dd` |
| `analysis/OlhoffCurrent/diagnostics/fixedmove_400_dynamics/evidence/fm_analysis.mat` | PASS | `b0bdd1a20efd3a8aa826c05b5b3ea19abd756b672cb1cc655d0fab4490da77f5` |
| `analysis/OlhoffCurrent/diagnostics/dynamical_regime/runs/runB_320x40.mat` | PASS | `196ce8c317648dd4121d81b2237e6bb7cce1813dcc8fe5812ca1c5e405737832` |
| `analysis/OlhoffCurrent/diagnostics/fixedmove_400_dynamics/runs/runC_400x50.mat` | PASS | `31bc06c5a0bd374e9105379ed930d20d4137cf958a8815482bc7eb0ee64ec7d0` |
| `analysis/OlhoffCurrent/diagnostics/two_branch_maturity_240/evidence/tb_analysis.mat` | PASS | `a06a7faab3872278e17e4b6d6187e7877e954fa89d3db8485607a2f56f5409d0` |
| `analysis/OlhoffCurrent/diagnostics/two_branch_maturity_240/runs/runD_240x30.mat` | PASS | `8cd55994e10c97037198172fe1271a9d7ae91e4cb32da553de3dd7fe9aee7e80` |

## Tests and preservation

`scripts/inventory.py` performs read-only input checks and writes only this new
audit directory. `evidence/inventory_tests.json` records the checks separately
from the incomplete evidence gate. The 17/5 prior test results are retained,
hash-valid evidence, not newly executed tests. The old suite includes a 160x20
scientific solve, so rerunning it wholesale would conflict with this task's scope.
Required new offline rule/floor tests remain pending.

The starting dirty entries are unchanged. No tracked production file was edited.
No previous manifest or study result was rewritten. No commit was created.
