# PROVENANCE — three_rung_promotion_validation_retry1

Part A of the retry: the **dependency-specific scientific provenance gate**.
This gate asks one narrow question —

> may the single authorized C320 three-rung scientific run proceed?

— and it is deliberately **narrower than full historical-study finalization**.
The distinction between *scientific validation dependencies* and *global /
historical evidence-package finalization* is the axis this whole document turns
on. Section 6 states it explicitly; Part H
(`PROMOTION_PROVENANCE.md`) applies the stricter gate.

## A0 — Inventory

| Field | Value |
|---|---|
| branch | `benchmark-methodology-r2` |
| HEAD at task start | `60f5b72519aeba942b650d5408339f7ffe6b978b` |
| MATLAB | `25.2.0.3042426 (R2025b) Update 1` |
| `maxNumCompThreads` (default) | 10 — the run forces 1 via `runtime.singleThread` |
| platform | `MACA64` (Darwin 25.6.0, arm64) |
| `+impl` tree hash | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` |
| `+impl` source files | 75, manifest **verified ok** |
| currentness | **`CURRENT`** |

### Pre-existing dirty paths (2 — both present before this task, neither touched by it)

```
 M analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C320x40_iterations.csv
 M analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C320x40_record.json
```

Both are `tOuter`-class drift; see section 5. Confirmed here against git HEAD:

```
runs/C320x40_iterations.csv   disk 6ac72fa880f2e9fb…   HEAD ff570d6e4d024f36…
runs/C320x40_record.json      disk 765604214a4074ab…   HEAD 3fc68c050a9d81b5…
```

### The prior stopped attempt — preserved unchanged

`diagnostics/three_rung_promotion_validation/` is **not modified by this retry**.
All seven of its files hash exactly to its own `FINAL_SHA256.txt`:

| File | SHA-256 | matches its FINAL_SHA256 |
|---|---|:--:|
| `C320_ORACLE.md` | `d09579b80c0d187b3800e71e22538a1c482e60d79505d44457ffc9d57e8b82d9` | ✅ |
| `CONTROLLER_RECOVERY.md` | `119082321bf5ba56ffc39a36c76f7608e18a9b2c645a033da70a76475261bdac` | ✅ |
| `PROVENANCE.md` | `cf76d496b822c88cc49be610587f07fdcd3a4a3560e0188c16e04a46191fe0a5` | ✅ |
| `REPORT.md` | `de2af57ad78e6967d93c5ec1c9942d2f2c7446a1a20c70b7edac790aec6bd7ef` | ✅ |
| `EVIDENCE.json` | `7cd57f6146f401c3bb4e8842775c6858648dd009bcdb9c7e5ba7516088af3fd5` | ✅ |
| `METRICS.json` | `71c06e79846b19bdf2bd3782b3fc4d0f0c4e5bd1fdd85b9bb0fc5bca72eb445d` | ✅ |
| `DATA_MANIFEST.json` | `97796babb16395dedd4ad43f06ac416384cffae3026cd5c8fcc8325048b8cf47` | ✅ |

Its terminal status — `THREE_RUNG_PROMOTION_PROVENANCE_FAIL`,
`PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED`,
`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`, **scientific runs = 0** — stands as
written and is not rewritten here.

Full machine-readable inventory, including the SHA-256 of every source and
document this retry depends on: `evidence/inventory.json`
(produced by `scripts/tr_inventory.m`).

## A1 — C320 oracle currentness → **`C320_ORACLE_DEPENDENCY_PASS`**

The stopped attempt's oracle study was **not repeated**. Only currentness and
hashes were re-checked, as directed.

1. **`C320_ORACLE.md` is unchanged** — `d09579b8…7e8b82d9`, identical to the
   digest its own `FINAL_SHA256.txt` recorded when it was frozen.
2. **The raw C320 trajectory still reproduces every frozen anchor.** Loaded
   `evidence/two_branch_controller_validation/C320x40_trajectory.mat`
   (`RHO` 12800×1600), recomputed with the study's `local_vecHash` convention
   (SHA-256 over `typecast(double(v(:)),'uint8')`, MATLAB column-major):

| Anchor | Expected (frozen in `C320_ORACLE.md`) | Recomputed | |
|---|---|---|:--:|
| final `rho` | `0348b288da711f2bdcd89263f7feaae33ccaedcb73a133a1a5e941421795bfb3` | identical | ✅ |
| `RHO[:,1:352]` | `b8c0f18d887c7f5a81ece452f0a4097a0cc1a67710a137d66e677fcfd78b5ba3` | identical | ✅ |
| `omega(1:2,1:352)` | `fd63608399edaede2c4f928a25d5478eacc3711f6d626ae770ad107e7f4c488d` | identical | ✅ |

3. **The trajectory's own `meta.implTree`** is
   `edbfe47eb32109a2…ffcf37ee152cb` — the implementation tree currently on
   disk. The oracle and the candidate will run against the *same* source.
4. **The frozen event structure reproduces from the raw container**, not only
   from the tracked CSV:
   `stageStarts = [1 275 314 353]`, `W=20 P=20 Wnp=10 tol=0.1`,
   `descents = [275 1 274 255; 314 2 313 294; 353 3 352 333]`,
   branches `A B B`.
5. **The S3 terminal state at iteration 352** recomputes to the frozen values
   (`omega1 = 166.42726927757769`, `omega2 = 203.58101247948915`,
   `volume = 0.49999874883108736`, `move = 0.01`, `stage = 3`, `multN = 2`,
   `nInner = 19`, `cumInner = 6498`).
6. **The oracle arm's configuration reconstructs bit-exactly.** `tr_config`
   replays `cv_config('C',320,40)`'s own recorded override list; the resulting
   four-rung config hashes to
   `2359a1112fcec9edd0971aae9a89508cd703ea3899770e7e85c850138c2550f4`, which is
   **exactly the `cfgHash` committed in `runs/C320x40_record.json`**. The
   candidate is therefore provably a one-field delta from the configuration that
   actually produced the oracle, not from a re-typed approximation of it.

## A2 — Frozen A/B recovery → **`A_B_DEFINITION_DEPENDENCY_PASS`**

Recovered from executable and frozen sources, never from prose. See
`CONTROLLER_RECOVERY.md` for the full rule, its indexing and its reset
semantics. Summary of the consistency check:

| Source | Role | Agreement |
|---|---|---|
| `+impl/architecture/+olh/+move/exhaustion.m` | the **online, executed** detector | authoritative |
| `diagnostics/two_branch_maturity_240/scripts/tb_branches.m` | the retrospective form | predicates, constants, windows, persistence, tolerance **identical** |
| `diagnostics/two_branch_maturity_240/PREREGISTRATION.md` | the frozen preregistration | SHA-256 `62748225253f85f6a2fbc1bad35489a2c201cd45ee64ff279601003354b73abd` — **exactly the digest `exhaustion.m` cites in its own PROVENANCE comment** |

```
W = 20   P = 20   Wnp = 10   tol = 0.05*sqrt(NE/3200)  ( = 0.1 at 320x40 )

A(k) = med20 cos(k) < 0  AND  med20 net(k) < 0.5  AND  amp(k) >= tol
B(k) = med20 cos(k) > 0  AND                            amp(k) <  tol
E(k) = A(k) OR B(k)
```

**No ambiguity.** `THREE_RUNG_CONTROLLER_DEFINITION_AMBIGUOUS` does not apply.
The one place where the two forms differ in *operand* — `amp` — is documented
rather than smoothed over: `CONTROLLER_RECOVERY.md` §3. It does not create
ambiguity for this retry, because the oracle and the candidate are both produced
by the **same online detector**, and that detector is what is being validated.

## A3 — Three-rung architecture dependency

Verified that the tracked, frozen architecture artifacts still state the
verdicts the retry relies on, and that **not one of them has changed**:
`git status` over `diagnostics/three_rung_architecture`,
`diagnostics/three_rung_resolution_240`, `diagnostics/two_branch_maturity_240`
and `+impl` is **empty — all clean at HEAD**.

| Verdict | Still asserted in |
|---|---|
| `THREE_RUNG_ARCHITECTURE_SUPPORTED` | `three_rung_architecture/PREREGISTRATION.md:331`, `evidence/PREREGISTRATION.frozen:331`, `three_rung_resolution_240/{EVIDENCE.json:13, METRICS.json:734, REPORT.md:309}` |
| `THREE_RUNG_POLICY_PREREGISTRATION_JUSTIFIED` | `three_rung_architecture/{PREREGISTRATION.md:348, REPORT.md:308}`, `three_rung_resolution_240/{EVIDENCE.json:14, METRICS.json:736, REPORT.md:23}` |
| `C240_THREE_RUNG_COUNTERFACTUAL_EXACT` | `three_rung_resolution_240/{EVIDENCE.json:15, METRICS.json:730, REPORT.md:130}` |
| `THRESHOLD_SPLITTING_CONCERN_RESOLVED` | `three_rung_resolution_240/{EVIDENCE.json:16, METRICS.json:732, REPORT.md:209}` |

Recorded explicitly, as directed:

```
C240 RAW                      : REMOTE_REQUIRED_EVIDENCE_NOT_LOCAL
C240 scientific architecture  : TRACKED/FROZEN_ARTIFACT_INTACT
```

This is **sufficient to permit the C320 scientific validation**. It is **not**
sufficient for final production promotion — that is Part H's question, and the
answer there is different.

## A4 — Static three-rung prefix proof → **`THREE_RUNG_PREFIX_STATIC_PROOF_PASS`**

See `SINGLE_FACTOR_AUDIT.md` §2 for the complete site-by-site audit of all 17
reads of `move.levels` / `moveLevels` in the promoted tree. The result: under
`stop.rule = 'stageExhaustion'` with p-continuation off and projection off,
**ladder length is computationally inert until the detector declares at
`stage == 3`.** It is a structural property of the code, not an empirical hope.

## 5 — `tOuter` classification (explicit, as directed)

`tOuter` is **nondeterministic runtime telemetry** — per-iteration wall time,
written by the benchmark instrumentation at `olhoffSolve.m:552` and, as its own
comment states, *"Nothing reads it back."* It cannot be reproduced bitwise
across runs, hosts or MATLAB builds, and no scientific quantity depends on it.

Therefore, for this retry:

> **`tOuter` SHALL NOT be part of scientific prefix-equivalence acceptance.**

Exact equivalence is required on scientific state, controller state,
optimization state and inner work — the other 54 columns of the telemetry CSV,
plus the raw `RHO` / `omega` anchors. Wall-clock timing is recorded and
reported, never asserted.

The prior attempt established, and this document does not re-derive, that the
two dirty tracked CSVs differ from HEAD **only** in `tOuter`: stripping that one
column makes both byte-identical
(`5f9f18960dfc4a990c15d1b101b256e006336a8d6ca41f560ec6d6b58f6bb53d`).

## 6 — The three provenance classes, kept apart

The repository's own `olhoffcurrent_evidence_gate` was run on every load-bearing
study. Its output separates cleanly into three classes, which must **not** be
collapsed into one generic FAIL. Full output: `evidence/gates.json`.

### Class A — locally regenerated containers (`CONTAINER_DIGEST_STALE / HOST-SPECIFIC`)

```
C160x20_trajectory.mat  recorded 4d11a2fd…  on disk 81244cf5…
C320x40_trajectory.mat  recorded c9d4d766…  on disk 4892c10e…
C400x50_trajectory.mat  recorded fa0e714c…  on disk 673be8c7…
```

These `.mat` **containers** differ. Their **scientific content does not**: the
prior attempt verified all three final-`rho` digests against the values
committed in git-tracked `runs/*_record.json`, and this retry independently
re-verified C320 down to the `RHO[:,1:352]` and `omega(1:2,1:352)` prefix
anchors (§A1). `evidence/` is git-ignored wholesale, so these files never
travelled with git and the copies here were produced locally.

**This is not `SCIENCE_INVALID`.** For C320 — the only one this retry consumes —
the scientific oracle is locally validated to the bit.

### Class B — C240 raw artifact not transferred to this host (`REMOTE_REQUIRED_EVIDENCE_NOT_LOCAL`)

```
analysis/OlhoffCurrent/evidence/three_rung_resolution_240/C240x30_trajectory.mat
  REQUIRED_MISSING   131 203 128 bytes
  sha256 183d7ce60d512fc2c045c3cb575404b00c8f0e223adaf97db86c9eef1fd50b0d
```

Produced on a different machine; `evidence/` is git-ignored, so it was never
transferred. **Absent locally ≠ lost.** It is not re-run and is not called
irrecoverable. The remedy is a file copy.

### Class C — genuinely stale, machine-independent hash bookkeeping

`two_branch_controller_validation/FINAL_SHA256.txt` (and, for `baselines.json`,
`three_rung_architecture/EVIDENCE.json` too) record digests that disagree with
the study's own **committed, clean** files:

| File | recorded | on disk **= git HEAD** |
|---|---|---|
| `PROVENANCE.md` | `42037f81…3ca90790` | `0e987a9b…35459a33` |
| `BASELINES.md` | `74629a16…d2299b21` | `2212a66a…baa4f0df` |
| `evidence/baselines.json` | `08b48346…10b4c571` | `a4b55671…8738f420e` |

All three are byte-identical to HEAD, so this reproduces on **any** machine. It
is a real bookkeeping defect, repairable by regenerating one hash file, and it
touches no scientific content. Status and proposed repair:
`PROVENANCE_REPAIR_STATUS.md`.

### Why these classes do not close the run-permission gate

A failing *historical evidence-package finalization* is a statement about
whether a whole study can still be re-verified end to end on this host. The
question this gate asks is narrower: **is the specific scientific dependency
this run consumes verified?** For the C320 oracle it is — cryptographically, in
§A1, against digests committed to git. Class A is container staleness with
proven-identical science; Class B concerns a study whose *scientific* verdict is
carried by tracked artifacts that are intact and clean; Class C is a hash file,
not a result.

Failure of the broader historical `FINAL_SHA256` package alone therefore does
**not** force this gate to fail. It does force **Part H** to fail, and it does.

## A5 — Gate verdict

| Requirement | Result |
|---|---|
| C320 oracle dependency | **PASS** (§A1) |
| A/B definition dependency | **PASS** (§A2) |
| architecture tracked/frozen dependency | **PASS** (§A3) |
| three-rung prefix static proof | **PASS** (§A4) |
| `+impl` integrity / currentness | **PASS** — tree `edbfe47e…`, 75 files, `CURRENT` |
| scientific source ambiguity | **none** |

# `DEPENDENCY_SPECIFIC_SCIENTIFIC_PROVENANCE_PASS`

The single authorized C320 three-rung scientific run may proceed.

**Final production promotion and nine-mesh campaign authorization remain
conditional** on closing the genuine remaining provenance defects (Classes B and
C, and the `tOuter` drift) before production freeze. Part H.
