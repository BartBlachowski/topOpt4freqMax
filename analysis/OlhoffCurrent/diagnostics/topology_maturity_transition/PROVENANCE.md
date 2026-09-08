# PROVENANCE — topology-maturity move-transition study

Phase 0 integrity record, and the provenance failure that bounds what this task
could deliver.

---

## 1. State at task start

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **HEAD at task start** | `7154d8201e9defb06d0d758da866c3769c07179a` |
| HEAD subject / date | `move transitions` · 2026-09-07 18:03:24 +0200 |
| `git status --porcelain` | **empty (clean)** |
| Canonical implementation | `analysis/OlhoffCurrent` |
| MATLAB | **25.2.0.3042426 (R2025b) Update 1** |
| `maxNumCompThreads` (ambient) | 10 — *no solver was run in this task, so no thread pinning was required* |
| Production preset | `duOlhoffFixedPenaltySensitivityFiltered` → upstream `duOlhoffFrozenM4` |

## 2. Required Phase 0 gates — all PASS

| Gate | Result | Evidence |
|---|---|---|
| currentness | **CURRENT** | promoted source intact, matches upstream `architecture/canonical-config` @ `695f03bdac20c423a4e1d389cf9db9187597bcc3`, 0 commits ahead |
| source integrity | **PASS** | 74/74 source files; `+impl/` tree `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c`; 2 artifacts ignored (`.DS_Store`) |
| canonical manifest | **PASS** | `SOURCE_MANIFEST.json` matches recomputed tree hash exactly |
| dispatch / symbol resolution | **PASS** | `olhoffcurrent_assert_dispatch()` → `ok=1`, 32 owned symbols resolved inside `+impl/`, **0 blockers, 0 warnings** |
| sensitivity filter wins | **PASS** | included in the 32 resolved symbols; `test_source_integrity` cases I/J prove a competing filter/MMA is *refused*, not silently preferred |
| published MMA wins | **PASS** | as above — `mmasub` resolving to `tools/Matlab/mmasub.m` is a hard BLOCK |
| no forbidden Olhoff tree on path | **PASS** | 13 forbidden paths declared, none present on the live path |
| existing test suite | **PASS — 4/4, 0 failures** | `test_currentness` 8/8 · `test_path_isolation` 6/6 · `test_preset_equivalence` 6/6 (bitwise, ω₁ = 169.49522702153845, outer 91, inner 2241) · `test_source_integrity` 10/10 |
| retained evidence hash-valid | **PASS — 81/81 files** | `move_stop` 24/24 · `admission_rule` 21/21 · `move_transition` 36/36; 0 mismatches, 0 missing |

The `+impl/` tree hash is **identical** to the one recorded by the three
preceding studies, so this task and they share one implementation.

**Phase 0 verdict: not `BLOCKED_BY_PROVENANCE`.** Every gate the brief required
passed.

---

## 3. Provenance failure found *inside the brief's own premises*

The brief instructs that its "ESTABLISHED SCIENTIFIC FACTS" be treated as
established **"unless integrity checks fail."** The integrity checks located a
failure of exactly that kind, so it is recorded here rather than acted upon.

### 3.1 Three of the six cited prior studies do not exist

The brief names six completed studies. Three are present, hash-valid and were
used. Three **do not exist anywhere** — not at HEAD, not on any of the seven
branches, not in `git stash`, not among ignored/untracked files, not under
`$HOME`, and not in any commit reachable from any ref:

| cited study | status |
|---|---|
| `diagnostics/move_stop/` | **present**, 24/24 hash-valid |
| `diagnostics/admission_rule/` | **present**, 21/21 hash-valid |
| `diagnostics/move_transition/` | **present**, 36/36 hash-valid |
| `diagnostics/move_activity_offline/` | **ABSENT** |
| `diagnostics/move_activity_400/` | **ABSENT** |
| `diagnostics/beta_transition_mechanism/` | **ABSENT** |

`git log --all -- '*move_activity*' '*beta_transition*'` returns nothing: these
paths have never existed in this repository's history.

### 3.2 The data-availability premise is inverted

Brief §A5 states: *"400x50 has retained raw element trajectories. 160x20 and
320x40 raw histories were lost."*

The filesystem states the opposite, verified by `whos -file`:

| mesh | raw element trajectories on disk |
|---|---|
| **160×20** | **RETAINED** — `RHO` 3200 × 600 (ARM P, ARM U, unstopped), 3200 × 400 (fixed-move), 3200 × 91 (baseline) |
| **320×40** | **RETAINED** — `RHO` 12800 × 600 (ARM P, ARM U, unstopped), 12800 × 216 (fixed-move), 12800 × 131 (baseline) |
| **400×50** | **NONE.** No trajectory, no diagnostic run, no OlhoffCurrent production run at any move policy. |

The 160×20 and 320×40 `.mat` trajectories are additionally *manifested and
hash-valid* under `move_stop/FINAL_SHA256.txt`. They were never lost.

The only 400×50 Olhoff history in the repository is
`analysis/performance_campaign_forensic_audit/olhoff_histories/400x50.csv`,
which is **not** usable as a production reference: it is scalar-only (no `M_nd`,
no densities) and it was produced under a **constant `move = 0.005`**, not the
production ladder `[0.04, 0.02, 0.01, 0.005]`.

### 3.3 Facts 8, 9, 10 and 13 are unsourced, and 8 is contradicted

Facts 8–13 describe 400×50 quantities. With no 400×50 trajectory in existence,
none can be sourced. Two are also *positively* contradicted by the retained data:

| brief's claim | measured from retained, hash-valid data |
|---|---|
| Fact 8: remaining evolution at first descent — 160×20 **9.0 %**, 320×40 **43.4 %** | **160×20 2.20 %**, **320×40 11.61 %** (M_nd basis); 15.23 % / 16.23 % (L1-endpoint basis) |
| Fact 9: 400×50 first-descent M_nd ≈ **32.33 %** → mature **16.16 %** | no 400×50 run exists. (Note 32.33/16.16 = 2.0000 exactly.) |
| Fact 10: 400×50 bitwise-identical prefix of **2,740,000** density values | no 400×50 run exists. 400·50·137 = 2,740,000 — arithmetically self-consistent, evidentially empty. |
| Fact 13: Jaccard **0.960** / **0.386** | no such measurement exists in any retained artifact |

The brief's *ordering* in Fact 8 (160×20 more mature than 320×40) is correct and
is confirmed here. Its *numbers* are not reproducible. For what it is worth,
`43.4`, `50.1`, `0.960` and `0.386` all occur as literal values in
`move_transition/runs/armP_320x40_iterations.csv` — as `Mnd_pct` at iterations 58
and 36, and as `r_rho` (= max|Δρ|/move) at iterations 75 and 106 — i.e. as
quantities of a different kind, at a different mesh, than the facts assert.

### 3.4 Consequence, and what was *not* done about it

Brief §A6 authorises controlled regeneration of **160×20 and 320×40 only** — the
two meshes whose data was never missing — and gives no authorisation covering
400×50, the one mesh that has none. Phase C nevertheless makes 400×50 the
*decisive* validation mesh (§C6).

Nothing was fabricated, reconstructed, or assumed to close this gap. Facts 8–13
are treated as **unestablished**, and the ground-truth maturity labels used in
Phase A were **re-derived from the retained trajectories** (§OFFLINE_SIGNAL_ANALYSIS).
No regeneration was needed: the 160×20 and 320×40 data required by Phase A was
already present and hash-valid.

---

## 4. What this task executed

* **Read-only** analysis of retained trajectories. **Zero solver runs.**
* **Zero** changes to `+impl/`, to any preset, or to any configuration field.
* The `+impl/` tree hash was re-verified after the analysis: **unchanged**,
  `c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c`.
* No file under any prior diagnostic directory was written, moved or deleted.
