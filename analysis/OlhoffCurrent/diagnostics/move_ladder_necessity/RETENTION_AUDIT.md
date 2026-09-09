# RETENTION AUDIT — why raw trajectories keep disappearing

A software/process audit. **No trajectory is reconstructed.**

---

## 1. The loss, measured

Every study's own `FINAL_SHA256.txt` was re-verified against the filesystem:

| study | hashed entries OK | mismatched | **MISSING** | declares evidence? |
|---|---|---|---|---|
| admission_rule | 21 | 0 | 0 | **NO** |
| beta_transition_mechanism | 26 | 0 | 0 | yes |
| dynamical_regime | 30 | 0 | **3** | **NO** |
| fixedmove_400_dynamics | 26 | 0 | **2** | **NO** |
| move_activity_400 | 37 | 0 | 0 | yes |
| move_activity_offline | 20 | 0 | 0 | **NO** |
| move_stop | 20 | 0 | **4** | **NO** |
| move_transition | 36 | 0 | 0 | **NO** |
| topology_maturity_transition | 14 | 0 | **1** | **NO** |
| two_branch_controller_validation | 71 | 0 | 0 | yes |
| two_branch_maturity_240 | 26 | 0 | **2** | **NO** |
| **total** | **327** | **0** | **12** | 3 of 11 |

**12 hash-manifested artifacts are gone, across 5 studies.** Nothing is
corrupted — zero mismatches. Files were lost, not damaged.

```
dynamical_regime              evidence/dr_analysis.mat, runs/runA_400x50.mat, runs/runB_320x40.mat
fixedmove_400_dynamics        evidence/fm_analysis.mat, runs/runC_400x50.mat
move_stop                     runs/{baseline,fixedmove}_{160x20,320x40}.mat
topology_maturity_transition  evidence/phaseA_stats.mat
two_branch_maturity_240       evidence/tb_analysis.mat, runs/runD_240x30.mat
```

**Correction to the previous report.** `two_branch_controller_validation/REPORT.md`
answer 6 reported *five* missing artifacts. That was the set that study happened
to probe. The true figure across the tree is **12 hash-manifested** files, plus a
further ~7 distinct trajectories referenced in prose or code that were never
hashed at all (`armP/armU_*`, `unstopped_*`, `move_transition/analysis.mat`).
The earlier number understated the loss and is corrected here.

## 2. Why — five findings

### R-F1 · The gate is opt-in, and opting out is invisible

`olhoffcurrent_evidence_gate` answers *"is the declared evidence still there?"*.
It can only ask that of a study that wrote an `EVIDENCE.json`. A study that never
writes one is not *failed* by the gate — it is **invisible** to it. All five
studies with losses have no `EVIDENCE.json`. That is the hole.

### R-F2 · A clean `FINAL_SHA256.txt` can mean "we never hashed the data"

`admission_rule` and `move_transition` show 0 missing — not because their raw
data survives, but because they never hashed it. `move_transition/code/mt_report.m`
loads `analysis.mat`; no such file exists and no manifest names it. A green
checkmark that is green because nothing was checked is worse than a red one.

### R-F3 · Adoption is a clean function of *when*, not of care

Commit order, oldest first:

```
2b89b07  move_stop                                   ) 
a1f2c6c  admission_rule                              )  no EVIDENCE.json
7154d82  move_transition                             )  5 of these lost data
9ad79be  topology_maturity_transition, dynamical_regime,
         fixedmove_400_dynamics, two_branch_maturity_240
b06050c  move_activity_offline                       )
--------------------------------------------------------------------------
5fbec9a  EVIDENCE_POLICY.md + gate + declare  <-- the machinery lands here
--------------------------------------------------------------------------
357d2c5  move_activity_400            EVIDENCE.json, 0 missing
b6014ba  beta_transition_mechanism    EVIDENCE.json, 0 missing
1438aa3  two_branch_controller_validation  EVIDENCE.json, 0 missing
```

**Every study finalized before the policy lost data or never hashed it. Every
study finalized after it has an `EVIDENCE.json` and has lost nothing.** The
policy works. It was simply never made mandatory, so it protects only the
studies that chose it.

### R-F4 · Hash files were not self-verifying

`FINAL_SHA256.txt` records digests of other files but nothing re-checks it. This
audit found the previous study's own hash file already stale: its
`DATA_MANIFEST.json` digest was computed before a later figure refresh updated
that manifest. Corrected in place, and the correction is recorded in that file
rather than made silently. A manifest nobody re-runs is a snapshot, not a check.

### R-F5 · Finalization is not atomic with committing

The previous study's `C400x50` outputs, its final documents and its
`EVIDENCE.json` are **still uncommitted** at the time of this audit. Its raw
trajectories live in the durable evidence root — untracked by design, which is
correct — but the *declaration* that makes them verifiable is untracked too. On a
fresh clone the study would currently present as incomplete. This is precisely
the shape of the original defect, one step earlier in the pipeline.

## 3. What was NOT the cause

* **Not gitignoring.** `.mat` is ignored deliberately and correctly — one archive
  already breached GitHub's 100 MB limit. Untracked is a storage decision. The
  policy's own words apply: *"untracked is a storage decision; undeclared was the
  defect."*
* **Not corruption.** 327/327 surviving digests match.
* **Not a cleanup script.** No such script exists in the tree. The files sat in
  git-ignored `runs/` and study-local `evidence/` directories, outside the
  durable evidence root, with nothing asserting their existence; any clone,
  checkout, prune or manual tidy would remove them without a signal.

## 4. The enforcement added

**`analysis/OlhoffCurrent/olhoffcurrent_finalization_gate.m`** — a fail-closed
finalization check. A study may not be called finalized unless:

| gate | requirement | closes |
|---|---|---|
| **G1** | `EVIDENCE.json` exists — *declaring nothing is not a passing state* | R-F1 |
| **G2** | every `required` declared artifact present and hash-valid | the original defect |
| **G3** | `FINAL_SHA256.txt` exists | R-F2 |
| **G4** | every digested line in it resolves and matches — the hash file is **self-verifying** | R-F4 |
| **G5** | no `.mat` the study *claims* to hold is absent | R-F2 |

G5 counts only claims: lines carrying a digest, and manifest entries carrying a
`sha256`. A study that *documents* an absent file in prose (as the previous study
does, in a "MISSING PRIOR EVIDENCE" section) is not penalised — penalising honest
disclosure would be exactly backwards.

**Scientifically inert.** It lives outside `+impl/`, so it cannot change the
canonical tree hash; it reads and hashes files and touches no optimizer,
configuration or trajectory. The `+impl` tree hash is unchanged by this task:
`edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`.

**`analysis/OlhoffCurrent/tests/test_finalization_gate.m`** — 12 checks:

```
A   compliant study                        -> PASS
B   no EVIDENCE.json (the lost-study state) -> FAIL
C   required artifact removed               -> FAIL
D   required artifact altered               -> FAIL
D2  restored                                -> PASS
E   no FINAL_SHA256.txt                     -> FAIL
F   FINAL_SHA256.txt gone stale             -> FAIL
G   names an absent .mat                    -> FAIL
H   the three real compliant studies        -> PASS  (x3)
I   no NEW study fails the gate             -> ledger check
```

Test I holds a ledger of the eight legacy studies that predate the policy. It
fails if that set **grows** — catching regressions without rewriting history.
Two bugs in the gate were found and fixed by these tests before it was accepted:
`exist()` resolving bare filenames via the MATLAB path (so `PROVENANCE.md` could
match the implementation-level file instead of the study's own), and G5 reading a
documented-absent line as a possession claim.

### The gate caught R-F4 in practice, during this task

Not a hypothetical. While finalizing *this* study, a figure was regenerated, its
`DATA_MANIFEST.json` digests were refreshed, and `FINAL_SHA256.txt` was left
holding the manifest's previous digest — the identical staleness found in the
previous study. The gate returned:

```
G1 declares=1  G2 required=1  G3 hashfile=1  G4 selfverify=0  G5 nomissingmat=1
```

The fix was to make the refresh order a fixed point (regenerate the manifest,
then recompute every hash-file line last, which is stable because
`DATA_MANIFEST.json` lists neither itself nor `FINAL_SHA256.txt`). G4 now passes.
An opt-in check nobody re-runs would have shipped that defect a second time.

## 5. What this does not fix

The eight legacy studies still fail G1. **Their history is not rewritten and
their lost data is not regenerated** — regeneration would cost hours of solver
time and, more importantly, would be new scientific runs, which this task
forbids. They are recorded in the ledger as permanently deficient. Anything that
depended on their raw element-level data stays unanswerable, and this audit says
so wherever it matters (see `EVIDENCE_INVENTORY.md`).
