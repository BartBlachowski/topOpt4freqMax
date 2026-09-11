# SOFTWARE_VALIDATION — Part C

Two suites. Neither is a scientific result and neither may be cited as one.

1. **The retry's own controller-mechanics suite**, `scripts/tr_tests.m` —
   29 tests, all passing.
2. **The repository's own test suite**, `analysis/OlhoffCurrent/tests/` —
   run unmodified, 6 files.

Machine-readable output: `evidence/software_tests.json`.

## 1. Controller mechanics — 29/29 PASS

The two mechanisms under test are driven **directly with synthetic detector
state**, so no test depends on an optimization outcome. Where an end-to-end
solve is needed, it runs at **48×6 (NE = 288)** — far below the 160×20
scientific floor — and its numbers are never interpreted. The scientific run
lock is respected: the candidate was solved exactly once, at 320×40.

### Descent and termination at each rung, both ladders (12 tests)

| ladder | stage | move | E declared | descend | CONVERGED |
|---|---|---|---|:--:|:--:|
| three-rung | 1 | 0.04 | yes | ✅ yes → 0.02 | no |
| three-rung | 2 | 0.02 | yes | ✅ yes → 0.01 | no |
| **three-rung** | **3** | **0.01** | **yes** | **no** | ✅ **yes** |
| four-rung | 1 | 0.04 | yes | yes → 0.02 | no |
| four-rung | 2 | 0.02 | yes | yes → 0.01 | no |
| four-rung | 3 | 0.01 | yes | yes → 0.005 | no |

and, at every stage of both ladders, **E not declared ⇒ the move is held and the
run does not stop.** This is the required distinction: the ladders agree
everywhere except at stage 3 with a declaration, where the three-rung policy
converges and the four-rung one descends.

### The removed rung (2 tests)

- three-rung ladder never yields `move = 0.005` — reachable moves are exactly
  `{0.04, 0.02, 0.01}`, driven through repeated declarations past the end of the
  ladder;
- `0.005` is absent from the level vector.

### beta has no authority (3 tests)

- a beta history engineered as a textbook stall **cannot descend** the ladder
  under `stageExhaustion` — stage stays 1, move stays 0.04;
- a beta stall **cannot terminate** — `admit(stage, declared = false) = false`;
- **control:** that *same* beta history **does** descend under the production
  `boundVariable` signal (stage 1 → 2). The suppression is real, not an inert
  history that would have done nothing anyway.

### Cap and failure behaviour (2 tests)

- a non-declared terminal stage does not stop — so a run that never declares
  runs to the cap;
- at 48×6 with `cap = 5`, the three-rung run is `CAP_HIT` at exactly 5
  iterations. `CAP_HIT` stays `CAP_HIT`.

### A / B / persistence / reset unchanged (4 tests)

- **the detector is ladder-blind**: `olh.move.exhaustion` takes no ladder
  argument, and two runs over identical synthetic input produce `isequaln`
  state;
- Branch B declares at exactly `P = 20` consecutive true, with
  `declIter = declBegin + 19`;
- frozen constants intact: `W = 20`, `P = 20`, `Wnp = 10`;
- **descent resets the detector window wholly to the new stage**:
  `stageStart` moves to the descent iteration, `cntA = cntB = 0`,
  `declared = false`, `declIter = NaN`.

### End-to-end mechanics at 48×6 (4 tests)

- three-rung never visits `move = 0.005`;
- three-rung final stage ≤ 3;
- **ladder length is inert over the common prefix** — `move` and `stage`
  identical between the two ladders across every commonly executed iteration;
- the cap is honoured and the status is one of `CONVERGED` / `CAP_HIT`.

### Configuration validation still refuses what it always refused (2 tests)

- a non-descending ladder raises `olh:config:ladderNotDescending`;
- `stop.rule = 'stageExhaustion'` still **requires** the matching
  `move.continuation.signal`, *"or a declaration below the last rung is never
  consumed."* This coupling was discovered while writing the suite and is now
  asserted rather than assumed.

# `THREE_RUNG_SOFTWARE_VALIDATION_PASS`

## 2. Repository test suite

| Test | failures |
|---|---|
| `test_currentness` | **0** |
| `test_source_integrity` | **0** — including all destructive cases; tree restored pristine, 75 files, `edbfe47e…` |
| `test_path_isolation` | **0** — every forbidden Olhoff tree still blocked |
| `test_preset_equivalence` | **0** — production preset vs frozen conference at 160×20, **bitwise** on `rho`, `omega1` and volume |
| `test_evidence_retention` | **0** |
| `test_finalization_gate` | **4** |

`test_preset_equivalence` executes a 160×20 solve. That is the **repository's
own frozen equivalence fixture** — production preset, four-rung ladder, not this
retry's candidate — and it is a pre-existing software test, not a scientific run
of the candidate. Recorded here so the run count is unambiguous:
**candidate scientific runs = 1** (`SCIENTIFIC_SAFETY.md`).

### The 4 failures, and why they are not this candidate's

`test_finalization_gate`'s own self-tests **A–G all pass** — the gate is
operational and fails closed, exactly as designed. The 4 failures are cases
**H** and **I**, which assert that real studies still pass the gate on this
host. They fail for the three provenance classes documented in
`PROVENANCE.md` §6 and `PROVENANCE_REPAIR_STATUS.md`: a container regenerated
locally, a file that lives on another machine, and a stale hash file. This is
the same regression the stopped attempt reported, unchanged by this retry, and
it is the reason Part H keeps promotion blocked.

One entry in case I is new and self-inflicted-by-timing: this study's own
directory appeared in the "new" list while it lacked an `EVIDENCE.json` and a
`FINAL_SHA256.txt`. Both are written at finalization, and the suite is re-run
afterwards; `FINAL_SHA256.txt` records the result.
