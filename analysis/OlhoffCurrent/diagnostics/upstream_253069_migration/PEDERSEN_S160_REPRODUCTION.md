# PEDERSEN_S160_REPRODUCTION — Parts 11 and 20B

```
PEDERSEN_PRESET_S160_REPRODUCTION_PASS
```

**Control.** The committed upstream result `repro/results/S160x20/res.mat` (git blob `93c6550980f0…`, identical in `6b08708` and `253069`, file SHA-256 `de7b15764509…`), read from the verified snapshot.

**Candidate.** `olhoffcurrent_config(160, 20, 'Preset', 'duOlhoffPedersenAdaptiveBoxSensitivityFiltered')` → `olhoffSolve`, exactly as production resolves it (cap 400, single thread, diagnostics off, verbose off).

| quantity | migrated OlhoffCurrent | committed S160x20 | upstream 253069 re-run |
|---|---|---|---|
| resolved scientific settings | 87 leaves | equal except `stop.rule` (absent at 6b08708), `runtime.verbose` (true), `runtime.name` | all 87 equal |
| status / termination | CONVERGED at outer 121: "‖drho‖₂ = 4.745e-02, max\|drho\| = 1.358e-02" | identical log (2 lines) | identical |
| cumulative inner / inner max / not converged | 2369 / 38 / 0 | 2369 | = |
| ω₁ / ω₂ / ω₃ | 169.210576386275 / 170.399975755933 / 321.397… | bitwise | bitwise |
| gap₁₂ | 0.7029 % | 0.70291 % | = |
| volume | 0.4999989839 | bitwise | = |
| ρ SHA-256 | `2a1c0d0afa18e763…` | ρ bitwise | = |
| M_nd (4·mean ρ(1−ρ)) / grey (0.1 < ρ < 0.9) / mid | 0.114616 / 0.141875 / 0.016875 | 0.11461565500445316 / 0.141875 (summary.json) | = |
| box trajectory `hist.move` (max box) | 0.1 throughout | bitwise | = |
| box trajectory `aux.moveMean` | 0.1 → 0.0942 (it. 3) → … → 0.008444 (it. 121) | bitwise | = |
| `aux.Mnd` history (121 values) | — | bitwise | = |
| every non-timing `hist` field (24) and `res.aux` | — | bitwise | bitwise |

**Comparison result.** `B_ped.post_vs_committed` found:

- no differing value;
- no field present only in the committed record;
- configuration differences limited to the three listed rows.

The migrated result adds only `hist.tOuter` (timing, excluded). Against UP the comparison is strict: identical field sets, all bitwise.

**Independent test.** `tests/test_named_preset_reproduction('pedersen')` checks against the tracked digest fixture `tests/fixtures/S160x20_reference.json`, generated from the committed `res.mat`: **PASS**, 10/10.

**Timing** (not a criterion). Wall 279 s with nine concurrent 160×20 solves on 10 cores; the committed sweep recorded 405.6 s with nine parallel runs. Neither number is a benchmark.
