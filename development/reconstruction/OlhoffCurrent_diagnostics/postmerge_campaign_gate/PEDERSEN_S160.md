# PEDERSEN_S160 — Step 5B

```
POSTMERGE_PEDERSEN_S160_PASS
```

**Run.** Exactly one 160×20 solve of `duOlhoffPedersenAdaptiveBoxSensitivityFiltered`, in the merged normal checkout:
- HEAD `b21483b`, `+impl` clean;
- configuration `olhoffcurrent_config(160, 20, 'Preset', name)`;
- single thread;
- wall time 229 s (not a science field).

**Evidence.** `analysis/OlhoffCurrent/evidence/postmerge_campaign_gate/ANCHOR_PEDERSEN_S160.mat` (git-ignored) and `evidence/ANCHOR_COMPARISON.json → pedersen`.

**Standard.** The same `mig_compare` standard as Step 5A.

## Result

| quantity | value |
|---|---|
| status / stop rule | CONVERGED / `designChange` (natural ε = 0.05·√(NE/3200) stop) |
| outer / inner | **121 / 2369** |
| ω₁ | **169.210576386275** |
| ω₂ | 170.399975755934 (gap 0.703 %) |
| volume | 0.499998983945 |
| M_nd | 11.4616 % |
| gray fraction | 0.141875 |
| ρ SHA-256 (raw bytes) | `2a1c0d0afa18e763…` |
| adaptive box | `hist.move` max 0.1 throughout; `aux.moveMean` 0.1 → 0.00844 |
| resolved config hash | `b1a5744df798ad624fcd0b8b4306888eb99816ee5ce60f51040e39e03e90d4f4` (= PROVENANCE.json event 2 `config_hashes["160x20"]`) |

## Comparisons

| reference | verdict | detail |
|---|---|---|
| **committed upstream S160x20** (`repro/results/S160x20/res.mat`, SHA-256 `de7b1576…`, identical to the blob at `Olhoff@6b0870850d74`) | **pass (established level)** | 0 differing result leaves, 0 only-in-reference. The config differs only in `stop.rule` (absent at 6b08708), `runtime.verbose` and `runtime.name`, exactly as established. `tOuter` excluded. |
| committed S160x20: individual histories | **equal** | `hist.move`, `aux.moveMean` (box trajectory), `aux.Mnd`, `hist.N` (multiplicity), `hist.nInner`, and the log are equal; status is equal |
| migration post-run `POST_PED.mat` | **bitwise, strict** | 0 / 0 / 0 / 0; cfg struct identical |
| upstream-snapshot run `UP_PED.mat` | **bitwise, strict** | 0 / 0 |
