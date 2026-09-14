# CAMPAIGN_AUTHORIZATION

```
OLHOFF_NINE_MESH_CAMPAIGN_AUTHORIZED
```

> The migrated OlhoffCurrent implementation has been reviewed, merged and independently revalidated at 160×20 for both the historical and the new production formulations. The exact Pedersen/adaptive production identity is frozen, all nine future configurations are prevalidated, and the definitive nine-mesh campaign is authorized.

**The campaign itself was NOT run.** No mesh above 160×20 was solved. The definitive nine-mesh campaign is a separate task.

## History of this decision

| attempt | date | verdict | reason | record |
|---|---|---|---|---|
| 1 | 2026-09-13 | BLOCKED | the A2 provenance rule was not fail-closed (P5, P9, P10) | `attempt1_blocked/` |
| 2 | 2026-09-13 | BLOCKED | after the hardening repair and merge, one new failure: `gray_kkt_forensic_audit` pinned the superseded manifest as required evidence | `attempt2_blocked_gray_kkt/` |
| 2 (resolved) | 2026-09-14 | **AUTHORIZED** | owner decision option 2 (the study's evidence record marks the pin historical); re-run confirms no failure remains that was not demonstrated before the merge | RESOLUTION_GRAY_KKT.md |

## Authorization conditions (all pass)

| condition | status |
|---|---|
| provenance repair accepted | PASS |
| P5 / P9 / P10 negative controls | PASS (all FAIL as required, from committed code) |
| full adversarial gate suite | PASS (39/39) |
| committed provenance self-contained | PASS |
| amendment review | PASS |
| migration review | PASS |
| merge | PASS (fast-forward to `b21483b`) |
| post-merge identity | PASS |
| post-merge tests | PASS (only verified pre-existing failures remain; RESOLUTION_GRAY_KKT.md) |
| historical S160 | PASS (bitwise) |
| Pedersen S160 | PASS (bitwise at established level) |
| nine-config preview | PASS |
| telemetry readiness | PASS |
| repository integrity | PASS |
| no scientific parameter retuned | PASS |

## Identity the campaign must use

Full detail is in `CAMPAIGN_IDENTITY.json`.

| | |
|---|---|
| branch / HEAD | `benchmark-methodology-r2` @ `b21483b158f58e05e7b56957f2fbe8e1d2891395` |
| upstream implementation | `253069262407885a8b759a9e721c4f0a7d3a397d`; `+impl` tree `4ba9a3ae10881344a0e60f2b8a8976c5ec9ceccf966bc2a3da4f37fb5aebffbf`; `SOURCE_MANIFEST.json` `aee44aa9172a8105314bbb9475e28bed8a53d4d154824a8748abf64a208b1836` |
| preset | `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` (selected by the harness's `confbench_olhoff_preset`) |

Config hashes, one per mesh:

| mesh | config hash |
|---|---|
| 160×20 | `b1a5744df798ad624fcd0b8b4306888eb99816ee5ce60f51040e39e03e90d4f4` |
| 240×30 | `2fac1384239527b1a0108e1d34725021cdc7c5a26f1939b602239415b34847e1` |
| 320×40 | `a1203d43efb240308cfcb520c92a6d9bc37a0c703c728fc2f3d298f779bc5f6e` |
| 400×50 | `e40c6ba16c9addbb587d08ecab29669f7db560d09b980ca62c2096856b849280` |
| 480×60 | `8e55d18251152ec2d26836593489c14f4842390d912d7c66f43f3b18730c83de` |
| 560×70 | `8b2533a3caf8432d40a54dbcbd8958900e333c410527d5547c53699b9e373ccf` |
| 640×80 | `cbe8dcf070d3e318a889867b5168fdbb09f1b6babfd46f981447d96e9a073aaf` |
| 720×90 | `2893ad47136fc9a7775f1f4d9ff1dcba38e7fe14271dd12b5727ff4bd8bb54f3` |
| 800×100 | `f9138743067afd0edce255ad26a6d7ae5c0f632a537516a4793eafecbacbe4cd` |

The campaign should refuse to start if HEAD, the `+impl` tree or any per-mesh config hash differs from these values.

## Carried caveats (unchanged by this gate)

- **Distinct formulation.** Pedersen stiffness + linear mass with an adaptive box is a distinct reconstruction, not "Olhoff 2007" and not a bug fix of the eq. (4b) formulation.
- **Heuristic stop.** The natural design-change stop is not a KKT certificate.
- **Phase 6 still deferred.** That covers the radius and low-density policy for other methods and the evaluator ω₁ policy.
- **Separate defect.** The p-continuation logging defect D1 remains separate.
- **Unchanged study verdict.** `gray_kkt_forensic_audit` records `PERFORMANCE_CAMPAIGN_BLOCKED_BY_OPTIMALITY_ISSUE` for the historical tree `edbfe47e`. That scientific verdict about the historical formulation is unchanged, and was not re-evaluated by this gate.
