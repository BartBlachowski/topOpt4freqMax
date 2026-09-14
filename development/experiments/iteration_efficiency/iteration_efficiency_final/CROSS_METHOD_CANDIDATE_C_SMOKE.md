# Cross-method Candidate-C smoke

Real lossless trajectories were evaluated for the three principal methods at
16x2 for three post-updates:

| method | evaluator | status | selected ordinal range | adaptive failures |
|---|---|---:|---:|---:|
| Proposed | Candidate C, E1/E2/E3 | PASS | 1 | 0 |
| Yuksel | Candidate C, E1/E2/E3 | PASS | 1 | 0 |
| Olhoff-LP | Candidate C, E1/E2/E3 | PASS | 1 | 0 |

Olhoff-MMA also passed when selected. The integration suite additionally replayed
the established 480x60/k=194 difficult state: E3 selected ordinal 13 after the
schedule `3→6→12→24` (three escalations). Thus the maximum ordinal exercised by
this integration was **13**, with zero unresolved structural-mode failures.

Selected ordinals are allowed to differ by method/state; only unresolved
classification is a failure.
