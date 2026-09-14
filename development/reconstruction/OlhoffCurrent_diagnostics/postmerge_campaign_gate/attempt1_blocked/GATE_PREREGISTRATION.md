# GATE_PREREGISTRATION — final merge and promotion gate

**Disclosure:** this file was written **after** Step 1 ran. It is a post-hoc record, not a new set of rules. The owner fixed the pass criteria, stop rules and verdict vocabulary in the task brief before anything ran. Nothing below changes, relaxes or reinterprets them.

## Fixed identities

| | |
|---|---|
| migration branch | `migration/olhoffcurrent-upstream-253069` |
| migration commit | `9b30ec45b038fb36e7cf20d57679b71cfd099fb3` |
| upstream implementation commit | `253069262407885a8b759a9e721c4f0a7d3a397d` |
| parent successful-science commit | `6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7` |
| historical anchor | `duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered`, 160×20: 91 / 2241 / ω₁ 169.495227021538 (bitwise) |
| production anchor | `duOlhoffPedersenAdaptiveBoxSensitivityFiltered`, 160×20: 121 / 2369 / ω₁ 169.210576386275 (bitwise) |

## Stop rules, as applied

1. **Amendment review FAIL** → `OLHOFF_NINE_MESH_CAMPAIGN_BLOCKED`, stop before merge.
2. **Gate rule** broader than necessary or not fail-closed → BLOCK.
3. **Commit review FAIL** → stop before merge.
4. **Merge conflict** without an already-reviewed resolution → stop.
5. **Any new test regression**, or either 160×20 anchor failing → BLOCK.
6. **No mesh above 160×20** is solved under any outcome. No preset, tolerance, radius, material law, controller, Proposed, Yuksel, evaluator or Phase-6 change is made.
7. **Steps not reached** get the `_FAIL` verdict of their pair, marked **NOT EXECUTED**. A step that did not run is never reported as passed.

## Authorization condition

`OLHOFF_NINE_MESH_CAMPAIGN_AUTHORIZED` only if every condition in the brief holds; otherwise `OLHOFF_NINE_MESH_CAMPAIGN_BLOCKED` with the exact blocker.
