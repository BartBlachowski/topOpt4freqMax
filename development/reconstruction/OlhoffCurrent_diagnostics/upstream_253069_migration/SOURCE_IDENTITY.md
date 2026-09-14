# SOURCE_IDENTITY — Part 1

```
UPSTREAM_253069_IDENTITY_PASS
```

Machine-readable: `evidence/source_identity.json`.

| item | verified value |
|---|---|
| repository | `/Users/piotrek/Programming/Matlab/Olhoff` |
| object type of `253069262407885a8b759a9e721c4f0a7d3a397d` | `commit` (`git cat-file -t`) |
| tree | `4571029f39b62181499ab3d4ffb6da89f1d30021` |
| parent | `6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7` — the only parent; equals the stated successful-science commit |
| parent tree / subject | `809c671e2d1b15232f8b75ca0f07237c0e8e37ac` / "Nine resolution test - ultimate results" |
| subject / date | "Promote optional stage-exhaustion controller and outer timing" / 2026-09-13T19:00:09+02:00 |
| branch containing it | `migration/upstream-olhoffcurrent-capabilities` (its head; checked out in a separate upstream worktree) |
| `git fsck --connectivity-only` | exit 0 |
| files changed vs parent | 50 (implementation 8, docs 3, tests 4 + fixture, audit tree 34); list in `evidence/source_identity.json` |

## Immutable snapshot

- `git archive 253069…` → tar SHA-256 `f911240340cab69b9806f98f9413d9038649fde5e14adf3bbe4d57f3ddb0dbeb` (458 311 680 bytes, git 2.50.1).
- Extracted into the session scratchpad and made read-only (`chmod -R a-w`).
- **All 1877 files blob-verified**: the git blob SHA-1 of every extracted file equals the commit's `ls-tree` entry. 0 mismatches.
- Every promoted byte and every upstream-side run in this study comes from that snapshot. Nothing was copied from, and no run was made in, the upstream working tree.

## Upstream working-tree state (recorded, excluded)

Checked out: `repro/natural-convergence @ 6b08708`. Dirty: ` M repro/PLAN_OLHOFFCURRENT_UPDATE.md`, the uncommitted §7 already noted by the scientific-delta audit. It was not read.

The migration plan was read from the committed object: `git show 253069:repro/PLAN_OLHOFFCURRENT_UPDATE.md`. It is identical at `6b08708` (`git diff --quiet` passed).

## Changed relative to 6b08708 (implementation and docs)

`architecture/olhoffSolve.m`, `+olh/+move/limit.m`, `+olh/+move/exhaustion.m` (new), `+olh/+config/{schema,validate,fromLegacy,toLegacy,describe}.m`, `docs/{CONFIG_REFERENCE,SCIENTIFIC_CONFIG_PROVENANCE,MIGRATION_FROM_LEGACY}.md`. Tests `test_stage_exhaustion`, `test_outer_timing`, `test_preset_resolution_unchanged` (+ fixture) and the audit tree `repro/audits/upstream_olhoffcurrent_capabilities/**` are not part of the executable subset.
