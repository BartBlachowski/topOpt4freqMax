# CONFIG_EQUIVALENCE — Phase 11

# `PROMOTED_PRODUCTION_CONFIG_EQUIVALENCE_NOT_ASSESSED_PROMOTION_NOT_PERFORMED`

Phase 11 compares the **promoted** production configuration against the
validated candidate. Promotion did not occur, so the left-hand side of that
comparison does not exist. Issuing `PASS` would assert equivalence with a
configuration that has never been created; issuing `FAIL` would report a defect
where none was found. Neither is true, so neither is issued.

## 1. What *was* established

Both configurations were resolved and compared field-wise across all 81 schema
leaves (`PROMOTION_DIFF_PLAN.md`). Against **production as it stands today**:

| Group | Result |
|---|---|
| scientific formulation (18 fields audited) | **identical** |
| policy group (12 fields audited) | 9 identical, **3 differ** — and those 3 are exactly the promotion delta |
| unexpected computational differences | **0** |

So the only distance between today's production and the validated candidate is
the intended change itself. Nothing unrelated has drifted.

## 2. The validated candidate is provably the tested one

```
re-resolved cfgHash        afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab
recorded by the C320 run   afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab
```

The comparison baseline is not a reconstruction; it is the configuration that
produced `CONVERGED @352`.

## 3. How a future Phase 11 must assert equivalence

**Not by comparing whole-config hashes.** After promotion the production
`cfgHash` will legitimately differ from the candidate's `afad9ea4…`, because
`runtime.maxOuter` (400 vs the study's 1600) and `runtime.diagnostics`
(false vs true) are harness settings that are *not* promoted and *are* included
in the hash. A whole-hash comparison would produce a spurious FAIL.

Equivalence must instead be asserted **field-wise over the computational set**:
every schema leaf except the declared harness/label fields
(`runtime.name`, `runtime.maxOuter`, `runtime.diagnostics`, `runtime.verbose`,
`runtime.singleThread`), which the schema itself documents as not read by the
solver mathematics.

The audit script that does this already exists —
`scripts/tr_policy_recovery.m` — and reports
`UNEXPECTED computational differences: 0` today. After promotion the three
POLICY rows should move to `same = 1` and the unexpected count must remain 0.
