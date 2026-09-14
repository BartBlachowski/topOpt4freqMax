# MIGRATION_GATE — Parts 24 and 25

```
OLHOFFCURRENT_MIGRATION_READY_WITH_NAMED_FORMULATION_SPLIT
```

## Why this gate

- **The formulations differ** (FORMULATION_COMPARISON: low-density stiffness and mass laws), so plain
  READY — which would let the migrated realization stand in for the current one — is excluded.
- **The source method is scientifically defensible:** Pedersen (2000) is the remedy Du & Olhoff §2.2
  name; eq.(2) is printed; the adaptive box is a class C reconstruction of the same standing as the
  ladder it replaces; its committed nine-mesh evidence recomputes exactly and its first five
  iterations reproduce bitwise on this host.
- **Its causal basis is established well enough to migrate *it*:** every shared kernel is bitwise
  identical, the controller is the first divergence, and the material law is shown necessary for the
  controller's clean behaviour (single factor + same state at 480, consistent at 240/800). The one
  open cell — Pedersen under the three-rung ladder — matters only for repairing the *old* preset,
  not for adding the new one, so REQUIRES_CAUSAL_TEST is not the right gate.
- **Nothing blocks it:** no source/target integrity failure, no uncommitted claimed evidence, no
  unexplained implementation difference.

## Conditions attached to the gate

1. Old preset `duOlhoffFixedPenaltySensitivityFiltered` → `duOlhoffFrozenM4` preserved with
   unchanged config hashes and its bitwise equivalence test.
2. New realization added under a distinct formulation-bearing name (MIGRATION_CLASSIFICATION §2);
   no in-place repointing; selection explicit and versioned.
3. Stage-exhaustion controller and `hist.tOuter` survive; the target becomes byte-identical to an
   accepted upstream commit (upstream-first, plan 2.4).
4. Promotion pins `6b08708` or a later explicitly accepted commit; the uncommitted §7 material is
   excluded.
5. PROVENANCE corrected (75 files / seven-file divergence) and pointing to this audit.
6. Caveats: heuristic ε-stop; formulation split; not bimodal beyond 160×20; evaluator model named in
   every table.
7. Phase 6 and all changes to Proposed/Yuksel stay out.

## Single next action (Part 25)

**C — first promote the shared upstream options (stage-exhaustion controller and `hist.tOuter`
instrumentation, both default-off) into the Olhoff repository on top of `6b08708`, verify there that
A1_frozen160, the validated C320 three-rung record and S160x20 reproduce bitwise, and only then
execute the modified migration.**

Why this and not A directly: six of the files the plan promotes changed on *both* sides. Merging them
inside `+impl` would make OlhoffCurrent diverge from every upstream commit in seven files while its
provenance already claims one. Doing the merge upstream first turns the promotion into a byte copy
plus one documented instrumentation line, which is the only way the migration can keep
OlhoffCurrent's "provably equal to an accepted upstream commit" identity. Not B: the remaining causal
test does not gate this migration. Not D: the source is defensible and wanted as the production
candidate, so it should enter OlhoffCurrent — as a second named preset, not as a replacement.

Not recommended as the next step: another nine-mesh sweep; the C480 SOCP treatment; Phase 6.
