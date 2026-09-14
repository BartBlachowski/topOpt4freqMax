# MIGRATION_CLASSIFICATION — Parts 19, 20, 21, 22

```
PRESERVE_OLD_OLHOFFCURRENT_PRESET
SOURCE_METHOD_REQUIRES_DISTINCT_NAMED_PRESET
```

## 1. Per-change classification (Part 19)

Classes: A paper fidelity, B standard method, C reconstruction, D new formulation, E software/reporting,
F bug fix. **No change qualifies as F**: the target's SIMP + eq.(4b), ladder controllers and stops are
internally consistent; none was shown incorrect.

| # | change (source 6b08708 unless noted) | files | class | decision | reason |
|---|---|---|---|---|---|
| 1 | Pedersen (2000) stiffness law + its derivative | `+olh/+material/stiffnessInterpolation.m` (new), `fem/assemble2D.m`, `algo/genGrad.m`, stiffness struct in `olhoffSolve.m` | B/D | **PROMOTE AS NEW NAMED PRESET** (code is inert under SIMP — proven bitwise; activation only through the new preset) | STRONG single-factor + same-state evidence; paper-named alternative; changes the relaxed problem |
| 2 | linear mass eq.(2) selection | preset field only | A-option/D | **PROMOTE AS NEW NAMED PRESET** | part of the validated realization; MODERATE evidence it is not the spike suppressor |
| 3 | adaptive per-element box | `+olh/+move/limit.m` (adaptive branch, 5th argument), vector box in `olhoffSolve.m`, config layer | C | **PROMOTE AS NEW NAMED PRESET** | first divergence, drives rate and termination; unstable with SIMP/4b, so never offered with the old law as a production candidate |
| 4 | ε-test without guards as the stop of the new realization | preset fields | C | **PROMOTE AS NEW NAMED PRESET, with caveat** | HEURISTIC_STOP; false stop observed under SIMP/4b; clean in all 18 Pedersen runs |
| 5 | `duOlhoffAdaptivePedersen` preset | `+olh/+presets/duOlhoffAdaptivePedersen.m`, `list.m` | C/D | **PROMOTE AS NEW NAMED PRESET** under a production name that states the formulation (§2) | verified sweep evidence; bitwise reproduction on this host |
| 6 | eigSolve options pass-through | `fem/eigSolve.m` | E | **PROMOTE NOW** | bitwise identical at defaults |
| 7 | `res.aux` (Mnd, moveMean); `hist.move` = max box | `olhoffSolve.m` | E | **PROMOTE NOW** | reporting; document that `hist.move` is the max for vector boxes |
| 8 | `hist.tOuter` timing instrumentation (target-local) | target `olhoffSolve.m` | E | **PROMOTE NOW** (must survive) | required by the benchmark accounting |
| 9 | stage-exhaustion controller (target-local) | target `exhaustion.m`, `limit.m` branch, `olhoffSolve.m`, config layer | C | **KEEP HISTORICAL ONLY** (must survive in `+impl`) | reproduces the C320/C480/C800 three-rung evidence; not used by either production preset |
| 10 | frozen `duOlhoffFrozenM4` / `duOlhoffFixedPenaltySensitivityFiltered` | presets | C | **KEEP HISTORICAL ONLY**, still resolvable with unchanged config hash | conference realization; bitwise anchors; all prior audits |
| 11 | `duOlhoffAdaptiveMove` (SIMP/4b + adaptive box) | preset | C | **KEEP HISTORICAL ONLY** (reclassify as EXPERIMENT_PRESET) | reproduces A2, C3, M1; spikes at 240/480/800 |
| 12 | `duOlhoffOuterAsymptotes`, `optimizer.inner.asymptoteHistory = 'outer'` | preset, `innerLoopRho.m` | C | **DO NOT PROMOTE** (code arrives inert with byte-copy; preset EXPERIMENT_PRESET) | refuted by the source itself (N1–N6) |
| 13 | `stop.guards.boxInactiveFraction`, `settledWindow`, `move.initial = Inf` | `olhoffSolve.m`, config | C/E | **DO NOT PROMOTE** as scientific options (inert, default off) | used only by refuted presets |
| 14 | plan Phase 3: repoint `olhoffcurrent_preset.m` to `duOlhoffAdaptivePedersen`, `historicalAliases = {}` | production files | — | **DO NOT PROMOTE** | silently changes the meaning of the production preset and erases lineage |
| 15 | uncommitted source §7 (ρ_min 1e−7, `maxCluster` 4 → 2, Pedersen/linear mass for all methods) | not in the audited commit | D | **DO NOT PROMOTE** | uncommitted, unvalidated; changes the new formulation again (`maxCluster` 2 → Jcalc 3 disables the ω_J multiplicity check) |
| 16 | Pedersen under the three-rung ladder | — | D | **NEEDS CAUSAL TEST** (only if someone wants to *repair* the old preset) | the missing cell; not needed to migrate #1–#5 |
| 17 | sweep tables' SIMP + eq.(4) re-evaluation | `repro/sweep_table.m` | E | reporting rule: every benchmark table must name its evaluator model | native vs re-evaluated ω differ (166.01 vs 166.3 at 480) |

## 2. Preset identity policy (Part 20)

**A. Preserve the old preset.** `duOlhoffFixedPenaltySensitivityFiltered` → `duOlhoffFrozenM4`
keeps its name, delegation and config hashes (480×60 `a49417d0…`; 320×40 `2a5b5009…`). The existing
`test_preset_equivalence` (bitwise vs the frozen conference record) stays and must still pass.

**B. Add the source realization under a distinct name.** Derived from what actually distinguishes it:

```
duOlhoffPedersenAdaptiveBoxSensitivityFiltered
  upstreamPreset      duOlhoffAdaptivePedersen   @ 6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7
  materialLaw         Pedersen (2000) eq.(5), linear below 0.1 (rho/100); mass eq.(2) linear, q = 1
  moveController      adaptive per-element box: initial 0.10, floor 0.002, x1.2 monotone / x0.7 reversal (outer history)
  stopRule            ||drho||_2 < 0.05 sqrt(NE/3200), no guards, no persistence
  unchanged           p = 3 fixed; Sigmund sensitivity filter on all f_sk at R = 0.06 b; N = 2 subspace with
                      offsets and off-diagonals; published MMA on the increment (tol 0.05, 5..500); rho0 0.5; rho_min 1e-3
  provenanceAliases   S160x20 ... S800x100 (repro/results, SWEEP_R06)
  formulationSplit    differs from duOlhoffFixedPenaltySensitivityFiltered in the low-density material law
  epistemicClass      reconstruction (class C controller, class B/D material law)
  mustNotBeLabelled   'Olhoff 2007'
```

(The plan's `duOlhoffAdaptiveBoxPedersenSensitivityFiltered` is acceptable if preferred; what is not
acceptable is any name without the material-law axis, or reuse of the old name.)

**C. Selection is explicit and versioned.** `olhoffcurrent_preset` must accept a preset name and keep
both registered; production scripts and the benchmark harness name the preset. Changing the default
is a recorded provenance event (PROVENANCE.json entry with old/new name, upstream commits, config
hashes, date), never an edit of the existing entry. Every result embeds preset name, upstream
preset, upstream commit and config hash. `historicalAliases` is never emptied.

## 3. Review of `repro/PLAN_OLHOFFCURRENT_UPDATE.md` (committed version) — Part 21

| phase | decision | reasons from this audit |
|---|---|---|
| **0** upstream committed & accepted | **APPROVE WITH MODIFICATION** | The commit exists and its evidence verifies (SOURCE_SWEEP_EVIDENCE_PASS; bitwise reproduction of S480 iterations 1–5 on this host). Modifications: pin exactly `6b08708` (or a later explicitly accepted commit) — the working tree now carries an uncommitted §7 that must not leak into a promotion; set PROVENANCE `source.branch` to `repro/natural-convergence` (the `canonical-config` branch model no longer matches currentness output); still run the 12 upstream anchors (not done here). |
| **1** promote the source files | **APPROVE WITH MODIFICATION** | The file set is exactly 15 modified + 4 new; the "unchanged" list verifies 20/20 identical; the shared numerical path is bitwise identical at 9 states. Modifications: reclassify `duOlhoffAdaptiveMove` and `duOlhoffOuterAsymptotes` as EXPERIMENT_PRESET (not production-eligible); treat `innerLoopRho` outer-asymptote code and the two stop guards as inert. "74 → 79" is wrong: the target has 75 files, so the result is 79 only if `exhaustion.m` is kept (it must be). |
| **2** re-apply target adaptations | **APPROVE WITH MODIFICATION — do 2.4 first** | Six files changed on both sides (`olhoffSolve.m`, `limit.m`, `schema/validate/fromLegacy/toLegacy.m`). A local three-way merge would enlarge the undocumented target-local divergence (PROVENANCE already says "one adaptation" while seven files differ). Promote stage exhaustion + `tOuter` upstream as default-off options first, so the target becomes byte-identical to an upstream commit; then prove bitwise: A1_frozen160, the validated C320 three-rung record, and S160x20. |
| **3** production preset | **APPROVE WITH MODIFICATION** (the in-place replacement is **REJECTED**) | Add the new named preset (§2) instead of repointing `olhoffcurrent_preset`; keep `historicalAliases`; caveat must state the formulation split, the heuristic ε-stop (false stop seen under SIMP/4b), "not bimodal beyond 160×20 at R = 0.06", and that sweep frequencies were re-evaluated under SIMP + eq.(4). 3.2 (cap 400) fine. |
| **4** integrity, provenance, tests | **APPROVE WITH MODIFICATION** | 4.3: do **not** replace `test_preset_equivalence` — keep the frozen test unchanged and add a second test for the new preset vs `S160x20`. 4.2: PROVENANCE must also correct the stale 74-file/one-adaptation statement and cite this audit. Copy S160x20 digests into `evidence/` per the evidence policy. |
| **5** benchmark harness | **APPROVE WITH MODIFICATION** | Assertions must be per preset (frozen assertions retained under the frozen preset, new ones added); 5.3 display name must mention the Pedersen formulation, not only "adaptive box"; add the evaluator-model column (native vs common). 5.6 per-iteration fits: approve (harness/reporting). |
| **6** other methods | **DEFER** | See §4. |
| order/gates step 8 (nine-mesh campaign) | **DEFER** | not an immediate step; requires the migration and the 160×20 preflight first |

**15-file promotion justified?** Yes as a *code* promotion (19 files, verified exact, inert for the old
presets), **no** as a *meaning* change of the production preset.
**Merge of local controller/timing edits?** Yes, both must survive; upstream-first preferred.
**Manifest/provenance updates?** Required, and must fix the existing inconsistency.
**Harness changes?** Approved with per-preset assertions.

## 4. Phase 6 stays separate (Part 22)

| item | kind | decision |
|---|---|---|
| 6.1 common physical radius 0.06 for Proposed and Yuksel (re-freeze profiles, new ids) | **benchmark-policy change** (alters frozen scientific parameters of two methods) | separate authorization; not implemented |
| 6.2 Pedersen linearization or Du–Olhoff cut-off in the Proposed solver | **scientific-method change** to another method | separate authorization; this audit's evidence concerns Olhoff only and does not transfer (different optimizer, filter builder, E_min 1e−9) |
| 6.3 report the common evaluator ω₁ for all methods | **reporting-only change** | may proceed independently |
| uncommitted §7.1 same-problem settings (Pedersen + linear mass for all, one H builder, ρ_min 1e−7, eigen modes) | mixed: benchmark-policy + scientific-method changes for all three methods, and a new unvalidated change of the Olhoff formulation | not part of the audited plan; separate authorization and validation |
| uncommitted §7.2 common stop (RMS 8.8e−4) and move 0.10 for all | scientific-method changes for Proposed/Yuksel | separate authorization |
| uncommitted §7.4 columns, per-iteration fits | reporting-only | may proceed independently |
