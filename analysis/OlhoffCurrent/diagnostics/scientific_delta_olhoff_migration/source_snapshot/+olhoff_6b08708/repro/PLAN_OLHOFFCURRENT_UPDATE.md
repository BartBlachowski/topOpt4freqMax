# Plan: bring `analysis/OlhoffCurrent` up to the adaptive-box / Pedersen realization

Goal: `examples/Performance/performance_comparison.m` produces the Olhoff column of
`repro/results/SWEEP_R06.md` (preset `duOlhoffAdaptivePedersen`, physical radius
0.06) under the production path guard, with the same provenance discipline the
tree has now. No file is changed by this plan; it lists what would change.

Reference state of the upstream work: branch `repro/natural-convergence` in
`/Users/piotrek/Programming/Matlab/Olhoff`, uncommitted (25 modified/added source
files plus `repro/`). Regression evidence: `run_all_tests` 0 failures, anchors
A1_frozen160 and A4_nodescent160 bitwise IDENTICAL, so the frozen preset is
untouched by the promotion.

---

## Phase 0 — upstream must be a committed, accepted state

The OlhoffCurrent provenance model records a commit hash and checks that it is an
ancestor of the upstream branch it names (`architecture/canonical-config`).

0.1 Commit the upstream branch (solver, config layer, presets, docs, `repro/`
    runner and tables; the 60 MB of `repro/results/*/res.mat` may be committed or
    listed in `EVIDENCE_MANIFEST.sha256` per the existing ≥ 5 MB rule).
0.2 Either merge `repro/natural-convergence` into `architecture/canonical-config`
    or change `PROVENANCE.json: source.branch` to the new branch. The
    `olhoffcurrent_currentness()` state model needs one named branch.
0.3 Re-run upstream `architecture/anchors/code/anchorReport()` (12 anchors) and
    record the digests in the promotion report — that is the "verified before
    promoting" step the last promotion documented.

## Phase 1 — promote the source into `+impl/` (74 → 79 files)

Byte-copy from upstream, then re-apply OlhoffCurrent's own adaptations (Phase 2).

| upstream file | status | why |
|---|---|---|
| `architecture/olhoffSolve.m` | MODIFIED | adaptive move (vector box), `res.aux`, stiffness struct, eig options |
| `architecture/+olh/+config/schema.m` | MODIFIED | 86 fields: `move.policy=adaptive`, `move.adaptive.*`, `stop.guards.boxInactiveFraction`, `stop.guards.settledWindow`, `optimizer.inner.asymptoteHistory`, `material.stiffness.model/linearBelow`, `move.initial` domain `[0 Inf]` |
| `architecture/+olh/+config/validate.m` | MODIFIED | rules for the fields above |
| `architecture/+olh/+config/fromLegacy.m`, `toLegacy.m` | MODIFIED | legacy spellings `SA`, `sAGrow/sAShrink`, `stiffModel/stiffLinearBelow`, `innerAsy`, `boxInactive`, `settledWindow` |
| `architecture/+olh/+config/describe.m` | MODIFIED | prints the new policies |
| `architecture/+olh/+move/limit.m` | MODIFIED | `'adaptive'` policy; 5th argument `rho` |
| `architecture/+olh/+material/stiffnessInterpolation.m` | NEW | SIMP / Pedersen (2000) eq. (5) |
| `architecture/+olh/+presets/duOlhoffAdaptiveMove.m`, `duOlhoffOuterAsymptotes.m`, `duOlhoffAdaptivePedersen.m`, `list.m` | NEW / MODIFIED | the realization and its two parents |
| `fem/assemble2D.m`, `fem/eigSolve.m`, `algo/genGrad.m`, `algo/innerLoopRho.m` | MODIFIED | stiffness struct; eig options; outer-asymptote option (default off) |
| `architecture/docs/*.md` | MODIFIED | CONFIG_REFERENCE (regenerated), PRESETS, SCIENTIFIC_CONFIG_PROVENANCE |

Unchanged and not re-copied: `mma/`, `mma_published/`, `filter/`, `fem/model2D.m`,
`fem/elemMats2D.m`, `fem/massScale.m`, `algo/innerLoop.m`, `algo/innerLoopLP.m`,
`algo/deltaLambda.m`, `algo/multRule.m`, `algo/moveControl.m`, `algo/useMMA.m`.

## Phase 2 — re-apply OlhoffCurrent's two local adaptations on top

OlhoffCurrent diverged from upstream `695f03b` in six files (stage-exhaustion
controller + `hist.tOuter`). Both must survive, because `olhoffcurrent_run.m`
reads `hist.tOuter` and the diagnostics record the exhaustion option.

2.1 `olhoffSolve.m`: port (a) the `tOuterTic`/`hist.tOuter` instrumentation and
    (b) the `useExhaustion` block, the `exhaustStop` terminal admission and the
    `res.exhaustion` record onto the new upstream solver. The new adaptive block
    passes `rho` to `olh.move.limit` and uses `mvMax`; the exhaustion block reads
    `hist.move(outer)` and `hist.stage(outer)`, both still written — no conflict
    expected, but the merged file must be diffed against both parents.
2.2 `+olh/+move/limit.m`: add the `stageExhaustion` branch (OlhoffCurrent) to the
    file that now also has the `adaptive` branch (upstream). The two are in
    different `case` arms.
2.3 `schema.m`, `validate.m`, `fromLegacy.m`, `toLegacy.m`: add OlhoffCurrent's
    `stop.rule` field and the `stageExhaustion` enum value to the upstream
    versions (three small hunks each).
2.4 Cleaner alternative to consider: promote the exhaustion controller upstream
    first (it is a labelled class-C option, default off), so that OlhoffCurrent
    becomes byte-identical to upstream except the timing line. This removes the
    "PROVENANCE.md says one adaptation but six files differ" inconsistency noted
    in the fidelity review.

## Phase 3 — production preset

3.1 `olhoffcurrent_preset.m`: `upstreamPreset = 'duOlhoffAdaptivePedersen'`;
    new production name in the same descriptive style, e.g.
    `duOlhoffAdaptiveBoxPedersenSensitivityFiltered`; label
    "Du-Olhoff reconstruction, adaptive box, Pedersen linearization, sensitivity
    filtered"; `historicalAliases = {}` (this realization has no audit code);
    `epistemicClass` unchanged (class C reconstruction);
    `mustNotBeLabelled = 'Olhoff 2007'` unchanged.
3.2 `olhoffcurrent_config.m`: no change needed. `MaxOuter` default 400 was enough
    at every mesh of the 0.06 sweep (max 246); keep 400.
3.3 `olhoffcurrent_caveat.m`: reword — the caveat currently says the outer count
    depends on a "move-limit continuation schedule"; now it depends on the
    per-element adaptive box and on the ε value, and the method is not bimodal
    beyond 160×20 at this radius. Add: stiffness interpolation follows Pedersen
    (2000), named in Du & Olhoff §2.2, and mass is eq. (2).
3.4 `olhoffcurrent_run.m`: works as is. Optional: report `res.aux.Mnd(end)` next
    to `final_grayness` (same quantity, already computed from `x`), and
    `final_move_limit` should be documented as the MAX per-element box.

## Phase 4 — integrity, provenance, tests inside OlhoffCurrent

4.1 Regenerate `SOURCE_MANIFEST.json` (`olhoffcurrent_source_manifest`) — 79 files,
    new tree hash.
4.2 `PROVENANCE.json` / `PROVENANCE.md`: new source commit, branch, promotion
    date, list of adaptations (timing + exhaustion, or timing only if 2.4 is
    done), new preset name, and a pointer to upstream `NOTES.md §32` and
    `repro/results/SWEEP_R06.md` as acceptance evidence.
4.3 `tests/test_preset_equivalence.m`: it compares the production preset at
    160×20/320×40 against the frozen conference record and WILL FAIL by design.
    Replace by: (a) `duOlhoffFrozenM4` still reproduces the frozen record
    bitwise (proves the promotion did not touch the old realization), and
    (b) the production preset reproduces the upstream sweep record `S160x20`
    (copy `rho`, `omega`, `nOuter`, `cumInner` digest from
    `repro/results/S160x20/res.mat` into `OlhoffCurrent/evidence/`).
4.4 `tests/test_currentness.m` item 5: passes once `list.m` contains the new
    preset. `test_path_isolation`: no new bare function names are introduced
    (`stiffnessInterpolation` is a package member), so the owned-name set is
    unchanged; run it anyway.
4.5 `analysis/OLHOFF_CURRENT_PROMOTION_REPORT.md`: new section with the
    live re-execution of `S160x20` under the promoted tree (the equivalent of
    the earlier A1 live check).

## Phase 5 — the benchmark harness (`examples/Performance`)

5.1 `conference_bench/confbench_preflight.m` §5b: five of the six field-by-field
    assertions encode the frozen realization and will FAIL:
    - "M4 multiplicity treatment, frozen subN" — still true (subspace, 2); keep.
    - "fixed physical filter R = 0.06" — still true; keep.
    - "genuine nested MMA": true (`innerVar='drho'`); keep.
    - "outer RMS stopping semantics": requires `outerGuard == 'settledmove'`;
      the new preset has `settledMove = false` (no ladder to settle). Change to
      `outerGuard == 'none'` and add `boxInactive` absent/0.
    - "S2 continuation realization as frozen": replace by
      `moveFamily == 'SA'`, `move == 0.10`, `moveMin == 0.002`,
      `sAGrow == 1.2`, `sAShrink == 0.7`.
    - add: `stiffModel == 'pedersen'`, `stiffLinearBelow == 0.1`,
      `massInterp == 'lin'`.
5.2 `performance_comparison.m` `printMethodSettings`: the Olhoff branch prints
    "continuation: %s ladder %s ..." from `mc.moveFamily/s2Levels`; print the
    adaptive box and the stiffness/mass models instead (`isfield` on
    `stiffModel`).
5.3 `conference_bench/confbench_display_name.m`: `'Du-Olhoff reconstruction (M4)'`
    → a label without the audit code, e.g. `'Du-Olhoff reconstruction (adaptive box)'`.
5.4 `conference_bench/confbench_caveats.m`: replace the two sentences about the
    "move-limit continuation schedule" with the adaptive-box statement and the
    bimodality caveat; keep "must not be labelled Olhoff 2007".
5.5 `conference_bench/confbench_run_case.m` / `confbench_export.m`: unchanged
    (they read the accounting struct). Optional additions: a `gap12_pct` and
    `Mnd` column in the detailed CSV are already there (`stopping.gap12_pct`,
    `grayness`).
5.6 `conference_bench/confbench_scaling_fit.m` / `confbench_complexity_plots.m`:
    add a fit of PER-ITERATION cost (eigensolve per outer, MMA per inner
    sub-iterate, SIMP per iteration) alongside the total-time fit, because the
    Olhoff outer count is not monotone in NE (121 → 93 → 246 across the sweep)
    and total time cannot follow a power law. This is a harness change, not
    an Olhoff change.

## Phase 6 — what the other two methods need for a fair table (outside OlhoffCurrent)

6.1 Same physical filter radius for Proposed and Yuksel: `rmin_element` 2.0/2.5
    in `analysis/three_method_parametric_study/study_base_config.m` and the
    frozen `profile_freeze_manifest.json` become `radius_units = 'physical'`,
    0.06. That is a scientific change to frozen profiles and needs its own
    re-freeze and profile ids.
6.2 Proposed solver (`analysis/ourApproach/Matlab/topopt_freq.m`): linear mass
    with `rho_min = 1e-9` produces spurious localized modes on grey designs
    (native ω₁ = 109 at 160×20 vs 154 under the evaluator). Adopt the same
    Pedersen linearization or the Du–Olhoff mass cut-off.
6.3 Report the evaluator ω₁ (E1) next to the native one for all three methods;
    the Olhoff native values under the Pedersen model differ by < 0.5 % from
    SIMP + eq. (4) on black-and-white designs (sweep table), but the column
    should still be the common one.

## Expected result of the promoted column (from `SWEEP_R06.md`)

| mesh | outer | inner | ω₁ (SIMP+eq.4) | gap | M_nd | eig/outer [s] | per inner it. [s] |
|---|---|---|---|---|---|---|---|
| 160×20 | 121 | 2369 | 169.7 | 0.7 % | 0.115 | 0.053 | 0.168 |
| 240×30 | 111 | 2077 | 167.6 | 11.7 % | 0.123 | 0.113 | 0.334 |
| 320×40 | 101 | 2001 | 166.1 | 17.5 % | 0.141 | 0.204 | 0.470 |
| 400×50 | 93 | 1913 | 166.7 | 18.9 % | 0.122 | 0.326 | 0.639 |
| 480×60 | 112 | 2137 | 166.3 | 22.4 % | 0.131 | 0.476 | 0.890 |
| 560×70 | 130 | 2371 | 166.1 | 24.3 % | 0.133 | 0.601 | 1.034 |
| 640×80 | 156 | 2868 | 166.0 | 24.2 % | 0.133 | 0.726 | 1.170 |
| 720×90 | 204 | 3752 | 165.7 | 22.2 % | 0.162 | 1.145 | 1.215 |
| 800×100 | 246 | 4650 | 165.8 | 18.2 % | 0.165 | 1.359 | 1.251 |

Wall times in the sweep were measured with nine runs in parallel; the benchmark
runs sequentially, so expect ~10 % lower per-iteration times (H1 solo: 1.12 s per
inner sub-iterate at 800×100 vs 1.25 s in the sweep).

## Order and gates

1. Phase 0 (commit, anchors) → 2. Phases 1–2 (copy + merge, diff against both
parents) → 3. Phase 4.1–4.2 (manifest, provenance) → 4. Phase 3 (preset) →
5. Phase 4.3–4.5 (tests, live re-execution of S160x20) → 6. Phase 5 (harness) →
7. one-mesh preflight run of `performance_comparison.m` at 160×20 → 8. nine-mesh
campaign. Phase 6 is independent and can precede or follow.
