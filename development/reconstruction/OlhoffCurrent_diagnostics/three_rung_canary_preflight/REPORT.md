# BOTTOM LINE

**The validated three-rung controller generalizes. The science does not follow it.**

Both canaries ran, both converged, and all twenty preregistered gates passed.
Across six meshes now spanning a factor of 25 in element count, the controller
produces exactly one structure: a single S1 declaration, then both lower rungs
at the minimum dwell of 39, terminal declaration on branch B with full
persistence. Nothing new appeared at 480×60 or at 800×100, no cap was
approached, and beta never held authority anywhere.

But under that same correct controller, held fixed, the **design gets steadily
worse with refinement**: M_nd 12.8 → 12.9 → 12.9 → 15.4 → 26.3 → 34.4 % across
160/240/320/400/480/800, accelerating, with ω₁ falling monotonically and neither
quantity showing an asymptote. The three-rung 800×100 design is as gray as the
*legacy* 480×60 design. And this is not stopping too early: at both canaries the
terminal window is flat to 0.014 % in ω₁ and 0.07 points in M_nd. The controller
stopped where the design had stopped moving; the design stopped moving while
still a third intermediate-density.

So the September 11 campaign's fine-mesh deterioration was **not** an artefact of
the wrong stopping rule. Running the right rule reproduces it. That redirects the
open question from stopping to spectral/discretization/formulation, and it is why
the nine-mesh campaign is blocked rather than authorized — running it now would
spend a day of compute to document, at nine meshes, a problem two canaries have
already documented.

## Final verdicts

```
THREE_RUNG_DEPLOYMENT_PREFLIGHT_PASS

C480_THREE_RUNG_CANARY_PASS

C800_THREE_RUNG_CONTROLLER_PASS
C800_ENDPOINT_SCIENTIFICALLY_SUSPICIOUS
C800_RUNTIME_BEHAVIOR_EXPLAINED

FIXED_WORK_SCALING_SANITY_PASS
NEXT_MODE_WARNING_MATERIAL_CONCERN

THREE_RUNG_CONTROLLER_GENERALIZES_BUT_FINE_MESH_SCIENCE_SUSPICIOUS

CORRECT_NINE_MESH_CAMPAIGN_BLOCKED
```

This is preregistered **Case 4** exactly as written before any run.

## The two canaries

| | 480×60 | 800×100 |
|---|---|---|
| status | CONVERGED | CONVERGED |
| outer / cap | 386 / 1600 | 468 / 1600 |
| inner MMA (non-converged) | 7 300 (0) | 10 404 (0) |
| S1 / S2 / S3 declarations | 308 / 347 / 386 | 390 / 429 / 468 |
| all branches | B, B, B | B, B, B |
| terminal amp/ε | 0.0524 | 0.0845 |
| ω₁ | 163.93225938567002 | 161.94583808332942 |
| gap12 | 0.12979791372346 | 3.4875831182270845e−05 |
| M_nd | 26.34156302529312 % | 34.41232077776818 % |
| next-mode warnings | 4 (1.04 %), iterations ≤ 12 | 81 (17.31 %), iterations 26–113 |
| wall | 2799.28 s | 7554.10 s |
| config hash | `03097a28…782e` = frozen | `7724af5e…0ffe` = frozen |

---

## Direct answers

**1. Did the preflight prove the correct controller was actually deployed?**
Yes. 47 of 47 field checks at each mesh, 0 blockers, from a runtime resolution
read back with `olh.config.getPath` after defaults → preset → overrides →
derived rules → validation. The config hashes were **frozen before MATLAB was
reachable on this host** and the runtime resolution reproduced them bit for bit,
so the assertion is a pre-run commitment that was met, not a value recorded
afterwards.

**2. What exact effective config was used?** `duOlhoffFrozenM4` plus
`move.levels = [0.04 0.02 0.01]`, `move.continuation.signal = stageExhaustion`,
`stop.rule = stageExhaustion`, cap 1600, `diagnostics = true`, single thread;
ε = 0.15 (480) / 0.25 (800), rminEl 3.6 / 6.0, free DOF 58 678 / 161 798 read
from the built model. p = 3 fixed, eq. (4b) mass with q = 1, sensitivity filter
on every f_sk at R = 0.06·b, projection off, fixed subspace N = 2 with diagonal
offsets and off-diagonals, published MMA on the increment. Full record in
`EFFECTIVE_CONFIG.json` (`RUNTIME_RESOLVED`), field table in
`CONFIG_ASSERTIONS.md`.

**3. Did 480×60 reach credible terminal E?** Yes. Terminal persistent E at
move 0.01, stage 3, branch B, full persistence P = 20, iteration 386. All ten
gates pass.

**4. At what S1/S2/S3 iterations and branches?** 480×60: S1 308 (B), S2 347 (B),
S3 386 (B). 800×100: S1 390 (B), S2 429 (B), S3 468 (B). Branch A never fired at
either mesh. Both totals are S1 + 78, the pattern all four validated meshes show.

**5. Was beta ever authoritative?** No, at either canary — asserted structurally
(`signal ≠ boundVariable`, `stop.rule ≠ designChange`), and `olh.config.validate`
refuses to resolve a half-applied policy. Beta was recorded as a diagnostic
(final 26 873.74 and 26 228.45). In **production as it stands today** beta does
still hold continuation authority, because promotion never happened.

**6. Did 480 expose any new controller pathology?** No pathology. It did expose
one thing the in-range set could not: **S1 length is not monotone in mesh** —
102, 206, 274, 388, then 308 at 480. My pre-run budget projection assumed
monotonicity and over-predicted by 65 %. It was explicitly not an acceptance
criterion and no gate read it.

**7. Was 800 reached?** Yes. Authorized only after 480 passed its gate, as
preregistered.

**8. Did 800 reach credible terminal E?** Yes — controller-wise. Terminal
persistent E at move 0.01, stage 3, branch B, full persistence, iteration 468,
no cap hit (headroom 1132). The *endpoint's scientific quality* is a separate
verdict: see 14 and 25.

**9. What exact S1/S2/S3 events occurred at 800?** S1 declared 390 (window
371–390, branch B, amp/ε 0.874, med₂₀cos 0.943); S2 declared 429 (410–429, B,
amp/ε 0.237, med₂₀cos 0.917); S3 declared 468 (449–468, B, amp/ε 0.0845,
med₂₀cos 0.821). Stage starts 1 / 391 / 430; two descents, both consumed by the
exhaustion rule at the iteration after declaration.

**10. Did 800 enter any low-amplitude cancellation hole?** No. Branch B requires
amp < ε **and** med₂₀cos > 0 — coherent motion, not merely small motion. All
three declarations carry high positive med₂₀cos (0.943, 0.917, 0.821) and
med₂₀net (0.958, 0.886, 0.903). Branch A, the cancellation branch, never fired.

**11. Was its endpoint substantially different from the old legacy endpoint?**
Yes, substantially and favourably. ω₁ 153.302 → 161.946 (+5.64 %); M_nd 50.66 →
34.41 % (−16.24 points); final move 0.02 → 0.01; 170 → 468 outer. Topology IoU
at threshold 0.5 is 0.7021 with 15.45 % of elements flipping side, so the
designs share a family but differ materially.

**12. Did M_nd improve?** Against legacy, yes at both canaries: −8.33 points at
480×60, −16.24 at 800×100. **In absolute terms it is worsening with mesh under
the correct controller**: 12.8, 12.9, 12.9, 15.4, 26.3, 34.4 % across
160/240/320/400/480/800. That is the study's central negative finding.

**13. Did ω₁ improve?** Against legacy, yes: +1.25 % at 480×60, +5.64 % at
800×100. Across meshes under the correct controller it falls monotonically —
169.98, 167.04, 166.43, 166.45, 163.93, 161.95 — with no asymptote.

**14. Did topology look more mature?** Against legacy at 800×100, clearly yes:
resolved black flanges, a defined central void, distinct diagonal members where
the legacy field is a gray blur (`FIG_12`). Across the two canaries the topology
family is the **same** (`FIG_11`) — the 800 design is the 480 design with larger
and grayer end regions. So topology is mesh-consistent; discreteness is not.

**15. How many multiple-J warnings occurred?** 480×60: 4 of 386 (1.04 %).
800×100: 81 of 468 (17.31 %). The legacy runs recorded **the same absolute
counts** — 4 of 164 and 81 of 170 — which is informative: both policies traverse
the same early coalescence phase, and only the length of what follows differs.

**16. Did next-mode warnings overlap terminal convergence?** **No, at either
canary.** At 800×100 the 81 warnings are a contiguous early transient, iterations
26–113; nothing in the remaining 355 iterations (76 % of the run), nothing in
stage 2 or 3, nothing in the declaration window 449–468. Endpoint gap23 = 1.362.
No stopping decision in this study was taken inside a spectrally ill-posed regime.

**17. Was total 800 runtime faster/slower than 720-like expectations?** The
question was posed because the *legacy* campaign showed 800×100 finishing faster
than 720×90. That inversion is fully explained (answer 21) and it **did not
recur**: the three-rung 800×100 canary ran 468 iterations and 7554 s, far above
anything in the legacy series, because the frozen rule requires a persistent
terminal window before it will stop.

**18. What was wall time per outer?** 7.2509 s at 480×60; 16.1378 s at 800×100.
Ratio 2.23 for a 2.78× mesh (exponent 0.83).

**19. What was eigensolver time per outer?** 0.27157 s at 480×60 (3.75 % of
total); 1.18725 s at 800×100 (7.36 %). This is **assembly + eigensolve** — the
two are not separable in this instrumentation, at any mesh.

**20. What was MMA time per outer?** 6.96098 s at 480×60 (**96.00 %**);
14.89901 s at 800×100 (**92.32 %**). The nested MMA dominates at both meshes, as
it does at all nine legacy meshes.

**21. If total runtime was unexpectedly low/high, why?** Nothing unexpected in
the canaries: `Σ tOuter` accounts for 99.98 % of total wall at both, and every
kernel scales as the legacy fits and the fixed-work benchmark predict. The
*legacy* 720 → 800 inversion decomposes exactly:
`0.8288 = 1.0872 × 0.7623` (total-wall ratio = per-outer ratio × iteration-count
ratio, reproduced to 16 digits) — the finer mesh was 8.7 % more expensive per
iteration and ran 23.8 % fewer iterations. A stopping-regime effect, not
anomalous scaling, and it bought a worse design.

**22. Did fixed-work kernel timings scale sensibly from 480 to 800?** Yes —
`FIXED_WORK_SCALING_SANITY_PASS`. Assembly+eigensolve ×4.826 (exponent
**1.541**, which is what 2D sparse direct factorization predicts), gradients
×3.387 (1.194), MMA sub-problem whole solve ×2.681 (**0.965**, essentially
linear). The benchmark independently reproduces the canaries' own stage-3
telemetry to within 0.8 % and 0.5 % on the eigensolve. The per-step figure
(exponent 0.359) is confounded by step count (14 vs 26) and is not the kernel
measure.

**23. Is three-rung cross-mesh generalization supported?** **Yes.** Six meshes,
NE 3 200 → 80 000, one frozen policy, identical structure every time, twenty
gates passed across the two canaries, no cap approached, no new pathology.

**24. Is the fine-mesh problem primarily a stopping/controller issue?** **No,
and this is the study's main result.** The correct stopping rule was run at both
canary meshes, behaved perfectly, and the endpoint still degrades. At both, the
terminal window is flat (ω₁ +0.0037 % and +0.014 %; M_nd −0.036 and −0.064
points over 20 iterations), so the runs did not stop early — the design stopped
moving while still gray.

**25. Or is a deeper spectral/multiplicity/discretization issue more plausible?**
**Yes, by elimination** — stopping has now been excluded as the explanation.
Which deeper cause is open. `CAMPAIGN_DECISION.md` §3 lists five untested
candidates: fixed p = 3 without continuation, the sensitivity filter at fixed
physical radius, absent projection, spectral crowding during coalescence, and
genuine continuum behaviour. This study tested none and prefers none. Note that
the timing evidence argues *against* the spectral candidate being the direct
cause: the warning regime resolves 355 iterations before the 800×100 endpoint.

**26. Is the full correct nine-mesh campaign authorized?** No.
`CORRECT_NINE_MESH_CAMPAIGN_BLOCKED`.

**27. If not, what exactly blocks it?** Three things, in order of importance:
(a) the unexplained M_nd growth under the correct controller — running nine
meshes now would document it nine times rather than explain it once;
(b) production still delegates to `duOlhoffFrozenM4`, so a campaign resolving
through `olhoffcurrent_config` would run the legacy policy again;
(c) retention and clean timing remain mutually exclusive under the present
`runtime.diagnostics` instrumentation, so one series cannot be both an evidence
and a performance campaign. `CAMPAIGN_DECISION.md` §5 gives the frozen
specification that clears (b) and (c); (a) needs a diagnosis, not a campaign.

**28. Were any scientific parameters changed after outcomes were seen?** No.
The preregistration is frozen at SHA-256
`25a2500b6e708202b51fba705cb8604c20de577f56f8707f1ac93ce6fec14c82`, with that
digest recorded in `evidence/FINAL_SHA256.PRERUN.txt`, written while the study
still stood at a preflight FAIL and zero runs existed. No threshold, window,
persistence length, tolerance, move value or cap was touched. The cap stayed at
the inherited 1600 even though the pre-run projection suggested 800×100 might
approach it.

**29. Were exactly one 480 and at most one 800 scientific runs executed?** Yes —
exactly one each. `cp_run.m` accepts only those two meshes and refuses every
other in code. No run was repeated, extended, restarted or repaired.

**30. Was the full nine-mesh campaign NOT executed?** Correct — not executed,
not started, not partially started. No 560×70, 640×80 or 720×90 run exists.

---

## Figures

All twelve required figures exist, from canary data.

| # | figure | file |
|---|---|---|
| 1 | ω₁ trajectory, 480 | `FIG_1_omega_480x60` |
| 2 | ω₁ trajectory, 800 | `FIG_2_omega_800x100` |
| 3 | M_nd / grayness | `FIG_3_discreteness_{480x60,800x100}` |
| 4 | move / stage timeline | `FIG_4_move_stage_{480x60,800x100}` |
| 5 | A/B/E timeline | `FIG_5_branch_timeline_{480x60,800x100}` |
| 6 | gap12 / gap23 | `FIG_6_gaps_{480x60,800x100}` |
| 7 | multiple-J warning timeline | `FIG_7_multiJ_{480x60,800x100}` |
| 8 | inner MMA per outer | `FIG_8_inner_{480x60,800x100}` |
| 9 | per-outer timing decomposition | `FIG_9_timing_{480x60,800x100}` |
| 10 | cumulative wall time | `FIG_10_cumwall_{480x60,800x100}` |
| 11 | 480 vs 800 final topology | `FIG_11_topology_480_vs_800` |
| 12 | legacy vs three-rung 800 topology | `FIG_12_topology_legacy_vs_three_rung_800` |

Three further figures come from retained legacy/historical records and are
labelled as such in their own titles: `FIG_A` (validated three-rung structure and
the pre-run budget extrapolation), `FIG_B` (legacy next-mode regime), `FIG_C`
(legacy 720 → 800 inversion).

## Artifact guide

Preregistration and gates: `PREREGISTRATION.md` (frozen; original at
`evidence/PREREGISTRATION.frozen`).
Deployment: `DEPLOYMENT_PREFLIGHT.md`, `CONFIG_ASSERTIONS.md`,
`EFFECTIVE_CONFIG.json`, `PREFLIGHT_MANIFEST.json`, `INSTRUMENTATION.md`.
Canaries: `C480_REPORT.md`, `C800_REPORT.md`.
Analysis: `PERFORMANCE_DECOMPOSITION.md`, `FIXED_WORK_TIMING.md`,
`MULTIPLICITY_WARNING_AUDIT.md`, `LEGACY_COMPARISON.md`.
Decision and the next task's frozen spec: `CAMPAIGN_DECISION.md`.
Provenance and integrity: `PROVENANCE.md`, `METRICS.json`, `DATA_MANIFEST.json`,
`EVIDENCE.json`, `FINAL_SHA256.txt`, `evidence/`.
Drivers: `scripts/` — `cp_run.m` is one command per canary.

# WHAT WE LEARNED

The validated three-rung controller generalizes cleanly beyond its validated
range. Six meshes, a 25× span in NE, one frozen policy, one structure every
time: S1 + 78, both lower rungs at the minimum dwell, terminal declaration on
branch B with full persistence and terminal amp/ε in a narrow band. Twenty gates
across two canaries, all passed. That question is settled, and it is the
question the September 11 campaign was supposed to answer and could not.

The fine-mesh degradation is **not** a stopping artefact. This is the finding
that changes the programme's direction: running the correct rule reproduces the
deterioration the legacy campaign showed, at flat terminal windows, with the
design having genuinely stopped moving. Whatever is wrong at fine mesh, the
stopping rule was not it.

Two mechanical findings worth carrying forward. The MMA sub-problem's cost per
step rises ×2.0–2.9 from the first rung to the last while step counts do not
explain it, so the terminal rungs cost 28–34 % of wall time on 17–20 % of the
iterations — and `mmasub`'s internal iterations are not instrumented, so the
cause is unresolved. And the eigensolve scales as NE^1.54, exactly what 2D
sparse direct factorization predicts, which is the strongest single check that
the numerical kernels are behaving.

# WHAT WE DID NOT LEARN

Why the design degrades under refinement. Five candidates are listed and none is
tested: fixed p = 3 without continuation, the sensitivity filter at fixed
physical radius, absent projection, spectral crowding during coalescence, and
the possibility that the continuum problem has no discrete optimum at this
volume fraction. Whether the controller behaves at 560×70, 640×80 or 720×90 —
not run, and not interpolated, though the canaries bracket the range. Why the
MMA sub-problem costs more per step at a tighter move limit. Whether the
next-mode warning regime during coalescence has any causal role — the timing
evidence argues against it, but that is an absence of correlation, not a proof.

# WHAT I WOULD DO NEXT

Diagnose the grayness before spending a campaign on it. Two of the five
candidates are cheap and need no controller change: p continuation and
projection are both already implemented and both currently disabled. Either can
be tested at 480×60 and 800×100 against the canary endpoints now in hand — which
is what makes those two endpoints worth keeping, and why they were retained in
full.

If that shows the grayness is a formulation choice rather than a discretization
limit, the nine-mesh campaign becomes the right experiment and its specification
is written and ready in `CAMPAIGN_DECISION.md` §5. If it shows the continuum
problem genuinely has no discrete optimum here, the programme should record that
as a negative result rather than run nine meshes to restate it.

Do not change the controller. Nothing in this study justifies touching A, B, the
windows, the persistence length, the ladder or the tolerance scaling — the
controller is the one part of the system that demonstrably works at every mesh
tested.
