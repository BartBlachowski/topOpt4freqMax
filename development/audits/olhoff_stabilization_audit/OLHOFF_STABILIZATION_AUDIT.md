# Final pre-campaign Olhoff stabilization audit

## Executive conclusion

A single causal policy works: after **100 consecutive native states with N=2
and gap12≤1%**, reduce the move limit once from 0.005 to 0.0025. The trigger
uses only the current/past native trajectory, never stops the optimizer, and is
identical on every mesh.

The selected S1 policy makes raw common-E1≤1% persistent at iterations
186/234/234/246 on 160×20, 240×30, 320×40 and 400×50, versus
459/219/1150/1104 for the faithful fixed-move control. It advances persistent
1% quality by 273, 916 and 858 iterations on the three excursion-prone meshes,
with a 15-iteration delay on the already-stable calibration mesh. It also makes
the native 1% bimodal gap persistent at 145/106/103/961 versus
1504/106/1548/1511.

At k=1600, S1 retains N=2, support-to-support connectivity and terminal raw
E1/E2/E3 loss within 0.5% of the same-mesh baseline reference. All terminal
binary E1/E2/E3 losses are within 1.1%. There are no LP failures or nonfinite
iterations. The stronger S2 fails binary validation at 400×50; S3 fails its LP
solve and is rejected with failure precedence intact.

S1 does **not** make the method natively convergent. Its final maximum update
still reaches the reduced move limit. The practical semantics are therefore a
fixed total work horizon of 1600 outer iterations, explicitly labelled **not
native convergence**.

## Freeze, experimental order and integrity

The audit was staged as required:

1. freeze branch, HEAD, dirty-tree inventory, MATLAB/thread settings and source
   hashes;
2. diagnose all four existing baseline trajectories before testing a variant;
3. freeze the trigger and S0–S3 family;
4. run S1–S3 only on 240×30;
5. reject S3 and freeze S1/S2 before opening hold-outs;
6. validate unchanged on 160×20, 320×40 and 400×50;
7. select S1, reject S2, and perform a no-solve nine-mesh preflight.

The branch is `benchmark-methodology-r2`, HEAD is
`cb6353feb941f12b2aaa927e622649e1ccc926f7`, and the worktree was already dirty.
All prior changes were preserved. MATLAB was 25.2.0.3042426 (R2025b) Update 1,
MACA64, with one computational thread per run. All 61 reproduction-manifest
entries match; zero mismatch and zero missing. No authoritative source under
`Matlab/reproduction2007/` was changed.

The audit-side S0 mirror reproduced the first 20 baseline updates bit-for-bit.
The extracted per-iteration common-E1 evaluator matched the unchanged R3
evaluator at every verification checkpoint with maximum absolute error zero.
See [`preregistration.json`](preregistration.json),
[`holdout_candidate_freeze.json`](holdout_candidate_freeze.json) and
[`provenance.json`](provenance.json).

## Table A — Baseline late-dynamics diagnosis

“Good” below means raw-E1 loss≤1% relative to the same mesh's baseline k=1600
state. Saturation is the fraction of elements taking at least 98% of the 0.005
move. Exit turnover is exact-count binary turnover from the preceding state.

| Mesh | First good | Persistent good | 1% exits | gap≤1% reopenings | saturation at exits / mature control | mean update RMS at exits / control | exit binary turnover | modal/solver events | diagnosed mechanism | hypothesis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 160×20 | 186 | 459 | 6 | 3 | 29.6% / 27.8% | 0.00272 / 0.00264 | 0.146% | N stays 2; no failure | shallow same-basin excursions; move association modest | PLAUSIBLE |
| 240×30 | 219 | 219 | 0 | 0 | n/a / 21.7% | n/a / 0.00233 | n/a | no failure | persistent quality but strong near-period-two update reversal | PLAUSIBLE |
| 320×40 | 218 | 1150 | 5 | 7 | 25.4% / 19.1% | 0.00249 / 0.00218 | 0.031% | N stays 2; no failure | coherent move-saturated bursts, including the k=1149 spike | SUPPORTED |
| 400×50 | 246 | 1104 | 3 | 7 | **37.3% / 16.0%** | 0.00300 / 0.00200 | 0.033% | one mode-order event; no persistent N/solver failure | one-step quality spikes at 41–51% move saturation | SUPPORTED |

All exit states remain support-to-support connected in both raw and binary
representations. LP flags stay successful, values stay finite, and N does not
change after the first 1% quality entry. The late mechanism is therefore not a
new simple-mode basin, connectivity collapse, or solver artifact. The major
excursions are coherent saturated update bursts within a repeatedly revisited
bimodal basin. Fixed move=0.005 is a material contributor, though modal/order
and active-set events determine when a burst occurs.

The detailed 47 transition records are preserved in
[`raw/baseline_excursions.csv`](raw/baseline_excursions.csv); mesh summaries are
in [`baseline_late_dynamics.csv`](baseline_late_dynamics.csv).

## Frozen causal policy

`native_bimodal_persistence_v1` is:

```text
condition(k) = (N(k) == 2) AND (gap12(k) <= 0.01)

if condition holds for 100 consecutive native evaluations:
    apply the next lower move limit to the current update
    reset the counter
else if condition fails:
    reset the counter
```

There is no objective coefficient, common-evaluator input, future reference,
topology look-ahead or mesh-dependent setting. The same trigger fires for S1
at 245, 206, 203 and 257 across the four meshes.

## Table B — Stabilization candidates

| Profile | Trigger | Move sequence | 240×30 calibration result | Hold-out selection | Final status |
|---|---|---|---|---|---|
| S0 | none | 0.005 | persistent 1% at 219; persistent 0.5% at 294 | control | faithful baseline |
| S1 | N=2 and gap≤1%, 100 consecutive | 0.005→0.0025 | clean; terminal raw-E1 loss 0.082%, binary-E1 loss 0.053%; persistent 1% at 234 | selected | **VALIDATED / selected profile** |
| S2 | same, counter reset after each stage | 0.005→0.0025→0.00125 | clean; terminal raw-E1 loss 0.087%, binary-E1 loss 0.066%; persistent 1% at 234 | selected as bounded stronger hold-out | rejected: 400×50 binary-E1 14.6%, binary-E2/E3 97.3% |
| S3 | same | 0.005→0.0025→0.00125→0.000625 | LP failure at outer 1316; last binary-E1 loss 54.3% | not authorized | **SOLVER_FAILURE** |

S3's final saved small update is not counted as stabilization. Its failure is
recorded above `VALID_STABILIZED_STATE` and it was never run on a hold-out.

## Table C — Cross-resolution validation

Times are measured cumulative optimization-loop telemetry. Development and
hold-out runs were parallelized across single-thread MATLAB processes, so the
absolute inter-run timing comparison is descriptive; the final campaign must
remeasure under its frozen timing protocol. Iteration crossings are unaffected.

| Profile | Mesh | first≤1% | persistent≤1% | first≤0.5% | persistent≤0.5% | persistent gap≤1% | terminal raw E1 loss | terminal binary E1 loss | connected raw/bin | solver | time to persistent 1% | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|---:|---|
| S0 | 160×20 | 186 | 459 | 226 | 843 | 1504 | 0.000% | 0.000% | yes/yes | clean | 21.26 s | control |
| S1 | 160×20 | 186 | **186** | 226 | 730 | **145** | 0.005% | 0.394% | yes/yes | clean | **14.83 s** | PASS |
| S2 | 160×20 | 186 | 186 | 226 | 1203 | 145 | 0.230% | 0.500% | yes/yes | clean | 16.75 s | not selected |
| S0 | 240×30 | 219 | 219 | 294 | 294 | 106 | 0.000% | 0.000% | yes/yes | clean | 24.46 s | control |
| S1 | 240×30 | 234 | 234 | 417 | 417 | 106 | 0.082% | 0.053% | yes/yes | clean | 27.68 s | PASS; calibration delay |
| S2 | 240×30 | 234 | 234 | 542 | 542 | 898 | 0.087% | 0.066% | yes/yes | clean | 27.73 s | not selected |
| S0 | 320×40 | 218 | 1150 | 250 | 1150 | 1548 | 0.000% | 0.000% | yes/yes | clean | 208.01 s | control |
| S1 | 320×40 | 234 | **234** | 299 | 1162 | **103** | −0.084% | 0.857% | yes/yes | clean | **64.22 s** | PASS |
| S2 | 320×40 | 234 | 234 | 299 | **299** | 103 | 0.254% | 0.634% | yes/yes | clean | 63.83 s | not selected |
| S0 | 400×50 | 246 | 1104 | 290 | 1510 | 1511 | 0.000% | 0.000% | yes/yes | clean¹ | 331.03 s | control |
| S1 | 400×50 | 246 | **246** | 321 | 1534 | **961** | 0.441% | 0.486% | yes/yes | clean¹ | **106.37 s** | PASS |
| S2 | 400×50 | 246 | 246 | 321 | 1529 | 271 | 0.373% | **14.599%** | yes/yes | clean¹ | 106.20 s | **FAIL binary quality** |

¹ S0, S1 and S2 share the same one-iteration volume residual 1.295×10⁻⁴ at
k=157, before either stabilization trigger. It returns to machine-zero and is
not caused by continuation. It is disclosed rather than silently erased.

S1 terminal E2/E3 raw losses track E1 and remain within 0.5%. Its worst
terminal binary evaluator loss is 1.085% (E2/E3, 320×40). Terminal binary
turnover from the baseline late topology is 0.62%, 1.03%, 3.55% and 4.65%.
Every S1 terminal state has N=2 and gap12 between 0.082% and 0.258%.

## Table D — Baseline versus selected S1

Positive reductions mean S1 is earlier/faster. Spectral difference is signed
terminal raw-E1 loss relative to baseline k=1600. Topology difference is
density RMS / exact-count binary turnover relative to the baseline late state.

| Mesh | baseline persistent 1% | S1 persistent 1% | iteration reduction | measured time reduction | spectral difference | topology difference |
|---|---:|---:|---:|---:|---:|---:|
| 160×20 | 459 | 186 | **273** | +6.44 s | +0.005% loss | 0.0357 / 0.62% |
| 240×30 | 219 | 234 | −15 | −3.22 s | +0.082% loss | 0.0518 / 1.03% |
| 320×40 | 1150 | 234 | **916** | +143.79 s | −0.084% loss | 0.1319 / 3.55% |
| 400×50 | 1104 | 246 | **858** | +224.66 s | +0.441% loss | 0.1702 / 4.65% |

For the stricter 0.5% band, S1 changes persistent entry by +113, −123, −12
and −24 iterations. Stabilization is validated for the previously declared 1%
practical-quality target, not claimed to dominate at every stricter band.

The selected state is in the same intended **bimodal, connected, high-common-
quality basin**, but it is not the identical late point. Native ω1 stays within
0.5%; ω3 differs by several percent on the larger meshes. The audit therefore
does not claim byte-identical reproduction after the deliberate policy action.

## Cost and stopping semantics

S1 measured total loop times to k=1600 are 85.60, 157.96, 335.50 and 539.33 s.
Eigensolve shares are 68.4%, 69.7%, 68.6% and 66.8%. Total work is slightly
more expensive than S0 at the same cap because the altered LP path changes
per-iteration cost; its value is much earlier *persistent useful quality* on
the unstable meshes.

No simple native stopping rule emerges. At k=1600, `max|Δrho|` remains exactly
the stabilized move 0.0025 and RMS updates remain about 0.0010–0.0013 on the
hold-outs. The profile therefore uses:

```text
FIXED_TOTAL_OUTER_WORK = 1600
stabilization trigger != convergence
endpoint != native convergence
```

This is deliberately different from a tuned convergence detector.

## Figures

- [raw-E1 baseline vs S1](figures/fig01_raw_E1_loss.png)
- [gap12 baseline vs S1](figures/fig02_gap12.png)
- [multiplicity N](figures/fig03_N.png)
- [move limit](figures/fig04_move.png)
- [density-update RMS](figures/fig05_dRms.png)
- [move-bound fraction](figures/fig06_move_bound_fraction.png)
- [binary topology turnover](figures/fig07_binary_topology_turnover.png)
- [time to persistent quality](figures/fig08_time_to_persistent_quality.png)
- [topology snapshots around entry/departure](figures/fig09_topology_snapshots.png)

The plots show that S1 removes the large raw-E1 spikes at 320×40 and 400×50,
but binary topology can still pass through immature intermediate states. That
is why terminal and meaningful checkpoint binary evaluation remains a required
campaign output.

## Table E — Final method readiness

| Method | Profile ID | Termination/work semantics | Cross-resolution validated? | Remaining blocker |
|---|---|---|:---:|---|
| Olhoff | `olhoff_fig3a_s1_native_bimodal_p100_move005_0025_fixed1600_v1` | causal one-stage stabilization + fixed 1600 outer work; **not convergence** | yes, four meshes | none; final campaign must use this profile rather than the legacy S0/rmin2 dispatcher default |
| Yuksel | `yuksel_practical_move01_tol001` | native two-stage practical profile; stage counts/times preserved | yes (`ROBUST`) | none |
| Proposed | `proposed_practical_move02_tol001` | native practical max-density-change profile | yes (`ROBUST`) | none; authoritative stricter profile remains separately resolution-sensitive |

The exact campaign manifest is
[`final_campaign_profile.json`](final_campaign_profile.json). It supersedes the
legacy default Olhoff row for the ultimate campaign; the existing
`performance_comparison.m` must not silently dispatch its old S0/rmin2 profile
under the new profile ID.

## Table F — Nine-resolution campaign gate

The preflight generated configurations and memory/schema estimates only. It did
not run optimization at 480×60 or above.

| Requirement | Result | Evidence |
|---|:---:|---|
| frozen reproduction manifest | PASS | 61/61 match |
| S0 mirror identity | PASS | 20-update bit identity |
| simple causal policy frozen before hold-out | PASS | preregistration + candidate freeze hashes |
| cross-resolution S1 quality/modal/topology validity | PASS | Tables C–D; CSV evidence |
| S1 solver health | PASS | zero LP/nonfinite failures; inherited k=157 volume event disclosed |
| explicit work semantics | PASS | fixed 1600, not convergence |
| Yuksel selected profile ready | PASS | frozen `ROBUST` hold-out result |
| Proposed selected practical profile ready | PASS | frozen `ROBUST` hold-out result |
| all nine configurations generated structurally | PASS | [`campaign_preflight.csv`](campaign_preflight.csv) |
| memory sanity at 800×100 | PASS | conservative 1.46 GB estimate on 64 GB host |
| result schema and timing telemetry | PASS | loop/eigen/gradient/inner/move/status fields present |
| common E1/E2/E3 raw/binary evaluator compatibility | PASS | arbitrary mesh dimensions; unchanged evaluator |
| censor/failure semantics | PASS | solver failure precedes valid fixed-work state; failed rows excluded from fits |
| forbidden high-resolution optimization solves absent | PASS | no 480×60–800×100 run artifact |
| final profile manifest frozen | PASS | selected and campaign profile JSON artifacts |

**FINAL NINE-RESOLUTION CAMPAIGN: GO**

## Answers to the fifteen final questions

1. **Why do trajectories leave/revisit good states?** Coherent, saturated LP
   update bursts and associated active/modal-order events continue inside the
   mature basin; they are not solver or connectivity failures.
2. **Is move=0.005 material?** Yes. Exit states have substantially more
   move-bound activity, most clearly at 320×40 and 400×50.
3. **Does causal reduction stabilize the basin?** S1 does at the 1% practical
   quality level and materially advances persistent modal quality.
4. **Without changing the basin?** It preserves the intended bimodal,
   connected, high-common-quality basin, but changes the path and exact late
   design; exact identity is not claimed.
5. **Does it generalize?** Yes, unchanged across calibration and all three
   hold-outs.
6. **Persistent bimodality?** Yes. S1 persistent gap≤1% occurs by
   145/106/103/961 and all terminal states have N=2.
7. **Raw and binary topology quality?** Yes for S1 at the fixed endpoint; S2
   fails binary quality and is rejected.
8. **Connectivity?** Preserved support-to-support in raw and binary fields.
9. **Solver health?** S1 has zero LP and nonfinite failures. The inherited
   pretrigger 400×50 volume event is disclosed.
10. **Time reduction to persistent 1%?** Descriptively +6.44, −3.22, +143.79
    and +224.66 s; campaign timing must be remeasured under the final protocol.
11. **Time reduction to persistent 0.5%?** It is not robust: −5.89, −13.76,
    −42.18 and −72.95 s in these measured runs despite one iteration-count gain.
12. **Simple native stop after stabilization?** No. The move-bound statistic
    remains saturated; fixed-work semantics are required.
13. **Exact Olhoff benchmark semantics?** Selected S1 profile, causal
    N=2/gap≤1%/100 trigger, move 0.005→0.0025, fixed total k=1600, explicitly
    not convergence.
14. **Are Yuksel and Proposed ready?** Yes, using their frozen robust practical
    profiles listed in Table E.
15. **Can the nine-resolution campaign launch?** Yes, using the frozen manifest
    and status/fit guardrails in this report.

## Evidence map

The required deliverables are:

- [`baseline_late_dynamics.csv`](baseline_late_dynamics.csv)
- [`stabilization_runs.csv`](stabilization_runs.csv)
- [`persistent_quality.csv`](persistent_quality.csv)
- [`modal_stability.csv`](modal_stability.csv)
- [`topology_stability.csv`](topology_stability.csv)
- [`solver_health.csv`](solver_health.csv)
- [`selected_profile.json`](selected_profile.json)
- [`campaign_gate.json`](campaign_gate.json)
- [`provenance.json`](provenance.json)

Raw MAT/CSV evidence and all analysis scripts are retained below this directory.
No Yuksel or Proposed source/configuration was modified. No 480×60, 560×70,
640×80, 720×90 or 800×100 optimization was run.

## Final verdict

Frozen profiles/work semantics:

- Olhoff — `olhoff_fig3a_s1_native_bimodal_p100_move005_0025_fixed1600_v1`:
  fixed 1600 outer work after causal stabilization; not native convergence.
- Yuksel — `yuksel_practical_move01_tol001`: frozen native two-stage practical
  semantics.
- Proposed — `proposed_practical_move02_tol001`: frozen native practical
  semantics.

The next authorized experiment is the full 160×20, 240×30, 320×40, 400×50,
480×60, 560×70, 640×80, 720×90 and 800×100 performance/scaling campaign.

**PRACTICAL OLHOFF STABILIZATION VALIDATED**

**FINAL NINE-RESOLUTION CAMPAIGN: GO**
