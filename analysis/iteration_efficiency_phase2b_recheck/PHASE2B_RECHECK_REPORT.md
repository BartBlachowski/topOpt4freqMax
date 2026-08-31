# Phase 2B-R — Olhoff single-precision trajectory representation qualification

Date: 2026-08-30
Branch `benchmark-methodology-r2`, HEAD `632e9b01811845709de33f93051fd853373ed5e1`
Governing specification: `PHASE2B_V3_SPEC.md` (v2 + ten repository-grounded patches)
Classification: precision qualification only. No production campaign, no methodology
change, no protected-source modification, no authorization token.

## Outcome

Single-precision storage of Olhoff trajectory densities is **not observationally
lossless** under the frozen methodology. It changes `b_ref`, `k_enter` and `k_cert` at
every frozen q level, and flips per-state quality classifications on genuine paired
evidence. The mechanism is identified, reproducible and systematic.

Reported separately per the binding ruling:

- **STRICT_BINARY_IDENTITY = FAIL** (44 of 45 genuine pairs identical; the one difference
  is fully explained and does not propagate).
- **FROZEN_DECISION_EQUIVALENCE = FAIL** (Q6, Q7, Q9, Q10, Q12).

Only the second governs. It fails.

## 1. Root cause

The frozen E2/E3 mass law is **discontinuous**:

    g(x) = x^6  for x <= 0.1,   g(x) = x  otherwise
    g(0.1^-) = 1e-6      g(0.1^+) = 0.1      ratio 1e5

Olhoff move-limit arithmetic (`rho = min(1,max(rhomin, rho+drho))`, move 0.005 then
0.0025 from rho0 = 0.5) drives densities onto the accumulated value
`0.099999999999999644729` and parks many elements there simultaneously. Since

    single(0.099999999999999644729) = 0.10000000149011611938  >  0.1

every such element silently crosses to the linear branch under single storage and its
mass rises by five orders of magnitude. Added mass lowers omega_1, so single storage
**systematically under-reports** E2 and E3.

This is not a rounding-magnitude effect. A density perturbation of 2.9e-8 produces a
frequency change of 2.27e-2 because the evaluator is discontinuous exactly where the
optimizer stalls. E1, whose mass law is linear, is unaffected: its maximum relative error
is 1.18e-10.

The prior attempt's unexplained ~2.67e-2 E2/E3 tail is this mechanism. It was **not** a
state-pairing artifact: the pre-existing `+ie2b` code already used the correct 2-D
`rho_snapshots(:,end)` and `(:,k+1)` accessors, so Patches 1 and 2 corrected the v2 prompt
text rather than that code. The tail was also inflated in presentation: it came from one
stalled 24x4 state repeated across many checkpoints, not from many independent states.

## 2. Evidence base

- **8 frozen production final-state pairs** (`res.rho` double vs `res.rho_snapshots(:,end)`
  single), available with no new runs. All eight verified per Patch 8: 2-D dimensions,
  `nDone+1` column count, and `isequal(rho_snapshots(:,end), single(res.rho))`. Statuses:
  four CAP_HIT, three SOLVER_FAILURE (480x60, 560x70, 640x80). 800x100 remains
  `RUN_ERROR / N/A / UNVERIFIABLE_AT_PRESENT`.
- **45 genuine paired intermediate states** at 96x12 from checkpoint-limited reruns of the
  unmodified protected source. Prefix bit-identity 45/45, cast identity 45/45.
- **One full 96x12 trajectory at horizon 3200**, on which the frozen reference,
  measurement-budget and persistence engines were run end to end.

Qualification meshes obey Patch 4 (even nely, 8:1 aspect). Per Patch 5, a_sig at 96x12 is
1.44 elements against 4–100 in production, so the topology gate applied here is
STRICTER_THAN_PRODUCTION.

### Identification of the double trajectory

For every at-risk element the double is either at or below 0.1 (x^6 branch) or above it
(linear branch); monotone rounding makes `single(value) == single(0.1)` the exact and
complete at-risk test. Forcing all at-risk elements onto the x^6 branch yields an upper
bracket Q_hi, and leaving them as stored yields Q_lo = Q_single; omega_1 is monotone
decreasing in element mass, so the true double satisfies Q_double in [Q_lo, Q_hi].

Measured on the 45 genuine pairs: **255 of 255 at-risk elements had a double value at or
below 0.1** — a 100% flip rate — and Q_double agreed with Q_hi to 6.9e-10 relative. The
upper bracket therefore *is* the double trajectory, verified rather than assumed, which
permitted the full 3200-state end-to-end comparison without a prohibitive O(B^2) sweep.

## 3. Numerical results

Evaluator error over the full 3200-state trajectory (`EVALUATOR_ERROR_SUMMARY.csv`):

| evaluator | median rel | p95 rel | max rel | max abs |
|---|---|---|---|---|
| E1 | 0 | 4.69e-11 | **1.18e-10** | 1.50e-08 |
| E2 | 0 | 3.08e-03 | **2.27e-02** | 2.95 |
| E3 | 0 | 3.08e-03 | **2.27e-02** | 2.95 |

Stratified by at-risk element count (`EVALUATOR_ERROR_STRATIFIED.csv`), which confirms the
mechanism is the sole source of error:

| stratum | states | max rel E2 |
|---|---|---|
| no at-risk elements | 1640 | **0** |
| 1–16 | 1188 | 2.69e-03 |
| 17–64 | 371 | 7.93e-03 |
| >= 65 | 1 | 2.27e-02 |

**Documented bound.** No bound below 2.3e-2 can be claimed for E2/E3. Against the frozen
acceptance bands — 2.0% at q = 0.98, 1.0% at q = 0.99, 0.5% at q = 0.995 — a 2.27e-2
perturbation exceeds all three. The robust-quality perturbation, after the reference
shift partially cancels, reaches 2.374e-3, which still exceeds the minimum observed
classification margins of 8.67e-5, 2.59e-5 and 5.77e-6.

## 4. Decision equivalence

| decision | double | single | identical |
|---|---|---|---|
| reference status | PASS | PASS | yes |
| **b_ref** | **2200** | **2100** | **NO** |
| B_meas | 3200 | 3200 | yes (only because B0 = B_ref = 3200 saturates the formula) |
| certification_tail_truncated | false | false | yes |
| **k_enter** q=0.980 | **233** | **232** | **NO** |
| **k_enter** q=0.990 | **315** | **309** | **NO** |
| **k_enter** q=0.995 | **609** | **524** | **NO** |
| **k_cert** q=0.980 | **332** | **331** | **NO** |
| **k_cert** q=0.990 | **414** | **408** | **NO** |
| **k_cert** q=0.995 | **708** | **623** | **NO** |
| per-state acceptance flips | — | — | 3 / 3 / 29 states |
| volume_pass, topology_pass, hard_gate_pass | — | — | yes, 45/45 |
| final status | PASS | PASS | yes |

The q = 0.995 certification location moves by **85 iterations**, 13.6% of its own value.
That is the quantity the iteration-efficiency study exists to measure.

Independently, on genuine paired states alone and holding Q_ref fixed at the
single-derived value to isolate the state-level effect, acceptance flips at k = 231
(q = 0.98) and at k = 433, 435, 441, 443, 447, 449, 501, 523 (q = 0.995).

## 5. Binary projection and topology

Per Patch 6, single rounding is monotone and cannot invert an ordering; it can only
collapse distinct doubles onto one float32. A binary difference therefore requires a
newly-created tie spanning the cutoff rank, resolved by the ascending-global-index
tiebreak. One of 45 pairs differed (k = 100, 2 elements): both differing elements lay
inside the 12-element single tie group at cutoff rank 576, and the double gap across the
cutoff was 0. Zero unexplained differences. The difference did not change volume_pass,
topology_pass or hard_gate_pass. Topology decision equivalence therefore **passes**, and
the binary difference is correctly diagnostic under the binding ruling.

## 6. Production-scale coverage

Between 38.8% and 78.3% of stored production states contain branch-ambiguous elements, up
to 12,477 in a single 720x90 state, against at most 192 in the worst qualification state.
Cutoff-tie states run 18.7%–88.0% of production states against 5.0% here. Because the
measured error grows monotonically with at-risk count, production error is expected to
**exceed** the 2.27e-2 measured here. The qualification evidence is not conservative with
respect to the failing mechanism. See `PRODUCTION_SCALE_RISK_ANALYSIS.md`.

## 7. Findings about the Phase-2A gate

`+ie2a/production_preflight.m:18` establishes only that the observer string is present in
`tools/Matlab/topopt_history_record.m`. Grep of the callers shows
`analysis/ourApproach/Matlab/topopt_freq.m`,
`analysis/YukselApproach/Matlab/top99neo_inertial_freq.m` and
`analysis/OlhoffApproach/Matlab/topFreqOptimization_MMA.m` call that recorder, while
`analysis/olhoff_stabilization_audit/olhoffOptStabilized.m` — the runner the frozen Olhoff
profile actually binds — does **not**. **PHASE-2A OLHOFF OBSERVABILITY CHECK WAS
INSUFFICIENT.** This is an implementation-gate finding, not a statement about the
numerical method. Phase-2A's own blocker was nevertheless correctly raised and correctly
held.

## 8. Repository integrity

All six protected numerical sources, three profile sources and six audit/normative records
re-hash to their Phase-2A recorded values; the contract matches its recorded SHA-256. No
protected source, frozen methodology, audit directory, frozen trajectory or Phase-2A
validation artifact was modified. The prior Phase-2B directory is preserved byte-for-byte;
its artifacts are classified in `EXISTING_EVIDENCE_DISPOSITION.csv` and none were
overwritten. Environment recorded in `ENVIRONMENT_PROVENANCE.json` (MATLAB R2025b, MACA64,
runner pinned to one thread).

## 9. Preflight and authorization

Qualification did not pass, so per Patch 9 and WP17:

- no qualification artifact was created at
  `analysis/iteration_efficiency_phase2a/validation_outputs/olhoff_new_trajectory_precision_qualification.json`;
- `production_preflight.m` was **not** edited; its SHA-256 is unchanged;
- `olhoff_lossless_trajectory` therefore continues to fail closed, which is correct;
- the production authorization token was not set.

## Required final summary

1. **Single cast location.** `olhoffOptStabilized.m:21` (`snapshots=NaN(NE,cfg.maxOuter+1,'single'); snapshots(:,1)=single(rho)`) and `:65` (`snapshots(:,outer+1)=single(rho)`).
2. **Feedback into optimization?** No. Storage-only; `snapshots` is write-only in the loop.
3. **Res.rho double?** Yes. 4. **rho_snapshots single?** Yes. 2-D, `NE x (nDone+1)`.
5. **Existing genuine paired endpoint states.** 8 (one per available frozen mesh).
6. **Historical Mo9 status.** Valid only for final-state component counts and detached areas; every spectral quantity is a NEW quantity whose repeat check is NOT PERFORMABLE. NOT EXTENDABLE.
7. **New-trajectory qualification status.** NOT QUALIFIED.
8. **Phase-2A Olhoff observer coverage valid?** No — insufficient; the stabilized runner never calls the instrumented recorder.
9. **Checkpoint/prefix identity.** PASS, 45/45 bit-identical, 45/45 cast identity.
10. **Genuine paired intermediate states.** 45, plus 8 frozen endpoint pairs.
11. **Density roundtrip error.** max 2.861e-08, median 0 (most states are float32-exact).
12. **STRICT_BINARY_IDENTITY.** FAIL — 1 of 45 pairs differs.
13. **Binary differences.** 2 elements of 1152 at k=100; 0 unexplained; 0 changing the gate.
14. **Did any binary difference change topology PASS/FAIL?** No. 45/45 identical on volume_pass, topology_pass and hard_gate_pass.
15. **Max relative E1 error.** 1.18e-10.
16. **Max relative E2 error.** 2.27e-02.
17. **Max relative E3 error.** 2.27e-02.
18. **Documented E1/E2/E3 bound.** E1 <= 1.2e-10; E2/E3 cannot be bounded below 2.3e-2. Robust-quality perturbation <= 2.374e-3 against minimum margins of 8.67e-5 / 2.59e-5 / 5.77e-6.
19. **Explanation of the ~2.67% tail.** Discontinuous E2/E3 mass law at x = 0.1 crossed by float32 rounding of 0.0999999999999996447; 100% of at-risk elements flip. Not a pairing artifact.
20. **Quality-classification flip?** Yes — 3, 3 and 29 states at q = 0.98, 0.99, 0.995.
21. **b_ref identity.** FAIL — 2200 vs 2100.
22. **B_meas identity.** PASS — 3200 both, but insensitive by construction (B0 = B_ref = 3200).
23. **Persistence exercised?** Yes, under frozen P = 100 with b_ref established at 2100/2200.
24. **k_enter identity.** FAIL at all three q levels.
25. **k_cert identity.** FAIL at all three q levels; q=0.995 moves by 85 iterations.
26. **Final-status identity.** PASS on this case.
27. **Production-scale risk coverage.** NOT COVERED — production carries 3x–65x more branch-ambiguous elements per state and 3.7x–17.6x the cutoff-tie rate.
28. **Unresolved extrapolation limitation.** Whether historical production at-risk elements sat below 0.1 is unrecoverable; the qualification's 100% flip rate suggests they did, but this cannot be proven retrospectively.
29. **Protected numerical sources unchanged?** Yes — all hashes verified before and after.
30. **May olhoff_lossless_trajectory pass preflight?** No. It must continue to fail closed.
31. **Production technically ready for independent final review?** No.
32. **Production authorization still absent?** Yes — token not set, campaign not started.
33. **Snapshot indexing verified by assertion?** Yes — ndims, column count, element count and `isequal(rho_snapshots(:,end), single(res.rho))` asserted on every run used.
34. **a_sig per qualification mesh.** 96x12 = 1.44 elements, STRICTER_THAN_PRODUCTION (production 4–100). See `A_SIG_REGIME.csv`.

---

# OLHOFF SINGLE-PRECISION TRAJECTORY NOT QUALIFIED
