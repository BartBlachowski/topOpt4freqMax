# WP15 — Binding qualification criteria, determined

Storage of NEW Olhoff trajectory density fields as IEEE single precision.

| # | Criterion | Result | Evidence |
|---|---|---|---|
| Q1 | single conversion is observational/storage-only, no feedback into optimization | **MET** | `PRECISION_PATH.md`; olhoffOptStabilized.m:10,21,65,88 |
| Q2 | genuine paired evidence without protected-source modification or fork | **MET** | 45 checkpoint-limited reruns of the unmodified source; `PREFIX_IDENTITY.csv` (45/45 bit-identical, 45/45 cast identity) |
| Q3 | every binary-field difference propagated through the topology analysis and explained | **MET** | `BINARY_DIFFERENCE_ANALYSIS.csv`; 1 difference in 45 pairs, 2 elements, both inside the 12-element single tie group at the cutoff; 0 unexplained |
| Q4 | frozen topology PASS/FAIL identical | **MET** | `TOPOLOGY_DECISION_EQUIVALENCE.csv`; volume_pass, topology_pass and hard_gate_pass identical at 45/45 |
| Q5 | explicit numerical E1/E2/E3 bounds established | **MET (bounds established; they are large)** | `EVALUATOR_ERROR_SUMMARY.csv` |
| Q6 | frozen spectral/quality PASS/FAIL classifications identical | **FAILED** | `QUALITY_DECISION_EQUIVALENCE.csv`; 3 / 3 / 29 states flip at q = 0.98 / 0.99 / 0.995 |
| Q7 | reference semantics exercised and `b_ref` identical | **EXERCISED, FAILED** | `REFERENCE_EQUIVALENCE.csv`; b_ref double 2200 vs single 2100 |
| Q8 | `B_meas` identical | **MET** | `B_MEAS_EQUIVALENCE.csv`; both 3200 — but only because Olhoff B0 = B_ref = 3200 saturates the formula, so B_meas is insensitive to the b_ref difference |
| Q9 | persistence exercised and `k_enter` identical | **EXERCISED, FAILED** | `PERSISTENCE_EQUIVALENCE.csv`; 233/232, 315/309, 609/524 |
| Q10 | `k_cert` identical | **FAILED** | 332/331, 414/408, 708/623 |
| Q11 | final status identical | **MET on this case** | both trajectories reach PASS at all three q levels within the horizon; the status *precedence* outcome coincides even though the certification locations differ |
| Q12 | production-scale risk contains no untested regime capable of changing decisions | **FAILED** | `PRODUCTION_SCALE_RISK_ANALYSIS.md`; production carries 3x–65x more branch-ambiguous elements per state than the worst qualification state |
| Q13 | checkpoint/prefix identity passes under the recorded environment | **MET** | `PREFIX_IDENTITY.csv`, `ENVIRONMENT_PROVENANCE.json` |

Exact binary-field identity is reported separately and is not a Q criterion:
**STRICT_BINARY_IDENTITY = FAIL** (44 of 45 pairs identical).
**FROZEN_DECISION_EQUIVALENCE = FAIL** on Q6, Q7, Q9, Q10, Q12.

Six criteria are unmet. Under WP16 any one of Q6, Q7, Q9, Q10 is sufficient for
NOT QUALIFIED.
