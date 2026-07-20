# A4 Recovery Phase 2 — Validation Record

| Validator | Result | Executed evidence |
|---|---|---|
| V-P2-1 — non-perturbation | **PASS (tiny integration)** | 40×5, four-iteration frozen arm with diagnostics on/off: bit-identical trajectory, endpoint, and topology. Production-scale instance remains V-P2-2. |
| V-P2-2 — `N=inf` bit identity | **PENDING PRODUCTION RERUN** | Driver stop gate implemented against the preserved exact endpoint, final change, mode data, and topology. |
| V-P2-3 — iteration-25/30 recovery | **PASS** | Iteration 25: rungs 20/40/80/160, `m_final=160`, index 49, MAC 0.9775288450, E-1. Iteration 30: rungs 20/40/80, `m_final=80`, index 37, MAC 0.9663501395, E-1. |
| V-P2-4 — screening symmetry | **PASS** | Identical design/reference fixture produced identical rung sequence, selected index, and decisions independent of arm metadata. |
| V-P2-5 — ladder determinism | **PASS** | Repeated fixture search reproduced complete final candidate records exactly. |
| V-P2-6 — finite-arm replay | **PENDING PRODUCTION RERUN** | Full finite-arm replay is implemented in the production driver. |
| V-P2-7 — classifier fixtures | **PASS** | E-0, E-1, E-2a, E-2b, E-3, E-4, E-5, deep-index E-1, and tie fixtures passed. |
| V-P2-8 — factor drift/hash | **PASS** | Base hash reproduced `fnv1a32_c141e407`; a distinct config produced a distinct hash. |
| V-P2-9 — Phase 1 regression | **PASS** | `test_a4_phase1`: 10/10. |

Additional executed validation: `test_a4_pipeline` 28/28; `test_a4_refresh` 22/22; `test_a4_classifier` 17/17; MATLAB Code Analyzer reported no parse errors; `git diff --check` passed.

Reproducibility and documentation checklist items that depend on actual production outputs—R-1 through R-5 and D-3 through D-5—remain pending the complete A4 rerun.
