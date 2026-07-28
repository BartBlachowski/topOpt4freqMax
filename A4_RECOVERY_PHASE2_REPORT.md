# A4 Recovery Phase 2 Report

- Specification: `A4_RECOVERY_PHASE2_SPECIFICATION.md`
- Base configuration hash: `fnv1a32_c141e407`
- Immutable frozen baseline: `/Users/piotrek/Programming/topOpt4freqMax/examples/Revision_v1/reference/a4/a4_topology_Ninf.csv`
- Immutable frozen baseline SHA-256: `9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806`
- Commit: `3542a2d`
- Run verdict: **COMPLETE**

## Per-arm measurement status

- N=inf: ACCEPTED_WITH_WARNING; events=25; max window=320; max index=49; deferrals=0/0; warnings=W-2,W-5.
- N=50: ACCEPTED_WITH_WARNING; events=23; max window=320; max index=49; deferrals=0/10; warnings=W-2,W-5.
- N=10: ACCEPTED_WITH_WARNING; events=62; max window=320; max index=40; deferrals=50/53; warnings=W-1,W-2,W-5.
- N=5: ACCEPTED_WITH_WARNING; events=239; max window=320; max index=20; deferrals=233/234; warnings=W-1,W-2,W-5.
- N=1: ACCEPTED_WITH_WARNING; events=1040; max window=320; max index=19; deferrals=1038/1040; warnings=W-1,W-2,W-5.
- N=50 pre-Phase-2 endpoint: 159.60117294919709; Phase-2 endpoint: 159.60129669888926; difference: +0.00012374969216466525 rad/s.

## Scope limitation

Phase 2 emits corrected measurements only. M-1, M-2, M-3, M-7 and M-9 remain open as specified in §7.6.
No campaign-level H0/H1 decision or manuscript claim is emitted.

## Screening-decision reconstruction

- N=inf, iteration 20, event 8: 1/160 candidates admissible; outcome SELECTED; selected index 43 from m_final=160; classes=E-1.
- N=inf, iteration 1, event 1: 0/320 candidates admissible; outcome REFERENCE_UNAVAILABLE; selected index 0 from m_final=320; classes=E-2a,E-4.
- N=inf, iteration 2, event 2: 0/320 candidates admissible; outcome REFERENCE_UNAVAILABLE; selected index 0 from m_final=320; classes=E-2a,E-4.

## Section 11 evidence

The following checklist is generated from run evidence; implementation fixtures do not substitute for production gates.
- [x] **S-1**
- [x] **S-2**
- [x] **S-3**
- [x] **S-4**
- [x] **S-5**
- [x] **S-6**
- [x] **S-7**
- [x] **S-8**
- [x] **S-9**
- [x] **I-1**
- [x] **I-2**
- [x] **I-3**
- [x] **I-4**
- [x] **I-5**
- [x] **I-6**
- [x] **I-7**
- [x] **I-8**
- [x] **I-9**
- [x] **V-P2-1**
- [x] **V-P2-2**
- [x] **V-P2-3**
- [x] **V-P2-4**
- [x] **V-P2-5**
- [x] **V-P2-6**
- [x] **V-P2-7**
- [x] **V-P2-8**
- [x] **V-P2-9**
- [x] **R-1**
- [x] **R-2**
- [x] **R-3**
- [x] **R-4**
- [x] **R-5**
- [x] **D-1**
- [x] **D-2**
- [x] **D-3**
- [x] **D-4**
- [x] **D-5**
- [x] **D-6**
