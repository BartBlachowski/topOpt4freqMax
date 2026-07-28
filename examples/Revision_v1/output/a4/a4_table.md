# Table A4-1 — Eigenpair-refresh study

Spec: `A4_RECOVERY_PHASE2_SPECIFICATION`. Base config hash: `fnv1a32_c141e407`. Commit: `3542a2d`.
Pre-declared equivalence margin delta = 5.0%.

`Δω₁ vs N=∞` is blank for UNAVAILABLE or REJECTED arms.

| N | ω₁_tracked | ω₁_min | ω₁_thresh | MAC | j* | iters | conv | scheduled/effective | grayness | feasibility | omitted ratio | status | warnings | Δω₁ vs N=∞ |
|---|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|---|---|---:|
| inf | 159.5656 | 159.5656 | 162.4677 | 0.9996 | 1 | 2000 | no | 0/0 | 0.0969 | 9.186e-05 | null | ACCEPTED_WITH_WARNING | W-2,W-5 | +0.00% |
| 50 | 159.6013 | 159.6013 | 162.4788 | 0.9996 | 1 | 540 | yes | 10/10 | 0.0966 | 3.710e-05 | null | ACCEPTED_WITH_WARNING | W-2,W-5 | +0.02% |
| 10 | 159.1229 | 98.5535 | 162.9134 | 0.9948 | 7 | 536 | yes | 53/3 | 0.1087 | 9.634e-05 | null | ACCEPTED_WITH_WARNING | W-1,W-2,W-5 | -0.28% |
| 5 | 158.6727 | 158.6727 | 162.7406 | 0.9994 | 1 | 1173 | yes | 234/1 | 0.1111 | 1.294e-04 | null | ACCEPTED_WITH_WARNING | W-1,W-2,W-5 | -0.56% |
| 1 | 157.6329 | 157.6329 | 161.4558 | 0.9997 | 1 | 1040 | yes | 1040/2 | 0.1219 | 7.958e-05 | null | ACCEPTED_WITH_WARNING | W-1,W-2,W-5 | -1.21% |

**Run verdict: COMPLETE**

**Scientific decision: not emitted in Phase 2 (§9.5).**

_Wall-clock time is recorded for provenance only and may not appear in any
performance claim (spec §4.5)._
