# Table A4-1 — Eigenpair-refresh study

Spec: `A4_RECOVERY_PHASE2_SPECIFICATION`. Base config hash: `fnv1a32_c141e407`. Commit: `03729b4`.
Pre-declared equivalence margin delta = 5.0%.

`Δω₁ vs N=∞` is blank for UNAVAILABLE or REJECTED arms.

| N | ω₁_tracked | ω₁_min | ω₁_thresh | MAC | j* | iters | conv | scheduled/effective | grayness | feasibility | omitted ratio | status | warnings | Δω₁ vs N=∞ |
|---|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|---|---|---:|
| inf | 159.5656 | 159.5656 | 162.4677 | 0.9996 | 1 | 2000 | no | 0/0 | 0.0969 | 9.186e-05 | null | ACCEPTED_WITH_WARNING | W-2,W-5 | +0.00% |

**Run verdict: HALTED**

**Scientific decision: not emitted in Phase 2 (§9.5).**

_Wall-clock time is recorded for provenance only and may not appear in any
performance claim (spec §4.5)._
