# Table A4-1 — Eigenpair-refresh study

Spec: `A4_SPECIFICATION_V3`. Base config hash: `fnv1a32_ffffffff`. Commit: `2c945de`.
Pre-declared equivalence margin delta = 5.0%.

`Δω₁ vs N=∞` is populated **only for clean arms** (Class B, or Class C/B1–B2 —
spec §7.6). It is left BLANK for B3/B4 and REJECTED arms — a contaminated or
unstable arm is disqualified as an accuracy reference and its endpoint must not
be read as one.

| N | ω₁_tracked | ω₁_min | ω₁_thresh | MAC | j* | iters | conv | refreshes | eigensolves | grayness | comps | class | Δω₁ vs N=∞ |
|---|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|---|---:|
| inf | 159.5656 | 159.5656 | 26.5193 | 0.9996 | 1 | 2000 | no | 0 | 2 | 0.0969 | 1 | ACCEPTED_WITH_BREAKDOWN/B4 |  |
| 50 | 159.6012 | 159.6012 | 26.4802 | 0.9996 | 1 | 541 | yes | 10 | 12 | 0.0966 | 1 | ACCEPTED |  |
| 10 | NaN | NaN | NaN | NaN | 0 | 0 | no | 0 | 0 | NaN | 0 | ACCEPTED_WITH_BREAKDOWN/B3 |  |
| 5 | NaN | NaN | NaN | NaN | 0 | 0 | no | 0 | 0 | NaN | 0 | ACCEPTED_WITH_BREAKDOWN/B3 |  |
| 1 | NaN | NaN | NaN | NaN | 0 | 0 | no | 0 | 0 | NaN | 0 | ACCEPTED_WITH_BREAKDOWN/B3 |  |

**Decision: INDETERMINATE**

the N=inf arm (the published method) is ACCEPTED_WITH_BREAKDOWN/B4 -- only a Class B arm may serve as the accuracy reference (spec §5.2). Report as an observation about the published method.

_Wall-clock time is recorded for provenance only and may not appear in any
performance claim (spec §4.5)._
