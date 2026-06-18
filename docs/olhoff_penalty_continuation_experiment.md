# Du & Olhoff (2007) CC beam — SIMP penalty-continuation experiment

**Date:** 2026-06-18
**Question:** Is SIMP penalty continuation (p: 1 → 3, stated in the paper as the
normal procedure) the missing mechanism that kept Du & Olhoff's optimization in
the *connected* basin and away from the disconnected two-block topology that the
fixed-p=3 `OlhoffApproachExact` implementation finds for the clamped-clamped beam?

**Verdict: NO — refuted.** Penalty continuation does not restore connectivity.
Best-supported interpretation: **C** — the disconnected topology is preferred
across the entire p = 1 → 3 path, so the published connected Fig. 3c design is not
the global optimum of the formulation as implemented.

---

## Setup (one variable changed)

The only difference between control and treatment is the penalty schedule.

| Setting | Value (both runs) |
|---|---|
| BC / mesh / volfrac | CC, 40×5, 0.5 |
| mass_mode | du2007_c1 (paper Eq. 4b) |
| sensitivity filter | rmin_elem = 2.5 |
| multiplicity | mult_tol = 0.01 (N=2 branch active) |
| step control | move_lim = 0.2, outer_move = 0.2, alpha = 0.5 |
| acceptance_check | false |
| initial field | uniform ρ = 0.5 |

- **Control:** `penal = 3` fixed (= the existing disconnected-design baseline).
- **Treatment:** `penal_schedule = repelem(1.0:0.2:3.0, 30)` then held at 3 to
  convergence (the only change).

The step-control settings (move 0.2, α 0.5) are essential: the committed
`run_clamped_clamped_exact.m` defaults (move_lim=Inf, α=1) instead collapse to a
singular mechanism (ω₁≈0.016). The step-control config is what reproduces the
disconnected near-paper design and matches `audit_optimizer_nochange.m`.

Code: one inert-by-default field `cfg.penal_schedule` added to
`topopt_freq_exact.m`; driver `run_cc_penalty_continuation.m`. Artifacts in
`results/penalty_continuation/`.

## Results

| | iter | p | ω₁ | ω₂ | ccSolid | volume |
|---|---|---|---|---|---|---|
| Control best | 164 | 3.0 | 641.7 (→790.7 thresh) | 646.9 | 2 | 0.468 |
| Control final | 300 | 3.0 | 239.2 | 303.7 | 4 | 0.499 |
| **Continuation best** | **29** | **1.0** | **1589.9 (→1617.6 thresh)** | 1621.7 | 2 | 0.296 |
| Continuation final | 630 | 3.0 | 274.4 | 356.6 | 2 | 0.500 |

- **ccSolid == 1 in 0 / 300 control iters and 0 / 630 continuation iters.**
  The continuation design is disconnected at iteration 1 and stays disconnected
  through the whole ramp, settling to a stable ccSolid = 2 design at p = 3.
- Topology images confirm genuine two-block disconnection (solid at both clamped
  walls, empty center) — not grey-mask fragmentation — including at p = 1.0.
- Neither design resembles the connected end-to-end X-brace of Fig. 3c.

## Decisive detail

At **p = 1.0** (linear stiffness, no grey penalty), the single highest ω₁ of the
whole experiment (≈1590) is achieved by the *most* disconnected, lowest-volume
design (two short stocky stubs, 30% volume). For clamped-clamped frequency
maximization, short stiff clamped stubs beat any spanning member, and this is true
even with penalization off. Penalty continuation cannot gate a basin that is
preferred at p = 1.

Enabling ingredients (orthogonal to p): clamped BCs (each stub has no rigid-body
mode → high cantilever frequency) + the paper's own localized-mode mass
suppression (du2007_c1) making the void inert.

## Caveat

The optimization is unstable with this configuration: ω₁ oscillates wildly
(73 → 1590), designs stay grey (final grey fraction ≈ 0.76), and neither run
reaches a clean 0–1 optimum. So this does not prove the disconnected design is the
rigorous global optimum. What is robust: penalty continuation is eliminated as the
connectivity-preserving mechanism. The instability points the remaining
investigation toward the eigenvalue-sensitivity / multiplicity handling and the
optimizer path — not toward fixed-p.
