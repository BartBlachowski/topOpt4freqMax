# Experiment — Independent FE verification of paper initial frequencies

**Script:** [`implementation/fe_verify.py`](../implementation/fe_verify.py) (clean-room; reuses no `OlhoffApproachExact` code)
**Date run:** Phase 4. Env: this repo's Python (numpy 2.4.2, scipy 1.17.0).
**Goal:** the paper's *own* first validation gate — does the spec-derived FE reproduce the published initial fundamental frequencies of the uniform `ρ=0.5` beam (SS 68.7, CS 104.1, CC 146.1)? At `ρ=0.5` all mass models (eq. 2/4/4a/4b) coincide (`ρ>0.1`), so this isolates FE + BC + mesh + SIMP power.

## Result (ω₁ at uniform ρ=0.5)

| mesh | SS p=3 | CS p=3 | CC p=3 | SS p=1 |
|---|---|---|---|---|
| 40×5 | 71.51 | 105.67 | 147.43 | 143.02 |
| 48×6 | **69.01** | **104.69** | **146.85** | 138.01 |
| 64×8 | **68.75** | **104.27** | **146.25** | 137.51 |
| 80×10 | 68.62 | 104.07 | 145.97 | 137.25 |
| **paper** | **68.7** | **104.1** | **146.1** | — |

`SS_corner` (pin the two bottom corners) gives ≈100–200 — clearly wrong.

## Findings (all independently derived)

1. **The FE core in the spec is correct.** With `p=3`, the mid-height-pin SS interpretation, and a mesh of ~48×6–64×8, all three targets reproduce to **<1%**. Mesh refinement converges monotonically from above toward the targets. ⇒ Spec sections 1–2, 8.1 and the [I] Q4 assumption are validated.

2. **SS = pin (ux,uy) at the mid-height node of each end edge** — NOT corner pins. This resolves spec ambiguity §11.13 for the SS case and independently confirms the existing code's `build_supports_exact` choice.

3. **The published initial frequencies are the SIMP-*penalized* (p=3) values, not the physical (p=1) values.** At `ρ=0.5`, `ω² ∝ 0.5^{p-1}`: p=1 gives ω ≈ 143/211/295 (≈2× too high), p=3 gives the published 68.7/104.1/146.1 (the data show the exact factor-of-2: 143.0→71.5, etc.). So the paper evaluated the initial design with `p=3` already in force.

   **Consequence for the continuation question (comparison doc D1):** the benchmark numbers are consistent with **p=3** being used for the reported quantities. This *weakens* the hypothesis that absence of `p:1→3` continuation is what breaks reproduction — the existing fixed-`p=3` code matches the initial frequencies exactly. Continuation may still aid *path stability*, but it is not required to match the reported frequencies, and the paper's own initial numbers are a p=3 evaluation.

4. **The benchmark mismatch is therefore confirmed optimizer-side, not FE-side.** A correct FE + uniform start already lands on the published initial frequencies; the divergence documented in Phase 3 arises only once the optimizer iterates. This matches the project memory note ("reproduction gap is optimizer-side").

## Caveat

This verifies only the FE/initial gate — the easy, decidable part. It says nothing about whether the *optimum* `456.4` (bimodal, connected) is reachable; that depends on the path-control and basin questions raised in Phase 2–3, which remain open and would require the full optimizer to settle.
