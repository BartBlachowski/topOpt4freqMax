# CAUSAL_ATTRIBUTION — Part 23

```
SOURCE_SUCCESS_PRIMARILY_FORMULATION
```

Issued by the preregistered rule (§10): share_material = ΔMaterial / ΔTotal = 0.1548 / 0.1327 =
**1.17 ≥ 0.75**, with M1 spike events and a false ε-stop recorded as "material law necessary for
stability under the source controller". The rule is applied as frozen; the qualifications below are
part of the verdict, not a revision of it.

## Ranked candidate explanations (for the successful 480×60 behaviour)

| rank | candidate | evidence level | what the evidence is |
|---|---|---|---|
| 1 | **Pedersen low-density stiffness treatment** | **STRONG CAUSAL EVIDENCE** | Single factor at 480 (M1 vs S480, configs differ only in the material law, bitwise prefix to k* = 6): spikes 11 → 0, false ε-stop at 64 → clean stop at 112, M_nd 0.285 → 0.131, broad gray core 1.19 → 0.004. Same-state: the M1 endpoint has three localized SIMP/4b modes (ω₁ 34.36, 100 % kinetic energy in 0.1 < ρ ≤ 0.3) that vanish under Pedersen (ω₁ 163.55). Consistent retained pairs: A2 vs S240 (13 vs 0 spikes; MODERATE), C3 vs S800 (spikes, killed; describe-level). |
| 2 | **Adaptive per-element box** | **STRONG CAUSAL EVIDENCE for rate and for the termination mechanism; EVIDENCE AGAINST as a sufficient cause of low grayness** | First divergence (D7 at ρ₀, everything upstream bitwise). M_nd at iteration 64: 0.285 (M1) vs 0.507 (C480) under the same law. But under SIMP/4b its endpoint (0.285) is not below the ladder's (0.263); ΔController = −0.022 (< 0.03 threshold). Box collapse (80 % at floor) is what let the ε-test fire in M1. |
| 3 | **Natural ε stopping** | **MODERATE CAUSAL EVIDENCE for iteration count; EVIDENCE AGAINST for sharpness or optimality** | S480 stopped 30+ iterations after M_nd had flattened; gray-fit physical KKT 0.364 (S480) vs 0.334 (C480) under a common law → HEURISTIC_STOP; the same rule stopped M1 inside a spike. Stop rule and box were changed together in M1 vs C480, so not separately isolated. |
| 4 | **Mass-law difference (eq.2 vs eq.4b)** | **MODERATE evidence against as the grayness/spike cause** (P1 vs P2 at 240, R = 0.0433: spikes 0/0, M_nd 0.081/0.085); affects outer count (191 vs 314). NOT ISOLATED at 480. | |
| 5 | innerLoop / problem-(25) realization | **IDENTICAL BETWEEN IMPLEMENTATIONS** | byte-identical code; bitwise same-state steps at 9 states |
| 6 | MMA accuracy / state / asymptotes / subsolv | **IDENTICAL BETWEEN IMPLEMENTATIONS** | same files and constants; attenuation relative to the box comparable (step/box RMS 0.49 vs 0.57 in the first 10 iterations); box-mediated differences belong to rank 2 |
| 7 | multiplicity handling | **IDENTICAL BETWEEN IMPLEMENTATIONS** | bitwise N, dOff, rows |
| 8 | sensitivity filter | **IDENTICAL BETWEEN IMPLEMENTATIONS** | bitwise filtered f_sk; the non-conservativity is still present in the source |
| 9 | FE assembly | **IDENTICAL BETWEEN IMPLEMENTATIONS** (given the law) | K, M SHA-256 equal |
| 10 | eigensolver | **IDENTICAL BETWEEN IMPLEMENTATIONS** | bitwise eigenpairs; defaults of the new opts equal the old constants |
| 11 | initialization | **IDENTICAL BETWEEN IMPLEMENTATIONS** | ρ₀ = 0.5 |
| 12 | other: filter radius, ε, ρ_min, volume, p, N | **IDENTICAL** | 70 of 96 config leaves equal in all four configs |
| — | Pedersen × three-rung ladder | **NOT ISOLATED** | no run exists; would decide whether the old preset's grayness is itself a low-density-law effect |

## Qualifications

1. **Interaction.** The spike pathology of SIMP/4b is observed under the adaptive box (M1, A2, C3)
   and not under the three-rung ladder at 480 (C480, 0 spikes). The evidence shows Pedersen is
   *necessary* for the source controller's clean behaviour; it does not show Pedersen would make
   the target controller's endpoint sharp.
2. **Rate vs endpoint.** The box triples early progress under either law. The "primarily formulation"
   verdict concerns the final-M_nd decomposition, as preregistered.
3. **Termination.** The natural-termination *mechanism* is the controller (box contraction + ε);
   its *credibility* is the formulation.

## One-sentence attribution

The source's success is the Pedersen low-density stiffness law making its adaptive-box + ε-stop
controller stable: all shared numerical machinery (FE, eigensolver, gradients, filter, multiplicity,
problem (25), MMA) is bitwise identical, the controller creates the first divergence and the speed,
and the material law is what prevents that controller from falling into localized-mode spikes and a
false gray stop.
