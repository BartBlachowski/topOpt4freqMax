# 480_MATCHED_EXPERIMENT — Parts 11, 12, 13, 17

## 1. Design (as preregistered, AUDIT_PREREGISTRATION §5–§6 + Amendment 1)

| level | source side | target side | runs executed |
|---|---|---|---|
| **M0 native** | committed `repro/results/S480x60` (`duOlhoffAdaptivePedersen`, retained hist/aux + final ρ) | retained C480 three-rung canary (full RHO/DRHO, 386 iterations) | 0 |
| **M1 matched common core** | source snapshot code, committed preset `duOlhoffAdaptiveMove` + committed sweep override `move.initial = 0.10` → the target's material law SIMP p = 3 + eq.(4b) under the source controller | retained C480 (the target cannot realize the adaptive box or Pedersen without code change) | **1** (source side) |

Budget used: one source-side 480×60 trajectory (M1, 395.7 s, launched 16:44:04, finished
16:50:45), zero target-side trajectories, no other mesh, no nine-mesh rerun, no SOCP run. Offline
same-state evaluations at 9 frozen states × 3 evaluators, zero ρ updates.

Preflight (fail-closed, `evaluations/m1_run/M1_preflight.json`): S480 vs M1 configs differ in
exactly `material.stiffness.model`, `material.mass.model`, `runtime.name`, `runtime.diagnostics`,
and resolver metadata `provenance.{preset,overrides,resolvedAt}` (Amendment 1, frozen before
launch, SHA-256 `9b90de7b…`). Every solver symbol resolved inside the snapshot only.

## 2. Validity checks on the M1 run

| check | result |
|---|---|
| ρ trajectory replayed from `res.diag.drho` ends bitwise at `res.rho` | PASS |
| replayed per-element box reproduces `hist.move` and `aux.moveMean` | PASS (bitwise) |
| replayed M_nd reproduces `aux.Mnd` | PASS (bitwise) |
| **P-prefix: M1 `hist`/`aux` bitwise equal to committed S480 for every k < k*** | **PREFIX_BITWISE_PASS** — k* = 6, first difference at 6 (ω, β, dx, vol, M_nd), `nInner`/`cumInner`/`moveMean` first differ at 7 |
| offline evaluator reproduces M1 in-run steps 1, 6, 12 bitwise | PASS |

The P-prefix pass simultaneously proves: (i) the snapshot reproduces committed evidence on this
host; (ii) the diagnostics recorder is inert; (iii) the material law is the only difference, and
it is inactive until an element reaches ρ ≤ 0.1.

## 3. Outcomes (native model of each run)

| | C480 target | M1 source code, SIMP/4b | S480 source |
|---|---|---|---|
| status / outer | CONVERGED (terminal E) / 386 | CONVERGED (ε-test) / **64, inside a spike state** | CONVERGED (ε-test) / 112 |
| final native ω₁ / ω₂ | 163.93 / 185.21 | **34.36 / 34.41** (Pedersen re-evaluation of the same design: 163.55 / 175.97) | 166.01 / 203.44 |
| gap12 | 13.0 % | 0.14 % (localized pair) | 22.5 % |
| M_nd | 0.2634 | 0.2855 | 0.1307 |
| gray / mid fraction | 0.287 / 0.119 | 0.318 / 0.132 | 0.151 / 0.031 |
| broad gray core fraction / area | 0.130 / 1.041 | 0.149 / 1.189 | 0.0006 / 0.004 |
| max gray depth / R | 6.11 | 6.27 | 1.11 |
| spike events | 0 | 11 | 0 |
| inner sub-iterations per outer (mean) | 18.9 | 19.3 | 19.1 |
| target canonical production endpoint (Sept 11, no trajectory) | 164 outer, ω₁ 161.906, M_nd 0.347 | | |

Figures: `fig01` final ρ, `fig02` S480 − C480, `fig03` histograms, `fig04`–`fig12`
trajectories, `fig19` stopping metric.

## 4. Telemetry coverage (Part 13)

Per outer iteration in `evaluations/trajectory_{C480,M1,S480}.csv`: ρ hash, ω₁–ω₃, λ₁, λ₂,
gap12, dOff, volume, M_nd, gray, mid, broad core, β, box max/mean/floor fraction, N, multJ,
max|Δρ|, ‖Δρ‖₂, sign-reversal fraction, bound and box saturation, predicted gain β − λ₁, realized
gain, gain ratio, stop metric vs ε, controller state, inner count, inner status, spike flag.
**Limitation:** S480 retains no ρ trajectory, so its per-iteration gray/mid/broad-core, ρ hash,
sign reversal and saturation are unavailable beyond the M1-identical prefix (iterations 1–5);
M_nd, box max/mean, ω, gap, β, Δρ norms and inner counts are available for all 112 iterations.
Full Δρ vectors: C480 all iterations (retained evidence), M1 all iterations
(`evaluations/m1_run/M1_480x60_trajectory.mat`). Sensitivities (raw and filtered f_sk, f_JJ),
problem-(25) rows, element energies and boxes at the 9 frozen states: `evaluations/same_state/`.

## 5. M_nd difference analysis (Part 17)

Milestones (M_nd):

| outer | 1 | 5 | 10 | 20 | 40 | 60 | 64 | 80 | 100 | 112 | 200 | 386 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C480 | 0.998 | 0.943 | 0.817 | 0.644 | 0.560 | 0.516 | 0.507 | 0.475 | 0.447 | 0.430 | 0.314 | 0.263 |
| M1 | 0.975 | 0.684 | 0.556 | 0.479 | 0.406 | 0.287 | 0.285 | — | — | — | — | — |
| S480 | 0.975 | 0.684 | 0.560 | 0.474 | 0.289 | 0.191 | 0.170 | 0.134 | 0.131 | 0.131 | — | — |

First crossing of M_nd ≤ 0.75 / 0.5 / 0.35 / 0.25 / 0.15: C480 13 / 68 / 160 / never / never;
M1 5 / 16 / 46 / never / never; S480 5 / 17 / 36 / 46 / 69.

Preregistered divergence onset (|ΔM_nd| ≥ 0.03 for 10 consecutive iterations):
**S480 vs C480 at outer 2; M1 vs C480 at outer 2; S480 vs M1 at outer 30.**

Answers:

- **Is target grayness created immediately by a different material model?** No. The two laws are
  identical until an element reaches ρ ≤ 0.1 (source trajectory: iteration 6; C480: iteration 12),
  yet S480 and C480 separate at iteration 2. The early separation is created by the box
  (0.10 vs 0.04), identically in M1.
- **Does it arise from inner-solver attenuation?** Not as a *difference*: the inner solver is
  bitwise identical and attenuates comparably relative to each box (SOLVER_COMPARISON §3).
- **Does the adaptive box sharpen topology?** It sharpens *faster* under both laws (M_nd at 64:
  0.285 M1 vs 0.507 C480), but under SIMP/4b it does not reach a sharper endpoint (0.285 vs 0.263)
  and it traps the run in localized-mode spikes.
- **Does Pedersen stiffness suppress gray/void pathologies?** Yes, under the adaptive box: M1 and
  S480 track each other to iteration ~25 and separate from iteration 30, after M1's first spikes (20,
  23); S480 then removes its broad gray core (area 1.19 → 0.004), 0 spikes vs 11.
- **Does stopping merely preserve an already-sharp design?** For S480, yes: M_nd was 0.134 at 80 and
  0.131 from 100 to 112. For C480, the terminal window was also flat (prior canary report).

Preregistered decomposition (§10) along C → M1 → S on final M_nd:
ΔTotal = 0.1327, ΔController = −0.0220, ΔMaterial = 0.1548, **share_material = 1.17 ≥ 0.75 →
PRIMARILY_FORMULATION**, with M1 spike events recorded (material law necessary for stability
under the source controller). Qualification required by the data: the controller's effect on
*rate* is large (matched-iteration M_nd 0.29 vs 0.51 at 64), and M1's endpoint is a premature stop;
the Pedersen × ladder cell is not isolated.
