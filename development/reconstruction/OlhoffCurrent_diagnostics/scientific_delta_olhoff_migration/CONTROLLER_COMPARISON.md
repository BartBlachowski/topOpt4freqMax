# CONTROLLER_COMPARISON — Part 8 (adaptive box vs three-rung ladder)

## 1. Source: `move.policy = 'adaptive'` (`+olh/+move/limit.m`, source 6b08708)

Recovered from code and verified by replay (the replayed box reproduces `hist.move` = max and
`aux.moveMean` = mean **bitwise** for all 64 M1 iterations; `scripts/sd_m1_post.m`).

```
outer 1:   d_e = move.initial (0.10) for every element; state rhoPrev1 = rho_0
outer 2:   d_e unchanged (0.10); rhoPrev2 = rho_0, rhoPrev1 = rho_1
outer k≥3: z_e = (rho_k,e − rho_k−1,e)(rho_k−1,e − rho_k−2,e)     rho = clamped designs actually visited
           f_e = 1.2 if z_e > 0,  0.7 if z_e < 0,  1 if z_e = 0
           d_e ← min(move.initial, max(move.minimum, f_e d_e))  = clamp to [0.002, 0.10]
inner:     box_e = [max(rho_min − rho_e, −d_e), min(1 − rho_e, d_e)]
record:    hist.move(k) = max_e d_e,  aux.moveMean(k) = mean_e d_e
```

| question | answer |
|---|---|
| initialization | 0.10 for all elements (sweep override of the parent preset's 0.04) |
| how it changes | ×1.2 after two same-sign outer steps, ×0.7 after a reversal, ×1 if either step is exactly 0 |
| trigger | the element's own sign pattern of its last two **outer** design changes |
| min / max | 0.002 / 0.10 — the ceiling is `move.initial`, so growth only recovers from contraction |
| local sign changes? | yes, per element |
| per-element or global? | per element (vector box) |
| stage exhaustion? | none |
| branch A/B maturity? | none |
| does β affect it? | no |
| natural ε termination acts instead? | yes: `stop.norm = l2`, `‖Δρ‖₂ < ε = 0.15`, `settledMove = false`, `boxInactiveFraction = 0`, no persistence |
| constants | the same 1.2 / 0.7 that published MMA uses for asyincr / asydecr, applied to the outer history |
| provenance | class C reconstruction (source's own label) |

## 2. Target: three-rung controller (C480 canary) and canonical production

| question | three-rung (C480) | canonical production |
|---|---|---|
| initialization | global 0.04 | global 0.04 |
| levels | [0.04, 0.02, 0.01] | [0.04, 0.02, 0.01, 0.005] |
| trigger to descend | stage-exhaustion declaration E = A ∨ B held P = 20 consecutive iterations; A: med₂₀ cos < 0 ∧ med₂₀ net < 0.5 ∧ ‖Δρ‖₂ ≥ ε; B: ‖Δρ‖₂ < ε ∧ med₂₀ cos > 0; windows stage-local (W = 20, Wnp = 10) | β stall: mean of last 10 β vs previous 10, relative gain < 5e−3, with dwell |
| local sign changes? | only through global cosine/net statistics | no |
| per-element or global? | global scalar | global scalar |
| stage exhaustion | yes | no |
| branch A/B maturity | yes (all C480 events branch B) | no |
| β | no authority | drives the ladder |
| termination | terminal E declaration at the last rung | ‖Δρ‖₂ < ε on a move unchanged from the previous iteration |
| C480 events | stages start at 1, 309, 348; terminal declaration at 386 (window 367–386) | Sept-11 production endpoint 164 iterations, stopped one iteration after a move change |

## 3. Behavioural comparison at 480×60 (measured)

| | C480 (ladder, SIMP/4b) | M1 (adaptive, SIMP/4b) | S480 (adaptive, Pedersen/eq.2) |
|---|---|---|---|
| first step at ρ₀ | ‖Δρ‖₂ 4.27, max 0.040 | 13.39, max 0.100 (cos 0.960 to C480's) | = M1 (bitwise) |
| M_nd at outer 10 / 20 / 40 / 64 | 0.817 / 0.644 / 0.560 / 0.507 | 0.556 / 0.479 / 0.406 / 0.285 | 0.560 / 0.474 / 0.289 / 0.170 |
| first outer with M_nd ≤ 0.5 / 0.35 / 0.25 | 68 / 160 / never | 16 / 46 / never | 17 / 36 / 46 |
| mean box at stop | 0.01 (scalar) | 0.0034; 80 % of elements at the 0.002 floor | 0.028 (max 0.10) |
| sign-reversal fraction (median over run) | 0.22 | 0.27 | n/a |
| spike events | 0 | 11 | 0 |
| outer at stop | 386 | 64 (inside a spike state) | 112 |

## 4. Reading

The adaptive box is **highly material**: it is the first divergence (D7) at ρ₀ and it roughly
triples early design progress under either material law. It is not "better because simpler":
under the target's SIMP/eq.(4b) law it produces spikes, a collapsing box and an ε-stop inside a
localized-mode state with an endpoint grayer than the ladder's (0.285 vs 0.263). Its clean
behaviour at every sweep mesh is observed only together with the Pedersen stiffness law.
