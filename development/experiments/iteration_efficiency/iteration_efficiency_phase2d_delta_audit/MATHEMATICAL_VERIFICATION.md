# WP2 / WP17 — Independent mathematical verification
READ_ONLY_INDEPENDENT_DELTA_AUDIT

Recomputed from the source definitions, in exact rational arithmetic (`fractions.Fraction`)
and in IEEE-754 double. Phase-2D's numbers were not consulted until after computation.
Script: `scripts/wp2_math.py`.

## Exact rational arithmetic at ρe = 1/10

| quantity | exact value in ℚ |
|---|---|
| Eq. (4) low branch, (1/10)^6 | `1/1000000` |
| Eq. (4) high branch, 1/10 | `1/10` |
| Eq. (4) multiplicative jump | **`100000` — exactly 1e5** |
| Eq. (4a) low branch, 1e5·(1/10)^6 | **`1/10`** |
| Eq. (4a) C0 residual | **`0` — exactly zero** |
| Eq. (4b) low branch, 6e5·(1/10)^6 − 5e6·(1/10)^7 | `1/10` |
| Eq. (4b) C0 residual | `0` |
| Eq. (4b) low-side slope, 6·6e5·(1/10)^5 + 7·(−5e6)·(1/10)^6 | `1` |

The `6.939e-17` "residual" reported by Phase 2D is an **IEEE-754 artifact of evaluating
`1e5*0.1^6`**, not a property of Eq. (4a). In exact arithmetic Eq. (4a) is continuous with
residual identically zero. Phase-2D's `EVALUATOR_UNIT_TESTS.md` states this correctly as
"one ULP"; the distinction is recorded here for precision.

## Branch-point table (IEEE-754 double)

| quantity | Eq. (4) | Eq. (4a) | Eq. (4b) |
|---|---|---|---|
| left limit g(0.1⁻) | 9.999999999999995e-07 | 0.09999999999999995 | 0.10000000000000003 |
| value g(0.1) (branch closed below) | 1.0000000000000004e-06 | 0.10000000000000003 | 0.09999999999999998 |
| right limit g(0.1⁺) | 0.10000000000000002 | 0.10000000000000002 | 0.10000000000000002 |
| absolute jump | **9.999900e-02** | 6.938894e-17 | −1.387779e-17 |
| multiplicative jump | **1.000000e+05** | 1.000000 | 1.000000 |
| left derivative | 6.000e-05 | **6** | **1** |
| right derivative | 1 | **1** | **1** |
| **C0 continuous** | **no** | **yes** | yes |
| **C1 continuous** | no | **no — slope jumps 6 → 1** | yes |

All of Phase-2D's arithmetic claims are **confirmed**.

## Continuity, differentiability and numerical stability are three different things

The audit request asks these to be distinguished. They are:

- **Continuity (C0).** Eq. (4) fails: a finite jump of factor 1e5. Eq. (4a) and (4b) hold.
- **Differentiability (C1).** Eq. (4) fails. **Eq. (4a) also fails** — the one-sided
  derivatives at 0.1 are 6 and 1. Only Eq. (4b) holds. Eq. (4a) must never be described as
  "smooth" without that qualification; Phase 2D correctly does not.
- **Numerical stability.** The governing property is not C0 or C1 but the **global Lipschitz
  constant**, because what the evaluator does to a perturbed density is bounded by
  `|g(a) − g(b)| ≤ L·|a − b|`:

| law | sup abs g' on the low branch | high branch | global Lipschitz constant |
|---|---|---|---|
| Eq. (4) | 6.0e-05 | 1 | **none — g is discontinuous** |
| Eq. (4a) | **6.000000** | 1 | **6** |
| Eq. (4b) | **2.775986** | 1 | **2.776** |

This is the correct formal statement of what the amendment buys: Eq. (4a) makes the common
mass law **globally Lipschitz with constant 6**, where Eq. (4) had no finite Lipschitz
constant at all. It is *not* a claim of smoothness.

Both amended low branches are non-negative and monotone increasing on [0, 0.1].

## WP17 — the C1 kink: does it matter?

**No, and the evidence is stronger than "no demonstrated harm".**

1. A post-hoc evaluator is never differentiated. No frozen rule consumes `dg/dρ`.
2. The residual perturbation under Eq. (4a) is **not set by the kink**. On 236 genuine paired
   states, amended E2 float32 error is **5.5960e-08** against **E1's 5.5949e-08** — and E1's
   mass law is linear with Lipschitz constant 1 and no branch at all. On the 160x20
   trajectory, amended E2 branch-side sensitivity is **2.6560e-10** against E1's
   **2.6252e-10**, a ratio of 1.012. The amended E2/E3 residual **is** the generic float32
   quantisation floor. Reducing the Lipschitz constant from 6 to 2.776 cannot reduce a
   quantity that is already at the floor of the branch-free control.
3. Therefore Eq. (4b) would deliver **no measurable stability improvement** over Eq. (4a).

**WP17 ruling: EQ4A_C1_KINK = ACCEPTABLE.** The kink is irrelevant to this use, and
adopting Eq. (4b) for C1's sake would have been unjustified.

**However** — and this is the finding that governs the audit — Eq. (4b) is not merely
unnecessary, it is **actively worse** on the axis that turns out to matter. At every density
in (0, 0.1) it places *more* mass in the void than Eq. (4a):

| ρ | Eq. (4) | Eq. (4a) | Eq. (4b) | (4b)/(4a) |
|---|---|---|---|---|
| 1e-3 | 1.000e-18 | 1.000e-13 | 5.950e-13 | 5.95 |
| 0.01 | 1.000e-12 | 1.000e-07 | 5.500e-07 | 5.50 |
| 0.05 | 1.563e-08 | 1.563e-03 | 5.469e-03 | 3.50 |
| 0.08 | 2.621e-07 | 2.621e-02 | 5.243e-02 | 2.00 |
| 0.1  | 1.000e-06 | 1.000e-01 | 1.000e-01 | 1.00 |

Measured at state k = 252 of the 160x20 production trajectory (dense LAPACK):

| variant | E2 ω₁ | E3 ω₁ |
|---|---|---|
| Eq. (4) | 166.487 (structural) | 166.487 (structural) |
| Eq. (4a) | **31.404 (spurious void mode)** | **22.390 (spurious)** |
| Eq. (4b) | **22.206 (spurious, worse)** | **15.832 (spurious, worse)** |

So the Phase-2D decision not to adopt Eq. (4b) was correct — for a better reason than the
one given.

## Does the amendment reinstate the localized-eigenmode problem Eq. (4) exists to suppress?

Element-level ratio of a void element's local frequency scale to the structural scale, for
the E3 model at its ρ = 1e-3 floor:

| law | void mass | sqrt(K/M) void | sqrt(K/M) solid | ratio |
|---|---|---|---|---|
| Eq. (4) | 1e-18 | 1.000e+08 | 3.162e+03 | 31623× |
| Eq. (4a) | 1e-13 | 3.162e+05 | 3.162e+03 | 100× |

At the *floor* the margin is still 100×, so Phase-2D's reasoning is right as far as it goes.
It does not go far enough: the pathology is not an isolated void element at the floor but a
**large connected gray region at ρ just below 0.1**, whose collective modes are far softer
than any single element. That case was not examined by Phase 2D and is the subject of
finding **D1**.
