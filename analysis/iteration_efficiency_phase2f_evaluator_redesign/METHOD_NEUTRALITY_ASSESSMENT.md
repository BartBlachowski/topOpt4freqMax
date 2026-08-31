# WP15 — Method neutrality of candidates C and D
PHASE 2F — EVIDENCE ONLY

The evidence base is asymmetric and this document keeps three categories strictly apart:

    DEMONSTRATED      measured on stored trajectories
    BY CONSTRUCTION   provable from the rule's definition, independent of data
    UNVERIFIED        cannot be checked because the trajectories do not exist

**Only Olhoff density trajectories exist in this repository.** Phase 2C established this and
Phase 2E re-verified it by enumerating every `.mat` artifact: the Proposed and Yuksel final
campaign runs retained checksums and scalar histories, not density fields. Nothing in this
phase can therefore *demonstrate* cross-method behaviour for any candidate.

## Candidate C — Eq. (4a) plus a modal-validity selection

### Does the rule reference method identity? — NO, BY CONSTRUCTION

The selection consumes exactly three things: the density field `x`, the evaluator's own
interpolation (`E_e`, `m_e`, `z_eff`), and the eigenpairs of the resulting pencil. There is
no method argument, no provenance field, no run-order dependence, no branch on which
optimizer produced the field. This is the same structural argument that makes the frozen
hard gate method-independent, and it is verifiable by reading the rule rather than by
sampling trajectories.

### Does it use native interpolation-specific information? — NO, BY CONSTRUCTION

`z_eff` is the density the evaluator's *own* law is evaluated at (E3's `max(x,1e-3)` clamp,
E1/E2's raw `x`). It is a property of the evaluator, applied identically to every field. No
native optimizer constant enters.

### Could one method's grayness systematically produce more rejected modes? — YES, AND THIS IS THE REAL QUESTION

The incidence of void-localised modes below the structure depends on how much soft,
massive, weakly-connected material a design carries. That is a genuine property of the
intermediate design, and different update laws traverse that regime differently. Two
readings are possible and they must not be conflated:

- **Reading 1 — physical.** A design carrying a large soft region genuinely *has* low-frequency
  local modes. Rejecting them and reporting the structural mode measures the load-bearing
  performance consistently for every method. Under this reading, differing rejection counts
  reflect differing designs, which is what a measuring instrument should reflect.
- **Reading 2 — bias.** If the *rejection rule itself* behaved differently for different
  grayness profiles — e.g. becoming ambiguous or failing to find a structural mode more often
  for one method — that would be instrument bias, not a property of the design.

Reading 1 is benign; Reading 2 is not. The two are distinguishable by evidence: Reading 2
shows up as ambiguous modes, escalation failures, or threshold-sensitive classification, and
those are measured in `MODAL_LOCALIZATION_DISTRIBUTIONS.csv`,
`STRUCTURAL_MODE_THRESHOLD_PLATEAUS.csv` and `MODE_COUNT_REQUIREMENTS.csv`. Where the
populations separate cleanly and the plateau is wide, the rule's *decisions* are insensitive
to grayness even though its *rejection counts* are not — which is precisely Reading 1.

**Status: neutrality BY CONSTRUCTION for the rule; the Reading-1/Reading-2 discrimination is
DEMONSTRATED on Olhoff only and UNVERIFIED for Proposed and Yuksel.**

## Candidate D — exact-count binary projection

### Is the projection identical for every method? — YES, BY CONSTRUCTION

`exact_count_binary(x, volfrac)` sorts by descending density with an increasing-global-index
tie-break and takes exactly `round(volfrac·N)` elements. No method-dependent input. It is
already the frozen topology gate's projection, so this neutrality property is inherited, not
newly claimed.

### Does it privilege methods that become binary earlier? — YES, AND THE SIGN IS NOT OBVIOUS

This is candidate D's central neutrality question, and the answer found here is not the
expected one. The naive worry is that a method reaching 0–1 sooner would score higher
sooner. The measured behaviour is the opposite and worse: **when the field is far from
binary, the projection is not merely approximate, it is arbitrary.** On the 160x20 Olhoff
trajectory:

| k | distinct density values | density gap at the cutoff | elements tied at the cutoff | binary ω₁ (E2) | gray lowest ω₁ |
|---|---|---|---|---|---|
| 1 | 2 | 1.0e-02 | **1600** | **0.0488** | 69.164 |
| 2 | 3 | **0.0** | 24 | 0.0497 | 69.932 |
| 10 | 11 | **0.0** | 20 | 0.0567 | 76.097 |
| 20 | 21 | **0.0** | 36 | 61.253 | 83.709 |
| 50 | 50 | 1.0e-02 | 32 | 97.918 | 105.667 |
| **100** | 99 | **0.0** | 30 | **1.4510** | 150.317 |
| 252 | 398 | 1.9e-04 | 1 | 169.327 | 31.404 |
| 800 | 624 | 1.8e-03 | 1 | 169.354 | 166.608 |
| 1600 | 699 | 4.2e-03 | 1 | 169.581 | 167.049 |

At k = 1 the field takes **two** distinct values and 1600 of 3200 elements sit exactly at the
cutoff: which half becomes solid is decided **entirely by the global index tie-break**. The
resulting ω₁ is 0.0488 — a near-mechanism, three orders below the gray value. At k = 100,
thirty elements tie and the binary ω₁ collapses to 1.45 against a gray value of 150.3.

This is method-relevant because *how long a method's field stays coarsely quantised* is a
property of its update law. Olhoff's move-limited LP updates hold densities on a small set
of accumulated values for many iterations, producing exactly these mass ties. A method whose
densities spread continuously would tie far less often. So candidate D's arbitrariness is
**not uniformly distributed across methods** — and, unlike candidate C's, the variation is
driven by an index tie-break rather than by any physical property of the design.

**A mitigating structural fact:** the frozen hard gate uses *the same* binary field and
requires left–right connectivity, so a degenerate projection should also fail the gate and
never reach the acceptance scan. Whether that guard is complete — whether any hard-gate-
PASSING state carries a degenerate binary ω₁ — is measured in
`BINARY_PROJECTION_STABILITY.csv` and `HARD_GATE_VS_MODAL_VALIDITY.csv`, and is reported in
`CANDIDATE_D_BINARY_ANALYSIS.md`. It must not be assumed.

**Status: projection neutrality BY CONSTRUCTION; tie-driven arbitrariness DEMONSTRATED on
Olhoff; its cross-method incidence UNVERIFIED.**

## What cannot be settled without new trajectories

For both candidates the *rule* is method-independent by construction, and that is the part
that can be established now. What cannot be established is the *empirical incidence* of each
candidate's failure mode across methods:

- candidate C: how often does a method's designs put a void mode below the structure?
- candidate D: how often does a method's density distribution produce cutoff ties?

Both are properties of the optimizers' intermediate fields, and neither can be measured
without Proposed and Yuksel density histories. See `NEXT_PHASE_REQUIREMENTS.md`.
