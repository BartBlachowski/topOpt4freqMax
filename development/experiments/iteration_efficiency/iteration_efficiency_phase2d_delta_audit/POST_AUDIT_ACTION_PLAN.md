# WP23 / WP24 — Post-audit action sequence
READ_ONLY_INDEPENDENT_DELTA_AUDIT — NOTHING BELOW WAS EXECUTED

The verdict is **DO NOT REFREEZE — AMENDMENT SCIENTIFICALLY OR TECHNICALLY UNSOUND**, so
WP23's approval sequence does not apply. WP24 applies: the minimum evidence and correction
needed. No broad redesign is proposed.

## The blocker, stated as narrowly as it can be

Not the reference-length gap (ruled non-blocking, `REFERENCE_PERSISTENCE_RULING.md`).
Not scope, provenance, mathematics, implementation, hard-gate invariance or collateral drift
(all pass). The blocker is one finding:

> **D1.** Under Eq. (4a), the common evaluator's `omega_raw_E2(1)` and `omega_raw_E3(1)` are
> spurious localized void modes rather than the structural first eigenfrequency on gray
> intermediate states — 34 of 1600 states at 160x20, with the value falling to 18.9% of the
> structural one, and comparable incidence at 240x30 and 320x40.

Phase 2C established that the source's tolerance of Eq. (4)'s discontinuity rests on premises
that fail for this study's estimand. The same is true of the source's tolerance of Eq. (4a):
the authors report "negligible differences in the final results" precisely because
"all intermediate values of the material density will approach 0 or 1 during the design
process". This study scores the intermediate values.

## Minimum evidence needed to close D1

**Step 1 — screening diagnostic (no optimizer, ~1 hour of compute).**
For each candidate low-density mass law, over every stored Olhoff density trajectory
(8 meshes, 358–1601 states each), compute for E1, E2 and E3 at every state:

    s(k) = fraction of mode-1 kinetic energy carried by elements with rho <= 0.1
    r(k) = omega_1^{candidate,E2 or E3}(k) / omega_1^{E1}(k)

A candidate is admissible only if `s(k) < 0.5` at every state with `hard_gate_pass`, and
`r(k)` stays within a declared tolerance of 1. This audit's `scripts/wp_spurious*.py` already
implement the diagnostic; it is cheap and it is the test that would have caught D1.

**Step 2 — settle the mass law.** The choice is the methodology owner's, not the auditor's.
This audit probed the option space empirically at the six worst affected 160x20 states plus
the converged final state (`REMEDY_FEASIBILITY_PROBE.csv`, 105 evaluations). The mechanism
that emerges is decisive and should guide the choice:

> **E2 and E3 need a strong low-density mass suppression precisely because they have weak
> stiffness floors.** E1 carries a stiffness floor of `1e-6` and is structural under *every*
> mass law tried, including a purely linear one. E2's floor is `1e-9` — a thousand times
> softer — and E3 has no additive mass floor at all, only a `1e-3` density clamp giving a void
> stiffness of `1e-2` against E1's `10`. Eq. (4)'s `x^6` suppression is what compensated for
> those weak floors. Any law that puts appreciable mass into the void breaks E2 and E3 while
> leaving E1 untouched.

| option | verdict on the evidence in hand |
|---|---|
| keep Eq. (4) | **refuted** by Phase 2C and reconfirmed here: 2.67e-2 float32 error, 439× the binding decision margin, on 48.7% of states |
| Eq. (4a) as amended | **refuted** by D1 — spurious in 6 of 7 probed states for both E2 and E3 |
| Eq. (4b) (C1) | **refuted** by D2 — spurious in the same 6, and *worse*: E2 ω₁ 21.3–23.2 against Eq. (4a)'s 33.7–37.6 |
| linear mass, Du & Olhoff Eq. (2) with q = 1, for E2/E3 | **refuted** — spurious in **all 7** probed states, including the converged final state k = 1600 (E2 ω₁ = 17.50, E3 = 12.41, void participation 1.0000). E1's clean behaviour does **not** transfer, exactly because of the stiffness-floor difference above. |
| Eq. (4a) **plus a declared modal-validity rule** | **not refuted.** Requires a participation threshold, an increased mode count (3 is provably insufficient — the structural mode sat at index 4 for E2 and 5 for E3 at k = 252), and a declared fallback when no admissible mode is found. A genuinely new methodological device needing its own justification and audit. |
| evaluate E2/E3 on the **exact-count binary field** the evaluator already computes | **not refuted, and clean in every probe.** All three evaluators are structural at all 7 states with void participation **0.0000** and ω₁ ≈ 169.3. The field carries no element near 0.1, so both the discontinuity and the spurious-mode defect vanish at once. Cost: it changes the estimand from raw to binary, a larger methodology change — but `omega_binary_E1/E2/E3` is already computed by the frozen evaluator at every state, so no new machinery is needed. |

The last two are the only survivors. Both are real methodology changes requiring their own
freeze cycle; neither is a two-line amendment.

**Step 3 — re-run the amendment validation** for whichever law is chosen, adding to
Phase-2D's experiment set the level-shift comparison and the modal-participation diagnostic
that finding D3 shows were missing.

**Step 4 — independent delta audit** of that revised amendment.

**Step 5 — refreeze**, executing every obligation in
`REFREEZE_IMPLEMENTATION_OBLIGATIONS.md`, including the five items **[added by this audit]**.

**Step 6 — new Olhoff precision qualification.** Requires a limited optimizer run; the
minimum adequate experiment is specified in `PRECISION_REQUALIFICATION_REQUIREMENTS.md`
(96x12, horizon 3200, reference run separate from measurement, float64 density snapshots
retained at every update, ~29 MB). Do not run it before Step 2 — a result obtained under a
law that is later replaced is worthless.

**Step 7 — independent review of that qualification.**

**Step 8 — production preflight.** It fails closed today on `olhoff_lossless_trajectory` and
will continue to until Step 6 lands.

**Step 9 — production authorization.** Not before Steps 1–8.

## What is already settled and need not be redone

- Source provenance for Eq. (4), (4a), (4b): verified against the primary PDF (WP1).
- Branch-point mathematics: verified in exact rational arithmetic (WP2).
- Implementation scope: exactly two functional lines, E1 untouched, no collateral drift
  (WP4, WP18).
- Hard-gate invariance: structural, not merely observed (WP10).
- Phase-2B historical classification: correct in both directions (WP14).
- Every Phase-2D stability measurement: independently reproduced (WP6, WP7, WP9).
- Decision margins on the one reference-length trajectory: newly measured here, reusable for
  any candidate law (WP11, WP12).

## What must not happen

- No refreeze on the current amendment.
- No 3200-update qualification run under Eq. (4a).
- No production authorization token.
- The two amended normative documents must not be reverted: their retirement of the
  native-identity claim is correct and remains correct under any continuous replacement law.
