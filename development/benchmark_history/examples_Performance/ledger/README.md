# Protocol ledger and freeze record

WP0 of [PLAN_two_table_redesign.md](../PLAN_two_table_redesign.md). These files exist so
that every number in Tables A, B and C has one configuration, one provenance, one timing
basis and one scientific claim — and so that "we froze this beforehand" is a checkable
statement about version control rather than an assertion in prose.

| File | Plan section | Contents |
|---|---|---|
| [protocol_ledger.json](protocol_ledger.json) | §2 | One record per (method, source case). Every setting cites the page, equation, table or figure that supports it, and carries a provenance label. |
| [table_schemas.json](table_schemas.json) | §3.2, §4.5, §7 | Every planned column: definition, unit, provenance rule, source. Plus the missing-data, signed-error, stage-count, delta and commensurability conventions. |
| [freeze_record.json](freeze_record.json) | §2.1 | Every quantity the plan describes as frozen or preregistered, plus an explicit list of what is *not* yet frozen and what blocks each one. |

## Provenance vocabulary

Exactly three labels, per §2:

- **`reported`** — explicitly stated in the source, with a citation.
- **`inferred`** — derived from source material. The inference formula and its uncertainty
  are recorded. Reported as an interval, never as an exact value.
- **`unavailable`** — neither reported nor defensibly inferable. Never silently filled from
  the controlled benchmark.

## How to use this

Before a campaign runs, commit the freeze record and note its hash. Cite that hash in the
manuscript. A frozen value changes only by a new commit that says what changed, why, and
what had already been observed at the time — §4.2.1 and §4.3 each permit exactly one such
revision, and both must be visible in history.

## Status: WP0 is PARTIAL

Three things are worth knowing before reading the ledger.

**The Proposed method is not frozen.** No manuscript is present in the repository, so its
native configuration cannot be cited to a source. §2 requires that freeze to happen *before*
Table A results are inspected, precisely so the native settings cannot be chosen
retrospectively. Until the author supplies it, no Table A row may be produced for the
Proposed method. This is the single blocking item for WP0's exit.

**`τ` and `εV` are working values, not preregistered ones.** §4.2.1 requires them to be
calibrated from measured `rV` and `d∞` floors. `rV` has never been logged by any run in
this repository, so no existing artifact supports `εV = 1e-3`. That calibration is WP4/WP8
work and the freeze record marks it as blocking.

**Two source-side facts were verified during WP0 and are worth flagging.**

1. *The 200-iteration cap belongs to a different algorithm.* Yuksel2025 p. 3206 states that
   the **dynamic comparison code** reduces its maximum density change to 0.01 and is
   terminated after 200 iterations. The proposed method's own static stages terminate on the
   0.01 max-density-change test of p. 3204 with no stated iteration limit. The benchmark
   configuration's `stage1_max_iters: 200` therefore imports a parameter from the method
   being compared against. This is direct source confirmation of the WP2 defect.

2. *The inferred iteration counts can be validated against a control.* Yuksel2025 never
   prints iteration counts, so they must be inferred as total runtime divided by runtime per
   iteration from Table 1 (p. 3207). Applying that same formula to the Dynamic Code rows
   recovers 200.2, 200.0 and 200.0 — against the 200 iterations the text states independently
   on p. 3206. The inference reproduces a known value on 3 of 3 control rows, which supports
   its use on the proposed-method rows. The resulting counts are 261, 208 and 180, with
   rounding intervals [232, 298], [196, 222] and [174, 187]. Table A prints the intervals.

The two source papers also turn out to share a benchmark problem: Yuksel2025 §6.1 states its
simply supported beam is the problem of Du and Olhoff (2007), with the same 8 m × 1 m domain,
`vf = 0.5`, `E₀ = 1e7 Pa`, `ν = 0.3`, `ρ₀ = 1 kg/m³` used by the Olhoff–Du beam examples. That
is the compatibility precondition §4.1 places on the primary Table B, and it is satisfied
rather than assumed.

## What the ledger does not do

It does not resolve the fidelity discrepancies it records — the Yuksel iteration-count gap,
the Olhoff–Du reproduction gap, the missing Yuksel post-processing step, or the documented
deviations of the benchmark Olhoff implementation from its source. Those are Table A results
and belong in Table A's discussion. The ledger's job is to make sure each one is attributable
to a specific setting with a specific provenance, rather than to an unrecorded choice.
