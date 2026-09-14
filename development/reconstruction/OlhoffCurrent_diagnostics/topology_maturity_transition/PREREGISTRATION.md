# PREREGISTRATION — NOT REACHED

This file exists so that the absence of a preregistration is part of the record
and cannot be back-filled later.

**No transition rule was preregistered by this task.** Phase B was never entered.

## Why

Phase B may only be entered once Phase A has nominated exactly one statistic
under §A10. Phase A nominated none:

> `TOPOLOGY_MATURITY_SIGNAL_NOT_IDENTIFIED`

The evidence is in [`OFFLINE_SIGNAL_ANALYSIS.md`](OFFLINE_SIGNAL_ANALYSIS.md).
In short: at the two states of known, opposite maturity — 160×20 at iteration 79
(2.20 % of topology evolution remaining) and 320×40 at iteration 130 (11.61 %
remaining) — every mesh-normalised, outlier-robust candidate returns essentially
the same value (ratios 1.04, 1.06, 1.14), and every candidate that does vary
reads *larger at the more mature mesh*. Across matched maturity levels the
mesh ratios swing by 37–81× and change sign.

§A10 is explicit that this outcome is terminal, and §B1 is explicit that a
threshold which cannot be defended without topology-performance fitting must not
be preregistered. Choosing one here would have meant selecting a cut on a
statistic already shown to be blind to the distinction it was supposed to make,
and defending it by the M_nd it produced — the exact procedure §B1 forbids.

## What was therefore NOT decided

No statistic, no normalisation, no threshold, no persistence/dwell, no transition
predicate, no terminal-admission redesign, and no semantic configuration schema.
`W = 10` was used for the offline analysis only, as §A2 directs, and was not
tuned.

## Consequence

Phases B, C and D were not executed. No causal run was performed at any mesh. No
production code was modified. The existing β-stall move-transition policy remains
in force unchanged, per the standing verdict
`KEEP_CURRENT_MOVE_POLICY_PENDING_REVIEW`.
