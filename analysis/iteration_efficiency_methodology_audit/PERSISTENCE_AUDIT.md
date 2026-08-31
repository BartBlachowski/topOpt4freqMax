# Persistence audit — `P = 100`, `k_enter`, `k_cert`

Audit-only. No Phase-1A document was modified.

## 1. Provenance of `P = 100`

Traced to the Olhoff stabilization policy, where it is not merely a default but an
enforced preregistration constant. `analysis/olhoff_stabilization_audit/olhoffOptStabilized.m`:

```matlab
defaults = struct('id','S0','move_sequence',baselineMove,'gap_threshold',0.01,'persistence',100);
...
assert(p.persistence==100 && p.gap_threshold==0.01, 'Policy differs from preregistration.');
```

and `selected_profile.json`:

```json
"stabilization_trigger": {
  "condition": "N == 2 && gap12 <= 0.01",
  "persistence_consecutive_iterations": 100,
  "counter_resets_on_false": true,
  "uses_future_information": false
}
```

The supporting evidence cited in the protocol — that prior audits found 50-step look-ahead
false positives — is evidence gathered **on Olhoff trajectories**. It is a sound basis for
`P = 100` for Olhoff. It is not evidence about Proposed or Yuksel.

**Verdict:** `P = 100` has been generalised from one method to three. The protocol's claim
that it "is evidence based" is true of one third of its application.

## 2. Magnitude of the burden, per method

From `examples/Performance/final_campaign/table1_performance.csv`:

| method | native counts across nine meshes | `P − 1 = 99` as a share of the run |
|---|---|---|
| Proposed | 107, 236, 207, 182, 219, 256, 309, 297, 330 | **30 % – 93 %** |
| Yuksel (S1+S2) | 244, 320, 572, 732, 1201, 1604, 2000*, 1966, 2000* | 5 % – 41 % |
| Olhoff | 1600 fixed (or 357/399/1066 on failure) | **6 %** |

\* iteration cap, not convergence.

The fixed additive burden is **5× to 15× larger in proportion to the work being measured**
for Proposed than for Olhoff. `FAIRNESS_RISK_REGISTER.md` F10 gets the direction right
("Adds a large fixed burden to short Proposed trajectories"); the magnitude is not stated
anywhere, and the magnitude is what a reader needs.

## 3. Consequence-by-consequence

**Compresses relative differences** — yes, mechanically, and most for the shortest series.
`k_cert` ratios are always closer to 1 than `k_enter` ratios.

**Changes method ranking** — **no.** `k_cert = k_enter + 99` for every method, so absolute
differences and rank order are preserved exactly. The protocol states this correctly and it
is worth stating loudly, because it is the reassuring half of the answer.

**Distorts scaling exponents** — **yes, quantified.** Using the frozen Olhoff
`persistent_raw_E1_1pct` series [186, 234, 234, 246] at
`N_e ∈ {3200, 7200, 12800, 20000}`:

| series | `C` | `p` | `R²_log` |
|---|---:|---:|---:|
| `k_enter` | 59.91 | **+0.1451** | 0.839 |
| `k_enter + 99` | 131.5 | **+0.0991** | 0.843 |

A **32 % reduction in the fitted exponent** from a bookkeeping convention. Note that
`R²_log` barely moves, so the distortion is invisible in the usual goodness-of-fit
diagnostic. The distortion grows as counts shrink relative to 99, i.e. it is largest for
Proposed.

**Disproportionately penalises expensive Olhoff outer iterations** — in **time**, yes; in
**iterations**, no. At 800x100 the frozen per-iteration costs are 1.844 s (Olhoff),
0.814 s (Proposed), 0.489 s (Yuksel), so the same 99 extra iterations cost 183 s / 81 s /
48 s. The protocol shows both quantities, which is the correct resolution — the two facts
point in opposite directions and neither should be suppressed.

**Interacts unfairly with Yuksel stage boundaries** — guarded correctly. Only Stage-2 states
are acceptance-eligible, so a window cannot straddle the handoff. Two residual interactions
are not guarded: the window can force Stage 2 past its native stop (handled by the observer
extension), and the Stage-1 budget change (Finding M6) moves Stage 2's starting design at
three meshes.

**Interacts with Proposed's native stopping** — yes, materially. Proposed's native run at
160x20 is **107** iterations and `k_cert ≥ 100` by construction, so certification will very
likely require running past the native stop. Mechanically this is fine — the observer
extension suppresses only native termination and
`examples/Performance/extension_invariance_validation.json` already demonstrates bitwise
prefix identity for all three methods at 160x20 (`max_abs_diff_xphys_at_stop = 0`,
`scalar_prefix_identical: true`, `bitwise: true`). Semantically it is odd: the reported
"minimum optimization work" can exceed the work the method performs when used normally.
This must be said wherever `k_cert` appears for Proposed (Finding Mo3).

**Unnecessarily conservative** — for Proposed and Yuksel, probably; for Olhoff, no.

## 4. Same `P` for all methods, or method-specific?

**Keep the same `P`.** The protocol's reasoning is correct and I endorse it without
qualification: `P` is an **evidence burden**, and varying it by method means demanding
different amounts of proof for the same claim. Worse, a method-specific `P` would be exactly
the kind of per-method knob the expected-result firewall exists to prevent — one that could
be adjusted after seeing counts, with a plausible-sounding justification available for any
value.

The cost of uniformity is the unequal proportional burden in §2. That cost is real, it is
disclosed, and it is the right trade. Report it numerically rather than removing it.

## 5. If `P` were re-derived, on what principle?

Not from desired iteration counts, and not from one method's stabilization policy.

The defensible principle is measurable **now**, from existing frozen data, before any
production run:

> `P` is the smallest window length for which the observed **false-certification rate** —
> the fraction of length-`P` all-pass windows that are subsequently followed by a gate exit
> — falls below a preregistered rate, measured across the pre-existing trajectories of
> **all three** methods.

The ingredients already exist. `build_stabilization_outputs.m` computes exactly this class
of statistic via its `crossings()` helper, which returns both the first crossing and the
number of later exits (`nExit`). The Olhoff trajectories alone supply 14 409 states with
per-state gate quantities. Proposed and Yuksel would need the new runs, so a fully
symmetric derivation is only possible after production — which is an argument for keeping
`P = 100` as a frozen convention now and deriving it properly in any successor study.

**If `P = 100` is retained unchanged, that is acceptable**, provided it is labelled what it
is: a convention inherited from the Olhoff stabilization policy, applied uniformly for
equal evidence burden, whose proportional cost differs by 5–15× across the three methods.

## 6. `k_enter` and `k_cert`

**Definitions.** `ACCEPTANCE_GATE_SPEC.md` §5:

```
k_enter = min { a >= 1 : A_e(k) = 1 for all k in [a, a+P-1] }
k_cert  = k_enter + P - 1
```

Both correct and unambiguous.

**Which answers which question.** `k_enter` answers "minimum iterations to *obtain* a proper
result". `k_cert` answers "minimum iterations to *certify* one". No ambiguity.

**Is presenting both sufficient?** In tables and figures, yes — the equal-prominence rule is
explicit and correct. In the abstract and captions, no rule exists, and that is where a
single number gets quoted. Minimum correction: bind any single-number statement of iteration
effort to name its endpoint and carry its quality context.

**Should either be the headline?** `k_enter` should lead, as the protocol says. `k_cert`
cannot: at `P = 100` it is dominated by an arbitrary convention, which for Proposed at
160x20 is comparable to the method's entire native run.

**Look-ahead bias in `k_enter` — two distinct forms.**

1. *`P`-window look-ahead.* `k_enter` cannot be known until `P` further states are observed.
   Structural, unavoidable, honestly framed as a retrospective maturation location, and
   paired with the prospective `k_cert`. **Not problematic.**
2. *Whole-horizon dependence.* `k_enter` depends on `Q_ref`, which is a maximum over windows
   within the *entire* observed horizon — including everything after `k_cert`. So `k_enter`
   consumes information from the far future of the run, not merely `P` states ahead, and it
   is a functional of `B0`. Since `B0` is 900 / 2000-per-stage / 3200, the references are not
   computed over comparable horizons. **Problematic, and undisclosed.** This is Finding C2.

The protocol's retrospective framing covers form 1 completely and is silent on form 2. The
two are not the same claim and should not share one disclosure sentence.
