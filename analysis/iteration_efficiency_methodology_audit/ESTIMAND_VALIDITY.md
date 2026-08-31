# Estimand validity audit — R and A

Audit-only. No Phase-1A document was modified.

## 1. What R actually measures

From `ACCEPTANCE_GATE_SPEC.md` §4:

```
Q_ref_mj = max over base-valid length-P windows W of ( min over k in W of Q(k) )
S_R(k)   = [ Q(k) >= (1 - delta_R) * Q_ref_mj ]
```

R is therefore **the work required to enter and hold the top `delta_R` band of a method's
own best sustained common-E1 floor, within the horizon that method was observed over.**

Three properties of that definition matter and only one is stated in the package.

### 1.1 Stated: R does not equalise quality

Correctly and repeatedly disclosed (protocol §"Research questions", `ACCEPTANCE_GATE_SPEC.md`
§4, `FAIRNESS_RISK_REGISTER.md` F03). Not a defect.

### 1.2 Unstated: `Q_ref` as a window minimum is a *good* choice, and it has a side effect

Using the window **minimum** rather than a peak or the last state is the right decision. It
makes the reference achievable for a full window by construction, so a method that
oscillates is not asked to hold a level it only touches transiently. Given that the Olhoff
selected profile is documented as never becoming stationary (`selected_profile.json`:
`simple_native_stopping_rule_identified: false`, "max density change remains at
move=0.0025"), this choice is what makes R applicable to that method at all. Credit where
due — this is better than the obvious alternatives.

The side effect: for a noisy trajectory the reference sits **below** the trajectory's own
peaks, so the acceptance band `[0.99·Q_ref, ∞)` is wide relative to that method's best. For
a smooth plateauing trajectory the reference sits **at** the plateau, so the band is
narrow relative to its best. Noise therefore buys a slightly easier bar. The effect is
bounded by the oscillation amplitude and is probably small here, but it is a preference
unrelated to quality and it should be named.

### 1.3 Unstated and material: R rewards early plateau, penalises late improvement

If a trajectory improves monotonically and stops improving at iteration `a`, then
`Q_ref` ≈ `Q(a)` and `k_enter` is close to the point where `Q` first came within
`delta_R` of `Q(a)` — early. If a trajectory is still improving at the horizon, `Q_ref` is
set by the *latest* window, and no earlier window can satisfy the band, so `k_enter` is
pushed toward the horizon.

This is a preference for **stationarity**, which is exactly the property the protocol
explicitly declined to test ("Persistent acceptability is deliberately not the same as
density stationarity", `ACCEPTANCE_GATE_SPEC.md` §5). The protocol avoids putting a
stationarity tolerance in the gate and then reintroduces stationarity through the back
door, in the definition of the reference.

The three methods differ systematically on this axis:

| method | native stop rule (frozen profile) | plateau behaviour |
|---|---|---|
| Proposed | `max\|x − x_old\| <= 0.01` on the raw design field | stops when it plateaus, by construction |
| Yuksel | Stage-2 `max\|dx\| < 0.01` after >= 2 iterations | same |
| Olhoff | none identified; fixed 1600-outer work horizon | documented as never stationary |

### 1.4 Unstated and critical: R is horizon-relative, and horizons differ 3.5×

`Q_ref` is quantified over "the final observer horizon". The horizons are
`B0` = 900 (Proposed), 2000 per stage (Yuksel), 3200 (Olhoff). A longer horizon can only
weakly increase `Q_ref`, hence weakly increase the bar, hence weakly increase `k_enter`.

The method with the smallest budget therefore gets the most easily satisfied
self-reference. The budget formula
`B0 = ceil_100(max(2·K_prior, K_prior + 5P))` was derived from prior work without reference
to the future ordering — that part is clean — but the *consequence* for `Q_ref`
comparability was never examined anywhere in the package.

Note the extension rule partially closes the loop in one direction: extension is refused if
a certified window already exists (`IMPLEMENTATION_REQUIREMENTS.md` §3 condition 4), so a
certification cannot be retroactively invalidated by a later improvement discovered through
extension. That is a good control. But it fixes the *procedure*, not the *comparability*:
it means "the first budget that certifies wins", and the budgets are unequal.

This is Finding **C2**. It is the one defect that goes to whether R can be compared across
methods at all, which is the study's entire purpose.

## 2. Answers to the six audit sub-questions

**Can R systematically reward a method whose final design is poor?** Yes — through §1.3
(early plateau) more strongly than through the low-ceiling mechanism the protocol names.
A method that stops improving early and badly gets an early `k_enter`; a method that keeps
improving to a high final quality gets a late one.

**Does requiring common-evaluator endpoint quality alongside R prevent misleading
interpretation?** Only partly. The quality is in Table 3 and F7, not Main Table 1. Given a
measured 6.1–7.2 % Du–Olhoff lead in common E1-raw ω₁ at every one of the nine frozen
meshes — seven times `delta_R` — a count-only headline table is not defensible.

**Is R comparable across methods with different reference endpoints?** As a statement about
each method's own maturation, yes. As a cross-method efficiency statement, no — and
currently not even as a like-for-like maturation statement, because of §1.4.

**What should R be called?**

| candidate | verdict |
|---|---|
| convergence efficiency | **no** — the protocol explicitly refuses a stationarity test |
| optimization efficiency | **no** — implies work per unit of optimization outcome |
| iteration efficiency | **no** — same, and it is the study's title, which compounds the problem |
| maturation efficiency | acceptable **only** with "self-referenced" in the same sentence |
| **self-referenced maturation work** | **recommended** — accurate, and hard to over-read |

**What claims can R support?** Per method and mesh: "under a common spectral evaluator and
a common structural-sanity gate, method M entered and held a state within `delta_R` of the
best quality it sustained on this trajectory after `k_enter` of its own method-level
iterations, reaching ω₁ = X (Y % of the best observed across the three methods)."

**What must R not support?** More efficient; faster to a given quality; better converging;
cheaper per unit quality; equally good solutions; fewer iterations implies less computation.

## 3. Is R suitable as the primary estimand?

**Yes in kind; not in its current definition.**

It is a legitimate primary quantity for a paper about iteration effort, and I am not
rejecting it for failing to equalise endpoint quality — that limitation is disclosed and
intentional, and the protocol handles it more honestly than most.

It is not currently suitable because:

- `Q_ref` is horizon-dependent with method-asymmetric horizons (C2 — CRITICAL);
- `delta_R` moves the estimand more than the method does (M1 — MAJOR): on the frozen
  Olhoff series the persistent crossing moves from [186, 234, 234, 246] at 1 % to
  [730, 417, 1162, 1534] at 0.5 %, and the fitted exponent from +0.145 to +0.479;
- the structural preferences in §1.2–1.3 are unstated (M2 — MAJOR).

Fix C2, disclose M1 and M2, and R becomes suitable.

**Does the proposed presentation prevent overclaiming?** Not yet. The mechanisms are all
present — Table 3, F7, the naming discipline, the risk register — but the one table a
reader will quote has no quality column, the only symmetric quality comparison is optional,
and nothing constrains the abstract.

## 4. Estimand A

**The conditional treatment is correct.** No `Omega_req(mesh)` exists in the repository; I
looked. Refusing to instantiate rather than mining endpoints for an attractive level is the
right call and should not be weakened under review pressure.

**Is the best-observed benchmark scientifically useful?** More useful than "optional"
implies. With a 6–7 % separation against a 1 % margin, it would very likely return
`QUALITY_NOT_REACHED` for Proposed and Yuksel at every mesh. That is a *result*. Reporting
it as a censored-status column is the honest way to show what the study can and cannot
distinguish.

**Does its terminology prevent it being mistaken for an engineering requirement?** Yes. The
prohibitions ("labelled best-observed benchmark, never A, absolute, engineering adequacy,
or a requirement") are explicit and repeated in three documents. Keep them all.

**Could a later implementation manufacture `Omega_req` from observed trajectories while
still calling A independent?** The prohibition is unambiguous but is enforced only by
prose. `IMPLEMENTATION_REQUIREMENTS.md` §6 lists no engine check that rejects an
`Omega_req` lacking a pre-production provenance hash. Minimum correction: require
`Omega_req` to carry a provenance record hashed into the protocol manifest before the first
production run, and have the acceptance engine hard-fail otherwise. No new scientific
choice; it is the same pattern the package already uses for profiles and source hashes.

## 5. Frozen quality evidence underlying M3

`examples/Performance/final_campaign/common_evaluators.csv`, common E1-raw ω₁:

| mesh | Olhoff | Yuksel | Proposed | Olhoff lead over best other |
|---|---:|---:|---:|---:|
| 160x20 | 166.745 | 157.167 | 153.675 | +6.1 % |
| 240x30 | 169.616 | 159.436 | 157.639 | +6.4 % |
| 320x40 | 170.227 | 160.690 | 158.763 | +5.9 % |
| 400x50 | 170.517 | 159.968 | 159.519 | +6.6 % |
| 480x60 | 170.244 | 160.551 | 160.254 | +6.0 % |
| 560x70 | 171.272 | 160.343 | 160.722 | +6.6 % |
| 640x80 | 172.559 | 160.696 | 160.852 | +7.3 % |
| 720x90 | 173.073 | 160.681 | 161.069 | +7.5 % |
| 800x100 | 172.945 | 160.854 | 161.365 | +7.2 % |

The ordering is identical under E2-raw and E3-raw at all nine meshes, so this is not an
artifact of the evaluator choice. Olhoff rows at 480x60/560x70/640x80 are pre-failure
states, which is exactly the case `Q_ref`'s best-sustained-floor construction was designed
to handle.
