# Phase 1C — response to the independent methodology audit

Status: **design only.** No Phase-2 implementation, no optimization run, no production
calculation, no modification of `performance_comparison.m`, of any MATLAB numerical
algorithm, of any frozen performance artifact, or of any frozen trajectory evidence. The
directory `analysis/iteration_efficiency_methodology_audit/` is immutable independent-review
evidence and was not touched.

The independent audit returned **NOT READY — MAJOR METHODOLOGICAL REVISION REQUIRED** with
2 CRITICAL, 8 MAJOR, 9 MODERATE and 5 MINOR findings. This document is the author's
one-to-one response. Machine-readable closure is in `PHASE1C_FINDING_CLOSURE.csv`.

## Disposition summary

| Severity | Count | ACCEPT | PARTIALLY ACCEPT | REJECT | Addressed in Phase 1C |
|---|---:|---:|---:|---:|---:|
| CRITICAL | 2 | 2 | 0 | 0 | 2 |
| MAJOR | 8 | 8 | 0 | 0 | 8 |
| MODERATE | 9 | 9 | 0 | 0 | 9 |
| MINOR | 5 | 5 | 0 | 0 | 5 |
| **Total** | **24** | **24** | **0** | **0** | **24** |

No finding is rejected. Two dispositions deliberately depart from the audit's stated
minimum correction, both in the stricter direction, and both are flagged for the delta
auditor:

- **Mo7** — the audit asked that `a_res=5` be renamed and its derivation disclosed.
  Phase 1C **retires the constant entirely** rather than renaming a one-method calibration.
- **Mo6** — the audit suggested replacing the T0 sensitivity with a per-component rule at
  the Proposed and Yuksel filter footprints (9 and 21 cells). Phase 1C **declines that
  specific mechanism**, because C1 and Mo7 together prohibit deriving any common topology
  scale from a native filter radius, and substitutes method-neutral FE patch scales.

## Read-only evidence used

Frozen evidence was re-read, never regenerated. Phase 1C additionally performed an
independent read-only recomputation of the repaired topology gate over the frozen Olhoff
snapshot trajectories (exact-count projection, four-connectivity, support footprints) to
test whether the proposed correction is logically satisfiable. This reproduced the
`TOPOLOGY_SANITY_SPEC.md` table rows at four meshes **exactly**, including the anchor case
the audit used to diagnose the aggregate clause:

| mesh | states | `a_sig` | support pass | repaired-gate pass | longest repaired run | final max detached |
|---|---:|---:|---:|---:|---:|---:|
| 160x20 | 1601 | 4 | 98.88% | 66.52% | 957 | 0 |
| 240x30 | 1601 | 9 | 98.88% | 93.07% | 1319 | 0 |
| 640x80 | 1067 | 64 | 98.50% | 95.03% | 925 | 4 |
| 720x90 | 1601 | 81 | 99.00% | 97.81% | 1517 | 8 |

These span the coarsest mesh (strictest `a_sig = 4`), the pilot mesh, the audit's diagnostic
anchor, and a fine mesh. The script is `verify_repaired_topology_gate.py` in this directory,
so the delta auditor can reproduce these numbers directly.

No optimizer was invoked and no trajectory was generated.

---

# WP0 — audit-response ledger

`PHASE1C_FINDING_CLOSURE.csv` carries all 24 findings with finding ID, severity, audit
statement, disposition, supporting evidence, the protocol change made, whether a new
scientific decision was required, whether Phase 1C resolves it, and residual risk. It is
one row per finding with no aggregation. Three findings (C1, C2, Mo7) required a genuine
new scientific choice; the remaining 21 were resolvable by correction, disclosure, or
reporting discipline.

---

# WP1 — C1: the topology gate

**Disposition: ACCEPT.** The audit's empirical demonstration at available meshes is decisive
and is not contested. At 640x80 the superseded T1 rule admitted only 0.6% of states with no
`P=100` window (longest run 5), although each detached component remained below the repaired
physical threshold; aggregate sub-resolution speckle drove rejection. The inherited
800x100 figures are not treated as reproduced evidence: that artifact is zero bytes and the
frozen endpoint is `RUN_ERROR`/E1 `N/A`.

### Baseline intent restored

`TOPOLOGY_SANITY_SPEC.md` Sec 1 now states that the gate answers only whether a topology
is grossly or pathologically invalid, and explicitly disclaims cleanliness, elegance,
fragmentation aesthetics, and speck count.

### The aggregate clause is removed

`sum(detached areas) <= a_res` is **deleted from the hard gate**. No physical argument
supports it: many individually negligible specks do not become one physical component
because their areas sum, and the clause's severity grew 25-fold across the mesh family for
purely bookkeeping reasons. Aggregate detached area, component count, and largest-connected-
component fraction are retained as **mandatory diagnostics**, so nothing is lost from the
record — only from the veto.

The baseline hard gate is now

```
H_T(k) = [C_required(k) = 1] AND [max_c A_c^detached(k) < A_sig]
```

i.e. required structural connectivity AND no individually significant detached component,
exactly the conceptual form the audit response was asked to investigate.

### Resolution scale — method-neutral, not filter-derived

The audit is correct that `a_res=5` was derived from Olhoff's `rmin=1.3` (the smallest of
three: Proposed 2.0 → 9 cells, Yuksel 2.5 → 21 cells) and that calling it `r_common` was
not a valid description. Phase 1C does not rename it and does not substitute another
arbitrary constant. **Native filter radii are method properties and cannot serve as a
neutral yardstick at all**, so the constant is retired and the scale is re-derived from the
fixed FE geometry, which is common to all three methods by construction.

Of the four candidate families the response was asked to consider, Phase 1C adopts a
combination of (B) and (D): a **physical-area criterion normalized to the fixed design
domain**, anchored to the coarsest production mesh's element geometry.

All elements are square because `8/nelx = 1/nely` across the family. With coarsest mesh
160x20, element area `A_e0 = 8/(160·20) = 0.0025`, define

```
A_sig = 4·A_e0 = 0.01
```

the area of a 2×2 Q4 patch on the coarsest production mesh — the smallest two-dimensional
Q4 patch containing an interior shared node. Single cells and one-cell-wide remnants do not
define a physical structural feature. At mesh `j`, a detached component is significant iff
`n_c · A_e(j) >= A_sig`, so the element-count threshold is `a_sig(j) = ceil(A_sig / A_e(j))`.

### Can an element-count threshold stay constant across 160x20 … 800x100?

**No, and this is the explicit reason the constant was rejected.** Physical element area
falls 25-fold across the family, so a constant five-element allowance represents 25× less
physical material at 800x100 than at 160x20 and its share of the solid volume falls
correspondingly. A constant *element count* is therefore a silently tightening gate; a
constant *physical area* is the mesh-invariant statement. The resulting counts are:

| mesh | 160x20 | 240x30 | 320x40 | 400x50 | 480x60 | 560x70 | 640x80 | 720x90 | 800x100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `a_sig` (elements) | 4 | 9 | 16 | 25 | 36 | 49 | 64 | 81 | 100 |

The physical threshold is constant; its element count grows with refinement, which is the
correct direction and the opposite of the superseded rule's behaviour.

### Connectivity — traced to the problem definition, not to terminology

The benchmark is the fixed 8×1 domain with **both translational degrees of freedom fixed at
the mid-height node of each end face**, and nothing else is prescribed. Tracing each method:

- Proposed applies a distributed semi-harmonic load;
- Yuksel changes from a point load in Stage 1 to a design-dependent inertial load in Stage 2;
- Olhoff's free-vibration eigenproblem has **no external loaded region at all**.

There is therefore **no common loaded region**, and support-to-loaded-region connectivity
cannot be imposed as a common hard condition without redefining at least one method. The
only physically required common condition is **support-to-support connectivity**: on the
exact-count binary field, `C_required = 1` iff one four-neighbour component intersects the
Q4 element footprints of both prescribed support nodes. Column-to-column
`left_right_connected` is demoted to a compatibility diagnostic. Support-node index
agreement across the three implementations (`floor` vs `round` of `nely/2`) holds only for
even `nely`, which is now an explicit stated precondition (finding Mi1).

### Validation on frozen trajectories only

`TOPOLOGY_SANITY_SPEC.md` Sec 6 reports pass fractions and longest persistence runs for
all available frozen Olhoff meshes; Phase 1C independently reproduced 160x20, 240x30,
640x80 and 720x90.
Against the four required verification criteria:

1. **Pathological states still fail.** Support connectivity is refused at 1–4.5% of states
   per mesh, concentrated in the early unformed phase; those states are rejected regardless
   of speck accounting.
2. **Specks no longer dominate acceptance.** At 640x80 the repaired gate admits 95.03% of
   states with the final state carrying largest detached area 4 against `a_sig=64`, where
   the superseded rule admitted 0.6% with longest run 5.
3. **No available method trajectory is structurally excluded by an irrelevant artifact.**
   At 640x80 the Olhoff trajectory that T1 removed is admitted, with longest run 925 versus
   5 before. The 800x100 case is unavailable and is not inferred.
4. **The criterion has a defensible mesh interpretation** — a fixed physical area, stated
   per mesh in element counts.

Two honesty limits are recorded rather than papered over. The Olhoff 800x100 artifact
`s1_800x100.mat` is zero bytes and its frozen endpoint is `RUN_ERROR`/E1 `N/A`; its topology
evidence is therefore `UNVERIFIABLE_AT_PRESENT`, with no pass fraction, run length, or final
measurement inferred. And **no all-state pass fraction exists for
Proposed or Yuksel at any mesh**, because the final campaign stored no trajectories for
them; only single diagnostic final fields are available (Proposed 160x20 and Yuksel 800x100,
both passing). That gap is classified in WP16 as requiring new instrumentation, not
inferred from endpoint CSVs.

The gate was not selected to make every method pass, and no variant was scored against an
iteration ranking.

---

# WP2 — C2: horizon dependence removed from R

**Disposition: ACCEPT.** The audit's causal chain is accepted in full: a shorter observer
horizon can only lower a best-sustained-floor reference, which lowers the relative bar,
which moves `k_enter` earlier — so the method with the smallest safety budget had the most
easily satisfied self-reference, and that quantity was the intended headline.

### The required property

Phase 1C splits the study into two phases with different jobs, specified in
`REFERENCE_QUALITY_SPEC.md`:

- **Reference phase** — a dedicated deterministic trajectory per (method, mesh) establishes
  and freezes a defensible sustained attainable quality, by a rule whose semantics contain
  no measurement horizon.
- **Measurement phase** — a separate trajectory from the identical initialization is
  scanned for the earliest attainment of a declared fraction of that already-frozen
  reference.

The measurement engine receives only the provenance-locked triplet
`(Q_ref_E1, Q_ref_E2, Q_ref_E3)` and cannot recompute it from its own horizon. Its budget
and any formulaic extension can affect whether an endpoint is *observed*; they cannot alter
any `Q_ref`.

### Alternatives evaluated

| Construction | Fairness / reproducibility | Runs & censoring | Post-hoc tuning risk | Decision |
|---|---|---|---|---|
| **A. Common reference horizon** | one cap for all, but the cap itself becomes the reference definition and freezes a still-improving method at an arbitrary point | one long reference run; cap and failure endpoints remain semantically unequal | cap value is a free parameter with visible consequences | rejected as baseline |
| **B. Stabilization rule** | identical stopping *semantics* for every method; the cap controls availability, not magnitude | separate reference run; honest `REFERENCE_NOT_ESTABLISHED` possible | constants fixed pre-production and applied to all three evaluators simultaneously | **selected baseline** |
| **C. Frozen terminal-quality reference** | requires no new run | terminal meanings differ across methods; Proposed/Yuksel fields were never stored; Olhoff is fixed-work and failure-censored | provenance is unequal by construction | rejected |
| **D. Reference as an explicit function of horizon** | fully transparent | leaves no single frozen denominator; multiplies results | none, but no primary estimand results | retained as supplementary diagnostic |

Construction C fails on evidence availability alone: the frozen campaign kept checksums,
not final density fields, for Proposed and Yuksel.

### The selected rule

Constants frozen before Phase 2, none of which depend on a method, mesh, ranking, or native
stopping rule: block length `P = 100`, look-back `L_ref = 5P = 500`, reference resolution
`epsilon_ref = 0.001`. The 0.1% resolution is one fifth of the tightest primary quality
deficit (0.5%); the look-back is five common persistence windows.

With `F_e(b)` the cumulative best sustained floor through update `b` over base-valid
`P`-windows, evaluate at block endpoints `b = tP` the relative late gain
`g_e(b) = [F_e(b) − F_e(b−L_ref)] / F_e(b)`, and freeze at the **first** block endpoint
where `g_e(b) <= epsilon_ref` **for all three evaluators simultaneously**:

```
b_ref = min { b : g_e(b) <= epsilon_ref for all e in {E1,E2,E3} }
Q_ref_e = F_e(b_ref)
```

This is a causal first-passage rule on the reference prefix. The offline engine must stop
its logical scan at the first qualifying `b_ref` and may not inspect later quality to pick a
different one. A later observer failure is reported but cannot revise a frozen reference.

**The decisive design element is that there is no fallback.** If no `b_ref` exists by
`B_ref = 3200`, the cell returns `REFERENCE_NOT_ESTABLISHED`; if a required solver
terminates first, it returns `REFERENCE_SOLVER_TERMINATION`. Because the cap never supplies
a reference *value*, no choice of cap can lower the quality bar. `B_ref` controls only
whether a reference exists, which is a visible censoring statement rather than a silent
change of denominator. That is precisely the property C2 demanded.

Reference stability is explicitly **not** optimizer stationarity and must not be called
convergence.

### Calibration accounting

Reference-phase updates and time are published as `N_reference` and `T_reference`, are fully
documented and reproducible, and are **never** charged to `k_enter`, `k_cert`, `T_enter`, or
`T_cert`. `TIMING_SPEC.md` gives `T_reference` its own calibration-cost panel.

A mandatory diagnostic reports what the superseded Phase-1A rule *would* have produced at
horizons 900, 1600, 2000 and 3200 wherever those prefixes exist. This tests C2 closure
empirically; it cannot select a different reference rule.

---

# WP3 — δ_R reassessed as a primary scientific parameter

**Disposition: ACCEPT (M1).** The audit's sensitivity is not a tolerance effect: crossings
move roughly 3–6× between 1% and 0.5%, and the fitted exponent moves from +0.145 to +0.479
while `R2_log` falls from 0.84 to 0.44. The threshold changes the answer more than the
method does. `δ_R = 1%` is therefore not declared to be the unique truth.

**Recommendation: (B) — the quality–effort family is the primary result**, with a single
landmark permitted only as a compact anchor that may never stand alone.

`QUALITY_EFFORT_SPEC.md` preregisters

```
q ∈ {0.980, 0.990, 0.995}      (deficits 2%, 1%, 0.5%)
```

as **co-primary**, not baseline-plus-sensitivity. These are exactly the Phase-1A baseline
and its two declared sensitivity values, elevated together; they were not chosen from any
cross-method ranking, and no ranking evidence at 98%/99.5% was consulted in selecting them.
Three is argued as the smallest scientifically useful set: two levels cannot distinguish a
curve from a line, and each additional level multiplies every table and figure while adding
freedom to select a favourable level after the fact. The 99% landmark may anchor prose and
compact plots.

The output is thereby a genuine relationship — **method-level iterations required versus
fraction of attainable sustained quality** — rather than a claim that one tolerance is
canonical. Where `p` depends materially on `q`, `SCALING_AND_FIGURE_SPEC.md` requires that
to be reported as a scientific result, not buried.

---

# WP4 — evaluator neutrality

**Disposition: ACCEPT (M4).** The audit's identification is accepted: E1 is Proposed's
interpolation up to floor values, E2 is Yuksel's piecewise `x^6` mass law, E3 is Olhoff's
`rho_min = 1e-3` clipped model, verified to 9–11 significant digits. E1 is no longer
described as an obviously neutral universal evaluator, and it is not swapped for E2 or E3,
which would only relocate the favouritism.

The mitigating result is incorporated explicitly and repeated wherever the symmetry
objection is raised: **the three evaluators agree within 0.429% and preserve the ordering
wherever all evaluator values are available.** E2 and E3 share the same piecewise `x^6`
mass law and differ only in stiffness floor. This bounds the practical size of the objection
without erasing it, and it is stated as endpoint evidence that does not automatically extend
to mid-trajectory states — which is why trajectory-level robustness is still required.

**Recommendation: option (A), implemented as a normalized minimum.** Acceptance requires the
relative-quality threshold under all three evaluators:

```
r_e(k)   = Q_e(k) / Q_ref_e
r_all(k) = min_e r_e(k)
S_q(k)   = [ r_all(k) >= q ]
```

This is exactly "require acceptance under all of E1/E2/E3", expressed as a single scalar.

Option (B) as literally written — `Q_robust(k) = min(E1(k), E2(k), E3(k))` in **absolute**
frequency units — was analysed and **rejected**. The three models carry different level
offsets, so an absolute minimum is dominated by whichever evaluator happens to sit lowest,
essentially always the same one, at every state and every mesh; that reintroduces a single
privileged model through the back door and makes the reference and the measurement depend on
an arbitrary units choice. The minimum over **dimensionless attainment ratios** has no such
degeneracy: each evaluator is compared only to its own frozen reference.

The residual pessimism is real and disclosed: the all-evaluator gate can only delay entry
relative to any single evaluator, and reference freezing likewise waits for the
slowest-stabilizing evaluator. The goal is robustness, so this cost is accepted and bounded
by publishing the per-evaluator endpoints beside it.

**E1/E2/E3 are preserved individually regardless**, as co-equal primary decompositions —
not sensitivities. For every `(method, mesh, q)` the study reports each evaluator-only
`k_enter_e/k_cert_e`, the robust pair, all three absolute endpoint qualities, and all three
ratios. Any disagreement in PASS/censoring or cross-method ordering is labelled
`MODEL_DEPENDENT`.

---

# WP5 — absolute quality differences are not hidden

**Disposition: ACCEPT (M3), numeric evidence corrected by N4.** Olhoff leads common raw-E1
endpoint `omega1` by 6.2–8.5% over Proposed and 5.9–7.7% over Yuksel across the eight meshes
with a complete method triple. Its 800x100 endpoint is `RUN_ERROR`/E1 `N/A`, so no
nine-mesh inference is available. R alone therefore cannot carry the paper.

Repairs to the paper logic:

- The **best-observed benchmark is now mandatory** (`REFERENCE_QUALITY_SPEC.md` Sec 8):
  `Q_BO_e,j = max_m Q_ref_e,mj`, with attainment reported at the same q levels. It remains
  a symmetric descriptive comparison and is never named A, absolute adequacy, or a
  requirement.
- **Main Table 1 carries quality**: E1/E2/E3 at entry, ratio to own reference, and ratio to
  best-observed sit in the same row as the counts. Main Table 2 exposes absolute reference
  quality directly.
- **Figure F1 is the primary scientific figure** and is a quality–effort representation:
  common spectral quality and sustained floors versus method-level iteration effort, with
  q-lines, entry/certification markers, native stop, and gate satisfaction. Figure F4 plots
  absolute reference/endpoint quality versus mesh with the corrected eight-mesh fact in its caption.
  F2 (counts) carries an in-axes note pointing at F3 (per-update cost).
- **No unqualified single number.** Any statement in an abstract, caption, table, or prose
  must name `q`, the evaluator semantics, and enter-vs-cert, and must carry absolute achieved
  quality in the same sentence, row, or adjacent panel. "Iterations to a proper result"
  without these qualifiers is prohibited.

**No `Omega_req` was manufactured.** A remains uninstantiated and is not required for R;
`A_NOT_INSTANTIATED` is printed. A is admissible only if a mesh-specific target with
independent provenance, content-hashed and predating production, is supplied — enforced by a
mechanical provenance gate that hard-fails on an absent or post-dated record. No study
trajectory may supply it. The two axes are named separately throughout: **relative maturation
efficiency** versus **absolute achieved solution quality**.

---

# WP6 — Yuksel factual error corrected

**Disposition: ACCEPT (M7).** `top99neo_inertial_freq.m:237` is `x = xPhys;` under the
comment "Use stage-1 outputs as initial guesses for stage 2", and line 242 deals `xPhys`
into the Stage-2 field. The Phase-1A statement that Stage 1's design variable "is not
returned into Stage 2" was false and is deleted, not softened.

The corrected statement in `ITERATION_ACCOUNTING_SPEC.md`: **Stage 1's filtered physical
field becomes Stage 2's design variable and Stage 1's displacement becomes the initial mode
estimate.** The design state is continuous across the handoff; only the raw/physical
identification undergoes a one-time re-filtering shift.

Stage 1 remains separately reported, on grounds that survive the correction:

- **distinct algorithmic role** — it prepares a mode estimate for the inertial stage;
- **distinct objective and update regime** — point-load compliance, not the
  inertial/eigenfrequency objective;
- **distinct timing characteristics** — no eigen-structure work, different cost per update;
- **methodological transparency** — a reader must be able to see how much of the
  chronological total precedes the objective actually under study.

The false rationale is not retained merely because the reporting decision survives.
Stage-2-only acceptance eligibility now rests solely on the objective-mismatch ground, which
Phase 1A also gave. The correction *strengthens* the chronological-sum justification, since
the density trajectory really is continuous across the handoff.

---

# WP7 — persistence revisited only where the audit requires

P was not casually redesigned. `PERSISTENCE_AUDIT.md` and findings Mo1, Mo3, F10 were the
inputs.

**Decision: keep common `P = 100`, with `P = 50/200` OAT rescans at every q level.** The
original justification survives in part and is re-stated honestly rather than repaired:
P=100 is a **convention inherited from Olhoff stabilization evidence and applied uniformly
to impose the same proof length on every method** — it is *not* a value derived from all
three methods, and Phase 1A's implication otherwise is withdrawn. Method-specific P remains
prohibited, because it would make the certification burden a tunable per-method quantity.

The repairs elsewhere do not disturb this. The repaired topology gate makes long persistent
windows attainable at fine meshes (longest repaired runs 925 at 640x80 versus 5 before), so
P=100 is no longer near-unsatisfiable. Horizon-independent references and the q-family change
*where* windows start, not how long proof must be.

The unequal proportional burden is mandatory context: `P−1` is roughly 30–93% of Proposed's
native run but about 6% of Olhoff's fixed horizon, and for Proposed certification may extend
past the native stop, so `k_native` is printed beside `k_cert` (Mo3).

**Both counts stay prominent.** `k_enter` is the primary retrospective maturation location
and leads count and scaling claims; `k_cert` is the paired prospective certification
location, kept at equal visual prominence in tables and quality–effort plots.

**The quality–effort curve uses persistent `k_enter`, with certified `k_cert` as its
companion panel.** Instantaneous crossings are diagnostic only and are never accepted
endpoints — a single-state crossing is exactly the transient the persistence rule exists to
exclude, and admitting it into the headline curve would reintroduce noise sensitivity at the
tightest q level where it is worst. Persistence semantics are identical across methods by
construction: the same P, the same window definition, the same base-valid requirement, with
only the method-specific validity condition differing, and that condition's satisfaction
index `k_gate` printed beside every endpoint (M8).

---

# WP8 — the primary scientific output

**Decision: (D), a specified combination, led by (C).**

The audit is right that "proper" has two dimensions. The study therefore reports:

1. **Primary — quality-versus-iteration curves (C)**: figure F1, per method and mesh, with
   E1/E2/E3 quality and sustained floors against native method-level updates, q-lines and
   endpoint markers. This is the object that answers the question directly.
2. **Co-primary tabulated landmarks (B)**: the `k_enter`/`k_cert` family at
   q = 98%, 99%, 99.5%, each row carrying absolute achieved quality.
3. **Mandatory absolute-quality exposure**: Main Table 2 and figure F4, reference and
   endpoint quality per evaluator with ratio-to-best-observed.

Option (A) — a single table at one relative threshold — is rejected outright by M1's
sensitivity. The experiment answers *how much optimization work is needed to obtain a given
fraction of the method's attainable quality*, and simultaneously shows *what absolute quality
the method ultimately attains*. **No scalar efficiency score may combine quality, counts, and
time**, and this prohibition is written into `QUALITY_EFFORT_SPEC.md` Sec 6 and
`SCALING_AND_FIGURE_SPEC.md` Sec 1.

---

# WP9 — timing architecture preserved

The auditor identified the timing architecture as one of the strongest parts of the protocol
and it was **not redesigned**. The three principles are intact: an observational phase
determines endpoints; clean lightweight fixed-horizon replays measure time; observer and I/O
overhead never contaminate timed work.

Changes are confined to what the repairs require:

- endpoints are now q-indexed, so replays run at every distinct robust `k_enter(q)` and
  `k_cert(q)`, deduplicating equal horizons;
- `T_reference` is added as separately reported calibration cost — the reference phase is
  real optimization work, is documented and reproducible, and is **never** folded into
  `T_enter`/`T_cert`;
- the no-mixing rule for the single-run descriptive fallback now spans methods, **meshes**,
  q levels, and endpoints (Mi3).

`T_gate_offline` and `T_observer_after_cert` remain disclosed but uncharged. Common evaluator
time is excluded equally for all three methods because it is experiment measurement, not part
of any native algorithm.

---

# WP10 — honest iteration accounting retained

The audit-validated distinctions are retained unchanged in substance:

- **Proposed** — OC method-level updates; and now the recorded fact (Mi5) that a Proposed
  iteration contains **no eigensolve** under the frozen solid-reference profile, against a
  frozen Olhoff eigensolve that was about 75% of outer-update cost at 800x100.
- **Yuksel** — Stage 1, Stage 2, and a chronological total that is explicitly qualified as
  chronological method-level update work, not homogeneous units.
- **Olhoff** — outer updates and LP calls, with genuine solver-internal iterations reported
  only where actually instrumented.

`nInner = 1` is never interpreted as one simplex/HiGHS iteration: `innerLoopLP.m` hard-codes
`st.nInner = 1` irrespective of solver work, so `sum(nInner)` is just the number of successful
LP calls. The 640x80 failure replay makes the distinction concrete — one `linprog` call, exit
flag 0, **38 reported LP solver iterations**, while the production convention would still
record one. No `outer + inner = total` metric is created.

---

# WP11 — scaling analysis repaired

`SCALING_AND_FIGURE_SPEC.md` is updated throughout.

- **Multiple q levels are carried into the fits**: the primary family is
  `k_enter(N_e | q) = C(q)·N_e^{p(q)}` at q = 98%, 99%, 99.5%, reported together, with
  E1/E2/E3-only fits as co-equal mandatory decompositions. Threshold dependence of `p` is
  not buried; where `p` changes materially with quality level **that is reported as a
  scientific result**.
- **`k_enter` scaling is primary; `k_cert` scaling is descriptive.** This is the explicit
  recommendation the WP asked for. Fitting `k_cert = k_enter + P − 1` to a power law fits a
  power law plus a constant, and the audit quantified the distortion at 32% on frozen Olhoff
  data, worst where counts are smallest — i.e. worst for Proposed. `k_cert` raw points remain
  a prominent companion; its fit, if shown, carries a convention caveat in the caption, and
  the pipeline verifies that fitting `k_cert − (P−1)` reproduces the `k_enter` fit exactly.
- **Common-support companion fits are mandatory** and cross-method exponent comparison
  outside common support is prohibited (M5). If common support has fewer than three meshes,
  the spec requires stating that no comparative exponent exists.
- **Leave-one-out `p` ranges accompany every fitted `p`** (Mo2), and preregistered
  `WEAKLY_IDENTIFIED` conditions (`R2_log < 0.80`; LOO range spans zero; LOO width exceeds
  `|p|`) prevent a poorly determined exponent from being quoted as a small one (Mo4).
- `C`, `p`, `R2_log`, `n_valid` and the fitted range are all retained, and are called
  **empirical scaling fits over the tested mesh range**. "Intrinsic complexity", "asymptotic
  complexity", "order-optimal", and unqualified "scales better" are prohibited.

---

# WP12 — tables and figures repaired

The package can no longer imply "fewer iterations = better method" without showing achieved
quality; `PROPOSED_TABLE_LAYOUTS.md` opens with that prohibition.

- **Main Table 1 (quality–effort landmarks)** — mesh, method, q, robust `k_enter`, robust
  `k_cert`, native stop, method-gate k, stage/chronological decomposition, E1/E2/E3 quality at
  entry, ratio to own reference, ratio to best-observed, status with subclass. Rows at
  q = 98/99/99.5 are co-primary. Yuksel keeps S1 / S2 / chronological total; Olhoff counts
  outer updates with LP calls footnoted; Proposed rows state the no-eigensolve fact.
- **Main Table 2 (quality and validity)** — reference freeze index, `Q_ref` under E1/E2/E3,
  stability gains, best-observed ratios, raw relative volume, support path, largest and
  aggregate detached physical area, `n_islands_all`, `A_sig`, method validity, status. Its
  caption carries the corrected eight-mesh fact.
- **Main Table 3 (timing)** — remains secondary: `T_enter`, `T_cert`, mean native update
  cost, with stage/method decomposition and reference cost in a separate block.

All eight mandatory figures are specified (F1 quality-versus-iterations; F2 `k_enter` versus
mesh by q; F3 mean update cost; F4 absolute quality versus mesh; F5 `T_enter` versus mesh;
F6 Yuksel stage decomposition; F7 Olhoff outer/LP decomposition; F8 accepted topology grid),
plus a descriptive `k_cert` companion. Compaction is recommended where redundancy exists:
**F2 and F3 are placed adjacent and share a caption** so counts are never read without cost,
and evaluator-component versions of F2 are companion panels rather than separate figures.
F7 is the one mandatory supplementary figure.

---

# WP13 — remaining MAJOR and MODERATE findings

All eight MAJOR findings are dispositioned above or here: **M1** (WP3), **M2** (below),
**M3** (WP5), **M4** (WP4), **M5** (WP11), **M6** (below), **M7** (WP6), **M8** (below).

**M2 — R's structural preferences.** ACCEPT. R rewards early plateau and trajectory noise
and penalises steady late improvement, and the methods sit at opposite ends of that axis:
Proposed and Yuksel terminate on a max-density-change test and plateau by construction, while
the selected Olhoff profile is documented as never becoming stationary. Phase 1A stated only
the weaker property. `QUALITY_EFFORT_SPEC.md` Sec 6 now states **all three** preferences
explicitly in the estimand definition, and mandatory sustained-floor and reference-stability
trajectories let a reader see whether a method plateaued, oscillated, or was still improving.
This cannot be engineered away — it is what "self-referenced" means — so it is disclosed and
counterweighted by the mandatory best-observed benchmark and F4.

**M6 — Yuksel Stage-1 cap change.** ACCEPT. Stage 1 hit its own 1000-update cap at 640x80,
720x90 and 800x100 in the frozen campaign; a 2000 budget lets it run longer and changes the
Stage-2 initial design at those three meshes. Raising a previously binding cap is not a
neutral safety horizon. Disclosed at both points of use — `ITERATION_ACCOUNTING_SPEC.md` and
`IMPLEMENTATION_REQUIREMENTS.md` Sec 3 — with the explicit statement that `N_stage1`, the
Stage-2 trajectory, and the chronological total at those meshes are **not comparable** to
frozen campaign values, and figure F6 marks them. Retaining the binding cap was rejected
because it would make Stage-1 reference stabilization unreachable by construction.

**M8 — asymmetric method-specific gates.** ACCEPT. Yuksel (Stage-2-only) and Olhoff (policy
stage 2, `N=2`, `gap12 <= 1%`) carry conditions that can only raise their counts; Proposed
carries none. The audit's recommendation is adopted exactly: **keep the asymmetry, make its
magnitude visible.** The first method-condition satisfaction index `k_gate` is reported beside
every endpoint, so a reader can see how much of `k_enter` is gate-imposed — the frozen 160x20
Olhoff policy trigger at outer iteration 245 being the worked example. Imposing false symmetry
would redefine the methods and was rejected.

**MODERATE findings affecting implementation semantics** — all nine are accepted and closed:
Mo1 (k_cert fit demoted, identity check added), Mo2 (LOO ranges), Mo3 (`k_native` beside
`k_cert`), Mo4 (`WEAKLY_IDENTIFIED` criteria), Mo5 (`SOLVER_TERMINATION` +
`GENERIC_LP_ITERATION_LIMIT_ONLY` subclass carried in every table and legend), Mo6 (T0 demoted
to a known strict diagnostic; permissive-direction FE-patch sensitivity substituted — see the
documented deviation), Mo7 (`a_res`/`r_common` retired entirely), Mo8 (LP instrumentation via a
diagnostic mirror **outside** the frozen `reproduction2007` tree, preserving
`repro2007_tree_hash.m`), Mo9 (historical float32 snapshot path explicitly permitted for
offline rescans on recorded verified equivalence, while new Phase-2 trajectories must be
double).

**MINOR findings** are handled as documentation corrections: Mi1 (even-`nely` precondition
stated), Mi2 (`H.rV` named as the gate's source quantity), Mi3 (no-mixing rule extended to
meshes), Mi4 (acceptance-engine compute and storage budgeted, and scaled for the added
reference phase), Mi5 (Proposed's no-eigensolve iteration recorded numerically).

---

# WP14 — expected-result firewall re-audit

The repaired protocol was re-read with the methods renamed **A**, **B**, **C** and the
question asked: does any remaining design choice systematically reward a particular kind of
method?

| Potential systematic reward | Status after repair | Mechanism |
|---|---|---|
| **Cheaper endpoint quality** | Not eliminated — inherent to a self-referenced estimand — but no longer concealed | R is explicitly relative-to-own-reference; absolute quality is mandatory in Main Tables 1 and 2 and figure F4; the best-observed benchmark is mandatory; the corrected 6.2–8.5% over Proposed / 5.9–7.7% over Yuksel gap across eight complete triples must be stated before new results, with Olhoff 800x100 `RUN_ERROR`/E1 `N/A` excluded |
| **Shorter reference horizon** | **Eliminated** | Reference generation is a separate trajectory with a method-independent first-passage stabilization rule and **no cap fallback**; a horizon can only determine whether a reference exists, never its value |
| **Native interpolation** | **Eliminated as a gate asymmetry** | Acceptance requires the threshold under all three evaluators via `min_e r_e`; per-evaluator endpoints are co-equal; E1's identity as Proposed's model is stated, not defended |
| **Smoother topology** | **Eliminated** | The gate tests gross pathology only: support connectivity plus no individually significant detached component, on a fixed physical scale derived from FE geometry rather than any native filter radius; aggregate speck area is diagnostic only |
| **Fewer algorithmic layers** | Not eliminated — it is the object of study — but not silently rewarded | Counts are declared to be method-level iteration effort in different units; per-update cost (F3) is adjacent to counts (F2) with an in-axes note; Proposed's no-eigensolve iteration and Olhoff's 75% eigensolve share are stated numerically; no `outer + inner` common metric exists |

The protocol remains valid under every outcome the WP enumerated. Concretely: it accommodates
Olhoff having the fewest outer iterations (counts are per-method units, never equated);
Proposed having the most method-level iterations (no ordering is assumed anywhere);
Yuksel having the lowest time (timing is secondary, platform-scoped, and separately reported);
Olhoff having the best quality (F4 and Main Table 2 exist to show exactly this, and the frozen
evidence already says so); **ranking changing with requested quality** (the q family is
co-primary precisely so this is visible rather than hidden); and **quality–effort curves
crossing** (F1 is the primary figure and a crossing is a reportable result, not an anomaly).

A `NOT_REACHED`, `REFERENCE_NOT_ESTABLISHED`, or `QUALITY_NOT_REACHED` outcome for any method
— including Proposed — is a valid publishable result. No design choice was tuned against a
ranking, and no threshold was selected after observing which method benefits: the three q
levels are the pre-existing Phase-1A baseline and sensitivities, `P` is unchanged, `A_sig` is
derived from FE geometry before any cross-method scan, and the reference constants are ratios
of quantities already fixed.

---

# WP15 — the repaired logical estimand

Frozen constants: `P = 100`; `L_ref = 500`; `epsilon_ref = 0.001`; `B_ref = 3200`;
`q ∈ {0.980, 0.990, 0.995}`; `A_sig = 0.01`; volume tolerance `1e-3`; Olhoff `gap12 <= 0.01`.

**State.** `X_mj(k)` is the return-equivalent physical density field after exactly `k`
completed method-level updates of method `m` at mesh `j`, including ordinary finalization had
the method stopped there. `X(0)` is recorded and never acceptable. `xb_mj(k)` is its
exact-count binary projection: `nSolid = round(Vf·Ne)` elements set solid in order of
decreasing density, exact ties broken by increasing global element index.

**Topology validity.**
```
C_required(k) = 1  iff a single four-connected component of xb(k) intersects
                       the Q4 element footprints of BOTH prescribed support nodes
a_sig(j)      = ceil( A_sig / A_e(j) ),        A_sig = 4·A_e(160x20) = 0.01
H_T(k)        = [C_required(k) = 1] AND [ max_c (detached component area) < A_sig ]
```
No aggregate-area veto. Requires even `nely`.

**Method-specific validity.**
```
H_method(k) = Proposed : TRUE
              Yuksel   : state lies in Stage 2
              Olhoff   : policyStage = 2 AND N = 2 AND gap12 <= 0.01
```
`k_gate` = first index at which `H_method` holds, reported beside every endpoint.

**Pointwise non-spectral gate.**
```
H_V(k)  = |mean(X(k)) − Vf| / Vf <= 1e-3          (relative form, source field H.rV)
H0(k)   = H_health(k) AND H_V(k) AND H_T(k) AND H_method(k)
```

**Reference quality** `Q_ref(m, mesh)` — from the **reference trajectory only**, per
evaluator `e ∈ {E1, E2, E3}`:
```
F_e(b)   = max over a of  min_{k∈[a,a+P−1]} Q_e(k),
           over windows [a, a+P−1] ⊆ [1,b] with H0(k)=1 throughout
g_e(b)   = [ F_e(b) − F_e(b−L_ref) ] / F_e(b),   evaluated at b = tP
b_ref    = min { b : g_e(b) <= epsilon_ref  for ALL e }
Q_ref_e  = F_e(b_ref)
```
If no such `b` exists by `B_ref`: `REFERENCE_NOT_ESTABLISHED`. If a required solver
terminates first: `REFERENCE_SOLVER_TERMINATION`. **No terminal-cap fallback exists**, which
is what makes `Q_ref` independent of the measurement horizon.

**Evaluator-robust spectral acceptance** — on the **measurement trajectory**, against the
frozen triplet:
```
r_e(k)   = Q_e(k) / Q_ref_e
r_all(k) = min_e r_e(k)
S_q(k)   = [ r_all(k) >= q ]
A_q(k)   = H0(k) AND S_q(k)
```

**Persistence and endpoints**, for each `q`:
```
k_enter(q) = min { a >= 1 : A_q(k) = TRUE for every k ∈ [a, a+P−1] }
k_cert(q)  = k_enter(q) + P − 1
```
Evaluator-only decompositions replace `S_q` with `[r_e >= q]`. Instantaneous crossings are
diagnostics and are never endpoints. `k_enter` is the primary maturation location; `k_cert`
is the paired certification location.

`Q_ref` no longer depends on any measurement horizon: it is produced by a separate trajectory
under a causal first-passage rule with no cap fallback, and the measurement engine receives
only the frozen, provenance-hashed triplet. A reader can reproduce every endpoint
classification from the above without interpretive judgment.

---

# WP16 — what the repaired protocol needs from Phase 2

Nothing below was executed. `EVIDENCE_AVAILABILITY_MATRIX.csv` carries the per-quantity
detail; this is the summary by class.

| Class | Quantities |
|---|---|
| **ALREADY AVAILABLE** | Frozen profiles and effective configurations; frozen final counts (as prior budget evidence only); Yuksel stage counts; Olhoff outer/`nInner` counts; Olhoff per-state densities (float32 snapshots), native spectra, policy/gap/`N`/LP flags and per-iteration timing at eight meshes; the endpoint evaluator table, with Olhoff 800x100 explicitly `RUN_ERROR`/E1 `N/A`; targeted-replay closure evidence |
| **DERIVABLE OFFLINE** | At the eight available Olhoff meshes: per-state E1/E2/E3 raw and binary spectra, exact-count topology metrics under the repaired gate, binary turnover, provisional q-level endpoints, and accepted-state images — all from existing snapshots once the constants are frozen, on the explicitly permitted historical single-precision path (Mo9). Olhoff 800x100 is unavailable and `UNVERIFIABLE_AT_PRESENT` |
| **NEEDS OBSERVATIONAL INSTRUMENTATION** | Return-equivalent per-state density recording for Proposed and Yuksel with state-index identity tests; Olhoff initialization/finalization timing boundaries; genuine `linprog` solver-internal iterations **via a diagnostic mirror outside the frozen `reproduction2007` tree** (Mo8); observer-on/observer-off prefix invariance checks |
| **NEEDS REFERENCE-PHASE RUN** | `Q_ref_E1/E2/E3` and `b_ref` for **all 27 cells**, plus `F_e(b)` trajectories, `g_e(b)` gains, the superseded-horizon diagnostic at 900/1600/2000/3200, `N_reference` and `T_reference`. This is new work created by the C2 repair. Olhoff's existing snapshots can supply a *provisional offline* reference for engineering validation, but not the production reference, because the reference trajectory must be a declared reference run under the frozen rule |
| **NEEDS TIMING REPLAY** | `T_enter(q)`, `T_cert(q)`, mean native update cost per method/stage, and per-method time decompositions, at every distinct endpoint horizon on the fixed reference platform |
| **NEEDS NEW OPTIMIZATION TRAJECTORY** | Complete instrumented measurement trajectories for Proposed and Yuksel at all nine meshes (the frozen campaign stored checksums, not fields); Olhoff measurement trajectories under the new recorder; the unavailable zero-byte 800x100 Olhoff artifact (`RUN_ERROR`/E1 `N/A`); and A endpoints, which do not exist and are not generated unless an independent `Omega_req` is provenance-locked first |

The immediate goal is methodology readiness, not data production. **None of these are
authorized by this document.**

---

# Delta trail

Superseded reasoning is retained rather than deleted:

- The T1 aggregate clause and `a_res = 5` / `r_common` are described in
  `TOPOLOGY_SANITY_SPEC.md` Sec 1 as superseded, with the reason and the evidence, before the
  replacement is given.
- The Phase-1A horizon-relative reference is described in `REFERENCE_QUALITY_SPEC.md` Sec 1
  as the defect being repaired, and Sec 7 requires a **mandatory diagnostic reproducing what
  it would have produced** at 900/1600/2000/3200.
- The false Yuksel handoff statement is replaced in place, with the corrected mechanism and
  the surviving rationale both stated (`ITERATION_ACCOUNTING_SPEC.md`).
- The δ_R baseline-plus-sensitivity structure is superseded by the co-primary q family, with
  the former baseline (99%) retained as a permitted compact anchor.
- E1's former description as a neutral universal evaluator is withdrawn explicitly rather
  than quietly rewritten.
- `SCALING_AND_FIGURE_SPEC.md` Sec 6 maps every existing performance artifact/concept to
  RETAIN / REDEFINE / SUPPLEMENT ONLY / RETIRE.

---

# Required final summary

**1. C1 disposition and repaired topology rule.** ACCEPT. The aggregate detached-area veto is
deleted; the hard gate is support-to-support connectivity AND no individually significant
detached component, with significance defined by a fixed physical area
`A_sig = 4·A_e(160x20) = 0.01` (a 2×2 Q4 patch on the coarsest production mesh), giving
`a_sig = 4…100` elements across the family. `a_res = 5` and the name `r_common` are retired,
not renamed. Aggregate area remains a mandatory diagnostic.

**2. C2 disposition and repaired `Q_ref`.** ACCEPT. Reference generation moves to a separate
trajectory. `Q_ref_e = F_e(b_ref)` where `b_ref` is the first block endpoint at which the
relative late gain over a 500-update look-back falls to ≤0.1% for all three evaluators
simultaneously. `B_ref = 3200` is a censoring boundary with **no fallback value**, so no
horizon can set the quality bar. Reference work is published as `N_reference`/`T_reference`
and never charged to `k_enter`/`k_cert`.

**3. Does R remain the primary estimand?** Yes, renamed **self-referenced maturation work**
and explicitly paired with absolute quality. A remains uninstantiated
(`A_NOT_INSTANTIATED`); the mandatory best-observed benchmark supplies the symmetric
cross-method quality comparison without pretending to be an engineering requirement.

**4. One δ_R or a quality–effort family?** The **family is primary**:
q ∈ {98%, 99%, 99.5%}, co-primary. A single landmark may anchor prose but may never stand
alone.

**5. Evaluator neutrality.** Acceptance requires the relative threshold under **all three**
evaluators, implemented as `min_e [Q_e(k)/Q_ref_e] >= q`. The absolute-units minimum was
analysed and rejected as covertly privileging one model. E1/E2/E3 endpoints are preserved as
co-equal decompositions; disagreement is labelled `MODEL_DEPENDENT`; the mitigating 0.429%
agreement with preserved ordering wherever values are available is stated wherever the objection is raised.

**6. Exposing the corrected eight-mesh Olhoff advantage.** The frozen common raw-E1 range is
6.2–8.5% over Proposed and 5.9–7.7% over Yuksel; Olhoff 800x100 is `RUN_ERROR`/E1 `N/A` and
is not inferred. Mandatory best-observed benchmark; absolute
E1/E2/E3 quality columns in Main Table 1; reference/endpoint quality in Main Table 2; figure
F4 with the fact in its caption; F1 as the primary quality-versus-effort figure; and a
prohibition on unqualified single-number statements.

**7. Corrected Yuksel semantics.** `x = xPhys` at line 237 carries Stage 1's filtered physical
field into Stage 2 as its design variable, with Stage-1 displacement as the initial mode
estimate: the design state is continuous, with a one-time re-filtering shift in raw/physical
identification. Stage-2-only eligibility now rests solely on objective mismatch.

**8. Persistence decision.** Common `P = 100` retained with `P = 50/200` rescans at every q,
re-described honestly as a uniform proof-length convention inherited from Olhoff evidence
rather than a value derived from all three methods. Method-specific P prohibited. Unequal
proportional burden disclosed.

**9. `k_enter`/`k_cert` roles.** `k_enter` is the primary retrospective maturation location
and leads count and scaling claims. `k_cert` is the paired prospective certification location,
equally prominent in tables and plots, but secondary for scaling because of the additive
`P−1`. Quality–effort curves use persistent `k_enter` with a `k_cert` companion; instantaneous
crossings are diagnostics only.

**10. Revised scaling strategy.** `k_enter(N_e | q)` fits are primary at all three q levels
with E1/E2/E3 decompositions; `k_cert` fits are descriptive with a convention caveat and an
identity check; common-support companion fits are mandatory and cross-method exponent
comparison outside common support is prohibited; LOO `p` ranges and `WEAKLY_IDENTIFIED`
labels are required; everything is called an empirical scaling fit over the tested range.

**11. Status of the 2 CRITICAL findings.** Both ACCEPTED and repaired at the design level:
C1 closed by evidence (repaired gate reproduced independently at four meshes);
C2 closed by construction, pending the reference-phase runs it creates.

**12. Status of the 8 MAJOR findings.** All eight ACCEPTED and addressed: M1 q-family;
M2 structural preferences stated and instrumented; M3 quality promoted to mandatory and into
Main Table 1; M4 all-evaluator acceptance; M5 common-support fits; M6 Stage-1 cap change
disclosed; M7 factual correction made; M8 `k_gate` reported beside every endpoint.

**13. MODERATE/MINOR closed or deferred.** All 9 MODERATE and all 5 MINOR are closed at the
design level; **none deferred**. Two carry documented deviations from the audit's suggested
mechanism (Mo6, Mo7), both stricter, both flagged for the delta auditor.

**14. Remaining scientific choices.** All are *made and frozen*, and all should be challenged
rather than left open: (a) `A_sig` as a 2×2 coarsest-mesh Q4 patch; (b) the reference
constants `L_ref = 500` and `epsilon_ref = 0.001`, and `B_ref = 3200`; (c) the three q levels;
(d) `P = 100` as an inherited convention; (e) retaining asymmetric method gates rather than
imposing false symmetry; (f) whether an independent `Omega_req` will ever be supplied — A
stays absent otherwise, which does not block R. Two evidence gaps remain and cannot be closed
by design: no Proposed/Yuksel trajectory evidence exists to validate the topology gate on
their fields, and the zero-byte 800x100 Olhoff artifact makes its topology row
`UNVERIFIABLE_AT_PRESENT` (`RUN_ERROR`/E1 `N/A`).

**15. Exact work required before implementation.** Delta audit of this package; then, if it
passes, freeze the protocol hash; then Phase A no-solve preflight including the acceptance
engine, exact-projection and component unit tests, and the diagnostic LP mirror with its
identity test; then Phase B smoke tests on non-production even-`nely` meshes; then Phase C
pilot at 240x30 with masked method identities. Only then reference-phase runs, measurement
trajectories, offline scan freeze, and timing replays. Compute and storage must be budgeted
for **two** trajectories per cell.

**16. Should the repaired protocol return to the independent auditor?** **Yes — mandatory.**
The auditor that issued C1 and C2 must receive this package for a delta audit before any
implementation. The delta audit should concentrate on: the `A_sig` derivation and its
permissiveness at fine meshes; whether the stabilization rule truly severs horizon dependence
or relocates it into `L_ref`/`epsilon_ref`; whether the all-evaluator minimum is adequately
symmetric; whether the two documented deviations (Mo6, Mo7) are acceptable; and whether the
Proposed/Yuksel topology evidence gap is tolerable before production.

---

# FINAL VERDICT

READY FOR INDEPENDENT DELTA AUDIT

*This is a request for delta audit, not a declaration of implementation readiness. No
implementation, optimization, production execution, or performance-campaign modification is
authorized by this document.*
