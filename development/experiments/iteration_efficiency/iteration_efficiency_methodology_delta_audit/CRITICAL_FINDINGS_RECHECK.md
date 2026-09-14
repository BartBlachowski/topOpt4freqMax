# Delta audit — CRITICAL findings recheck

Read-only verification. No file in the Phase-1C package, the original audit, MATLAB, or the
frozen campaign was modified. No optimizer was run.

Both CRITICAL findings are **CLOSED**. Neither closure rests on the author's ledger: each was
re-derived from the repaired specification text plus an independent recomputation over frozen
evidence.

---

## C1 — topology gate

### The repaired rule, read from the specification

`TOPOLOGY_SANITY_SPEC.md` Sec 5 and the normative expression in `ACCEPTANCE_GATE_SPEC.md`
Sec 6 agree:

```
H_T(k) = [C_required(k) = 1] AND [ max_c (detached component area) < A_sig ]
         # no aggregate-area veto
A_sig  = 4 * A_e(160x20) = 0.01
a_sig(j) = ceil( A_sig / A_e(j) )      -> 4, 9, 16, 25, 36, 49, 64, 81, 100
```

This is the "required connectivity AND no individually significant detached component" form
the repair was asked to investigate, with the aggregate clause removed rather than relaxed.

### 1-2. Method neutrality

`a_res = 5` and the label `r_common` are retired, not renamed. The scale is now derived from
FE element geometry on the coarsest production mesh, which is a property of the shared
benchmark discretisation and not of any optimizer. No native filter radius (Olhoff 1.3,
Proposed 2.0, Yuksel 2.5) enters the definition. **Genuinely method-neutral.**

### 3. Behaviour under mesh refinement — verified numerically

The original objection was that a constant element count silently tightens with refinement.
Recomputed as a share of solid volume (`Vf * 8 = 4.0`):

| mesh | `A_e` | `a_sig` | `A_sig` / solid | retired 5-element rule / solid |
|---|---:|---:|---:|---:|
| 160x20 | 2.50e-03 | 4 | 0.2500% | 0.3125% |
| 320x40 | 6.25e-04 | 16 | 0.2500% | 0.0781% |
| 640x80 | 1.56e-04 | 64 | 0.2500% | 0.0195% |
| 800x100 | 1.00e-04 | 100 | 0.2500% | 0.0125% |

The repaired threshold is **exactly mesh-invariant**; the retired rule swung by 25x. The
defensible-mesh-interpretation criterion is met.

### 4. Specks no longer dominate acceptance — verified

Independent union-find recomputation (`independent_gate_recheck.py`, deliberately not the
author's `scipy.ndimage` path) over the frozen Olhoff trajectory at 640x80, 1067 states:

| rule | pass fraction | longest run |
|---|---:|---:|
| T1 (per-component **and** aggregate < 5) | 0.56% | 5 |
| per-component < 5 only | 45.74% | — |
| repaired gate (`a_sig = 64`) | 95.03% | 925 |

The final state carries `det_max = 4`, `det_tot = 20` — **the original audit's own diagnostic
anchor, reproduced exactly**. The 0.56% vs 45.74% contrast confirms the aggregate clause, not
the per-component clause, was the binding constraint.

The author's Sec 6 table also reproduces exactly at 160x20 (66.52% / 957), 240x30 (93.07% /
1319) and 720x90 (97.81% / 1517).

### 5. Gross pathology still fails — verified by probe

Real states: of 1067 at 640x80, 53 are rejected — 16 on support and significant-detached
together, 37 on significant-detached alone. The first significant-detached rejection is at
iteration 55 with a **64-element** detached component. The gate is not vacuous.

Synthetic probes on the final accepted state (topology logic only; volume deliberately not
preserved):

| probe | result | required |
|---|---|---|
| severed mid-span | support fails | fail ✓ |
| left support isolated | support fails | fail ✓ |
| 12x12 detached blob (144 el) | **rejected**, 144 >= 64 | fail ✓ |
| 7x7 detached speck (49 el) | accepted, 49 < 64 | pass ✓ |
| 40 isolated 3x3 specks (aggregate 380 el) | accepted, max component 9 | pass ✓ |

The last row is the intended semantics made concrete: aggregate 380 elements — 76x the
retired `a_res = 5` — passes, while a single 144-element component fails. The gate answers
"is there a significant detached body?", not "is this geometrically clean."

**Residual, raised as N3 (MODERATE, non-blocking):** the deletion is operative, not theoretical.
Among *passing* states at 640x80 the aggregate detached area reaches 674 elements (2.633% of
solid volume; median 65, p95 148). That is defensible for a gross-pathology gate — the spectral
gate is the real backstop, since material spent on disconnected specks cannot support `omega1`
— but the quantity appears in no paper-facing table. It must be shown for accepted endpoints.

### 6. Required connectivity — traced to the problem definition

The benchmark fixes both translational DOF at the mid-height node of each end face and
prescribes nothing else. Verified in source: `model2D.m` case `'mid'` uses `rowSS = nely/2+1`
and **errors on odd `nely`**; `study_evaluate_design.m` fixes `2*nL+1, 2*nL+2, 2*nR+1, 2*nR+2`
at `jMid = round(nely/2)` in the first and last node columns.

Support-to-**support** is correct and support-to-load is correctly *not* imposed: Proposed
uses a distributed semi-harmonic load, Yuksel switches to a design-dependent inertial load,
and Olhoff's free-vibration eigenproblem has **no external loaded region at all**. There is no
common loaded region to connect to. Imposing one would redefine at least one method. The
even-`nely` precondition is now stated (Mi1).

### 7. Exact-count projection neutrality

Verified against the repository's own evaluator: `study_evaluate_design.m` uses
`nSolid = round(volfrac*numel(x))` and `sortrows([-x,(1:numel(x))'],[1 2])` — exact count,
volume-preserving, ties broken by increasing global index. The spec matches this exactly, and
my independent implementation reproduces it. The projection is applied identically to all
methods and depends on no method parameter. Gray-field methods are affected more than crisp
ones, which is disclosed (F06) and is a property of any binary topology test, not of this
repair.

### Deliberate departures

**Mo7 — retire `a_res` rather than rename it: ACCEPTED.** Strictly stronger than the minimum
correction I originally requested. The rename would have left a one-method calibration in
place with better labelling; retirement removes it. The quantitative check above confirms the
replacement actually fixes the objection rather than relabelling it.

**Mo6 — decline the filter-footprint sensitivity: ACCEPTED.** My original suggestion (a
per-component rule at the Proposed and Yuksel footprints, 9 and 21 cells) is incompatible with
the C1/Mo7 closure, which prohibits deriving any common topology scale from a native filter
radius. Adopting it would have reintroduced the defect. The substituted 1x1/3x3 FE patch
scales satisfy the finding's actual requirement — a sensitivity that can discriminate in the
**permissive** direction — which is what my original T0 objection was about. The author also
did the other half of my either/or: T0's known outcome is stated up front.

**C1: CLOSED.**

---

## C2 — horizon-independent reference quality

### The repaired construction

`REFERENCE_QUALITY_SPEC.md` Sec 3-6:

```
F_e(b)  = max over base-valid P-windows in [1,b] of the window minimum of Q_e
g_e(b)  = [F_e(b) - F_e(b-L_ref)] / F_e(b),  evaluated at b = tP
b_ref   = min { b : g_e(b) <= epsilon_ref for ALL e in {E1,E2,E3} }
Q_ref_e = F_e(b_ref)
```
with `P=100`, `L_ref=500`, `epsilon_ref=0.001`, `B_ref=3200`.

### Point-by-point

- **Reference-phase semantics.** Reference generation is a dedicated trajectory, separate from
  measurement. The measurement engine receives only the frozen, provenance-hashed triplet and
  is required to be structurally incapable of recomputing it from its own horizon
  (`IMPLEMENTATION_REQUIREMENTS.md` Sec 6, items 7-8).
- **Stopping rule.** Causal first-passage. The engine must stop its logical scan at the first
  qualifying `b_ref` and may not inspect later quality to choose a different one. A later
  improvement or failure cannot revise a frozen reference.
- **Censoring and safety budget.** `B_ref` is a censoring boundary only. **There is no fallback
  to the best floor at the cap** — the single most important property. Failure to stabilise
  yields `REFERENCE_NOT_ESTABLISHED`; a pre-`b_ref` solver stop yields
  `REFERENCE_SOLVER_TERMINATION` with backend subclass.
- **Method-specific iteration structure.** The count unit is declared per method (Proposed OC
  updates; Yuksel Stage-2 updates after the separately counted Stage 1; Olhoff outer updates)
  and the spec states these are not equal computational work.
- **Reproducibility.** Deterministic run from the frozen initialisation/profile, with
  fingerprint verification against the measurement trajectory at shared counts and at every
  reported endpoint; a mismatch is declared an implementation failure, not a new reference.
- **Separation of work.** `N_reference` and `T_reference` are published and explicitly never
  charged to `k_enter`, `k_cert`, `T_enter`, `T_cert` (`TIMING_SPEC.md` Sec 1).

### The decisive question

> If method labels were hidden, could one method receive an easier R target merely because of
> how long or how differently its reference trajectory was generated?

**No.** The generating rule is identical for all three methods and contains no method-specific
constant, no native stopping rule, and no assigned horizon. The Phase-1A causal chain
(shorter horizon → lower `Q_ref` → lower bar → earlier `k_enter`) is severed at its first
link: a shorter horizon cannot lower `Q_ref`, because the cap supplies no value. It can only
produce `REFERENCE_NOT_ESTABLISHED`, which removes the cell visibly rather than making its
target easier. Being censored is never an advantage.

The reference procedure does not introduce a *new asymmetric horizon*: `B_ref = 3200` is
uniform, and `L_ref`/`epsilon_ref` are ratios of already-frozen common quantities.

**Residual, correctly scoped to M2 rather than C2:** a method that plateaus early stabilises
early and freezes a lower reference, so it does get an easier target. But that follows from
the method's own behaviour under a common label-blind rule, not from an arbitrary assignment.
That is the disclosed, accepted nature of a self-referenced estimand (M2), and it is exposed
by the mandatory sustained-floor trajectories and the mandatory best-observed benchmark.

**C2: CLOSED** as a reference-construction defect.

### But the separation created a new defect

Separating the phases introduced an inconsistency that could not exist in Phase 1A, where the
reference lived inside the measurement horizon. The reference horizon is a uniform
`B_ref = 3200`, while measurement horizons are `900 / 2000 / 3200`, and nothing requires the
measurement horizon to cover `b_ref` (whose earliest possible value is 600). Worse, the single
permitted extension requires the sustained floor to be **still improving** — the exact logical
negation of the stabilisation condition defining `b_ref` — so a stabilised-but-uncertified
cell is denied its extension precisely because it stabilised.

Exposure is method-correlated: Proposed 900 vs a 3200 reference horizon (largest), Yuksel
2000, Olhoff 3200 (none). Raised as **N1 (MAJOR, BLOCKING)**. This does not reopen C2; it is a
budget-harmonisation defect in the repair's implementation contract.
