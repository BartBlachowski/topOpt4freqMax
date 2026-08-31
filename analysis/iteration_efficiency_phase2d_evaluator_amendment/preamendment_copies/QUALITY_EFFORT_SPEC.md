# Phase 1C quality–effort specification

Status: binding design for independent delta audit; no result is produced here.

## 1. Primary output

R remains the primary estimand, renamed **self-referenced maturation work**. A single
1%-deficit landmark is no longer the primary scientific result. The primary output is the
small preregistered quality–effort family

\[
q\in\{0.980,\;0.990,\;0.995\},
\]

equivalently deficits `delta_R in {2%,1%,0.5%}`. These are exactly the Phase-1A baseline
and sensitivity values, elevated together after the audit showed 3–6x count changes. They
were not selected from future cross-method rankings. The 99% landmark may anchor prose
and compact plots, but it may never stand alone as the primary conclusion.

The experiment answers: **how many native method-level design updates are needed to enter
and certify a regime attaining each declared fraction of the method's independently
frozen sustained reference quality?** It simultaneously shows the absolute quality of
that reference and of every accepted endpoint.

## 2. Evaluator-symmetric acceptance

No evaluator is privileged. E1 matches Proposed's interpolation up to floor values, E2
matches Yuksel's, and E3 matches Olhoff's. E2 and E3 share the same piecewise `x^6` mass law
and differ only in stiffness floor, so the three-evaluator minimum is closer to two-way in
evidential terms. Frozen endpoint evidence mitigates but does not erase this symmetry issue:
their raw omega1 values differ by at most 0.429% and preserve the same ordering wherever all
three evaluator values are available.

For each evaluator, define the normalized reference attainment

\[
r_e(k)=Q_e(k)/Q^{ref}_e,\qquad e\in\{E1,E2,E3\}.
\]

Define robust relative attainment

\[
r_{all}(k)=\min\{r_{E1}(k),r_{E2}(k),r_{E3}(k)\}.
\]

The spectral gate at level `q` is

\[
S_q(k)=[r_{all}(k)\ge q],
\]

which is exactly equivalent to requiring the evaluator-relative threshold under **all
three** models. Taking the minimum of absolute frequencies was rejected because arbitrary
level offsets could let one evaluator dominate; the minimum of normalized attainment
ratios is symmetric and interpretable.

E1/E2/E3 results remain co-equal primary decompositions. For every `(method,mesh,q)`,
report each evaluator-only `k_enter_e/k_cert_e`, the all-evaluator robust pair, all three
absolute endpoint qualities, and all three ratios. Label a conclusion `MODEL_DEPENDENT`
whenever evaluator choice changes PASS/censoring or a cross-method ordering. Numeric
differences are shown even when the qualitative conclusion agrees.

## 3. Complete endpoint definition

Let `H0_mj(k)` be the pointwise non-spectral gate:

\[
H0=H_{health}\land H_V\land H_T\land H_{method}.
\]

For each quality level `q`,

\[
A_q(k)=H0(k)\land S_q(k).
\]

With common `P=100`:

\[
k_{enter}(q)=\min\{a\ge1:A_q(k)=1\;\forall k\in[a,a+P-1]\},
\]

\[
k_{cert}(q)=k_{enter}(q)+P-1.
\]

The same formulas define evaluator-only decompositions by replacing `S_q` with
`[r_e>=q]`. No instantaneous crossing is an accepted endpoint. Instantaneous quality
curves and first crossings are shown only as diagnostics; persistent `k_enter` and
certified `k_cert` are the reported landmarks.

## 4. Roles of the two counts

- `k_enter(q)` is the primary retrospective **maturation location** and leads count and
  scaling claims.
- `k_cert(q)` is the paired prospective **certification location**. It stays prominent in
  tables and quality–effort plots, but its power-law scaling is secondary because
  `k_cert=k_enter+P-1` adds a convention-dependent constant.

Any single-number statement in an abstract, caption, table, or prose must name `q`, the
evaluator semantics, and `enter` or `cert`, and must carry absolute achieved quality in
the same sentence, row, or adjacent panel. “Iterations to a proper result” without these
qualifiers is prohibited.

## 5. Required quality context

Every R landmark is paired with:

- E1/E2/E3 raw omega1 at entry and certification;
- each evaluator's frozen reference and attainment ratio;
- ratio to the mandatory best-observed reference for that mesh/evaluator;
- method-specific-gate satisfaction index, native stop, volume, topology metrics, and
  status;
- the sustained-floor/reference trajectory showing whether the method plateaued,
  oscillated, or was still improving.

Main figures include `Q_e(k)` and sustained-floor trajectories versus method-level update
count, with q-level lines and endpoint markers. Thus fast attainment of a lower reference
cannot be read as unconditional superiority.

Frozen context must be stated before new results: Olhoff's common raw-E1 endpoint `omega1`
is 6.2–8.5% above Proposed and 5.9–7.7% above Yuksel across the eight meshes with a complete
method triple. The Olhoff 800x100 endpoint is `RUN_ERROR` and E1 is `N/A`; it is explicitly
excluded, not inferred. The new reference study may update numerical endpoints; it cannot
erase or omit this pre-existing evidence.

## 6. Structural limitations of R

R is intentionally self-relative and has structural preferences:

- an early plateau can yield an early fraction-of-reference crossing;
- oscillation lowers a sustained-window floor relative to transient peaks;
- steady late improvement delays the crossing and may prevent reference stabilization.

The window minimum is retained because it prevents a transient peak from becoming the
reference. The separate stabilization procedure removes arbitrary measurement-horizon
dependence but does not make R an equal-quality comparison. Sustained-floor trajectories
and the mandatory best-observed benchmark expose these effects.

R does not support claims of equal quality, convergence, optimizer independence,
hardware-independent speed, transfer to other benchmarks, or intrinsic/asymptotic
complexity. No scalar “overall efficiency score” may combine quality, counts, and time.

## 7. Persistence decision

Keep common `P=100` and offline `P=50/200` sensitivity at every q level. P=100 is described
accurately as a convention inherited from Olhoff's stabilization evidence and applied
uniformly to impose the same proof length—not as a value derived from all three methods.
Frozen counts show `P-1` is roughly 30–93% of Proposed's native run versus about 6% of
Olhoff's fixed horizon; this unequal proportional burden is mandatory context.

Method-specific P is prohibited. A successor study may derive P from a preregistered
false-certification-rate target across frozen trajectories of all methods, but Phase 1C
does not use future production data to retune it.

## 8. Method-specific validity and gate-imposed floors

- Proposed: no additional gate; satisfaction index is `k_gate=1` if other pointwise
  conditions pass. A Proposed loop update contains no eigensolve under the frozen solid
  reference profile.
- Yuksel: only Stage-2 states are eligible because Stage 1 optimizes compliance under a
  point load, not the inertial/eigenfrequency objective. Report Stage-2 start as the gate
  satisfaction index.
- Olhoff: policy stage 2, native `N=2`, and `gap12<=1%` must hold. Report the first index
  where this complete method gate holds. Frozen 160x20 evidence puts the policy trigger
  at update 245, making the gate-imposed floor visible.

These asymmetric conditions are retained because false symmetry would redefine the
methods. Their count-raising effect is printed beside every endpoint.
