# Phase 1C acceptance gate and endpoint specification

Status: repaired design for independent delta audit; no Phase-2 run is authorized.

## 1. State convention

`X_mj(k)` is the return-equivalent physical density field after exactly `k` completed
method-level updates and ordinary finalization if the method stopped there. `X(0)` is
recorded but cannot be accepted. Phase 2 records global/native-stage indices, pre/post
mapping, and fingerprints and proves them against checkpoint-stopped runs.

## 2. Pointwise non-spectral gate

\[
H0_m(k)=H_{health}(k)\land H_V(k)\land H_T(k)\land H_{m,specific}(k).
\]

- `H_health`: every required solve/update through state k is finite with accepted flags.
- `H_V`: `H.rV=abs(mean(X)-Vf)/Vf <= 1e-3` on the raw field. The engine computes this
  explicit relative formula; it must not use the evaluator's absolute
  `volume_residual=mean(X)-Vf` field by mistake.
- `H_T`: exact-count binary support connectivity and no physically significant detached
  component under `TOPOLOGY_SANITY_SPEC.md`; aggregate detached area is diagnostic only.
- Proposed method condition: true.
- Yuksel method condition: state lies in Stage 2. Stage 1 is excluded because it optimizes
  point-load compliance rather than the Stage-2 inertial/eigenfrequency objective.
- Olhoff method condition: policy stage 2, native `N=2`, and
  `gap12=(omega2-omega1)/omega1<=0.01` on the same return-equivalent state.

Report the first method-condition satisfaction index `k_gate` beside every endpoint.
This exposes the asymmetric gate-imposed floor rather than pretending the formulations
have symmetric internal validity conditions.

## 3. Frozen spectral references

For E1/E2/E3, reference generation is completely defined by
`REFERENCE_QUALITY_SPEC.md`. The measurement scan receives the provenance-locked triplet

\[
(Q^{ref}_{E1,mj},Q^{ref}_{E2,mj},Q^{ref}_{E3,mj})
\]

from a separate reference trajectory. It cannot use its own safety horizon to modify the
triplet. `REFERENCE_NOT_ESTABLISHED` or a pre-freeze solver termination prevents R for
that cell; no terminal-cap fallback is allowed.

E1/E2/E3 are the Proposed-like, Yuksel-like, and Olhoff-like interpolation models,
respectively—not a neutral model plus two subordinate alternatives. Their frozen endpoint
spread is at most 0.429% with identical ordering wherever all evaluator values are available,
but trajectory timing must still be evaluated under all three.

## 4. Primary R quality family

For `e in {E1,E2,E3}`:

\[
r_e(k)=Q_e(k)/Q^{ref}_e,
\qquad r_{all}(k)=\min_e r_e(k).
\]

For every primary level

\[
q\in\{0.980,0.990,0.995\},
\]

define

\[
S_q(k)=[r_{all}(k)\ge q],\qquad A_q(k)=H0_m(k)\land S_q(k).
\]

The all-evaluator minimum is over dimensionless attainment ratios. It is equivalent to
requiring all three evaluator-specific thresholds and avoids the arbitrary scale dominance
created by `min(E1,E2,E3)` in absolute units. Evaluator-only `A_q^e` and endpoints are
also mandatory co-equal decompositions.

## 5. Persistence and exact endpoints

Keep common `P=100`. For each q:

\[
k_{enter}(q)=\min\{a\ge1:A_q(k)=1\ \forall k=a,\ldots,a+P-1\},
\]

\[
k_{cert}(q)=k_{enter}(q)+P-1.
\]

`k_enter` is the primary retrospective maturation location. `k_cert` is the paired
prospective certification location. Both stay in main tables; scaling interpretation
leads with `k_enter`. Instantaneous crossings are diagnostic only.

P=100 is a frozen equal-evidence convention inherited from Olhoff evidence, not a value
derived from all three methods. Use P=50/200 OAT rescans at all q levels. The additive
term preserves successful rank and absolute count differences but compresses ratios,
costs unequal seconds, and distorts a power-law exponent. Proposed certification may
extend past its native stop; therefore `k_native` is mandatory beside `k_cert`.

## 6. Normative assembled expression

The acceptance engine implements this expression without interpretive judgment:

```text
X_mj(k)       := return-equivalent raw physical field after k completed native updates
xb_mj(k)      := exact-count binary projection with stable global-index tie break

H_health(k)   := required solves/updates through k are finite and flag-successful
H_V(k)        := abs(mean(X(k))-Vf)/Vf <= 1e-3
H_T(k)        := support_component(xb(k)) exists
                  AND max physical area of each detached component < A_sig=0.01
                  # no aggregate-area veto
H_method(k)   := Proposed: TRUE
                 Yuksel: Stage 2
                 Olhoff: policyStage=2 AND N=2 AND gap12<=0.01
H0(k)         := H_health AND H_V AND H_T AND H_method

Qref_e        := first-passage stabilized reference from the independent reference run
                 under REFERENCE_QUALITY_SPEC.md; never from measurement horizon
r_e(k)        := Q_e_raw(k)/Qref_e, e in {E1,E2,E3}
S_q(k)        := min_e r_e(k) >= q, q in {0.980,0.990,0.995}
A_q(k)        := H0(k) AND S_q(k)

k_enter(q)    := first a>=1 with A_q(k)=TRUE for every k in [a,a+P-1]
k_cert(q)     := k_enter(q)+P-1, P=100
```

Every reference constant, q level, topology physical scale, method gate, and P value is
protocol-hashed before production.

## 7. Conditional A and best-observed benchmark

A exists only for an independently justified `Omega_req(mesh)` whose source/hash predates
production and is accepted by a mechanical provenance gate. Otherwise emit
`A_NOT_INSTANTIATED`.

The best-observed benchmark is mandatory but non-engineering. For each evaluator,
`Q_BO_e,j=max_m Qref_e,mj`; report q-level attainment and status. It is never named A,
absolute adequacy, or a requirement.

## 8. Status precedence and required qualifiers

For each q/evaluator semantics:

1. `PASS` / `PASS_WITH_LATER_SOLVER_TERMINATION` if finite certification exists;
2. `REFERENCE_SOLVER_TERMINATION` or `REFERENCE_NOT_ESTABLISHED` if reference fails;
3. `SOLVER_TERMINATION` before measurement certification, always with backend subclass;
4. `INVALID_TOPOLOGY` if topology is the limiting persistent gate;
5. `QUALITY_NOT_REACHED` if base-valid states persist but q does not;
6. `PERSISTENT_NONACCEPTANCE` for recurrent pointwise exits;
7. `OTHER`, reason mandatory.

For the known Olhoff LP event, every table/legend must carry
`GENERIC_LP_ITERATION_LIMIT_ONLY: dual-simplex-highs returned exit flag 0 in the recorded MATLAB version`.
This is not generalized to failure of the Du–Olhoff formulation.

## 9. Mandatory disclosures and OAT rescans

- quality levels 98%, 99%, 99.5% are co-primary, not baseline plus hidden sensitivity;
- E1/E2/E3 component endpoints and robust-all endpoint;
- P=50/100/200;
- volume tolerances 5e-4/1e-3/2e-3;
- topology 1x1/2x2/3x3 coarsest-FE physical scales;
- Olhoff native gap 0.5%/1%/2%;
- mandatory best-observed status;
- sustained-floor and reference-stability trajectories.

Use OAT rescans only. No Cartesian search, result-dependent preferred q/evaluator, or
new optimization solely for a sensitivity is allowed.

## Phase 2H evaluator amendment

At each state, Q is the lowest unanimously valid structural-mode frequency under Candidate
C for each of E1/E2/E3. Validity requires all three strict modal tests: `voidKE<0.5`,
`voidSE<0.5`, and `densityParticipation>0.5`; IPR is nonbinding. Search is adaptive from
3 modes by doubling without a scientific ceiling. No valid mode is a fail-closed
`STRUCTURAL_MODE_NOT_FOUND`. The binary projection cannot satisfy or fail a quality gate.
