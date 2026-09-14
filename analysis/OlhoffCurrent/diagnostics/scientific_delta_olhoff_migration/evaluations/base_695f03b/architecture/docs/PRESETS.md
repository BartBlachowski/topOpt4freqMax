# PRESETS

Named realizations. A preset populates canonical configuration fields and
contains **no mathematics**, no run state and no solver call.

Every preset resolves through `olh.config.resolve`:

```matlab
cfg = olh.config.resolve('duOlhoffFrozenM4');
cfg = olh.config.resolve('projected', 'domain.mesh.nelx', 320, 'domain.mesh.nely', 40);
```

The mesh-scaled tolerance is re-derived after overrides, so changing the mesh
automatically rescales ε.

## Classification

| Preset | Classification | Historical label(s) |
|---|---|---|
| `duOlhoffFrozenM4` | SCIENTIFIC_PRESET | TMA, B0, REG160, "frozen M4", the conference realization |
| `duOlhoffMatureM4` | EXPERIMENT_PRESET | Bmature, R2 |
| `restorationLadderGuard` | EXPERIMENT_PRESET | R1 |
| `noDescentFixedMove` | EXPERIMENT_PRESET | nodescent |
| `pContinuationCoupled` | EXPERIMENT_PRESET | P1 |
| `pContinuationDecoupled` | EXPERIMENT_PRESET | PD1 |
| `pMassCompatible` | EXPERIMENT_PRESET | PM1 |
| `projectionIdentity` | EXPERIMENT_PRESET | D160 |
| `projected` | EXPERIMENT_PRESET | T160, T240, T320, T800 |
| `legacyBinaryDiagonal` | SCIENTIFIC_PRESET | the pre-M4 reconstruction |

No preset is an OBSOLETE_ALIAS: every historical realization is scientifically
distinct from the others, and the two candidates for "runtime only" — B0 vs
HISTORICAL (identical mathematics, different runner) and the mesh variants of a
single realization — are *not* presets at all. A mesh is an override.

`Bmature` appears in three audits with different iteration caps. Those are
**runtime** differences and resolve to the same preset plus a
`runtime.maxOuter` override.

---

## duOlhoffFrozenM4

```
mass model:       Eq. (4b), the C1 model                                    [A]
penalization:     p = 3, held FIXED                                         [A/C]
filter:           Sigmund (1997) SENSITIVITY filter                         [A]
  radius:         R = 0.06 physical                                         [C]
  applied to:     every f_sk, not only the diagonal                         [C]
projection:       disabled                                                   -
multiplicity:     fixed subspace, size 2, no classifier                      [C]
  (25d) form:     diagonal offsets diag(lambda_j - lambda_n) retained        [C]
  off-diagonals:  retained, full determinant                                 [A]
optimizer:        published MMA (Svanberg Sept-2007), on the increment        [A]
move policy:      staged ladder [0.04 0.02 0.01 0.005]                       [C]
  descends when:  the bound variable beta stalls over a window of 10          [C]
stopping:         ||d(design)||_2 < eps                                      [A/B]
  guard:          only on an iteration where the move limit did not change    [C]
  eps:            0.05 at 160x20, scaled 0.05*sqrt(NE/3200)                   [C]
provenance:       RECONSTRUCTION PRESET
```

**This is not a published Du–Olhoff realization.** It is the reconstruction the
conference results were produced with. Two of its choices depart from the
paper's own statements:

* §2.1 says p is "normally assigned values increasing from 1 to 3"; this preset
  fixes p = 3, on the numerical evidence that the reported initial
  eigenfrequencies fit p = 3 and not p = 1.
* The paper *draws* corner supports; its *numbers* fit mid-height supports with
  axial restraint at both ends. The numbers won.

It also carries a **known deficiency deliberately**: under a move ladder,
‖Δρ‖∞ ≤ mv_k, so an iteration that lowers the move limit mechanically lowers the
measured step with no change in the design. `settledMove` suppresses the symptom;
it does not remove the cause (`audit_termination_mesh_admission`, verdict
`S2_CONTINUATION_DEFECT`). **Fix architecture, not history.**

## duOlhoffMatureM4

```
parent:           duOlhoffFrozenM4
departure:        + stop.guards.maxDesignChange                              [D]
classification:   post-publication safeguard
```
Convergence is asserted only once `max|Δ(design)|` has fallen below `eps/√NE`.
Without it the frozen rule can stop while the design is still moving at a level
the ladder has merely stopped resolving.

## restorationLadderGuard

```
parent:           duOlhoffFrozenM4
departure:        + stop.guards.ladderExhausted                              [D]
```
Convergence is asserted only once no *remaining* ladder level exceeds `eps/√NE`.
A statement about the **schedule**, where `maxDesignChange` is a statement about
the **design**. They answer different questions and are separate fields.

## noDescentFixedMove

```
parent:           duOlhoffFrozenM4
departure:        move.policy = fixed, move.initial = 0.04                   [C]
```
The move limit is held at the ladder's own first level — no new constant. With a
fixed move the `settledMove` guard is vacuously true at every iteration, so the
stopping rule of §3.5.1 is evaluated exactly as written, with no schedule
artefact to suppress.

## pContinuationCoupled

```
parent:           duOlhoffMatureM4
departure:        + p schedule [1 2 3], driver = moveLadderStage
provenance:       running a p schedule is the paper's own stated practice   [A]
                  the schedule values are class B, the transition rule class C
```
p is indexed by the move-ladder stage, so the transition is the stall event the
move controller already computes and no new numerical constant enters.

## pContinuationDecoupled

```
parent:           pContinuationCoupled
departure:        driver = ownCounter                                        [C]
```
p advances on the same stall *event* but keeps its own index: while p is below
its final value the stall is consumed by the p controller instead of the ladder,
the move is restored to the ladder's first level and the ladder's re-arm clock is
reset. **Why it exists:** under the coupled driver p and the move limit cannot be
varied independently, so a p-continuation result is confounded with a
move-schedule result.

## pMassCompatible

```
parent:           pContinuationDecoupled
departure:        + mass continuation, low-p model = Eq. (2)                 [D]
```
While p is below its final value the printed linear model (2) is in force; the
terminal model (4b) resumes when p first reaches its final value, and the final
analysis always uses (4b). Rationale: (4)'s low-density cut-off exists to
suppress spurious localized modes arising when the stiffness/mass ratio collapses
at p=3, q=1 (§2.2); at p=1 that ratio is not small and the cut-off is arguably
unwarranted. **Both mass models are class A; scheduling between them is class D.**

## projectionIdentity

```
parent:           duOlhoffMatureM4
departure:        filter.type = density, projection enabled at beta = 0      [D]
classification:   CONTROL, not a treatment
```
At β = 0 the tanh operator is the exact identity, so what remains is only the
change of formulation: design variable z, **density** filter instead of the
sensitivity filter, sensitivities reaching z by the chain rule. This is what
separates the two class-D departures from each other — the filter switch (D2)
from the projection itself (D1). Without it, any difference between a frozen and
a projected run confounds the two.

## projected

```
parent conceptual basis: the frozen reconstruction (via duOlhoffMatureM4)
projection:       enabled, tanh Heaviside                                    [D]
  beta schedule:  1 -> 2 -> 4 -> 8, advancing on the outer convergence event [D]
  eta:            0.5                                                        [D]
filter:           DENSITY filter                                            [D]
sensitivities:    complete chain rule to z, applied to every f_sk and f_JJ   [D]
volume:           (25e) evaluated exactly at z+dz, value and gradient paired [D]
classification:   POST-PUBLICATION MODIFICATION / NEW REALIZATION
```

**Du & Olhoff did not publish this.** `projection`, `Heaviside` and
`density filter` occur **zero** times in Du & Olhoff (2007) and in Olhoff & Du
(2014), and the operator is absent from the Krog & Olhoff lineage; it postdates
the 2007 paper. Do not present any projected result as theirs.

Note the stopping-field consequence, now explicit: `stop.field` is the **design
variable**, so under projection the outer test monitors Δz, not Δρ_phys. The
physical change is recorded as `hist.dxPhys2`. This differs in *meaning* from the
frozen realization, where the design variable *is* the density.

`T800` is this preset at 800×100 — an override, not a separate realization.

## legacyBinaryDiagonal

```
parent:           duOlhoffFrozenM4
departures:       multiplicity.method = binary, tolerance 0.02              [C]
                  multiplicity.diagonalOffsets = false -> (25d) AS PRINTED   [A]
                  filter.applyTo = diagonal                                  [C]
                  move.policy = fixed at 0.05, no settledMove guard          [C]
classification:   SCIENTIFIC_PRESET
```
The other end of the option space, and the only preset that runs (25d) exactly as
printed — which assumes the N eigenvalues are *exactly* equal. Retained as a
scientific alternative, not as an obsolete alias, and it exercises code paths the
frozen realization never reaches.
