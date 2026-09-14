# PYMORPHOGEN_PORT_MAP

How the canonical MATLAB architecture lines up with a future Python port in
`/Users/piotrek/Programming/Python/pyMorphoGen`.

**Nothing is ported here.** This is a readiness review: which modules map onto
which future interface, where pyMorphoGen already has the abstraction, and which
MATLAB decisions would make the port harder if left alone.

---

## 1. What pyMorphoGen already has

Inspected at `source/design/`:

| pyMorphoGen | Shape | Relevance |
|---|---|---|
| `SimpInterpolation` (`simp.py`) | dataclass with `stiffness_scale`, `mass_scale`, and **both** derivatives | the right shape already: value and derivative together |
| `DensityFilter` (`filters.py`) | sparse `H`, `Hs`, `apply`, and an adjoint `chain` | the density-filter half; the *sensitivity* filter is a different operator |
| `DensityField` (`density_field.py`) | owns the design vector, no FEM knowledge | the design-variable/physical-density distinction has a home |
| `config.py`, `problem.py`, `builder.py` | dataclass configuration | the canonical schema maps onto dataclasses directly |
| `optimizers/mma.py`, `maths/mma.py` | MMA | the inner optimizer |

The MATLAB architecture was shaped so these map one-to-one.

## 2. Interface map

| Future interface | MATLAB module | Notes for the port |
|---|---|---|
| `MaterialInterpolation` | `olh.material.massInterpolation`, plus SIMP stiffness in `assemble2D` | **pyMorphoGen's `SimpInterpolation.mass_scale` is the LINEAR model only** — `M = M₀·[ρmin + ρ(1−ρmin)]`. Du & Olhoff's (4)/(4a)/(4b) are piecewise with a low-density cut-off and are **not** representable by it. This is the single largest gap. |
| `DensityTransform` | `filter/projectDensity.m`, `filter/projDensityField.m` | tanh Heaviside + the affine floor `ρmin+(1−ρmin)P`. Value and derivative are already returned together. |
| `Filter` | `filter/prepFilter.m`, `filter/applyFilter.m`, `filter/projChain.m` | **Two distinct operators.** pyMorphoGen's `DensityFilter.chain` is the *adjoint of the density filter*, not the Sigmund (1997) sensitivity filter, which is `H(ρ·df)/(Hs·max(1e-3,ρ))` — nonlinear in ρ and not an adjoint of anything. Porting one and calling it "the filter" would silently change the formulation. |
| `MultiplicityHandler` | `olh.multi.detect` + `algo/deltaLambda.m` | Two separable concerns: the detector, and the subeigenvalue problem (25d) with or without diagonal offsets. Keep them separate in Python too — the MATLAB coupling was an accident and is documented as such. |
| `Optimizer` | `algo/innerLoop.m`, `algo/innerLoopRho.m`, `mma_published/` | The inner problem's variables are `[Δρ; β]` — β is an MMA design variable, not a separate quantity. |
| `ContinuationPolicy` | `olh.move.limit`, plus the p and projection controllers in `olhoffSolve` | **Three controllers, not one.** See §4. |
| `ConvergencePolicy` | the stopping block of `olhoffSolve` | metric, threshold, three guards, continuation veto, cap, failure — six separable concerns. |
| `Configuration` | `olh.config.*` | schema → dataclasses; `validate` → `__post_init__` plus cross-field checks; presets → classmethods. |

## 3. What ports cleanly

* **The schema.** 80 fields with types, domains, defaults and provenance
  classes. It becomes nested frozen dataclasses almost mechanically, and the
  provenance class belongs in each field's metadata.
* **Presets.** Pure functions `cfg -> cfg` with no state, no mathematics and no
  I/O. They become classmethods or a registry.
* **Mass interpolation.** One function, value and derivative together,
  dispatching on a semantic model name, with the printed constants isolated and
  guarded. It is already the shape the port wants.
* **The multiplicity detector.** Stateless for `binary`/`subspace`; the state for
  `latch`/`hysteresis` is one scalar.
* **The move policies.** State is a small named struct.
* **The digest-based regression harness.** `anchorRecord`/`anchorDigests` are
  language-independent in concept: hash the IEEE-754 bytes of every scientific
  quantity, keep log text out of the equality standard. A Python port can be
  validated against the *same reference artifacts*.

## 4. What must not be flattened in the port

**Three continuation controllers, not one.** They differ in exactly the ways a
generic `ContinuationPolicy` would hide:

| | watches | advances | can veto convergence |
|---|---|---|---|
| move | a stalling signal over a window | the ladder index | no |
| penalization | the move controller's stall event | p | **yes**, until p is final |
| projection | the outer convergence event itself | β_proj | **yes**, it consumes it |

A single abstraction would need a "can this controller reject a convergence
decision" flag anyway, at which point it has not abstracted the difference — it
has relabelled it. Port them as three classes with a shared *reporting*
interface (`state`, `transitions`, `terminal`), not a shared control interface.

**The two filters are different operators.** See the `Filter` row above.

**Design variable vs physical density.** `DensityField` is the right home, but
the port must keep three named fields (`z`, `z̃`, `ρ_phys`), and the convergence
policy must *state* which one it monitors. This is precisely where the MATLAB
code silently changed meaning under projection.

**Status precedence.** `SOLVER_FAILURE > CONVERGED > CAP_HIT > STOPPED_OTHER`
must be explicit, not inferred from whether a loop broke early.

## 5. Gaps in pyMorphoGen that the port will hit

1. **Mass interpolation is linear-only.** `SimpInterpolation.mass_scale` cannot
   express (4)/(4a)/(4b). A `MassInterpolation` protocol with `eq2/eq4/eq4a/eq4b`
   implementations is needed, with the printed coefficients guarded exactly as
   `olh.material.massInterpolation` guards them.
2. **No sensitivity filter.** Only the density filter and its adjoint exist.
3. **No multiple-eigenvalue machinery.** The generalized gradients `f_sk`, the
   subeigenvalue problem (25d) in its erratum form, the diagonal-offset variant
   and the constraint-gradient reconstruction `∂Δλ_j/∂Δρ_e = Σ v_js v_jk (f_sk)_e`
   have no counterpart. This is the scientific core and the largest body of work.
4. **No bound formulation.** The inner problem maximizes β subject to (25b–f),
   with β among the design variables. `simp_mma_optimization.py` is a compliance
   driver.
5. **Determinism.** `eigSolve` pins ARPACK's start vector because mode ordering
   near a degeneracy is otherwise non-deterministic — and the degeneracy is the
   phenomenon under study. Any Python eigensolver needs the same treatment, and
   it must be *configuration*, not a hidden constant.

## 6. Sequencing a future port

1. Configuration: schema → dataclasses; validation; presets. Testable with no FEM.
2. `MassInterpolation` (four models) — validate against this tree's unit tests,
   including the continuity claims at ρ = 0.1.
3. `Filter`: both operators, kept distinct.
4. `DensityTransform`: tanh projection and chain rule.
5. FE model and eigensolver, with a pinned start vector.
6. Generalized gradients and (25d), with and without diagonal offsets.
7. Inner bound-formulation problem on MMA.
8. The three continuation controllers.
9. Convergence policy with explicit status precedence.
10. Regression against **the same anchor artifacts**, at the tolerance a
    cross-language port genuinely permits — which is *not* bitwise, and must be
    preregistered rather than discovered.

## 7. MATLAB decisions left alone that would hinder a port

Recorded, not fixed, because fixing them would change trajectories:

* `useMMA` selects the MMA variant by **mutating the global MATLAB path**. There
  is no Python analogue; the port should inject the optimizer.
* The inner loop hard-codes MMA's `a₀,a,c,d` and the β box `xmax=5`. These are
  reconstruction choices (class C) with no configuration path.
* `eigSolve`'s tolerance, iteration cap and Krylov factor are hard-coded; the
  canonical schema has fields for them but `olhoffSolve` still passes only the
  solver name, because plumbing them through would change nothing today and
  risks a trajectory change for no benefit. **The port should wire them.**
* The top88 filter's `max(1e-3, ρ)` normalization guard is published behaviour
  and must be carried over verbatim, not "cleaned up".
