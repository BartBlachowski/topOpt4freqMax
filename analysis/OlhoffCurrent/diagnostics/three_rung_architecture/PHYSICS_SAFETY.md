# PHYSICS SAFETY — does `move = 0.005` buy anything the metrics hide?

Phase 17. The final rung cannot be removed merely because `M_nd` and `ω₁` barely
change, if it delivers a material physics or multiplicity benefit. This document
asks that question directly and answers it.

---

## 1. The preregistered physics gate (A6 = controller-study P10, verbatim)

> subspace size 2 throughout, `ω₂ > ω₁`, all `ω` finite, no NaN/Inf, no
> non-converged inner solve.

Evaluated over the **three-rung prefix** `[1, kE(3)]` — the only iterations the
three-rung policy would execute:

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| subspace size `N = 2` at every iteration | ✅ | ✅ | ✅ |
| `ω₂ > ω₁` at every iteration | ✅ | ✅ | ✅ |
| all `ω` and `M_nd` finite, no NaN/Inf | ✅ | ✅ | ✅ |
| non-converged inner solves | 0 | 0 | 0 |
| **A6 at S3** | **PASS** | **PASS** | **PASS** |

The same holds over the full four-rung run, so nothing is lost by stopping early.

`degen` — near-degeneracy hits inside the multiplicity-aware subspace — is
non-zero at essentially every outer iteration on every mesh (180 / 352 / 466 over
the three-rung prefixes; 219 / 1600 / 505 over the full runs). That is the
**expected** behaviour of this formulation: the double-eigenvalue treatment exists
precisely because the two lowest modes cluster. It is **not** part of the
preregistered gate, in this study or in the controller study that froze it, and
is reported descriptively only.

## 2. Mode structure at S3 versus F

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| `ω₁` S3 → F | 169.9766 → 170.0110 (**+0.02025 %**) | 166.4273 → 166.4189 (**−0.00504 %**) | 166.4427 → 166.4562 (**+0.00815 %**) |
| `ω₂` S3 → F | 171.5258 → 171.4302 (−0.056 %) | 203.5810 → 203.6763 (+0.047 %) | 201.5026 → 201.5433 (+0.020 %) |
| relative gap at S3 | 0.009120 | 0.223232 | 0.210643 |
| relative gap at F | 0.008347 | 0.223878 | 0.210789 |
| Δ gap (F − S3) | −0.000767 | +0.000634 | +0.000146 |
| subspace size at S3 / F | 2 / 2 | 2 / 2 | 2 / 2 |
| mode order | unchanged | unchanged | unchanged |
| minimum gap over `[S3, F]` | 0.008347 | 0.223229 | 0.210643 |

**No mode crossing, no multiplicity change, no loss of separation anywhere.** The
subspace size is 2 at S3 and at F on all three meshes, and `ω₂ > ω₁` throughout
both the prefix and the continuation.

At 160×20 the gap narrows slightly across rung 4 (0.00912 → 0.00835) as `ω₁`
rises and `ω₂` falls inside the two-mode cluster. Under the frozen threshold this
is explicitly **not material**: gap magnitude alone is not a materiality
criterion, because the objective is `ω₁`, and both states remain non-degenerate
with `N = 2`. At the fine meshes the gap is ≈ 0.21–0.22 and moves by under 0.0007.

At 320×40, notably, rung 4 moves `ω₁` in the **wrong** direction (−0.0084).

## 3. Volume feasibility

Gate A5: `|volume − 0.5| ≤ 1e-4`. Materiality: worsening by `≥ 1e-5`.

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| volume at S3 | 0.499996770 | 0.499998749 | 0.499999113 |
| \|volume − 0.5\| at S3 | 3.23e-06 | 1.25e-06 | 8.87e-07 |
| volume at F | 0.499999134 | 0.499999617 | 0.499999432 |
| \|volume − 0.5\| at F | 8.66e-07 | 3.83e-07 | 5.68e-07 |
| change across rung 4 | −2.36e-06 (**improves**) | −8.68e-07 (improves) | −3.19e-07 (improves) |
| **material?** | no | no | no |
| **A5 at S3** | **PASS** | **PASS** | **PASS** |

Every value is at least an order of magnitude inside the gate, and rung 4 improves
feasibility slightly on all three meshes — by amounts one to two orders of
magnitude below the 1e-5 materiality bar.

## 4. Final topology

`figures/F9_topology_S3_vs_F.png` shows S3, F and their difference per mesh.

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| mean \|Δρ_e\| (S3 → F) | 0.000424 | 0.001723 | 0.000208 |
| RMS Δρ | 0.001695 | 0.005056 | 0.000387 |
| max \|Δρ_e\| | 0.03697 | 0.04167 | 0.00393 |
| Δ gray fraction | −0.000625 | −0.000781 | 0.000000 |
| Δ mid-density fraction | 0.000000 | +0.000156 | 0.000000 |
| **material (bar 0.01)?** | no | no | no |

The largest per-element change anywhere is 0.042 at 320×40; the mean change is
6–48× below the bar. At 400×50 the gray and mid fractions are **bit-identical**
between S3 and F. On all three meshes the S3 and F designs are the same structure.

Distance to production, where production's density field survives:

| | 400×50 only |
|---|---|
| mean \|Δρ_e\| S1 → P | 0.13137 |
| mean \|Δρ_e\| S2 → P | 0.13247 |
| mean \|Δρ_e\| **S3 → P** | **0.13269** |
| mean \|Δρ_e\| F → P | 0.13284 |

S3 is essentially as far from production's design as F is; the large structural
difference is bought by rung 1, not by anything below it. The 400×50 production
density hash recomputed here matches `baselines.json` exactly
(`0d8b799c77b86f62c9113e88331052b0431c4917ad2740fd651834c040eedaaf`).

Production density fields for **160×20 and 320×40 are UNAVAILABLE**
(`baselines.json`: `rho_available = false`, raw `.mat` lost). Those distances are
reported as a gap and are **not** imputed.

## 5. Conclusion

> Does the final `move = 0.005` rung provide a scientifically meaningful mode or
> multiplicity refinement that is invisible in `M_nd` and `ω₁`?

**No.** Subspace size, mode order, mode separation and volume feasibility are
unchanged, uncontested and well inside every gate at S3 on all three meshes. The
gap changes by less than 0.0008 anywhere, in a direction that is not even
consistent across meshes, and gap magnitude is explicitly not a materiality
criterion under the inherited thresholds.

**There is no blocking physics evidence for retaining rung 4.** Stated plainly, as
the brief requires: rung 4 provides no material multiplicity benefit, no material
volume benefit and no material topology benefit on any primary mesh.

The reason this audit does not return `SUPPORTED` lies elsewhere entirely — in the
threshold-splitting finding of `RUNG_VALUE_DECOMPOSITION.md` §4 — and is not a
physics-safety failure.
