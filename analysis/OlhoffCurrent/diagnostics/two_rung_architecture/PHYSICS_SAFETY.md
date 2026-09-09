# PHYSICS SAFETY — does anything below `move = 0.02` matter for the physics?

The question Phase 15 poses: do rungs 3 and 4 provide a scientifically
meaningful mode/multiplicity refinement that is invisible in `M_nd`? If they do,
that is blocking evidence for the two-rung architecture regardless of grayness.

---

## 1. The preregistered physics gate (A6 = controller-study P10, verbatim)

> subspace size 2 throughout, `ω₂ > ω₁`, all `ω` finite, no NaN/Inf, no
> non-converged inner solve.

Evaluated over the **two-rung prefix** `[1, kE(2)]` — the only iterations the
two-rung policy would execute:

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| subspace size `N = 2` at every iteration | ✅ | ✅ | ✅ |
| `ω₂ > ω₁` at every iteration | ✅ | ✅ | ✅ |
| all `ω` finite, no NaN/Inf in `ω₁`, `ω₂`, `M_nd` | ✅ | ✅ | ✅ |
| non-converged inner solves | 0 | 0 | 0 |
| **A6** | **PASS** | **PASS** | **PASS** |

`degen` (near-degeneracy hits inside the multiplicity-aware subspace) is non-zero
at essentially every outer iteration on every mesh, in the two-rung prefix and in
the four-rung run alike. That is the **expected** behaviour of this formulation —
the double-eigenvalue treatment exists precisely because the two lowest modes
cluster — and it is **not** part of the preregistered gate, in this study or in
the controller study that froze it. It is reported descriptively only:
141/219 (160×20 prefix/full), 313/1600 (320×40), 427/505 (400×50) hits.

## 2. Mode structure at S2 versus F

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| `ω₁` S2 → F | 169.8175 → 170.0110 (+0.1139 %) | 166.4163 → 166.4189 (+0.0016 %) | 166.4355 → 166.4562 (+0.0124 %) |
| `ω₂` S2 → F | 171.9416 → 171.4302 (−0.297 %) | 203.5528 → 203.6763 (+0.061 %) | 201.4595 → 201.5433 (+0.042 %) |
| relative gap S2 | 0.012513 | 0.223143 | 0.210436 |
| relative gap F | 0.008347 | 0.223878 | 0.210789 |
| Δ gap (F − S2) | −0.004160 | +0.000723 | +0.000353 |
| subspace size at S2 / F | 2 / 2 | 2 / 2 | 2 / 2 |
| mode order | unchanged | unchanged | unchanged |
| minimum gap over `[S2, F]` | 0.008347 | 0.223143 | 0.210436 |

**No mode crossing, no multiplicity change, no loss of separation anywhere.**
The subspace size is 2 at S2 and at F on all three meshes, and `ω₂ > ω₁`
throughout.

At 160×20 rungs 3+4 *narrow* the gap, from 0.0125 to 0.0083 — the two modes move
closer together as `ω₁` is pushed up and `ω₂` comes down. Under the frozen
threshold this is explicitly **not material**: gap magnitude alone is not a
materiality criterion, because the objective is `ω₁`, and both states remain
firmly non-degenerate with `N = 2`. It is worth noting only as the mechanism
behind the 160×20 `ω₁` gain: rungs 3+4 there are still trading `ω₂` for `ω₁`
inside the two-mode cluster.

At the fine meshes the gap is ≈ 0.21–0.22 and moves by less than 0.001 across
rungs 3+4. Nothing physical is happening below `move = 0.02` there.

## 3. Volume feasibility

Preregistered gate A5: `|volume − 0.5| ≤ 1e-4`. Materiality: worsening by
`≥ 1e-5` between two states.

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| volume at S2 | 0.499996991 | 0.499999904 | 0.499999879 |
| \|volume − 0.5\| at S2 | 3.01e-06 | 9.59e-08 | 1.21e-07 |
| volume at F | 0.499999134 | 0.499999617 | 0.499999432 |
| \|volume − 0.5\| at F | 8.66e-07 | 3.83e-07 | 5.68e-07 |
| change across rungs 3+4 | −2.14e-06 (improves) | +2.87e-07 (worsens) | +4.47e-07 (worsens) |
| **material?** | no | no | no |
| **A5 at S2** | **PASS** | **PASS** | **PASS** |

Every value is two to three orders of magnitude inside the gate, and every
change across rungs 3+4 is an order of magnitude below the 1e-5 materiality bar.
Volume feasibility does not depend on the lower rungs.

## 4. Final topology

`figures/F9_topology_S2_vs_F.png` shows S2, F and their difference for each
mesh. Quantitatively:

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| mean \|Δρ_e\| (S2 → F) | 0.001432 | 0.002036 | 0.000598 |
| RMS Δρ | 0.005960 | 0.005803 | 0.001236 |
| max \|Δρ_e\| | 0.1124 | 0.0497 | 0.0145 |
| Δ gray fraction | +0.000625 | −0.000625 | −0.000400 |
| Δ mid-density fraction | +0.000625 | +0.000156 | −0.000100 |
| **material (bar 0.01)?** | no | no | no |

The largest per-element change anywhere is 0.11 at 160×20, confined to a handful
of elements along interior member edges; the mean change is 7× below the bar. On
all three meshes the S2 and F designs are the same structure.

Distance to production, where production's density field survives:

| | 400×50 only |
|---|---|
| mean \|Δρ_e\| S1 → P | 0.13137 |
| mean \|Δρ_e\| **S2 → P** | **0.13247** |
| mean \|Δρ_e\| F → P | 0.13284 |

S2 is essentially as far from production's design as F is — the large structural
difference is bought by rung 1, not by anything below it.
Production density fields for **160×20 and 320×40 are UNAVAILABLE**
(`baselines.json`: `rho_available = false`, raw `.mat` lost). Those distances are
reported as a gap and are **not** imputed. The 400×50 production density hash
recomputed here, `0d8b799c77b86f62c9113e88331052b0431c4917ad2740fd651834c040eedaaf`,
matches the value recorded in `baselines.json` exactly.

## 5. Conclusion

> Do rungs 3 and 4 provide a scientifically meaningful mode or multiplicity
> refinement that is invisible in `M_nd`?

**No.** Subspace size, mode order, mode separation and volume feasibility are
unchanged, uncontested and well inside every gate at S2 on all three meshes.
There is **no blocking physics evidence** for retaining rungs 3 and 4.

The one place rungs 3+4 do change something real — `ω₁` at 160×20, +0.1139 % —
is an *objective* effect, not a multiplicity effect, and it is recorded as such
in `RUNG_VALUE_DECOMPOSITION.md` §2. It is the sole reason this audit does not
return `SUPPORTED`; it is not a physics-safety failure.
