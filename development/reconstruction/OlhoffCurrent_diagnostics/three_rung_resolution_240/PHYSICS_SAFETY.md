# PHYSICS SAFETY — does `move = 0.005` buy physics that `M_nd` hides at 240×30?

Phase 16. Rung 4 cannot be removed if it provides a material physics or
multiplicity benefit invisible in the headline metrics. This asks directly.

---

## 1. The inherited physics gate, over the three-rung prefix `[1, 284]`

The gate (controller study P10, verbatim): subspace size 2 throughout,
`ω₂ > ω₁`, all `ω` finite, no NaN/Inf, no non-converged inner solve.

| | prefix `[1, 284]` | full run `[1, 1358]` |
|---|---|---|
| subspace size `N = 2` at every iteration | ✅ | ✅ |
| `ω₂ > ω₁` at every iteration | ✅ | ✅ |
| all `ω₁`, `ω₂`, `M_nd` finite; no NaN/Inf | ✅ | ✅ |
| non-converged inner solves | **0** | **0** |
| **gate** | **PASS** | **PASS** |

The three-rung prefix passes the gate on its own, so nothing is lost by
terminating at `S3`.

`degen` — near-degeneracy hits inside the multiplicity-aware subspace — is
non-zero at essentially every outer iteration (284 over the prefix, 1358 over the
full run). That is the **expected** behaviour of this formulation: the
double-eigenvalue treatment exists precisely because the two lowest modes
cluster. It is **not** part of the frozen gate, here or in the study that froze
it, and is reported descriptively only.

## 2. Mode structure, `S3` versus `F`

| | `S3` @284 | `F` @1358 | change |
|---|---|---|---|
| `ω₁` | 167.038463 | 167.049693 | **+0.011230 (+0.00672 %)** |
| `ω₂` | 197.234272 | 197.029405 | −0.204867 (−0.1039 %) |
| relative gap `(ω₂−ω₁)/ω₁` | 0.180772 | 0.179466 | **−0.001306** |
| subspace size `N` | 2 | 2 | **0** |
| mode order | — | — | **unchanged** |
| minimum gap over `[S3, F]` | — | — | 0.179425 |

**No mode crossing, no multiplicity change, no loss of separation anywhere.**
The gap narrows by 0.0013 — `ω₁` rises slightly while `ω₂` falls slightly inside
the two-mode cluster — and remains at 0.179, two orders of magnitude clear of
degeneracy.

Under the inherited thresholds gap magnitude alone is **explicitly not** a
materiality criterion, because the objective is `ω₁`. Recorded so the reasoning
is visible rather than assumed.

## 3. Volume feasibility

| | `S3` | `F` | change | bar |
|---|---|---|---|---|
| volume | 0.499999552 | 0.499999201 | — | — |
| \|volume − 0.5\| | **4.48e-07** | **7.99e-07** | +3.51e-07 | worsening ≥ 1e-5 is material |
| acceptance gate \|volume − 0.5\| ≤ 1e-4 | **PASS** | **PASS** | | |

Both states sit more than two orders of magnitude inside the acceptance gate, and
rung 4's effect on feasibility is 28× below the materiality bar.

## 4. Final topology

`figures/F11_topology_S3_vs_F.png`.

| | value | bar |
|---|---|---|
| mean \|Δρ_e\| (`S3 → F`) | 0.002120 | 0.01 |
| RMS Δρ | 0.006131 | — |
| max \|Δρ_e\| | 0.06693 | — |
| Δ gray fraction | −0.000556 | 0.01 |
| Δ mid-density fraction | +0.000278 | 0.01 |
| **material?** | **no** | |

The mean per-element change is 4.7× below the bar and the gray/mid fractions move
by less than a thousandth. `S3` and `F` are the same structure.

## 5. No 240×30 production baseline exists

`baselines.json` covers 160×20, 320×40 and 400×50 only. Production-relative `ω₁`
and `M_nd` gates at this mesh are therefore reported **`UNAVAILABLE`** and are
**not** imputed from another mesh, interpolated, or replaced by a proxy. Every
absolute quantity and every `S3`-relative quantity in this study is unaffected;
only the production comparison is missing, and it is missing because the evidence
does not exist rather than because it was not computed.

## 6. Conclusion

> Does the final `move = 0.005` rung provide a scientifically meaningful mode or
> multiplicity refinement invisible in `M_nd` and `ω₁`?

**No.** Subspace size, mode order, mode separation and volume feasibility are
unchanged and well inside every gate at `S3`. The gap moves by 0.0013 — not a
materiality criterion, and in the same direction the objective moves.

**There is no blocking physics evidence for retaining rung 4 at 240×30**, and
this matches the finding at all three previously audited meshes.
