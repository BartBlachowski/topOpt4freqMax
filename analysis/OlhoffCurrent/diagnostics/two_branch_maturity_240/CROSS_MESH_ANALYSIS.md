# CROSS_MESH_ANALYSIS — 160×20, 240×30, 320×40, 400×50

The required Phase 13 table, and the one clean cross-mesh law it exposes.

160/320/400 quantities are taken directly from the hash-valid
`fixedmove_400_dynamics/evidence/fm_analysis.mat` — the same numbers those
studies reported. 240×30 is new and withheld.

---

## 1. The required table

| | **160×20** | **240×30** (withheld) | **320×40** | **400×50** |
|---|---|---|---|---|
| NE | 3 200 | **7 200** | 12 800 | 20 000 |
| `tol(NE) = 0.05·√(NE/3200)` | 0.050 | **0.075** | 0.100 | 0.125 |
| **fixed-move endpoint type** | high-amplitude cancelling | **low-amplitude cancelling** | cancelling (≈ tol) | coherent converged |
| **first Branch-A event** | **83** | **never** | **255** | never |
| **first Branch-B event** | never | **187** | never | **369** |
| **selected exhaustion event** | **83 (A)** | **187 (B)** | **255 (A)** | **369 (B)** |
| native-stop event | never | 147 | 216 | 369 |
| **β-stall first fires** | **79** | **92** | **130** | **138** |
| production first descent | 79 | *92 (inferred)* | 130 | 138 |
| M_nd at exhaustion | 13.233 | **12.839** | 13.024 | 16.159 |
| **`remUseful` at event** | **2.017 %** | **−0.711 %** | **−0.581 %** | **1.107 %** |
| `postRelImp` at event | 14.185 % | **0.029 %** | 0.162 % | 6.872 % |
| post-event M_nd | 13.23 → **11.36** (improves) | 12.84 → 13.45 (worsens) | 13.02 → 13.52 (worsens) | 16.16 → **15.05** (improves) |
| post-event ω₁ gain | +0.462 % | +0.0008 % | +0.008 % | +0.034 % |
| confirmation tail | 517 | **1013** | 945 | 831 |
| `boundFrac` at event | 0.0550 | **0.00000** | 0.00000 | 0.00000 |
| `cosθ` at event (median₂₀) | −0.635 | **+0.217** | −0.535 | +0.994 |
| `net/path` at event (median₂₀) | 0.487 | **0.847** | 0.500 | 0.973 |
| normalized amplitude `‖Δρ‖₂/tol` at event | 11.24 | **0.676** | 1.102 | 0.993 |
| `max|Δρ|/move` at event | 0.9995 | 0.4232 | 0.7039 | 0.2582 |
| **terminal `max|Δρ|/move`** | 0.9993 | **0.6807** | 0.6966 | 0.0308 |
| **terminal `‖Δρ‖₂/tol`** | **6.92** | **0.83** | **0.98** | **0.095** |
| terminal median `cosθ` | −0.951 | **−0.917** | −0.965 | +0.997 |
| terminal `net/path` | 0.090 | **0.202** | 0.132 | 0.992 |
| fixed-move final M_nd | 11.458 | 13.451 | 13.524 | 15.224 |
| fixed-move final ω₁ | 169.575 | 167.013 | 166.329 | 166.158 |

*240×30 production first descent is **inferred**, not run: the fixed-move and
production arms are bitwise identical until production first descends, and
production descends on β-stall, which first fires at 92 on this arm. The archived
scalar record (104 outer iterations, two agreeing campaigns) is consistent with
descending at 92 and converging ~12 iterations later — the same descent-to-stop
gap seen at 400×50 (138 → 139), 320×40 (130 → 131) and 160×20 (79 → 91). No
second run was made.*

---

## 2. The one clean law: β-stall precedes exhaustion, and the gap grows with refinement

| mesh | NE | β-stall (= production descent) | exhaustion event | **gap** |
|---|---|---|---|---|
| 160×20 | 3 200 | 79 | 83 | **+4** |
| **240×30** | **7 200** | **92** | **187** | **+95** |
| 320×40 | 12 800 | 130 | 255 | **+125** |
| 400×50 | 20 000 | 138 | 369 | **+231** |

Strictly monotone in `NE`, and **the withheld mesh lands exactly where the
sequence requires** — between 160×20's +4 and 320×40's +125. This was not fitted;
the exhaustion events come from a rule frozen before 240×30 was run.

This is the sharpest available statement of why the production trigger is wrong
and why it gets worse with refinement: β measures *objective* stationarity, which
arrives almost immediately after the coarse mesh's design stops improving but
long before the fine meshes' designs do.

## 3. Where does 240×30 sit? (Phase 12)

**It is a hybrid, and it is the reason the union is needed.**

* **Endpoint** — like 160×20 and 320×40: a cancelling cycle
  (`cosθ = −0.917`, `net/path = 0.202`).
* **Classification** — like 400×50: **Branch B**, because it first passes through
  a coherent amplitude-converged phase.
* **Amplitude scale** — its own: terminal `‖Δρ‖₂ = 0.83 × tol`, *below* its own
  threshold, where 160×20 sits at 6.92 × tol and 400×50 at 0.095 × tol.
* **Pathway** — new: **converge coherently, then re-excite into a low-amplitude
  cycle.** No training mesh did this.

So 240×30 does not resemble any single training mesh. It resembles 400×50 on the
way in and 160×20/320×40 at the end.

## 4. Is there a monotone regime progression in NE?

Partly, and the exception matters.

Terminal `‖Δρ‖₂/tol`: 6.92 → **0.83** → 0.98 → 0.095. Not monotone — 240×30 sits
slightly *below* 320×40. Terminal `max|Δρ|/move`: 0.999 → **0.681** → 0.697 →
0.031, again with 240×30 marginally below 320×40.

So refinement does drive the endpoint from "full-amplitude limit cycle" toward
"coherent convergence", but **not smoothly**: 240×30 and 320×40 are effectively
the same endpoint state (`≈0.7` move utilisation, `≈0.9` cancellation, amplitude
just under `tol`), and the transition to genuine convergence happens somewhere
between 320×40 and 400×50.

The practical consequence is that **the two-branch structure is not a
coarse-vs-fine dichotomy**. Both branches can be relevant at the same mesh at
different times — which 240×30 demonstrates directly.

## 5. Component coverage (Phase 14)

| | 160×20 | 240×30 | 320×40 | 400×50 | coverage |
|---|---|---|---|---|---|
| Branch A alone | **83** | never | **255** | never | 2 / 4 |
| Branch B alone | never | **187** | never | **369** | 2 / 4 |
| **Union** | **83** | **187** | **255** | **369** | **4 / 4** |

Each component covers exactly half; the union covers all four with one frozen
rule and no mesh-specific constant. `remUseful` at every event lies in
**[−0.71 %, +2.02 %]**.

## 6. What the union does *not* yet handle

* **Branch A is blind to low-amplitude cancellation.** At 240×30 the cancellation
  signature holds for 492 iterations but Branch A's `‖Δρ‖₂ ≥ tol` clause is
  satisfied simultaneously on only **6**. 320×40 was caught only because its
  cancellation began while amplitude was still marginally above `tol`.
* Consequently the union's success at 240×30 depended on Branch B firing first.
  It did so robustly (410 consecutive iterations of its predicate afterwards), but
  a trajectory that entered low-amplitude cancellation *without* a prior
  20-iteration coherent-converged window would satisfy neither branch. **No
  observed mesh does this** — it is a structural gap, not an observed failure.
* **160×20 remains the weak case**: 14.19 % relative M_nd improvement still
  available at the event, disclosed before the test.
