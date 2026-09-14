# MULTIPLICITY_WARNING_AUDIT — Part H

## 1. What the warning is

`olhoffSolve` computes `J = n + N` (n = 1, fixed subspace N = 2, so J = 3) and
raises

```matlab
multJ = (J+1 <= Jcalc) && abs(w(J+1)-w(J))/w(J) < multiplicity.tolerance
```

i.e. it flags that **ω₄ lies within 5 % of ω₃**. Constraint (25b) is written for
a *simple* ω_J; when ω_J is itself nearly multiple the constraint's premise
fails. The solver logs the iteration and continues — **no procedure is
defined**, by the paper or by this reconstruction. A disclosed gap, not a bug.

Two things are distinct and must not be conflated: the **cluster** ω₁/ω₂
(imposed at fixed N = 2, measured by gap12) and the **next mode** ω₃/ω₄ (what
this warning is about, and outside the treated subspace).

## 2. Canary measurements

### 480×60 three-rung

| | |
|---|---|
| warnings | **4 of 386 iterations (1.04 %)** |
| first warning | iteration 12 |
| by stage | stage 1: 4 of 308 (1.30 %) · stage 2: 0 of 39 · stage 3: **0 of 39** |
| overlap with the terminal declaration window (367–386) | **none** |
| gap12 at endpoint | 0.12980 |
| gap23 at endpoint | 1.16726 |
| gap12 range over the run | 0.00336 – 2.68475 |
| multiplicity N at endpoint | 2 |

The warning regime at 480×60 is **confined to early stage 1** and vanishes
entirely once the design organizes. The stopping decision was taken 374
iterations after the last warning, in a spectrally clean window.

### 800×100 three-rung

| | |
|---|---|
| warnings | **81 of 468 iterations (17.31 %)** |
| first warning | iteration 26 |
| **last warning** | **iteration 113** |
| by stage | stage 1: 81 of 390 (20.77 %) · stage 2: 0 of 39 · stage 3: **0 of 39** |
| overlap with the terminal declaration window (449–468) | **none** |
| iterations after the last warning | **355 (76 % of the run)** |
| gap12 at endpoint | 3.4875831182270845e−05 |
| gap23 at endpoint | 1.3616912123784417 |
| multiplicity N at endpoint | 2 |

The warnings form a **contiguous early transient** — iterations 26–113, during
mode coalescence — and then stop completely. `figures/FIG_7_multiJ_800x100.*`
shows the block and the cumulative fraction decaying across the rest of the run.

## 3. Answers to Part H's five diagnostic questions, at 800×100

| question | answer |
|---|---|
| appears early? | **Yes** — first at 26, in the coalescence phase |
| increases late? | **No** — last at 113; zero in the final 355 iterations |
| overlaps terminal E? | **No** — the declaration window 449–468 is clean |
| correlates with branch A/B behaviour? | **No** — all three declarations are branch B, 335+ iterations after the last warning; branch A never fired |
| correlates with objective/topology deterioration? | **Not demonstrable** — the regime resolves *before* the phase in which the endpoint is determined, so no timing relationship supports a link |

## 4. Legacy comparison

Legacy campaign (beta / four-rung / `designChange`, cap 400), from
`nine_mesh_campaign_audit/MASTER_TABLE.csv`; figure
`figures/FIG_B_legacy_next_mode_warning_regime.*`.

| mesh | NE | outer | warnings | % | gap12 | gap23 |
|---|---|---|---|---|---|---|
| 160×20 | 3 200 | 91 | 2 | 2.20 | 0.0145 | 1.081 |
| 240×30 | 7 200 | 104 | 3 | 2.88 | 0.1618 | 0.861 |
| 320×40 | 12 800 | 131 | 1 | 0.76 | 0.1074 | 0.982 |
| 400×50 | 20 000 | 139 | 2 | 1.44 | 0.0774 | 1.216 |
| 480×60 | 28 800 | 164 | 4 | 2.44 | 0.0697 | 1.333 |
| 560×70 | 39 200 | 190 | 4 | 2.11 | 0.0290 | 1.434 |
| 640×80 | 51 200 | 199 | 7 | 3.52 | 0.0090 | 1.475 |
| 720×90 | 64 800 | 223 | 43 | 19.28 | 0.0092 | 1.497 |
| 800×100 | 80 000 | 170 | 81 | 47.65 | 0.0060 | 1.734 |

Direct canary-vs-legacy comparison at the two canary meshes:

| mesh | legacy | three-rung canary |
|---|---|---|
| 480×60 | 4 of 164 (2.44 %) | 4 of 386 (**1.04 %**) |
| 800×100 | **81** of 170 (47.65 %) | **81** of 468 (**17.31 %**) |

The **absolute counts are identical at both meshes** — 4 and 4, 81 and 81. That
is not a coincidence and it is informative: both policies start from the same
uniform design and run stage 1 at move 0.04, so they traverse the same early
coalescence phase and accumulate the same warnings there. The two policies then
diverge in how long they continue afterwards, which is why the *fractions*
differ so much.

The consequence is the important part. The legacy 800×100 run stopped at
iteration 170 — only 57 iterations after its last warning — at M_nd 50.66 %. The
three-rung canary continued to 468, ending 355 iterations clear of the regime,
at M_nd 34.41 %. **The warning regime is a property of the early trajectory, not
of the controller**, and the controller determines only how far past it the run
gets.

## 5. What is still not established

* **No causality.** Nothing links warning incidence to endpoint grayness,
  objective loss or topology. At 800×100 the regime ends 355 iterations before
  termination, so the timing does not support a direct link either way.
* **No mechanism for the gap.** (25b) assumes a simple ω_J and no procedure
  exists for the multiple case, in the paper or in this reconstruction. This
  audit records incidence; it proposes no treatment.
* **Nothing was changed.** The multiplicity treatment, subspace size,
  `multiplicity.tolerance`, `eigen.maxCluster` and the diagonal-offset /
  off-diagonal choices are untouched, as `PREREGISTRATION.md` §3 requires.

## 6. A separate observation: gap12 becomes degenerate at fine mesh

Distinct from the next-mode warning, and not a defect. At the three-rung
endpoints gap12 is 0.2232 (320×40), 0.1298 (480×60), **3.49e−05** (800×100) —
ω₁ and ω₂ agree to five significant figures at the finest mesh. Legacy shows the
same direction (0.0697 → 0.0060 from 480 to 800).

A bimodal optimum is the **expected physics** of eigenfrequency maximization,
and the fixed N = 2 subspace treatment with diagonal offsets and off-diagonal
terms exists precisely for it; `multN` is 2 at the endpoint. It is recorded here
because it is a clear mesh trend — the optimum becomes bimodal only at fine
mesh — and because it must not be confused with the ω₃/ω₄ crowding that the
next-mode warning is about.

## 7. Verdict

```
NEXT_MODE_WARNING_MATERIAL_CONCERN
```

with its scope stated precisely, because the phrase could otherwise be
over-read:

**Material** because the incidence grows 17-fold between the two canaries
(1.04 % → 17.31 %) over a 2.8× refinement; because during those iterations
constraint (25b) is applied with its premise violated and **no procedure is
defined** for that case; because those iterations sit in the coalescence phase
that selects which topology the run converges to; and because it happens at
exactly the mesh where endpoint quality degrades most.

**Not** material to the termination decision, and this is equally firm: the
regime ends at iteration 113, the terminal declaration window 449–468 is clean,
gap23 = 1.36 at the endpoint, and branch A never fired. Nothing about the
stopping decision at either canary was taken inside a spectrally ill-posed
regime.

**No causal claim.** "Material concern" here means it warrants attention before
a nine-mesh campaign, not that it has been shown to cause the grayness. The
timing evidence — regime resolved 355 iterations before the endpoint — is if
anything evidence *against* a direct link, and is recorded as such.
