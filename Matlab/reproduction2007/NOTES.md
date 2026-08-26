# Reproduction notes — Du & Olhoff (2007)

Running record of what has been **settled by evidence**, what is a **reconstruction**,
and what is still open. Every claim here is backed by a script under `runs/`.

---

## 1. Olhoff & Du (2014) — checked first (CLAUDE.md §6 "cheapest possible win")

| Gap | Closed by 2014? |
|---|---|
| Erratum form of the subeigenvalue problem | **YES** — 2014 eq. (19d)/(20f,g) print `det\|f_sk'Δρ − δ_sk Δ(ω²)\| = 0` already corrected. Independent confirmation of SMO 34:545. |
| Filter radius | **NO** — the 2014 paper contains **zero** occurrences of "filter". |
| Multiplicity tolerance | NO |
| Move limit | NO |
| Mesh (NE) | NO |
| MMA vs LP | Restates *"can be solved using the MMA method (Svanberg 1987) **or** a linear programming algorithm"* — consistent with CLAUDE.md §5: MMA + full coupling is the baseline, LP is the option. |

Net: the erratum is now double-sourced; every other unknown in CLAUDE.md §4 stands.

---

## 2. Fig. 4a read from the clean PDF — CLAUDE.md §3 RESOLVED

Rendered at 600 dpi from `docs/…2007….pdf` p.100. **The first plotted marker is
iteration 1, not iteration 0.** Axis-calibrated readings at the first marker:

    ω₁ ≈ 71    ω₂ ≈ 245    ω₃ ≈ 428

So the ≈430 value in CLAUDE.md §3 was read correctly, and the paper genuinely has
**no mode between ~245 and ~428**.

### Resolution: mid-height supports **with axial restraint at both ends**

Initial design (uniform ρ=0.5), plane-stress Q4, from `runs/sweep_supports.m`:

| support | axial | ω₁ | ω₂ | ω₃ | verdict |
|---|---|---|---|---|---|
| mid | one  | 68.40 | 248.0 **E** | 253.4 B | two modes crammed at ~250; **order swaps with mesh** ✗ |
| **mid** | **both** | **68.40** | **253.4 B** | **420.8 E** | matches 68.7 / ~245 / ~428 ✓ |
| corner | one  | 64.53 | 168.4 E | 275.0 B | −6.1% ✗ |
| corner | both | 95.50 | 195.9 B | 363.1 E | +39% (arch action) ✗ |

(B = bending, E = extensional, at 160×20.)

This resolves **both** problems in CLAUDE.md §3: the spurious low extensional mode
was an artifact of the pin+roller idealization, and with axial restraint at both
ends the ω₂/ω₃ ordering is stable from 64×8 through 240×30 — the mesh-dependent
mode swap is gone, so J = n+N is no longer mesh-dependent.

**Caveat, recorded deliberately:** Figs. 2a and 3a *draw* the supports at the bottom
corners. The drawing contradicts the numbers. The numbers were followed; both
idealizations remain switchable (`cfg.support`, `cfg.axial`).

### Mesh
mid+both gives ω₁ = 68.75 (64×8), 68.62 (80×10), 68.40 (160×20), 68.32 (240×30)
against the paper's 68.7 — so the paper's mesh was ≈64×8–80×10. Per the run
constraints we work at 160×20 and validate at 240×30; the residual −0.4…−0.6% is
pure mesh convergence, not a modelling error. All three BC cases show the same
small bias (a: −0.44%, b: −0.36%, c: −0.36% at 160×20).

---

## 3. Verified numerically

- **Eq. (19) generalized gradients** — central-FD check on 3 modes × 5 elements,
  worst relative error 1.7e-4 (FD truncation limited), most 1e-6…1e-8.
  `runs/test_gradients.m` (a).
- **Eq. (25d) in the erratum form** — on a *genuinely* degenerate pair (square plate
  clamped all round, relative gap 2.8e-13) the 2×2 subeigenvalue problem predicts the
  eigenvalue increments to 1.4e-5 relative error. Dropping the off-diagonals gives
  11%–250% errors. **The off-diagonal coupling is not a refinement — it changes the
  answer.** `runs/test_gradients.m` (b).
- **Mass interpolation (4)/(4a)/(4b)** — CLAUDE.md §2's algebra confirmed: (4) jumps
  by 0.1 at ρ=0.1; (4a) is C⁰ (jump 7e-9); (4b) is C¹ (jump 2e-9, slope 1 both sides).
- **Element matrices** — Q4 and Q6 both rank 5 (= 8 − 3 rigid-body modes); element
  mass sums exactly to ρ·dx·dy·t per translational direction.

---

## 4. Reconstruction — the inner-loop convergence criterion

The paper's Fig. 1 asks only *"Increments Δρ_e converged?"* and never gives a test.
This is not a free choice — a naive test silently breaks the algorithm:

| criterion | behaviour |
|---|---|
| absolute, `max\|Δx_step\| < 1e-4` | at move=0.01 the inner loop **exits after 1 sub-iterate** having travelled 4.6e-5 → the outer loop then declares convergence at iteration 1. Silent no-op. |
| relative to move limit, `< tol·move` | still exits at the `minInner` floor for move ≤ 0.01, because the MMA step *grows before it decays*. |
| **relative to the accumulated increment**, `max\|Δx_step\| / max\|Δρ\| < tol` | **scale-invariant**: 95/99/89/106/120 sub-iterates for move = 0.1/0.05/0.02/0.01/0.005, reaching the move limit in every case. |

The third is what is implemented, flagged in the source as a reconstruction.

---

## 5. Efficiency observation (contradicts the §6 complexity expectation)

At 160×20 (NE=3200, J=n+N modes), measured shares of wall-clock:

    eigensolve  ~2–3 %      generalized gradients  ~0.1 %      inner loop  ~97 %

CLAUDE.md §6 lists the eigensolve as the dominant outer cost. At this mesh it is not
— the inner loop (≈90–120 MMA sub-iterates × O(NE) each) dominates by ~30×. The
crossover to eigensolve-dominance would need a far larger NE or many more modes.
`maxNumCompThreads(1)` throughout, as required.

---

## 6. The filter radius decides the multiplicity — main result so far

`r_min` is never stated in the paper. Sweeping it (LP route, 160×20, move=0.005,
tolMult=0.05) gives a **sharp transition**, not a gradual trend:

| r_min (el) | ω₁ | ω₂ | ω₃ | gap | Mnd | bimodal |
|---|---|---|---|---|---|---|
| 1.10 | **170.13** | 170.76 | 312.3 | 0.37% | 0.106 | **100%** |
| 1.20 | 168.24 | 168.60 | **286.01** | 0.22% | 0.132 | **100%** |
| 1.50 | 166.49 | 167.36 | 260.0 | 0.52% | 0.153 | **100%** |
| 2.00 | 163.77 | 172.02 | 297.1 | 5.04% | 0.201 | 0% |
| 2.50 | 159.49 | 167.52 | 318.0 | 5.04% | 0.282 | 0% |
| 3.00 | ~156 | ~164 | ~325 | 5.3% | 0.34 | 0% |

Paper: ω₁ = ω₂ = **174.7** (bimodal), ω₃ = **284.9**.

Same sweep at the conclusive mesh, **240×30**:

| r_min (el) | ω₁ | ω₂ | ω₃ | gap | Mnd | bimodal |
|---|---|---|---|---|---|---|
| 1.10 | 171.13 | 172.00 | 345.1 | 0.50% | 0.086 | **100%** |
| **1.30** | **170.47** | **170.87** | **285.19** | **0.23%** | 0.103 | **100%** |
| 1.50 | 169.63 | 179.10 | 295.8 | 5.59% | 0.117 | 0% |
| 1.80 | 167.87 | 172.77 | 257.3 | 2.92% | 0.120 | 100% |
| 2.20 | 166.98 | 178.99 | 291.6 | 7.19% | 0.138 | 0% |

**r_min = 1.30 at 240×30 is the best single reproduction: ω₃ = 285.19 against
284.9 (+0.1%), bimodal with a 0.23% gap, ω₁ = 170.47 (−2.4%).**

Note the best-fitting r_min is ≈1.2–1.3 *elements* at both meshes rather than a
fixed physical length (0.06 at 160×20 vs 0.043 at 240×30). The filter is therefore
acting as minimal nearest-neighbour smoothing, not as a physical length scale, and
the member thickness is being set by the mesh. That is a caveat on mesh-independence,
not a result to be proud of — but it is what reproduces the paper.

- At r_min ≥ 2 elements the optimum is **not bimodal at all** — the algorithm
  converges to a stable point with a 5% gap. CLAUDE.md §7 says the multiplicity is
  the sharper test; that test turns out to be governed by the one parameter the
  paper never reports.
- r_min = 1.20 reproduces **ω₃ = 286.01 against 284.9 (0.4%)**.
- The move limit is **not** the determinant: with r_min = 3, ω₁ = 155.6 / 156.1 /
  156.0 / 155.8 for move = 0.02 / 0.01 / 0.005 / 0.002. Step-converged, still not
  bimodal. So the failure at large r_min is not a step-size artifact.

### Independent validation of the FE model — the paper's own topology

Fig. 3a was digitized from the PDF (`runs/digitize_fig3.m`; the drawn domain
outline is located from the row profile, its 8 px pen width measured, and the
outline **erased** by inpainting from the first interior line; recovered domain
aspect 8.039 against the exact 8). Evaluated in our FE model:

| mesh | ω₁ | ω₂ | ω₃ |
|---|---|---|---|
| 160×20 | 163.2 | 167.6 | 280.3 |
| 240×30 | 166.5 | 168.7 | **287.0** |
| paper | 174.7 | 174.7 | **284.9** |

ω₃ matches to 0.7%. This separates the two possible causes of an ω₁ shortfall:
**the FE model and the support idealization are correct**, and any shortfall is in
the optimizer. (Binary projections of the digitized figure collapse to ω ≈ 2 —
thresholding a printed scan disconnects thin members — so the grey area-average is
the meaningful comparison.)

---

## 7. The LP route needs an actual LP solver — and is ~75× cheaper

Imposing eq. (22), `f_sk'Δρ = 0` for s≠k, **cannot be done inside MMA**. MMA is an
interior-point method; two-sided inequalities leave the feasible set with empty
interior. Measured: `subsolv` RCOND ≈ 9e-18, 97 771 singular-matrix warnings, and
the design freezes (max|Δρ| collapses from 0.02 to 0.0013 the moment N=2 engages).

Implemented instead as a genuine LP (`algo/innerLoopLP.m`, `linprog`), which is what
Krog & Olhoff (1999) and §3.5.3 actually describe. Cost per outer iteration:

| route | inner solves / outer | 300 outer iterations |
|---|---|---|
| MMA + full (25d) coupling | ≈ 90–120 MMA sub-iterates | ≈ 20 min |
| LP + eq. (22) | **1** linprog | **16 s** |

Both reach a similar design at r_min = 3 (ω₁ ≈ 156 either way), so at this mesh the
LP route buys ~75× at no accuracy cost. Note this inverts the CLAUDE.md §6 cost
model twice over: the inner loop dominates the MMA route (97% of wall-clock, vs the
eigensolve's 2–3%), and the LP route removes the inner loop almost entirely (then
the eigensolve becomes 70%).

**The MMA baseline does not converge once N ≥ 2**: the inner loop hits its cap
(300 sub-iterates, `conv = NO`) whenever the modes coalesce — exactly the
non-differentiability CLAUDE.md §5 predicts, since at the solution the Δλ_j
themselves coalesce and the eigenvectors of A become undefined. This is a property
of the reconstruction, and it is the concrete reason the LP route exists.

---

## 8. Undefined case (25b) — it actually happens

CLAUDE.md §5 says to log, not patch, the case where ω_J is itself multiple. It
occurs: **21 times** in the 1600-iteration 240×30 run (J = 2 and J = 3). Logged in
`res.log`, no patch applied.

---

## 8b. The residual ω₁ shortfall is FE discretization, not a worse optimum

One optimized design was evaluated as the *same physical field* on progressively
refined meshes (each element split k×k, densities copied — identical geometry, so
every difference is discretization). `runs/test_discretization.m`:

| mesh | r_min=1.10 design | r_min=1.20 design |
|---|---|---|
| 160×20 | 170.13 | 168.23 |
| 320×40 | 166.09 | 164.71 |
| 480×60 | 164.77 | 163.72 |
| 640×80 | 164.10 | 163.27 |

Richardson fit `ω₁(nely) = ω∞ + C·nely^(−p)` gives p = 0.99 and 1.29 (first order —
the design's boundary is pixel-jagged, not smooth). Extrapolated **back** to the
mesh range that reproduces the paper's own initial frequency to 0.1%:

| mesh | r_min=1.10 | r_min=1.20 |
|---|---|---|
| 64×8 | 181.9 | 181.7 |
| **80×10** | **178.0** | **176.8** |
| **96×12** | **175.3** | **173.8** |
| 160×20 | 170.1 | 168.2 |

**The paper's 174.7 sits inside that band.** So the −2.6% at 160×20 is not a worse
local optimum — it is that 160×20 is ~3% less stiff than the ~80×10 the paper's
ω₁⁰ = 68.7 points to. No optimization was run below 160×20; this is a post-hoc
evaluation of one fixed design.

Caveat, stated rather than buried: ω₃ does **not** match at the same mesh. The
r_min = 1.20 design gives ω₃ = 286.0 at 160×20 against the paper's 284.9 (0.4%),
but extrapolated to 80×10 it would read ≈305. ω₁ and ω₃ therefore do not both match
at one mesh — consistent with CLAUDE.md §7's warning that these optima are strongly
non-unique.

---

## 8c. Fig. 4 iteration history reproduced

The paper's Fig. 4 (2007 Fig. 4a / 2014 Fig. 4) was extracted at 600 dpi to
`docs/figs/paper_fig4_hist.png` and the run history plotted in the same style and
on the same axes (`algo/plotHistory.m`, `algo/compareHistory.m`). Comparisons:
`results/FIG4_definitive_vs_paper_80.png` (matched 0–80 axis) and `..._full.png`.

The **move limit sets the pace** of that figure, and it is unstated. At 240×30,
r_min = 1.3, LP route:

| move | coalescence at iter | ω₂ peak | behaviour |
|---|---|---|---|
| 0.05 | 13 | 310 @ 5 | too fast |
| 0.03 | 18 | 311 @ 7 | matches the paper's timing, but **breaks connectivity** |
| **0.02** | **27** | 312 @ 10 | **smooth, no dropouts — used for the figure** |
| 0.01 | 54 | 312 @ 20 | too slow |

Paper: coalescence ≈ 20, ω₂ peaks ≈ 325 @ 7, ω₃ peaks ≈ 527 @ 9, converged by ~60–80.

What is reproduced: the three-phase structure is the same — ω₁ rises monotonically
from 68, ω₂ rises to a peak then falls into ω₁, ω₃ rises to a peak then decays, and
ω₁/ω₂ **coalesce near iteration 20–25 and stay coalesced** (N = 2 for 375 of 400
outer iterations, bimodal in 100% of the last 50).

What differs, stated rather than smoothed over:
- our peaks land at iteration ~14 against the paper's 7–9, i.e. the paper's early
  steps are larger than any move limit that keeps our run connected;
- our ω₃ shows a sharp drop at iteration ~23–26 (a mode change) where the paper's
  declines smoothly;
- final ω₁ = 170.3, ω₂ = 175.1 — the pair straddles the paper's 174.7 rather than
  sitting on it.

### A failure mode worth recording: move limit vs connectivity

At move = 0.03 the run collapses to ω₁ ≈ 2 at iterations 47–62 and again after 88.
That is not a spurious low-density mode — eq. (4)'s ρ⁶ mass already suppresses
those (at ρ_min the stiffness/mass ratio is ~1e9, which pushes those modes *up*).
It is a genuine mechanism: the design has been cut into a disconnected island that
carries mass with no load path. The paper's Fig. 4 is perfectly smooth, so their
step control kept the design connected throughout. Logged, not patched.

---

## 9. Reproduction status

| quantity | paper | reproduced | mesh |
|---|---|---|---|
| ω₁⁰ (a) | 68.7 | 68.40 / 68.32 | 160×20 / 240×30 |
| ω₁⁰ (b) | 104.1 | 103.73 | 160×20 |
| ω₁⁰ (c) | 146.1 | 145.57 | 160×20 |
| ω₃ of the optimum | 284.9 | 286.01 | 160×20, r_min=1.2 |
| ω₁ optimum | 174.7 | 170.13 (−2.6%) | 160×20, r_min=1.1 |
| ω₁ optimum | 174.7 | **170.47** (−2.4%) | 240×30, r_min=1.3 |
| ω₃ of the optimum | 284.9 | **285.19** (+0.1%) | 240×30, r_min=1.3 |
| ω₁ optimum, same design read at ~80×10–96×12 | 174.7 | **173.8 – 178.0** | extrapolated, §8b |
| multiplicity at optimum | bimodal | **bimodal, 100% of the last 50 iters** | both |
| increase over initial | +154% | +148.7% | 160×20 |

Topology comparisons against the printed figure: `results/*_vs_paper.png`. The
architecture matches (lens envelope, end X-bracing, central void, diamond cells);
our design tapers to the mid-height support where the paper's keeps material at the
end corners — the residual difference.

---

## 10. Open

- ~~Remaining ~3% on ω₁~~ **explained** — FE discretization, see §8b. Our design
  read at the paper's own mesh gives 173.8–178.0 against 174.7.
- ω₃ and ω₁ do not match at a single common mesh (§8b caveat). Worth one more
  look: it may indicate the paper's design differs in the thin end members, where
  ours tapers to the mid-height support and theirs keeps full-depth ends.
- MMA baseline at small r_min still running; needed because CLAUDE.md §5 makes
  MMA + full coupling the reproduction and the LP route the labelled alternative.
- `filterMode` **swept** (LP, 160×20, r_min=1.2): diag → ω₃ = 286.01, all → 280.80,
  none → 419.97; all three stay bimodal. `diag` is marginally closer to the paper's
  284.9 but the two filtered branches are not decisively separated by this example.
  Not filtering at all is clearly wrong for ω₃.
- Cases (b), (c) optima; max-ω₂; gap problem (26); 3D plate; bimaterial: not started.
