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

---

## 11. The inner-loop MMA-call explosion — root-caused (2026-09-03)

The MMA baseline's ~90–300 mmasub calls per outer iteration (§7) had three
stacked causes, none of them the paper's algorithm:

1. **`mma/mmasub.m` is not Svanberg's published code.** The copy as received
   carries local modifications `move = 1.0`, `asyinit = 0.01` (published
   September-2007 values: `0.5`, `0.5`; verified against two independent
   verbatim copies — gistmeto/EduTO and arjendeetman's port of the smoptit.se
   distribution). The same edit is present in every other mmasub copy in this
   user's project lineage, so it propagated from a common ancestor.
   `asyinit = 0.01` shrinks the initial asymptote spread 50×; asymptotes grow
   at 1.2×/iter, so ~20 sub-iterates are burned recovering the published
   spread. Per the user's decision `mma/` stays byte-identical as received;
   `mma_published/` holds the published-constants copy and `cfg.mmaVariant`
   ('published' = baseline | 'asfound') selects at run time (`algo/useMMA.m`).
   The resolved mmasub path is recorded in every run.

2. **The sub-problem optimum is a bang-bang vertex, and MMA approaches
   vertices asymptotically.** Profiled at iteration 1 (case a, 160×20,
   move=0.01): the exact LP solution of sub-problem (25) has 100% of elements
   at ±move. MMA is functionally converged after ~10–20 sub-iterates (β within
   0.3% of the LP optimum) but the argmax is a near-degenerate plateau, so the
   iterate drifts (step ~1/k) and the relative step criterion at tolInner=0.01
   fires only at ~85. ~80% of all mmasub calls polished an already-converged
   increment.

3. **At N = 2 the (25d) sub-problem optimum sits exactly at the coalescence
   of the Δλ_j** (measured: Δλ₂−Δλ₁ driven to ~0.02% of λ), where the
   eigenvector basis of A flips freely (rotations of 60–85° between
   consecutive sub-iterates). Gradients chatter, relstep hovers at 0.05–0.15
   for 100+ sub-iterates, and tolInner=0.01 is reachable only near ~250–300 —
   the cap-hit mechanism of §7. β is nevertheless monotone-ish and within
   ~0.2% of its 300-iterate value by sub-iterate ~30.

**Resolution (recorded, swept):** tolInner = 0.05 with minInner = 5,
maxInner = 500 (safety, never observed to fire). Measured at 160×20,
move=0.02: ~8–20 mmasub calls/outer at N=1, ~15–40 at N=2, zero cap hits.
tolInner sweep 0.10/0.05/0.02/0.01 → mean nInner 7.8/16.9/44.1/81.7; looser
tolerance = smaller effective outer steps (trajectory pace trades against
move). tolInner=0.10 distorts the trajectory (26% deviation), 0.05 keeps it
within ~7% of the tol=0.01 reference at equal iteration count.

## 12. Outer convergence criterion — the printed norm

Fig. 1 tests ‖Δρ‖ < ε — a vector norm. The previous implementation tested
max|Δρ| < 1e-3, which with bang-bang inner increments equals the move limit at
every iteration and **can never fire** (the 800-iteration BASE_mma run rode
maxdrho = 0.0099 to the cap with ω₁ stagnant for hundreds of iterations —
pure stagnation, cap-hit finish). `olhoffOpt` now implements the Euclidean
norm (`cfg.outerNorm='l2'`; 'max' kept as labelled alternative) and records
`hist.dxNorm2`. ‖Δρ‖₂ decays as the design binarizes (only gray boundary
elements keep moving), which is what lets the printed test fire naturally.
ε is unstated in the paper → calibrated per mesh, recorded per run.

## 13. The post-coalescence joint ascent — the live discrepancy (2026-09-03)

The paper's Fig. 4 axes run to 80 (case a) and 100 (cases b, c) iterations, and
in every case the coalesced pair CLIMBS after merging: a 160→175, b 235→289
(+23%), c ~355→456 (+28%). Fig. 4b/4c panels extracted from the clean PDF to
`docs/figs/paper_fig4b_hist.png`, `paper_fig4c_hist.png`.

With the tuned inner loop (§11) the pre-coalescence trajectories reproduce well
(move 0.03–0.05 matches the paper's early pace), but the joint ascent stalls:
one-step probe at the stalled case-c state shows predicted Δλ gain +26.2,
realized +2.1 (8%) — a chatter equilibrium, not convergence. Findings:

- **Persistent MMA asymptotes refuted.** Running the sub-problem in design
  coordinates with solver state carried across outer iterations
  (`algo/innerLoopRho.m`, `cfg.innerVar='rho'`) makes every case WORSE
  (a: 156/174.5 gap 12%; c: oscillates wildly, gap 32%).
- **No move limit refuted (again).** move=inf destroys both tested cases
  (ω₁ → 0.1–0.3 within 5 iterations), reproducing the postmortem's box-vertex
  destruction in controlled conditions. The authors MUST have used an
  unstated step restriction; a hard move box is the minimal reconstruction.
- **tolMult smaller is worse:** 0.005 loses the joint ascent entirely
  (the (25b) J-constraint alone does not substitute for the (25c)/(25d)
  machinery). 0.05 is the working value — the paper's "very small tolerance"
  does NOT reproduce its own figures in our reconstruction.
- **Eq. (4)'s discontinuity at rho=0.1 is NOT 'negligible':** §2.2 claims
  (4)/(4a)/(4b) give negligible differences; measured, eq. (4) stalls case c
  at 394 while C¹-smooth (4b) restores ascent (410+, still rising at 100).
  ~100–140 elements cross rho=0.1 per iteration at move=0.05, each injecting
  an unmodeled mass jump.
- **Filtering ALL f_sk (not just the diagonal) also restores ascent** for
  case c (409 rising vs 394 stalled): consistent smoothing of the coupled
  model matters. CLAUDE.md §5 said 'which one reproduces Fig. 4a is a result'
  — for case c the answer is 'all' (or 4b, or both).
- **Filter radius does not rescue case c** (rminEl 1.6/2.0 at eq. (4): 362–383).
- Stalled case-c topology is a waisted-lens (case-a-like) family; the paper's
  Fig. 3c keeps full-depth edge chords with a continuous X-lattice web. The
  stall is basin lock-in; 4b/'all' unlock the restructuring.

## 14. Paper's own printed designs, digitized and evaluated (b, c)

Extending §6b's control to Figs. 3b/3c (`runs/digitize_fig3.m`, grey mode):

| design | 160×20 | 240×30 | paper claims |
|---|---|---|---|
| Fig. 3a | 163.2/167.6/280.3 | 166.5/168.7/287.0 | 174.7 bimodal |
| Fig. 3b | 254.8/273.5 | 253.8/274.1 | 288.7 bimodal |
| Fig. 3c | 231.6/422.9 | 230.5/418.2 | 456.4 bimodal |

The paper's own 3b design digitizes to **255/273 — our optimizer's 268/283 at
160×20 matches or exceeds the printed design**, so the case-b "shortfall" vs
288.7 is print/digitization/mesh, not the optimizer. The 3c digitization
collapses (231.6): the printed design's fine X-web — including the *gray*
central X visible in the figure — does not survive scan thresholding; no
verdict for c by this route.

## 15. lambda-tilde: the paper says omega_n^2, not the cluster mean

§3.5.1: "In the second step of the main loop, we set lambda~ = omega_n^2".
`olhoffOpt` used mean(lambda(cluster)); fixed to lambda_n (identical at exact
degeneracy, up to tolMult apart at detection). Measured effect at tolMult=0.05:
within run-to-run noise (case b 267.7 vs 268.9), but the printed form is kept.

## 16. Single mmasub call per outer iteration — the architecture that shows
## the paper's convergence signature

Testing the classical MMA-driver architecture (one mmasub per outer iteration,
asymptotes/xold persistent across iterations, fresh FE data each step;
`cfg.innerVar='rho'`, `cfg.maxInner=1`) against the text-faithful multi-call
frozen-sub-problem loop:

|  | multi-call frozen (25) | single-call persistent |
|---|---|---|
| stability at move=inf | destroyed (ω→0.1) | **stable, all cases** |
| history shape | chatter, flat-noisy tail | **smooth, decaying steps** |
| ‖Δρ‖₂ tail | 0.24–1.4 floor, never decays | **decays to 0.06–0.34** |
| printed ε-test ‖Δρ‖<ε | can never fire | **fires naturally** |
| coalescence gap | 2.7–8% | **0.4–2.9%** |
| ceilings at 120 it (a/b/c) | 166–170 / 268 / 406–420 | 165 / 253 / 380 (still rising) |

The single-call runs show every qualitative signature of Fig. 4 (fast early
steps — case b ω₁ 104→191 by iter 5, matching the paper's 104→185 by 2–3;
smooth monotone approach; genuinely tight bimodality; convergence inside the
80/100-iteration budget) but settle a few percent lower. The multi-call loop
reaches higher ω₁ but cannot terminate naturally and never tightens the pair.
Note (25d)'s off-diagonal machinery is inert in a single call from Δρ=0 (A=0,
basis undefined → diagonal f_jj); it only acts in multi-call inner loops.

**Verdict on the text-vs-code question:** §3.5 describes iterating the frozen
sub-problem (25) to convergence; the figures behave like the classical
one-call-per-iteration MMA driver. Both realizations are implemented and
labelled; which one the authors ran cannot be settled from the paper alone.

## 17. Partial inner convergence without a move limit is the worst of both

Completing the architecture matrix: maxInner=5 (a few mmasub calls per outer
iteration, persistent asymptotes, move=inf) destroys all three cases —
a: 141.8 (dips to 5.6), b: 199.0 (dips to 0.2), c: 69.9 (dips to 0.2), with
||drho||_2 in the 2-11 range throughout. Partial convergence of the frozen
sub-problem takes a large step in a direction the frozen model only trusts
locally, with nothing to restrict it.

The step restriction must come from EITHER an explicit move box (the 'strict'
profile) OR the MMA asymptotes acting alone with exactly one call per FE
re-analysis (the 'fig4' profile). Mixing them — a few calls, asymptotes not yet
adapted, no box — is unstable. This closes the architecture matrix:

| calls/outer | step control | outcome |
|---|---|---|
| ~15-40 (to inner conv.) | hard move box | chatters, eps can't fire, higher w1 |
| ~15-40 (to inner conv.) | none (move=inf) | destroyed |
| 5 (partial) | none (move=inf) | destroyed |
| **1** | **asymptotes only** | **stable, smooth, converges naturally** |

## 18. Fig. 4 shape vs Fig. 3 values — they are NOT reachable together (2026-09-03)

The user's question — can we get convergence in 80/100 iterations AND a plot
matching Fig. 4? — is answered separately for shape and for value.

Fig. 4a/4b/4c read from the clean PDF (600 dpi, `docs/figs/paper_fig4*.png`):
case a runs to 80 iterations, b and c to 100. In every case omega_2 RISES to a
peak then DESCENDS into omega_1, omega_3 rises to a peak then decays, and the
coalesced pair keeps climbing afterwards.

**Shape: reproduced.** Single mmasub call per outer iteration, persistent
asymptotes, move box 0.10, mass (4b), all f_sk filtered:

| feature | paper (a) | ours (a) |
|---|---|---|
| omega_2 peak | 325 @ 7 | 296 @ 8 |
| omega_3 peak | 527 @ 9 | 505 @ 12 |
| coalescence | ~20 | 24 |
| natural termination | (not stated) | YES, ~160 iterations |

**Value: not at the same setting.** That configuration converges to
omega_1 = 151.4 (-13%). Removing the move box raises it to 166.8 (-4.5%),
converged at 128 iterations, bimodal to 0.87% — but then the first ~8
iterations oscillate violently (omega_3 420->150->420), which Fig. 4 does not
show. Sweeping the move box between these extremes trades one for the other:

| move | w2 peak | coalescence | final w1 | shape verdict |
|---|---|---|---|---|
| 0.05 | 302 @ 22 | 60 | 141.3 | too slow |
| **0.10** | **296 @ 8** | **24** | **151.4** | **best shape match** |
| 0.15 | 290 @ 4 | 15 | 157.1 | peak too early |
| 0.20 | 280 @ 3 | 12 | 158.5 | peak too early |
| 0.30 | 273 @ 2 | 7 | 159.9 | transient gone |
| 0.50 | 253 @ 1 | 6 | 167.1 | no rise at all |
| inf | 253 @ 1 | 4 | 166.8 | violent oscillation |

The paper's Fig. 4a has BOTH a late, high omega_2 peak (325@7) AND a final
174.7. No move-limit value in our reconstruction delivers both; the ceiling
rises monotonically as the transient is destroyed. This is a genuine,
recorded gap, not a tuning failure -- and it is the sharpest remaining
evidence that some element of the authors' step control is not reconstructed.

Three profiles are therefore shipped and labelled: `shape`, `freq`, `strict`
(`runs/run_fig2.m`). Every figure names its profile.


## 19. Deliverable runs — status 2026-09-03

`runs/run_fig2a.m`, `run_fig2b.m`, `run_fig2c.m` (thin wrappers over
`runs/run_fig2.m`). All at 160x20, r_min = 1.2 el, tolMult = 0.05,
`mmaVariant='published'`, `maxNumCompThreads(1)`.

| case | profile | omega_1 | paper | dev | gap (bimodal) | outer | MMA calls | natural conv | wall |
|---|---|---|---|---|---|---|---|---|---|
| a | shape | 151.4 | 174.7 | -13.3% | 0.15% | 160 | 160 | YES | 16.2 s |
| a | freq  | 166.8 | 174.7 | -4.5%  | 0.87% | 128 | 128 | YES | 13.2 s |
| b | shape | 226.8 | 288.7 | -21.4% | 0.11% | 169 | 169 | YES | 14.9 s |
| b | freq  | 262.7 | 288.7 | -9.0%  | 1.29% | 130 | 130 | YES | 16.4 s |
| c | shape | 374.9 | 456.4 | -17.9% | 0.08% | 225 | 225 | YES | 22.9 s |
| c | freq  | 415.8 | 456.4 | -8.9%  | 1.07% | 110 | 110 | YES | 13.4 s |

**RETRACTED as an efficiency claim (see sec. 20/21).** These runs are the
comparator profiles, not Du & Olhoff's algorithm, so their iteration and
MMA-call counts are NOT this paper's complexity and must not be compared
against other algorithms. The earlier statement here -- "~500-800x reduction
in MMA calls vs the previous baseline" -- is withdrawn: it compared a
different algorithm against the reproduction, which is exactly the
substitution CLAUDE.md forbids. The table is kept only as the record of what
the comparators do.

Initial frequencies reproduce to 0.4%: 68.40/103.73/145.57 vs 68.7/104.1/146.1.
All optima are bimodal, tightly so (0.08-1.3%), which CLAUDE.md sec. 7 names as
the sharper test.

Remaining gap: final omega_1 is 4.5-9% low ('freq') or 13-21% low ('shape').
NOTES sec. 18 shows these are the two ends of one trade-off curve controlled by
the unstated move limit, and that the paper's Fig. 4a sits off that curve
(it has both the late high omega_2 peak AND the high final value).
Case-b's printed Fig. 3b design digitizes to 254.8/273.5 in our FE model
(NOTES sec. 14) -- BELOW our 262.7 -- so for b the residual is largely in the
printed-figure/FE comparison, not the optimizer.

## 20. CORRECTION — one MMA call per outer iteration is NOT the paper's method

Sections 16-19 explored a single-call architecture and reported it favourably
because it converges naturally. That framing was wrong for a reproduction
study and is corrected here. `run_fig2` now defaults to `strict` and warns on
the comparator profiles.

**It contradicts the text.** Sec. 3.5.1 p.98: the third step "consists of an
inner loop that solves a suboptimization problem. **Upon convergence**, this
inner loop delivers optimum values of increments Δρ_e", with "K, M, ω_j, φ_j,
N, R, λ̃, λ̂, and the f_sk vectors ... **kept fixed in the third step**". With
one call per outer iteration the FE data is refreshed every call, so nothing
is held fixed and Fig. 1's box 3 + its No-branch are unrealized.

**It disables (25d), the paper's central contribution.** A single call starts
at Δρ = 0, so A(Δρ) = 0 in `deltaLambda`: the sub-eigenvalue problem is
degenerate, the eigenvector basis is undefined, the identity fallback applies,
and the constraint gradients reduce to the diagonal f_jj. Consequences:
- the erratum form of (25d) never acts;
- the model is the off-diagonals-dropped form that §3.4.2 permits ONLY under
  eq. (22) — measured 11-250% wrong at multiplicity (§3, `test_gradients.m`);
- eq. (22) is not imposed either, so it is neither the MMA baseline nor the
  labelled LP alternative — it is a third, unsanctioned scheme.

It yields plausible numbers because MMA is conservative, not because it is
the method. Its iteration and MMA-call counts must NOT be quoted as this
paper's complexity.

**Standing status of the actual reproduction (`strict`):** reaches comparable
or slightly higher ω₁ than the comparators, keeps the full (25d) machinery
live at ~15-40 mmasub calls per outer iteration, and does NOT terminate
naturally — ‖Δρ‖₂ chatters at a floor above any usable ε. That non-termination
is the honest headline finding about the paper-as-written plus our
reconstruction of its unstated step control, and §11's inner-loop fixes
(published MMA constants, tolInner = 0.05) still reduce its cost from
~100-300 calls/iteration to ~15-40 without changing the algorithm.

## 21. What may and may not be counted as efficiency, in this study

The measured quantity is the cost of *Du & Olhoff's* algorithm. Changes fall
into three classes and only the first is neutral with respect to that
measurement.

**(1) Corrections — restore the paper/lineage as specified. Keep.**
- `mmasub` published Sept-2007 constants instead of the local `move = 1.0`,
  `asyinit = 0.01` (sec. 11). Same reasoning CLAUDE.md sec. 6 applies to
  GCMMA: a non-published MMA changes iteration counts, i.e. the quantity
  being measured. "MMA (Svanberg 1987)" must mean Svanberg's MMA.
- Outer test `||drho||_2 < eps` instead of `max|drho| < eps` (sec. 12).
  Fig. 1 prints a vector norm; the max-norm reading could never fire.
- `lambda~ = omega_n^2` instead of the cluster mean (sec. 15), per sec. 3.5.1.
These change measured counts, but they change them TOWARD the specified
algorithm. Each is reported with its before/after.

**(2) Unstated parameters — sweep and report the CURVE, never a point.**
`tolInner` is the clearest case. It is never given in the paper, and it sets
the inner cost almost by itself (sec. 11, at 160x20 move 0.02):

| tolInner | mean mmasub calls / outer | max deviation of the omega history vs tol=0.01 |
|---|---|---|
| 0.10 | 7.8  | 26%  |
| 0.05 | 16.9 | 7.2% |
| 0.02 | 44.1 | 3.5% |
| 0.01 | 81.7 | reference |

Quoting 16.9 as "the" per-iteration cost would be choosing a number the paper
does not fix. **The honest efficiency result is that Du & Olhoff's
per-iteration cost is undetermined by the paper across at least a 10x band,
because the inner-loop convergence criterion is never stated.** Any table
comparing this algorithm with others must carry that band, or fix tolInner by
a stated rule and say so. Same for the move limit, r_min, and the
multiplicity tolerance.

**(3) Substitutions — forbidden, however well they perform.**
Collapsing step 3 to a single MMA call (sec. 20). It is faster and it
converges, and it is not this algorithm: it never forms A(Δρ) != 0, so the
(25d) coupling that the paper exists to introduce never acts. Retained only
as a labelled comparator, with a runtime warning, and excluded from every
complexity statement.

**Consequence for the complexity study.** The per-iteration cost model of
CLAUDE.md sec. 6 must be reported with the inner loop parameterized by its
call count k:
    outer: assembly O(NE) + eigensolve(J modes) + gradients O(NE*N^2)
    inner: k * [ A(drho) O(NE*N^2) + N^3 + gradients O(NE*N^3) + MMA O(NE) ]
with k measured, its tolInner dependence stated, and the observation from
sec. 5 that at 160x20 the inner loop dominates wall-clock (~97%) rather than
the eigensolve.

## 22. Reproduction baseline — `strict` profile, the paper's algorithm

`run_fig2a/b/c` at 160x20, r_min = 1.2 el, tolMult = 0.05, move = 0.03,
mass (4b), all f_sk filtered, published MMA constants, tolInner = 0.05,
maxNumCompThreads(1). Full (25d) erratum coupling live throughout.

| case | omega_1 | paper | dev | omega_2 | gap | outer | mmasub calls | calls/outer | wall | inner share |
|---|---|---|---|---|---|---|---|---|---|---|
| a | 165.5 | 174.7 | -5.2%  | 174.1 | 5.16% | 150 (cap) | 2716 | 18.1 | 127 s | 95% |
| b | 267.3 | 288.7 | -7.4%  | 281.7 | 5.40% | 150 (cap) | 3242 | 21.6 | 173 s | 97% |
| c | 387.0 | 456.4 | -15.2% | 424.8 | 9.78% | 150 (cap) | 3888 | 25.9 | 273 s | 98% |

These are the numbers admissible in a cross-algorithm efficiency comparison,
subject to the tolInner band of sec. 21.

Two defects stand, unpatched:
- **No natural termination.** All three hit the budget. Sec. 3.5.2 solved
  literally is SLP-with-a-move-box (MMA reduced to a solver for the frozen
  sub-problem), which oscillates about the optimum rather than contracting,
  so ||drho||_2 never falls below any usable eps.
- **The pair does not stay coalesced** (5.2 / 5.4 / 9.8% final gap, against
  the paper's "all bimodal"). CLAUDE.md sec. 7 names multiplicity as the
  sharper test, so this is the more serious of the two.

Note the ordering against the comparators: `strict` gives HIGHER omega_1 than
either single-call profile in every case (165.5/267.3/387.0 vs 151.4/226.8/374.9
and 166.8/262.7/415.8 -- c is the exception, where `freq` is higher). The
substituted architecture was never buying accuracy; it was buying termination.

Next avenue, and to be reported as reconstruction not improvement: a
CONTRACTING move limit. SLP with a shrinking trust box is the standard
contemporaneous remedy for exactly this oscillation and sits in the authors'
lineage (Krog & Olhoff 1999). It would let the printed eps-test fire without
touching the inner loop. To be swept and recorded per CLAUDE.md sec. 4.

## 23. Defect (2) diagnosed — the multiplicity tolerance induces a limit cycle

The `strict` runs' final gaps are not converging to some value that happens to
exceed 5%; they are PINNED to tolMult itself:

| case | N=2 in | gap<5% first @ | gap<5% in last 50 | last 10 gaps (%) |
|---|---|---|---|---|
| a | 54/150 | 19 | 9/50  | 5.0 5.0 5.2 5.0 5.1 5.0 5.1 5.0 5.2 5.0 |
| b | 64/150 | 17 | 0/50  | 5.5 5.6 5.4 5.6 5.5 5.6 5.4 5.6 5.4 5.6 |
| c | 108/150 | 17 | 32/50 | 4.6 4.0 3.9 3.7 3.6 3.2 3.3 3.1 4.7 6.8 |

Mechanism: the detection rule is a hard binary switch. Gap < tolMult -> N=2,
the coupled (25d) model runs and separates the pair slightly; gap > tolMult ->
N=1, the simple model pulls them back together. The design genuinely sits at a
near-degenerate point and the ALGORITHM chatters across the threshold, exactly
the branching CLAUDE.md sec. 4 warns of.

So the pair does coalesce (all three cases reach gap<5% by iteration 17-19,
matching the paper's "soon coalesces"); what fails is STAYING there, and the
failure is in the detection rule, not the optimizer.

This sharpens the tolMult dilemma already noted in sec. 13:
- tolMult = 0.05: coalescence achieved but chatters at the threshold;
- tolMult = 0.005 (nearer the paper's "predefined, very small tolerance"):
  the joint ascent is lost entirely -- N=2 essentially never fires.
Neither reproduces "all bimodal" as a stable end state. The paper gives no
value, and no value in our reconstruction gives its reported behaviour.

Candidate remedies, all reconstruction and all to be swept, not adopted:
hysteresis on the switch (join at one tolerance, leave at a looser one -- the
rebuilt legacy solver used 2%/5%, see the post-mortem sec. 2.3); or the
contracting move limit of sec. 22, since a shrinking step would damp the
separation that trips the switch. The two defects may be one phenomenon.

---

## 24. Step-control audit — verdict STEP_CONTROL_PARTIAL (2026-09-03)

Full record: `audit_stepcontrol/` (WP1 evidence, WP3b/c retrospective analysis,
WP4 Pareto, WP5 sensitivity, WP6 verdict + out-of-sample addendum, manifest).

**WP1.** "move limit", "trust region", "step size" occur **zero times** in
Du & Olhoff 2007 and zero times in Olhoff & Du 2014. The only bound on Δρ is
the box (25f). The outer test is on "**the norm** of the vector Δρ" — the norm
is NOT specified; our ℓ₂ reading is reconstruction (`cfg.outerNorm`, 'max' kept).
Krog & Olhoff (now in `docs/`) Eq.(103) writes the perturbation as
`a+Δa = a+ε·e` with ε "a small positive scalar which gives the magnitude" —
class-B support for bounding magnitude, with no value. Their abstract's
"restrictions on the vector of design changes" is about DIRECTION (the
off-diagonal equalities), not magnitude, and must not be cited for a move limit.

**Fixed move cannot terminate at any ε** (retrospective, no reruns). ‖Δρ‖₂ is
non-monotone: dips to 0.15 at iteration 45, then rises 2.7–3.6× and floors
(`k^(−0.01)`). Reachable stopping iterations are only {1…42} (move 0.03) and
{7…21} (move 0.05). **This refutes the "spectrally settled, slowly
density-converging" explanation of Fig. 4's horizon**: Krog & Olhoff's "very
slow final convergence" is monotone-sluggish, ours is oscillation at a floor.

**All varying-move families terminate naturally** (S1/S2/S3), keeping the
genuine nested loop (15.6–21.5 calls/outer), and without costing final quality
(best ω₁ = 169.88, −2.76%, beats the non-terminating baseline's −3.97%).

**Out-of-sample 240×30**, profile frozen on 160×20 evidence alone, predictions
checked inside the run: ω₁ 163.32 (−6.51%), gap 5.004%, **114 outer CONV**,
17.3 calls/outer, 213 s. Three of four predictions PASS; ω₁ FAILS by 0.51 pp —
and refining the mesh made ω₁ *worse* by 1.6 pp vs 160×20, which is the
direction §8b predicts (the paper's 174.7 points to ≈80×10–96×12). No
re-tuning was done after seeing it.

**Two defects are now shown to be mesh-independent properties of the
reconstruction, not artifacts:**
- multiplicity stays pinned to tolMult (gap 5.004% at 240×30; 4.79–5.19% across
  all 18 runs at 160×20) — the §23 threshold limit cycle;
- the Fig. 4 transient and the final ω₁ are not jointly reachable; move₀
  controls both and they want opposite values (0.03 → transient matches,
  0.04 → horizon matches, neither → 174.7).

**The transient/value trade-off is NOT caused by the inner-loop architecture.**
It appeared identically with the one-call comparators and survives unchanged
under the genuine nested algorithm, so it lives in the FE model, filter, mass
law, or multiplicity handling.

**WP5 cost is not identifiable from the paper.** All four points completed
(tolInner 0.10 / 0.05 / 0.02 / 0.01 → 9.1 / 21.9 / 119.3 / 173.8 calls per
outer, a **19.1× cost range**, wall 364 s → 6894 s). ω₁ spans 166.04–169.27
(spread 1.92% of mean), coalescence 12–13, transient peaks identical.
Stopping iteration at ε = 0.05 is **81 / 85 / 84 / 66** — three of four cluster
at 81–85 but the tightest tolerance stops ~20 iterations earlier, so the
horizon is O(80) spanning **66–85**, not the tight 81–85 band reported when
only three points were available. Caveat: the final *topology* is NOT robust
(mean |Δρ| = 0.148–0.156, 32–33% of elements differ by >0.1) at any tolerance,
even where the scalars agree.

**Timing, both meshes: the inner loop dominates** (95% at 160×20, 96% at
240×30; eigensolve 3–4%). CLAUDE.md §6's expectation that the eigensolve is the
dominant outer cost is contradicted at both resolutions.

`mma/` verified byte-identical to the WP0 baseline (sha256 54c16800…).
Nine-mesh campaign not launched.

---

## 25. Residual-discrepancy audit — verdict MULTIPLICITY_THRESHOLD_CAUSAL (2026-09-03)

Full record: `audit_residual_discrepancy/` (WP0 frozen state, WP1 evidence,
WP2 pre-registration + results, WP2B mechanism, WP2C termination, WP3 analytic,
REPORT, two manifests). Stopped at Gate A; WP4–WP6 not run; nine-mesh campaign
not launched; frozen 240×30 not modified or re-tuned.

**The terminal gap IS the tolerance.** Frozen 160×20 realization, `tolMult` the
only varied field, five pre-registered points:

| tolMult | 0.030 | 0.040 | 0.050 | 0.060 | 0.070 |
|---|---|---|---|---|---|
| terminal g₁₂ | 2.996% | 4.007% | 5.016% | 5.724% | 6.955% |
| g₁₂/tolMult | 0.999 | 1.002 | 1.003 | 0.954 | 0.994 |
| ω₁ | 158.84 | 166.83 | 166.05 | 168.44 | 167.80 |

Fit `g₁₂ = 0.9636·tolMult + 0.0012`, **r = 0.9975**; 5/5 ratios in [0.85,1.15];
gap spread / tolMult spread = 0.990. §23's pinning is now causal, not descriptive.

**Mechanism — the switch creates its own attractor.** From existing histories,
all five runs: mean Δgap **< 0 for every N=1 iteration** (−0.0006…−0.0097,
56–100% closing) and **> 0 for every N=2 iteration** (+0.0025…+0.0044, 84–94%
opening). Since N=1 is chosen above the threshold and N=2 below, the threshold is
a stable fixed point of the switched dynamics. Explicit period-2 cycle in the
frozen run, iterations 36–49 and 68–80.

**Why 5% is the worst possible place to sit.** Predicted vs realized Δλ₁ along the
realized step: N=1 branch median **realized/predicted = 0.116** (over-predicts the
gain ~9×); N=2 branch **0.786**. At a 5% gap the single-mode linearization is
already invalid, but the coupled model is not selected. Off-diagonal/diagonal of
A_sk = f_skᵀΔρ: median 0.054 — small but non-zero, so the branches are genuinely
different models. cos(Δρ_{k−1},Δρ_k): +0.65 within N=2, +0.06 within N=1.

**WP1, the number the lineage does give.** Krog & Olhoff (`docs/978-3-7091-2566-3_9.pdf`,
§5.3 p.306): *"How small this numerical difference between two eigenvalues should
be before they should be considered a multiple, still remains uncertain. For all
the examples in the present section a **relative tolerance of 10⁻⁴** was used."*
Class B, Olhoff's own practice, **500× tighter than our 0.05**. Du & Olhoff's
"predefined, very small tolerance" (p.98) is qualitative only; the test form
(relative ω difference, recomputed in step 1) is class A and we match it; no
hysteresis is mentioned anywhere. Olhoff & Du 2014 re-audited for all five
suspects: **one** relevant sentence, identical to 2007 — it closes nothing.

**§24's natural termination and this pinning are one phenomenon.** Every run —
including the frozen 240×30 — was still switching branches within its last 5
iterations when ε fired (240×30: `21222122`, gaps 4.98 4.99 5.01 4.98 4.99). The
S2 ladder damps the *chatter*, not the design. That is why the 81–85 horizon was
robust across a 13× inner-cost range while ω₁ and the topology were not.

**Filter — a real defect, but not the cause of the greying.** WP6's frozen
realization is documented as `rminEl=1.3`; the 160×20 evidence base actually ran
**1.2** and only the 240×30 OOS ran 1.3, with `rminPhys=[]`, so the *physical*
radius fell 0.060 → 0.0433 (**−27.8%**) across the refinement. Holding 1.3 to
800×100 would shrink it 3.3× — the nine-mesh series as specified would sweep the
filter along with the mesh. **Settle this before unblocking it.** But greying is
not filter-driven: 240×30 is greyer (M_nd 0.215→0.311, +44%; G2 +68%), yet at
fixed mesh and filter `tolMult` alone spans M_nd 0.152–0.371, and the 160×20
tolMult=0.03 run (0.371) is greyer than the frozen 240×30 (0.311).

**Not closed.** ω₁ stays in 158.8–168.4 across the whole probe (best −3.6% at
tolMult=0.06). Pinning explains why the pair never stabilises and why runs stop
where they do; it does not produce 174.7. §8b's discretization account remains the
leading explanation for the level and is untested here. The −6.51% 240×30 miss is
now known to have been delivered from inside an active limit cycle *and* at a
27.8% smaller physical filter radius — two confounds the WP6 addendum did not
have; that does not reopen the step-control verdict.

**Hygiene.** One code change: `cfg.diag` per-iteration recorder in `olhoffOpt.m`,
additive and default-off. `runs/test_diagflag.m` proves bitwise identity with the
flag on and off. The edit landed while `tolMult=0.07` was still running, so
`runs/wp2_verify070.m` re-ran that point and required bitwise equality — PASS.
`audit_stepcontrol/` evidence 38/38 OK; `mma/mmasub.m` still `54c16800…`.

**Next experiment (needs authorization): test multiplicity RULES, not values.**
"Very small" (A), 10⁻⁴ (B), and "0.005 loses the joint ascent" (§13) are only
jointly consistent if the memoryless binary test is itself the wrong
reconstruction. Compare, pre-registered: binary at 10⁻⁴ as the lineage control;
hysteretic join/leave; continuous cluster membership — on gap stability, ω₁, grey
index and termination, predicting in advance that removing the switching
asymmetry removes the pinning. Not a search for 174.7.

---

## 26. Multiplicity-reconstruction audit — verdict MULTIPLICITY_RECONSTRUCTION_SUCCESS (2026-09-03)

Full record: `audit_multiplicity_reconstruction/` (WP0 frozen state, WP1 source
evidence, WP2 switching audit, WP3 candidates, WP4 no-rerun replay, WP5
pre-registration + results, WP6 basis invariance, REPORT, two manifests, four
figures). Three non-baseline runs at 160×20. No 240×30 run. Nine-mesh campaign
not launched. Filter defect untouched. `mma/mmasub.m` unchanged (`54c16800…`).

**The classifier is anti-aligned with the need.** Zero-rerun reconstruction of
the frozen run (eigenvalue error 0.000e+00, N matched 80/80). The quantity that
decides whether the simple model is valid is the step-induced coupling relative
to the separation, c = |f₁₂ᵀΔρ|/(λ₂−λ₁), **not** the gap: Spearman
ρ(c, realized/predicted) = **−0.677**, ρ(gap, c) = −0.035.

| branch | n | median c | frac c>1 | real/pred Δλ₁ |
|---|---|---|---|---|
| N=1 (simple used) | 30 | **1.177** | **60 %** | 0.209 |
| N=2 (coupled used) | 36 | **0.0069** | 0 % | 0.680 |

The coupled model is switched on where it changes the mode-1 prediction by 2 %
(a₂₂ < a₁₁ on 0/36 iterations, so it does not even re-sort the diagonal) and off
where the modes mix completely. **(25d) has no representation of λ₂−λ₁** — the
paper's own derivation assumes exact degeneracy — so its mode-2 constraint is
structurally unable to bind (slack +9.6 % λ₁ always) and it cannot close a gap.
λ̃ = λ_n in (19) also inflates the upper mode's predicted gain by **+11.3 %** at
a 5 % gap; eq. (24)'s own λ_j removes it.

**Three candidates, pre-registered, 160×20, only the multiplicity treatment
varied. Regression first: the patched code reproduces `S2_lad004` BITWISE.**

| run | ω₁ | terminal g₁₂ | outer | branch changes | last-20 | M_nd | real/pred | median c |
|---|---|---|---|---|---|---|---|---|
| M0 binary 0.05 | 166.05 | 5.02 % | 80 | 23 | 9 | 0.215 | 0.234 | 0.015 |
| M1 binary 1e−4 | 164.77 | 3.86 % | 134 | **0** | 0 | 0.196 | **0.060** | 0.652 |
| M2 latch 0.05 | 166.80 | **20.08 %** rising | 101 | 1 | 0 | 0.137 | 0.664 | 0.002 |
| **M4 subspace N=2** | **169.48** | **1.48 %** flat | 90 | **0** | 0 | **0.134** | **0.548** | 0.044 |

**M1 = `TIGHT_THRESHOLD_FAIL`**: N=2 never fires in 134 iterations; pinning gone
(g/tol = 386) but no coalescence. **M2 = `PERSISTENCE_FREEZE`**: latching drives
the pair from 4.2 % to 20.1 %, still rising — the coupled branch has no
gap-closing mechanism. M3 hysteresis implemented but not run; bounded by the
M0/M2 endpoints to a `HYSTERESIS_LIMIT_CYCLE`.

**M4 — the reconstruction that works.** (25d) with the actual separation
retained on its diagonal, built only from the paper's own equations:
`B_ss = λ_s + f_ss(λ_s)ᵀΔρ` [eq. 24], `B_sk = f_sk(λ̃)ᵀΔρ` [eq. 19],
`Δλ_j = μ_j(B) − λ_j`. **Both printed limits exact** (degenerate → (25d),
verified 0.000e+00; separated → (20)/(24), 2.2e−13), so a well-separated mode in
the window changes nothing and **the classifier is removed, not retuned**. Class
**C** — written in no source. Mechanism: β ≤ λ₁+μ_min is *concave*, so
mode-mixing directions are penalised; M4 holds c at 0.044 and **never exceeds 1
in 89 iterations**. **Krog & Olhoff's f_skᵀΔa = 0 emerges from maximizing μ_min
rather than being imposed.** M1↔M4 is single-factor (mode-3 constraint never
active, min slack 3.40 λ₁): the one difference is f₁₂ᵀΔρ.

**WP6 basis invariance.** Rotating the pair by θ: with `filterMode='all'` the
coupled Δλ and its gradients are invariant to **3–7e−12**. With
`filterMode='diag'` they drift by up to **20 %** — CLAUDE.md §5's open filtering
question is settled on principle: `'diag'` is ill-posed for a multiple
eigenvalue. The basis sensitivity lives in the **N=1** branch (f₁₁ᵀΔρ varies by
a factor 25). Realized eigenvector rotation tracks c at ρ = **+0.926**; median
14° on N=1 iterations vs 1° on N=2. `BASIS_INSTABILITY` **not supported**.

**WP1 — the 10⁻⁴ does not transfer as a setting.** Krog & Olhoff (132)/(136)
bound **every** eigenvalue of the subspectrum with λ_j + f_jjᵀΔa ≥ β regardless
of N; their detector only adds the f_skᵀΔa = 0 equalities. It never gates the
ascent mechanism, which is why 10⁻⁴ is safe there and fatal here. Olhoff & Du
2014 **drops the tolerance sentence entirely** (grep for toleran|very small|
predefined returns 0).

**TWO PREVIOUS CONCLUSIONS REVISED.**
1. **§13's "tolMult = 0.005 loses the joint ascent entirely" does not
   reproduce.** At 1e−4 under the S2 realization, ω₁ = 164.77 — within 0.8 % of
   the 0.05 run, natural convergence in 134 outer. What a tight tolerance loses
   is the *coalescence*, not the ascent.
2. **§23/§25: the gap oscillation is NOT caused by the branch switching.** M1
   oscillates around 5–6 % for ninety iterations with **zero** transitions
   (std 0.885 % over k=30–70, 39 sign flips) against M0's std 0.991 % with 12
   transitions. The oscillation comes from the N=1 branch's large box-vertex
   steps; the classifier pins *where* it settles. The pinning fit itself
   (g = 0.9636·tolMult + 0.0012, r = 0.9975) stands unchanged.

**Also recorded.** A genuine localized-mode pair at ω = 85.63/85.93 appeared for
one iteration in M2 (k=61) and vanished at k=62 — **dense LAPACK reproduces it
to 1.4e−7**, so it is real, not an ARPACK artifact (28.6 % of elements below
ρ = 0.01 there). First observed instance of CLAUDE.md §5's undefined case
(the constrained index set silently jumped to different physical modes); the
algorithm recovered in one step. Logged, not patched.

**ω₁ is secondary and the deficit is NOT closed.** 169.48 (−2.99 %) is the best
of the four and in the band of the best step-control run (169.88). §8b's
discretization account is untouched.

**Next experiment, and it is the load-bearing one:** M4 at `subN = 3` and
`subN = 4`. If the classifier really has been made inconsequential, the result
must be insensitive to the window size; if ω₁, the gap or the topology move
materially, M4 is a differently-parameterised classifier rather than the removal
of one. ~4 minutes, 160×20. Not run here (the audit was capped at three
non-baseline treatments).

---

## 27. M4 spectral-window confirmation — verdict SUBSPACE_WINDOW_INVARIANT (2026-09-04)

Full record: `audit_multiplicity_subN_confirmation/` (WP0 freeze + pre-registered
prediction, WP1 code semantics, WP3 comparison, REPORT, two manifests, three
figures). **Exactly two runs**, 160×20, `subN = 3` and `subN = 4`, against the
frozen `subN = 2`. Nothing tuned, no third window, M4 unmodified. No solver code
was written; `algo/`, `fem/`, `filter/`, `mma/` byte-identical throughout.
`mma/mmasub.m` still `54c16800…`.

**The gate this closes.** §26 named the `subN` sweep as the load-bearing
falsification test of M4: if the result moves with the window size, M4 is a
differently-parameterised classifier rather than the removal of one.

**Single-factor proof.** `cfg.subN` is read **exactly once in the whole solver**
(`algo/multRule.m:67`), asserted by counting executable occurrences across
`algo/`, `fem/`, `filter/` (total = 1). `Jcalc = n+Nmax = 5` is fixed before the
rule is called, so all three runs solve the *same* eigenproblem; `eigSolve` is
deterministic; S2 move control reads only `hist.beta` (asserted: `mv` identical
for `hist.N=[1 1 1]` vs `[2 2 2]`); `lamref`, `nvar`, `Vtot`, `f0val`, the inner
stopping test (density block only) and the outer test are all N-free. Only
`m = N+2` and `J = n+N` follow N. Both runs asserted at launch to differ from the
frozen cfg in `{subN, name}` only.

**Result — the prediction held.**

| subN | ω₁ | ω₂ | ω₃ | g₁₂ | M_nd | outer | trans | MMA |
|---|---|---|---|---|---|---|---|---|
| 2 (frozen) | 169.4806 | 171.9831 | 357.72 | 1.4766 % | 0.1339 | 90 CONV | 0 | 2219 |
| 3 | 169.6415 | 171.7733 | 363.57 | 1.2566 % | 0.1328 | 89 CONV | 0 | 2204 |
| 4 | 169.6325 | 171.7574 | 342.23 | 1.2526 % | 0.1330 | 91 CONV | 0 | 2313 |

ω₁ spread **0.095 %**; gap 1.25–1.48 % with flat trends; zero branch transitions
everywhere; ω₁(k) trajectories superimposed to **1.7–2.8e−3 rms**; g₁₂(k)
correlates at **1.0000**; density correlation **0.9933 / 0.9977**, mean |Δρ|
0.016 / 0.009; topologies visually indistinguishable, all Fig. 3a's
eye–X–X–void–X–X–eye.

**The decisive number: the window effect SATURATES.** 2→3 moves ω₁ by +0.161 and
g₁₂ by −0.220 pp; 3→4 moves them by −0.009 and −0.004 pp — **ratios 0.056 and
0.018**. A re-parameterised classifier would keep moving with each added mode;
a genuine local-subspace treatment converges in the window, which is observed.

**The added modes never activate.** Minimum normalised constraint slack over the
whole run: mode 3 = 3.369 / 3.314 / 2.992, mode 4 = 8.511 / 8.442, mode 5 =
5.309 — all ≥ 3.0 λ₁ against mode 2's median slack of 0.030, a two-decade
separation. Promoting mode 3 from the linear (25b) form to the coupled (25c)
form left its slack essentially unchanged (3.369 → 3.314): the analytic
separated-limit property of the diagonal-offset form, confirmed in a live run.
**`HIGHER_MODE_ACTIVATION` not supported.** No modal ordering changes (MAC of
mode 1 never below 0.5) in any run.

**Calibration for "small".** NOTES §24: varying `tolInner` over a 19× cost range
— concluded *not* to alter the science — moved the topology by mean |Δρ|
0.148–0.156 with 32–33 % of elements >0.1. The `subN` effect is 0.009–0.016 with
1.7–4.8 % >0.1, an order of magnitude smaller, and it does **not** grow with the
window (2↔4 is smaller than 2↔3), so it is non-uniqueness scatter, not drift.

**Reported against the result, not tuned away:** `subN = 4` has 3 iterations of
91 with coupling indicator c > 1 (1.19, 1.18, 1.54) where `subN` = 2 and 3 have
none — versus the old classifier's N=1 branch at median c = 1.177 with 60 %
above 1. And `subN = 4`'s ω₃ differs 6 % from the baseline, in a mode whose
slack never falls below 3.0 λ₁.

**Two log-only artifacts recorded, not changed:** at `subN = 4`, `N = Nmax` so
`olhoffOpt.m:68` emits *"detected N=4 >= Nmax=4, J may be truncated"* every
iteration — misleading under `multRule='subspace'` (N is set by fiat, and
nothing is truncated: `J = 5 = Jcalc`); and `multJ` targets a different pair at
each `subN`. Neither has numerical effect; left alone so the three runs stay
comparable. No implementation bug found.

**Consequence.** `MULTIPLICITY_RECONSTRUCTION_SUCCESS` is **STRENGTHENED on
Question A** (a coherent multiplicity treatment that removes threshold pinning —
its last free parameter is shown not to carry the result) and **unaffected on
Question B** (whether this is Du & Olhoff's implementation). **M4 remains class
C.** Invariance to `subN` is evidence of internal coherence, not of authorship.

**No further experiment is justified by this result** — the gate is closed and
M4 has no remaining free parameter of its own. The deferred queue is unchanged:
filter-radius definition (still the nine-mesh blocker), a 240×30 M4 run (now
better motivated, but settle the filter radius first, since that is exactly the
parameter whose meaning changes under refinement), the ω₁ deficit, bimaterial
and 3D.

---

## 28. Conference admission gate — verdict CONFERENCE_ADMISSION_FAIL (2026-09-04)

Full record: `audit_conference_admission/` (WP1/WP2 filter semantics, WP4 frozen
realization, WP5/WP6 results, REPORT, two manifests, two figures). **Exactly one
240×30 run.** No tuning after the result. `mma/mmasub.m` still `54c16800…`; zero
solver code written; only `NOTES.md` modified outside the new directory. All 44
`.mat` evidence files across all five audit trees verify byte-for-byte.

`performance_comparison.m` and the Proposed/Yuksel implementations are **not in
this project** (nor under `~/Programming/Matlab` to depth 3).

### Job 1 — filter semantics: RESOLVED and FROZEN (a valid deliverable)

The defect was **configuration, not code**: `olhoffOpt.m:24-27` already converts
`rminPhys → rminEl`, exactly (asserted: `dx == dy` at every mesh in the series);
runs simply set `rminEl` and left `rminPhys` empty. Historical values, read from
the `.mat`: **every** 160×20 run used `rminEl = 1.2` → r_phys = **0.06000**; the
one previous 240×30 used 1.3 → **0.04333** (−27.8 %; ×5.0 shrink projected to
800×100).

**Frozen convention: fixed physical radius r_phys = 0.06·b, `rminEl = 0.06·nely`**
(class A by implication from the paper's own "mesh-independent" wording; the
*value* 0.06 is class C, carried forward because it is the physical radius of the
entire 160×20 evidence base — **not** chosen by comparing ω₁).

| mesh | 160×20 | 240×30 | 320×40 | 400×50 | 480×60 | 560×70 | 640×80 | 720×90 | 800×100 |
|---|---|---|---|---|---|---|---|---|---|
| rminEl | 1.2 | 1.8 | 2.4 | 3.0 | 3.6 | 4.2 | 4.8 | 5.4 | 6.0 |

**No bridge run was needed — proved, not assumed:** `prepFilter(160,20,1.2)` and
`prepFilter(160,20,0.06/(1/20))` give bitwise-identical `H` and `Hs`
(nnz = 15640), so the frozen M4 160×20 result already uses the final semantics.

### Job 2 — the 240×30 admission run: FAILED

    w1 = 167.0646  w2 = 194.0663  w3 = 361.1765 | g12 = 16.1624% | 103 outer CONV
    0 branch transitions | MMA 2317 (22.5/outer) | M_nd 0.1564 | vol err -8.7e-07
    wall 223 s = eig 3.1% + grad 0.2% + inner 96.5%

| | 160×20 ref | 240×30 |
|---|---|---|
| ω₁ | 169.481 (−2.99 %) | 167.065 (−4.37 %) |
| ω₂ | 171.983 | **194.066** |
| terminal g₁₂ | 1.477 % | **16.162 %** |
| slope g₁₂ last 20 | −6.4e−05 flat | **+1.26e−03 rising** |
| M_nd | 0.1339 | 0.1564 |

**Both meshes coalesce identically, then 240×30 de-coalesces.** g₁₂ < 2 % at
k = 16 on both; minima 0.415 % / 0.366 %. 240×30 then re-crosses 2 % at k = 51,
5 % at k = 62, 10 % at k = 72, ending at 16.14 % **still rising**. Over the last
30 iterations dω₁ = +0.034/iter vs **dω₂ = +0.349/iter** — mode 2 climbs 10×
faster than the objective. √β tracks ω₁ correctly (+0.009), so the bound
formulation is fine; the design left the bimodal ridge. Mode 2's slack binds
early (2e−4) then rises to ≈0.35 from k ≈ 50, exactly when ω₁ stalls.

**Coupling indicator** median 0.0045 (last 20: 0.0028), realized/predicted 0.991
— formally excellent, but only **because the pair separated**: at a 16 % gap the
diagonal-offset model correctly reduces to the simple model, so **M4's coupled
machinery is inert over the second half of the run.**

**Threshold pathology did NOT return** (no threshold exists in `'subspace'`,
0 branch transitions, 16.16 % = 3.23× the old attractor, not equal to it).
**Higher modes stayed inactive** (mode 3 slack ≥ 3.63 λ₁) — `subN = 2` is not
invalidated. **No modal ordering changes** on either mesh.

**Greyness: the filter fix works** — M_nd 0.156 vs **0.311** for the old 240×30
(shrunken filter + old classifier). Not the reason for the verdict.

**Topology: interior bracing coarsened from four X-cells to two on a FINER mesh
at the SAME physical filter radius**, with grey end regions. Refinement at fixed
regularization length should preserve or refine cells, not simplify them.

### Why FAIL, not PARTIAL

Three pre-declared FAIL clauses are met: (1) **convergence mechanism** — the
density norm converged but the spectrum did not; termination occurred *during*
monotone spectral drift, so "termination after multiplicity stabilization" — the
criterion the multiplicity audit used to certify M4 — fails; (2) **topology
qualitatively collapses**; (3) **the performance number would be misleading** in
a comparison whose whole point is mesh refinement, since at 160×20 the method
gives the paper's bimodal optimum and one refinement step later gives a unimodal
design with a 16 % gap.

### What is NOT concluded — and the uncomfortable possibility

M4 is **not** shown to be wrong; all 160×20 verdicts stand. The filter convention
is **not** shown to be wrong. The cause is **not** identified and was not
investigated (scope). Three untested candidates, and the first cuts *against* the
reference: **at 160×20 the frozen radius gives `rminEl = 1.20 < √2`, the weakest
possible 5-point stencil, while 240×30 gives 1.80 and admits diagonals — so the
160×20 M4 validation may itself rest on an under-resolved filter, in which case
240×30 is the more trustworthy point and the earlier result is the anomaly.**
(This caveat was recorded in WP2 *before* the run.) Also: basin non-uniqueness;
S2 ladder frozen on 160×20 evidence.

**`PERFORMANCE_COMPARISON_INTEGRATION = NOT_AUTHORIZED`.** Prior verdicts are not
retracted — each was, and remains, a statement about the reconstruction *at
160×20*. Next question, needing its own authorization: is the 160×20 reference or
the 240×30 run the anomaly, i.e. does M4's stable coalescence depend on an
under-resolved filter? Until that is settled, neither mesh's numbers belong in a
cross-method comparison.

### 28a. WP6A diagnostic — FILTER_SUPPORT_CAUSAL_SUPPORTED (2026-09-04)

`audit_conference_admission/diagnostic_filter_support/`. **DIAGNOSTIC ONLY** —
`performance_eligible=false`, `mesh_convergence_evidence=false`,
`diagnostic_only=true`, set on the result struct and in
`DIAGNOSTIC_MARKERS.json`. Enters no comparison, no convergence evidence, no
runtime fit, no frozen config.

One run: **160×20 with rminEl = 1.8** — the 240×30 discrete filter support at the
reference mesh, asserted to differ from the frozen reference in `{rminEl,name}`
only. **Not refinement**: at this mesh rminEl = 1.8 means r_phys = 0.09, not 0.06.

| | REF 160×20 r1.2 | **DIAG 160×20 r1.8** | ADM 240×30 r1.8 |
|---|---|---|---|
| terminal g₁₂ | 1.477 % | **6.995 %** | 16.162 % |
| slope last 20 | −6.4e−5 flat | **+1.00e−3 rising** | +1.26e−3 rising |
| re-cross 5 % | **never** | **k=72** | k=62 |
| ω₂/ω₁ | 1.0148 | **1.0700** | 1.1616 |
| dω₂/dω₁ last 30 | 1.4 | **4.4** | 11.8 |
| c median last 20 | 0.1409 | **0.0101** | 0.0028 |
| mode-2 slack last 20 | 0.0343 | **0.1331** | 0.3356 |
| **interior cells** | **four (paper)** | **two** | **two** |
| M_nd | 0.1339 | 0.2045 | 0.1564 |

**Both topology and spectral behaviour reproduced, concordantly.** DIAG lands in
the 240×30 architecture (one X-cell per side, enlarged void, grey ends) *and*
its dynamics (coalesce <1 %, hold, de-coalesce monotonically, still rising at
termination). REF reproduces neither. Five observables order monotonically
REF → DIAG → ADM with the intervention intermediate in every one; the g₁₂ curves
of DIAG and ADM track closely to k ≈ 55. Mode 3 stays ≥ 3.17 λ₁ from binding in
all three; no modal ordering changes anywhere.

**Incomplete**: DIAG reaches ~half the separation (6.99 vs 16.16 %), onset 10
iterations later, and **greyness is discordant and non-monotone** (M_nd 0.2045,
the greyest of the three). **Confound**: 1.2 → 1.8 at fixed mesh changes stencil
geometry (diagonals enter above √2) *and* physical radius (0.06 → 0.09)
together, so this run cannot say which attribute carries the effect.

**Secondary verdict `FILTER_SUPPORT_CAUSAL_SUPPORTED`** — filter support
contributes causally to the bifurcation. NOT proof of mesh convergence, NOT
proof the filter is the only cause, NOT a remedy: rminEl = 1.8 not adopted, no
neighbouring radius tried, frozen convention unchanged, M4 / step control /
subN / stopping rules / MMA untouched. **Does not override
CONFERENCE_ADMISSION_FAIL**, which stands. It does sharpen §28's open question:
whether 160×20 or 240×30 is the anomaly is now known to be a question about the
filter as much as about the mesh.

---

## 29. Mesh-consistent filter & three-mesh admission audit — verdict FILTER_DISCRETIZATION_STILL_UNSAFE (2026-09-04)

Full record: `audit_filter_mesh_admission/`. WP0 verified all nine prior
manifests and all 51 `.mat` evidence files; MMA and every solver source
unchanged; **zero solver code written**.

### The filter question is CLOSED, affirmatively

`prepFilter` with `rmin = R/h` satisfies `H = (1/h)·max(0, R − d_phys)` (verified
to 2.2e−16), and the `1/h` cancels in the row normalisation. **The
implementation already is the fixed continuous cone `w(r)=max(0,R−r)` sampled at
element centres.** Freezing `r_phys = 0.06` and deriving `rminEl = 0.06·nely` is
therefore a *zero-code-change* correction, and it is the smallest justified one:
cell-integrated weights were evaluated analytically and are **not** uniformly
better (240×30, λ=2R: point-sampled 0.0167 vs cell-integrated 0.0380).

Convergence proved, not assumed: cross-mesh L2 on a common physical grid is
**monotone over all nine meshes** (1.74e−2 → … → 0); order 2.15 against the
exact continuous convolution; 1.81 for the impulse kernel; constants preserved
to 0.00e+00, interior linears to 1e−14, boundary error → the correct limit R/π.

**No shell transition is abrupt.** `H(rminEl)` is Lipschitz (|dσ/drm| ≤ 3.95):
the cone's linear falloff makes every shell enter with weight exactly 0. Shells
create only derivative **kinks** — the √2 kink (−1.173) is **2.2× the next
largest**. `rminEl ≤ 1` gives `H = I` exactly (symbol ≡ 1), and the symbol at
λ=2R then runs 0.996 (rm 1.001) → **0.627 (1.2)** → 0.566 (√2) → **0.436 (1.8)**
→ **0.448 (2.4)** → 0.446 (6.5), against a continuous target of 0.4457. The
secant slope over 1.001→1.2 is **−1.856, the steepest anywhere on the axis**:
`(1,√2)` is **not** a plateau but the interval of maximum sensitivity to the
radius, where the operator is still most of the way from "no filter" to the
intended one. It settles only beyond `rminEl ≈ 3`.

### §28's open question is ANSWERED, from filter analysis alone

**160×20 is the anomaly.** It is the only mesh in the nine-mesh series inside
that interval (`rminEl = 1.2 < √2`), sitting on the plunge (symbol deviation
**+0.181** against −0.0095 at 240×30 and +0.0022 at 320×40). Its operator misses the frozen continuous
filter by **41 % of the symbol value at λ = 2R**, is 13.7 % anisotropic,
misplaces 29.8 % of the kernel mass, and passes 20 % of the element checkerboard
— against 3.7 % / 2.6 % / 8.6 % / 2.2 % at 240×30. **No quadrature can fix it**:
at `rminEl = 1.2` the support has only *two distinct radii*. Recorded as an
a-priori prediction in `WP3_continuous_filter_FREEZE.md` §3.7 *before* any run.

This also explains WP6A: `rminEl` 1.2 → 1.8 at fixed mesh moved the operator
across the largest kink on the whole radius axis.

### The three-mesh sequence (160×20 and 240×30 reused, bitwise-proved; 320×40 new)

| | 160×20 | 240×30 | 320×40 |
|---|---|---|---|
| rminEl | 1.2 | 1.8 | 2.4 |
| ω₁ | 169.4806 | 167.0646 | 165.9468 |
| terminal g₁₂ | **1.477 %** | **16.162 %** | **10.674 %** |
| M_nd | 0.1339 | 0.1564 | **0.2340** |
| outer / MMA | 90 / 2219 | 103 / 2317 | 130 / 2596 |

g₁₂, ω₂/ω₁ and the gap slope all shrink by **2.68×** on the second step (sign
reversed); ω₁ by 2.16×. Topology: 240 vs 320 agree **93.5 %** (corr 0.952)
against 88.7 % for 160 vs 240 — the family change (four X-cells → two) happens
**entirely on the first step**, the step where the operator changed 10.9×.
**Outcome A, benign discretization convergence** — the second step crosses two
shells and changes nothing qualitative.

**M4 generalises.** Coupling median-last-20 0.1409 → 0.0028 → **0.0006**, **zero
`c>1` episodes anywhere**, realized/predicted Δλ₁ **0.548 → 0.991 → 1.057** (the
first-order model gets *better* under refinement), mode-3 slack min 3.37 → 3.63
→ **3.81**, no modal reordering in 323 outer iterations, zero branch
transitions, no threshold pinning (`tolMult` is not read by `'subspace'`).

### What actually fails: the STOPPING RULE, not the filter

Replaying `M_nd(k)` from the recorded `Δρ` (exact to 0.00e+00):

| slope over last 20 | 160×20 | 240×30 | 320×40 |
|---|---|---|---|
| M_nd whole domain | **−3.1e−04 flat** | −1.6e−03 | −1.5e−03 |
| M_nd end regions | **−7.1e−04 flat** | −3.7e−03 | −3.1e−03 |

**160×20 had flattened onto a plateau before ε fired; neither finer run had.**
S2 rung reached at iteration: 160×20 `0.02@79 · 0.01@90`; 240×30
`0.02@92 · 0.01@103`; 320×40 `0.02@130 · 0.01@**never**`. Both coarser runs stop
on their first iteration at move = 0.01; **320×40 stops on its first iteration
at move = 0.02 and never gets a finer rung** — at 240×30 that last rung alone
removed 7.6 % of M_nd in 11 iterations.

The grey is a growing wedge in the end triangles peaking 0.5–1.0 from each
support (M_nd 0.195 → 0.369 → **0.597**), **not** at the support node (0.0008 /
0.092 / 0.085 within 0.125 of it). Chords and support diagonals are crisp at
every mesh; the **interior bracing of the end triangle never forms** at the finer
meshes. Signature of an unfinished run, not of a converged optimum.

`ε ∝ √NE` correctly fixes the per-element tolerance (8.839e−04 at all three).
What it does not fix is how far down the S2 ladder a run gets first — and the
ladder is what drives ρ to 0/1. Both are class-C reconstructions frozen on
160×20 evidence.

### Verdict

`FILTER_DISCRETIZATION_STILL_UNSAFE` — **by clause 3 only** ("160/240/320 do not
represent a controlled refinement sequence"). Clauses 1, 2 and 4 are refuted
with evidence. `PERFORMANCE_COMPARISON_INTEGRATION = NOT_AUTHORIZED`: quality
numbers are read off still-improving designs and cost numbers are *defined* by
the mis-scaled stop, so a scaling exponent fitted to 90/103/130 would measure
the stopping rule, not the method.

**Next authorization required** (not taken here — altering step control and
stopping rules is outside this audit's scope): make the ε / S2-ladder pair
mesh-consistent, then re-test whether g₁₂ and M_nd converge across 160/240/320.
Until then no mesh's numbers belong in a cross-method comparison.

---

## 30. Termination / S2 continuation audit — verdict S2_CONTINUATION_DEFECT (2026-09-04)

Full record: `audit_termination_mesh_admission/`. WP0 verified all eleven prior
manifests and all 59 `.mat` evidence files; the previous audit's end manifest
verifies at **0 mismatches**.

### The outer-norm question is CLOSED — it was already correct

`tolOuter = 0.05·√(NE/3200)` means

    ‖Δρ‖₂ < tolOuter  ⟺  d_RMS = ‖Δρ‖₂/√NE < 0.05/√3200 = 8.838835e−04

identical at 160×20 / 240×30 / 320×40 to **1.084e−19**. The brief's "equivalence
option" was already implemented. And `d_RMS` is the best-collapsing measure:
late-stage mesh spread **1.343** vs 2.041 (raw ℓ²), 2.345 (max), 1.431 (mean).

### The logical defect, and the fix

**All three historical runs terminated on the FIRST iteration after an S2 rung
transition** (3 of 3). `d_RMS` had sat on a flat plateau at 2.0–2.8 ε for ~11
iterations, then crossed below ε in the single iteration where the move halved
(drop factors 2.75 / 3.81 / 2.90). `d_RMS/mv` was constant along each plateau and
`max|Δρ|` sat at the move bound: the optimizer is move-limited, so `‖Δρ‖→0`
reported the ladder, not the design.

Minimal correction, frozen before any rerun, **no new numerical constant**:

    converged(k) ⇔ ‖Δρ_k‖₂ < ε(NE)  AND  k ≥ 2  AND  mv_k == mv_{k−1}

`cfg.outerGuard = 'settledmove'`; default stays `'none'`. 23 lines in
`algo/olhoffOpt.m` + one field in `algo/defaultCfg.m`. Six tests pass; iterations
1…K_old of all three reruns are **bitwise identical** to the recorded runs
(`max|Δ(Δρ)| = 0.0e+00`).

### Confirmation — and the falsified prediction

| | 160×20 | 240×30 | 320×40 |
|---|---|---|---|
| outer (old → new) | 90 → **91** | 103 → **104** | 130 → **131** |
| ω₁ | 169.4952 | 167.0704 | 165.9508 |
| g₁₂ | 1.454 % | 16.184 % | 10.738 % |
| M_nd | 0.1340 | 0.1560 | 0.2336 |
| terminal move | 0.010 | 0.010 | **0.020** |
| M_nd slope last 20 | **−2.45e−04** | −1.52e−03 | −1.41e−03 |

Nine of ten pre-registered predictions hold. **Prediction 4 is FALSIFIED**:
M_nd slopes improved by only 5.8 % and 3.4 %, and still differ **6.2×** between
160×20 and 240×30.

**WP10 counterfactual: the old stop was harmless tail polishing.** Δω₁ ≤ 0.009 %,
ΔM_nd ≤ 0.25 %, topologies agree to 99.98 %, no element moved by >0.01. §28–29's
diagnosis of the *mechanism* was right, but repairing it buys one iteration. The
runs were stopping tens of iterations early, not one.

### What is actually responsible: S2's β-driven schedule

Ladder occupancy: 160×20 `0.04@1(78) 0.02@79(11) 0.01@90(2)`; 240×30
`0.04@1(91) 0.02@92(11) 0.01@103(2)`; **320×40 `0.04@1(129) 0.02@130(2)`** — it
spent 98.5 % of its run at the coarsest move and never reached 0.01.

The trigger is identical everywhere (β's 10-iteration relative improvement
crossing `s2Tol = 5e−3`). What differs is *when*: at 320×40 the objective
**resurges** at k ≈ 90 (4.09e−2, up from 1.55e−2) and only decays past threshold
at k = 130. Design states at those descents: M_nd 0.1344 / 0.1693 / 0.2340.

And which rung catches the ε crossing is an accident: at 240×30 the 0.02-rung
step sat at 2.17 ε (forcing a further descent), at 320×40 its first step there
was already 0.92 ε (so it stopped). Rungs are a factor 2 apart, so the crossing
lands on different rungs at different meshes.

**The ladder's schedule is set by the objective's improvement history, which is
not synchronized with the design's approach to 0/1.**

### Unchanged and re-confirmed at all three meshes

M4: coupling median-last-20 0.1409 / 0.0027 / 0.0007, **zero c>1**,
realized/predicted 0.533 / 0.985 / **1.058**, mode-3 slack min 3.37 / 3.63 /
3.81, no reordering in 326 iterations. Filter: `rminEl` 1.2 / 1.8 / 2.4,
`r_phys = 0.06`, operators identical. Topology: 0.9999 / 0.9999 / 0.9998
agreement with the old-stop designs; cross-mesh 0.8865 / **0.9353** / 0.8440 —
identical to §29. The end-triangle grey wedge (ends M_nd 0.152 / 0.250 / 0.414)
is unchanged.

### Verdict

`S2_CONTINUATION_DEFECT` · `PERFORMANCE_COMPARISON_INTEGRATION = NOT_AUTHORIZED`.

Cost claims remain blocked: 91/104/131 outer iterations are set by when β's
improvement crossed 5e−3 (k = 79/92/130), so a scaling fit would measure that
schedule. Wall times (127 / 234 / 374 s) are for the first time mutually
comparable — sequential, single-threaded, no concurrent job — but inherit that
limit and the 19× `tolInner` range.

**Next authorization required** (not taken; the gate says stop): a continuation
schedule driven by the *design* rather than by β, then a re-test of whether M_nd
and g₁₂ converge across 160/240/320.

---

## 31. S2 design-driven continuation remedy — verdict S2_LADDER_ITSELF_DEFECTIVE (2026-09-04)

Full record: `audit_s2_design_continuation/`. WP0 verified all thirteen prior
manifests and all 67 `.mat` evidence files; the previous audit's end manifest
verifies at **0 mismatches**.

### The rule form is settled by evidence: it must be DIFFERENTIAL

Median over k = 20–70 (all meshes on the 0.04 rung, like-for-like):

| measure | 160×20 | 240×30 | 320×40 | spread |
|---|---|---|---|---|
| d_RMS | 1.063e−2 | 7.143e−3 | 4.870e−3 | 2.182 |
| u = d_RMS/m | 2.657e−1 | 1.786e−1 | 1.218e−1 | 2.182 |
| q = ‖Δρ‖₂²/(‖Δρ‖₁·m) | 6.678e−1 | 3.674e−1 | 2.462e−1 | **2.712** |
| max\|Δρ\|/m | 0.9992 | 0.9968 | 0.9905 | **1.009** |
| f_active | 0.5513 | 0.5953 | 0.6964 | 1.263 |

**The best-collapsing measures carry no signal** — `max|Δρ|/m` is saturated at 1
(the move bound binds at the maximum essentially always), `f_active` *rises* as
the move shrinks. **Every informative measure is mesh-dependent by 2.2–2.8×**;
`q`, built to be invariant to *how many* elements move, collapses **worse** than
`u`, and `f_active` is nearly mesh-invariant — so the spread is genuine
amplitude, not participation: at a finer mesh the per-element update is a
genuinely smaller fraction of the move limit (`q ∝ h^1.44`). **No absolute
threshold can be mesh-consistent.**

### The frozen rule (no new constant): swap the stall signal β → d_RMS

    descend when (w1 − w2)/|w1| < s2Tol , w1,w2 = means of d_RMS over the two
    preceding 10-iteration windows        (s2Window, s2Tol, ladder all reused)

`cfg.s2Signal`, default legacy `'beta'`. Pre-registered at
`cdf88b2d1aa6ac73a322a7d0bf1e0d261143a06ecf83f97bf3a366b15227a9b1`,
2026-09-04T09:54:51Z, before `moveControl.m` was edited. Seven tests pass; runs
are **bitwise identical** to the TMA runs up to the first descent (n = 41/45/53).

### It fixed continuation and destroyed the science

| mesh | old transitions | new | old M_nd | new M_nd | ΔM_nd | Δω₁ |
|---|---|---|---|---|---|---|
| 160×20 | 79, 90 | **42, 57** | 0.1340 | **0.2779** | **+107 %** | −2.65 % |
| 240×30 | 92, 103 | **46** | 0.1560 | **0.4226** | **+171 %** | −5.93 % |
| 320×40 | 130 | **54** | 0.2336 | **0.4466** | **+91 %** | −6.00 % |

Outer iterations 91/104/131 → **86/54/59**. Pre-registered scoring: P1 **PASS**
(decision-state u spread 1.85/1.94 vs 3.89 old), P2 **PASS** (M_nd-slope spread
**2.351** vs 6.206), P4/P5/P6 PASS, P3 FAIL (terminal moves 0.01/0.02/0.02),
**P7 FAIL catastrophically**. M4 healthy everywhere (0 % c>1, no reordering,
mode-3 slack ≥ 3.37 λ₁) — no new pathology.

### Mechanism — two causes, honestly split

1. **`d_RMS` trend is a bad usefulness proxy.** It stops decreasing exactly when
   the design enters its large topological-reorganization phase — when the coarse
   move is *most* productive. Its windowed relative decrease swings −0.81 … +0.60
   on the 0.04 rung; the rule fires on the first non-decrease.
2. **The ladder makes that unrecoverable and terminal.** Monotone and
   irreversible, so an early descent cannot be undone; and `d_RMS ≈ u·m` with
   `u ≈ 0.03–0.09` late, so once `m` ≤ 0.02 the frozen absolute ε = 8.839e−4 is
   met almost automatically. **Early descent ⇒ early termination.**

Cause 2 is why the verdict names the ladder: **with this ladder and this absolute
ε, no trigger firing earlier than β's can be adopted at all**, whatever its
mesh-consistency. This exact failure was registered in advance
(`PREREGISTRATION.md` §3.2 item 6) with its verdict mapping.

### Consequence for cost claims

The *same* solver, filter, M4 and outer tolerance give **91/104/131** outer
iterations under one continuation trigger and **86/54/59** under another, with no
change to the physics. **No scaling exponent may be fitted to either set.**

### Verdict

`S2_LADDER_ITSELF_DEFECTIVE` · `PERFORMANCE_COMPARISON_INTEGRATION = NOT_AUTHORIZED`.
The new signal is **not adopted** (`s2Signal` defaults to `'beta'`), so every
prior result remains reproducible bit for bit. A future remedy must address the
ladder's irreversibility or ε's coupling to the move scale — both outside this
task's scope.

---

## 32. Natural convergence without a ladder: adaptive per-element move box — 240×30 and 800×100 (2026-09-12/13)

Full record: `repro/results/<run>/` (res.mat with cfg, describe.txt, summary.json,
hist_vs_paper.png, topo_vs_paper.png), `repro/results/SUMMARY.md`. Runner:
`repro/run_repro.m`; common evaluation: `repro/evaluate_runs.m`. Branch
`repro/natural-convergence`. Anchors A1_frozen160 and A4_nodescent160 re-run
after every code change: science digest **IDENTICAL**; `run_all_tests` 0 failures.

### Code changes (all default-off; the frozen realization is bitwise unchanged)

- `optimizer.inner.asymptoteHistory = 'outer'` (innerLoopRho): MMA asymptotes
  formed from ρ_k, ρ_k−1, ρ_k−2 and HELD through the inner loop. **Refuted**:
  holding the asymptotes removes the inner loop's own damping — inner caps of
  500 sub-iterates at every tested box (N1–N5, killed). The pre-existing
  `variable='design'` path (asymptotes on the inner sequence) collapses its
  asymptotes and "converges" at iteration 29 on a grey blob (N6, M_nd 0.70).
- `move.policy = 'adaptive'` (olh.move.limit, preset `duOlhoffAdaptiveMove`):
  per-element box d_e, Svanberg's asyincr/asydecr (1.2 / 0.7) applied on the
  OUTER design history, clamped to [move.minimum, move.initial]; the inner loop
  is the frozen one (asymptotes reset per outer iteration). **This is what
  delivers natural termination**: the printed test ‖Δρ‖₂ < ε fires on its own
  in every 240×30 run, with no ladder and no stall detector.
- `stop.guards.boxInactiveFraction`, `stop.guards.settledWindow`,
  `move.initial = Inf` admitted, eigensolver fields passed through, `res.aux.Mnd`.

### 240×30, simply supported (Fig. 2a). Paper: 174.7 / 174.7 / 284.9, bimodal

| run | mass | cut-off | box d₀ | r_min | outer | ω₁ / ω₂ / ω₃ (eq. 4, 0.1) | gap | M_nd | ω₁ within 1 % of peak @ |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 4b | 0.1 | 0.04 | 1.8 el | 68 CONV | 157.9 / 159.7 / 366 | 1.1 % | 0.41 | 51 |
| A2 | 4b | 0.1 | 0.10 | 1.8 el | 84 CONV | 164.2 / 167.9 / 175 | 2.2 % | 0.15 | 66 |
| A3 | 4b | 0.1 | 0.20 | 1.8 el | 332 CONV | 167.4 / 197.5 / 367 | 18 % | 0.12 | 81 |
| A5 | 4b, **binary N** | 0.1 | 0.10 | 1.8 el | 201 CONV | 167.8 / 185.3 / 303 | 10 % | 0.12 | 122 |
| **B1** | 4b | 0.1 | 0.10 | **1.3 el** | 324 CONV | **171.5 / 180.1 / 303** | 5.0 % | **0.079** | 75 |
| B3 | 4b | 0.1 | 0.06 | 1.3 el | 230 CONV | 171.2 / 196.8 / 389 | 15 % | 0.088 | 46 |
| B5 | 4b, binary N | 0.1 | 0.10 | 1.3 el | 154 CONV | 171.1 / 179.0 / 216 | 4.6 % | 0.083 | 69 |
| D1 | **4** | 0.1 | 0.10 | 1.3 el | 254 CONV | 170.7 / 178.9 / 300 | 4.8 % | 0.085 | 50 |
| D2 | **4a** | 0.1 | 0.10 | 1.3 el | 296 CONV | 171.9 / 178.7 / 297 | 4.0 % | 0.075 | 62 |
| **F1** | 4 | **0.2** | 0.10 | 1.3 el | 337 CONV | **169.8 / 176.8 / 293** | 4.1 % | 0.094 | **34** |
| F2 | 4 | 0.3 | 0.10 | 1.3 el | 150 CONV | 159.6 / 175.6 / 555 | 10 % | 0.26 | 50 |

Every run terminates on the printed ε-test (RMS 8.8e−4 per element). What is
reproduced at 240×30 with the 1.3-element radius: **the Fig. 3a topology**
(outer chords, end triangles, X-web; M_nd 0.08–0.09, i.e. black-and-white),
**ω₁ = 171–172 (−2 %)**, ω₃ = 293–303 (paper 285), **ω₁ within 1 % of its
final value by iteration 34–75** — the paper's horizon. The early trajectory
matches Fig. 4a (ω₂ peak 309 @ 3, ω₃ peak 467 @ 4, coalescence @ 8).

What is NOT reproduced: (i) the **bimodal end state**. Every run coalesces to
< 1.5 % and stays there for 40–100 iterations, then the pair opens to 4–5 %
(binary detection) or 4–18 % (fixed window, larger boxes) and never closes
again. In B1/D1/F1 the opening coincides with a **localized-mode spike**
(ω₁ momentarily 20–110 rad/s). Spikes occur under (4), (4a), (4b) and a 0.2
cut-off alike, so they live at ρ ≈ 0.1–0.4 — the large grey end wedges — not
below the cut-off. Cut-off 0.3 (F2) removes them but leaves the wedges grey
(mass-free grey material is never penalised). (ii) The **iteration count**:
the ε-test fires at 150–340, not ~80–100, because the tail is spent resolving
the last 5–10 % of grey elements while boxes sit at the 0.002 floor (ε_L2 =
0.075 is only reachable when < 20 % of elements still move at the floor).

### 800×100, same physical radius 0.0433 (= 1.3 el at 240×30 = 4.33 el here)

| run | mass / cut-off | box | ε rule | outer | ω₁ / ω₂ / ω₃ | gap | M_nd |
|---|---|---|---|---|---|---|---|
| C1 | 4b / 0.1 | 0.10 | mesh-scaled 0.25 | 69 CONV | 156.2 / 157.2 / 407 | 0.6 % | **0.45** (false stop, grey) |
| C2 | 4b / 0.1 | 0.06 | mesh-scaled 0.25 | 355 CONV | 168.2 / 168.6 / 217 | 0.3 % | 0.17 (grey ends) |
| C3 | 4b / 0.1, R = 0.06 | 0.10 | mesh-scaled | killed @64 | 71 / 73 / 93 | — | localized-mode collapse |
| E2 | 4 / 0.1 | 0.10 | mesh-scaled 0.25 | 105 CONV | 120 / 166 / 176 | — | 0.29 (stopped inside a spike) |
| E1 | 4 / 0.1 | 0.10 | 0.0246 (h^1.44) | 400 CAP | 167.7 / 221.9 / 459 | 32 % | 0.12 |
| **G1** | **4 / 0.2** | 0.10 | 0.0246 (h^1.44) | 400 CAP | **168.3 / 181.6 / 260** | 7.9 % | **0.096** |

G1 is the 800×100 result: black-and-white (M_nd 0.096), ω₁ = 168.3 (−3.7 %),
ω₁ within 1 % of its final value by iteration 79, coalesced to ≤ 1.2 % until
iteration 150, then opened to 8 % over the next 250 iterations while M_nd
stayed at 0.095. The topology is the same family as Fig. 3a (outer chords,
end triangles, X-web) but with fewer, thicker web members than at 240×30 —
the optimum is non-unique, as CLAUDE.md §7 warns. Under the mesh-scaled ε
(0.25) G1 would have stopped at iteration 191 with ω 169.3 / 176.0, gap 4.0 %,
M_nd 0.095, i.e. essentially the same design: at this mesh the *default* rule
gives natural termination once the localized modes are suppressed (cut-off
0.2), whereas with cut-off 0.1 the same rule stops inside a spike (E2) or on a
grey design (C1).

### Standing findings

1. **The move ladder is not needed for termination.** Svanberg's contraction
   rule on a per-element box gives natural convergence in every run; the
   ladder's S2_CONTINUATION_DEFECT is gone (no stop ever coincides with a
   schedule change, because there is none).
2. **Bimodality is lost late, and localized modes are the trigger.** This is
   the sharpest remaining gap to the paper, which reports "all bimodal". The
   paper's own remedy (mass cut-off at 0.1) does not cover ρ ≈ 0.1–0.4 grey
   regions; the paper's other remedy, Pedersen (2000) linearised stiffness, is
   not implemented and is the next candidate.
3. **ε is not mesh-consistent** (C1 vs G1): the same RMS tolerance stops a grey
   design at 69 iterations at 800×100 and a black-and-white one at 324 at
   240×30. The h^1.44 rescaling of §31 is too tight at 800×100 (never met in
   400). The reachable ε is set by the adaptive floor × the number of
   still-oscillating elements, not by the mesh.
4. The 1.3-element / 0.043 physical radius reproduces ω₃ and the Fig. 3a member
   layout; the frozen 0.06 does not (A2: ω₃ = 175, grey end patches).

### Addendum 2026-09-13 — Pedersen (2000) stiffness linearization (`docs/s001580050130.pdf`)

Implemented as `material.stiffness.model = 'pedersen'` (olh.material.
stiffnessInterpolation; eq. (5) of Pedersen: mass linear, stiffness ρ^p above
`linearBelow`, linear ρ·ρ₀^(p−1) below, printed ρ₀ = 0.1 ⇒ ρ/100). SIMP path
bitwise unchanged (anchor A1 IDENTICAL, suite 0 failures); eq. (19) gradients
verified by FD inside the linear band (rel. err. ≤ 1e−4). Pedersen's second
ingredient (ignoring low-density nodes in the eigenvector convergence test of
his inverse iteration) has no counterpart in a LAPACK/ARPACK solve.

| run | stiffness | mass | outer | native ω₁/ω₂/ω₃ | re-evaluated SIMP + eq.(4) | gap | M_nd | ω₁ settled by |
|---|---|---|---|---|---|---|---|---|
| **P1** | Pedersen 0.1 | eq. (2) linear | 191 CONV | 170.4 / 178.7 / 302 | 171.0 / 178.9 / 300 | 4.6 % | 0.081 | 41 |
| P2 | Pedersen 0.1 | eq. (4) 0.1 | 314 CONV | 170.7 / 178.3 / 299 | 170.7 / 178.0 / 294 | 4.3 % | 0.085 | 29 |
| P3 | Pedersen **0.3** (class C) | eq. (2) | 271 CONV | 168.8 / 168.8 / 361 | **157.3 / 169.3 / 222** | 7.7 % | 0.113 | 33 |

- **P1 is the cleanest trajectory of the whole study**: no localized-mode spike
  at any iteration, smooth monotone ω₁, natural termination at 191, Fig. 3a
  topology, M_nd 0.08. The paper's own cited remedy therefore removes the spikes
  that (4)/(4a)/(4b) leave. Yet the pair still opens from ~1 % at iteration 15
  to 4.6 % at the end **without any spike event**. Conclusion revised: the late
  loss of bimodality is intrinsic to the fixed-window (25d) reconstruction (the
  coupled model has no gap-closing mechanism, NOTES §26), not to localized modes.
- P3's tight bimodality (0.03 % native) does not survive re-evaluation under the
  printed models (157.3 / 169.3): with ρ₀ = 0.3 the 5.6 % of elements at
  0.1 < ρ < 0.3 carry stiffness ρ/11 instead of ρ³ and the "bimodal optimum" is
  an artifact of that band. Rejected.
- 800×100 with P1's settings (H1) launched; result lands in
  `repro/results/H1_800_ped01_lin/` when finished.

**Standing gap after this study: the bimodal end state of Fig. 3a/4a is not
reproduced by any variant that stays within the printed models; every natural
convergence ends 4–5 % apart (fixed window) or pinned at the tolerance (binary).**

**H1 — 800×100, Pedersen 0.1 + linear mass, box 0.10, R = 0.0433, mesh-scaled ε
(2026-09-13).** CONVERGED naturally at 291 (‖Δρ‖₂ = 0.235 < 0.25), ω₁ = 167.8
(−4.0 %), ω₂ = 215.7, ω₃ = 407, M_nd 0.105, spike-free history like P1, ω₁
within 1 % of its final value by iteration 96. Re-evaluated under SIMP + eq. (4):
168.0 / 215.8 / 405. Fig. 3a topology family (chords, end triangles, X-web),
black-and-white. The pair coalesced to 0.8 % by iteration 50 and then opened
steadily to 28 % — the same intrinsic separation as at 240×30, larger here. At
800×100 the printed test fires on the mesh-scaled ε with a black-and-white design
in this configuration, unlike C1/E2 (cut-off 0.1 SIMP, which stopped grey or
inside a spike): the spike-free model is what makes the default ε rule usable at
this mesh.

### Nine-mesh sweep, common physical radius 0.06 — preset `duOlhoffAdaptivePedersen` (2026-09-13)

`repro/results/SWEEP_R06.md` (+ .csv), runs `repro/results/S<mesh>/`. All nine
meshes run in parallel, single-threaded each, so wall times are mutually
comparable but inflated by memory contention relative to a solo run (H1 solo:
1.12 s per inner iteration at 800×100 vs 1.25 here).

| mesh | outer | inner/outer | ω₁ / ω₂ / ω₃ (SIMP + eq. 4) | gap | M_nd | ω₁ settled @ | eig/outer [s] | per inner it. [s] |
|---|---|---|---|---|---|---|---|---|
| 160×20 | 121 | 19.6 | 169.7 / 170.8 / 318 | **0.7 %** | 0.115 | 41 | 0.053 | 0.168 |
| 240×30 | 111 | 18.7 | 167.6 / 187.2 / 309 | 11.7 % | 0.123 | 32 | 0.113 | 0.334 |
| 320×40 | 101 | 19.8 | 166.1 / 195.3 / 350 | 17.5 % | 0.141 | 45 | 0.204 | 0.470 |
| 400×50 | 93 | 20.6 | 166.7 / 198.2 / 338 | 18.9 % | 0.122 | 43 | 0.326 | 0.639 |
| 480×60 | 112 | 19.1 | 166.3 / 203.6 / 368 | 22.4 % | 0.131 | 46 | 0.476 | 0.890 |
| 560×70 | 130 | 18.2 | 166.1 / 206.5 / 386 | 24.3 % | 0.133 | 59 | 0.601 | 1.034 |
| 640×80 | 156 | 18.4 | 166.0 / 206.1 / 387 | 24.2 % | 0.133 | 66 | 0.726 | 1.170 |
| 720×90 | 204 | 18.4 | 165.7 / 202.6 / 384 | 22.2 % | 0.162 | 85 | 1.145 | 1.215 |
| 800×100 | 246 | 18.9 | 165.8 / 195.9 / 407 | 18.2 % | 0.165 | 171 | 1.359 | 1.251 |

- **Every row terminates on the printed ε-test** (no cap, no ladder). No
  localized-mode spike anywhere in the nine histories.
- ω₁ is smooth and nearly flat: 169.7 → 166 → 165.8 (−2.9 … −5.1 % vs 174.7),
  the residual being FE discretization (§8b) plus the coarser element radius.
- M_nd 0.115–0.165, mildly increasing with mesh; the grey band is one physical
  radius wide, so its area fraction is roughly mesh-independent as expected.
- Bimodal only at 160×20 (1.2 el). The gap opens with the element-count radius
  (1.8 el → 12 %, 2.4 → 18 %, ≥ 3 → 19–24 %), matching the §6 threshold of
  ~1.5–2 elements; see the explanation recorded in the reply of 2026-09-13
  (member-size effect: cheap thin diagonals raise ω₂ for free at fine meshes,
  so max min(ω₁, ω₂) is unimodal there).
- Cost columns are smooth and monotone: eigensolve per outer 0.053 → 1.36 s,
  inner MMA per sub-iterate 0.168 → 1.25 s, inner share 94–98 %; inner
  sub-iterates per outer 18–21 at every mesh (tolInner 0.05). The outer count
  is NOT monotone in NE (121, 111, 101, 93, then rising to 246) because the
  per-element box makes the physical evolution rate mesh-dependent; total wall
  time must not be fitted as a power law of NE, per-iteration cost may.

Second sweep launched with `filter.radiusElements = 1.3` at every mesh
(`repro/results/Rel<mesh>/`, table `SWEEP_R13EL.md`) to test whether a
mesh-dependent radius keeps the bimodality across the sweep.

### Nine-mesh sweep, radius fixed at 1.3 ELEMENTS — same preset otherwise (2026-09-13)

`repro/results/SWEEP_R13EL.md` (+ .csv), runs `repro/results/Rel<mesh>/`.
Frequencies re-evaluated under SIMP + eq. (4), cut-off 0.1.

| mesh | outer | ω₁ / ω₂ / ω₃ | gap | M_nd | ω₁ settled @ | eig/outer [s] | per inner it. [s] |
|---|---|---|---|---|---|---|---|
| 160×20 | 262 | 168.2 / 168.7 / 255 | **0.3 %** | 0.123 | 41 | 0.044 | 0.178 |
| 240×30 | 191 | 171.0 / 178.9 / 300 | 4.6 % | 0.081 | 41 | 0.103 | 0.334 |
| 320×40 | 108 | 172.1 / 172.8 / 263 | **0.4 %** | 0.072 | 38 | 0.210 | 0.497 |
| 400×50 | 173 | 172.8 / 181.6 / 318 | 5.0 % | 0.065 | 45 | 0.350 | 0.703 |
| 480×60 | 228 | 173.9 / 191.7 / 361 | 10.2 % | 0.057 | 51 | 0.510 | 0.868 |
| 560×70 | 257 | 174.2 / 191.3 / 348 | 9.8 % | 0.055 | 57 | 0.664 | 0.905 |
| 640×80 | 291 | 174.9 / 193.7 / 366 | 10.7 % | 0.052 | 65 | 0.836 | 1.033 |
| 720×90 | 378 | 175.0 / 198.1 / 386 | 13.2 % | 0.047 | 79 | 1.321 | 1.072 |
| 800×100 | 400 CAP | 175.2 / 213.2 / 494 | 21.7 % | 0.055 | 94 | 1.585 | 1.146 |

Against the 0.06 sweep:
- **ω₁ now reaches the paper's 174.7** from 560×70 upward (175.2 at 800×100)
  and rises monotonically with mesh, because the design may use thinner
  members; under 0.06 it fell to 165.8. Grey fraction halves (M_nd 0.05–0.07
  vs 0.12–0.17). So the "separate radius strategy" wins on ω₁ and on greyness.
- **Bimodality is still not kept.** Only 160×20 and 320×40 end bimodal
  (0.3 %, 0.4 %); the gap is non-monotone (4.6, 0.4, 5.0, 10, 10, 11, 13, 22 %),
  i.e. at a fixed element radius the outcome flips between the nearly degenerate
  bimodal and unimodal optima (§32 reasons, point 3) and the unimodal wins ever
  more clearly with mesh. The 800×100 design has a fine multi-member web
  (thin diagonals raise ω₂ to 213 for free).
- Cost: outer count grows with mesh (108 → 400) and 800×100 hits the cap
  (‖Δρ‖₂ 0.36 vs ε 0.25 — the many thin members keep more elements moving at
  the box floor); inner sub-iterates per outer 17–23; per-iteration costs
  smooth (eig 0.044 → 1.59 s, inner sub-iterate 0.18 → 1.15 s).
- The topology is no longer comparable across rows (member thickness follows
  the element size), which is the price of the element-radius policy.

**Recommendation for the performance comparison.** Neither policy gives
"bimodal at every mesh". The physical radius 0.06 gives comparable topologies,
monotone ω₁ (169.7 → 165.8), monotone M_nd, natural termination at every mesh,
and a gap that grows monotonically with mesh — a smooth table with an honest
"not bimodal beyond 160×20" caveat. The element radius 1.3 gives ω₁ at the
paper's value on fine meshes and half the grey, but a non-monotone gap, a cap
hit at 800×100 and mesh-dependent topologies. For a cross-method cost
comparison the physical radius is the defensible choice, provided the other
two methods use the same physical radius.
