# Du & Olhoff (2007) — exact reproduction

## Objective

Reproduce the algorithm of Du & Olhoff, *Topological design of freely vibrating continuum
structures for maximum values of simple and multiple eigenfrequencies and frequency gaps*,
Struct Multidisc Optim 34:91–110 (2007).

**Exactness requirement.** This is a reproduction study covering results *and* efficiency and
complexity. Improved, modernised or "equivalent but better" substitutions are not acceptable.
Where the paper is ambiguous, resolve it from sources of 2007 or earlier in the authors' own
lineage (Svanberg 1987, Sigmund 1997, Krog & Olhoff 1999, Seyranian, Lund & Olhoff 1994),
never from later literature. Where the paper is silent, treat the quantity as a swept control
parameter and record it per run — do not pick a "sensible" value and bury it.

---

## 0. Status pointer (added by the reproduction work)

`NOTES.md` is the running evidence log. As of 2026-08-25:

- **§3 is RESOLVED.** Fig. 4a read from the clean PDF (first marker is iteration 1,
  not 0) gives ω ≈ 71 / 245 / 428. Mid-height supports **with ux restrained at both
  ends** reproduce 68.40 / 253.4 (bending) / 420.8 (extensional) and remove the
  mesh-dependent ω₂/ω₃ swap. Corner supports fail at −6% or +39%. Note the paper
  *draws* corner supports — the drawing and the numbers disagree; the numbers won.
- **The erratum is double-sourced.** Olhoff & Du (2014) prints (19d) already corrected.
  The 2014 paper closes nothing else — it has zero mentions of filtering.
- **Eq. (19) and the erratum form of (25d) are verified against finite differences**;
  dropping the off-diagonals mispredicts degenerate increments by 11–250%.
- Run constraints in force: mesh never below 160×20, conclusive test at 240×30,
  every test compared against the paper's topology *image* as well as its numbers.

---

## 1. MANDATORY: use the erratum

`docs/` contains the Publisher's Erratum (SMO 34:545, DOI 10.1007/s00158-007-0167-6).

As printed in the main article, equations **(25d), (26f), (26g)** are wrong — the `Δ` is missing:

    printed:   det | f_sk^T Δρ  −  δ_sk (ω²) |  = 0        WRONG
    correct:   det | f_sk^T Δρ  −  δ_sk Δ(ω²) | = 0        USE THIS

The erratum also corrects the Fig. 2 caption symbols (ω^φ → ω⁰). Implementing the printed
form reproduces nothing; it is not a well-posed problem.

---

## 2. What is settled

### Robust (verified numerically, factor-of-two effect)

- **p = 3 and q = 1, fixed from iteration 0.** §2.1 says p is "normally assigned values
  increasing from 1 to 3 during the optimization process", but the reported initial
  eigenfrequencies are only consistent with p = 3 held constant. p = 1 is off by roughly a
  factor of two. Do **not** implement continuation unless a run demonstrably requires it.
- Initial design: uniform ρ_e = 0.5 everywhere. Volume fraction α = 50%.
- 2D material: E = 1e7, ν = 0.3, ρ_m = 1. Domain a = 8, b = 1, plane stress.
- 3D material: E = 1e11, ν = 0.3, ρ_m = 7800. Plate a = b = 20, t = 1, eight-node solid
  elements with Wilson incompatible displacement modes.
- ρ_min = 1e-3 (eq. 7e), single-material case.

### Verified algebraically

The mass-interpolation coefficients in (4a)/(4b) are internally consistent and can be coded
verbatim: at ρ = 0.1, c₀ρ⁶ = 1e5·1e-6 = 0.1 gives C⁰; c₁ρ⁶ + c₂ρ⁷ = 0.6 − 0.5 = 0.1 with
slope 6c₁ρ⁵ + 7c₂ρ⁶ = 36 − 35 = 1 gives C¹. Baseline is (4); (4a)/(4b) are the paper's own
stated cross-check and should reproduce it to within "negligible differences" (§2.2) — verify
that claim rather than assuming it.

### Tentative — do NOT treat as established

- **Support idealization in 2D.** Supports applied at *mid-height of the end faces* reproduce
  ω₁⁰ = 68.7 and 104.1 to within 0.2%; bottom-corner supports give 65.1 (5% low). BUT this is a
  5% effect and plain Q4 shear locking is comfortably within that band. If the elements were
  incompatible-mode or Q8, corner supports could match equally well. **Confounded with element
  formulation — keep both as switchable options.**
- Mesh ~64×8 to 80×10 for the 2D beam reproduces ω₁⁰ well, but ω₁ varies < 0.3% across
  40×5–160×20, so this is weak evidence. NE is never reported anywhere in the paper.

---

## 3. Unresolved discrepancy — investigate early

At iteration 0 for the simply supported 2D beam (Fig. 2a), a plane-stress Q4 model with the
settled parameters gives:

| mode | computed (80×10) | character            | Fig. 4a at iter. 0 |
|------|------------------|----------------------|--------------------|
| ω₁   | 68.6             | 1st bending          | 68.7  ✓            |
| ω₂   | 255.2            | 2nd bending          | ≈255  ✓            |
| ω₃   | 258.5            | **extensional**      | ≈430  ✗            |
| ω₄   | 520.2            | 3rd bending          | —                  |

Two problems:

1. The paper appears to have no extensional mode below ~430. Ours sits at ~258. Candidate
   explanations: axial restraint at both ends, different element formulation, or a misreading
   of the figure (the ≈430 value was read off a low-resolution scan and is uncertain — **read
   the initial red-triangle value in Fig. 4a from the clean PDF in `docs/`, it discriminates
   between candidate models**).
2. At 160×20 the extensional mode (248) and 2nd bending (253) **swap order**. With n = 1 and
   N = 2 at the optimum, J = n + N = 3, so the mode receiving bound constraint (25b) is
   mesh-dependent — right in the region where the ω₁/ω₂ coalescence occurs. This is
   algorithmically live, not cosmetic.

---

## 4. Unknowns to sweep (never hard-code silently)

| Parameter | Status | Why it matters |
|---|---|---|
| Filter radius r_min | **Never stated in the paper, for any example** | Strongest determinant of member thickness in every figure |
| Multiplicity tolerance | "predefined, very small tolerance" (§3.5.1), value never given | Decides N and R every iteration; branches the algorithm |
| Move limit on Δρ | Not stated | (18) is only locally valid; no move limit may diverge |
| Off-diagonal switch | See §5 below | Changes inner-loop complexity from O(NE·N³) to O(NE·N) |
| Element formulation (2D) | "plane stress elements", nothing more | Confounded with support placement, see §2 |
| Mesh (NE) | Never reported | See §3 |
| Inner-loop convergence, outer ε | Not stated | Iteration counts, i.e. the efficiency result |

Each run must write a config recording all of these.

---

## 5. The central algorithmic gap — read this before coding the inner loop

§3.5.2 is explicit that the independent variables are β (or β₁, β₂) and Δρ_e, and that the
Δ(ω²_j) are **dependent**. So the inner loop must, per MMA sub-iterate, evaluate them from
(25d): they are the eigenvalues of the N×N symmetric matrix

    A(Δρ),   A_sk = f_sk^T Δρ,     s,k = n .. n+N-1

The paper never states how the constraint gradients are formed. The reconstruction is

    ∂Δλ_j/∂Δρ_e = Σ_{s,k} v_js v_jk (f_sk)_e

with v_j the eigenvector of A. **This is a reconstruction, not the authors' text** — document it
as such. Note it is itself non-differentiable when the Δλ_j coalesce inside the inner loop.

**Baseline configuration = full nonlinear coupling + MMA.** §3.5.3 states MMA was used. The
linear-programming route (imposing f_sk^T Δρ = 0 for s≠k, per Krog & Olhoff 1999) is presented
in the final paragraph as something one *may* additionally do — grammatically an option, not a
report of what was run. Implement it as a labelled second configuration, not as the reproduction.

Also unresolved: the filter takes one sensitivity vector, but the multiple case has N(N+1)/2
vectors f_sk. Filtering only the diagonal f_jj versus all of them are different algorithms.
Keep both branches switchable; which one reproduces Fig. 4a is a result.

### Other undefined cases — log, do not patch

- Formulation (15a–c) constrains only j ≥ n. Nothing prevents ω_{n−1} overtaking ω_n. The paper
  is silent. Do nothing; if it breaks, that is a finding about the method.
- (25b) assumes the J-th eigenfrequency is simple. If it comes up multiple, there is no defined
  procedure. Log the occurrence.
- Bimaterial: box (7e) with ρ_min = 1e-3 is inherited but arguably should be 0..1. Unstated.
- Fig. 10: m₀ is "the total structural mass of the plate" — ambiguous between the full solid and
  the 50% design. Fig. 9 separately uses m_b = "given mass of total structural material".
- How concentrated masses m_c attach (single node vs distributed patch) is unstated.

---

## 6. Environment

MATLAB, end to end. Available in the working folder:

- `mmasub.m`, `subsolv.m` — Svanberg. **Use `mmasub`, not GCMMA.** GCMMA is the 1995/2002
  globally-convergent variant and would change iteration counts, i.e. the quantity being
  measured. Licensed to the requester: keep local, do not commit to a public repo.
- `top88.m` — Sigmund et al. Offers both sensitivity and density filtering. **Use the
  sensitivity filter (`ft = 1`)**; the paper cites Sigmund (1997) and filters sensitivities.
  Note top88's normalisation `max(1e-3, x(:)).*Hs` — that guard is part of published behaviour,
  keep it rather than cleaning it up.
- `docs/` — the 2007 paper, the erratum, and **Olhoff & Du (2014)**. Check the 2014 paper first:
  it may restate this method with more implementation detail and could close the filter-radius
  or multiplicity-tolerance gaps directly. This is the cheapest possible win.

### Numerical hygiene

- **Use `eig(K,M)` on the reduced free-DOF system, not `eigs`.** ARPACK starts from a random
  vector, so mode ordering near degeneracies is non-deterministic between runs — and there is a
  near-degeneracy sitting on top of the coalescence being studied (§3). Dense LAPACK costs
  nothing at ~800 elements. If 3D meshes later force `eigs`, fix the start vector explicitly.
- **`maxNumCompThreads(1)` for all timing runs**, and report it. Otherwise BLAS multithreading
  thresholds smear the inner-loop scaling measurement and the measured exponent will be wrong in
  a way that looks like a real result.
- β (or β₁, β₂) is an MMA design variable alongside Δρ, not a separate quantity.

### Per-iteration complexity (target of the efficiency study)

Outer: FE assembly O(NE); eigensolve for J = n+N modes (dominant); generalized gradients (19)
O(NE·N²) (+ O(NE·R²) for problem 26); filter O(NE·n_r).
Inner, per MMA sub-iterate: form A(Δρ) O(NE·N²); N×N eigensolve O(N³) negligible; constraint
gradients **O(NE·N³)** ← bottleneck; MMA subproblem O(NE).

---

## 7. Validation targets

Frequencies (2D beam, Figs. 2–9):

- initial: 68.7 / 104.1 / 146.1
- max ω₁: 174.7 / 288.7 / 456.4  (+154% / +177% / +212%, all reported bimodal)
- max ω₂: 598.3 / 732.8 / 849.0  (all bimodal)
- mean-eigenvalue baseline (§4.2): ω₁ = 161.7 (simple), ω₂ = 444.5, ω₃ = 805.6;
  compare against ω₂ₐ = 174.7, ω₃ₐ = 284.9 of the optimum design
- gap example (Fig. 9): final ω₃−ω₂ = 810

Frequencies (3D plate, Figs. 10–19):

- initial: 8.1 / 31.1 / 3.5;  four-corner-plus-centre support case 24.6 (bimodal)
- max ω₁: 16.4 / 65.4 / 9.7 (all unimodal); centre-supported case 60.3 (bimodal)
- max ω₂: 46.0 (trimodal) / 155.4 (bimodal) / 39.8 (bimodal)
- bimaterial max ω₄/ω₅/ω₆: 243.8 (bimodal) / 249.7 (unimodal) / 353.2 (bimodal)
- bimaterial gap (Fig. 19): (ω₃−ω₂)_opt = 31.7

**The reported multiplicities are the sharper test.** Frequency values are relatively forgiving;
whether ω₁ and ω₂ actually coalesce and *stay* coalesced depends directly on the multiplicity
tolerance and on whether off-diagonal coupling was retained. Prioritise reproducing Fig. 4a's
coalescence over matching 174.7 to three digits.

Expect topologies to differ visibly in thin members even when frequencies match — these optima
are strongly non-unique. The paper's own Fig. 6 vs Fig. 3a comparison shows the sensitivity.

### Suspected typo — do not chase

§4.4 text reports the Fig. 9 gap of 810 as "no less than 548%" larger than the initial gap. That
implies an initial gap of ~125, but Fig. 9c shows ω₂ starting near 300 and ω₃/ω₄ near 830, i.e.
an initial gap of ~530 → an increase of ~53%. Most likely a misplaced decimal (54.8%). Verify
against the clean figure, but do not tune to hit 548%.

---

## 8. Suggested layout

    mma/      untouched, read-only: mmasub.m, subsolv.m
    filter/   top88-derived sensitivity filter, ft=1
    fem/      Q4 plane stress (switchable incompatible modes); 8-node solid + Wilson modes
    algo/     §3.5 outer loop and inner loop; erratum forms of (25d)/(26f)/(26g)
    runs/     one config per experiment: filter radius, multiplicity tolerance, move limit,
              off-diagonal switch, element formulation, mesh, thread count
    docs/     2007 paper, erratum, Olhoff & Du 2014

Every figure produced must be traceable to a config in `runs/`.
