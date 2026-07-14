# MASS_INTERPOLATION_DECISION

**Status: ACCEPTED. This is the authoritative project decision on the mass interpolation model.**
**Date:** 2026-07-14
**Supersedes:** every prior statement in this repository that the proposed method uses a global
mass penalization exponent `d = 3`, and every statement that attributes the observed
localized modes to the mass interpolation law.

---

## 1. Decision

> **The proposed method interpolates element mass LINEARLY in the element density, with a
> `ρ_min` regularization floor:**
>
> ```
> m_e(x_e) = [ ρ_min + x_e (ρ_0 − ρ_min) ] · m_e^0
> ```
>
> **The stiffness is penalized with SIMP `p = 3`; the mass is NOT penalized.**
> **No low-density (Du–Olhoff) correction branch is applied.**

This is what both the MATLAB and the Python solvers have always done. **The implementation is
correct. The manuscript was wrong.** The manuscript is corrected to describe the implemented
method; no solver, configuration, or numerical result is changed.

**Two things that were conflated must be kept apart from now on:**

| | |
|---|---|
| **The global mass interpolation law** | How element mass depends on density across the *whole* density range. Ours is **linear**. |
| **A low-density correction branch** | A *separate* device that alters the mass law only for `x_e ≤ 0.1`, to suppress void-mass modes. Du & Olhoff have one. **We do not.** |

The manuscript's `d = 3` arose by collapsing the second into the first. That is the error.

---

## 2. Evidence

### 2.1 What the implementation actually does

| Evidence | Finding |
|---|---|
| [`topopt_freq.m:51`](analysis/ourApproach/Matlab/topopt_freq.m#L51) | `pmass = localOpt(runCfg, 'pmass', 1.0)` — default **1** |
| [`topopt_freq.py:134`](analysis/ourApproach/Python/topopt_freq.py#L134) | `pmass = float(run_cfg.get("pmass", 1.0))` — **identical default** |
| [`our_mass_interpolation.m`](analysis/ourApproach/Matlab/our_mass_interpolation.m) | `power` mode: `m = x.^pmass`; then `rho_e = rho_min + m·(rho0 − rho_min)` |
| Active configs (`clamped_beam_200x25.json`, `clamped_beam_400x50.json`, `BuildingTopOptFreq.json`) | **none sets `pmass`** → all inherit the linear default |
| Active experiment scripts (`exp2_authoritative_sweep.m`, `exp3_authoritative_mesh_convergence.m`, `exp2b_building.m`) | **none sets `pmass`** |

**Every production result in this repository was computed with linear mass.**

### 2.2 Provenance — neither source was a decision

| Evidence | Finding |
|---|---|
| `git log -S 'p = d = 3' -- paper/main.tex` | single commit **`78b26e1 "Ai paper writter"`** |
| `git log -S 'x_e^d' -- paper/main.tex` | **same commit** |
| `git log -S "'pmass', 1.0"` | **`ffcb81c "elastic2D implemented"`** — an unrelated feature commit |
| `git log -S "'pmass', 3"` | **no commit, ever** |
| Gate A0 "Authoritative Formulation" (A0-F1 … A0-F6) | pins the reference design, `ω₀²`, `ρ_nodal` scaling, mode normalization, and load sensitivity — **never mentions the mass interpolation** |

The manuscript's exponent was introduced by a drafting pass; the solver's default was inherited
from an unrelated commit. **The mass model has never been the subject of a project decision.
This document is that decision.**

---

## 3. Mathematical justification

### 3.1 A global `d = p` makes frequency independent of density

For a uniform density field `x = c` (no passive elements), `K(c) ∝ c^p K̂` and `M(c) ∝ c^d M̂`, so

```
ω²(c) ∝ c^(p − d) · ω²(solid)
```

- **`d = p = 3` ⟹ `ω²(c) ∝ c⁰ = 1`.** A uniform structure would have **the same natural
  frequencies at any density.** A 1%-dense beam would ring exactly like the solid beam, and
  adding material would confer no frequency benefit whatsoever. For a method whose purpose is
  frequency maximization by material placement, this is degenerate.
- **`d = 1` ⟹ `ω²(c) ∝ c²`, `ω ∝ c`.** Densifying raises frequency. Physical.

### 3.2 A global `d = 3` contradicts the volume interpretation

`x_e` is the material **volume fraction** of the element, and the constraint `Σ x_e V_e ≤ V_f` is
**linear** in `x`. Under `m ∝ x³`, a design at `x = 0.5` would consume 50% of the volume budget
while weighing 12.5% of solid. Volume and mass would describe two different structures.

### 3.3 A global `d = 3` falsifies the inertial load

The method's premise is `f(x) = ω₀² M(x) Φ₀` — **an inertial force**, justified by Rayleigh's
principle. If `M(x)` is not the physical mass, `f` is not an inertial force and the Rayleigh
argument collapses. Linear mass is what makes the load mean what the paper says it means.

**All three arguments are independent, and all three select `d = 1`.**

---

## 4. Literature justification

### 4.1 Du & Olhoff (2007) use linear mass in the physical range

Their Eq. 4b is implemented in this repository as `du2007_c1`
([`our_mass_interpolation.m`](analysis/ourApproach/Matlab/our_mass_interpolation.m)):

```
m = x                       for x > 0.1        ← LINEAR
m = c1·x⁶ + c2·x⁷           for x ≤ 0.1        (c1 = 6e5, c2 = −5e6)
```

The mass law is **linear wherever the material is physically present**. The high exponent applies
**only** to the void branch. The manuscript's `d = 6` referred to that branch's leading power and
was mis-transcribed into a global exponent.

### 4.2 Du & Olhoff's own published initial frequencies require linear mass

From the archived FE verification
([`initial_frequency_verification.md`](analysis/OlhoffApproachExactOpus/experiments/initial_frequency_verification.md)):

> *"At `ρ=0.5`, `ω² ∝ 0.5^{p−1}` … p=3 gives the published 68.7/104.1/146.1 … p=1 gives
> ω ≈ 143/211/295 (**≈2× too high**)."*
> *"At `ρ=0.5` **all mass models (eq. 2/4/4a/4b) coincide** (`ρ>0.1`)."*

The exponent `p − 1` **is** `p − d` with `d = 1`. Under a global `d = 3` the exponent would be
`0`, and the uniform `ρ = 0.5` start would return the **solid** frequencies — exactly the
143/211/295 values that are **twice** the published ones. **The reference benchmark itself is
reproducible only with linear mass.**

*(This artifact lives in the retired `OlhoffApproachExactOpus` tree. It is cited here as a
diagnostic FE check on a literature mass law, never as reviewer evidence. The conclusion also
follows from §3.1 without it.)*

### 4.3 Yuksel — the method's direct ancestor — uses linear mass

[`top99neo_inertial_freq.m:356`](analysis/YukselApproach/Matlab/top99neo_inertial_freq.m#L356):
`rhoe = rho_min + (rho0 - rho_min) * xSolid;`

---

## 5. Rejected alternatives

| Rejected | Why |
|---|---|
| **Global `d = p = 3`** (the manuscript's equation) | Unphysical (§3.1–3.3); fails Du & Olhoff's published initial frequencies by **2×** (§4.2); **never executed anywhere in this repository** — there is not one artifact for it. |
| **"Adopt `d = 3` and rerun everything"** | Would spend the entire remaining campaign installing a model refuted by the paper's own reference benchmark. Rejected on scientific grounds, not cost. |
| **Global `d = 6`** | The S1 mitigation value. It is a *low-density device applied globally*; it rescued the tracked mode (MAC 0.786 → 0.974) but left 9 of 10 modes localized and increased component fragmentation (126 → 198). Not a global mass law, and not adopted as one. |
| **Adopting Du & Olhoff's low-density branch (`du2007_c1`)** | **Tested and refuted** on EXP3 400×50 (`archive/diagnostics/eq4b_hypothesis_test/`): the run **capped 2000/2000**, failed the A5 lowest-mode check (physical mode was mode 2), doubled grayness (0.061 → 0.131), and still left 9 localized modes. **Not adopted.** The option remains available and is recorded as an open question (§9). |

---

## 6. The localized-mode question — what may and may not be claimed

**It may NOT be claimed that any mass exponent eliminates the localized modes.** The repository
has tested three mass models on the same EXP3 400×50 case, and **none of them removes the family**:

| | `pmass = 1` (linear) | `pmass = 6` | Du–Olhoff Eq. 4b |
|---|---:|---:|---:|
| `ω₁` (rad/s) | 64.39 | 131.93 | 77.01 |
| tracked MAC | 0.786 | 0.974 | 0.924 |
| grayness | 0.061 | 0.054 | **0.131** |
| iterations | 1750 | 1579 | **2000 (capped)** |
| A5 lowest-mode check | — | pass | **fail (mode 2)** |
| **localized modes (of 10)** | **8** | **9** | **9** |

**What the current evidence indicates instead.** In the diagnosed fine-mesh designs, the modes
flagged "localized low-density":

- carry **`low_density_kinetic_fraction = 0.0000`** — essentially **no kinetic energy in the
  low-density material** — at **both** `pmass = 1` **and** `pmass = 6`;
- are flagged on their **strain**-energy fraction (0.99+) in low-density elements;
- have their kinetic energy on **solid components that are not connected to the supports**
  (`dominant_component_touches_both_supports: false`; component ids 22/81/82/83/114/115, never
  the support-connected component 1);
- occur in designs with **126 (pmass=1) and 198 (pmass=6) disconnected solid components**.

This is **not** the classical void-mass mechanism (which would put kinetic energy *in* the void).
It is consistent with **solid islands resonating on a weak-void foundation** — the isolated-island
and point-connection modes already discussed in the cited literature (Deng et al., 2024).

> **This is stated as current evidence, not as a proven theorem.** We have not established that
> disconnection is the sole mechanism, we have not isolated the role of `E_min`, and the mass
> exponent demonstrably does affect *which* mode is lowest. The claim is an association supported
> by the diagnostics above, and it is presented as such in the manuscript.

---

## 7. Repository impact

**No solver, algorithm, configuration, or numerical result changes.** The implementation was
already correct. The changes are documentary:

- **Manuscript** — the mass equation, the eigenvalue-sensitivity equation, the §4.1 material
  paragraph, and the localized-mode paragraph (see §8).
- **Project documents** — every statement asserting a global `d = 3`, and every statement
  attributing the localized modes to the mass interpolation, is corrected.
- **Gate A0** — must be extended: it never pinned the mass interpolation. **This document is that
  pin.** Any future config setting `pmass ≠ 1` (other than an explicitly labelled diagnostic) is a
  deviation from the declared method.

---

## 8. Manuscript impact

| Location | Change |
|---|---|
| Eq. (1), [main.tex:211-214](paper/main.tex#L211-L214) | `m_e = x_e^d m_e^0`, `p = d = 3` → **linear mass with `ρ_min` floor**, matching the code exactly. |
| Eigenvalue sensitivity, [main.tex:228-235](paper/main.tex#L228-L235) | `− λ_j · d · x_e^{d−1} · φᵀm_e^0φ` → `− λ_j · (ρ_0 − ρ_min) · φᵀm_e^0φ` (the linear-mass derivative, independent of `x_e`). |
| §4.1 material paragraph, [main.tex:344](paper/main.tex#L344) | *"The SIMP penalization exponent is `p = 3` for both stiffness and mass"* → `p = 3` penalizes **stiffness only**; mass is linear. |
| Localized-mode paragraph, [main.tex:237-238](paper/main.tex#L237-L238) | The `d ≫ p` / `d = p = 3` conflation is removed. The global law and Du & Olhoff's low-density branch are explicitly distinguished; the claim *"without inducing localized modes"* is **withdrawn**; the disconnected-component evidence is stated, hedged. |

**No reported frequency, MAC value, gain, table, or figure changes** — they were all computed with
the linear law that is now correctly described.

---

## 9. Computational impact

> ## **Zero. Not one hour of the remaining campaign is invalidated.**

EXP2, EXP2b, EXP3, S1, and CR2 all ran with linear mass — the model now declared authoritative.
The manuscript is being corrected *to match the results*, not the results recomputed to match the
manuscript. No rerun is triggered by this decision.

---

## 10. Remaining open questions

These are **not** resolved by this decision and must not be presented as resolved.

1. **Should a low-density correction branch be adopted at all?** Du & Olhoff's Eq. 4b was tested
   and refuted on EXP3 400×50 (capped, A5 fail, grayness doubled). The method currently has **no
   remedy** for the artefact it names. This is now an openly stated limitation, not a solved
   problem.
2. **What actually causes the localized modes?** The disconnected-component association is
   evidence, not proof. The role of `E_min` (`10⁻⁶E₀` for the clamped/building cases, `10⁻⁹E₀` for
   the SS beam) has not been isolated, and no connectivity constraint has been tested.
3. **The `x_e` lower bound.** The manuscript declares `x_e ∈ [10⁻³, 1]`; the interaction of that
   bound with `ρ_min` and `E_min` in the void regime is undocumented.
4. **`ρ_min` is not uniform across the examples** (`10⁻⁹ρ₀` in §4.1, `10⁻⁶` in §4.2/§4.3). This is
   faithfully reported but never justified.
5. **S1's scientific goal remains unmet.** No tested configuration produces a clean spectrum. This
   decision does not change that; it changes only the *explanation* that may be offered for it.

**Not affected by this decision:** the EXP1/EXP5 retirement, the `OlhoffApproachExact` retirement,
the absence of a cross-code performance benchmark, and the A4 specification's structural findings
(single-factor design, true-`ω₁` endpoint, the missing refresh capability on the `semi_harmonic`
path).
