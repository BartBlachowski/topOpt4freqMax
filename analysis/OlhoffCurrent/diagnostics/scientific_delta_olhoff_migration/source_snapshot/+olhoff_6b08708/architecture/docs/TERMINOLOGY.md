# TERMINOLOGY

Names used in this project, what they actually denote, and which of them are
ours rather than Du & Olhoff's.

---

## 1. "M4" — the most dangerous name in the tree

**"M4" does not mean mass model (4), (4a) or (4b).**

`M0`–`M4` are the *multiplicity-treatment candidate indices* of
`audit_multiplicity_reconstruction`. From `WP5_preregistration.md` and
`algo/multRule.m`:

| Label | `cfg.multRule` | What it is |
|---|---|---|
| `M0` | `binary`, `tolMult=0.05` | the memoryless relative-difference test — the original frozen reconstruction |
| `M1` | `binary`, `tolMult=1e-4` | same test at the Krog & Olhoff §5.3 threshold |
| `M2` | `latch` | one-way monotone latch; N never decreases |
| `M3` | `hyst` | two-threshold hysteresis (never run) |
| **`M4`** | **`subspace`, `subN=2`** | **no classifier at all: N fixed at 2, and the subeigenvalue problem carries the actual eigenvalue separation `diag(λ_j−λ_n)` on its diagonal** |

So "M4" is a statement about **multiplicity treatment and the form of eq. (25d)**.
The mass model is an entirely separate axis that happens to share the digit 4
because Du & Olhoff's mass equations are numbered (4), (4a), (4b).

That collision is the reason this file exists. In `audit_m4_topology_restoration`,
`projected_m4`, `frozen_m4`, `du_olhoff_mature_m4` the token `m4` means
*multiplicity candidate 4*, and a reader who assumes it means *eq. (4)* will draw
the wrong conclusion about what was run — the frozen realization uses mass model
**(4b)**, not (4).

### 1.1 What "M4" has come to mean colloquially

By the end of the audit programme "the M4 realization" was used for the whole
frozen conference configuration:

> mass model (4b) **+** p=3 fixed **+** physical filter R=0.06 applied to *all*
> f_sk **+** multiplicity `subspace` with `subN=2` and diagonal offsets **+**
> published MMA on Δρ **+** S2 move ladder on the β signal **+** `settledMove`
> stopping guard **+** ε = 0.05·√(NE/3200) **+** `eigs` with a fixed start vector

That is nine independent choices carried by one two-character token, only one of
which the token actually names.

### 1.2 Ruling

* The mass axis gets semantic names: `eq2`, `eq4`, `eq4a`, `eq4b`.
* The multiplicity axis gets semantic names: `binary`, `latch`, `hysteresis`,
  `subspace`.
* The bundle of nine choices gets a **preset name**, and the preset is the only
  place the bundle exists.
* `M4` survives only as a documented alias in `PRESETS.md`, never in solver code.

**Selecting `eq4b` does not make the solver "M4", and selecting `subspace` does
not tell you the mass model.**

---

## 2. The two βs

| Symbol | Meaning | Where |
|---|---|---|
| `β` | the **bound-formulation variable** of eq. (25a): the scalar being maximized, `max β` subject to `β ≤ ω_j² + Δ(ω_j²)`. A design variable of the inner MMA problem, alongside Δρ. | `hist.beta`, `st.beta`, `cfg.s2Signal='beta'` |
| `β_proj` | the **projection sharpness** of the tanh Heaviside operator. Class D, absent from every source. | `cfg.projection.betaSchedule`, `hist.projBeta` |

They are unrelated. `filter/projectDensity.m` already carries this warning in its
header; the canonical schema keeps them in different branches
(`cfg.projection.beta.*` vs the bound variable, which is never configuration at all).

A third, unrelated β exists in MMA's internals (`subsolv.m`). It never reaches
configuration.

---

## 3. Realization labels that are ours, not the paper's

| Token | Origin | Means |
|---|---|---|
| `S0`,`S1`,`S2`,`S3` | `audit_stepcontrol` | move-limit policy families: fixed, geometric contraction, staged ladder, trust-ratio. **The paper has no move limit at all.** |
| `M0`…`M4` | `audit_multiplicity_reconstruction` | multiplicity candidates, §1 above. |
| `R1`,`R2` | `audit_m4_topology_restoration` | preregistered stopping safeguards: ladder-exhausted and max-design-change. |
| `B0` | same | the *unguarded* baseline re-run, used to prove bitwise reproduction. |
| `Bmature` | `audit_p_continuation` onward | the R2-guarded baseline, i.e. "the frozen realization allowed to finish". |
| `P1` | `audit_p_continuation` | p-continuation **coupled** to the ladder stage. |
| `PD1` | `audit_m4_p_continuation_decoupled` | p-continuation **decoupled** onto its own counter. |
| `PM1` | `audit_pm1_printed_mass_continuation` | PD1 **plus** the printed low-p mass model. |
| `D160` | `audit_m4_projection_invariance` | projection ON at β_proj=0 — the *identity* control that isolates the filter switch from the projection. |
| `T160/T240/T320/T800` | same, and `audit_projection_800x100` | the projection treatment. |
| `REG160` | same | the default-off bitwise regression gate. |
| `TMA` | `audit_termination_mesh_admission` | the audit that froze `CFG(1..3)`; "the TMA realization" = the conference realization. |
| `nodescent` | `audit_s2_final_conference_restoration` | S0 with the move pinned at the ladder's own first level. |

None of these appears in Du & Olhoff (2007) or Olhoff & Du (2014).

---

## 4. Density fields — names that must stay distinct

The projection work introduced three fields where the frozen realization has one.
Using `rho` for all of them is how the stopping test silently changed the object
it monitors (`ARCHITECTURE_VARIANT_INVENTORY.md` §2.6).

| Canonical name | Symbol | Definition | Without projection |
|---|---|---|---|
| **design variable** | `z` | the optimizer's variable, box [0,1] | *is* the density; `z ≡ ρ` |
| **filtered density** | `z̃` | `(H z)/Hs`, the density filter | does not exist |
| **physical density** | `ρ_phys` | `ρ_min + (1−ρ_min)·P(z̃; β_proj, η)` | `= z` |
| **FE density** | — | what `assemble2D` receives | always `ρ_phys` |

`res.rho` is **always the physical density actually used by the FE model**. Under
projection `res.z`, `res.zTilde` and `res.rhoPhys` are recorded alongside it.

Consequently:
* `hist.dxOuter`, `hist.dxNorm2` measure **Δ(design variable)** in both formulations.
* `hist.dxPhys2` measures **Δρ_phys**, and is recorded but never read.
* Du & Olhoff (P4) monitor Δρ where ρ *is* the design variable, so the frozen
  realization's stopping field is unambiguous. The projection runs monitor Δz.
  **This is a real difference in meaning and the presets must state it.**

---

## 5. Words the sources do not contain

Verified by text extraction over both PDFs:

| Term | Du & Olhoff (2007) | Olhoff & Du (2014) |
|---|---|---|
| `filter` | 1 (sensitivities) | **0** |
| `move limit` / `trust region` / `step size` | **0** | **0** |
| `tolerance` (numeric value) | qualitative only | **0** |
| `continuation` | **0** | — |
| `projection` / `Heaviside` | **0** | — |
| `density filter` | **0** | — |

Anything named by a term in this table is ours. Say so.
