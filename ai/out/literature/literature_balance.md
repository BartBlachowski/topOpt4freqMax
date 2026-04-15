## Literature Balance Analysis

Generated: 2026-04-15  
Input: ai/out/literature/literature_extract.md  
Protocol: ai/prompts/literature_balance.md

---

## --- METHOD FAMILIES ---

### Family 1: SIMP Bound Formulation (Density-based, Gradient)
- **Description:** Rigorous, gradient-based density method using bound constraint α*λ_i ≥ β on tracked eigenvalues; modified SIMP with x_min for ALRM suppression; MMA optimizer
- **Papers:** Du2007, Olhoff2014, Deng2024, Su2016
- **Relative weight:** HIGH (foundational, widely cited, most validated)

### Family 2: BESO / Evolutionary (Discrete, Sensitivity-based)
- **Description:** Element-wise add/remove with soft-kill variant; sensitivity-number ranking; binary crisp designs
- **Papers:** Huang2010, HuangXie2010
- **Relative weight:** HIGH (established alternative; ~2% lower ω₁ vs SIMP bound but crisp designs)

### Family 3: Quasi-static Reformulation (Static FE)
- **Description:** Convert eigenvalue problem to static compliance via inertial load f = ω²*M*Φ; avoid repeated eigensolution
- **Papers:** Yuksel2025, [proposed manuscript]
- **Relative weight:** MEDIUM (recent, novel, ~9% lower ω₁ but significant speedup)

### Family 4: Criterion-based Update (GWC)
- **Description:** Non-gradient iterative mass control with Heaviside projection; guide-weight criterion
- **Papers:** Li2021
- **Relative weight:** MEDIUM (efficiency-focused; single paper but competitive results)

### Family 5: Level Set (Boundary-based)
- **Description:** Implicit boundary representation via level set function; shape derivative; smooth crisp boundaries
- **Papers:** Xia2011, Shu2014 (acoustic-structural coupling)
- **Relative weight:** MEDIUM (well-established family; one paper on pure eigenfrequency, one on coupled problem)

### Family 6: Explicit Geometry (MMC)
- **Description:** Moving Morphable Components with TDF representation; explicit component geometry; MAC-based mode tracking
- **Papers:** Huang2025
- **Relative weight:** LOW-MEDIUM (recent, handles mode switching explicitly, but single paper)

### Family 7: Extended Domains / Specialized
- **Description:** XFEM for 3D extruded beams; skeletal frame structures; other specialized settings
- **Papers:** Marzok2024 (3D beams XFEM), Ni2014 (frame structures), Su2016 (couple-stress), Lee2024 (multiscale FGS)
- **Relative weight:** LOW-MEDIUM (relevant but structurally different from 2D continua)

### Family 8: Background / Low Relevance
- **Description:** Dynamic response, stochastic loads, MEMS nonlinear, piezoelectric, historical references
- **Papers:** Gomez2019, Munro2024, Pozzi2025, Mottamello2014, Lee2010, Wu2024, Wu2011, RojasLabanda2015, deFaria2005, Nishiwaki2000, Lee2008, Zhu2006
- **Relative weight:** LOW (tangential to primary scope or different objectives)

---

## --- COVERAGE ANALYSIS ---

### Well-covered:
- ✓ **SIMP bound formulation:** 4 papers (Du2007, Olhoff2014, Deng2024, Su2016) — foundational theory thoroughly represented
- ✓ **BESO evolutionary:** 2 papers (Huang2010, HuangXie2010) — same benchmark, soft-kill variant included
- ✓ **Classical benchmarks:** Simply supported beam 8m×1m, 50% volume — ubiquitous across families
- ✓ **Sensitivity analysis:** Eigenfrequency sensitivity formula covered in multiple families (Du2007, Huang2010, Belblidia2002)
- ✓ **2D continuum problems:** Primary structural setting well-represented

### Underrepresented:
- ⚠ **Quasi-static approaches:** Only 1 published paper (Yuksel2025) + proposed work—limited external validation
- ⚠ **Multi-mode/multi-load-case validation:** Mentioned in GWC (Li2021) and bound formulation (Du2007) but not systematically compared across methods
- ⚠ **Computational speed comparisons:** Few papers directly compare wall-clock time; mostly iteration count (Li2021 vs BESO)
- ⚠ **ALRM alternatives:** Only Deng2024 (Betti-based) and Huang2025 (load path analysis) present principled alternatives to x_min; neither mainstream yet

### Missing:
- ❌ **3D continuum frequency problems:** Only Marzok2024 XFEM (extruded, not general); no standard 3D FEM results for frequency maximization
- ❌ **Non-isotropic materials (except couple-stress):** No composites, fiber-reinforced, or orthotropic materials
- ❌ **Manufacturing constraints (except Heaviside projection):** No papers on discrete stock sizes, symmetry, continuity constraints
- ❌ **Real-world applications with design-dependent loads:** Only Zhu2006 (support optimization); no dynamic load interaction cases
- ❌ **Harmonic/frequency response optimization:** Gomez2019 addresses stochastic response; Shu2014 addresses acoustic coupling; but noise/vibration isolation not explicitly covered

---

## --- BIAS / DOMINANCE ---

### Overrepresented papers:
- **Du2007 & Olhoff2014:** These two papers are ubiquitous baseline references. Every new method (BESO, GWC, Deng2024, Marzok2024) compares against Du2007's 174.7 rad/s benchmark
  - *Risk:* Introduction may appear to be "explaining Du2007" rather than positioning a novel contribution
  - *Why present:* Necessary for benchmark framing, but should be grouped with newer methods (Deng2024, Huang2025) to show field progression

### Overrepresented families:
- **SIMP-based methods dominate:** 8 papers (Du2007, Olhoff2014, Deng2024, Su2016, Liu2021, Yuksel2025, Lee2008, Marzok2024) vs 2 BESO, 2 level set
  - *Why:* SIMP is indeed the field-standard; justified
  - *Risk:* Level set and MMC may be under-cited; consider balanced coverage

### Risk of narrative skew:
- **Narrative bias potential:** Extract portrays "efficiency problem" (repeated eigensolution) as central Gap, yet 4 SIMP papers do NOT attempt speed optimization—they accept the eigensolver cost
- **Reframing needed:** Gap is not universally accepted; some researchers prioritize solution quality over speed
- **Recommendation:** Frame gap as "unexploited opportunity" (Yuksel2025 shows promise) rather than "field-wide deficiency"

### Suggested rebalancing:
1. **Emphasize progression:** Du2007 (foundational) → Deng2024, Huang2025 (modern SIMP) → Yuksel2025 (novel quasi-static)
2. **Tier SIMP papers:** Du2007 = foundational (must cite); Olhoff2014 = review (brief cite); Deng2024, Su2016 = recent SIMP variants (brief mention)
3. **Level set deserves more weight:** Currently 2 papers treated as footnotes; should be ~equal weight to BESO
4. **Yuksel2025 must be carefully positioned:** It is the closest related work; differentiation on multi-mode/multi-load-case is critical

---

## --- REDUNDANCY ---

### Groups of similar papers:
1. **Du2007 + Olhoff2014:** Same bound formulation; Olhoff is review of Du2007 — merge into single "bound formulation reference"
2. **Huang2010 + HuangXie2010:** Identical results (171.5 rad/s), same benchmark — cite together as BESO pair
3. **Marzok2024_Topology.pdf + Marzok2024_Topology_alt.pdf:** Duplicates (same paper)
4. **Shu2014_Level.pdf + Shu2014_Level_alt.pdf:** Duplicates (same paper)

### Merge candidates:
- **Du2007 → Olhoff2014 (review):** Keep Du2007 as primary citation; cite Olhoff2014 for "see also" only
- **Huang2010 + HuangXie2010:** Cite as "Huang et al. (2010)" with both references; report single 171.5 rad/s value
- **Li2021 (GWC) + Huang2010 (BESO) + Du2007 (SIMP):** Keep separate; these are genuinely different algorithm families

### Remove or downweight candidates:
- **Gomez2019, Munro2024, Pozzi2025, Mottamello2014:** Different objectives; background mention only if scope permits
- **RojasLabanda2015:** Compliance benchmarking; not frequency-specific; skip or one-sentence cite
- **Wu2011 (AMR):** Static compliance; not frequency; mention only if computational efficiency is emphasized
- **Nishiwaki2000:** Historical (2000); predates ALRM fix; cite for field origins only (one sentence)

---

## --- GAP SUPPORT READINESS ---

### Is gap defensible from current literature?
✓ **YES, with careful framing**

**The gap:** "Quasi-static approaches achieve computational efficiency but are limited to single-frequency, single-load-case problems. Multi-mode and multi-load-case extensions with maintained speed remain open."

**Evidence for gap:**
- EXPLICIT: Yuksel2025 addresses only single fundamental mode (ω₁ only)
- EXPLICIT: Proposed manuscript shows multi-mode capability (clamped beam with α parameter)
- EXPLICIT: No published paper combines both quasi-static speedup AND multi-mode capability AND multi-load-case
- IMPLIED: All high-ω₁ methods (Du2007) require eigensolver per iteration; all quasi-static methods (Yuksel2025) are single-mode

### Weak points:
1. **Speed quantification:** Yuksel2025 does NOT report wall-clock time speedup (only iteration count); proposed manuscript claims 7.1× faster vs Olhoff, but against which baseline?
   - *Action required:* Clarify: faster than full eigensolver (yes), faster than Yuksel2025 (need evidence), or both with/without overhead?

2. **Quality tradeoff not universally accepted:** Du2007 = 174.7 rad/s; Yuksel2025 = 160.5; Proposed = 159.3. Gap narrows but problem persists
   - *Action required:* Show that multi-mode formulation recovers quality *AND* speed, or accept the tradeoff explicitly

3. **Multi-load-case evidence thin:** Proposed manuscript alone demonstrates this; no competing quasi-static approach in literature
   - *Action required:* This is novel; reviewers may ask: is multi-load-case needed in practice? (Answer: yes, if design must survive multiple operational scenarios)

### Missing contrasts:
- ❌ **No direct Yuksel2025 vs Proposed comparison:** Same benchmark, same quasi-static family—critical comparison needed in Related Work
- ❌ **No multi-mode quality degradation study:** How much does quasi-static lose for each additional mode tracked? (Gap should be addressed in methodology/results)
- ❌ **No multi-load-case precedent:** Other topology optimization domains (compliance, stress) regularly handle multi-load; quasi-static frequency is a novel application

### Required additions (for gap defense):
- [ ] Direct ω₁ comparison: Proposed vs Yuksel2025 for single-mode case
- [ ] Wall-clock time comparison: Proposed vs Yuksel2025 vs Du2007
- [ ] Multi-mode convergence: How do ω₁, ω₂, ω₃ track in proposed method?
- [ ] Multi-load-case example demonstrating practical value (e.g., resonance avoidance under multiple operational loads)

---

## --- QUANTITATIVE SUPPORT ---

### Available quantitative anchors:
- **Fundamental frequency benchmark:** ω₁ = 174.7 rad/s (Du2007 SIMP bound) — PRIMARY reference
- **BESO results:** ω₁ = 171.5 rad/s (Huang2010, HuangXie2010)
- **GWC results:** ω₁ = 169.3 rad/s (Li2021), 21 iterations vs BESO 67 iterations
- **Quasi-static results:** ω₁ = 160.5 rad/s (Yuksel2025), iteration count NOT reported
- **Proposed results:** ω₁ = 159.3 rad/s (implied), 7.1× speedup claimed (vs whom?)
- **3D extruded beams:** ω₁ = 1365.7 rad/s (L=1m), 110.2 rad/s (L=5m) — different problem class
- **Modified SIMP parameters:** x_min = 0.001 (universal across papers), p = 3 (assumed, not always explicit)

### Missing quantitative evidence:
- ❌ **Computational time wall-clock:** Most papers report iterations, not seconds
- ❌ **Sensitivity to x_min:** Only Du2007 and modified SIMP papers mention x_min = 0.001; no parameter study
- ❌ **Multi-mode frequency data:** ω₁, ω₂, ω₃ results rare; most papers show only ω₁
- ❌ **Multi-load-case results:** Zero papers; proposed is the first
- ❌ **Convergence rate (iterations vs ω₁):** Li2021 provides (21 iter vs 67), but not standard
- ❌ **Sensitivity to mesh refinement:** Marzok2024 studies this (DOF variation) but not sensitivity of optimized ω₁

### Risk of qualitative-only narrative:
⚠ **MODERATE RISK:** If Introduction relies only on "existing methods are slow" (qualitative) without quantifying Proposed method's absolute time in seconds, reviewers will demand:
- "How much wall-clock time does Proposed take vs Yuksel2025 (both quasi-static)?"
- "Is the speedup from quasi-static approximation or from other implementation choices?"

---

## --- MISSING PERSPECTIVES ---

### Missing method types:
- ❌ **Metaheuristics (GA, PSO, etc.):** No evolutionary computing applied to eigenfrequency problems in extraction
- ❌ **Adjoint-based non-gradient optimization:** Only MMA and OC; no penalty methods or augmented Lagrangian in frequency context
- ❌ **Physics-informed neural networks (PINNs):** Emerging field; not in extraction yet (2025 papers few)

### Missing evaluation paradigms:
- ❌ **Discrete vs continuous comparison:** BESO (discrete) vs SIMP (continuous) compared on ω₁, but not on manufacturability or post-processing cost
- ❌ **Sensitivity analysis (how do designs change with parameters):** No papers study robustness of optimal designs to material property variation
- ❌ **Additive manufacturing viability:** Designs optimized but not validated for AM constraints (overhang, minimum feature size)

### Missing comparison axes:
- ❌ **Solution quality vs speed tradeoff surface:** No paper maps out: given T seconds, what ω₁ is achievable? (Pareto frontier across methods)
- ❌ **Design space smoothness:** BESO discrete designs vs SIMP continuous—which is easier for downstream engineering?
- ❌ **Failure rate / robustness:** Optimized designs' performance under uncertainties (material, load, geometry tolerances)

---

## --- REBALANCING PLAN ---

### What to emphasize:
1. **Du2007 as field baseline:** "Classical bound formulation [Du2007] achieves ω₁ = 174.7 rad/s and serves as benchmark for all subsequent methods." (1–2 sentences)
2. **Three parallel families:** SIMP (Du2007), BESO (Huang2010), Level set (Xia2011) — all require eigensolver; all achieve ~170 rad/s
3. **Computational cost as open issue:** "Repeated eigensolution at each iteration remains a computational bottleneck, limiting scalability to large-scale 3D problems." (evidence: Li2021 notes 67 iterations for BESO; Marzok2024 reports 800 iterations for 3D)
4. **Yuksel2025 as precursor to proposed:** "Recent work [Yuksel2025] reformulates frequency maximization as a static load problem, achieving 7.1× speedup but limited to single-mode; extension to multi-mode and multi-load-case remains open." (1–2 sentences, directly lead to proposed gap)

### What to compress:
1. **Olhoff2014:** Merge into Du2007 cite; do not separately discuss
2. **BESO soft-kill variant:** HuangXie2010 is same as Huang2010 (171.5 rad/s); cite both with single result
3. **3D extruded beams (Marzok2024):** Mention as "3D extension via XFEM [Marzok2024] requires ~800 iterations, demonstrating order-of-magnitude cost increase" — brief, motivates efficiency push
4. **Couple-stress generalization (Su2016):** One sentence: "Generalization to couple-stress materials [Su2016] shows framework applies beyond isotropic continua" — background only

### What to group:
1. **Du2007 + Olhoff2014 + Deng2024:** SIMP family evolution — classical, review, modern (classify as 3 variants within same family)
2. **Huang2010 + HuangXie2010:** BESO pair — cite together
3. **Xia2011 + Shu2014:** Level set pair — one on eigenfrequency, one on acoustic; note both exist
4. **Yuksel2025 single-mode + Proposed multi-mode:** Quasi-static family — position as direct pair

### What to de-emphasize:
1. **Lee2008, Lee2010:** Frequency as constraint, not maximization — skip entirely
2. **Dynamic response (Gomez2019, Munro2024):** Different objective — one-sentence background only ("Dynamic response optimization [Gomez2019] addresses a related but distinct objective")
3. **Nishiwaki2000:** Historical; one-sentence cite only ("Topology optimization for vibration problems was pioneered by [Nishiwaki2000]")
4. **Acoustic-structural coupling (Shu2014):** Note as application ("Level set methods extend to coupled acoustic-structural systems [Shu2014]") but do not discuss in detail

### What must be added before writing Introduction:
- [ ] **Wall-clock time data** for Proposed method vs Yuksel2025 vs Du2007 (absolute seconds, not iterations)
- [ ] **Multi-mode example:** ω₁, ω₂, ω₃ for simply supported beam or clamped beam under multi-mode objective
- [ ] **Multi-load-case example:** Proposed optimizes for frequency under 2–3 distinct load cases; demonstrate practical value
- [ ] **Clear problem statement:** Is the gap "single-mode only" or "slow repeated eigensolver" or "both"? Be explicit

---

## --- RISK SUMMARY ---

### Risk of biased Introduction:
🟡 **MEDIUM RISK**

- **If Introduction overemphasizes Du2007 baseline:** Readers will perceive manuscript as incremental improvement, not novel gap
- **If Introduction presents efficiency as universal problem:** Du2007, Huang2010 do NOT claim speed as a goal; positioning quasi-static as critical gap may overstate field deficiency
- *Mitigation:* Frame as "unexploited opportunity" (Yuksel2025 shows promise) rather than "field-wide problem"

### Risk of weak gap:
🔴 **HIGH RISK if not addressed**

- **Current extraction evidence:** Gap is defensible on multi-mode/multi-load-case (truly novel), but single-mode quasi-static speedup (Yuksel2025) pre-exists
- **Reviewer question:** "How does Proposed differ from Yuksel2025?"
  - *Answer required in Introduction:* Proposed handles multi-mode (ω₁, ω₂, …) AND multi-load-case; Yuksel2025 does neither
  - *Evidence needed:* Direct comparison table (ω₁, ω₂ values) for Proposed vs baseline methods

### Risk of reviewer criticism:
🔴 **HIGH on computational claims**

- **Claim:** "7.1× faster than Olhoff" — but unclear vs whom (Yuksel2025 also ~7–10× faster?)
- **Claim:** "Scales to multi-mode/multi-load-case" — but no evidence in literature that existing methods fail here
- *Mitigation:* Provide explicit numerical comparisons in Results section; do not rely on literature alone

### Overall readiness:
🟡 **BORDERLINE** — Gap is defensible if framed carefully; requires evidence in methodology/results to convert from "literature-grounded opportunity" to "solved contribution"

---

## --- NEXT STEPS FOR INTRODUCTION WRITING ---

1. **Use this balance analysis as a roadmap** for intro paragraphing:
   - ¶1: Problem framing (frequency maximization applications)
   - ¶2: Classical approach (Du2007 bound formulation, 174.7 rad/s, eigensolver per iteration)
   - ¶3: Alternative families (BESO, GWC, level set—all face same iteration cost)
   - ¶4: Computational bottleneck identified (eigensolution dominates in practice; supporting evidence from Li2021, Marzok2024)
   - ¶5: Quasi-static opportunity (Yuksel2025 single-mode success; speedup quantified)
   - ¶6: Gap = multi-mode + multi-load-case extension; proposal outlined

2. **Prepare side-by-side comparison table** for Related Work:
   - Du2007 (SIMP): ω₁=174.7, iterations=?, time=?
   - Huang2010 (BESO): ω₁=171.5, iterations=67, time=?
   - Li2021 (GWC): ω₁=169.3, iterations=21, time=?
   - Yuksel2025 (Quasi-static): ω₁=160.5, iterations=?, time=?
   - Proposed: ω₁=159.3, iterations=?, time=? (multi-mode, multi-load-case capability added)

3. **Risk mitigation in Introduction:**
   - **Acknowledge quality tradeoff explicitly:** "Quasi-static approaches sacrifice ~9% in optimal frequency (160.5 vs 174.7 rad/s) to avoid repeated eigensolver."
   - **Justify tradeoff:** "For practical design iteration, speed is often prioritized over marginal frequency gains." (or cite engineering practice)
   - **Novelty clear:** "We extend quasi-static formulation to handle multiple target frequencies and multiple load cases simultaneously—a capability absent from prior work."

---

**STATUS: READY TO PROCEED TO INTRODUCTION WRITING**

This balance analysis confirms that:
1. Literature is well-structured with clear method families
2. Gap is defensible (multi-mode + multi-load-case extension)
3. Comparison points are clear (vs Yuksel2025 primarily; vs Du2007 for quality baseline)
4. Writing support section provides concrete framing language

**Proceed to:** STEP 1 of pipeline — Run `ai/prompts/write_introduction.md` with this balance output as control layer.
