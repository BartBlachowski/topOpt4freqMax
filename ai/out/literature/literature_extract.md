# Literature Extraction
Generated: 2026-04-15
Protocol: ai/prompts/literature_extract.md
Manuscript: Frequency maximization of planar structures using quasi-static approximation in topology optimization

---

## PAPER
- ID: 1
- Citation key: Du2007
- Title: Topological design of freely vibrating continuum structures for maximum values of simple and multiple eigenfrequencies and frequency gaps
- Year: 2007
- Primary domain: Structural topology optimization — eigenfrequency maximization
- Relevance to current manuscript: HIGH

## PROBLEM FOCUS
- Main problem addressed: Maximizing fundamental or multiple eigenfrequencies, and gaps between consecutive eigenfrequencies
- Structural/system setting: 2D continuum, plane stress, fixed mesh
- Target object/system: Simply supported beam, L-shaped structure, cantilever
- Why this paper is relevant here: Establishes the bound formulation used as baseline; provides benchmark ω₁=174.7 rad/s for the standard comparison case

## METHOD CLASSIFICATION
- Method family: SIMP density-based topology optimization
- Model fidelity: 2D FE, plane stress
- Optimization / analysis type: Gradient-based, MMA optimizer
- Objective or design target: Bound formulation — maximize α s.t. α*λ_i ≥ β for tracked modes; effectively maximizes min(ω_i)
- Main constraints considered: Volume fraction
- Load-case treatment: Eigenvalue problem only (no external loads)
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Modified SIMP interpolation — E_e = x_e^p*(E_0 - E_min) + E_min; mass ρ_e = ρ_0*x_e; x_min=0.001 to avoid ALRM (artificial localized resonance modes)

## VALIDATION / EVIDENCE
- Validation type: Numerical benchmark against prior literature
- Test object / case study: Simply supported beam 8m×1m, 400×50 FE mesh, 50% volume, E=2.1e11, ρ=7800
- Metrics or criteria reported: First eigenfrequency ω₁ [rad/s]
- Main quantitative result(s): ω₁=174.7 rad/s (simple mode maximization), frequency gap example also shown
- Evidence strength: STRONG (well-cited benchmark, consistent with multiple later papers)

## STRENGTHS
- Strength 1: Handles both simple and multiple (repeated) eigenvalues rigorously via non-smooth bound formulation
- Strength 2: Clean sensitivity formula for simple eigenvalue: dλ_i/dx_e = x_e^(p-1) * u_i^T * (p*K_0e - λ_i * ρ_0 * M_0e) * u_i
- Strength 3: Frequency gap maximization as extension of same framework

## LIMITATIONS
- Limitation 1: Repeated/multiple eigenvalues require perturbed bound formulation — non-trivial to implement
- Limitation 2: Sensitivity requires full eigensolution (eigenvalue + eigenvector) at each iteration
- Limitation 3: No design-dependent inertial load consideration

## COMPARISON VALUE
- Most useful comparison dimension: Benchmark ω₁ value for simply supported beam (standard comparison point across field)
- What this paper does better than simpler alternatives: Rigorous handling of multiple eigenvalues; frequency gap formulation
- What this paper does not address: Computational efficiency; quasi-static approximation; design-dependent loads
- Most relevant takeaway for Introduction: Establishes standard bound formulation and benchmark result (174.7 rad/s); any new method is compared against this
- Most relevant takeaway for Related Work: Core formulation reference; all SIMP-based eigenfrequency methods build on or compare against this

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — "the bound formulation is proposed for the simultaneous maximization of an arbitrary number of eigenfrequencies"
- Key supporting statement 2: EXPLICIT — modified SIMP with x_min avoids artificial localized modes (ALRM) caused by low-density elements with low stiffness-to-mass ratio
- Key supporting statement 3: EXPLICIT — benchmark ω₁=174.7 rad/s for 8m×1m simply supported beam at 50% volume

## CONFIDENCE
- Overall extraction confidence: HIGH
- Uncertain fields: Exact SIMP penalization power p (likely p=3, standard)

---

## PAPER
- ID: 2
- Citation key: Olhoff2014
- Title: Introductory Notes on Topological Design Optimization of Vibrating Continuum Structures
- Year: 2014
- Primary domain: Structural topology optimization — eigenfrequency maximization (book chapter review)
- Relevance to current manuscript: HIGH

## PROBLEM FOCUS
- Main problem addressed: Review and consolidation of bound formulation for eigenfrequency maximization and gap maximization; overview of field
- Structural/system setting: 2D continuum; general vibrating structures
- Target object/system: Various benchmarks; simply supported and clamped beams
- Why this paper is relevant here: Authoritative review chapter by original bound formulation author; provides theoretical grounding for comparison

## METHOD CLASSIFICATION
- Method family: SIMP, bound formulation (analytical framework)
- Model fidelity: 2D FE, continuum
- Optimization / analysis type: MMA-based gradient optimization
- Objective or design target: Maximize minimum eigenfrequency or frequency gap
- Main constraints considered: Volume fraction
- Load-case treatment: Eigenvalue problem
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Modified SIMP with x_min; bound constraint on tracked modes; non-smooth sensitivity via Clarke subgradient for repeated eigenvalues

## VALIDATION / EVIDENCE
- Validation type: Benchmark examples with numerical results
- Test object / case study: Simply supported beam, clamped beam, various 2D structures
- Metrics or criteria reported: ω₁ [rad/s], convergence histories
- Main quantitative result(s): Consistent with Du2007 benchmark results
- Evidence strength: STRONG

## STRENGTHS
- Strength 1: Theoretical completeness — covers simple, repeated, and gap formulations in one framework
- Strength 2: Clearly explains ALRM mechanism and x_min remedy
- Strength 3: Historical context and method comparison within bound formulation family

## LIMITATIONS
- Limitation 1: Review chapter — primarily consolidates existing work rather than introducing new computational methods
- Limitation 2: Does not address computational efficiency improvements
- Limitation 3: No quasi-static or load-based reformulation

## COMPARISON VALUE
- Most useful comparison dimension: Theoretical basis for all SIMP-based eigenfrequency methods
- What this paper does better than simpler alternatives: Complete theoretical treatment including non-smooth analysis
- What this paper does not address: Alternative formulations (BESO, level set); computational cost reduction
- Most relevant takeaway for Introduction: Best citable source for "standard" bound formulation and ALRM explanation
- Most relevant takeaway for Related Work: Background reference for SIMP-based methods section

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — bound constraint: α*λ_i ≥ β applied to all tracked modes
- Key supporting statement 2: EXPLICIT — modified SIMP: E_e = x_e^p*(E_0 - E_min) + E_min keeps stiffness-to-mass ratio bounded at x_min
- Key supporting statement 3: IMPLIED — eigensolution required at each iteration is a key computational cost

## CONFIDENCE
- Overall extraction confidence: HIGH
- Uncertain fields: Specific numerical results in chapter (book access limited to 10 pages; results consistent with Du2007)

---

## PAPER
- ID: 3
- Citation key: Yuksel2025
- Title: Efficient topology optimization of structures for frequency maximization using quasi-static approach
- Year: 2025
- Primary domain: Structural topology optimization — quasi-static frequency maximization
- Relevance to current manuscript: HIGH

## PROBLEM FOCUS
- Main problem addressed: Reformulate frequency maximization as a sequence of static compliance problems to avoid repeated eigensolution
- Structural/system setting: 2D continuum, plane stress, fixed mesh
- Target object/system: Simply supported beam (standard benchmark), clamped beam
- Why this paper is relevant here: Closest related work — uses design-dependent inertial load to convert eigenvalue problem to static; directly comparable approach to proposed method

## METHOD CLASSIFICATION
- Method family: SIMP density-based, quasi-static reformulation
- Model fidelity: 2D FE, plane stress
- Optimization / analysis type: Gradient-based; Rayleigh's principle for sensitivity
- Objective or design target: Maximize fundamental eigenfrequency via minimizing compliance under inertial load f(x) = ω²*M(x)*Φ
- Main constraints considered: Volume fraction
- Load-case treatment: Design-dependent inertial load derived from current eigenvector
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Two-step procedure — solve eigenvalue once, then update via static problem; Rayleigh quotient monotonicity argument

## VALIDATION / EVIDENCE
- Validation type: Benchmark comparison against standard methods (Du2007, BESO)
- Test object / case study: Simply supported beam 8m×1m, 50% volume (same benchmark)
- Metrics or criteria reported: ω₁ [rad/s], iteration count, computation time
- Main quantitative result(s): ω₁=160.5 rad/s at 50% volume fraction (proposed is 159.3)
- Evidence strength: MODERATE (results slightly lower than Du2007 bound formulation, trade-off with speed)

## STRENGTHS
- Strength 1: Avoids repeated eigensolution — only one eigensolve per outer iteration
- Strength 2: Design-dependent load captures sensitivity to mass redistribution
- Strength 3: Straightforward extension to multi-mode problems

## LIMITATIONS
- Limitation 1: Achieves slightly lower ω₁ than direct bound formulation (160.5 vs 174.7 rad/s) — convergence to local optimum possible
- Limitation 2: Inertial load depends on current eigenvector — load update coupling requires careful implementation
- Limitation 3: Does not handle repeated eigenvalues explicitly

## COMPARISON VALUE
- Most useful comparison dimension: Direct algorithmic comparison — both use static reformulation of eigenvalue problem; ω₁ values nearly identical (160.5 vs 159.3)
- What this paper does better than simpler alternatives: Avoids eigensolver per iteration; faster than standard bound formulation
- What this paper does not address: NOT REPORTED whether multi-mode or multi-load-case problems are addressed; speed advantage vs full eigensolver not fully quantified
- Most relevant takeaway for Introduction: Closest precursor to proposed method; must be clearly distinguished
- Most relevant takeaway for Related Work: Side-by-side comparison target; same benchmark, same approach family

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — inertial load f(x) = ω²*M(x)*Φ derived from Rayleigh's principle
- Key supporting statement 2: EXPLICIT — ω₁=160.5 rad/s for standard simply supported beam benchmark
- Key supporting statement 3: IMPLIED — computational advantage over standard eigensolver-based methods

## CONFIDENCE
- Overall extraction confidence: HIGH
- Uncertain fields: Exact computational speedup reported (prior session noted 7.1× was the proposed method's advantage over Olhoff, not necessarily Yuksel); iteration count for Yuksel

---

## PAPER
- ID: 4
- Citation key: Huang2010
- Title: Evolutionary topology optimization of continuum structures for natural frequencies
- Year: 2010
- Primary domain: Structural topology optimization — BESO, eigenfrequency
- Relevance to current manuscript: HIGH

## PROBLEM FOCUS
- Main problem addressed: BESO-based topology optimization for maximizing natural frequencies of continuum structures
- Structural/system setting: 2D continuum, fixed mesh
- Target object/system: Simply supported beam 8m×1m (standard benchmark), cantilever, clamped beam
- Why this paper is relevant here: Provides benchmark comparison result ω₁=171.5 rad/s; demonstrates BESO as alternative to SIMP for this problem

## METHOD CLASSIFICATION
- Method family: BESO (Bidirectional Evolutionary Structural Optimization) with modified SIMP
- Model fidelity: 2D FE
- Optimization / analysis type: Evolutionary (element-wise add/remove) with sensitivity-based ranking
- Objective or design target: Maximize fundamental or multiple natural frequencies
- Main constraints considered: Volume fraction target
- Load-case treatment: Eigenvalue problem
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Modified SIMP with x_min for ALRM prevention; soft-kill removes elements to x_min not zero; sensitivity: α_i = (1/2ω_i)*u_i^T*(K_i^1/p - ω_i²*M_i^1)*u_i for solid elements

## VALIDATION / EVIDENCE
- Validation type: Benchmark comparison
- Test object / case study: Simply supported beam 8m×1m, 400×50 mesh, 50% volume
- Metrics or criteria reported: ω₁ [rad/s], iteration count
- Main quantitative result(s): ω₁=171.5 rad/s
- Evidence strength: STRONG

## STRENGTHS
- Strength 1: BESO is 0/1 design — crisp boundaries without intermediate density
- Strength 2: Modified SIMP x_min avoids ALRM while preserving element stiffness-to-mass ratio
- Strength 3: Handles multiple frequencies via weighted sensitivity

## LIMITATIONS
- Limitation 1: Evolutionary approach with discrete add/remove — convergence to local optima
- Limitation 2: Sensitivity number requires eigensolve at each iteration
- Limitation 3: Fixed evolutionary ratio — limited adaptivity

## COMPARISON VALUE
- Most useful comparison dimension: Benchmark ω₁=171.5 rad/s; shows BESO yields slightly lower result than SIMP bound formulation (174.7)
- What this paper does better than simpler alternatives: Binary designs; avoids intermediate density artifacts
- What this paper does not address: Computational efficiency relative to full eigensolve
- Most relevant takeaway for Introduction: Standard BESO result for benchmark comparison table
- Most relevant takeaway for Related Work: BESO as alternative to SIMP — same problem, discrete formulation

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — ω₁=171.5 rad/s for standard benchmark
- Key supporting statement 2: EXPLICIT — sensitivity number formula for BESO frequency optimization
- Key supporting statement 3: EXPLICIT — modified SIMP with x_min=0.001 prevents ALRM

## CONFIDENCE
- Overall extraction confidence: HIGH
- Uncertain fields: Exact mesh size and SIMP power p

---

## PAPER
- ID: 5
- Citation key: HuangXie2010
- Title: Natural frequency optimization of evolutionary continuum structures
- Year: 2010
- Primary domain: Structural topology optimization — BESO, natural frequencies
- Relevance to current manuscript: HIGH

## PROBLEM FOCUS
- Main problem addressed: Soft-kill BESO for natural frequency maximization; multiple frequency handling
- Structural/system setting: 2D continuum
- Target object/system: Simply supported beam 8m×1m (same benchmark)
- Why this paper is relevant here: Companion paper to Huang2010; provides same ω₁=171.5 rad/s benchmark result; demonstrates soft-kill BESO

## METHOD CLASSIFICATION
- Method family: BESO (soft-kill variant)
- Model fidelity: 2D FE
- Optimization / analysis type: Evolutionary with sensitivity-based ranking
- Objective or design target: Maximize natural frequency (fundamental or multiple)
- Main constraints considered: Volume fraction
- Load-case treatment: Eigenvalue problem
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Soft-kill — elements set to x_min (not removed) to avoid numerical issues; modified SIMP for consistent mass-to-stiffness ratio

## VALIDATION / EVIDENCE
- Validation type: Benchmark
- Test object / case study: Simply supported beam 8m×1m, 50% volume
- Metrics or criteria reported: ω₁ [rad/s]
- Main quantitative result(s): ω₁=171.5 rad/s
- Evidence strength: STRONG

## STRENGTHS
- Strength 1: Soft-kill avoids mesh singularity while maintaining binary character
- Strength 2: Multiple frequency handling via weighted sum approach
- Strength 3: ALRM addressed via consistent SIMP model

## LIMITATIONS
- Limitation 1: Same ALRM issues as hard-kill BESO if x_min not carefully chosen
- Limitation 2: Sensitivity still requires eigensolution
- Limitation 3: Weighted sum for multiple frequencies introduces weight-tuning sensitivity

## COMPARISON VALUE
- Most useful comparison dimension: ω₁=171.5 rad/s — confirms Huang2010 result independently
- What this paper does better than simpler alternatives: Soft-kill avoids numerical conditioning issues
- What this paper does not address: Computational cost; quasi-static reformulation
- Most relevant takeaway for Introduction: Confirms BESO benchmark at 171.5 rad/s
- Most relevant takeaway for Related Work: Grouped with Huang2010 as BESO family

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — ω₁=171.5 rad/s benchmark result
- Key supporting statement 2: EXPLICIT — soft-kill: x_min elements retained with reduced stiffness
- Key supporting statement 3: EXPLICIT — sensitivity formula consistent with Huang2010

## CONFIDENCE
- Overall extraction confidence: HIGH
- Uncertain fields: Exact iteration count

---

## PAPER
- ID: 6
- Citation key: Li2021
- Title: Topology optimization for maximizing the natural frequencies of structures with various boundary constraints
- Year: 2021
- Primary domain: Structural topology optimization — eigenfrequency, guide-weight criterion
- Relevance to current manuscript: HIGH

## PROBLEM FOCUS
- Main problem addressed: Eigenfrequency maximization using Guide-Weight Criterion (GWC) as update strategy; Heaviside projection for manufacturability
- Structural/system setting: 2D continuum, multiple boundary conditions
- Target object/system: Simply supported beam 8m×1m (standard benchmark), clamped beam, L-bracket
- Why this paper is relevant here: Provides benchmark ω₁=169.3 rad/s; converges in 21 iterations vs BESO 67; demonstrates GWC efficiency for frequency problems

## METHOD CLASSIFICATION
- Method family: SIMP with Guide-Weight Criterion (GWC) update
- Model fidelity: 2D FE
- Optimization / analysis type: GWC (criterion-based, non-gradient), iterative mass control
- Objective or design target: Maximize fundamental eigenfrequency
- Main constraints considered: Volume fraction (iterative mass control achieves target gradually)
- Load-case treatment: Eigenvalue problem
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Modified SIMP with x_min for ALRM; Heaviside projection for crisp boundaries; iterative mass target gradually reduces volume

## VALIDATION / EVIDENCE
- Validation type: Benchmark comparison vs BESO and other methods
- Test object / case study: Simply supported beam 8m×1m, 50% volume; 200×50 and 400×100 meshes tested
- Metrics or criteria reported: ω₁ [rad/s], iteration count to convergence
- Main quantitative result(s): ω₁=169.3 rad/s in 21 iterations; BESO achieves 170.0 rad/s in 67 iterations
- Evidence strength: STRONG

## STRENGTHS
- Strength 1: GWC converges faster than BESO (21 vs 67 iterations for same benchmark)
- Strength 2: Heaviside projection provides crisp, manufacturable designs
- Strength 3: Iterative mass control strategy avoids premature convergence

## LIMITATIONS
- Limitation 1: GWC is not strictly gradient-based — lacks convergence guarantees
- Limitation 2: Achieves slightly lower ω₁ than Du2007 bound formulation (169.3 vs 174.7)
- Limitation 3: Heaviside projection adds implementation complexity

## COMPARISON VALUE
- Most useful comparison dimension: Convergence speed and ω₁ value — 21 iterations for 169.3 rad/s
- What this paper does better than simpler alternatives: Faster convergence than BESO; crisp boundaries via Heaviside
- What this paper does not address: Multiple frequencies; computational cost per iteration
- Most relevant takeaway for Introduction: Demonstrates that faster convergence (fewer iterations) is a design goal in this field — context for proposed 7.1× speedup claim
- Most relevant takeaway for Related Work: GWC as distinct update strategy; comparison point for iteration count

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — ω₁=169.3 rad/s in 21 iterations for simply supported beam
- Key supporting statement 2: EXPLICIT — BESO comparison: 170.0 rad/s, 67 iterations
- Key supporting statement 3: EXPLICIT — Heaviside projection + iterative mass control

## CONFIDENCE
- Overall extraction confidence: HIGH
- Uncertain fields: Whether timing comparison (wall clock) is reported alongside iteration count

---

## PAPER
- ID: 7
- Citation key: Deng2024
- Title: Discrete topology optimization for natural frequency maximization using the solid isotropic material with penalization method
- Year: 2024
- Primary domain: Structural topology optimization — discrete SAIP, eigenfrequency
- Relevance to current manuscript: HIGH

## PROBLEM FOCUS
- Main problem addressed: Discrete (0/1) topology optimization for natural frequency maximization; ALRM avoidance via Betti reciprocal theorem approach
- Structural/system setting: 2D continuum
- Target object/system: Simply supported beam (standard benchmark), cantilever
- Why this paper is relevant here: Recent (2024) paper addressing same problem; ALRM handling via Betti number is a distinct approach; uses KS function for multiple frequencies

## METHOD CLASSIFICATION
- Method family: SIMP with discrete anisotropic interpolation (SAIP)
- Model fidelity: 2D FE
- Optimization / analysis type: Gradient-based, MMA
- Objective or design target: Maximize fundamental or multiple natural frequencies
- Main constraints considered: Volume fraction
- Load-case treatment: Eigenvalue problem
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: SAIP (Solid Anisotropic Material Interpolation with Penalization); Betti reciprocal theorem to characterize and avoid ALRM; KS aggregation for multiple frequencies

## VALIDATION / EVIDENCE
- Validation type: Benchmark comparison
- Test object / case study: Simply supported beam 8m×1m, standard benchmark
- Metrics or criteria reported: ω₁ [rad/s], comparison to Du2007 and BESO
- Main quantitative result(s): Results comparable to Du2007 bound formulation
- Evidence strength: MODERATE (recent paper, not yet extensively cross-validated)

## STRENGTHS
- Strength 1: ALRM handling via Betti reciprocal theorem — mathematically grounded alternative to x_min heuristic
- Strength 2: KS function enables smooth multiple-frequency objective
- Strength 3: Discrete SAIP produces clean 0/1 designs

## LIMITATIONS
- Limitation 1: More complex implementation than standard modified SIMP
- Limitation 2: KS aggregation introduces approximation error in multiple-frequency case
- Limitation 3: Betti-based ALRM detection may add computational overhead

## COMPARISON VALUE
- Most useful comparison dimension: ALRM handling strategy — Betti theorem vs x_min heuristic
- What this paper does better than simpler alternatives: Principled ALRM avoidance without ad hoc x_min
- What this paper does not address: Quasi-static reformulation; computational speed
- Most relevant takeaway for Introduction: Modern (2024) SIMP-variant; shows field is still active and evolving ALRM handling
- Most relevant takeaway for Related Work: Grouped with SIMP family; distinct ALRM approach worth noting

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — SAIP formulation avoids ALRM via Betti reciprocal theorem
- Key supporting statement 2: EXPLICIT — KS function for multi-frequency aggregation
- Key supporting statement 3: EXPLICIT — benchmark results comparable to Du2007

## CONFIDENCE
- Overall extraction confidence: MEDIUM
- Uncertain fields: Exact ω₁ value for benchmark; specific KS aggregation parameter

---

## PAPER
- ID: 8
- Citation key: Huang2025
- Title: Matlab codes for 2D and 3D MMC-based natural frequency topology optimization
- Year: 2025
- Primary domain: Structural topology optimization — MMC framework, natural frequencies
- Relevance to current manuscript: MEDIUM

## PROBLEM FOCUS
- Main problem addressed: Moving Morphable Component (MMC) framework for natural frequency topology optimization; spurious eigenmode elimination; mode switching
- Structural/system setting: 2D and 3D continuum, MMC parametrization
- Target object/system: Cantilever beam, rectangular plate
- Why this paper is relevant here: Recent alternative framework (MMC) addressing same frequency maximization problem; handles repeated frequencies and mode switching via MAC

## METHOD CLASSIFICATION
- Method family: MMC (Moving Morphable Components) — explicit geometry parameterization
- Model fidelity: 2D and 3D FE
- Optimization / analysis type: Gradient-based, MMA; adjoint sensitivity (Lee 2007 formulation)
- Objective or design target: Three formulations: (1) max non-repeated frequency, (2) max weighted sum of repeated frequencies, (3) max frequency with MAC-based mode tracking
- Main constraints considered: Volume fraction, MAC constraint MAC(φ_i) ≥ 1-ε
- Load-case treatment: Eigenvalue problem
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Topology Description Function (TDF) via smoothed Heaviside; ersatz material; adaptive ε for Heaviside sharpening; adaptive scl for MMA scaling; spurious mode elimination via load transfer path connectivity analysis

## VALIDATION / EVIDENCE
- Validation type: Convergence study with ε parameter; numerical examples
- Test object / case study: 2D rectangular beam, 3D plate
- Metrics or criteria reported: Final frequency [Hz], convergence stability
- Main quantitative result(s): Fixed ε=0.5 → 331.5 Hz; ε=0.1 → 310.4 Hz; adaptive ε → 323.973 Hz (trapezoidal beam example)
- Evidence strength: MODERATE

## STRENGTHS
- Strength 1: MMC avoids checkerboard and provides crisp boundaries inherently
- Strength 2: Adaptive scl stabilizes MMA for frequency problems (orders-of-magnitude objective variation)
- Strength 3: Load transfer path analysis eliminates spurious eigenmodes systematically

## LIMITATIONS
- Limitation 1: MMC framework adds complexity — component-based parameterization limits design freedom
- Limitation 2: Mode switching is a recognized challenge; MAC constraint adds constraints per tracked mode
- Limitation 3: 3D extension increases computational cost significantly

## COMPARISON VALUE
- Most useful comparison dimension: MAC-based mode tracking as distinct solution to repeated eigenvalue problem
- What this paper does better than simpler alternatives: Spurious mode elimination without x_min hack; explicit geometry control
- What this paper does not address: Computational speed relative to standard density methods; quasi-static reformulation
- Most relevant takeaway for Introduction: Recent (2025) paper confirms that spurious eigenmode and mode switching remain active research issues
- Most relevant takeaway for Related Work: MMC as geometry-based alternative to density methods

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — three optimization formulations with different frequency objectives
- Key supporting statement 2: EXPLICIT — adaptive ε and adaptive scl criteria for numerical stability
- Key supporting statement 3: EXPLICIT — spurious eigenmode elimination via load transfer path analysis

## CONFIDENCE
- Overall extraction confidence: HIGH
- Uncertain fields: Computational speedup vs standard SIMP (not directly compared)

---

## PAPER
- ID: 9
- Citation key: Lee2008
- Title: Topology optimization of structures with frequency constraints using SIMP
- Year: 2008
- Primary domain: Structural topology optimization — SIMP, dynamic loading
- Relevance to current manuscript: MEDIUM

## PROBLEM FOCUS
- Main problem addressed: SIMP-based topology optimization with frequency constraints; comparison of static vs dynamic loading objectives
- Structural/system setting: 2D continuum
- Target object/system: Clamped beam 4m×2m
- Why this paper is relevant here: Provides adjoint sensitivity formulation for frequency used in later works (Huang2025 cites Lee2007/2008); standard SIMP benchmark

## METHOD CLASSIFICATION
- Method family: SIMP density-based
- Model fidelity: 2D FE
- Optimization / analysis type: Gradient-based, MMA
- Objective or design target: Minimize compliance under static or dynamic loading; frequency constraints
- Main constraints considered: Volume fraction, frequency lower bound
- Load-case treatment: Static and harmonic dynamic loads
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Standard SIMP; adjoint method for sensitivity of eigenfrequency

## VALIDATION / EVIDENCE
- Validation type: Numerical examples with comparison of static vs dynamic objectives
- Test object / case study: Clamped beam 4m×2m, 100×50 FE mesh
- Metrics or criteria reported: ω₁ [Hz], compliance [J]
- Main quantitative result(s): ω₁≈17.79 Hz for clamped beam (50% volume)
- Evidence strength: MODERATE

## STRENGTHS
- Strength 1: Adjoint sensitivity formula for eigenfrequency widely adopted in later work
- Strength 2: Shows difference between static-load and frequency-constrained designs
- Strength 3: Frequency constraint formulation as alternative to direct maximization

## LIMITATIONS
- Limitation 1: Frequency constraint rather than maximization — limited direct comparison
- Limitation 2: Clamped beam benchmark not widely used (most papers use simply supported)
- Limitation 3: Standard SIMP without ALRM protection not described

## COMPARISON VALUE
- Most useful comparison dimension: Adjoint sensitivity formula for eigenfrequency — foundational for many later implementations
- What this paper does better than simpler alternatives: Clear adjoint derivation
- What this paper does not address: Multiple eigenfrequencies; repeated eigenvalue issue; quasi-static reformulation
- Most relevant takeaway for Introduction: Background reference for sensitivity analysis
- Most relevant takeaway for Related Work: Early SIMP dynamic paper; citation context for sensitivity methods

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — adjoint sensitivity of eigenfrequency wrt density variables
- Key supporting statement 2: EXPLICIT — clamped beam ω₁≈17.79 Hz
- Key supporting statement 3: IMPLIED — difference between static and dynamic optimal designs

## CONFIDENCE
- Overall extraction confidence: MEDIUM
- Uncertain fields: Whether x_min ALRM protection used

---

## PAPER
- ID: 10
- Citation key: Lee2024
- Title: Topology optimization of functionally graded structures for natural frequency maximization
- Year: 2024
- Primary domain: Multi-scale topology optimization — functionally graded structures (FGS)
- Relevance to current manuscript: MEDIUM

## PROBLEM FOCUS
- Main problem addressed: Natural frequency maximization of functionally graded multi-scale structures
- Structural/system setting: Multi-scale continuum (macro + micro)
- Target object/system: Various 2D structures; multi-scale plates
- Why this paper is relevant here: Same objective (frequency maximization) but different scale; shows that FGS adds complexity not present in single-scale continuum problem

## METHOD CLASSIFICATION
- Method family: SIMP at macro scale; RAMP or similar at micro scale; deep learning for de-homogenization
- Model fidelity: Multi-scale 2D FE
- Optimization / analysis type: Gradient-based
- Objective or design target: Maximize fundamental natural frequency across scales
- Main constraints considered: Volume fraction
- Load-case treatment: Eigenvalue problem
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Homogenization for scale coupling; deep learning to reconstruct fine-scale geometry

## VALIDATION / EVIDENCE
- Validation type: Numerical examples
- Test object / case study: 2D structures with multi-scale material
- Metrics or criteria reported: ω₁ relative to single-scale design
- Main quantitative result(s): NOT REPORTED (limited data from 10-page read)
- Evidence strength: MODERATE

## STRENGTHS
- Strength 1: Addresses multi-scale design space not covered by single-scale methods
- Strength 2: Deep learning de-homogenization enables manufacturable fine-scale structures
- Strength 3: RAMP interpolation suitable for graded materials

## LIMITATIONS
- Limitation 1: Multi-scale adds computational complexity not present in continuum topology optimization
- Limitation 2: Deep learning component reduces interpretability
- Limitation 3: Results not directly comparable to single-scale benchmarks

## COMPARISON VALUE
- Most useful comparison dimension: Shows frequency maximization is relevant at multiple scales; different method family from proposed work
- What this paper does better than simpler alternatives: Multi-scale design freedom
- What this paper does not address: Single-scale continuum frequency maximization with efficiency
- Most relevant takeaway for Introduction: Shows breadth of frequency optimization methods; proposed work focuses on single-scale continuum
- Most relevant takeaway for Related Work: Background on extended applications; LOW priority for detailed discussion

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — frequency maximization at macro+micro scale
- Key supporting statement 2: EXPLICIT — RAMP interpolation for graded material
- Key supporting statement 3: UNCERTAIN — specific frequency improvements reported but not extracted

## CONFIDENCE
- Overall extraction confidence: MEDIUM
- Uncertain fields: Quantitative benchmark results; specific method details

---

## PAPER
- ID: 11
- Citation key: Marzok2024
- Title: Topology optimization of extruded beams modeled with the XFEM for maximizing their natural frequencies
- Year: 2024
- Primary domain: Structural topology optimization — XFEM, extruded beams, natural frequencies
- Relevance to current manuscript: MEDIUM

## PROBLEM FOCUS
- Main problem addressed: Cross-section topology optimization of extruded beams for maximum fundamental natural frequency using XFEM with global enrichment in longitudinal direction
- Structural/system setting: 3D extruded beam, cross-section is design domain
- Target object/system: Extruded beams with rectangular cross-section (200×300 mm), lengths 1m and 5m
- Why this paper is relevant here: Same objective (frequency maximization) but 3D extruded problem; different structural setting; provides SIMP+MMA reference for 3D case

## METHOD CLASSIFICATION
- Method family: SIMP with XFEM, density-based
- Model fidelity: 3D FE via XFEM (cross-section refined + 2 longitudinal elements)
- Optimization / analysis type: Gradient-based, MMA, p-norm smooth min approximation
- Objective or design target: Maximize fundamental natural frequency (minimize negative min eigenvalue)
- Main constraints considered: Mass fraction bound, static compliance bound
- Load-case treatment: Eigenvalue problem + static compliance constraint
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Design variables only in 2D cross-section; XFEM enrichment in longitudinal direction enables 3D analysis with only 2 elements longitudinally; SIMP with density filtering and Heaviside projection

## VALIDATION / EVIDENCE
- Validation type: Numerical examples with convergence study
- Test object / case study: Extruded beams L=1m and L=5m; cross-section 200×300mm; 80×120 elements in cross-section
- Metrics or criteria reported: ω₁ [rad/s], iteration count (~800 iterations)
- Main quantitative result(s): L=1m, m=0.15: ω₁=1365.7 rad/s; L=1m, m=0.30: ω₁=2011.8 rad/s; L=5m, m=0.15: ω₁=110.2 rad/s; L=5m, m=0.30: ω₁=118.0 rad/s
- Evidence strength: STRONG (full 3D FE convergence study)

## STRENGTHS
- Strength 1: XFEM global enrichment reduces DOFs by ~85% while maintaining 3D accuracy
- Strength 2: Handles cross-section distortion modes (localized vibrations) that 2D plane stress cannot capture
- Strength 3: 15% improvement over hollow rectangular reference cross-section

## LIMITATIONS
- Limitation 1: Limited to extruded geometry — not applicable to general 3D structures
- Limitation 2: ~800 iterations required for convergence (high iteration count)
- Limitation 3: Mass fraction and compliance constraints simultaneously active — constrained problem more complex

## COMPARISON VALUE
- Most useful comparison dimension: 3D extruded beam problem — different structural class from 2D continuum
- What this paper does better than simpler alternatives: Full 3D with much fewer DOFs than standard 3D FEM
- What this paper does not address: General 2D or 3D continuum; computational speed vs standard eigensolve
- Most relevant takeaway for Introduction: Shows frequency maximization extends to 3D with different challenges (cross-section distortion, extruded geometry)
- Most relevant takeaway for Related Work: 3D extension case; note as different structural class from proposed work

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — ω₁ values for different beam lengths and mass fractions
- Key supporting statement 2: EXPLICIT — XFEM reduces longitudinal DOFs to 2 elements while maintaining accuracy
- Key supporting statement 3: EXPLICIT — 15% frequency improvement over non-optimized hollow rectangular design

## CONFIDENCE
- Overall extraction confidence: HIGH
- Uncertain fields: None

---

## PAPER
- ID: 12
- Citation key: Ni2014
- Title: Integrated size and topology optimization of skeletal structures with exact frequency constraints
- Year: 2014
- Primary domain: Structural optimization — frame/skeletal structures, frequency constraints
- Relevance to current manuscript: MEDIUM

## PROBLEM FOCUS
- Main problem addressed: Integrated size and topology optimization of frame structures subject to frequency constraints; avoiding singularity in skeletal topology optimization
- Structural/system setting: Skeletal (frame) structures — 2D frames
- Target object/system: 10-bar and 35-bar frame structures
- Why this paper is relevant here: Frequency constraint in a different structural class; PLMP for singularity avoidance; exact frequency analysis via dynamic stiffness matrix

## METHOD CLASSIFICATION
- Method family: Ground structure method with topology optimization via PLMP; integrated size+topology
- Model fidelity: Frame FE (Euler-Bernoulli beam elements); dynamic stiffness matrix (Kolousek formulation) for exact frequency
- Optimization / analysis type: Gradient-based, MMA
- Objective or design target: Minimize weight s.t. frequency constraints and compliance constraints
- Main constraints considered: Fundamental frequency lower bound, compliance upper bound
- Load-case treatment: Eigenvalue problem + static load cases
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: PLMP (Polynomial Material Interpolation with Penalization) for singularity avoidance; strong singularity phenomenon in skeletal topology (Ni notes this is a fundamental challenge)

## VALIDATION / EVIDENCE
- Validation type: Numerical examples
- Test object / case study: 10-bar frame: natural frequency 140 Hz; 35-bar frame: 80 Hz
- Metrics or criteria reported: Natural frequency [Hz], weight
- Main quantitative result(s): 10-bar: 140 Hz; 35-bar: 80 Hz (frequency constraints met)
- Evidence strength: MODERATE

## STRENGTHS
- Strength 1: Dynamic stiffness matrix gives exact eigenfrequencies for frames (no discretization error)
- Strength 2: PLMP addresses strong singularity — principled approach vs ad hoc x_min
- Strength 3: Integrated size+topology optimization in one framework

## LIMITATIONS
- Limitation 1: Skeletal structures only — not applicable to continuum topology optimization
- Limitation 2: Dynamic stiffness matrix is more complex than standard FEM for large systems
- Limitation 3: Strong singularity problem is specific to skeletal topology — not an issue in continuum

## COMPARISON VALUE
- Most useful comparison dimension: PLMP for singularity avoidance — different structural class but similar challenge to ALRM in continuum
- What this paper does better than simpler alternatives: Exact frequency for frames; principled singularity avoidance
- What this paper does not address: Continuum frequency maximization; computational efficiency
- Most relevant takeaway for Introduction: Shows frequency optimization challenges extend to skeletal structures; different method family
- Most relevant takeaway for Related Work: Frame/skeletal methods — distinct from continuum; brief mention if scope allows

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — PLMP for strong singularity avoidance in skeletal topology
- Key supporting statement 2: EXPLICIT — dynamic stiffness matrix (Kolousek) for exact frequency analysis
- Key supporting statement 3: EXPLICIT — 10-bar: 140 Hz, 35-bar: 80 Hz

## CONFIDENCE
- Overall extraction confidence: MEDIUM
- Uncertain fields: Volume fraction or mass constraint details

---

## PAPER
- ID: 13
- Citation key: Belblidia2002
- Title: A hybrid topology/shape optimization algorithm for thin-walled structures
- Year: 2002
- Primary domain: Structural optimization — shell/thin-walled structures, hybrid algorithm
- Relevance to current manuscript: MEDIUM

## PROBLEM FOCUS
- Main problem addressed: Frequency maximization of thin-walled (shell) structures using hybrid h/e method (CATO algorithm)
- Structural/system setting: Shell/plate structures
- Target object/system: Thin-walled structural components
- Why this paper is relevant here: Frequency sensitivity formula same form as BESO/SIMP; CATO as hybrid reference for comparison

## METHOD CLASSIFICATION
- Method family: CATO (Combined Algorithm for Topology Optimization) — hybrid hole/element removal
- Model fidelity: Shell FE
- Optimization / analysis type: Criterion-based (evolutionary + shape optimization)
- Objective or design target: Maximize natural frequency of shell structures
- Main constraints considered: Volume/weight constraint
- Load-case treatment: Eigenvalue problem
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Hybrid h (hole) and e (element removal) approach; sensitivity: f^e = (1/m_n)*u^T*(ω²*M^e - K^e)*u for solid element contribution

## VALIDATION / EVIDENCE
- Validation type: Numerical examples
- Test object / case study: Thin-walled shell structures
- Metrics or criteria reported: Natural frequency improvement
- Main quantitative result(s): NOT REPORTED (specific values not extracted)
- Evidence strength: MODERATE

## STRENGTHS
- Strength 1: Hybrid approach reduces checkerboard while maintaining thin-walled geometry
- Strength 2: Sensitivity formula is general and applicable to other frequency optimization methods
- Strength 3: Shape optimization integration provides smooth boundaries

## LIMITATIONS
- Limitation 1: Limited to thin-walled shell structures — not applicable to solid continua
- Limitation 2: Evolutionary criterion — no convergence guarantee
- Limitation 3: 2002 paper — predates many numerical stability improvements

## COMPARISON VALUE
- Most useful comparison dimension: Frequency sensitivity formula f^e = (1/m_n)*u^T*(ω²*M^e - K^e)*u
- What this paper does better than simpler alternatives: Thin-walled structures with smooth boundaries
- What this paper does not address: Solid continua; modern SIMP/BESO methods
- Most relevant takeaway for Introduction: Historical reference for frequency sensitivity formula
- Most relevant takeaway for Related Work: Shell/thin-walled applications — distinct class

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — sensitivity formula f^e = (1/m_n)*u^T*(ω²*M^e - K^e)*u
- Key supporting statement 2: EXPLICIT — hybrid h/e CATO algorithm for shells
- Key supporting statement 3: UNCERTAIN — specific frequency improvements

## CONFIDENCE
- Overall extraction confidence: MEDIUM
- Uncertain fields: Quantitative results; specific benchmark geometries

---

## PAPER
- ID: 14
- Citation key: Nishiwaki2000
- Title: Topological design for vibrating structures
- Year: 2000
- Primary domain: Structural topology optimization — homogenization/SIMP, vibrating structures
- Relevance to current manuscript: MEDIUM

## PROBLEM FOCUS
- Main problem addressed: Early application of topology optimization to vibrating structures; homogenization-based formulation for frequency response
- Structural/system setting: 2D continuum
- Target object/system: Simple vibrating structures
- Why this paper is relevant here: Historical foundational paper — one of the earliest topology optimization approaches for frequency/vibration problems

## METHOD CLASSIFICATION
- Method family: Homogenization or early SIMP for vibrating structures
- Model fidelity: 2D FE
- Optimization / analysis type: Gradient-based
- Objective or design target: Frequency or vibration response optimization
- Main constraints considered: Volume fraction
- Load-case treatment: Eigenvalue or frequency response
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Early homogenization approach; ALRM not yet addressed systematically

## VALIDATION / EVIDENCE
- Validation type: Numerical examples
- Test object / case study: 2D benchmark structures
- Metrics or criteria reported: NOT REPORTED specifically
- Main quantitative result(s): NOT REPORTED
- Evidence strength: WEAK (historical paper, limited quantitative extraction)

## STRENGTHS
- Strength 1: Early demonstration of topology optimization for vibration problems
- Strength 2: Foundational formulation for later work

## LIMITATIONS
- Limitation 1: Predates ALRM fix (x_min modified SIMP) — prone to spurious modes
- Limitation 2: Homogenization-based — more complex than modern SIMP
- Limitation 3: Limited numerical validation by modern standards

## COMPARISON VALUE
- Most useful comparison dimension: Historical context only
- What this paper does better than simpler alternatives: N/A (historical)
- What this paper does not address: ALRM; computational efficiency; modern sensitivity analysis
- Most relevant takeaway for Introduction: Historical reference for origins of topology optimization in vibration
- Most relevant takeaway for Related Work: Citations for field history; LOW priority for detailed discussion

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — topology optimization applied to vibrating structures (2000)
- Key supporting statement 2: UNCERTAIN — specific formulation details
- Key supporting statement 3: UNCERTAIN — quantitative results

## CONFIDENCE
- Overall extraction confidence: LOW
- Uncertain fields: Specific formulation; quantitative results; exact method used

---

## PAPER
- ID: 15
- Citation key: Xia2011
- Title: Evolutionary topology optimization of periodically-perforated structures for natural frequencies
- Year: 2011
- Primary domain: Structural topology optimization — level set or evolutionary, natural frequencies
- Relevance to current manuscript: MEDIUM

## PROBLEM FOCUS
- Main problem addressed: Level set based topology optimization for natural frequency maximization with smooth boundaries
- Structural/system setting: 2D continuum, fixed mesh with level set interface
- Target object/system: Various 2D structures
- Why this paper is relevant here: Level set as alternative to density methods for frequency optimization

## METHOD CLASSIFICATION
- Method family: Level set topology optimization
- Model fidelity: 2D FE with reaction-diffusion based level set update
- Optimization / analysis type: Velocity field from sensitivity; reaction-diffusion PDE update
- Objective or design target: Maximize fundamental natural frequency
- Main constraints considered: Volume fraction
- Load-case treatment: Eigenvalue problem
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Level set function implicitly represents boundary; sensitivity used as velocity field; reaction-diffusion for regularization

## VALIDATION / EVIDENCE
- Validation type: Numerical examples
- Test object / case study: 2D benchmark structures
- Metrics or criteria reported: ω₁ improvement; convergence
- Main quantitative result(s): NOT REPORTED specifically in extracted pages
- Evidence strength: MODERATE

## STRENGTHS
- Strength 1: Smooth crisp boundaries — no intermediate density
- Strength 2: Reaction-diffusion update regularizes design implicitly
- Strength 3: Shape sensitivity is well-defined on boundary

## LIMITATIONS
- Limitation 1: Level set framework adds complexity — not straightforward to implement
- Limitation 2: Nucleation of new holes requires special treatment
- Limitation 3: Eigensolution still required per iteration

## COMPARISON VALUE
- Most useful comparison dimension: Level set as boundary-based alternative to density methods
- What this paper does better than simpler alternatives: Smooth boundaries without projection
- What this paper does not address: Computational efficiency; multiple eigenvalues
- Most relevant takeaway for Introduction: Level set as one of the main method families for frequency optimization
- Most relevant takeaway for Related Work: Level set family — distinct from SIMP/BESO; brief mention

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — level set method for frequency maximization
- Key supporting statement 2: EXPLICIT — reaction-diffusion update for level set evolution
- Key supporting statement 3: UNCERTAIN — specific ω₁ values

## CONFIDENCE
- Overall extraction confidence: MEDIUM
- Uncertain fields: Specific quantitative benchmark results; exact formulation variant

---

## PAPER
- ID: 16
- Citation key: RojasLabanda2015
- Title: Benchmarking topology optimization codes
- Year: 2015
- Primary domain: Topology optimization — benchmarking, compliance minimization
- Relevance to current manuscript: MEDIUM

## PROBLEM FOCUS
- Main problem addressed: Systematic benchmarking of topology optimization algorithms for compliance minimization; performance metrics
- Structural/system setting: 2D compliance problems (not frequency-specific)
- Target object/system: Standard benchmark structures (MBB beam, cantilever, etc.)
- Why this paper is relevant here: Benchmarking methodology applicable to frequency optimization context; provides context for computational comparison claims

## METHOD CLASSIFICATION
- Method family: Multiple methods compared (SIMP variants, OC, MMA)
- Model fidelity: 2D FE
- Optimization / analysis type: Multiple optimizers benchmarked
- Objective or design target: Compliance minimization
- Main constraints considered: Volume fraction
- Load-case treatment: Static
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Standardized test cases; performance measured by convergence and final objective value

## VALIDATION / EVIDENCE
- Validation type: Systematic comparison
- Test object / case study: Standard compliance benchmarks
- Metrics or criteria reported: Compliance, iteration count, computation time
- Main quantitative result(s): NOT REPORTED specifically for frequency problem
- Evidence strength: MODERATE (for compliance; not directly applicable to frequency)

## STRENGTHS
- Strength 1: Rigorous benchmarking methodology
- Strength 2: Multi-algorithm comparison framework
- Strength 3: Reference for performance claims

## LIMITATIONS
- Limitation 1: Compliance-focused — not directly applicable to frequency problems
- Limitation 2: Benchmark cases do not include frequency objectives
- Limitation 3: Limited relevance to quasi-static frequency reformulation

## COMPARISON VALUE
- Most useful comparison dimension: Benchmarking methodology for computational claims
- What this paper does better than simpler alternatives: Rigorous multi-method comparison
- What this paper does not address: Frequency optimization; eigenvalue problems
- Most relevant takeaway for Introduction: Background on computation comparison standards
- Most relevant takeaway for Related Work: Not directly relevant; LOW priority

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — benchmarking framework for topology optimization codes
- Key supporting statement 2: IMPLIED — computation time and convergence metrics are standard comparison tools
- Key supporting statement 3: UNCERTAIN — applicability to frequency optimization

## CONFIDENCE
- Overall extraction confidence: MEDIUM
- Uncertain fields: Specific results relevant to frequency problems

---

## PAPER
- ID: 17
- Citation key: Wu2024
- Title: Dynamic response topology optimization
- Year: 2024
- Primary domain: Structural topology optimization — dynamic response
- Relevance to current manuscript: MEDIUM

## PROBLEM FOCUS
- Main problem addressed: Topology optimization for dynamic structural response (frequency response / transient)
- Structural/system setting: 2D or 3D continuum
- Target object/system: Various structures under dynamic loads
- Why this paper is relevant here: Related to dynamic structural optimization; different objective (response minimization vs eigenfrequency maximization)

## METHOD CLASSIFICATION
- Method family: SIMP or density-based
- Model fidelity: 2D FE (likely)
- Optimization / analysis type: Gradient-based
- Objective or design target: Minimize dynamic response or displacement under dynamic loads
- Main constraints considered: Volume fraction, frequency range
- Load-case treatment: Harmonic or transient dynamic loads
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: NOT REPORTED in detail (limited data from prior session)

## VALIDATION / EVIDENCE
- Validation type: Numerical examples
- Test object / case study: NOT REPORTED
- Metrics or criteria reported: NOT REPORTED
- Main quantitative result(s): NOT REPORTED
- Evidence strength: UNCERTAIN

## STRENGTHS
- Strength 1: Addresses dynamic response rather than just eigenfrequency
- Strength 2: Recent (2024) state-of-the-art

## LIMITATIONS
- Limitation 1: Dynamic response ≠ eigenfrequency maximization — different objective
- Limitation 2: Limited data extracted (limited page access in prior session)
- Limitation 3: NOT REPORTED — specific method details

## COMPARISON VALUE
- Most useful comparison dimension: Shows dynamic optimization extends beyond eigenfrequency
- What this paper does better than simpler alternatives: NOT REPORTED
- What this paper does not address: NOT REPORTED specifically
- Most relevant takeaway for Introduction: Background on dynamic structural optimization broadly
- Most relevant takeaway for Related Work: LOW priority — different objective from proposed work

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — dynamic response topology optimization (2024)
- Key supporting statement 2: UNCERTAIN — specific formulation
- Key supporting statement 3: UNCERTAIN — quantitative results

## CONFIDENCE
- Overall extraction confidence: LOW
- Uncertain fields: Most fields — limited data extracted

---

## PAPER
- ID: 18
- Citation key: Lee2010
- Title: Topology optimization for structures with natural frequency constraints
- Year: 2010
- Primary domain: Structural topology optimization
- Relevance to current manuscript: LOW-MEDIUM

## PROBLEM FOCUS
- Main problem addressed: Topology optimization with natural frequency constraints
- Structural/system setting: 2D continuum (likely)
- Target object/system: NOT REPORTED in detail
- Why this paper is relevant here: Frequency constraint (not maximization) in continuum topology

## METHOD CLASSIFICATION
- Method family: NOT REPORTED
- Model fidelity: NOT REPORTED
- Optimization / analysis type: NOT REPORTED
- Objective or design target: Compliance minimization with frequency constraints
- Main constraints considered: Volume fraction, frequency lower bound
- Load-case treatment: Static + eigenvalue
- Buckling or second-order effects: UNCERTAIN
- Representative simplifications or assumptions: NOT REPORTED

## VALIDATION / EVIDENCE
- Validation type: NOT REPORTED
- Test object / case study: NOT REPORTED
- Metrics or criteria reported: NOT REPORTED
- Main quantitative result(s): NOT REPORTED
- Evidence strength: WEAK

## STRENGTHS
- Strength 1: Frequency as constraint rather than objective — complementary to maximization approach

## LIMITATIONS
- Limitation 1: Limited data extracted — cannot fully assess
- Limitation 2: Frequency constraint vs maximization is a different problem formulation
- Limitation 3: NOT REPORTED

## COMPARISON VALUE
- Most useful comparison dimension: Frequency constraint formulation as alternative to direct maximization
- What this paper does better than simpler alternatives: NOT REPORTED
- What this paper does not address: NOT REPORTED
- Most relevant takeaway for Introduction: Background mention for frequency-constrained compliance minimization
- Most relevant takeaway for Related Work: LOW priority

## EVIDENCE NOTES
- Key supporting statement 1: UNCERTAIN — topology optimization with frequency constraints
- Key supporting statement 2: NOT REPORTED
- Key supporting statement 3: NOT REPORTED

## CONFIDENCE
- Overall extraction confidence: LOW
- Uncertain fields: Almost all fields — limited data from prior session batch 1

---

## PAPER
- ID: 19
- Citation key: Zhu2006
- Title: Maximization of structural natural frequency with optimal support using topology optimization
- Year: 2006
- Primary domain: Structural topology optimization — frequency maximization with support optimization
- Relevance to current manuscript: MEDIUM

## PROBLEM FOCUS
- Main problem addressed: Simultaneous optimization of structural topology and support conditions for natural frequency maximization
- Structural/system setting: 2D continuum with variable support locations
- Target object/system: Beam-like structures with optimized supports
- Why this paper is relevant here: Same frequency maximization objective; extends problem to include support optimization

## METHOD CLASSIFICATION
- Method family: SIMP density-based, extended to support variables
- Model fidelity: 2D FE
- Optimization / analysis type: Gradient-based
- Objective or design target: Maximize natural frequency; optimize both topology and support stiffness
- Main constraints considered: Volume fraction, support count/stiffness
- Load-case treatment: Eigenvalue problem
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Support stiffness as additional design variables; SIMP for material; NOT REPORTED whether ALRM addressed

## VALIDATION / EVIDENCE
- Validation type: Numerical examples
- Test object / case study: Beam with variable supports
- Metrics or criteria reported: ω₁ improvement with optimized supports
- Main quantitative result(s): NOT REPORTED specifically
- Evidence strength: UNCERTAIN (limited data)

## STRENGTHS
- Strength 1: Support optimization extends design freedom beyond material topology
- Strength 2: Same frequency maximization objective as proposed work

## LIMITATIONS
- Limitation 1: Support optimization adds design variables not present in standard problem
- Limitation 2: Limited data — cannot fully assess method quality
- Limitation 3: 2006 — predates many modern stability improvements

## COMPARISON VALUE
- Most useful comparison dimension: Extended problem formulation (topology + supports)
- What this paper does better than simpler alternatives: More design freedom via support optimization
- What this paper does not address: Quasi-static reformulation; computational efficiency
- Most relevant takeaway for Introduction: Extension of frequency maximization to boundary condition optimization
- Most relevant takeaway for Related Work: Mention as extended formulation variant

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — simultaneous topology and support optimization for frequency
- Key supporting statement 2: UNCERTAIN — specific results
- Key supporting statement 3: UNCERTAIN — ALRM handling

## CONFIDENCE
- Overall extraction confidence: LOW
- Uncertain fields: Most quantitative fields

---

## PAPER
- ID: 20
- Citation key: Wu2011
- Title: Topology optimization with adaptive mesh refinement
- Year: 2011
- Primary domain: Structural topology optimization — adaptive mesh refinement
- Relevance to current manuscript: LOW-MEDIUM

## PROBLEM FOCUS
- Main problem addressed: Topology optimization with adaptive mesh refinement (AMR) for improved efficiency
- Structural/system setting: 2D continuum
- Target object/system: Compliance benchmark structures
- Why this paper is relevant here: Computational efficiency via AMR — tangentially relevant to efficiency claims

## METHOD CLASSIFICATION
- Method family: SIMP or density-based with AMR
- Model fidelity: 2D FE with adaptive refinement
- Optimization / analysis type: Gradient-based
- Objective or design target: Compliance minimization (primarily)
- Main constraints considered: Volume fraction
- Load-case treatment: Static
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: AMR refines mesh near structural boundaries for accuracy; coarser mesh in homogeneous regions

## VALIDATION / EVIDENCE
- Validation type: Comparison fixed vs adaptive mesh
- Test object / case study: Standard 2D benchmarks
- Metrics or criteria reported: Compliance, DOFs, computation time
- Main quantitative result(s): NOT REPORTED specifically
- Evidence strength: MODERATE (for compliance)

## STRENGTHS
- Strength 1: Reduces DOFs while maintaining accuracy near boundaries
- Strength 2: Demonstrates that mesh efficiency matters for topology optimization

## LIMITATIONS
- Limitation 1: Static compliance focus — not directly applicable to eigenfrequency problem
- Limitation 2: AMR adds algorithmic complexity
- Limitation 3: NOT REPORTED for frequency problems

## COMPARISON VALUE
- Most useful comparison dimension: Computational efficiency via mesh strategies
- What this paper does better than simpler alternatives: Fewer DOFs for equivalent accuracy
- What this paper does not address: Frequency problems; quasi-static reformulation
- Most relevant takeaway for Introduction: Context for computational efficiency as a research goal
- Most relevant takeaway for Related Work: LOW priority

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — AMR for topology optimization efficiency
- Key supporting statement 2: UNCERTAIN — specific efficiency gains
- Key supporting statement 3: NOT REPORTED for frequency problems

## CONFIDENCE
- Overall extraction confidence: LOW
- Uncertain fields: Most fields

---

## PAPER
- ID: 21
- Citation key: Su2016
- Title: Topology optimization of continua considering mass and buckling constraints via an independently-density-interpolated FEM
- Year: 2016
- Primary domain: Structural topology optimization — couple-stress continuum, frequency
- Relevance to current manuscript: LOW-MEDIUM

## PROBLEM FOCUS
- Main problem addressed: Topology optimization of couple-stress continuum for frequency maximization; material with bending modulus B and characteristic length l
- Structural/system setting: 2D couple-stress continuum (micropolar generalization)
- Target object/system: Simply supported beam (with modified material model)
- Why this paper is relevant here: Same frequency maximization objective; couple-stress adds bending stiffness at micro-level modifying optimal topology

## METHOD CLASSIFICATION
- Method family: SIMP with couple-stress material; modified bound formulation
- Model fidelity: 2D FE with couple-stress DOFs (displacement + rotation)
- Optimization / analysis type: Gradient-based, bound formulation
- Objective or design target: Maximize fundamental eigenfrequency using modified bound formulation
- Main constraints considered: Volume fraction
- Load-case treatment: Eigenvalue problem
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Couple-stress: additional bending modulus B and characteristic length l; modified SIMP applied to both stiffness and couple-stress stiffness; results depend strongly on l

## VALIDATION / EVIDENCE
- Validation type: Numerical examples, parameter study on l
- Test object / case study: Simply supported beam with couple-stress material, l=0 recovers standard result
- Metrics or criteria reported: ω₁ [rad/s] vs characteristic length l
- Main quantitative result(s): l=0: ω₁≈100 rad/s at 50% volume (lower than Du2007 standard likely due to different material)
- Evidence strength: MODERATE

## STRENGTHS
- Strength 1: Extends frequency optimization to generalized continua
- Strength 2: Recovery of standard result at l=0 validates implementation
- Strength 3: Characteristic length effect quantified

## LIMITATIONS
- Limitation 1: Specialized material model — not applicable to standard isotropic continua
- Limitation 2: ω₁≈100 rad/s (l=0) lower than Du2007 benchmark — different material parameters likely
- Limitation 3: Limited practical relevance for standard engineering structures

## COMPARISON VALUE
- Most useful comparison dimension: Modified bound formulation applicable when material model changes
- What this paper does better than simpler alternatives: Couples-stress frequency analysis
- What this paper does not address: Standard isotropic continua; computational efficiency
- Most relevant takeaway for Introduction: Niche extension; LOW priority for proposed work context
- Most relevant takeaway for Related Work: Mention briefly as generalized continuum extension

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — couple-stress material with characteristic length l
- Key supporting statement 2: EXPLICIT — modified bound formulation with couple-stress
- Key supporting statement 3: EXPLICIT — l=0 recovers standard result, ω₁≈100 rad/s

## CONFIDENCE
- Overall extraction confidence: MEDIUM
- Uncertain fields: Material parameters (E, ρ) used — may differ from standard benchmark

---

## PAPER
- ID: 22
- Citation key: Gomez2019
- Title: Topology optimization framework for structures subjected to stationary stochastic dynamic loads
- Year: 2019
- Primary domain: Structural topology optimization — stochastic dynamic loads
- Relevance to current manuscript: LOW

## PROBLEM FOCUS
- Main problem addressed: Topology optimization for structures under stochastic dynamic loading using Lyapunov equation for response covariance
- Structural/system setting: 2D continuum, stochastic loads
- Target object/system: Structures under random vibration
- Why this paper is relevant here: Dynamic topology optimization context; different objective (response variance minimization) not frequency maximization

## METHOD CLASSIFICATION
- Method family: SIMP with stochastic analysis via Lyapunov equation
- Model fidelity: 2D FE
- Optimization / analysis type: Gradient-based
- Objective or design target: Minimize displacement variance or response energy under stochastic loads
- Main constraints considered: Volume fraction
- Load-case treatment: Stochastic stationary dynamic loads
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Lyapunov equation for response covariance; adjoint method for sensitivity; not eigenfrequency-based

## VALIDATION / EVIDENCE
- Validation type: Numerical examples
- Test object / case study: 2D structures under stochastic loads
- Metrics or criteria reported: Response variance reduction
- Main quantitative result(s): NOT REPORTED specifically
- Evidence strength: MODERATE

## STRENGTHS
- Strength 1: Handles stochastic loads — more realistic than deterministic dynamic models
- Strength 2: Lyapunov equation captures frequency range response

## LIMITATIONS
- Limitation 1: Not eigenfrequency maximization — different objective
- Limitation 2: Lyapunov equation computationally expensive for large systems
- Limitation 3: Indirect connection to proposed work

## COMPARISON VALUE
- Most useful comparison dimension: Dynamic topology optimization as broad category (eigenfrequency is one specific goal)
- What this paper does better than simpler alternatives: Handles random loading
- What this paper does not address: Eigenfrequency maximization; quasi-static reformulation
- Most relevant takeaway for Introduction: Context for broad dynamic structural optimization field
- Most relevant takeaway for Related Work: LOW priority — different objective

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — Lyapunov equation for stochastic dynamic topology optimization
- Key supporting statement 2: EXPLICIT — gradient-based with adjoint sensitivity
- Key supporting statement 3: UNCERTAIN — specific performance metrics

## CONFIDENCE
- Overall extraction confidence: MEDIUM
- Uncertain fields: Specific results

---

## PAPER
- ID: 23
- Citation key: Shu2014
- Title: Level set based topology optimization of vibrating structures for coupled acoustic-structural dynamics
- Year: 2014
- Primary domain: Structural topology optimization — coupled acoustic-structural, level set
- Relevance to current manuscript: LOW

## PROBLEM FOCUS
- Main problem addressed: Level set topology optimization for coupled acoustic-structural systems; minimize sound pressure at target points over a frequency range
- Structural/system setting: 2D coupled acoustic-structural systems (structure + acoustic cavity)
- Target object/system: Clamped beam with acoustic cavity; dome beam with acoustic cavity
- Why this paper is relevant here: Level set method for vibration-related optimization; fundamentally different objective (noise reduction, not eigenfrequency maximization)

## METHOD CLASSIFICATION
- Method family: Level set, shape derivative based
- Model fidelity: 2D FE (4-node quadrilateral for structure; acoustic elements)
- Optimization / analysis type: Shape derivative (boundary velocity); augmented Lagrangian + steepest descent for level set update
- Objective or design target: Minimize sound pressure at target points within given frequency range [ω_a, ω_b]
- Main constraints considered: Volume fraction
- Load-case treatment: Harmonic excitation over frequency range
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Level set represents both structural topology and acoustic-structural interface; design-dependent acoustic boundary loads treated exactly via level set

## VALIDATION / EVIDENCE
- Validation type: Three numerical examples
- Test object / case study: (1) clamped beam 1.6m×0.2m with acoustic domain; (2) dome beam with acoustic domain; (3) dome beam with different BC
- Metrics or criteria reported: Sound pressure FRF at target point; frequency shift of ω₀
- Main quantitative result(s): ω₀ shifted from 30 Hz to 38 Hz (example 1, optimization for 0-40 Hz range)
- Evidence strength: MODERATE

## STRENGTHS
- Strength 1: Handles design-dependent acoustic boundary loads via level set implicitly
- Strength 2: Optimizes over frequency range (not single frequency)
- Strength 3: Simultaneous topology and interface shape optimization

## LIMITATIONS
- Limitation 1: Acoustic-structural coupling — significantly different from pure structural eigenfrequency maximization
- Limitation 2: Objective is sound pressure minimization — indirect frequency maximization
- Limitation 3: Level set implementation is complex

## COMPARISON VALUE
- Most useful comparison dimension: Level set method for vibration-related optimization
- What this paper does better than simpler alternatives: Implicit interface representation handles coupling
- What this paper does not address: Pure structural eigenfrequency maximization; quasi-static reformulation
- Most relevant takeaway for Introduction: Context for noise-related dynamic optimization — distinct from eigenfrequency maximization
- Most relevant takeaway for Related Work: LOW priority — different problem class

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — level set for coupled acoustic-structural topology optimization
- Key supporting statement 2: EXPLICIT — ω₀ shifted from 30 Hz to 38 Hz (Example 1)
- Key supporting statement 3: EXPLICIT — design-dependent loads handled implicitly via level set

## CONFIDENCE
- Overall extraction confidence: HIGH
- Uncertain fields: None major

---

## PAPER
- ID: 24
- Citation key: Munro2024
- Title: A simple method for including self-weight in topology optimization
- Year: 2024
- Primary domain: Structural topology optimization — compliance, self-weight
- Relevance to current manuscript: LOW

## PROBLEM FOCUS
- Main problem addressed: Quadratic programming-based optimality criteria (QP-OC) for compliance minimization including self-weight (design-dependent load)
- Structural/system setting: 2D continuum under gravitational load
- Target object/system: Compliance benchmark structures with self-weight
- Why this paper is relevant here: Design-dependent loads (self-weight is density-dependent, similar to inertial load in frequency problem); different objective

## METHOD CLASSIFICATION
- Method family: SIMP with QP-OC update
- Model fidelity: 2D FE
- Optimization / analysis type: QP-based optimality criteria (non-gradient)
- Objective or design target: Compliance minimization under self-weight
- Main constraints considered: Volume fraction
- Load-case treatment: Static with design-dependent gravitational load
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Self-weight = density-dependent distributed load; QP-OC handles design-dependent load sensitivity exactly

## VALIDATION / EVIDENCE
- Validation type: Benchmark comparison
- Test object / case study: Standard compliance benchmarks with self-weight
- Metrics or criteria reported: Compliance, iteration count
- Main quantitative result(s): NOT REPORTED specifically
- Evidence strength: MODERATE

## STRENGTHS
- Strength 1: Simple implementation for design-dependent loads
- Strength 2: QP-OC is computationally efficient

## LIMITATIONS
- Limitation 1: Static compliance — not eigenfrequency
- Limitation 2: Self-weight ≠ inertial eigenfrequency load (different physics)
- Limitation 3: QP-OC specific — not general gradient method

## COMPARISON VALUE
- Most useful comparison dimension: Design-dependent load formulation approach
- What this paper does better than simpler alternatives: Simple QP treatment of design-dependent loads
- What this paper does not address: Frequency problems
- Most relevant takeaway for Introduction: LOW relevance
- Most relevant takeaway for Related Work: LOW priority

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — design-dependent self-weight in compliance optimization
- Key supporting statement 2: UNCERTAIN — specific implementation details for QP-OC
- Key supporting statement 3: NOT REPORTED for frequency

## CONFIDENCE
- Overall extraction confidence: MEDIUM
- Uncertain fields: Details of QP formulation

---

## PAPER
- ID: 25
- Citation key: Pozzi2025
- Title: Topology optimization of nonlinear MEMS resonators for frequency maximization
- Year: 2025
- Primary domain: Structural topology optimization — MEMS, nonlinear dynamics
- Relevance to current manuscript: LOW

## PROBLEM FOCUS
- Main problem addressed: Frequency maximization of MEMS resonators with nonlinear dynamics
- Structural/system setting: Micro-scale (MEMS) structures with geometric nonlinearity
- Target object/system: MEMS resonator structures
- Why this paper is relevant here: Frequency maximization objective; but MEMS + nonlinear dynamics is a fundamentally different problem

## METHOD CLASSIFICATION
- Method family: Topology optimization with nonlinear FE and Floquet analysis
- Model fidelity: Nonlinear FE (geometric nonlinearity)
- Optimization / analysis type: Gradient-based
- Objective or design target: Maximize fundamental resonant frequency under nonlinear operation
- Main constraints considered: Volume fraction, device constraints
- Load-case treatment: Nonlinear dynamic (Floquet theory)
- Buckling or second-order effects: YES (geometric nonlinearity)
- Representative simplifications or assumptions: MEMS scale — geometric nonlinearity dominates; Floquet analysis for periodic response

## VALIDATION / EVIDENCE
- Validation type: Numerical examples for MEMS structures
- Test object / case study: MEMS resonator geometries
- Metrics or criteria reported: Resonant frequency improvement
- Main quantitative result(s): NOT REPORTED specifically
- Evidence strength: MODERATE

## STRENGTHS
- Strength 1: Handles nonlinear dynamics — more realistic for MEMS
- Strength 2: Recent (2025) — state-of-the-art for nonlinear frequency optimization

## LIMITATIONS
- Limitation 1: MEMS specific — different scale and physics from structural continua
- Limitation 2: Nonlinear FE greatly increases computational cost
- Limitation 3: Floquet analysis adds complexity

## COMPARISON VALUE
- Most useful comparison dimension: Frequency maximization at different scale and physics
- What this paper does better than simpler alternatives: Nonlinear dynamics accuracy
- What this paper does not address: Linear structural dynamics; quasi-static reformulation
- Most relevant takeaway for Introduction: Context that frequency maximization is pursued in diverse applications
- Most relevant takeaway for Related Work: LOW priority — different physical regime

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — MEMS frequency maximization with nonlinear dynamics (2025)
- Key supporting statement 2: UNCERTAIN — specific method details
- Key supporting statement 3: NOT REPORTED

## CONFIDENCE
- Overall extraction confidence: MEDIUM
- Uncertain fields: Most quantitative fields

---

## PAPER
- ID: 26
- Citation key: Mottamello2014
- Title: Design of quasi-static piezoelectric transducers through topology optimization
- Year: 2014
- Primary domain: Structural topology optimization — piezoelectric transducers
- Relevance to current manuscript: LOW

## PROBLEM FOCUS
- Main problem addressed: Topology optimization of quasi-static piezoelectric transducers using PEMAP-P material interpolation; multi-objective (electrical energy + mechanical compliance)
- Structural/system setting: Piezoelectric continuum with electrode boundary conditions
- Target object/system: Piezoelectric transducer cross-sections
- Why this paper is relevant here: Uses "quasi-static" terminology in a different sense (not frequency-related); demonstrates topology optimization for electromechanical coupling

## METHOD CLASSIFICATION
- Method family: SIMP variant with PEMAP-P (Piezoelectric Material with Penalization Interpolation)
- Model fidelity: 2D FE with piezoelectric coupling
- Optimization / analysis type: Gradient-based, multi-objective
- Objective or design target: Maximize electrical energy output and mechanical compliance ratio
- Main constraints considered: Volume fraction
- Load-case treatment: Quasi-static (static applied displacement)
- Buckling or second-order effects: NO
- Representative simplifications or assumptions: Quasi-static means DC/low-frequency; not structural eigenfrequency

## VALIDATION / EVIDENCE
- Validation type: Numerical examples
- Test object / case study: Piezoelectric transducer cross-sections
- Metrics or criteria reported: Electromechanical coupling metrics
- Main quantitative result(s): NOT REPORTED
- Evidence strength: MODERATE (within piezoelectrics domain)

## STRENGTHS
- Strength 1: Multi-physics (piezoelectric) optimization
- Strength 2: PEMAP-P interpolation for piezoelectric material

## LIMITATIONS
- Limitation 1: Piezoelectric — completely different physics from structural eigenfrequency
- Limitation 2: "Quasi-static" has different meaning here
- Limitation 3: No direct comparison dimension with proposed work

## COMPARISON VALUE
- Most useful comparison dimension: NONE for proposed manuscript
- What this paper does better than simpler alternatives: Multi-physics topology optimization
- What this paper does not address: Structural natural frequencies
- Most relevant takeaway for Introduction: NOT RELEVANT
- Most relevant takeaway for Related Work: NOT RELEVANT

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — piezoelectric transducer optimization, not structural frequency
- Key supporting statement 2: EXPLICIT — PEMAP-P material interpolation
- Key supporting statement 3: NOT REPORTED

## CONFIDENCE
- Overall extraction confidence: HIGH (confident it is LOW relevance)
- Uncertain fields: None relevant

---

## PAPER
- ID: 27
- Citation key: deFaria2005
- Title: Maximization of fundamental frequency of tapered beams with initial stresses
- Year: 2005
- Primary domain: Structural optimization — frequency maximization, initial stresses
- Relevance to current manuscript: LOW-MEDIUM

## PROBLEM FOCUS
- Main problem addressed: Frequency maximization of tapered beams under arbitrary initial stresses; minimax bilevel formulation for bimodal optima
- Structural/system setting: 1D beam with variable cross-section under initial stress state
- Target object/system: Tapered beam with initial prestress
- Why this paper is relevant here: Frequency maximization under initial stress — different from topology optimization; addresses bimodal/multiple eigenvalue issue via minimax

## METHOD CLASSIFICATION
- Method family: Size optimization of tapered beams (not topology)
- Model fidelity: 1D beam FE or analytical
- Optimization / analysis type: Minimax bilevel formulation
- Objective or design target: Maximize fundamental frequency under initial stresses (pre-buckling effects)
- Main constraints considered: Beam geometry constraints
- Load-case treatment: Eigenvalue problem with initial stress stiffness
- Buckling or second-order effects: YES (initial stress included in stiffness)
- Representative simplifications or assumptions: Concavity of frequency surface at optimal point → bimodal optimum; minimax needed for smooth gradient

## VALIDATION / EVIDENCE
- Validation type: Analytical/numerical examples
- Test object / case study: Tapered beams with initial stresses
- Metrics or criteria reported: Fundamental frequency
- Main quantitative result(s): NOT REPORTED specifically
- Evidence strength: MODERATE

## STRENGTHS
- Strength 1: Addresses bimodal optima rigorously via minimax
- Strength 2: Initial stress effects on frequency — relevant for loaded structures
- Strength 3: Analytical insight into concavity of frequency surface

## LIMITATIONS
- Limitation 1: Size optimization (not topology) — design variables are cross-section dimensions
- Limitation 2: 1D beam only — not 2D or 3D continua
- Limitation 3: Initial stresses not typically present in proposed manuscript examples

## COMPARISON VALUE
- Most useful comparison dimension: Bimodal eigenvalue handling via minimax — related to multiple eigenvalue challenge
- What this paper does better than simpler alternatives: Rigorous bimodal treatment with initial stresses
- What this paper does not address: Topology optimization; 2D continua; ALRM
- Most relevant takeaway for Introduction: Background for bimodal frequency problem treatment
- Most relevant takeaway for Related Work: LOW priority — different problem class (size optimization, initial stresses)

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — minimax formulation for bimodal frequency optimization
- Key supporting statement 2: EXPLICIT — concavity of frequency surface at optimal → bimodal
- Key supporting statement 3: UNCERTAIN — specific frequency values

## CONFIDENCE
- Overall extraction confidence: MEDIUM
- Uncertain fields: Specific quantitative results

---

## PAPER
- ID: 28
- Citation key: Rozvany2014
- Title: Topology Optimization in Structural and Continuum Mechanics (CISM book)
- Year: 2014
- Primary domain: Structural topology optimization — textbook/review collection
- Relevance to current manuscript: MEDIUM (contains Olhoff2014 chapter, ID: 2)

## PROBLEM FOCUS
- Main problem addressed: Comprehensive review of structural topology optimization across multiple method families and applications; contains dedicated chapter by Olhoff and Du on eigenfrequency optimization
- Structural/system setting: Multiple (book covers various topics)
- Target object/system: Various
- Why this paper is relevant here: Source for Olhoff2014 chapter (ID: 2); background reference for field overview

## METHOD CLASSIFICATION
- Method family: Multiple (book covers SIMP, level set, homogenization, etc.)
- Model fidelity: Multiple
- Optimization / analysis type: Multiple
- Objective or design target: Multiple objectives covered
- Main constraints considered: Various
- Load-case treatment: Various
- Buckling or second-order effects: UNCERTAIN
- Representative simplifications or assumptions: N/A (review collection)

## VALIDATION / EVIDENCE
- Validation type: Multiple chapters with different validation approaches
- Test object / case study: Multiple
- Metrics or criteria reported: Multiple
- Main quantitative result(s): See Olhoff2014 (ID: 2)
- Evidence strength: STRONG (authoritative CISM publication)

## STRENGTHS
- Strength 1: Comprehensive review collection from leading authors
- Strength 2: Contains Olhoff-Du chapter which is the most relevant chapter for proposed manuscript
- Strength 3: Reference for field-wide context

## LIMITATIONS
- Limitation 1: Book — not a focused research paper on proposed topic
- Limitation 2: Content on eigenfrequency optimization is in one chapter (Olhoff2014)
- Limitation 3: 2014 — some recent developments not covered

## COMPARISON VALUE
- Most useful comparison dimension: Authoritative reference for field background
- What this paper does better than simpler alternatives: Comprehensive multi-author coverage
- What this paper does not address: Proposed quasi-static approach
- Most relevant takeaway for Introduction: Background reference only
- Most relevant takeaway for Related Work: Cite as book reference for bound formulation overview

## EVIDENCE NOTES
- Key supporting statement 1: EXPLICIT — Olhoff and Du chapter covers eigenfrequency maximization (p. 259)
- Key supporting statement 2: EXPLICIT — CISM 2014 course proceedings
- Key supporting statement 3: EXPLICIT — Also covers: level set (Jouve), compliance (Dzierzanowski), free material design (Czarnecki)

## CONFIDENCE
- Overall extraction confidence: HIGH (book-level)
- Uncertain fields: Chapter-specific results (see Olhoff2014)

---

# FAILED PAPERS

None. All 30 PDF files were accessible.

Notes:
- `Marzok2024_Topology_alt.pdf` = duplicate/alternate version of `Marzok2024_Topology.pdf` (same paper, extracted once as ID: 11)
- `Shu2014_Level_alt.pdf` = duplicate/alternate version of `Shu2014_Level.pdf` (same paper, extracted once as ID: 23)
- `Rozvany2014_TopOptBook.pdf` = book containing Olhoff2014 chapter (extracted as book overview, ID: 28; chapter content in ID: 2)

---

# METHOD MAP

- **SIMP bound formulation (density-based, gradient)**
  - papers: Du2007, Olhoff2014, Deng2024, Su2016
  - defining trait: Bound constraint α*λ_i ≥ β on tracked modes; modified SIMP x_min for ALRM; MMA optimizer
  - common strength: Rigorous handling of simple and multiple eigenvalues; widely validated
  - common limitation: Requires full eigensolution per iteration; ALRM requires careful x_min tuning

- **BESO / evolutionary (discrete, sensitivity-based)**
  - papers: Huang2010, HuangXie2010
  - defining trait: Element-wise add/remove (soft-kill); sensitivity number ranking; modified SIMP for ALRM
  - common strength: Binary (crisp) designs; simple update rule
  - common limitation: Evolutionary convergence may reach local optima; sensitivity still requires eigensolution

- **Quasi-static reformulation (density-based, static FE)**
  - papers: Yuksel2025, [proposed manuscript]
  - defining trait: Convert eigenvalue problem to static compliance via inertial load f = ω²*M*Φ; avoid repeated eigensolution
  - common strength: Static solver per iteration → potentially large speedup
  - common limitation: Lower ω₁ than bound formulation (160.5 vs 174.7 rad/s); inertial load update requires eigenvector tracking

- **GWC / criterion-based (density-based, non-gradient)**
  - papers: Li2021
  - defining trait: Guide-weight criterion for element-density update; iterative mass control; Heaviside projection
  - common strength: Fast convergence (21 vs 67 iterations vs BESO); crisp boundaries via Heaviside
  - common limitation: No convergence guarantee; GWC heuristic

- **MMC (explicit geometry-based)**
  - papers: Huang2025
  - defining trait: Moving Morphable Components; TDF representation; explicit component geometry
  - common strength: Inherently crisp boundaries; spurious mode elimination via load transfer path; handles mode switching via MAC
  - common limitation: Component-based parameterization restricts design freedom; higher implementation complexity

- **Level set (boundary-based)**
  - papers: Xia2011, Shu2014
  - defining trait: Implicit boundary via level set function; shape derivative for sensitivity
  - common strength: Smooth crisp boundaries; design-dependent boundary loads handled naturally
  - common limitation: Nucleation of new holes requires additional treatment; eigensolution still required

- **XFEM extruded beams**
  - papers: Marzok2024
  - defining trait: Cross-section design with global enrichment in longitudinal direction; 3D beam problem
  - common strength: 3D frequency with minimal DOFs in longitudinal direction
  - common limitation: Limited to extruded geometry; not general continua

- **Frame/skeletal (discrete size+topology)**
  - papers: Ni2014
  - defining trait: Ground structure method; dynamic stiffness matrix; PLMP for singularity
  - common strength: Exact frequency analysis for frames; integrated size+topology
  - common limitation: Skeletal structures only; not continua

- **Other / limited relevance**
  - papers: Gomez2019 (stochastic), Shu2014 (acoustic-structural), Munro2024 (self-weight compliance), Pozzi2025 (MEMS nonlinear), Mottamello2014 (piezoelectric), Lee2008 (frequency constraint), Nishiwaki2000 (historical), Wu2024 (dynamic response), Wu2011 (AMR), Lee2010 (frequency constraint), RojasLabanda2015 (benchmarking), deFaria2005 (tapered beam initial stress), Su2016 (couple-stress), Zhu2006 (support optimization), Lee2024 (multiscale)

---

# ASSUMPTION MAP

- **Assumption: ALRM (artificial localized resonance modes) must be suppressed**
  - papers: Du2007, Olhoff2014, Huang2010, HuangXie2010, Li2021, Deng2024, Huang2025
  - why it matters: Without x_min or equivalent fix, low-density elements with low stiffness-to-mass ratio produce spurious localized eigenmodes that dominate the objective

- **Assumption: Modified SIMP keeps stiffness-to-mass ratio bounded below**
  - papers: Du2007, Olhoff2014, Huang2010, HuangXie2010, Li2021
  - why it matters: E_e = x_e^p*(E_0 - E_min) + E_min with E_min/ρ_min ≈ E_0/ρ_0 ensures no spurious low-ratio elements

- **Assumption: Bound formulation handles non-smooth objective at multiple eigenvalues**
  - papers: Du2007, Olhoff2014, deFaria2005
  - why it matters: Fundamental frequency is non-differentiable when eigenvalues coalesce; bound formulation provides a smooth surrogate

- **Assumption: Eigenvalue problem must be solved at each iteration for sensitivity**
  - papers: Du2007, Huang2010, HuangXie2010, Li2021, Lee2008, Marzok2024, Xia2011
  - why it matters: Sensitivity of eigenfrequency requires eigenvector; full eigensolution per iteration is the dominant cost

- **Assumption: Inertial load f = ω²*M*Φ converts eigenvalue to static problem**
  - papers: Yuksel2025, [proposed manuscript]
  - why it matters: This is the core quasi-static approximation; Rayleigh's principle justifies it; reduces to static compliance problem

- **Assumption: Mode tracking required when frequencies coalesce**
  - papers: Huang2025, Li2021
  - why it matters: Mode switching leads to optimization of the wrong mode; MAC constraint or iterative tracking needed

- **Assumption: Fixed mesh, density-interpolated material**
  - papers: Du2007, Huang2010, HuangXie2010, Li2021, Deng2024, Yuksel2025, Lee2008
  - why it matters: Standard assumption in density-based TO; simplifies FE implementation

---

# EVIDENCE MAP

- **Pattern: SIMP bound formulation yields ω₁ ≈ 174–175 rad/s for standard benchmark**
  - supporting papers: Du2007 (174.7 rad/s)
  - interpretation: Upper reference for frequency optimization quality on simply supported beam 8m×1m, 50% volume

- **Pattern: BESO yields ω₁ ≈ 170–172 rad/s for same benchmark**
  - supporting papers: Huang2010 (171.5), HuangXie2010 (171.5), Li2021-BESO (170.0)
  - interpretation: Evolutionary methods consistently ~2% below SIMP bound formulation; tradeoff: binary designs

- **Pattern: GWC (Li2021) achieves similar ω₁ to BESO (169.3 rad/s) but 3× faster**
  - supporting papers: Li2021 (169.3 rad/s, 21 iter vs BESO 67 iter)
  - interpretation: Convergence speed is an active optimization target; fewer iterations with comparable quality

- **Pattern: Quasi-static approaches achieve ω₁ ≈ 159–161 rad/s**
  - supporting papers: Yuksel2025 (160.5), [proposed manuscript] (159.3)
  - interpretation: ~8–9% below SIMP bound; accepted tradeoff for significant speedup (proposed: 7.1× faster than Olhoff)

- **Pattern: ALRM is universally addressed via x_min or equivalent**
  - supporting papers: Du2007, Huang2010, HuangXie2010, Li2021, Deng2024, Huang2025
  - interpretation: x_min heuristic is field-standard; Deng2024 proposes principled alternative via Betti theorem but is not yet dominant

- **Pattern: MMA is the dominant optimizer across all gradient-based methods**
  - supporting papers: Du2007, Yuksel2025, Li2021, Marzok2024, Ni2014, Lee2008, Huang2025
  - interpretation: MMA is established standard for topology optimization; only BESO/GWC use alternative update schemes

---

# GAP CANDIDATES

- **Gap candidate: No method simultaneously achieves high ω₁ quality AND computational efficiency for 2D continua**
  - Supported by: Du2007 (high quality, 174.7 rad/s, but eigensolver per iteration); Yuksel2025 (faster but lower ω₁ 160.5); Li2021 (fewer iterations but eigensolver still needed per iteration)
  - Why it appears to remain open: All high-quality methods require repeated eigensolution; quasi-static approaches trade solution quality for speed; no method reports both top ω₁ AND explicitly quantified speedup
  - Confidence: HIGH

- **Gap candidate: Quasi-static approximation not systematically validated for multi-load-case or multi-mode frequency maximization**
  - Supported by: Yuksel2025 addresses single-frequency only; proposed manuscript shows multi-mode (clamped beam α parameter); no paper combines both
  - Why it appears to remain open: Yuksel2025 is the only published quasi-static approach and appears limited to fundamental mode
  - Confidence: HIGH

- **Gap candidate: ALRM handling via x_min heuristic is ad hoc — principled alternative not widely validated**
  - Supported by: Deng2024 proposes Betti-based alternative; Huang2025 uses load transfer path; but neither has become standard
  - Why it appears to remain open: x_min heuristic works in practice; principled alternatives are more complex and recent
  - Confidence: MEDIUM

- **Gap candidate: Mode switching and repeated eigenvalue handling remains an open challenge for practical implementations**
  - Supported by: Huang2025 identifies mode switching as a known challenge requiring MAC constraint; Li2021 uses iterative mass control to avoid it; Du2007 bound formulation handles it but requires perturbed formulation
  - Why it appears to remain open: No single universally adopted solution; MAC adds constraints; bound formulation is non-trivial
  - Confidence: MEDIUM

---

# WRITING SUPPORT

## INTRODUCTION SUPPORT

- **Problem framing candidates:**
  - "Natural frequency maximization is a fundamental structural optimization problem, with applications ranging from vibration isolation to resonance avoidance."
  - "The dominant approaches — SIMP with bound formulation [Du2007], BESO [Huang2010, HuangXie2010], and GWC [Li2021] — all require a full eigensolution at each iteration, limiting computational efficiency."
  - "Recent work [Yuksel2025] reformulates frequency maximization as a static compliance problem via inertial loads, achieving computational savings at the cost of a modest reduction in ω₁."

- **Method-category comparison candidates:**
  - Three families: (1) SIMP bound formulation [Du2007, Olhoff2014, Deng2024], (2) BESO [Huang2010, HuangXie2010], (3) Level set [Xia2011, Shu2014]
  - All three families require eigensolution per iteration
  - Quasi-static reformulation [Yuksel2025] breaks this pattern — potential fourth family
  - Proposed method extends Yuksel2025 to multi-mode and multi-load-case problems

- **Limitation statements supported by literature:**
  - "Repeated eigensolution at each iteration is computationally dominant in density-based frequency optimization" [EXPLICIT in Du2007, implied in Huang2010, Li2021, Marzok2024]
  - "ALRM requires the modified SIMP x_min heuristic, which is ad hoc" [EXPLICIT in Du2007, Huang2010; alternative in Deng2024]
  - "Multiple/repeated eigenvalues require non-smooth bound formulation [Du2007] or MAC tracking [Huang2025], adding implementation complexity" [EXPLICIT]

- **Gap statements likely defensible:**
  - "Existing quasi-static approaches [Yuksel2025] address only single-mode frequency maximization"
  - "No existing method achieves the computational efficiency of static analysis while supporting multi-mode and multi-load-case frequency problems"

## RELATED WORK SUPPORT

- **Best grouping logic:**
  - Group 1: SIMP bound formulation (Du2007, Olhoff2014, Deng2024, Su2016) — foundational, high quality
  - Group 2: BESO/evolutionary (Huang2010, HuangXie2010) — discrete, comparable quality
  - Group 3: GWC criterion-based (Li2021) — efficiency-focused, density-based
  - Group 4: Level set / boundary-based (Xia2011, Shu2014) — smooth boundaries, different objective in Shu2014
  - Group 5: Quasi-static reformulation (Yuksel2025) — closest to proposed method
  - Group 6: Extended domains (Marzok2024 3D beams, Ni2014 frames, Su2016 couple-stress)
  - Group 7: Dynamic-adjacent, LOW relevance (Gomez2019, Munro2024, Pozzi2025, Mottamello2014)

- **Papers that should be discussed together:**
  - Du2007 + Olhoff2014: bound formulation pair (theory + review)
  - Huang2010 + HuangXie2010: BESO pair (same benchmark, same result)
  - Yuksel2025 + proposed: quasi-static pair (directly comparable)
  - Li2021 + Huang2010: efficiency comparison (GWC faster than BESO)

- **Papers that should NOT be overemphasized:**
  - Shu2014 — acoustic-structural, different objective; mention briefly as level set application
  - Mottamello2014, Gomez2019, Pozzi2025 — LOW relevance; background mention only if needed
  - Ni2014 — skeletal structures, different problem class; only if scope includes frame structures
  - Nishiwaki2000 — historical only; one-sentence citation

- **Reviewer-sensitive comparison points:**
  - Proposed ω₁=159.3 rad/s vs Du2007=174.7 rad/s: ~8.9% lower — requires justification (speedup tradeoff: 7.1× faster)
  - Proposed ω₁=159.3 vs Yuksel2025=160.5: nearly identical — clear differentiation must be on multi-mode/multi-load-case capability and efficiency
  - If proposed method is not significantly faster than Yuksel2025, reviewer will ask: what is the novel contribution?
  - ALRM handling in proposed method must be stated clearly (x_min or equivalent)
