# SCIENTIFIC_INTERPRETATION (Part 20)

The answers are conservative. Evidence is cited by file. "Native" means the Pedersen/linear-mass model the solver optimizes.

### 1. Does the migrated Pedersen/adaptive implementation retain the smooth nine-mesh behaviour of the upstream source sweep?

**Yes, exactly.** At all nine meshes it reproduces the upstream sweep bit for bit: final ρ, every native and eq. (4) frequency, outer and inner counts, and the complete ω and nested-iteration histories (SOURCE_SWEEP_COMPARISON.md). "Retains" is therefore not an approximation. The migrated production path, run through `performance_comparison.m`, performs the same computation. The qualitative behaviour, including the features below that were not examined in the sweep, is inherited in full.

### 2. Is ω₁ approximately mesh-stable?

**Empirically, yes.**

- From 160x20 to 800x100, ω₁ falls 2.23 % (169.21 → 165.43).
- Over the five finest meshes its range is 0.35 % (0.59 rad/s).
- Adjacent changes from 480x60 upward are ≤ 0.14 %, and 720x90 and 800x100 agree to 0.005 %.

Three caveats apply:

- the sequence is not monotone (320x40 lies 0.36 % below 400x50);
- ω₁ still drifts slightly downward to 720x90;
- at 800x100 ω₁ was still rising slowly when the run stopped (TERMINATION_AUDIT.md).

This is an empirical mesh trend, not proven mesh convergence (MESH_TREND.md).

### 3. Does fine-mesh grayness remain controlled?

**Bounded, but not mesh-invariant.**

- M_nd stays at 0.115–0.141 and the gray fraction at 0.140–0.166 up to 640x80. Both step up by about 22 % at 720x90 (0.162 / 0.186) and hold at 800x100 (0.165 / 0.188).
- That is 1.44× and 1.32× the 160x20 values, inside the preregistered 2× bound.
- At every mesh the final M_nd is the minimum of its own history, so the designs were still clearing gray material when the stop fired.
- The cause of the step at the finest meshes is not established (TOPOLOGY_AUDIT.md).

### 4. Does adaptive-box natural termination remain credible at all nine meshes?

**Formally, yes; beyond that, with limits.**

- All nine runs end NATIVE_CONVERGED on ‖Δρ‖₂ < ε, with no cap, no guard, no nested-MMA failure, and a box that never collapsed to its floor.
- Each stop is the first crossing of a slowly decaying design-change norm, at 0.95–0.996 of ε.
- The largest per-element box stays at its ceiling throughout, so the floor criterion caught nothing.
- The designs are still becoming less gray at the stop, and at 800x100 ω₁ is still rising by about 0.1 % per 20 iterations.

The stop is a consistent heuristic applied at a constant per-element RMS. It is not evidence of a settled optimum (TERMINATION_AUDIT.md).

### 5. Are there any CAP_HIT endpoints?

**No.** The maximum is 246 of 400 outer iterations, and the runner reports `cap_summary.any_cap_hit = false`.

### 6. Are there any localized low-density mode collapses?

**None observed.**

- There are zero sudden ω₁ drops by the existing 0.7 criterion. The largest one-step ω₁ decrease anywhere is 0.71 %.
- At every mesh the terminal mode 1 is the fundamental bending mode, and only 0.3–0.7 % of its kinetic energy lies in the ≈ 41 % of elements with ρ ≤ 0.1.
- The common E1/E2/E3 evaluator selects ordinal 1 as the structural mode at all nine meshes (SPECTRAL_AUDIT.md).

### 7. Does the terminal spectral gap vary systematically with mesh?

**Yes, non-monotonically.** It goes 0.70 % → 11.8 → 17.7 → 19.0 → 22.5 → 24.5 (560x70) → 24.3 → 22.4 → 18.3 % (800x100). The variation comes from ω₂, which rises 21 % to 560x70 and then falls 5.2 %, while ω₁ stays nearly constant. Every run passes through a near-coalescence of ω₁ and ω₂ at outer 8–12. From 240x30 up the pair then separates, and at finer meshes the separation happens later: gap < 0.05 last occurs at outer 31 for 240x30 and at outer 86 for 800x100.

### 8. Is the coarse 160x20 near-multiplicity still exceptional?

**Yes.** 160x20 is the only mesh whose ω₁/ω₂ pair never separates: gap < 0.05 in 114 of 121 iterations, ending at a native gap of 0.70 % (0.65 % in the eq. (4) re-evaluation). It is an effectively bimodal terminal state. Every other mesh ends at a gap of 11.8–24.5 %. 160x20 is also the only fragmented design, with 16 components at the 0.5 threshold, and it has the coarsest filter (1.2 elements).

### 9. Are 720x90 or 800x100 anomalous?

- **In ω₁: no.**
- **In other respects: yes, qualitatively.** The two finest meshes differ from the 400x50–640x80 plateau in several ways:
  - about 22 % more grayness;
  - 204 and 246 outer iterations, against 93–156;
  - longer near-multiplicity phases;
  - falling ω₂ and gap;
  - different layouts: extra gray end-bay braces at 720x90, left–right asymmetry at 800x100.
- **Asymmetry elsewhere.** 320x40 is also strongly asymmetric, and it sits below the ω₁ trend.

None of this violates a preregistered flag, so it is a documented regime change rather than a failure. The designs are bitwise identical to the historical sweep, so these features are properties of the formulation, not artefacts of this campaign.

### 10. Does the formulation provide a defensible production Olhoff baseline for the benchmark comparison?

**Yes, as an observational, reproducible and computationally characterized baseline.** That is the preregistered rule outcome `OLHOFF_PRODUCTION_BASELINE_VALIDATED`. It holds under these explicit qualifications:

- **What it rests on.**
  - frozen identity, verified before, during and after every solve;
  - execution through the real benchmark runner;
  - natural termination at all nine meshes, with no CAP_HIT and no inner failure;
  - an empirically stable ω₁ at the 0.35 % level over the finest five meshes;
  - no localized-mode collapse;
  - exact reproduction of an independent earlier sweep;
  - clean, fully decomposed timing.
- **What it does not rest on and does not show.**
  - KKT stationarity. The stop is heuristic, and the sensitivity-filtered subproblem is not shown to be optimal for the physical problem (HISTORICAL_KKT_CONTEXT.md).
  - Mesh convergence of the topology, grayness, ω₂ or the gap.
  - Insensitivity to ε. The designs were still changing slowly at the stop.
  - Equivalence to Du & Olhoff (2007). This is a distinct, labelled reconstruction.
- **How it must be used.** Cross-method comparisons must name their frequency model: native versus the common evaluator. They must also report the 160x20 bimodality, the fine-mesh grayness step, and the non-monotone outer count alongside any time or scaling claim.
