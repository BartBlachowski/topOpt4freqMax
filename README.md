# Frequency maximization of planar structures: code for the paper

This repository reproduces every table and figure of

> P. Tauzowski, B. Błachowski, *Frequency Maximization of Planar Structures Using
> Quasi-Static Approximation in Topology Optimization*, Applied Sciences (revised version).

It contains three topology-optimization methods for the fundamental eigenfrequency of
2D plane-stress structures, and one script for each result of the paper:

| Method | File | In the paper |
| --- | --- | --- |
| Proposed quasi-static method | `solvers/proposed/proposed_freq.m` | Section 2.2, Algorithm 3 |
| Yuksel & Yilmaz (2025), two static stages | `solvers/yuksel/yuksel_freq.m` | Section 2.1, Algorithm 2 |
| Du & Olhoff (2007), bound formulation | `solvers/du_olhoff/du_olhoff.m` | Section 2.1, Algorithm 1, Section 3.1 |

## Quick start

MATLAB R2025b (R2023b or newer should work) with the Image Processing Toolbox (used by
the Yuksel–Yilmaz filter). From the repository root:

```matlab
setup_paths
run_all(true)          % every script on coarse meshes, about 25 min
```

`run_all` without arguments reproduces everything at the paper's resolution; this takes
many hours because the Du–Olhoff runs of Tables 2 and 4 dominate. Each script can be run
on its own, and most accept `true` for a quick check. Results are written to `results/`.

## Which script reproduces what

| Paper item | Script (`paper/`) | Full run time* |
| --- | --- | --- |
| Figures 1–4: steps of the method | `fig1_4_method_steps` | < 1 min |
| Section 2.3: λ̂ versus ω₁, modal mass m_v | `sec2_3_spectral_check` | 3 min |
| Table 1: load sensitivity, optimizer, volume constraint | `table1_gradient_ablation` | 5 min |
| Table 2, Figures 5–8: nine-mesh benchmark | `table2_fig8_benchmark` | 2.5 h |
| Tables 3–4: native versus matched settings | `tables3_4_matched_settings` | 2 h |
| Section 4.1: tolerance 10⁻³ and cost breakdown | `sec4_1_tolerance_check` | 20 min |
| Figure 9: frequency histories (240×30) | `fig9_frequency_histories` | 3 min |
| Tables 5–8, Figures 10–11: load weighting α | `tables5_8_fig10_11_alpha_sweep` | 5 min |
| Section 4.3: choice of the reference mode | `sec4_3_reference_mode_study` | 1 min |
| Table 9, Figures 12–13: comparison with Du–Olhoff | `table9_fig12_13_direct_comparison` | 45 min |
| Section 3.2: finite-difference check of Eq. (13) | `checks/check_full_derivative_fd` | 1 min |
| Section 4.2: sign invariance of Eq. (14) | `checks/check_sign_invariance` | 1 min |
| Sections 4.1–4.2: sensitivity of the mode selection | `checks/check_mode_selection` | 15 min |
| Section 3.1: Du–Olhoff against the published benchmark | `checks/check_du_olhoff_reference` | 3 min |

\*Single thread on an Apple-silicon laptop. Wall times depend on the machine; iteration
counts and frequencies do not. Scripts that compare with a printed value report the
difference, for example `gap ss_beam_400x50_vf50  -4.08 (paper -4.1 %)`.

## Layout

```
setup_paths.m        adds the folders below to the MATLAB path
configs/             the three examples: problem_ss_beam, problem_clamped_beam, problem_building
solvers/proposed/    proposed method (MMA only for the Table 1 ablation)
solvers/yuksel/      Yuksel–Yilmaz method
solvers/du_olhoff/   Du–Olhoff bound formulation (with Svanberg's MMA, 2007 version)
solvers/common/      Q4 mesh, assembly, stagnation_x stop, common modal evaluator, MAC, plots
paper/               one script per table or figure; paper/checks: verification scripts
tests/               comparison of these solvers with the research code
```

## Conventions shared by all scripts

- **Elements** are numbered column by column from the bottom-left corner
  (`e = ely + elx*nely + 1`). The Du–Olhoff and Yuksel–Yilmaz solvers number internally
  from the top (as in top88/top99neo) and return their designs in this common numbering.
- **Stopping rule**: `stagnation_x` of Section 2.5 with W = 10 and ε_x = ε_M = 10⁻²
  (`solvers/common/stagnation_x.m`) for every method.
- **Reported frequencies** are the lowest *structural* frequencies of the final design
  under the common evaluator (`solvers/common/structural_modes.m`): SIMP stiffness with
  E_min = 10⁻⁶E₀, linear mass, modes localized in elements with x ≤ 0.1 excluded
  (Section 2.4).

## Relation to the research code

The results in the paper were computed with the research repository at tag
`applsci-revision-1` (branch `revision-applsci`), which also holds the run logs. The
solvers here are compact versions of the same algorithms; `tests/test_ports_against_research_code.m`
compares them with that tag: the Du–Olhoff solver (simply supported beam, clamped beam,
building) and the Yuksel–Yilmaz solver give bit-for-bit the same designs and iteration
counts; the proposed solver gives the same iteration counts and designs within 1.1·10⁻⁸
(the research code assembles the stiffness from the lower triangle only).

## References

- J. Du, N. Olhoff, Struct. Multidisc. Optim. 34 (2007) 91–110, doi:10.1007/s00158-007-0101-y
- O. Yuksel, C. Yilmaz, Engineering Computations 42 (2025) 3194–3214, doi:10.1108/EC-03-2025-0278
- K. Svanberg, Int. J. Numer. Meth. Eng. 24 (1987) 359–373 (MMA)
- N. L. Pedersen, Struct. Multidisc. Optim. 20 (2000) 2–11
- O. Sigmund, Mech. Struct. Mach. 25 (1997) 493–524 (sensitivity filter)
