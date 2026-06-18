# CC square-element mesh study assessment

Experiment: `run_cc_square_mesh_study.m`

Tools:

- MATLAB: `/Applications/MATLAB_R2025b.app/bin/matlab`
- Python analysis: `.venv/bin/python`

Constraint observed: the solver and optimizer code were not modified. The run
script copied the audit-baseline settings from `run_cc_meshcompare_b.m` and
changed only `cfg.nelx,cfg.nely`.

Important convention: `rmin_elem=2.5` was held fixed as the existing solver
setting. Since the code defines filter radius in element units, the physical
filter radius decreases with refinement. This is the literal "do not modify
filter radius" interpretation for this code path.

## Results

| mesh | initial omega1 | initial omega2 | initial omega3 | best iter | omega1 best | omega2 best | N best | final omega1 | final omega2 | N final | volume final | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80x10 | 145.968 | 364.269 | 622.685 | 43 | 325.432 | 340.434 | 1 | 292.481 | 384.353 | 1 | 0.4999 | 22.4 s |
| 160x20 | 145.569 | 363.049 | 622.558 | 34 | 329.695 | 420.396 | 1 | 327.041 | 432.049 | 1 | 0.4998 | 50.8 s |
| 320x40 | 145.458 | 362.715 | 622.513 | 54 | 386.738 | 459.371 | 1 | 383.412 | 455.548 | 1 | 0.4999 | 184.9 s |

## Morphology

| mesh | raw 8-components | effective 8-components | connected? | ccSolid | central density | braces | topology class | Fig. 3c similarity |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 80x10 | 5 | 2 | no | 0.033 | 0.209 | yes, weak/fragmentary | disconnected blocks | low, 0.359 |
| 160x20 | 3 | 1 | yes | 0.194 | 0.322 | yes | connected cross-braced truss | high, 0.875 |
| 320x40 | 5 | 1 | yes | 0.249 | 0.300 | yes | connected cross-braced truss | high, 0.891 |

`effective 8-components` ignores tiny islands below 0.5 percent of the mesh
or 10 elements, whichever is larger. Raw components are still reported because
the designs contain small isolated specks, but the structural load path is
dominated by one component for 160x20 and 320x40.

## Convergence table requested

| mesh | connected? | ccSolid | omega1best | omega1final | Nfinal |
|---|---:|---:|---:|---:|---:|
| 80x10 | no | 0.033 | 325.432 | 292.481 | 1 |
| 160x20 | yes | 0.194 | 329.695 | 327.041 | 1 |
| 320x40 | yes | 0.249 | 386.738 | 383.412 | 1 |

## Answers

### A. Does topology converge toward the connected Fig. 3c class?

Yes. The transition occurs between 80x10 and 160x20. The 80x10 design still
contains two effective blocks and a weak central span. Both 160x20 and 320x40
have one effective connected component, central-span material, and diagonal
brace signatures. The topology class therefore converges toward the connected
cross-braced Fig. 3c branch.

### B. Does the disconnected branch disappear, weaken, or remain dominant?

Along the unchanged optimizer trajectory, it weakens and stops dominating once
the mesh reaches 160x20. The 80x10 case still selects the disconnected branch,
but 160x20 and 320x40 select a connected branch. The central-span solid measure
increases from 0.033 to 0.194 to 0.249, which is direct evidence that the
central connection strengthens with resolution.

This does not prove that no disconnected feasible design can have higher
frequency under the aggressive `du2007_c1` mass model. It says that the
unchanged optimizer path no longer settles into the disconnected branch at the
finer square-element meshes.

### C. Does the best frequency trend toward the published value 456.4?

Partially. Best omega1 rises from 325.4 to 329.7 to 386.7, so the trend is
upward, but it is still below 456.4 at 320x40. The second frequency is already
near the paper value at 320x40 (`omega2 best=459.4`, `omega2 final=455.5`), but
the first frequency has not coalesced with it. The frequency trend supports
convergence in the right direction, but not full numerical reproduction.

### D. Does multiplicity emerge naturally at finer resolution?

Not within these 80-iteration runs. All final and best designs report `N=1`.
The 320x40 final pair is closer (`omega1=383.4`, `omega2=455.5`) but still not
bimodal. Multiplicity may require more optimizer progress, different path
control, or an undocumented paper numerical choice; it did not emerge from mesh
resolution alone under the frozen settings.

### E. Is there evidence that 40x5 was too coarse to represent the intended optimum?

Yes. The square-element refinement sequence shows that even 80x10 remains too
coarse for a stable connected central load path, while 160x20 and 320x40 can
represent a connected cross-braced truss. The initial frequencies remain close
to the paper's CC initial value of 146.1 rad/s:

- 80x10: 145.968
- 160x20: 145.569
- 320x40: 145.458

This validates the FE/BC scale to within about 0.5 percent while showing that
topology branch selection is mesh-sensitive.

## Final assessment

1. Topology-convergence conclusion: the connected Fig. 3c morphology is the
   mesh-resolved branch selected by the unchanged optimizer once the mesh is at
   least 160x20. The 80x10 case still disconnects; 160x20 and 320x40 are
   connected cross-braced trusses.

2. Frequency-convergence conclusion: frequency convergence is incomplete.
   Omega1 improves at 320x40 and trends upward, but remains about 16 percent
   below 456.4. Omega2 is already near 456.4, indicating the expected target
   cluster is nearby but not yet coalesced.

3. Estimated probability that Fig. 3c is the mesh-resolved optimum of the paper
   formulation: 0.65. The topology evidence is strong, but the frequency and
   multiplicity evidence is not yet decisive. If "optimum" is interpreted as
   the connected morphology rather than exact eigenvalue/multiplicity, the
   probability is closer to 0.8. If interpreted strictly as `omega1=omega2=456.4`
   under the exact reported algorithm, the current evidence is closer to 0.55.

4. Recommendation: investigate optimizer path before going to much finer
   meshes. A 320x40 run is already feasible and gives the connected branch, so
   pure topology convergence is largely answered. The unresolved issue is why
   omega1 does not coalesce with omega2 by 80 iterations. The next highest-value
   step is not a new fix, but a controlled path/convergence audit on 320x40:
   inspect histories, run longer only if that is classified as the same
   stopping-budget extension, and identify whether the 80-iteration cap is
   preventing multiplicity. If strict "mesh only" must continue, run 640x80
   only after accepting the expected cost increase.
