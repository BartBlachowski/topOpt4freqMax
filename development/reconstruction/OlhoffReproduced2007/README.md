# Olhoff 2007 reproduced method on the Yuksel benchmark cases

This directory provides a small, readable entry point for the Du--Olhoff 2007
method imported under `Matlab/reproduction2007/`, on either of that tree's two
inner solvers -- the Eq. (22) LP route (default) or the paper-literal MMA route.
It solves the same three structural problems exposed by
`analysis/YukselApproach/Matlab`:

- simply supported `8 x 1` beam;
- fixed--pinned `8 x 1` beam;
- `15 x 10` cantilever with a concentrated mass at the right-edge midpoint.

The frozen reproduction tree is not modified or copied.  The implementation
installs that tree through `repro2007_paths`, reuses its element, eigensolver,
mass-interpolation, sensitivity-filter, generalized-gradient, LP, and MMA
routines unmodified, and contains only the forward-model extension needed for
the cantilever and nondesign point mass.

## Entry points

Add `Matlab/` below this directory to the MATLAB path and run one of:

```matlab
run_olhoff_simply_supported
run_olhoff_fixed_pinned
run_olhoff_cantilever
```

The main function is:

```matlab
[rho, omega, info] = topopt_olhoff_reproduced2007( ...
    nelx, nely, volfrac, penal, rmin, move, maxit, bcType, runCfg);
```

`bcType` is `"simply"`, `"fixedPinned"`, or `"cantilever"`.  The runners use
the same meshes, dimensions, volume fractions, filter radii, material data,
and boundary-condition idealizations as the corresponding Yuksel runners.
They retain the Olhoff reproduction's method-specific Eq. (4) mass law,
`rho_min = 1e-3`, the Eq. (22) LP update by default, `move = 0.005`, and
multiplicity tolerance `0.05`.

## Choosing the optimizer

All three runners expose the step-3 inner solver of the Du--Olhoff Fig. 1 loop
through an `optimizer` variable:

| value   | route |
| ------- | ----- |
| `"lp"`  | default; Eq. (22) LP route after Krog & Olhoff (1999), one `linprog` solve per outer iteration |
| `"mma"` | the paper-literal MMA inner loop on problem (25) with the full Eq. (25d) coupling |

Edit the line near the top of the runner, or set the variable before calling it
(the runners `clearvars -except optimizer`, so a preset value survives):

```matlab
optimizer = "mma";
run_olhoff_fixed_pinned
```

`"mma"` is the clean-room study's *labelled baseline*, not its successful
configuration: `NOTES.md` 7 records that it does not converge once `N >= 2`.  It
also costs up to `max_inner = 300` MMA sub-iterates per outer iteration where
the LP route takes one `linprog` solve, so the iteration caps in the runners --
which were chosen for the LP route -- are far more expensive on it.  Measured on
the fixed--pinned `320 x 40` runner: about 110 sub-iterates and 33 s per outer
iteration, against 1.3 s for LP.

Through the function interface the same switch is `runCfg.optimizer`, alongside
`off_diag`, `max_inner`, `tol_inner`, and `min_inner`:

```matlab
runCfg = struct('optimizer','mma','max_inner',100);
```

`off_diag` selects the MMA constraint set (`true`, the default, is the full
Eq. (25d) coupling; `false` imposes the Eq. (22) off-diagonal conditions as
inequality pairs).  It is rejected on the LP route, where Eq. (22) is built into
`innerLoopLP`.  `info.optimizer`, `info.route`, and `info.source.method` record
which route ran.

This is the constant-move clean-room reproduction, not the later S1
stabilization profile in `analysis/olhoff_stabilization_audit/`.
