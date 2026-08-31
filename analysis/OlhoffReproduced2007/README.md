# Olhoff 2007 reproduced method on the Yuksel benchmark cases

This directory provides a small, readable entry point for the Du--Olhoff 2007
Eq. (22) LP method imported under `Matlab/reproduction2007/`.  It solves the
same three structural problems exposed by `analysis/YukselApproach/Matlab`:

- simply supported `8 x 1` beam;
- fixed--pinned `8 x 1` beam;
- `15 x 10` cantilever with a concentrated mass at the right-edge midpoint.

The frozen reproduction tree is not modified or copied.  The implementation
installs that tree through `repro2007_paths`, reuses its element, eigensolver,
mass-interpolation, sensitivity-filter, generalized-gradient, and LP routines,
and contains only the forward-model extension needed for the cantilever and
nondesign point mass.

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
`rho_min = 1e-3`, Eq. (22) LP update, `move = 0.005`, and multiplicity tolerance
`0.05`.

This is the constant-move clean-room reproduction, not the later S1
stabilization profile in `analysis/olhoff_stabilization_audit/`.
