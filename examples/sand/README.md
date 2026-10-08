# examples/sand — reproduction runners for Toebat & Feppon (2026)

*Local constraints in topology optimization: a Simultaneous Analysis and Design (SAND)
approach*, Struct. Multidisc. Optim. 69:212, `docs/s00158-026-04412-9.pdf`.
Heat sink on the unit square, P1 density and temperature, minimize volume subject to
T ≤ Tmax = 300 °C at every node, solved with the Null Space Optimizer.

The authors' reference implementation (`tools/SAND/ex13_heat_SAND.py`, "SAND exact") is
used unmodified; everything else here is a thin layer on top of it.

## Paper ↔ runner map

| paper item | runner | note |
|---|---|---|
| Fig. 4/5, Table 2 row "SAND exact" | `run_sand_exact.py` | authors' code as is |
| Fig. 6/7, Table 2 rows "SAND-ε …" | `run_sand_exact.py --alpha eps`, `run_strategies_comparison.py` | α = 1e-8, Eq. (4.12) |
| Table 2 (all strategies), Fig. 4–7 | `run_strategies_comparison.py` | **"NAND exact" excluded**, see below |
| Table 3 (p-norm sweep) | `run_pnorm_sweep.py` | NAND aggregation, p ∈ {1,10,20,30,40,50} |
| Table 4, Fig. 10–13 (QP solvers) | `run_qp_solver_comparison.py` | OSQP, QPALM, PIQP here; MOSEK/Gurobi/CPLEX auto-added if importable |
| Fig. 8/9 (metric weight α) | `run_alpha_sweep.py` | the paper does not list its α values |
| Table 5, Fig. 14–16 (200×200) | any runner with `--N 200` | ~10× the 100×100 cost |
| — | `run_check_derivatives.py` | FD check of every Jacobian, run this first |
| Fig. 4/5-style side-by-side panels for any group of finished cases | `plot_comparison.py <group_dir>` | writes `fig_designs.png`, `fig_histories.png` |

Paper settings are the defaults (Sect. 4.2): `dt=0.05`, `alphaJ=alphaC=1`,
`itnormalisation=50`, `maxit=500`, `K=0.01`, QPALM, `tol_qp=1e-8`, 100×100 mesh.
`--quick` switches to 30×30 / 20 iterations for a smoke test.

## Setup

```bash
bash tools/SAND/setup_sand_env.sh --freefem     # Python stack into .venv + FreeFEM 4.15 (macOS arm64)
.venv/bin/python examples/sand/run_check_derivatives.py
.venv/bin/python examples/sand/run_sand_exact.py --quick
```

`sand_env.py` finds `FreeFem++` under `/Applications/FreeFem++.app` automatically, or via
`FREEFEM_BIN`. Outputs go to `examples/sand/results/<runner>/<tag>/` (git-ignored):
`summary.json`, `history.csv`, `rho.npy`, `T.npy`, `design.png`, `history.png`, plus a
`table*.csv/.md` in the paper's layout for the multi-case runners.

## Cost on this machine (Apple Silicon, 100×100, n = 10 201)

| QP solver | s / iteration | 500 iterations |
|---|---|---|
| PIQP | 8.6 | ≈ 1.2 h |
| OSQP | 12.6 | ≈ 1.8 h |
| QPALM (paper default) | 17.7 | ≈ 2.5 h |

Paper (Xeon Gold 6240, Table 2/4): 28 s/iter QPALM, 15 s PIQP, 23 s OSQP. NAND aggregation
is roughly 10× cheaper per iteration. 200×200 took the authors 49 h for SAND exact.

## Deviations and caveats

* **Initial design.** The paper says ρ₀ is uniform with max T exactly Tmax; the authors'
  shipped script uses ρ₀ = 0.4 (max T ≈ 1.05 Tmax at 30×30). Default `--init script`
  reproduces the shipped code; `--init paper` bisects the uniform density to hit Tmax.
* **maxit.** The shipped script stops at 150; the paper and these runners use 500.
* **Aggregation strategies are re-implemented** (not shipped upstream): SAND aggregation
  (Eq. 4.7/4.8) and NAND aggregation with the adjoint (Eq. 4.5/4.6), on the authors' FE
  spaces (κ in P3, κ′ in P2). Verified by `run_check_derivatives.py`; not verified against
  the authors' own numbers.
* **"NAND exact" is not provided.** It needs n sparse solves and dense n×n Jacobians per
  iteration (16 GB and 57 s/iter at 100×100 in the paper; infeasible at 200×200). The paper
  itself substitutes "SAND-ε exact" for it, which `--alpha eps` gives.
* **pypardiso** is a scipy shim here (no MKL on arm64). The optimizer only uses it for the
  `linear_system` range-space method; the paper uses `method_xiC="qp"`, so no effect.
* Results are **QP-solver dependent** (paper Table 4, Fig. 10) and the paper notes
  asymmetric round-off; expect designs to match qualitatively, numbers to a few percent.
