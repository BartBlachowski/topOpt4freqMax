# tools/SAND

Upstream material for the SAND heat-sink paper (Toebat & Feppon, SMO 69:212, 2026,
`docs/s00158-026-04412-9.pdf`).

| file | provenance |
|---|---|
| `ex13_heat_SAND.py` | byte-identical copy of `nullspace_optimizer/examples/topopt_examples/ex13_heat_SAND.py`, GitLab `florian.feppon/null-space-optimizer`, branch `public-master`, package 1.3.0 (last upstream fix 2026-05-07). **Do not edit**: the runners import it. |
| `setup_sand_env.sh` | installs the Python stack into `.venv` and, with `--freefem`, FreeFEM 4.15 on macOS arm64 |
| `pypardiso_shim.py` | scipy-backed stand-in for `pypardiso` (needs Intel MKL, unavailable on arm64) |

Runners that reproduce the paper's tables/figures live in `examples/sand/`.
