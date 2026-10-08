"""Shim for `pypardiso` on platforms without Intel MKL (e.g. Apple Silicon).

nullspace_optimizer 1.3.0 imports `pypardiso` unconditionally in
optimizers/nullspace/utils.py and calls `pypardiso.spsolve(G, rhs)` for the
KKT linear systems of the null-space / range-space steps.  MKL (and therefore
the real pypardiso) is not available for arm64 macOS, so this module provides
the same entry point backed by scipy's SuperLU.  The optimizer already falls
back to LSQR when spsolve raises, so behaviour on singular systems is
unchanged.  Installed by tools/SAND/setup_sand_env.sh.
"""
import numpy as _np
import scipy.sparse as _sp
import scipy.sparse.linalg as _spla

__version__ = "0.0-scipy-shim"


def spsolve(A, b, *args, **kwargs):
    A = _sp.csc_matrix(A)
    x = _spla.spsolve(A, b)
    x = _np.asarray(x)
    if _np.isnan(x).any():
        raise RuntimeError("scipy spsolve returned NaN (singular matrix?)")
    return x


class PyPardisoSolver:  # minimal API surface, in case anything instantiates it
    def __init__(self, *a, **k):
        pass

    def solve(self, A, b):
        return spsolve(A, b)

    def factorize(self, A):
        self._lu = _spla.splu(_sp.csc_matrix(A))
        return self

    def free_memory(self, *a, **k):
        pass
