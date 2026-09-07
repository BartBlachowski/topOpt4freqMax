"""ma4_traj -- read the retained element-level trajectory.

MAT v7.3 is HDF5.  MATLAB writes arrays column-major, so an NE x nOuter MATLAB
matrix appears to h5py with shape (nOuter, NE); every accessor here returns the
MATLAB orientation so callers think in (element, iteration) as the solver does.
"""
import os
import numpy as np
import h5py

REPO = '/Users/piotrek/Programming/topOpt4freqMax'
EVID = os.path.join(REPO, 'analysis/OlhoffCurrent/evidence/move_activity_400')


def path(arm, nelx=400, nely=50):
    return os.path.join(EVID, f'{arm}400_{nelx}x{nely}_trajectory.mat')


class Traj:
    """RHO/DRHO as (NE, nOuter), plus move and the scalar histories."""

    def __init__(self, arm, nelx=400, nely=50):
        self.arm, self.nelx, self.nely = arm, nelx, nely
        self.NE = nelx * nely
        self.f = h5py.File(path(arm, nelx, nely), 'r')
        self.nOuter = self.f['RHO'].shape[0]
        self.move = np.array(self.f['move']).ravel()

    def rho(self, k):
        """Physical density at 1-based outer iteration k, shape (NE,)."""
        return np.array(self.f['RHO'][k - 1, :])

    def drho(self, k):
        """Applied increment at 1-based outer iteration k, shape (NE,)."""
        return np.array(self.f['DRHO'][k - 1, :])

    def u(self, k):
        """Utilisation u_e(k) = |drho_e(k)| / move(k)  -- move(k), see METRICS."""
        return np.abs(self.drho(k)) / self.move[k - 1]

    def grid(self, v):
        """Element vector -> (nely, nelx) image, matching MATLAB reshape order."""
        return np.asarray(v).reshape(self.nelx, self.nely).T

    def hist(self, name):
        return np.array(self.f['hist'][name]).ravel()

    def close(self):
        self.f.close()
