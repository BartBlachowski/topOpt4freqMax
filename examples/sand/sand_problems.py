"""Optimization problems of Toebat & Feppon (2026), SMO 69:212, Sect. 4.1.

Three of the paper's four strategies are provided, all on the same P1 heat-sink
discretization produced by the authors' code (``tools/SAND/ex13_heat_SAND.py``):

* ``HeatSANDExact``       -- Sect. 4.1.4 "SAND exact".  A thin subclass of the
                             authors' ``Heat_TO_SAND`` that only exposes the metric
                             weight alpha of Eq. (4.10): the authors' balanced choice
                             Eq. (4.11) (default), the degenerate choice Eq. (4.12)
                             (alpha = eps = 1e-8, "SAND-eps"), or any number.
* ``HeatSANDAggregation`` -- Sect. 4.1.3 "SAND aggregation": the nodal temperature
                             constraints are replaced by the single p-norm (4.7).
* ``HeatNANDAggregation`` -- Sect. 4.1.2 "NAND aggregation": density-only problem
                             (4.4) with the adjoint sensitivity (4.5)-(4.6).

"NAND exact" (Sect. 4.1.1) is deliberately NOT provided: it needs n linear solves
per iteration and dense n x n constraint Jacobians (16 GB at 100x100 in the paper,
infeasible at 200x200), and the paper itself replaces it by "SAND-eps exact".

Rescaling convention (paper Sect. 4.1.3, authors' code): the heat source is Q/Tmax
so the stored temperature is T~ = T/Tmax and every temperature bound reads <= 1.
Physical temperatures are recovered by dividing by ``problem.rescale``.

The two aggregation classes are new code (not shipped by the authors).  Their
Jacobians are verified by ``run_check_derivatives.py``.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from pyfreefem import FreeFemRunner
from nullspace_optimizer import Optimizable, memoize

import sand_env  # noqa: F401  (adds tools/SAND to sys.path)
from ex13_heat_SAND import (  # the authors' reference implementation, unmodified
    CONST_KAPPA_F, CONST_KAPPA_S, CONST_Q, SIMP_P,
    Heat_TO_SAND, init_filter_matrix, init_mesh, retract_state, solve_state,
)

EPS_DEGENERATE = 1e-8          # Eq. (4.12), value stated in Sect. 4.1.3
PNORM_DEFAULT = 10             # Sect. 4.1.2: "we set p = 10 unless stated otherwise"


# --------------------------------------------------------------------------- #
#  Initial design
# --------------------------------------------------------------------------- #
def max_temperature_uniform(Th, vfrac: float) -> float:
    """Physical max T for a uniform density ``vfrac`` (unscaled source Q)."""
    T = retract_state(Th, vfrac * np.ones(Th.nv, dtype=float))
    return float(np.max(T))


def uniform_density_for_Tmax(Th, maxT: float, lo=1e-3, hi=1.0, iters=40, tol=1e-6) -> float:
    """Sect. 4.2: 'The initial guess rho0 is a uniform density that is determined so
    that the maximum temperature is exactly Tmax'.  Bisection on the volume fraction;
    max T decreases monotonically with rho because kappa(rho) is increasing."""
    f_lo = max_temperature_uniform(Th, lo) - maxT
    f_hi = max_temperature_uniform(Th, hi) - maxT
    if f_hi > 0:
        raise ValueError(f"Even a full design exceeds Tmax={maxT}: max T = {f_hi + maxT:.2f}")
    if f_lo < 0:
        return lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = max_temperature_uniform(Th, mid) - maxT
        if abs(f_mid) < tol * maxT:
            return mid
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
#  p-norm aggregation pieces (FreeFEM), shared by SAND and NAND aggregation
# --------------------------------------------------------------------------- #
@memoize()
def pnorm_of_state(Th, T, pn: int):
    """h(T) = (int_D T^p dx)^(1/p) and its Frechet derivative row D_T h, Eq. (4.8)."""
    script = """
    IMPORT "io.edp"
    mesh Th = importMesh("Th");
    fespace Fh1(Th,P1);
    Fh1 T; T[] = importArray("T");
    real pn = $pn;
    real hp = int2d(Th)(T^pn);
    real h  = hp^(1./pn);
    varf vDh(dummy, dT) = int2d(Th)(h^(1-pn)*T^(pn-1)*dT);
    real[int] Dh = vDh(0, Fh1);
    exportVar(h);
    exportArray(Dh);
    """
    runner = FreeFemRunner(script)
    runner.import_variables(Th=Th, T=T)
    return runner.execute({"pn": pn})


@memoize()
def solve_state_nand_aggregation(Th, rho, pn: int, rescale: float):
    """NAND aggregation: state T(rho), cost, p-norm h and its adjoint sensitivity.

    Implements Eqs. (4.5)-(4.6) of the paper with the same FE spaces the authors use
    for kappa (P3) and kappa' (P2) in ``ex13_heat_SAND.solve_state``.
    """
    script = """
    IMPORT "io.edp"
    load "Element_P3"
    mesh Th = importMesh("Th");
    fespace Fh1(Th,P1);
    fespace Fh2(Th,P2);
    fespace Fh3(Th,P3);

    Fh1 rho; rho[] = importArray("rho");
    real p = $p;
    real kappaf = $kappaf;
    real kappas = $kappas;
    real pn = $pn;
    Fh3 kappa  = rho^p*(kappaf-kappas)+kappas;
    Fh2 dkappa = p*rho^(p-1)*(kappaf-kappas);
    func Q = $Q;
    macro grad(u) [dx(u),dy(u)] //

    // state (3.4)
    Fh1 T, S, R;
    solve heat(T,S) = int2d(Th)(kappa*grad(T)'*grad(S)) - int2d(Th)(Q*S) + on(5,T=0);

    // aggregation (4.4)
    real hp = int2d(Th)(T^pn);
    real h  = hp^(1./pn);

    // adjoint (4.6):  int kappa grad S . grad r = int T^(p-1) r
    solve adj(S,R) = int2d(Th)(kappa*grad(S)'*grad(R)) - int2d(Th)(T^(pn-1)*R) + on(5,S=0);

    // cost and derivatives
    real vol0 = int2d(Th)(1.);
    real J = int2d(Th)(rho/vol0);
    varf vDJ(dummy, drho) = int2d(Th)(drho/vol0);
    real[int] DJ = vDJ(0, Fh1);

    // (4.5):  Dh . drho = - h^(1-p) int kappa'(rho) grad T . grad S  drho
    varf vDh(dummy, drho) = int2d(Th)(-h^(1-pn)*dkappa*(grad(T)'*grad(S))*drho);
    real[int] Dh = vDh(0, Fh1);

    exportVar(J);
    exportVar(h);
    exportArray(T[]);
    exportArray(DJ);
    exportArray(Dh);
    """
    runner = FreeFemRunner(script)
    runner.import_variables(Th=Th, rho=rho)
    return runner.execute({"p": SIMP_P, "Q": CONST_Q * rescale, "kappaf": CONST_KAPPA_F,
                           "kappas": CONST_KAPPA_S, "pn": pn})


# --------------------------------------------------------------------------- #
#  SAND exact  (authors' class + explicit alpha)
# --------------------------------------------------------------------------- #
class HeatSANDExact(Heat_TO_SAND):
    """Sect. 4.1.4.  ``alpha``: 'balanced' (Eq. 4.11, authors' default), 'eps'
    (Eq. 4.12, alpha = 1e-8, the paper's "SAND-eps"), or a float used verbatim as the
    weight of the identity block acting on the rescaled temperature T/Tmax.

    ``init``: 'script' keeps the authors' uniform rho0 = 0.4; 'paper' bisects the
    uniform density so that max T = Tmax exactly (Sect. 4.2); a float is used as is.
    """

    label = "SAND exact"

    def __init__(self, n=100, alpha="balanced", init="script", gamma=2, maxT=300.0, plot=False):
        vfrac0 = 0.4 if init == "script" else init
        if init == "paper":
            Th_tmp = init_mesh(n)
            vfrac0 = uniform_density_for_Tmax(Th_tmp, maxT)
        super().__init__(n=n, vfrac0=vfrac0, gamma=gamma, maxT=maxT, plot=plot)
        self.n = n
        self.vfrac0 = float(vfrac0)
        self.alpha_balanced = float(sp.linalg.norm(self.rhoFilter) / np.sqrt(self.Th.nv))  # Eq. (4.11)
        if alpha == "balanced":
            self.alpha = self.alpha_balanced
        elif alpha in ("eps", "epsilon", "degenerate"):
            self.alpha = EPS_DEGENERATE
        else:
            self.alpha = float(alpha)
        I_nv = sp.eye(self.Th.nv, format="csc")
        self.IP = sp.block_diag((self.rhoFilter, I_nv * self.alpha), format="csc")  # Eq. (4.10)

    # ---- diagnostics shared by the runners --------------------------------
    def split(self, x):
        return x[: self.nbRho], x[self.nbRho:]

    def physical_T(self, x):
        return self.split(x)[1] / self.rescale

    def state_residual(self, x):
        return float(np.linalg.norm(self.G(x)))

    def max_T_violation(self, x):
        """max(T - Tmax) in physical units, Fig. 5/8/12/15/16 right panels."""
        return float(np.max(self.physical_T(x)) - self.maxT)

    def aggregation_value(self, x):
        return None

    def accept(self, params, results):
        super().accept(params, results)
        x = results["x"][-1]
        results.setdefault("maxT_minus_Tmax", []).append(self.max_T_violation(x))
        results.setdefault("state_residual", []).append(self.state_residual(x))

    def describe(self):
        return dict(strategy=self.label, n_vertices=int(self.Th.nv), N=self.n, maxT=self.maxT,
                    alpha=self.alpha, alpha_balanced=self.alpha_balanced, vfrac0=self.vfrac0,
                    n_variables=int(2 * self.Th.nv), n_equalities=int(self.Th.nv),
                    n_inequalities=int(3 * self.Th.nv))


# --------------------------------------------------------------------------- #
#  SAND aggregation  (4.7)
# --------------------------------------------------------------------------- #
class HeatSANDAggregation(HeatSANDExact):
    """Sect. 4.1.3.  Same variables (rho, T) and state constraint as SAND exact, but the
    n nodal bounds are replaced by one p-norm constraint h(T/Tmax) <= 1."""

    label = "SAND aggregation"

    def __init__(self, n=100, alpha="balanced", init="script", pnorm=PNORM_DEFAULT, **kw):
        super().__init__(n=n, alpha=alpha, init=init, **kw)
        self.pnorm = int(pnorm)

    def _agg(self, x):
        return pnorm_of_state(self.Th, self.split(x)[1], self.pnorm)

    def H(self, x):
        rho = self.split(x)[0]
        return np.concatenate((-rho, rho - 1.0, [self._agg(x)["h"] - 1.0]))

    def dH(self, x):
        I = sp.eye(self.nbRho, format="csc")
        Dh_T = sp.csr_matrix(np.asarray(self._agg(x)["Dh"], dtype=float)[None, :])
        DHr = sp.vstack((-I, I, sp.csr_matrix((1, self.nbRho))))
        DHT = sp.vstack((sp.csr_matrix((2 * self.nbRho, self.nbT)), Dh_T))
        return sp.hstack((DHr, DHT), format="csc")

    def aggregation_value(self, x):
        """h - 1 in rescaled units (what the optimizer sees, Table 2 column 'h')."""
        return float(self._agg(x)["h"] - 1.0)

    def accept(self, params, results):
        # Do NOT call Heat_TO_SAND.accept: it indexes H[2n:] as nodal temperatures.
        if self.plot:
            self.show_state(results["x"][-1], fig_title="Iteration " + str(results["it"][-1]))
        params["normalisation_norm"] = lambda rhoT: np.linalg.norm(rhoT[: self.nbRho], np.inf)
        x = results["x"][-1]
        results.setdefault("maxT_minus_Tmax", []).append(self.max_T_violation(x))
        results.setdefault("state_residual", []).append(self.state_residual(x))
        results.setdefault("aggregation", []).append(self.aggregation_value(x))

    def describe(self):
        d = super().describe()
        d.update(pnorm=self.pnorm, n_inequalities=int(2 * self.Th.nv + 1))
        return d


# --------------------------------------------------------------------------- #
#  NAND aggregation  (4.4) with adjoint (4.5)-(4.6)
# --------------------------------------------------------------------------- #
class HeatNANDAggregation(Optimizable):
    """Sect. 4.1.2.  Design variable rho only; T(rho) is solved inside every evaluation.
    Metric A^NAND = Helmholtz matrix (3.5), identical to the rho-block of the SAND metric."""

    label = "NAND aggregation"

    def __init__(self, n=100, init="script", pnorm=PNORM_DEFAULT, gamma=2, maxT=300.0, plot=False):
        self.n = n
        self.maxT = float(maxT)
        self.pnorm = int(pnorm)
        self.plot = plot
        self.Th = init_mesh(n)
        self.nbRho = self.Th.nv
        self.rhoFilter = init_filter_matrix(self.Th, gamma)
        self.rescale = 1.0 / self.maxT
        if init == "paper":
            vfrac0 = uniform_density_for_Tmax(self.Th, self.maxT)
        elif init == "script":
            vfrac0 = 0.4
        else:
            vfrac0 = float(init)
        self.vfrac0 = float(vfrac0)
        self.rho0 = self.vfrac0 * np.ones(self.nbRho, dtype=float)
        self.alpha = None
        self.alpha_balanced = None

    def solve(self, rho):
        return solve_state_nand_aggregation(self.Th, rho, self.pnorm, self.rescale)

    # ---- Optimizable interface --------------------------------------------
    def x0(self):
        return self.rho0

    def J(self, rho):
        return self.solve(rho)["J"]

    def H(self, rho):
        return np.concatenate((-rho, rho - 1.0, [self.solve(rho)["h"] - 1.0]))

    def dJ(self, rho):
        return np.asarray(self.solve(rho)["DJ"], dtype=float)

    def dH(self, rho):
        I = sp.eye(self.nbRho, format="csc")
        Dh = sp.csr_matrix(np.asarray(self.solve(rho)["Dh"], dtype=float)[None, :])
        return sp.vstack((-I, I, Dh), format="csc")

    def inner_product(self, rho):
        return self.rhoFilter

    def retract(self, rho, drho):
        return np.clip(rho + drho, 0.0, 1.0)

    def accept(self, params, results):
        rho = results["x"][-1]
        results.setdefault("maxT_minus_Tmax", []).append(self.max_T_violation(rho))
        results.setdefault("aggregation", []).append(self.aggregation_value(rho))
        if self.plot:
            import matplotlib.pyplot as plt
            from pymedit import P1Function
            fig, ax = plt.subplots(1, 2)
            P1Function(self.Th, rho).plot(fig=fig, ax=ax[0], cmap="gray_r", vmin=0, vmax=1)
            P1Function(self.Th, self.physical_T(rho)).plot(fig=fig, ax=ax[1], cmap="turbo", type_plot="tricontourf")
            plt.pause(0.1)

    # ---- diagnostics shared by the runners --------------------------------
    def split(self, rho):
        return rho, np.asarray(self.solve(rho)["T[]"], dtype=float)

    def physical_T(self, rho):
        return self.split(rho)[1] / self.rescale

    def state_residual(self, rho):
        return 0.0  # the state is solved exactly inside every evaluation

    def max_T_violation(self, rho):
        return float(np.max(self.physical_T(rho)) - self.maxT)

    def aggregation_value(self, rho):
        return float(self.solve(rho)["h"] - 1.0)

    def describe(self):
        return dict(strategy=self.label, n_vertices=int(self.Th.nv), N=self.n, maxT=self.maxT,
                    alpha=None, alpha_balanced=None, vfrac0=self.vfrac0, pnorm=self.pnorm,
                    n_variables=int(self.Th.nv), n_equalities=0, n_inequalities=int(2 * self.Th.nv + 1))


# --------------------------------------------------------------------------- #
#  Factory used by the runners
# --------------------------------------------------------------------------- #
STRATEGIES = {
    "sand_exact":            dict(cls=HeatSANDExact,       alpha="balanced", paper_name="SAND exact"),
    "sand_eps_exact":        dict(cls=HeatSANDExact,       alpha="eps",      paper_name="SAND-eps exact"),
    "sand_aggregation":      dict(cls=HeatSANDAggregation, alpha="balanced", paper_name="SAND aggregation"),
    "sand_eps_aggregation":  dict(cls=HeatSANDAggregation, alpha="eps",      paper_name="SAND-eps aggregation"),
    "nand_aggregation":      dict(cls=HeatNANDAggregation, alpha=None,       paper_name="NAND aggregation"),
}


def make_problem(strategy: str, n: int, init="script", pnorm=PNORM_DEFAULT, alpha=None, maxT=300.0):
    spec = STRATEGIES[strategy]
    cls = spec["cls"]
    if cls is HeatNANDAggregation:
        return cls(n=n, init=init, pnorm=pnorm, maxT=maxT)
    a = spec["alpha"] if alpha is None else alpha
    if cls is HeatSANDAggregation:
        return cls(n=n, alpha=a, init=init, pnorm=pnorm, maxT=maxT)
    return cls(n=n, alpha=a, init=init, maxT=maxT)
