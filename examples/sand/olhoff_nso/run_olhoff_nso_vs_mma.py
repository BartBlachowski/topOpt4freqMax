#!/usr/bin/env python
"""A/B: Null Space Optimizer vs MMA on the Du-Olhoff fundamental-frequency problem.

Same FE model, same sensitivities, same filter as analysis/Olhoff's production preset
(duOlhoffPedersenAdaptiveBoxSensitivityFiltered, see results/olhoff_ref_160x20_cfg.json);
only the optimizer differs:

  --optimizer nso   Null Space Optimizer (nullspace_optimizer.nlspace_solve), single loop,
                    bound formulation with Jc eigenvalue constraints, box and volume as
                    inequality constraints, optional Krog-Olhoff off-diagonal equalities.
  --optimizer mma   single-loop MMA (the package's Svanberg port) on the same class.

The MATLAB production solver (nested MMA with the (25d) sub-eigenvalue coupling, adaptive
per-element move box) is the third arm; its history is produced by
results/olhoff_ref_<mesh>.mat (scratchpad/run_olhoff_ref.m) and merged by compare_olhoff_nso.py.

    python run_olhoff_nso_vs_mma.py --optimizer nso --dt 0.02
    python run_olhoff_nso_vs_mma.py --optimizer nso --dt 0.02 --offdiag
    python run_olhoff_nso_vs_mma.py --optimizer mma --move 0.1
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")          # MATLAB arm runs single-threaded
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from olhoff_fe import OlhoffBeam
from olhoff_problem import OlhoffFreqProblem

PRODUCTION = dict(a=8.0, b=1.0, t=1.0, E=1e7, nu=0.3, rhom=1.0, bc="a", support="mid", axial="both",
                  stiffness="pedersen", p=3.0, linear_below=0.1, mass="lin",
                  rmin_phys=0.06, rhomin=1e-3)
VOLFRAC, RHO0 = 0.5, 0.5


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--optimizer", choices=["nso", "mma"], default="nso")
    ap.add_argument("--nelx", type=int, default=160); ap.add_argument("--nely", type=int, default=20)
    ap.add_argument("--Jc", type=int, default=4, help="eigenvalue constraints carried (eigen.maxCluster = 4)")
    ap.add_argument("--maxit", type=int, default=400, help="Olhoff's runtime.maxOuter")
    ap.add_argument("--dt", type=float, default=0.02, help="NSO step: max|drho| per iteration while normalized")
    ap.add_argument("--itnorm", type=int, default=50, help="NSO itnormalisation (paper: 50); -1 = always")
    ap.add_argument("--K", type=float, default=0.1)
    ap.add_argument("--qp-solver", default="piqp")
    ap.add_argument("--metric", choices=["euclid", "helmholtz"], default="euclid")
    ap.add_argument("--no-filter", action="store_true", help="drop the Sigmund sensitivity filter")
    ap.add_argument("--offdiag", action="store_true", help="Krog-Olhoff f_12' xi = 0 equality on the fixed N=2 subspace (production: N fixed at 2)")
    ap.add_argument("--gap-tol", type=float, default=0.05, help="multiplicity.tolerance of the production preset")
    ap.add_argument("--move", type=float, default=0.1, help="MMA move limit (production adaptive box starts at 0.1)")
    ap.add_argument("--init-rho", default=None, help="warm start: .npy density, or 'ref' = results/olhoff_ref_<mesh>.mat final design")
    ap.add_argument("--stop-omega1", type=float, default=None, help="predictor mode: stop at the first iterate with omega1 >= this")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args(argv)

    model = OlhoffBeam(a.nelx, a.nely, **PRODUCTION)
    prob = OlhoffFreqProblem(model, volfrac=VOLFRAC, rho0=RHO0, Jc=a.Jc, filter_sens=not a.no_filter,
                             bounds_in_H=(a.optimizer == "nso"), offdiag_equality=a.offdiag,
                             gap_tol=a.gap_tol, metric=a.metric)
    if a.init_rho:
        if a.init_rho == "ref":
            import scipy.io as sio
            rho_init = np.asarray(sio.loadmat(os.path.join(a.out, f"olhoff_ref_{a.nelx}x{a.nely}.mat"), squeeze_me=True)["rho"], dtype=float).ravel()
        else:
            rho_init = np.load(a.init_rho)
        prob.set_initial(rho_init)
    prob.stop_omega1 = a.stop_omega1
    w0 = prob.omegas(prob.x0())
    tag = a.tag or (f"{a.optimizer}_{a.nelx}x{a.nely}_" + (f"dt{a.dt}_itn{a.itnorm}_{a.metric}" + ("_offdiag" if a.offdiag else "")
                                                           if a.optimizer == "nso" else f"move{a.move}") + ("_nofilter" if a.no_filter else ""))
    os.makedirs(a.out, exist_ok=True)
    print(f"=== {tag}: NE={model.nele}, free dofs={len(model.free)}, omega0={w0[:3]}, lamref={prob.lamref:.3f}", flush=True)

    t0 = time.time()
    if a.optimizer == "nso":
        from nullspace_optimizer import nlspace_solve
        params = dict(dt=a.dt, alphaJ=1.0, alphaC=1.0, maxit=a.maxit, K=a.K,
                      itnormalisation=(a.maxit + 1 if a.itnorm < 0 else a.itnorm),
                      qp_solver=a.qp_solver, tol_qp=1e-8, method_xiC="qp", qp_saturate_slack=True,
                      save_only_N_iterations=2, save_only_Q_constraints=a.Jc + 1, tol=1e-9)
        res = nlspace_solve(prob, params)
    else:
        from nullspace_optimizer.optimizers.MMA.MMA import mma_solve
        params = dict(maxit=a.maxit, tol=1e-9, move=a.move, tight_move=True, save_only_N_iterations=3)
        # mma_solve broadcasts l/u against an (n,1) column: pass them as (n,1) arrays
        l = np.concatenate((prob.rhomin * np.ones(prob.NE), [0.0]))[:, None]
        u = np.concatenate((np.ones(prob.NE), [10.0]))[:, None]
        res = mma_solve(prob, l, u, params)
    wall = time.time() - t0

    x = res["x"][-1]
    rho, beta = prob.split(x)
    w = prob.omegas(x)
    hist = {k: np.asarray(res[k], dtype=float).tolist() for k in
            ("omega1", "omega2", "omega3", "beta_omega", "vol", "Mnd", "drho_l2", "drho_max", "J", "time") if k in res}
    hist["it"] = list(map(int, res["it"]))
    drho = np.asarray(hist["drho_l2"], dtype=float)
    eps_l2 = 0.05 * np.sqrt(model.nele / 3200)            # Olhoff's mesh-scaled stop threshold
    native_stop = next((i for i, d in enumerate(drho) if i > 0 and np.isfinite(d) and d < eps_l2), None)
    summary = dict(tag=tag, optimizer=a.optimizer, args=vars(a), NE=model.nele, iterations=int(hist["it"][-1]),
                   omega=w[:4].tolist(), omega1=float(w[0]), gap12_pct=float(100 * (w[1] - w[0]) / w[0]),
                   beta_omega=float(np.sqrt(max(beta * prob.lamref, 0))), vol=float(rho.mean()),
                   Mnd=prob.m.grayness(rho), wall_s=wall, s_per_iter=wall / max(1, hist["it"][-1]),
                   eig_solves=prob.n_eig, eig_time_s=prob.t_eig,
                   olhoff_eps_l2=eps_l2, first_iter_below_olhoff_eps=native_stop,
                   omega1_at_native_stop=(hist["omega1"][native_stop] if native_stop is not None else None),
                   omega0=w0[:3].tolist())
    with open(os.path.join(a.out, tag + ".json"), "w") as fh:
        json.dump(dict(summary=summary, history=hist), fh, indent=1)
    np.save(os.path.join(a.out, tag + "_rho.npy"), rho)
    import scipy.io as sio
    sio.savemat(os.path.join(a.out, tag + "_rho.mat"), dict(rho=rho, omega=w[:4], iterations=hist["it"][-1], wall_s=wall))
    print(f"    done it={summary['iterations']} omega1={w[0]:.4f} omega2={w[1]:.4f} gap={summary['gap12_pct']:.2f}% "
          f"vol={rho.mean():.4f} Mnd={summary['Mnd']*100:.2f}% wall={wall:.0f}s ({summary['s_per_iter']:.2f} s/it, "
          f"eig {prob.t_eig:.0f}s/{prob.n_eig} solves) first ||drho||2<{eps_l2:.3f} at it {native_stop}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
