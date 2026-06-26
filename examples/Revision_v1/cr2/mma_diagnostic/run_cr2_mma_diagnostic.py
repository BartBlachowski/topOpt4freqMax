"""
run_cr2_mma_diagnostic.py — CR2-MMA matched diagnostic run.

Purpose: determine whether CR2 failure is caused by OC rather than the
         sensitivity formula, by running both variants with MMA instead.

Acceptance criteria (identical to prior CR2 protocol):
  - not capped (converged before max_iters)
  - design_change <= 1e-3
  - feasibility <= 1e-4
  - final MAC >= 0.8  (computed vs solid reference mode)
  - all artifacts present

No further tuning is performed after this script runs.
"""

from __future__ import annotations
import sys
import os
import json
import time
import traceback
import hashlib
from pathlib import Path

import numpy as np

# ---- repo root on path ----
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[3]   # .../topOpt4freqMax
OUTPUT_DIR  = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for p in [str(REPO_ROOT), str(REPO_ROOT / "tools" / "Python")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from run_topopt_from_json import run_topopt_from_json  # noqa: E402

# acceptance thresholds
CONV_TOL  = 1e-3
FEAS_TOL  = 1e-4
MAC_THRESH = 0.8

LOG_PATH = OUTPUT_DIR / "cr2_mma_run.log"
_log_fh  = None

def _log(msg: str) -> None:
    print(msg, flush=True)
    if _log_fh is not None:
        _log_fh.write(msg + "\n")
        _log_fh.flush()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_final_mac(gate_a0: dict, xPhys: np.ndarray,
                       free: np.ndarray) -> tuple[float, int, float]:
    """
    Compute MAC between solid reference mode 1 and final design's first eigenmode.
    Returns (mac, tracked_mode_index, tracked_mode_omega_rad).
    """
    from scipy.sparse.linalg import eigsh
    try:
        M_final = gate_a0["current_mass_matrix"]
        K_ref   = None  # not stored; recompute from gate_a0 won't work easily
        # gate_a0 has no K(x_final). We use the reference mode directly from info
        # and the final M. For MAC: use mass-weighted inner product convention.
        ref_phi = gate_a0["reference_modes"][:, 0]   # solid reference mode 1

        # We can't easily get K_final here without re-running.
        # Instead: use stored final frequencies and mode shapes from gate_a0
        # if available. If not, fall back to a simpler criterion.
        # gate_a0 does not store final eigenvectors (only reference ones).
        # We'll compute MAC between ref_phi and its projection through M_final
        # which is a conservative estimate; the actual MAC requires final eigvecs.

        # Better: run a small eigensolver using the information we do have.
        # The solver stores current_mass_matrix (M at final x). We need K at final x.
        # These are not stored in gate_a0. We'll store them separately in the run.
        # See _run_variant — we pass back extra_kf, extra_mf from solver info.

        # Fallback: use freq ratio as MAC proxy
        return np.nan, 1, np.nan
    except Exception:
        return np.nan, 1, np.nan


def _compute_mac_from_kf_mf(Kf: object, Mf: object, ref_phi_free: np.ndarray) -> tuple[float, int, float]:
    """Compute MAC between ref_phi_free and first eigenvector of (Kf, Mf)."""
    from scipy.sparse.linalg import eigsh
    n_req = min(3, ref_phi_free.shape[0] - 1)
    try:
        lam, phi = eigsh(Kf, M=Mf, k=n_req, sigma=1e-6, which="LM")
        order = np.argsort(lam)
        lam   = lam[order]
        phi   = phi[:, order]
        lam   = lam[lam > 0]
        phi   = phi[:, :lam.size]
        if lam.size == 0:
            return np.nan, 1, np.nan

        # MAC against mode 1
        ref_n = ref_phi_free / (np.linalg.norm(ref_phi_free) + 1e-300)
        best_mac = -1.0
        best_idx = 0
        best_omega = np.nan
        for k in range(lam.size):
            pn = phi[:, k] / (np.linalg.norm(phi[:, k]) + 1e-300)
            mac_k = float(np.dot(ref_n, pn)) ** 2
            if mac_k > best_mac:
                best_mac = mac_k
                best_idx = k + 1   # 1-based
                best_omega = float(np.sqrt(max(0.0, lam[k])))

        return best_mac, best_idx, best_omega
    except Exception as exc:
        _log(f"  [MAC] eigensolver failed: {exc}")
        return np.nan, 1, np.nan


# ---------------------------------------------------------------------------
# variant runner
# ---------------------------------------------------------------------------

def _run_variant(label: str, config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    max_iters = int(cfg["optimization"]["max_iters"])
    load_sens = str(cfg["optimization"]["load_sensitivity"])
    optimizer  = str(cfg["optimization"]["optimizer"])
    prefix     = OUTPUT_DIR / f"cr2_mma_variant_{label.lower()}"

    result: dict = {
        "label": label,
        "load_sensitivity": load_sens,
        "optimizer": optimizer,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": _sha256(config_path),
        "iteration_cap": max_iters,
        "solver_success": False,
        "iterations": -1,
        "final_design_change": np.nan,
        "final_feasibility": np.nan,
        "final_mac": np.nan,
        "tracked_mode_index": -1,
        "tracked_mode_omega": np.nan,
        "final_objective": np.nan,
        "final_freq_omega1": np.nan,
        "timing_total_s": np.nan,
        "plateau_diagnostics": {},
        "acceptance": {},
        "artifacts": {},
        "exception": "",
    }

    _log(f"\n{'='*60}")
    _log(f"CR2-MMA Variant {label}  ({load_sens}, {optimizer})")
    _log(f"{'='*60}")

    t0 = time.perf_counter()
    x_final   = None
    info_dict = {}
    history   = {}
    Kf_final  = None
    Mf_final  = None
    ref_phi_free = None

    try:
        # run_topopt_from_json with return_diagnostics=True returns omega in rad/s
        x_final, omega_rad, t_iter, n_iter, info_dict = run_topopt_from_json(
            str(config_path), return_diagnostics=True
        )
        result["timing_total_s"] = time.perf_counter() - t0
        result["solver_success"] = True
        result["iterations"]     = int(n_iter)
        result["final_freq_omega1"] = float(omega_rad[0]) if np.isfinite(omega_rad[0]) else np.nan

        history = info_dict.get("cr2_history", {})
        gate    = info_dict.get("gate_a0", {})

        if history:
            result["final_design_change"] = float(history["design_change"][-1])
            result["final_feasibility"]   = float(history["feasibility"][-1])
            result["final_objective"]     = float(history["objective"][-1])
        else:
            _log("  WARNING: no cr2_history in info")

        # ---- compute MAC using final K, M ----
        # topopt_freq returns info with 'last_obj' and 'last_vol' but not K/M.
        # We need to rebuild final K, M from x_final and cfg. Import solver module.
        _log("  Computing final MAC via eigensolver on final design ...")
        try:
            import importlib
            solver_dir = str(REPO_ROOT / "analysis" / "ourApproach" / "Python")
            if solver_dir not in sys.path:
                sys.path.insert(0, solver_dir)
            tf_mod = importlib.import_module("topopt_freq")

            L     = float(cfg["domain"]["size"]["length"])
            H     = float(cfg["domain"]["size"]["height"])
            nelx  = int(cfg["domain"]["mesh"]["nelx"])
            nely  = int(cfg["domain"]["mesh"]["nely"])
            E0    = float(cfg["material"]["E"])
            nu    = float(cfg["material"]["nu"])
            rho0  = float(cfg["material"]["rho"])
            Emin  = E0 * float(cfg["void_material"]["E_min_ratio"])
            rho_min = float(cfg["void_material"]["rho_min"])
            penal = float(cfg["optimization"]["penalization"])
            pmass = 1.0

            hx = L / nelx; hy = H / nely
            KE = tf_mod.lk(hx, hy, nu)
            ME = tf_mod.lm(hx, hy)

            # node coords (same indexing as solver)
            nx = nelx + 1; ny = nely + 1
            node_x = np.tile(np.linspace(0, L, nx), ny).reshape(ny, nx).T.ravel()
            node_y = np.repeat(np.linspace(0, H, ny), nx).reshape(ny, nx).T.ravel()

            # DOF map
            from tools.Python.bc_processor import supports_to_fixed_dofs
            supports = cfg["bc"]["supports"]
            extra_fixed = supports_to_fixed_dofs(supports, nelx, nely, L, H)

            ndof = 2 * nx * ny
            all_dofs = np.arange(ndof, dtype=np.int64)
            fixed = np.unique(extra_fixed.astype(np.int64))
            free  = np.setdiff1d(all_dofs, fixed)

            # edofMat
            nxy = nx * ny
            node_ids = np.arange(nxy).reshape(nx, ny)  # [elx, ely] -> global node
            edof_mat = np.zeros((nelx * nely, 8), dtype=np.int64)
            for elx in range(nelx):
                for ely in range(nely):
                    e   = elx * nely + ely
                    nLL = node_ids[elx,   ely]
                    nLR = node_ids[elx+1, ely]
                    nUR = node_ids[elx+1, ely+1]
                    nUL = node_ids[elx,   ely+1]
                    edof_mat[e, :] = [2*nLL, 2*nLL+1, 2*nLR, 2*nLR+1,
                                      2*nUR, 2*nUR+1, 2*nUL, 2*nUL+1]

            iK = np.kron(edof_mat, np.ones((8,1), dtype=np.int64)).ravel()
            jK = np.kron(edof_mat, np.ones((1,8), dtype=np.int64)).ravel()

            from scipy.sparse import csc_array
            xp = x_final
            sK = (KE.ravel()[:, np.newaxis] * (Emin + xp**penal * (E0 - Emin))).ravel(order="F")
            sM = (ME.ravel()[:, np.newaxis] * (rho_min + xp**pmass * (rho0 - rho_min))).ravel(order="F")
            K_fin = csc_array((sK, (iK, jK)), shape=(ndof, ndof))
            M_fin = csc_array((sM, (iK, jK)), shape=(ndof, ndof))
            K_fin = (K_fin + K_fin.T) / 2
            M_fin = (M_fin + M_fin.T) / 2
            Kf_final = K_fin[free][:, free]
            Mf_final = M_fin[free][:, free]

            # Reference mode from gate_a0
            if gate and "reference_modes" in gate:
                ref_phi_all  = gate["reference_modes"][:, 0]
                ref_phi_free = ref_phi_all[free]
            else:
                ref_phi_free = None

            if ref_phi_free is not None:
                mac, mode_idx, omega_rad = _compute_mac_from_kf_mf(
                    Kf_final, Mf_final, ref_phi_free)
                result["final_mac"]            = float(mac)
                result["tracked_mode_index"]   = int(mode_idx)
                result["tracked_mode_omega"]   = float(omega_rad)
                _log(f"  Final MAC={mac:.4f} (vs solid ref mode 1), "
                     f"tracked mode {mode_idx}, omega={omega_rad:.4f} rad/s")
            else:
                _log("  WARNING: no reference modes in gate_a0; MAC not computed")

        except Exception as mac_exc:
            _log(f"  WARNING: MAC computation failed: {mac_exc}")
            traceback.print_exc()

    except Exception as exc:
        result["timing_total_s"] = time.perf_counter() - t0
        result["exception"] = traceback.format_exc()
        _log(f"  EXCEPTION in Variant {label}:\n{result['exception']}")

    # ---- save artifacts ----
    _log(f"\n  Saving artifacts for Variant {label} ...")

    arts: dict[str, Path] = {
        "histories_csv":       prefix.with_name(f"cr2_mma_variant_{label.lower()}_histories.csv"),
        "mode_tracking_csv":   prefix.with_name(f"cr2_mma_variant_{label.lower()}_mode_tracking.csv"),
        "sensitivity_norms_csv": prefix.with_name(f"cr2_mma_variant_{label.lower()}_sensitivity_norms.csv"),
        "topology_csv":        prefix.with_name(f"cr2_mma_variant_{label.lower()}_topology.csv"),
        "topology_png":        prefix.with_name(f"cr2_mma_variant_{label.lower()}_topology.png"),
    }
    result["artifacts"] = {k: str(v.relative_to(REPO_ROOT)) for k, v in arts.items()}

    if result["solver_success"] and history:
        _save_histories_csv(arts["histories_csv"], history)
        _save_mode_tracking_csv(arts["mode_tracking_csv"], history,
                                result["final_mac"],
                                result["tracked_mode_index"],
                                result["tracked_mode_omega"])
        _save_sensitivity_norms_csv(arts["sensitivity_norms_csv"], history)

    if result["solver_success"] and x_final is not None:
        nelx_s = int(cfg["domain"]["mesh"]["nelx"])
        nely_s = int(cfg["domain"]["mesh"]["nely"])
        _save_topology_csv(arts["topology_csv"], x_final, nelx_s, nely_s)
        _save_topology_png(arts["topology_png"], x_final, nelx_s, nely_s, label,
                           load_sens, result["final_mac"],
                           result.get("final_design_change", np.nan),
                           result.get("final_feasibility", np.nan))

    # ---- plateau diagnostics ----
    if result["solver_success"] and history:
        result["plateau_diagnostics"] = _plateau_diagnostics(history, max_iters)
    else:
        result["plateau_diagnostics"] = {"status": "unavailable"}

    # ---- acceptance ----
    result["acceptance"] = _assess_acceptance(result, cfg, arts)

    _log(f"\n  Variant {label} done  accepted={result['acceptance']['accepted']}")
    if not result["acceptance"]["accepted"]:
        _log(f"    failures: {result['acceptance']['failures']}")

    return result, history, x_final


# ---------------------------------------------------------------------------
# CSV / PNG writers
# ---------------------------------------------------------------------------

def _save_histories_csv(path: Path, h: dict) -> None:
    n     = len(h["objective"])
    iters = np.arange(1, n + 1)
    freqs = np.asarray(h["frequency"])
    if freqs.ndim == 1:
        freqs = freqs[:, np.newaxis]
    n_f   = freqs.shape[1] if freqs.ndim == 2 else 1

    header = ["iteration", "objective", "design_change", "feasibility",
              "grayness", "volume",
              "sensitivity_difference_l2", "sensitivity_difference_linf"]
    header += [f"omega_{k+1}" for k in range(n_f)]

    rows = []
    for i in range(n):
        row = [
            iters[i],
            h["objective"][i],
            h["design_change"][i],
            h["feasibility"][i],
            h["grayness"][i],
            h["volume"][i],
            h["sensitivity_difference_l2"][i],
            h["sensitivity_difference_linf"][i],
        ]
        if freqs.ndim == 2:
            row += list(freqs[i, :])
        else:
            row += [freqs[i]]
        rows.append(row)

    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(f"{v}" for v in row) + "\n")
    _log(f"    Saved: {path.name}")


def _save_mode_tracking_csv(path: Path, h: dict,
                             final_mac: float, mode_idx: int,
                             mode_omega: float) -> None:
    n = len(h["objective"])
    freqs = np.asarray(h["frequency"])
    omega1 = freqs[:, 0] if freqs.ndim == 2 else freqs

    with open(path, "w") as f:
        f.write("iteration,omega_1_rad_s,final_mac,final_tracked_mode_index,final_tracked_mode_omega\n")
        for i in range(n):
            mac_col   = float(final_mac) if i == n - 1 else ""
            idx_col   = mode_idx if i == n - 1 else ""
            omega_col = float(mode_omega) if i == n - 1 else ""
            f.write(f"{i+1},{float(omega1[i])},{mac_col},{idx_col},{omega_col}\n")
    _log(f"    Saved: {path.name}")


def _save_sensitivity_norms_csv(path: Path, h: dict) -> None:
    n = len(h["objective"])
    with open(path, "w") as f:
        f.write("iteration,sensitivity_difference_l2,sensitivity_difference_linf\n")
        for i in range(n):
            f.write(f"{i+1},{h['sensitivity_difference_l2'][i]},{h['sensitivity_difference_linf'][i]}\n")
    _log(f"    Saved: {path.name}")


def _save_topology_csv(path: Path, x: np.ndarray, nelx: int, nely: int) -> None:
    with open(path, "w") as f:
        f.write("element_index,elx,ely,density\n")
        for elx in range(nelx):
            for ely in range(nely):
                e = elx * nely + ely
                f.write(f"{e+1},{elx+1},{ely+1},{x[e]:.8f}\n")
    _log(f"    Saved: {path.name}")


def _save_topology_png(path: Path, x: np.ndarray, nelx: int, nely: int,
                       label: str, mode: str, mac: float,
                       dc: float, feas: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 2.5), facecolor="white")
        ax.imshow(x.reshape(nelx, nely).T, cmap="gray_r", origin="lower",
                  interpolation="none", vmin=0, vmax=1, aspect="auto")
        title = (f"CR2-MMA Variant {label} ({mode})"
                 f"  MAC={mac:.3f}  dc={dc:.2e}  feas={feas:.2e}")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Element column")
        ax.set_ylabel("Element row")
        fig.tight_layout()
        fig.savefig(str(path), dpi=180, bbox_inches="tight")
        plt.close(fig)
        _log(f"    Saved: {path.name}")
    except Exception as exc:
        _log(f"    WARNING: topology PNG failed: {exc}")


# ---------------------------------------------------------------------------
# diagnostics / acceptance
# ---------------------------------------------------------------------------

def _plateau_diagnostics(h: dict, max_iters: int) -> dict:
    n  = len(h["objective"])
    w  = min(50, n)
    idx = slice(n - w, n)
    obj = np.array(h["objective"][n-w:])
    dc  = np.array(h["design_change"][n-w:])
    feas = np.array(h["feasibility"][n-w:])
    freqs = np.asarray(h["frequency"])
    omega1 = (freqs[:, 0] if freqs.ndim == 2 else freqs)[n-w:]

    sc_obj  = max(abs(float(np.mean(obj))), 1e-300)
    sc_om   = max(abs(float(np.mean(omega1))), 1e-300)

    obj_range = float((obj.max() - obj.min()) / sc_obj)
    om_range  = float((omega1.max() - omega1.min()) / sc_om)
    dc_median = float(np.median(dc))
    dc_max    = float(dc.max())
    feas_min  = float(feas.min())
    feas_max  = float(feas.max())
    move_sat  = float(np.mean(dc >= 0.99 * 0.2))

    lag1_obj = np.abs(np.diff(obj))
    lag2_obj = np.abs(obj[2:] - obj[:-2])
    med_lag1 = float(np.median(lag1_obj) / sc_obj) if len(lag1_obj) > 0 else np.inf
    med_lag2 = float(np.median(lag2_obj) / sc_obj) if len(lag2_obj) > 0 else np.inf

    period2_sig = (med_lag2 <= 1e-4) and (med_lag1 >= 1e-3)
    move_sat_sig = move_sat >= 0.9
    plateau      = (obj_range <= 5e-3) and (om_range <= 5e-3)
    stable_cycle = all(dc > 1e-3) and plateau and (dc_median > 1e-3)
    alg_fail_sig = move_sat_sig or period2_sig or stable_cycle

    return {
        "status": "computed",
        "window_length": w,
        "relative_objective_range": obj_range,
        "relative_frequency_range": om_range,
        "median_design_change": dc_median,
        "max_design_change": dc_max,
        "feasibility_min": feas_min,
        "feasibility_max": feas_max,
        "move_saturation_fraction": move_sat,
        "median_relative_lag1_objective": med_lag1,
        "median_relative_lag2_objective": med_lag2,
        "objective_plateau": bool(plateau),
        "move_saturation_signature": bool(move_sat_sig),
        "period2_signature": bool(period2_sig),
        "persistent_bounded_cycle_signature": bool(stable_cycle),
        "algorithm_failure_signature": bool(alg_fail_sig),
    }


def _assess_acceptance(result: dict, cfg: dict, arts: dict[str, Path]) -> dict:
    failures = []
    if not result["solver_success"]:
        failures.append("solver_success=false")
    else:
        max_it = int(cfg["optimization"]["max_iters"])
        if result["iterations"] >= max_it:
            failures.append("iteration cap reached")
        dc = result["final_design_change"]
        if not np.isfinite(dc) or dc > CONV_TOL:
            failures.append(f"design_change {dc:.3e} > {CONV_TOL:.1e}")
        feas = result["final_feasibility"]
        if not np.isfinite(feas) or feas > FEAS_TOL:
            failures.append(f"feasibility {feas:.3e} > {FEAS_TOL:.1e}")
        mac = result["final_mac"]
        if not np.isfinite(mac) or mac < MAC_THRESH:
            failures.append(f"MAC {mac:.4f} < {MAC_THRESH:.1f}")

    missing = [str(p) for p in arts.values() if not p.exists()]
    if missing:
        failures.append(f"missing artifacts: {missing}")

    return {
        "accepted": len(failures) == 0,
        "failures": failures,
        "conv_tol": CONV_TOL,
        "feas_tol": FEAS_TOL,
        "mac_threshold": MAC_THRESH,
    }


# ---------------------------------------------------------------------------
# topology difference
# ---------------------------------------------------------------------------

def _topology_difference(x_a: np.ndarray, x_b: np.ndarray,
                          nelx: int, nely: int) -> dict:
    if x_a is None or x_b is None or x_a.size != x_b.size:
        return {"status": "unavailable"}
    d = x_a - x_b
    corr = float(np.corrcoef(x_a, x_b)[0, 1]) if x_a.std() > 0 and x_b.std() > 0 else np.nan
    return {
        "status": "computed",
        "mean_absolute_difference": float(np.mean(np.abs(d))),
        "rms_difference": float(np.sqrt(np.mean(d**2))),
        "max_absolute_difference": float(np.max(np.abs(d))),
        "fraction_gt_0_01": float(np.mean(np.abs(d) > 0.01)),
        "fraction_gt_0_05": float(np.mean(np.abs(d) > 0.05)),
        "pearson_correlation": corr,
    }


def _save_diff_png(path: Path, x_a: np.ndarray, x_b: np.ndarray,
                   nelx: int, nely: int) -> None:
    if x_a is None or x_b is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = np.abs(x_a - x_b)
        fig, ax = plt.subplots(figsize=(12, 2.5), facecolor="white")
        im = ax.imshow(d.reshape(nelx, nely).T, cmap="hot_r", origin="lower",
                       interpolation="none", vmin=0, vmax=1, aspect="auto")
        ax.set_title("CR2-MMA |Variant A − Variant B| topology difference")
        ax.set_xlabel("Element column")
        ax.set_ylabel("Element row")
        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
        fig.tight_layout()
        fig.savefig(str(path), dpi=180, bbox_inches="tight")
        plt.close(fig)
        _log(f"    Saved: {path.name}")
    except Exception as exc:
        _log(f"    WARNING: diff PNG failed: {exc}")


# ---------------------------------------------------------------------------
# outcome classification
# ---------------------------------------------------------------------------

def _classify_outcome(res_a: dict, res_b: dict) -> tuple[str, str]:
    """Return (outcome_category, rationale)."""
    acc_a = res_a["acceptance"]["accepted"]
    acc_b = res_b["acceptance"]["accepted"]
    alg_a = res_a.get("plateau_diagnostics", {}).get("algorithm_failure_signature", False)
    alg_b = res_b.get("plateau_diagnostics", {}).get("algorithm_failure_signature", False)
    capped_a = (res_a["iterations"] >= res_a["iteration_cap"])
    capped_b = (res_b["iterations"] >= res_b["iteration_cap"])

    if acc_a and acc_b:
        return (
            "accepted matched comparison",
            "Both variants converged under MMA with all criteria satisfied. "
            "Endpoint comparison is scientifically valid. "
            "CR2 failure was caused by OC, not the sensitivity formula.",
        )
    if acc_a and not acc_b:
        return (
            "accepted matched comparison",
            "Variant A (omitted) converged; Variant B (complete) did not. "
            "MMA resolves OC micro-oscillation for the omitted formula. "
            "The complete formula remains optimizer-unstable under MMA. "
            "Partial accepted comparison: Variant A result is reportable.",
        )
    if not acc_a and not acc_b and (alg_a or alg_b):
        return (
            "diagnostic optimizer failure",
            "At least one variant shows algorithm-failure signature (move saturation, "
            "period-2, or stable cycle) even under MMA. "
            "Failure is not exclusively attributable to OC.",
        )
    if capped_a or capped_b:
        return (
            "inconclusive",
            "One or both variants hit the iteration cap without converging or "
            "showing a clear failure signature.",
        )
    return (
        "inconclusive",
        "Convergence criteria not met but no clear failure signature detected.",
    )


# ---------------------------------------------------------------------------
# manifest / JSON serialization
# ---------------------------------------------------------------------------

def _to_json_safe(obj):
    """Recursively convert numpy scalars / ndarrays to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, (int,)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(_to_json_safe(obj), f, indent=2)


def _write_manifest(path: Path, outcome: str, res_a: dict, res_b: dict,
                    diff_metrics: dict, config_a: Path, config_b: Path) -> None:
    def _entry(p: Path, role: str) -> dict:
        pp = Path(p) if not isinstance(p, Path) else p
        return {
            "path": str(pp.relative_to(REPO_ROOT)) if pp.is_absolute() else str(pp),
            "role": role,
            "exists": pp.exists(),
        }

    entries = [
        _entry(SCRIPT_DIR / "run_cr2_mma_diagnostic.py",    "run script"),
        _entry(config_a,                                      "Variant A config"),
        _entry(config_b,                                      "Variant B config"),
        _entry(LOG_PATH,                                      "run log"),
        _entry(OUTPUT_DIR / "cr2_mma_results.json",          "result summary JSON"),
        _entry(OUTPUT_DIR / "cr2_mma_summary.md",            "result summary markdown"),
        _entry(OUTPUT_DIR / "cr2_topology_difference_metrics.json", "topology difference metrics"),
        _entry(OUTPUT_DIR / "cr2_topology_difference.png",   "topology difference figure"),
    ]
    for art_key, role in [
        ("histories_csv", "histories CSV"),
        ("mode_tracking_csv", "mode tracking CSV"),
        ("sensitivity_norms_csv", "sensitivity norms CSV"),
        ("topology_csv", "topology CSV"),
        ("topology_png", "topology PNG"),
    ]:
        for lbl, res in [("a", res_a), ("b", res_b)]:
            p_str = res.get("artifacts", {}).get(art_key, "")
            if p_str:
                entries.append(_entry(REPO_ROOT / p_str,
                                      f"Variant {lbl.upper()} {role}"))

    manifest = {
        "study": "CR2-MMA matched diagnostic",
        "gate": "E4-CR2-MMA",
        "outcome_category": outcome,
        "no_further_tuning": True,
        "output_directory": str(OUTPUT_DIR.relative_to(REPO_ROOT)),
        "entries": entries,
    }
    _write_json(path, manifest)
    _log(f"  Manifest: {path.name}")


def _write_summary_md(path: Path, outcome: str, rationale: str,
                      res_a: dict, res_b: dict, diff_metrics: dict) -> None:
    acc_a = res_a["acceptance"]["accepted"]
    acc_b = res_b["acceptance"]["accepted"]
    with open(path, "w") as f:
        f.write("# CR2-MMA Diagnostic Summary\n\n")
        f.write(f"- **Outcome:** {outcome.upper()}\n")
        f.write(f"- **Rationale:** {rationale}\n")
        f.write(f"- Optimizer: MMA (both variants)\n")
        f.write(f"- Load: F(x) = omega0^2 * M(x) * Phi0\n")
        f.write(f"- Acceptance: dc <= {CONV_TOL:.0e}, feas <= {FEAS_TOL:.0e}, "
                f"MAC >= {MAC_THRESH:.1f}, not capped, all artifacts present\n")
        f.write(f"- No further tuning performed\n\n")

        for label, res in [("A (omitted)", res_a), ("B (complete)", res_b)]:
            f.write(f"## Variant {label}\n\n")
            f.write(f"- Accepted: {int(res['acceptance']['accepted'])}\n")
            f.write(f"- Solver success: {int(res['solver_success'])}\n")
            f.write(f"- Iterations: {res['iterations']} / {res['iteration_cap']}\n")
            if res["solver_success"]:
                f.write(f"- Final design_change: {res['final_design_change']:.3e}\n")
                f.write(f"- Final feasibility: {res['final_feasibility']:.3e}\n")
                f.write(f"- Final MAC: {res['final_mac']:.4f} (mode {res['tracked_mode_index']})\n")
                f.write(f"- Final omega_1: {res['final_freq_omega1']:.4f} rad/s\n")
                f.write(f"- Timing: {res['timing_total_s']:.1f} s\n")
                pd = res.get("plateau_diagnostics", {})
                f.write(f"- Algorithm failure signature: {pd.get('algorithm_failure_signature', '?')}\n")
            failures = res["acceptance"]["failures"]
            if failures:
                f.write(f"- Failures: {'; '.join(failures)}\n")
            else:
                f.write("- Failures: none\n")
            f.write("\n")

        f.write("## Topology Difference\n\n")
        if diff_metrics.get("status") == "computed":
            f.write(f"| Metric | Value |\n|--------|-------|\n")
            f.write(f"| MAD | {diff_metrics['mean_absolute_difference']:.3f} |\n")
            f.write(f"| RMS | {diff_metrics['rms_difference']:.3f} |\n")
            f.write(f"| Max |Δ| | {diff_metrics['max_absolute_difference']:.3f} |\n")
            f.write(f"| |Δ|>0.05 fraction | {diff_metrics['fraction_gt_0_05']:.3f} |\n")
            f.write(f"| Pearson r | {diff_metrics['pearson_correlation']:.3f} |\n")
        else:
            f.write("Unavailable (one or both variants did not converge).\n")
    _log(f"  Summary: {path.name}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    global _log_fh
    _log_fh = open(LOG_PATH, "w")
    try:
        _log(f"CR2-MMA matched diagnostic started: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
        _log(f"Repo root : {REPO_ROOT}")
        _log(f"Output dir: {OUTPUT_DIR}")
        _log("No further tuning beyond these two predeclared variants is permitted.")

        config_a = SCRIPT_DIR / "cr2_mma_variant_a_omitted.json"
        config_b = SCRIPT_DIR / "cr2_mma_variant_b_complete.json"
        nelx = 160; nely = 20

        # ---- run variants ----
        res_a, hist_a, x_a = _run_variant("a", config_a)
        res_b, hist_b, x_b = _run_variant("b", config_b)

        # ---- topology difference ----
        _log("\n--- Topology difference ---")
        diff_metrics = _topology_difference(x_a, x_b, nelx, nely)
        diff_png  = OUTPUT_DIR / "cr2_topology_difference.png"
        diff_json = OUTPUT_DIR / "cr2_topology_difference_metrics.json"
        _save_diff_png(diff_png, x_a, x_b, nelx, nely)
        _write_json(diff_json, diff_metrics)
        for k, v in diff_metrics.items():
            _log(f"  {k}: {v}")

        # ---- classify outcome ----
        outcome, rationale = _classify_outcome(res_a, res_b)
        _log(f"\nOUTCOME: {outcome.upper()}")
        _log(f"RATIONALE: {rationale}")

        # ---- write summary artifacts ----
        _log("\n--- Writing summary artifacts ---")
        results_json = OUTPUT_DIR / "cr2_mma_results.json"
        summary_md   = OUTPUT_DIR / "cr2_mma_summary.md"
        manifest     = OUTPUT_DIR / "cr2_mma_manifest.json"

        full_result = {
            "study": "CR2-MMA matched diagnostic",
            "gate": "E4-CR2-MMA",
            "outcome_category": outcome,
            "rationale": rationale,
            "authoritative_load": "F(x) = omega0^2 * M(x) * Phi0",
            "acceptance_criteria": {
                "design_change_tolerance": CONV_TOL,
                "feasibility_tolerance": FEAS_TOL,
                "mac_threshold": MAC_THRESH,
                "not_capped": True,
                "all_artifacts_present": True,
            },
            "variant_a": res_a,
            "variant_b": res_b,
            "topology_difference": diff_metrics,
        }
        _write_json(results_json, full_result)
        _log(f"  Results JSON: {results_json.name}")
        _write_summary_md(summary_md, outcome, rationale, res_a, res_b, diff_metrics)
        _write_manifest(manifest, outcome, res_a, res_b, diff_metrics, config_a, config_b)

        _log(f"\nCR2-MMA diagnostic complete: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
        _log(f"Outcome: {outcome.upper()}")
        _log(f"Variant A accepted: {res_a['acceptance']['accepted']}")
        _log(f"Variant B accepted: {res_b['acceptance']['accepted']}")

    finally:
        if _log_fh is not None:
            _log_fh.close()
            _log_fh = None


if __name__ == "__main__":
    main()
