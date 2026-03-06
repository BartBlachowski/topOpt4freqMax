"""
run_topopt_from_json.py — Run ourApproach topology optimization from a JSON task file.

Mirrors tools/Matlab/run_topopt_from_json.m (ourApproach branch only).

Usage
-----
From Python:
    from tools.Python.run_topopt_from_json import run_topopt_from_json
    x, omega, t_iter, n_iter = run_topopt_from_json("path/to/config.json")

From command line:
    python tools/Python/run_topopt_from_json.py examples/HingedBeam/BeamTopOptFreq.json
"""

from __future__ import annotations
import sys
import os
import re
from pathlib import Path
import numpy as np

# Ensure repository root is on the path so sibling packages are importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.Python.config_loader import (
    load_json_config, req_num, req_int, req_str,
    has_field_path, get_field_path, parse_bool,
)
from tools.Python.bc_processor import supports_to_fixed_dofs
from tools.Python.passive_regions import parse_passive_regions
from tools.Python.load_cases import validate_load_cases


def _save_frequency_iteration_plot(
    freq_iter_omega: np.ndarray,
    approach_name: str,
    nelx: int,
    nely: int,
    repo_root: str,
) -> None:
    """Save Figure-6-style frequency history plot to <repo>/results."""
    arr = np.asarray(freq_iter_omega, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 1:
        return

    n_iter = arr.shape[0]
    if arr.shape[1] < 3:
        padded = np.full((n_iter, 3), np.nan, dtype=float)
        padded[:, : arr.shape[1]] = arr
        arr = padded
    else:
        arr = arr[:, :3]

    out_dir = Path(repo_root) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    name_safe = re.sub(r"[^\w\-]", "_", str(approach_name))
    png_path = out_dir / f"{name_safe}_{nelx}x{nely}_freq_iterations.png"

    try:
        import matplotlib
        # Prefer a non-interactive backend in headless/script usage, but do not
        # fail if a backend has already been selected earlier in the process.
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
        x_iter = np.arange(1, n_iter + 1, dtype=int)
        colors = ["#0072BD", "#D95319", "#77AC30"]
        for j in range(3):
            ax.plot(x_iter, arr[:, j], "-", linewidth=1.6, color=colors[j], label=rf"$\omega_{{{j+1}}}$")

        ax.set_xlabel("Outer iteration")
        ax.set_ylabel("Frequency (rad/s)")
        ax.set_title(f"{approach_name} frequency history")
        ax.grid(True)
        ax.legend(loc="best")
        if n_iter == 1:
            ax.set_xlim(0.5, 1.5)
        else:
            ax.set_xlim(1, n_iter)
        fig.tight_layout()
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved frequency iteration plot: {png_path}")
    except Exception as exc:
        print(f"Warning: failed to save frequency iteration plot ({exc}).")


def run_topopt_from_json(json_input: str | dict) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Run topology optimization for ourApproach from a JSON config.

    Parameters
    ----------
    json_input : str or dict
        Path to JSON file, or an already-decoded config dict.

    Returns
    -------
    x : (nelx*nely,) float array
        Final physical density field.
    omega : (3,) float array
        First three circular frequencies [rad/s].
    t_iter : float
        Average time per optimization iteration [s].
    n_iter : int
        Number of iterations executed.
    """
    cfg = load_json_config(json_input)

    # ---- Required fields ----
    L = req_num(cfg, ["domain", "size", "length"], "domain.size.length")
    H = req_num(cfg, ["domain", "size", "height"], "domain.size.height")
    nelx = req_int(cfg, ["domain", "mesh", "nelx"], "domain.mesh.nelx")
    nely = req_int(cfg, ["domain", "mesh", "nely"], "domain.mesh.nely")

    E0 = req_num(cfg, ["material", "E"], "material.E")
    nu = req_num(cfg, ["material", "nu"], "material.nu")
    rho0 = req_num(cfg, ["material", "rho"], "material.rho")
    E_min_ratio = req_num(cfg, ["void_material", "E_min_ratio"], "void_material.E_min_ratio")
    rho_min = req_num(cfg, ["void_material", "rho_min"], "void_material.rho_min")
    Emin = E0 * E_min_ratio

    approach_raw = req_str(cfg, ["optimization", "approach"], "optimization.approach").strip()
    approach = approach_raw.lower()
    if approach != "ourapproach":
        raise ValueError(
            f'run_topopt_from_json.py only supports "ourApproach" (got "{approach_raw}").'
        )

    volfrac = req_num(cfg, ["optimization", "volume_fraction"], "optimization.volume_fraction")
    penal = req_num(cfg, ["optimization", "penalization"], "optimization.penalization")
    move = req_num(cfg, ["optimization", "move_limit"], "optimization.move_limit")
    max_iters = req_int(cfg, ["optimization", "max_iters"], "optimization.max_iters")
    conv_tol = req_num(cfg, ["optimization", "convergence_tol"], "optimization.convergence_tol")

    filter_type = req_str(cfg, ["optimization", "filter", "type"], "optimization.filter.type").lower()
    filter_radius = req_num(cfg, ["optimization", "filter", "radius"], "optimization.filter.radius")
    radius_units = req_str(
        cfg, ["optimization", "filter", "radius_units"], "optimization.filter.radius_units"
    ).lower()

    optimizer_type = "OC"
    if has_field_path(cfg, ["optimization", "optimizer"]):
        optimizer_type = str(get_field_path(cfg, ["optimization", "optimizer"])).upper().strip()
        if optimizer_type not in {"OC", "MMA"}:
            raise ValueError(f'optimization.optimizer must be "OC" or "MMA" (got "{optimizer_type}").')

    # ---- Filter radius conversion ----
    dx = L / nelx
    if radius_units == "element":
        rmin_phys = filter_radius * dx
    elif radius_units == "physical":
        rmin_phys = filter_radius
    else:
        raise ValueError(
            f'optimization.filter.radius_units must be "element" or "physical" (got "{radius_units}").'
        )

    # ---- Boundary conditions ----
    supports = cfg.get("bc", {}).get("supports", [])
    extra_fixed_dofs = supports_to_fixed_dofs(supports, nelx, nely, L, H)

    # ---- Passive regions ----
    pas_s, pas_v = parse_passive_regions(cfg, nelx, nely, L, H)
    print(f"Passive regions: {pas_s.size} solid, {pas_v.size} void elements")

    # ---- Load cases ----
    load_cases = None
    if has_field_path(cfg, ["domain", "load_cases"]):
        raw_lc = get_field_path(cfg, ["domain", "load_cases"])
        load_cases = validate_load_cases(raw_lc, "domain.load_cases")

    # ---- Filter type (ft) ----
    filter_map = {"sensitivity": 0, "density": 1}
    ft = filter_map.get(filter_type, None)
    if ft is None:
        raise ValueError(
            f'optimization.filter.type must be "sensitivity" or "density" (got "{filter_type}").'
        )

    use_heaviside = False
    if has_field_path(cfg, ["optimization", "filter", "heaviside"]):
        use_heaviside = parse_bool(
            get_field_path(cfg, ["optimization", "filter", "heaviside"]),
            "optimization.filter.heaviside",
        )

    # ---- Optional flags ----
    harmonic_normalize = True
    if has_field_path(cfg, ["optimization", "harmonic_normalize"]):
        harmonic_normalize = parse_bool(
            get_field_path(cfg, ["optimization", "harmonic_normalize"]),
            "optimization.harmonic_normalize",
        )

    semi_baseline = "solid"
    if has_field_path(cfg, ["optimization", "semi_harmonic_baseline"]):
        semi_baseline = str(get_field_path(cfg, ["optimization", "semi_harmonic_baseline"])).lower()

    semi_rho_src = "x"
    if has_field_path(cfg, ["optimization", "semi_harmonic_rho_source"]):
        semi_rho_src = str(get_field_path(cfg, ["optimization", "semi_harmonic_rho_source"])).lower()

    visualize_live = True
    if has_field_path(cfg, ["postprocessing", "visualize_live"]):
        visualize_live = parse_bool(
            get_field_path(cfg, ["postprocessing", "visualize_live"]),
            "postprocessing.visualize_live",
        )

    save_frq_iter = False
    if has_field_path(cfg, ["postprocessing", "save_frequency_iterations"]):
        save_frq_iter = parse_bool(
            get_field_path(cfg, ["postprocessing", "save_frequency_iterations"]),
            "postprocessing.save_frequency_iterations",
        )

    # ---- Build run_cfg ----
    run_cfg = {
        "E0": E0,
        "Emin": Emin,
        "nu": nu,
        "rho0": rho0,
        "rho_min": rho_min,
        "move": move,
        "conv_tol": conv_tol,
        "max_iters": max_iters,
        "optimizer": optimizer_type,
        "extra_fixed_dofs": extra_fixed_dofs,
        "pas_s": pas_s,
        "pas_v": pas_v,
        "harmonic_normalize": harmonic_normalize,
        "semi_harmonic_baseline": semi_baseline,
        "semi_harmonic_rho_source": semi_rho_src,
        "visualize_live": visualize_live,
        "save_frq_iterations": save_frq_iter,
        "use_heaviside": use_heaviside,
    }
    if load_cases is not None:
        run_cfg["load_cases"] = load_cases

    # ---- Call solver ----
    solver_dir = os.path.join(_REPO_ROOT, "ourApproach", "Python")
    if solver_dir not in sys.path:
        sys.path.insert(0, solver_dir)
    from topopt_freq import topopt_freq

    x_out, f_hz, t_iter, n_iter, info = topopt_freq(
        nelx, nely, volfrac, penal, rmin_phys, ft, L, H, run_cfg
    )

    if save_frq_iter:
        freq_hist = info.get("freq_iter_omega") if isinstance(info, dict) else None
        if freq_hist is None:
            print(
                "Warning: postprocessing.save_frequency_iterations=true, "
                f'but no iteration history was returned by "{approach_raw}".'
            )
        else:
            _save_frequency_iteration_plot(freq_hist, approach_raw, nelx, nely, _REPO_ROOT)

    omega = np.full(3, np.nan, dtype=float)
    valid = np.isfinite(f_hz)
    omega[valid] = 2.0 * np.pi * f_hz[valid]
    return x_out, omega, t_iter, n_iter


# =============================================================================
# CLI entry point
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_topopt_from_json.py <path/to/config.json>")
        sys.exit(1)
    json_path = sys.argv[1]
    x, omega, t_iter, n_iter = run_topopt_from_json(json_path)
    print(f"\nDone. n_iter={n_iter}, t_iter={t_iter:.3f}s")
    print(f"omega = {omega} rad/s")
    print(f"f     = {omega / (2*np.pi)} Hz")
