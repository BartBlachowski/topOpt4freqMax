"""Environment glue for the SAND heat-sink runners (Toebat & Feppon 2026).

Responsibilities, all side-effect free except for PATH / sys.path edits:

* make the authors' reference script ``tools/SAND/ex13_heat_SAND.py`` importable
  (it is kept byte-identical to the upstream GitLab file; nothing here edits it);
* put a ``FreeFem++`` binary on PATH (pyfreefem invokes it by name);
* select a non-interactive matplotlib backend when no display is available;
* report the versions of every component so a run is attributable.

Install notes are in ``examples/sand/README.md`` and ``tools/SAND/setup_sand_env.sh``.
"""
from __future__ import annotations

import glob
import importlib
import os
import platform
import shutil
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOOLS_SAND = os.path.join(REPO_ROOT, "tools", "SAND")
if TOOLS_SAND not in sys.path:          # at import time: sand_problems imports ex13_heat_SAND directly
    sys.path.insert(0, TOOLS_SAND)

_FREEFEM_CANDIDATES = (
    "/Applications/FreeFem++.app/Contents/ff-*/bin",
    os.path.expanduser("~/Applications/FreeFem++.app/Contents/ff-*/bin"),
    "/usr/local/bin",
    "/opt/homebrew/bin",
    "/usr/bin",
)


def ensure_freefem_on_path() -> str:
    """Return the path of a working ``FreeFem++`` executable, extending PATH if needed."""
    env_bin = os.environ.get("FREEFEM_BIN")
    if env_bin and os.path.isfile(env_bin):
        os.environ["PATH"] = os.path.dirname(env_bin) + os.pathsep + os.environ.get("PATH", "")
        return env_bin
    found = shutil.which("FreeFem++")
    if found:
        return found
    for pattern in _FREEFEM_CANDIDATES:
        for d in sorted(glob.glob(pattern), reverse=True):
            exe = os.path.join(d, "FreeFem++")
            if os.path.isfile(exe) and os.access(exe, os.X_OK):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                return exe
    raise RuntimeError(
        "FreeFem++ not found. Install FreeFEM (macOS arm64: the v4.15 Apple-Silicon .dmg, "
        "copied to /Applications/FreeFem++.app and de-quarantined), or set FREEFEM_BIN. "
        "See examples/sand/README.md."
    )


def freefem_version(exe: str) -> str:
    """Best-effort version string: the ``ff-x.y.z`` directory name, else the banner."""
    parts = exe.split(os.sep)
    for p in parts:
        if p.startswith("ff-"):
            return p[3:]
    try:
        out = subprocess.run([exe, "-nw", "-ne"], capture_output=True, text=True, timeout=30)
        for line in (out.stdout + out.stderr).splitlines():
            if "version" in line.lower():
                return line.strip()
    except Exception:  # noqa: BLE001 - diagnostics only
        pass
    return "unknown"


def setup(headless: bool | None = None) -> dict:
    """Prepare imports and return a metadata dict describing the environment."""
    if TOOLS_SAND not in sys.path:
        sys.path.insert(0, TOOLS_SAND)
    if headless is None:
        headless = not os.environ.get("DISPLAY") and sys.platform != "darwin" or os.environ.get("SAND_HEADLESS") == "1"
    if headless or os.environ.get("MPLBACKEND") is None:
        # Agg is always safe for file output; interactive viewing is not the runners' job.
        os.environ.setdefault("MPLBACKEND", "Agg")
    exe = ensure_freefem_on_path()

    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "scipy", "nullspace_optimizer", "pyfreefem", "pymedit",
                "qpalm", "piqp", "osqp", "cvxopt", "pypardiso"):
        try:
            m = importlib.import_module(mod)
            versions[mod] = str(getattr(m, "__version__", "present"))
        except Exception as exc:  # noqa: BLE001
            versions[mod] = f"MISSING ({type(exc).__name__})"
    versions["FreeFem++"] = freefem_version(exe)
    versions["FreeFem++_path"] = exe
    try:
        versions["git_commit"] = subprocess.run(
            ["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        versions["git_commit"] = "unknown"
    return versions


def available_qp_solvers() -> list[str]:
    """QP solvers importable in this environment, in the paper's Table 4 order."""
    table4 = [("osqp", "osqp"), ("qpalm", "qpalm"), ("piqp", "piqp"),
              ("mosek", "mosek"), ("gurobi", "gurobipy"), ("cplex", "cplex")]
    out = []
    for name, mod in table4:
        try:
            importlib.import_module(mod)
            out.append(name)
        except Exception:  # noqa: BLE001
            pass
    return out
