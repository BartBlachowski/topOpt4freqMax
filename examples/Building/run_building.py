"""
Run the Building frequency-maximization example.

Configuration: BuildingTopOptFreq.json
    - 5 x 15 m building, clamped at base (horizontal line y=0)
    - 80 x 240 mesh, volfrac=0.3, MMA optimizer, sensitivity filter
    - Five rectangular passive solid regions (columns + floors)
    - Semi-harmonic load cases (mode 1 active, mode 2 zero-weight)
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.Python.run_topopt_from_json import run_topopt_from_json

JSON_FILE = os.path.join(_HERE, "BuildingTopOptFreq.json")

if __name__ == "__main__":
    x, omega, t_iter, n_iter = run_topopt_from_json(JSON_FILE)
    print(f"\nBuilding done.")
    print(f"  Iterations : {n_iter}")
    print(f"  Time/iter  : {t_iter:.3f} s")
    print(f"  omega (rad/s): {omega}")
    print(f"  f     (Hz)   : {omega / 6.283185307:.6g}")
