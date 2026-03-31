"""
Run the cantilever beam minimum-compliance example (elastic2D approach).

Configuration: CantileverBeam.json
    - 2 × 1 m cantilever, clamped at x=0
    - 60×30 mesh, volfrac=0.4, OC optimizer, sensitivity filter
    - Tip load: 1 N downward at (2.0, 0.5)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.Python.run_topopt_from_json import run_topopt_from_json

JSON_FILE = os.path.join(_HERE, "CantileverBeam.json")

if __name__ == "__main__":
    x, c_or_omega, t_iter, n_iter = run_topopt_from_json(JSON_FILE)
    print(f"\nCantilever done.")
    print(f"  Iterations : {n_iter}")
    print(f"  Time/iter  : {t_iter:.3f} s")
    print(f"  Compliance : {float(c_or_omega[0]):.6g}")
