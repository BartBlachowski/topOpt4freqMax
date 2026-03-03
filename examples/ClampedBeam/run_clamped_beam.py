"""
Run the Clamped Beam frequency-maximization example.

Configuration: BeamTopOptFreq.json
    - 8 x 1 m beam, clamped at both ends (full vertical-line BCs)
    - 400 x 50 mesh, volfrac=0.5, MMA optimizer, sensitivity filter
    - Semi-harmonic load cases (mode 1 active, mode 2 zero-weight)
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.Python.run_topopt_from_json import run_topopt_from_json

JSON_FILE = os.path.join(_HERE, "BeamTopOptFreq.json")

if __name__ == "__main__":
    x, omega, t_iter, n_iter = run_topopt_from_json(JSON_FILE)
    print(f"\nClamped Beam done.")
    print(f"  Iterations : {n_iter}")
    print(f"  Time/iter  : {t_iter:.3f} s")
    print(f"  omega (rad/s): {omega}")
    print(f"  f     (Hz)   : {omega / 6.283185307:.6g}")
