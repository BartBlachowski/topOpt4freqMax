"""
Benchmark runner for Olhoff & Du (2014) 2D beam cases.
Runs CC, CS, SS boundary conditions and reports initial/final eigenfrequencies.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from topFreqOptimization_MMA import topFreqOptimization_MMA


def _print_block(name, data):
    omega = data["omega"]
    n = min(3, len(omega))
    vals = " ".join(f"{w:8.2f}" for w in omega[:n])
    print(f"{name} eigenfreqs (rad/s): {vals}")


def main():
    base_cfg = dict(
        L=8,
        H=1,
        nelx=240,
        nely=30,
        volfrac=0.5,
        penal=3.0,
        maxiter=300,
        J=3,
        E0=1e7,
        rho0=1.0,
        rho_min=1e-6,
        nu=0.3,
        t=1.0,
    )
    base_cfg["rmin"] = 2 * base_cfg["L"] / base_cfg["nelx"]
    base_cfg["Emin"] = max(1e-6 * base_cfg["E0"], 1e-3)
    opts = dict(doDiagnostic=True, diagnosticOnly=False, diagModes=5)

    paper = {
        "CC": dict(init=146.1, opt=456.4),
        "CS": dict(init=104.1, opt=288.7),
        "SS": dict(init=68.7, opt=174.7),
    }
    cases = [
        dict(code="CC", label="Clamped-Clamped"),
        dict(code="CS", label="Clamped-Simply"),
        dict(code="SS", label="Simply-Simply"),
    ]

    results = []
    for idx, case in enumerate(cases):
        cfg = dict(base_cfg)
        cfg["supportType"] = case["code"]

        print(f"\n================== {case['label']} ==================")
        print(f"Paper: init={paper[case['code']]['init']:.1f}, opt={paper[case['code']]['opt']:.1f}")
        print("=========================================")

        t0 = time.time()
        omega_best, x_phys_best, diag_out = topFreqOptimization_MMA(cfg, opts)
        elapsed = time.time() - t0

        results.append(
            dict(
                code=case["code"],
                label=case["label"],
                diag=diag_out,
                omega=omega_best,
                xPhys=x_phys_best,
                time=elapsed,
            )
        )

        print(f"\n--- {case['label']} Summary ---")
        _print_block("Initial", diag_out["initial"])
        _print_block("Final", diag_out["final"])
        print(f"Volume: {np.mean(x_phys_best):.4f} (target {cfg['volfrac']:.2f})")
        print(f"Grayness: {np.mean(4 * x_phys_best * (1 - x_phys_best)):.4f}")
        print(f"Time: {elapsed:.1f} sec")

        fig, ax = plt.subplots(figsize=(4, 1))
        fig.canvas.manager.set_window_title(f"Olhoff {case['code']} topology")
        ax.imshow(
            1 - x_phys_best.reshape(cfg["nely"], cfg["nelx"]),
            cmap="gray",
            aspect="equal",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.axis("off")
        ax.set_title(
            f"{case['code']}: omega1={omega_best:.1f} (paper: {paper[case['code']]['opt']:.1f})"
        )
        fig.tight_layout()

    print("\n\n========== FINAL BENCHMARK SUMMARY ==========")
    print("BC   | Init(code) | Init(paper) | Opt(code) | Opt(paper) | Improve")
    print("-----|------------|-------------|-----------|------------|--------")
    for result in results:
        p = paper[result["code"]]
        improv = (result["diag"]["final"]["omega"][0] / result["diag"]["initial"]["omega"][0] - 1) * 100
        print(
            f"{result['code']}  | {result['diag']['initial']['omega'][0]:10.1f} | "
            f"{p['init']:11.1f} | {result['diag']['final']['omega'][0]:9.1f} | "
            f"{p['opt']:10.1f} | {improv:5.0f}%"
        )
    print("==============================================")

    plt.show()


if __name__ == "__main__":
    main()
