#!/usr/bin/env python
"""Predictor-corrector budget: NSO (to omega1 >= threshold) + production corrector vs cold production.

Reads results/pred_<mesh>_<tag>_rho.mat (NSO predictor: wall_s, iterations, omega),
results/olhoff_corrector_<mesh>_pc_<tag>.mat (MATLAB corrector) and results/olhoff_ref_<mesh>.mat.
"""
import glob
import os
import re
import sys

import numpy as np
import scipy.io as sio

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main(mesh="160x20"):
    ref = sio.loadmat(os.path.join(RES, f"olhoff_ref_{mesh}.mat"), squeeze_me=True)
    rows = [dict(arm="cold production (baseline)", pred_it=0, pred_s=0.0, pred_w1=float(np.atleast_2d(ref["hist_omega"])[0, 0] if np.atleast_2d(ref["hist_omega"]).shape[0] > 1 else np.nan),
                 corr_it=int(ref["nOuter"]), corr_s=float(ref["wall_s"]), inner=int(np.sum(ref["hist_nInner"])),
                 w1=float(ref["omega"][0]), gap=float(100 * (ref["omega"][1] - ref["omega"][0]) / ref["omega"][0]),
                 Mnd=float(100 * ref["Mnd"]), status=str(ref["status"]))]
    for cp in sorted(glob.glob(os.path.join(RES, f"olhoff_corrector_{mesh}_pc_*.mat"))):
        tag = re.search(r"_pc_(.+)\.mat$", cp).group(1)
        pp = os.path.join(RES, f"pred_{mesh}_{tag}_rho.mat")
        if not os.path.isfile(pp):
            continue
        p = sio.loadmat(pp, squeeze_me=True); c = sio.loadmat(cp, squeeze_me=True)
        rows.append(dict(arm=f"NSO {tag} -> production", pred_it=int(p["iterations"]), pred_s=float(p["wall_s"]),
                         pred_w1=float(np.atleast_1d(p["omega"])[0]), corr_it=int(c["nOuter"]), corr_s=float(c["wall_s"]),
                         inner=int(np.sum(c["hist_nInner"])), w1=float(c["omega"][0]),
                         gap=float(100 * (c["omega"][1] - c["omega"][0]) / c["omega"][0]), Mnd=float(100 * c["Mnd"]),
                         status=str(c["status"])))
    hdr = "| arm | NSO it | NSO s | ω₁ handed over | corrector outer it | corrector inner it | corrector s | TOTAL s | final ω₁ | gap % | M_nd % | status |"
    print(hdr); print("|" + "---|" * 12)
    for r in rows:
        print(f"| {r['arm']} | {r['pred_it']} | {r['pred_s']:.0f} | {r['pred_w1']:.2f} | {r['corr_it']} | {r['inner']} | {r['corr_s']:.0f} | "
              f"**{r['pred_s'] + r['corr_s']:.0f}** | {r['w1']:.3f} | {r['gap']:.2f} | {r['Mnd']:.1f} | {r['status']} |")
    with open(os.path.join(RES, f"predictor_corrector_{mesh}.md"), "w") as fh:
        fh.write(hdr + "\n|" + "---|" * 12 + "\n")
        for r in rows:
            fh.write(f"| {r['arm']} | {r['pred_it']} | {r['pred_s']:.0f} | {r['pred_w1']:.2f} | {r['corr_it']} | {r['inner']} | {r['corr_s']:.0f} | "
                     f"{r['pred_s'] + r['corr_s']:.0f} | {r['w1']:.3f} | {r['gap']:.2f} | {r['Mnd']:.1f} | {r['status']} |\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
