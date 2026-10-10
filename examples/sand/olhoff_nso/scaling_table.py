#!/usr/bin/env python
"""Scaling table: production vs NSO per mesh (iterations, omega1, M_nd, wall, s/iter, time to reach
95 % / 98 % of that mesh's production omega1)."""
import glob
import json
import os
import re

import numpy as np
import scipy.io as sio

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def first_at(w, t, th):
    i = next((k for k, v in enumerate(w) if v >= th), None)
    return ("-", "-") if i is None else (i, f"{t[i]:.0f}")


rows = []
for rp in sorted(glob.glob(os.path.join(RES, "olhoff_ref_*x*.mat"))):
    mesh = re.search(r"olhoff_ref_(\d+x\d+)\.mat", rp).group(1)
    r = sio.loadmat(rp, squeeze_me=True)
    om = np.atleast_2d(r["hist_omega"]); om = om if om.shape[0] > om.shape[1] else om.T
    w1p = float(r["omega"][0]); tO = np.cumsum(np.asarray(r["hist_tOuter"], float).ravel())
    arms = [("production nested MMA", om[:, 0], tO, int(r["nOuter"]), str(r["status"]), float(r["omega"][1]),
             float(100 * r["Mnd"]), float(r["wall_s"]), float(np.sum(r["hist_tInner"])))]
    for jp in sorted(glob.glob(os.path.join(RES, f"nso_{mesh}_dt*_itn-1_euclid.json"))):
        d = json.load(open(jp)); s = d["summary"]; h = d["history"]
        w = np.asarray(h["omega1"]); t = np.linspace(0, s["wall_s"], len(w))
        arms.append((os.path.basename(jp)[:-5].replace(f"_{mesh}", ""), w, t, s["iterations"], "cap" if s["iterations"] >= 400 else "stop",
                     s["omega"][1], 100 * s["Mnd"], s["wall_s"], 0.0))
    for name, w, t, it, st, w2, mnd, wall, tin in arms:
        i95, t95 = first_at(w, t, 0.95 * w1p); i98, t98 = first_at(w, t, 0.98 * w1p)
        rows.append((mesh, name, it, st, f"{w[-1]:.2f}", f"{100*(w2-w[-1])/w[-1]:.1f}", f"{mnd:.1f}", f"{wall:.0f}", f"{wall/it:.2f}",
                     f"{100*tin/wall:.0f}" if tin else "-", i95, t95, i98, t98))
hdr = ["mesh", "arm", "iters", "status", "ω₁", "gap %", "M_nd %", "wall s", "s/iter", "inner MMA %", "it to 95% ω₁ᵖ", "s", "it to 98% ω₁ᵖ", "s"]
md = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in rows)
print(md); open(os.path.join(RES, "scaling.md"), "w").write(md)
