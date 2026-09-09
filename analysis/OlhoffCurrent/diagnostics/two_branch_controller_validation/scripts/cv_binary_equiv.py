#!/usr/bin/env python3
"""Binary-equivalence probe: does the MATLAB build change alter the trajectory?

The frozen preregistration and the C160/C320 candidate runs were produced under
MATLAB 25.2.0.2998904 (R2025b base).  The only R2025b install now present is
25.2.0.3042426 (Update 1).  This compares a rerun produced under Update 1 with
the committed run produced under the base build, column by column, at full
double precision.

It answers exactly one question and does not interpret it:
    are the two runs bitwise identical, and if not, where do they first diverge?
"""
import csv, json, sys, math
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent


def load(p):
    with open(p) as f:
        rows = list(csv.DictReader(f))
    return rows[0].keys(), rows


def cmp_csv(orig, new):
    ko, ro = load(orig)
    kn, rn = load(new)
    out = {"origRows": len(ro), "newRows": len(rn),
           "columnsIdentical": list(ko) == list(kn)}
    if not out["columnsIdentical"]:
        out["onlyInOrig"] = sorted(set(ko) - set(kn))
        out["onlyInNew"] = sorted(set(kn) - set(ko))
        return out
    n = min(len(ro), len(rn))
    firstDiff = None
    worst = {}
    for i in range(n):
        for k in ko:
            a, b = ro[i][k], rn[i][k]
            if a == b:
                continue
            try:
                fa, fb = float(a), float(b)
            except ValueError:
                fa = fb = None
            if fa is not None and math.isnan(fa) and math.isnan(fb):
                continue
            d = abs(fa - fb) if fa is not None else float("inf")
            rel = d / max(abs(fa), abs(fb), 1e-300) if fa is not None else float("inf")
            if k not in worst or rel > worst[k][0]:
                worst[k] = (rel, d, i + 1, a, b)
            if firstDiff is None:
                firstDiff = {"row": i + 1, "column": k, "orig": a, "new": b,
                             "absDiff": d, "relDiff": rel}
    out["bitwiseIdentical"] = firstDiff is None and len(ro) == len(rn)
    out["firstDifference"] = firstDiff
    out["nColumnsDiffering"] = len(worst)
    out["worstPerColumn"] = {k: {"relDiff": v[0], "absDiff": v[1], "firstRow": v[2],
                                 "orig": v[3], "new": v[4]}
                             for k, v in sorted(worst.items(),
                                                key=lambda kv: -kv[1][0])[:12]}
    return out


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "C160x20"
    orig = STUDY / "evidence" / "rerun_20260909" / f"{tag}_iterations.ORIG.csv"
    new = STUDY / "runs" / f"{tag}_iterations.csv"
    res = {"tag": tag, "origFile": str(orig.relative_to(STUDY)),
           "newFile": str(new.relative_to(STUDY)),
           "origMatlab": "25.2.0.2998904 (R2025b)",
           "newMatlab": "25.2.0.3042426 (R2025b) Update 1"}
    res.update(cmp_csv(orig, new))
    print(json.dumps(res, indent=1)[:4000])
    outf = STUDY / "evidence" / "rerun_20260909" / f"{tag}_binary_equiv.json"
    outf.write_text(json.dumps(res, indent=1))
    print("wrote", outf)
