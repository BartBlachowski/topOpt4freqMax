#!/usr/bin/env python3
"""Independent anchor, mode-identity, and stored precision-pair checks.

No optimizer is invoked.  Phase-2F is imported only as a numerical FE engine;
all outputs are confined to the Phase-2G directory.  A dense LAPACK solve is
used as an independent eigensolver cross-check at the critical k=252 anchor.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase2g-matplotlib")
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
F2 = REPO / "analysis/iteration_efficiency_phase2f_evaluator_redesign"
sys.path.insert(0, str(F2 / "scripts"))
from modal_engine import exact_count_binary, interp, mesh_data, modes  # noqa: E402


def write_csv(name, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else ["status", "reason"]
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def topology_metrics(x, nx, ny):
    x = np.asarray(x, float).ravel()
    xb = exact_count_binary(x, .5)
    solid = xb.reshape(nx, ny).T.astype(bool)
    labels = np.zeros_like(solid, dtype=np.int32)
    sizes = []
    cid = 0
    for c in range(nx):
        for r in range(ny):
            if not solid[r, c] or labels[r, c]:
                continue
            cid += 1; count = 0; todo = deque([(r, c)]); labels[r, c] = cid
            while todo:
                rr, cc = todo.popleft(); count += 1
                for rn, cn in ((rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                    if 0 <= rn < ny and 0 <= cn < nx and solid[rn, cn] and labels[rn, cn] == 0:
                        labels[rn, cn] = cid; todo.append((rn, cn))
            sizes.append(count)
    support_rows = np.unique([ny // 2 - 1, ny // 2])
    left = set(labels[support_rows, 0]) - {0}; right = set(labels[support_rows, -1]) - {0}
    span = left & right
    required = next(iter(span)) if len(span) == 1 else 0
    detached = [s for i, s in enumerate(sizes, 1) if i != required]
    area = 8.0 / (nx * ny)
    vol_pass = abs(x.mean() - .5) / .5 <= .001
    topo_pass = len(span) == 1 and all(s * area < .01 for s in detached)
    return {
        "binary": xb, "n_solid": int(xb.sum()), "target_n_solid": int(round(.5 * x.size)),
        "binary_volume": float(xb.mean()), "raw_volume": float(x.mean()), "volume_pass": bool(vol_pass),
        "n_components": cid, "required_connected": len(span) == 1, "required_component_label": required,
        "component_sizes": sizes, "detached_components": len(detached),
        "max_detached_elements": max(detached, default=0), "topology_pass": bool(topo_pass),
        "hard_gate_pass": bool(vol_pass and topo_pass), "labels": labels,
    }


def modal_diag(r, tau=.1):
    low = r["zeff"] <= tau
    vke = r["ke_n"][low].sum(axis=0)
    vse = r["se_n"][low].sum(axis=0)
    dwp = r["ke_n"].T @ r["zeff"]
    ipr = (r["ke_n"] ** 2).sum(axis=0)
    return vke, vse, dwp, ipr


def first_struct(r):
    vke, vse, dwp, ipr = modal_diag(r)
    votes = (vke < .5).astype(int) + (vse < .5).astype(int) + (dwp > .5).astype(int)
    ix = np.flatnonzero(votes == 3)
    j = int(ix[0]) if ix.size else -1
    return j, vke, vse, dwp, ipr


def load_state(mesh, state):
    nx, ny = map(int, mesh.split("x"))
    with h5py.File(REPO / f"examples/Performance/final_campaign/raw/olhoff/s1_{mesh}.mat", "r") as f:
        x = np.float64(f["res/rho_snapshots"][state])
    return np.clip(x, 0, 1), nx, ny


def dense_crosscheck(x, nx, ny, nm=14):
    md = mesh_data(nx, ny); ee, rr, _ = interp(x, "E2", "eq4a")
    K = sp.coo_matrix(((md["KEv"][None, :] * ee[:, None]).ravel(), (md["rows"], md["cols"])),
                      shape=(md["ndof"], md["ndof"])).tocsr()
    M = sp.coo_matrix(((md["MEv"][None, :] * rr[:, None]).ravel(), (md["rows"], md["cols"])),
                      shape=(md["ndof"], md["ndof"])).tocsr()
    free = md["free"]
    kd = ((K + K.T) * .5)[free][:, free].toarray()
    mdense = ((M + M.T) * .5)[free][:, free].toarray()
    vals = la.eigh(kd, mdense, eigvals_only=True, subset_by_index=[0, nm - 1], driver="gvx")
    return np.sqrt(np.maximum(vals, 0))


def anchor_rows(do_dense=True):
    z = np.load(F2 / "scripts/survey.npz")
    gate = np.load(F2 / "scripts/gate_full.npz")
    taus = z["TAUS"]; ti = int(np.argmin(abs(taus - .1)))
    configs = [k[:-6] for k in z.files if k.endswith("|omega")]

    # Locate an observed >12 case without assuming its identity.
    escalated = []
    for cfg in configs:
        if not (cfg.endswith("|E1|linear") or cfg.endswith("|E2|eq4a") or cfg.endswith("|E3|eq4a")):
            continue
        for k, n in zip(z[f"{cfg}|k"], z[f"{cfg}|nmodes"]):
            if n > 12:
                escalated.append((cfg, int(k), int(n)))
    first_esc = min(escalated, key=lambda a: (a[1], a[0])) if escalated else None

    # rho~=0.1 parking state, searched on the complete 160x20 history.
    with h5py.File(REPO / "examples/Performance/final_campaign/raw/olhoff/s1_160x20.mat", "r") as f:
        X = f["res/rho_snapshots"][()]
    parked_counts = np.sum(np.abs(np.float64(X) - .1) <= 2e-7, axis=1)
    park_k = int(np.argmax(parked_counts[1:]) + 1)

    anchors = [
        ("early_gray", "160x20", 1),
        ("rho_0p1_parking", "160x20", park_k),
        ("known_void_mode", "160x20", 252),
        ("simple_well_behaved", "160x20", 1600),
        ("severe_binary_near_mechanism", "400x50", 833),
        ("worst_binary_ratio", "160x20", 639),
        ("late_converged_like", "400x50", 1577),
    ]
    if first_esc:
        anchors.append(("first_observed_gt12_escalation", first_esc[0].split("|")[0], first_esc[1]))

    rows = []
    for kind, mesh, state in anchors:
        x, nx, ny = load_state(mesh, state)
        top = topology_metrics(x, nx, ny)
        nmode = 24 if kind == "first_observed_gt12_escalation" else 14
        for model, law, cand in (("E1", "linear", "C"), ("E2", "eq4", "A"),
                                  ("E2", "eq4a", "B/C"), ("E3", "eq4a", "B/C")):
            r = modes(x, nx, ny, model, law, k=nmode)
            j, vke, vse, dwp, ipr = first_struct(r)
            rows.append({
                "anchor": kind, "mesh": mesh, "state": state, "candidate_context": cand,
                "density_representation": "gray", "evaluator": model, "mass_law": law,
                "modes_computed": nmode, "algebraic_lowest_omega": float(r["omega"][0]),
                "first_structural_ordinal": j + 1 if j >= 0 else -1,
                "selected_structural_omega": float(r["omega"][j]) if j >= 0 else float("nan"),
                "selected_voidKE": float(vke[j]) if j >= 0 else float("nan"),
                "selected_voidSE": float(vse[j]) if j >= 0 else float("nan"),
                "selected_density_participation": float(dwp[j]) if j >= 0 else float("nan"),
                "selected_IPR": float(ipr[j]) if j >= 0 else float("nan"),
                "rho_0p1_parked_elements": int(np.sum(np.abs(x - .1) <= 2e-7)),
                "hard_gate_pass": top["hard_gate_pass"], "topology_pass": top["topology_pass"],
                "binary_exact_count": top["n_solid"], "binary_components": top["n_components"],
                "binary_detached_components": top["detached_components"],
                "source": "PHASE2G_RECOMPUTATION_USING_READ_ONLY_FE_ENGINE",
            })
        xb = top["binary"]
        rb = modes(xb, nx, ny, "E2", "eq4a", k=6)
        j, vke, vse, dwp, ipr = first_struct(rb)
        rows.append({
            "anchor": kind, "mesh": mesh, "state": state, "candidate_context": "D",
            "density_representation": "exact_count_binary", "evaluator": "E2", "mass_law": "eq4a",
            "modes_computed": 6, "algebraic_lowest_omega": float(rb["omega"][0]),
            "first_structural_ordinal": j + 1 if j >= 0 else -1,
            "selected_structural_omega": float(rb["omega"][j]) if j >= 0 else float("nan"),
            "selected_voidKE": float(vke[j]) if j >= 0 else float("nan"),
            "selected_voidSE": float(vse[j]) if j >= 0 else float("nan"),
            "selected_density_participation": float(dwp[j]) if j >= 0 else float("nan"),
            "selected_IPR": float(ipr[j]) if j >= 0 else float("nan"),
            "rho_0p1_parked_elements": int(np.sum(np.abs(x - .1) <= 2e-7)),
            "hard_gate_pass": top["hard_gate_pass"], "topology_pass": top["topology_pass"],
            "binary_exact_count": top["n_solid"], "binary_components": top["n_components"],
            "binary_detached_components": top["detached_components"],
            "source": "PHASE2G_RECOMPUTATION_USING_READ_ONLY_FE_ENGINE",
        })

    dense = {"performed": False}
    if do_dense:
        x, nx, ny = load_state("160x20", 252)
        t0 = time.time()
        wd = dense_crosscheck(x, nx, ny, 14)
        wa = modes(x, nx, ny, "E2", "eq4a", k=14)["omega"]
        dense = {"performed": True, "mesh": "160x20", "state": 252,
                 "solver": "dense scipy.linalg.eigh generalized LAPACK gvx, no shift/no Arnoldi",
                 "dense_omega": wd.tolist(), "sparse_omega": wa.tolist(),
                 "max_relative_difference": float(np.max(abs(wd - wa) / wd)), "seconds": time.time() - t0}
    if do_dense or not (OUT / "dense_crosscheck.json").exists():
        (OUT / "dense_crosscheck.json").write_text(json.dumps(dense, indent=2) + "\n")
    return rows


def binary_figure_and_details():
    x, nx, ny = load_state("400x50", 833)
    top = topology_metrics(x, nx, ny); xb = top["binary"]
    rb = modes(xb, nx, ny, "E2", "eq4a", k=6, return_vectors=True)
    rg = modes(x, nx, ny, "E2", "eq4a", k=12)
    jg, vg, *_ = first_struct(rg)
    vb, seb, dwpb, iprb = modal_diag(rb)
    labels = top["labels"]
    req = top["required_component_label"]
    vals = np.sort(x)[::-1]; nsolid = int(round(.5 * x.size))
    cut = vals[nsolid-1]; gap = vals[nsolid-1] - vals[nsolid]
    ntied = int(np.sum(x == cut))
    rows = []
    for j in range(6):
        ke = rb["ke_n"][:, j].reshape(nx, ny).T
        required_share = float(ke[labels == req].sum())
        detached_share = float(sum(ke[labels == i].sum() for i in range(1, top["n_components"] + 1) if i != req))
        rows.append({
            "mesh": "400x50", "state": 833, "mode_ordinal": j + 1, "binary_omega": float(rb["omega"][j]),
            "binary_voidKE": float(vb[j]), "binary_voidSE": float(seb[j]),
            "binary_density_participation": float(dwpb[j]), "binary_IPR": float(iprb[j]),
            "gray_structural_omega": float(rg["omega"][jg]), "gray_structural_ordinal": jg + 1,
            "binary_exact_solid_count": top["n_solid"], "binary_volume": top["binary_volume"],
            "cut_density": float(cut), "cut_gap": float(gap), "cut_ties": ntied,
            "support_connectivity": top["required_connected"], "n_components": top["n_components"],
            "detached_components": top["detached_components"], "hard_gate_pass": top["hard_gate_pass"],
            "component_sizes_elements": ";".join(map(str, top["component_sizes"])),
            "required_spanning_component_KE_share": required_share,
            "detached_solid_components_KE_share": detached_share,
            "interpretation": "PROJECTION_CREATED_DETACHED_SOLID_ISLAND_MECHANISM_NOT_VOID_LOCALIZED_NOT_EIGENSOLVER_ERROR",
        })
    write_csv("BINARY_NEAR_MECHANISM_MODES.csv", rows)

    fig, ax = plt.subplots(2, 4, figsize=(14, 5.5), constrained_layout=True)
    ax[0, 0].imshow(x.reshape(nx, ny).T, origin="lower", cmap="gray_r", vmin=0, vmax=1, aspect="auto")
    ax[0, 0].set_title("actual gray density")
    ax[1, 0].imshow(xb.reshape(nx, ny).T, origin="lower", cmap="gray_r", vmin=0, vmax=1, aspect="auto")
    ax[1, 0].set_title("exact-count binary topology")
    for j in range(6):
        a = ax[j // 3, 1 + j % 3]
        en = rb["ke_n"][:, j].reshape(nx, ny).T
        im = a.imshow(np.log10(np.maximum(en, 1e-12)), origin="lower", cmap="inferno", vmin=-10, vmax=-1, aspect="auto")
        a.set_title(f"binary mode {j+1}: omega={rb['omega'][j]:.2f}\nvoidKE={vb[j]:.1e}")
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax[:, 1:].ravel().tolist(), label="log10 element kinetic-energy share", shrink=.8)
    fig.savefig(OUT / "figures/binary_near_mechanism_400x50_k833.png", dpi=180)
    plt.close(fig)


def mode_identity():
    xall_path = REPO / "examples/Performance/final_campaign/raw/olhoff/s1_160x20.mat"
    gate = np.load(F2 / "scripts/gate_full.npz")
    hk = {int(k): bool(v) for k, v in zip(gate["160x20|GATE2|k"], gate["160x20|GATE2|hard"])}
    rows = []
    with h5py.File(xall_path, "r") as f:
        X = f["res/rho_snapshots"]
        prev_u = prev_sel = prev_omega = None
        for k in range(1, X.shape[0]):
            x = np.clip(np.float64(X[k]), 0, 1)
            r = modes(x, 160, 20, "E2", "eq4a", k=12, return_vectors=True)
            sel, vke, vse, dwp, ipr = first_struct(r)
            if prev_u is not None and sel >= 0 and prev_sel >= 0:
                a = prev_u[:, prev_sel]; b = r["U"]
                mac = (a @ b) ** 2 / ((a @ a) * np.einsum("ij,ij->j", b, b))
                best = int(np.argmax(mac))
                gap_below = ((r["omega"][sel] - r["omega"][sel-1]) / r["omega"][sel]) if sel > 0 else float("nan")
                rows.append({
                    "mesh": "160x20", "evaluator": "E2", "mass_law": "eq4a", "state": k,
                    "hard_gate_pass": hk.get(k, False), "selected_ordinal": sel + 1,
                    "previous_selected_ordinal": prev_sel + 1, "ordinal_changed": sel != prev_sel,
                    "selected_omega": float(r["omega"][sel]),
                    "relative_frequency_step": abs(float(r["omega"][sel]) - prev_omega) / prev_omega,
                    "MAC_to_previous_selected": float(mac[sel]), "best_MAC_partner_ordinal": best + 1,
                    "best_MAC": float(mac[best]), "selected_is_best_MAC_partner": best == sel,
                    "gap_to_rejected_mode_below": gap_below,
                    "selected_voidKE": float(vke[sel]), "classification_margin": float(.5 - vke[sel]),
                    "ambiguous_transition": bool(mac[sel] < .5 and abs(float(r["omega"][sel]) - prev_omega) / prev_omega > .02),
                    "MAC_note": "EUCLIDEAN_MAC_DIAGNOSTIC;MODE_TRACKING_DOES_NOT_DEFINE_SELECTION",
                })
            prev_u, prev_sel = r["U"], sel
            prev_omega = float(r["omega"][sel]) if sel >= 0 else float("nan")
            if k % 400 == 0: print(f"MAC {k}/{X.shape[0]-1}", flush=True)
    write_csv("MODE_IDENTITY_AUDIT.csv", rows)
    return rows


def precision_pairs():
    qdir = REPO / "analysis/iteration_efficiency_phase2b_precision/qualification_runs"
    cases = [("gray_full_24x4_h200_paired_states.mat", 24, 4),
             ("s1_transition_96x12_h320_paired_states.mat", 96, 12)]
    rows = []
    for fn, nx, ny in cases:
        with h5py.File(qdir / fn, "r") as f:
            xd = f["x_double"][()]; xs = np.float64(f["x_single"][()]); it = f["pairIterations"][()].ravel().astype(int)
        for i, k in enumerate(it):
            td = topology_metrics(xd[i], nx, ny); ts = topology_metrics(xs[i], nx, ny)
            for model, law in (("E1", "linear"), ("E2", "eq4a"), ("E3", "eq4a")):
                rd = modes(xd[i], nx, ny, model, law, k=12)
                rs = modes(xs[i], nx, ny, model, law, k=12)
                jd, vd, *_ = first_struct(rd); js, vs, *_ = first_struct(rs)
                wd = float(rd["omega"][jd]) if jd >= 0 else float("nan")
                ws = float(rs["omega"][js]) if js >= 0 else float("nan")
                rows.append({
                    "source": fn, "mesh": f"{nx}x{ny}", "state": int(k), "evaluator": model,
                    "double_selected_ordinal": jd + 1 if jd >= 0 else -1,
                    "single_selected_ordinal": js + 1 if js >= 0 else -1,
                    "ordinal_invariant": jd == js, "double_selected_omega": wd, "single_selected_omega": ws,
                    "selected_omega_relative_change": abs(ws-wd)/wd if np.isfinite(wd) and wd > 0 else float("nan"),
                    "double_selected_voidKE": float(vd[jd]) if jd >= 0 else float("nan"),
                    "single_selected_voidKE": float(vs[js]) if js >= 0 else float("nan"),
                    "hard_gate_double": td["hard_gate_pass"], "hard_gate_single": ts["hard_gate_pass"],
                    "hard_gate_invariant": td["hard_gate_pass"] == ts["hard_gate_pass"],
                })
    write_csv("PRECISION_PAIR_AUDIT.csv", rows)
    return {
        "records": len(rows), "ordinal_changes": sum(r["ordinal_invariant"] is False for r in rows),
        "hard_gate_changes": sum(r["hard_gate_invariant"] is False for r in rows),
        "max_selected_omega_relative_change": max(r["selected_omega_relative_change"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-dense", action="store_true"); ap.add_argument("--skip-mac", action="store_true")
    args = ap.parse_args()
    rows = anchor_rows(not args.skip_dense); write_csv("ANCHOR_REPRODUCTION.csv", rows)
    binary_figure_and_details()
    mac = [] if args.skip_mac else mode_identity()
    ps = precision_pairs()
    result = {"anchors": len(rows), "mode_identity_transitions": len(mac), "precision": ps}
    (OUT / "verification_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
