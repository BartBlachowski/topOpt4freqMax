#!/usr/bin/env python3
"""nmp_finalize.py -- RUN_INDEX.json, DATA_MANIFEST.json, FINAL_SHA256.txt.  Read-only on raw data."""
import hashlib, json, os

REPO = "/Users/piotrek/Programming/topOpt4freqMax"
D = os.path.join(REPO, "analysis/OlhoffCurrent/diagnostics/nine_mesh_pedersen_production")
MESHES = ["160x20", "240x30", "320x40", "400x50", "480x60", "560x70", "640x80", "720x90", "800x100"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, REPO)


def walk(root, skip=()):
    out = []
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            if f == ".DS_Store":
                continue
            p = os.path.join(dp, f)
            if any(os.path.relpath(p, root).startswith(s) for s in skip):
                continue
            out.append(p)
    return out


lock = json.load(open(os.path.join(D, "CAMPAIGN_LOCK.json")))
out_root = lock["output_root_abs"]
smoke_root = json.load(open(os.path.join(D, "SMOKE_LOCK.json")))["output_root_abs"]
logs = os.path.join(D, "logs")
launch = json.load(open(os.path.join(logs, "campaign_LAUNCH.json")))
end = json.load(open(os.path.join(logs, "campaign_END.json")))
ev = json.load(open(os.path.join(D, "EVIDENCE.json")))
X = json.load(open(os.path.join(D, "evidence/EXTRACT.json")))
by = {r["mesh"]: r for r in (X["runs"] if isinstance(X["runs"], list) else [X["runs"]])}

index = {
    "schema": "nmp_run_index/1",
    "campaign": "definitive Olhoff nine-mesh Pedersen production campaign",
    "preset": lock["preset"], "head": lock["head"], "impl_tree_sha256": lock["impl_tree_sha256"],
    "authoritative_runner": lock["runner"]["path"],
    "launch_local": launch["start_local"], "end_local": end["end_local"], "exit_code": end["exit_code"],
    "command": launch["command"], "output_root": rel(out_root),
    "attempts": 1, "retries": [], "repeated_runs": [],
    "warmup": {"run_dir": rel(os.path.join(out_root, "runs", "warmup_48x6")),
               "note": "performance_comparison.m's own discarded 48x6, 5-outer warm-up; never an observation"},
    "runs": [],
    "non_scientific_pre_campaign": {
        "smoke_output_root": rel(smoke_root),
        "smoke_logs": [rel(os.path.join(logs, f)) for f in sorted(os.listdir(logs)) if f.startswith("smoke_")],
        "synthetic_endpath": "analysis/OlhoffCurrent/diagnostics/nine_mesh_pedersen_production/evidence/SYNTHETIC_ENDPATH.json",
        "note": "mechanics only; excluded from every table, fit and verdict"},
}
for i, m in enumerate(MESHES):
    r = by[m]
    rd = os.path.join(out_root, "runs", m)
    index["runs"].append({
        "order": i + 1, "mesh": m, "run_dir": rel(rd), "status": r["runner"]["status"],
        "precheck_when": r["precheck"]["when"], "precheck_pass": r["precheck"]["pass"],
        "tap_when": r["tap"]["when"], "postcheck_when": r["postcheck"]["when"], "postcheck_pass": r["postcheck"]["pass"],
        "config_hash": r["config_hash"], "rho_sha256": r["rho_sha256"],
        "files": {f: sha(os.path.join(rd, f)) for f in sorted(os.listdir(rd)) if f != ".DS_Store"},
        "topology_png": rel(os.path.join(out_root, "topologies", f"topology_olhoff_{m}.png"))})
json.dump(index, open(os.path.join(D, "RUN_INDEX.json"), "w"), indent=1)

raw = walk(out_root)
diag = [p for p in walk(D) if os.path.basename(p) not in ("FINAL_SHA256.txt", "DATA_MANIFEST.json")]
manifest = {
    "schema": "nmp_data_manifest/1",
    "raw_production_output_root": rel(out_root),
    "raw_production_files": [{"path": rel(p), "bytes": os.path.getsize(p), "sha256": sha(p)} for p in raw],
    "smoke_output_root": rel(smoke_root),
    "smoke_files": [{"path": rel(p), "bytes": os.path.getsize(p), "sha256": sha(p)} for p in walk(smoke_root)],
    "diagnostics_files": [{"path": rel(p), "bytes": os.path.getsize(p), "sha256": sha(p)} for p in diag],
    "read_only_inputs": [
        {"path": lock["frozen_sources"]["campaign_identity"], "sha256": lock["frozen_sources"]["campaign_identity_sha256"]},
        {"path": lock["frozen_sources"]["nine_mesh_configs"], "sha256": lock["frozen_sources"]["nine_mesh_configs_sha256"]},
        {"path": "analysis/OlhoffCurrent/diagnostics/scientific_delta_olhoff_migration/evaluations/sweep_verification.json",
         "sha256": sha(os.path.join(REPO, "analysis/OlhoffCurrent/diagnostics/scientific_delta_olhoff_migration/evaluations/sweep_verification.json"))},
    ] + [{"path": rel(p), "sha256": sha(p)} for p in
         [os.path.join(REPO, "analysis/OlhoffCurrent/diagnostics/scientific_delta_olhoff_migration/source_snapshot/+olhoff_6b08708/repro/results", "S" + m, "res.mat") for m in MESHES]],
    "verdicts": ev["verdicts"],
    "note": "Raw production outputs stay in the output root; only derived evidence (JSON, CSV, 800x100 grids, figures) lives here.",
}
json.dump(manifest, open(os.path.join(D, "DATA_MANIFEST.json"), "w"), indent=1)

lines = []
for p in sorted(walk(D)):
    if os.path.basename(p) == "FINAL_SHA256.txt":
        continue
    lines.append(f"{sha(p)}  {rel(p)}")
for p in raw:
    lines.append(f"{sha(p)}  {rel(p)}")
open(os.path.join(D, "FINAL_SHA256.txt"), "w").write("\n".join(lines) + "\n")
print(len(lines), "hashed lines;", len(raw), "raw production files")
