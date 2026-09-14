#!/usr/bin/env python3
"""nmp_make_lock.py smoke|campaign -- write <MODE>_LOCK.json and <MODE>_LOCK.json.sha256.

The lock pins everything the observation hooks and the launcher re-check:
HEAD, branch, +impl tree, SOURCE_MANIFEST bytes, the production preset, the nine
frozen config hashes (taken from the frozen gate files and required to equal the
Part 1 identity check), the runner file both as committed and as edited, the
unified diff of that edit, olhoffcurrent_run.m and the three hook lines, every
script of this study, the preregistration, and the output locations.
Read-only on the repository.
"""
import hashlib, json, os, re, subprocess, sys, datetime

REPO = "/Users/piotrek/Programming/topOpt4freqMax"
D = os.path.join(REPO, "analysis/OlhoffCurrent/diagnostics/nine_mesh_pedersen_production")
GATE = os.path.join(REPO, "analysis/OlhoffCurrent/diagnostics/postmerge_campaign_gate")
RUNNER_REL = "examples/Performance/performance_comparison.m"
RUN_REL = "analysis/OlhoffCurrent/olhoffcurrent_run.m"
MESHES = ["160x20", "240x30", "320x40", "400x50", "480x60", "560x70", "640x80", "720x90", "800x100"]


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    with open(p, "rb") as f:
        return sha_bytes(f.read())


def git(*args):
    return subprocess.run(["git", "--no-pager", "-C", REPO, *args], check=True,
                          capture_output=True).stdout


def main(mode):
    assert mode in ("smoke", "campaign")
    ident = json.load(open(os.path.join(GATE, "CAMPAIGN_IDENTITY.json")))
    nmc = json.load(open(os.path.join(GATE, "NINE_MESH_CONFIGS.json")))
    chk = json.load(open(os.path.join(D, "evidence/IDENTITY_CHECK.json")))
    assert chk["verdict"] == "NINE_MESH_CAMPAIGN_IDENTITY_PASS", chk["verdict"]

    hashes = {}
    for m in MESHES:
        a = ident["nine_config_hashes"][m]
        b = [c for c in nmc["configs"] if c["mesh"] == m][0]["config_hash"]
        c = [x for x in chk["meshes"] if x["mesh"] == m][0]["hash_solve_route"]
        assert a == b == c, (m, a, b, c)
        hashes[m] = a

    runner_abs = os.path.join(REPO, RUNNER_REL)
    committed = git("show", "HEAD:" + RUNNER_REL)
    edited = open(runner_abs, "rb").read()
    patch = git("diff", "HEAD", "--", RUNNER_REL)
    assert patch, "runner is not edited"
    tracked = [x for x in git("diff", "--name-only", "HEAD").decode().split("\n") if x]
    assert tracked == [RUNNER_REL], tracked
    m = re.search(rb"^cfg\.runLabel\s*=\s*'([^']+)';", edited, re.M)
    label = m.group(1).decode()
    out_root = os.path.join(REPO, "examples/Performance/conference_benchmark", label)

    patch_path = os.path.join(D, "evidence", f"{mode}_runner_edit.patch")
    with open(patch_path, "wb") as f:
        f.write(patch)

    scripts = sorted(os.listdir(os.path.join(D, "scripts")))
    prereg = os.path.join(D, "PREREGISTRATION.md")
    lock = {
        "schema": "nmp_lock/1",
        "mode": mode,
        "created_local": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo_abs": REPO,
        "head": git("rev-parse", "HEAD").decode().strip(),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
        "upstream_commit": ident["implementation"]["upstream_commit"],
        "impl_tree_sha256": ident["implementation"]["impl_tree_sha256"],
        "source_manifest_sha256": ident["implementation"]["source_manifest_sha256"],
        "preset": ident["production"]["preset"],
        "upstream_preset": ident["production"]["upstream_preset"],
        "meshes": MESHES,
        "nine_config_hashes": hashes,
        "frozen_sources": {
            "campaign_identity": os.path.relpath(os.path.join(GATE, "CAMPAIGN_IDENTITY.json"), REPO),
            "campaign_identity_sha256": sha_file(os.path.join(GATE, "CAMPAIGN_IDENTITY.json")),
            "nine_mesh_configs": os.path.relpath(os.path.join(GATE, "NINE_MESH_CONFIGS.json"), REPO),
            "nine_mesh_configs_sha256": sha_file(os.path.join(GATE, "NINE_MESH_CONFIGS.json")),
            "identity_check": "analysis/OlhoffCurrent/diagnostics/nine_mesh_pedersen_production/evidence/IDENTITY_CHECK.json",
            "identity_check_sha256": sha_file(os.path.join(D, "evidence/IDENTITY_CHECK.json")),
            "identity_check_verdict": chk["verdict"],
        },
        "runner": {
            "path": RUNNER_REL,
            "committed_blob_sha1": git("rev-parse", "HEAD:" + RUNNER_REL).decode().strip(),
            "committed_sha256": sha_bytes(committed),
            "edited_sha256": sha_bytes(edited),
            "edit_patch": os.path.relpath(patch_path, REPO),
            "edit_patch_sha256": sha_bytes(patch),
            "run_label": label,
        },
        "allowed_tracked_changes": [RUNNER_REL],
        "olhoffcurrent_run": {
            "path": RUN_REL,
            "sha256": sha_file(os.path.join(REPO, RUN_REL)),
            "committed_blob_sha1": git("rev-parse", "HEAD:" + RUN_REL).decode().strip(),
            "hook_lines": {"precheck": 124, "tap": 128, "postcheck": 249},
        },
        "output_root_abs": out_root,
        "run_root_abs": os.path.join(out_root, "runs"),
        "logs_dir_abs": os.path.join(D, "logs"),
        "matlab_cwd_abs": os.path.join(D, "logs", f"matlab_cwd_{mode}"),
        "matlab_binary": "/Applications/MATLAB_R2025b.app/bin/matlab",
        "preregistration": {
            "path": os.path.relpath(prereg, REPO),
            "sha256": sha_file(prereg),
        },
        "scripts": [{"path": os.path.relpath(os.path.join(D, "scripts", s), REPO),
                     "sha256": sha_file(os.path.join(D, "scripts", s))}
                    for s in scripts if not s.startswith(".")],
    }
    assert lock["head"] == ident["repository"]["merged_head"]
    lock_path = os.path.join(D, f"{mode.upper()}_LOCK.json")
    assert not os.path.exists(lock_path), f"{lock_path} exists; locks are never rewritten"
    with open(lock_path, "w") as f:
        json.dump(lock, f, indent=1)
        f.write("\n")
    h = sha_file(lock_path)
    with open(lock_path + ".sha256", "w") as f:
        f.write(f"{h}  {os.path.basename(lock_path)}\n")
    print(lock_path, h)


if __name__ == "__main__":
    main(sys.argv[1])
