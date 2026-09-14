#!/usr/bin/env python3
"""Verify the extracted source snapshot byte-for-byte against the committed
git blobs at the audited source commit, and write a SHA-256 file manifest.

Read-only with respect to the source repository: it only runs `git ls-tree`.
"""
import hashlib, json, os, subprocess, sys

SRC = "/Users/piotrek/Programming/Matlab/Olhoff"
COMMIT = "6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(HERE, "source_snapshot", "+olhoff_6b08708")
OUT = os.path.join(HERE, "evaluations", "source_snapshot_manifest.json")


def git_blob_sha1(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def main():
    tree = subprocess.run(["git", "-C", SRC, "ls-tree", "-r", "-l", COMMIT],
                          check=True, capture_output=True, text=True).stdout
    committed = {}
    for line in tree.splitlines():
        meta, path = line.split("\t", 1)
        mode, typ, oid, size = meta.split()
        committed[path] = (oid, int(size))

    files, mismatches = [], []
    for root, _, names in os.walk(SNAP):
        for n in names:
            full = os.path.join(root, n)
            rel = os.path.relpath(full, SNAP)
            data = open(full, "rb").read()
            sha1 = git_blob_sha1(data)
            sha256 = hashlib.sha256(data).hexdigest()
            ok = rel in committed and committed[rel][0] == sha1
            if not ok:
                mismatches.append(rel)
            files.append({"path": rel, "size": len(data), "sha256": sha256,
                          "git_blob": sha1, "matches_commit": ok})
    files.sort(key=lambda f: f["path"])
    tree_lines = "".join(f"{f['path']}  {f['sha256']}\n" for f in files)
    out = {
        "source_repository": SRC,
        "source_commit": COMMIT,
        "procedure": "git archive --format=tar <commit> -- <subset paths>; tar -x",
        "n_files": len(files),
        "n_committed_files_total_at_commit": len(committed),
        "n_mismatch_vs_commit_blobs": len(mismatches),
        "mismatches": mismatches,
        "snapshot_tree_sha256": hashlib.sha256(tree_lines.encode()).hexdigest(),
        "files": files,
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("files", len(files), "mismatches", len(mismatches),
          "tree", out["snapshot_tree_sha256"])
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
