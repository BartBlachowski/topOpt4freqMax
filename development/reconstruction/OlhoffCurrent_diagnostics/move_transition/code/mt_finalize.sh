#!/bin/sh
# Hash manifest of every committed artifact of the move-transition study.
# Run from the repository root.  *.mat is intentionally excluded: it is
# gitignored raw solver state, fully reproducible from code/.
#
# The authoritative OlhoffCurrent source-tree hash is produced by
# olhoffcurrent_source_manifest and is recorded in METRICS.json; it is quoted
# here for convenience only.
BASE=analysis/OlhoffCurrent/diagnostics/move_transition
{
  echo "# FINAL_SHA256 -- move-transition experiment"
  echo "# generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# repository HEAD at freeze: $(git rev-parse HEAD)"
  echo "# OlhoffCurrent +impl tree (from source manifest, verified after the runs):"
  echo "#   c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c  (74 files, UNCHANGED)"
  echo
  find "$BASE" -type f ! -name '*.mat' ! -name 'FINAL_SHA256.txt' ! -name '.DS_Store' \
    | sort | xargs shasum -a 256
} > "$BASE/FINAL_SHA256.txt"
cat "$BASE/FINAL_SHA256.txt"
