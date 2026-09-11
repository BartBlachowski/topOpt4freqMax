#!/bin/sh
# tr_seal -- write FINAL_SHA256.txt over every artifact of this study.
# Run LAST.  Paths are study-relative for study files and repo-relative for the
# git-ignored raw trajectory, matching the convention the finalization gate
# resolves.
set -e
STUDY="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$STUDY/../.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
cd "$STUDY"

{
  echo "FINAL_SHA256 -- three_rung_promotion_validation_retry1"
  echo "=============================================================================="
  echo "generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "branch    $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  echo "HEAD      $(git -C "$REPO" rev-parse HEAD)"
  echo "implTree  $(python3 -c "import json,sys; print(json.load(open('METRICS.json'))['run']['implTree'])")"
  echo ""
  echo "scientific runs 1   (C320x40_three_rung, CONVERGED @352)"
  echo ""
  echo "--- study files (study-relative) ---"
  find . -type f ! -name 'FINAL_SHA256.txt' ! -name '.DS_Store' \
    | sed 's|^\./||' | LC_ALL=C sort \
    | while IFS= read -r f; do shasum -a 256 "$f"; done
  echo ""
  echo "--- raw evidence (repo-relative, git-ignored) ---"
  T="analysis/OlhoffCurrent/evidence/three_rung_promotion_validation_retry1/C320x40_three_rung_trajectory.mat"
  if [ -f "$REPO/$T" ]; then
    printf '%s  %s   (%s bytes)\n' \
      "$(shasum -a 256 "$REPO/$T" | cut -d' ' -f1)" "$T" \
      "$(stat -f %z "$REPO/$T")"
  else
    printf 'MISSING                                                           %s\n' "$T"
  fi
} > FINAL_SHA256.txt

echo "wrote $STUDY/FINAL_SHA256.txt ($(wc -l < FINAL_SHA256.txt | tr -d ' ') lines)"
