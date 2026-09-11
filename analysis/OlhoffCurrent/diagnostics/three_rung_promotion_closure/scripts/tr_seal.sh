#!/bin/sh
# tr_seal -- write FINAL_SHA256.txt over every artifact of this closure study.
# Run LAST.  This study has no raw scientific evidence of its own: it executed
# zero optimization runs.
set -e
STUDY="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$STUDY/../../../.." && pwd)"
cd "$STUDY"
{
  echo "FINAL_SHA256 -- three_rung_promotion_closure"
  echo "=============================================================================="
  echo "generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "branch    $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  echo "HEAD      $(git -C "$REPO" rev-parse HEAD)"
  echo "implTree  $(python3 -c "import json;print(json.load(open('METRICS.json'))['implTree'])")"
  echo ""
  echo "scientific runs 0   nine-mesh runs 0   promoted: no"
  echo ""
  echo "--- study files (study-relative) ---"
  find . -type f ! -name 'FINAL_SHA256.txt' ! -name '.DS_Store' \
    | sed 's|^\./||' | LC_ALL=C sort \
    | while IFS= read -r f; do shasum -a 256 "$f"; done
  echo ""
  echo "--- reused immutable evidence (repo-relative; NOT produced here) ---"
  for T in \
    "analysis/OlhoffCurrent/diagnostics/three_rung_promotion_validation_retry1/METRICS.json" \
    "analysis/OlhoffCurrent/diagnostics/three_rung_promotion_validation_retry1/FINAL_SHA256.txt" \
    "analysis/OlhoffCurrent/evidence/three_rung_promotion_validation_retry1/C320x40_three_rung_trajectory.mat"
  do
    if [ -f "$REPO/$T" ]; then
      printf '%s  %s\n' "$(shasum -a 256 "$REPO/$T" | cut -d' ' -f1)" "$T"
    else
      printf 'MISSING                                                           %s\n' "$T"
    fi
  done
} > FINAL_SHA256.txt
echo "wrote $STUDY/FINAL_SHA256.txt ($(wc -l < FINAL_SHA256.txt | tr -d ' ') lines)"
