#!/bin/bash
# nmp_launch.sh smoke|campaign
#
# Launches examples/Performance/performance_comparison.m -- the authoritative
# production entry point -- in ONE non-interactive MATLAB process, with the
# observation breakpoints armed first (nmp_arm_hooks) in that same process.
#
# Refuses to start unless: the lock file matches its recorded SHA-256; HEAD is the
# locked HEAD; the runner file is exactly the locked run-selection edit; the only
# tracked file differing from HEAD is that runner; the runner output root and this
# mode's stdout log do not exist yet.
#
# caffeinate -i -s keeps the machine from idle/system sleep for the duration; it
# does not change process priority or CPU affinity.
set -u
REPO=/Users/piotrek/Programming/topOpt4freqMax
D=$REPO/analysis/OlhoffCurrent/diagnostics/nine_mesh_pedersen_production
MATLAB=/Applications/MATLAB_R2025b.app/bin/matlab
RUNNER=$REPO/examples/Performance/performance_comparison.m
MODE="${1:-}"
case "$MODE" in
  smoke)    LOCK=$D/SMOKE_LOCK.json ;;
  campaign) LOCK=$D/CAMPAIGN_LOCK.json ;;
  *) echo "usage: $0 smoke|campaign" >&2; exit 2 ;;
esac

jget() { python3 -c 'import json,sys
d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."): d=d[k]
print(d)' "$LOCK" "$1"; }

LOCKSHA=$(awk '{print $1}' "$LOCK.sha256")
[ "$(shasum -a 256 "$LOCK" | awk '{print $1}')" = "$LOCKSHA" ] || { echo "REFUSE: lock hash mismatch" >&2; exit 10; }
[ "$(jget mode)" = "$MODE" ] || { echo "REFUSE: lock mode is not $MODE" >&2; exit 10; }
[ "$(git -C "$REPO" rev-parse HEAD)" = "$(jget head)" ] || { echo "REFUSE: HEAD differs from lock" >&2; exit 12; }
[ "$(shasum -a 256 "$RUNNER" | awk '{print $1}')" = "$(jget runner.edited_sha256)" ] || { echo "REFUSE: runner is not the locked edit" >&2; exit 11; }
DIRTY=$(git -C "$REPO" diff --name-only HEAD)
[ "$DIRTY" = "examples/Performance/performance_comparison.m" ] || { echo "REFUSE: tracked changes are [$DIRTY]" >&2; exit 14; }
OUT=$(jget output_root_abs)
[ ! -e "$OUT" ] || { echo "REFUSE: output root exists: $OUT" >&2; exit 13; }
LOG=$D/logs/${MODE}_performance_comparison_stdout.log
[ ! -e "$LOG" ] || { echo "REFUSE: log exists: $LOG" >&2; exit 15; }
CWD=$(jget matlab_cwd_abs)
mkdir -p "$CWD"
[ -z "$(ls -A "$CWD")" ] || { echo "REFUSE: MATLAB working folder not empty: $CWD" >&2; exit 16; }

STMT="addpath('$D/scripts'); addpath('$REPO/analysis/OlhoffCurrent'); nmp_arm_hooks(); run('$RUNNER'); nmp_after_run();"
START=$(date +%Y-%m-%dT%H:%M:%S%z)
python3 - "$D/logs/${MODE}_LAUNCH.json" "$START" "$STMT" "$LOCK" "$LOCKSHA" "$CWD" "$LOG" "$MATLAB" <<'PY'
import json, sys, platform
p, start, stmt, lock, locksha, cwd, log, matlab = sys.argv[1:9]
json.dump({"schema": "nmp_launch/1", "start_local": start, "cwd": cwd, "stdout_stderr_log": log,
           "env": {"NMP_LOCK": lock, "NMP_LOCK_SHA256": locksha},
           "command": ["/usr/bin/caffeinate", "-i", "-s", matlab, "-batch", stmt],
           "shell_form": 'cd "%s" && NMP_LOCK="%s" NMP_LOCK_SHA256=%s /usr/bin/caffeinate -i -s %s -batch "%s" > "%s" 2>&1' % (cwd, lock, locksha, matlab, stmt, log),
           "host": platform.node(), "platform": platform.platform()}, open(p, "w"), indent=1)
PY
echo "launch $MODE at $START; log $LOG"
cd "$CWD" || exit 17
NMP_LOCK="$LOCK" NMP_LOCK_SHA256="$LOCKSHA" /usr/bin/caffeinate -i -s "$MATLAB" -batch "$STMT" > "$LOG" 2>&1
RC=$?
END=$(date +%Y-%m-%dT%H:%M:%S%z)
python3 -c 'import json,sys; json.dump({"schema":"nmp_end/1","start_local":sys.argv[2],"end_local":sys.argv[3],"exit_code":int(sys.argv[4])}, open(sys.argv[1],"w"), indent=1)' \
  "$D/logs/${MODE}_END.json" "$START" "$END" "$RC"
echo "end $MODE at $END exit=$RC"
exit $RC
