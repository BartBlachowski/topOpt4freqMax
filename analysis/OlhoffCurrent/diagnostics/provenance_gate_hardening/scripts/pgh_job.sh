#!/bin/bash
# pgh_job.sh NAME KIND -- one logged MATLAB batch block of the hardening audit.
# The log is written to a temporary file and moved into logs/ only when the job
# ends, so a running job never alters a file the study's FINAL_SHA256.txt lists.
H="$(cd "$(dirname "$0")/.." && pwd)"
name="$1"; kind="$2"
start=$(date +%s)
cd "${TMPDIR:-/tmp}" || exit 9
tmp="$(mktemp -t pgh_${name})"
/Applications/MATLAB_R2025b.app/bin/matlab -batch "addpath('$H/scripts'); pgh_run('$kind', '$H/evidence/$name.json')" > "$tmp" 2>&1
rc=$?
mv "$tmp" "$H/logs/$name.log.txt"
echo "$name exit=$rc secs=$(($(date +%s)-start)) end=$(date +%H:%M:%S)" >> "$H/logs/_done.txt"
exit $rc
