#!/bin/bash
# pmg_job.sh NAME 'MATLAB statement' -- one logged MATLAB batch job of the post-merge gate.
# The log goes to a temporary file and is moved into logs/ only when the job ends, so a
# running job never alters a file listed in the study's FINAL_SHA256.txt.
G=/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffCurrent/diagnostics/postmerge_campaign_gate
name="$1"; cmd="$2"
start=$(date +%s)
cd /private/tmp/claude-501/-Users-piotrek-Programming-topOpt4freqMax/84bbf569-dc04-4771-be4d-d9d2da2b8566/scratchpad/run || exit 9
tmp="$(mktemp -t pmg_${name})"
/Applications/MATLAB_R2025b.app/bin/matlab -batch "addpath('$G/scripts'); $cmd" > "$tmp" 2>&1
rc=$?
mv "$tmp" "$G/logs/$name.log.txt"
echo "$name exit=$rc secs=$(($(date +%s)-start)) end=$(date +%H:%M:%S)" >> "$G/logs/_done.txt"
exit $rc
