#!/bin/bash
# mig_job.sh NAME 'MATLAB statement'  -- one MATLAB batch job of the migration audit, logged.
W=/Users/piotrek/Programming/topOpt4freqMax-migration-253069
D=$W/analysis/OlhoffCurrent/diagnostics/upstream_253069_migration
EV=$W/analysis/OlhoffCurrent/evidence/upstream_253069_migration
name="$1"; cmd="$2"
start=$(date +%s)
/Applications/MATLAB_R2025b.app/bin/matlab -batch "addpath('$D/scripts'); $cmd" > "$EV/logs/$name.log" 2>&1
rc=$?
echo "$name exit=$rc secs=$(($(date +%s)-start)) end=$(date +%H:%M:%S)" >> "$EV/logs/_done.txt"
exit $rc
