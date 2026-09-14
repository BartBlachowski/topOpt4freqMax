#!/bin/zsh
# usage: bg_launch_one.sh <arm> <nelx> <nely>
set -u
ARM=$1; NX=$2; NY=$3
ROOT=/Users/piotrek/Programming/topOpt4freqMax
SC=$ROOT/docs/bimodality_gap/scripts
LOG=$ROOT/docs/bimodality_gap/logs/BG_${ARM}_${NX}x${NY}.log
cd $SC
/usr/bin/caffeinate -i -s /Applications/MATLAB_R2025b.app/bin/matlab -batch "addpath('$SC'); A=bg_arms(); k=find(strcmp({A.name},'$ARM'),1); assert(~isempty(k),'unknown arm'); a=A(k); bg_run_arm(a.name,$NX,$NY,a.preset,a.overrides,a.maxOuter,'$ROOT/docs/bimodality_gap/runs');" > $LOG 2>&1
echo "exit=$? $ARM ${NX}x${NY}" >> $LOG
