#!/bin/zsh
# Second batch (box-ceiling arms), 5 at a time.  Launched 2026-09-14 11:07 while the
# last run of batch 1 (budget400 400x50) and the 640x80/800x100 continuations were still running.
cd "$(dirname "$0")"
date > ../logs/QUEUE2_START.txt
xargs -P 5 -L 1 ./bg_launch_one.sh < bg_queue2.txt
date > ../logs/QUEUE2_END.txt
