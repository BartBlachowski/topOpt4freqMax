#!/bin/zsh
# Runs every line of bg_queue.txt ("arm nelx nely") through bg_launch_one.sh, 9 at a time.
cd "$(dirname "$0")"
date > ../logs/QUEUE_START.txt
xargs -P 9 -L 1 ./bg_launch_one.sh < bg_queue.txt
date > ../logs/QUEUE_END.txt
