#!/bin/bash
# Run v2 learned steering phases in order, e.g.:
#   bash start_steering_v2.command setup e1
#   bash start_steering_v2.command e2            (after the checkpoint is selected)
#   bash start_steering_v2.command test e3 e4    (after alpha is selected)
# Stops at the first failed phase. Log: steering_v2/logs/all.out
cd "$(dirname "$0")"
mkdir -p steering_v2/logs
[ $# -gt 0 ] || { sed -n '2,6p' "$0"; exit 1; }
caffeinate -i -w $$ &
{
  for phase in "$@"; do
    echo "== $phase start $(date '+%F %T') =="
    if ! bash run_steering_v2.sh "$phase"; then
      echo "== $phase FAILED $(date '+%F %T') =="
      exit 1
    fi
    echo "== $phase done $(date '+%F %T') =="
  done
  echo "== ALL DONE ($*) $(date '+%F %T') =="
} 2>&1 | tee -a steering_v2/logs/all.out
