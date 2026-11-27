#!/bin/bash
# The bench_views.py sweep, every case in its own process. JSON lines go to views.jsonl.
# usage: run_views.sh <checkout under test> <python> <out dir>
# A case or check that fails writes a line with "failed": true; its output goes to views.log.
set -u -o pipefail
PORT=$1
PY=$2
OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
: > "$OUT/views.jsonl"
: > "$OUT/views.log"
echo "load before: $(uptime)" > "$OUT/views_load.txt"
case_() {
  (cd /tmp && PYTHONPATH=$PORT/python timeout 2400 $PY -W ignore "$HERE/bench_views.py" run "$@" 2>&1 | tee -a "$OUT/views.log" | grep -E '^\{' >> "$OUT/views.jsonl") \
    || echo "{\"workload\": \"$1\", \"engine\": \"$4\", \"rows\": $2, \"groups_target\": $3, \"failed\": true}" >> "$OUT/views.jsonl"
}
for NG in "1000000 20000" "5000000 200000" "20000000 200000"; do
  set -- $NG
  WORKLOADS="view derived"
  # Python set cells: 20M rows of them do not fit beside the pandas path.
  [ "$1" -le 5000000 ] && WORKLOADS="$WORKLOADS view-sets"
  for W in $WORKLOADS; do
    PYTHONPATH=$PORT/python timeout 2400 $PY -W ignore "$HERE/bench_views.py" check "$W" "$1" "$2" 2>&1 | tee -a "$OUT/views.log" | grep -E '^\{' >> "$OUT/views.jsonl" \
      || echo "{\"check\": \"failed\", \"workload\": \"$W\", \"rows\": $1, \"groups_target\": $2}" >> "$OUT/views.jsonl"
    for E in pandas port; do case_ "$W" "$1" "$2" "$E"; done
  done
  case_ dask "$1" "$2" port-cluster
done
echo "load after: $(uptime)" >> "$OUT/views_load.txt"
