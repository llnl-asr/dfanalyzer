#!/bin/bash
# Every case in its own process, one after the other. JSON lines go to results.jsonl.
# usage: run.sh <checkout under test> <python> <out dir>
# The pandas case calls fix_dtypes and set_main_metrics from tests/pandas_reference.py. The base
# package itself does not import against the current dftracer-utils, so both cases run from this
# checkout.
set -u -o pipefail
PORT=$1
PY=$2
OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
: > "$OUT/results.jsonl"
: > "$OUT/results.log"
echo "load before: $(uptime)" > "$OUT/load.txt"
for NG in "1000000 20000" "5000000 200000" "20000000 200000"; do
  set -- $NG
  PYTHONPATH=$PORT/python timeout 2400 $PY -W ignore "$HERE/bench_main_view.py" check "$1" "$2" 2>&1 | tee -a "$OUT/results.log" | grep -E '^\{' >> "$OUT/results.jsonl" \
    || echo "{\"check\": \"failed\", \"rows\": $1, \"groups_target\": $2}" >> "$OUT/results.jsonl"
  for E in pandas port; do
    (cd /tmp && PYTHONPATH=$PORT/python timeout 2400 $PY -W ignore "$HERE/bench_main_view.py" run "$1" "$2" "$E" 2>&1 | tee -a "$OUT/results.log" | grep -E '^\{' >> "$OUT/results.jsonl") \
      || echo "{\"engine\": \"$E\", \"rows\": $1, \"groups_target\": $2, \"failed\": true}" >> "$OUT/results.jsonl"
  done
done
echo "load after: $(uptime)" >> "$OUT/load.txt"
