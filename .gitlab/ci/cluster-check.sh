#!/bin/bash
# External-cluster DFAnalyzer check (from ci.yml). Runs inside the container.
set -e
dfanalyzer-cluster cluster=local cluster.processes=False \
  +cluster.protocol=tcp +cluster.worker_class=distributed.nanny.Nanny > cluster.log 2>&1 &
cluster_pid=$!

# Wait for the scheduler address to appear in the log (file will contain only the address)
for i in {1..30}; do
  if grep -q '^tcp://' cluster.log; then break; fi
  sleep 1
done
scheduler_address=$(grep '^tcp://' cluster.log | tail -1)
echo "Scheduler address: $scheduler_address"

# Run analysis commands using the external cluster. The dftracer/ai run is
# facts-enabled (output=console on the time_range axis) so it both validates the
# analysis AND prints the Analysis Facts table to the CI log -- a single run, no
# duplicate.
dfanalyzer analyzer=dftracer analyzer/preset=ai trace_path=tests/data/extracted/dftracer-ai \
  cluster=external cluster.restart_on_connect=False cluster.scheduler_address=$scheduler_address \
  view_types=[time_range] facts.enabled=true facts.eval_mode=rule \
  facts.eval_rule_file=tests/data/ci_facts_rules.yaml output=console 2>&1 | tee facts_console.log
# The dftracer/ai run above is facts-enabled; verify facts reached the console.
grep -q "Analysis Facts (" facts_console.log || { echo "ERROR: no analysis facts printed to console"; exit 1; }

# Kill the cluster process
kill $cluster_pid || true
wait $cluster_pid 2>/dev/null || true
