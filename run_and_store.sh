#!/bin/bash
# Run HELM evaluation and store results in the database.
# Usage: ./run_and_store.sh <suite-name> [conf-file] [max-eval-instances]

set -e

SUITE=${1:?Usage: ./run_and_store.sh <suite-name> [conf-file] [max-instances]}
CONF=${2:-run_specs_test.conf}
MAX_INSTANCES=${3:-600}

echo "=== HELM Evaluation ==="
echo "Suite: $SUITE"
echo "Config: $CONF"
echo "Max instances: $MAX_INSTANCES"
echo ""

# Run HELM evaluation
PYTHONPATH=. helm-run --conf-paths "$CONF" --suite "$SUITE" \
  --local-path . --max-eval-instances "$MAX_INSTANCES"

echo ""
echo "=== Storing Results ==="

# Store each run directory's results
for run_dir in benchmark_output/runs/"$SUITE"/*/; do
  if [ -f "$run_dir/scenario_state.json" ]; then
    echo "Storing: $run_dir"
    python store_helm_results.py --run-dir "$run_dir" --suite "$SUITE"
    echo ""
  fi
done

echo "=== All done ==="
