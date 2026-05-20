#!/usr/bin/env bash
# Run 5 independent MLMC experiments with different random seeds.
# Upload colab_prior_work_runner_true_caida_tuned.py to the pod first.
# Usage: bash run_5seeds_runpod.sh
set -e
SCRIPT="colab_prior_work_runner_true_caida_tuned.py"
export PRIOR_WORK_PROFILE="colab_tuned"
export PRIOR_WORK_T="3.0"
export PRIOR_WORK_EPSILONS="0.1,0.05,0.02"
export PRIOR_WORK_CAIDA_MAX_NODES="500"
for i in 0 1 2 3 4; do
  echo "=== Run $i ==="
  PRIOR_WORK_RUN_INDEX=$i python3 -u "$SCRIPT"
done
echo "=== All 5 runs complete ==="
