from __future__ import annotations

import glob
import os
import sys


run_index = int(sys.argv[1])

os.environ["PRIOR_WORK_PROFILE"] = "colab_tuned"
os.environ["PRIOR_WORK_T"] = "3.0"
os.environ["PRIOR_WORK_EPSILONS"] = "0.1,0.05,0.02"
os.environ["PRIOR_WORK_CAIDA_MAX_NODES"] = "500"
os.environ["PRIOR_WORK_RUN_INDEX"] = str(sys.argv[1])
os.environ["PRIOR_WORK_OUTPUT_DIR"] = f"/tmp/mlmc_results/run{run_index:02d}"

os.makedirs(os.environ["PRIOR_WORK_OUTPUT_DIR"], exist_ok=True)

runner_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "results",
    "results",
    "prior_work_kaggle_tuned_bundle_20260215_233444",
    "colab_prior_work_runner_true_caida_tuned.py",
)

exec(open(runner_path).read())

for filename in sorted(glob.glob(os.path.join(os.environ["PRIOR_WORK_OUTPUT_DIR"], "*.csv"))):
    print(f"=== FILE: {filename} ===")
    with open(filename) as f:
        print(f.read())
