#!/usr/bin/env bash
# =============================================================================
# run_extended_eps_runpod.sh
# Phase B — RunPod A100 pod-side setup + experiment launcher
#
# Usage (run this script ON the pod, from the project root):
#   bash scripts/run_extended_eps_runpod.sh [--output-dir DIR] [--dry-run]
#
# What it does:
#   1. Validates GPU environment (requires A100 or any CUDA device)
#   2. Installs Python dependencies (cupy-cuda12x scipy pandas numpy tqdm)
#   3. Creates the output directory structure
#   4. Launches experiments/exp_extended_epsilon.py with run2-equivalent config
#   5. Tails the progress log until DONE / ERROR
#
# Estimated wall-clock time on A100: ~25–45 min for 15 jobs (3 scenarios × 5 ε)
# Estimated RunPod cost at $2.19/hr:  ~$0.90–$1.65  (well within $3.00 budget)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override via CLI args)
# ---------------------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-/root/results/extended_eps}"
EPSILONS="0.1 0.05 0.02 0.01 0.005"
SCENARIOS="synthetic_n100 synthetic_n500 real_caida_asrel2_20260101_n500"
SEEDS="42"
CAP_MC=1000000
CAP_MLMC=500000
DRY_RUN=0
SKIP_INSTALL=0
NO_CAIDA=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXP_SCRIPT="${PROJECT_ROOT}/experiments/exp_extended_epsilon.py"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)   OUTPUT_DIR="$2";  shift 2 ;;
    --dry-run)      DRY_RUN=1;        shift   ;;
    --skip-install) SKIP_INSTALL=1;   shift   ;;
    --no-caida)     NO_CAIDA=1;       shift   ;;
    --epsilons)     EPSILONS="$2";    shift 2 ;;
    --scenarios)    SCENARIOS="$2";   shift 2 ;;
    --seeds)        SEEDS="$2";       shift 2 ;;
    --cap-mc)       CAP_MC="$2";      shift 2 ;;
    --cap-mlmc)     CAP_MLMC="$2";    shift 2 ;;
    -h|--help)
      sed -n '2,20p' "${BASH_SOURCE[0]}"
      exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 2 ;;
  esac
done

COST_PER_HOUR=2.19
RUN_START_TS=$(date +%s)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
warn() { echo "[$(date '+%H:%M:%S')] WARNING: $*" >&2; }
die()  { echo "[$(date '+%H:%M:%S')] FATAL: $*" >&2; exit 1; }

elapsed_cost() {
  local now elapsed_s hours cost
  now=$(date +%s)
  elapsed_s=$(( now - RUN_START_TS ))
  hours=$(echo "scale=4; ${elapsed_s} / 3600" | bc)
  cost=$(echo "scale=4; ${hours} * ${COST_PER_HOUR}" | bc)
  printf "elapsed=%ds  estimated_cost=\$%.4f\n" "${elapsed_s}" "${cost}"
}

# ---------------------------------------------------------------------------
# Step 1 — GPU environment check
# ---------------------------------------------------------------------------
log "=== Step 1: GPU environment check ==="

if ! command -v nvidia-smi &>/dev/null; then
  warn "nvidia-smi not found — assuming CUDA is available via container drivers"
else
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1 || echo "unknown")
  GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 || echo "?")
  CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || echo "?")
  log "  GPU      : ${GPU_NAME}"
  log "  VRAM     : ${GPU_MEM} MiB"
  log "  Driver   : ${CUDA_VER}"
  if echo "${GPU_NAME}" | grep -qi "A100"; then
    log "  [OK] A100 detected — optimal configuration"
  else
    warn "  Non-A100 GPU detected; results may differ from paper values"
    warn "  The script will continue — cap_mc/cap_mlmc may need reducing for smaller GPUs"
  fi
fi

# Quick Python/CUDA check
python3 -c "import sys; assert sys.version_info >= (3,8), 'Python 3.8+ required'" \
  || die "Python 3.8+ required"
log "  Python: $(python3 --version)"

# ---------------------------------------------------------------------------
# Step 2 — Install Python dependencies
# ---------------------------------------------------------------------------
log ""
log "=== Step 2: Install dependencies ==="

if [[ "${SKIP_INSTALL}" -eq 1 ]]; then
  log "  --skip-install: skipping pip install"
else
  log "  Installing cupy-cuda12x scipy pandas numpy tqdm ..."

  # Determine CuPy variant from CUDA version
  CUDA_VER_SHORT=$(python3 -c "
import subprocess, re
try:
    out = subprocess.check_output(['nvcc','--version'], text=True)
    m = re.search(r'release (\d+)\.(\d+)', out)
    if m:
        major = int(m.group(1))
        print('cuda11x' if major == 11 else 'cuda12x')
    else:
        print('cuda12x')
except Exception:
    print('cuda12x')
" 2>/dev/null || echo "cuda12x")

  CUPY_PKG="cupy-${CUDA_VER_SHORT}"
  log "  CuPy package: ${CUPY_PKG}"

  pip install --quiet --upgrade pip
  pip install --quiet "${CUPY_PKG}" scipy pandas numpy tqdm
  log "  [OK] dependencies installed"
fi

# Verify CuPy import (skip in dry-run mode — no GPU needed for matrix preview)
if [[ "${DRY_RUN}" -eq 0 ]]; then
  python3 -c "import cupy as cp; log=lambda m: print(m); log(f'  CuPy {cp.__version__}  device: {cp.cuda.runtime.getDeviceProperties(0)[\"name\"].decode()}')" \
    || die "CuPy import failed — check CUDA version compatibility"
fi

# ---------------------------------------------------------------------------
# Step 3 — Output directory + logging setup
# (skip in dry-run mode — just show the experiment matrix then exit)
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN}" -eq 1 ]]; then
  log ""
  log "=== Experiment matrix (DRY RUN) ==="
  read -ra EPS_ARR   <<< "${EPSILONS}"
  read -ra SCEN_ARR  <<< "${SCENARIOS}"
  read -ra SEEDS_ARR <<< "${SEEDS}"
  TOTAL_JOBS=$(( ${#SCEN_ARR[@]} * ${#EPS_ARR[@]} * ${#SEEDS_ARR[@]} ))
  log "  Scenarios (${#SCEN_ARR[@]}): ${SCENARIOS}"
  log "  Epsilons  (${#EPS_ARR[@]}):  ${EPSILONS}"
  log "  Seeds     (${#SEEDS_ARR[@]}): ${SEEDS}"
  log "  cap_mc    : ${CAP_MC}    cap_mlmc: ${CAP_MLMC}"
  log "  Total jobs: ${TOTAL_JOBS}"
  EST_MIN=$(( TOTAL_JOBS * 2 ))
  EST_COST=$(echo "scale=4; ${EST_MIN} / 60 * ${COST_PER_HOUR}" | bc)
  log "  Estimated time : ~${EST_MIN} min"
  log "  Estimated cost : ~\$${EST_COST} / \$3.00 budget"
  log ""
  log "  DRY RUN — exiting without running."
  exit 0
fi

log ""
log "=== Step 3: Output directory setup ==="

mkdir -p "${OUTPUT_DIR}"
LOG_PATH="${OUTPUT_DIR}/run_progress.log"
CSV_PATH="${OUTPUT_DIR}/extended_epsilon_results.csv"
JSON_PATH="${OUTPUT_DIR}/run_summary.json"

log "  Output dir : ${OUTPUT_DIR}"
log "  Progress log: ${LOG_PATH}"
log "  CSV output  : ${CSV_PATH}"

# Save environment snapshot
python3 - <<PYEOF
import json, platform, subprocess
env = {
    "script":       "run_extended_eps_runpod.sh",
    "python":       platform.python_version(),
    "platform":     platform.platform(),
}
try:
    import cupy as cp
    env["cupy"] = cp.__version__
    props = cp.cuda.runtime.getDeviceProperties(0)
    env["gpu_name"] = props["name"].decode()
    env["compute_capability"] = f"{props['major']}.{props['minor']}"
    env["global_mem_gb"] = f"{props['totalGlobalMem'] / 1e9:.1f}"
except Exception as e:
    env["cupy_error"] = str(e)
try:
    smi = subprocess.check_output(
        ["nvidia-smi","--query-gpu=name,driver_version,cuda_version",
         "--format=csv,noheader,nounits"], text=True).strip()
    env["nvidia_smi"] = smi
except Exception:
    pass
out = "${OUTPUT_DIR}/env_snapshot.json"
with open(out, "w") as f:
    json.dump(env, f, indent=2)
print(f"  Env snapshot -> {out}")
PYEOF

# ---------------------------------------------------------------------------
# Step 4 — Experiment matrix summary
# ---------------------------------------------------------------------------
log ""
log "=== Step 4: Experiment matrix ==="

# Convert space-separated strings to arrays
read -ra EPS_ARR    <<< "${EPSILONS}"
read -ra SCEN_ARR   <<< "${SCENARIOS}"
read -ra SEEDS_ARR  <<< "${SEEDS}"
TOTAL_JOBS=$(( ${#SCEN_ARR[@]} * ${#EPS_ARR[@]} * ${#SEEDS_ARR[@]} ))

log "  Scenarios (${#SCEN_ARR[@]}): ${SCENARIOS}"
log "  Epsilons  (${#EPS_ARR[@]}):  ${EPSILONS}"
log "  Seeds     (${#SEEDS_ARR[@]}): ${SEEDS}"
log "  cap_mc    : ${CAP_MC}    cap_mlmc: ${CAP_MLMC}"
log "  Total jobs: ${TOTAL_JOBS}"
log "  $(elapsed_cost)"

# Rough wall-time estimate: ~2 min/job on A100 (conservative)
EST_MIN=$(( TOTAL_JOBS * 2 ))
EST_COST=$(echo "scale=4; ${EST_MIN} / 60 * ${COST_PER_HOUR}" | bc)
log "  Estimated time : ~${EST_MIN} min"
log "  Estimated cost : ~\$${EST_COST} / \$3.00 budget"



# ---------------------------------------------------------------------------
# Step 5 — Run the experiment
# ---------------------------------------------------------------------------
log ""
log "=== Step 5: Running exp_extended_epsilon.py ==="
log "  Monitor from another terminal:"
log "    bash ${PROJECT_ROOT}/scripts/monitor_runpod.sh ${LOG_PATH}"
log ""

# Build argument list
CAIDA_FLAG=""
[[ "${NO_CAIDA}" -eq 1 ]] && CAIDA_FLAG="--no-caida-download"

# Run with unbuffered output so log is written immediately
python3 -u "${EXP_SCRIPT}" \
  --output-dir   "${OUTPUT_DIR}" \
  --epsilons     ${EPSILONS} \
  --scenarios    ${SCENARIOS} \
  --seeds        ${SEEDS} \
  --cap-mc       "${CAP_MC}" \
  --cap-mlmc     "${CAP_MLMC}" \
  ${CAIDA_FLAG} \
  2>&1 | tee -a "${LOG_PATH}"

EXIT_CODE=${PIPESTATUS[0]}

# ---------------------------------------------------------------------------
# Step 6 — Post-run summary
# ---------------------------------------------------------------------------
log ""
log "=== Step 6: Run summary ==="
log "  $(elapsed_cost)"

if [[ ${EXIT_CODE} -ne 0 ]]; then
  die "exp_extended_epsilon.py exited with code ${EXIT_CODE} — check ${LOG_PATH}"
fi

if [[ ! -f "${CSV_PATH}" ]]; then
  die "CSV not found at ${CSV_PATH} — experiment may have produced no rows"
fi

# Print key stats from the CSV
python3 - <<PYEOF
import pandas as pd, sys

csv_path  = "${CSV_PATH}"
json_path = "${JSON_PATH}"

df = pd.read_csv(csv_path)
total   = len(df)
eq_acc  = df["equal_accuracy_ci_targeted"].astype(str).str.lower().isin({"true","1"}).sum()
scens   = df["scenario"].nunique()
eps_cnt = df["epsilon"].nunique()

print(f"  Total rows          : {total}")
print(f"  equal_accuracy=True : {eq_acc} / {total}")
print(f"  Scenarios covered   : {scens}")
print(f"  Epsilon values      : {sorted(df['epsilon'].unique().tolist())}")
print()

# Check W8 numbers (abstract claims 12.91x speedup, 257.72x cost ratio at eps=0.01)
W8_SPEEDUP = 12.91
W8_COST    = 257.72
reliable = df[df["equal_accuracy_ci_targeted"].astype(str).str.lower().isin({"true","1"})]
row_eps001 = reliable[
    reliable["epsilon"].between(0.009, 0.011) &
    (reliable["scenario"] == "synthetic_n500")
]

print("  W8 verification (synthetic_n500, eps=0.01):")
if row_eps001.empty:
    print("    [WARN] No equal-accuracy rows for synthetic_n500 eps=0.01 — W8 unverified")
else:
    r = row_eps001.iloc[0]
    spd  = float(r["speedup_runtime"])
    cost = float(r["cost_ratio_mc_over_mlmc"])
    spd_ok  = spd  >= W8_SPEEDUP * 0.90
    cost_ok = cost >= W8_COST    * 0.90
    print(f"    speedup_runtime        : {spd:.2f}x  (paper: {W8_SPEEDUP}x)  {'[OK]' if spd_ok  else '[LOWER]'}")
    print(f"    cost_ratio_mc_over_mlmc: {cost:.2f}x (paper: {W8_COST}x) {'[OK]' if cost_ok else '[LOWER]'}")
    if spd_ok and cost_ok:
        print("    => W8 numbers REPRODUCED (within 10%)")
    else:
        print("    => W8 numbers not reproduced at 90% threshold")
        print("       (may still be acceptable — check equal_accuracy_ci_targeted)")

print()
print("  Top speedup rows:")
top = reliable.nlargest(5, "speedup_runtime")[
    ["scenario","epsilon","speedup_runtime","cost_ratio_mc_over_mlmc","equal_accuracy_ci_targeted"]
]
print(top.to_string(index=False))
PYEOF

log ""
log "=== DONE ==="
log "  Results  : ${CSV_PATH}"
log "  Summary  : ${JSON_PATH}"
log "  Log      : ${LOG_PATH}"
log ""
log "  Next steps:"
log "    1. scp -r pod:${OUTPUT_DIR} results/results/runpod_a100_extended/"
log "    2. python3 scripts/post_process_extended.py"
log "    3. python3 paper/gen_figures_a100.py"
