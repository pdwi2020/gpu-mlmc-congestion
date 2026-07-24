#!/usr/bin/env bash
# RunPod 4×RTX 3090 — all multi-GPU experiments (Runs 2, 3, 4, 7)
# Run this script inside the pod after repo is cloned.
# Usage: bash scripts/runpod_run_all.sh
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "  RunPod multi-GPU experiment suite"
echo "  Hardware: 4×RTX 3090 (PCIe, NCCL)"
echo "  Repo: $REPO_ROOT"
echo "============================================================"
echo ""

# ── Install dependencies ────────────────────────────────────────
echo "[SETUP] Installing Python dependencies..."
pip install -q torch torchrun networkx pymetis numpy matplotlib scipy 2>&1 | tail -3
echo "[SETUP] Done."
echo ""

# ── Verify NCCL / torchrun ──────────────────────────────────────
echo "[CHECK] GPU count:"
python3 -c "import torch; print(f'  {torch.cuda.device_count()} GPUs detected'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

mkdir -p results/large_scale results/halo_sweep results/overlap

# ── RUN 2 — Multi-GPU large-n weak scaling ──────────────────────
echo "============================================================"
echo "  RUN 2: Multi-GPU large-n weak scaling (ε=0.05, G=4)"
echo "============================================================"
torchrun --nproc_per_node=4 scripts/run_large_scale_scaling.py \
    --mode distributed \
    --n-nodes 5000 10000 25000 50000 \
    --out-dir results/large_scale
echo ""

# ── RUN 3 — Halo-K sweep ────────────────────────────────────────
echo "============================================================"
echo "  RUN 3: Halo-exchange interval sweep K=1,2,4,8 (G=4)"
echo "============================================================"
torchrun --nproc_per_node=4 scripts/run_halo_sweep.py \
    --distributed \
    --n-nodes 2000 \
    --k-values 1 2 4 8 \
    --n-seeds 5 \
    --out-dir results/halo_sweep
echo ""

# ── RUN 4a — Overlap benchmark 2-GPU ───────────────────────────
echo "============================================================"
echo "  RUN 4a: Blocking vs overlapped comm (G=2)"
echo "============================================================"
torchrun --nproc_per_node=2 scripts/run_overlap_benchmark.py \
    --distributed \
    --n-nodes 2000 \
    --n-reps 5 \
    --out-dir results/overlap
echo ""

# ── RUN 4b — Overlap benchmark 4-GPU ───────────────────────────
echo "============================================================"
echo "  RUN 4b: Blocking vs overlapped comm (G=4)"
echo "============================================================"
torchrun --nproc_per_node=4 scripts/run_overlap_benchmark.py \
    --distributed \
    --n-nodes 2000 \
    --n-reps 5 \
    --out-dir results/overlap
echo ""

# ── RUN 7 — Nsight Systems timeline ────────────────────────────
echo "============================================================"
echo "  RUN 7: Nsight Systems timeline capture"
echo "============================================================"
if command -v nsys &> /dev/null; then
    nsys profile \
        --output results/overlap/nsight_timeline \
        --trace cuda,nccl,nvtx \
        --force-overwrite true \
        torchrun --nproc_per_node=4 scripts/run_overlap_benchmark.py \
            --distributed --n-nodes 2000 --n-reps 1 --out-dir results/overlap
    echo "[RUN 7] Nsight trace saved to results/overlap/nsight_timeline.nsys-rep"
else
    echo "[RUN 7] nsys not found — skipping Nsight capture."
    echo "  Install with: apt-get install -y nsight-systems-cli"
    echo "  Or use: pip install nvidia-nsight-systems"
fi
echo ""

# ── Summary ─────────────────────────────────────────────────────
echo "============================================================"
echo "  ALL RUNS COMPLETE"
echo "============================================================"
echo "Output files:"
find results/large_scale results/halo_sweep results/overlap \
    -name "*.json" -o -name "*.csv" -o -name "*.nsys-rep" 2>/dev/null | sort
echo ""
echo "Next step: copy results/ back to local machine:"
echo "  rsync -avz --include='*.json' --include='*.csv' --include='*.nsys-rep' \\"
echo "    --include='*/' --exclude='*' \\"
echo "    root@<POD_IP>:<PORT>:$(pwd)/results/ results/"
