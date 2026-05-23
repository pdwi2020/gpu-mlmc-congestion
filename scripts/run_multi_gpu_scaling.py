"""
Multi-GPU MLMC scaling benchmark.

Measures weak and strong scaling of MultiGPUMLMC across world_size ∈ {1, 2, 4}.

Weak scaling:  n_nodes grows proportionally with world_size; runtime/GPU should be flat.
Strong scaling: n_nodes fixed, world_size increases; target ≥1.5x speedup per doubling.

In single-process mode (no torchrun), simulates multiple ranks sequentially using the
round-robin partitioner, which gives the correct local workload but skips actual
inter-process communication — sufficient for profiling partition overhead.

Usage:
    # Single-process benchmark (no distributed setup needed)
    python scripts/run_multi_gpu_scaling.py

    # True multi-GPU (4 GPUs, requires nccl/gloo + torchrun)
    torchrun --nproc_per_node=4 scripts/run_multi_gpu_scaling.py --distributed

Outputs:
    results/multi_gpu/scaling_results.json
    results/multi_gpu/scaling_plot.png
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gpu.parallel_mc import MultiGPUMLMC


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------

def erdos_renyi_adj(n: int, p: float = 0.05, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    adj = rng.random((n, n)) < p
    adj = ((adj + adj.T) > 0).astype(float)
    np.fill_diagonal(adj, 0.0)
    return adj


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def bench_rank(adj, world_size, rank, epsilon, L_max, N_pilot, seed):
    """Run one rank's MLMC and return elapsed time and estimate."""
    sim = MultiGPUMLMC(adj, world_size=world_size, rank=rank, seed=seed + rank)
    t0 = time.perf_counter()
    result = sim.mlmc_estimate_multigpu(epsilon=epsilon, L_max=L_max, N_pilot=N_pilot)
    elapsed = time.perf_counter() - t0
    del sim
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return elapsed, result


def bench_all_ranks(adj, world_size, epsilon, L_max, N_pilot, seed):
    """Simulate all ranks sequentially (single-process approximation)."""
    times = []
    for rank in range(world_size):
        t, _ = bench_rank(adj, world_size, rank, epsilon, L_max, N_pilot, seed)
        times.append(t)
    return times


# ---------------------------------------------------------------------------
# Weak scaling: n_nodes ∝ world_size
# ---------------------------------------------------------------------------

def weak_scaling(base_n, world_sizes, epsilon, L_max, N_pilot, seed):
    print("\n=== Weak Scaling (n_nodes ∝ world_size) ===")
    print(f"{'world_size':>12} {'n_nodes':>10} {'max_rank_time':>16} {'avg_rank_time':>16}")
    print("-" * 58)
    results = []
    for ws in world_sizes:
        n = base_n * ws
        adj = erdos_renyi_adj(n, p=max(0.02, 5.0/n), seed=seed)
        times = bench_all_ranks(adj, ws, epsilon, L_max, N_pilot, seed)
        max_t = max(times)
        avg_t = sum(times) / len(times)
        print(f"{ws:12d} {n:10d} {max_t:16.3f}s {avg_t:16.3f}s")
        results.append({'world_size': ws, 'n_nodes': n,
                        'max_time': max_t, 'avg_time': avg_t, 'times': times})
    return results


# ---------------------------------------------------------------------------
# Strong scaling: n_nodes fixed, world_size varies
# ---------------------------------------------------------------------------

def strong_scaling(n, world_sizes, epsilon, L_max, N_pilot, seed):
    print(f"\n=== Strong Scaling (n_nodes={n}) ===")
    print(f"{'world_size':>12} {'max_rank_time':>16} {'speedup':>10}")
    print("-" * 42)
    results = []
    baseline_t = None
    for ws in world_sizes:
        adj = erdos_renyi_adj(n, p=max(0.02, 5.0/n), seed=seed)
        times = bench_all_ranks(adj, ws, epsilon, L_max, N_pilot, seed)
        max_t = max(times)
        if baseline_t is None:
            baseline_t = max_t
        speedup = baseline_t / max_t
        print(f"{ws:12d} {max_t:16.3f}s {speedup:10.2f}x")
        results.append({'world_size': ws, 'n_nodes': n,
                        'max_time': max_t, 'speedup': speedup, 'times': times})
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MultiGPUMLMC scaling benchmark")
    parser.add_argument("--base-n", type=int, default=1250,
                        help="Base number of nodes for weak scaling (×world_size gives 1250/2500/5000)")
    parser.add_argument("--strong-n", type=int, default=5000,
                        help="Fixed node count for strong scaling")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--L-max", type=int, default=4)
    parser.add_argument("--N-pilot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/multi_gpu")
    parser.add_argument("--distributed", action="store_true",
                        help="Initialize torch.distributed (for torchrun)")
    args = parser.parse_args()

    if args.distributed:
        import torch
        import torch.distributed as dist
        # torchrun sets LOCAL_RANK; each process sees exactly one GPU remapped to cuda:0
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Distributed mode: rank {rank}/{world_size} local_rank={local_rank}", flush=True)
        adj = erdos_renyi_adj(args.strong_n, seed=args.seed)
        t0 = time.perf_counter()
        sim = MultiGPUMLMC(adj, world_size=world_size, rank=rank, seed=args.seed)
        result = sim.mlmc_estimate_multigpu(args.epsilon, args.L_max, args.N_pilot)
        elapsed = time.perf_counter() - t0
        # Synchronize then print per-rank timing (avoids all_reduce device type issues)
        dist.barrier()
        print(f"[rank {rank}] time={elapsed:.3f}s  estimate={result['estimate']:.4f}", flush=True)
        dist.destroy_process_group()
        return

    os.makedirs(args.out_dir, exist_ok=True)

    weak_results = weak_scaling(
        args.base_n, args.world_sizes,
        args.epsilon, args.L_max, args.N_pilot, args.seed)

    strong_results = strong_scaling(
        args.strong_n, args.world_sizes,
        args.epsilon, args.L_max, args.N_pilot, args.seed)

    out = {'weak_scaling': weak_results, 'strong_scaling': strong_results,
           'args': vars(args)}
    out_path = os.path.join(args.out_dir, "scaling_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ws_w = [r['world_size'] for r in weak_results]
        t_w = [r['max_time'] for r in weak_results]
        ax1.plot(ws_w, t_w, 'o-', color='tab:blue')
        ax1.axhline(t_w[0], color='gray', linestyle='--', label='Ideal (flat)')
        ax1.set_xlabel("World size (GPUs)")
        ax1.set_ylabel("Max rank time (s)")
        ax1.set_title("Weak Scaling")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ws_s = [r['world_size'] for r in strong_results]
        sp_s = [r['speedup'] for r in strong_results]
        ideal = [ws / ws_s[0] for ws in ws_s]
        ax2.plot(ws_s, sp_s, 'o-', color='tab:orange', label='Actual')
        ax2.plot(ws_s, ideal, '--', color='gray', label='Ideal linear')
        ax2.set_xlabel("World size (GPUs)")
        ax2.set_ylabel("Speedup")
        ax2.set_title(f"Strong Scaling (n={args.strong_n})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        png_path = os.path.join(args.out_dir, "scaling_plot.png")
        fig.savefig(png_path, dpi=150)
        print(f"Saved: {png_path}")
    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
