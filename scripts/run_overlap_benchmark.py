"""
Blocking vs overlapped communication benchmark (C1 + C2).

Measures:
  C1) Overlap Gain = (T_blocking - T_overlapped) / T_blocking × 100
  C2) Hidden communication fraction H = 1 - T_wait / T_sync_allreduce

Method:
  - BLOCKING mode:  timing_enabled=True  → all_reduce is synchronous, no overlap
  - OVERLAPPED mode: timing_enabled=False → all_reduce is async, handle.wait()
                     in the next step allows compute/comm overlap

The hidden fraction H is estimated by running in BLOCKING mode to get the true
all_reduce time, then comparing to wall-clock under OVERLAPPED mode:
  H_est ≈ (T_block_wall - T_overlap_wall) / T_allreduce_from_block_mode

Usage (torchrun, 4 GPUs, requires NCCL):
    torchrun --nproc_per_node=4 scripts/run_overlap_benchmark.py --distributed

Usage (2 GPUs):
    torchrun --nproc_per_node=2 scripts/run_overlap_benchmark.py --distributed

Output:
    results/overlap/overlap_benchmark.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gpu.parallel_mc import MultiGPUMLMC


def ba_adj(n: int, m: int = 3, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=float)
    for i in range(min(m, n)):
        for j in range(i):
            adj[i, j] = adj[j, i] = 1.0
    degrees = np.maximum(adj.sum(axis=1), 1.0)
    for new_node in range(m, n):
        probs_existing = degrees[:new_node]
        probs_existing = probs_existing / probs_existing.sum()
        targets = rng.choice(new_node, size=min(m, new_node), replace=False, p=probs_existing)
        for tgt in targets:
            adj[new_node, tgt] = adj[tgt, new_node] = 1.0
        degrees[new_node] = len(targets)
        degrees[targets] += 1.0
    return adj


def bench_one(adj, world_size, rank, epsilon, L_max, N_pilot, seed,
              blocking: bool, n_reps: int = 3):
    """Run n_reps timed MLMC estimates in blocking or overlapped comm mode."""
    times = []
    allreduce_times = []

    for rep in range(n_reps):
        sim = MultiGPUMLMC(adj, world_size=world_size, rank=rank, seed=seed + rep)
        sim.timing_enabled = blocking  # True → synchronous, False → async overlap

        # Warmup
        sim.mlmc_estimate_multigpu(epsilon=0.1, L_max=2, N_pilot=10)
        sim.reset_timers()

        t0 = time.perf_counter()
        sim.mlmc_estimate_multigpu(epsilon=epsilon, L_max=L_max, N_pilot=N_pilot)
        elapsed = time.perf_counter() - t0

        ratio = sim.comm_compute_ratio()
        times.append(elapsed)
        allreduce_times.append(ratio["comm_s"])

        del sim
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        "elapsed_mean_s": float(np.mean(times)),
        "elapsed_std_s": float(np.std(times)),
        "allreduce_mean_s": float(np.mean(allreduce_times)),
        "n_reps": n_reps,
        "blocking": blocking,
    }


def main():
    parser = argparse.ArgumentParser(description="Blocking vs overlapped comm benchmark (C1+C2)")
    parser.add_argument("--n-nodes", type=int, default=2000)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--L-max", type=int, default=4)
    parser.add_argument("--N-pilot", type=int, default=50)
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/overlap")
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    world_size, rank = 1, 0
    if args.distributed:
        import torch
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    adj = ba_adj(args.n_nodes, m=3, seed=args.seed)

    print(f"Benchmarking G={world_size}, n={args.n_nodes}, eps={args.epsilon}", flush=True)

    # --- BLOCKING (synchronous, no overlap) ---
    print("  [1/2] Blocking comm...", flush=True)
    blk = bench_one(adj, world_size, rank, args.epsilon,
                    args.L_max, args.N_pilot, args.seed,
                    blocking=True, n_reps=args.n_reps)
    print(f"    T_block = {blk['elapsed_mean_s']:.3f}s  "
          f"T_allreduce = {blk['allreduce_mean_s']:.4f}s", flush=True)

    # --- OVERLAPPED (async, hidden comm) ---
    print("  [2/2] Overlapped comm...", flush=True)
    ov = bench_one(adj, world_size, rank, args.epsilon,
                   args.L_max, args.N_pilot, args.seed,
                   blocking=False, n_reps=args.n_reps)
    print(f"    T_overlap = {ov['elapsed_mean_s']:.3f}s", flush=True)

    # --- Derived metrics ---
    T_block = blk["elapsed_mean_s"]
    T_overlap = ov["elapsed_mean_s"]
    T_allreduce = blk["allreduce_mean_s"]

    overlap_gain_pct = 100 * (T_block - T_overlap) / T_block if T_block > 0 else 0.0

    # Hidden fraction: how much of the all_reduce was hidden by overlap
    # H ≈ (T_block - T_overlap) / T_allreduce  (clamped to [0,1])
    hidden_frac = min(1.0, max(0.0,
        (T_block - T_overlap) / T_allreduce if T_allreduce > 0 else 0.0))

    visible_comm_s = T_allreduce * (1 - hidden_frac)

    out = {
        "hardware": f"{world_size}×GPU PCIe",
        "n_nodes": args.n_nodes,
        "epsilon": args.epsilon,
        "world_size": world_size,
        "blocking": blk,
        "overlapped": ov,
        "overlap_gain_pct": overlap_gain_pct,
        "hidden_comm_frac": hidden_frac,
        "visible_comm_s": visible_comm_s,
        "total_allreduce_s": T_allreduce,
        "note": (
            f"Overlap gain = (T_block - T_overlap)/T_block = "
            f"({T_block:.3f} - {T_overlap:.3f})/{T_block:.3f} = {overlap_gain_pct:.1f}%. "
            f"Hidden fraction H ≈ {100*hidden_frac:.1f}% of all_reduce hidden by async overlap."
        ),
    }

    if rank == 0:
        out_path = os.path.join(args.out_dir, f"overlap_benchmark_{world_size}gpu.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\nResults saved: {out_path}")
        print(f"  Overlap Gain:  {overlap_gain_pct:.1f}%")
        print(f"  Hidden fraction H: {100*hidden_frac:.1f}%")

    if args.distributed:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
