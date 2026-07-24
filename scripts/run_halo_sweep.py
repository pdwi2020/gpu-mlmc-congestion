"""
Halo-exchange interval sweep (B4).

Sweeps halo_exchange_every K ∈ {1, 2, 4, 8} on the 4×RTX3090 pod.
For each K, measures:
  - Wall-clock runtime (post-warmup)
  - Estimator mean and std across seeds (bias check)
  - Comm/compute breakdown (timing_enabled=True)

K=1 is fully correct (exchange every step). K>1 introduces bounded stale-ghost
error ≤ L_lip * K * dt per step; if accuracy change is negligible, the runtime
reduction is a contribution.

Usage (torchrun, 4 GPUs):
    torchrun --nproc_per_node=4 scripts/run_halo_sweep.py --distributed

Usage (single-process simulation):
    python scripts/run_halo_sweep.py

Output:
    results/halo_sweep/halo_k_sweep.json
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


def bench_k(adj, K, world_size, rank, epsilon, L_max, N_pilot, seed, n_seeds=3):
    """Run n_seeds replications with halo_exchange_every=K, return timing + estimates."""
    times = []
    estimates = []
    phases_list = []

    for s in range(n_seeds):
        sim = MultiGPUMLMC(adj, world_size=world_size, rank=rank,
                           seed=seed + s, halo_exchange_every=K)
        sim.timing_enabled = True

        # Warmup
        sim.mlmc_estimate_multigpu(epsilon=0.1, L_max=2, N_pilot=10)
        sim.reset_timers()

        t0 = time.perf_counter()
        res = sim.mlmc_estimate_multigpu(epsilon=epsilon, L_max=L_max, N_pilot=N_pilot)
        elapsed = time.perf_counter() - t0

        ratio = sim.comm_compute_ratio()
        times.append(elapsed)
        estimates.append(res.get("estimate", float("nan")))
        phases_list.append(ratio.get("phases", {}))

        del sim
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    est_arr = [e for e in estimates if not np.isnan(e)]
    return {
        "K": K,
        "elapsed_mean_s": float(np.mean(times)),
        "elapsed_std_s": float(np.std(times)),
        "estimate_mean": float(np.mean(est_arr)) if est_arr else None,
        "estimate_std": float(np.std(est_arr)) if est_arr else None,
        "n_seeds": n_seeds,
        "phases_mean": {
            k: float(np.mean([p.get(k, 0) for p in phases_list]))
            for k in ["sde_pct", "bookkeeping_pct", "halo_pct", "sync_pct"]
        },
        "world_size": world_size,
        "epsilon": epsilon,
    }


def run_sweep(args, world_size: int = 1, rank: int = 0):
    os.makedirs(args.out_dir, exist_ok=True)
    adj = ba_adj(args.n_nodes, m=3, seed=args.seed)

    results = []
    k1_estimate = None

    for K in args.k_values:
        print(f"  K={K}  n={args.n_nodes}  G={world_size}", flush=True)
        row = bench_k(adj, K, world_size, rank, args.epsilon,
                      args.L_max, args.N_pilot, args.seed, args.n_seeds)

        # Bias relative to K=1 baseline
        if K == 1:
            k1_estimate = row["estimate_mean"]
        if k1_estimate is not None and row["estimate_mean"] is not None:
            row["bias_pct"] = 100 * abs(row["estimate_mean"] - k1_estimate) / (abs(k1_estimate) + 1e-12)
        else:
            row["bias_pct"] = None

        # Runtime reduction relative to K=1
        if results:
            row["runtime_reduction_pct"] = 100 * (1 - row["elapsed_mean_s"] / results[0]["elapsed_mean_s"])
        else:
            row["runtime_reduction_pct"] = 0.0

        results.append(row)
        print(f"    → {row['elapsed_mean_s']:.3f}s  est={row['estimate_mean']:.4f}"
              f"  bias={row.get('bias_pct','?'):.2f}%  rt-reduc={row['runtime_reduction_pct']:.1f}%",
              flush=True)

    out_path = os.path.join(args.out_dir, "halo_k_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Halo-exchange interval sweep (B4)")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--n-nodes", type=int, default=2000)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--L-max", type=int, default=4)
    parser.add_argument("--N-pilot", type=int, default=50)
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Seeds per K for bias estimation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/halo_sweep")
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    if args.distributed:
        import torch
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            run_sweep(args, world_size=world_size, rank=rank)
        dist.barrier()
        dist.destroy_process_group()
    else:
        run_sweep(args, world_size=1, rank=0)


if __name__ == "__main__":
    main()
