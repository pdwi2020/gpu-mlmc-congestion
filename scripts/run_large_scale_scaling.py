"""
Large-scale + tight-epsilon scaling experiment (B1).

Runs a TARGETED (n, epsilon) matrix — NOT the full Cartesian product — to
support two independent paper claims without wasting GPU time:

  Claim 1 — efficiency recovers at large n:
      n ∈ {10000, 25000, 50000}  ×  epsilon = 0.05 only
      (reviewer needs runtime-vs-n, not tight accuracy at large n)

  Claim 2 — O(epsilon^{-2}) complexity slope:
      n ∈ {1000, 5000}  ×  epsilon ∈ {0.05, 0.01, 0.005, 0.002}

  Bridge point (optional, ~30 min):
      n = 10000  ×  epsilon = 0.01

The expensive cross-terms (n >= 10000 × epsilon <= 0.01) are omitted:
they cost 10-100 GPU-hours and produce no data point used in the paper.

Single-GPU usage (A100):
    python scripts/run_large_scale_scaling.py --mode single --out-dir results/large_scale

Multi-GPU usage (torchrun, 4 GPUs, RTX3090 pod):
    torchrun --nproc_per_node=4 scripts/run_large_scale_scaling.py \\
        --mode distributed --out-dir results/large_scale

Output:
    results/large_scale/single_gpu_results.json
    results/large_scale/multi_gpu_<G>gpu_results.json
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
    """Barabási–Albert adjacency matrix (degree-weighted, symmetric)."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=float)
    # Seed graph: complete graph on m nodes
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


def warmup(sim, epsilon, L_max, N_pilot):
    """One short warmup run to initialize NCCL and CUDA kernels."""
    sim.mlmc_estimate_multigpu(epsilon=0.1, L_max=2, N_pilot=10)


def build_run_pairs(n_nodes_large, epsilons_tight, n_bridge=10000, eps_bridge=0.01,
                    include_bridge=True):
    """Return the targeted (n, epsilon) pairs — not the full Cartesian product.

    Claim 1: large n × eps=0.05 only  (efficiency-vs-n curve)
    Claim 2: moderate n × all tight epsilons  (complexity-slope curve)
    Bridge:  n=10000 × eps=0.01  (connects the two curves)
    """
    pairs = []
    # Claim 2: moderate n, all epsilons
    for n in [1000, 5000]:
        for eps in epsilons_tight:
            pairs.append((n, eps))
    # Bridge point
    if include_bridge and n_bridge not in [1000, 5000]:
        pairs.append((n_bridge, eps_bridge))
    # Claim 1: large n, eps=0.05 only
    for n in n_nodes_large:
        if n not in [1000, 5000]:
            pairs.append((n, 0.05))
    return pairs


def run_single_gpu(args):
    """Sweep targeted (n, epsilon) pairs on a single GPU."""
    os.makedirs(args.out_dir, exist_ok=True)
    results = []

    pairs = build_run_pairs(
        n_nodes_large=args.n_nodes,
        epsilons_tight=args.epsilons,
        n_bridge=args.bridge_n,
        eps_bridge=args.bridge_eps,
        include_bridge=args.bridge,
    )
    print(f"Running {len(pairs)} targeted (n, ε) pairs:", flush=True)
    for n, eps in pairs:
        print(f"  n={n:,}  ε={eps}", flush=True)
    print(flush=True)

    for n, eps in pairs:
        print(f"→ n={n:,}  ε={eps}", flush=True)
        adj = ba_adj(n, m=3, seed=args.seed)
        sim = MultiGPUMLMC(adj, world_size=1, rank=0, seed=args.seed)

        # Warmup
        warmup(sim, eps, args.L_max, args.N_pilot)

        # Timed run
        t0 = time.perf_counter()
        res = sim.mlmc_estimate_multigpu(epsilon=eps, L_max=args.L_max, N_pilot=args.N_pilot)
        elapsed = time.perf_counter() - t0

        # Memory
        peak_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated() / 1e6
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        row = {
            "n_nodes": n,
            "epsilon": eps,
            "elapsed_s": elapsed,
            "estimate": res.get("estimate", None),
            "peak_mem_mb": peak_mb,
            "hardware": "single_gpu",
            "seed": args.seed,
        }
        results.append(row)
        print(f"  done: {elapsed:.3f}s  est={row['estimate']:.4f}  mem={peak_mb:.1f}MB",
              flush=True)

        del sim
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    out_path = os.path.join(args.out_dir, "single_gpu_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")


def run_distributed(args):
    """True multi-GPU via torchrun. Runs targeted large-n pairs at eps=0.05."""
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    os.makedirs(args.out_dir, exist_ok=True)
    results = []

    # Distributed: only large-n at eps=0.05 (efficiency-recovery claim)
    large_n = [n for n in args.n_nodes if n >= 5000]
    pairs = [(n, 0.05) for n in large_n]

    if rank == 0:
        print(f"Distributed G={world_size}: running {len(pairs)} pairs", flush=True)
        for n, eps in pairs:
            print(f"  n={n:,}  ε={eps}", flush=True)

    for n, eps in pairs:
        adj = ba_adj(n, m=3, seed=args.seed)
        sim = MultiGPUMLMC(adj, world_size=world_size, rank=rank, seed=args.seed + rank)

        # Warmup (excludes NCCL init overhead from timing)
        warmup(sim, eps, args.L_max, args.N_pilot)
        dist.barrier()

        t0 = time.perf_counter()
        res = sim.mlmc_estimate_multigpu(epsilon=eps, L_max=args.L_max, N_pilot=args.N_pilot)
        elapsed = time.perf_counter() - t0
        dist.barrier()

        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()

        if rank == 0:
            row = {
                "n_nodes": n,
                "epsilon": eps,
                "elapsed_s": elapsed,
                "estimate": res.get("estimate", None),
                "world_size": world_size,
                "peak_mem_mb_per_gpu": peak_mb,
                "seed": args.seed,
            }
            results.append(row)
            print(f"  n={n:,}  G={world_size}  {elapsed:.3f}s", flush=True)

        del sim
        torch.cuda.empty_cache()

    if rank == 0:
        out_path = os.path.join(args.out_dir, f"multi_gpu_{world_size}gpu_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"Saved: {out_path}")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Large-scale MLMC scaling (B1) — targeted (n, ε) pairs only")
    parser.add_argument("--mode", choices=["single", "distributed"], default="single")
    parser.add_argument("--n-nodes", type=int, nargs="+",
                        default=[1000, 5000, 10000, 25000, 50000],
                        help="Graph sizes. Single-GPU: moderate n gets all ε, "
                             "large n gets ε=0.05 only. Distributed: large n only.")
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.05, 0.01, 0.005, 0.002],
                        help="Epsilon values for moderate n (n≤5000) only")
    parser.add_argument("--bridge", action="store_true", default=True,
                        help="Include bridge point n=10000, ε=0.01 (default: on)")
    parser.add_argument("--no-bridge", dest="bridge", action="store_false")
    parser.add_argument("--bridge-n", type=int, default=10000)
    parser.add_argument("--bridge-eps", type=float, default=0.01)
    parser.add_argument("--L-max", type=int, default=4)
    parser.add_argument("--N-pilot", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/large_scale")
    args = parser.parse_args()

    if args.mode == "single":
        run_single_gpu(args)
    else:
        run_distributed(args)


if __name__ == "__main__":
    main()
