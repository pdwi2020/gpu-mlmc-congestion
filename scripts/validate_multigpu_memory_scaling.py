"""
Validate memory scaling of MultiGPUMLMC across graph sizes and partition counts.

For n in {500, 1000, 2000, 5000, 10000} and G in {1, 2, 4}:
  - Measure allocated memory before and after MultiGPUMLMC init
  - Compare against analytical model: mem ≈ n² / G × bytes_per_entry + halo_overhead
  - Report halo overhead fraction

Saves results to results/multi_gpu/memory_scaling.json.
"""

import json
import os
import sys
import time
import tracemalloc

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gpu.parallel_mc import MultiGPUMLMC  # noqa: E402


def ba_adjacency(n: int, m: int = 3, seed: int = 42) -> np.ndarray:
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    return nx.to_numpy_array(g, dtype=np.float32)


def measure_memory_bytes(func):
    """Run func(), return (result, peak_bytes) using tracemalloc."""
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    result = func()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    peak_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
    return result, max(peak_bytes, 0)


def run_validation():
    n_vals = [500, 1000, 2000, 5000, 10000]
    g_vals = [1, 2, 4]
    # 4 bytes per float32 entry; adjacency + influence tensors (≈ 2 copies)
    BYTES_PER_ENTRY = 8.0
    N_PATHS = 32  # small batch to limit memory during measurement

    results = []

    print(f"{'n':>6} {'G':>3} {'mem_MB':>10} {'model_MB':>10} {'halo%':>8} {'n_halo':>8}")
    print("-" * 50)

    for n in n_vals:
        adj = ba_adjacency(n, m=3, seed=42)

        for G in g_vals:
            def _build():
                return MultiGPUMLMC(
                    adjacency_matrix=adj,
                    world_size=G,
                    rank=0,
                    influence_strength=0.1,
                    decay_rate=0.5,
                    noise_intensity=0.1,
                )

            sim, alloc_bytes = measure_memory_bytes(_build)
            mem_mb = alloc_bytes / (1024 ** 2)

            # Analytical model: n² / G float32 entries × 2 copies
            n_local = len(sim._local_nodes)
            model_bytes = (n * n_local) * BYTES_PER_ENTRY
            model_mb = model_bytes / (1024 ** 2)

            n_halo = len(sim._halo_edges)
            # Halo fraction: halo_edges / total_local_edges
            local_edges = int(adj[np.ix_(sim._local_nodes, sim._local_nodes)].sum())
            total_incident = int(adj[sim._local_nodes, :].sum())
            halo_pct = 100.0 * n_halo / max(total_incident, 1)

            print(f"{n:>6} {G:>3} {mem_mb:>10.2f} {model_mb:>10.2f} "
                  f"{halo_pct:>7.1f}% {n_halo:>8}")

            results.append({
                "n": n,
                "G": G,
                "n_local_nodes": n_local,
                "n_halo_edges": n_halo,
                "measured_mem_mb": round(mem_mb, 3),
                "model_mem_mb": round(model_mb, 3),
                "halo_pct": round(halo_pct, 2),
                "partitioner": "degree-weighted-striped" if G > 1 else "trivial",
            })

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "multi_gpu",
        "memory_scaling.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"description": "MultiGPUMLMC memory scaling", "rows": results}, f, indent=2)
    print(f"\nSaved to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    run_validation()
