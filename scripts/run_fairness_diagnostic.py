"""ANA-MLMC fairness diagnostic: per-node CI sharpening vs node centrality.

The Section V-C ANA benchmark only measures the global estimator CI half-width.
Reviewer feedback (resolved in Section V-H, Limitations) flagged that the more
operator-relevant question is *which* nodes get sharpened estimates under
different weight configurations. This script measures per-node CI half-widths
on a fixed Barabasi-Albert n=300 scale-free graph and compares them across
five named ANA configurations plus the standard Giles baseline.

Caveat surfaced by this diagnostic: the current ANA-MLMC implementation
chooses the global sample budget N_l (per level) to minimise a weighted
variance, but every node within a level receives the same N_l samples.
Per-node CI therefore sharpens *uniformly* by sqrt(N_l_baseline / N_l_ANA),
not differentially across nodes. This script verifies the prediction
empirically and produces the figure used in Section V-C / V-H.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gpu.parallel_mc import GPUAdaptiveNetworkAwareMLMC, GPUCoupledPropagationMLMC
from network.topology import TopologyGenerator, centrality_weights


def _per_node_ci_from_paths(paths: np.ndarray, level: int = 0, N_per_level: List[int] | None = None) -> np.ndarray:
    """Compute per-node 95% CI half-widths from MLMC level statistics.

    Approximation: dominated by the finest level's statistic, which is what the
    framework also reports. Returns shape (n,).
    """
    var_per_node = paths.var(axis=0, ddof=1)
    n_paths = paths.shape[0]
    half_width = 1.96 * np.sqrt(var_per_node / max(n_paths, 1))
    return half_width


def run_one(adjacency: np.ndarray, *, simulator: str, weights: Tuple[float, float, float] | None,
            epsilon: float, T: float, base_dt: float, L_max: int, pilot_samples: int, seed: int) -> Dict:
    """Run a single sampler config and capture per-node statistics from level 0."""
    n_nodes = adjacency.shape[0]
    if simulator == "baseline":
        sim = GPUCoupledPropagationMLMC(adjacency_matrix=adjacency, seed=seed)
        full = sim.mlmc_estimate(epsilon=epsilon, T=T, base_dt=base_dt,
                                 L_max=L_max, pilot_samples=pilot_samples, verbose=False)
        n_finest = full["level_stats"][0]["n_samples"]
    else:
        assert weights is not None
        sim = GPUAdaptiveNetworkAwareMLMC(
            adjacency_matrix=adjacency, seed=seed,
            weight_centrality=weights[0], weight_variance=weights[1], weight_sla=weights[2],
        )
        full = sim.mlmc_estimate_weighted(epsilon=epsilon, T=T, base_dt=base_dt,
                                          L_max=L_max, pilot_samples=pilot_samples, verbose=False)
        n_finest = full["level_stats"][0]["n_samples"]

    # Re-draw n_finest paths at the finest level to capture per-node stats.
    # _run_level_state_tensors is defined on the parent class GPUCoupledPropagationMLMC
    # and returns torch tensors of shape (n_nodes, n_samples) on the device.
    c_fine, _c_coarse = sim._run_level_state_tensors(level=0, n_samples=int(n_finest),
                                                     T=T, base_dt=base_dt)
    # Move to CPU/numpy and orient as (n_samples, n_nodes).
    endpoints = c_fine.detach().cpu().numpy().T
    per_node_ci = _per_node_ci_from_paths(endpoints)
    return {
        "total_cost": float(full["total_cost"]),
        "n_finest_paths": int(n_finest),
        "per_node_ci": per_node_ci,
        "estimate": float(full["estimate"]),
        "endpoints_mean": endpoints.mean(axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--base-dt", type=float, default=0.1)
    parser.add_argument("--L-max", type=int, default=4)
    parser.add_argument("--pilot-samples", type=int, default=50)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "fairness_diagnostic")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    configs: List[Tuple[str, Tuple[float, float, float] | None]] = [
        ("baseline_giles",   None),
        ("centrality_only",  (1.0, 0.0, 0.0)),
        ("variance_only",    (0.0, 1.0, 0.0)),
        ("sla_only",         (0.0, 0.0, 1.0)),
        ("default_mix",      (0.4, 0.4, 0.2)),
    ]

    rows: List[Dict] = []
    per_config: Dict[str, List[np.ndarray]] = {name: [] for name, _ in configs}

    centralities_per_seed: List[np.ndarray] = []

    for seed in range(args.seeds):
        graph = TopologyGenerator(seed=seed).generate_barabasi_albert(n_nodes=args.n, m=3).get_largest_component()
        adjacency = graph.get_adjacency_matrix()
        centrality = centrality_weights(graph, kind="pagerank")
        # If get_largest_component reduced node count, align centrality length with adjacency.
        if centrality.shape[0] != adjacency.shape[0]:
            centrality = centrality[: adjacency.shape[0]]
        centralities_per_seed.append(centrality)

        for name, weights in configs:
            sim_kind = "baseline" if name == "baseline_giles" else "ana"
            t0 = time.perf_counter()
            res = run_one(adjacency, simulator=sim_kind, weights=weights,
                          epsilon=args.epsilon, T=args.T, base_dt=args.base_dt,
                          L_max=args.L_max, pilot_samples=args.pilot_samples, seed=seed)
            elapsed = time.perf_counter() - t0
            per_config[name].append(res["per_node_ci"])
            rows.append({
                "config": name, "seed": seed,
                "total_cost": res["total_cost"],
                "n_finest_paths": res["n_finest_paths"],
                "median_per_node_ci": float(np.median(res["per_node_ci"])),
                "mean_per_node_ci":   float(np.mean(res["per_node_ci"])),
                "elapsed_s": elapsed,
            })
            print(f"  seed={seed} cfg={name:<16} cost={res['total_cost']:>9.0f}  median_node_ci={np.median(res['per_node_ci']):.5f}  elapsed={elapsed:.2f}s")

    # Per-seed aggregation: stack per-node CIs across seeds (shape: n_seeds, n).
    summary: Dict[str, Dict] = {}
    for name in per_config:
        stack = np.stack(per_config[name], axis=0)  # (n_seeds, n)
        mean_per_node = stack.mean(axis=0)
        summary[name] = {
            "median_per_node_ci": float(np.median(mean_per_node)),
            "mean_per_node_ci":   float(np.mean(mean_per_node)),
            "p95_per_node_ci":    float(np.percentile(mean_per_node, 95)),
            "p5_per_node_ci":     float(np.percentile(mean_per_node, 5)),
            "n_seeds":            stack.shape[0],
            "n_nodes":            stack.shape[1],
        }

    # Compute sharpening ratio (ANA / baseline) per node, averaged over seeds.
    baseline_stack = np.stack(per_config["baseline_giles"], axis=0).mean(axis=0)
    sharpening: Dict[str, np.ndarray] = {}
    for name in per_config:
        if name == "baseline_giles":
            continue
        ana_stack = np.stack(per_config[name], axis=0).mean(axis=0)
        sharpening[name] = ana_stack / np.maximum(baseline_stack, 1e-12)

    # Centrality alignment (use seed 0's centrality for ranking)
    centrality_ref = centralities_per_seed[0]

    csv_path = args.output / "fairness_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    json_path = args.output / "fairness_summary.json"
    json_path.write_text(json.dumps({
        "epsilon": args.epsilon, "n_nodes": args.n,
        "T": args.T, "L_max": args.L_max, "pilot_samples": args.pilot_samples,
        "summary": summary,
    }, indent=2))
    np.savez(args.output / "fairness_arrays.npz",
             centrality=centrality_ref,
             baseline_per_node_ci=baseline_stack,
             **{f"sharpening_{n}": s for n, s in sharpening.items()})

    print(f"\n=== Per-config summary (n={args.n}, eps={args.epsilon}, {args.seeds} seeds) ===")
    print(f"{'config':<20}{'median_CI':>14}{'mean_CI':>14}{'p5_CI':>14}{'p95_CI':>14}")
    for name in [n for n, _ in configs]:
        s = summary[name]
        print(f"{name:<20}{s['median_per_node_ci']:>14.5f}{s['mean_per_node_ci']:>14.5f}{s['p5_per_node_ci']:>14.5f}{s['p95_per_node_ci']:>14.5f}")

    print("\n=== ANA / baseline per-node CI ratio (mean +/- SD across nodes) ===")
    print(f"{'config':<20}{'mean ratio':>14}{'sd':>10}{'min':>10}{'max':>10}")
    for name, s in sharpening.items():
        print(f"{name:<20}{s.mean():>14.4f}{s.std():>10.4f}{s.min():>10.4f}{s.max():>10.4f}")

    print(f"\nWrote {csv_path}, {json_path}, fairness_arrays.npz")


if __name__ == "__main__":
    main()
