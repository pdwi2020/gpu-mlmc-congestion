"""
Run baseline GPU-MLMC versus Adaptive Network-Aware MLMC.

Run from repo root:
    python3 scripts/run_ana_mlmc_experiment.py --network er --n 500
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gpu.parallel_mc import GPUAdaptiveNetworkAwareMLMC, GPUCoupledPropagationMLMC
from network.topology import NetworkGraph, TopologyGenerator, load_caida_topology


logger = logging.getLogger(__name__)


def _parse_epsilons(raw: str) -> List[float]:
    """Parse a comma-separated epsilon list."""
    epsilons = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not epsilons:
        raise ValueError("At least one epsilon is required")
    return epsilons


def _limit_network(network: NetworkGraph, n_nodes: int) -> NetworkGraph:
    """Keep the highest-degree n-node induced subgraph."""
    if network.n_nodes <= n_nodes:
        return network

    selected = [
        node
        for node, _ in sorted(network.graph.degree(), key=lambda item: item[1], reverse=True)[:n_nodes]
    ]
    limited = NetworkGraph(directed=network.graph.is_directed())
    limited.graph = network.graph.subgraph(selected).copy()
    return limited.get_largest_component()


def _load_network(kind: str, n_nodes: int, seed: int) -> NetworkGraph:
    """Load or synthesize the requested network."""
    generator = TopologyGenerator(seed=seed)
    if kind == "er":
        p = max(0.01, min(0.05, 8.0 / max(float(n_nodes), 1.0)))
        network = generator.generate_erdos_renyi(n_nodes=n_nodes, p=p)
        return network.get_largest_component()

    if kind == "ba":
        network = generator.generate_barabasi_albert(n_nodes=n_nodes, m=3)
        return network.get_largest_component()

    caida_dir = ROOT / "datasets" / "caida"
    candidates = sorted(
        list(caida_dir.glob("*.as-rel2.txt.bz2"))
        + list(caida_dir.glob("*.as-rel2.txt.gz"))
        + list(caida_dir.glob("*.as-rel2.txt"))
    )
    if candidates:
        network = load_caida_topology(candidates[-1], as_undirected=True, largest_component=True)
        return _limit_network(network, n_nodes)

    logger.warning("No local CAIDA AS-REL2 file found; using Barabasi-Albert fallback")
    return generator.generate_barabasi_albert(n_nodes=n_nodes, m=3)


def _run_baseline(adjacency: np.ndarray,
                  epsilon: float,
                  seed: int,
                  T: float,
                  base_dt: float,
                  L_max: int,
                  pilot_samples: int) -> Dict:
    """Run baseline coupled GPU-MLMC and include runtime."""
    simulator = GPUCoupledPropagationMLMC(adjacency_matrix=adjacency, seed=seed)
    started = time.perf_counter()
    result = simulator.mlmc_estimate(
        epsilon=epsilon,
        T=T,
        base_dt=base_dt,
        L_max=L_max,
        pilot_samples=pilot_samples,
        verbose=False,
    )
    result["runtime_s"] = time.perf_counter() - started
    return result


def _run_ana(adjacency: np.ndarray,
             epsilon: float,
             seed: int,
             T: float,
             base_dt: float,
             L_max: int,
             pilot_samples: int) -> Dict:
    """Run GPU ANA-MLMC and include runtime."""
    simulator = GPUAdaptiveNetworkAwareMLMC(adjacency_matrix=adjacency, seed=seed)
    started = time.perf_counter()
    result = simulator.mlmc_estimate_weighted(
        epsilon=epsilon,
        T=T,
        base_dt=base_dt,
        L_max=L_max,
        pilot_samples=pilot_samples,
        verbose=False,
    )
    result["runtime_s"] = time.perf_counter() - started
    return result


def _result_row(method: str,
                network_kind: str,
                n_nodes: int,
                epsilon: float,
                seed: int,
                result: Dict) -> Dict:
    """Flatten a simulator result for CSV output."""
    return {
        "method": method,
        "network": network_kind,
        "n_nodes": n_nodes,
        "epsilon": epsilon,
        "seed": seed,
        "total_cost": float(result["total_cost"]),
        "runtime_s": float(result["runtime_s"]),
        "estimate": float(result["estimate"]),
        "ci_lower": float(result["ci_lower"]),
        "ci_upper": float(result["ci_upper"]),
    }


def _write_outputs(rows: List[Dict], output_dir: Path) -> None:
    """Write CSV rows and JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ana_mlmc_results.csv"
    json_path = output_dir / "ana_mlmc_summary.json"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_rows": len(rows),
        "methods": sorted({row["method"] for row in rows}),
        "mean_by_method": {},
    }
    for method in summary["methods"]:
        method_rows = [row for row in rows if row["method"] == method]
        summary["mean_by_method"][method] = {
            "total_cost": float(np.mean([row["total_cost"] for row in method_rows])),
            "runtime_s": float(np.mean([row["runtime_s"] for row in method_rows])),
            "estimate": float(np.mean([row["estimate"] for row in method_rows])),
        }

    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


def _print_table(rows: List[Dict]) -> None:
    """Print a compact baseline versus ANA comparison table."""
    print()
    print("network epsilon seed baseline_cost ana_cost cost_ratio baseline_s ana_s")
    grouped = {}
    for row in rows:
        key = (row["network"], row["epsilon"], row["seed"])
        grouped.setdefault(key, {})[row["method"]] = row

    for (network_kind, epsilon, seed), pair in sorted(grouped.items()):
        baseline = pair["baseline"]
        ana = pair["ana"]
        ratio = baseline["total_cost"] / ana["total_cost"] if ana["total_cost"] > 0 else float("inf")
        print(
            f"{network_kind:7s} {epsilon:<7g} {seed:<4d} "
            f"{baseline['total_cost']:<13.2f} {ana['total_cost']:<9.2f} "
            f"{ratio:<10.3f} {baseline['runtime_s']:<10.3f} {ana['runtime_s']:<.3f}"
        )


def main() -> None:
    """Parse CLI arguments and run the ANA-MLMC experiment."""
    parser = argparse.ArgumentParser(description="Run ANA-MLMC benchmark")
    parser.add_argument("--network", choices=["er", "ba", "caida"], default="er")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--epsilons", default="0.05,0.02,0.01")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "ana_mlmc")
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--base-dt", type=float, default=0.1)
    parser.add_argument("--L-max", type=int, default=4)
    parser.add_argument("--pilot-samples", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    epsilons = _parse_epsilons(args.epsilons)
    rows = []

    for seed in range(args.seeds):
        network = _load_network(args.network, args.n, seed)
        adjacency = network.get_adjacency_matrix().astype(np.float32)
        for epsilon in epsilons:
            baseline = _run_baseline(
                adjacency, epsilon, seed, args.T, args.base_dt, args.L_max, args.pilot_samples
            )
            ana = _run_ana(
                adjacency, epsilon, seed, args.T, args.base_dt, args.L_max, args.pilot_samples
            )
            if ana["total_cost"] > baseline["total_cost"] * 1.05:
                logger.warning(
                    "ANA cost exceeded baseline by more than 5%% for seed=%s epsilon=%s",
                    seed,
                    epsilon,
                )

            rows.append(_result_row("baseline", args.network, network.n_nodes, epsilon, seed, baseline))
            rows.append(_result_row("ana", args.network, network.n_nodes, epsilon, seed, ana))

    _write_outputs(rows, args.output)
    _print_table(rows)


if __name__ == "__main__":
    main()
