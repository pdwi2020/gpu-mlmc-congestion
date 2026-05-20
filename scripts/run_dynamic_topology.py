#!/usr/bin/env python3
"""Run the dynamic topology validation experiment on a CAIDA subgraph."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "datasets"))

from caida.loader import CAIDATopologyLoader
from gpu.parallel_mc import GPUCoupledPropagationMLMC
from network.dynamic_traffic import DynamicArrivalProcess
from network.link_failures import LinkFailureSchedule
from network.sde import CongestionPropagationSDE
from network.topology import NetworkGraph


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--network", type=str, default="caida", choices=["caida"])
    parser.add_argument("--T", type=float, default=24.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/dynamic_topology/"))
    parser.add_argument("--caida-date", type=str, default="20260101")
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--l-max", type=int, default=3)
    parser.add_argument("--pilot-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--event-window", type=float, default=0.5)
    return parser.parse_args()


def load_caida_subgraph(n_nodes: int, date: str, seed: int) -> NetworkGraph:
    """Load a CAIDA topology and return a relabeled n-node subgraph."""
    loader = CAIDATopologyLoader(data_dir=REPO_ROOT / "datasets" / "caida")
    network = loader.load_topology(
        date=date,
        download_if_missing=True,
        as_undirected=True,
        largest_component=True,
        add_link_properties=False,
        seed=seed,
    )
    return extract_degree_subgraph(network, n_nodes)


def extract_degree_subgraph(network: NetworkGraph, n_nodes: int) -> NetworkGraph:
    """Extract a high-degree relabeled subgraph from a NetworkGraph."""
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if network.n_nodes < n_nodes:
        raise ValueError(f"Requested {n_nodes} nodes but topology has {network.n_nodes}")

    ranked_nodes = sorted(network.graph.degree(), key=lambda item: item[1], reverse=True)
    selected = [node for node, _ in ranked_nodes[:n_nodes]]
    subgraph_nx = network.graph.subgraph(selected).copy()
    if subgraph_nx.number_of_edges() == 0:
        raise ValueError("Selected CAIDA subgraph has no edges")

    relabeled = nx.convert_node_labels_to_integers(subgraph_nx, ordering="default")
    subgraph = NetworkGraph(directed=network.graph.is_directed())
    subgraph.graph = relabeled
    return subgraph


def make_dynamic_arrivals(T: float) -> DynamicArrivalProcess:
    """Create the 24-hour diurnal traffic process with three burst events."""
    return DynamicArrivalProcess(
        lambda_base=0.15,
        amplitude=0.3,
        period_hours=24.0,
        burst_rate=3.0 / T,
        burst_duration=0.35,
        burst_magnitude=6.0,
        jitter_std=0.02,
        n_bursts=3,
        global_bursts=True,
        burst_node_fraction=0.2,
    )


def compute_ci_evolution(paths: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-node mean and mean +/- 2 sigma bands across seed paths."""
    mean = paths.mean(axis=0)
    ddof = 1 if paths.shape[0] > 1 else 0
    sigma = paths.std(axis=0, ddof=ddof)
    return {
        "mean": mean,
        "ci_lower": mean - 2.0 * sigma,
        "ci_upper": mean + 2.0 * sigma,
        "ci_width": 4.0 * sigma,
    }


def event_mask(time_grid: np.ndarray, event_times: List[float], window: float) -> np.ndarray:
    """Return a boolean mask for time points near any event."""
    mask = np.zeros(time_grid.shape, dtype=bool)
    for event_time in event_times:
        mask |= np.abs(time_grid - event_time) <= window
    return mask


def summarize_ci_widths(
    time_grid: np.ndarray,
    total_width: np.ndarray,
    burst_onsets: List[float],
    failure_times: List[float],
    event_window: float,
) -> Tuple[float, float]:
    """Summarize peak CI width near failures and during quiet periods."""
    failure_near = event_mask(time_grid, failure_times, event_window)
    burst_near = event_mask(time_grid, burst_onsets, event_window)
    quiet = ~(failure_near | burst_near)

    peak_failure = float(total_width[failure_near].max()) if np.any(failure_near) else 0.0
    peak_quiet = float(total_width[quiet].max()) if np.any(quiet) else 0.0
    return peak_failure, peak_quiet


def plot_ci_evolution(
    time_grid: np.ndarray,
    total_mean: np.ndarray,
    total_sigma: np.ndarray,
    burst_onsets: List[float],
    failure_times: List[float],
    figure_path: Path,
) -> None:
    """Write the dynamic topology CI evolution figure."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    ci_lower = total_mean - 2.0 * total_sigma
    ci_upper = total_mean + 2.0 * total_sigma
    ci_width = ci_upper - ci_lower

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10.5, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_top.plot(time_grid, total_mean, color="#1f2937", linewidth=1.8, label="Mean")
    ax_top.fill_between(
        time_grid,
        ci_lower,
        ci_upper,
        color="#4f83cc",
        alpha=0.28,
        linewidth=0.0,
        label="95% CI",
    )
    for onset in burst_onsets:
        ax_top.axvline(onset, color="#b91c1c", alpha=0.55, linewidth=1.0)
    for failure_time in failure_times:
        ax_top.axvline(failure_time, color="#5b21b6", alpha=0.55, linewidth=1.0, linestyle="--")
    ax_top.set_ylabel("Total network congestion")
    ax_top.legend(loc="upper left", frameon=False)
    ax_top.grid(alpha=0.25)

    ax_bottom.plot(time_grid, ci_width, color="#0f766e", linewidth=1.5)
    ax_bottom.set_xlabel("Time (hours)")
    ax_bottom.set_ylabel("CI width")
    ax_bottom.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def run_seed(
    seed: int,
    adjacency: np.ndarray,
    graph: NetworkGraph,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, Any], List[float], List[float]]:
    """Run one dynamic seed and return the path plus MLMC summary metadata."""
    n_steps = int(args.T / args.dt)
    interval_times = np.arange(n_steps, dtype=float) * args.dt
    arrival_process = make_dynamic_arrivals(args.T)
    lambda_t = arrival_process.generate(
        n_nodes=adjacency.shape[0],
        T=args.T,
        dt=args.dt,
        seed=seed,
    ).astype(np.float32)
    failure_schedule = LinkFailureSchedule.schedule_failures(
        graph=graph,
        T=args.T,
        dt=args.dt,
        n_failures=5,
        seed=seed,
        recovery_time=2.0,
    )
    adjacency_t = failure_schedule.as_time_series(interval_times).astype(np.float32)

    mlmc = GPUCoupledPropagationMLMC(adjacency_matrix=adjacency.astype(np.float32), seed=seed)
    mlmc_result = mlmc.mlmc_estimate(
        epsilon=args.epsilon,
        T=args.T,
        base_dt=args.dt,
        L_max=args.l_max,
        pilot_samples=args.pilot_samples,
        metric="sum_congestion",
        verbose=False,
        lambda_t=lambda_t,
        adjacency_t=adjacency_t,
    )

    sde = CongestionPropagationSDE(
        adjacency_matrix=adjacency,
        influence_strength=0.1,
        decay_rate=0.5,
        noise_intensity=0.1,
    )
    _, path = sde.simulate_with_dynamic_inputs(
        T=args.T,
        dt=args.dt,
        lambda_t=lambda_t,
        adjacency_t=adjacency_t,
        seed=seed,
    )
    burst_onsets = [float(value) for value in arrival_process.last_burst_onsets]
    failure_times = [float(event.start_time) for event in failure_schedule.events]
    return path, mlmc_result, burst_onsets, failure_times


def main() -> None:
    """Run the dynamic topology experiment and write outputs."""
    args = parse_args()
    start = time.perf_counter()
    output_dir = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = load_caida_subgraph(n_nodes=args.n, date=args.caida_date, seed=args.seed)
    adjacency = graph.get_adjacency_matrix().astype(np.float32)

    seed_values = [args.seed + idx for idx in range(args.n_seeds)]
    paths = []
    mlmc_results = []
    all_bursts: List[float] = []
    all_failures: List[float] = []
    for seed in seed_values:
        path, mlmc_result, burst_onsets, failure_times = run_seed(seed, adjacency, graph, args)
        paths.append(path)
        mlmc_results.append(mlmc_result)
        all_bursts.extend(burst_onsets)
        all_failures.extend(failure_times)

    path_array = np.stack(paths, axis=0)
    n_steps = int(args.T / args.dt)
    time_grid = np.linspace(0.0, args.T, n_steps + 1)
    ci = compute_ci_evolution(path_array)

    npz_path = output_dir / "ci_evolution.npz"
    np.savez_compressed(
        npz_path,
        time=time_grid,
        mean=ci["mean"],
        ci_lower=ci["ci_lower"],
        ci_upper=ci["ci_upper"],
        ci_width=ci["ci_width"],
        seeds=np.array(seed_values, dtype=int),
    )

    total_paths = path_array.sum(axis=2)
    total_mean = total_paths.mean(axis=0)
    total_sigma = total_paths.std(axis=0, ddof=1 if args.n_seeds > 1 else 0)
    total_width = 4.0 * total_sigma
    unique_bursts = sorted({round(value, 6) for value in all_bursts})
    unique_failures = sorted({round(value, 6) for value in all_failures})
    peak_failure, peak_quiet = summarize_ci_widths(
        time_grid=time_grid,
        total_width=total_width,
        burst_onsets=unique_bursts,
        failure_times=unique_failures,
        event_window=args.event_window,
    )

    figure_path = REPO_ROOT / "paper" / "figures" / "dynamic_topology_ci_evolution.png"
    plot_ci_evolution(
        time_grid=time_grid,
        total_mean=total_mean,
        total_sigma=total_sigma,
        burst_onsets=unique_bursts,
        failure_times=unique_failures,
        figure_path=figure_path,
    )

    runtime = time.perf_counter() - start
    summary = {
        "n_seeds": args.n_seeds,
        "seeds": seed_values,
        "network": args.network,
        "n_nodes": args.n,
        "T": args.T,
        "dt": args.dt,
        "epsilon": args.epsilon,
        "runtime_seconds": runtime,
        "peak_ci_width_near_failures": peak_failure,
        "peak_ci_width_quiet": peak_quiet,
        "burst_onsets": unique_bursts,
        "failure_times": unique_failures,
        "ci_evolution_npz": str(npz_path),
        "figure": str(figure_path),
        "mlmc_results": mlmc_results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "Dynamic topology summary: "
        f"seeds={args.n_seeds}, runtime={runtime:.2f}s, "
        f"peak_ci_width_near_failures={peak_failure:.6g}, "
        f"peak_ci_width_quiet={peak_quiet:.6g}"
    )


if __name__ == "__main__":
    main()
