"""
CPU-MLMC Baseline Timing Script
Runs MLMCSimulator (CPU/NumPy) on synthetic ER graphs and compares timings
against the GPU results already stored in results/results/tables/.
Saves output to results/results/tables/cpu_vs_gpu_mlmc.json
"""

import sys
import os
import json
import csv
import time
from pathlib import Path

# Resolve paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from network.topology import TopologyGenerator
from network.traffic import PoissonTraffic
from simulation.mlmc import MLMCSimulator

GPU_CSV = REPO_ROOT / "results/results/tables/colab_prior_work_run_true_caida_rows.csv"
OUTPUT_JSON = REPO_ROOT / "results/results/tables/cpu_vs_gpu_mlmc.json"


def load_gpu_timings():
    """Load GPU-MLMC timings from the Colab/A100 results CSV."""
    rows = {}
    if not GPU_CSV.exists():
        print(f"WARNING: GPU CSV not found at {GPU_CSV}")
        return rows
    with open(GPU_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["scenario"], float(row["epsilon"]))
            rows[key] = {
                "gpu_mc_time_s": float(row["mc_runtime_s"]),
                "gpu_mlmc_time_s": float(row["mlmc_runtime_s"]),
                "gpu_cost_ratio": float(row["cost_ratio_mc_over_mlmc"]),
                "gpu_mlmc_estimate": float(row["mlmc_estimate"]),
            }
    return rows


def run_cpu_mlmc(n_nodes, epsilon, T=5.0, base_dt=0.1, L_max=4, n_seeds=2):
    """
    Time CPU-MLMC for a given network size and accuracy target.
    Returns mean wall-clock time and estimate over n_seeds runs.
    """
    times = []
    estimates = []
    for seed in range(n_seeds):
        gen = TopologyGenerator(seed=seed)
        network = gen.generate_erdos_renyi(n_nodes=n_nodes, p=0.3)
        network.set_link_properties(seed=seed)
        traffic = PoissonTraffic(rate=5.0, seed=seed)
        sim = MLMCSimulator(seed=seed)
        try:
            t0 = time.perf_counter()
            result = sim.estimate(
                network=network,
                traffic=traffic,
                epsilon=epsilon,
                T=T,
                base_dt=base_dt,
                L_max=L_max,
            )
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            estimates.append(result.mean)
        except Exception as e:
            print(f"  WARNING: seed={seed} failed: {e}")
    if not times:
        return None
    return {
        "cpu_mlmc_mean_time_s": round(sum(times) / len(times), 4),
        "cpu_mlmc_std_time_s": round((max(times) - min(times)) / 2, 4),
        "cpu_mlmc_estimate": round(sum(estimates) / len(estimates), 6),
        "n_seeds_ok": len(times),
    }


def main():
    print("=" * 60)
    print("CPU-MLMC Baseline Timing Experiment")
    print("=" * 60)

    gpu_rows = load_gpu_timings()

    # Configurations: (n_nodes, epsilon, scenario_key, L_max)
    configs = [
        (100, 0.10, "synthetic_n100", 4),
        (100, 0.05, "synthetic_n100", 4),
        (100, 0.02, "synthetic_n100", 5),
        (500, 0.10, "synthetic_n500", 4),
        (500, 0.05, "synthetic_n500", 5),
        (500, 0.02, "synthetic_n500", 5),
    ]

    results = []
    for n_nodes, epsilon, scenario_key, L_max in configs:
        print(f"\nRunning CPU-MLMC: n={n_nodes}, ε={epsilon}, L_max={L_max} ...")
        cpu_result = run_cpu_mlmc(n_nodes=n_nodes, epsilon=epsilon, L_max=L_max, n_seeds=2)

        gpu_key = (scenario_key, epsilon)
        gpu_data = gpu_rows.get(gpu_key, {})

        cpu_time = cpu_result["cpu_mlmc_mean_time_s"] if cpu_result else None
        gpu_time = gpu_data.get("gpu_mlmc_time_s")

        speedup = None
        if cpu_time and gpu_time and gpu_time > 0:
            speedup = round(cpu_time / gpu_time, 2)

        row = {
            "scenario": scenario_key,
            "n_nodes": n_nodes,
            "epsilon": epsilon,
            "cpu_mlmc_time_s": cpu_time,
            "cpu_mlmc_std_s": cpu_result["cpu_mlmc_std_time_s"] if cpu_result else None,
            "gpu_mlmc_time_s": round(gpu_time, 4) if gpu_time else None,
            "gpu_mc_time_s": round(gpu_data.get("gpu_mc_time_s", 0), 4) if gpu_data else None,
            "cpu_to_gpu_speedup": speedup,
            "gpu_cost_ratio": gpu_data.get("gpu_cost_ratio"),
            "cpu_mlmc_estimate": cpu_result["cpu_mlmc_estimate"] if cpu_result else None,
            "gpu_mlmc_estimate": gpu_data.get("gpu_mlmc_estimate"),
        }
        results.append(row)
        print(
            f"  CPU-MLMC: {cpu_time:.3f}s  |  GPU-MLMC: {gpu_time:.4f}s  |  Speedup: {speedup}x"
            if cpu_time and gpu_time
            else f"  CPU-MLMC: {cpu_time}  |  GPU-MLMC: {gpu_time}"
        )

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Scenario':<20} {'ε':>6} {'CPU-MLMC(s)':>12} {'GPU-MLMC(s)':>12} {'GPU Speedup':>12} {'CostRatio':>10}")
    print("-" * 80)
    for r in results:
        c = f"{r['cpu_mlmc_time_s']:.3f}" if r["cpu_mlmc_time_s"] else "N/A"
        g = f"{r['gpu_mlmc_time_s']:.4f}" if r["gpu_mlmc_time_s"] else "N/A"
        sp = f"{r['cpu_to_gpu_speedup']:.1f}x" if r["cpu_to_gpu_speedup"] else "N/A"
        cr = f"{r['gpu_cost_ratio']:.1f}" if r["gpu_cost_ratio"] else "N/A"
        print(f"{r['scenario']:<20} {r['epsilon']:>6} {c:>12} {g:>12} {sp:>12} {cr:>10}")
    print("=" * 80)

    # Save
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDONE — saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
