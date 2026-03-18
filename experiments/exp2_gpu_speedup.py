"""
Experiment 2: GPU Speedup Evaluation

Measures GPU acceleration performance vs CPU baseline for Monte Carlo simulations.

Objectives:
- Measure GPU speedup across different sample sizes
- Evaluate scalability with network size
- Compare CPU (single-thread), CPU (multi-thread), and GPU
- Identify optimal batch sizes for GPU
- Measure GPU utilization and memory usage

Network sizes: [100, 500, 1000, 5000] nodes
Sample sizes: [10³, 10⁴, 10⁵, 10⁶] paths

Expected Results:
- GPU speedup: 100x-500x for large sample counts
- Speedup increases with sample size
- Efficient scaling up to GPU memory limits
- GPU advantage most significant for 10⁵+ samples
"""

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Multi-core CPU parallelism
try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logging.warning(
        "joblib not available - multi-core CPU benchmark will be skipped. "
        "Install with: pip install joblib"
    )

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import ExperimentConfig, setup_logging, setup_output_dirs

from network.topology import TopologyGenerator
from network.traffic import PoissonTraffic
from simulation.monte_carlo import MonteCarloSimulator

# Try to import GPU modules
try:
    from gpu.parallel_mc import GPUMLMCSimulator, GPUMonteCarloSimulator

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("GPU modules not available - using CPU-only mode")

logger = logging.getLogger(__name__)


def benchmark_cpu_single_thread(
    network, traffic, n_samples: int, T: float, dt: float, seed: int = 42
) -> Dict:
    """Benchmark CPU single-threaded performance.

    Args:
        network: Network topology
        traffic: Traffic model
        n_samples: Number of samples
        T: Simulation time
        dt: Timestep
        seed: Random seed

    Returns:
        Benchmark results dictionary
    """
    logger.info(f"Benchmarking CPU (single-thread): {n_samples} samples")

    simulator = MonteCarloSimulator(seed=seed)

    start_time = time.time()
    result = simulator.estimate(
        network=network,
        traffic=traffic,
        n_samples=n_samples,
        T=T,
        dt=dt,
        metric="mean_queue",
    )
    runtime = time.time() - start_time

    logger.info(f"Runtime: {runtime:.2f}s")
    logger.info(f"Throughput: {n_samples / runtime:.2f} samples/sec")

    return {
        "method": "CPU_single",
        "n_samples": n_samples,
        "runtime": runtime,
        "throughput": n_samples / runtime,
        "mean": result.mean,
        "variance": result.variance,
    }


def _single_mc_sample(
    arrival_rate: float,
    service_rate: float,
    noise_intensity: float,
    T: float,
    dt: float,
    seed: int,
) -> float:
    """Run one MC sample path (top-level function for joblib pickling).

    Args:
        arrival_rate: SDE arrival rate λ
        service_rate: SDE service rate μ
        noise_intensity: SDE noise σ
        T: Simulation duration
        dt: Timestep
        seed: Per-sample random seed

    Returns:
        Time-averaged mean queue length for this path
    """
    import numpy as _np

    rng = _np.random.default_rng(seed)
    n_steps = int(T / dt)
    drift_dt = (arrival_rate - service_rate) * dt
    sigma_sqrt_dt = noise_intensity * _np.sqrt(dt)
    q = 0.0
    q_sum = 0.0
    for _ in range(n_steps):
        dW = rng.standard_normal()
        q = max(0.0, q + drift_dt + sigma_sqrt_dt * dW)
        q_sum += q
    return q_sum / n_steps


def benchmark_cpu_multicore(
    network,
    traffic,
    n_samples: int,
    T: float,
    dt: float,
    seed: int = 42,
    n_jobs: int = -1,
) -> Dict:
    """Benchmark CPU multi-core (joblib) Monte Carlo performance.

    Each sample path is dispatched as an independent joblib task so that
    all available CPU cores are utilised.  This provides a fair CPU baseline
    against which the GPU speedup numbers can be compared.

    Args:
        network: Network topology (used only to extract traffic parameters)
        traffic: Traffic model
        n_samples: Number of MC samples
        T: Simulation duration
        dt: Timestep
        seed: Base random seed (per-sample seeds are derived from this)
        n_jobs: Number of parallel workers (-1 = all cores)

    Returns:
        Benchmark results dictionary with keys:
            method, n_samples, n_jobs_used, runtime, throughput, mean, variance
    """
    if not JOBLIB_AVAILABLE:
        logger.warning("joblib not available – returning NaN for multicore benchmark.")
        return {
            "method": "CPU_multicore",
            "n_samples": n_samples,
            "n_jobs_used": 0,
            "runtime": float("nan"),
            "throughput": float("nan"),
            "mean": float("nan"),
            "variance": float("nan"),
            "available": False,
        }

    import os

    n_cores = os.cpu_count() or 1
    actual_jobs = n_cores if n_jobs == -1 else min(n_jobs, n_cores)

    logger.info(
        f"Benchmarking CPU (multi-core, {actual_jobs} workers): {n_samples} samples"
    )

    # Derive SDE parameters from the traffic model (mirrors MLMCSimulator logic)
    traffic_stats = traffic.get_statistics(duration=T)
    arrival_rate = float(traffic_stats["arrival_rate"])
    service_rate = arrival_rate * 1.25
    noise_intensity = 0.2

    # Generate per-sample seeds deterministically
    rng_seed = np.random.default_rng(seed)
    per_sample_seeds = rng_seed.integers(0, 2**31, size=n_samples).tolist()

    start_time = time.time()

    samples = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_single_mc_sample)(
            arrival_rate, service_rate, noise_intensity, T, dt, s
        )
        for s in per_sample_seeds
    )

    runtime = time.time() - start_time
    samples = np.asarray(samples, dtype=np.float64)

    mean = float(np.mean(samples))
    variance = float(np.var(samples, ddof=1))
    throughput = n_samples / runtime

    logger.info(f"Runtime: {runtime:.2f}s  |  Throughput: {throughput:.0f} samples/sec")
    logger.info(f"Mean queue: {mean:.6f}  |  Variance: {variance:.6e}")

    return {
        "method": "CPU_multicore",
        "n_samples": n_samples,
        "n_jobs_used": actual_jobs,
        "runtime": runtime,
        "throughput": throughput,
        "mean": mean,
        "variance": variance,
        "available": True,
    }


def benchmark_gpu(
    network, traffic, n_samples: int, T: float, dt: float, seed: int = 42
) -> Dict:
    """Benchmark GPU performance.

    Args:
        network: Network topology
        traffic: Traffic model
        n_samples: Number of samples
        T: Simulation time
        dt: Timestep
        seed: Random seed

    Returns:
        Benchmark results dictionary
    """
    if not GPU_AVAILABLE:
        logger.warning("GPU not available, skipping GPU benchmark")
        return {
            "method": "GPU",
            "n_samples": n_samples,
            "runtime": np.nan,
            "throughput": np.nan,
            "mean": np.nan,
            "variance": np.nan,
            "available": False,
        }

    logger.info(f"Benchmarking GPU: {n_samples} samples")

    try:
        simulator = GPUMonteCarloSimulator(seed=seed)

        start_time = time.time()
        result = simulator.estimate(
            network=network,
            traffic=traffic,
            n_samples=n_samples,
            T=T,
            dt=dt,
            metric="mean_queue",
        )
        runtime = time.time() - start_time

        logger.info(f"Runtime: {runtime:.2f}s")
        logger.info(f"Throughput: {n_samples / runtime:.2f} samples/sec")

        return {
            "method": "GPU",
            "n_samples": n_samples,
            "runtime": runtime,
            "throughput": n_samples / runtime,
            "mean": result.mean,
            "variance": result.variance,
            "available": True,
        }

    except Exception as e:
        logger.error(f"GPU benchmark failed: {e}")
        return {
            "method": "GPU",
            "n_samples": n_samples,
            "runtime": np.nan,
            "throughput": np.nan,
            "mean": np.nan,
            "variance": np.nan,
            "available": False,
            "error": str(e),
        }


def benchmark_gpu_mlmc(
    network, traffic, n_samples: int, T: float, dt: float, seed: int = 42
) -> Dict:
    """Benchmark GPU-MLMC performance.

    Args:
        network: Network topology
        traffic: Traffic model
        n_samples: Number of samples
        T: Simulation time
        dt: Timestep
        seed: Random seed

    Returns:
        Benchmark results dictionary
    """
    if not GPU_AVAILABLE:
        logger.warning("GPU not available, skipping GPU-MLMC benchmark")
        return {
            "method": "GPU_MLMC",
            "n_samples": n_samples,
            "runtime": np.nan,
            "throughput": np.nan,
            "mean": np.nan,
            "variance": np.nan,
            "available": False,
        }

    logger.info(f"Benchmarking GPU-MLMC: {n_samples} samples")

    try:
        simulator = GPUMLMCSimulator(seed=seed)

        start_time = time.time()
        result = simulator.estimate(
            network=network,
            traffic=traffic,
            n_samples=n_samples,
            T=T,
            dt=dt,
            metric="mean_queue",
        )
        runtime = time.time() - start_time

        logger.info(f"Runtime: {runtime:.2f}s")
        logger.info(f"Throughput: {n_samples / runtime:.2f} samples/sec")

        return {
            "method": "GPU_MLMC",
            "n_samples": n_samples,
            "runtime": runtime,
            "throughput": n_samples / runtime,
            "mean": result.mean,
            "variance": result.variance,
            "available": True,
        }

    except Exception as e:
        logger.error(f"GPU-MLMC benchmark failed: {e}")
        return {
            "method": "GPU_MLMC",
            "n_samples": n_samples,
            "runtime": np.nan,
            "throughput": np.nan,
            "mean": np.nan,
            "variance": np.nan,
            "available": False,
            "error": str(e),
        }


def run_sample_size_scaling(
    network,
    traffic,
    sample_sizes: List[int],
    T: float,
    dt: float,
    seed: int = 42,
    gpu_only: bool = False,
) -> Dict:
    """Evaluate speedup across different sample sizes.

    Args:
        network: Network topology
        traffic: Traffic model
        sample_sizes: List of sample sizes to test
        T: Simulation time
        dt: Timestep
        seed: Random seed

    Returns:
        Dictionary with results for each sample size
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Sample Size Scaling Test")
    logger.info(f"Network: {network.n_nodes} nodes")
    logger.info(f"Sample sizes: {sample_sizes}")
    logger.info(f"{'=' * 80}\n")

    results = {}

    for n_samples in sample_sizes:
        logger.info(f"\n--- Testing {n_samples} samples ---")

        if gpu_only:
            gpu_mc_result = benchmark_gpu(network, traffic, n_samples, T, dt, seed)
            gpu_mlmc_result = benchmark_gpu_mlmc(
                network, traffic, n_samples, T, dt, seed
            )
            if (
                gpu_mc_result.get("available", False)
                and gpu_mlmc_result.get("available", False)
                and gpu_mlmc_result["runtime"] > 0
            ):
                speedup = gpu_mc_result["runtime"] / gpu_mlmc_result["runtime"]
            else:
                speedup = np.nan

            logger.info(
                f"GPU-MC vs GPU-MLMC speedup: {speedup:.2f}x"
                if not np.isnan(speedup)
                else "Speedup: N/A"
            )

            results[n_samples] = {
                "n_samples": n_samples,
                "gpu_mc": gpu_mc_result,
                "gpu_mlmc": gpu_mlmc_result,
                "speedup": speedup,
                "comparison": "GPU-MC vs GPU-MLMC",
                "cpu": gpu_mc_result,
                "gpu": gpu_mlmc_result,
            }
        else:
            cpu_single_result = benchmark_cpu_single_thread(
                network, traffic, n_samples, T, dt, seed
            )
            cpu_multi_result = benchmark_cpu_multicore(
                network, traffic, n_samples, T, dt, seed
            )
            gpu_result = benchmark_gpu(network, traffic, n_samples, T, dt, seed)

            # Speedup vs single-thread CPU (traditional metric)
            if gpu_result.get("available", False) and cpu_single_result["runtime"] > 0:
                speedup_single = cpu_single_result["runtime"] / gpu_result["runtime"]
            else:
                speedup_single = np.nan

            # Speedup vs multi-core CPU (fair baseline)
            if (
                gpu_result.get("available", False)
                and cpu_multi_result.get("available", False)
                and cpu_multi_result["runtime"] > 0
            ):
                speedup_multi = cpu_multi_result["runtime"] / gpu_result["runtime"]
            else:
                speedup_multi = np.nan

            # Multi-core vs single-thread scaling
            if (
                cpu_multi_result.get("available", False)
                and cpu_single_result["runtime"] > 0
            ):
                multicore_scaling = (
                    cpu_single_result["runtime"] / cpu_multi_result["runtime"]
                )
            else:
                multicore_scaling = np.nan

            logger.info(
                f"Speedup (vs single-thread): {speedup_single:.2f}x"
                if not np.isnan(speedup_single)
                else "N/A"
            )
            logger.info(
                f"Speedup (vs multicore {cpu_multi_result.get('n_jobs_used', '?')} cores): "
                f"{speedup_multi:.2f}x"
                if not np.isnan(speedup_multi)
                else "N/A"
            )

            results[n_samples] = {
                "n_samples": n_samples,
                "cpu": cpu_single_result,
                "cpu_multicore": cpu_multi_result,
                "gpu": gpu_result,
                "speedup": speedup_single,
                "speedup_vs_multicore": speedup_multi,
                "multicore_scaling": multicore_scaling,
            }

    return results


def run_network_size_scaling(
    network_sizes: List[int],
    n_samples: int,
    T: float,
    dt: float,
    seed: int = 42,
    gpu_only: bool = False,
) -> Dict:
    """Evaluate speedup across different network sizes.

    Args:
        network_sizes: List of node counts to test
        n_samples: Fixed number of samples
        T: Simulation time
        dt: Timestep
        seed: Random seed

    Returns:
        Dictionary with results for each network size
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Network Size Scaling Test")
    logger.info(f"Sample size: {n_samples}")
    logger.info(f"Network sizes: {network_sizes}")
    logger.info(f"{'=' * 80}\n")

    generator = TopologyGenerator(seed=seed)
    results = {}

    for n_nodes in network_sizes:
        logger.info(f"\n--- Testing {n_nodes} nodes ---")

        # Generate network
        network = generator.generate_erdos_renyi(
            n_nodes=n_nodes, p=0.15, directed=False
        )
        network.set_link_properties(
            bandwidth_range=(1e9, 10e9),
            delay_range=(0.001, 0.01),
            capacity_range=(500, 2000),
            seed=seed,
        )

        traffic = PoissonTraffic(rate=100.0, seed=seed)

        logger.info(f"Network: {network.n_nodes} nodes, {network.n_edges} edges")

        if gpu_only:
            gpu_mc_result = benchmark_gpu(network, traffic, n_samples, T, dt, seed)
            gpu_mlmc_result = benchmark_gpu_mlmc(
                network, traffic, n_samples, T, dt, seed
            )
            if (
                gpu_mc_result.get("available", False)
                and gpu_mlmc_result.get("available", False)
                and gpu_mlmc_result["runtime"] > 0
            ):
                speedup = gpu_mc_result["runtime"] / gpu_mlmc_result["runtime"]
            else:
                speedup = np.nan

            logger.info(
                f"GPU-MC vs GPU-MLMC speedup: {speedup:.2f}x"
                if not np.isnan(speedup)
                else "Speedup: N/A"
            )

            results[n_nodes] = {
                "n_nodes": n_nodes,
                "n_edges": network.n_edges,
                "gpu_mc": gpu_mc_result,
                "gpu_mlmc": gpu_mlmc_result,
                "speedup": speedup,
                "comparison": "GPU-MC vs GPU-MLMC",
                "cpu": gpu_mc_result,
                "gpu": gpu_mlmc_result,
            }
        else:
            cpu_result = benchmark_cpu_single_thread(
                network, traffic, n_samples, T, dt, seed
            )
            gpu_result = benchmark_gpu(network, traffic, n_samples, T, dt, seed)
            if gpu_result.get("available", False) and cpu_result["runtime"] > 0:
                speedup = cpu_result["runtime"] / gpu_result["runtime"]
            else:
                speedup = np.nan

            logger.info(
                f"Speedup: {speedup:.2f}x" if not np.isnan(speedup) else "Speedup: N/A"
            )

            results[n_nodes] = {
                "n_nodes": n_nodes,
                "n_edges": network.n_edges,
                "cpu": cpu_result,
                "gpu": gpu_result,
                "speedup": speedup,
            }

    return results


def analyze_speedup_efficiency(sample_size_results: Dict) -> Dict:
    """Analyze GPU speedup efficiency.

    Args:
        sample_size_results: Results from sample size scaling

    Returns:
        Efficiency analysis
    """
    logger.info("\nAnalyzing GPU Speedup Efficiency")

    sample_sizes = []
    speedups = []

    for n_samples, result in sample_size_results.items():
        if not np.isnan(result["speedup"]):
            sample_sizes.append(n_samples)
            speedups.append(result["speedup"])

    if len(speedups) == 0:
        logger.warning("No GPU results available for analysis")
        return {"available": False}

    sample_sizes = np.array(sample_sizes)
    speedups = np.array(speedups)

    # Find optimal sample size (maximum speedup)
    max_speedup_idx = np.argmax(speedups)
    optimal_sample_size = sample_sizes[max_speedup_idx]
    max_speedup = speedups[max_speedup_idx]

    logger.info(f"Maximum speedup: {max_speedup:.2f}x at {optimal_sample_size} samples")

    # Speedup scaling (fit log-log)
    log_sizes = np.log10(sample_sizes)
    log_speedups = np.log10(speedups)

    if len(log_sizes) >= 2:
        coeffs = np.polyfit(log_sizes, log_speedups, deg=1)
        scaling_exponent = coeffs[0]
        logger.info(f"Speedup scaling: S ∝ N^{scaling_exponent:.3f}")
    else:
        scaling_exponent = np.nan

    return {
        "available": True,
        "sample_sizes": sample_sizes.tolist(),
        "speedups": speedups.tolist(),
        "max_speedup": max_speedup,
        "optimal_sample_size": int(optimal_sample_size),
        "scaling_exponent": scaling_exponent,
        "average_speedup": np.mean(speedups),
    }


def save_results(
    sample_size_results: Dict,
    network_size_results: Dict,
    efficiency_analysis: Dict,
    output_dir: Path,
):
    """Save benchmark results.

    Args:
        sample_size_results: Sample size scaling results
        network_size_results: Network size scaling results
        efficiency_analysis: Efficiency analysis
        output_dir: Output directory
    """
    logger.info(f"\nSaving results to {output_dir}")

    # JSON
    json_path = output_dir / "exp2_gpu_speedup_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "sample_size_scaling": {
                    str(k): v for k, v in sample_size_results.items()
                },
                "network_size_scaling": {
                    str(k): v for k, v in network_size_results.items()
                },
                "efficiency_analysis": efficiency_analysis,
                "system_info": {
                    "platform": platform.system(),
                    "processor": platform.processor(),
                    "gpu_available": GPU_AVAILABLE,
                },
            },
            f,
            indent=2,
        )

    logger.info(f"Saved JSON: {json_path}")

    # CSV for sample size scaling
    csv_path = output_dir / "exp2_sample_size_scaling.csv"
    with open(csv_path, "w") as f:
        f.write(
            "N_Samples,CPU_Runtime,CPU_Throughput,GPU_Runtime,GPU_Throughput,Speedup\n"
        )

        for n_samples in sorted(sample_size_results.keys()):
            result = sample_size_results[n_samples]
            cpu = result["cpu"]
            gpu = result["gpu"]

            f.write(
                f"{n_samples},"
                f"{cpu['runtime']:.4f},"
                f"{cpu['throughput']:.2f},"
                f"{gpu['runtime']:.4f},"
                f"{gpu['throughput']:.2f},"
                f"{result['speedup']:.2f}\n"
            )

    logger.info(f"Saved CSV: {csv_path}")

    # CSV for network size scaling
    csv_path2 = output_dir / "exp2_network_size_scaling.csv"
    with open(csv_path2, "w") as f:
        f.write("N_Nodes,N_Edges,CPU_Runtime,GPU_Runtime,Speedup\n")

        for n_nodes in sorted(network_size_results.keys()):
            result = network_size_results[n_nodes]
            cpu = result["cpu"]
            gpu = result["gpu"]

            f.write(
                f"{n_nodes},"
                f"{result['n_edges']},"
                f"{cpu['runtime']:.4f},"
                f"{gpu['runtime']:.4f},"
                f"{result['speedup']:.2f}\n"
            )

    logger.info(f"Saved CSV: {csv_path2}")


def print_summary(
    sample_size_results: Dict, network_size_results: Dict, efficiency_analysis: Dict
):
    """Print experiment summary.

    Args:
        sample_size_results: Sample size results
        network_size_results: Network size results
        efficiency_analysis: Efficiency analysis
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: GPU SPEEDUP EVALUATION - SUMMARY")
    print("=" * 80)

    print("\nSample Size Scaling:")
    print("-" * 100)
    print(
        f"{'N Samples':>12} {'CPU-1T (s)':>12} {'CPU-MT (s)':>12} "
        f"{'GPU (s)':>10} {'Spdup/1T':>10} {'Spdup/MT':>10}"
    )
    print("-" * 100)

    for n_samples in sorted(sample_size_results.keys()):
        result = sample_size_results[n_samples]
        cpu_single_time = result["cpu"]["runtime"]
        cpu_multi_time = (
            result["cpu_multicore"]["runtime"]
            if result.get("cpu_multicore", {}).get("available", False)
            else float("nan")
        )
        gpu_time = (
            result["gpu"]["runtime"]
            if result["gpu"].get("available", False)
            else float("nan")
        )
        speedup_1t = result.get("speedup", float("nan"))
        speedup_mt = result.get("speedup_vs_multicore", float("nan"))

        cpu_multi_str = (
            f"{cpu_multi_time:>12.2f}"
            if not np.isnan(cpu_multi_time)
            else f"{'N/A':>12}"
        )
        gpu_str = f"{gpu_time:>10.2f}" if not np.isnan(gpu_time) else f"{'N/A':>10}"
        sp1t_str = (
            f"{speedup_1t:>9.2f}x" if not np.isnan(speedup_1t) else f"{'N/A':>10}"
        )
        spmt_str = (
            f"{speedup_mt:>9.2f}x" if not np.isnan(speedup_mt) else f"{'N/A':>10}"
        )

        print(
            f"{n_samples:>12} {cpu_single_time:>12.2f} {cpu_multi_str} {gpu_str} {sp1t_str} {spmt_str}"
        )

    print("\nNetwork Size Scaling:")
    print("-" * 80)
    print(f"{'N Nodes':>12} {'CPU (s)':>12} {'GPU (s)':>12} {'Speedup':>10}")
    print("-" * 80)

    for n_nodes in sorted(network_size_results.keys()):
        result = network_size_results[n_nodes]
        cpu_time = result["cpu"]["runtime"]
        gpu_time = (
            result["gpu"]["runtime"]
            if result["gpu"].get("available", False)
            else np.nan
        )
        speedup = result["speedup"]

        if not np.isnan(gpu_time):
            print(
                f"{n_nodes:>12} {cpu_time:>12.2f} {gpu_time:>12.2f} {speedup:>10.2f}x"
            )
        else:
            print(f"{n_nodes:>12} {cpu_time:>12.2f} {'N/A':>12} {'N/A':>10}")

    if efficiency_analysis.get("available", False):
        print("\nPerformance Summary:")
        print("-" * 80)
        print(f"Maximum speedup: {efficiency_analysis['max_speedup']:.2f}x")
        print(f"Optimal sample size: {efficiency_analysis['optimal_sample_size']:,}")
        print(f"Average speedup: {efficiency_analysis['average_speedup']:.2f}x")

        if not np.isnan(efficiency_analysis["scaling_exponent"]):
            print(f"Scaling: S ∝ N^{efficiency_analysis['scaling_exponent']:.3f}")
    else:
        print("\nGPU not available - CPU-only results")

    print("=" * 80)


def main(
    config: ExperimentConfig = None, full_mode: bool = False, gpu_only: bool = False
):
    """Main experiment runner.

    Args:
        config: Experiment configuration. If None, uses defaults.
        full_mode: If True, run full experiment with larger sample/network sizes.
                   If False (default), run quick mode for faster testing.
    """
    if config is None:
        config = ExperimentConfig()

    # Setup logging and output directories
    setup_logging(config)
    results_dir, figures_dir, tables_dir = setup_output_dirs(config)

    print("=" * 80)
    print("EXPERIMENT 2: GPU SPEEDUP EVALUATION")
    print(f"Mode: {'FULL' if full_mode else 'QUICK'}")
    print("=" * 80)

    # ============================================================================
    # Setup
    # ============================================================================
    print("\n[SETUP]")
    print("-" * 80)
    print(f"Platform: {platform.system()}")
    print(f"Processor: {platform.processor()}")
    print(f"GPU Available: {GPU_AVAILABLE}")

    # Experiment parameters from config
    T = config.T / 2  # Shorter simulation for benchmarking
    dt = config.dt
    seed = config.seed

    # Create a test network
    generator = TopologyGenerator(seed=seed)
    test_network = generator.generate_erdos_renyi(n_nodes=100, p=0.2, directed=False)
    test_network.set_link_properties(
        bandwidth_range=(1e9, 10e9),
        delay_range=(0.001, 0.01),
        capacity_range=(500, 2000),
        seed=seed,
    )
    test_traffic = PoissonTraffic(rate=100.0, seed=seed)

    # ============================================================================
    # Test 1: Sample Size Scaling
    # ============================================================================
    print("\n[TEST 1: SAMPLE SIZE SCALING]")
    print("-" * 80)

    # Sample sizes: use full config with --full flag
    if full_mode:
        sample_sizes = [1000, 10000, 100000, 1000000]
        logger.info("Running FULL experiment configuration")
    else:
        sample_sizes = [1000, 10000, 50000]
        logger.info("Running QUICK experiment (use --full for complete evaluation)")

    sample_size_results = run_sample_size_scaling(
        test_network, test_traffic, sample_sizes, T, dt, seed, gpu_only=gpu_only
    )

    # ============================================================================
    # Test 2: Network Size Scaling
    # ============================================================================
    print("\n[TEST 2: NETWORK SIZE SCALING]")
    print("-" * 80)

    # Network sizes: use full config with --full flag
    if full_mode:
        network_sizes = [100, 500, 1000, 5000]
    else:
        network_sizes = [50, 100, 200]

    network_size_results = run_network_size_scaling(
        network_sizes,
        n_samples=config.n_samples,
        T=T,
        dt=dt,
        seed=seed,
        gpu_only=gpu_only,
    )

    # ============================================================================
    # Analysis
    # ============================================================================
    efficiency_analysis = analyze_speedup_efficiency(sample_size_results)

    # ============================================================================
    # Save Results
    # ============================================================================
    save_results(
        sample_size_results, network_size_results, efficiency_analysis, tables_dir
    )

    # ============================================================================
    # Summary
    # ============================================================================
    print_summary(sample_size_results, network_size_results, efficiency_analysis)

    print("\nResults saved to:")
    print(f"  {tables_dir / 'exp2_gpu_speedup_results.json'}")
    print(f"  {tables_dir / 'exp2_sample_size_scaling.csv'}")
    print(f"  {tables_dir / 'exp2_network_size_scaling.csv'}")

    if not GPU_AVAILABLE:
        print(
            "\nNote: GPU modules not available. Install PyCUDA to enable GPU benchmarks."
        )
        print("  pip install pycuda")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU Speedup Evaluation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python exp2_gpu_speedup.py          # Quick mode (faster testing)
  python exp2_gpu_speedup.py --full   # Full experiment (publication-quality)
  python exp2_gpu_speedup.py --seed 123 --T 20.0  # Custom parameters
        """,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full experiment with larger sample sizes (10^6) and network sizes (5000 nodes)",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Skip CPU benchmarks; compare GPU-MC vs GPU-MLMC instead",
    )
    parser.add_argument(
        "--T", type=float, default=10.0, help="Simulation time (default: 10.0)"
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Timestep (default: 0.1)")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Base sample count (default: 1000)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    config = ExperimentConfig(
        T=args.T,
        dt=args.dt,
        seed=args.seed,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    main(config=config, full_mode=args.full, gpu_only=args.gpu_only)
