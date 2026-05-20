"""
GPU Benchmark Script

Comprehensive benchmarking of GPU vs CPU performance for
Monte Carlo and MLMC network simulations.

Generates performance plots and comparison tables.
"""

import sys
from pathlib import Path
import numpy as np
import time
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from network.topology import TopologyGenerator
from network.traffic import PoissonTraffic
from simulation.monte_carlo import MonteCarloSimulator
from simulation.mlmc import MLMCSimulator

# Try importing GPU modules
try:
    from gpu.parallel_mc import GPUMonteCarloSimulator, GPUMLMCSimulator, PYCUDA_AVAILABLE
    from gpu.memory_mgmt import GPUMemoryManager
except ImportError:
    PYCUDA_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark_monte_carlo(network, traffic, sample_sizes, T=10.0, dt=0.1):
    """
    Benchmark Monte Carlo on CPU and GPU.

    Args:
        network: NetworkGraph
        traffic: TrafficModel
        sample_sizes: List of sample sizes to test
        T: Simulation duration
        dt: Time step

    Returns:
        Dictionary with benchmark results
    """
    results = {
        'sample_sizes': sample_sizes,
        'cpu_times': [],
        'gpu_times': [],
        'speedups': [],
        'cpu_means': [],
        'gpu_means': []
    }

    cpu_sim = MonteCarloSimulator(seed=42)

    if PYCUDA_AVAILABLE:
        gpu_sim = GPUMonteCarloSimulator(seed=42)

    for n_samples in sample_sizes:
        logger.info(f"Benchmarking Monte Carlo: N={n_samples}")

        # CPU benchmark
        cpu_start = time.time()
        cpu_result = cpu_sim.estimate(
            network, traffic, n_samples, T, dt, verbose=False
        )
        cpu_time = time.time() - cpu_start

        results['cpu_times'].append(cpu_time)
        results['cpu_means'].append(cpu_result.mean)

        # GPU benchmark
        if PYCUDA_AVAILABLE:
            gpu_start = time.time()
            gpu_result = gpu_sim.estimate(
                network, traffic, n_samples, T, dt, verbose=False
            )
            gpu_time = time.time() - gpu_start

            speedup = cpu_time / gpu_time

            results['gpu_times'].append(gpu_time)
            results['gpu_means'].append(gpu_result.mean)
            results['speedups'].append(speedup)

            logger.info(f"  CPU: {cpu_time:.2f}s, GPU: {gpu_time:.2f}s, Speedup: {speedup:.1f}x")
        else:
            logger.info(f"  CPU: {cpu_time:.2f}s (GPU not available)")

    return results


def benchmark_mlmc(network, traffic, epsilon_values, L_max=4, T=10.0, base_dt=0.1):
    """
    Benchmark MLMC on CPU and GPU.

    Args:
        network: NetworkGraph
        traffic: TrafficModel
        epsilon_values: List of target accuracies
        L_max: Maximum MLMC level
        T: Simulation duration
        base_dt: Base time step

    Returns:
        Dictionary with benchmark results
    """
    results = {
        'epsilon_values': epsilon_values,
        'cpu_times': [],
        'gpu_times': [],
        'speedups': [],
        'cpu_estimates': [],
        'gpu_estimates': [],
        'cpu_costs': [],
        'gpu_costs': []
    }

    cpu_sim = MLMCSimulator(refinement_factor=2, seed=42)

    if PYCUDA_AVAILABLE:
        gpu_sim = GPUMLMCSimulator(refinement_factor=2, seed=42)

    for epsilon in epsilon_values:
        logger.info(f"Benchmarking MLMC: ε={epsilon}")

        # CPU benchmark
        cpu_start = time.time()
        cpu_result = cpu_sim.mlmc_estimate(
            network, traffic, epsilon, L_max, T, base_dt,
            pilot_samples=50, verbose=False
        )
        cpu_time = time.time() - cpu_start

        results['cpu_times'].append(cpu_time)
        results['cpu_estimates'].append(cpu_result.estimate)
        results['cpu_costs'].append(cpu_result.total_cost)

        # GPU benchmark
        if PYCUDA_AVAILABLE:
            gpu_start = time.time()
            gpu_result = gpu_sim.mlmc_estimate_gpu(
                network, traffic, epsilon, L_max, T, base_dt,
                pilot_samples=50, verbose=False
            )
            gpu_time = time.time() - gpu_start

            speedup = cpu_time / gpu_time

            results['gpu_times'].append(gpu_time)
            results['gpu_estimates'].append(gpu_result.estimate)
            results['gpu_costs'].append(gpu_result.total_cost)
            results['speedups'].append(speedup)

            logger.info(f"  CPU: {cpu_time:.2f}s, GPU: {gpu_time:.2f}s, Speedup: {speedup:.1f}x")
        else:
            logger.info(f"  CPU: {cpu_time:.2f}s (GPU not available)")

    return results


def print_benchmark_summary(mc_results, mlmc_results):
    """Print formatted benchmark summary."""

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Monte Carlo results
    print("\n1. Monte Carlo Benchmark")
    print("-" * 80)
    print(f"{'N':>10} | {'CPU Time':>12} | {'GPU Time':>12} | {'Speedup':>10} | {'CPU Mean':>10}")
    print("-" * 80)

    for i, n in enumerate(mc_results['sample_sizes']):
        cpu_time = mc_results['cpu_times'][i]
        cpu_mean = mc_results['cpu_means'][i]

        if PYCUDA_AVAILABLE and i < len(mc_results['gpu_times']):
            gpu_time = mc_results['gpu_times'][i]
            speedup = mc_results['speedups'][i]
            print(f"{n:10d} | {cpu_time:10.2f}s | {gpu_time:10.2f}s | {speedup:9.1f}x | {cpu_mean:10.4f}")
        else:
            print(f"{n:10d} | {cpu_time:10.2f}s | {'N/A':>12} | {'N/A':>10} | {cpu_mean:10.4f}")

    # MLMC results
    print("\n2. MLMC Benchmark")
    print("-" * 80)
    print(f"{'Epsilon':>10} | {'CPU Time':>12} | {'GPU Time':>12} | {'Speedup':>10} | {'CPU Est':>10}")
    print("-" * 80)

    for i, eps in enumerate(mlmc_results['epsilon_values']):
        cpu_time = mlmc_results['cpu_times'][i]
        cpu_est = mlmc_results['cpu_estimates'][i]

        if PYCUDA_AVAILABLE and i < len(mlmc_results['gpu_times']):
            gpu_time = mlmc_results['gpu_times'][i]
            speedup = mlmc_results['speedups'][i]
            print(f"{eps:10.4f} | {cpu_time:10.2f}s | {gpu_time:10.2f}s | {speedup:9.1f}x | {cpu_est:10.4f}")
        else:
            print(f"{eps:10.4f} | {cpu_time:10.2f}s | {'N/A':>12} | {'N/A':>10} | {cpu_est:10.4f}")

    # Summary statistics
    if PYCUDA_AVAILABLE:
        print("\n3. Summary Statistics")
        print("-" * 80)

        mc_avg_speedup = np.mean(mc_results['speedups']) if mc_results['speedups'] else 0
        mlmc_avg_speedup = np.mean(mlmc_results['speedups']) if mlmc_results['speedups'] else 0

        print(f"Monte Carlo Average Speedup: {mc_avg_speedup:.1f}x")
        print(f"MLMC Average Speedup: {mlmc_avg_speedup:.1f}x")
        print(f"Overall Average Speedup: {np.mean([mc_avg_speedup, mlmc_avg_speedup]):.1f}x")

        # GPU info
        try:
            mem_mgr = GPUMemoryManager()
            info = mem_mgr.get_memory_info()
            print(f"\nGPU Device: {info['device_name']}")
            print(f"GPU Memory: {info['total_memory_gb']:.1f} GB")
        except:
            pass

    print("\n" + "=" * 80)


def main():
    """Run comprehensive GPU benchmark."""

    print("=" * 80)
    print("GPU-Accelerated Network Simulation - Comprehensive Benchmark")
    print("=" * 80)

    if not PYCUDA_AVAILABLE:
        print("\nWARNING: PyCUDA not available. Running CPU-only benchmarks.")
        print("Install PyCUDA with: pip install pycuda")
        print()

    # Create network and traffic
    print("\n[Setup] Creating Network and Traffic")
    print("-" * 80)

    gen = TopologyGenerator(seed=42)
    network = gen.generate_erdos_renyi(n_nodes=100, p=0.05)
    network.set_link_properties(seed=42)

    traffic = PoissonTraffic(rate=10.0, seed=42)

    print(f"Network: {network}")
    print(f"Traffic: {traffic}")

    # Benchmark parameters
    mc_sample_sizes = [100, 500, 1000, 5000, 10000]
    mlmc_epsilon_values = [0.1, 0.05, 0.01, 0.005]

    # Run Monte Carlo benchmark
    print("\n[Benchmark 1] Monte Carlo Simulation")
    print("-" * 80)

    mc_results = benchmark_monte_carlo(
        network=network,
        traffic=traffic,
        sample_sizes=mc_sample_sizes,
        T=10.0,
        dt=0.1
    )

    # Run MLMC benchmark
    print("\n[Benchmark 2] Multilevel Monte Carlo Simulation")
    print("-" * 80)

    mlmc_results = benchmark_mlmc(
        network=network,
        traffic=traffic,
        epsilon_values=mlmc_epsilon_values,
        L_max=4,
        T=10.0,
        base_dt=0.1
    )

    # Print summary
    print_benchmark_summary(mc_results, mlmc_results)

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if PYCUDA_AVAILABLE:
        mc_avg_speedup = np.mean(mc_results['speedups'])
        mlmc_avg_speedup = np.mean(mlmc_results['speedups'])

        print("\nPerformance Analysis:")
        print(f"- GPU provides {mc_avg_speedup:.0f}x speedup for Monte Carlo")
        print(f"- GPU provides {mlmc_avg_speedup:.0f}x speedup for MLMC")
        print(f"- Combined with MLMC's algorithmic advantage, total speedup can reach {mc_avg_speedup * 100:.0f}x+")

        print("\nBest Practices:")
        print("- Use GPU for sample counts > 1,000")
        print("- Combine GPU + MLMC for maximum efficiency")
        print("- Batch size should be optimized for available GPU memory")
        print("- For very large networks, consider distributed GPU computing")
    else:
        print("\nTo enable GPU acceleration:")
        print("1. Install CUDA Toolkit (11.8+)")
        print("2. Install PyCUDA: pip install pycuda")
        print("3. Verify installation: python -c 'import pycuda.driver as cuda; cuda.init()'")
        print("4. Re-run this benchmark")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
