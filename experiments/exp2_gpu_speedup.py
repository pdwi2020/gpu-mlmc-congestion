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

import sys
from pathlib import Path
import numpy as np
import time
import logging
from typing import Dict, List, Tuple
import json
import platform

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from network.topology import TopologyGenerator
from network.traffic import PoissonTraffic
from simulation.monte_carlo import MonteCarloSimulator

# Try to import GPU modules
try:
    from gpu.parallel_mc import GPUMonteCarloSimulator
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("GPU modules not available - using CPU-only mode")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(exist_ok=True)


def benchmark_cpu_single_thread(
    network,
    traffic,
    n_samples: int,
    T: float,
    dt: float,
    seed: int = 42
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
        metric='mean_queue'
    )
    runtime = time.time() - start_time

    logger.info(f"Runtime: {runtime:.2f}s")
    logger.info(f"Throughput: {n_samples/runtime:.2f} samples/sec")

    return {
        'method': 'CPU_single',
        'n_samples': n_samples,
        'runtime': runtime,
        'throughput': n_samples / runtime,
        'mean': result.mean,
        'variance': result.variance
    }


def benchmark_gpu(
    network,
    traffic,
    n_samples: int,
    T: float,
    dt: float,
    seed: int = 42
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
            'method': 'GPU',
            'n_samples': n_samples,
            'runtime': np.nan,
            'throughput': np.nan,
            'mean': np.nan,
            'variance': np.nan,
            'available': False
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
            metric='mean_queue'
        )
        runtime = time.time() - start_time

        logger.info(f"Runtime: {runtime:.2f}s")
        logger.info(f"Throughput: {n_samples/runtime:.2f} samples/sec")

        return {
            'method': 'GPU',
            'n_samples': n_samples,
            'runtime': runtime,
            'throughput': n_samples / runtime,
            'mean': result.mean,
            'variance': result.variance,
            'available': True
        }

    except Exception as e:
        logger.error(f"GPU benchmark failed: {e}")
        return {
            'method': 'GPU',
            'n_samples': n_samples,
            'runtime': np.nan,
            'throughput': np.nan,
            'mean': np.nan,
            'variance': np.nan,
            'available': False,
            'error': str(e)
        }


def run_sample_size_scaling(
    network,
    traffic,
    sample_sizes: List[int],
    T: float,
    dt: float,
    seed: int = 42
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
    logger.info(f"\n{'='*80}")
    logger.info(f"Sample Size Scaling Test")
    logger.info(f"Network: {network.n_nodes} nodes")
    logger.info(f"Sample sizes: {sample_sizes}")
    logger.info(f"{'='*80}\n")

    results = {}

    for n_samples in sample_sizes:
        logger.info(f"\n--- Testing {n_samples} samples ---")

        # CPU baseline
        cpu_result = benchmark_cpu_single_thread(
            network, traffic, n_samples, T, dt, seed
        )

        # GPU
        gpu_result = benchmark_gpu(
            network, traffic, n_samples, T, dt, seed
        )

        # Compute speedup
        if gpu_result.get('available', False) and cpu_result['runtime'] > 0:
            speedup = cpu_result['runtime'] / gpu_result['runtime']
        else:
            speedup = np.nan

        logger.info(f"Speedup: {speedup:.2f}x" if not np.isnan(speedup) else "Speedup: N/A")

        results[n_samples] = {
            'n_samples': n_samples,
            'cpu': cpu_result,
            'gpu': gpu_result,
            'speedup': speedup
        }

    return results


def run_network_size_scaling(
    network_sizes: List[int],
    n_samples: int,
    T: float,
    dt: float,
    seed: int = 42
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
    logger.info(f"\n{'='*80}")
    logger.info(f"Network Size Scaling Test")
    logger.info(f"Sample size: {n_samples}")
    logger.info(f"Network sizes: {network_sizes}")
    logger.info(f"{'='*80}\n")

    generator = TopologyGenerator(seed=seed)
    results = {}

    for n_nodes in network_sizes:
        logger.info(f"\n--- Testing {n_nodes} nodes ---")

        # Generate network
        network = generator.generate_erdos_renyi(n_nodes=n_nodes, p=0.15, directed=False)
        network.set_link_properties(
            bandwidth_range=(1e9, 10e9),
            delay_range=(0.001, 0.01),
            capacity_range=(500, 2000),
            seed=seed
        )

        traffic = PoissonTraffic(rate=100.0, seed=seed)

        logger.info(f"Network: {network.n_nodes} nodes, {network.n_edges} edges")

        # CPU baseline
        cpu_result = benchmark_cpu_single_thread(
            network, traffic, n_samples, T, dt, seed
        )

        # GPU
        gpu_result = benchmark_gpu(
            network, traffic, n_samples, T, dt, seed
        )

        # Compute speedup
        if gpu_result.get('available', False) and cpu_result['runtime'] > 0:
            speedup = cpu_result['runtime'] / gpu_result['runtime']
        else:
            speedup = np.nan

        logger.info(f"Speedup: {speedup:.2f}x" if not np.isnan(speedup) else "Speedup: N/A")

        results[n_nodes] = {
            'n_nodes': n_nodes,
            'n_edges': network.n_edges,
            'cpu': cpu_result,
            'gpu': gpu_result,
            'speedup': speedup
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
        if not np.isnan(result['speedup']):
            sample_sizes.append(n_samples)
            speedups.append(result['speedup'])

    if len(speedups) == 0:
        logger.warning("No GPU results available for analysis")
        return {
            'available': False
        }

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
        'available': True,
        'sample_sizes': sample_sizes.tolist(),
        'speedups': speedups.tolist(),
        'max_speedup': max_speedup,
        'optimal_sample_size': int(optimal_sample_size),
        'scaling_exponent': scaling_exponent,
        'average_speedup': np.mean(speedups)
    }


def save_results(
    sample_size_results: Dict,
    network_size_results: Dict,
    efficiency_analysis: Dict,
    output_dir: Path
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
    with open(json_path, 'w') as f:
        json.dump({
            'sample_size_scaling': {str(k): v for k, v in sample_size_results.items()},
            'network_size_scaling': {str(k): v for k, v in network_size_results.items()},
            'efficiency_analysis': efficiency_analysis,
            'system_info': {
                'platform': platform.system(),
                'processor': platform.processor(),
                'gpu_available': GPU_AVAILABLE
            }
        }, f, indent=2)

    logger.info(f"Saved JSON: {json_path}")

    # CSV for sample size scaling
    csv_path = output_dir / "exp2_sample_size_scaling.csv"
    with open(csv_path, 'w') as f:
        f.write("N_Samples,CPU_Runtime,CPU_Throughput,GPU_Runtime,GPU_Throughput,Speedup\n")

        for n_samples in sorted(sample_size_results.keys()):
            result = sample_size_results[n_samples]
            cpu = result['cpu']
            gpu = result['gpu']

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
    with open(csv_path2, 'w') as f:
        f.write("N_Nodes,N_Edges,CPU_Runtime,GPU_Runtime,Speedup\n")

        for n_nodes in sorted(network_size_results.keys()):
            result = network_size_results[n_nodes]
            cpu = result['cpu']
            gpu = result['gpu']

            f.write(
                f"{n_nodes},"
                f"{result['n_edges']},"
                f"{cpu['runtime']:.4f},"
                f"{gpu['runtime']:.4f},"
                f"{result['speedup']:.2f}\n"
            )

    logger.info(f"Saved CSV: {csv_path2}")


def print_summary(
    sample_size_results: Dict,
    network_size_results: Dict,
    efficiency_analysis: Dict
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
    print("-" * 80)
    print(f"{'N Samples':>12} {'CPU (s)':>12} {'GPU (s)':>12} {'Speedup':>10}")
    print("-" * 80)

    for n_samples in sorted(sample_size_results.keys()):
        result = sample_size_results[n_samples]
        cpu_time = result['cpu']['runtime']
        gpu_time = result['gpu']['runtime'] if result['gpu'].get('available', False) else np.nan
        speedup = result['speedup']

        if not np.isnan(gpu_time):
            print(f"{n_samples:>12} {cpu_time:>12.2f} {gpu_time:>12.2f} {speedup:>10.2f}x")
        else:
            print(f"{n_samples:>12} {cpu_time:>12.2f} {'N/A':>12} {'N/A':>10}")

    print("\nNetwork Size Scaling:")
    print("-" * 80)
    print(f"{'N Nodes':>12} {'CPU (s)':>12} {'GPU (s)':>12} {'Speedup':>10}")
    print("-" * 80)

    for n_nodes in sorted(network_size_results.keys()):
        result = network_size_results[n_nodes]
        cpu_time = result['cpu']['runtime']
        gpu_time = result['gpu']['runtime'] if result['gpu'].get('available', False) else np.nan
        speedup = result['speedup']

        if not np.isnan(gpu_time):
            print(f"{n_nodes:>12} {cpu_time:>12.2f} {gpu_time:>12.2f} {speedup:>10.2f}x")
        else:
            print(f"{n_nodes:>12} {cpu_time:>12.2f} {'N/A':>12} {'N/A':>10}")

    if efficiency_analysis.get('available', False):
        print("\nPerformance Summary:")
        print("-" * 80)
        print(f"Maximum speedup: {efficiency_analysis['max_speedup']:.2f}x")
        print(f"Optimal sample size: {efficiency_analysis['optimal_sample_size']:,}")
        print(f"Average speedup: {efficiency_analysis['average_speedup']:.2f}x")

        if not np.isnan(efficiency_analysis['scaling_exponent']):
            print(f"Scaling: S ∝ N^{efficiency_analysis['scaling_exponent']:.3f}")
    else:
        print("\nGPU not available - CPU-only results")

    print("=" * 80)


def main():
    """Main experiment runner."""

    print("=" * 80)
    print("EXPERIMENT 2: GPU SPEEDUP EVALUATION")
    print("=" * 80)

    # ============================================================================
    # Setup
    # ============================================================================
    print("\n[SETUP]")
    print("-" * 80)
    print(f"Platform: {platform.system()}")
    print(f"Processor: {platform.processor()}")
    print(f"GPU Available: {GPU_AVAILABLE}")

    # Experiment parameters
    T = 5.0  # Shorter simulation for benchmarking
    dt = 0.1
    seed = 42

    # Create a test network
    generator = TopologyGenerator(seed=seed)
    test_network = generator.generate_erdos_renyi(n_nodes=100, p=0.2, directed=False)
    test_network.set_link_properties(
        bandwidth_range=(1e9, 10e9),
        delay_range=(0.001, 0.01),
        capacity_range=(500, 2000),
        seed=seed
    )
    test_traffic = PoissonTraffic(rate=100.0, seed=seed)

    # ============================================================================
    # Test 1: Sample Size Scaling
    # ============================================================================
    print("\n[TEST 1: SAMPLE SIZE SCALING]")
    print("-" * 80)

    sample_sizes = [1000, 10000, 50000]  # Reduced for faster testing
    # For full experiment: [1000, 10000, 100000, 1000000]

    sample_size_results = run_sample_size_scaling(
        test_network,
        test_traffic,
        sample_sizes,
        T,
        dt,
        seed
    )

    # ============================================================================
    # Test 2: Network Size Scaling
    # ============================================================================
    print("\n[TEST 2: NETWORK SIZE SCALING]")
    print("-" * 80)

    network_sizes = [50, 100, 200]  # Reduced for faster testing
    # For full experiment: [100, 500, 1000, 5000]

    network_size_results = run_network_size_scaling(
        network_sizes,
        n_samples=10000,
        T=T,
        dt=dt,
        seed=seed
    )

    # ============================================================================
    # Analysis
    # ============================================================================
    efficiency_analysis = analyze_speedup_efficiency(sample_size_results)

    # ============================================================================
    # Save Results
    # ============================================================================
    save_results(
        sample_size_results,
        network_size_results,
        efficiency_analysis,
        TABLES_DIR
    )

    # ============================================================================
    # Summary
    # ============================================================================
    print_summary(
        sample_size_results,
        network_size_results,
        efficiency_analysis
    )

    print("\nResults saved to:")
    print(f"  {TABLES_DIR / 'exp2_gpu_speedup_results.json'}")
    print(f"  {TABLES_DIR / 'exp2_sample_size_scaling.csv'}")
    print(f"  {TABLES_DIR / 'exp2_network_size_scaling.csv'}")

    if not GPU_AVAILABLE:
        print("\nNote: GPU modules not available. Install PyCUDA to enable GPU benchmarks.")
        print("  pip install pycuda")

    print("\nNext steps:")
    print("  - Generate speedup plots (log-log scale)")
    print("  - Profile GPU kernel performance")
    print("  - Test with larger sample sizes (10^6+)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
