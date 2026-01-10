"""
Experiment 3: Uncertainty Quantification Demonstration

Demonstrates uncertainty-aware network performance analysis with confidence intervals.

Objectives:
- Quantify uncertainty in delay, queue length, and utilization metrics
- Generate uncertainty bands for time-series predictions
- Compare deterministic vs stochastic predictions
- Visualize prediction confidence intervals
- Demonstrate bootstrap confidence intervals

Network: Email-Eu-core (SNAP) or synthetic
Traffic: MAWI-based bursty traffic
Metrics: End-to-end delay, queue length, link utilization
Confidence level: 95%

Expected Results:
- Well-calibrated confidence intervals
- Uncertainty bands showing prediction variability
- Significant difference from deterministic predictions
- Actionable insights for network planning
"""

import sys
from pathlib import Path
import numpy as np
import time
import logging
from typing import Dict, List, Tuple
import json

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "datasets"))

from network.topology import TopologyGenerator
from network.traffic import BurstyTraffic
from network.sde import QueueDynamicsSDE
from simulation.monte_carlo import MonteCarloSimulator
from metrics.delay import DelayCalculator
from metrics.congestion import CongestionAnalyzer
from metrics.uncertainty import UncertaintyQuantifier
from datasets.synthetic.generator import SyntheticBenchmarkGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(exist_ok=True)


def run_stochastic_simulation(
    network,
    traffic,
    n_samples: int,
    T: float,
    dt: float,
    seed: int = 42
) -> Dict:
    """Run stochastic Monte Carlo simulation.

    Args:
        network: Network topology
        traffic: Traffic model
        n_samples: Number of MC samples
        T: Simulation time
        dt: Timestep
        seed: Random seed

    Returns:
        Dictionary with simulation results and samples
    """
    logger.info(f"Running stochastic simulation: {n_samples} samples")

    simulator = MonteCarloSimulator(seed=seed)

    # Run simulation
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

    logger.info(f"Mean queue length: {result.mean:.4f}")
    logger.info(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    logger.info(f"Runtime: {runtime:.2f}s")

    # Generate sample trajectories for uncertainty band
    logger.info("Generating sample trajectories...")

    n_timesteps = int(T / dt)
    sample_trajectories = np.zeros((n_samples, n_timesteps))

    # Simple queue dynamics for each sample
    sde = QueueDynamicsSDE(
        arrival_rate=traffic.rate if hasattr(traffic, 'rate') else 100.0,
        service_rate=120.0,
        noise_intensity=0.5,
        seed=seed
    )

    for i in range(n_samples):
        trajectory = sde.simulate_path(T=T, dt=dt, q0=0.0, seed=seed+i)
        sample_trajectories[i] = trajectory

    return {
        'result': result,
        'runtime': runtime,
        'sample_trajectories': sample_trajectories,
        'times': np.linspace(0, T, n_timesteps)
    }


def compute_deterministic_prediction(
    arrival_rate: float,
    service_rate: float,
    T: float,
    dt: float
) -> Dict:
    """Compute deterministic prediction using M/M/1 formula.

    Args:
        arrival_rate: Arrival rate λ
        service_rate: Service rate μ
        T: Simulation time
        dt: Timestep

    Returns:
        Deterministic prediction
    """
    logger.info("Computing deterministic prediction (M/M/1)")

    # M/M/1 steady-state
    rho = arrival_rate / service_rate
    if rho >= 1.0:
        logger.warning(f"System unstable: ρ = {rho:.2f} >= 1")
        expected_queue = np.inf
    else:
        expected_queue = rho / (1 - rho)

    logger.info(f"Utilization: ρ = {rho:.3f}")
    logger.info(f"Expected queue length: {expected_queue:.4f}")

    # Constant trajectory (steady-state)
    n_timesteps = int(T / dt)
    times = np.linspace(0, T, n_timesteps)
    trajectory = np.full(n_timesteps, expected_queue)

    return {
        'utilization': rho,
        'expected_queue': expected_queue,
        'trajectory': trajectory,
        'times': times
    }


def analyze_delay_uncertainty(
    network,
    stochastic_results: Dict,
    confidence_level: float = 0.95
) -> Dict:
    """Analyze delay uncertainty with confidence intervals.

    Args:
        network: Network topology
        stochastic_results: Stochastic simulation results
        confidence_level: Confidence level

    Returns:
        Delay uncertainty analysis
    """
    logger.info("\nAnalyzing Delay Uncertainty")

    delay_calc = DelayCalculator(network, confidence_level=confidence_level)

    # Generate delay samples (synthetic for demonstration)
    np.random.seed(42)
    mean_queue = stochastic_results['result'].mean
    std_queue = np.sqrt(stochastic_results['result'].variance)

    # Delay approximation: D ≈ Q/μ (Little's Law)
    service_rate = 120.0
    mean_delay = mean_queue / service_rate
    std_delay = std_queue / service_rate

    delay_samples = np.random.normal(mean_delay, std_delay, size=1000)
    delay_samples = np.maximum(delay_samples, 0)  # Non-negative

    # Compute delay metrics
    delay_metrics = delay_calc.estimate_delay_distribution(delay_samples)

    logger.info(f"Mean delay: {delay_metrics.mean_delay*1000:.2f} ms")
    logger.info(f"Std delay: {delay_metrics.std_delay*1000:.2f} ms")
    logger.info(f"95% CI: [{delay_metrics.ci_lower*1000:.2f}, {delay_metrics.ci_upper*1000:.2f}] ms")
    logger.info(f"P95 delay: {delay_metrics.percentiles['p95']*1000:.2f} ms")
    logger.info(f"P99 delay: {delay_metrics.percentiles['p99']*1000:.2f} ms")

    return {
        'metrics': delay_metrics.summary(),
        'samples': delay_samples
    }


def analyze_congestion_uncertainty(
    network,
    stochastic_results: Dict,
    threshold: float = 0.8
) -> Dict:
    """Analyze congestion with uncertainty.

    Args:
        network: Network topology
        stochastic_results: Stochastic simulation results
        threshold: Congestion threshold

    Returns:
        Congestion uncertainty analysis
    """
    logger.info("\nAnalyzing Congestion Uncertainty")

    congestion_analyzer = CongestionAnalyzer(network, congestion_threshold=threshold)

    # Use sample trajectories
    sample_trajectories = stochastic_results['sample_trajectories']
    times = stochastic_results['times']

    n_samples, n_timesteps = sample_trajectories.shape
    n_nodes = network.n_nodes

    # Expand to network-wide (replicate for demonstration)
    queue_states = np.zeros((n_samples, n_timesteps, n_nodes))
    for i in range(n_samples):
        for j in range(n_nodes):
            noise = np.random.normal(0, 0.2, n_timesteps)
            queue_states[i, :, j] = sample_trajectories[i] + noise

    # Analyze congestion
    mean_queue_states = np.mean(queue_states, axis=0)  # [n_timesteps, n_nodes]

    congestion_metrics = congestion_analyzer.analyze_simulation_congestion(
        mean_queue_states,
        arrival_rates=100.0,
        service_rates=120.0,
        times=times
    )

    logger.info(f"Mean queue length: {congestion_metrics.mean_queue_length:.2f}")
    logger.info(f"Max queue length: {congestion_metrics.max_queue_length:.2f}")
    logger.info(f"Mean utilization: {congestion_metrics.mean_utilization:.3f}")
    logger.info(f"Congestion events: {len(congestion_metrics.congestion_events)}")

    # Compute congestion probability
    from metrics.congestion import compute_congestion_probability

    congestion_threshold_queue = 10.0
    congestion_prob = compute_congestion_probability(
        queue_states.reshape(-1),
        congestion_threshold_queue
    )

    logger.info(f"P(Queue > {congestion_threshold_queue}): {congestion_prob:.3f}")

    return {
        'metrics': congestion_metrics.summary(),
        'congestion_probability': congestion_prob,
        'threshold': congestion_threshold_queue
    }


def compute_uncertainty_bands(
    stochastic_results: Dict,
    confidence_level: float = 0.95
) -> Dict:
    """Compute uncertainty bands for queue evolution.

    Args:
        stochastic_results: Stochastic simulation results
        confidence_level: Confidence level

    Returns:
        Uncertainty band results
    """
    logger.info("\nComputing Uncertainty Bands")

    uq = UncertaintyQuantifier(
        confidence_level=confidence_level,
        n_bootstrap=1000,
        random_seed=42
    )

    sample_trajectories = stochastic_results['sample_trajectories']
    times = stochastic_results['times']

    # Compute uncertainty band
    uncertainty_band = uq.compute_uncertainty_band(
        sample_trajectories,
        confidence_level=confidence_level,
        method='quantile'
    )

    logger.info(f"Time points: {len(uncertainty_band.times)}")
    logger.info(f"Mean trajectory range: [{np.min(uncertainty_band.mean):.2f}, {np.max(uncertainty_band.mean):.2f}]")
    logger.info(f"Average CI width: {np.mean(uncertainty_band.width()):.2f}")
    logger.info(f"Max CI width: {np.max(uncertainty_band.width()):.2f}")

    # Relative uncertainty
    rel_width = uncertainty_band.relative_width()
    avg_rel_uncertainty = np.nanmean(rel_width) * 100

    logger.info(f"Average relative uncertainty: {avg_rel_uncertainty:.1f}%")

    return {
        'times': times.tolist(),
        'mean': uncertainty_band.mean.tolist(),
        'lower': uncertainty_band.lower.tolist(),
        'upper': uncertainty_band.upper.tolist(),
        'average_width': float(np.mean(uncertainty_band.width())),
        'average_relative_uncertainty': float(avg_rel_uncertainty)
    }


def compare_with_deterministic(
    stochastic_results: Dict,
    deterministic_results: Dict,
    uncertainty_band: Dict
) -> Dict:
    """Compare stochastic predictions with deterministic baseline.

    Args:
        stochastic_results: Stochastic simulation results
        deterministic_results: Deterministic prediction
        uncertainty_band: Uncertainty band

    Returns:
        Comparison results
    """
    logger.info("\nComparing Stochastic vs Deterministic")

    stochastic_mean = stochastic_results['result'].mean
    deterministic_value = deterministic_results['expected_queue']

    difference = abs(stochastic_mean - deterministic_value)
    relative_diff = 100 * difference / deterministic_value if deterministic_value > 0 else np.inf

    logger.info(f"Stochastic mean: {stochastic_mean:.4f}")
    logger.info(f"Deterministic: {deterministic_value:.4f}")
    logger.info(f"Absolute difference: {difference:.4f}")
    logger.info(f"Relative difference: {relative_diff:.1f}%")

    # Check if deterministic value is within CI
    ci_lower = stochastic_results['result'].ci_lower
    ci_upper = stochastic_results['result'].ci_upper
    deterministic_in_ci = (deterministic_value >= ci_lower) and (deterministic_value <= ci_upper)

    logger.info(f"Deterministic within 95% CI: {deterministic_in_ci}")

    return {
        'stochastic_mean': stochastic_mean,
        'deterministic_value': deterministic_value,
        'absolute_difference': difference,
        'relative_difference_percent': relative_diff,
        'deterministic_in_ci': deterministic_in_ci
    }


def save_results(
    stochastic_results: Dict,
    deterministic_results: Dict,
    delay_analysis: Dict,
    congestion_analysis: Dict,
    uncertainty_band: Dict,
    comparison: Dict,
    output_dir: Path
):
    """Save all results.

    Args:
        stochastic_results: Stochastic results
        deterministic_results: Deterministic results
        delay_analysis: Delay analysis
        congestion_analysis: Congestion analysis
        uncertainty_band: Uncertainty band
        comparison: Comparison results
        output_dir: Output directory
    """
    logger.info(f"\nSaving results to {output_dir}")

    # JSON (exclude large arrays)
    json_path = output_dir / "exp3_uncertainty_quantification_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'stochastic': {
                'mean': stochastic_results['result'].mean,
                'variance': stochastic_results['result'].variance,
                'ci_lower': stochastic_results['result'].ci_lower,
                'ci_upper': stochastic_results['result'].ci_upper,
                'runtime': stochastic_results['runtime']
            },
            'deterministic': {
                'utilization': deterministic_results['utilization'],
                'expected_queue': deterministic_results['expected_queue']
            },
            'delay_analysis': delay_analysis['metrics'],
            'congestion_analysis': congestion_analysis['metrics'],
            'uncertainty_band': uncertainty_band,
            'comparison': comparison
        }, f, indent=2)

    logger.info(f"Saved JSON: {json_path}")

    # Save uncertainty band data (NPZ for plots)
    npz_path = output_dir / "exp3_uncertainty_band.npz"
    np.savez(
        npz_path,
        times=np.array(uncertainty_band['times']),
        mean=np.array(uncertainty_band['mean']),
        lower=np.array(uncertainty_band['lower']),
        upper=np.array(uncertainty_band['upper'])
    )

    logger.info(f"Saved NPZ: {npz_path}")


def print_summary(
    stochastic_results: Dict,
    delay_analysis: Dict,
    congestion_analysis: Dict,
    uncertainty_band: Dict,
    comparison: Dict
):
    """Print experiment summary.

    Args:
        stochastic_results: Stochastic results
        delay_analysis: Delay analysis
        congestion_analysis: Congestion analysis
        uncertainty_band: Uncertainty band
        comparison: Comparison
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: UNCERTAINTY QUANTIFICATION - SUMMARY")
    print("=" * 80)

    print("\nQueue Length Prediction:")
    print("-" * 80)
    result = stochastic_results['result']
    print(f"Mean: {result.mean:.4f}")
    print(f"Std: {np.sqrt(result.variance):.4f}")
    print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print(f"CI Width: {result.ci_upper - result.ci_lower:.4f}")

    print("\nDelay Prediction:")
    print("-" * 80)
    delay_metrics = delay_analysis['metrics']
    print(f"Mean delay: {delay_metrics['mean']*1000:.2f} ms")
    print(f"Std delay: {delay_metrics['std']*1000:.2f} ms")
    print(f"P95 delay: {delay_metrics['percentiles']['p95']*1000:.2f} ms")
    print(f"P99 delay: {delay_metrics['percentiles']['p99']*1000:.2f} ms")

    print("\nCongestion Analysis:")
    print("-" * 80)
    cong_metrics = congestion_analysis['metrics']
    print(f"Mean queue: {cong_metrics['mean_queue_length']:.2f}")
    print(f"Max queue: {cong_metrics['max_queue_length']:.2f}")
    print(f"Mean utilization: {cong_metrics['mean_utilization']:.3f}")
    print(f"Congestion probability: {congestion_analysis['congestion_probability']:.3f}")

    print("\nUncertainty Quantification:")
    print("-" * 80)
    print(f"Average CI width: {uncertainty_band['average_width']:.2f}")
    print(f"Relative uncertainty: {uncertainty_band['average_relative_uncertainty']:.1f}%")

    print("\nStochastic vs Deterministic:")
    print("-" * 80)
    print(f"Stochastic: {comparison['stochastic_mean']:.4f}")
    print(f"Deterministic: {comparison['deterministic_value']:.4f}")
    print(f"Difference: {comparison['relative_difference_percent']:.1f}%")
    print(f"Deterministic in CI: {comparison['deterministic_in_ci']}")

    print("\nKey Insights:")
    print("-" * 80)
    if uncertainty_band['average_relative_uncertainty'] > 10:
        print("- Significant prediction uncertainty requires stochastic modeling")
    if not comparison['deterministic_in_ci']:
        print("- Deterministic prediction outside confidence interval")
    print("- Uncertainty bands provide actionable confidence ranges")
    print("- Network planning should account for prediction variability")

    print("=" * 80)


def main():
    """Main experiment runner."""

    print("=" * 80)
    print("EXPERIMENT 3: UNCERTAINTY QUANTIFICATION")
    print("=" * 80)

    # ============================================================================
    # Setup
    # ============================================================================
    print("\n[SETUP]")
    print("-" * 80)

    # Create network
    generator = TopologyGenerator(seed=42)
    network = generator.generate_erdos_renyi(n_nodes=50, p=0.2, directed=False)
    network.set_link_properties(
        bandwidth_range=(1e9, 10e9),
        delay_range=(0.001, 0.01),
        capacity_range=(500, 2000),
        seed=42
    )

    # Bursty traffic
    traffic = BurstyTraffic(
        on_rate=150.0,
        off_rate=50.0,
        burst_duration=1.0,
        idle_duration=0.5,
        seed=42
    )

    print(f"Network: {network.n_nodes} nodes, {network.n_edges} edges")
    print(f"Traffic: Bursty (on={traffic.on_rate}, off={traffic.off_rate})")

    # Parameters
    n_samples = 1000
    T = 20.0
    dt = 0.1
    confidence_level = 0.95
    seed = 42

    print(f"\nParameters:")
    print(f"  Samples: {n_samples}")
    print(f"  Simulation time: {T}s")
    print(f"  Timestep: {dt}s")
    print(f"  Confidence level: {confidence_level*100:.0f}%")

    # ============================================================================
    # Simulations
    # ============================================================================

    # Stochastic
    print("\n[STOCHASTIC SIMULATION]")
    print("-" * 80)
    stochastic_results = run_stochastic_simulation(
        network, traffic, n_samples, T, dt, seed
    )

    # Deterministic
    print("\n[DETERMINISTIC PREDICTION]")
    print("-" * 80)
    deterministic_results = compute_deterministic_prediction(
        arrival_rate=traffic.on_rate,
        service_rate=120.0,
        T=T,
        dt=dt
    )

    # ============================================================================
    # Analysis
    # ============================================================================

    # Delay uncertainty
    delay_analysis = analyze_delay_uncertainty(
        network, stochastic_results, confidence_level
    )

    # Congestion uncertainty
    congestion_analysis = analyze_congestion_uncertainty(
        network, stochastic_results, threshold=0.8
    )

    # Uncertainty bands
    uncertainty_band = compute_uncertainty_bands(
        stochastic_results, confidence_level
    )

    # Comparison
    comparison = compare_with_deterministic(
        stochastic_results, deterministic_results, uncertainty_band
    )

    # ============================================================================
    # Save Results
    # ============================================================================
    save_results(
        stochastic_results,
        deterministic_results,
        delay_analysis,
        congestion_analysis,
        uncertainty_band,
        comparison,
        TABLES_DIR
    )

    # ============================================================================
    # Summary
    # ============================================================================
    print_summary(
        stochastic_results,
        delay_analysis,
        congestion_analysis,
        uncertainty_band,
        comparison
    )

    print("\nResults saved to:")
    print(f"  {TABLES_DIR / 'exp3_uncertainty_quantification_results.json'}")
    print(f"  {TABLES_DIR / 'exp3_uncertainty_band.npz'}")

    print("\nNext steps:")
    print("  - Generate uncertainty band plots")
    print("  - Create delay distribution histograms")
    print("  - Visualize congestion heatmaps")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
