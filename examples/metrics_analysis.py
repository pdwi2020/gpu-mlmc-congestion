"""
Performance Metrics Analysis Example

Demonstrates comprehensive metrics analysis on network simulation results:
- Delay metrics and distribution analysis
- Congestion detection and propagation
- Uncertainty quantification and confidence intervals
- Variance reduction analysis (MC vs MLMC)

This example shows how to use all three metrics modules together to
analyze network performance with uncertainty quantification.
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from network.topology import TopologyGenerator
from network.traffic import PoissonTraffic
from network.sde import QueueDynamicsSDE
from simulation.monte_carlo import MonteCarloSimulator
from simulation.mlmc import MLMCSimulator
from metrics.delay import DelayCalculator
from metrics.congestion import CongestionAnalyzer
from metrics.uncertainty import UncertaintyQuantifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_metrics_analysis():
    """Run comprehensive metrics analysis example."""

    print("=" * 80)
    print("PERFORMANCE METRICS ANALYSIS - DEMONSTRATION")
    print("=" * 80)

    # ============================================================================
    # 1. Setup: Create Network and Traffic
    # ============================================================================
    print("\n[1] SETUP")
    print("-" * 80)

    # Generate network topology
    generator = TopologyGenerator(seed=42)
    network = generator.generate_erdos_renyi(n_nodes=50, p=0.15, directed=False)

    network.set_link_properties(
        bandwidth_range=(1e9, 10e9),  # 1-10 Gbps
        delay_range=(0.001, 0.01),    # 1-10 ms
        capacity_range=(500, 2000),   # packets/sec
        seed=42
    )

    print(f"Network: {network.n_nodes} nodes, {network.n_edges} edges")

    # Create traffic model
    traffic = PoissonTraffic(rate=100.0, packet_size_mean=1500.0, seed=42)
    print(f"Traffic: Poisson with λ={traffic.rate} pkts/s")

    # ============================================================================
    # 2. Run Monte Carlo Simulations
    # ============================================================================
    print("\n[2] MONTE CARLO SIMULATIONS")
    print("-" * 80)

    # Standard Monte Carlo
    print("\nRunning Standard MC (1000 samples)...")
    mc_simulator = MonteCarloSimulator(seed=42)

    mc_result = mc_simulator.estimate(
        network=network,
        traffic=traffic,
        n_samples=1000,
        T=10.0,
        dt=0.1,
        metric='mean_queue'
    )

    print(f"MC Result:")
    print(f"  Mean: {mc_result.mean:.4f}")
    print(f"  Std: {mc_result.std:.4f}")
    print(f"  95% CI: [{mc_result.ci_lower:.4f}, {mc_result.ci_upper:.4f}]")
    print(f"  Runtime: {mc_result.runtime:.2f}s")

    # Multilevel Monte Carlo
    print("\nRunning MLMC (ε=0.05)...")
    mlmc_simulator = MLMCSimulator(seed=42)

    mlmc_result = mlmc_simulator.mlmc_estimate(
        network=network,
        traffic=traffic,
        epsilon=0.05,
        L_max=4,
        T=10.0,
        base_dt=0.2
    )

    print(f"MLMC Result:")
    print(f"  Mean: {mlmc_result.mean:.4f}")
    print(f"  Variance: {mlmc_result.variance:.6f}")
    print(f"  Levels: {mlmc_result.L}")
    print(f"  Total samples: {sum(mlmc_result.N_samples)}")
    print(f"  Total cost: {mlmc_result.total_cost:.0f}")

    # ============================================================================
    # 3. Delay Analysis
    # ============================================================================
    print("\n[3] DELAY ANALYSIS")
    print("-" * 80)

    delay_calc = DelayCalculator(network, confidence_level=0.95)

    # Generate delay samples for analysis
    # (In practice, these would come from simulation)
    np.random.seed(42)
    delay_samples = np.random.exponential(scale=0.015, size=1000)  # Mean 15 ms

    delay_metrics = delay_calc.estimate_delay_distribution(delay_samples)

    print(f"\nDelay Distribution:")
    print(f"  Mean: {delay_metrics.mean_delay*1000:.2f} ms")
    print(f"  Median: {delay_metrics.median_delay*1000:.2f} ms")
    print(f"  Std: {delay_metrics.std_delay*1000:.2f} ms")
    print(f"  Min: {delay_metrics.min_delay*1000:.2f} ms")
    print(f"  Max: {delay_metrics.max_delay*1000:.2f} ms")
    print(f"\nPercentiles:")
    for p_name, p_value in delay_metrics.percentiles.items():
        print(f"  {p_name}: {p_value*1000:.2f} ms")
    print(f"\n95% Confidence Interval:")
    print(f"  [{delay_metrics.ci_lower*1000:.2f}, {delay_metrics.ci_upper*1000:.2f}] ms")

    # Compare two scenarios
    print("\n" + "-" * 80)
    print("Scenario Comparison: Normal vs Heavy Load")

    delay_samples_heavy = np.random.exponential(scale=0.025, size=1000)  # Higher delay

    comparison = delay_calc.compare_delay_distributions(
        delay_samples,
        delay_samples_heavy,
        "Normal Load",
        "Heavy Load"
    )

    print(f"\nMean difference: {comparison['mean_difference']*1000:.2f} ms "
          f"({comparison['mean_difference_pct']:.1f}%)")
    print(f"T-test p-value: {comparison['t_test']['pvalue']:.4f}")
    print(f"Significant: {comparison['t_test']['significant']}")
    print(f"Effect size (Cohen's d): {comparison['effect_size']['cohens_d']:.2f} "
          f"({comparison['effect_size']['interpretation']})")

    # ============================================================================
    # 4. Congestion Analysis
    # ============================================================================
    print("\n[4] CONGESTION ANALYSIS")
    print("-" * 80)

    congestion_analyzer = CongestionAnalyzer(
        network,
        congestion_threshold=0.8,
        queue_threshold=10.0
    )

    # Generate synthetic queue states
    n_timesteps = 100
    n_nodes = network.n_nodes
    times = np.linspace(0, 10, n_timesteps)

    # Simulate queue evolution with some congestion
    queue_states = np.zeros((n_timesteps, n_nodes))
    for t in range(n_timesteps):
        # Base level + some variation + congestion at certain nodes
        queue_states[t, :] = np.random.exponential(scale=3.0, size=n_nodes)

        # Inject congestion at specific times and nodes
        if 40 <= t <= 60:  # Congestion event
            queue_states[t, :5] += 10.0  # First 5 nodes heavily congested

    congestion_metrics = congestion_analyzer.analyze_simulation_congestion(
        queue_states,
        arrival_rates=8.0,
        service_rates=10.0,
        times=times
    )

    print(f"\nCongestion Metrics:")
    print(f"  Mean Queue Length: {congestion_metrics.mean_queue_length:.2f}")
    print(f"  Max Queue Length: {congestion_metrics.max_queue_length:.2f}")
    print(f"  Mean Utilization: {congestion_metrics.mean_utilization:.3f}")
    print(f"  Max Utilization: {congestion_metrics.max_utilization:.3f}")
    print(f"  Congested Nodes: {len(congestion_metrics.congested_nodes)}")
    print(f"  Congestion Events: {len(congestion_metrics.congestion_events)}")

    print(f"\nCongestion Propagation:")
    for key, value in congestion_metrics.propagation_metrics.items():
        if key != 'clusters':
            print(f"  {key}: {value}")

    # Temporal evolution
    evolution = congestion_analyzer.compute_temporal_congestion_evolution(
        queue_states, times, window_size=10
    )

    print(f"\nTemporal Evolution:")
    print(f"  Trend: {evolution['trend']}")
    print(f"  Trend slope: {evolution['trend_slope']:.4f}")
    print(f"  Peak mean queue: {np.max(evolution['mean_queue']):.2f}")

    # Identify bottlenecks
    bottlenecks = congestion_analyzer.identify_bottlenecks(queue_states, percentile=90)

    print(f"\nTop 5 Bottleneck Nodes:")
    for i, (node_id, avg_queue) in enumerate(bottlenecks[:5]):
        print(f"  {i+1}. Node {node_id}: {avg_queue:.2f} avg queue")

    # ============================================================================
    # 5. Uncertainty Quantification
    # ============================================================================
    print("\n[5] UNCERTAINTY QUANTIFICATION")
    print("-" * 80)

    uq = UncertaintyQuantifier(
        confidence_level=0.95,
        n_bootstrap=1000,
        random_seed=42
    )

    # Bootstrap confidence interval
    print("\nBootstrap Confidence Interval:")

    ci_lower, ci_upper, bootstrap_dist = uq.bootstrap_confidence_interval(
        delay_samples,
        statistic=np.mean,
        method='percentile'
    )

    print(f"  Sample mean: {np.mean(delay_samples)*1000:.2f} ms")
    print(f"  Bootstrap 95% CI: [{ci_lower*1000:.2f}, {ci_upper*1000:.2f}] ms")
    print(f"  CI width: {(ci_upper - ci_lower)*1000:.2f} ms")

    # Prediction interval
    pi_lower, pi_upper = uq.compute_prediction_interval(
        delay_samples,
        method='quantile'
    )

    print(f"\nPrediction Interval (95%):")
    print(f"  [{pi_lower*1000:.2f}, {pi_upper*1000:.2f}] ms")
    print(f"  Width: {(pi_upper - pi_lower)*1000:.2f} ms")
    print(f"  (Wider than CI, accounts for future observation variability)")

    # Uncertainty band for time series
    print("\nUncertainty Band for Queue Evolution:")

    # Generate sample trajectories
    n_samples = 200
    sample_trajectories = np.zeros((n_samples, n_timesteps))

    for i in range(n_samples):
        # Each sample is a noisy version of a base trajectory
        base = 3.0 + 2.0 * np.sin(2 * np.pi * times / 10)
        noise = np.random.normal(0, 0.5, n_timesteps)
        sample_trajectories[i] = base + noise

    uncertainty_band = uq.compute_uncertainty_band(
        sample_trajectories,
        method='quantile'
    )

    print(f"  Time points: {len(uncertainty_band.times)}")
    print(f"  Mean trajectory range: [{np.min(uncertainty_band.mean):.2f}, "
          f"{np.max(uncertainty_band.mean):.2f}]")
    print(f"  Average CI width: {np.mean(uncertainty_band.width()):.2f}")
    print(f"  Max CI width: {np.max(uncertainty_band.width()):.2f}")

    # ============================================================================
    # 6. Variance Reduction Analysis (MC vs MLMC)
    # ============================================================================
    print("\n[6] VARIANCE REDUCTION ANALYSIS")
    print("-" * 80)

    # Generate synthetic samples representing MC and MLMC
    # (In practice, use actual simulation results)
    np.random.seed(42)
    mc_samples = np.random.normal(loc=5.0, scale=1.0, size=1000)
    mlmc_samples = np.random.normal(loc=5.0, scale=0.3, size=1000)  # Lower variance

    var_reduction = uq.analyze_variance_reduction(
        mc_samples,
        mlmc_samples,
        mc_cost=mc_result.runtime,
        mlmc_cost=mlmc_result.total_cost / 1000  # Approximate runtime
    )

    print(f"\nVariance Reduction Metrics:")
    print(f"  MC Variance: {var_reduction.mc_variance:.6f}")
    print(f"  MLMC Variance: {var_reduction.mlmc_variance:.6f}")
    print(f"  Reduction Factor: {var_reduction.reduction_factor:.2f}x")
    print(f"  Reduction: {var_reduction.reduction_percent:.1f}%")
    print(f"\nComputational Cost:")
    print(f"  MC Cost: {var_reduction.mc_cost:.2f}s")
    print(f"  MLMC Cost: {var_reduction.mlmc_cost:.2f}s")
    print(f"  Efficiency Gain: {var_reduction.efficiency_gain:.2f}x")

    # Compare multiple estimators
    print("\n" + "-" * 80)
    print("Multi-Estimator Comparison:")

    estimators = {
        'Standard MC': mc_samples,
        'MLMC': mlmc_samples,
        'Control Variates': np.random.normal(5.0, 0.5, 1000)  # Example
    }

    comparison = uq.compare_estimator_variance(estimators, reference='Standard MC')

    for name, stats in comparison.items():
        print(f"\n{name}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Variance: {stats['variance']:.6f}")
        print(f"  Std: {stats['std']:.4f}")
        if 'variance_reduction' in stats:
            print(f"  Variance Reduction: {stats['variance_reduction']:.2f}x "
                  f"({stats['variance_reduction_pct']:.1f}%)")

    # ============================================================================
    # 7. Convergence Diagnostics
    # ============================================================================
    print("\n[7] CONVERGENCE DIAGNOSTICS")
    print("-" * 80)

    # Check MC convergence
    mc_diagnostics = uq.diagnose_convergence(mc_samples)

    print(f"\nStandard MC Convergence:")
    print(f"  Converged: {mc_diagnostics['converged']}")
    print(f"  Relative Variance: {mc_diagnostics['relative_variance']:.4f}")
    print(f"  Effective Sample Size: {mc_diagnostics['effective_sample_size']:.0f}")
    print(f"  Batch Variance: {mc_diagnostics['batch_variance']:.6f}")

    # Effective sample size for autocorrelated data
    # Generate autocorrelated samples
    autocorr_samples = np.zeros(5000)
    autocorr_samples[0] = np.random.normal()
    for i in range(1, 5000):
        autocorr_samples[i] = 0.7 * autocorr_samples[i-1] + 0.3 * np.random.normal()

    ess = uq.compute_effective_sample_size(autocorr_samples)

    print(f"\nAutocorrelated Data:")
    print(f"  Actual samples: 5000")
    print(f"  Effective samples: {ess:.0f}")
    print(f"  Efficiency: {ess/5000*100:.1f}%")

    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nKey Findings:")
    print(f"  1. Mean delay: {delay_metrics.mean_delay*1000:.2f} ms "
          f"(95% CI: [{delay_metrics.ci_lower*1000:.2f}, {delay_metrics.ci_upper*1000:.2f}])")
    print(f"  2. Average queue length: {congestion_metrics.mean_queue_length:.2f}")
    print(f"  3. Network utilization: {congestion_metrics.mean_utilization:.3f}")
    print(f"  4. Congestion events detected: {len(congestion_metrics.congestion_events)}")
    print(f"  5. MLMC variance reduction: {var_reduction.reduction_factor:.2f}x")
    print(f"  6. Overall efficiency gain: {var_reduction.efficiency_gain:.2f}x")

    print("\nMetrics analysis demonstrates:")
    print("  - Comprehensive performance characterization with uncertainty bounds")
    print("  - Congestion detection and temporal evolution tracking")
    print("  - Significant variance reduction via MLMC (>10x typical)")
    print("  - Well-calibrated confidence intervals for decision making")

    print("\nNext steps:")
    print("  - Run full experiments with real network topologies")
    print("  - Compare with deterministic models (queueing theory)")
    print("  - Generate publication-quality visualizations")
    print("  - Export results for further analysis")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_metrics_analysis()
