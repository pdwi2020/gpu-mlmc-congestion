"""
Basic Simulation Example

Demonstrates Monte Carlo and MLMC simulation for network performance estimation.

This script shows:
1. Creating a synthetic network topology
2. Defining traffic models
3. Running Monte Carlo simulation
4. Running MLMC simulation
5. Comparing efficiency gains
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from network.topology import TopologyGenerator
from network.traffic import PoissonTraffic, BurstyTraffic
from simulation.monte_carlo import MonteCarloSimulator
from simulation.mlmc import MLMCSimulator


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run basic simulation examples."""

    print("=" * 80)
    print("GPU-Accelerated MLMC Network Simulation - Basic Example")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Create Network Topology
    # -------------------------------------------------------------------------
    print("\n[Step 1] Creating Network Topology")
    print("-" * 80)

    gen = TopologyGenerator(seed=42)

    # Generate small Erdős-Rényi random network
    network = gen.generate_erdos_renyi(n_nodes=100, p=0.05)

    # Add synthetic link properties
    network.set_link_properties(
        bandwidth_range=(100.0, 1000.0),  # Mbps
        delay_range=(1.0, 10.0),           # ms
        capacity_range=(100.0, 500.0),     # packets
        seed=42
    )

    print(f"Network created: {network}")
    print(f"Summary: {network.summary()}")

    # -------------------------------------------------------------------------
    # Step 2: Define Traffic Model
    # -------------------------------------------------------------------------
    print("\n[Step 2] Defining Traffic Model")
    print("-" * 80)

    # Poisson traffic (simple model)
    traffic_poisson = PoissonTraffic(
        rate=10.0,               # packets/time unit
        mean_packet_size=1500.0, # bytes
        seed=42
    )

    # Bursty traffic (more realistic)
    traffic_bursty = BurstyTraffic(
        on_rate=50.0,            # packets/time unit during ON
        mean_on_duration=0.5,    # time units
        mean_off_duration=0.5,   # time units
        seed=42
    )

    print(f"Poisson traffic: {traffic_poisson}")
    print(f"Bursty traffic: {traffic_bursty}")

    # Use Poisson for simplicity in this example
    traffic = traffic_poisson

    # -------------------------------------------------------------------------
    # Step 3: Standard Monte Carlo Simulation
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running Standard Monte Carlo Simulation")
    print("-" * 80)

    mc_simulator = MonteCarloSimulator(seed=42)

    mc_result = mc_simulator.estimate(
        network=network,
        traffic=traffic,
        n_samples=200,           # Number of independent samples
        T=10.0,                  # Simulation duration
        dt=0.1,                  # Time step
        metric='mean_queue',     # Metric to estimate
        confidence_level=0.95,
        verbose=True
    )

    print(f"\n{mc_result.summary()}")
    print(f"Estimated mean queue length: {mc_result.mean:.4f}")
    print(f"Standard deviation: {mc_result.std:.4f}")
    print(f"95% Confidence Interval: [{mc_result.ci_lower:.4f}, {mc_result.ci_upper:.4f}]")
    print(f"Computational cost: {mc_result.computational_cost:.2e} timesteps")

    # -------------------------------------------------------------------------
    # Step 4: Multilevel Monte Carlo (MLMC) Simulation
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Multilevel Monte Carlo (MLMC) Simulation")
    print("-" * 80)

    mlmc_simulator = MLMCSimulator(refinement_factor=2, seed=42)

    mlmc_result = mlmc_simulator.mlmc_estimate(
        network=network,
        traffic=traffic,
        epsilon=0.01,            # Target accuracy
        L_max=4,                 # Maximum levels (0,1,2,3,4)
        T=10.0,                  # Simulation duration
        base_dt=0.1,             # Coarsest time step
        metric='mean_queue',
        pilot_samples=50,        # Samples for variance estimation
        confidence_level=0.95,
        verbose=True
    )

    print(f"\n{mlmc_result.summary()}")
    print(f"Estimated mean queue length: {mlmc_result.estimate:.4f}")
    print(f"Variance: {mlmc_result.variance:.6e}")
    print(f"√MSE: {np.sqrt(mlmc_result.mse):.6e}")
    print(f"95% Confidence Interval: [{mlmc_result.ci_lower:.4f}, {mlmc_result.ci_upper:.4f}]")
    print(f"Computational cost: {mlmc_result.total_cost:.2e} timesteps")

    # Print level-wise statistics
    print("\nMLMC Level Statistics:")
    for stats in mlmc_result.level_stats:
        print(f"  {stats}")

    # -------------------------------------------------------------------------
    # Step 5: Compare MC and MLMC Efficiency
    # -------------------------------------------------------------------------
    print("\n[Step 5] Comparing Monte Carlo and MLMC Efficiency")
    print("-" * 80)

    comparison = mlmc_simulator.compare_with_standard_mc(
        network=network,
        traffic=traffic,
        epsilon=0.01,
        L_max=4,
        T=10.0,
        base_dt=0.1,
        metric='mean_queue'
    )

    print("\nEfficiency Comparison:")
    print(f"  MLMC estimate:     {comparison['mlmc_estimate']:.6f}")
    print(f"  MLMC cost:         {comparison['mlmc_cost']:.2e} timesteps")
    print(f"  MC cost:           {comparison['mc_cost']:.2e} timesteps")
    print(f"  Speedup factor:    {comparison['speedup']:.2f}x")
    print(f"  Cost reduction:    {comparison['cost_reduction']*100:.1f}%")

    # -------------------------------------------------------------------------
    # Step 6: Convergence Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 6] Monte Carlo Convergence Analysis")
    print("-" * 80)

    convergence = mc_simulator.convergence_test(
        network=network,
        traffic=traffic,
        sample_sizes=[50, 100, 200, 500, 1000],
        T=10.0,
        dt=0.1,
        metric='mean_queue',
        n_trials=5
    )

    print("\nConvergence Results (as N increases):")
    print(f"{'N':>6} | {'Mean':>10} | {'Std':>10} | {'CI Width':>10}")
    print("-" * 48)
    for i, n in enumerate(convergence['sample_sizes']):
        print(f"{n:6d} | {convergence['means'][i]:10.4f} | "
              f"{convergence['stds'][i]:10.4f} | {convergence['ci_widths'][i]:10.4f}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Simulation Complete!")
    print("=" * 80)

    print("\nKey Findings:")
    print(f"1. Network performance metric (mean queue length): {mlmc_result.estimate:.4f}")
    print(f"2. MLMC achieved {comparison['speedup']:.1f}x speedup over standard MC")
    print(f"3. MLMC reduced computational cost by {comparison['cost_reduction']*100:.0f}%")
    print(f"4. Confidence interval width decreases with more samples (MC convergence)")

    print("\nNext Steps:")
    print("- Experiment with different network topologies (scale-free, small-world)")
    print("- Try bursty traffic models for more realistic scenarios")
    print("- Test on larger networks (1000+ nodes)")
    print("- Add GPU acceleration for massive speedups")
    print("- Validate against real datasets (SNAP, CAIDA, MAWI)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
