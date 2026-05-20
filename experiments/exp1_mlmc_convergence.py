"""
Experiment 1: MLMC Convergence Analysis

Validates Multilevel Monte Carlo convergence rates and error-cost scaling.

Objectives:
- Demonstrate MLMC convergence rate: MSE ∝ Cost^(-1)
- Compare with standard MC: MSE ∝ Cost^(-0.5) for SDEs
- Verify theoretical complexity: MLMC O(ε^-2) vs MC O(ε^-3)
- Analyze level-wise variance decay: V_l ∝ M^(-αl)
- Measure computational cost savings

Network: Email-Eu-core (SNAP) or synthetic
Traffic: Poisson with controlled parameters
Levels: L = 0, 1, 2, 3, 4, 5
Target accuracies: ε = [0.1, 0.05, 0.01, 0.005, 0.001]

Expected Results:
- MLMC cost reduction: 10x-100x for tight tolerances
- Variance decay: α ≈ 2 for Euler-Maruyama
- Optimal sample allocation follows theory
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
from network.traffic import PoissonTraffic
from simulation.monte_carlo import MonteCarloSimulator
from simulation.mlmc import MLMCSimulator
from simulation.discretization import MLMCHierarchy
from datasets.synthetic.generator import SyntheticBenchmarkGenerator
from config import ExperimentConfig, parse_args, setup_logging, setup_output_dirs

logger = logging.getLogger(__name__)


def run_mc_convergence_test(
    network,
    traffic,
    target_epsilons: List[float],
    T: float = 10.0,
    dt: float = 0.01,
    seed: int = 42
) -> Dict:
    """Run standard MC convergence test.

    For each target accuracy ε, compute required samples N = Var/ε²
    and measure actual MSE and cost.

    Args:
        network: Network topology
        traffic: Traffic model
        target_epsilons: List of target accuracies
        T: Simulation time
        dt: Timestep
        seed: Random seed

    Returns:
        Dictionary with results for each epsilon
    """
    logger.info("Running Standard MC Convergence Test")
    logger.info(f"Target accuracies: {target_epsilons}")

    mc_simulator = MonteCarloSimulator(seed=seed)
    results = {}

    # First, estimate variance with pilot run
    logger.info("Running pilot to estimate variance...")
    pilot_samples = 1000
    pilot_result = mc_simulator.estimate(
        network=network,
        traffic=traffic,
        n_samples=pilot_samples,
        T=T,
        dt=dt,
        metric='mean_queue'
    )
    estimated_variance = pilot_result.variance

    logger.info(f"Estimated variance from pilot: {estimated_variance:.6e}")

    for epsilon in target_epsilons:
        logger.info(f"\n--- MC with ε = {epsilon} ---")

        # Required samples: N = Var / ε²
        n_required = int(np.ceil(estimated_variance / (epsilon ** 2)))
        n_required = max(n_required, 100)  # Minimum 100 samples

        logger.info(f"Required samples: {n_required}")

        # Run simulation
        start_time = time.time()
        result = mc_simulator.estimate(
            network=network,
            traffic=traffic,
            n_samples=n_required,
            T=T,
            dt=dt,
            metric='mean_queue'
        )
        runtime = time.time() - start_time

        # Compute MSE (use variance as proxy since true value unknown)
        # MSE ≈ Var/N
        mse = result.variance / n_required

        logger.info(f"Mean: {result.mean:.4f}")
        logger.info(f"Variance: {result.variance:.6e}")
        logger.info(f"MSE: {mse:.6e}")
        logger.info(f"Runtime: {runtime:.2f}s")
        logger.info(f"Cost (timesteps): {n_required * int(T/dt)}")

        results[epsilon] = {
            'method': 'MC',
            'epsilon': epsilon,
            'n_samples': n_required,
            'mean': result.mean,
            'variance': result.variance,
            'mse': mse,
            'runtime': runtime,
            'cost': n_required * int(T/dt),  # Total timesteps
            'ci_lower': result.ci_lower,
            'ci_upper': result.ci_upper
        }

    return results


def run_mlmc_convergence_test(
    network,
    traffic,
    target_epsilons: List[float],
    L_max: int = 5,
    T: float = 10.0,
    base_dt: float = 0.2,
    seed: int = 42
) -> Dict:
    """Run MLMC convergence test.

    For each target accuracy ε, run MLMC with automatic sample allocation
    and measure MSE and cost.

    Args:
        network: Network topology
        traffic: Traffic model
        target_epsilons: List of target accuracies
        L_max: Maximum level
        T: Simulation time
        base_dt: Base timestep (coarsest level)
        seed: Random seed

    Returns:
        Dictionary with results for each epsilon
    """
    logger.info("Running MLMC Convergence Test")
    logger.info(f"Target accuracies: {target_epsilons}")
    logger.info(f"Maximum level: L_max = {L_max}")

    mlmc_simulator = MLMCSimulator(seed=seed)
    results = {}

    for epsilon in target_epsilons:
        logger.info(f"\n--- MLMC with ε = {epsilon} ---")

        # Run MLMC
        start_time = time.time()
        result = mlmc_simulator.mlmc_estimate(
            network=network,
            traffic=traffic,
            epsilon=epsilon,
            L_max=L_max,
            T=T,
            base_dt=base_dt,
            metric='mean_queue'
        )
        runtime = time.time() - start_time

        logger.info(f"Mean: {result.mean:.4f}")
        logger.info(f"Variance: {result.variance:.6e}")
        logger.info(f"Levels used: L = {result.L}")
        logger.info(f"Samples per level: {result.N_samples}")
        logger.info(f"Total cost: {result.total_cost:.0f}")
        logger.info(f"Runtime: {runtime:.2f}s")

        # Compute MSE
        mse = result.variance

        results[epsilon] = {
            'method': 'MLMC',
            'epsilon': epsilon,
            'L': result.L,
            'N_samples': result.N_samples,
            'mean': result.mean,
            'variance': result.variance,
            'mse': mse,
            'runtime': runtime,
            'cost': result.total_cost,
            'level_variances': result.level_variances.tolist() if hasattr(result.level_variances, 'tolist') else result.level_variances,
            'level_costs': result.level_costs.tolist() if hasattr(result.level_costs, 'tolist') else result.level_costs
        }

    return results


def analyze_variance_decay(mlmc_results: Dict) -> Dict:
    """Analyze level-wise variance decay.

    Theoretical: V_l ∝ M^(-αl) where α ≈ 2 for Euler-Maruyama

    Args:
        mlmc_results: MLMC results dictionary

    Returns:
        Variance decay analysis
    """
    logger.info("\nAnalyzing Variance Decay")

    # Use results from smallest epsilon (most levels)
    epsilon_min = min(mlmc_results.keys())
    result = mlmc_results[epsilon_min]

    level_variances = np.array(result['level_variances'])
    L = result['L']

    # Fit V_l = C * M^(-α*l)
    # log(V_l) = log(C) - α*l*log(M)
    levels = np.arange(L + 1)
    log_variances = np.log(level_variances + 1e-16)  # Add small value to avoid log(0)

    # Linear regression
    valid = np.isfinite(log_variances)
    if np.sum(valid) >= 2:
        coeffs = np.polyfit(levels[valid], log_variances[valid], deg=1)
        alpha = -coeffs[0] / np.log(2)  # Assuming M=2
        C = np.exp(coeffs[1])

        logger.info(f"Variance decay rate: α = {alpha:.3f}")
        logger.info(f"Theoretical: α ≈ 2 for Euler-Maruyama")
        logger.info(f"Constant: C = {C:.6e}")
    else:
        alpha = np.nan
        C = np.nan

    return {
        'alpha': alpha,
        'C': C,
        'levels': levels.tolist(),
        'variances': level_variances.tolist(),
        'log_variances': log_variances.tolist()
    }


def compare_convergence_rates(mc_results: Dict, mlmc_results: Dict) -> Dict:
    """Compare convergence rates: MSE vs Cost.

    Theoretical:
    - MC: MSE ∝ Cost^(-0.5) for i.i.d. samples
    - MLMC: MSE ∝ Cost^(-1) for SDEs with Euler-Maruyama

    Args:
        mc_results: MC results
        mlmc_results: MLMC results

    Returns:
        Convergence rate analysis
    """
    logger.info("\nComparing Convergence Rates")

    # Extract data
    mc_costs = np.array([r['cost'] for r in mc_results.values()])
    mc_mses = np.array([r['mse'] for r in mc_results.values()])

    mlmc_costs = np.array([r['cost'] for r in mlmc_results.values()])
    mlmc_mses = np.array([r['mse'] for r in mlmc_results.values()])

    # Fit MSE = C * Cost^(-β)
    # log(MSE) = log(C) - β*log(Cost)

    # MC fit
    log_mc_costs = np.log(mc_costs)
    log_mc_mses = np.log(mc_mses)
    mc_coeffs = np.polyfit(log_mc_costs, log_mc_mses, deg=1)
    mc_beta = -mc_coeffs[0]
    mc_C = np.exp(mc_coeffs[1])

    logger.info(f"MC convergence rate: β = {mc_beta:.3f}")
    logger.info(f"  Theoretical: β ≈ 0.5 for i.i.d.")

    # MLMC fit
    log_mlmc_costs = np.log(mlmc_costs)
    log_mlmc_mses = np.log(mlmc_mses)
    mlmc_coeffs = np.polyfit(log_mlmc_costs, log_mlmc_mses, deg=1)
    mlmc_beta = -mlmc_coeffs[0]
    mlmc_C = np.exp(mlmc_coeffs[1])

    logger.info(f"MLMC convergence rate: β = {mlmc_beta:.3f}")
    logger.info(f"  Theoretical: β ≈ 1.0 for SDEs with MLMC")

    # Speedup factor
    speedup_factors = mc_costs / mlmc_costs
    avg_speedup = np.mean(speedup_factors)

    logger.info(f"\nAverage cost reduction: {avg_speedup:.2f}x")

    return {
        'mc': {
            'beta': mc_beta,
            'C': mc_C,
            'costs': mc_costs.tolist(),
            'mses': mc_mses.tolist()
        },
        'mlmc': {
            'beta': mlmc_beta,
            'C': mlmc_C,
            'costs': mlmc_costs.tolist(),
            'mses': mlmc_mses.tolist()
        },
        'speedup_factors': speedup_factors.tolist(),
        'average_speedup': avg_speedup
    }


def save_results(
    mc_results: Dict,
    mlmc_results: Dict,
    variance_decay: Dict,
    convergence_comparison: Dict,
    output_dir: Path
):
    """Save all results to JSON and CSV files.

    Args:
        mc_results: MC results
        mlmc_results: MLMC results
        variance_decay: Variance decay analysis
        convergence_comparison: Convergence rate comparison
        output_dir: Output directory
    """
    logger.info(f"\nSaving results to {output_dir}")

    # Save to JSON
    json_path = output_dir / "exp1_mlmc_convergence_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'mc_results': {str(k): v for k, v in mc_results.items()},
            'mlmc_results': {str(k): v for k, v in mlmc_results.items()},
            'variance_decay': variance_decay,
            'convergence_comparison': convergence_comparison
        }, f, indent=2)

    logger.info(f"Saved JSON: {json_path}")

    # Save comparison table to CSV
    csv_path = output_dir / "exp1_cost_comparison.csv"
    with open(csv_path, 'w') as f:
        f.write("Epsilon,MC_Samples,MC_Cost,MC_MSE,MLMC_Levels,MLMC_Cost,MLMC_MSE,Speedup\n")

        for epsilon in sorted(mc_results.keys()):
            mc_r = mc_results[epsilon]
            mlmc_r = mlmc_results[epsilon]

            speedup = mc_r['cost'] / mlmc_r['cost']

            f.write(
                f"{epsilon},"
                f"{mc_r['n_samples']},"
                f"{mc_r['cost']},"
                f"{mc_r['mse']:.6e},"
                f"{mlmc_r['L']},"
                f"{mlmc_r['cost']:.0f},"
                f"{mlmc_r['mse']:.6e},"
                f"{speedup:.2f}\n"
            )

    logger.info(f"Saved CSV: {csv_path}")


def print_summary(
    mc_results: Dict,
    mlmc_results: Dict,
    convergence_comparison: Dict
):
    """Print experiment summary.

    Args:
        mc_results: MC results
        mlmc_results: MLMC results
        convergence_comparison: Convergence comparison
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: MLMC CONVERGENCE ANALYSIS - SUMMARY")
    print("=" * 80)

    print("\nCost Comparison:")
    print("-" * 80)
    print(f"{'Epsilon':>10} {'MC Cost':>12} {'MLMC Cost':>12} {'Speedup':>10}")
    print("-" * 80)

    for epsilon in sorted(mc_results.keys()):
        mc_cost = mc_results[epsilon]['cost']
        mlmc_cost = mlmc_results[epsilon]['cost']
        speedup = mc_cost / mlmc_cost

        print(f"{epsilon:>10.4f} {mc_cost:>12.0f} {mlmc_cost:>12.0f} {speedup:>10.2f}x")

    print("\nConvergence Rates:")
    print("-" * 80)
    mc_beta = convergence_comparison['mc']['beta']
    mlmc_beta = convergence_comparison['mlmc']['beta']
    avg_speedup = convergence_comparison['average_speedup']

    print(f"MC convergence rate:     β = {mc_beta:.3f} (theory: 0.5)")
    print(f"MLMC convergence rate:   β = {mlmc_beta:.3f} (theory: 1.0)")
    print(f"Average cost reduction:  {avg_speedup:.2f}x")

    print("\nKey Findings:")
    print("-" * 80)
    if mlmc_beta > mc_beta:
        print("✓ MLMC achieves faster convergence than standard MC")
    if avg_speedup > 5:
        print(f"✓ MLMC provides significant cost savings ({avg_speedup:.1f}x)")
    if abs(mlmc_beta - 1.0) < 0.2:
        print("✓ MLMC convergence rate matches theory (β ≈ 1.0)")

    print("=" * 80)


def main(config: ExperimentConfig = None):
    """Main experiment runner.

    Args:
        config: Experiment configuration. If None, uses defaults.
    """
    if config is None:
        config = ExperimentConfig()

    # Setup logging and output directories
    setup_logging(config)
    results_dir, figures_dir, tables_dir = setup_output_dirs(config)

    print("=" * 80)
    print("EXPERIMENT 1: MLMC CONVERGENCE ANALYSIS")
    print("=" * 80)

    # ============================================================================
    # Setup
    # ============================================================================
    print("\n[SETUP]")
    print("-" * 80)

    # Create synthetic benchmark scenario
    generator = SyntheticBenchmarkGenerator(seed=config.seed)
    scenario = generator.generate_stable_queue_scenario(
        arrival_rate=8.0,
        service_rate=10.0,
        noise_intensity=0.5
    )

    network = scenario['network']
    traffic = scenario['traffic']

    print(f"Network: {network.n_nodes} nodes, {network.n_edges} edges")
    print(f"Traffic: {traffic}")
    print(f"Ground truth utilization: {scenario['ground_truth']['utilization']:.2f}")
    print(f"Expected queue length: {scenario['ground_truth']['expected_queue_length']:.2f}")

    # Experiment parameters from config
    target_epsilons = config.target_epsilons
    T = config.T
    dt_mc = config.dt / 10  # Finer timestep for MC
    base_dt_mlmc = config.dt * 2  # Coarser base for MLMC
    L_max = config.L_max
    seed = config.seed

    print(f"\nParameters:")
    print(f"  Target accuracies: {target_epsilons}")
    print(f"  Simulation time: T = {T}s")
    print(f"  MC timestep: dt = {dt_mc}s")
    print(f"  MLMC base timestep: {base_dt_mlmc}s")
    print(f"  MLMC max level: L = {L_max}")

    # ============================================================================
    # Run Experiments
    # ============================================================================

    # Standard MC
    print("\n[STANDARD MONTE CARLO]")
    print("-" * 80)
    mc_results = run_mc_convergence_test(
        network, traffic, target_epsilons, T, dt_mc, seed
    )

    # MLMC
    print("\n[MULTILEVEL MONTE CARLO]")
    print("-" * 80)
    mlmc_results = run_mlmc_convergence_test(
        network, traffic, target_epsilons, L_max, T, base_dt_mlmc, seed
    )

    # ============================================================================
    # Analysis
    # ============================================================================

    # Variance decay
    variance_decay = analyze_variance_decay(mlmc_results)

    # Convergence rate comparison
    convergence_comparison = compare_convergence_rates(mc_results, mlmc_results)

    # ============================================================================
    # Save Results
    # ============================================================================
    save_results(
        mc_results,
        mlmc_results,
        variance_decay,
        convergence_comparison,
        tables_dir
    )

    # ============================================================================
    # Summary
    # ============================================================================
    print_summary(mc_results, mlmc_results, convergence_comparison)

    print("\nResults saved to:")
    print(f"  {tables_dir / 'exp1_mlmc_convergence_results.json'}")
    print(f"  {tables_dir / 'exp1_cost_comparison.csv'}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    config = parse_args(description="MLMC Convergence Analysis Experiment")
    main(config)
