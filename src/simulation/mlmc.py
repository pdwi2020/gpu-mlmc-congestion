"""
Multilevel Monte Carlo (MLMC) Simulation Module

Implementation of MLMC method for efficient uncertainty quantification
in network simulations.

Key features:
- Hierarchical discretization levels
- Coupled path generation for variance reduction
- Optimal sample allocation
- Adaptive level selection

Classes:
    MLMCSimulator: Main MLMC simulator
    MLMCResult: Container for MLMC results with level-wise statistics
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from dataclasses import dataclass, field
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from network.topology import NetworkGraph
from network.traffic import TrafficModel
from network.sde import QueueDynamicsSDE
from simulation.discretization import (
    MLMCHierarchy,
    generate_coupled_noise,
    get_timestep
)


logger = logging.getLogger(__name__)


@dataclass
class MLMCLevelStats:
    """Statistics for a single MLMC level."""
    level: int
    n_samples: int
    dt: float
    mean_Y: float  # Mean of Y_l
    var_Y: float   # Variance of Y_l
    mean_diff: float  # Mean of (Y_l - Y_{l-1})
    var_diff: float   # Variance of (Y_l - Y_{l-1})
    cost_per_sample: float
    total_cost: float

    def __repr__(self) -> str:
        return (f"MLMCLevel(l={self.level}, N={self.n_samples}, "
                f"E[Y_l-Y_{{l-1}}]={self.mean_diff:.6f}, "
                f"V[Y_l-Y_{{l-1}}]={self.var_diff:.6f})")


@dataclass
class MLMCResult:
    """
    Container for MLMC simulation results.

    Attributes:
        estimate: Final MLMC estimate
        variance: Total variance
        mse: Mean squared error estimate
        level_stats: Statistics for each level
        total_cost: Total computational cost
        L_max: Maximum level used
        epsilon: Target accuracy
    """
    estimate: float
    variance: float
    mse: float
    level_stats: List[MLMCLevelStats]
    total_cost: float
    L_max: int
    epsilon: float
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    confidence_level: float = 0.95
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"MLMCResult(L={self.L_max}, estimate={self.estimate:.4f}, "
            f"√MSE={np.sqrt(self.mse):.4f}, cost={self.total_cost:.2e})"
        )

    def variance_breakdown(self) -> str:
        """Return variance contribution by level."""
        lines = ["Variance breakdown by level:"]
        for stats in self.level_stats:
            var_contribution = stats.var_diff / stats.n_samples
            lines.append(f"  Level {stats.level}: V={var_contribution:.6e}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


class MLMCSimulator:
    """
    Multilevel Monte Carlo simulator for network performance estimation.

    Implements the MLMC algorithm with:
    - Coupled path generation across levels
    - Optimal sample allocation based on variance and cost
    - Adaptive level selection
    """

    def __init__(self,
                 refinement_factor: int = 2,
                 seed: Optional[int] = None):
        """
        Initialize MLMC simulator.

        Args:
            refinement_factor: Time discretization refinement factor M
            seed: Random seed for reproducibility
        """
        self.refinement_factor = refinement_factor
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def run_coupled_paths(self,
                         network: NetworkGraph,
                         traffic: TrafficModel,
                         level: int,
                         T: float,
                         base_dt: float,
                         metric: str = 'mean_queue',
                         seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Run coupled coarse and fine paths for MLMC level l.

        Returns (Y_l, Y_{l-1}) where:
        - Y_l is the metric value with time step dt_l
        - Y_{l-1} is the metric value with time step dt_{l-1}

        For level 0, Y_{-1} is defined as 0.

        Args:
            network: NetworkGraph object
            traffic: TrafficModel object
            level: MLMC level index
            T: Simulation duration
            base_dt: Base time step (dt_0)
            metric: Metric to compute
            seed: Random seed

        Returns:
            Tuple of (Y_fine, Y_coarse)
        """
        if seed is not None:
            np.random.seed(seed)

        # Get time steps for this level
        dt_fine = get_timestep(level, base_dt, self.refinement_factor)

        if level == 0:
            # Level 0: only compute Y_0
            Y_fine = self._simulate_single_level(
                network, traffic, T, dt_fine, metric, seed
            )
            Y_coarse = 0.0  # Y_{-1} = 0 by convention
        else:
            # Level l > 0: compute coupled (Y_l, Y_{l-1})
            dt_coarse = get_timestep(level - 1, base_dt, self.refinement_factor)

            Y_fine, Y_coarse = self._simulate_coupled_levels(
                network, traffic, T, dt_fine, dt_coarse, metric, seed
            )

        return Y_fine, Y_coarse

    def _simulate_single_level(self,
                               network: NetworkGraph,
                               traffic: TrafficModel,
                               T: float,
                               dt: float,
                               metric: str,
                               seed: Optional[int]) -> float:
        """Simulate single discretization level."""
        # Get traffic parameters
        traffic_stats = traffic.get_statistics(duration=T)
        arrival_rate = traffic_stats['arrival_rate']
        service_rate = arrival_rate * 1.25  # Slightly overprovisioned

        # Create queue SDE
        queue_sde = QueueDynamicsSDE(
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            noise_intensity=0.2
        )

        # Simulate path
        time, queue_length = queue_sde.simulate_path(T=T, dt=dt, q0=0.0, seed=seed)

        # Compute metric
        if metric == 'mean_queue':
            return np.mean(queue_length)
        elif metric == 'max_queue':
            return np.max(queue_length)
        elif metric == 'final_queue':
            return queue_length[-1]
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _simulate_coupled_levels(self,
                                 network: NetworkGraph,
                                 traffic: TrafficModel,
                                 T: float,
                                 dt_fine: float,
                                 dt_coarse: float,
                                 metric: str,
                                 seed: Optional[int]) -> Tuple[float, float]:
        """Simulate coupled fine and coarse paths."""
        # Get traffic parameters
        traffic_stats = traffic.get_statistics(duration=T)
        arrival_rate = traffic_stats['arrival_rate']
        service_rate = arrival_rate * 1.25

        # Create queue SDE
        queue_sde = QueueDynamicsSDE(
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            noise_intensity=0.2
        )

        # Simulate coupled paths
        time_fine, q_fine, q_coarse = queue_sde.simulate_coupled_paths(
            T=T,
            dt_coarse=dt_coarse,
            dt_fine=dt_fine,
            q0=0.0,
            seed=seed
        )

        # Compute metric for both paths
        if metric == 'mean_queue':
            Y_fine = np.mean(q_fine)
            Y_coarse = np.mean(q_coarse)
        elif metric == 'max_queue':
            Y_fine = np.max(q_fine)
            Y_coarse = np.max(q_coarse)
        elif metric == 'final_queue':
            Y_fine = q_fine[-1]
            Y_coarse = q_coarse[-1]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return Y_fine, Y_coarse

    def estimate_level_variance(self,
                                network: NetworkGraph,
                                traffic: TrafficModel,
                                level: int,
                                T: float,
                                base_dt: float,
                                metric: str,
                                n_samples: int = 100) -> Tuple[float, float, float]:
        """
        Estimate variance and cost for a single MLMC level.

        Args:
            network: NetworkGraph object
            traffic: TrafficModel object
            level: MLMC level
            T: Simulation duration
            base_dt: Base time step
            metric: Metric to compute
            n_samples: Number of samples for variance estimation

        Returns:
            Tuple of (mean_diff, var_diff, cost_per_sample)
        """
        differences = np.zeros(n_samples)

        for i in range(n_samples):
            sample_seed = (self.seed + i) if self.seed is not None else None
            Y_fine, Y_coarse = self.run_coupled_paths(
                network, traffic, level, T, base_dt, metric, sample_seed
            )
            differences[i] = Y_fine - Y_coarse

        mean_diff = np.mean(differences)
        var_diff = np.var(differences, ddof=1)

        # Cost: number of timesteps for fine level
        dt_fine = get_timestep(level, base_dt, self.refinement_factor)
        cost_per_sample = T / dt_fine

        return mean_diff, var_diff, cost_per_sample

    def compute_optimal_samples(self,
                                variances: List[float],
                                costs: List[float],
                                epsilon: float) -> List[int]:
        """
        Compute optimal number of samples per level.

        Uses the MLMC theorem for optimal allocation:
            N_l = (2/ε²) * √(V_l / C_l) * Σ_k √(V_k * C_k)

        Args:
            variances: List of variances V_l for each level
            costs: List of costs C_l per sample for each level
            epsilon: Target accuracy ε

        Returns:
            List of sample counts [N_0, N_1, ..., N_L]
        """
        L = len(variances) - 1

        # Compute sum term: Σ_k √(V_k * C_k)
        sum_term = np.sum([np.sqrt(v * c) for v, c in zip(variances, costs)])

        # Compute N_l for each level
        N_samples = []
        for l in range(L + 1):
            if variances[l] <= 0 or costs[l] <= 0:
                N_samples.append(1)
                continue

            N_l = (2.0 / epsilon ** 2) * np.sqrt(variances[l] / costs[l]) * sum_term
            N_l = int(np.ceil(N_l))
            N_l = max(1, N_l)  # At least 1 sample
            N_samples.append(N_l)

        return N_samples

    def mlmc_estimate(self,
                     network: NetworkGraph,
                     traffic: TrafficModel,
                     epsilon: float,
                     L_max: int,
                     T: float = 10.0,
                     base_dt: float = 0.1,
                     metric: str = 'mean_queue',
                     pilot_samples: int = 100,
                     confidence_level: float = 0.95,
                     verbose: bool = True) -> MLMCResult:
        """
        Run MLMC estimation with adaptive sample allocation.

        Args:
            network: NetworkGraph object
            traffic: TrafficModel object
            epsilon: Target accuracy (MSE ≈ ε²)
            L_max: Maximum number of levels
            T: Simulation duration
            base_dt: Base time step (dt_0)
            metric: Metric to estimate
            pilot_samples: Initial samples for variance estimation
            confidence_level: Confidence level for CI
            verbose: Print progress

        Returns:
            MLMCResult object
        """
        if verbose:
            logger.info(f"Starting MLMC simulation: ε={epsilon}, L_max={L_max}")

        # Step 1: Pilot run to estimate variances and costs
        if verbose:
            logger.info(f"Step 1: Pilot run with {pilot_samples} samples per level")

        variances = []
        costs = []
        mean_diffs_pilot = []

        for l in range(L_max + 1):
            mean_diff, var_diff, cost = self.estimate_level_variance(
                network, traffic, l, T, base_dt, metric, pilot_samples
            )
            variances.append(var_diff)
            costs.append(cost)
            mean_diffs_pilot.append(mean_diff)

            if verbose:
                logger.info(f"  Level {l}: V={var_diff:.6e}, C={cost:.2e}, "
                           f"E[Y_l-Y_{{l-1}}]={mean_diff:.6e}")

        # Step 2: Compute optimal sample allocation
        if verbose:
            logger.info("Step 2: Computing optimal sample allocation")

        optimal_N = self.compute_optimal_samples(variances, costs, epsilon)

        if verbose:
            for l, N_l in enumerate(optimal_N):
                logger.info(f"  Level {l}: N={N_l}")

        # Step 3: Generate additional samples to reach optimal N
        if verbose:
            logger.info("Step 3: Generating additional samples")

        level_stats = []
        total_cost = 0.0

        # Storage for all samples
        all_diffs = [[] for _ in range(L_max + 1)]

        for l in range(L_max + 1):
            n_needed = optimal_N[l] - pilot_samples
            n_total = optimal_N[l]

            # Collect differences
            diffs = []

            for i in range(n_total):
                sample_seed = (self.seed + l * 10000 + i) if self.seed is not None else None
                Y_fine, Y_coarse = self.run_coupled_paths(
                    network, traffic, l, T, base_dt, metric, sample_seed
                )
                diffs.append(Y_fine - Y_coarse)

            diffs = np.array(diffs)

            # Compute level statistics
            mean_Y = np.mean([d + mean_diffs_pilot[l-1] if l > 0 else d for d in diffs])
            var_Y = np.var(diffs, ddof=1)
            mean_diff = np.mean(diffs)
            var_diff = np.var(diffs, ddof=1)

            dt_l = get_timestep(l, base_dt, self.refinement_factor)
            cost_per_sample = T / dt_l
            level_cost = cost_per_sample * n_total
            total_cost += level_cost

            stats = MLMCLevelStats(
                level=l,
                n_samples=n_total,
                dt=dt_l,
                mean_Y=mean_Y,
                var_Y=var_Y,
                mean_diff=mean_diff,
                var_diff=var_diff,
                cost_per_sample=cost_per_sample,
                total_cost=level_cost
            )

            level_stats.append(stats)
            all_diffs[l] = diffs

            if verbose:
                logger.info(f"  Level {l} complete: {stats}")

        # Step 4: Compute final MLMC estimate
        estimate = sum([stats.mean_diff for stats in level_stats])

        # Total variance
        variance = sum([stats.var_diff / stats.n_samples for stats in level_stats])

        # MSE estimate (variance + bias²)
        # Bias from discretization: assume weak order 1
        dt_finest = get_timestep(L_max, base_dt, self.refinement_factor)
        bias_estimate = dt_finest  # Conservative estimate
        mse = variance + bias_estimate ** 2

        # Confidence interval
        from scipy import stats as sp_stats
        alpha = 1 - confidence_level
        z_value = sp_stats.norm.ppf(1 - alpha / 2)
        margin = z_value * np.sqrt(variance)
        ci_lower = estimate - margin
        ci_upper = estimate + margin

        result = MLMCResult(
            estimate=estimate,
            variance=variance,
            mse=mse,
            level_stats=level_stats,
            total_cost=total_cost,
            L_max=L_max,
            epsilon=epsilon,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            metadata={
                'T': T,
                'base_dt': base_dt,
                'metric': metric,
                'refinement_factor': self.refinement_factor
            }
        )

        if verbose:
            logger.info(f"MLMC complete: {result.summary()}")
            logger.info(f"Estimate: {estimate:.6f} ± {np.sqrt(variance):.6f}")
            logger.info(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
            logger.info(result.variance_breakdown())

        return result

    def compare_with_standard_mc(self,
                                network: NetworkGraph,
                                traffic: TrafficModel,
                                epsilon: float,
                                L_max: int,
                                T: float = 10.0,
                                base_dt: float = 0.1,
                                metric: str = 'mean_queue') -> Dict:
        """
        Compare MLMC with standard Monte Carlo.

        Args:
            network: NetworkGraph object
            traffic: TrafficModel object
            epsilon: Target accuracy
            L_max: Maximum MLMC level
            T: Simulation duration
            base_dt: Base time step
            metric: Metric to estimate

        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing MLMC with standard Monte Carlo")

        # Run MLMC
        mlmc_result = self.mlmc_estimate(
            network, traffic, epsilon, L_max, T, base_dt, metric, verbose=False
        )

        # Standard MC: use finest discretization
        dt_finest = get_timestep(L_max, base_dt, self.refinement_factor)

        # For MC, to achieve same variance as MLMC, need:
        # N_MC = V / ε²
        # where V is total variance at finest level
        V_finest = mlmc_result.level_stats[-1].var_Y
        N_mc = int(np.ceil(V_finest / epsilon ** 2))

        mc_cost = N_mc * (T / dt_finest)

        comparison = {
            'mlmc_estimate': mlmc_result.estimate,
            'mlmc_variance': mlmc_result.variance,
            'mlmc_cost': mlmc_result.total_cost,
            'mc_variance_target': epsilon ** 2,
            'mc_n_samples': N_mc,
            'mc_cost': mc_cost,
            'speedup': mc_cost / mlmc_result.total_cost,
            'cost_reduction': 1 - (mlmc_result.total_cost / mc_cost)
        }

        logger.info(f"MLMC cost: {mlmc_result.total_cost:.2e}")
        logger.info(f"MC cost (estimated): {mc_cost:.2e}")
        logger.info(f"Speedup: {comparison['speedup']:.2f}x")
        logger.info(f"Cost reduction: {comparison['cost_reduction']*100:.1f}%")

        return comparison


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("MLMC Simulation Module - Example Usage")
    print("=" * 60)

    # Create network and traffic
    from network.topology import TopologyGenerator
    from network.traffic import PoissonTraffic

    gen = TopologyGenerator(seed=42)
    network = gen.generate_erdos_renyi(n_nodes=50, p=0.1)
    network.set_link_properties(seed=42)

    traffic = PoissonTraffic(rate=10.0, seed=42)

    print(f"\nNetwork: {network}")
    print(f"Traffic: {traffic}")

    # Run MLMC estimation
    print("\n1. MLMC Estimation")
    print("-" * 60)

    simulator = MLMCSimulator(refinement_factor=2, seed=42)
    result = simulator.mlmc_estimate(
        network=network,
        traffic=traffic,
        epsilon=0.01,
        L_max=4,
        T=10.0,
        base_dt=0.1,
        metric='mean_queue',
        pilot_samples=50
    )

    print(f"\nFinal estimate: {result.estimate:.6f}")
    print(f"Variance: {result.variance:.6e}")
    print(f"√MSE: {np.sqrt(result.mse):.6e}")
    print(f"Total cost: {result.total_cost:.2e}")

    # Compare with standard MC
    print("\n2. MLMC vs Standard MC")
    print("-" * 60)

    comparison = simulator.compare_with_standard_mc(
        network=network,
        traffic=traffic,
        epsilon=0.01,
        L_max=4,
        T=10.0,
        base_dt=0.1
    )

    print(f"\nMLMC cost: {comparison['mlmc_cost']:.2e}")
    print(f"MC cost: {comparison['mc_cost']:.2e}")
    print(f"Speedup: {comparison['speedup']:.2f}x")
    print(f"Cost reduction: {comparison['cost_reduction']*100:.1f}%")

    print("\n" + "=" * 60)
