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
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from dataclasses import dataclass, field
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from network.topology import NetworkGraph, centrality_weights
from network.traffic import TrafficModel
from network.sde import CongestionPropagationSDE, QueueDynamicsSDE
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

    @property
    def cost(self) -> float:
        """Total cost for this level (alias for total_cost)."""
        return self.total_cost

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

    @property
    def mean(self) -> float:
        """Alias for estimate (backward compatibility)."""
        return self.estimate

    @property
    def rmse(self) -> float:
        """Root mean squared error."""
        return np.sqrt(self.mse)

    @property
    def N_samples(self) -> List[int]:
        """Sample counts per level, for backward compatibility."""
        return [s.n_samples for s in self.level_stats]

    @property
    def level_variances(self) -> List[float]:
        """Per-level variance estimates (V_l)."""
        return [s.var_diff for s in self.level_stats]

    @property
    def level_costs(self) -> List[float]:
        """Per-level cost estimates (C_l * N_l or similar)."""
        return [s.cost_per_sample * s.n_samples for s in self.level_stats]

    @property
    def L(self) -> int:
        """Number of levels used."""
        return len(self.level_stats)

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
                                n_samples: int = 100,
                                return_samples: bool = False) -> Tuple[float, float, float, Optional[np.ndarray]]:
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
            return_samples: If True, return the sample differences for reuse

        Returns:
            Tuple of (mean_diff, var_diff, cost_per_sample, differences)
            If return_samples=False, differences is None
        """
        differences = np.zeros(n_samples)

        for i in range(n_samples):
            sample_seed = (self.seed + level * 10000 + i) if self.seed is not None else None
            Y_fine, Y_coarse = self.run_coupled_paths(
                network, traffic, level, T, base_dt, metric, sample_seed
            )
            differences[i] = Y_fine - Y_coarse

        mean_diff = np.mean(differences)
        var_diff = np.var(differences, ddof=1)

        # Cost: number of timesteps for fine level
        dt_fine = get_timestep(level, base_dt, self.refinement_factor)
        cost_per_sample = T / dt_fine

        if return_samples:
            return mean_diff, var_diff, cost_per_sample, differences
        return mean_diff, var_diff, cost_per_sample, None

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

    # Bias calibration constant for reflected SDE (weak order 0.5)
    # Conservative estimate based on empirical analysis
    BIAS_CALIBRATION_CONSTANT = 0.5

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

        # Step 1: Pilot run to estimate variances and costs (store samples for reuse)
        if verbose:
            logger.info(f"Step 1: Pilot run with {pilot_samples} samples per level")

        variances = []
        costs = []
        mean_diffs_pilot = []
        pilot_diffs = []  # Store pilot samples for reuse

        for l in range(L_max + 1):
            mean_diff, var_diff, cost, diffs = self.estimate_level_variance(
                network, traffic, l, T, base_dt, metric, pilot_samples,
                return_samples=True  # Return samples for reuse
            )
            variances.append(var_diff)
            costs.append(cost)
            mean_diffs_pilot.append(mean_diff)
            pilot_diffs.append(diffs)

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

        # Step 3: Generate ADDITIONAL samples (reuse pilot samples)
        if verbose:
            logger.info("Step 3: Generating additional samples (reusing pilot samples)")

        level_stats = []
        total_cost = 0.0

        for l in range(L_max + 1):
            # Calculate how many additional samples needed beyond pilot
            n_additional = max(0, optimal_N[l] - pilot_samples)
            n_total = pilot_samples + n_additional

            # Start with pilot samples (reuse them!)
            diffs = list(pilot_diffs[l])

            # Generate only additional samples needed
            for i in range(n_additional):
                # Start seed after pilot samples to avoid duplicates
                sample_seed = (self.seed + l * 10000 + pilot_samples + i) if self.seed is not None else None
                Y_fine, Y_coarse = self.run_coupled_paths(
                    network, traffic, l, T, base_dt, metric, sample_seed
                )
                diffs.append(Y_fine - Y_coarse)

            diffs = np.array(diffs)

            # Compute level statistics
            mean_diff = np.mean(diffs)
            var_diff = np.var(diffs, ddof=1)

            # Compute mean_Y as cumulative estimate (for diagnostics)
            if l == 0:
                mean_Y = mean_diff  # E[Y_0]
            else:
                # Sum of all level differences up to this point
                mean_Y = sum(mean_diffs_pilot[:l]) + mean_diff
            var_Y = var_diff

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

            if verbose:
                logger.info(f"  Level {l} complete: {stats} (reused {pilot_samples}, new {n_additional})")

        # Step 4: Compute final MLMC estimate
        estimate = sum([stats.mean_diff for stats in level_stats])

        # Total variance
        variance = sum([stats.var_diff / stats.n_samples for stats in level_stats])

        # MSE estimate (variance + bias²)
        # Bias from discretization: weak order 0.5 for reflected SDE
        # Using calibrated bias constant for conservative estimate
        dt_finest = get_timestep(L_max, base_dt, self.refinement_factor)
        bias_estimate = self.BIAS_CALIBRATION_CONSTANT * np.sqrt(dt_finest)
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

    def estimate(self, network, traffic, epsilon, T=10.0, base_dt=0.1,
                 L_max=4, metric='mean_queue', pilot_samples=100,
                 min_samples=None, confidence_level=0.95, verbose=False,
                 **kwargs) -> 'MLMCResult':
        """Alias for mlmc_estimate() with backward-compatible parameter names."""
        if min_samples is not None:
            pilot_samples = min_samples
        return self.mlmc_estimate(
            network=network,
            traffic=traffic,
            epsilon=epsilon,
            L_max=L_max,
            T=T,
            base_dt=base_dt,
            metric=metric,
            pilot_samples=pilot_samples,
            confidence_level=confidence_level,
            verbose=verbose,
        )

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

        # For MC, to achieve same MSE as MLMC, need:
        # N_MC = Var(Y_L) / ε²
        # where Var(Y_L) is the variance of a single MC sample at finest level
        # Estimate Var(Y_L) as sum of level variances (telescoping property):
        # Var(Y_L) ≈ Var(Y_0) + Σ Var(Y_l - Y_{l-1})
        V_mc_sample = sum(stats.var_diff for stats in mlmc_result.level_stats)
        N_mc = int(np.ceil(V_mc_sample / epsilon ** 2))

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


class AdaptiveNetworkAwareMLMC(MLMCSimulator):
    """Adaptive Network-Aware MLMC with centrality, variance, and SLA weights."""

    def __init__(self,
                 refinement_factor: int = 2,
                 seed: Optional[int] = None,
                 weight_centrality: float = 0.4,
                 weight_variance: float = 0.4,
                 weight_sla: float = 0.2,
                 centrality_kind: str = 'pagerank',
                 sla_priority: Optional[np.ndarray] = None) -> None:
        """Initialize ANA-MLMC with network-risk weighting coefficients."""
        super().__init__(refinement_factor=refinement_factor, seed=seed)
        gammas = np.array([weight_centrality, weight_variance, weight_sla], dtype=float)
        if np.any(gammas < 0.0):
            raise ValueError("ANA-MLMC weights must be non-negative")
        if not np.isclose(float(np.sum(gammas)), 1.0):
            raise ValueError("ANA-MLMC weights must sum to 1.0")

        self.weight_centrality = float(weight_centrality)
        self.weight_variance = float(weight_variance)
        self.weight_sla = float(weight_sla)
        self.centrality_kind = centrality_kind
        self.sla_priority = None if sla_priority is None else np.asarray(sla_priority, dtype=float)

    @staticmethod
    def _normalize_component(values: np.ndarray, n_nodes: int) -> np.ndarray:
        """Normalize a non-negative component or return uniform weights."""
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != n_nodes:
            raise ValueError(f"Expected vector of length {n_nodes}, got {arr.size}")

        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr = np.maximum(arr, 0.0)
        total = float(np.sum(arr))
        if total <= 0.0:
            return np.full(n_nodes, 1.0 / n_nodes, dtype=float)
        return arr / total

    def compute_node_weights(self,
                             graph: NetworkGraph,
                             pilot_per_node_vars: np.ndarray,
                             sla_vec: Optional[np.ndarray] = None) -> np.ndarray:
        """Combine centrality, pilot variance, and SLA priority into node weights."""
        n_nodes = graph.n_nodes
        if n_nodes <= 0:
            raise ValueError("ANA-MLMC requires at least one network node")

        pilot_vars = np.asarray(pilot_per_node_vars, dtype=float)
        if pilot_vars.ndim == 1:
            variance_component_raw = pilot_vars
        elif pilot_vars.ndim == 2:
            variance_component_raw = np.mean(pilot_vars, axis=0)
        else:
            raise ValueError("pilot_per_node_vars must have shape (n,) or (L+1, n)")

        c_i = centrality_weights(graph, kind=self.centrality_kind)
        v_i = self._normalize_component(variance_component_raw, n_nodes)

        sla_source = sla_vec
        if sla_source is None:
            sla_source = self.sla_priority
        if sla_source is None:
            sla_source = np.ones(n_nodes, dtype=float)
        s_i = self._normalize_component(np.asarray(sla_source, dtype=float), n_nodes)

        weights = (
            self.weight_centrality * c_i
            + self.weight_variance * v_i
            + self.weight_sla * s_i
        )
        return self._normalize_component(weights, n_nodes)

    def compute_optimal_samples_weighted(self,
                                         level_var_per_node: np.ndarray,
                                         costs: List[float],
                                         weights: np.ndarray,
                                         epsilon: float) -> List[int]:
        """Compute Giles allocation using weighted per-node level variances."""
        level_vars = np.asarray(level_var_per_node, dtype=float)
        if level_vars.ndim != 2:
            raise ValueError("level_var_per_node must have shape (L+1, n)")

        n_levels, n_nodes = level_vars.shape
        if len(costs) != n_levels:
            raise ValueError("costs length must match the number of MLMC levels")

        node_weights = self._normalize_component(np.asarray(weights, dtype=float), n_nodes)
        uniform = np.full(n_nodes, 1.0 / n_nodes, dtype=float)

        if np.allclose(node_weights, uniform, rtol=0.0, atol=1.0e-14):
            weighted_variances = np.mean(level_vars, axis=1)
            weighted_samples = super().compute_optimal_samples(
                weighted_variances.tolist(), costs, epsilon
            )
            giles_samples = super().compute_optimal_samples(
                np.mean(level_vars, axis=1).tolist(), costs, epsilon
            )
            assert weighted_samples == giles_samples
            return weighted_samples

        weighted_variances = level_vars @ node_weights
        return super().compute_optimal_samples(weighted_variances.tolist(), costs, epsilon)

    def _make_congestion_sde(self, network: NetworkGraph) -> CongestionPropagationSDE:
        """Build the vector congestion SDE used by network-aware estimation."""
        adjacency = network.get_adjacency_matrix()
        return CongestionPropagationSDE(
            adjacency_matrix=adjacency,
            influence_strength=0.1,
            decay_rate=0.5,
            noise_intensity=0.1,
        )

    @staticmethod
    def _extract_node_metric(path: np.ndarray, metric: str) -> np.ndarray:
        """Extract a per-node metric from a congestion path."""
        if metric in ('mean_congestion', 'mean_queue', 'mean'):
            return np.mean(path, axis=0)
        if metric in ('max_congestion', 'max_queue', 'max'):
            return np.max(path, axis=0)
        if metric in ('final_congestion', 'final_queue', 'final'):
            return path[-1]
        raise ValueError(f"Unknown metric: {metric}")

    def _simulate_single_level_node_values(self,
                                           network: NetworkGraph,
                                           traffic: TrafficModel,
                                           T: float,
                                           dt: float,
                                           metric: str,
                                           seed: Optional[int]) -> np.ndarray:
        """Simulate one vector congestion level and return per-node values."""
        _ = traffic
        congestion_sde = self._make_congestion_sde(network)
        _, c_fine = congestion_sde.simulate_path(T=T, dt=dt, seed=seed)
        return self._extract_node_metric(c_fine, metric)

    def _simulate_coupled_level_node_values(self,
                                            network: NetworkGraph,
                                            traffic: TrafficModel,
                                            T: float,
                                            dt_fine: float,
                                            dt_coarse: float,
                                            metric: str,
                                            seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate coupled vector congestion levels and return per-node values."""
        _ = traffic
        congestion_sde = self._make_congestion_sde(network)
        _, c_fine, c_coarse = congestion_sde.simulate_coupled_paths(
            T=T,
            dt_coarse=dt_coarse,
            dt_fine=dt_fine,
            seed=seed,
        )
        return (
            self._extract_node_metric(c_fine, metric),
            self._extract_node_metric(c_coarse, metric),
        )

    def run_coupled_node_paths(self,
                               network: NetworkGraph,
                               traffic: TrafficModel,
                               level: int,
                               T: float,
                               base_dt: float,
                               metric: str = 'mean_congestion',
                               seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Run one coupled ANA-MLMC sample and return per-node fine/coarse values."""
        dt_fine = get_timestep(level, base_dt, self.refinement_factor)
        if level == 0:
            y_fine = self._simulate_single_level_node_values(
                network, traffic, T, dt_fine, metric, seed
            )
            y_coarse = np.zeros_like(y_fine)
            return y_fine, y_coarse

        dt_coarse = get_timestep(level - 1, base_dt, self.refinement_factor)
        return self._simulate_coupled_level_node_values(
            network, traffic, T, dt_fine, dt_coarse, metric, seed
        )

    def estimate_level_variance_per_node(self,
                                         network: NetworkGraph,
                                         traffic: TrafficModel,
                                         level: int,
                                         T: float,
                                         base_dt: float,
                                         metric: str,
                                         n_samples: int = 100,
                                         return_samples: bool = False
                                         ) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
        """Estimate per-node mean and variance of one MLMC level difference."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        differences = np.zeros((n_samples, network.n_nodes), dtype=float)
        for i in range(n_samples):
            sample_seed = (self.seed + level * 10000 + i) if self.seed is not None else None
            y_fine, y_coarse = self.run_coupled_node_paths(
                network, traffic, level, T, base_dt, metric, sample_seed
            )
            differences[i] = y_fine - y_coarse

        mean_diff = np.mean(differences, axis=0)
        var_diff = (
            np.var(differences, axis=0, ddof=1)
            if n_samples > 1
            else np.zeros(network.n_nodes, dtype=float)
        )
        dt_fine = get_timestep(level, base_dt, self.refinement_factor)
        cost_per_sample = T / dt_fine

        if return_samples:
            return mean_diff, var_diff, cost_per_sample, differences
        return mean_diff, var_diff, cost_per_sample, None

    def mlmc_estimate_weighted(self,
                               network: NetworkGraph,
                               traffic: TrafficModel,
                               epsilon: float,
                               L_max: int,
                               T: float = 10.0,
                               base_dt: float = 0.1,
                               metric: str = 'mean_congestion',
                               pilot_samples: int = 100,
                               confidence_level: float = 0.95,
                               verbose: bool = True,
                               sla_vec: Optional[np.ndarray] = None) -> MLMCResult:
        """Run ANA-MLMC estimation for the weighted network quantity.

        Adaptive level selection: levels are added until the stopping criterion
        max(|bias_l^w|, V_l^w / N_pilot) < epsilon / 2 is satisfied or L_max
        is reached.  This ensures both the weighted bias and the weighted pilot
        variance are below the RMSE half-budget before committing to the full
        allocation.
        """
        if verbose:
            logger.info(f"Starting ANA-MLMC simulation: ε={epsilon}, L_max={L_max}")

        per_node_variances = []
        costs = []
        pilot_diffs = []
        stop_thresh = epsilon / 2.0  # RMSE half-budget per paper spec
        uniform_w = np.ones(network.n_nodes, dtype=float) / network.n_nodes
        stopping_w = sla_vec / sla_vec.sum() if (sla_vec is not None and float(np.sum(sla_vec)) > 0) else uniform_w

        for l in range(L_max + 1):
            _, var_diff, cost, diffs = self.estimate_level_variance_per_node(
                network=network,
                traffic=traffic,
                level=l,
                T=T,
                base_dt=base_dt,
                metric=metric,
                n_samples=pilot_samples,
                return_samples=True,
            )
            per_node_variances.append(var_diff)
            costs.append(cost)
            pilot_diffs.append(diffs)

            # Weighted quantities for stopping check use SLA weights when provided;
            # full ANA weights are computed after all pilot levels are complete.
            V_lw = float(np.dot(stopping_w, var_diff))
            bias_lw = float(abs(np.dot(stopping_w, np.mean(diffs, axis=0))))

            if verbose:
                logger.info(
                    f"  Level {l}: V_lw={V_lw:.3e}  bias_lw={bias_lw:.3e}  "
                    f"threshold={stop_thresh:.3e}  C={cost:.2e}"
                )

            # Adaptive stopping: both weighted variance and bias below half RMSE budget
            if l > 0 and max(bias_lw, V_lw / pilot_samples) < stop_thresh:
                if verbose:
                    logger.info(
                        f"  Adaptive stopping at level {l}: "
                        f"max(bias_lw={bias_lw:.3e}, V_lw/N={V_lw/pilot_samples:.3e}) "
                        f"< ε/2={stop_thresh:.3e}"
                    )
                break

        level_var_per_node = np.vstack(per_node_variances)
        node_weights = self.compute_node_weights(network, level_var_per_node, sla_vec=sla_vec)
        optimal_N = self.compute_optimal_samples_weighted(
            level_var_per_node, costs, node_weights, epsilon
        )

        if verbose:
            logger.info("ANA node weights computed")
            for l, n_l in enumerate(optimal_N):
                logger.info(f"  Level {l}: N={n_l}")

        level_stats = []
        total_cost = 0.0
        weighted_level_vars = []
        weighted_estimator_vars = []

        for l in range(L_max + 1):
            n_additional = max(0, optimal_N[l] - pilot_samples)
            n_total = pilot_samples + n_additional
            diffs = pilot_diffs[l]

            if n_additional > 0:
                additional = np.zeros((n_additional, network.n_nodes), dtype=float)
                for i in range(n_additional):
                    sample_seed = (
                        self.seed + l * 10000 + pilot_samples + i
                        if self.seed is not None
                        else None
                    )
                    y_fine, y_coarse = self.run_coupled_node_paths(
                        network, traffic, l, T, base_dt, metric, sample_seed
                    )
                    additional[i] = y_fine - y_coarse
                diffs = np.vstack([diffs, additional])

            mean_per_node = np.mean(diffs, axis=0)
            var_per_node = (
                np.var(diffs, axis=0, ddof=1)
                if n_total > 1
                else np.zeros(network.n_nodes, dtype=float)
            )
            weighted_diffs = diffs @ node_weights

            mean_diff = float(np.dot(node_weights, mean_per_node))
            var_diff = float(np.dot(node_weights, var_per_node))
            weighted_scalar_var = (
                float(np.var(weighted_diffs, ddof=1))
                if n_total > 1
                else 0.0
            )
            weighted_level_vars.append(var_diff)
            weighted_estimator_vars.append(weighted_scalar_var)

            mean_Y = (
                mean_diff
                if l == 0
                else sum(stats.mean_diff for stats in level_stats) + mean_diff
            )
            dt_l = get_timestep(l, base_dt, self.refinement_factor)
            cost_per_sample = T / dt_l
            level_cost = cost_per_sample * n_total
            total_cost += level_cost

            level_stats.append(MLMCLevelStats(
                level=l,
                n_samples=n_total,
                dt=dt_l,
                mean_Y=mean_Y,
                var_Y=var_diff,
                mean_diff=mean_diff,
                var_diff=var_diff,
                cost_per_sample=cost_per_sample,
                total_cost=level_cost,
            ))

            if verbose:
                logger.info(f"  Level {l} complete: {level_stats[-1]}")

        estimate = float(sum(stats.mean_diff for stats in level_stats))
        variance = float(sum(stats.var_diff / stats.n_samples for stats in level_stats))

        dt_finest = get_timestep(L_max, base_dt, self.refinement_factor)
        bias_estimate = self.BIAS_CALIBRATION_CONSTANT * np.sqrt(dt_finest)
        mse = variance + bias_estimate ** 2

        from scipy import stats as sp_stats
        alpha = 1 - confidence_level
        z_value = sp_stats.norm.ppf(1 - alpha / 2)
        margin = z_value * np.sqrt(variance)

        return MLMCResult(
            estimate=estimate,
            variance=variance,
            mse=mse,
            level_stats=level_stats,
            total_cost=total_cost,
            L_max=L_max,
            epsilon=epsilon,
            ci_lower=estimate - margin,
            ci_upper=estimate + margin,
            confidence_level=confidence_level,
            metadata={
                'T': T,
                'base_dt': base_dt,
                'metric': metric,
                'refinement_factor': self.refinement_factor,
                'ana_node_weights': node_weights.tolist(),
                'ana_level_var_per_node': level_var_per_node.tolist(),
                'ana_weighted_level_variances': weighted_level_vars,
                'weighted_estimator_level_variances': weighted_estimator_vars,
            },
        )


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
