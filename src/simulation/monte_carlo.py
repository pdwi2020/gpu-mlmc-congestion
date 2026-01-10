"""
Monte Carlo Simulation Module

Standard Monte Carlo estimator for network performance metrics
with uncertainty quantification.

Classes:
    MonteCarloSimulator: Single-level Monte Carlo simulation
    NetworkSimulationResult: Container for simulation results
"""

import numpy as np
from typing import Dict, Optional, Callable, List, Tuple
import logging
from dataclasses import dataclass, field
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from network.topology import NetworkGraph
from network.traffic import TrafficModel
from network.sde import QueueDynamicsSDE, CongestionPropagationSDE


logger = logging.getLogger(__name__)


@dataclass
class NetworkSimulationResult:
    """
    Container for network simulation results.

    Attributes:
        samples: Array of metric values from all sample paths (N,)
        mean: Sample mean
        variance: Sample variance
        std: Standard deviation
        ci_lower: Lower confidence interval bound
        ci_upper: Upper confidence interval bound
        confidence_level: Confidence level (e.g., 0.95)
        n_samples: Number of samples
        computational_cost: Cost measure (e.g., number of timesteps)
    """
    samples: np.ndarray
    mean: float
    variance: float
    std: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_samples: int
    computational_cost: float
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"MonteCarloResult(N={self.n_samples}, "
            f"mean={self.mean:.4f}±{self.std:.4f}, "
            f"CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}])"
        )

    def __repr__(self) -> str:
        return self.summary()


class MonteCarloSimulator:
    """
    Standard Monte Carlo simulator for network performance estimation.

    Generates independent sample paths and computes statistics.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize Monte Carlo simulator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def run_single_path(self,
                       network: NetworkGraph,
                       traffic: TrafficModel,
                       T: float,
                       dt: float,
                       metric: str = 'mean_queue',
                       seed: Optional[int] = None) -> float:
        """
        Run single sample path simulation.

        Args:
            network: NetworkGraph object
            traffic: TrafficModel object
            T: Simulation duration
            dt: Time step
            metric: Metric to compute ('mean_queue', 'max_queue', 'delay', etc.)
            seed: Random seed for this path

        Returns:
            Metric value for this sample path
        """
        if seed is not None:
            np.random.seed(seed)

        # For simplicity, simulate queue at a single representative link
        # In full implementation, this would simulate entire network

        # Get traffic statistics
        traffic_stats = traffic.get_statistics(duration=T)
        arrival_rate = traffic_stats['arrival_rate']

        # Assume service rate based on network capacity
        # In practice, this would come from network.get_link_properties()
        service_rate = arrival_rate * 1.25  # 25% overprovisioned

        # Create queue SDE
        queue_sde = QueueDynamicsSDE(
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            noise_intensity=0.2
        )

        # Simulate path
        time, queue_length = queue_sde.simulate_path(T=T, dt=dt, q0=0.0, seed=seed)

        # Compute requested metric
        if metric == 'mean_queue':
            return np.mean(queue_length)
        elif metric == 'max_queue':
            return np.max(queue_length)
        elif metric == 'final_queue':
            return queue_length[-1]
        elif metric == 'time_average_queue':
            # Time-weighted average
            return np.trapz(queue_length, time) / T
        elif metric == 'delay':
            # Approximate delay using Little's Law: E[W] = E[Q] / λ
            mean_queue = np.mean(queue_length)
            return mean_queue / arrival_rate if arrival_rate > 0 else 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def estimate(self,
                network: NetworkGraph,
                traffic: TrafficModel,
                n_samples: int,
                T: float,
                dt: float,
                metric: str = 'mean_queue',
                confidence_level: float = 0.95,
                verbose: bool = True) -> NetworkSimulationResult:
        """
        Run Monte Carlo estimation with multiple sample paths.

        Args:
            network: NetworkGraph object
            traffic: TrafficModel object
            n_samples: Number of independent sample paths
            T: Simulation duration
            dt: Time step
            metric: Metric to estimate
            confidence_level: Confidence level for CI (default: 0.95)
            verbose: Print progress

        Returns:
            NetworkSimulationResult object
        """
        if verbose:
            logger.info(f"Starting Monte Carlo simulation: N={n_samples}, T={T}, dt={dt}")

        samples = np.zeros(n_samples)

        # Generate independent samples
        for i in range(n_samples):
            # Use different seed for each sample
            sample_seed = (self.seed + i) if self.seed is not None else None
            samples[i] = self.run_single_path(
                network=network,
                traffic=traffic,
                T=T,
                dt=dt,
                metric=metric,
                seed=sample_seed
            )

            if verbose and (i + 1) % max(1, n_samples // 10) == 0:
                logger.info(f"  Completed {i + 1}/{n_samples} samples")

        # Compute statistics
        mean = np.mean(samples)
        variance = np.var(samples, ddof=1)  # Unbiased variance
        std = np.sqrt(variance)

        # Confidence interval
        ci_lower, ci_upper = self.compute_confidence_interval(
            samples, confidence_level
        )

        # Computational cost: total number of timesteps
        n_timesteps = int(T / dt)
        computational_cost = n_samples * n_timesteps

        result = NetworkSimulationResult(
            samples=samples,
            mean=mean,
            variance=variance,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            n_samples=n_samples,
            computational_cost=computational_cost,
            metadata={
                'T': T,
                'dt': dt,
                'metric': metric,
                'n_timesteps': n_timesteps
            }
        )

        if verbose:
            logger.info(f"Monte Carlo complete: {result.summary()}")

        return result

    def compute_confidence_interval(self,
                                   samples: np.ndarray,
                                   confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Compute confidence interval using Central Limit Theorem.

        Args:
            samples: Sample values (N,)
            confidence_level: Confidence level (default: 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        from scipy import stats

        n = len(samples)
        mean = np.mean(samples)
        std_error = np.std(samples, ddof=1) / np.sqrt(n)

        # Use t-distribution for finite samples
        alpha = 1 - confidence_level
        t_value = stats.t.ppf(1 - alpha / 2, df=n - 1)

        ci_lower = mean - t_value * std_error
        ci_upper = mean + t_value * std_error

        return ci_lower, ci_upper

    def estimate_mse(self,
                    true_value: float,
                    network: NetworkGraph,
                    traffic: TrafficModel,
                    n_samples: int,
                    T: float,
                    dt: float,
                    metric: str = 'mean_queue',
                    n_trials: int = 100) -> Dict:
        """
        Estimate Mean Squared Error (MSE) of the Monte Carlo estimator.

        Useful for validating convergence rates.

        Args:
            true_value: True value of the metric (if known)
            network: NetworkGraph object
            traffic: TrafficModel object
            n_samples: Number of samples per trial
            T: Simulation duration
            dt: Time step
            metric: Metric to estimate
            n_trials: Number of independent trials

        Returns:
            Dictionary with MSE statistics
        """
        estimates = np.zeros(n_trials)

        logger.info(f"Estimating MSE: {n_trials} trials with N={n_samples} each")

        for trial in range(n_trials):
            result = self.estimate(
                network=network,
                traffic=traffic,
                n_samples=n_samples,
                T=T,
                dt=dt,
                metric=metric,
                verbose=False
            )
            estimates[trial] = result.mean

        # Compute MSE components
        bias = np.mean(estimates) - true_value
        variance = np.var(estimates, ddof=1)
        mse = bias ** 2 + variance

        mse_stats = {
            'mse': mse,
            'bias': bias,
            'variance': variance,
            'rmse': np.sqrt(mse),
            'estimates': estimates,
            'true_value': true_value
        }

        logger.info(f"MSE Analysis: MSE={mse:.6e}, bias={bias:.6e}, var={variance:.6e}")

        return mse_stats

    def convergence_test(self,
                        network: NetworkGraph,
                        traffic: TrafficModel,
                        sample_sizes: List[int],
                        T: float,
                        dt: float,
                        metric: str = 'mean_queue',
                        n_trials: int = 10) -> Dict:
        """
        Test convergence of Monte Carlo estimator as N increases.

        Args:
            network: NetworkGraph object
            traffic: TrafficModel object
            sample_sizes: List of sample sizes to test
            T: Simulation duration
            dt: Time step
            metric: Metric to estimate
            n_trials: Number of trials per sample size

        Returns:
            Dictionary with convergence statistics
        """
        results = {
            'sample_sizes': sample_sizes,
            'means': [],
            'stds': [],
            'ci_widths': [],
            'costs': []
        }

        for n_samples in sample_sizes:
            trial_means = []
            trial_stds = []
            trial_ci_widths = []

            for trial in range(n_trials):
                result = self.estimate(
                    network=network,
                    traffic=traffic,
                    n_samples=n_samples,
                    T=T,
                    dt=dt,
                    metric=metric,
                    verbose=False
                )
                trial_means.append(result.mean)
                trial_stds.append(result.std)
                trial_ci_widths.append(result.ci_upper - result.ci_lower)

            results['means'].append(np.mean(trial_means))
            results['stds'].append(np.mean(trial_stds))
            results['ci_widths'].append(np.mean(trial_ci_widths))
            results['costs'].append(result.computational_cost)

            logger.info(f"N={n_samples}: mean={results['means'][-1]:.4f}, "
                       f"std={results['stds'][-1]:.4f}, "
                       f"CI width={results['ci_widths'][-1]:.4f}")

        return results


class NetworkMetricsCalculator:
    """
    Calculate various network performance metrics from simulation results.
    """

    @staticmethod
    def end_to_end_delay(queue_states: np.ndarray,
                        arrival_rate: float,
                        network: NetworkGraph,
                        path: List[int]) -> float:
        """
        Compute end-to-end delay along a path.

        Uses Little's Law at each hop: delay = queue_length / service_rate

        Args:
            queue_states: Queue lengths at each node (n_nodes,)
            arrival_rate: Traffic arrival rate
            network: NetworkGraph object
            path: List of node IDs forming the path

        Returns:
            End-to-end delay
        """
        total_delay = 0.0

        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i + 1]

            # Queue delay at this hop
            queue_length = queue_states[src] if src < len(queue_states) else 0.0

            # Service rate (from link capacity)
            edge_attrs = network.get_edge_attributes(src, dst)
            service_rate = edge_attrs.get('service_rate', arrival_rate * 1.5)

            # Little's Law: W = Q / μ
            queue_delay = queue_length / service_rate if service_rate > 0 else 0.0

            # Propagation delay
            prop_delay = edge_attrs.get('delay', 1.0)

            total_delay += queue_delay + prop_delay

        return total_delay

    @staticmethod
    def link_utilization(arrival_rate: float,
                        service_rate: float) -> float:
        """
        Compute link utilization: ρ = λ / μ

        Args:
            arrival_rate: Packet arrival rate
            service_rate: Link service rate

        Returns:
            Utilization (0 to 1+)
        """
        return arrival_rate / service_rate if service_rate > 0 else float('inf')

    @staticmethod
    def packet_loss_probability(queue_length: float,
                                buffer_size: float) -> float:
        """
        Estimate packet loss probability.

        Args:
            queue_length: Average queue length
            buffer_size: Buffer capacity

        Returns:
            Loss probability (0 to 1)
        """
        if buffer_size <= 0:
            return 1.0

        # Simple approximation: loss when queue exceeds buffer
        utilization = queue_length / buffer_size
        return max(0.0, min(1.0, utilization - 1.0))


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Monte Carlo Simulation Module - Example Usage")
    print("=" * 60)

    # Create example network
    from network.topology import TopologyGenerator
    from network.traffic import PoissonTraffic

    gen = TopologyGenerator(seed=42)
    network = gen.generate_erdos_renyi(n_nodes=50, p=0.1)
    network.set_link_properties(seed=42)

    print(f"\nNetwork: {network}")

    # Create traffic model
    traffic = PoissonTraffic(rate=10.0, seed=42)
    print(f"Traffic: {traffic}")

    # Run Monte Carlo estimation
    print("\n1. Single Monte Carlo Estimation")
    print("-" * 60)

    simulator = MonteCarloSimulator(seed=42)
    result = simulator.estimate(
        network=network,
        traffic=traffic,
        n_samples=100,
        T=10.0,
        dt=0.1,
        metric='mean_queue'
    )

    print(f"\nResult: {result.summary()}")
    print(f"Mean queue length: {result.mean:.4f} ± {result.std:.4f}")
    print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print(f"Computational cost: {result.computational_cost:.2e} timesteps")

    # Test convergence
    print("\n2. Convergence Test")
    print("-" * 60)

    convergence = simulator.convergence_test(
        network=network,
        traffic=traffic,
        sample_sizes=[10, 50, 100, 500, 1000],
        T=10.0,
        dt=0.1,
        n_trials=5
    )

    print("\nConvergence results:")
    for i, n in enumerate(convergence['sample_sizes']):
        print(f"  N={n:4d}: mean={convergence['means'][i]:.4f}, "
              f"CI width={convergence['ci_widths'][i]:.4f}")

    print("\n" + "=" * 60)
