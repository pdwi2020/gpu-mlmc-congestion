"""
Delay Metrics Module

Compute end-to-end delay metrics for network simulations:
- End-to-end delay using Little's Law
- Path-based delay computation
- Delay distribution estimation from Monte Carlo samples
- Confidence intervals and percentiles
- Statistical analysis of delay measurements

Mathematical Background:
- Little's Law: W = Q/μ (average delay = average queue length / service rate)
- Path delay: D_path = Σ_{i∈path} (D_prop,i + D_queue,i + D_trans,i)
  where:
    D_prop = propagation delay
    D_queue = queueing delay
    D_trans = transmission delay (packet_size / bandwidth)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from scipy import stats
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DelayMetrics:
    """Container for delay measurement results.

    Attributes:
        mean_delay: Average end-to-end delay
        median_delay: Median delay (p50)
        std_delay: Standard deviation
        min_delay: Minimum observed delay
        max_delay: Maximum observed delay
        percentiles: Dictionary of percentile values (p50, p90, p95, p99)
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        confidence_level: Confidence level used (e.g., 0.95)
        n_samples: Number of samples used
        samples: Raw delay samples (optional)
    """
    mean_delay: float
    median_delay: float
    std_delay: float
    min_delay: float
    max_delay: float
    percentiles: Dict[str, float]
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_samples: int
    samples: Optional[np.ndarray] = None

    def summary(self) -> Dict:
        """Return summary dictionary."""
        return {
            'mean': self.mean_delay,
            'median': self.median_delay,
            'std': self.std_delay,
            'min': self.min_delay,
            'max': self.max_delay,
            'percentiles': self.percentiles,
            'confidence_interval': (self.ci_lower, self.ci_upper),
            'confidence_level': self.confidence_level,
            'n_samples': self.n_samples
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"DelayMetrics(\n"
            f"  Mean: {self.mean_delay:.4f}\n"
            f"  Median: {self.median_delay:.4f}\n"
            f"  Std: {self.std_delay:.4f}\n"
            f"  Range: [{self.min_delay:.4f}, {self.max_delay:.4f}]\n"
            f"  P95: {self.percentiles.get('p95', np.nan):.4f}\n"
            f"  P99: {self.percentiles.get('p99', np.nan):.4f}\n"
            f"  {self.confidence_level*100:.0f}% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"  N={self.n_samples}\n"
            f")"
        )


class DelayCalculator:
    """Calculate end-to-end delay metrics for network simulations.

    This class provides methods to compute various delay metrics including
    queueing delay, propagation delay, transmission delay, and total
    end-to-end delay across network paths.

    Args:
        network: NetworkGraph instance
        confidence_level: Confidence level for intervals (default: 0.95)
    """

    def __init__(self, network, confidence_level: float = 0.95):
        """Initialize delay calculator.

        Args:
            network: NetworkGraph instance with topology and link properties
            confidence_level: Confidence level for CI (0 < α < 1)
        """
        self.network = network
        self.confidence_level = confidence_level

        # Cache shortest paths for efficiency
        self._path_cache = {}

        logger.info(f"Initialized DelayCalculator for network with {network.n_nodes} nodes")

    def compute_queueing_delay(
        self,
        queue_length: float,
        service_rate: float
    ) -> float:
        """Compute queueing delay using Little's Law.

        Little's Law: W = Q/μ
        where:
            W = average time in system (delay)
            Q = average number in system (queue length)
            μ = service rate

        Args:
            queue_length: Average queue length
            service_rate: Service rate (packets/second)

        Returns:
            Average queueing delay in seconds
        """
        if service_rate <= 0:
            raise ValueError(f"Service rate must be positive, got {service_rate}")

        delay = queue_length / service_rate
        return max(0.0, delay)  # Ensure non-negative

    def compute_propagation_delay(
        self,
        distance: float,
        propagation_speed: float = 2e8  # 2/3 speed of light in fiber
    ) -> float:
        """Compute propagation delay.

        D_prop = distance / propagation_speed

        Args:
            distance: Physical distance in meters
            propagation_speed: Signal propagation speed (default: 2e8 m/s for fiber)

        Returns:
            Propagation delay in seconds
        """
        return distance / propagation_speed

    def compute_transmission_delay(
        self,
        packet_size: float,
        bandwidth: float
    ) -> float:
        """Compute transmission delay.

        D_trans = packet_size / bandwidth

        Args:
            packet_size: Packet size in bits
            bandwidth: Link bandwidth in bits/second

        Returns:
            Transmission delay in seconds
        """
        if bandwidth <= 0:
            raise ValueError(f"Bandwidth must be positive, got {bandwidth}")

        return packet_size / bandwidth

    def compute_link_delay(
        self,
        u: int,
        v: int,
        queue_length: float = 0.0,
        packet_size: float = 1500 * 8  # 1500 bytes = 12000 bits
    ) -> float:
        """Compute total delay for a single link.

        Total delay = D_queue + D_prop + D_trans

        Args:
            u: Source node
            v: Destination node
            queue_length: Current queue length at node u
            packet_size: Packet size in bits (default: 1500 bytes)

        Returns:
            Total link delay in seconds
        """
        # Get link properties
        edge_data = self.network.graph.get_edge_data(u, v)
        if edge_data is None:
            raise ValueError(f"No edge exists between {u} and {v}")

        # Extract properties with defaults
        bandwidth = edge_data.get('bandwidth', 1e9)  # 1 Gbps default
        propagation_delay = edge_data.get('delay', 1e-3)  # 1 ms default
        capacity = edge_data.get('capacity', 1000)  # packets/sec

        # Compute components
        d_queue = self.compute_queueing_delay(queue_length, capacity)
        d_prop = propagation_delay  # Already stored
        d_trans = self.compute_transmission_delay(packet_size, bandwidth)

        total_delay = d_queue + d_prop + d_trans
        return total_delay

    def compute_path_delay(
        self,
        path: List[int],
        queue_states: Optional[Dict[int, float]] = None,
        packet_size: float = 1500 * 8
    ) -> float:
        """Compute end-to-end delay along a path.

        D_path = Σ_{i∈path} D_link,i

        Args:
            path: List of node IDs forming the path
            queue_states: Dictionary mapping node IDs to queue lengths
            packet_size: Packet size in bits

        Returns:
            Total path delay in seconds
        """
        if len(path) < 2:
            return 0.0

        if queue_states is None:
            queue_states = {}

        total_delay = 0.0

        # Sum delay for each link in path
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            queue_length = queue_states.get(u, 0.0)

            link_delay = self.compute_link_delay(u, v, queue_length, packet_size)
            total_delay += link_delay

        return total_delay

    def compute_end_to_end_delay(
        self,
        source: int,
        destination: int,
        queue_states: Optional[Dict[int, float]] = None,
        packet_size: float = 1500 * 8
    ) -> float:
        """Compute end-to-end delay between source and destination.

        Uses shortest path and sums link delays.

        Args:
            source: Source node ID
            destination: Destination node ID
            queue_states: Current queue lengths at each node
            packet_size: Packet size in bits

        Returns:
            End-to-end delay in seconds
        """
        # Get shortest path (cached)
        cache_key = (source, destination)
        if cache_key not in self._path_cache:
            path = self.network.compute_shortest_path(source, destination)
            self._path_cache[cache_key] = path
        else:
            path = self._path_cache[cache_key]

        if path is None:
            logger.warning(f"No path exists between {source} and {destination}")
            return np.inf

        return self.compute_path_delay(path, queue_states, packet_size)

    def estimate_delay_distribution(
        self,
        delay_samples: np.ndarray,
        compute_percentiles: List[float] = [50, 90, 95, 99]
    ) -> DelayMetrics:
        """Estimate delay distribution from Monte Carlo samples.

        Computes statistics and confidence intervals from delay samples.

        Args:
            delay_samples: Array of delay measurements (shape: [n_samples])
            compute_percentiles: List of percentile values to compute

        Returns:
            DelayMetrics object with statistics
        """
        if len(delay_samples) == 0:
            raise ValueError("delay_samples cannot be empty")

        # Filter out infinite values
        valid_samples = delay_samples[np.isfinite(delay_samples)]
        n_valid = len(valid_samples)

        if n_valid == 0:
            raise ValueError("All samples are infinite (no valid paths)")

        if n_valid < len(delay_samples):
            logger.warning(
                f"{len(delay_samples) - n_valid} samples were infinite and excluded"
            )

        # Compute statistics
        mean_delay = np.mean(valid_samples)
        median_delay = np.median(valid_samples)
        std_delay = np.std(valid_samples, ddof=1) if n_valid > 1 else 0.0
        min_delay = np.min(valid_samples)
        max_delay = np.max(valid_samples)

        # Compute percentiles
        percentiles = {}
        for p in compute_percentiles:
            percentiles[f'p{int(p)}'] = np.percentile(valid_samples, p)

        # Compute confidence interval
        ci_lower, ci_upper = self.compute_confidence_interval(
            valid_samples,
            self.confidence_level
        )

        return DelayMetrics(
            mean_delay=mean_delay,
            median_delay=median_delay,
            std_delay=std_delay,
            min_delay=min_delay,
            max_delay=max_delay,
            percentiles=percentiles,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            n_samples=n_valid,
            samples=valid_samples
        )

    def compute_confidence_interval(
        self,
        samples: np.ndarray,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Compute confidence interval for delay samples.

        Uses t-distribution for small samples (<30) and normal for large samples.

        Args:
            samples: Array of delay measurements
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(samples)
        if n < 2:
            return (np.mean(samples), np.mean(samples))

        mean = np.mean(samples)
        sem = stats.sem(samples)  # Standard error of mean

        # Use t-distribution for small samples
        if n < 30:
            t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n - 1)
            margin = t_critical * sem
        else:
            # Use normal distribution for large samples
            z_critical = stats.norm.ppf((1 + confidence_level) / 2)
            margin = z_critical * sem

        ci_lower = mean - margin
        ci_upper = mean + margin

        return (ci_lower, ci_upper)

    def compute_all_pairs_delay(
        self,
        queue_states: Optional[Dict[int, float]] = None,
        packet_size: float = 1500 * 8,
        sample_size: Optional[int] = None
    ) -> Dict[Tuple[int, int], float]:
        """Compute delay for all pairs of nodes.

        Args:
            queue_states: Current queue lengths
            packet_size: Packet size in bits
            sample_size: If provided, only compute for random sample of pairs

        Returns:
            Dictionary mapping (source, dest) to delay
        """
        nodes = list(self.network.graph.nodes())
        n = len(nodes)

        delays = {}

        # Generate pairs
        if sample_size is not None and sample_size < n * (n - 1):
            # Random sample
            rng = np.random.default_rng()
            pairs = []
            for _ in range(sample_size):
                src, dst = rng.choice(nodes, size=2, replace=False)
                pairs.append((src, dst))
        else:
            # All pairs
            pairs = [(src, dst) for src in nodes for dst in nodes if src != dst]

        for src, dst in pairs:
            try:
                delay = self.compute_end_to_end_delay(
                    src, dst, queue_states, packet_size
                )
                delays[(src, dst)] = delay
            except Exception as e:
                logger.warning(f"Failed to compute delay for {src}->{dst}: {e}")
                delays[(src, dst)] = np.inf

        return delays

    def analyze_delay_from_simulation(
        self,
        simulation_result,
        source: int,
        destination: int,
        packet_size: float = 1500 * 8
    ) -> DelayMetrics:
        """Analyze delay from simulation results.

        Extracts delay across all Monte Carlo samples and computes statistics.

        Args:
            simulation_result: NetworkSimulationResult or MLMCResult object
            source: Source node ID
            destination: Destination node ID
            packet_size: Packet size in bits

        Returns:
            DelayMetrics with distribution analysis
        """
        # Check if result has samples
        if not hasattr(simulation_result, 'samples') or simulation_result.samples is None:
            raise ValueError("Simulation result must contain raw samples")

        # If result has queue_samples attribute (time series)
        if hasattr(simulation_result, 'queue_samples'):
            # queue_samples shape: [n_samples, n_timesteps, n_nodes]
            queue_samples = simulation_result.queue_samples
            n_samples, n_timesteps, n_nodes = queue_samples.shape

            # Use final timestep queue states
            delays = []
            for i in range(n_samples):
                queue_states = {node: queue_samples[i, -1, node] for node in range(n_nodes)}
                delay = self.compute_end_to_end_delay(
                    source, destination, queue_states, packet_size
                )
                delays.append(delay)

            delays = np.array(delays)

        else:
            # Assume samples are scalar delay measurements
            delays = simulation_result.samples

        return self.estimate_delay_distribution(delays)

    def compare_delay_distributions(
        self,
        samples1: np.ndarray,
        samples2: np.ndarray,
        label1: str = "Distribution 1",
        label2: str = "Distribution 2"
    ) -> Dict:
        """Compare two delay distributions statistically.

        Performs statistical tests to determine if distributions differ significantly.

        Args:
            samples1: First set of delay samples
            samples2: Second set of delay samples
            label1: Label for first distribution
            label2: Label for second distribution

        Returns:
            Dictionary with comparison results
        """
        # Compute metrics for each
        metrics1 = self.estimate_delay_distribution(samples1)
        metrics2 = self.estimate_delay_distribution(samples2)

        # Perform t-test (means different?)
        t_stat, t_pvalue = stats.ttest_ind(samples1, samples2)

        # Perform Mann-Whitney U test (distributions different?)
        u_stat, u_pvalue = stats.mannwhitneyu(samples1, samples2, alternative='two-sided')

        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(samples1) - 1) * metrics1.std_delay**2 +
             (len(samples2) - 1) * metrics2.std_delay**2) /
            (len(samples1) + len(samples2) - 2)
        )
        cohens_d = (metrics1.mean_delay - metrics2.mean_delay) / pooled_std if pooled_std > 0 else 0.0

        return {
            'label1': label1,
            'label2': label2,
            'metrics1': metrics1.summary(),
            'metrics2': metrics2.summary(),
            'mean_difference': metrics1.mean_delay - metrics2.mean_delay,
            'mean_difference_pct': 100 * (metrics1.mean_delay - metrics2.mean_delay) / metrics2.mean_delay,
            't_test': {
                'statistic': t_stat,
                'pvalue': t_pvalue,
                'significant': t_pvalue < 0.05
            },
            'mann_whitney': {
                'statistic': u_stat,
                'pvalue': u_pvalue,
                'significant': u_pvalue < 0.05
            },
            'effect_size': {
                'cohens_d': cohens_d,
                'interpretation': self._interpret_cohens_d(cohens_d)
            }
        }

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def export_delay_samples(
        self,
        delay_samples: np.ndarray,
        filepath: Union[str, Path],
        metadata: Optional[Dict] = None
    ):
        """Export delay samples to file for further analysis.

        Args:
            delay_samples: Array of delay measurements
            filepath: Output file path (.npz format)
            metadata: Optional metadata dictionary
        """
        filepath = Path(filepath)

        # Compute metrics
        metrics = self.estimate_delay_distribution(delay_samples)

        # Save with metadata
        np.savez(
            filepath,
            samples=delay_samples,
            mean=metrics.mean_delay,
            median=metrics.median_delay,
            std=metrics.std_delay,
            percentiles=metrics.percentiles,
            ci_lower=metrics.ci_lower,
            ci_upper=metrics.ci_upper,
            confidence_level=metrics.confidence_level,
            n_samples=metrics.n_samples,
            metadata=metadata if metadata is not None else {}
        )

        logger.info(f"Exported {len(delay_samples)} delay samples to {filepath}")


def compute_delay_variance_reduction(
    mc_samples: np.ndarray,
    mlmc_samples: np.ndarray
) -> Dict:
    """Compute variance reduction achieved by MLMC over standard MC.

    Args:
        mc_samples: Standard MC delay samples
        mlmc_samples: MLMC delay samples

    Returns:
        Dictionary with variance reduction metrics
    """
    mc_var = np.var(mc_samples, ddof=1)
    mlmc_var = np.var(mlmc_samples, ddof=1)

    var_reduction_factor = mc_var / mlmc_var if mlmc_var > 0 else np.inf
    var_reduction_pct = 100 * (1 - mlmc_var / mc_var) if mc_var > 0 else 0.0

    return {
        'mc_variance': mc_var,
        'mlmc_variance': mlmc_var,
        'reduction_factor': var_reduction_factor,
        'reduction_percent': var_reduction_pct,
        'mc_std': np.sqrt(mc_var),
        'mlmc_std': np.sqrt(mlmc_var)
    }
