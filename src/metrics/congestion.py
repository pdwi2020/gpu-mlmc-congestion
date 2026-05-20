"""
Congestion Metrics Module

Compute congestion-related metrics for network simulations:
- Queue length statistics
- Link utilization (ρ = λ/μ)
- Congestion event detection and analysis
- Congestion propagation across the network
- Bottleneck identification

Mathematical Background:
- Utilization: ρ = λ/μ (arrival rate / service rate)
- Stability condition: ρ < 1 for stable queues
- Heavy load: ρ > 0.8 typically indicates high congestion risk
- Congestion propagation: C_i influences C_j through routing
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass
import logging
from collections import defaultdict, deque
from scipy import stats
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CongestionMetrics:
    """Container for congestion measurement results.

    Attributes:
        mean_queue_length: Average queue length across network
        max_queue_length: Maximum queue length observed
        mean_utilization: Average link utilization
        max_utilization: Maximum link utilization
        congested_nodes: Set of congested node IDs
        congested_links: Set of congested link tuples
        congestion_events: List of detected congestion events
        propagation_metrics: Metrics about congestion spread
        n_samples: Number of samples used
        threshold: Congestion threshold used
    """
    mean_queue_length: float
    max_queue_length: float
    mean_utilization: float
    max_utilization: float
    congested_nodes: Set[int]
    congested_links: Set[Tuple[int, int]]
    congestion_events: List[Dict]
    propagation_metrics: Dict
    n_samples: int
    threshold: float

    def summary(self) -> Dict:
        """Return summary dictionary."""
        return {
            'mean_queue_length': self.mean_queue_length,
            'max_queue_length': self.max_queue_length,
            'mean_utilization': self.mean_utilization,
            'max_utilization': self.max_utilization,
            'n_congested_nodes': len(self.congested_nodes),
            'n_congested_links': len(self.congested_links),
            'n_events': len(self.congestion_events),
            'propagation': self.propagation_metrics,
            'threshold': self.threshold
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"CongestionMetrics(\n"
            f"  Mean Queue: {self.mean_queue_length:.2f}\n"
            f"  Max Queue: {self.max_queue_length:.2f}\n"
            f"  Mean Utilization: {self.mean_utilization:.3f}\n"
            f"  Max Utilization: {self.max_utilization:.3f}\n"
            f"  Congested Nodes: {len(self.congested_nodes)}\n"
            f"  Congested Links: {len(self.congested_links)}\n"
            f"  Events: {len(self.congestion_events)}\n"
            f"  Threshold: {self.threshold}\n"
            f")"
        )


@dataclass
class CongestionEvent:
    """Represents a single congestion event.

    Attributes:
        node_id: Node where congestion occurred
        start_time: Event start time
        end_time: Event end time (None if ongoing)
        duration: Event duration
        peak_queue: Peak queue length during event
        severity: Severity level (0-1 scale)
    """
    node_id: int
    start_time: float
    end_time: Optional[float]
    duration: float
    peak_queue: float
    severity: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'peak_queue': self.peak_queue,
            'severity': self.severity
        }


class CongestionAnalyzer:
    """Analyze congestion patterns in network simulations.

    This class provides methods to detect, quantify, and analyze congestion
    across network nodes and links, including temporal evolution and spatial
    propagation patterns.

    Args:
        network: NetworkGraph instance
        congestion_threshold: Utilization threshold for congestion (default: 0.8)
        queue_threshold: Queue length threshold for congestion detection
    """

    def __init__(
        self,
        network,
        congestion_threshold: float = 0.8,
        queue_threshold: Optional[float] = None
    ):
        """Initialize congestion analyzer.

        Args:
            network: NetworkGraph instance
            congestion_threshold: Utilization threshold (ρ > threshold = congested)
            queue_threshold: Absolute queue length threshold (optional)
        """
        self.network = network
        self.congestion_threshold = congestion_threshold
        self.queue_threshold = queue_threshold

        logger.info(
            f"Initialized CongestionAnalyzer with threshold ρ > {congestion_threshold}"
        )

    def compute_queue_lengths(
        self,
        simulation_states: np.ndarray
    ) -> np.ndarray:
        """Extract queue lengths from simulation states.

        Args:
            simulation_states: Array of simulation states
                Shape: [n_samples, n_timesteps, n_nodes] or [n_timesteps, n_nodes]

        Returns:
            Queue length array (same shape as input)
        """
        # States typically contain queue lengths directly
        # May need to extract from more complex state representations
        return np.maximum(0.0, simulation_states)  # Ensure non-negative

    def compute_link_utilization(
        self,
        arrival_rates: Union[float, np.ndarray],
        service_rates: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Compute link utilization: ρ = λ/μ.

        Args:
            arrival_rates: Arrival rate(s) λ
            service_rates: Service rate(s) μ

        Returns:
            Utilization array ρ
        """
        arrival_rates = np.asarray(arrival_rates)
        service_rates = np.asarray(service_rates)

        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            utilization = arrival_rates / service_rates
            utilization = np.where(service_rates > 0, utilization, 0.0)

        return utilization

    def detect_congestion_events(
        self,
        queue_time_series: np.ndarray,
        times: np.ndarray,
        node_id: int,
        threshold: Optional[float] = None
    ) -> List[CongestionEvent]:
        """Detect congestion events in queue time series.

        An event starts when queue exceeds threshold and ends when it drops below.

        Args:
            queue_time_series: Queue length over time (shape: [n_timesteps])
            times: Time points corresponding to queue values
            node_id: Node ID for event labeling
            threshold: Queue threshold (uses self.queue_threshold if None)

        Returns:
            List of CongestionEvent objects
        """
        if threshold is None:
            if self.queue_threshold is None:
                # Use mean + 2*std as threshold
                threshold = np.mean(queue_time_series) + 2 * np.std(queue_time_series)
            else:
                threshold = self.queue_threshold

        events = []
        in_event = False
        event_start_time = None
        event_start_idx = None
        event_peak = 0.0

        for i, (t, q) in enumerate(zip(times, queue_time_series)):
            if q > threshold and not in_event:
                # Event starts
                in_event = True
                event_start_time = t
                event_start_idx = i
                event_peak = q

            elif q > threshold and in_event:
                # Event continues, update peak
                event_peak = max(event_peak, q)

            elif q <= threshold and in_event:
                # Event ends
                in_event = False
                event_end_time = t
                duration = event_end_time - event_start_time

                # Compute severity (normalized peak)
                severity = min(1.0, event_peak / (2 * threshold))

                event = CongestionEvent(
                    node_id=node_id,
                    start_time=event_start_time,
                    end_time=event_end_time,
                    duration=duration,
                    peak_queue=event_peak,
                    severity=severity
                )
                events.append(event)

                event_peak = 0.0

        # Handle ongoing event at end
        if in_event:
            event = CongestionEvent(
                node_id=node_id,
                start_time=event_start_time,
                end_time=None,
                duration=times[-1] - event_start_time,
                peak_queue=event_peak,
                severity=min(1.0, event_peak / (2 * threshold))
            )
            events.append(event)

        return events

    def identify_congested_nodes(
        self,
        utilization: Dict[int, float],
        threshold: Optional[float] = None
    ) -> Set[int]:
        """Identify nodes with utilization above threshold.

        Args:
            utilization: Dictionary mapping node IDs to utilization values
            threshold: Congestion threshold (uses self.congestion_threshold if None)

        Returns:
            Set of congested node IDs
        """
        if threshold is None:
            threshold = self.congestion_threshold

        congested = {
            node_id for node_id, util in utilization.items()
            if util > threshold
        }

        return congested

    def identify_congested_links(
        self,
        link_utilization: Dict[Tuple[int, int], float],
        threshold: Optional[float] = None
    ) -> Set[Tuple[int, int]]:
        """Identify links with utilization above threshold.

        Args:
            link_utilization: Dictionary mapping (u,v) to utilization
            threshold: Congestion threshold

        Returns:
            Set of congested link tuples
        """
        if threshold is None:
            threshold = self.congestion_threshold

        congested = {
            (u, v) for (u, v), util in link_utilization.items()
            if util > threshold
        }

        return congested

    def measure_congestion_spread(
        self,
        congested_nodes: Set[int],
        reference_time: Optional[float] = None
    ) -> Dict:
        """Measure how congestion spreads across the network.

        Analyzes the spatial distribution and connectivity of congested nodes.

        Args:
            congested_nodes: Set of congested node IDs
            reference_time: Time point for this measurement (optional)

        Returns:
            Dictionary with propagation metrics
        """
        if len(congested_nodes) == 0:
            return {
                'n_congested': 0,
                'congestion_fraction': 0.0,
                'largest_cluster_size': 0,
                'n_clusters': 0,
                'avg_cluster_size': 0.0,
                'reference_time': reference_time
            }

        n_total = self.network.n_nodes
        n_congested = len(congested_nodes)

        # Find clusters of congested nodes (connected components)
        clusters = self._find_congested_clusters(congested_nodes)

        cluster_sizes = [len(cluster) for cluster in clusters]
        largest_cluster = max(cluster_sizes) if cluster_sizes else 0
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0.0

        return {
            'n_congested': n_congested,
            'congestion_fraction': n_congested / n_total,
            'largest_cluster_size': largest_cluster,
            'n_clusters': len(clusters),
            'avg_cluster_size': avg_cluster_size,
            'clusters': clusters,
            'reference_time': reference_time
        }

    def _find_congested_clusters(
        self,
        congested_nodes: Set[int]
    ) -> List[Set[int]]:
        """Find clusters of connected congested nodes.

        Uses BFS to identify connected components in the subgraph of congested nodes.

        Args:
            congested_nodes: Set of congested node IDs

        Returns:
            List of clusters (each cluster is a set of node IDs)
        """
        unvisited = set(congested_nodes)
        clusters = []

        while unvisited:
            # Start new cluster with arbitrary unvisited node
            start_node = next(iter(unvisited))
            cluster = set()

            # BFS to find all connected congested nodes
            queue = deque([start_node])
            cluster.add(start_node)
            unvisited.remove(start_node)

            while queue:
                current = queue.popleft()

                # Check neighbors
                neighbors = self.network.graph.neighbors(current)
                for neighbor in neighbors:
                    if neighbor in unvisited:
                        cluster.add(neighbor)
                        unvisited.remove(neighbor)
                        queue.append(neighbor)

            clusters.append(cluster)

        return clusters

    def analyze_simulation_congestion(
        self,
        queue_states: np.ndarray,
        arrival_rates: Optional[Union[float, np.ndarray]] = None,
        service_rates: Optional[Union[float, np.ndarray]] = None,
        times: Optional[np.ndarray] = None
    ) -> CongestionMetrics:
        """Comprehensive congestion analysis from simulation results.

        Args:
            queue_states: Queue length time series
                Shape: [n_samples, n_timesteps, n_nodes] or [n_timesteps, n_nodes]
            arrival_rates: Arrival rates (scalar or array)
            service_rates: Service rates (scalar or array)
            times: Time points (optional)

        Returns:
            CongestionMetrics object
        """
        # Handle different input shapes
        if queue_states.ndim == 3:
            # Multiple samples: average over samples
            mean_queue_over_samples = np.mean(queue_states, axis=0)  # [n_timesteps, n_nodes]
            max_queue_over_samples = np.max(queue_states, axis=0)
            n_samples = queue_states.shape[0]
        elif queue_states.ndim == 2:
            mean_queue_over_samples = queue_states
            max_queue_over_samples = queue_states
            n_samples = 1
        else:
            raise ValueError(f"queue_states must be 2D or 3D, got shape {queue_states.shape}")

        n_timesteps, n_nodes = mean_queue_over_samples.shape

        if times is None:
            times = np.arange(n_timesteps)

        # Compute queue statistics
        mean_queue_length = np.mean(mean_queue_over_samples)
        max_queue_length = np.max(max_queue_over_samples)

        # Compute utilization if rates provided
        if arrival_rates is not None and service_rates is not None:
            utilization = self.compute_link_utilization(arrival_rates, service_rates)

            if np.isscalar(utilization):
                mean_utilization = utilization
                max_utilization = utilization
                node_utilization = {i: utilization for i in range(n_nodes)}
            else:
                mean_utilization = np.mean(utilization)
                max_utilization = np.max(utilization)
                if np.ndim(utilization) == 0:
                    node_utilization = {i: float(utilization) for i in range(n_nodes)}
                else:
                    node_utilization = {i: utilization[i] for i in range(len(utilization))}

            # Identify congested nodes
            congested_nodes = self.identify_congested_nodes(node_utilization)
        else:
            mean_utilization = np.nan
            max_utilization = np.nan
            congested_nodes = set()

        # Detect events for each node
        all_events = []
        for node_id in range(n_nodes):
            node_queue_series = mean_queue_over_samples[:, node_id]
            events = self.detect_congestion_events(
                node_queue_series, times, node_id
            )
            all_events.extend(events)

        # Measure propagation
        propagation_metrics = self.measure_congestion_spread(congested_nodes)

        # Placeholder for link congestion (requires traffic matrix)
        congested_links = set()

        return CongestionMetrics(
            mean_queue_length=mean_queue_length,
            max_queue_length=max_queue_length,
            mean_utilization=mean_utilization,
            max_utilization=max_utilization,
            congested_nodes=congested_nodes,
            congested_links=congested_links,
            congestion_events=[e.to_dict() for e in all_events],
            propagation_metrics=propagation_metrics,
            n_samples=n_samples,
            threshold=self.congestion_threshold
        )

    def compute_temporal_congestion_evolution(
        self,
        queue_states: np.ndarray,
        times: np.ndarray,
        window_size: int = 10
    ) -> Dict:
        """Analyze how congestion evolves over time.

        Args:
            queue_states: Queue states over time [n_timesteps, n_nodes]
            times: Time points
            window_size: Moving average window size

        Returns:
            Dictionary with temporal metrics
        """
        n_timesteps, n_nodes = queue_states.shape

        # Compute network-wide congestion over time
        total_queue_over_time = np.sum(queue_states, axis=1)  # [n_timesteps]
        mean_queue_over_time = np.mean(queue_states, axis=1)
        max_queue_over_time = np.max(queue_states, axis=1)

        # Compute moving averages
        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'valid') / w

        if window_size > 1 and n_timesteps >= window_size:
            smoothed_mean = moving_average(mean_queue_over_time, window_size)
            smoothed_max = moving_average(max_queue_over_time, window_size)
            smoothed_times = times[window_size-1:]
        else:
            smoothed_mean = mean_queue_over_time
            smoothed_max = max_queue_over_time
            smoothed_times = times

        # Detect trend (increasing/decreasing congestion)
        if len(smoothed_mean) > 2:
            # Linear regression on smoothed data
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                smoothed_times, smoothed_mean
            )
            trend = 'increasing' if slope > 0 else 'decreasing'
        else:
            slope = 0.0
            trend = 'stable'

        return {
            'times': times,
            'total_queue': total_queue_over_time,
            'mean_queue': mean_queue_over_time,
            'max_queue': max_queue_over_time,
            'smoothed_times': smoothed_times,
            'smoothed_mean': smoothed_mean,
            'smoothed_max': smoothed_max,
            'trend': trend,
            'trend_slope': slope,
            'window_size': window_size
        }

    def identify_bottlenecks(
        self,
        queue_states: np.ndarray,
        percentile: float = 95
    ) -> List[Tuple[int, float]]:
        """Identify network bottlenecks based on queue lengths.

        Args:
            queue_states: Queue states [n_timesteps, n_nodes] or [n_samples, n_timesteps, n_nodes]
            percentile: Percentile for bottleneck threshold (default: 95th)

        Returns:
            List of (node_id, avg_queue_length) sorted by severity
        """
        # Average over time and samples
        if queue_states.ndim == 3:
            avg_queues = np.mean(queue_states, axis=(0, 1))  # [n_nodes]
        else:
            avg_queues = np.mean(queue_states, axis=0)  # [n_nodes]

        # Find threshold
        threshold = np.percentile(avg_queues, percentile)

        # Identify bottlenecks
        bottlenecks = [
            (node_id, avg_queue)
            for node_id, avg_queue in enumerate(avg_queues)
            if avg_queue >= threshold
        ]

        # Sort by severity (descending)
        bottlenecks.sort(key=lambda x: x[1], reverse=True)

        return bottlenecks

    def export_congestion_heatmap(
        self,
        queue_states: np.ndarray,
        times: np.ndarray,
        filepath: Union[str, Path]
    ):
        """Export congestion heatmap data.

        Args:
            queue_states: Queue states [n_timesteps, n_nodes]
            times: Time points
            filepath: Output file path (.npz format)
        """
        filepath = Path(filepath)

        # Compute statistics
        mean_over_time = np.mean(queue_states, axis=1)
        max_over_time = np.max(queue_states, axis=1)
        mean_over_nodes = np.mean(queue_states, axis=0)

        np.savez(
            filepath,
            queue_states=queue_states,
            times=times,
            mean_over_time=mean_over_time,
            max_over_time=max_over_time,
            mean_over_nodes=mean_over_nodes
        )

        logger.info(f"Exported congestion heatmap to {filepath}")


def compute_congestion_probability(
    queue_samples: np.ndarray,
    threshold: float
) -> float:
    """Compute probability of congestion (queue > threshold).

    Args:
        queue_samples: Queue length samples
        threshold: Congestion threshold

    Returns:
        Probability estimate (fraction of samples exceeding threshold)
    """
    congested_fraction = np.mean(queue_samples > threshold)
    return congested_fraction


def compare_congestion_scenarios(
    metrics1: CongestionMetrics,
    metrics2: CongestionMetrics,
    label1: str = "Scenario 1",
    label2: str = "Scenario 2"
) -> Dict:
    """Compare congestion between two scenarios.

    Args:
        metrics1: First scenario metrics
        metrics2: Second scenario metrics
        label1: Label for first scenario
        label2: Label for second scenario

    Returns:
        Comparison dictionary
    """
    return {
        'label1': label1,
        'label2': label2,
        'mean_queue_diff': metrics1.mean_queue_length - metrics2.mean_queue_length,
        'mean_queue_pct_change': 100 * (
            metrics1.mean_queue_length - metrics2.mean_queue_length
        ) / metrics2.mean_queue_length if metrics2.mean_queue_length > 0 else np.inf,
        'max_queue_diff': metrics1.max_queue_length - metrics2.max_queue_length,
        'utilization_diff': metrics1.mean_utilization - metrics2.mean_utilization,
        'congested_nodes_diff': len(metrics1.congested_nodes) - len(metrics2.congested_nodes),
        'events_diff': len(metrics1.congestion_events) - len(metrics2.congestion_events)
    }
