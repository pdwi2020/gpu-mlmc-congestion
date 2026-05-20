"""
Traffic Modeling Module

This module provides classes for generating and modeling network traffic
with various stochastic characteristics.

Classes:
    TrafficModel: Abstract base class for traffic models
    PoissonTraffic: Poisson arrival process
    BurstyTraffic: On-Off bursty traffic model
    MAWIBasedTraffic: Traffic model based on MAWI trace statistics
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, List, Union
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class TrafficModel(ABC):
    """
    Abstract base class for traffic models.

    All traffic models should implement methods for generating
    arrival times, packet sizes, and computing traffic matrices.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize traffic model.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    @abstractmethod
    def generate_arrivals(self, duration: float) -> np.ndarray:
        """
        Generate packet arrival times.

        Args:
            duration: Simulation duration

        Returns:
            Array of arrival times
        """
        pass

    @abstractmethod
    def generate_packet_sizes(self, n_packets: int) -> np.ndarray:
        """
        Generate packet sizes.

        Args:
            n_packets: Number of packets

        Returns:
            Array of packet sizes (in bytes)
        """
        pass

    def get_arrival_rate(self, duration: float) -> float:
        """
        Estimate average arrival rate.

        Args:
            duration: Time window for estimation

        Returns:
            Average arrival rate (packets/time unit)
        """
        arrivals = self.generate_arrivals(duration)
        return len(arrivals) / duration

    def get_statistics(self, duration: float = 100.0) -> Dict:
        """
        Compute traffic statistics.

        Args:
            duration: Sampling duration

        Returns:
            Dictionary with statistics (rate, variance, burstiness, etc.)
        """
        arrivals = self.generate_arrivals(duration)
        n_arrivals = len(arrivals)

        # Compute inter-arrival times
        if n_arrivals > 1:
            inter_arrivals = np.diff(arrivals)
            mean_inter_arrival = np.mean(inter_arrivals)
            std_inter_arrival = np.std(inter_arrivals)
            cv = std_inter_arrival / mean_inter_arrival if mean_inter_arrival > 0 else 0
        else:
            mean_inter_arrival = duration
            std_inter_arrival = 0
            cv = 0

        # Packet sizes
        sizes = self.generate_packet_sizes(n_arrivals)

        stats = {
            'arrival_rate': n_arrivals / duration,
            'n_arrivals': n_arrivals,
            'mean_inter_arrival': mean_inter_arrival,
            'std_inter_arrival': std_inter_arrival,
            'coefficient_of_variation': cv,
            'mean_packet_size': np.mean(sizes),
            'std_packet_size': np.std(sizes),
        }

        return stats


class PoissonTraffic(TrafficModel):
    """
    Poisson arrival process traffic model.

    Arrivals follow a Poisson process with constant rate λ.
    Inter-arrival times are exponentially distributed.

    This is the simplest traffic model, commonly used in
    classical queueing theory (M/M/1, M/M/c queues).
    """

    def __init__(self,
                 rate: float,
                 mean_packet_size: float = 1500.0,
                 packet_size_std: float = 200.0,
                 seed: Optional[int] = None):
        """
        Initialize Poisson traffic model.

        Args:
            rate: Arrival rate λ (packets/time unit)
            mean_packet_size: Mean packet size (bytes)
            packet_size_std: Standard deviation of packet size
            seed: Random seed
        """
        super().__init__(seed)
        self.rate = rate
        self.mean_packet_size = mean_packet_size
        self.packet_size_std = packet_size_std

    def generate_arrivals(self, duration: float) -> np.ndarray:
        """
        Generate Poisson arrival times.

        Args:
            duration: Simulation duration

        Returns:
            Array of arrival times
        """
        if self.rate == 0.0:
            return np.array([], dtype=np.float64)

        # Expected number of arrivals
        expected_arrivals = int(self.rate * duration * 1.2)  # Add buffer

        # Generate exponential inter-arrival times
        inter_arrivals = np.random.exponential(1.0 / self.rate, expected_arrivals)

        # Cumulative sum gives arrival times
        arrivals = np.cumsum(inter_arrivals)

        # Keep only arrivals within duration
        arrivals = arrivals[arrivals <= duration]

        return arrivals

    def generate_packet_sizes(self, n_packets: int) -> np.ndarray:
        """
        Generate packet sizes from normal distribution.

        Args:
            n_packets: Number of packets

        Returns:
            Array of packet sizes (bytes)
        """
        sizes = np.random.normal(self.mean_packet_size, self.packet_size_std, n_packets)

        # Ensure positive sizes
        sizes = np.maximum(sizes, 64.0)  # Minimum Ethernet frame size

        return sizes

    def __repr__(self) -> str:
        return f"PoissonTraffic(rate={self.rate}, mean_size={self.mean_packet_size})"


class BurstyTraffic(TrafficModel):
    """
    Bursty On-Off traffic model.

    Traffic alternates between ON and OFF states.
    - ON state: Packets arrive at high rate
    - OFF state: No packets arrive

    This models bursty Internet traffic more realistically
    than Poisson processes.
    """

    def __init__(self,
                 on_rate: float,
                 mean_on_duration: float,
                 mean_off_duration: float,
                 mean_packet_size: float = 1500.0,
                 packet_size_std: float = 200.0,
                 seed: Optional[int] = None):
        """
        Initialize bursty traffic model.

        Args:
            on_rate: Packet arrival rate during ON state
            mean_on_duration: Mean duration of ON periods
            mean_off_duration: Mean duration of OFF periods
            mean_packet_size: Mean packet size (bytes)
            packet_size_std: Std dev of packet size
            seed: Random seed
        """
        super().__init__(seed)
        self.on_rate = on_rate
        self.mean_on_duration = mean_on_duration
        self.mean_off_duration = mean_off_duration
        self.mean_packet_size = mean_packet_size
        self.packet_size_std = packet_size_std

        # Effective average rate
        cycle_duration = mean_on_duration + mean_off_duration
        on_fraction = mean_on_duration / cycle_duration
        self.effective_rate = on_rate * on_fraction

    def generate_arrivals(self, duration: float) -> np.ndarray:
        """
        Generate bursty arrival times.

        Args:
            duration: Simulation duration

        Returns:
            Array of arrival times
        """
        arrivals = []
        t = 0.0

        while t < duration:
            # Generate ON period duration
            on_duration = np.random.exponential(self.mean_on_duration)

            # Generate arrivals during ON period
            on_end = min(t + on_duration, duration)

            # Poisson arrivals during ON
            n_arrivals = np.random.poisson(self.on_rate * on_duration)
            on_arrivals = np.sort(np.random.uniform(t, on_end, n_arrivals))
            arrivals.extend(on_arrivals)

            t = on_end

            # Generate OFF period duration
            off_duration = np.random.exponential(self.mean_off_duration)
            t += off_duration

        return np.array(arrivals)

    def generate_packet_sizes(self, n_packets: int) -> np.ndarray:
        """
        Generate packet sizes.

        Args:
            n_packets: Number of packets

        Returns:
            Array of packet sizes
        """
        sizes = np.random.normal(self.mean_packet_size, self.packet_size_std, n_packets)
        sizes = np.maximum(sizes, 64.0)
        return sizes

    def __repr__(self) -> str:
        return (f"BurstyTraffic(on_rate={self.on_rate}, "
                f"mean_on={self.mean_on_duration}, mean_off={self.mean_off_duration})")


class MAWIBasedTraffic(TrafficModel):
    """
    Traffic model based on MAWI trace statistics.

    Extracts statistical parameters from MAWI traffic traces
    and generates synthetic traffic matching those characteristics.

    Note: This class provides parameter fitting from traces.
    Actual PCAP parsing requires scapy (optional dependency).
    """

    def __init__(self,
                 arrival_rate: float,
                 burstiness: float,
                 mean_packet_size: float,
                 packet_size_std: float,
                 seed: Optional[int] = None):
        """
        Initialize MAWI-based traffic model.

        Args:
            arrival_rate: Mean arrival rate (packets/time unit)
            burstiness: Burstiness coefficient (CV of inter-arrivals)
            mean_packet_size: Mean packet size (bytes)
            packet_size_std: Std dev of packet size
            seed: Random seed
        """
        super().__init__(seed)
        self.arrival_rate = arrival_rate
        self.burstiness = burstiness
        self.mean_packet_size = mean_packet_size
        self.packet_size_std = packet_size_std

        # Configure underlying model based on burstiness
        if burstiness > 1.5:
            # High burstiness: use bursty On-Off model
            self._use_bursty_model = True
            # Calibrate On-Off parameters
            self._calibrate_onoff_model()
        else:
            # Low burstiness: use Poisson model
            self._use_bursty_model = False

    def _calibrate_onoff_model(self):
        """
        Calibrate On-Off model parameters from burstiness.

        Uses moment matching to fit On-Off parameters.
        """
        # Simple heuristic calibration
        # For more accurate fitting, use MAWI trace analysis
        self.on_rate = self.arrival_rate * 2.0
        self.mean_on_duration = 0.5  # seconds
        self.mean_off_duration = self.mean_on_duration * (self.on_rate / self.arrival_rate - 1)

        logger.info(f"Calibrated On-Off model: on_rate={self.on_rate:.2f}, "
                   f"on_dur={self.mean_on_duration:.3f}, off_dur={self.mean_off_duration:.3f}")

    def generate_arrivals(self, duration: float) -> np.ndarray:
        """
        Generate arrivals matching MAWI statistics.

        Args:
            duration: Simulation duration

        Returns:
            Array of arrival times
        """
        if self._use_bursty_model:
            # Use bursty On-Off model
            arrivals = []
            t = 0.0

            while t < duration:
                on_duration = np.random.exponential(self.mean_on_duration)
                on_end = min(t + on_duration, duration)

                n_arrivals = np.random.poisson(self.on_rate * on_duration)
                on_arrivals = np.sort(np.random.uniform(t, on_end, n_arrivals))
                arrivals.extend(on_arrivals)

                t = on_end
                off_duration = np.random.exponential(self.mean_off_duration)
                t += off_duration

            return np.array(arrivals)
        else:
            # Use Poisson model
            expected_arrivals = int(self.arrival_rate * duration * 1.2)
            inter_arrivals = np.random.exponential(1.0 / self.arrival_rate, expected_arrivals)
            arrivals = np.cumsum(inter_arrivals)
            arrivals = arrivals[arrivals <= duration]
            return arrivals

    def generate_packet_sizes(self, n_packets: int) -> np.ndarray:
        """
        Generate packet sizes matching MAWI distribution.

        Args:
            n_packets: Number of packets

        Returns:
            Array of packet sizes
        """
        # Use log-normal distribution for more realistic size distribution
        # (Internet packet sizes are often log-normally distributed)
        mu = np.log(self.mean_packet_size)
        sigma = self.packet_size_std / self.mean_packet_size

        sizes = np.random.lognormal(mu, sigma, n_packets)
        sizes = np.clip(sizes, 64.0, 9000.0)  # MTU constraints

        return sizes

    @staticmethod
    def from_pcap(pcap_path: Union[str, Path],
                  seed: Optional[int] = None) -> 'MAWIBasedTraffic':
        """
        Extract traffic parameters from PCAP file.

        Requires scapy package (optional dependency).

        Args:
            pcap_path: Path to PCAP file
            seed: Random seed

        Returns:
            MAWIBasedTraffic instance

        Raises:
            ImportError: If scapy is not installed
        """
        try:
            from scapy.all import rdpcap, IP
        except ImportError:
            raise ImportError(
                "scapy is required for PCAP processing. "
                "Install with: pip install scapy"
            )

        pcap_path = Path(pcap_path)
        if not pcap_path.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

        logger.info(f"Extracting traffic statistics from {pcap_path}")

        # Read packets
        packets = rdpcap(str(pcap_path))

        # Extract timestamps and sizes
        timestamps = []
        sizes = []

        for pkt in packets:
            if IP in pkt:
                timestamps.append(float(pkt.time))
                sizes.append(len(pkt))

        timestamps = np.array(timestamps)
        sizes = np.array(sizes)

        # Compute statistics
        duration = timestamps[-1] - timestamps[0]
        arrival_rate = len(timestamps) / duration

        # Inter-arrival times
        inter_arrivals = np.diff(timestamps)
        mean_inter = np.mean(inter_arrivals)
        std_inter = np.std(inter_arrivals)
        burstiness = std_inter / mean_inter if mean_inter > 0 else 1.0

        # Packet sizes
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)

        logger.info(f"Extracted statistics: rate={arrival_rate:.2f} pkt/s, "
                   f"burstiness={burstiness:.2f}, mean_size={mean_size:.0f} bytes")

        return MAWIBasedTraffic(
            arrival_rate=arrival_rate,
            burstiness=burstiness,
            mean_packet_size=mean_size,
            packet_size_std=std_size,
            seed=seed
        )

    def __repr__(self) -> str:
        return (f"MAWIBasedTraffic(rate={self.arrival_rate}, "
                f"burstiness={self.burstiness:.2f}, "
                f"mean_size={self.mean_packet_size:.0f})")


def create_traffic_matrix(network_graph,
                         traffic_model: TrafficModel,
                         n_flows: int,
                         duration: float,
                         seed: Optional[int] = None) -> Dict:
    """
    Generate traffic matrix for network simulation.

    Creates source-destination traffic flows across the network.

    Args:
        network_graph: NetworkGraph object
        traffic_model: TrafficModel instance
        n_flows: Number of flows to generate
        duration: Simulation duration
        seed: Random seed

    Returns:
        Dictionary with flow information:
            - 'sources': List of source node IDs
            - 'destinations': List of destination node IDs
            - 'arrivals': List of arrival time arrays (one per flow)
            - 'sizes': List of packet size arrays (one per flow)
    """
    if seed is not None:
        np.random.seed(seed)

    nodes = list(network_graph.nodes)
    n_nodes = len(nodes)

    if n_nodes < 2:
        raise ValueError("Network must have at least 2 nodes for traffic matrix")

    sources = []
    destinations = []
    arrivals = []
    sizes = []

    for _ in range(n_flows):
        # Random source and destination
        src, dst = np.random.choice(nodes, size=2, replace=False)

        # Generate traffic for this flow
        flow_arrivals = traffic_model.generate_arrivals(duration)
        flow_sizes = traffic_model.generate_packet_sizes(len(flow_arrivals))

        sources.append(src)
        destinations.append(dst)
        arrivals.append(flow_arrivals)
        sizes.append(flow_sizes)

    traffic_matrix = {
        'sources': sources,
        'destinations': destinations,
        'arrivals': arrivals,
        'sizes': sizes,
        'n_flows': n_flows,
        'duration': duration
    }

    logger.info(f"Created traffic matrix: {n_flows} flows over {duration} time units")

    return traffic_matrix


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Traffic Modeling Module - Example Usage")
    print("=" * 60)

    # Example 1: Poisson Traffic
    print("\n1. Poisson Traffic Model")
    print("-" * 60)

    poisson = PoissonTraffic(rate=10.0, seed=42)
    arrivals = poisson.generate_arrivals(duration=100.0)
    sizes = poisson.generate_packet_sizes(len(arrivals))

    stats = poisson.get_statistics(duration=100.0)
    print(f"Arrival rate: {stats['arrival_rate']:.2f} packets/time unit")
    print(f"Mean inter-arrival: {stats['mean_inter_arrival']:.4f}")
    print(f"Coefficient of variation: {stats['coefficient_of_variation']:.4f}")
    print(f"Mean packet size: {stats['mean_packet_size']:.0f} bytes")

    # Example 2: Bursty Traffic
    print("\n2. Bursty Traffic Model")
    print("-" * 60)

    bursty = BurstyTraffic(
        on_rate=50.0,
        mean_on_duration=0.5,
        mean_off_duration=0.5,
        seed=42
    )

    stats = bursty.get_statistics(duration=100.0)
    print(f"Arrival rate: {stats['arrival_rate']:.2f} packets/time unit")
    print(f"Coefficient of variation: {stats['coefficient_of_variation']:.4f}")
    print(f"Effective rate: {bursty.effective_rate:.2f} packets/time unit")

    # Example 3: MAWI-Based Traffic
    print("\n3. MAWI-Based Traffic Model")
    print("-" * 60)

    mawi = MAWIBasedTraffic(
        arrival_rate=100.0,
        burstiness=2.5,
        mean_packet_size=800.0,
        packet_size_std=400.0,
        seed=42
    )

    stats = mawi.get_statistics(duration=100.0)
    print(f"Arrival rate: {stats['arrival_rate']:.2f} packets/time unit")
    print(f"Coefficient of variation: {stats['coefficient_of_variation']:.4f}")
    print(f"Mean packet size: {stats['mean_packet_size']:.0f} bytes")

    print("\n" + "=" * 60)
