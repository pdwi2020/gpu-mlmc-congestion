"""
Shared pytest fixtures for GPU-Accelerated MLMC Network Modeling tests.

This module provides common fixtures used across multiple test files to avoid
duplication and ensure consistent test setup.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from network.topology import TopologyGenerator, NetworkGraph
from network.traffic import PoissonTraffic


@pytest.fixture
def simple_network():
    """Create a simple line graph network for basic tests.

    Returns:
        NetworkGraph: A 5-node line graph with uniform link properties
    """
    generator = TopologyGenerator(seed=42)
    network = generator.generate_line_graph(n_nodes=5)

    # Set uniform link properties
    network.set_link_properties(
        bandwidth_range=(1e9, 1e9),      # 1 Gbps
        delay_range=(0.001, 0.001),      # 1 ms
        capacity_range=(1000, 1000),     # 1000 pkt/s
        seed=42
    )

    return network


@pytest.fixture
def erdos_renyi_network():
    """Create an Erdos-Renyi random network for simulation tests.

    Returns:
        NetworkGraph: A 20-node random graph with p=0.2
    """
    generator = TopologyGenerator(seed=42)
    network = generator.generate_erdos_renyi(n_nodes=20, p=0.2)
    network.set_link_properties(seed=42)
    return network


@pytest.fixture
def poisson_traffic():
    """Create a Poisson traffic model for tests.

    Returns:
        PoissonTraffic: Traffic model with rate=5.0
    """
    return PoissonTraffic(rate=5.0, seed=42)


@pytest.fixture
def setup_network_traffic(erdos_renyi_network, poisson_traffic):
    """Combined fixture providing network and traffic for simulation tests.

    Returns:
        Tuple[NetworkGraph, PoissonTraffic]: Network and traffic model
    """
    return erdos_renyi_network, poisson_traffic


# Pytest markers configuration
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "network: marks tests requiring network access"
    )
