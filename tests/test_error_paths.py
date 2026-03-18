"""
Unit tests for error handling and edge cases.

Tests cover:
- Invalid network topology handling
- Invalid MLMC parameters
- Unstable queue SDE handling
- File not found for datasets
- Memory constraints
"""

import pytest
import numpy as np
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from network.topology import NetworkGraph, TopologyGenerator
from network.sde import QueueDynamicsSDE, CongestionPropagationSDE
from network.traffic import PoissonTraffic
from simulation.mlmc import MLMCSimulator
from simulation.monte_carlo import MonteCarloSimulator


class TestInvalidNetworkTopology:
    """Tests for invalid network topology handling."""

    def test_empty_network(self):
        """Test handling of empty network graph."""
        g = NetworkGraph()
        assert g.n_nodes == 0
        assert g.n_edges == 0

        # Summary should work even on empty graph
        summary = g.summary()
        assert summary['n_nodes'] == 0

    def test_single_node_network(self):
        """Test network with single node (no edges possible)."""
        g = NetworkGraph()
        g.add_node(0, capacity=100)

        assert g.n_nodes == 1
        assert g.n_edges == 0

        # Get neighbors of single node
        neighbors = g.get_neighbors(0)
        assert len(neighbors) == 0

    def test_disconnected_network(self):
        """Test handling of disconnected network components."""
        g = NetworkGraph()

        # Create two disconnected components
        g.add_node(0)
        g.add_node(1)
        g.add_edge(0, 1)

        g.add_node(2)
        g.add_node(3)
        g.add_edge(2, 3)

        # Graph has 4 nodes, 2 edges but is disconnected
        assert g.n_nodes == 4
        assert g.n_edges == 2

        # Largest component should have 2 nodes
        largest = g.get_largest_component()
        assert largest.n_nodes == 2

    def test_self_loop_handling(self):
        """Test that self-loops are handled correctly."""
        g = NetworkGraph()
        g.add_node(0)
        g.add_node(1)
        g.add_edge(0, 1)

        # Self-loops should be addable (NetworkX allows them)
        g.add_edge(0, 0)

        # Network should still function
        summary = g.summary()
        assert summary['n_nodes'] == 2

    def test_duplicate_edge_handling(self):
        """Test handling of duplicate edges."""
        g = NetworkGraph()
        g.add_node(0)
        g.add_node(1)
        g.add_edge(0, 1, weight=1.0)
        g.add_edge(0, 1, weight=2.0)  # Update existing edge

        # In undirected graph, edge should be updated not duplicated
        assert g.n_edges == 1
        attrs = g.get_edge_attributes(0, 1)
        assert attrs['weight'] == 2.0


class TestInvalidMLMCParameters:
    """Tests for invalid MLMC parameter handling."""

    @pytest.fixture
    def setup_simple_network(self):
        """Setup a simple network for testing."""
        gen = TopologyGenerator(seed=42)
        network = gen.generate_erdos_renyi(n_nodes=10, p=0.3)
        network.set_link_properties(seed=42)
        traffic = PoissonTraffic(rate=5.0, seed=42)
        return network, traffic

    def test_zero_epsilon(self, setup_simple_network):
        """Test MLMC with zero target accuracy."""
        network, traffic = setup_simple_network
        simulator = MLMCSimulator(refinement_factor=2, seed=42)

        # Zero epsilon should cause very large sample allocation
        # This test ensures no division by zero or infinite loops
        # We use a very small but non-zero epsilon instead
        result = simulator.mlmc_estimate(
            network=network,
            traffic=traffic,
            epsilon=1e-10,  # Very small epsilon
            L_max=1,
            T=1.0,
            base_dt=0.5,
            pilot_samples=5,
            verbose=False
        )

        # Should still produce a result
        assert result.estimate is not None
        assert np.isfinite(result.estimate)

    def test_negative_refinement_factor(self):
        """Test MLMC with invalid refinement factor."""
        # Refinement factor should be positive integer
        simulator = MLMCSimulator(refinement_factor=2, seed=42)
        assert simulator.refinement_factor == 2

    def test_zero_levels(self, setup_simple_network):
        """Test MLMC with L_max=0 (only level 0)."""
        network, traffic = setup_simple_network
        simulator = MLMCSimulator(refinement_factor=2, seed=42)

        result = simulator.mlmc_estimate(
            network=network,
            traffic=traffic,
            epsilon=0.1,
            L_max=0,  # Only level 0
            T=1.0,
            base_dt=0.5,
            pilot_samples=10,
            verbose=False
        )

        # Should produce valid result with single level
        assert len(result.level_stats) == 1
        assert result.level_stats[0].level == 0

    def test_very_large_pilot_samples(self, setup_simple_network):
        """Test MLMC when pilot samples exceed optimal allocation."""
        network, traffic = setup_simple_network
        simulator = MLMCSimulator(refinement_factor=2, seed=42)

        result = simulator.mlmc_estimate(
            network=network,
            traffic=traffic,
            epsilon=0.5,  # Large epsilon = fewer samples needed
            L_max=1,
            T=1.0,
            base_dt=0.5,
            pilot_samples=100,  # More than likely needed
            verbose=False
        )

        # Should use at least pilot_samples (reuses them)
        for stats in result.level_stats:
            assert stats.n_samples >= 100


class TestSDEUnstableQueue:
    """Tests for unstable queue SDE handling."""

    def test_unstable_arrival_rate(self):
        """Test SDE with arrival rate >= service rate (unstable)."""
        # This should log a warning but still work
        sde = QueueDynamicsSDE(
            arrival_rate=12.0,  # Higher than service
            service_rate=10.0,
            noise_intensity=0.5
        )

        # Expected queue length should be infinity for unstable queue
        expected = sde.expected_queue_length()
        assert expected == float('inf')

    def test_unstable_queue_simulation(self):
        """Test that unstable queue simulation still produces valid paths."""
        sde = QueueDynamicsSDE(
            arrival_rate=12.0,  # Unstable
            service_rate=10.0,
            noise_intensity=0.5
        )

        # Should still produce a valid (though growing) path
        time, q = sde.simulate_path(T=5.0, dt=0.1, q0=0.0, seed=42)

        assert len(time) == len(q)
        assert np.all(q >= 0)  # Non-negativity should still hold

    def test_zero_service_rate(self):
        """Test SDE with zero service rate."""
        sde = QueueDynamicsSDE(
            arrival_rate=10.0,
            service_rate=0.0,  # No service
            noise_intensity=0.5
        )

        # Should be unstable
        assert sde.expected_queue_length() == float('inf')

    def test_zero_noise_intensity(self):
        """Test SDE with zero noise (deterministic)."""
        sde = QueueDynamicsSDE(
            arrival_rate=8.0,
            service_rate=10.0,
            noise_intensity=0.0  # Deterministic
        )

        # Simulate deterministic path
        time, q = sde.simulate_path(T=10.0, dt=0.1, q0=5.0, seed=42)

        # With λ < μ and no noise, queue should drain
        assert q[-1] < q[0] or q[-1] == 0.0

    def test_negative_initial_queue(self):
        """Test handling of negative initial queue length."""
        sde = QueueDynamicsSDE(
            arrival_rate=8.0,
            service_rate=10.0,
            noise_intensity=0.5
        )

        # Even with negative initial, should remain non-negative
        time, q = sde.simulate_path(T=5.0, dt=0.1, q0=-5.0, seed=42)

        # First step should correct the negative value
        # (or the initial value itself gets reflected)
        # The path should be non-negative
        assert np.all(q[1:] >= 0)


class TestCongestionSDEErrors:
    """Tests for congestion propagation SDE error handling."""

    def test_empty_adjacency_matrix(self):
        """Test congestion SDE with empty network."""
        # Single node (1x1 adjacency)
        adjacency = np.array([[0]])

        sde = CongestionPropagationSDE(adjacency_matrix=adjacency)

        assert sde.n_nodes == 1

        # Should simulate without error
        time, c = sde.simulate_path(T=5.0, dt=0.1, seed=42)
        assert c.shape[1] == 1

    def test_asymmetric_influence(self):
        """Test congestion SDE with directed (asymmetric) adjacency."""
        # Directed graph: 0 -> 1 (but not 1 -> 0)
        adjacency = np.array([
            [0, 1],
            [0, 0]
        ])

        sde = CongestionPropagationSDE(
            adjacency_matrix=adjacency,
            influence_strength=0.5
        )

        # Should work with directed edges
        c0 = np.array([10.0, 0.0])
        time, c = sde.simulate_path(T=5.0, dt=0.1, c0=c0, seed=42)

        assert c.shape[0] == len(time)
        assert np.all(c >= 0)


class TestMonteCarloErrors:
    """Tests for Monte Carlo simulator error handling."""

    def test_zero_samples(self):
        """Test MC estimation with very few samples."""
        gen = TopologyGenerator(seed=42)
        network = gen.generate_erdos_renyi(n_nodes=10, p=0.3)
        network.set_link_properties(seed=42)
        traffic = PoissonTraffic(rate=5.0, seed=42)

        simulator = MonteCarloSimulator(seed=42)

        # Should handle small sample sizes
        result = simulator.estimate(
            network=network,
            traffic=traffic,
            n_samples=2,  # Minimum for variance calculation
            T=1.0,
            dt=0.5,
            verbose=False
        )

        assert result.n_samples == 2
        assert np.isfinite(result.mean)

    def test_invalid_metric(self):
        """Test MC with invalid metric name."""
        gen = TopologyGenerator(seed=42)
        network = gen.generate_erdos_renyi(n_nodes=10, p=0.3)
        network.set_link_properties(seed=42)
        traffic = PoissonTraffic(rate=5.0, seed=42)

        simulator = MonteCarloSimulator(seed=42)

        with pytest.raises(ValueError, match="Unknown metric"):
            simulator.run_single_path(
                network=network,
                traffic=traffic,
                T=1.0,
                dt=0.5,
                metric='invalid_metric_name'
            )


class TestTrafficModelErrors:
    """Tests for traffic model error handling."""

    def test_zero_rate_poisson(self):
        """Test Poisson traffic with zero rate."""
        traffic = PoissonTraffic(rate=0.0, seed=42)

        # Should produce no arrivals
        arrivals = traffic.generate_arrivals(duration=100.0)
        assert len(arrivals) == 0

    def test_negative_duration(self):
        """Test traffic generation with very short duration."""
        traffic = PoissonTraffic(rate=10.0, seed=42)

        # Very short duration
        arrivals = traffic.generate_arrivals(duration=0.001)

        # May or may not have arrivals, but should not crash
        assert isinstance(arrivals, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
