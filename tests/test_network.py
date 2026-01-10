"""
Unit tests for network module (topology, SDE, traffic).

Tests cover:
- NetworkGraph creation and manipulation
- SNAP and CAIDA dataset loaders
- Synthetic topology generation
- SDE simulation
- Traffic model generation
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from network.topology import (
    NetworkGraph,
    TopologyGenerator,
    create_example_network
)
from network.sde import (
    QueueDynamicsSDE,
    CongestionPropagationSDE,
    generate_coupled_brownian_increments
)
from network.traffic import (
    PoissonTraffic,
    BurstyTraffic,
    MAWIBasedTraffic,
    create_traffic_matrix
)


class TestNetworkGraph:
    """Tests for NetworkGraph class."""

    def test_initialization(self):
        """Test graph initialization."""
        # Undirected graph
        g = NetworkGraph(directed=False)
        assert g.n_nodes == 0
        assert g.n_edges == 0
        assert not g.graph.is_directed()

        # Directed graph
        g_dir = NetworkGraph(directed=True)
        assert g_dir.graph.is_directed()

    def test_add_nodes_edges(self):
        """Test adding nodes and edges."""
        g = NetworkGraph()

        # Add nodes
        g.add_node(0, capacity=100)
        g.add_node(1, capacity=200)
        assert g.n_nodes == 2

        # Add edge
        g.add_edge(0, 1, bandwidth=1000, delay=10)
        assert g.n_edges == 1

        # Check attributes
        edge_attrs = g.get_edge_attributes(0, 1)
        assert edge_attrs['bandwidth'] == 1000
        assert edge_attrs['delay'] == 10

    def test_set_link_properties(self):
        """Test setting synthetic link properties."""
        g = NetworkGraph()
        g.add_node(0)
        g.add_node(1)
        g.add_edge(0, 1)

        g.set_link_properties(seed=42)

        props = g.get_link_properties()
        assert 'bandwidth' in props
        assert 'delay' in props
        assert 'capacity' in props
        assert len(props['bandwidth']) == 1

    def test_shortest_paths(self):
        """Test shortest path computation."""
        g = NetworkGraph()
        for i in range(5):
            g.add_node(i)

        # Create path: 0-1-2-3-4
        for i in range(4):
            g.add_edge(i, i + 1)

        paths = g.compute_shortest_paths(source=0)
        assert 4 in paths
        assert len(paths[4]) == 5  # Path length from 0 to 4

    def test_get_neighbors(self):
        """Test getting node neighbors."""
        g = NetworkGraph()
        g.add_node(0)
        g.add_node(1)
        g.add_node(2)
        g.add_edge(0, 1)
        g.add_edge(0, 2)

        neighbors = g.get_neighbors(0)
        assert len(neighbors) == 2
        assert 1 in neighbors
        assert 2 in neighbors

    def test_largest_component(self):
        """Test extracting largest connected component."""
        g = NetworkGraph()

        # Create two components
        # Component 1: 0-1-2 (3 nodes)
        for i in range(3):
            g.add_node(i)
        g.add_edge(0, 1)
        g.add_edge(1, 2)

        # Component 2: 3-4 (2 nodes)
        g.add_node(3)
        g.add_node(4)
        g.add_edge(3, 4)

        largest = g.get_largest_component()
        assert largest.n_nodes == 3
        assert largest.n_edges == 2

    def test_summary(self):
        """Test network summary statistics."""
        g = NetworkGraph()
        for i in range(5):
            g.add_node(i)
        g.add_edge(0, 1)
        g.add_edge(1, 2)

        summary = g.summary()
        assert summary['n_nodes'] == 5
        assert summary['n_edges'] == 2
        assert 'avg_degree' in summary


class TestTopologyGenerator:
    """Tests for TopologyGenerator class."""

    def test_erdos_renyi(self):
        """Test Erdős-Rényi graph generation."""
        gen = TopologyGenerator(seed=42)
        g = gen.generate_erdos_renyi(n_nodes=50, p=0.1)

        assert g.n_nodes == 50
        assert g.n_edges > 0  # With p=0.1, should have some edges

    def test_barabasi_albert(self):
        """Test Barabási-Albert graph generation."""
        gen = TopologyGenerator(seed=42)
        g = gen.generate_barabasi_albert(n_nodes=50, m=3)

        assert g.n_nodes == 50
        # BA graph with m=3 should have (n-m)*m edges
        expected_edges = (50 - 3) * 3
        assert g.n_edges == expected_edges

    def test_watts_strogatz(self):
        """Test Watts-Strogatz graph generation."""
        gen = TopologyGenerator(seed=42)
        g = gen.generate_watts_strogatz(n_nodes=20, k=4, p=0.1)

        assert g.n_nodes == 20

    def test_random_regular(self):
        """Test random regular graph generation."""
        gen = TopologyGenerator(seed=42)
        g = gen.generate_random_regular(n_nodes=20, d=4)

        assert g.n_nodes == 20
        # All nodes should have degree 4
        for node in g.nodes:
            assert g.get_degree(node) == 4

    def test_hierarchical(self):
        """Test hierarchical topology generation."""
        gen = TopologyGenerator(seed=42)
        g = gen.generate_hierarchical(n_levels=3, branching_factor=2)

        # Should have 1 + 2 + 4 = 7 nodes
        assert g.n_nodes == 7
        assert g.n_edges == 6  # Tree with 7 nodes has 6 edges


class TestCreateExampleNetwork:
    """Tests for create_example_network convenience function."""

    def test_create_erdos_renyi(self):
        """Test creating example Erdős-Rényi network."""
        g = create_example_network('erdos_renyi', 'small')
        assert g.n_nodes == 50

    def test_create_barabasi_albert(self):
        """Test creating example Barabási-Albert network."""
        g = create_example_network('barabasi_albert', 'medium')
        assert g.n_nodes == 500


class TestQueueDynamicsSDE:
    """Tests for QueueDynamicsSDE class."""

    def test_initialization(self):
        """Test SDE initialization."""
        sde = QueueDynamicsSDE(
            arrival_rate=8.0,
            service_rate=10.0,
            noise_intensity=0.5
        )

        assert sde.arrival_rate == 8.0
        assert sde.service_rate == 10.0
        assert sde.noise_intensity == 0.5

    def test_drift_diffusion(self):
        """Test drift and diffusion terms."""
        sde = QueueDynamicsSDE(
            arrival_rate=8.0,
            service_rate=10.0,
            noise_intensity=0.5
        )

        # Drift should be λ - μ = 8 - 10 = -2
        drift = sde.drift(q=5.0, t=0.0)
        assert drift == -2.0

        # Diffusion should be σ = 0.5
        diffusion = sde.diffusion(q=5.0, t=0.0)
        assert diffusion == 0.5

    def test_euler_maruyama_step(self):
        """Test single Euler-Maruyama integration step."""
        sde = QueueDynamicsSDE(
            arrival_rate=8.0,
            service_rate=10.0,
            noise_intensity=0.5
        )

        q0 = 5.0
        q1 = sde.euler_maruyama_step(q=q0, t=0.0, dt=0.01, dw=0.0)

        # With dw=0, should be deterministic
        expected = q0 + (8.0 - 10.0) * 0.01
        assert abs(q1 - expected) < 1e-10

    def test_simulate_path(self):
        """Test simulating full queue dynamics path."""
        sde = QueueDynamicsSDE(
            arrival_rate=8.0,
            service_rate=10.0,
            noise_intensity=0.1
        )

        time, q = sde.simulate_path(T=10.0, dt=0.1, q0=5.0, seed=42)

        assert len(time) == len(q)
        assert time[0] == 0.0
        assert time[-1] == 10.0
        assert q[0] == 5.0
        assert np.all(q >= 0)  # Queue length should be non-negative

    def test_simulate_coupled_paths(self):
        """Test coupled path simulation for MLMC."""
        sde = QueueDynamicsSDE(
            arrival_rate=8.0,
            service_rate=10.0,
            noise_intensity=0.5
        )

        time_fine, q_fine, q_coarse = sde.simulate_coupled_paths(
            T=10.0,
            dt_coarse=0.1,
            dt_fine=0.01,
            q0=5.0,
            seed=42
        )

        assert len(q_fine) == len(q_coarse)
        # Paths should be different but correlated
        assert not np.allclose(q_fine, q_coarse)

    def test_expected_queue_length(self):
        """Test theoretical expected queue length."""
        sde = QueueDynamicsSDE(
            arrival_rate=8.0,
            service_rate=10.0,
            noise_intensity=0.0
        )

        expected_q = sde.expected_queue_length()
        # For λ=8, μ=10: ρ = 0.8, E[Q] = ρ/(1-ρ) = 4
        assert abs(expected_q - 4.0) < 1e-10


class TestCongestionPropagationSDE:
    """Tests for CongestionPropagationSDE class."""

    def test_initialization(self):
        """Test congestion SDE initialization."""
        adjacency = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])

        sde = CongestionPropagationSDE(
            adjacency_matrix=adjacency,
            influence_strength=0.2,
            decay_rate=0.5,
            noise_intensity=0.1
        )

        assert sde.n_nodes == 3
        assert sde.influence_strength == 0.2

    def test_drift_diffusion(self):
        """Test drift and diffusion for congestion model."""
        adjacency = np.array([
            [0, 1],
            [1, 0]
        ])

        sde = CongestionPropagationSDE(
            adjacency_matrix=adjacency,
            influence_strength=0.2,
            decay_rate=0.5,
            noise_intensity=0.1
        )

        c = np.array([5.0, 3.0])
        drift = sde.drift(c, t=0.0)

        assert len(drift) == 2
        # Each node influenced by neighbor minus self-decay

        diffusion = sde.diffusion(c, t=0.0)
        assert len(diffusion) == 2
        assert np.all(diffusion == 0.1)

    def test_simulate_path(self):
        """Test congestion propagation simulation."""
        adjacency = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])

        sde = CongestionPropagationSDE(
            adjacency_matrix=adjacency,
            influence_strength=0.2,
            decay_rate=0.5,
            noise_intensity=0.1
        )

        c0 = np.array([10.0, 0.0, 0.0])  # Initial congestion at node 0
        time, c = sde.simulate_path(T=10.0, dt=0.1, c0=c0, seed=42)

        assert c.shape[0] == len(time)
        assert c.shape[1] == 3
        assert np.all(c >= 0)  # Non-negative congestion

    def test_inject_congestion(self):
        """Test congestion injection."""
        adjacency = np.eye(3)  # Isolated nodes
        sde = CongestionPropagationSDE(adjacency_matrix=adjacency)

        c = np.zeros(3)
        c_new = sde.inject_congestion(c, node_ids=[0, 2], intensity=5.0)

        assert c_new[0] == 5.0
        assert c_new[1] == 0.0
        assert c_new[2] == 5.0


class TestCoupledBrownianIncrements:
    """Tests for coupled Brownian increment generation."""

    def test_coupled_increments(self):
        """Test generating coupled Brownian increments."""
        dw_fine, dw_coarse = generate_coupled_brownian_increments(
            dt_coarse=0.1,
            dt_fine=0.01,
            n_steps_fine=1000,
            dim=1,
            seed=42
        )

        assert dw_fine.shape == (1000, 1)
        assert dw_coarse.shape == (100, 1)

        # Verify coupling: sum of 10 fine increments = coarse increment
        for i in range(100):
            fine_sum = np.sum(dw_fine[i * 10:(i + 1) * 10])
            assert abs(fine_sum - dw_coarse[i]) < 1e-10


class TestPoissonTraffic:
    """Tests for PoissonTraffic class."""

    def test_initialization(self):
        """Test Poisson traffic initialization."""
        traffic = PoissonTraffic(rate=10.0, seed=42)
        assert traffic.rate == 10.0

    def test_generate_arrivals(self):
        """Test Poisson arrival generation."""
        traffic = PoissonTraffic(rate=10.0, seed=42)
        arrivals = traffic.generate_arrivals(duration=100.0)

        # Should have approximately rate * duration arrivals
        assert 800 < len(arrivals) < 1200  # Allow some variance

        # Arrivals should be sorted and within duration
        assert np.all(np.diff(arrivals) >= 0)
        assert np.all(arrivals <= 100.0)

    def test_generate_packet_sizes(self):
        """Test packet size generation."""
        traffic = PoissonTraffic(
            rate=10.0,
            mean_packet_size=1500.0,
            packet_size_std=200.0,
            seed=42
        )

        sizes = traffic.generate_packet_sizes(n_packets=1000)

        assert len(sizes) == 1000
        assert np.all(sizes >= 64)  # Minimum Ethernet frame size
        # Mean should be approximately 1500
        assert 1400 < np.mean(sizes) < 1600

    def test_get_statistics(self):
        """Test traffic statistics computation."""
        traffic = PoissonTraffic(rate=10.0, seed=42)
        stats = traffic.get_statistics(duration=100.0)

        assert 'arrival_rate' in stats
        assert 'coefficient_of_variation' in stats
        # Poisson CV should be approximately 1
        assert 0.8 < stats['coefficient_of_variation'] < 1.2


class TestBurstyTraffic:
    """Tests for BurstyTraffic class."""

    def test_initialization(self):
        """Test bursty traffic initialization."""
        traffic = BurstyTraffic(
            on_rate=50.0,
            mean_on_duration=0.5,
            mean_off_duration=0.5,
            seed=42
        )

        assert traffic.on_rate == 50.0
        # Effective rate should be on_rate * (on / (on + off)) = 25.0
        assert abs(traffic.effective_rate - 25.0) < 1e-10

    def test_generate_arrivals(self):
        """Test bursty arrival generation."""
        traffic = BurstyTraffic(
            on_rate=50.0,
            mean_on_duration=1.0,
            mean_off_duration=1.0,
            seed=42
        )

        arrivals = traffic.generate_arrivals(duration=100.0)

        # Should have some arrivals
        assert len(arrivals) > 0
        assert np.all(arrivals <= 100.0)


class TestMAWIBasedTraffic:
    """Tests for MAWIBasedTraffic class."""

    def test_initialization_low_burstiness(self):
        """Test MAWI traffic with low burstiness (Poisson-like)."""
        traffic = MAWIBasedTraffic(
            arrival_rate=100.0,
            burstiness=1.0,  # Low burstiness
            mean_packet_size=800.0,
            packet_size_std=200.0,
            seed=42
        )

        assert not traffic._use_bursty_model  # Should use Poisson

    def test_initialization_high_burstiness(self):
        """Test MAWI traffic with high burstiness (On-Off model)."""
        traffic = MAWIBasedTraffic(
            arrival_rate=100.0,
            burstiness=2.5,  # High burstiness
            mean_packet_size=800.0,
            packet_size_std=200.0,
            seed=42
        )

        assert traffic._use_bursty_model  # Should use bursty model

    def test_generate_arrivals(self):
        """Test MAWI-based arrival generation."""
        traffic = MAWIBasedTraffic(
            arrival_rate=100.0,
            burstiness=2.0,
            mean_packet_size=800.0,
            packet_size_std=200.0,
            seed=42
        )

        arrivals = traffic.generate_arrivals(duration=100.0)
        assert len(arrivals) > 0

    def test_generate_packet_sizes(self):
        """Test MAWI-based packet size generation."""
        traffic = MAWIBasedTraffic(
            arrival_rate=100.0,
            burstiness=1.5,
            mean_packet_size=800.0,
            packet_size_std=400.0,
            seed=42
        )

        sizes = traffic.generate_packet_sizes(n_packets=1000)
        assert len(sizes) == 1000
        assert np.all(sizes >= 64)
        assert np.all(sizes <= 9000)  # MTU constraint


class TestCreateTrafficMatrix:
    """Tests for create_traffic_matrix function."""

    def test_traffic_matrix_creation(self):
        """Test creating traffic matrix for network."""
        # Create small network
        gen = TopologyGenerator(seed=42)
        network = gen.generate_erdos_renyi(n_nodes=10, p=0.3)

        # Create traffic model
        traffic = PoissonTraffic(rate=10.0, seed=42)

        # Generate traffic matrix
        tm = create_traffic_matrix(
            network_graph=network,
            traffic_model=traffic,
            n_flows=5,
            duration=10.0,
            seed=42
        )

        assert tm['n_flows'] == 5
        assert len(tm['sources']) == 5
        assert len(tm['destinations']) == 5
        assert len(tm['arrivals']) == 5
        assert len(tm['sizes']) == 5

        # Sources and destinations should be different
        for src, dst in zip(tm['sources'], tm['destinations']):
            assert src != dst


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
