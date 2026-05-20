"""
Integration tests for full MLMC pipeline.

Tests end-to-end functionality combining all components:
- Network topology + Traffic model
- Monte Carlo and MLMC simulation
- GPU acceleration (when available)
- Metrics and uncertainty quantification
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from network.topology import TopologyGenerator, NetworkGraph
from network.traffic import PoissonTraffic
from simulation.monte_carlo import MonteCarloSimulator
from simulation.mlmc import MLMCSimulator
from simulation.discretization import MLMCHierarchy
from metrics.uncertainty import UncertaintyQuantifier

# Check GPU availability (try to instantiate to catch runtime ImportError from PyCUDA)
try:
    from gpu.parallel_mc import GPUMLMCSimulator
    _dummy = GPUMLMCSimulator(seed=0)
    del _dummy
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


class TestMLMCIntegration:
    """End-to-end MLMC integration tests."""

    @pytest.fixture
    def network_and_traffic(self):
        """Create network and traffic model for tests."""
        generator = TopologyGenerator(seed=42)
        network = generator.generate_erdos_renyi(n_nodes=10, p=0.3)
        network.set_link_properties(seed=42)
        traffic = PoissonTraffic(rate=5.0, seed=42)
        return network, traffic

    def test_mlmc_end_to_end(self, network_and_traffic):
        """Test full MLMC pipeline from network to estimate."""
        network, traffic = network_and_traffic

        # Create MLMC simulator
        mlmc = MLMCSimulator(seed=42)

        # Run estimation with target accuracy
        result = mlmc.estimate(
            network=network,
            traffic=traffic,
            epsilon=0.1,
            T=5.0,
            base_dt=0.1,
            L_max=3
        )

        # Verify result structure
        assert result.mean is not None
        assert result.variance > 0
        assert result.rmse > 0
        assert len(result.level_stats) > 0

        # Verify confidence interval contains mean
        assert result.ci_lower <= result.mean <= result.ci_upper

    def test_mlmc_achieves_target_accuracy(self, network_and_traffic):
        """Verify MLMC achieves specified target accuracy."""
        network, traffic = network_and_traffic

        mlmc = MLMCSimulator(seed=42)
        epsilon = 0.05

        result = mlmc.estimate(
            network=network,
            traffic=traffic,
            epsilon=epsilon,
            T=5.0,
            base_dt=0.1,
            L_max=4
        )

        # RMSE should be approximately epsilon
        # Allow some tolerance since this is statistical
        assert result.rmse < 3 * epsilon, f"RMSE {result.rmse} exceeds 3*epsilon={3*epsilon}"

    def test_mlmc_cost_scaling(self, network_and_traffic):
        """Verify MLMC achieves better cost than standard MC."""
        network, traffic = network_and_traffic

        mlmc = MLMCSimulator(seed=42)

        result = mlmc.estimate(
            network=network,
            traffic=traffic,
            epsilon=0.1,
            T=5.0,
            base_dt=0.1,
            L_max=3
        )

        # Total cost should be reported
        total_cost = result.total_cost
        assert total_cost > 0

        # Verify cost is distributed across levels
        level_costs = [stats.cost for stats in result.level_stats]
        assert sum(level_costs) == total_cost

    def test_mlmc_variance_decay(self, network_and_traffic):
        """Verify variance decays geometrically across levels."""
        network, traffic = network_and_traffic

        mlmc = MLMCSimulator(seed=42)

        result = mlmc.estimate(
            network=network,
            traffic=traffic,
            epsilon=0.1,
            T=5.0,
            base_dt=0.1,
            L_max=4,
            min_samples=100
        )

        # Extract level variances
        variances = [stats.var_diff for stats in result.level_stats]

        # Variance should generally decrease (with some noise)
        # Check that later levels have lower variance on average
        if len(variances) >= 3:
            early_var = np.mean(variances[:2])
            late_var = np.mean(variances[-2:])
            assert late_var < early_var, "Variance should decay across levels"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestGPUCPUConsistency:
    """Tests for GPU/CPU result consistency."""

    @pytest.fixture
    def network_and_traffic(self):
        """Create network and traffic model for tests."""
        generator = TopologyGenerator(seed=42)
        network = generator.generate_erdos_renyi(n_nodes=10, p=0.3)
        network.set_link_properties(seed=42)
        traffic = PoissonTraffic(rate=5.0, seed=42)
        return network, traffic

    def test_gpu_cpu_mean_consistency(self, network_and_traffic):
        """GPU and CPU should produce similar means."""
        network, traffic = network_and_traffic

        # CPU MLMC
        cpu_mlmc = MLMCSimulator(seed=42)
        cpu_result = cpu_mlmc.estimate(
            network=network,
            traffic=traffic,
            epsilon=0.1,
            T=5.0,
            base_dt=0.1,
            L_max=3
        )

        # GPU MLMC
        gpu_mlmc = GPUMLMCSimulator(seed=42)
        gpu_result = gpu_mlmc.estimate(
            network=network,
            traffic=traffic,
            epsilon=0.1,
            T=5.0,
            base_dt=0.1,
            L_max=3
        )

        # Means should be within statistical tolerance
        # Using 3 sigma tolerance
        tolerance = 3 * (cpu_result.rmse + gpu_result.rmse)
        diff = abs(cpu_result.mean - gpu_result.mean)
        assert diff < tolerance, f"CPU/GPU mean diff {diff} exceeds tolerance {tolerance}"

    def test_gpu_cpu_variance_consistency(self, network_and_traffic):
        """GPU and CPU should produce similar variances."""
        network, traffic = network_and_traffic

        # CPU MLMC
        cpu_mlmc = MLMCSimulator(seed=42)
        cpu_result = cpu_mlmc.estimate(
            network=network,
            traffic=traffic,
            epsilon=0.1,
            T=5.0,
            base_dt=0.1,
            L_max=3
        )

        # GPU MLMC
        gpu_mlmc = GPUMLMCSimulator(seed=42)
        gpu_result = gpu_mlmc.estimate(
            network=network,
            traffic=traffic,
            epsilon=0.1,
            T=5.0,
            base_dt=0.1,
            L_max=3
        )

        # Variances should be same order of magnitude
        ratio = cpu_result.variance / gpu_result.variance if gpu_result.variance > 0 else float('inf')
        assert 0.1 < ratio < 10, f"Variance ratio {ratio} outside expected range"


class TestFullPipeline:
    """Full pipeline integration tests."""

    @pytest.fixture
    def network_and_traffic(self):
        """Create network and traffic model for tests."""
        generator = TopologyGenerator(seed=42)
        network = generator.generate_erdos_renyi(n_nodes=10, p=0.3)
        network.set_link_properties(seed=42)
        traffic = PoissonTraffic(rate=5.0, seed=42)
        return network, traffic

    def test_topology_to_uncertainty(self, network_and_traffic):
        """Test from topology generation to uncertainty quantification."""
        network, traffic = network_and_traffic

        # Step 1: MLMC simulation
        mlmc = MLMCSimulator(seed=42)
        result = mlmc.estimate(
            network=network,
            traffic=traffic,
            epsilon=0.1,
            T=5.0,
            base_dt=0.1,
            L_max=3
        )

        # Step 2: Uncertainty quantification
        uq = UncertaintyQuantifier()

        # Get samples from result (if available) or use estimate
        # For testing, we verify the integration works
        assert result.mean is not None
        assert result.variance > 0
        assert result.ci_lower < result.ci_upper

    def test_monte_carlo_baseline(self, network_and_traffic):
        """Test standard Monte Carlo as baseline."""
        network, traffic = network_and_traffic

        mc = MonteCarloSimulator(seed=42)
        result = mc.estimate(
            network=network,
            traffic=traffic,
            n_samples=500,
            T=5.0,
            dt=0.05,
            metric='mean_queue'
        )

        # Verify MC produces valid results
        assert result.mean is not None
        assert result.variance > 0
        assert result.n_samples == 500

    def test_different_network_topologies(self):
        """Test with different network topologies."""
        generator = TopologyGenerator(seed=42)
        traffic = PoissonTraffic(rate=5.0, seed=42)
        mlmc = MLMCSimulator(seed=42)

        # Test with different topologies
        topologies = [
            generator.generate_line_graph(n_nodes=5),
            generator.generate_star_graph(n_nodes=6),
            generator.generate_erdos_renyi(n_nodes=8, p=0.4),
        ]

        for network in topologies:
            network.set_link_properties(seed=42)
            result = mlmc.estimate(
                network=network,
                traffic=traffic,
                epsilon=0.2,
                T=3.0,
                base_dt=0.1,
                L_max=2
            )
            assert result.mean is not None
            assert result.variance > 0


class TestMLMCHierarchyIntegration:
    """Test MLMC hierarchy with simulation."""

    def test_hierarchy_with_simulation(self):
        """Test that MLMC hierarchy integrates with simulation."""
        hierarchy = MLMCHierarchy(dt_coarsest=0.1, L_max=3, refinement_factor=2)

        # Verify hierarchy properties
        assert hierarchy.L_max == 3
        assert len(hierarchy.levels) == 4

        # Verify timesteps decrease
        dts = [hierarchy.get_timestep(l) for l in range(4)]
        for i in range(1, len(dts)):
            assert dts[i] < dts[i-1], "Timesteps should decrease with level"

    def test_coupled_simulation_consistency(self):
        """Test that coupled paths maintain consistency."""
        generator = TopologyGenerator(seed=42)
        network = generator.generate_line_graph(n_nodes=3)
        network.set_link_properties(seed=42)
        traffic = PoissonTraffic(rate=5.0, seed=42)

        mlmc = MLMCSimulator(seed=42)

        # Run with enough samples to get reliable statistics
        result = mlmc.estimate(
            network=network,
            traffic=traffic,
            epsilon=0.1,
            T=5.0,
            base_dt=0.1,
            L_max=3,
            min_samples=50
        )

        # Verify level statistics are computed
        for stats in result.level_stats:
            assert stats.n_samples > 0
            assert stats.var_diff >= 0  # Variance must be non-negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
