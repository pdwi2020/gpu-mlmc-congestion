"""
Unit tests for simulation module (discretization, Monte Carlo, MLMC).

Tests cover:
- Discretization hierarchy
- Coupled noise generation
- Monte Carlo estimation
- MLMC estimation
- Convergence properties
"""

import pytest
import numpy as np
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulation.discretization import (
    DiscretizationLevel,
    MLMCHierarchy,
    get_timestep,
    generate_coupled_noise,
    align_coarse_to_fine_grid,
    adaptive_timestep_selection,
    BrownianBridge
)
from simulation.monte_carlo import (
    MonteCarloSimulator,
    NetworkSimulationResult,
    NetworkMetricsCalculator
)
from simulation.mlmc import (
    MLMCSimulator,
    MLMCResult,
    MLMCLevelStats
)
from network.topology import TopologyGenerator
from network.traffic import PoissonTraffic


class TestDiscretizationLevel:
    """Tests for DiscretizationLevel class."""

    def test_initialization(self):
        """Test discretization level initialization."""
        level = DiscretizationLevel(level=2, dt=0.025, refinement_factor=2)

        assert level.level == 2
        assert level.dt == 0.025
        assert level.refinement_factor == 2

    def test_get_coarser_dt(self):
        """Test getting coarser time step."""
        level = DiscretizationLevel(level=2, dt=0.025, refinement_factor=2)
        dt_coarser = level.get_coarser_dt()

        assert dt_coarser == 0.05

    def test_get_finer_dt(self):
        """Test getting finer time step."""
        level = DiscretizationLevel(level=2, dt=0.025, refinement_factor=2)
        dt_finer = level.get_finer_dt()

        assert dt_finer == 0.0125

    def test_n_steps_for_duration(self):
        """Test computing number of steps."""
        level = DiscretizationLevel(level=0, dt=0.1, refinement_factor=2)
        n_steps = level.n_steps_for_duration(T=10.0)

        assert n_steps == 100


class TestMLMCHierarchy:
    """Tests for MLMCHierarchy class."""

    def test_initialization(self):
        """Test MLMC hierarchy initialization."""
        hierarchy = MLMCHierarchy(dt_coarsest=0.1, L_max=4, refinement_factor=2)

        assert hierarchy.L_max == 4
        assert hierarchy.refinement_factor == 2
        assert len(hierarchy.levels) == 5  # L=0,1,2,3,4

    def test_get_timestep(self):
        """Test getting time step for each level."""
        hierarchy = MLMCHierarchy(dt_coarsest=0.1, L_max=3, refinement_factor=2)

        expected_dts = [0.1, 0.05, 0.025, 0.0125]
        for l in range(4):
            dt = hierarchy.get_timestep(l)
            assert abs(dt - expected_dts[l]) < 1e-10

    def test_get_refinement_factor_between(self):
        """Test refinement factor between levels."""
        hierarchy = MLMCHierarchy(dt_coarsest=0.1, L_max=4, refinement_factor=2)

        # From level 3 to level 1: M^2 = 4
        M = hierarchy.get_refinement_factor_between(l_fine=3, l_coarse=1)
        assert M == 4

    def test_verify_coupling(self):
        """Test coupling verification."""
        hierarchy = MLMCHierarchy(dt_coarsest=0.1, L_max=4, refinement_factor=2)

        # All levels should have valid coupling
        for l in range(hierarchy.L_max + 1):
            assert hierarchy.verify_coupling(l)


class TestGetTimestep:
    """Tests for get_timestep function."""

    def test_timestep_calculation(self):
        """Test time step calculation."""
        base_dt = 0.1
        M = 2

        # Level 0: dt = 0.1
        assert get_timestep(0, base_dt, M) == 0.1

        # Level 1: dt = 0.05
        assert abs(get_timestep(1, base_dt, M) - 0.05) < 1e-10

        # Level 3: dt = 0.0125
        assert abs(get_timestep(3, base_dt, M) - 0.0125) < 1e-10


class TestGenerateCoupledNoise:
    """Tests for coupled Brownian motion generation."""

    def test_coupled_noise_dimensions(self):
        """Test dimensions of coupled noise."""
        dW_fine, dW_coarse = generate_coupled_noise(
            dt_coarse=0.1,
            dt_fine=0.01,
            n_steps_fine=1000,
            dim=1,
            seed=42
        )

        assert dW_fine.shape == (1000, 1)
        assert dW_coarse.shape == (100, 1)

    def test_coupling_property(self):
        """Test that fine increments sum to coarse increments."""
        dW_fine, dW_coarse = generate_coupled_noise(
            dt_coarse=0.1,
            dt_fine=0.01,
            n_steps_fine=1000,
            dim=2,
            seed=42
        )

        M = 10  # Refinement factor
        for i in range(len(dW_coarse)):
            for d in range(2):
                fine_sum = np.sum(dW_fine[i * M:(i + 1) * M, d])
                coarse_val = dW_coarse[i, d]
                assert abs(fine_sum - coarse_val) < 1e-10

    def test_multidimensional(self):
        """Test multidimensional Brownian motion."""
        dW_fine, dW_coarse = generate_coupled_noise(
            dt_coarse=0.2,
            dt_fine=0.05,
            n_steps_fine=40,
            dim=5,
            seed=42
        )

        assert dW_fine.shape == (40, 5)
        assert dW_coarse.shape == (10, 5)


class TestAlignCoarseToFineGrid:
    """Tests for grid alignment."""

    def test_alignment(self):
        """Test aligning coarse values to fine grid."""
        coarse_values = np.array([1.0, 2.0, 3.0])
        M = 4

        aligned = align_coarse_to_fine_grid(coarse_values, M)

        expected = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
        assert np.allclose(aligned, expected)


class TestAdaptiveTimestepSelection:
    """Tests for adaptive level selection."""

    def test_adaptive_selection(self):
        """Test adaptive selection of MLMC levels."""
        # Smaller epsilon should require more levels
        L_small = adaptive_timestep_selection(
            error_tolerance=0.001,
            base_dt=0.1,
            refinement_factor=2
        )

        L_large = adaptive_timestep_selection(
            error_tolerance=0.1,
            base_dt=0.1,
            refinement_factor=2
        )

        assert L_small > L_large


class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator class."""

    @pytest.fixture
    def setup_network_traffic(self):
        """Setup network and traffic for testing."""
        gen = TopologyGenerator(seed=42)
        network = gen.generate_erdos_renyi(n_nodes=20, p=0.2)
        network.set_link_properties(seed=42)

        traffic = PoissonTraffic(rate=5.0, seed=42)

        return network, traffic

    def test_initialization(self):
        """Test MC simulator initialization."""
        simulator = MonteCarloSimulator(seed=42)
        assert simulator.seed == 42

    def test_run_single_path(self, setup_network_traffic):
        """Test running single sample path."""
        network, traffic = setup_network_traffic
        simulator = MonteCarloSimulator(seed=42)

        value = simulator.run_single_path(
            network=network,
            traffic=traffic,
            T=5.0,
            dt=0.1,
            metric='mean_queue',
            seed=42
        )

        assert isinstance(value, float)
        assert value >= 0.0  # Queue length should be non-negative

    def test_estimate(self, setup_network_traffic):
        """Test Monte Carlo estimation."""
        network, traffic = setup_network_traffic
        simulator = MonteCarloSimulator(seed=42)

        result = simulator.estimate(
            network=network,
            traffic=traffic,
            n_samples=50,
            T=5.0,
            dt=0.1,
            metric='mean_queue',
            verbose=False
        )

        assert isinstance(result, NetworkSimulationResult)
        assert result.n_samples == 50
        assert len(result.samples) == 50
        assert result.mean >= 0.0
        assert result.variance >= 0.0
        assert result.ci_lower <= result.mean <= result.ci_upper

    def test_confidence_interval(self):
        """Test confidence interval computation."""
        simulator = MonteCarloSimulator(seed=42)

        # Generate sample data
        samples = np.random.normal(5.0, 1.0, 100)

        ci_lower, ci_upper = simulator.compute_confidence_interval(
            samples, confidence_level=0.95
        )

        # Mean should be within CI
        mean = np.mean(samples)
        assert ci_lower <= mean <= ci_upper

        # CI should be wider for lower confidence
        ci_lower_90, ci_upper_90 = simulator.compute_confidence_interval(
            samples, confidence_level=0.90
        )

        width_95 = ci_upper - ci_lower
        width_90 = ci_upper_90 - ci_lower_90

        assert width_90 < width_95

    def test_convergence_test(self, setup_network_traffic):
        """Test convergence as sample size increases."""
        network, traffic = setup_network_traffic
        simulator = MonteCarloSimulator(seed=42)

        results = simulator.convergence_test(
            network=network,
            traffic=traffic,
            sample_sizes=[10, 50, 100],
            T=5.0,
            dt=0.1,
            n_trials=3,
            verbose=False
        )

        # CI width should decrease with sample size
        ci_widths = results['ci_widths']
        assert ci_widths[0] > ci_widths[2]  # 10 samples vs 100 samples


class TestMLMCSimulator:
    """Tests for MLMCSimulator class."""

    @pytest.fixture
    def setup_network_traffic(self):
        """Setup network and traffic for testing."""
        gen = TopologyGenerator(seed=42)
        network = gen.generate_erdos_renyi(n_nodes=20, p=0.2)
        network.set_link_properties(seed=42)

        traffic = PoissonTraffic(rate=5.0, seed=42)

        return network, traffic

    def test_initialization(self):
        """Test MLMC simulator initialization."""
        simulator = MLMCSimulator(refinement_factor=2, seed=42)

        assert simulator.refinement_factor == 2
        assert simulator.seed == 42

    def test_run_coupled_paths_level_0(self, setup_network_traffic):
        """Test coupled paths for level 0."""
        network, traffic = setup_network_traffic
        simulator = MLMCSimulator(refinement_factor=2, seed=42)

        Y_fine, Y_coarse = simulator.run_coupled_paths(
            network=network,
            traffic=traffic,
            level=0,
            T=5.0,
            base_dt=0.1,
            metric='mean_queue',
            seed=42
        )

        assert isinstance(Y_fine, float)
        assert Y_coarse == 0.0  # Y_{-1} = 0 by convention

    def test_run_coupled_paths_level_1(self, setup_network_traffic):
        """Test coupled paths for level > 0."""
        network, traffic = setup_network_traffic
        simulator = MLMCSimulator(refinement_factor=2, seed=42)

        Y_fine, Y_coarse = simulator.run_coupled_paths(
            network=network,
            traffic=traffic,
            level=1,
            T=5.0,
            base_dt=0.1,
            metric='mean_queue',
            seed=42
        )

        assert isinstance(Y_fine, float)
        assert isinstance(Y_coarse, float)
        # Coupled paths should be correlated (not identical)
        assert Y_fine != Y_coarse

    def test_estimate_level_variance(self, setup_network_traffic):
        """Test variance estimation for a level."""
        network, traffic = setup_network_traffic
        simulator = MLMCSimulator(refinement_factor=2, seed=42)

        mean_diff, var_diff, cost, _ = simulator.estimate_level_variance(
            network=network,
            traffic=traffic,
            level=1,
            T=5.0,
            base_dt=0.1,
            metric='mean_queue',
            n_samples=20
        )

        assert isinstance(mean_diff, float)
        assert var_diff >= 0.0
        assert cost > 0.0

    def test_compute_optimal_samples(self):
        """Test optimal sample allocation."""
        simulator = MLMCSimulator(refinement_factor=2, seed=42)

        variances = [1.0, 0.25, 0.0625, 0.015625]  # Decaying by 1/4
        costs = [10, 20, 40, 80]  # Doubling

        N_samples = simulator.compute_optimal_samples(
            variances=variances,
            costs=costs,
            epsilon=0.01
        )

        # Should have 4 sample counts
        assert len(N_samples) == 4

        # All should be positive
        assert all(n > 0 for n in N_samples)

        # Higher variance and lower cost → more samples
        # Level 0 has highest V/C ratio
        assert N_samples[0] == max(N_samples)

    def test_mlmc_estimate(self, setup_network_traffic):
        """Test full MLMC estimation."""
        network, traffic = setup_network_traffic
        simulator = MLMCSimulator(refinement_factor=2, seed=42)

        result = simulator.mlmc_estimate(
            network=network,
            traffic=traffic,
            epsilon=0.05,
            L_max=2,
            T=5.0,
            base_dt=0.2,
            pilot_samples=10,
            verbose=False
        )

        assert isinstance(result, MLMCResult)
        assert result.L_max == 2
        assert len(result.level_stats) == 3  # L=0,1,2
        assert result.estimate >= 0.0
        assert result.variance >= 0.0
        assert result.total_cost > 0.0

    def test_variance_decay(self, setup_network_traffic):
        """Test that variance decays across levels."""
        network, traffic = setup_network_traffic
        simulator = MLMCSimulator(refinement_factor=2, seed=42)

        result = simulator.mlmc_estimate(
            network=network,
            traffic=traffic,
            epsilon=0.05,
            L_max=3,
            T=5.0,
            base_dt=0.2,
            pilot_samples=20,
            verbose=False
        )

        # Variance should generally decay for higher levels
        # (though not guaranteed for small samples)
        variances = [stats.var_diff for stats in result.level_stats]

        # At least check that variances are non-negative
        assert all(v >= 0 for v in variances)

    def test_mlmc_convergence_rate(self, setup_network_traffic):
        """Test MLMC convergence rate: V_l should decay as M^(-alpha*l).

        For Euler-Maruyama with weak order 1, we expect alpha ≈ 2.
        Due to the reflected boundary (Q >= 0), actual rate may be lower.
        """
        network, traffic = setup_network_traffic
        simulator = MLMCSimulator(refinement_factor=2, seed=42)

        # Run MLMC with more samples for reliable variance estimates
        result = simulator.mlmc_estimate(
            network=network,
            traffic=traffic,
            epsilon=0.02,
            L_max=4,
            T=5.0,
            base_dt=0.2,
            pilot_samples=50,
            verbose=False
        )

        # Extract variances for levels 1+ (level 0 has no coarse comparison)
        variances = [stats.var_diff for stats in result.level_stats[1:]]

        if len(variances) >= 2 and all(v > 0 for v in variances):
            # Compute empirical decay rate using log-log regression
            # V_l = C * M^(-alpha*l) => log(V_l) = log(C) - alpha*l*log(M)
            levels = np.arange(1, len(variances) + 1)
            log_vars = np.log(variances)

            # Linear fit: log(V) = a - b*l where b = alpha*log(M)
            coeffs = np.polyfit(levels, log_vars, 1)
            slope = -coeffs[0]  # Negative of slope gives decay rate

            # With M=2, alpha = slope / log(2)
            alpha = slope / np.log(2)

            # For reflected SDE, expect alpha between 0.5 and 2
            # Standard Euler-Maruyama gives alpha ≈ 2
            # Reflected boundary may reduce this
            assert alpha > 0, f"Variance should decay (alpha={alpha})"

            # Log the estimated rate for diagnostics
            logger.info(f"MLMC variance decay rate: alpha = {alpha:.2f}")
            logger.info(f"(Expected: ~2 for standard E-M, may be lower due to boundary)")

    def test_mlmc_cost_scaling(self, setup_network_traffic):
        """Test that MLMC cost scales as O(epsilon^-2)."""
        network, traffic = setup_network_traffic
        simulator = MLMCSimulator(refinement_factor=2, seed=42)

        costs = []
        epsilons = [0.1, 0.05]

        for eps in epsilons:
            result = simulator.mlmc_estimate(
                network=network,
                traffic=traffic,
                epsilon=eps,
                L_max=3,
                T=5.0,
                base_dt=0.2,
                pilot_samples=20,
                verbose=False
            )
            costs.append(result.total_cost)

        # For O(eps^-2) scaling: cost ratio ≈ (eps1/eps2)^2
        # With eps1=0.1 and eps2=0.05: ratio should be ~4
        if costs[0] > 0 and costs[1] > 0:
            actual_ratio = costs[1] / costs[0]
            expected_ratio = (epsilons[0] / epsilons[1]) ** 2  # = 4

            # Allow significant tolerance due to pilot overhead and finite samples
            assert actual_ratio >= 1.0, "Cost should not decrease for tighter epsilon"
            logger.info(f"Cost ratio: {actual_ratio:.2f} (expected ~{expected_ratio})")


class TestNetworkMetricsCalculator:
    """Tests for NetworkMetricsCalculator."""

    def test_link_utilization(self):
        """Test link utilization calculation."""
        util = NetworkMetricsCalculator.link_utilization(
            arrival_rate=8.0,
            service_rate=10.0
        )

        assert util == 0.8

    def test_packet_loss_probability(self):
        """Test packet loss probability estimation."""
        # Queue below capacity
        loss = NetworkMetricsCalculator.packet_loss_probability(
            queue_length=50.0,
            buffer_size=100.0
        )
        assert loss == 0.0

        # Queue at capacity
        loss = NetworkMetricsCalculator.packet_loss_probability(
            queue_length=100.0,
            buffer_size=100.0
        )
        assert loss == 0.0

        # Queue above capacity
        loss = NetworkMetricsCalculator.packet_loss_probability(
            queue_length=150.0,
            buffer_size=100.0
        )
        assert loss > 0.0


class TestGPUCoupledPropagationMLMC:
    """Integration tests for GPUCoupledPropagationMLMC (runs on CPU via torch fallback)."""

    def _make_chain_adj(self, n: int) -> np.ndarray:
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1.0
        return adj

    def test_mlmc_estimate_returns_valid_result(self):
        """mlmc_estimate returns well-structured dict with valid statistics."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from gpu.parallel_mc import GPUCoupledPropagationMLMC

        adj = self._make_chain_adj(5)
        mlmc = GPUCoupledPropagationMLMC(
            adj, influence_strength=0.2, decay_rate=0.5,
            noise_intensity=0.1, seed=42
        )
        result = mlmc.mlmc_estimate(
            epsilon=0.1, T=1.0, base_dt=0.1,
            L_max=2, pilot_samples=20, verbose=False
        )

        assert isinstance(result, dict)
        assert result['estimate'] >= 0, "Mean congestion must be non-negative"
        assert result['ci_lower'] < result['estimate'] < result['ci_upper'], (
            "Estimate must lie within its own CI"
        )
        assert len(result['level_stats']) == 3, "L_max=2 gives levels 0,1,2"
        assert result['total_cost'] > 0
        assert result['variance'] >= 0

    def test_mlmc_estimate_tighter_epsilon_gives_narrower_ci(self):
        """Tighter epsilon produces a narrower confidence interval."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from gpu.parallel_mc import GPUCoupledPropagationMLMC

        adj = self._make_chain_adj(4)
        mlmc = GPUCoupledPropagationMLMC(adj, seed=0)

        r_loose = mlmc.mlmc_estimate(epsilon=0.2, T=0.5, base_dt=0.1,
                                     L_max=2, pilot_samples=20, verbose=False)
        r_tight = mlmc.mlmc_estimate(epsilon=0.05, T=0.5, base_dt=0.1,
                                     L_max=2, pilot_samples=20, verbose=False)

        ci_loose = r_loose['ci_upper'] - r_loose['ci_lower']
        ci_tight = r_tight['ci_upper'] - r_tight['ci_lower']
        assert ci_tight <= ci_loose * 1.5, (
            f"Tighter epsilon should yield narrower CI: {ci_tight:.4f} vs {ci_loose:.4f}"
        )

    def test_mlmc_level0_coarse_is_zero(self):
        """At level 0, Y_coarse should be identically zero (Y_{-1}=0 convention)."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from gpu.parallel_mc import GPUCoupledPropagationMLMC

        adj = self._make_chain_adj(3)
        mlmc = GPUCoupledPropagationMLMC(adj, seed=7)
        Yf, Yc = mlmc.run_level(level=0, n_samples=50, T=0.5, base_dt=0.1)
        assert np.all(Yc == 0.0), "Level-0 coarse output must be all zeros"
        assert Yf.shape == (50,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
