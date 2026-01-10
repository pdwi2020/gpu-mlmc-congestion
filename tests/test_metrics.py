"""
Unit tests for performance metrics modules.

Tests cover:
- Delay metrics computation
- Congestion detection and analysis
- Uncertainty quantification
- Bootstrap confidence intervals
- Variance reduction analysis
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics.delay import DelayCalculator, DelayMetrics, compute_delay_variance_reduction
from metrics.congestion import (
    CongestionAnalyzer, CongestionMetrics, CongestionEvent,
    compute_congestion_probability, compare_congestion_scenarios
)
from metrics.uncertainty import (
    UncertaintyQuantifier, UncertaintyBand, VarianceReductionMetrics,
    compute_coverage_probability
)
from network.topology import NetworkGraph, TopologyGenerator


class TestDelayCalculator:
    """Tests for delay metrics."""

    @pytest.fixture
    def simple_network(self):
        """Create simple test network."""
        generator = TopologyGenerator(seed=42)
        network = generator.generate_line_graph(n_nodes=5)

        # Set uniform link properties
        network.set_link_properties(
            bandwidth_range=(1e9, 1e9),  # 1 Gbps
            delay_range=(0.001, 0.001),  # 1 ms
            capacity_range=(1000, 1000),  # 1000 pkt/s
            seed=42
        )

        return network

    def test_initialization(self, simple_network):
        """Test delay calculator initialization."""
        calc = DelayCalculator(simple_network, confidence_level=0.95)
        assert calc.network == simple_network
        assert calc.confidence_level == 0.95

    def test_compute_queueing_delay(self, simple_network):
        """Test Little's Law: W = Q/μ."""
        calc = DelayCalculator(simple_network)

        # Q = 10, μ = 5 → W = 2
        delay = calc.compute_queueing_delay(queue_length=10.0, service_rate=5.0)
        assert delay == pytest.approx(2.0)

        # Q = 0 → W = 0
        delay = calc.compute_queueing_delay(queue_length=0.0, service_rate=10.0)
        assert delay == 0.0

        # Invalid service rate
        with pytest.raises(ValueError):
            calc.compute_queueing_delay(10.0, service_rate=0.0)

    def test_compute_transmission_delay(self, simple_network):
        """Test transmission delay: D = L/B."""
        calc = DelayCalculator(simple_network)

        # 1500 bytes = 12000 bits, 1 Gbps = 1e9 bps
        packet_size = 1500 * 8  # bits
        bandwidth = 1e9  # bps
        expected_delay = packet_size / bandwidth  # 12 microseconds

        delay = calc.compute_transmission_delay(packet_size, bandwidth)
        assert delay == pytest.approx(expected_delay)

    def test_compute_link_delay(self, simple_network):
        """Test total link delay computation."""
        calc = DelayCalculator(simple_network)

        # Test link 0->1
        delay = calc.compute_link_delay(0, 1, queue_length=5.0)

        # Should include queueing + propagation + transmission
        assert delay > 0.0
        assert delay < 1.0  # Should be < 1 second for small queue

        # Test non-existent link
        with pytest.raises(ValueError):
            calc.compute_link_delay(0, 99)

    def test_compute_path_delay(self, simple_network):
        """Test path delay computation."""
        calc = DelayCalculator(simple_network)

        # Line graph: 0-1-2-3-4
        path = [0, 1, 2, 3, 4]
        queue_states = {i: 5.0 for i in range(5)}

        delay = calc.compute_path_delay(path, queue_states)

        # Should be sum of 4 link delays
        assert delay > 0.0

        # Empty path
        delay = calc.compute_path_delay([0], queue_states)
        assert delay == 0.0

    def test_compute_end_to_end_delay(self, simple_network):
        """Test end-to-end delay with shortest path."""
        calc = DelayCalculator(simple_network)

        queue_states = {i: 2.0 for i in range(5)}

        # Delay from node 0 to node 4 (should use shortest path)
        delay = calc.compute_end_to_end_delay(0, 4, queue_states)
        assert delay > 0.0
        assert np.isfinite(delay)

        # Same source and destination
        delay = calc.compute_end_to_end_delay(0, 0, queue_states)
        assert delay == 0.0

    def test_estimate_delay_distribution(self, simple_network):
        """Test delay distribution estimation."""
        calc = DelayCalculator(simple_network)

        # Generate synthetic delay samples
        np.random.seed(42)
        samples = np.random.exponential(scale=0.01, size=1000)  # Mean 10 ms

        metrics = calc.estimate_delay_distribution(samples)

        assert isinstance(metrics, DelayMetrics)
        assert metrics.mean_delay == pytest.approx(0.01, abs=0.001)
        assert metrics.median_delay > 0
        assert metrics.std_delay > 0
        assert metrics.min_delay >= 0
        assert metrics.max_delay > metrics.min_delay
        assert metrics.ci_lower < metrics.ci_upper
        assert metrics.n_samples == 1000
        assert 'p95' in metrics.percentiles

    def test_estimate_delay_with_infinite_values(self, simple_network):
        """Test handling of infinite delays (no path)."""
        calc = DelayCalculator(simple_network)

        samples = np.array([0.01, 0.02, np.inf, 0.015, np.inf])

        # Should filter out infinite values
        metrics = calc.estimate_delay_distribution(samples)
        assert metrics.n_samples == 3
        assert np.isfinite(metrics.mean_delay)

        # All infinite
        samples_inf = np.array([np.inf, np.inf, np.inf])
        with pytest.raises(ValueError, match="no valid paths"):
            calc.estimate_delay_distribution(samples_inf)

    def test_confidence_interval(self, simple_network):
        """Test confidence interval computation."""
        calc = DelayCalculator(simple_network)

        np.random.seed(42)
        samples = np.random.normal(loc=10.0, scale=2.0, size=100)

        ci_lower, ci_upper = calc.compute_confidence_interval(samples, 0.95)

        assert ci_lower < np.mean(samples) < ci_upper
        assert ci_upper - ci_lower > 0

    def test_compare_delay_distributions(self, simple_network):
        """Test statistical comparison of delay distributions."""
        calc = DelayCalculator(simple_network)

        np.random.seed(42)
        samples1 = np.random.normal(loc=10.0, scale=2.0, size=200)
        samples2 = np.random.normal(loc=12.0, scale=2.0, size=200)  # Different mean

        comparison = calc.compare_delay_distributions(
            samples1, samples2, "Scenario A", "Scenario B"
        )

        assert 't_test' in comparison
        assert 'mann_whitney' in comparison
        assert 'effect_size' in comparison
        assert comparison['mean_difference'] != 0
        assert 'significant' in comparison['t_test']

    def test_variance_reduction(self):
        """Test variance reduction computation."""
        np.random.seed(42)
        mc_samples = np.random.normal(loc=10.0, scale=2.0, size=1000)
        mlmc_samples = np.random.normal(loc=10.0, scale=0.5, size=1000)  # Lower variance

        result = compute_delay_variance_reduction(mc_samples, mlmc_samples)

        assert result['mc_variance'] > result['mlmc_variance']
        assert result['reduction_factor'] > 1.0
        assert result['reduction_percent'] > 0


class TestCongestionAnalyzer:
    """Tests for congestion metrics."""

    @pytest.fixture
    def simple_network(self):
        """Create simple test network."""
        generator = TopologyGenerator(seed=42)
        return generator.generate_erdos_renyi(n_nodes=10, p=0.3, directed=False)

    def test_initialization(self, simple_network):
        """Test congestion analyzer initialization."""
        analyzer = CongestionAnalyzer(simple_network, congestion_threshold=0.8)
        assert analyzer.network == simple_network
        assert analyzer.congestion_threshold == 0.8

    def test_compute_queue_lengths(self, simple_network):
        """Test queue length extraction."""
        analyzer = CongestionAnalyzer(simple_network)

        # Simulation states: [n_samples, n_timesteps, n_nodes]
        states = np.random.exponential(scale=2.0, size=(100, 50, 10))

        queues = analyzer.compute_queue_lengths(states)

        assert queues.shape == states.shape
        assert np.all(queues >= 0)  # Non-negative

    def test_compute_link_utilization(self, simple_network):
        """Test utilization: ρ = λ/μ."""
        analyzer = CongestionAnalyzer(simple_network)

        # Scalar rates
        util = analyzer.compute_link_utilization(arrival_rates=8.0, service_rates=10.0)
        assert util == pytest.approx(0.8)

        # Array rates
        arrival_rates = np.array([5.0, 9.0, 12.0])
        service_rates = np.array([10.0, 10.0, 10.0])
        utils = analyzer.compute_link_utilization(arrival_rates, service_rates)

        assert utils[0] == pytest.approx(0.5)
        assert utils[1] == pytest.approx(0.9)
        assert utils[2] == pytest.approx(1.2)

        # Handle division by zero
        utils = analyzer.compute_link_utilization(5.0, 0.0)
        assert utils == 0.0

    def test_detect_congestion_events(self, simple_network):
        """Test congestion event detection."""
        analyzer = CongestionAnalyzer(simple_network, queue_threshold=5.0)

        # Create queue time series with congestion events
        times = np.linspace(0, 10, 100)
        queue = np.zeros(100)

        # Event 1: t=2 to t=4
        queue[20:40] = 8.0

        # Event 2: t=6 to t=8
        queue[60:80] = 10.0

        events = analyzer.detect_congestion_events(queue, times, node_id=3, threshold=5.0)

        assert len(events) == 2
        assert isinstance(events[0], CongestionEvent)
        assert events[0].node_id == 3
        assert events[0].duration > 0
        assert events[0].peak_queue >= 8.0

    def test_identify_congested_nodes(self, simple_network):
        """Test congested node identification."""
        analyzer = CongestionAnalyzer(simple_network, congestion_threshold=0.8)

        utilization = {
            0: 0.5,   # Not congested
            1: 0.85,  # Congested
            2: 0.95,  # Congested
            3: 0.7    # Not congested
        }

        congested = analyzer.identify_congested_nodes(utilization)

        assert 1 in congested
        assert 2 in congested
        assert 0 not in congested
        assert 3 not in congested

    def test_measure_congestion_spread(self, simple_network):
        """Test congestion propagation measurement."""
        analyzer = CongestionAnalyzer(simple_network)

        # No congestion
        metrics = analyzer.measure_congestion_spread(set())
        assert metrics['n_congested'] == 0
        assert metrics['congestion_fraction'] == 0.0

        # Some nodes congested
        congested_nodes = {0, 1, 2}
        metrics = analyzer.measure_congestion_spread(congested_nodes)

        assert metrics['n_congested'] == 3
        assert metrics['congestion_fraction'] == 3 / simple_network.n_nodes
        assert metrics['largest_cluster_size'] > 0
        assert 'clusters' in metrics

    def test_analyze_simulation_congestion(self, simple_network):
        """Test comprehensive congestion analysis."""
        analyzer = CongestionAnalyzer(simple_network, congestion_threshold=0.8)

        # Create synthetic queue states
        n_timesteps = 100
        n_nodes = simple_network.n_nodes

        queue_states = np.random.exponential(scale=3.0, size=(n_timesteps, n_nodes))
        times = np.linspace(0, 10, n_timesteps)

        metrics = analyzer.analyze_simulation_congestion(
            queue_states,
            arrival_rates=8.0,
            service_rates=10.0,
            times=times
        )

        assert isinstance(metrics, CongestionMetrics)
        assert metrics.mean_queue_length > 0
        assert metrics.max_queue_length >= metrics.mean_queue_length
        assert metrics.mean_utilization == pytest.approx(0.8)
        assert len(metrics.congestion_events) >= 0

    def test_compute_temporal_evolution(self, simple_network):
        """Test temporal congestion evolution analysis."""
        analyzer = CongestionAnalyzer(simple_network)

        # Increasing congestion over time
        n_timesteps = 100
        n_nodes = simple_network.n_nodes
        times = np.linspace(0, 10, n_timesteps)

        queue_states = np.zeros((n_timesteps, n_nodes))
        for t in range(n_timesteps):
            # Linearly increasing
            queue_states[t, :] = 2.0 + 0.05 * t + np.random.normal(0, 0.5, n_nodes)

        evolution = analyzer.compute_temporal_congestion_evolution(
            queue_states, times, window_size=10
        )

        assert 'mean_queue' in evolution
        assert 'trend' in evolution
        assert evolution['trend'] == 'increasing'
        assert evolution['trend_slope'] > 0

    def test_identify_bottlenecks(self, simple_network):
        """Test bottleneck identification."""
        analyzer = CongestionAnalyzer(simple_network)

        n_timesteps = 100
        n_nodes = simple_network.n_nodes

        # Most nodes have low queue, some have high
        queue_states = np.random.exponential(scale=1.0, size=(n_timesteps, n_nodes))
        queue_states[:, 3] += 10.0  # Node 3 is a bottleneck
        queue_states[:, 7] += 8.0   # Node 7 is a bottleneck

        bottlenecks = analyzer.identify_bottlenecks(queue_states, percentile=90)

        assert len(bottlenecks) > 0
        # Should include nodes 3 and 7
        bottleneck_ids = [b[0] for b in bottlenecks]
        assert 3 in bottleneck_ids or 7 in bottleneck_ids

    def test_compute_congestion_probability(self):
        """Test congestion probability computation."""
        np.random.seed(42)
        queue_samples = np.random.exponential(scale=5.0, size=1000)
        threshold = 10.0

        prob = compute_congestion_probability(queue_samples, threshold)

        # Should be approximately 1 - CDF(threshold)
        assert 0 <= prob <= 1

    def test_compare_congestion_scenarios(self, simple_network):
        """Test scenario comparison."""
        analyzer = CongestionAnalyzer(simple_network)

        # Create two scenarios
        queue1 = np.random.exponential(scale=2.0, size=(50, 10))
        queue2 = np.random.exponential(scale=4.0, size=(50, 10))  # More congested

        metrics1 = analyzer.analyze_simulation_congestion(queue1)
        metrics2 = analyzer.analyze_simulation_congestion(queue2)

        comparison = compare_congestion_scenarios(metrics1, metrics2, "Light", "Heavy")

        assert 'mean_queue_diff' in comparison
        assert 'mean_queue_pct_change' in comparison
        assert comparison['mean_queue_diff'] < 0  # metrics1 has lower queue


class TestUncertaintyQuantifier:
    """Tests for uncertainty quantification."""

    def test_initialization(self):
        """Test UQ initialization."""
        uq = UncertaintyQuantifier(confidence_level=0.95, n_bootstrap=500, random_seed=42)
        assert uq.confidence_level == 0.95
        assert uq.n_bootstrap == 500

    def test_bootstrap_confidence_interval(self):
        """Test bootstrap CI computation."""
        uq = UncertaintyQuantifier(n_bootstrap=1000, random_seed=42)

        np.random.seed(42)
        samples = np.random.normal(loc=10.0, scale=2.0, size=200)

        ci_lower, ci_upper, bootstrap_dist = uq.bootstrap_confidence_interval(
            samples, statistic=np.mean, method='percentile'
        )

        assert ci_lower < np.mean(samples) < ci_upper
        assert len(bootstrap_dist) == 1000
        assert ci_upper - ci_lower > 0

    def test_bootstrap_methods(self):
        """Test different bootstrap methods."""
        uq = UncertaintyQuantifier(n_bootstrap=500, random_seed=42)

        np.random.seed(42)
        samples = np.random.normal(loc=10.0, scale=2.0, size=100)

        # Percentile method
        ci1 = uq.bootstrap_confidence_interval(samples, method='percentile')[:2]

        # Basic method
        ci2 = uq.bootstrap_confidence_interval(samples, method='basic')[:2]

        # Both should give reasonable intervals
        assert ci1[0] < ci1[1]
        assert ci2[0] < ci2[1]

    def test_prediction_interval(self):
        """Test prediction interval computation."""
        uq = UncertaintyQuantifier(random_seed=42)

        np.random.seed(42)
        samples = np.random.normal(loc=10.0, scale=2.0, size=200)

        # Prediction interval (wider than CI)
        pi_lower, pi_upper = uq.compute_prediction_interval(samples, method='quantile')

        # CI for comparison
        ci_lower, ci_upper, _ = uq.bootstrap_confidence_interval(samples)

        # PI should be wider
        assert pi_lower <= ci_lower
        assert pi_upper >= ci_upper

    def test_compute_uncertainty_band(self):
        """Test uncertainty band for time series."""
        uq = UncertaintyQuantifier(confidence_level=0.95, random_seed=42)

        # Generate sample trajectories
        np.random.seed(42)
        n_samples = 100
        n_timesteps = 50

        # All samples follow similar trajectory with noise
        t = np.linspace(0, 10, n_timesteps)
        true_trajectory = np.sin(t)

        samples = np.zeros((n_samples, n_timesteps))
        for i in range(n_samples):
            samples[i] = true_trajectory + np.random.normal(0, 0.2, n_timesteps)

        band = uq.compute_uncertainty_band(samples, method='quantile')

        assert isinstance(band, UncertaintyBand)
        assert len(band.times) == n_timesteps
        assert len(band.mean) == n_timesteps
        assert len(band.lower) == n_timesteps
        assert len(band.upper) == n_timesteps
        assert np.all(band.lower <= band.upper)

        # Mean should be close to true trajectory
        assert np.allclose(band.mean, true_trajectory, atol=0.5)

    def test_analyze_variance_reduction(self):
        """Test variance reduction analysis."""
        uq = UncertaintyQuantifier()

        np.random.seed(42)
        mc_samples = np.random.normal(loc=10.0, scale=2.0, size=1000)
        mlmc_samples = np.random.normal(loc=10.0, scale=0.5, size=1000)  # Lower var

        mc_cost = 100.0
        mlmc_cost = 50.0

        metrics = uq.analyze_variance_reduction(
            mc_samples, mlmc_samples, mc_cost, mlmc_cost
        )

        assert isinstance(metrics, VarianceReductionMetrics)
        assert metrics.mc_variance > metrics.mlmc_variance
        assert metrics.reduction_factor > 1.0
        assert metrics.reduction_percent > 0
        assert metrics.efficiency_gain > metrics.reduction_factor  # Because MLMC is cheaper

    def test_compare_estimator_variance(self):
        """Test estimator variance comparison."""
        uq = UncertaintyQuantifier(n_bootstrap=500, random_seed=42)

        np.random.seed(42)
        estimators = {
            'MC': np.random.normal(10.0, 2.0, 1000),
            'MLMC': np.random.normal(10.0, 0.5, 1000),
            'QMC': np.random.normal(10.0, 0.3, 1000)
        }

        results = uq.compare_estimator_variance(estimators, reference='MC')

        assert 'MC' in results
        assert 'MLMC' in results
        assert 'variance_reduction' in results['MLMC']
        assert results['MLMC']['variance_reduction'] > 1.0

    def test_effective_sample_size(self):
        """Test ESS computation for autocorrelated samples."""
        uq = UncertaintyQuantifier()

        # Independent samples
        np.random.seed(42)
        independent = np.random.normal(0, 1, 1000)
        ess_indep = uq.compute_effective_sample_size(independent)

        # Should be close to n
        assert ess_indep > 900

        # Autocorrelated samples
        autocorrelated = np.zeros(1000)
        autocorrelated[0] = np.random.normal()
        for i in range(1, 1000):
            autocorrelated[i] = 0.9 * autocorrelated[i-1] + 0.1 * np.random.normal()

        ess_auto = uq.compute_effective_sample_size(autocorrelated)

        # Should be much less than n
        assert ess_auto < 500

    def test_diagnose_convergence(self):
        """Test convergence diagnostics."""
        uq = UncertaintyQuantifier()

        # Converged samples (large n, i.i.d.)
        np.random.seed(42)
        samples = np.random.normal(10.0, 2.0, 10000)

        diagnostics = uq.diagnose_convergence(samples)

        assert 'converged' in diagnostics
        assert 'relative_variance' in diagnostics
        assert 'effective_sample_size' in diagnostics

        # Small sample (not converged)
        small_samples = np.random.normal(10.0, 2.0, 10)
        diagnostics_small = uq.diagnose_convergence(small_samples)

        assert diagnostics_small['converged'] == False

    def test_coverage_probability(self):
        """Test coverage probability computation."""
        np.random.seed(42)

        # Generate data with known distribution
        true_values = np.full(100, 10.0)
        ci_lower = np.random.normal(9.0, 0.5, 100)
        ci_upper = np.random.normal(11.0, 0.5, 100)

        coverage = compute_coverage_probability(true_values, ci_lower, ci_upper)

        # Should be high since intervals are wide
        assert 0 <= coverage <= 1
        assert coverage > 0.8


class TestIntegration:
    """Integration tests combining multiple metrics."""

    def test_full_metrics_pipeline(self):
        """Test complete metrics analysis pipeline."""
        # Create network
        generator = TopologyGenerator(seed=42)
        network = generator.generate_erdos_renyi(n_nodes=20, p=0.2)

        # Initialize all analyzers
        delay_calc = DelayCalculator(network, confidence_level=0.95)
        congestion_analyzer = CongestionAnalyzer(network, congestion_threshold=0.8)
        uq = UncertaintyQuantifier(confidence_level=0.95, n_bootstrap=200, random_seed=42)

        # Generate synthetic simulation data
        np.random.seed(42)
        n_samples = 100
        n_timesteps = 50
        n_nodes = network.n_nodes

        queue_states = np.random.exponential(scale=2.0, size=(n_samples, n_timesteps, n_nodes))

        # Analyze congestion
        # Use mean over samples for congestion analysis
        mean_queue_states = np.mean(queue_states, axis=0)
        congestion_metrics = congestion_analyzer.analyze_simulation_congestion(
            mean_queue_states,
            arrival_rates=8.0,
            service_rates=10.0
        )

        assert congestion_metrics.mean_queue_length > 0

        # Compute uncertainty band
        # Time evolution of mean queue
        mean_queue_over_nodes = np.mean(queue_states, axis=2)  # [n_samples, n_timesteps]
        uncertainty_band = uq.compute_uncertainty_band(mean_queue_over_nodes)

        assert uncertainty_band.mean.shape[0] == n_timesteps
        assert np.all(uncertainty_band.lower <= uncertainty_band.upper)

        # Delay analysis
        # Generate delay samples
        delay_samples = np.random.exponential(scale=0.02, size=n_samples)
        delay_metrics = delay_calc.estimate_delay_distribution(delay_samples)

        assert delay_metrics.mean_delay > 0
        assert delay_metrics.ci_lower < delay_metrics.ci_upper


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
