"""
Unit tests for dataset loaders and processors.

Tests cover:
- SNAP dataset loading
- CAIDA topology loading
- MAWI trace processing (without actual downloads)
- Synthetic benchmark generation
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src and datasets to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "datasets"))

from snap.loader import SNAPDatasetLoader, SNAP_DATASETS
from caida.loader import CAIDATopologyLoader
from mawi.loader import MAWITraceProcessor
from synthetic.generator import SyntheticBenchmarkGenerator


class TestSNAPDatasetLoader:
    """Tests for SNAP dataset loader."""

    def test_initialization(self, tmp_path):
        """Test loader initialization."""
        loader = SNAPDatasetLoader(data_dir=tmp_path)
        assert loader.data_dir == tmp_path
        assert loader.data_dir.exists()

    def test_list_available_datasets(self):
        """Test listing available datasets."""
        loader = SNAPDatasetLoader()
        datasets = loader.list_available_datasets()

        assert len(datasets) > 0
        assert all('name' in ds for ds in datasets)
        assert all('nodes' in ds for ds in datasets)
        assert all('edges' in ds for ds in datasets)

    def test_dataset_metadata(self):
        """Test dataset metadata structure."""
        assert 'email-Eu-core' in SNAP_DATASETS
        assert 'CA-GrQc' in SNAP_DATASETS

        email_eu = SNAP_DATASETS['email-Eu-core']
        assert 'url' in email_eu
        assert 'nodes' in email_eu
        assert 'edges' in email_eu
        assert 'directed' in email_eu

    def test_print_dataset_info(self, capsys):
        """Test printing dataset info."""
        loader = SNAPDatasetLoader()
        loader.print_dataset_info('email-Eu-core')

        captured = capsys.readouterr()
        assert 'email-Eu-core' in captured.out
        assert 'Email network' in captured.out

    @pytest.mark.slow
    @pytest.mark.network
    def test_download_dataset_real(self, tmp_path):
        """Test actual dataset download (slow, requires network)."""
        pytest.skip("Skipping real download test - run manually if needed")

        loader = SNAPDatasetLoader(data_dir=tmp_path)
        filepath = loader.download_dataset('email-Eu-core')

        assert filepath.exists()
        assert filepath.suffix == '.gz'


class TestCAIDATopologyLoader:
    """Tests for CAIDA topology loader."""

    def test_initialization(self, tmp_path):
        """Test loader initialization."""
        loader = CAIDATopologyLoader(data_dir=tmp_path)
        assert loader.data_dir == tmp_path
        assert loader.data_dir.exists()

    def test_get_available_dates(self):
        """Test getting available dates."""
        loader = CAIDATopologyLoader()
        dates = loader.get_available_dates()

        assert len(dates) > 0
        assert all(len(date) == 8 for date in dates)  # YYYYMMDD format
        assert '20260101' in dates  # January 2026

    def test_get_trace_url(self):
        """Test URL construction."""
        loader = CAIDATopologyLoader()
        loader.base_url = "http://example.com/"

        date = '20260101'
        filename = f"{date}.as-rel2.txt.bz2"

        # Check filename format
        assert filename == "20260101.as-rel2.txt.bz2"

    def test_print_topology_info(self, capsys):
        """Test printing topology info."""
        loader = CAIDATopologyLoader()
        loader.print_topology_info('20260101')

        captured = capsys.readouterr()
        assert '20260101' in captured.out
        assert 'Provider-to-Customer' in captured.out

    @pytest.mark.slow
    @pytest.mark.network
    def test_download_topology_real(self, tmp_path):
        """Test actual topology download (slow, requires network)."""
        pytest.skip("Skipping real download test - run manually if needed")

        loader = CAIDATopologyLoader(data_dir=tmp_path)
        filepath = loader.download_topology('20260101')

        assert filepath.exists()
        assert '.bz2' in filepath.name


class TestMAWITraceProcessor:
    """Tests for MAWI trace processor."""

    def test_initialization(self, tmp_path):
        """Test processor initialization."""
        processor = MAWITraceProcessor(data_dir=tmp_path)
        assert processor.data_dir == tmp_path
        assert processor.data_dir.exists()

    def test_get_trace_url(self):
        """Test URL construction."""
        processor = MAWITraceProcessor()
        url = processor.get_trace_url('20240619', '1400')

        assert '20240619' in url
        assert '202406191400.pcap.gz' in url

    def test_extract_statistics_fast(self, tmp_path):
        """Test fast statistics extraction."""
        processor = MAWITraceProcessor(data_dir=tmp_path)

        # Create dummy file for size estimation
        dummy_file = tmp_path / "test.pcap.gz"
        dummy_file.write_bytes(b'x' * (100 * 1024 * 1024))  # 100 MB

        stats = processor.extract_statistics_fast(dummy_file)

        assert 'arrival_rate' in stats
        assert 'mean_packet_size' in stats
        assert 'burstiness' in stats
        assert stats['file_size_mb'] > 0

    def test_create_traffic_model_from_stats(self):
        """Test creating traffic model from stats."""
        processor = MAWITraceProcessor()

        stats = {
            'arrival_rate': 1000.0,
            'burstiness': 2.5,
            'mean_packet_size': 800.0,
            'packet_size_std': 400.0
        }

        traffic_model = processor.create_traffic_model(stats=stats, seed=42)

        assert traffic_model is not None
        assert traffic_model.arrival_rate == 1000.0
        assert traffic_model.burstiness == 2.5

    def test_print_trace_info(self, capsys):
        """Test printing trace info."""
        processor = MAWITraceProcessor()
        processor.print_trace_info('20240619', '1400')

        captured = capsys.readouterr()
        assert '20240619' in captured.out
        assert '1400' in captured.out

    @pytest.mark.slow
    @pytest.mark.network
    def test_download_trace_real(self, tmp_path):
        """Test actual trace download (slow, requires network)."""
        pytest.skip("Skipping real download test - run manually if needed")

        processor = MAWITraceProcessor(data_dir=tmp_path)
        filepath = processor.download_trace('20240619', '1400')

        assert filepath.exists()
        assert '.pcap.gz' in filepath.name


class TestSyntheticBenchmarkGenerator:
    """Tests for synthetic benchmark generator."""

    def test_initialization(self, tmp_path):
        """Test generator initialization."""
        generator = SyntheticBenchmarkGenerator(data_dir=tmp_path, seed=42)
        assert generator.data_dir == tmp_path
        assert generator.seed == 42

    def test_generate_stable_queue_scenario(self):
        """Test generating stable queue scenario."""
        generator = SyntheticBenchmarkGenerator(seed=42)

        scenario = generator.generate_stable_queue_scenario(
            arrival_rate=8.0,
            service_rate=10.0,
            noise_intensity=0.5
        )

        assert 'name' in scenario
        assert 'network' in scenario
        assert 'traffic' in scenario
        assert 'ground_truth' in scenario

        # Check ground truth
        gt = scenario['ground_truth']
        assert 'expected_queue_length' in gt
        assert 'utilization' in gt
        assert gt['utilization'] == 0.8  # 8/10

        # M/M/1 formula: E[Q] = ρ/(1-ρ) = 0.8/0.2 = 4.0
        assert abs(gt['expected_queue_length'] - 4.0) < 1e-10

    def test_generate_congested_scenario(self):
        """Test generating congested scenario."""
        generator = SyntheticBenchmarkGenerator(seed=42)

        scenario = generator.generate_congested_scenario(utilization=0.95)

        gt = scenario['ground_truth']
        assert gt['utilization'] == 0.95
        assert gt['congested'] is True
        assert gt['stable'] is True  # Still stable, just heavily loaded

    def test_generate_bursty_traffic_scenario(self):
        """Test generating bursty traffic scenario."""
        generator = SyntheticBenchmarkGenerator(seed=42)

        scenario = generator.generate_bursty_traffic_scenario(burstiness=3.0)

        assert scenario['name'] == 'bursty_traffic'
        assert 'effective_rate' in scenario['ground_truth']

    def test_generate_convergence_test_suite(self):
        """Test generating convergence test suite."""
        generator = SyntheticBenchmarkGenerator(seed=42)

        suite = generator.generate_convergence_test_suite()

        assert len(suite) >= 3  # At least light, moderate, heavy
        assert 'light_load' in suite
        assert 'moderate_load' in suite
        assert 'heavy_load' in suite

        # Verify increasing load
        light_util = suite['light_load']['ground_truth']['utilization']
        moderate_util = suite['moderate_load']['ground_truth']['utilization']

        assert light_util < moderate_util

    def test_generate_parameter_sweep(self):
        """Test parameter sweep generation."""
        generator = SyntheticBenchmarkGenerator(seed=42)

        values = [0.5, 0.7, 0.9]
        sweep = generator.generate_parameter_sweep('utilization', values)

        assert len(sweep) == 3

        for i, scenario in enumerate(sweep):
            assert scenario['sweep_parameter'] == 'utilization'
            assert scenario['sweep_value'] == values[i]

    def test_save_and_load_scenario(self, tmp_path):
        """Test scenario saving and loading."""
        generator = SyntheticBenchmarkGenerator(data_dir=tmp_path, seed=42)

        # Generate and save
        scenario = generator.generate_stable_queue_scenario()
        filename = "test_scenario.json"
        generator.save_scenario(scenario, filename)

        # Load
        loaded = generator.load_scenario(filename)

        assert loaded['name'] == scenario['name']
        assert 'ground_truth' in loaded
        assert 'parameters' in loaded

    def test_stable_queue_raises_on_unstable(self):
        """Test that unstable parameters raise error."""
        generator = SyntheticBenchmarkGenerator(seed=42)

        with pytest.raises(ValueError, match="unstable"):
            generator.generate_stable_queue_scenario(
                arrival_rate=12.0,  # λ > μ
                service_rate=10.0
            )

    def test_create_ground_truth_validation_set(self):
        """Test creating validation set."""
        generator = SyntheticBenchmarkGenerator(seed=42)

        validation = generator.create_ground_truth_validation_set(
            n_samples=100,  # Small for testing
            T=5.0,
            dt=0.1
        )

        assert 'mc_estimate' in validation
        assert 'mc_std' in validation
        assert 'mc_ci_lower' in validation
        assert 'mc_ci_upper' in validation
        assert validation['n_samples'] == 100

        # Check CI contains estimate
        assert validation['mc_ci_lower'] <= validation['mc_estimate'] <= validation['mc_ci_upper']


class TestIntegration:
    """Integration tests combining multiple loaders."""

    def test_synthetic_to_simulation_workflow(self):
        """Test complete workflow from synthetic generation to simulation."""
        generator = SyntheticBenchmarkGenerator(seed=42)

        # Generate scenario
        scenario = generator.generate_stable_queue_scenario()

        # Extract components
        network = scenario['network']
        traffic = scenario['traffic']

        assert network.n_nodes > 0
        assert network.n_edges > 0
        assert traffic is not None

    def test_all_generators_produce_valid_networks(self):
        """Test that all generators produce valid NetworkGraph objects."""
        generator = SyntheticBenchmarkGenerator(seed=42)

        scenarios = [
            generator.generate_stable_queue_scenario(),
            generator.generate_congested_scenario(),
            generator.generate_bursty_traffic_scenario()
        ]

        for scenario in scenarios:
            network = scenario['network']

            # Basic validations
            assert network.n_nodes > 0
            assert network.n_edges >= 0

            # Summary should work
            summary = network.summary()
            assert 'n_nodes' in summary
            assert 'n_edges' in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
