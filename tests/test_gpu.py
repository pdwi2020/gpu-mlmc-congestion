"""
Unit tests for GPU acceleration modules.

Tests cover:
- GPU memory management
- CUDA kernel compilation and execution
- GPU Monte Carlo simulation
- GPU MLMC simulation
- CPU-GPU consistency

Note: These tests require PyCUDA and a CUDA-capable GPU.
If GPU is not available, tests will be skipped.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try importing GPU modules
try:
    from gpu.memory_mgmt import (
        GPUMemoryManager,
        GPUMemoryPool,
        estimate_memory_requirements,
        optimize_batch_size,
        PYCUDA_AVAILABLE
    )
    from gpu.cuda_kernels import (
        GPUQueueSimulator,
        CUDAKernelCompiler,
        get_optimal_block_size
    )
    from gpu.parallel_mc import (
        GPUMonteCarloSimulator,
        GPUMLMCSimulator
    )
    GPU_TESTS_AVAILABLE = PYCUDA_AVAILABLE
except ImportError:
    GPU_TESTS_AVAILABLE = False

from network.topology import TopologyGenerator
from network.traffic import PoissonTraffic


# Skip all tests if GPU not available
pytestmark = pytest.mark.skipif(
    not GPU_TESTS_AVAILABLE,
    reason="GPU/PyCUDA not available"
)


@pytest.mark.gpu
class TestGPUMemoryManager:
    """Tests for GPU memory management."""

    def test_initialization(self):
        """Test GPU memory manager initialization."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        manager = GPUMemoryManager()
        assert manager.device_name is not None
        assert manager.total_memory > 0
        assert manager.free_memory > 0

    def test_allocate_device_array(self):
        """Test GPU array allocation."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        manager = GPUMemoryManager()

        arr = manager.allocate_device_array(
            shape=(100, 50),
            dtype=np.float32,
            name="test_array"
        )

        assert arr.shape == (100, 50)
        assert arr.dtype == np.float32

        manager.free_device_array(arr)

    def test_transfer_to_device(self):
        """Test host-to-device transfer."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        manager = GPUMemoryManager()

        host_array = np.random.randn(50, 30).astype(np.float32)
        device_array = manager.transfer_to_device(host_array, name="test_transfer")

        assert device_array.shape == host_array.shape

        manager.free_device_array(device_array)

    def test_transfer_roundtrip(self):
        """Test host → device → host roundtrip."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        manager = GPUMemoryManager()

        original = np.random.randn(100, 80).astype(np.float32)
        device_array = manager.transfer_to_device(original)
        retrieved = manager.transfer_to_host(device_array)

        assert np.allclose(original, retrieved, rtol=1e-6)

        manager.free_device_array(device_array)

    def test_get_memory_info(self):
        """Test memory info retrieval."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        manager = GPUMemoryManager()
        info = manager.get_memory_info()

        assert 'device_name' in info
        assert 'total_memory_gb' in info
        assert 'free_memory_gb' in info
        assert info['total_memory_gb'] > 0


@pytest.mark.gpu
class TestGPUMemoryPool:
    """Tests for GPU memory pool."""

    def test_pool_reuse(self):
        """Test memory pool array reuse."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        manager = GPUMemoryManager()
        pool = GPUMemoryPool(manager)

        # Get array from pool (will allocate)
        arr1 = pool.get_array(shape=(100, 50), dtype=np.float32)
        arr1_id = id(arr1)

        # Return to pool
        pool.return_array(arr1)

        # Get again (should reuse)
        arr2 = pool.get_array(shape=(100, 50), dtype=np.float32)
        arr2_id = id(arr2)

        # Should be same object (reused)
        assert arr1_id == arr2_id

        pool.clear_pool()


@pytest.mark.gpu
class TestMemoryUtilities:
    """Tests for memory utility functions."""

    def test_estimate_memory_requirements(self):
        """Test memory estimation."""
        estimates = estimate_memory_requirements(
            n_paths=1000,
            n_timesteps=500,
            n_nodes=10,
            dtype=np.float32
        )

        assert 'total_gb' in estimates
        assert estimates['total_gb'] > 0
        assert estimates['total_with_overhead_gb'] > estimates['total_gb']

    def test_optimize_batch_size(self):
        """Test batch size optimization."""
        batch_size = optimize_batch_size(
            total_samples=10000,
            n_timesteps=1000,
            n_nodes=100,
            available_memory_gb=10.0,
            dtype=np.float32
        )

        assert batch_size > 0
        assert batch_size <= 10000


@pytest.mark.gpu
class TestCUDAKernels:
    """Tests for CUDA kernel compilation and execution."""

    def test_kernel_compilation(self):
        """Test CUDA kernel compilation."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        compiler = CUDAKernelCompiler()
        assert 'simulate_queue_dynamics' in compiler.kernels
        assert 'compute_queue_metrics' in compiler.kernels
        assert 'simulate_coupled_paths' in compiler.kernels

    def test_get_optimal_block_size(self):
        """Test optimal block size detection."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        block_size = get_optimal_block_size()
        assert block_size > 0
        assert block_size <= 1024  # Max block size


@pytest.mark.gpu
class TestGPUQueueSimulator:
    """Tests for GPU queue simulator."""

    def test_initialization(self):
        """Test GPU simulator initialization."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        simulator = GPUQueueSimulator()
        assert simulator.compiler is not None

    def test_simulate_paths(self):
        """Test parallel path simulation on GPU."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        simulator = GPUQueueSimulator()

        results = simulator.simulate_paths(
            n_paths=100,
            n_timesteps=100,
            arrival_rate=8.0,
            service_rate=10.0,
            noise_intensity=0.5,
            dt=0.01,
            metric='mean'
        )

        assert len(results) == 100
        assert np.all(results >= 0)  # Queue lengths should be non-negative

    def test_different_metrics(self):
        """Test different metric computations."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        simulator = GPUQueueSimulator()

        for metric in ['mean', 'max', 'final']:
            results = simulator.simulate_paths(
                n_paths=50,
                n_timesteps=100,
                arrival_rate=8.0,
                service_rate=10.0,
                noise_intensity=0.5,
                dt=0.01,
                metric=metric
            )

            assert len(results) == 50
            assert np.all(np.isfinite(results))

    def test_coupled_paths_mlmc(self):
        """Test MLMC coupled path simulation."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        simulator = GPUQueueSimulator()

        fine_results, coarse_results = simulator.simulate_coupled_paths_mlmc(
            n_paths=100,
            n_timesteps_fine=200,
            arrival_rate=8.0,
            service_rate=10.0,
            noise_intensity=0.5,
            dt_fine=0.01,
            dt_coarse=0.02
        )

        assert len(fine_results) == 100
        assert len(coarse_results) == 100

        # Paths should be correlated but not identical
        correlation = np.corrcoef(fine_results, coarse_results)[0, 1]
        assert correlation > 0.5  # Should be positively correlated


@pytest.mark.gpu
class TestGPUMonteCarloSimulator:
    """Tests for GPU Monte Carlo simulator."""

    @pytest.fixture
    def setup_network_traffic(self):
        """Setup network and traffic."""
        gen = TopologyGenerator(seed=42)
        network = gen.generate_erdos_renyi(n_nodes=20, p=0.2)
        network.set_link_properties(seed=42)

        traffic = PoissonTraffic(rate=5.0, seed=42)

        return network, traffic

    def test_initialization(self):
        """Test GPU MC simulator initialization."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        simulator = GPUMonteCarloSimulator()
        assert simulator.gpu_simulator is not None

    def test_estimate(self, setup_network_traffic):
        """Test GPU Monte Carlo estimation."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        network, traffic = setup_network_traffic
        simulator = GPUMonteCarloSimulator()

        result = simulator.estimate(
            network=network,
            traffic=traffic,
            n_samples=200,
            T=5.0,
            dt=0.1,
            verbose=False
        )

        assert result.n_samples == 200
        assert result.mean >= 0.0
        assert result.variance >= 0.0
        assert result.ci_lower <= result.mean <= result.ci_upper

        # Check GPU metadata
        assert 'gpu_time_seconds' in result.metadata
        assert result.metadata['gpu_time_seconds'] > 0

    def test_batching(self, setup_network_traffic):
        """Test batched GPU execution."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        network, traffic = setup_network_traffic
        simulator = GPUMonteCarloSimulator()

        # Force small batch size
        result = simulator.estimate(
            network=network,
            traffic=traffic,
            n_samples=150,
            T=3.0,
            dt=0.1,
            batch_size=50,  # Process in 3 batches
            verbose=False
        )

        assert result.n_samples == 150
        assert result.metadata['batch_size'] == 50
        assert result.metadata['n_batches'] == 3


@pytest.mark.gpu
class TestGPUMLMCSimulator:
    """Tests for GPU MLMC simulator."""

    @pytest.fixture
    def setup_network_traffic(self):
        """Setup network and traffic."""
        gen = TopologyGenerator(seed=42)
        network = gen.generate_erdos_renyi(n_nodes=20, p=0.2)
        network.set_link_properties(seed=42)

        traffic = PoissonTraffic(rate=5.0, seed=42)

        return network, traffic

    def test_initialization(self):
        """Test GPU MLMC simulator initialization."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        simulator = GPUMLMCSimulator(refinement_factor=2)
        assert simulator.refinement_factor == 2

    def test_run_coupled_paths_gpu(self, setup_network_traffic):
        """Test GPU coupled path generation."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        network, traffic = setup_network_traffic
        simulator = GPUMLMCSimulator(refinement_factor=2)

        # Level 0
        Y_fine, Y_coarse = simulator.run_coupled_paths_gpu(
            network=network,
            traffic=traffic,
            level=0,
            n_samples=50,
            T=5.0,
            base_dt=0.2
        )

        assert len(Y_fine) == 50
        assert np.all(Y_coarse == 0)  # Y_{-1} = 0

        # Level 1
        Y_fine, Y_coarse = simulator.run_coupled_paths_gpu(
            network=network,
            traffic=traffic,
            level=1,
            n_samples=50,
            T=5.0,
            base_dt=0.2
        )

        assert len(Y_fine) == 50
        assert len(Y_coarse) == 50
        # Should be correlated
        assert not np.array_equal(Y_fine, Y_coarse)

    def test_mlmc_estimate_gpu(self, setup_network_traffic):
        """Test full GPU MLMC estimation."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        network, traffic = setup_network_traffic
        simulator = GPUMLMCSimulator(refinement_factor=2)

        result = simulator.mlmc_estimate_gpu(
            network=network,
            traffic=traffic,
            epsilon=0.05,
            L_max=2,
            T=5.0,
            base_dt=0.2,
            pilot_samples=20,
            verbose=False
        )

        assert result.L_max == 2
        assert len(result.level_stats) == 3  # L=0,1,2
        assert result.estimate >= 0.0
        assert result.variance >= 0.0

        # Check GPU metadata
        assert result.metadata['device'] == 'GPU'
        assert result.metadata['gpu_time_seconds'] > 0


@pytest.mark.gpu
@pytest.mark.slow
class TestCPUGPUConsistency:
    """Test consistency between CPU and GPU implementations."""

    @pytest.fixture
    def setup_network_traffic(self):
        """Setup network and traffic."""
        gen = TopologyGenerator(seed=42)
        network = gen.generate_erdos_renyi(n_nodes=20, p=0.2)
        network.set_link_properties(seed=42)

        traffic = PoissonTraffic(rate=5.0, seed=42)

        return network, traffic

    def test_monte_carlo_consistency(self, setup_network_traffic):
        """Test GPU MC gives similar results to CPU MC."""
        if not GPU_TESTS_AVAILABLE:
            pytest.skip("GPU not available")

        from simulation.monte_carlo import MonteCarloSimulator

        network, traffic = setup_network_traffic

        # CPU simulation
        cpu_sim = MonteCarloSimulator(seed=42)
        cpu_result = cpu_sim.estimate(
            network, traffic, n_samples=500, T=5.0, dt=0.1, verbose=False
        )

        # GPU simulation
        gpu_sim = GPUMonteCarloSimulator(seed=42)
        gpu_result = gpu_sim.estimate(
            network, traffic, n_samples=500, T=5.0, dt=0.1, verbose=False
        )

        # Results should be statistically similar (not identical due to RNG)
        # Check if means are within 3 standard errors
        pooled_std = np.sqrt(cpu_result.variance / 500 + gpu_result.variance / 500)
        diff = abs(cpu_result.mean - gpu_result.mean)

        assert diff < 3 * pooled_std, (
            f"CPU and GPU results differ significantly: "
            f"CPU={cpu_result.mean:.4f}, GPU={gpu_result.mean:.4f}, diff={diff:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "gpu"])
