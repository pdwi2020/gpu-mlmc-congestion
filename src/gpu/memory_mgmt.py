"""
GPU Memory Management Module

Provides utilities for efficient GPU memory allocation, transfer, and management
for large-scale Monte Carlo simulations.

Classes:
    GPUMemoryManager: Manage GPU device memory allocation and transfers
    GPUMemoryPool: Memory pool for reusable allocations
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# Check if PyCUDA is available
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda import gpuarray
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    logger.warning("PyCUDA not available. GPU acceleration disabled.")


class GPUMemoryManager:
    """
    Manage GPU device memory for Monte Carlo simulations.

    Handles allocation, transfer, and deallocation of GPU memory
    with proper error handling and memory tracking.
    """

    def __init__(self):
        """Initialize GPU memory manager."""
        self.allocated_arrays = {}  # Track allocated arrays
        self.total_allocated = 0     # Total bytes allocated

        if not PYCUDA_AVAILABLE:
            raise ImportError(
                "PyCUDA is required for GPU acceleration. "
                "Install with: pip install pycuda"
            )

        # Get GPU device info
        self.device = cuda.Device(0)
        self.device_name = self.device.name()
        self.total_memory = self.device.total_memory()
        self.free_memory, self.total_memory = cuda.mem_get_info()

        logger.info(f"GPU Device: {self.device_name}")
        logger.info(f"Total Memory: {self.total_memory / 1e9:.2f} GB")
        logger.info(f"Free Memory: {self.free_memory / 1e9:.2f} GB")

    def allocate_device_array(self,
                             shape: Tuple[int, ...],
                             dtype: np.dtype = np.float32,
                             name: Optional[str] = None) -> gpuarray.GPUArray:
        """
        Allocate GPU device array.

        Args:
            shape: Array shape
            dtype: Data type (default: float32 for GPU efficiency)
            name: Optional name for tracking

        Returns:
            GPUArray object
        """
        if not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA not available")

        # Calculate size
        size_bytes = np.prod(shape) * np.dtype(dtype).itemsize

        # Check if enough memory
        free_mem, _ = cuda.mem_get_info()
        if size_bytes > free_mem:
            raise MemoryError(
                f"Insufficient GPU memory. Required: {size_bytes/1e9:.2f} GB, "
                f"Available: {free_mem/1e9:.2f} GB"
            )

        # Allocate
        device_array = gpuarray.zeros(shape, dtype=dtype)

        # Track allocation
        array_id = id(device_array)
        self.allocated_arrays[array_id] = {
            'array': device_array,
            'shape': shape,
            'dtype': dtype,
            'size_bytes': size_bytes,
            'name': name or f"array_{array_id}"
        }
        self.total_allocated += size_bytes

        logger.debug(f"Allocated GPU array: {shape}, {dtype}, "
                    f"{size_bytes/1e6:.2f} MB, name={name}")

        return device_array

    def transfer_to_device(self,
                          host_array: np.ndarray,
                          name: Optional[str] = None) -> gpuarray.GPUArray:
        """
        Transfer host array to GPU device.

        Args:
            host_array: NumPy array on host
            name: Optional name for tracking

        Returns:
            GPUArray on device
        """
        if not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA not available")

        # Convert to appropriate dtype if needed (float32 preferred on GPU)
        if host_array.dtype != np.float32 and host_array.dtype != np.float64:
            host_array = host_array.astype(np.float32)

        device_array = gpuarray.to_gpu(host_array)

        # Track allocation
        array_id = id(device_array)
        size_bytes = host_array.nbytes
        self.allocated_arrays[array_id] = {
            'array': device_array,
            'shape': host_array.shape,
            'dtype': host_array.dtype,
            'size_bytes': size_bytes,
            'name': name or f"array_{array_id}"
        }
        self.total_allocated += size_bytes

        logger.debug(f"Transferred to GPU: {host_array.shape}, "
                    f"{size_bytes/1e6:.2f} MB, name={name}")

        return device_array

    def transfer_to_host(self, device_array: gpuarray.GPUArray) -> np.ndarray:
        """
        Transfer GPU array back to host.

        Args:
            device_array: GPUArray on device

        Returns:
            NumPy array on host
        """
        if not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA not available")

        host_array = device_array.get()

        logger.debug(f"Transferred to host: {host_array.shape}, "
                    f"{host_array.nbytes/1e6:.2f} MB")

        return host_array

    def free_device_array(self, device_array: gpuarray.GPUArray):
        """
        Free GPU device array.

        Args:
            device_array: GPUArray to free
        """
        array_id = id(device_array)

        if array_id in self.allocated_arrays:
            info = self.allocated_arrays[array_id]
            self.total_allocated -= info['size_bytes']
            del self.allocated_arrays[array_id]
            logger.debug(f"Freed GPU array: {info['name']}, "
                        f"{info['size_bytes']/1e6:.2f} MB")

        # PyCUDA will handle actual deallocation via garbage collection
        del device_array

    def free_all(self):
        """Free all tracked GPU arrays."""
        array_ids = list(self.allocated_arrays.keys())
        for array_id in array_ids:
            info = self.allocated_arrays[array_id]
            del info['array']
            del self.allocated_arrays[array_id]

        self.total_allocated = 0
        logger.info("Freed all GPU arrays")

    def get_memory_info(self) -> Dict:
        """
        Get current GPU memory usage info.

        Returns:
            Dictionary with memory statistics
        """
        free_mem, total_mem = cuda.mem_get_info()

        return {
            'device_name': self.device_name,
            'total_memory_gb': total_mem / 1e9,
            'free_memory_gb': free_mem / 1e9,
            'used_memory_gb': (total_mem - free_mem) / 1e9,
            'tracked_allocated_gb': self.total_allocated / 1e9,
            'n_tracked_arrays': len(self.allocated_arrays),
            'utilization_percent': ((total_mem - free_mem) / total_mem) * 100
        }

    def print_memory_summary(self):
        """Print GPU memory usage summary."""
        info = self.get_memory_info()

        print("=" * 60)
        print("GPU Memory Summary")
        print("=" * 60)
        print(f"Device: {info['device_name']}")
        print(f"Total Memory: {info['total_memory_gb']:.2f} GB")
        print(f"Used Memory: {info['used_memory_gb']:.2f} GB")
        print(f"Free Memory: {info['free_memory_gb']:.2f} GB")
        print(f"Utilization: {info['utilization_percent']:.1f}%")
        print(f"Tracked Arrays: {info['n_tracked_arrays']}")
        print(f"Tracked Allocation: {info['tracked_allocated_gb']:.2f} GB")
        print("=" * 60)

    def __del__(self):
        """Cleanup on destruction."""
        self.free_all()


class GPUMemoryPool:
    """
    Memory pool for efficient reuse of GPU allocations.

    Reduces overhead of repeated allocations/deallocations
    for same-sized arrays.
    """

    def __init__(self, manager: GPUMemoryManager):
        """
        Initialize memory pool.

        Args:
            manager: GPUMemoryManager instance
        """
        self.manager = manager
        self.pools: Dict[Tuple, List[gpuarray.GPUArray]] = {}

    def get_array(self,
                  shape: Tuple[int, ...],
                  dtype: np.dtype = np.float32) -> gpuarray.GPUArray:
        """
        Get array from pool or allocate new one.

        Args:
            shape: Array shape
            dtype: Data type

        Returns:
            GPUArray (reused or newly allocated)
        """
        key = (shape, dtype)

        # Check if available in pool
        if key in self.pools and len(self.pools[key]) > 0:
            array = self.pools[key].pop()
            logger.debug(f"Reused array from pool: {shape}, {dtype}")
            return array

        # Allocate new array
        array = self.manager.allocate_device_array(shape, dtype)
        logger.debug(f"Allocated new array for pool: {shape}, {dtype}")
        return array

    def return_array(self, array: gpuarray.GPUArray):
        """
        Return array to pool for reuse.

        Args:
            array: GPUArray to return
        """
        shape = array.shape
        dtype = array.dtype
        key = (shape, dtype)

        if key not in self.pools:
            self.pools[key] = []

        self.pools[key].append(array)
        logger.debug(f"Returned array to pool: {shape}, {dtype}")

    def clear_pool(self):
        """Clear all pooled arrays."""
        for key, arrays in self.pools.items():
            for array in arrays:
                self.manager.free_device_array(array)
        self.pools.clear()
        logger.info("Cleared memory pool")


def estimate_memory_requirements(n_paths: int,
                                 n_timesteps: int,
                                 n_nodes: int,
                                 dtype: np.dtype = np.float32) -> Dict:
    """
    Estimate GPU memory requirements for simulation.

    Args:
        n_paths: Number of parallel sample paths
        n_timesteps: Number of time steps per path
        n_nodes: Number of network nodes
        dtype: Data type

    Returns:
        Dictionary with memory estimates
    """
    itemsize = np.dtype(dtype).itemsize

    # Main arrays needed
    states_size = n_paths * n_timesteps * n_nodes * itemsize  # State history
    noise_size = n_paths * n_timesteps * n_nodes * itemsize   # Random noise
    metrics_size = n_paths * itemsize                         # Output metrics

    total_size = states_size + noise_size + metrics_size

    # Add 20% overhead for intermediate arrays
    total_size_with_overhead = total_size * 1.2

    estimates = {
        'n_paths': n_paths,
        'n_timesteps': n_timesteps,
        'n_nodes': n_nodes,
        'dtype': str(dtype),
        'states_gb': states_size / 1e9,
        'noise_gb': noise_size / 1e9,
        'metrics_gb': metrics_size / 1e9,
        'total_gb': total_size / 1e9,
        'total_with_overhead_gb': total_size_with_overhead / 1e9
    }

    return estimates


def optimize_batch_size(total_samples: int,
                        n_timesteps: int,
                        n_nodes: int,
                        available_memory_gb: float,
                        dtype: np.dtype = np.float32) -> int:
    """
    Compute optimal batch size for GPU given memory constraints.

    Args:
        total_samples: Total number of samples needed
        n_timesteps: Time steps per sample
        n_nodes: Network nodes
        available_memory_gb: Available GPU memory (GB)
        dtype: Data type

    Returns:
        Optimal batch size
    """
    # Memory per sample (bytes)
    itemsize = np.dtype(dtype).itemsize
    memory_per_sample = n_timesteps * n_nodes * itemsize * 2  # states + noise

    # Available memory (bytes), use 80% to be safe
    available_bytes = available_memory_gb * 1e9 * 0.8

    # Compute batch size
    batch_size = int(available_bytes / memory_per_sample)

    # Clamp to reasonable range
    batch_size = max(1, min(batch_size, total_samples))

    logger.info(f"Optimal batch size: {batch_size} samples "
               f"(memory per sample: {memory_per_sample/1e6:.2f} MB)")

    return batch_size


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("GPU Memory Management - Example Usage")
    print("=" * 60)

    if not PYCUDA_AVAILABLE:
        print("\nPyCUDA not available. Install with: pip install pycuda")
        print("GPU acceleration will not be available.")
    else:
        # Initialize manager
        print("\n1. Initialize GPU Memory Manager")
        print("-" * 60)
        manager = GPUMemoryManager()
        manager.print_memory_summary()

        # Allocate arrays
        print("\n2. Allocate GPU Arrays")
        print("-" * 60)

        # Small array
        arr1 = manager.allocate_device_array(
            shape=(1000, 100),
            dtype=np.float32,
            name="test_array_1"
        )
        print(f"Allocated arr1: {arr1.shape}, {arr1.dtype}")

        # Transfer from host
        host_data = np.random.randn(500, 200).astype(np.float32)
        arr2 = manager.transfer_to_device(host_data, name="test_array_2")
        print(f"Transferred arr2: {arr2.shape}, {arr2.dtype}")

        # Check memory
        print("\n3. Memory Usage After Allocation")
        print("-" * 60)
        manager.print_memory_summary()

        # Transfer back to host
        print("\n4. Transfer Back to Host")
        print("-" * 60)
        result = manager.transfer_to_host(arr2)
        print(f"Retrieved array: {result.shape}, max diff: {np.max(np.abs(result - host_data))}")

        # Estimate memory requirements
        print("\n5. Memory Requirements Estimation")
        print("-" * 60)
        estimates = estimate_memory_requirements(
            n_paths=10000,
            n_timesteps=1000,
            n_nodes=100,
            dtype=np.float32
        )
        print(f"For 10,000 paths × 1,000 timesteps × 100 nodes:")
        print(f"  States: {estimates['states_gb']:.2f} GB")
        print(f"  Noise: {estimates['noise_gb']:.2f} GB")
        print(f"  Total (with overhead): {estimates['total_with_overhead_gb']:.2f} GB")

        # Optimize batch size
        print("\n6. Batch Size Optimization")
        print("-" * 60)
        info = manager.get_memory_info()
        batch_size = optimize_batch_size(
            total_samples=100000,
            n_timesteps=1000,
            n_nodes=100,
            available_memory_gb=info['free_memory_gb'],
            dtype=np.float32
        )
        print(f"Optimal batch size: {batch_size} samples")

        # Cleanup
        print("\n7. Cleanup")
        print("-" * 60)
        manager.free_device_array(arr1)
        manager.free_device_array(arr2)
        manager.print_memory_summary()

    print("\n" + "=" * 60)
