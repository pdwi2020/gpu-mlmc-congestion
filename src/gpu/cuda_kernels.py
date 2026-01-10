"""
CUDA Kernels for GPU-Accelerated Network Simulation

Implements CUDA kernels for parallel Monte Carlo simulation of network
queue dynamics and congestion propagation.

Each thread simulates one independent sample path, enabling massive parallelism.
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Check PyCUDA availability
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    from pycuda import gpuarray
    import pycuda.curand as curand
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    logger.warning("PyCUDA not available. GPU kernels will not be compiled.")


# CUDA kernel source code for queue dynamics simulation
QUEUE_DYNAMICS_KERNEL = """
__global__ void simulate_queue_dynamics(
    float* queue_states,        // Output: [n_paths, n_timesteps]
    const float* noise,          // Input: Random noise [n_paths, n_timesteps]
    const float arrival_rate,
    const float service_rate,
    const float noise_intensity,
    const float dt,
    const int n_paths,
    const int n_timesteps
)
{
    // Global thread ID = sample path index
    int path_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_id >= n_paths) return;

    // Each thread simulates one independent path
    float q = 0.0f;  // Initial queue length

    // Compute drift and diffusion
    float drift = arrival_rate - service_rate;
    float diffusion = noise_intensity;

    // Store initial state
    queue_states[path_id * n_timesteps + 0] = q;

    // Euler-Maruyama integration
    for (int t = 1; t < n_timesteps; t++) {
        // Get noise for this timestep
        float dW = noise[path_id * n_timesteps + t];

        // Update: q_new = q + drift*dt + diffusion*dW
        float q_new = q + drift * dt + diffusion * dW;

        // Enforce non-negativity constraint
        q_new = fmaxf(0.0f, q_new);

        // Store state
        queue_states[path_id * n_timesteps + t] = q_new;

        // Update for next iteration
        q = q_new;
    }
}
"""


# CUDA kernel for computing metrics from queue states
COMPUTE_METRICS_KERNEL = """
__global__ void compute_queue_metrics(
    float* metrics,              // Output: [n_paths, n_metrics]
    const float* queue_states,   // Input: [n_paths, n_timesteps]
    const int n_paths,
    const int n_timesteps,
    const int metric_type        // 0=mean, 1=max, 2=final
)
{
    int path_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_id >= n_paths) return;

    const float* path_data = &queue_states[path_id * n_timesteps];

    if (metric_type == 0) {
        // Mean queue length
        float sum = 0.0f;
        for (int t = 0; t < n_timesteps; t++) {
            sum += path_data[t];
        }
        metrics[path_id] = sum / n_timesteps;

    } else if (metric_type == 1) {
        // Max queue length
        float max_val = path_data[0];
        for (int t = 1; t < n_timesteps; t++) {
            max_val = fmaxf(max_val, path_data[t]);
        }
        metrics[path_id] = max_val;

    } else if (metric_type == 2) {
        // Final queue length
        metrics[path_id] = path_data[n_timesteps - 1];
    }
}
"""


# CUDA kernel for parallel coupled path simulation (MLMC)
COUPLED_PATHS_KERNEL = """
__global__ void simulate_coupled_paths(
    float* fine_metrics,         // Output: [n_paths]
    float* coarse_metrics,       // Output: [n_paths]
    const float* noise_fine,     // Input: [n_paths, n_timesteps_fine]
    const float arrival_rate,
    const float service_rate,
    const float noise_intensity,
    const float dt_fine,
    const float dt_coarse,
    const int n_paths,
    const int n_timesteps_fine,
    const int refinement_factor  // M = dt_coarse / dt_fine
)
{
    int path_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_id >= n_paths) return;

    const float drift = arrival_rate - service_rate;
    const float diffusion = noise_intensity;

    // Simulate fine path
    float q_fine = 0.0f;
    float sum_fine = 0.0f;

    for (int t = 0; t < n_timesteps_fine; t++) {
        float dW_fine = noise_fine[path_id * n_timesteps_fine + t];
        q_fine = q_fine + drift * dt_fine + diffusion * dW_fine;
        q_fine = fmaxf(0.0f, q_fine);
        sum_fine += q_fine;
    }

    fine_metrics[path_id] = sum_fine / n_timesteps_fine;

    // Simulate coarse path using aggregated noise
    float q_coarse = 0.0f;
    float sum_coarse = 0.0f;
    int n_timesteps_coarse = n_timesteps_fine / refinement_factor;

    for (int t_coarse = 0; t_coarse < n_timesteps_coarse; t_coarse++) {
        // Aggregate M fine noise increments
        float dW_coarse = 0.0f;
        for (int i = 0; i < refinement_factor; i++) {
            int t_fine = t_coarse * refinement_factor + i;
            dW_coarse += noise_fine[path_id * n_timesteps_fine + t_fine];
        }

        q_coarse = q_coarse + drift * dt_coarse + diffusion * dW_coarse;
        q_coarse = fmaxf(0.0f, q_coarse);

        // Add to sum M times to match fine grid
        sum_coarse += q_coarse * refinement_factor;
    }

    coarse_metrics[path_id] = sum_coarse / n_timesteps_fine;
}
"""


class CUDAKernelCompiler:
    """
    Compile and manage CUDA kernels for network simulation.
    """

    def __init__(self):
        """Initialize kernel compiler."""
        if not PYCUDA_AVAILABLE:
            raise ImportError("PyCUDA required for GPU kernels")

        self.kernels = {}
        self._compile_kernels()

    def _compile_kernels(self):
        """Compile all CUDA kernels."""
        logger.info("Compiling CUDA kernels...")

        try:
            # Compile queue dynamics kernel
            mod_queue = SourceModule(QUEUE_DYNAMICS_KERNEL)
            self.kernels['simulate_queue_dynamics'] = mod_queue.get_function(
                "simulate_queue_dynamics"
            )
            logger.info("Compiled: simulate_queue_dynamics")

            # Compile metrics kernel
            mod_metrics = SourceModule(COMPUTE_METRICS_KERNEL)
            self.kernels['compute_queue_metrics'] = mod_metrics.get_function(
                "compute_queue_metrics"
            )
            logger.info("Compiled: compute_queue_metrics")

            # Compile coupled paths kernel
            mod_coupled = SourceModule(COUPLED_PATHS_KERNEL)
            self.kernels['simulate_coupled_paths'] = mod_coupled.get_function(
                "simulate_coupled_paths"
            )
            logger.info("Compiled: simulate_coupled_paths")

            logger.info("All CUDA kernels compiled successfully")

        except Exception as e:
            logger.error(f"Failed to compile CUDA kernels: {e}")
            raise

    def get_kernel(self, name: str):
        """Get compiled kernel by name."""
        if name not in self.kernels:
            raise ValueError(f"Unknown kernel: {name}")
        return self.kernels[name]


class GPUQueueSimulator:
    """
    GPU-accelerated queue dynamics simulator using CUDA kernels.
    """

    def __init__(self):
        """Initialize GPU queue simulator."""
        if not PYCUDA_AVAILABLE:
            raise ImportError("PyCUDA required for GPU simulation")

        self.compiler = CUDAKernelCompiler()
        self.rng = curand.XORWOWRandomNumberGenerator()

    def simulate_paths(self,
                      n_paths: int,
                      n_timesteps: int,
                      arrival_rate: float,
                      service_rate: float,
                      noise_intensity: float,
                      dt: float,
                      metric: str = 'mean',
                      block_size: int = 256) -> np.ndarray:
        """
        Simulate multiple independent queue dynamics paths on GPU.

        Args:
            n_paths: Number of parallel sample paths
            n_timesteps: Number of time steps per path
            arrival_rate: Packet arrival rate λ
            service_rate: Service rate μ
            noise_intensity: Noise coefficient σ
            dt: Time step
            metric: Metric to compute ('mean', 'max', 'final')
            block_size: CUDA block size

        Returns:
            Array of metric values (n_paths,)
        """
        # Allocate device arrays
        queue_states = gpuarray.zeros((n_paths, n_timesteps), dtype=np.float32)
        metrics_out = gpuarray.zeros(n_paths, dtype=np.float32)

        # Generate random noise on GPU
        noise = gpuarray.empty((n_paths, n_timesteps), dtype=np.float32)
        self.rng.fill_normal(noise)

        # Scale noise by sqrt(dt) for Brownian increments
        noise = noise * np.sqrt(dt)

        # Configure grid
        grid_size = (n_paths + block_size - 1) // block_size

        # Launch queue simulation kernel
        kernel = self.compiler.get_kernel('simulate_queue_dynamics')
        kernel(
            queue_states,
            noise,
            np.float32(arrival_rate),
            np.float32(service_rate),
            np.float32(noise_intensity),
            np.float32(dt),
            np.int32(n_paths),
            np.int32(n_timesteps),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )

        # Compute metrics
        metric_type = {'mean': 0, 'max': 1, 'final': 2}[metric]
        metrics_kernel = self.compiler.get_kernel('compute_queue_metrics')
        metrics_kernel(
            metrics_out,
            queue_states,
            np.int32(n_paths),
            np.int32(n_timesteps),
            np.int32(metric_type),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )

        # Transfer results back to host
        results = metrics_out.get()

        logger.debug(f"Simulated {n_paths} paths on GPU with {n_timesteps} timesteps")

        return results

    def simulate_coupled_paths_mlmc(self,
                                   n_paths: int,
                                   n_timesteps_fine: int,
                                   arrival_rate: float,
                                   service_rate: float,
                                   noise_intensity: float,
                                   dt_fine: float,
                                   dt_coarse: float,
                                   block_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate coupled fine and coarse paths for MLMC on GPU.

        Args:
            n_paths: Number of parallel paths
            n_timesteps_fine: Number of fine time steps
            arrival_rate: Arrival rate
            service_rate: Service rate
            noise_intensity: Noise intensity
            dt_fine: Fine time step
            dt_coarse: Coarse time step
            block_size: CUDA block size

        Returns:
            Tuple of (fine_metrics, coarse_metrics)
        """
        # Verify refinement
        refinement_factor = int(dt_coarse / dt_fine)
        if not np.isclose(dt_coarse, refinement_factor * dt_fine):
            raise ValueError("dt_coarse must be integer multiple of dt_fine")

        # Allocate output arrays
        fine_metrics = gpuarray.zeros(n_paths, dtype=np.float32)
        coarse_metrics = gpuarray.zeros(n_paths, dtype=np.float32)

        # Generate fine noise on GPU
        noise_fine = gpuarray.empty((n_paths, n_timesteps_fine), dtype=np.float32)
        self.rng.fill_normal(noise_fine)
        noise_fine = noise_fine * np.sqrt(dt_fine)

        # Configure grid
        grid_size = (n_paths + block_size - 1) // block_size

        # Launch coupled simulation kernel
        kernel = self.compiler.get_kernel('simulate_coupled_paths')
        kernel(
            fine_metrics,
            coarse_metrics,
            noise_fine,
            np.float32(arrival_rate),
            np.float32(service_rate),
            np.float32(noise_intensity),
            np.float32(dt_fine),
            np.float32(dt_coarse),
            np.int32(n_paths),
            np.int32(n_timesteps_fine),
            np.int32(refinement_factor),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )

        # Transfer to host
        fine_results = fine_metrics.get()
        coarse_results = coarse_metrics.get()

        logger.debug(f"Simulated {n_paths} coupled MLMC paths on GPU")

        return fine_results, coarse_results


def get_optimal_block_size(device_id: int = 0) -> int:
    """
    Get optimal CUDA block size for device.

    Args:
        device_id: GPU device ID

    Returns:
        Optimal block size (typically 128, 256, or 512)
    """
    if not PYCUDA_AVAILABLE:
        return 256  # Default

    device = cuda.Device(device_id)
    max_threads_per_block = device.get_attribute(
        cuda.device_attribute.MAX_THREADS_PER_BLOCK
    )

    # Use 256 as good default (balance between occupancy and flexibility)
    block_size = min(256, max_threads_per_block)

    return block_size


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("CUDA Kernels - Example Usage")
    print("=" * 60)

    if not PYCUDA_AVAILABLE:
        print("\nPyCUDA not available. Install with: pip install pycuda")
    else:
        # Initialize simulator
        print("\n1. Initialize GPU Queue Simulator")
        print("-" * 60)
        simulator = GPUQueueSimulator()
        print("GPU simulator initialized")

        # Simulate paths
        print("\n2. Simulate Queue Dynamics on GPU")
        print("-" * 60)

        results = simulator.simulate_paths(
            n_paths=10000,
            n_timesteps=1000,
            arrival_rate=8.0,
            service_rate=10.0,
            noise_intensity=0.5,
            dt=0.01,
            metric='mean'
        )

        print(f"Simulated {len(results)} paths")
        print(f"Mean queue length: {np.mean(results):.4f}")
        print(f"Std: {np.std(results):.4f}")
        print(f"Min: {np.min(results):.4f}, Max: {np.max(results):.4f}")

        # MLMC coupled paths
        print("\n3. MLMC Coupled Paths on GPU")
        print("-" * 60)

        fine_metrics, coarse_metrics = simulator.simulate_coupled_paths_mlmc(
            n_paths=5000,
            n_timesteps_fine=1000,
            arrival_rate=8.0,
            service_rate=10.0,
            noise_intensity=0.5,
            dt_fine=0.01,
            dt_coarse=0.02
        )

        print(f"Fine path mean: {np.mean(fine_metrics):.4f}")
        print(f"Coarse path mean: {np.mean(coarse_metrics):.4f}")
        print(f"Mean difference: {np.mean(fine_metrics - coarse_metrics):.4f}")
        print(f"Variance of difference: {np.var(fine_metrics - coarse_metrics):.6e}")

        # Optimal block size
        print("\n4. Device Configuration")
        print("-" * 60)
        block_size = get_optimal_block_size()
        print(f"Optimal block size: {block_size}")

    print("\n" + "=" * 60)
