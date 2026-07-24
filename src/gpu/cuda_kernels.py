"""
CUDA Kernels for GPU-Accelerated Network Simulation

Implements CUDA kernels for parallel Monte Carlo simulation of network
queue dynamics and congestion propagation.

Each thread simulates one independent sample path, enabling massive parallelism.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Check PyCUDA availability
try:
    import pycuda.autoinit
    try:
        import pycuda.curand as curand
    except ImportError:
        import pycuda.curandom as curand  # newer pycuda renamed the module
    import pycuda.driver as cuda
    from pycuda import gpuarray
    from pycuda.compiler import SourceModule

    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    logger.warning("PyCUDA not available. GPU kernels will not be compiled.")


# CUDA kernel source code for queue dynamics simulation
#
# Memory Layout — Transposed [n_timesteps, n_paths] for Coalesced Access:
# Element (t, path_id) is stored at queue_states[t * n_paths + path_id].
# Consecutive threads (consecutive path_id values) therefore access consecutive
# memory addresses at each timestep, achieving fully coalesced global-memory
# reads and writes and maximising DRAM bandwidth utilisation.
# All kernels in this file use this transposed layout consistently.
#
QUEUE_DYNAMICS_KERNEL = """
__global__ void simulate_queue_dynamics(
    float* queue_states,        // Output: [n_timesteps, n_paths]  -- TRANSPOSED for coalesced access
    const float* noise,          // Input: Random noise [n_timesteps, n_paths]  -- TRANSPOSED
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

    // Each thread simulates one independent path.
    // Layout: [n_timesteps, n_paths] so consecutive threads (consecutive path_id)
    // access consecutive memory addresses at each timestep -- fully coalesced.
    float q = 0.0f;  // Initial queue length

    // Compute drift and diffusion
    float drift = arrival_rate - service_rate;
    float diffusion = noise_intensity;

    // Store initial state: index = 0 * n_paths + path_id
    queue_states[path_id] = q;

    // Euler-Maruyama integration
    for (int t = 1; t < n_timesteps; t++) {
        // Coalesced read: consecutive threads read consecutive addresses
        float dW = noise[t * n_paths + path_id];

        // Update: q_new = q + drift*dt + diffusion*dW
        float q_new = q + drift * dt + diffusion * dW;

        // Enforce non-negativity constraint (reflecting boundary)
        q_new = fmaxf(0.0f, q_new);

        // Coalesced write
        queue_states[t * n_paths + path_id] = q_new;

        // Update for next iteration
        q = q_new;
    }
}
"""


# CUDA kernel for computing metrics from queue states
COMPUTE_METRICS_KERNEL = """
__global__ void compute_queue_metrics(
    float* metrics,              // Output: [n_paths]
    const float* queue_states,   // Input: [n_timesteps, n_paths]  -- TRANSPOSED layout
    const int n_paths,
    const int n_timesteps,
    const int metric_type        // 0=mean, 1=max, 2=final
)
{
    int path_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_id >= n_paths) return;

    // Transposed layout: element (t, path_id) lives at index t*n_paths + path_id.
    // Consecutive threads read/write consecutive addresses -- fully coalesced.

    if (metric_type == 0) {
        // Mean queue length
        float sum = 0.0f;
        for (int t = 0; t < n_timesteps; t++) {
            sum += queue_states[t * n_paths + path_id];
        }
        metrics[path_id] = sum / n_timesteps;

    } else if (metric_type == 1) {
        // Max queue length
        float max_val = queue_states[0 * n_paths + path_id];
        for (int t = 1; t < n_timesteps; t++) {
            max_val = fmaxf(max_val, queue_states[t * n_paths + path_id]);
        }
        metrics[path_id] = max_val;

    } else if (metric_type == 2) {
        // Final queue length
        metrics[path_id] = queue_states[(n_timesteps - 1) * n_paths + path_id];
    }
}
"""


# CUDA kernel for parallel coupled path simulation (MLMC)
COUPLED_PATHS_KERNEL = """
__global__ void simulate_coupled_paths(
    float* fine_metrics,         // Output: [n_paths]
    float* coarse_metrics,       // Output: [n_paths]
    const float* noise_fine,     // Input: [n_timesteps_fine, n_paths]  -- TRANSPOSED for coalesced access
    const float arrival_rate,
    const float service_rate,
    const float noise_intensity,
    const float dt_fine,
    const float dt_coarse,
    const int n_paths,
    const int n_timesteps_fine,
    const int refinement_factor, // M = dt_coarse / dt_fine
    const int metric_type        // 0=mean, 1=max, 2=final
)
{
    int path_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_id >= n_paths) return;

    const float drift = arrival_rate - service_rate;
    const float diffusion = noise_intensity;

    // Simulate fine path
    float q_fine = 0.0f;
    float sum_fine = 0.0f;
    float max_fine = 0.0f;

    // Transposed layout: noise_fine[t * n_paths + path_id] -- fully coalesced reads
    for (int t = 0; t < n_timesteps_fine; t++) {
        float dW_fine = noise_fine[t * n_paths + path_id];
        q_fine = q_fine + drift * dt_fine + diffusion * dW_fine;
        q_fine = fmaxf(0.0f, q_fine);
        sum_fine += q_fine;
        max_fine = fmaxf(max_fine, q_fine);
    }

    if (metric_type == 0) {
        fine_metrics[path_id] = sum_fine / n_timesteps_fine;
    } else if (metric_type == 1) {
        fine_metrics[path_id] = max_fine;
    } else {
        fine_metrics[path_id] = q_fine;
    }

    // Simulate coarse path using aggregated noise
    float q_coarse = 0.0f;
    float sum_coarse = 0.0f;
    float max_coarse = 0.0f;
    int n_timesteps_coarse = n_timesteps_fine / refinement_factor;

    for (int t_coarse = 0; t_coarse < n_timesteps_coarse; t_coarse++) {
        // Aggregate M fine noise increments (transposed: noise_fine[t_fine * n_paths + path_id])
        float dW_coarse = 0.0f;
        for (int i = 0; i < refinement_factor; i++) {
            int t_fine = t_coarse * refinement_factor + i;
            dW_coarse += noise_fine[t_fine * n_paths + path_id];
        }

        q_coarse = q_coarse + drift * dt_coarse + diffusion * dW_coarse;
        q_coarse = fmaxf(0.0f, q_coarse);

        // Piecewise-constant interpolation: assume queue length constant between
        // coarse timesteps. This gives comparable mean metrics between levels.
        // sum_coarse = Σ(q_coarse_t * M) / N_fine = Σ(q_coarse_t) / N_coarse
        sum_coarse += q_coarse * refinement_factor;
        max_coarse = fmaxf(max_coarse, q_coarse);
    }

    if (metric_type == 0) {
        coarse_metrics[path_id] = sum_coarse / n_timesteps_fine;
    } else if (metric_type == 1) {
        coarse_metrics[path_id] = max_coarse;
    } else {
        coarse_metrics[path_id] = q_coarse;
    }
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
            self.kernels["simulate_queue_dynamics"] = mod_queue.get_function(
                "simulate_queue_dynamics"
            )
            logger.info("Compiled: simulate_queue_dynamics")

            # Compile metrics kernel
            mod_metrics = SourceModule(COMPUTE_METRICS_KERNEL)
            self.kernels["compute_queue_metrics"] = mod_metrics.get_function(
                "compute_queue_metrics"
            )
            logger.info("Compiled: compute_queue_metrics")

            # Compile coupled paths kernel
            mod_coupled = SourceModule(COUPLED_PATHS_KERNEL)
            self.kernels["simulate_coupled_paths"] = mod_coupled.get_function(
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

    Performance Notes:
        - GPU occupancy is typically 50-75% due to register pressure from
          loop-carried dependencies in the Euler-Maruyama integration.
        - Memory bandwidth is often the bottleneck for large simulations.
        - For optimal performance, use n_paths >= 10,000 and batch processing.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize GPU queue simulator.

        Args:
            seed: Random seed for reproducibility (affects GPU RNG state)
        """
        if not PYCUDA_AVAILABLE:
            raise ImportError("PyCUDA required for GPU simulation")

        self.compiler = CUDAKernelCompiler()
        # Note: XORWOW has limited statistical quality for high-precision work.
        # For production use requiring better RNG, consider generating noise
        # on CPU with numpy and transferring to GPU, or using cuRAND host API
        # with MRG32k3a generator.
        self.rng = curand.XORWOWRandomNumberGenerator(offset=seed if seed else 0)

    def simulate_paths(
        self,
        n_paths: int,
        n_timesteps: int,
        arrival_rate: float,
        service_rate: float,
        noise_intensity: float,
        dt: float,
        metric: str = "mean",
        block_size: int = 256,
    ) -> np.ndarray:
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
        # Allocate device arrays — transposed layout [n_timesteps, n_paths] for
        # coalesced access: consecutive threads (consecutive path_id) read/write
        # consecutive memory addresses at every timestep.
        queue_states = gpuarray.zeros((n_timesteps, n_paths), dtype=np.float32)
        metrics_out = gpuarray.zeros(n_paths, dtype=np.float32)

        # Generate random noise on GPU — same transposed layout
        noise = gpuarray.empty((n_timesteps, n_paths), dtype=np.float32)
        self.rng.fill_normal(noise)

        # Scale noise by sqrt(dt) for Brownian increments (keep float32)
        noise = noise * np.float32(np.sqrt(dt))

        # Configure grid
        grid_size = (n_paths + block_size - 1) // block_size

        # Launch queue simulation kernel
        kernel = self.compiler.get_kernel("simulate_queue_dynamics")
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
            grid=(grid_size, 1),
        )

        # Synchronize and check for errors
        cuda.Context.synchronize()

        # Compute metrics
        metric_type = {"mean": 0, "max": 1, "final": 2}[metric]
        metrics_kernel = self.compiler.get_kernel("compute_queue_metrics")
        metrics_kernel(
            metrics_out,
            queue_states,
            np.int32(n_paths),
            np.int32(n_timesteps),
            np.int32(metric_type),
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )

        # Synchronize and check for errors
        cuda.Context.synchronize()

        # Transfer results back to host
        results = metrics_out.get()

        logger.debug(f"Simulated {n_paths} paths on GPU with {n_timesteps} timesteps")

        return results

    def simulate_coupled_paths_mlmc(
        self,
        n_paths: int,
        n_timesteps_fine: int,
        arrival_rate: float,
        service_rate: float,
        noise_intensity: float,
        dt_fine: float,
        dt_coarse: float,
        metric: str = "mean",
        block_size: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            metric: Metric to compute ('mean', 'max', 'final')
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

        # Generate fine noise on GPU — transposed layout [n_timesteps_fine, n_paths]
        # so that at each timestep t, noise_fine[t * n_paths + path_id] is contiguous
        # for all path_id values — fully coalesced reads in the CUDA kernel.
        noise_fine = gpuarray.empty((n_timesteps_fine, n_paths), dtype=np.float32)
        self.rng.fill_normal(noise_fine)
        noise_fine = noise_fine * np.float32(np.sqrt(dt_fine))

        # Configure grid
        grid_size = (n_paths + block_size - 1) // block_size
        metric_type = {"mean": 0, "max": 1, "final": 2}[metric]

        # Launch coupled simulation kernel
        kernel = self.compiler.get_kernel("simulate_coupled_paths")
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
            np.int32(metric_type),
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
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


def benchmark_memory_layout(
    n_paths: int = 65536,
    n_timesteps: int = 512,
    n_repeats: int = 5,
    block_size: int = 256,
) -> dict:
    """
    Benchmark the bandwidth improvement from the transposed [n_timesteps, n_paths]
    memory layout versus the legacy [n_paths, n_timesteps] layout.

    Compiles both kernel variants, times them over ``n_repeats`` runs, and
    reports effective memory bandwidth (GB/s) and throughput (samples/s) for
    each layout so the improvement can be quoted in the paper.

    Args:
        n_paths:      Number of parallel sample paths.
        n_timesteps:  Time steps per path.
        n_repeats:    Number of timed repetitions (results are averaged).
        block_size:   CUDA block size.

    Returns:
        Dictionary with keys:
            ``legacy_bw_GBps``   – bandwidth for old [n_paths, n_timesteps] layout
            ``transposed_bw_GBps`` – bandwidth for new [n_timesteps, n_paths] layout
            ``speedup``          – transposed / legacy throughput ratio
            ``legacy_ms``        – mean wall-clock time (ms) for legacy layout
            ``transposed_ms``    – mean wall-clock time (ms) for transposed layout
    """
    if not PYCUDA_AVAILABLE:
        logger.warning("PyCUDA not available – skipping memory layout benchmark.")
        return {}

    import time

    # ------------------------------------------------------------------ #
    # Legacy kernel source (original [n_paths, n_timesteps] row-major)   #
    # ------------------------------------------------------------------ #
    LEGACY_KERNEL = """
__global__ void simulate_queue_dynamics_legacy(
    float* queue_states,
    const float* noise,
    const float arrival_rate,
    const float service_rate,
    const float noise_intensity,
    const float dt,
    const int n_paths,
    const int n_timesteps
)
{
    int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= n_paths) return;

    float q    = 0.0f;
    float drift     = arrival_rate - service_rate;
    float diffusion = noise_intensity;

    queue_states[path_id * n_timesteps] = q;

    for (int t = 1; t < n_timesteps; t++) {
        float dW   = noise[path_id * n_timesteps + t];   // strided (non-coalesced)
        float q_new = q + drift * dt + diffusion * dW;
        q_new = fmaxf(0.0f, q_new);
        queue_states[path_id * n_timesteps + t] = q_new; // strided write
        q = q_new;
    }
}
"""

    # ------------------------------------------------------------------ #
    # Transposed kernel source (new [n_timesteps, n_paths] column-major)  #
    # ------------------------------------------------------------------ #
    TRANSPOSED_KERNEL = """
__global__ void simulate_queue_dynamics_transposed(
    float* queue_states,
    const float* noise,
    const float arrival_rate,
    const float service_rate,
    const float noise_intensity,
    const float dt,
    const int n_paths,
    const int n_timesteps
)
{
    int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= n_paths) return;

    float q    = 0.0f;
    float drift     = arrival_rate - service_rate;
    float diffusion = noise_intensity;

    queue_states[path_id] = q;

    for (int t = 1; t < n_timesteps; t++) {
        float dW    = noise[t * n_paths + path_id];        // coalesced read
        float q_new = q + drift * dt + diffusion * dW;
        q_new = fmaxf(0.0f, q_new);
        queue_states[t * n_paths + path_id] = q_new;       // coalesced write
        q = q_new;
    }
}
"""

    logger.info("Compiling benchmark kernels …")
    mod_legacy = SourceModule(LEGACY_KERNEL)
    mod_transposed = SourceModule(TRANSPOSED_KERNEL)
    fn_legacy = mod_legacy.get_function("simulate_queue_dynamics_legacy")
    fn_transposed = mod_transposed.get_function("simulate_queue_dynamics_transposed")

    grid_size = (n_paths + block_size - 1) // block_size

    # Simulation parameters
    arrival_rate = np.float32(1.0)
    service_rate = np.float32(1.25)
    noise_intensity = np.float32(0.2)
    dt = np.float32(0.01)

    # Each kernel reads noise (n_paths*n_timesteps floats) and writes states
    # (n_paths*n_timesteps floats) → total bytes per run:
    bytes_per_run = 2 * n_paths * n_timesteps * 4  # float32 = 4 bytes

    # ------------------------------------------------------------------
    # Benchmark legacy layout
    # ------------------------------------------------------------------
    rng = curand.XORWOWRandomNumberGenerator(offset=0)

    legacy_times = []
    for _ in range(n_repeats):
        states_legacy = gpuarray.zeros((n_paths, n_timesteps), dtype=np.float32)
        noise_legacy = gpuarray.empty((n_paths, n_timesteps), dtype=np.float32)
        rng.fill_normal(noise_legacy)
        noise_legacy = noise_legacy * np.float32(np.sqrt(dt))

        cuda.Context.synchronize()
        t0 = time.perf_counter()
        fn_legacy(
            states_legacy,
            noise_legacy,
            arrival_rate,
            service_rate,
            noise_intensity,
            dt,
            np.int32(n_paths),
            np.int32(n_timesteps),
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )
        cuda.Context.synchronize()
        legacy_times.append(time.perf_counter() - t0)

    legacy_ms = np.mean(legacy_times) * 1e3
    legacy_bw = bytes_per_run / (np.mean(legacy_times) * 1e9)  # GB/s
    legacy_tput = n_paths / np.mean(legacy_times)  # paths/s

    # ------------------------------------------------------------------
    # Benchmark transposed layout
    # ------------------------------------------------------------------
    transposed_times = []
    for _ in range(n_repeats):
        states_transposed = gpuarray.zeros((n_timesteps, n_paths), dtype=np.float32)
        noise_transposed = gpuarray.empty((n_timesteps, n_paths), dtype=np.float32)
        rng.fill_normal(noise_transposed)
        noise_transposed = noise_transposed * np.float32(np.sqrt(dt))

        cuda.Context.synchronize()
        t0 = time.perf_counter()
        fn_transposed(
            states_transposed,
            noise_transposed,
            arrival_rate,
            service_rate,
            noise_intensity,
            dt,
            np.int32(n_paths),
            np.int32(n_timesteps),
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )
        cuda.Context.synchronize()
        transposed_times.append(time.perf_counter() - t0)

    transposed_ms = np.mean(transposed_times) * 1e3
    transposed_bw = bytes_per_run / (np.mean(transposed_times) * 1e9)
    transposed_tput = n_paths / np.mean(transposed_times)

    speedup = transposed_tput / legacy_tput

    result = {
        "n_paths": n_paths,
        "n_timesteps": n_timesteps,
        "legacy_ms": round(legacy_ms, 3),
        "transposed_ms": round(transposed_ms, 3),
        "legacy_bw_GBps": round(legacy_bw, 2),
        "transposed_bw_GBps": round(transposed_bw, 2),
        "speedup": round(speedup, 3),
    }

    logger.info(
        f"Memory layout benchmark  |  "
        f"legacy={legacy_ms:.1f} ms ({legacy_bw:.1f} GB/s)  |  "
        f"transposed={transposed_ms:.1f} ms ({transposed_bw:.1f} GB/s)  |  "
        f"speedup={speedup:.2f}x"
    )
    print(
        f"\n{'=' * 60}\n"
        f"Memory Layout Benchmark Results\n"
        f"{'=' * 60}\n"
        f"  n_paths     : {n_paths:,}\n"
        f"  n_timesteps : {n_timesteps:,}\n"
        f"  Repeats     : {n_repeats}\n"
        f"  Legacy   [n_paths, n_timesteps] : {legacy_ms:7.2f} ms  |  {legacy_bw:6.2f} GB/s\n"
        f"  Transposed [n_timesteps,n_paths]: {transposed_ms:7.2f} ms  |  {transposed_bw:6.2f} GB/s\n"
        f"  Speedup     : {speedup:.2f}x\n"
        f"{'=' * 60}\n"
    )
    return result


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
            metric="mean",
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
            dt_coarse=0.02,
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
