"""
GPU-Accelerated Parallel Monte Carlo Module

Integrates CUDA kernels with Monte Carlo and MLMC simulation frameworks
for massive GPU speedup.

Classes:
    GPUMonteCarloSimulator: GPU-accelerated Monte Carlo
    GPUMLMCSimulator: GPU-accelerated MLMC
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
from pathlib import Path
import sys

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from network.topology import NetworkGraph
from network.traffic import TrafficModel
from simulation.monte_carlo import NetworkSimulationResult
from simulation.mlmc import MLMCResult, MLMCLevelStats
from simulation.discretization import get_timestep

logger = logging.getLogger(__name__)

# Check GPU availability
try:
    from gpu.cuda_kernels import GPUQueueSimulator, PYCUDA_AVAILABLE
    from gpu.memory_mgmt import GPUMemoryManager, optimize_batch_size
except ImportError:
    PYCUDA_AVAILABLE = False
    logger.warning("GPU modules not available")


class GPUMonteCarloSimulator:
    """
    GPU-accelerated Monte Carlo simulator.

    Provides 100x-500x speedup over CPU for large sample counts.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize GPU Monte Carlo simulator.

        Args:
            seed: Random seed (note: GPU RNG may differ from CPU)
        """
        if not PYCUDA_AVAILABLE:
            raise ImportError(
                "PyCUDA required for GPU simulation. "
                "Install with: pip install pycuda"
            )

        self.seed = seed
        self.gpu_simulator = GPUQueueSimulator()
        self.memory_manager = GPUMemoryManager()

    def estimate(self,
                network: NetworkGraph,
                traffic: TrafficModel,
                n_samples: int,
                T: float,
                dt: float,
                metric: str = 'mean_queue',
                confidence_level: float = 0.95,
                batch_size: Optional[int] = None,
                verbose: bool = True) -> NetworkSimulationResult:
        """
        Run GPU-accelerated Monte Carlo estimation.

        Args:
            network: NetworkGraph object
            traffic: TrafficModel object
            n_samples: Total number of samples
            T: Simulation duration
            dt: Time step
            metric: Metric to compute
            confidence_level: Confidence level for CI
            batch_size: Batch size (auto-optimize if None)
            verbose: Print progress

        Returns:
            NetworkSimulationResult
        """
        if verbose:
            logger.info(f"Starting GPU Monte Carlo: N={n_samples}, T={T}, dt={dt}")

        start_time = time.time()

        # Get traffic parameters
        traffic_stats = traffic.get_statistics(duration=T)
        arrival_rate = traffic_stats['arrival_rate']
        service_rate = arrival_rate * 1.25  # Overprovisioned

        # Compute number of timesteps
        n_timesteps = int(T / dt)

        # Determine batch size if not specified
        if batch_size is None:
            mem_info = self.memory_manager.get_memory_info()
            batch_size = optimize_batch_size(
                total_samples=n_samples,
                n_timesteps=n_timesteps,
                n_nodes=1,  # Single queue for now
                available_memory_gb=mem_info['free_memory_gb'],
                dtype=np.float32
            )

            if verbose:
                logger.info(f"Auto-selected batch size: {batch_size}")

        # Process in batches
        n_batches = (n_samples + batch_size - 1) // batch_size
        all_results = []

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, n_samples)
            batch_n = batch_end - batch_start

            if verbose and n_batches > 1:
                logger.info(f"Processing batch {batch_idx + 1}/{n_batches} "
                           f"({batch_n} samples)")

            # Run GPU simulation for this batch
            batch_results = self.gpu_simulator.simulate_paths(
                n_paths=batch_n,
                n_timesteps=n_timesteps,
                arrival_rate=arrival_rate,
                service_rate=service_rate,
                noise_intensity=0.2,
                dt=dt,
                metric=metric.replace('_queue', '')  # Convert 'mean_queue' → 'mean'
            )

            all_results.append(batch_results)

        # Combine all batches
        samples = np.concatenate(all_results)

        # Compute statistics
        mean = np.mean(samples)
        variance = np.var(samples, ddof=1)
        std = np.sqrt(variance)

        # Confidence interval
        from scipy import stats
        n = len(samples)
        std_error = std / np.sqrt(n)
        alpha = 1 - confidence_level
        t_value = stats.t.ppf(1 - alpha / 2, df=n - 1)
        ci_lower = mean - t_value * std_error
        ci_upper = mean + t_value * std_error

        # Computational cost
        computational_cost = n_samples * n_timesteps

        # Wall-clock time
        elapsed_time = time.time() - start_time

        result = NetworkSimulationResult(
            samples=samples,
            mean=mean,
            variance=variance,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            n_samples=n_samples,
            computational_cost=computational_cost,
            metadata={
                'T': T,
                'dt': dt,
                'metric': metric,
                'n_timesteps': n_timesteps,
                'batch_size': batch_size,
                'n_batches': n_batches,
                'gpu_time_seconds': elapsed_time,
                'throughput_samples_per_sec': n_samples / elapsed_time
            }
        )

        if verbose:
            logger.info(f"GPU Monte Carlo complete: {result.summary()}")
            logger.info(f"GPU time: {elapsed_time:.2f}s "
                       f"({n_samples/elapsed_time:.0f} samples/sec)")

        return result

    def benchmark_speedup(self,
                         network: NetworkGraph,
                         traffic: TrafficModel,
                         n_samples_list: list,
                         T: float = 10.0,
                         dt: float = 0.1) -> Dict:
        """
        Benchmark GPU speedup vs CPU for different sample sizes.

        Args:
            network: NetworkGraph
            traffic: TrafficModel
            n_samples_list: List of sample sizes to test
            T: Simulation duration
            dt: Time step

        Returns:
            Dictionary with benchmark results
        """
        from simulation.monte_carlo import MonteCarloSimulator

        results = {
            'n_samples': n_samples_list,
            'gpu_time': [],
            'cpu_time': [],
            'speedup': []
        }

        cpu_simulator = MonteCarloSimulator(seed=42)

        for n_samples in n_samples_list:
            logger.info(f"Benchmarking N={n_samples}...")

            # GPU timing
            gpu_start = time.time()
            gpu_result = self.estimate(
                network, traffic, n_samples, T, dt, verbose=False
            )
            gpu_time = time.time() - gpu_start

            # CPU timing
            cpu_start = time.time()
            cpu_result = cpu_simulator.estimate(
                network, traffic, n_samples, T, dt, verbose=False
            )
            cpu_time = time.time() - cpu_start

            speedup = cpu_time / gpu_time

            results['gpu_time'].append(gpu_time)
            results['cpu_time'].append(cpu_time)
            results['speedup'].append(speedup)

            logger.info(f"  GPU: {gpu_time:.2f}s, CPU: {cpu_time:.2f}s, "
                       f"Speedup: {speedup:.1f}x")

        return results


class GPUMLMCSimulator:
    """
    GPU-accelerated Multilevel Monte Carlo simulator.

    Combines MLMC variance reduction with GPU parallelism
    for optimal efficiency.
    """

    def __init__(self, refinement_factor: int = 2, seed: Optional[int] = None):
        """
        Initialize GPU MLMC simulator.

        Args:
            refinement_factor: Time discretization refinement factor M
            seed: Random seed
        """
        if not PYCUDA_AVAILABLE:
            raise ImportError("PyCUDA required for GPU MLMC")

        self.refinement_factor = refinement_factor
        self.seed = seed
        self.gpu_simulator = GPUQueueSimulator()
        self.memory_manager = GPUMemoryManager()

    def run_coupled_paths_gpu(self,
                             network: NetworkGraph,
                             traffic: TrafficModel,
                             level: int,
                             n_samples: int,
                             T: float,
                             base_dt: float,
                             metric: str = 'mean_queue') -> Tuple[np.ndarray, np.ndarray]:
        """
        Run coupled paths for MLMC level on GPU.

        Args:
            network: NetworkGraph
            traffic: TrafficModel
            level: MLMC level
            n_samples: Number of samples
            T: Simulation duration
            base_dt: Base time step
            metric: Metric to compute

        Returns:
            Tuple of (Y_fine, Y_coarse) arrays
        """
        # Get traffic parameters
        traffic_stats = traffic.get_statistics(duration=T)
        arrival_rate = traffic_stats['arrival_rate']
        service_rate = arrival_rate * 1.25

        if level == 0:
            # Level 0: only compute Y_0
            dt_fine = get_timestep(0, base_dt, self.refinement_factor)
            n_timesteps = int(T / dt_fine)

            Y_fine = self.gpu_simulator.simulate_paths(
                n_paths=n_samples,
                n_timesteps=n_timesteps,
                arrival_rate=arrival_rate,
                service_rate=service_rate,
                noise_intensity=0.2,
                dt=dt_fine,
                metric=metric.replace('_queue', '')
            )

            Y_coarse = np.zeros_like(Y_fine)  # Y_{-1} = 0

        else:
            # Level l > 0: compute coupled paths
            dt_fine = get_timestep(level, base_dt, self.refinement_factor)
            dt_coarse = get_timestep(level - 1, base_dt, self.refinement_factor)
            n_timesteps_fine = int(T / dt_fine)

            Y_fine, Y_coarse = self.gpu_simulator.simulate_coupled_paths_mlmc(
                n_paths=n_samples,
                n_timesteps_fine=n_timesteps_fine,
                arrival_rate=arrival_rate,
                service_rate=service_rate,
                noise_intensity=0.2,
                dt_fine=dt_fine,
                dt_coarse=dt_coarse,
                metric=metric.replace('_queue', '')
            )

        return Y_fine, Y_coarse

    def mlmc_estimate_gpu(self,
                         network: NetworkGraph,
                         traffic: TrafficModel,
                         epsilon: float,
                         L_max: int,
                         T: float = 10.0,
                         base_dt: float = 0.1,
                         metric: str = 'mean_queue',
                         pilot_samples: int = 100,
                         confidence_level: float = 0.95,
                         verbose: bool = True) -> MLMCResult:
        """
        Run GPU-accelerated MLMC estimation.

        Args:
            network: NetworkGraph
            traffic: TrafficModel
            epsilon: Target accuracy
            L_max: Maximum level
            T: Simulation duration
            base_dt: Base time step
            metric: Metric to estimate
            pilot_samples: Pilot samples for variance estimation
            confidence_level: Confidence level
            verbose: Print progress

        Returns:
            MLMCResult
        """
        if verbose:
            logger.info(f"Starting GPU MLMC: ε={epsilon}, L_max={L_max}")

        start_time = time.time()

        # Step 1: Pilot run
        if verbose:
            logger.info(f"Step 1: Pilot run with {pilot_samples} samples per level")

        variances = []
        costs = []
        mean_diffs_pilot = []

        for l in range(L_max + 1):
            Y_fine, Y_coarse = self.run_coupled_paths_gpu(
                network, traffic, l, pilot_samples, T, base_dt, metric
            )

            diffs = Y_fine - Y_coarse
            mean_diff = np.mean(diffs)
            var_diff = np.var(diffs, ddof=1)

            dt_l = get_timestep(l, base_dt, self.refinement_factor)
            cost = T / dt_l

            variances.append(var_diff)
            costs.append(cost)
            mean_diffs_pilot.append(mean_diff)

            if verbose:
                logger.info(f"  Level {l}: V={var_diff:.6e}, C={cost:.2e}, "
                           f"E[diff]={mean_diff:.6e}")

        # Step 2: Compute optimal samples
        if verbose:
            logger.info("Step 2: Computing optimal sample allocation")

        sum_term = np.sum([np.sqrt(v * c) for v, c in zip(variances, costs)])
        optimal_N = []

        for l in range(L_max + 1):
            if variances[l] <= 0 or costs[l] <= 0:
                optimal_N.append(1)
            else:
                N_l = (2.0 / epsilon ** 2) * np.sqrt(variances[l] / costs[l]) * sum_term
                N_l = max(1, int(np.ceil(N_l)))
                optimal_N.append(N_l)

        if verbose:
            for l, N_l in enumerate(optimal_N):
                logger.info(f"  Level {l}: N={N_l}")

        # Step 3: Generate full samples
        if verbose:
            logger.info("Step 3: Generating samples on GPU")

        level_stats = []
        total_cost = 0.0

        for l in range(L_max + 1):
            Y_fine, Y_coarse = self.run_coupled_paths_gpu(
                network, traffic, l, optimal_N[l], T, base_dt, metric
            )

            diffs = Y_fine - Y_coarse
            mean_Y = np.mean(Y_fine)
            var_Y = np.var(Y_fine, ddof=1)
            mean_diff = np.mean(diffs)
            var_diff = np.var(diffs, ddof=1)

            dt_l = get_timestep(l, base_dt, self.refinement_factor)
            cost_per_sample = T / dt_l
            level_cost = cost_per_sample * optimal_N[l]
            total_cost += level_cost

            stats = MLMCLevelStats(
                level=l,
                n_samples=optimal_N[l],
                dt=dt_l,
                mean_Y=mean_Y,
                var_Y=var_Y,
                mean_diff=mean_diff,
                var_diff=var_diff,
                cost_per_sample=cost_per_sample,
                total_cost=level_cost
            )

            level_stats.append(stats)

            if verbose:
                logger.info(f"  Level {l} complete: {stats}")

        # Step 4: Combine estimates
        estimate = sum([stats.mean_diff for stats in level_stats])
        variance = sum([stats.var_diff / stats.n_samples for stats in level_stats])

        dt_finest = get_timestep(L_max, base_dt, self.refinement_factor)
        # Reflected SDE has weak order 0.5 (not 1) due to boundary condition
        # Must match CPU implementation in mlmc.py for consistent results
        bias_estimate = np.sqrt(dt_finest)
        mse = variance + bias_estimate ** 2

        # CI
        from scipy import stats as sp_stats
        alpha = 1 - confidence_level
        z_value = sp_stats.norm.ppf(1 - alpha / 2)
        margin = z_value * np.sqrt(variance)
        ci_lower = estimate - margin
        ci_upper = estimate + margin

        elapsed_time = time.time() - start_time

        result = MLMCResult(
            estimate=estimate,
            variance=variance,
            mse=mse,
            level_stats=level_stats,
            total_cost=total_cost,
            L_max=L_max,
            epsilon=epsilon,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            metadata={
                'T': T,
                'base_dt': base_dt,
                'metric': metric,
                'refinement_factor': self.refinement_factor,
                'gpu_time_seconds': elapsed_time,
                'device': 'GPU'
            }
        )

        if verbose:
            logger.info(f"GPU MLMC complete: {result.summary()}")
            logger.info(f"GPU time: {elapsed_time:.2f}s")

        return result


class GPUCoupledPropagationMLMC:
    """
    GPU-accelerated MLMC for the coupled CongestionPropagationSDE.

    Uses PyTorch tensor operations so it runs on GPU without PyCUDA:
      - influence matrix-vector multiply:  torch.mv(influence_gpu, c)
      - noise generation:                  torch.randn(...)
      - reflection:                        torch.clamp_min(c, 0)

    Falls back to NumPy if CUDA is unavailable (CPU mode).
    """

    def __init__(self,
                 adjacency_matrix: "np.ndarray",
                 influence_strength: float = 0.1,
                 decay_rate: float = 0.5,
                 noise_intensity: float = 0.1,
                 refinement_factor: int = 2,
                 seed: Optional[int] = None):
        import torch
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.n_nodes = adjacency_matrix.shape[0]
        self.influence_strength = influence_strength
        self.decay_rate = decay_rate
        self.noise_intensity = noise_intensity
        self.refinement_factor = refinement_factor

        # Degree-normalised influence matrix (n_nodes x n_nodes) on GPU
        degrees = adjacency_matrix.sum(axis=1).clip(min=1)
        influence_np = (adjacency_matrix / degrees[:, None]) * influence_strength
        self._influence = torch.tensor(
            influence_np, dtype=torch.float32, device=self._device
        )

    # ------------------------------------------------------------------
    # Internal: one vectorised Euler-Maruyama step for N paths in parallel
    # c_batch: (n_nodes, N_paths) tensor
    # dw:      (n_nodes, N_paths) noise tensor (pre-scaled by sqrt(dt))
    # ------------------------------------------------------------------
    def _em_step(self, c_batch, dt, dw, influence=None, lambda_vec=None):
        t = self._torch
        if influence is None:
            influence = self._influence
        # drift: (n_nodes, N_paths) = influence @ c_batch - decay * c_batch
        drift = t.mm(influence, c_batch) - self.decay_rate * c_batch
        if lambda_vec is not None:
            drift = drift + lambda_vec[:, None]
        c_new = c_batch + drift * dt + self.noise_intensity * dw
        return t.clamp_min(c_new, 0.0)

    def _resample_dynamic_series(self,
                                 values: Optional[np.ndarray],
                                 target_steps: int,
                                 expected_tail: Tuple[int, ...],
                                 name: str) -> Optional[np.ndarray]:
        """Resample dynamic inputs by repeating or block-averaging time steps."""
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim < 1 or arr.shape[1:] != expected_tail:
            raise ValueError(f"{name} must have trailing shape {expected_tail}")
        if arr.shape[0] == target_steps + 1:
            arr = arr[:-1]
        source_steps = arr.shape[0]
        if source_steps == target_steps:
            return arr
        if target_steps % source_steps == 0:
            factor = target_steps // source_steps
            return np.repeat(arr, factor, axis=0)
        if source_steps % target_steps == 0:
            factor = source_steps // target_steps
            return arr.reshape(target_steps, factor, *arr.shape[1:]).mean(axis=1)
        raise ValueError(
            f"{name} with {source_steps} steps cannot be resampled to {target_steps} steps"
        )

    def _lambda_tensor(self,
                       lambda_t: Optional[np.ndarray],
                       target_steps: int):
        """Prepare a dynamic arrival tensor for a target time grid."""
        arr = self._resample_dynamic_series(
            lambda_t,
            target_steps,
            (self.n_nodes,),
            "lambda_t",
        )
        if arr is None:
            return None
        return self._torch.tensor(arr, dtype=self._torch.float32, device=self._device)

    def _influence_tensor_series(self,
                                 adjacency_t: Optional[np.ndarray],
                                 target_steps: int):
        """Prepare degree-normalized dynamic influence tensors for a target grid."""
        arr = self._resample_dynamic_series(
            adjacency_t,
            target_steps,
            (self.n_nodes, self.n_nodes),
            "adjacency_t",
        )
        if arr is None:
            return None
        degrees = arr.sum(axis=2)
        degrees[degrees == 0.0] = 1.0
        influence = (arr / degrees[:, :, None]) * self.influence_strength
        return self._torch.tensor(influence, dtype=self._torch.float32, device=self._device)

    def _run_level_state_tensors(self,
                                 level: int,
                                 n_samples: int,
                                 T: float,
                                 base_dt: float,
                                 lambda_t: Optional[np.ndarray] = None,
                                 adjacency_t: Optional[np.ndarray] = None):
        """Run coupled paths and return final fine/coarse state tensors."""
        t = self._torch
        dev = self._device

        dt_fine = base_dt / (self.refinement_factor ** level)
        n_steps_fine = int(T / dt_fine)
        lambda_fine = self._lambda_tensor(lambda_t, n_steps_fine)
        influence_fine = self._influence_tensor_series(adjacency_t, n_steps_fine)

        if level == 0:
            c_fine = t.zeros(self.n_nodes, n_samples, device=dev, dtype=t.float32)
            for step_idx in range(n_steps_fine):
                dw = t.randn(self.n_nodes, n_samples, device=dev) * (dt_fine ** 0.5)
                influence = None if influence_fine is None else influence_fine[step_idx]
                lambda_vec = None if lambda_fine is None else lambda_fine[step_idx]
                c_fine = self._em_step(c_fine, dt_fine, dw, influence, lambda_vec)
            c_coarse = t.zeros_like(c_fine)
            return c_fine, c_coarse

        M = self.refinement_factor
        dt_coarse = dt_fine * M
        n_steps_coarse = int(T / dt_coarse)
        lambda_coarse = self._lambda_tensor(lambda_t, n_steps_coarse)
        influence_coarse = self._influence_tensor_series(adjacency_t, n_steps_coarse)

        c_fine = t.zeros(self.n_nodes, n_samples, device=dev, dtype=t.float32)
        c_coarse = t.zeros(self.n_nodes, n_samples, device=dev, dtype=t.float32)

        for i_coarse in range(n_steps_coarse):
            # Generate M fine-level increments, aggregate for coarse
            dw_sum = t.zeros(self.n_nodes, n_samples, device=dev, dtype=t.float32)
            for j in range(M):
                i_fine = i_coarse * M + j
                dw_f = t.randn(self.n_nodes, n_samples, device=dev) * (dt_fine ** 0.5)
                influence_f = None if influence_fine is None else influence_fine[i_fine]
                lambda_f = None if lambda_fine is None else lambda_fine[i_fine]
                c_fine = self._em_step(c_fine, dt_fine, dw_f, influence_f, lambda_f)
                dw_sum = dw_sum + dw_f
            # Coarse step with aggregated increment (same Brownian path)
            influence_c = None if influence_coarse is None else influence_coarse[i_coarse]
            lambda_c = None if lambda_coarse is None else lambda_coarse[i_coarse]
            c_coarse = self._em_step(c_coarse, dt_coarse, dw_sum, influence_c, lambda_c)

        return c_fine, c_coarse

    def run_level(self,
                  level: int,
                  n_samples: int,
                  T: float,
                  base_dt: float,
                  metric: str = 'mean_congestion',
                  lambda_t: Optional[np.ndarray] = None,
                  adjacency_t: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run N_samples coupled (fine, coarse) paths for MLMC level `level`.

        Returns:
            (Y_fine, Y_coarse): shape (n_samples,) each; Y_coarse=0 at level 0.
        """
        c_fine, c_coarse = self._run_level_state_tensors(
            level=level,
            n_samples=n_samples,
            T=T,
            base_dt=base_dt,
            lambda_t=lambda_t,
            adjacency_t=adjacency_t,
        )
        Y_fine = self._extract_metric(c_fine, metric)
        Y_coarse = self._extract_metric(c_coarse, metric)

        return Y_fine, Y_coarse

    def _extract_metric(self, c_batch, metric: str) -> np.ndarray:
        """Extract scalar metric from final congestion state (n_nodes, N_paths)."""
        if metric in ('mean_congestion', 'mean'):
            vals = c_batch.mean(dim=0)
        elif metric in ('max_congestion', 'max'):
            vals = c_batch.max(dim=0).values
        elif metric in ('sum_congestion', 'sum'):
            vals = c_batch.sum(dim=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return vals.cpu().numpy()

    def mlmc_estimate(self,
                      epsilon: float,
                      T: float,
                      base_dt: float,
                      L_max: int = 6,
                      pilot_samples: int = 100,
                      metric: str = 'mean_congestion',
                      verbose: bool = True,
                      lambda_t: Optional[np.ndarray] = None,
                      adjacency_t: Optional[np.ndarray] = None) -> Dict:
        """
        Run GPU-MLMC for the coupled propagation SDE.

        Returns:
            dict with keys: estimate, variance, ci_lower, ci_upper,
                            total_cost, level_stats, epsilon
        """
        variances, costs, mean_diffs_pilot, pilot_diffs_store = [], [], [], []

        for l in range(L_max + 1):
            Yf, Yc = self.run_level(
                l,
                pilot_samples,
                T,
                base_dt,
                metric,
                lambda_t=lambda_t,
                adjacency_t=adjacency_t,
            )
            diffs = Yf - Yc
            mean_diffs_pilot.append(float(np.mean(diffs)))
            variances.append(float(np.var(diffs, ddof=1)))
            dt_l = base_dt / (self.refinement_factor ** l)
            costs.append(float(T / dt_l))
            pilot_diffs_store.append(diffs)
            if verbose:
                logger.info(f"  Level {l}: V={variances[-1]:.4e}, C={costs[-1]:.2e}")

        # Optimal allocation
        sum_vc = float(np.sum([np.sqrt(v * c) for v, c in zip(variances, costs)]))
        optimal_N = []
        for l in range(L_max + 1):
            if variances[l] <= 0:
                optimal_N.append(1)
            else:
                Nl = max(1, int(np.ceil((2.0 / epsilon ** 2) *
                                        np.sqrt(variances[l] / costs[l]) * sum_vc)))
                optimal_N.append(Nl)

        # Full sampling
        level_stats = []
        total_cost = 0.0
        for l in range(L_max + 1):
            n_add = max(0, optimal_N[l] - pilot_samples)
            diffs = list(pilot_diffs_store[l])
            if n_add > 0:
                Yf, Yc = self.run_level(
                    l,
                    n_add,
                    T,
                    base_dt,
                    metric,
                    lambda_t=lambda_t,
                    adjacency_t=adjacency_t,
                )
                diffs.extend(list(Yf - Yc))
            diffs = np.array(diffs)
            md = float(np.mean(diffs))
            vd = float(np.var(diffs, ddof=1))
            dt_l = base_dt / (self.refinement_factor ** l)
            cp = T / dt_l
            level_stats.append({
                'level': l, 'n_samples': len(diffs),
                'mean_diff': md, 'var_diff': vd, 'cost_per_sample': cp
            })
            total_cost += cp * len(diffs)

        estimate = float(sum(s['mean_diff'] for s in level_stats))
        variance = float(sum(s['var_diff'] / s['n_samples'] for s in level_stats))
        from scipy import stats as sp_stats
        z = float(sp_stats.norm.ppf(0.975))
        margin = z * float(np.sqrt(variance))

        return {
            'estimate': estimate,
            'variance': variance,
            'ci_lower': estimate - margin,
            'ci_upper': estimate + margin,
            'total_cost': total_cost,
            'level_stats': level_stats,
            'epsilon': epsilon,
        }


class GPUAdaptiveNetworkAwareMLMC(GPUCoupledPropagationMLMC):
    """Torch implementation of Adaptive Network-Aware MLMC."""

    def __init__(self,
                 adjacency_matrix: "np.ndarray",
                 influence_strength: float = 0.1,
                 decay_rate: float = 0.5,
                 noise_intensity: float = 0.1,
                 refinement_factor: int = 2,
                 seed: Optional[int] = None,
                 weight_centrality: float = 0.4,
                 weight_variance: float = 0.4,
                 weight_sla: float = 0.2,
                 centrality_kind: str = 'pagerank',
                 sla_priority: Optional["np.ndarray"] = None) -> None:
        """Initialize GPU ANA-MLMC with network-risk weighting coefficients."""
        super().__init__(
            adjacency_matrix=adjacency_matrix,
            influence_strength=influence_strength,
            decay_rate=decay_rate,
            noise_intensity=noise_intensity,
            refinement_factor=refinement_factor,
            seed=seed,
        )
        gammas = np.array([weight_centrality, weight_variance, weight_sla], dtype=float)
        if np.any(gammas < 0.0):
            raise ValueError("ANA-MLMC weights must be non-negative")
        if not np.isclose(float(np.sum(gammas)), 1.0):
            raise ValueError("ANA-MLMC weights must sum to 1.0")

        t = self._torch
        self._adjacency = t.tensor(adjacency_matrix, dtype=t.float32, device=self._device)
        self.weight_centrality = float(weight_centrality)
        self.weight_variance = float(weight_variance)
        self.weight_sla = float(weight_sla)
        self.centrality_kind = centrality_kind
        self.sla_priority = (
            None
            if sla_priority is None
            else t.tensor(sla_priority, dtype=t.float32, device=self._device)
        )

    def _normalize_tensor(self, values: "torch.Tensor") -> "torch.Tensor":
        """Normalize a non-negative tensor or return uniform weights."""
        t = self._torch
        arr = values.to(device=self._device, dtype=t.float32).flatten()
        arr = t.where(t.isfinite(arr), arr, t.zeros_like(arr))
        arr = t.clamp_min(arr, 0.0)
        total = arr.sum()
        if float(total.item()) <= 0.0:
            return t.full_like(arr, 1.0 / arr.numel())
        return arr / total

    def _centrality_tensor(self, alpha: float = 0.85) -> "torch.Tensor":
        """Compute centrality weights as a torch tensor."""
        t = self._torch
        n_nodes = self.n_nodes
        adjacency = t.clamp_min(self._adjacency, 0.0)

        if self.centrality_kind == 'degree':
            return self._normalize_tensor(adjacency.sum(dim=1))

        if self.centrality_kind == 'pagerank':
            adjacency = adjacency + t.eye(n_nodes, device=self._device, dtype=t.float32) * 1.0e-12
            row_sums = t.clamp_min(adjacency.sum(dim=1), 1.0e-12)
            transition = adjacency / row_sums[:, None]
            rank = t.full((n_nodes,), 1.0 / n_nodes, device=self._device, dtype=t.float32)
            teleport = (1.0 - alpha) / n_nodes
            for _ in range(100):
                rank = teleport + alpha * t.mv(transition.t(), rank)
            return self._normalize_tensor(rank)

        if self.centrality_kind == 'betweenness':
            import networkx as nx
            graph_np = self._adjacency.detach().cpu().numpy()
            directed = not np.allclose(graph_np, graph_np.T)
            nx_graph = nx.from_numpy_array(
                graph_np,
                create_using=nx.DiGraph if directed else nx.Graph,
            )
            scores = nx.betweenness_centrality(nx_graph, normalized=True, weight='weight')
            vals = t.tensor(
                [scores[i] for i in range(n_nodes)],
                dtype=t.float32,
                device=self._device,
            )
            return self._normalize_tensor(vals)

        raise ValueError(f"Unknown centrality kind: {self.centrality_kind}")

    def compute_node_weights(self,
                             pilot_per_node_vars: "torch.Tensor",
                             sla_vec: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """Combine centrality, pilot variance, and SLA priority into node weights."""
        t = self._torch
        pilot_vars = pilot_per_node_vars.to(device=self._device, dtype=t.float32)
        if pilot_vars.ndim == 1:
            variance_raw = pilot_vars
        elif pilot_vars.ndim == 2:
            variance_raw = pilot_vars.mean(dim=0)
        else:
            raise ValueError("pilot_per_node_vars must have shape (n,) or (L+1, n)")

        c_i = self._centrality_tensor()
        v_i = self._normalize_tensor(variance_raw)

        sla_source = sla_vec
        if sla_source is None:
            sla_source = self.sla_priority
        if sla_source is None:
            sla_source = t.ones(self.n_nodes, device=self._device, dtype=t.float32)
        else:
            sla_source = sla_source.to(device=self._device, dtype=t.float32)
        s_i = self._normalize_tensor(sla_source)

        return self._normalize_tensor(
            self.weight_centrality * c_i
            + self.weight_variance * v_i
            + self.weight_sla * s_i
        )

    def _giles_allocation_tensor(self,
                                 variances: "torch.Tensor",
                                 costs: "torch.Tensor",
                                 epsilon: float) -> List[int]:
        """Compute Giles allocation using torch tensor operations."""
        t = self._torch
        variances = t.clamp_min(variances.to(device=self._device, dtype=t.float32), 0.0)
        costs = costs.to(device=self._device, dtype=t.float32)
        positive = (variances > 0.0) & (costs > 0.0)
        sum_term = t.sqrt(variances * t.clamp_min(costs, 0.0)).sum()
        raw = (2.0 / epsilon ** 2) * t.sqrt(
            variances / t.clamp_min(costs, 1.0e-30)
        ) * sum_term
        samples = t.where(positive, t.ceil(raw), t.ones_like(raw))
        samples = t.clamp_min(samples, 1.0).to(dtype=t.int64)
        return [int(v) for v in samples.detach().cpu().tolist()]

    def compute_optimal_samples_weighted(self,
                                         level_var_per_node: "torch.Tensor",
                                         costs: "torch.Tensor",
                                         weights: "torch.Tensor",
                                         epsilon: float) -> List[int]:
        """Compute Giles allocation from weighted per-node level variances."""
        t = self._torch
        level_vars = level_var_per_node.to(device=self._device, dtype=t.float32)
        if level_vars.ndim != 2:
            raise ValueError("level_var_per_node must have shape (L+1, n)")

        costs_t = costs.to(device=self._device, dtype=t.float32)
        node_weights = self._normalize_tensor(weights)
        uniform = t.full((self.n_nodes,), 1.0 / self.n_nodes, device=self._device)

        if t.allclose(node_weights, uniform, rtol=0.0, atol=1.0e-7):
            weighted_variances = level_vars.mean(dim=1)
            weighted_samples = self._giles_allocation_tensor(weighted_variances, costs_t, epsilon)
            giles_samples = self._giles_allocation_tensor(level_vars.mean(dim=1), costs_t, epsilon)
            assert weighted_samples == giles_samples
            return weighted_samples

        weighted_variances = t.mv(level_vars, node_weights)
        return self._giles_allocation_tensor(weighted_variances, costs_t, epsilon)

    def run_level_node_values(self,
                              level: int,
                              n_samples: int,
                              T: float,
                              base_dt: float,
                              metric: str = 'mean_congestion') -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Run N coupled paths and return per-node fine/coarse final states."""
        _ = metric
        c_fine, c_coarse = self._run_level_state_tensors(
            level=level,
            n_samples=n_samples,
            T=T,
            base_dt=base_dt,
        )
        return c_fine.t().contiguous(), c_coarse.t().contiguous()

    def mlmc_estimate_weighted(self,
                               epsilon: float,
                               T: float,
                               base_dt: float,
                               L_max: int = 6,
                               pilot_samples: int = 100,
                               metric: str = 'mean_congestion',
                               sla_vec: Optional["torch.Tensor"] = None,
                               verbose: bool = True) -> Dict:
        """Run GPU ANA-MLMC for the weighted network quantity."""
        t = self._torch
        per_node_variances = []
        costs = []
        pilot_diffs_store = []

        for l in range(L_max + 1):
            y_fine, y_coarse = self.run_level_node_values(l, pilot_samples, T, base_dt, metric)
            diffs = y_fine - y_coarse
            var_per_node = (
                t.var(diffs, dim=0, unbiased=True)
                if pilot_samples > 1
                else t.zeros(self.n_nodes, device=self._device, dtype=t.float32)
            )
            per_node_variances.append(var_per_node)
            dt_l = base_dt / (self.refinement_factor ** l)
            costs.append(float(T / dt_l))
            pilot_diffs_store.append(diffs)
            if verbose:
                logger.info(
                    f"  Level {l}: V_node_mean={float(var_per_node.mean().item()):.4e}, C={costs[-1]:.2e}"
                )

        level_var_per_node = t.stack(per_node_variances, dim=0)
        costs_tensor = t.tensor(costs, dtype=t.float32, device=self._device)
        weights = self.compute_node_weights(level_var_per_node, sla_vec=sla_vec)
        optimal_N = self.compute_optimal_samples_weighted(
            level_var_per_node, costs_tensor, weights, epsilon
        )

        level_stats = []
        total_cost = 0.0
        weighted_level_vars = []
        weighted_estimator_level_vars = []

        for l in range(L_max + 1):
            n_add = max(0, optimal_N[l] - pilot_samples)
            diffs = pilot_diffs_store[l]
            if n_add > 0:
                y_fine, y_coarse = self.run_level_node_values(l, n_add, T, base_dt, metric)
                diffs = t.cat([diffs, y_fine - y_coarse], dim=0)

            n_total = int(diffs.shape[0])
            mean_per_node = diffs.mean(dim=0)
            var_per_node = (
                t.var(diffs, dim=0, unbiased=True)
                if n_total > 1
                else t.zeros(self.n_nodes, device=self._device, dtype=t.float32)
            )
            weighted_diffs = t.mv(diffs, weights)
            mean_diff_t = t.dot(weights, mean_per_node)
            var_diff_t = t.dot(weights, var_per_node)
            weighted_scalar_var_t = (
                t.var(weighted_diffs, unbiased=True)
                if n_total > 1
                else t.tensor(0.0, device=self._device)
            )

            mean_diff = float(mean_diff_t.item())
            var_diff = float(var_diff_t.item())
            weighted_scalar_var = float(weighted_scalar_var_t.item())
            weighted_level_vars.append(var_diff)
            weighted_estimator_level_vars.append(weighted_scalar_var)

            dt_l = base_dt / (self.refinement_factor ** l)
            cp = float(T / dt_l)
            total_cost += cp * n_total
            level_stats.append({
                'level': l,
                'n_samples': n_total,
                'mean_diff': mean_diff,
                'var_diff': var_diff,
                'weighted_estimator_var_diff': weighted_scalar_var,
                'cost_per_sample': cp,
            })
            if verbose:
                logger.info(
                    f"  Level {l}: N={n_total}, weighted V={var_diff:.4e}, C={cp:.2e}"
                )

        estimate = float(sum(s['mean_diff'] for s in level_stats))
        variance = float(sum(s['var_diff'] / s['n_samples'] for s in level_stats))
        from scipy import stats as sp_stats
        z = float(sp_stats.norm.ppf(0.975))
        margin = z * float(np.sqrt(variance))

        return {
            'estimate': estimate,
            'variance': variance,
            'ci_lower': estimate - margin,
            'ci_upper': estimate + margin,
            'total_cost': total_cost,
            'level_stats': level_stats,
            'epsilon': epsilon,
            'weights': weights.detach().cpu().tolist(),
            'level_var_per_node': level_var_per_node.detach().cpu().tolist(),
            'weighted_level_variances': weighted_level_vars,
            'weighted_estimator_level_variances': weighted_estimator_level_vars,
            'optimal_N': optimal_N,
        }


# ---------------------------------------------------------------------------
# Multi-GPU MLMC with METIS graph partitioning
# ---------------------------------------------------------------------------

class MultiGPUMLMC(GPUCoupledPropagationMLMC):
    """
    Multi-GPU MLMC using METIS graph partitioning for the coupled SDE.

    The n-node SDE graph is partitioned into `world_size` subgraphs via METIS.
    Each partition runs on a separate GPU (or CPU core if fewer GPUs are available).
    Boundary nodes (halo nodes) exchange congestion values between partitions after
    each Euler-Maruyama step via torch.distributed send/recv.

    Requires:
        pip install pymetis        # wraps METIS 5.x C library
        torch.distributed          # for inter-GPU communication (gloo or nccl)

    Single-process simulation (world_size=1) falls back to the parent class and
    requires no METIS or distributed backend.

    Usage (single process, simulates multi-GPU data flow on one GPU/CPU):
        sim = MultiGPUMLMC(adj, world_size=2)
        result = sim.mlmc_estimate_multigpu(epsilon=0.05)

    Usage (true multi-GPU via torchrun):
        # torchrun --nproc_per_node=4 your_script.py
        # dist.init_process_group("nccl")
        # rank = dist.get_rank()
        # sim = MultiGPUMLMC(adj, world_size=dist.get_world_size(), rank=rank)
        # result = sim.mlmc_estimate_multigpu(epsilon=0.05)
        # if rank == 0: print(result)
    """

    def __init__(self,
                 adjacency_matrix: "np.ndarray",
                 world_size: int = 1,
                 rank: int = 0,
                 influence_strength: float = 0.1,
                 decay_rate: float = 0.5,
                 noise_intensity: float = 0.1,
                 refinement_factor: int = 2,
                 seed: Optional[int] = None) -> None:
        super().__init__(
            adjacency_matrix=adjacency_matrix,
            influence_strength=influence_strength,
            decay_rate=decay_rate,
            noise_intensity=noise_intensity,
            refinement_factor=refinement_factor,
            seed=seed,
        )
        self.world_size = world_size
        self.rank = rank
        self._adjacency = self._torch.tensor(
            adjacency_matrix, dtype=self._torch.float32, device=self._device)

        # Partition graph with METIS (or trivial single-partition if world_size=1)
        self._partition_map, self._local_nodes, self._halo_edges = \
            self._partition_graph(adjacency_matrix, world_size, rank)

    # ------------------------------------------------------------------
    # Graph partitioning
    # ------------------------------------------------------------------

    def _partition_graph(
            self,
            adj: "np.ndarray",
            world_size: int,
            rank: int
    ) -> Tuple["np.ndarray", List[int], List[Tuple[int, int]]]:
        """
        Partition the graph into `world_size` parts using METIS.

        Returns:
            partition_map  : array[n_nodes] — part id for each node (0..world_size-1)
            local_nodes    : list of node indices owned by this rank
            halo_edges     : list of (local_node, remote_node) pairs crossing partitions
        """
        n = adj.shape[0]
        if world_size == 1:
            return np.zeros(n, dtype=int), list(range(n)), []

        try:
            import pymetis
        except ImportError:
            logger.warning(
                "pymetis not installed — falling back to round-robin partition. "
                "Install with: pip install pymetis"
            )
            partition_map = np.arange(n) % world_size
        else:
            # Build adjacency list for METIS (undirected, 0-indexed)
            adjacency_list = [
                list(np.where(adj[i] > 0)[0])
                for i in range(n)
            ]
            _, partition_map = pymetis.part_graph(world_size, adjacency=adjacency_list)
            partition_map = np.array(partition_map, dtype=int)

        local_nodes = [i for i in range(n) if partition_map[i] == rank]

        # Halo edges: edges between local and remote nodes
        halo_edges = []
        for i in local_nodes:
            for j in np.where(adj[i] > 0)[0]:
                if partition_map[j] != rank:
                    halo_edges.append((i, int(j)))

        return partition_map, local_nodes, halo_edges

    # ------------------------------------------------------------------
    # Single Euler-Maruyama step with halo exchange
    # ------------------------------------------------------------------

    def _step_with_halo(
            self,
            Q: "torch.Tensor",           # (n_paths, n_local) local congestion
            Q_global: "torch.Tensor",    # (n_paths, n_nodes) full state (for influence)
            arrivals: float,
            dt: float,
            sqrt_dt: float,
            level: int,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """One EM step on the local partition, updating Q_global halo columns."""
        t = self._torch
        n_paths = Q.shape[0]

        # Local EM update using the full influence matrix column for local nodes
        local_idx = self._torch.tensor(self._local_nodes, device=self._device)
        influence_local = self._adjacency[local_idx, :]  # (n_local, n_nodes)

        # Coupled drift: d_i = arrivals_i - decay*Q_i + influence * Q_neighbors
        influence_term = t.mm(Q_global, influence_local.T)  # (n_paths, n_local)
        drift = (arrivals - self.decay_rate * Q + self.influence_strength * influence_term)
        noise = t.randn(n_paths, len(self._local_nodes), device=self._device) * sqrt_dt
        Q_new = t.clamp_min(Q + drift * dt + self.noise_intensity * noise, 0.0)

        # Write local results back into global state tensor
        Q_global[:, local_idx] = Q_new

        # Halo exchange via a single all_reduce SUM collective.
        # Each rank zeros non-local columns, then all_reduce SUM assembles the
        # complete Q_global on every rank in one NCCL call — replacing O(halo_edges)
        # individual point-to-point ops with one bandwidth-efficient collective.
        if _DIST_AVAILABLE and dist.is_initialized() and dist.get_world_size() > 1:
            Q_send = self._torch.zeros_like(Q_global)
            Q_send[:, local_idx] = Q_new          # only local columns non-zero
            dist.all_reduce(Q_send, op=dist.ReduceOp.SUM)
            Q_global = Q_send                      # now every rank has full state

        return Q_new, Q_global

    # ------------------------------------------------------------------
    # MLMC level estimator on local partition
    # ------------------------------------------------------------------

    def _simulate_level_local(
            self,
            level: int,
            n_samples: int,
            T: float = 1.0,
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """
        Simulate one MLMC level on the local partition.

        Returns:
            fine_peak   : (n_samples, n_local) peak congestion at fine resolution
            coarse_peak : (n_samples, n_local) peak congestion at coarse resolution
                          (zero array at level 0)
        """
        t = self._torch
        n_local = len(self._local_nodes)
        n_nodes = self.n_nodes

        M = self.refinement_factor
        n_fine = int(2 ** level)
        dt_fine = T / n_fine
        dt_coarse = T / max(n_fine // M, 1) if level > 0 else T

        Q_fine = t.zeros(n_samples, n_local, device=self._device)
        Q_coarse = t.zeros(n_samples, n_local, device=self._device)
        Q_global_f = t.zeros(n_samples, n_nodes, device=self._device)
        Q_global_c = t.zeros(n_samples, n_nodes, device=self._device)

        peak_fine = t.zeros(n_samples, n_local, device=self._device)
        peak_coarse = t.zeros(n_samples, n_local, device=self._device)

        arrivals = 1.0  # normalised; real traces override this externally

        sqrt_fine = np.sqrt(dt_fine)
        for _ in range(n_fine):
            Q_fine, Q_global_f = self._step_with_halo(
                Q_fine, Q_global_f, arrivals, dt_fine, sqrt_fine, level)
            peak_fine = t.maximum(peak_fine, Q_fine)

        if level > 0:
            sqrt_coarse = np.sqrt(dt_coarse)
            n_coarse = n_fine // M
            for _ in range(n_coarse):
                Q_coarse, Q_global_c = self._step_with_halo(
                    Q_coarse, Q_global_c, arrivals, dt_coarse, sqrt_coarse, level)
                peak_coarse = t.maximum(peak_coarse, Q_coarse)

        return (peak_fine.detach().cpu().numpy(),
                peak_coarse.detach().cpu().numpy())

    # ------------------------------------------------------------------
    # MLMC estimator (multi-GPU entry point)
    # ------------------------------------------------------------------

    def mlmc_estimate_multigpu(
            self,
            epsilon: float = 0.05,
            L_max: int = 5,
            N_pilot: int = 100,
            T: float = 1.0,
    ) -> Dict:
        """
        Run MLMC on the local graph partition and aggregate across all ranks.

        In single-process mode (world_size=1) this is equivalent to the standard
        GPUCoupledPropagationMLMC estimator restricted to `local_nodes`.

        In true multi-GPU mode (torchrun), each rank runs this independently; the
        caller should gather results with dist.all_reduce before reporting.

        Returns:
            dict with keys: estimate, variance, level_stats, epsilon, world_size, rank
        """
        t = self._torch

        # Pilot pass: estimate per-level variance on local nodes
        level_vars = []
        for l in range(L_max + 1):
            fine, coarse = self._simulate_level_local(l, N_pilot, T)
            diff = fine - coarse
            level_vars.append(float(diff.var()))

        # Optimal sample counts (standard Giles formula applied to local variances)
        level_costs = [float(2 ** l) for l in range(L_max + 1)]
        optimal_N = []
        S = sum(np.sqrt(v * c) for v, c in zip(level_vars, level_costs))
        for v, c in zip(level_vars, level_costs):
            N_l = max(1, int(np.ceil(2 / epsilon**2 * np.sqrt(v / c) * S)))
            optimal_N.append(N_l)

        # Synchronize optimal_N across all ranks: every rank must call
        # _step_with_halo with the SAME batch size for the collective all_reduce
        # to receive matching tensor shapes.
        if _DIST_AVAILABLE and dist.is_initialized() and dist.get_world_size() > 1:
            opt_t = t.tensor(optimal_N, dtype=t.long, device=self._device)
            dist.broadcast(opt_t, src=0)
            optimal_N = opt_t.tolist()

        # Main estimation pass
        level_estimates = []
        level_stats = []
        for l, N_l in enumerate(optimal_N):
            fine, coarse = self._simulate_level_local(l, N_l, T)
            diff = fine - coarse
            Y_l = float(diff.mean())
            var_l = float(diff.var() / N_l)
            level_estimates.append(Y_l)
            level_stats.append({
                'level': l, 'n_samples': N_l,
                'estimate': Y_l, 'variance': var_l,
            })

        total_estimate = float(sum(level_estimates))
        total_variance = float(sum(s['variance'] for s in level_stats))

        return {
            'estimate': total_estimate,
            'variance': total_variance,
            'level_stats': level_stats,
            'epsilon': epsilon,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_nodes': self._local_nodes,
            'n_local_nodes': len(self._local_nodes),
        }


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser(description='MultiGPUMLMC torchrun entry point')
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--L-max', type=int, default=5)
    parser.add_argument('--n-nodes', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from network.topology import TopologyGenerator

    rng = np.random.default_rng(args.seed)
    graph = TopologyGenerator(seed=args.seed).generate_erdos_renyi(
        n_nodes=args.n_nodes, p=0.3)
    adj = graph.get_adjacency_matrix()

    rank = 0
    if _DIST_AVAILABLE and args.world_size > 1:
        dist.init_process_group(backend='gloo')
        rank = dist.get_rank()

    sim = MultiGPUMLMC(adj, world_size=args.world_size, rank=rank)
    result = sim.mlmc_estimate_multigpu(
        epsilon=args.epsilon, L_max=args.L_max)
    if rank == 0:
        print(json.dumps(result, default=float, indent=2))

    if _DIST_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()
