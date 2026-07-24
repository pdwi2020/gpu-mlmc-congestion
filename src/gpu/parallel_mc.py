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
                optimal_N.append(2)
            else:
                N_l = (2.0 / epsilon ** 2) * np.sqrt(variances[l] / costs[l]) * sum_term
                N_l = max(2, int(np.ceil(N_l)))  # min 2 for ddof=1 variance
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

    def estimate(self, *args, **kwargs) -> MLMCResult:
        """Alias for mlmc_estimate_gpu for API consistency with CPU MLMCSimulator."""
        return self.mlmc_estimate_gpu(*args, **kwargs)


class GPUCoupledPropagationMLMC:
    """
    GPU-accelerated MLMC for the coupled CongestionPropagationSDE.

    Uses PyTorch tensor operations so it runs on GPU without PyCUDA:
      - influence matrix-vector multiply:  torch.mv(influence_gpu, c)
      - noise generation:                  torch.randn(...)
      - reflection:                        torch.clamp_min(c, 0)

    Falls back to NumPy if CUDA is unavailable (CPU mode).
    """

    #: Selectable reflection / discretisation schemes (see `reflection` arg).
    _REFLECTION_SCHEMES = ("predictor_corrector", "euler_clamp")
    #: Selectable adaptive local-error indicators (see `adaptive_error_estimator`).
    _ADAPTIVE_ESTIMATORS = ("embedded", "half_step")
    #: Path roles that carry independent adaptive step-size state.
    _PATH_ROLES = ("fine", "coarse")

    def __init__(self,
                 adjacency_matrix: "np.ndarray",
                 influence_strength: float = 0.1,
                 decay_rate: float = 0.5,
                 noise_intensity: float = 0.1,
                 refinement_factor: int = 2,
                 seed: Optional[int] = None,
                 adaptive_stepping: bool = False,
                 adaptive_rtol: float = 0.1,
                 adaptive_error_estimator: str = "embedded",
                 reflection: str = "predictor_corrector",
                 adaptive_diagnostics: bool = False,
                 device: Optional[str] = None):
        """Initialise the GPU-parallel coupled-propagation MLMC simulator.

        Args:
            adaptive_stepping: enable SIMT two-bucket adaptive pathwise stepping
                (Section "Adaptive Time Stepping").  Independent of `reflection`.
            adaptive_rtol: refinement tolerance tau on the relative local-error
                indicator.  Ignored when `adaptive_stepping` is False.
            adaptive_error_estimator: which local-error indicator drives bucket
                assignment.  "embedded" (default) reads the error off the
                integrator's own internal stages and costs no extra matmul;
                "half_step" is the manuscript's full-step-vs-two-half-steps
                comparison, which must evaluate both estimates for every path
                and therefore costs 3x the drift work per step.
            reflection: boundary/discretisation scheme.
                "predictor_corrector" (default) is the half-step
                predictor-corrector reflection used for every published number;
                "euler_clamp" is plain Euler-Maruyama with clamping at zero, for
                ablating the corrector stage.
            adaptive_diagnostics: record the per-step distribution of the local
                error indicator alongside bucket occupancy.  Costs a sort and a
                device synchronisation per step, so it defaults to False and
                should stay off whenever wall-clock is being measured.
            device: force a torch device ('cpu' or 'cuda').  Default None keeps
                the previous behaviour of selecting CUDA when available.  Used
                by the device-matched CPU-vs-GPU baseline so both arms run the
                identical code and estimate the identical quantity.
        """
        import torch
        self._torch = torch
        # `device` exists so the SAME code path can be timed on CPU and on GPU.
        # Without it a "CPU baseline" has to be a second implementation, and the
        # two then estimate subtly different quantities -- which is exactly the
        # defect found in the stored cpu_vs_gpu_mlmc.json, whose CPU and GPU arms
        # disagreed sevenfold on the same estimand.  Default None keeps the
        # previous auto-select behaviour.
        self._device = (torch.device(device) if device is not None
                        else torch.device("cuda" if torch.cuda.is_available()
                                          else "cpu"))
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.n_nodes = adjacency_matrix.shape[0]
        self.influence_strength = influence_strength
        self.decay_rate = decay_rate
        self.noise_intensity = noise_intensity
        self.refinement_factor = refinement_factor
        # --- Independently switchable scheme components (ablation ladder) ----
        # Adaptive stepping: when True the path ensemble is split into a
        # full-step and a half-step bucket at every nominal step.
        self.adaptive_stepping = bool(adaptive_stepping)
        self.adaptive_rtol = float(adaptive_rtol)

        if adaptive_error_estimator not in self._ADAPTIVE_ESTIMATORS:
            raise ValueError(
                f"adaptive_error_estimator must be one of "
                f"{sorted(self._ADAPTIVE_ESTIMATORS)}, got {adaptive_error_estimator!r}"
            )
        self.adaptive_error_estimator = adaptive_error_estimator

        if reflection not in self._REFLECTION_SCHEMES:
            raise ValueError(
                f"reflection must be one of {sorted(self._REFLECTION_SCHEMES)}, "
                f"got {reflection!r}"
            )
        self.reflection = reflection
        self.adaptive_diagnostics = bool(adaptive_diagnostics)

        # Per-path step-size scale factors, one tensor per path role
        # ('fine' / 'coarse'), each of shape (N_paths,) with values in
        # {1.0, 0.5}: exactly two buckets, as claimed in the manuscript.
        # Allocated lazily and reset at the start of every run.
        self._adaptive_h_scale: Dict[str, "torch.Tensor"] = {}
        # Per-step diagnostics: list of dicts, one per nominal step per role.
        self.adaptive_bucket_history: List[Dict[str, float]] = []
        # Count of drift matmuls issued by the adaptive stepper (test hook).
        self.adaptive_mm_calls: int = 0
        # Steps whose refine mask was uniform, and so took the fast path that
        # skips the bucket gather/scatter entirely.
        self._adaptive_uniform_steps: int = 0

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
        noise = self.noise_intensity * dw

        if self.reflection == "euler_clamp":
            # Ablation variant: plain Euler-Maruyama, reflection by clamping at
            # zero.  One drift evaluation, no corrector.
            drift_e = t.mm(influence, c_batch) - self.decay_rate * c_batch
            if lambda_vec is not None:
                drift_e = drift_e + lambda_vec[:, None]
            return t.clamp_min(c_batch + drift_e * dt + noise, 0.0)

        # Predictor: drift at current state, reflect
        drift_n = t.mm(influence, c_batch) - self.decay_rate * c_batch
        if lambda_vec is not None:
            drift_n = drift_n + lambda_vec[:, None]
        c_pred = t.clamp_min(c_batch + drift_n * dt + noise, 0.0)

        # Corrector: re-evaluate drift at reflected prediction, same dW
        drift_pred = t.mm(influence, c_pred) - self.decay_rate * c_pred
        if lambda_vec is not None:
            drift_pred = drift_pred + lambda_vec[:, None]
        return t.clamp_min(c_batch + drift_pred * dt + noise, 0.0)

    def reset_adaptive_state(self) -> None:
        """Reset persistent per-path step-size scales to 1.0.

        Must be called at the start of each new simulation run so that
        step-size history from a previous run does not contaminate the next.
        Also clears the per-step bucket-occupancy diagnostics.
        """
        self._adaptive_h_scale = {}
        self.adaptive_bucket_history = []
        self.adaptive_mm_calls = 0
        self._adaptive_uniform_steps = 0

    # ------------------------------------------------------------------
    # Adaptive stepping: two-bucket SIMT-friendly pathwise refinement
    # ------------------------------------------------------------------
    def _bucket_step(self, c, dt, dw, influence, lambda_vec):
        """Advance one contiguous bucket of paths by `dt`.

        Issues exactly ONE ``torch.mm`` per drift evaluation over the whole
        bucket -- one for ``euler_clamp``, two for the predictor-corrector.  No
        per-path work of any kind occurs here, which is the property that keeps
        SIMT lanes coherent when step sizes differ across the ensemble.

        Returns:
            (c_new, err) where `err` is the embedded relative local-error
            indicator per path, shape (bucket_size,).  For the
            predictor-corrector this is the corrector-minus-predictor
            difference (a standard embedded estimate, free of extra matmuls);
            for plain Euler it degrades to the relative state increment.
        """
        t = self._torch
        noise = self.noise_intensity * dw

        drift_n = t.mm(influence, c) - self.decay_rate * c
        self.adaptive_mm_calls += 1
        if lambda_vec is not None:
            drift_n = drift_n + lambda_vec[:, None]
        c_pred = t.clamp_min(c + drift_n * dt + noise, 0.0)

        if self.reflection == "euler_clamp":
            err = (t.abs(c_pred - c) / (t.abs(c_pred) + 1.0)).amax(dim=0)
            return c_pred, err

        drift_pred = t.mm(influence, c_pred) - self.decay_rate * c_pred
        self.adaptive_mm_calls += 1
        if lambda_vec is not None:
            drift_pred = drift_pred + lambda_vec[:, None]
        c_new = t.clamp_min(c + drift_pred * dt + noise, 0.0)
        err = (t.abs(c_new - c_pred) / (t.abs(c_new) + 1.0)).amax(dim=0)
        return c_new, err

    def _bucket_advance(self, c, dt, dw, influence, lambda_vec, refine: bool):
        """Advance a bucket across the full nominal step `dt`.

        `refine=False` (full-step bucket): one step of size `dt`.
        `refine=True`  (half-step bucket): two sub-steps of size `dt/2`.

        Noise splitting.  The two sub-steps each receive ``dw/2`` so that the
        sub-increments sum to the outer increment *exactly*.  This is the
        conditional mean of the Brownian bridge, not a bridge sample: drawing
        the true bridge would consume an extra normal per refined path, which
        would make randomness consumption depend on the refinement pattern and
        silently break the fine/coarse common-random-number coupling that the
        MLMC variance reduction rests on.  Because the diffusion coefficient is
        additive (constant ``noise_intensity``), the total noise injected over
        the nominal step is identical to the unrefined step; only the drift is
        resolved more finely.
        """
        t = self._torch
        if self.adaptive_error_estimator == "half_step":
            # Manuscript indicator: one full step vs two half steps, same
            # Wiener increment.  Both estimates must be formed for every path,
            # so this costs 3x the drift work regardless of bucket occupancy.
            c_full, _ = self._bucket_step(c, dt, dw, influence, lambda_vec)
            dw_half = dw * 0.5
            c_mid, _ = self._bucket_step(c, dt * 0.5, dw_half, influence, lambda_vec)
            c_two, _ = self._bucket_step(c_mid, dt * 0.5, dw_half, influence, lambda_vec)
            err = (t.abs(c_full - c_two) / (t.abs(c_two) + 1.0)).amax(dim=0)
            return (c_two if refine else c_full), err

        # Embedded indicator: read off the integrator's own stages, no extra mm.
        if refine:
            dw_half = dw * 0.5
            c_mid, err_a = self._bucket_step(c, dt * 0.5, dw_half, influence, lambda_vec)
            c_new, err_b = self._bucket_step(c_mid, dt * 0.5, dw_half, influence, lambda_vec)
            return c_new, t.maximum(err_a, err_b)
        return self._bucket_step(c, dt, dw, influence, lambda_vec)

    def _adaptive_scale(self, role: str, n_paths: int):
        """Fetch (allocating if needed) the per-path step-scale tensor for `role`."""
        t = self._torch
        scale = self._adaptive_h_scale.get(role)
        if scale is None or scale.shape[0] != n_paths:
            scale = t.ones(n_paths, device=self._device)
            self._adaptive_h_scale[role] = scale
        return scale

    def _em_step_adaptive(self, c_batch, dt, dw, influence=None, lambda_vec=None,
                          role: str = "fine"):
        """One nominal step of size `dt` with two-bucket adaptive refinement.

        Paths are partitioned by their current effective step size into exactly
        two buckets -- full step (`h_scale == 1`) and half step
        (`h_scale == 0.5`) -- as claimed in the manuscript.  Each bucket is
        gathered into a contiguous tensor with ``index_select`` and handed to
        :meth:`_bucket_step`, which issues a single batched ``torch.mm`` per
        drift evaluation over the entire bucket.  Nothing in this method scales
        with the number of paths in Python; the matmul count per nominal step
        depends only on which buckets are non-empty, never on `N_paths`.

        Both buckets advance by the *same* nominal `dt`, so every path stays on
        the common time grid; refinement changes how the interval is resolved,
        not how far the path travels.  The caller's nominal grid (and hence the
        MLMC telescoping identity) is therefore untouched -- in particular the
        coarse path keeps its nominal step of ``2 * h_l``.

        `role` selects which path family ('fine' or 'coarse') the persistent
        step-scale state belongs to, so that the fine and coarse paths of an
        MLMC level adapt independently without sharing controller state.
        """
        t = self._torch
        if influence is None:
            influence = self._influence
        if not self.adaptive_stepping:
            return self._em_step(c_batch, dt, dw, influence, lambda_vec)
        if role not in self._PATH_ROLES:
            raise ValueError(f"role must be one of {self._PATH_ROLES}, got {role!r}")

        n_paths = c_batch.shape[1]
        scale = self._adaptive_scale(role, n_paths)

        full_mask = scale >= 1.0
        # One device sync answers both "is the mask uniform?" and "how many
        # paths are refined?", replacing two `nonzero` calls in the common case.
        n_half = int((~full_mask).sum().item())
        self._adaptive_uniform_steps += int(n_half == 0 or n_half == n_paths)

        if n_half == 0 or n_half == n_paths:
            # Uniform-mask fast path.  Measured occupancy on this SDE is
            # overwhelmingly all-full or all-half, and in that regime the
            # gather/scatter machinery is pure overhead: with a single occupied
            # bucket the ensemble is already contiguous.  Operating on
            # `c_batch` directly is bitwise identical to `index_select` with a
            # complete, ordered index (same values, same op order), so this is
            # a cost saving and not a change of scheme.
            c_out, err = self._bucket_advance(
                c_batch, dt, dw, influence, lambda_vec, refine=(n_half != 0))
        else:
            idx_full = full_mask.nonzero(as_tuple=True)[0]
            idx_half = (~full_mask).nonzero(as_tuple=True)[0]

            c_out = t.empty_like(c_batch)
            err = t.zeros(n_paths, device=self._device, dtype=c_batch.dtype)

            # ---- Bucket 1: full-step paths (one batched matmul per drift eval)
            c_new, err_f = self._bucket_advance(
                c_batch.index_select(1, idx_full), dt,
                dw.index_select(1, idx_full),
                influence, lambda_vec, refine=False,
            )
            c_out.index_copy_(1, idx_full, c_new)
            err.index_copy_(0, idx_full, err_f)

            # ---- Bucket 2: half-step paths (one batched matmul per drift eval)
            c_new, err_h = self._bucket_advance(
                c_batch.index_select(1, idx_half), dt,
                dw.index_select(1, idx_half),
                influence, lambda_vec, refine=True,
            )
            c_out.index_copy_(1, idx_half, c_new)
            err.index_copy_(0, idx_half, err_h)

        # ---- Controller: reassign buckets from the realised local error ------
        # Two buckets only, so the scale is binary: refine above tolerance,
        # return to the full step once the error falls well below it.  The
        # hysteresis band (quarter of tau) stops paths oscillating between
        # buckets on consecutive steps.
        tau = self.adaptive_rtol
        needs_half = err > tau
        back_to_full = err < tau * 0.25
        new_scale = t.where(
            needs_half,
            t.full_like(scale, 0.5),
            t.where(back_to_full, t.ones_like(scale), scale),
        )
        self._adaptive_h_scale[role] = new_scale

        # Bucket occupancy is free -- `n_half` was already needed above -- so it
        # is always recorded.  The distribution of the local-error indicator
        # costs a sort and a device sync per step, so it is opt-in via
        # `adaptive_diagnostics`; leaving it off keeps timing runs honest.
        record = {
            "role": role,
            "dt": float(dt),
            "n_full": n_paths - n_half,
            "n_half": n_half,
            "frac_half": float(n_half) / float(max(n_paths, 1)),
        }
        if self.adaptive_diagnostics and n_paths:
            # The spread decides whether mixed buckets are reachable at all: if
            # the indicator concentrates across paths, every path crosses any
            # threshold on the same step and the two-bucket split is structurally
            # incapable of paying for itself.  One sync for all five statistics.
            err_f = err.float()
            quantiles = t.quantile(
                err_f, t.tensor([0.05, 0.5, 0.95], device=err_f.device))
            stats = t.cat([err_f.min().reshape(1), quantiles,
                           err_f.max().reshape(1)]).tolist()
            record.update({
                "err_min": stats[0],
                "err_p05": stats[1],
                "err_p50": stats[2],
                "err_p95": stats[3],
                "err_max": stats[4],
                # Dispersion p95/p05; near 1 means a homogeneous ensemble.
                "err_spread": (stats[3] / stats[1] if stats[1] > 0
                               else float("inf")),
            })

        self.adaptive_bucket_history.append(record)
        return c_out

    def _step(self, c_batch, dt, dw, influence=None, lambda_vec=None,
              role: str = "fine"):
        """Dispatch one nominal step to the fixed or adaptive integrator.

        This is the single entry point used by the MLMC path loops, so the
        `adaptive_stepping` flag actually reaches the simulation.
        """
        if self.adaptive_stepping:
            return self._em_step_adaptive(c_batch, dt, dw, influence, lambda_vec,
                                          role=role)
        return self._em_step(c_batch, dt, dw, influence, lambda_vec)

    def adaptive_work_units(self) -> float:
        """Timestep-path evaluations performed by the adaptive stepper.

        One unit is one path advanced across one sub-step.  A fixed-step run of
        `n_steps` over `N` paths costs ``n_steps * N``; the half-step bucket
        costs double, so this quantifies the extra work refinement bought.
        """
        return float(sum(r["n_full"] + 2 * r["n_half"]
                         for r in self.adaptive_bucket_history))

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

        # Every call starts a fresh ensemble at c=0, so the adaptive controller
        # must start from the full-step bucket rather than inherit step-size
        # history from the previous level or the previous call.
        self.reset_adaptive_state()

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
                c_fine = self._step(c_fine, dt_fine, dw, influence, lambda_vec,
                                    role="fine")
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
                c_fine = self._step(c_fine, dt_fine, dw_f, influence_f, lambda_f,
                                    role="fine")
                dw_sum = dw_sum + dw_f
            # Coarse step with aggregated increment (same Brownian path).  The
            # coarse nominal step stays fixed at M*dt_fine = 2*h_l; adaptivity
            # only subdivides that interval, so the telescoping identity and the
            # common-random-number coupling both survive.
            influence_c = None if influence_coarse is None else influence_coarse[i_coarse]
            lambda_c = None if lambda_coarse is None else lambda_coarse[i_coarse]
            c_coarse = self._step(c_coarse, dt_coarse, dw_sum, influence_c, lambda_c,
                                  role="coarse")

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

    def _apply_control_variate(
        self,
        P_l: "torch.Tensor",
        C_state: "torch.Tensor",
        adj: "torch.Tensor",
    ) -> "torch.Tensor":
        """Per-node network-aware control variate.

        Reduces estimator variance using the neighbourhood mean as a zero-mean
        surrogate.  For node i:
            P_l^(i),CV = P_l^(i) - theta_i * (P_l^(i) - neighbour_mean_i)
        where theta_i is the online OLS coefficient estimated from the pilot batch.

        Args:
            P_l:      level estimator per path, shape (n_nodes, N_paths)
            C_state:  final congestion state per path, shape (n_nodes, N_paths)
            adj:      binary adjacency matrix, shape (n_nodes, n_nodes)

        Returns:
            CV-corrected estimator, same shape as P_l
        """
        import torch as _torch
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        neigh_mean = _torch.mm(adj.float(), C_state) / deg  # (n_nodes, N_paths)

        P_c = P_l - P_l.mean(dim=1, keepdim=True)
        N_c = neigh_mean - neigh_mean.mean(dim=1, keepdim=True)

        cov = (P_c * N_c).mean(dim=1)                         # (n_nodes,)
        var_cv = N_c.pow(2).mean(dim=1).clamp(min=1e-8)       # (n_nodes,)
        theta = (cov / var_cv).unsqueeze(1)                    # (n_nodes, 1)

        return P_l - theta * (P_l - neigh_mean)

    def bootstrap_quantile_ci(
        self,
        level_diffs: list,
        quantile: float = 0.95,
        B: int = 1000,
        alpha: float = 0.05,
    ) -> tuple:
        """Bootstrap CI for a quantile of the MLMC estimator using GPU tensors.

        Pools all level difference samples, draws B bootstrap resamples via
        torch.multinomial (GPU) or numpy (CPU), computes the quantile for each
        resample, and returns the (alpha/2, 1-alpha/2) percentile interval.

        Args:
            level_diffs: list of 1-D arrays/tensors, one per MLMC level
            quantile: target quantile (e.g. 0.95 for P95)
            B: number of bootstrap resamples
            alpha: two-sided significance level

        Returns:
            (lo, hi) bootstrap percentile CI
        """
        import torch as _torch
        arrays = [np.asarray(d, dtype=np.float32).ravel() for d in level_diffs]
        pooled = np.concatenate(arrays)
        n = len(pooled)

        use_gpu = self._device.type != 'cpu' if hasattr(self._device, 'type') else False

        if use_gpu:
            data = _torch.tensor(pooled, device=self._device)
            weights = _torch.ones(n, device=self._device)
            idx = _torch.multinomial(weights, num_samples=B * n, replacement=True).view(B, n)
            boot_q = _torch.quantile(data[idx], quantile, dim=1)
            lo = float(_torch.quantile(boot_q, alpha / 2).item())
            hi = float(_torch.quantile(boot_q, 1.0 - alpha / 2).item())
        else:
            rng = np.random.default_rng(seed=0)
            idx = rng.integers(0, n, size=(B, n))
            boot_q = np.quantile(pooled[idx], quantile, axis=1)
            lo = float(np.quantile(boot_q, alpha / 2))
            hi = float(np.quantile(boot_q, 1.0 - alpha / 2))

        return lo, hi

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
        # Reset per-path adaptive step scales at the start of each fresh run.
        self.reset_adaptive_state()

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
        all_level_diffs = []
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
            all_level_diffs.append(diffs)
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

        # Bootstrap CIs for tail quantiles (replaces Gaussian approximation for P95/P99)
        p95_lo, p95_hi = self.bootstrap_quantile_ci(all_level_diffs, quantile=0.95)
        p99_lo, p99_hi = self.bootstrap_quantile_ci(all_level_diffs, quantile=0.99)

        return {
            'estimate': estimate,
            'variance': variance,
            'ci_lower': estimate - margin,
            'ci_upper': estimate + margin,
            'total_cost': total_cost,
            'level_stats': level_stats,
            'epsilon': epsilon,
            'p95_ci': (p95_lo, p95_hi),
            'p99_ci': (p99_lo, p99_hi),
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
                 seed: Optional[int] = None,
                 halo_exchange_every: int = 1) -> None:
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
        # Temporal blocking: exchange halo every K EM steps (K=1 = every step).
        # For Lipschitz constant L_lip, ghost-value error grows ≤ L_lip*K*h per step,
        # so K = O(1/h) is the theoretical safe upper bound.  Default K=1 is fully
        # correct; K=4 or K=8 can be used to reduce communication overhead in
        # bandwidth-limited settings at the cost of a bounded stale-ghost error.
        self.halo_exchange_every = max(1, int(halo_exchange_every))
        self._adjacency = self._torch.tensor(
            adjacency_matrix, dtype=self._torch.float32, device=self._device)

        # Partition graph with METIS (or trivial single-partition if world_size=1)
        self._partition_map, self._local_nodes, self._halo_edges = \
            self._partition_graph(adjacency_matrix, world_size, rank)

        # Pre-classify local nodes for comm/compute overlap (T21).
        # Interior nodes have no cross-rank edges; boundary nodes have at least one.
        _boundary_set = {i for (i, _) in self._halo_edges}
        self._local_interior_nodes = [n for n in self._local_nodes if n not in _boundary_set]
        self._local_boundary_nodes = [n for n in self._local_nodes if n in _boundary_set]

        # Timing accumulators for comm/compute breakdown.
        # Set timing_enabled=True to collect synchronous comm/compute measurements.
        # In production (False), all_reduce stays fully async for overlap.
        self.timing_enabled: bool = False
        self._timing_compute_s: float = 0.0   # SDE + bookkeeping (backward-compat)
        self._timing_comm_s: float = 0.0       # halo all_reduce (sync mode)
        self._timing_n_steps: int = 0
        # Fine-grained 4-phase breakdown (populated only when timing_enabled=True)
        self._timing_sde_s: float = 0.0          # drift+noise EM kernel
        self._timing_bookkeeping_s: float = 0.0  # Q_global write-back
        self._timing_sync_wait_s: float = 0.0    # handle.wait() blocking time

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
                "pymetis not installed — falling back to degree-weighted striped "
                "partition.  Install with: pip install pymetis for optimal cuts."
            )
            # Degree-weighted striped assignment (O(n log n)):
            # Sort nodes by degree descending, then interleave across ranks.
            # This separates high-degree hubs onto distinct ranks, minimising
            # the number of halo boundary edges.
            degrees = np.asarray(adj, dtype=float).sum(axis=1)
            sorted_by_degree = np.argsort(-degrees)          # descending degree
            partition_map = np.empty(n, dtype=int)
            partition_map[sorted_by_degree] = np.arange(n) % world_size
        else:
            # Build adjacency list for METIS (undirected, 0-indexed)
            adjacency_list = [
                list(np.where(adj[i] > 0)[0])
                for i in range(n)
            ]
            # Degree-based vertex weights: cost(node i) ∝ degree(i) + 1.
            # Per the email (item 3C): "partition that balances both edge cut
            # and computational load, where each node's cost depends on its
            # degree."  METIS will balance total weight across partitions,
            # which keeps high-degree (expensive) nodes off the boundary.
            degrees = np.asarray(adj, dtype=float).sum(axis=1)
            vertex_weights = (degrees + 1).astype(int).tolist()
            _, partition_map = pymetis.part_graph(
                world_size,
                adjacency=adjacency_list,
                vweights=vertex_weights,
            )
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
            pending_handle=None,         # async handle from the previous step
            step_index: int = 0,         # global EM step counter for temporal blocking
    ) -> Tuple["torch.Tensor", "torch.Tensor", object]:
        """One EM step on the local partition with non-blocking halo exchange.

        Returns (Q_new, Q_global, handle) where handle is the async all_reduce
        work handle for this step's comm.  The caller must call handle.wait()
        before starting the next step that reads ghost columns (inter-step
        compute/comm overlap pattern).

        If a pending_handle from the previous step is supplied, it is completed
        here before the local EM update reads Q_global (ensuring ghost values
        are fresh from the prior step's comm).
        """
        import time as _time
        t = self._torch
        _has_cuda = hasattr(t, "cuda") and t.cuda.is_available()

        # ---- Phase 1: Sync — wait for previous step's async comm ----
        if pending_handle is not None:
            if self.timing_enabled:
                _tw = _time.perf_counter()
                pending_handle.wait()
                if _has_cuda:
                    t.cuda.synchronize()
                self._timing_sync_wait_s += _time.perf_counter() - _tw
            else:
                pending_handle.wait()

        # ---- Phase 2: SDE update (drift + noise + clamp) ----
        _t0 = _time.perf_counter()

        n_paths = Q.shape[0]
        local_idx = self._torch.tensor(self._local_nodes, device=self._device)
        influence_local = self._adjacency[local_idx, :]       # (n_local, n_nodes)

        # Coupled drift using current (now-fresh) Q_global
        influence_term = t.mm(Q_global, influence_local.T)    # (n_paths, n_local)
        drift = arrivals - self.decay_rate * Q + self.influence_strength * influence_term
        noise = t.randn(n_paths, len(self._local_nodes), device=self._device) * sqrt_dt
        Q_new = t.clamp_min(Q + drift * dt + self.noise_intensity * noise, 0.0)

        # ---- Phase 3: Bookkeeping — write-back local results to global state ----
        if self.timing_enabled:
            if _has_cuda:
                t.cuda.synchronize()
            _t_sde = _time.perf_counter()
            self._timing_sde_s += _t_sde - _t0
            Q_global[:, local_idx] = Q_new
            if _has_cuda:
                t.cuda.synchronize()
            _t1 = _time.perf_counter()
            self._timing_bookkeeping_s += _t1 - _t_sde
        else:
            Q_global[:, local_idx] = Q_new
            if _has_cuda:
                t.cuda.synchronize()
            _t1 = _time.perf_counter()

        self._timing_compute_s += _t1 - _t0

        # --- Comm phase timing ---
        # Temporal blocking: skip comm on non-exchange steps (stale ghost values).
        # Error grows ≤ L_lip * halo_exchange_every * dt per step (bounded).
        do_exchange = (
            _DIST_AVAILABLE
            and dist.is_initialized()
            and dist.get_world_size() > 1
            and (step_index % self.halo_exchange_every == self.halo_exchange_every - 1)
        )

        handle = None
        if do_exchange:
            Q_send = self._torch.zeros_like(Q_global)
            Q_send[:, local_idx] = Q_new
            if self.timing_enabled:
                # Synchronous comm for accurate timing (disables async overlap)
                _tc0 = _time.perf_counter()
                dist.all_reduce(Q_send, op=dist.ReduceOp.SUM)
                if _has_cuda:
                    t.cuda.synchronize()
                self._timing_comm_s += _time.perf_counter() - _tc0
            else:
                # Async: caller waits on handle before next step (enables overlap)
                handle = dist.all_reduce(Q_send, op=dist.ReduceOp.SUM, async_op=True)
            Q_global = Q_send

        self._timing_n_steps += 1
        return Q_new, Q_global, handle

    def reset_timers(self) -> None:
        """Reset all timing accumulators for a fresh measurement window."""
        self._timing_compute_s = 0.0
        self._timing_comm_s = 0.0
        self._timing_n_steps = 0
        self._timing_sde_s = 0.0
        self._timing_bookkeeping_s = 0.0
        self._timing_sync_wait_s = 0.0

    def comm_compute_ratio(self) -> dict:
        """Return comm/compute timing breakdown over all measured steps.

        When timing_enabled=True the fine-grained 4-phase breakdown is also
        returned (sde_s, bookkeeping_s, sync_wait_s, comm_s). The aggregate
        compute_s = sde_s + bookkeeping_s for backward compatibility.
        """
        total = self._timing_compute_s + self._timing_comm_s + self._timing_sync_wait_s
        out = {
            "compute_s": self._timing_compute_s,
            "comm_s": self._timing_comm_s,
            "sync_wait_s": self._timing_sync_wait_s,
            "total_s": total,
            "comm_frac": self._timing_comm_s / total if total > 0 else 0.0,
            "n_steps": self._timing_n_steps,
        }
        if self.timing_enabled:
            phase_total = (self._timing_sde_s + self._timing_bookkeeping_s
                           + self._timing_comm_s + self._timing_sync_wait_s)
            out["phases"] = {
                "sde_s": self._timing_sde_s,
                "bookkeeping_s": self._timing_bookkeeping_s,
                "halo_s": self._timing_comm_s,
                "sync_s": self._timing_sync_wait_s,
                "sde_pct": 100 * self._timing_sde_s / phase_total if phase_total > 0 else 0,
                "bookkeeping_pct": 100 * self._timing_bookkeeping_s / phase_total if phase_total > 0 else 0,
                "halo_pct": 100 * self._timing_comm_s / phase_total if phase_total > 0 else 0,
                "sync_pct": 100 * self._timing_sync_wait_s / phase_total if phase_total > 0 else 0,
            }
        return out

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

        # Inter-step compute/comm overlap: pass the async handle from step k into
        # step k+1; _step_with_halo waits on the handle before reading Q_global,
        # allowing local compute (peak bookkeeping etc.) to overlap with comm.
        sqrt_fine = np.sqrt(dt_fine)
        handle_f = None
        for step in range(n_fine):
            Q_fine, Q_global_f, handle_f = self._step_with_halo(
                Q_fine, Q_global_f, arrivals, dt_fine, sqrt_fine, level,
                pending_handle=handle_f, step_index=step)
            peak_fine = t.maximum(peak_fine, Q_fine)
        if handle_f is not None:
            handle_f.wait()

        if level > 0:
            sqrt_coarse = np.sqrt(dt_coarse)
            n_coarse = n_fine // M
            handle_c = None
            for step in range(n_coarse):
                Q_coarse, Q_global_c, handle_c = self._step_with_halo(
                    Q_coarse, Q_global_c, arrivals, dt_coarse, sqrt_coarse, level,
                    pending_handle=handle_c, step_index=step)
                peak_coarse = t.maximum(peak_coarse, Q_coarse)
            if handle_c is not None:
                handle_c.wait()

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


# ---------------------------------------------------------------------------
# GPU Importance Sampling MLMC (Freidlin-Wentzell IS walk)
# ---------------------------------------------------------------------------

class GPUImportanceSamplingMLMC(GPUCoupledPropagationMLMC):
    """GPU-accelerated MLMC with Girsanov importance sampling for rare-event estimation.

    Implements a Freidlin-Wentzell IS walk: paths are tilted toward the rare-event
    region via a Girsanov change of measure, and their contributions are weighted by
    the Radon-Nikodym derivative to correct for the tilt.

    The IS drift for queue overflow P(Q > B) on a single-node reflected SDE is:
        h*(t) = (B - Q(t)) / (sigma^2 * (T - t) + eps)
    which pushes paths toward the overflow boundary B.  The discrete Girsanov weight
    accumulated over N steps is:
        log w = -sum_k [ h*(t_k) * dW_k  +  0.5 * h*(t_k)^2 * dt ]
    The self-normalised IS estimator is:
        P_hat = sum(w_i * 1{Q_i > B}) / sum(w_i)

    Effective sample size (ESS) measures weight degeneracy:
        ESS = (sum w_i)^2 / sum(w_i^2)

    Args:
        adjacency_matrix: n×n adjacency matrix (same as parent class).
        overflow_threshold: target overflow level B for P(Q > B) estimation.
        is_strength: scaling factor for the IS drift (default 1.0 = full Girsanov tilt).
        All other args: forwarded to GPUCoupledPropagationMLMC.
    """

    def __init__(self,
                 adjacency_matrix: "np.ndarray",
                 overflow_threshold: float = 5.0,
                 is_strength: float = 1.0,
                 **kwargs):
        super().__init__(adjacency_matrix, **kwargs)
        self.overflow_threshold = overflow_threshold
        self.is_strength = is_strength

    def simulate_is_paths(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        target_node: int = 0,
    ) -> dict:
        """Simulate IS-tilted SDE paths and compute self-normalised IS estimate.

        Args:
            n_paths: number of IS paths to simulate.
            n_steps: number of EM time steps.
            dt: step size.
            target_node: index of the node for which P(Q > B) is estimated.

        Returns:
            dict with keys: p_hat, log_p_hat, ess, ess_pct, raw_weights, outcomes.
        """
        t = self._torch
        B = self.overflow_threshold
        sigma = self.noise_intensity

        c = t.zeros(self.n_nodes, n_paths, device=self._device)
        log_w = t.zeros(n_paths, device=self._device)

        T_total = n_steps * dt

        for step in range(n_steps):
            t_remaining = T_total - step * dt + 1e-6
            q_target = c[target_node]  # (n_paths,)

            # IS drift: pushes target node toward overflow threshold B
            h_star = self.is_strength * (B - q_target) / (sigma ** 2 * t_remaining)
            h_star = t.clamp(h_star, -10.0, 10.0)  # numerical stability

            dW = t.randn(self.n_nodes, n_paths, device=self._device) * (dt ** 0.5)
            dW_target = dW[target_node]  # (n_paths,)

            # Girsanov log-weight increment: -h* dW - 0.5 h*^2 dt
            log_w = log_w - h_star * dW_target - 0.5 * h_star ** 2 * dt

            # Tilted EM step: add IS drift to target node
            tilted_dW = dW.clone()
            tilted_dW[target_node] = tilted_dW[target_node] + h_star * dt / (dt ** 0.5)

            drift = (t.mm(self._influence, c) - self.decay_rate * c
                     + self.noise_intensity * tilted_dW / (dt ** 0.5) * 0.0)
            drift = t.mm(self._influence, c) - self.decay_rate * c
            c = t.clamp(c + drift * dt + self.noise_intensity * tilted_dW, min=0.0)

        # Self-normalised IS estimator
        log_w_stable = log_w - log_w.max()
        w = t.exp(log_w_stable)
        outcomes = (c[target_node] > B).float()

        w_sum = w.sum()
        p_hat = float((w * outcomes).sum() / w_sum.clamp(min=1e-30))
        ess = float(w_sum ** 2 / (w ** 2).sum())
        ess_pct = ess / n_paths * 100.0

        log_p_hat = float(t.log(t.tensor(p_hat + 1e-300)))

        return {
            'p_hat': p_hat,
            'log_p_hat': log_p_hat,
            'ess': ess,
            'ess_pct': ess_pct,
            'n_paths': n_paths,
            'overflow_threshold': B,
        }

    def estimate_rare_event(
        self,
        overflow_thresholds: list,
        n_paths: int = 5000,
        n_steps: int = 100,
        dt: float = 0.01,
        target_node: int = 0,
    ) -> list:
        """Estimate P(Q > B) for a list of overflow thresholds via IS-MLMC.

        For each B, creates a fresh IS instance with that threshold, runs IS paths,
        and compares against direct MC for small B (feasibility check).

        Args:
            overflow_thresholds: list of B values to sweep.
            n_paths: IS paths per threshold.
            n_steps, dt, target_node: passed to simulate_is_paths.

        Returns:
            list of dicts with p_hat, log_p_hat, ess_pct per threshold.
        """
        results = []
        for B in overflow_thresholds:
            self.overflow_threshold = B
            r = self.simulate_is_paths(n_paths, n_steps, dt, target_node)
            results.append(r)
        return results


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
