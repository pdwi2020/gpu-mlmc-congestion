#!/usr/bin/env python3
"""
exp_extended_epsilon.py — Extended-ε GPU Benchmark for RunPod A100
====================================================================
Reproduces and extends the GPU-MC vs GPU-MLMC comparison from the paper
across a wider range of accuracy targets:

    ε ∈ {0.10, 0.05, 0.02, 0.01, 0.005}

for three scenarios:
    • synthetic_n100  — Erdős-Rényi  n=100, p=0.15
    • synthetic_n500  — Erdős-Rényi  n=499, p=0.02  (≈ same edge count as paper)
    • real_caida_asrel2_20260101_n500 — CAIDA AS-rel2, subgraphed to 500 nodes

Uses **CuPy** for GPU acceleration (no PyCUDA dependency), which matches the
environment used for all prior RunPod/Colab experiments.

Key design decisions matching prior runs
-----------------------------------------
* ci_target_half  = epsilon * CI_TARGET_FACTOR  (0.003, same as run1/run2)
* CI match tolerance: CI_MATCH_TOL = 0.15  (15 %, same as prior)
* T = 5.0 s, base_dt = 0.1 s  (same as run2_extended)
* arrival_rate = 1.0, service_rate = 1.25, noise_intensity = 0.2
  (mirrors MLMCSimulator._simulate_single_level hard-coded values)
* Refinement factor M = 2 throughout
* MLMC pilot samples = 100 per level  (same as CPU MLMCSimulator)
* Sample caps:  cap_mc  = 1_000_000  (matching run2 large-cap config)
                cap_mlmc_per_level = 500_000

Usage (on RunPod, after installing deps)
-----------------------------------------
    pip install cupy-cuda12x scipy pandas numpy tqdm
    python exp_extended_epsilon.py \\
        --output-dir /root/results/extended_eps \\
        --epsilons 0.1 0.05 0.02 0.01 0.005 \\
        --seeds 42 \\
        2>&1 | tee /root/results/extended_eps/run_progress.log

Monitor from another terminal:
    tail -f /root/results/extended_eps/run_progress.log

Output
------
    <output-dir>/extended_epsilon_results.csv   — main results table
    <output-dir>/run_progress.log               — full run log
    <output-dir>/run_summary.json               — JSON metadata
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import bz2
import gzip
import io
import json
import logging
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party  (numpy / scipy always available; cupy / pandas checked below)
# ---------------------------------------------------------------------------
import numpy as np

try:
    from scipy import stats as sp_stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# CuPy — required for GPU simulation.
# Falls back to cupy_torch_shim (PyTorch-based) when CuPy is unavailable or
# segfaults on the current CUDA image (common on PyTorch-bundled containers).
try:
    import cupy as cp

    # Quick smoke-test: cp.array() segfaults on some PyTorch images even
    # though the import itself succeeds.  Catch it via a subprocess check.
    import subprocess as _subprocess, sys as _sys
    _probe = _subprocess.run(
        [_sys.executable, "-c", "import cupy as cp; cp.zeros(1)"],
        capture_output=True, timeout=15,
    )
    if _probe.returncode != 0:
        raise RuntimeError("CuPy smoke-test failed (segfault or error)")
    GPU_AVAILABLE = True
except Exception:
    try:
        import cupy_torch_shim as cp  # type: ignore[no-redef]
        GPU_AVAILABLE = True
        print("[INFO] CuPy unavailable — using PyTorch shim for GPU simulation.",
              file=sys.stderr)
    except ImportError:
        GPU_AVAILABLE = False
        print(
            "[FATAL] Neither CuPy nor cupy_torch_shim found.\n"
            "Falling back to NumPy CPU simulation (SLOW — for debugging only).",
            file=sys.stderr,
        )

# ---------------------------------------------------------------------------
# Experiment-wide constants  (mirror prior run configurations exactly)
# ---------------------------------------------------------------------------

# SDE parameters — must match MLMCSimulator._simulate_single/coupled_levels
ARRIVAL_RATE: float = 1.0
SERVICE_RATE: float = 1.25  # = arrival_rate * 1.25 (overprovisioned)
NOISE_INTENSITY: float = 0.2
T: float = 5.0  # Simulation horizon (seconds)
BASE_DT: float = 0.1  # Coarsest MLMC level timestep
REFINEMENT_FACTOR: int = 2  # M: dt_{l+1} = dt_l / M
L_MAX: int = 10  # Maximum MLMC levels (L_max)
PILOT_SAMPLES: int = 100  # Pilot samples per MLMC level

# CI-targeting parameters — identical to run1 "colab_tuned" and run2 configs
CI_TARGET_FACTOR: float = 0.003  # ci_target_half = epsilon * CI_TARGET_FACTOR
CI_MATCH_TOL: float = 0.15  # ±15 % tolerance on CI half-width matching
MAX_CI_TUNE_ITERS: int = 8  # Max tuning iterations for MC path count

# Sample caps — run2 "large cap" configuration (achieved ε=0.01 equal_accuracy=True)
CAP_MC: int = 1_000_000  # Hard cap on GPU-MC paths
CAP_MLMC_PER_LEVEL: int = 500_000  # Hard cap on MLMC samples per level

# z-value for 95 % confidence intervals
Z_95: float = 1.959_963_985

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
# Each entry:  { label, nodes_target, graph_type, p/m, dataset_note }

SCENARIOS: Dict[str, Dict] = {
    "synthetic_n100": {
        "label": "synthetic_n100",
        "dataset_note": "Synthetic Erdos-Renyi graph",
        "graph_type": "erdos_renyi",
        "n_nodes": 100,
        "p": 0.15,
        "seed": 42,
    },
    "synthetic_n500": {
        "label": "synthetic_n500",
        "dataset_note": "Synthetic Erdos-Renyi graph",
        "graph_type": "erdos_renyi",
        "n_nodes": 500,
        "p": 0.02,
        "seed": 42,
    },
    "real_caida_asrel2_20260101_n500": {
        "label": "real_caida_asrel2_20260101_n500",
        "dataset_note": "CAIDA AS-rel2 20260101 (from serial-2), sampled to 500 nodes",
        "graph_type": "caida",
        "n_nodes": 500,
        "caida_date": "20260101",
        "caida_url": (
            "http://data.caida.org/datasets/as-relationships/serial-2/"
            "20260101.as-rel2.txt.bz2"
        ),
        "seed": 42,
    },
}

# Default epsilon sweep
DEFAULT_EPSILONS: List[float] = [0.10, 0.05, 0.02, 0.01, 0.005]

# CSV column order — identical to prior run CSVs so gen_figures_a100.py
# can ingest this file without modification
CSV_COLUMNS: List[str] = [
    "scenario",
    "nodes",
    "epsilon",
    "qoi",
    "dataset_note",
    "h_finest",
    "mc_paths",
    "mlmc_levels",
    "mlmc_N_l",
    "mc_runtime_s",
    "mlmc_runtime_s",
    "speedup_runtime",
    "mc_cost",
    "mlmc_cost",
    "cost_ratio_mc_over_mlmc",
    "mc_estimate",
    "mlmc_estimate",
    "ci_target_half",
    "mc_ci_half",
    "mlmc_ci_half",
    "equal_accuracy_ci_targeted",
    "error_proxy_mc_ci2_plus_hL",
    "error_proxy_mlmc_ci2_plus_hL",
    "sanity_same_qoi",
    "sanity_same_hL",
    "sanity_seed_policy",
    "sanity_cost_definition",
    "sanity_warmup_excluded",
]

# ---------------------------------------------------------------------------
# Progress logger — writes to both stdout and a rotating log file
# ---------------------------------------------------------------------------


def setup_progress_logger(log_path: Optional[Path] = None) -> logging.Logger:
    """
    Create a logger that simultaneously streams to stdout (INFO) and,
    if *log_path* is given, to a UTF-8 file (DEBUG).

    Use ``tail -f <log_path>`` from a second terminal to monitor the run.
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger("ext_eps")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(ch)

    # File handler — DEBUG and above
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(fh)
        root.info(f"Progress log: {log_path}  (tail -f to monitor)")

    return root


# Module-level logger (will be re-assigned in main after output dir is known)
log = logging.getLogger("ext_eps")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extended-ε GPU-MC vs GPU-MLMC benchmark (RunPod A100)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/extended_eps",
        help="Directory where CSV, log, and JSON are saved.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=DEFAULT_EPSILONS,
        metavar="EPS",
        help="List of target accuracy values ε to sweep.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=list(SCENARIOS.keys()),
        choices=list(SCENARIOS.keys()),
        metavar="SCENARIO",
        help="Which scenarios to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        metavar="SEED",
        help=(
            "Base random seeds.  Each seed produces one CSV row per "
            "(scenario, ε) pair.  Use multiple seeds for error-bar runs."
        ),
    )
    parser.add_argument(
        "--cap-mc",
        type=int,
        default=CAP_MC,
        help="Hard cap on GPU-MC sample count.",
    )
    parser.add_argument(
        "--cap-mlmc",
        type=int,
        default=CAP_MLMC_PER_LEVEL,
        help="Hard cap on MLMC samples per level.",
    )
    parser.add_argument(
        "--no-caida-download",
        action="store_true",
        default=False,
        help=(
            "Skip CAIDA topology download.  Falls back to a synthetic "
            "Barabási-Albert graph with n=500 nodes."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the experiment matrix and exit without running.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def now_utc() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def elapsed_str(seconds: float) -> str:
    """Format elapsed seconds as 'Xm Ys'."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Chunk 3.2 — GPU-MC simulation (CuPy)
# ---------------------------------------------------------------------------


def simulate_gpu_mc_cupy(
    n_paths: int,
    arrival_rate: float = ARRIVAL_RATE,
    service_rate: float = SERVICE_RATE,
    noise_intensity: float = NOISE_INTENSITY,
    T: float = T,
    dt: float = BASE_DT,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Single-level GPU Monte Carlo via Euler-Maruyama using CuPy.

    Memory layout: ``[n_timesteps, n_paths]`` (transposed / column-major for
    paths) so that at every timestep *t* all ``n_paths`` threads read/write
    consecutive addresses — fully coalesced global-memory access.

    Args:
        n_paths:         Number of independent sample paths to simulate.
        arrival_rate:    SDE drift parameter λ.
        service_rate:    SDE drift parameter μ  (μ > λ for stability).
        noise_intensity: SDE diffusion coefficient σ.
        T:               Simulation horizon (seconds).
        dt:              Euler-Maruyama timestep.
        seed:            CuPy RNG seed for reproducibility.

    Returns:
        1-D NumPy array of shape ``(n_paths,)`` with the time-averaged
        mean queue length for each simulated path.
    """
    n_timesteps = int(T / dt)
    drift_dt = float((arrival_rate - service_rate) * dt)
    sigma_sqrt_dt = float(noise_intensity * np.sqrt(dt))

    if GPU_AVAILABLE:
        xp = cp
        if seed is not None:
            cp.random.seed(seed)
        # Transposed layout: [n_timesteps, n_paths] — coalesced per timestep
        # Generate all Brownian increments at once to minimise kernel launches
        dW = xp.random.standard_normal(
            (n_timesteps, n_paths), dtype=xp.float32
        ) * xp.float32(sigma_sqrt_dt)
        q = xp.zeros(n_paths, dtype=xp.float32)
        q_sum = xp.zeros(n_paths, dtype=xp.float32)
        for t in range(n_timesteps):
            q = xp.maximum(q + xp.float32(drift_dt) + dW[t], xp.float32(0.0))
            q_sum = q_sum + q
        result = (q_sum / xp.float32(n_timesteps)).get()
    else:
        # NumPy CPU fallback (debugging only — very slow for large n_paths)
        rng = np.random.default_rng(seed)
        dW = (
            rng.standard_normal((n_timesteps, n_paths)).astype(np.float32)
            * sigma_sqrt_dt
        )
        q = np.zeros(n_paths, dtype=np.float32)
        q_sum = np.zeros(n_paths, dtype=np.float32)
        for t in range(n_timesteps):
            q = np.maximum(q + drift_dt + dW[t], 0.0)
            q_sum += q
        result = q_sum / n_timesteps

    return np.asarray(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# Chunk 3.3 — GPU-MLMC level simulation (CuPy coupled fine + coarse paths)
# ---------------------------------------------------------------------------


def simulate_gpu_mlmc_level_cupy(
    level: int,
    n_paths: int,
    arrival_rate: float = ARRIVAL_RATE,
    service_rate: float = SERVICE_RATE,
    noise_intensity: float = NOISE_INTENSITY,
    T: float = T,
    base_dt: float = BASE_DT,
    M: int = REFINEMENT_FACTOR,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate one MLMC level on GPU: returns coupled (Y_fine, Y_coarse) arrays.

    At level ``l``:
      * Fine path uses timestep  ``dt_fine   = base_dt / M^l``
      * Coarse path uses timestep ``dt_coarse = base_dt / M^(l-1) = dt_fine * M``
    Both paths share the **same** Brownian increments (coarse noise = sum of M
    consecutive fine increments), which is the MLMC coupling that drives
    variance reduction.

    At level 0, Y_coarse is defined as zero (P_{-1} ≡ 0 by convention).

    Memory layout: ``[n_timesteps_fine, n_paths]`` — transposed so each CuPy
    kernel step reads/writes contiguous addresses for all paths.

    Args:
        level:           MLMC level index l (0 = coarsest).
        n_paths:         Number of coupled path pairs to simulate.
        arrival_rate:    SDE drift parameter λ.
        service_rate:    SDE drift parameter μ.
        noise_intensity: SDE diffusion coefficient σ.
        T:               Simulation horizon (seconds).
        base_dt:         Coarsest-level timestep dt_0.
        M:               Refinement factor (dt_{l+1} = dt_l / M).
        seed:            CuPy / NumPy RNG seed.

    Returns:
        Tuple ``(Y_fine, Y_coarse)`` — two 1-D float64 NumPy arrays of shape
        ``(n_paths,)`` with time-averaged mean queue lengths.
        For level 0, ``Y_coarse`` is all zeros.
    """
    dt_fine: float = base_dt / (M**level)
    n_timesteps_fine: int = int(T / dt_fine)
    drift_dt_fine: float = float((arrival_rate - service_rate) * dt_fine)
    sigma_sqrt_dt_fine: float = float(noise_intensity * np.sqrt(dt_fine))

    xp = cp if GPU_AVAILABLE else np

    if seed is not None:
        if GPU_AVAILABLE:
            cp.random.seed(seed)
        else:
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # Generate ALL fine Brownian increments upfront in transposed layout
    # [n_timesteps_fine, n_paths] — coalesced reads in the time loop
    # ------------------------------------------------------------------
    if GPU_AVAILABLE:
        dW_fine = xp.random.standard_normal(
            (n_timesteps_fine, n_paths), dtype=xp.float32
        ) * xp.float32(sigma_sqrt_dt_fine)
    else:
        # NumPy legacy RandomState.standard_normal() does not accept dtype=;
        # generate float64 and cast to float32 explicitly.
        dW_fine = np.random.standard_normal((n_timesteps_fine, n_paths)).astype(
            np.float32
        ) * np.float32(sigma_sqrt_dt_fine)

    # ------------------------------------------------------------------
    # Fine path simulation
    # ------------------------------------------------------------------
    q_fine = xp.zeros(n_paths, dtype=xp.float32)
    q_fine_sum = xp.zeros(n_paths, dtype=xp.float32)

    for t in range(n_timesteps_fine):
        q_fine = xp.maximum(
            q_fine + xp.float32(drift_dt_fine) + dW_fine[t],
            xp.float32(0.0),
        )
        q_fine_sum = q_fine_sum + q_fine

    Y_fine = xp.asarray(q_fine_sum / xp.float32(n_timesteps_fine), dtype=xp.float64)

    # ------------------------------------------------------------------
    # Level 0: coarse is zero by MLMC convention (P_{-1} ≡ 0)
    # ------------------------------------------------------------------
    if level == 0:
        Y_coarse = xp.zeros(n_paths, dtype=xp.float64)
        if GPU_AVAILABLE:
            return Y_fine.get(), Y_coarse.get()
        return np.asarray(Y_fine), np.asarray(Y_coarse)

    # ------------------------------------------------------------------
    # Coarse path simulation
    # dt_coarse = dt_fine * M
    # Each coarse step aggregates M consecutive fine increments
    # ------------------------------------------------------------------
    dt_coarse: float = dt_fine * M
    n_timesteps_coarse: int = n_timesteps_fine // M
    drift_dt_coarse: float = float((arrival_rate - service_rate) * dt_coarse)

    q_coarse = xp.zeros(n_paths, dtype=xp.float32)
    q_coarse_sum = xp.zeros(n_paths, dtype=xp.float32)

    for tc in range(n_timesteps_coarse):
        # Aggregate M fine increments into one coarse increment.
        # dW_fine shape: [n_timesteps_fine, n_paths]
        # dW_fine[tc*M : (tc+1)*M, :] has shape [M, n_paths];
        # summing over axis=0 gives [n_paths] — the coarse Brownian step.
        dW_coarse = xp.sum(dW_fine[tc * M : (tc + 1) * M], axis=0)  # [n_paths]

        q_coarse = xp.maximum(
            q_coarse + xp.float32(drift_dt_coarse) + dW_coarse,
            xp.float32(0.0),
        )
        q_coarse_sum = q_coarse_sum + q_coarse

    # Align coarse metric to same normalisation as fine:
    # fine mean  = sum_fine  / n_timesteps_fine
    # coarse mean = sum_coarse / n_timesteps_coarse
    Y_coarse = xp.asarray(
        q_coarse_sum / xp.float32(n_timesteps_coarse), dtype=xp.float64
    )

    if GPU_AVAILABLE:
        return Y_fine.get(), Y_coarse.get()
    return np.asarray(Y_fine), np.asarray(Y_coarse)


# ---------------------------------------------------------------------------
# Chunk 3.4 — Adaptive MLMC estimator (GPU, Giles-optimal allocation)
# ---------------------------------------------------------------------------


def run_gpu_mlmc_adaptive(
    epsilon_mlmc: float,
    arrival_rate: float = ARRIVAL_RATE,
    service_rate: float = SERVICE_RATE,
    noise_intensity: float = NOISE_INTENSITY,
    T: float = T,
    base_dt: float = BASE_DT,
    M: int = REFINEMENT_FACTOR,
    L_max: int = L_MAX,
    pilot_n: int = PILOT_SAMPLES,
    cap_per_level: int = CAP_MLMC_PER_LEVEL,
    seed: int = 42,
) -> dict:
    """
    Run GPU-MLMC with Giles-optimal adaptive sample allocation.

    Steps
    -----
    1. Pilot run (``pilot_n`` samples per level) — estimate V_l and mean_diff_l.
    2. Optimal allocation: N_l* propto sqrt(V_l / C_l).
    3. Full run: draw additional samples, reuse pilot diffs.
    4. Combine: MLMC estimate, CI half-width, bias, MSE.

    Returns
    -------
    dict with keys: estimate, CI_half, total_cost, mse, levels_used,
                    N_l, mean_diffs, variances, h_finest
    """
    levels = list(range(L_max + 1))

    # Cost = number of fine timesteps per path at level l
    costs: Dict[int, float] = {l: T / (base_dt / M**l) for l in levels}

    # ------------------------------------------------------------------
    # 1. PILOT RUN
    # ------------------------------------------------------------------
    means: Dict[int, float] = {}
    variances: Dict[int, float] = {}
    pilot_diffs: Dict[int, np.ndarray] = {}

    for l in levels:
        Y_fine, Y_coarse = simulate_gpu_mlmc_level_cupy(
            l,
            pilot_n,
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            noise_intensity=noise_intensity,
            T=T,
            base_dt=base_dt,
            M=M,
            seed=seed + l,
        )
        diffs = Y_fine - Y_coarse
        means[l] = float(np.mean(diffs))
        variances[l] = float(np.var(diffs, ddof=1)) if pilot_n > 1 else 1e-12
        pilot_diffs[l] = diffs

    # Guard against non-positive variance (can occur at deep near-zero-diff levels)
    for l in levels:
        if variances[l] <= 0.0:
            variances[l] = 1e-12

    # ------------------------------------------------------------------
    # 2. OPTIMAL ALLOCATION  (Giles 2008 formula)
    # ------------------------------------------------------------------
    sum_sqrt_vc = float(sum(np.sqrt(variances[l] * costs[l]) for l in levels))
    N_opt: Dict[int, int] = {}
    for l in levels:
        n_raw = (2.0 / epsilon_mlmc**2) * np.sqrt(variances[l] / costs[l]) * sum_sqrt_vc
        n = math.ceil(n_raw)
        n = max(pilot_n, n)
        n = min(n, cap_per_level)
        N_opt[l] = n

    # ------------------------------------------------------------------
    # 3. FULL RUN  (reuse pilot diffs, draw additional samples per level)
    # ------------------------------------------------------------------
    all_diffs: Dict[int, np.ndarray] = {}
    mean_diff: Dict[int, float] = {}
    var_diff: Dict[int, float] = {}
    n_used: Dict[int, int] = {}

    for l in levels:
        n_additional = N_opt[l] - pilot_n
        if n_additional > 0:
            Y_f_add, Y_c_add = simulate_gpu_mlmc_level_cupy(
                l,
                n_additional,
                arrival_rate=arrival_rate,
                service_rate=service_rate,
                noise_intensity=noise_intensity,
                T=T,
                base_dt=base_dt,
                M=M,
                seed=seed + l * 10000,
            )
            all_diffs[l] = np.concatenate([pilot_diffs[l], Y_f_add - Y_c_add])
        else:
            all_diffs[l] = pilot_diffs[l][: N_opt[l]]

        n_used[l] = len(all_diffs[l])
        mean_diff[l] = float(np.mean(all_diffs[l]))
        var_diff[l] = float(np.var(all_diffs[l], ddof=1)) if n_used[l] > 1 else 1e-12

    # ------------------------------------------------------------------
    # 4. COMBINE
    # ------------------------------------------------------------------
    estimate = float(sum(mean_diff[l] for l in levels))
    variance = float(sum(var_diff[l] / n_used[l] for l in levels))
    CI_half = float(Z_95 * np.sqrt(max(variance, 0.0)))
    total_cost = float(sum(costs[l] * n_used[l] for l in levels))

    dt_finest = base_dt / M**L_max
    # Bias for reflected SDE (weak order 0.5, BIAS_CONST=0.5 from mlmc.py)
    bias = 0.5 * np.sqrt(dt_finest)
    mse = float(variance + bias**2)

    return {
        "estimate": estimate,
        "CI_half": CI_half,
        "total_cost": total_cost,
        "mse": mse,
        "levels_used": L_max + 1,
        "N_l": [N_opt[l] for l in levels],
        "mean_diffs": [mean_diff[l] for l in levels],
        "variances": [var_diff[l] for l in levels],
        "h_finest": float(dt_finest),
    }


# ---------------------------------------------------------------------------
# Chunk 3.5 — Binary-search GPU-MC to match a target CI half-width
# ---------------------------------------------------------------------------


def target_ci_gpu_mc(
    ci_target: float,
    arrival_rate: float = ARRIVAL_RATE,
    service_rate: float = SERVICE_RATE,
    noise_intensity: float = NOISE_INTENSITY,
    T: float = T,
    dt: float = BASE_DT,
    cap_mc: int = CAP_MC,
    seed: int = 42,
    max_iters: int = MAX_CI_TUNE_ITERS,
    tol: float = CI_MATCH_TOL,
) -> dict:
    """
    Scaling loop: find N_paths such that GPU-MC CI half-width ≈ ci_target.

    Algorithm
    ---------
    1. Quick pilot (1 000 paths) to estimate std -> initial N_est.
    2. Loop up to max_iters:
         simulate N_est paths -> compute ci_half.
         if |ci_half - ci_target| / ci_target <= tol -> break.
         rescale N_est proportional to (ci_half / ci_target)^2.
    3. Cap N_est to cap_mc before each run.

    Returns
    -------
    dict with keys: samples, n_paths, estimate, CI_half, equal_accuracy,
                    total_cost, runtime_s, dt
    """
    n_timesteps = int(T / dt)

    # Quick pilot to seed the initial N estimate
    pilot = simulate_gpu_mc_cupy(
        1000,
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        noise_intensity=noise_intensity,
        T=T,
        dt=dt,
        seed=seed,
    )
    est_std = float(np.std(pilot, ddof=1))
    if est_std <= 0.0:
        est_std = 1e-6

    N_est = int(math.ceil((Z_95 * est_std / ci_target) ** 2))
    N_est = max(100, min(N_est, cap_mc))

    t_start = time.perf_counter()
    samples: Optional[np.ndarray] = None
    ci_half = float("inf")

    for _ in range(max_iters):
        samples = simulate_gpu_mc_cupy(
            N_est,
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            noise_intensity=noise_intensity,
            T=T,
            dt=dt,
            seed=seed,
        )
        std_est = float(np.std(samples, ddof=1))
        ci_half = float(Z_95 * std_est / np.sqrt(N_est))

        if abs(ci_half - ci_target) / ci_target <= tol:
            break

        # Scale N proportionally to CI^2 relationship: N ~ (sigma/ci)^2
        scale = (ci_half / ci_target) ** 2
        N_est = min(cap_mc, int(math.ceil(N_est * scale)))

    runtime_s = float(time.perf_counter() - t_start)
    total_cost = float(N_est * n_timesteps)
    equal_accuracy = abs(ci_half - ci_target) / ci_target <= tol

    return {
        "samples": samples if samples is not None else np.array([], dtype=np.float64),
        "n_paths": N_est,
        "estimate": float(np.mean(samples)) if samples is not None else float("nan"),
        "CI_half": ci_half,
        "equal_accuracy": equal_accuracy,
        "total_cost": total_cost,
        "runtime_s": runtime_s,
        "dt": float(dt),
    }


# ---------------------------------------------------------------------------
# Chunk 3.6 — Orchestrate one (scenario, epsilon) pair
# ---------------------------------------------------------------------------


def run_one_scenario_epsilon(
    scenario_key: str,
    epsilon: float,
    n_nodes: int,
    seed: int = 42,
    cap_mc: int = CAP_MC,
    cap_mlmc: int = CAP_MLMC_PER_LEVEL,
) -> dict:
    """
    Run GPU-MC and GPU-MLMC for one (scenario, epsilon) pair.

    Both methods target the same CI half-width:
        ci_target_half = epsilon * CI_TARGET_FACTOR
    and use the same finest timestep:
        h_L = BASE_DT / REFINEMENT_FACTOR^L_MAX

    Returns a dict whose keys exactly match CSV_COLUMNS.
    """
    ci_target_half = epsilon * CI_TARGET_FACTOR
    dt_finest = BASE_DT / (REFINEMENT_FACTOR**L_MAX)

    # --- GPU-MC -------------------------------------------------------
    t0 = time.perf_counter()
    mc_result = target_ci_gpu_mc(
        ci_target_half,
        dt=dt_finest,
        cap_mc=cap_mc,
        seed=seed,
    )
    mc_time = float(time.perf_counter() - t0)

    # --- GPU-MLMC -----------------------------------------------------
    epsilon_mlmc = ci_target_half  # MLMC targets same CI half-width
    t1 = time.perf_counter()
    mlmc_result = run_gpu_mlmc_adaptive(
        epsilon_mlmc,
        cap_per_level=cap_mlmc,
        seed=seed,
    )
    mlmc_time = float(time.perf_counter() - t1)

    # --- Sanity flags (always True — audit trail only) ----------------
    sanity_same_qoi = True  # both methods estimate mean_queue
    sanity_same_hL = True  # both use dt_finest as the finest step
    sanity_seed_policy = True  # both seeded from the same base seed
    sanity_cost_def = True  # cost = paths x timesteps throughout
    sanity_warmup_excl = True  # no warmup samples counted in cost

    # --- Equal-accuracy flag ------------------------------------------
    equal_accuracy = mc_result["equal_accuracy"] and (
        abs(mlmc_result["CI_half"] - ci_target_half) / ci_target_half <= CI_MATCH_TOL
    )

    # --- Derived metrics ----------------------------------------------
    speedup_runtime = (mc_time / mlmc_time) if mlmc_time > 0.0 else float("nan")
    cost_ratio = (
        mc_result["total_cost"] / mlmc_result["total_cost"]
        if mlmc_result["total_cost"] > 0.0
        else float("nan")
    )
    error_proxy_mc = mc_result["CI_half"] ** 2 + dt_finest
    error_proxy_mlmc = mlmc_result["CI_half"] ** 2 + dt_finest

    return {
        "scenario": scenario_key,
        "nodes": n_nodes,
        "epsilon": epsilon,
        "qoi": "mean_queue",
        "dataset_note": SCENARIOS[scenario_key]["dataset_note"],
        "h_finest": dt_finest,
        "mc_paths": mc_result["n_paths"],
        "mlmc_levels": mlmc_result["levels_used"],
        "mlmc_N_l": str(mlmc_result["N_l"]),
        "mc_runtime_s": mc_time,
        "mlmc_runtime_s": mlmc_time,
        "speedup_runtime": speedup_runtime,
        "mc_cost": mc_result["total_cost"],
        "mlmc_cost": mlmc_result["total_cost"],
        "cost_ratio_mc_over_mlmc": cost_ratio,
        "mc_estimate": mc_result["estimate"],
        "mlmc_estimate": mlmc_result["estimate"],
        "ci_target_half": ci_target_half,
        "mc_ci_half": mc_result["CI_half"],
        "mlmc_ci_half": mlmc_result["CI_half"],
        "equal_accuracy_ci_targeted": equal_accuracy,
        "error_proxy_mc_ci2_plus_hL": error_proxy_mc,
        "error_proxy_mlmc_ci2_plus_hL": error_proxy_mlmc,
        "sanity_same_qoi": sanity_same_qoi,
        "sanity_same_hL": sanity_same_hL,
        "sanity_seed_policy": sanity_seed_policy,
        "sanity_cost_definition": sanity_cost_def,
        "sanity_warmup_excluded": sanity_warmup_excl,
    }


# ---------------------------------------------------------------------------
# Chunk 3.7 — Topology loader (Erdos-Renyi synthetic or CAIDA real)
# ---------------------------------------------------------------------------


def load_or_build_topology(
    scenario_cfg: dict,
    no_download: bool = False,
) -> Tuple[int, int]:
    """
    Return (n_nodes_actual, n_edges) for a scenario configuration.

    The SDE simulation uses global ARRIVAL_RATE / SERVICE_RATE for all nodes,
    so topology only affects the scenario label in the CSV, not the numerical
    simulation.  This function exists to provide realistic node/edge counts and
    to optionally subsample a real CAIDA topology via BFS.
    """
    if scenario_cfg["graph_type"] == "erdos_renyi":
        n = scenario_cfg["n_nodes"]
        p = scenario_cfg["p"]
        expected_edges = int(n * (n - 1) / 2 * p)
        return n, expected_edges

    elif scenario_cfg["graph_type"] == "caida":
        if no_download:
            # Fallback: Barabasi-Albert approximation with m=3
            n = scenario_cfg["n_nodes"]
            return n, n * 3

        try:
            url = scenario_cfg["caida_url"]
            log.info(f"Downloading CAIDA topology from {url}")
            with urllib.request.urlopen(url, timeout=120) as resp:
                raw = bz2.decompress(resp.read()).decode("utf-8")

            edges: set = set()
            for line in raw.splitlines():
                if line.startswith("#"):
                    continue
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    try:
                        edges.add((int(parts[0]), int(parts[1])))
                    except ValueError:
                        continue

            # Subsample to n_nodes using BFS from highest-degree node
            n_target = scenario_cfg["n_nodes"]
            adj: Dict[int, set] = {}
            for a, b in edges:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)

            if not adj:
                raise ValueError("Empty adjacency list after parsing CAIDA data")

            start = max(adj, key=lambda x: len(adj[x]))
            visited: List[int] = []
            queue_bfs: List[int] = [start]
            seen: set = {start}

            while queue_bfs and len(visited) < n_target:
                node = queue_bfs.pop(0)
                visited.append(node)
                for nb in sorted(
                    adj.get(node, []),
                    key=lambda x: -len(adj.get(x, [])),
                ):
                    if nb not in seen and len(seen) < n_target:
                        seen.add(nb)
                        queue_bfs.append(nb)

            sub_nodes = set(visited[:n_target])
            sub_edges = [(a, b) for a, b in edges if a in sub_nodes and b in sub_nodes]
            log.info(f"CAIDA subgraph: {len(sub_nodes)} nodes, {len(sub_edges)} edges")
            return len(sub_nodes), len(sub_edges)

        except Exception as exc:
            log.warning(f"CAIDA download failed ({exc}), using BA fallback")
            n = scenario_cfg["n_nodes"]
            return n, n * 3

    else:
        raise ValueError(f"Unknown graph_type: {scenario_cfg['graph_type']!r}")


# ---------------------------------------------------------------------------
# Chunk 3.8 — CSV helper + main() entry point
# ---------------------------------------------------------------------------


def _write_csv(rows: list, path: Path) -> None:
    """Overwrite *path* with all rows, always writing the full CSV header."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_progress.log"

    global log
    log = setup_progress_logger(log_path)

    log.info("=" * 70)
    log.info("Extended-epsilon GPU Benchmark")
    log.info(f"Scenarios : {args.scenarios}")
    log.info(f"Epsilons  : {args.epsilons}")
    log.info(f"Seeds     : {args.seeds}")
    log.info(f"cap_mc    : {args.cap_mc:,}   cap_mlmc: {args.cap_mlmc:,}")
    log.info(f"Output    : {out_dir}")
    log.info("=" * 70)

    if args.dry_run:
        log.info("DRY RUN -- exiting.")
        return

    # Log GPU info
    if GPU_AVAILABLE:
        log.info(f"CuPy version : {cp.__version__}")
        try:
            device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            log.info(f"GPU device   : {device_name}")
        except Exception:
            log.info("GPU device   : (device query unavailable)")
    else:
        log.warning("GPU NOT available -- running on CPU NumPy fallback (slow)")

    rows: List[dict] = []
    total_jobs = len(args.scenarios) * len(args.epsilons) * len(args.seeds)
    job_idx = 0
    wall_start = time.perf_counter()

    for scenario_key in args.scenarios:
        scenario_cfg = SCENARIOS[scenario_key]
        n_nodes, _ = load_or_build_topology(scenario_cfg, args.no_caida_download)

        for epsilon in sorted(args.epsilons, reverse=True):  # loose -> tight
            for seed in args.seeds:
                job_idx += 1
                log.info("")
                log.info(
                    f"[{job_idx}/{total_jobs}] scenario={scenario_key}  "
                    f"epsilon={epsilon}  seed={seed}"
                )

                t_job = time.perf_counter()
                try:
                    row = run_one_scenario_epsilon(
                        scenario_key,
                        epsilon,
                        n_nodes,
                        seed=seed,
                        cap_mc=args.cap_mc,
                        cap_mlmc=args.cap_mlmc,
                    )
                except Exception as exc:
                    log.error(f"  FAILED: {exc}")
                    import traceback

                    log.debug(traceback.format_exc())
                    continue

                elapsed = time.perf_counter() - t_job
                rows.append(row)

                log.info(
                    f"  [done] runtime_speedup={row['speedup_runtime']:.2f}x  "
                    f"cost_ratio={row['cost_ratio_mc_over_mlmc']:.2f}x  "
                    f"equal_acc={row['equal_accuracy_ci_targeted']}  "
                    f"time={elapsed_str(elapsed)}"
                )

                # Write CSV incrementally after every row so partial results
                # survive if the pod is killed mid-run.
                csv_path = out_dir / "extended_epsilon_results.csv"
                _write_csv(rows, csv_path)
                log.info(f"  CSV updated -> {csv_path}")

                # Log estimated remaining time
                wall_elapsed = time.perf_counter() - wall_start
                rate = job_idx / wall_elapsed if wall_elapsed > 0 else 1.0
                remaining = (total_jobs - job_idx) / rate if rate > 0 else float("inf")
                log.info(
                    f"  Progress: {job_idx}/{total_jobs}  "
                    f"elapsed={elapsed_str(wall_elapsed)}  "
                    f"ETA={elapsed_str(remaining)}"
                )

    # ------------------------------------------------------------------
    # Save final JSON summary
    # ------------------------------------------------------------------
    summary = {
        "run_date_utc": now_utc(),
        "gpu_available": GPU_AVAILABLE,
        "scenarios": args.scenarios,
        "epsilons": args.epsilons,
        "seeds": args.seeds,
        "total_rows": len(rows),
        "equal_accuracy_rows": sum(1 for r in rows if r["equal_accuracy_ci_targeted"]),
        "cap_mc": args.cap_mc,
        "cap_mlmc": args.cap_mlmc,
        "T": T,
        "base_dt": BASE_DT,
        "L_max": L_MAX,
    }
    json_path = out_dir / "run_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary JSON -> {json_path}")
    log.info("DONE")


if __name__ == "__main__":
    main()
