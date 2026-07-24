"""
Shared baseline utilities, and Quasi-Monte Carlo / variance-reduction
estimators, for the "Coupled Congestion Propagation SDE" (paper
Sec. "Coupled Congestion Propagation SDE", Eq. 6-7):

    dC_i(t) = ( sum_j alpha_ij C_j(t) - beta_i C_i(t) ) dt + sigma_i dW_i(t),
    alpha_ij = alpha * A_ij / deg(i)

This module is the shared home (per the IEEE Access resubmission baseline
task) for code used by BOTH scripts/run_des_baseline.py and
scripts/run_qmc_baseline.py, so that every baseline reports timings under an
identical topology, seed set, target functional, and thread configuration --
required for the apples-to-apples comparison in Reviewer 1's concern 5
("More baselines should be added ... CPU-MLMC, optimized discrete-event
simulation, existing GPU network simulators, or other uncertainty
quantification methods"). scripts/run_cpu_mlmc_baseline.py also imports the
provenance/topology/threading helpers from here for consistency.

Two things live here that are NOT simple imports from src/network/sde.py or
src/simulation/mlmc.py, both for the same underlying reason: the public API
of CongestionPropagationSDE does not support injecting both an *externally
controlled* Brownian increment array (needed for Sobol/antithetic
constructions) and an *exogenous arrival forcing* lambda_i at the same time
(only the protected `_em_step_with_inputs` does, and this file does not
import or edit src/network/sde.py's internals):

  1. `coupled_em_step` -- an independent re-implementation of the paper's
     Eq. 7 predictor-corrector Euler-Maruyama step, verified line-for-line
     against CongestionPropagationSDE._em_step_with_inputs, extended to
     accept an explicit external dW array and an exogenous lambda_vec.
  2. `coupled_mlmc_estimate` -- a lambda-forced coupled-model MLMC estimator
     that reuses the EXISTING, unmodified `MLMCSimulator.compute_optimal_samples`
     (Giles 2008 allocation) from src/simulation/mlmc.py for the allocation
     math, but drives it with `coupled_em_step` paths so it can share
     (lambda, alpha, beta, sigma) with the DES and QMC baselines. This is
     the "same functional the MLMC estimates" reference that
     scripts/run_des_baseline.py measures time-to-epsilon against, because
     AdaptiveNetworkAwareMLMC in mlmc.py hardcodes alpha=0.1/beta=0.5/
     sigma=0.1 with NO lambda forcing inside `_make_congestion_sde` (a
     protected method of a class this task is not permitted to edit), so it
     cannot represent the "moderate load" (lambda_i > 0) scenario the DES
     baseline is required to model.
"""
from __future__ import annotations

import glob
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ============================================================================
# Threading / CPU provenance
# ============================================================================
def set_blas_threads(n: Optional[int]) -> Dict:
    """Pin BLAS/OMP thread counts and report what was actually applied.

    Sets the usual thread-count environment variables (effective for
    OpenBLAS/MKL/Accelerate *if set before those libraries are first
    touched* -- callers should invoke this before `import numpy`/`scipy`
    where possible) and, best-effort, applies threadpoolctl at runtime so a
    thread count is also enforced for pools already initialised.
    """
    if n is None or n <= 0:
        n = os.cpu_count() or 1
    n = int(n)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)

    pools = []
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(n)
        pools = [{"internal_api": d.get("internal_api"),
                  "num_threads": d.get("num_threads"),
                  "prefix": d.get("prefix")}
                 for d in threadpoolctl.threadpool_info()]
    except ImportError:
        pools = []

    return {"requested_threads": n, "os_cpu_count": os.cpu_count(),
            "threadpoolctl_pools": pools}


def prescan_threads_arg(argv: List[str], default: Optional[int] = None) -> Optional[int]:
    """Read --threads from argv WITHOUT importing argparse/numpy, so thread
    env vars can be set before numpy/scipy/torch are first imported."""
    for i, tok in enumerate(argv):
        if tok == "--threads" and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                return default
        if tok.startswith("--threads="):
            try:
                return int(tok.split("=", 1)[1])
            except ValueError:
                return default
    return default


def cpu_model_string() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


# ============================================================================
# Provenance (matches scripts/run_adaptive_stepping_ablation.py's schema,
# extended with CPU/thread fields since these baselines run CPU-only)
# ============================================================================
def git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(
            ["git", "-C", str(REPO_ROOT), "diff", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def build_provenance(config: Dict, thread_info: Dict, device: str = "cpu") -> Dict:
    import numpy
    import scipy
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "not-installed"
    return {
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "cpu_model": cpu_model_string(),
        "os_cpu_count": os.cpu_count(),
        "thread_config": thread_info,
        "torch_version": torch_version,
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "config": config,
    }


def provenance_comment_lines(prov: Dict) -> List[str]:
    tc = prov.get("thread_config", {})
    return [
        f"# git_sha={prov['git_sha']}",
        f"# timestamp_utc={prov['timestamp_utc']}",
        f"# device={prov['device']}",
        f"# cpu_model={prov['cpu_model']}  os_cpu_count={prov['os_cpu_count']}"
        f"  requested_threads={tc.get('requested_threads')}",
        f"# torch={prov['torch_version']} numpy={prov['numpy_version']}"
        f" scipy={prov['scipy_version']} python={prov['python_version']}",
        f"# config={json.dumps(prov['config'], sort_keys=True, default=str)}",
    ]


# ============================================================================
# Topology loading (mirrors scripts/run_ablation_ladder.py's load_topology:
# never silently relabels a synthetic fallback as 'caida')
# ============================================================================
def load_topology(kind: str, n_nodes: int, seed: int,
                   p_er: float = 0.1) -> Tuple[np.ndarray, str, Optional[str]]:
    """Return (adjacency[float32], topology_used, note).

    kind='er' generates Erdos-Renyi(n_nodes, p_er). kind='caida' loads the
    largest local CAIDA AS-REL2 file under datasets/caida/ if present
    (degree-capped to n_nodes), and otherwise falls back to Erdos-Renyi with
    topology_used='er_fallback_no_caida_data' and an explanatory note --
    results are never labelled 'caida' when they are actually synthetic.
    """
    from network.topology import NetworkGraph, TopologyGenerator, load_caida_topology

    generator = TopologyGenerator(seed=seed)
    if kind == "er":
        network = generator.generate_erdos_renyi(n_nodes=n_nodes, p=p_er).get_largest_component()
        return network.get_adjacency_matrix().astype(np.float32), "er", None
    if kind == "ba":
        network = generator.generate_barabasi_albert(n_nodes=n_nodes, m=3).get_largest_component()
        return network.get_adjacency_matrix().astype(np.float32), "ba", None
    if kind == "ws":
        network = generator.generate_watts_strogatz(n_nodes=n_nodes, k=10,
                                                    p=0.1).get_largest_component()
        return network.get_adjacency_matrix().astype(np.float32), "ws", None
    if kind == "rr":
        network = generator.generate_random_regular(n_nodes=n_nodes, d=10).get_largest_component()
        return network.get_adjacency_matrix().astype(np.float32), "rr", None

    if kind != "caida":
        raise ValueError(f"unknown topology kind {kind!r}")

    caida_dir = REPO_ROOT / "datasets" / "caida"
    candidates: List[str] = []
    if caida_dir.is_dir():
        for ext in ("*.as-rel2.txt.bz2", "*.as-rel2.txt.gz", "*.as-rel2.txt"):
            candidates.extend(glob.glob(str(caida_dir / ext)))
    if candidates:
        candidates.sort()
        network = load_caida_topology(candidates[-1], as_undirected=True, largest_component=True)
        if network.n_nodes > n_nodes:
            selected = [nid for nid, _ in
                        sorted(network.graph.degree(), key=lambda kv: kv[1], reverse=True)[:n_nodes]]
            limited = NetworkGraph(directed=network.graph.is_directed())
            limited.graph = network.graph.subgraph(selected).copy()
            network = limited.get_largest_component()
        return network.get_adjacency_matrix().astype(np.float32), "caida", None

    note = ("no local CAIDA AS-REL2 file under datasets/caida/; fell back to "
            "Erdos-Renyi -- these rows are NOT real CAIDA topology data")
    network = generator.generate_erdos_renyi(n_nodes=n_nodes, p=p_er).get_largest_component()
    return network.get_adjacency_matrix().astype(np.float32), "er_fallback_no_caida_data", note


def influence_matrix_from_adjacency(adjacency: np.ndarray, alpha: float) -> np.ndarray:
    """alpha_ij = alpha * A_ij / deg(i) (paper Eq. 6); matches
    CongestionPropagationSDE._compute_influence_from_adjacency exactly."""
    degrees = adjacency.sum(axis=1).astype(float)
    degrees[degrees == 0] = 1.0
    influence = adjacency.astype(float) / degrees[:, None]
    influence *= alpha
    return influence


# ============================================================================
# Checkpointing (append-only JSONL, matches
# scripts/run_adaptive_stepping_ablation.py's schema/fingerprint pattern)
# ============================================================================
def config_fingerprint(cfg: Dict) -> str:
    payload = {k: v for k, v in sorted(cfg.items()) if k != "seeds"}
    return json.dumps(payload, sort_keys=True, default=str)


def load_checkpoint(path: str, fingerprint: str, schema_version: int) -> Dict:
    done: Dict = {}
    if not os.path.exists(path):
        return done
    stale = 0
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print("  [checkpoint] ignoring truncated trailing record", flush=True)
                continue
            if (record.get("schema_version") != schema_version
                    or record.get("config_fingerprint") != fingerprint):
                stale += 1
                continue
            done[record["unit_id"]] = record
    if stale:
        print(f"  [checkpoint] ignoring {stale} record(s) from a different "
              f"schema or configuration", flush=True)
    return done


def append_checkpoint(path: str, record: Dict) -> None:
    with open(path, "a") as handle:
        handle.write(json.dumps(record, default=float) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ============================================================================
# Coupled congestion-SDE step with explicit dW and lambda forcing
# ============================================================================
def coupled_em_step(c: np.ndarray, influence_matrix: np.ndarray, decay_rate,
                     sigma, dt: float, dw: np.ndarray,
                     lambda_vec: Optional[np.ndarray] = None) -> np.ndarray:
    """Predictor-corrector Euler-Maruyama step for the coupled congestion SDE
    (paper Eq. 7). `decay_rate` and `sigma` may be scalars or (n,) arrays.

    Independent re-implementation of
    CongestionPropagationSDE._em_step_with_inputs's formula (verified
    against src/network/sde.py, read but not imported for this call site)
    because the public API does not support both an externally supplied dW
    (needed for QMC/antithetic constructions) and an exogenous lambda_vec at
    the same time.
    """
    diffusion_term = sigma * dw
    drift_n = influence_matrix @ c - decay_rate * c
    if lambda_vec is not None:
        drift_n = drift_n + lambda_vec
    c_pred = np.maximum(0.0, c + drift_n * dt + diffusion_term)

    drift_p = influence_matrix @ c_pred - decay_rate * c_pred
    if lambda_vec is not None:
        drift_p = drift_p + lambda_vec
    return np.maximum(0.0, c + drift_p * dt + diffusion_term)


def simulate_functional(influence_matrix: np.ndarray, decay_rate, sigma,
                         lambda_vec: np.ndarray, T: float, dt: float,
                         dw_path: np.ndarray, warmup_frac: float = 0.2,
                         terminal: bool = True) -> float:
    """Euler-Maruyama-integrate the coupled SDE along an explicit Brownian
    path `dw_path` (shape (n_steps, n_nodes), already sqrt(dt)-scaled) and
    return a scalar functional Y.

    `terminal=True` (default): Y = mean_over_nodes(C(T)), the terminal-state
    functional canonical in MLMC complexity theory (Giles 2008) and already
    a recognised metric in this codebase (src.simulation.mlmc's
    'final_congestion'/'final_queue'). Used as THE target functional shared
    by the DES, QMC/MC/antithetic/CV, and coupled-MLMC-reference baselines,
    because it carries substantially more per-sample variance than a
    time-averaged functional: empirically, time-and-node-averaging over a
    T=20 horizon crushed per-replication variance so far (internal law of
    large numbers over ~320 post-warmup samples x n_nodes) that every
    Monte-Carlo method converged in ~1 sample regardless of variance-
    reduction technique -- correct, but useless for illustrating work-to-
    target-accuracy differences between methods.

    `terminal=False`: legacy time-and-node-averaged functional over the
    post-warmup window (kept for scripts/experiments that want the
    lower-variance convention).
    """
    n_nodes = influence_matrix.shape[0]
    n_steps = dw_path.shape[0]
    warmup_steps = int(warmup_frac * n_steps)
    c = np.zeros(n_nodes, dtype=float)
    acc, count = 0.0, 0
    for k in range(n_steps):
        c = coupled_em_step(c, influence_matrix, decay_rate, sigma, dt,
                             dw_path[k], lambda_vec)
        if not terminal and k >= warmup_steps:
            acc += float(np.mean(c))
            count += 1
    if terminal:
        return float(np.mean(c))
    return acc / max(count, 1)


def simulate_coupled_paths_with_lambda(
        influence_matrix: np.ndarray, decay_rate, sigma,
        lambda_vec: np.ndarray, T: float, dt_coarse: float, dt_fine: float,
        seed: Optional[int] = None, rng: Optional[np.random.Generator] = None,
        warmup_frac: float = 0.2, terminal: bool = True) -> Tuple[float, float]:
    """MLMC-coupled (fine, coarse) sample of the lambda-forced coupled SDE.

    Mirrors CongestionPropagationSDE.simulate_coupled_paths's Brownian-path
    sharing scheme (fine steps use dW ~ N(0,dt_fine); coarse steps sum the M
    matching fine increments so sqrt(dt_coarse)=sqrt(M*dt_fine) in
    distribution) so the level-difference telescoping identity holds, but
    drives coupled_em_step (with lambda_vec) instead of the class's
    protected step. Returns scalar (Y_fine, Y_coarse) functionals; see
    `simulate_functional` for the terminal-vs-time-averaged convention.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    n_nodes = influence_matrix.shape[0]
    M = int(round(dt_coarse / dt_fine))
    if not np.isclose(dt_coarse, M * dt_fine):
        raise ValueError("dt_coarse must be an integer multiple of dt_fine")

    n_steps_fine = int(T / dt_fine)
    n_steps_coarse = int(T / dt_coarse)
    dw_fine = rng.normal(0.0, np.sqrt(dt_fine), (n_steps_fine, n_nodes))

    c_fine = np.zeros(n_nodes)
    c_coarse = np.zeros(n_nodes)
    warmup_fine = int(warmup_frac * n_steps_fine)
    warmup_coarse = int(warmup_frac * n_steps_coarse)
    acc_fine, acc_coarse = 0.0, 0.0
    n_fine_kept, n_coarse_kept = 0, 0

    # Independent per-grid time-averages (matches MLMCSimulator's own
    # _simulate_coupled_levels convention: Y_fine=mean(q_fine),
    # Y_coarse=mean(q_coarse) computed separately over each grid).
    for i_c in range(n_steps_coarse):
        dw_coarse = dw_fine[i_c * M:(i_c + 1) * M].sum(axis=0)
        for j in range(M):
            i_f = i_c * M + j
            c_fine = coupled_em_step(c_fine, influence_matrix, decay_rate,
                                      sigma, dt_fine, dw_fine[i_f], lambda_vec)
            if not terminal and i_f >= warmup_fine:
                acc_fine += float(np.mean(c_fine))
                n_fine_kept += 1
        c_coarse = coupled_em_step(c_coarse, influence_matrix, decay_rate,
                                    sigma, dt_coarse, dw_coarse, lambda_vec)
        if not terminal and i_c >= warmup_coarse:
            acc_coarse += float(np.mean(c_coarse))
            n_coarse_kept += 1

    if terminal:
        return float(np.mean(c_fine)), float(np.mean(c_coarse))
    return acc_fine / max(n_fine_kept, 1), acc_coarse / max(n_coarse_kept, 1)


# ============================================================================
# Lambda-forced coupled-model MLMC reference estimator
# ============================================================================
def coupled_mlmc_estimate(adjacency: np.ndarray, lambda_vec: np.ndarray,
                           alpha: float, beta, sigma, T: float,
                           base_dt: float, L_max: int, epsilon: float,
                           pilot_samples: int, seed: int,
                           refinement_factor: int = 2,
                           confidence_level: float = 0.95) -> Dict:
    """MLMC estimate of E[Y] for the lambda-forced coupled congestion SDE,
    Y = time-and-node-averaged congestion (same functional as
    `simulate_functional`/the DES and QMC baselines).

    Uses `simulate_coupled_paths_with_lambda` for path generation (so
    (lambda, alpha, beta, sigma) can be shared with the DES/QMC baselines,
    unlike AdaptiveNetworkAwareMLMC which hardcodes alpha=0.1/beta=0.5/
    sigma=0.1 with no lambda support), but reuses the EXISTING, unmodified
    `MLMCSimulator.compute_optimal_samples` (Giles 2008 allocation) from
    src/simulation/mlmc.py for the sample-allocation math, rather than
    re-deriving it.
    """
    from simulation.discretization import get_timestep
    from simulation.mlmc import MLMCSimulator

    influence = influence_matrix_from_adjacency(adjacency, alpha)
    n_nodes = adjacency.shape[0]
    beta_arr = np.full(n_nodes, beta, dtype=float) if np.isscalar(beta) else np.asarray(beta, float)
    sigma_arr = np.full(n_nodes, sigma, dtype=float) if np.isscalar(sigma) else np.asarray(sigma, float)

    allocator = MLMCSimulator(refinement_factor=refinement_factor, seed=seed)

    def level_diffs(level: int, n: int, base_seed: int) -> np.ndarray:
        rng = np.random.default_rng(base_seed)
        dt_fine = get_timestep(level, base_dt, refinement_factor)
        diffs = np.empty(n)
        if level == 0:
            for i in range(n):
                y_fine, _ = simulate_coupled_paths_with_lambda(
                    influence, beta_arr, sigma_arr, lambda_vec, T,
                    dt_coarse=dt_fine, dt_fine=dt_fine, rng=rng)
                diffs[i] = y_fine
            return diffs
        dt_coarse = get_timestep(level - 1, base_dt, refinement_factor)
        for i in range(n):
            y_fine, y_coarse = simulate_coupled_paths_with_lambda(
                influence, beta_arr, sigma_arr, lambda_vec, T,
                dt_coarse=dt_coarse, dt_fine=dt_fine, rng=rng)
            diffs[i] = y_fine - y_coarse
        return diffs

    variances, costs, mean_diffs_pilot, pilot_diffs = [], [], [], []
    for l in range(L_max + 1):
        diffs = level_diffs(l, pilot_samples, seed + l * 10_000)
        variances.append(float(np.var(diffs, ddof=1)) if pilot_samples > 1 else 0.0)
        mean_diffs_pilot.append(float(np.mean(diffs)))
        dt_fine = get_timestep(l, base_dt, refinement_factor)
        costs.append(T / dt_fine)
        pilot_diffs.append(diffs)

    optimal_N = allocator.compute_optimal_samples(variances, costs, epsilon)

    level_records = []
    total_cost = 0.0
    for l in range(L_max + 1):
        n_extra = max(0, optimal_N[l] - pilot_samples)
        extra = level_diffs(l, n_extra, seed + l * 10_000 + pilot_samples) if n_extra else np.empty(0)
        diffs = np.concatenate([pilot_diffs[l], extra])
        n_total = len(diffs)
        mean_diff = float(np.mean(diffs))
        var_diff = float(np.var(diffs, ddof=1)) if n_total > 1 else 0.0
        dt_fine = get_timestep(l, base_dt, refinement_factor)
        cost_per_sample = T / dt_fine
        level_cost = cost_per_sample * n_total
        total_cost += level_cost
        level_records.append({
            "level": l, "n_samples": n_total, "dt": dt_fine,
            "mean_diff": mean_diff, "var_diff": var_diff,
            "cost_per_sample": cost_per_sample, "total_cost": level_cost,
        })

    estimate = float(sum(r["mean_diff"] for r in level_records))
    variance = float(sum(r["var_diff"] / r["n_samples"] for r in level_records
                          if r["n_samples"] > 0))
    from scipy import stats as sp_stats
    z = float(sp_stats.norm.ppf(1 - (1 - confidence_level) / 2))
    ci_halfwidth = z * float(np.sqrt(variance))

    return {
        "estimate": estimate, "variance": variance,
        "ci_halfwidth": ci_halfwidth, "total_cost": total_cost,
        "level_stats": level_records, "epsilon": epsilon, "L_max": L_max,
    }


# ============================================================================
# QMC / variance-reduction primitives
# ============================================================================
def sobol_normals(n_points: int, dim: int, seed: int) -> np.ndarray:
    """(n_points, dim) standard-normal draws from an Owen-scrambled Sobol
    sequence (scipy.stats.qmc.Sobol default scrambling is Owen scrambling).

    Uses `random_base2` when n_points is an exact power of two (recommended
    by scipy for Sobol's balance properties); falls back to `.random(n)`
    otherwise (still valid, weaker balance guarantee).
    """
    from scipy.stats import qmc
    from scipy.stats import norm

    engine = qmc.Sobol(d=dim, scramble=True, seed=seed)
    log2n = np.log2(n_points)
    if np.isclose(log2n, round(log2n)):
        u = engine.random_base2(m=int(round(log2n)))
    else:
        u = engine.random(n_points)
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    return norm.ppf(u)


def loglog_slope(xs: List[float], ys: List[float]) -> float:
    """Least-squares slope of log(y) vs log(x); NaN if fewer than 2 usable
    (finite, positive) points."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if good.sum() < 2:
        return float("nan")
    lx, ly = np.log(x[good]), np.log(y[good])
    design = np.vstack([lx, np.ones_like(lx)]).T
    slope, _ = np.linalg.lstsq(design, ly, rcond=None)[0]
    return float(slope)


def fluid_limit_mean(influence_matrix: np.ndarray, decay_rate, lambda_vec: np.ndarray,
                      T: float, dt: float, warmup_frac: float = 0.2,
                      terminal: bool = True) -> float:
    """Exact discrete-time expectation of the coupled process under the SAME
    Euler-Maruyama recursion used by `coupled_em_step`, but with the noise
    term switched off (E[dW]=0, so the predictor-corrector reduces to a
    deterministic linear recursion). Used as the ANALYTIC control-variate
    mean E[X] -- computed once via this recursion (not simulated), avoiding
    Monte Carlo noise in the control's baseline and any Euler discretisation
    mismatch against the actual simulated control-variate paths (which use
    the identical recursion, just with dw != 0). `terminal` must match the
    convention used for the simulated control-variate samples (see
    `simulate_functional`); default True (terminal-state functional).
    """
    n_nodes = influence_matrix.shape[0]
    n_steps = int(T / dt)
    warmup_steps = int(warmup_frac * n_steps)
    c = np.zeros(n_nodes)
    zero_dw = np.zeros(n_nodes)
    acc, count = 0.0, 0
    for k in range(n_steps):
        c = coupled_em_step(c, influence_matrix, decay_rate, 0.0, dt, zero_dw, lambda_vec)
        if not terminal and k >= warmup_steps:
            acc += float(np.mean(c))
            count += 1
    if terminal:
        return float(np.mean(c))
    return acc / max(count, 1)
