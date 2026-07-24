"""
Ablation and accuracy-work study for SIMT two-bucket adaptive pathwise stepping
(Section "Adaptive Time Stepping and Multi-GPU Halo Exchange", contribution 3).

Background.  `GPUCoupledPropagationMLMC._em_step_adaptive` was dead code until
it was wired into the MLMC path loop: it sat behind a constructor flag that no
code path consulted, so every previously published number was produced with the
feature inert.  This script measures the manuscript's claims from scratch.

The decisive question is NOT whether adaptive stepping is cheaper than fixed
stepping at a matched nominal step size -- that only measures controller
overhead, and an adaptive scheme is not supposed to win there.  It is whether
adaptive stepping reaches a target ACCURACY for less work and less wall clock
than a uniform grid.  `--experiment frontier` answers that; it is the gate.

Structural ceiling, worth stating before reading any number: with exactly two
buckets, a step costs (1 + f) path-updates per path where f is the half-step
occupancy, against 2 for a uniformly halved grid.  So the best achievable
speed-up over "just halve the step everywhere" is 2/(1+f) <= 2x, and it is
only realised if refining a fraction f of paths buys the accuracy of refining
all of them.

Experiments (--experiment):
    grid      ablation grid x regimes: alpha, estimate, wall, work, per-step
              bucket occupancy, local-error dispersion, matmul vs path count
    frontier  error-vs-work and error-vs-wall-clock at matched accuracy,
              against a fine-grid reference solution.  THE GATE.
    size      occupancy and error dispersion vs network size, testing whether
              a larger n helps or hurts bucket heterogeneity
    all       all three (default)

Axes swept in `grid`:
    adaptive_stepping        {False, True}
    adaptive_rtol            {1e-3, 1e-4, 1e-5}          (adaptive only)
    adaptive_error_estimator {embedded, half_step}       (adaptive only)
    reflection               {predictor_corrector, euler_clamp}
    regime                   see REGIMES below

Usage:
    python scripts/run_adaptive_stepping_ablation.py --quick
    python scripts/run_adaptive_stepping_ablation.py --seeds 0 1 2 --device cuda
    python scripts/run_adaptive_stepping_ablation.py --experiment frontier

Output (under --out, default results/adaptive_stepping):
    adaptive_stepping_ablation.json   full results + provenance
    summary.csv                       one row per (regime, configuration, seed)
    bucket_occupancy.csv              long format, one row per nominal step
    matmul_scaling.csv                torch.mm count vs path count
    accuracy_work.csv                 the gate: error vs work vs wall clock
    gate_verdict.csv                  per regime, does adaptive beat fixed
    size_sweep.csv                    occupancy and dispersion vs n
    checkpoint.jsonl                  append-only resume log

Checkpointed per work unit and resumable, so a reclaimed rented GPU costs at
most one unit.
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch  # noqa: E402
from gpu.parallel_mc import GPUCoupledPropagationMLMC  # noqa: E402

#: Bumped whenever the shape of a checkpoint record changes.  A resume that
#: finds a different version discards the log rather than mixing schemas.
SCHEMA_VERSION = 4

RTOL_DEFAULTS = [1e-3, 1e-4, 1e-5]
ESTIMATORS = ["embedded", "half_step"]
REFLECTIONS = ["predictor_corrector", "euler_clamp"]
PATH_COUNTS = [32, 128, 512, 2048, 8192]
PATH_COUNTS_QUICK = [32, 128, 512]
SIZE_SWEEP = [8, 32, 128, 512]
SIZE_SWEEP_QUICK = [8, 32]

# Tolerances for the frontier sweep: wide enough to bracket "never refines"
# through "always refines" on every regime.  The interesting tolerance is the
# one that produces sustained mixed occupancy, and it is regime-dependent, so
# the grid stays dense rather than assuming where that lands.
FRONTIER_RTOLS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6]
FRONTIER_RTOLS_QUICK = [1e-2, 1e-3, 1e-5]

# A speed-up has to clear this margin to count as a win rather than as noise.
WIN_MARGIN = 1.05

# ---------------------------------------------------------------------------
# Problem regimes.
#
# The manuscript motivates the scheme with heterogeneity: "some sample paths
# approach near-saturation (Q_i -> cap) and require finer temporal resolution
# ... while other paths evolve smoothly".  Two caveats that shape these
# regimes, both worth stating plainly:
#
#  1. `GPUCoupledPropagationMLMC` has NO upper capacity barrier.  The only
#     boundary is the reflecting barrier at zero.  "Near-saturation" as the
#     manuscript describes it cannot be constructed in this code without
#     adding a ceiling, so `rho` below sets the operating point (lambda/decay
#     relative to a nominal unit capacity) and nothing saturates.
#  2. The refine decision is per PATH, but the local-error indicator is a max
#     over NODES.  Making nodes heterogeneous therefore does not by itself make
#     paths heterogeneous -- every path sees a max over the same node set, and
#     maxima concentrate as n grows.  Path-level spread has to come from the
#     realised state, which is why the regimes below vary the noise level and
#     how hard the ensemble sits against the reflecting barrier.
# ---------------------------------------------------------------------------
REGIMES = {
    "homogeneous_chain": {
        "topology": "chain", "n_nodes": 12, "lam_profile": "uniform",
        "lam": 0.4, "noise_intensity": 0.1,
        "note": "original benchmark; known degenerate occupancy",
    },
    "rho090": {
        "topology": "er", "n_nodes": 64, "lam_profile": "uniform",
        "lam": 0.45, "noise_intensity": 0.2, "rho_nominal": 0.90,
        "note": "uniform high operating point",
    },
    "rho095": {
        "topology": "er", "n_nodes": 64, "lam_profile": "uniform",
        "lam": 0.475, "noise_intensity": 0.2, "rho_nominal": 0.95,
    },
    "rho099": {
        "topology": "er", "n_nodes": 64, "lam_profile": "uniform",
        "lam": 0.495, "noise_intensity": 0.2, "rho_nominal": 0.99,
    },
    "heterogeneous_rho": {
        "topology": "er", "n_nodes": 64, "lam_profile": "log_spaced",
        "lam": (0.005, 0.495), "noise_intensity": 0.2,
        "note": "per-node rho spanning 0.01 to 0.99 within one network",
    },
    "bimodal_load": {
        "topology": "er", "n_nodes": 64, "lam_profile": "bimodal",
        "lam": (0.0, 0.495), "noise_intensity": 0.4,
        "note": "half the nodes idle, half near the top of the range",
    },
    "reflection_active": {
        "topology": "er", "n_nodes": 64, "lam_profile": "uniform",
        "lam": 0.02, "noise_intensity": 0.5,
        "note": "ensemble pinned against the reflecting barrier at zero",
    },
    "heavy_noise": {
        "topology": "er", "n_nodes": 64, "lam_profile": "uniform",
        "lam": 0.2, "noise_intensity": 1.0,
        "note": "broad state distribution across paths",
    },
}
QUICK_REGIMES = ["homogeneous_chain", "heterogeneous_rho", "reflection_active"]
FRONTIER_REGIMES = ["homogeneous_chain", "heterogeneous_rho", "bimodal_load",
                    "reflection_active", "rho099"]


# ----------------------------------------------------------------- device ---
def select_device(requested: str) -> torch.device:
    """Resolve --device, preferring cuda > mps > cpu when set to 'auto'."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple MPS ({platform.processor() or platform.machine()})"
    return platform.processor() or platform.machine() or "cpu"


def synchronize(device: torch.device) -> None:
    """Block until queued device work has finished, so timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ------------------------------------------------------------- provenance ---
def git_sha() -> str:
    repo = os.path.join(os.path.dirname(__file__), '..')
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(
            ["git", "-C", repo, "diff", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def build_provenance(device: torch.device, config: dict) -> dict:
    return {
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_name": device_name(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "config": config,
    }


def provenance_comment_lines(prov: dict) -> list:
    """Provenance header for CSV files; read back with pandas comment='#'."""
    return [
        f"# git_sha={prov['git_sha']}",
        f"# timestamp_utc={prov['timestamp_utc']}",
        f"# device={prov['device']} ({prov['device_name']})",
        f"# torch={prov['torch_version']} numpy={prov['numpy_version']}"
        f" python={prov['python_version']}",
        f"# config={json.dumps(prov['config'], sort_keys=True, default=str)}",
    ]


# ------------------------------------------------------------- simulation ---
def chain_adj(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


def er_adj(n: int, p: float = 0.08, seed: int = 42) -> np.ndarray:
    """Erdos-Renyi graph, connected up by a spanning chain so no node is isolated."""
    rng = np.random.default_rng(seed)
    adj = (rng.random((n, n)) < p).astype(np.float32)
    adj = ((adj + adj.T) > 0).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


def regime_adjacency(regime: dict, seed: int = 42) -> np.ndarray:
    n = regime["n_nodes"]
    if regime["topology"] == "chain":
        return chain_adj(n)
    return er_adj(n, seed=seed)


def regime_lambda(regime: dict) -> np.ndarray:
    """Per-node arrival rates, shape (1, n_nodes), resampled onto each grid."""
    n = regime["n_nodes"]
    profile, lam = regime["lam_profile"], regime["lam"]
    if profile == "uniform":
        values = np.full(n, float(lam), dtype=np.float32)
    elif profile == "log_spaced":
        lo, hi = lam
        values = np.logspace(np.log10(lo), np.log10(hi), n).astype(np.float32)
    elif profile == "bimodal":
        lo, hi = lam
        values = np.where(np.arange(n) % 2 == 0, float(lo), float(hi)
                          ).astype(np.float32)
    else:
        raise ValueError(f"unknown lam_profile {profile!r}")
    return values.reshape(1, n)


def make_sim(params: dict, regime: dict, cfg: dict, seed: int,
             device: torch.device,
             diagnostics: bool = False) -> GPUCoupledPropagationMLMC:
    """Build a simulator pinned to `device`.

    The constructor hardcodes 'cuda if available else cpu', so the device is
    overridden here rather than in the library; every internal allocation reads
    `self._device`, and the influence matrix is the only pre-existing tensor.
    """
    kwargs = dict(
        influence_strength=cfg["influence_strength"],
        decay_rate=cfg["decay_rate"],
        noise_intensity=regime["noise_intensity"],
        seed=seed,
        adaptive_stepping=params["adaptive_stepping"],
        reflection=params["reflection"],
        adaptive_diagnostics=diagnostics,
    )
    if params["adaptive_stepping"]:
        kwargs["adaptive_rtol"] = params["adaptive_rtol"]
        kwargs["adaptive_error_estimator"] = params["adaptive_error_estimator"]

    sim = GPUCoupledPropagationMLMC(regime_adjacency(regime), **kwargs)
    if sim._device != device:
        sim._device = device
        sim._influence = sim._influence.to(device)
    return sim


def fixed_work_units(cfg: dict, level: int) -> float:
    """Timestep-path evaluations a fixed-step run of `level` would cost."""
    dt_fine = cfg["base_dt"] / (2 ** level)
    n_fine = int(cfg["T"] / dt_fine)
    n_coarse = 0 if level == 0 else int(cfg["T"] / (dt_fine * 2))
    return float((n_fine + n_coarse) * cfg["n_samples"])


def decay_exponent(variances: list, skip: int = 1) -> float:
    """Slope of log2(V_l) vs l over levels >= skip, returned as positive alpha."""
    usable = [(l, v) for l, v in enumerate(variances) if l >= skip and v > 0]
    if len(usable) < 2:
        return float("nan")
    levels = np.array([l for l, _ in usable], dtype=float)
    logv = np.log2(np.array([v for _, v in usable], dtype=float))
    design = np.vstack([levels, np.ones_like(levels)]).T
    slope = np.linalg.lstsq(design, logv, rcond=None)[0][0]
    return -float(slope)


def loglog_fit(x: list, y: list):
    """Least-squares fit of log(y) = a*log(x) + b; returns a callable and slope."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    good = np.isfinite(lx) & np.isfinite(ly)
    if good.sum() < 2:
        return None, float("nan")
    design = np.vstack([lx[good], np.ones(good.sum())]).T
    slope, intercept = np.linalg.lstsq(design, ly[good], rcond=None)[0]
    return (lambda value: float(np.exp(slope * np.log(value) + intercept)),
            float(slope))


# ---------------------------------------------------------------- measure ---
def measure_matmul_scaling(params: dict, regime: dict, cfg: dict,
                           device: torch.device, path_counts: list) -> list:
    """torch.mm calls for one nominal step, as the path count grows.

    The count must stay flat.  If it tracks `n_paths`, the implementation is
    doing per-path work and the SIMT divergence claim is unsupported.  Buckets
    are forced to a 50/50 split so both are genuinely occupied.
    """
    real_mm = torch.mm
    rows = []
    n_nodes = regime["n_nodes"]

    for n_paths in path_counts:
        sim = make_sim(params, regime, cfg, seed=0, device=device)
        c = torch.rand(n_nodes, n_paths, device=device)
        dw = torch.randn(n_nodes, n_paths, device=device) * 0.1
        calls = []

        def counting_mm(a, b, *args, **kwargs):
            calls.append(tuple(b.shape))
            return real_mm(a, b, *args, **kwargs)

        if params["adaptive_stepping"]:
            mixed = torch.where(
                torch.arange(n_paths, device=device) % 2 == 0,
                torch.ones(n_paths, device=device),
                torch.full((n_paths,), 0.5, device=device))
            sim._adaptive_h_scale["fine"] = mixed
            torch.mm = counting_mm
            try:
                sim._em_step_adaptive(c, cfg["base_dt"], dw, role="fine")
            finally:
                torch.mm = real_mm
        else:
            torch.mm = counting_mm
            try:
                sim._em_step(c, cfg["base_dt"], dw)
            finally:
                torch.mm = real_mm

        rows.append({"n_paths": n_paths, "mm_calls": len(calls),
                     "mm_operand_widths": sorted(s[1] for s in calls)})
    return rows


def occupancy_summary(trace: list) -> dict:
    """Is mixed occupancy reachable, and does it persist?

    `mixed_step_fraction` counts steps where both buckets are non-empty.
    `sustained_mixed_fraction` demands a genuine split (5%-95%) in the second
    half of the run, which is the only regime where the two-bucket design can
    do anything a uniform step change could not.
    """
    if not trace:
        return {"mixed_step_fraction": 0.0, "sustained_mixed_fraction": 0.0,
                "frac_half_mean": 0.0, "err_spread_median": float("nan")}
    frac = np.array([r["frac_half"] for r in trace], dtype=float)
    mixed = (frac > 0.0) & (frac < 1.0)
    tail = slice(len(frac) // 2, None)
    sustained = (frac[tail] > 0.05) & (frac[tail] < 0.95)
    spreads = [r["err_spread"] for r in trace
               if "err_spread" in r and np.isfinite(r["err_spread"])]
    return {
        "mixed_step_fraction": float(mixed.mean()),
        "sustained_mixed_fraction": float(sustained.mean()),
        "frac_half_mean": float(frac.mean()),
        "err_spread_median": float(np.median(spreads)) if spreads else float("nan"),
    }


def measure_grid_unit(params: dict, regime: dict, cfg: dict, seed: int,
                      device: torch.device) -> dict:
    """One (regime, configuration, seed) unit of the ablation grid.

    Runs the level sweep twice: once with diagnostics on to collect occupancy
    and error dispersion, and once with them off for the timing, so that the
    reported wall clock is not inflated by measurement machinery.
    """
    lam = regime_lambda(regime)

    sim = make_sim(params, regime, cfg, seed, device, diagnostics=True)
    variances, means, work_units, occupancy = [], [], 0.0, []

    for level in range(cfg["L_max"] + 1):
        y_fine, y_coarse = sim.run_level(
            level, cfg["n_samples"], cfg["T"], cfg["base_dt"], lambda_t=lam)
        diffs = y_fine - y_coarse
        variances.append(float(np.var(diffs, ddof=1)))
        means.append(float(np.mean(diffs)))
        # run_level resets the adaptive state per call, so accumulate here.
        work_units += (sim.adaptive_work_units() if params["adaptive_stepping"]
                       else fixed_work_units(cfg, level))

        if params["adaptive_stepping"]:
            # Every level and both roles, one row per step: recording only one
            # level would hide refinement that fires at coarser levels, where
            # the larger dt inflates the local-error indicator.
            per_role_step = {}
            for record in sim.adaptive_bucket_history:
                role = record["role"]
                per_role_step[role] = per_role_step.get(role, 0) + 1
                occupancy.append({"level": level, "step": per_role_step[role],
                                  **record})

    # Timing pass, diagnostics off.
    timed = make_sim(params, regime, cfg, seed, device, diagnostics=False)
    timing_level = cfg["timing_level"]
    timed.run_level(timing_level, max(32, cfg["n_samples"] // 8), cfg["T"],
                    cfg["base_dt"], lambda_t=lam)  # warm up
    synchronize(device)
    t0 = time.perf_counter()
    for _ in range(cfg["repeats"]):
        timed.run_level(timing_level, cfg["n_samples"], cfg["T"],
                        cfg["base_dt"], lambda_t=lam)
    synchronize(device)
    timed_wall = (time.perf_counter() - t0) / cfg["repeats"]

    uniform_steps = timed._adaptive_uniform_steps
    total_steps = max(len(timed.adaptive_bucket_history), 1)

    fine_trace = [r for r in occupancy if r["role"] == "fine"]
    return {
        "seed": seed,
        "alpha": decay_exponent(variances, skip=cfg["alpha_skip"]),
        "estimate": float(np.sum(means)),
        "level_variances": variances,
        "level_means": means,
        "timed_level": timing_level,
        "timed_wall_s": timed_wall,
        "work_units": work_units,
        "uniform_mask_step_fraction": float(uniform_steps) / total_steps,
        "occupancy_trace": occupancy,
        "occupancy_summary": occupancy_summary(fine_trace),
    }


# --------------------------------------------------- accuracy-work frontier --
def drive_paths(sim, increments, n_steps: int, T: float, lam, n_nodes: int,
                device: torch.device):
    """Advance an ensemble on `n_steps` using block sums of fine increments.

    Every scheme sees the SAME underlying Brownian path, so the differences
    measured downstream are discretisation error and nothing else.
    """
    block = increments.shape[2] // n_steps
    dt = T / n_steps
    c = torch.zeros(n_nodes, increments.shape[1], device=device)
    sim.reset_adaptive_state()
    for i in range(n_steps):
        dw = increments[:, :, i * block:(i + 1) * block].sum(dim=2)
        c = sim._step(c, dt, dw, None, lam, role="fine")
    return c


def frontier_point(params: dict, regime: dict, cfg: dict, seed: int,
                   device: torch.device, increments, reference,
                   n_steps: int, lam, repeats: int) -> dict:
    """Strong error, work and wall clock for one scheme setting."""
    n_nodes = regime["n_nodes"]
    n_paths = increments.shape[1]

    sim = make_sim(params, regime, cfg, seed, device, diagnostics=False)
    result = drive_paths(sim, increments, n_steps, cfg["T"], lam, n_nodes, device)
    error = float(torch.sqrt(((result - reference) ** 2).mean()).item())

    if params["adaptive_stepping"]:
        work = sim.adaptive_work_units()
        frac_half = float(np.mean([r["frac_half"]
                                   for r in sim.adaptive_bucket_history]))
    else:
        work = float(n_steps * n_paths)
        frac_half = 0.0

    synchronize(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        drive_paths(sim, increments, n_steps, cfg["T"], lam, n_nodes, device)
    synchronize(device)
    wall = (time.perf_counter() - t0) / repeats

    return {
        "n_steps": n_steps,
        "dt": cfg["T"] / n_steps,
        "error": error,
        "work_units": work,
        "wall_s": wall,
        "frac_half": frac_half,
    }


def measure_frontier_unit(regime_name: str, regime: dict, cfg: dict, seed: int,
                          device: torch.device) -> dict:
    """Error-vs-work and error-vs-wall-clock at matched accuracy.

    A reference solution is computed on a very fine fixed grid driven by the
    same Brownian path.  Both schemes are then evaluated against it, and the
    fixed-step curve is fitted so that the work a uniform grid would need to
    reach each adaptive point's accuracy can be read off directly.
    """
    n_nodes = regime["n_nodes"]
    n_paths = cfg["frontier_paths"]
    n_ref = cfg["frontier_ref_steps"]
    T = cfg["T"]

    torch.manual_seed(seed)
    lam = torch.tensor(regime_lambda(regime)[0], device=device)
    increments = (torch.randn(n_nodes, n_paths, n_ref, device=device)
                  * ((T / n_ref) ** 0.5))

    reference_params = {"adaptive_stepping": False, "adaptive_rtol": None,
                        "adaptive_error_estimator": None,
                        "reflection": "predictor_corrector"}
    reference_sim = make_sim(reference_params, regime, cfg, seed, device)
    reference = drive_paths(reference_sim, increments, n_ref, T, lam, n_nodes,
                            device)

    fixed_points, adaptive_points = [], []
    for n_steps in cfg["frontier_step_counts"]:
        point = frontier_point(reference_params, regime, cfg, seed, device,
                               increments, reference, n_steps, lam,
                               cfg["frontier_repeats"])
        point["scheme"] = "fixed"
        point["rtol"] = None
        fixed_points.append(point)

    for estimator in cfg["frontier_estimators"]:
        for n_steps in cfg["frontier_base_step_counts"]:
            for rtol in cfg["frontier_rtols"]:
                params = {"adaptive_stepping": True, "adaptive_rtol": rtol,
                          "adaptive_error_estimator": estimator,
                          "reflection": "predictor_corrector"}
                point = frontier_point(params, regime, cfg, seed, device,
                                       increments, reference, n_steps, lam,
                                       cfg["frontier_repeats"])
                point["scheme"] = f"adaptive_{estimator}"
                point["rtol"] = rtol
                adaptive_points.append(point)

    # Fit the uniform-grid cost of a given accuracy, then ask what each
    # adaptive point would have cost on a uniform grid at the same accuracy.
    work_of_error, work_slope = loglog_fit(
        [p["error"] for p in fixed_points], [p["work_units"] for p in fixed_points])
    wall_of_error, wall_slope = loglog_fit(
        [p["error"] for p in fixed_points], [p["wall_s"] for p in fixed_points])

    errors = [p["error"] for p in fixed_points]
    lo, hi = min(errors), max(errors)
    for point in adaptive_points:
        inside = lo <= point["error"] <= hi
        point["within_fixed_range"] = bool(inside)
        if work_of_error and inside:
            point["fixed_work_at_same_error"] = work_of_error(point["error"])
            point["work_speedup_vs_fixed"] = (
                point["fixed_work_at_same_error"] / point["work_units"])
        else:
            point["fixed_work_at_same_error"] = None
            point["work_speedup_vs_fixed"] = None
        if wall_of_error and inside:
            point["fixed_wall_at_same_error"] = wall_of_error(point["error"])
            point["wall_speedup_vs_fixed"] = (
                point["fixed_wall_at_same_error"] / point["wall_s"])
        else:
            point["fixed_wall_at_same_error"] = None
            point["wall_speedup_vs_fixed"] = None

    comparable = [p for p in adaptive_points if p["work_speedup_vs_fixed"]]
    best_work = max((p["work_speedup_vs_fixed"] for p in comparable), default=None)
    best_wall = max((p["wall_speedup_vs_fixed"] for p in comparable
                     if p["wall_speedup_vs_fixed"]), default=None)
    best_point = max(comparable, key=lambda p: p["work_speedup_vs_fixed"],
                     default=None)

    # A win has to clear a margin: a 1.6% work saving is measurement noise, not
    # evidence.  Wall clock is the operational criterion -- saving work while
    # taking longer is not a contribution anyone can use -- so the headline
    # verdict tracks wall clock and work is reported alongside it.
    wins_work = bool(best_work and best_work > WIN_MARGIN)
    wins_wall = bool(best_wall and best_wall > WIN_MARGIN)
    if wins_wall:
        verdict = "adaptive wins (work and wall)" if wins_work else \
            "adaptive wins on wall clock"
    elif wins_work:
        verdict = "work only; wall clock worse"
    else:
        verdict = "uniform grid cheaper"

    return {
        "seed": seed,
        "regime": regime_name,
        "fixed_points": fixed_points,
        "adaptive_points": adaptive_points,
        "fixed_work_slope": work_slope,
        "fixed_wall_slope": wall_slope,
        "n_comparable_points": len(comparable),
        "best_work_speedup": best_work,
        "best_wall_speedup": best_wall,
        "best_point_rtol": best_point["rtol"] if best_point else None,
        "best_point_scheme": best_point["scheme"] if best_point else None,
        "best_point_frac_half": best_point["frac_half"] if best_point else None,
        "win_margin": WIN_MARGIN,
        "adaptive_wins_on_work": wins_work,
        "adaptive_wins_on_wall": wins_wall,
        "verdict": verdict,
    }


# -------------------------------------------------------------- size sweep --
def measure_size_unit(regime_name: str, regime: dict, cfg: dict, seed: int,
                      device: torch.device, n_nodes: int) -> dict:
    """Occupancy and error dispersion as the network grows.

    Tests a structural objection: the refine decision is per path but the error
    indicator is a max over nodes, and maxima concentrate.  If dispersion falls
    as n grows, a bigger network makes mixed buckets LESS reachable, not more.
    """
    sized = dict(regime, n_nodes=n_nodes)
    lam = regime_lambda(sized)
    params = {"adaptive_stepping": True, "adaptive_rtol": cfg["size_rtol"],
              "adaptive_error_estimator": "embedded",
              "reflection": "predictor_corrector"}
    sim = make_sim(params, sized, cfg, seed, device, diagnostics=True)
    sim.run_level(cfg["size_level"], cfg["size_paths"], cfg["T"],
                  cfg["base_dt"], lambda_t=lam)

    trace = [r for r in sim.adaptive_bucket_history if r["role"] == "fine"]
    summary = occupancy_summary(trace)
    return {"seed": seed, "regime": regime_name, "n_nodes": n_nodes,
            **summary,
            "err_p05_median": float(np.median([r["err_p05"] for r in trace])),
            "err_p95_median": float(np.median([r["err_p95"] for r in trace]))}


# ------------------------------------------- MLMC bias / variance / cost ----
# Notation.  This section follows Giles (2008), which is the convention the
# manuscript's Theorem 1 uses.  Beware that it is the OPPOSITE of the loose
# usage elsewhere in this study, where "alpha" has meant the variance slope:
#
#   |E[P_l - P_{l-1}]| ~ 2^(-alpha_weak * l)     weak (bias) error       (alpha)
#   V_l = Var(P_l - P_{l-1}) ~ 2^(-beta_var * l) level-difference variance (beta)
#   C_l ~ 2^(gamma_cost * l)                     cost per sample          (gamma)
#
# Theorem 1's complexity depends on the sign of beta_var - gamma_cost, so the
# comparison that matters is not any single exponent but the total cost to
# reach a target MSE, computed below.

def mlmc_exponent(values: list, skip: int, sign: float) -> float:
    """Slope of log2(|values|) against level, over levels >= skip.

    `sign` is -1 for decaying quantities (variance, weak error) so that a
    healthy scheme returns a positive exponent, and +1 for cost.
    """
    usable = [(l, abs(v)) for l, v in enumerate(values)
              if l >= skip and np.isfinite(v) and abs(v) > 0]
    if len(usable) < 2:
        return float("nan")
    levels = np.array([l for l, _ in usable], dtype=float)
    logv = np.log2(np.array([v for _, v in usable], dtype=float))
    design = np.vstack([levels, np.ones_like(levels)]).T
    slope = np.linalg.lstsq(design, logv, rcond=None)[0][0]
    return sign * float(slope)


def mlmc_cost_to_epsilon(means: list, variances: list, costs: list,
                         alpha_weak: float, epsilons: list) -> list:
    """Total cost to reach MSE <= epsilon^2, by Giles's optimal allocation.

    Splits the MSE budget evenly between bias and variance.  The smallest level
    L whose extrapolated remaining bias |E[P_L - P_{L-1}]| / (2^alpha - 1) fits
    the bias half is selected, then

        N_l = ceil( 2 eps^-2 sqrt(V_l / C_l) * sum_k sqrt(V_k C_k) )

    which minimises sum_l N_l C_l subject to sum_l V_l / N_l <= eps^2 / 2.
    Cost is in timestep-path evaluations, so the adaptive scheme is charged for
    the sub-steps its refinement actually performed.
    """
    rows = []
    n_levels = len(means)
    for eps in epsilons:
        bias_budget = eps / np.sqrt(2.0)
        chosen_L = None
        for L in range(1, n_levels):
            if not np.isfinite(alpha_weak) or alpha_weak <= 0:
                break
            remaining = abs(means[L]) / max(2.0 ** alpha_weak - 1.0, 1e-12)
            if remaining <= bias_budget:
                chosen_L = L
                break
        if chosen_L is None:
            rows.append({"epsilon": eps, "levels_used": None,
                         "total_cost": None, "attainable": False})
            continue

        v = np.array(variances[:chosen_L + 1], dtype=float)
        c = np.array(costs[:chosen_L + 1], dtype=float)
        scale = float(np.sum(np.sqrt(np.maximum(v, 0.0) * c)))
        n_l = np.ceil(2.0 * eps ** -2 * np.sqrt(np.maximum(v, 0.0) / c) * scale)
        n_l = np.maximum(n_l, 1.0)
        rows.append({
            "epsilon": eps,
            "levels_used": chosen_L,
            "total_cost": float(np.sum(n_l * c)),
            "samples_per_level": [int(x) for x in n_l],
            "attainable": True,
        })
    return rows


def measure_mlmc_unit(regime_name: str, regime: dict, scheme: dict, cfg: dict,
                      seed: int, device: torch.device) -> dict:
    """Per-level bias, variance and measured cost for one scheme.

    This is the estimator-level criterion.  Adaptive MLMC in the Hoel et al.
    line is justified by what refinement does to V_l and to the bias, not by
    pathwise strong error, so this is the measurement that decides whether
    contribution (3) survives as a variance-reduction claim.
    """
    lam = regime_lambda(regime)
    params = dict(scheme["params"])
    n_samples = cfg["mlmc_samples"]

    means, variances, costs, direct, direct_var = [], [], [], [], []
    occupancy = []

    for level in range(cfg["mlmc_L_max"] + 1):
        sim = make_sim(params, regime, cfg, seed * 1000 + level, device,
                       diagnostics=params["adaptive_stepping"])
        y_fine, y_coarse = sim.run_level(level, n_samples, cfg["T"],
                                         cfg["base_dt"], lambda_t=lam)
        diffs = y_fine - y_coarse
        means.append(float(np.mean(diffs)))
        variances.append(float(np.var(diffs, ddof=1)))
        direct.append(float(np.mean(y_fine)))
        direct_var.append(float(np.var(y_fine, ddof=1)))

        if params["adaptive_stepping"]:
            costs.append(sim.adaptive_work_units() / n_samples)
            frac = [r["frac_half"] for r in sim.adaptive_bucket_history]
            occupancy.append(float(np.mean(frac)) if frac else 0.0)
        else:
            dt_fine = cfg["base_dt"] / (2 ** level)
            n_fine = int(cfg["T"] / dt_fine)
            n_coarse = 0 if level == 0 else int(cfg["T"] / (dt_fine * 2))
            costs.append(float(n_fine + n_coarse))
            occupancy.append(0.0)

    skip = cfg["mlmc_skip"]
    alpha_weak = mlmc_exponent(means, skip, -1.0)
    beta_var = mlmc_exponent(variances, skip, -1.0)
    gamma_cost = mlmc_exponent(costs, skip, +1.0)

    # Telescoping check: the running sum of level differences must reproduce a
    # direct estimate of E[P_l].  If adaptivity had broken the identity, the
    # cost-to-epsilon numbers below would be meaningless.
    # Each level is run with an independent seed, so the telescoped sum and the
    # direct estimate are independent and their difference has variance
    # (sum of the level-difference variances + the direct variance) / N.  Using
    # the single-level standard error here would overstate the discrepancy by
    # more than an order of magnitude, because Var(P_l) >> Var(P_l - P_{l-1}).
    telescoped = np.cumsum(means)
    telescope_gap = [float(abs(telescoped[l] - direct[l]))
                     for l in range(len(means))]
    mc_se = [float(np.sqrt((float(np.sum(variances[:l + 1])) + direct_var[l])
                           / n_samples))
             for l in range(len(means))]

    return {
        "seed": seed,
        "regime": regime_name,
        "scheme": scheme["name"],
        "params": params,
        "level_means": means,
        "level_variances": variances,
        "level_costs": costs,
        "level_direct_means": direct,
        "level_direct_variances": direct_var,
        "level_frac_half": occupancy,
        "telescope_gap": telescope_gap,
        "telescope_mc_se": mc_se,
        "alpha_weak": alpha_weak,
        "beta_var": beta_var,
        "gamma_cost": gamma_cost,
        "beta_minus_gamma": beta_var - gamma_cost,
        "cost_to_epsilon": mlmc_cost_to_epsilon(means, variances, costs,
                                                alpha_weak, cfg["mlmc_epsilons"]),
    }


def build_mlmc_schemes(cfg: dict) -> list:
    """Uniform baseline plus adaptive at a spread of tolerances.

    The tolerance is swept rather than hand-picked: the regime that shows
    sustained mixed occupancy is adaptivity's best case, and picking its
    tolerance after seeing the answer would be tuning toward an outcome.
    """
    schemes = [{
        "name": "uniform",
        "params": {"adaptive_stepping": False, "adaptive_rtol": None,
                   "adaptive_error_estimator": None,
                   "reflection": "predictor_corrector"},
    }]
    for rtol in cfg["mlmc_rtols"]:
        schemes.append({
            "name": f"adaptive_embedded_rtol{rtol:g}",
            "params": {"adaptive_stepping": True, "adaptive_rtol": rtol,
                       "adaptive_error_estimator": "embedded",
                       "reflection": "predictor_corrector"},
        })
    schemes.append({
        "name": "adaptive_halfstep_rtol1e-03",
        "params": {"adaptive_stepping": True, "adaptive_rtol": 1e-3,
                   "adaptive_error_estimator": "half_step",
                   "reflection": "predictor_corrector"},
    })
    return schemes


# ------------------------------------------------------------------ sweep ---
def build_configurations(rtols: list) -> list:
    """The ablation grid, with a stable config_id used as the resume key."""
    configurations = []
    for reflection in REFLECTIONS:
        configurations.append({
            "config_id": f"fixed__{reflection}",
            "params": {"adaptive_stepping": False, "adaptive_rtol": None,
                       "adaptive_error_estimator": None,
                       "reflection": reflection},
        })
    for reflection in REFLECTIONS:
        for estimator in ESTIMATORS:
            for rtol in rtols:
                configurations.append({
                    "config_id": (f"adaptive__{reflection}__{estimator}__"
                                  f"rtol{rtol:g}"),
                    "params": {"adaptive_stepping": True,
                               "adaptive_rtol": rtol,
                               "adaptive_error_estimator": estimator,
                               "reflection": reflection},
                })
    return configurations


def config_fingerprint(cfg: dict) -> str:
    """Identity of the experimental setup, excluding the seed list."""
    payload = {k: v for k, v in sorted(cfg.items()) if k != "seeds"}
    return json.dumps(payload, sort_keys=True, default=str)


def load_checkpoint(path: str, cfg: dict) -> dict:
    """Read completed units keyed by unit id.

    Tolerates a torn final line (the process can be killed mid-write on a
    reclaimed GPU) and refuses records written under a different schema or a
    different experimental configuration, so a resumed run cannot silently mix
    incomparable measurements.
    """
    done = {}
    if not os.path.exists(path):
        return done

    fingerprint = config_fingerprint(cfg)
    stale = 0
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print("  [checkpoint] ignoring truncated trailing record",
                      flush=True)
                continue
            if (record.get("schema_version") != SCHEMA_VERSION
                    or record.get("config_fingerprint") != fingerprint):
                stale += 1
                continue
            done[record["unit_id"]] = record

    if stale:
        print(f"  [checkpoint] ignoring {stale} record(s) from a different "
              f"schema or configuration", flush=True)
    return done


def append_checkpoint(path: str, record: dict) -> None:
    """Append one completed unit and force it to disk before continuing."""
    with open(path, "a") as handle:
        handle.write(json.dumps(record, default=float) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def aggregate(results: list, keys: list) -> dict:
    def stats(key):
        values = np.array([r.get(key, np.nan) for r in results], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {"mean": float("nan"), "sd": float("nan"), "n": 0}
        return {"mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "n": int(values.size)}
    return {key: stats(key) for key in keys}


def run_all(args, cfg: dict, device: torch.device) -> dict:
    checkpoint_path = os.path.join(args.out, "checkpoint.jsonl")
    if args.no_resume and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("  [checkpoint] --no-resume: cleared previous log", flush=True)

    done = load_checkpoint(checkpoint_path, cfg)
    if done:
        print(f"  [checkpoint] resuming, {len(done)} unit(s) already complete",
              flush=True)

    fingerprint = config_fingerprint(cfg)

    def execute(unit_id: str, fn, label: str):
        if unit_id in done:
            print(f"  {label}: cached", flush=True)
            return done[unit_id]["result"]
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        record = {"schema_version": SCHEMA_VERSION,
                  "config_fingerprint": fingerprint, "unit_id": unit_id,
                  "result": result, "unit_wall_s": elapsed}
        append_checkpoint(checkpoint_path, record)
        done[unit_id] = record
        print(f"  {label}  ({elapsed:.1f}s)", flush=True)
        return result

    output = {"grid": [], "frontier": [], "size": [], "mlmc": []}
    configurations = build_configurations(cfg["rtols"])
    path_counts = PATH_COUNTS_QUICK if args.quick else PATH_COUNTS

    # ---- experiment: ablation grid ------------------------------------
    if args.experiment in ("grid", "all"):
        print("\n" + "#" * 100)
        print("# EXPERIMENT 1/3: ablation grid")
        print("#" * 100, flush=True)
        for regime_name in cfg["regimes"]:
            regime = REGIMES[regime_name]
            print(f"\n=== regime {regime_name} "
                  f"(n={regime['n_nodes']}, sigma={regime['noise_intensity']}) ===",
                  flush=True)
            for configuration in configurations:
                config_id = configuration["config_id"]
                print(f"[{config_id}]", flush=True)
                seed_results = []
                for seed in cfg["seeds"]:
                    unit = f"grid|{regime_name}|{config_id}|{seed}"

                    def run(p=configuration["params"], r=regime, s=seed):
                        return measure_grid_unit(p, r, cfg, s, device)

                    summary = execute(unit, run, f"  seed {seed}")
                    seed_results.append(summary)

                occ = [r["occupancy_summary"] for r in seed_results]
                output["grid"].append({
                    "regime": regime_name,
                    "config_id": config_id,
                    "params": configuration["params"],
                    "seeds": seed_results,
                    "aggregate": aggregate(seed_results, [
                        "alpha", "estimate", "timed_wall_s", "work_units",
                        "uniform_mask_step_fraction"]),
                    "occupancy_aggregate": aggregate(occ, [
                        "mixed_step_fraction", "sustained_mixed_fraction",
                        "frac_half_mean", "err_spread_median"]),
                    "matmul_scaling": measure_matmul_scaling(
                        configuration["params"], regime, cfg, device,
                        path_counts),
                })

    # ---- experiment: accuracy-work frontier (the gate) ----------------
    if args.experiment in ("frontier", "all"):
        print("\n" + "#" * 100)
        print("# EXPERIMENT 2/3: accuracy-work frontier (THE GATE)")
        print("#" * 100, flush=True)
        for regime_name in cfg["frontier_regimes"]:
            regime = REGIMES[regime_name]
            print(f"\n=== regime {regime_name} ===", flush=True)
            for seed in cfg["seeds"]:
                unit = f"frontier|{regime_name}|{seed}"

                def run(rn=regime_name, r=regime, s=seed):
                    return measure_frontier_unit(rn, r, cfg, s, device)

                result = execute(unit, run, f"  seed {seed}")
                best = result["best_work_speedup"]
                print(f"    best work speedup vs uniform grid at matched "
                      f"accuracy: "
                      f"{'n/a' if best is None else f'{best:.3f}x'}", flush=True)
                output["frontier"].append(result)

    # ---- experiment: MLMC bias / variance / cost (estimator criterion) --
    if args.experiment in ("mlmc", "all"):
        print("\n" + "#" * 100)
        print("# EXPERIMENT: MLMC bias, variance and cost-to-epsilon")
        print("#" * 100, flush=True)
        schemes = build_mlmc_schemes(cfg)
        for regime_name in cfg["mlmc_regimes"]:
            regime = REGIMES[regime_name]
            print(f"\n=== regime {regime_name} ===", flush=True)
            for scheme in schemes:
                for seed in cfg["seeds"]:
                    unit = f"mlmc|{regime_name}|{scheme['name']}|{seed}"

                    def run(rn=regime_name, r=regime, sc=scheme, s=seed):
                        return measure_mlmc_unit(rn, r, sc, cfg, s, device)

                    result = execute(unit, run,
                                     f"  {scheme['name']:<32} seed {seed}")
                    output["mlmc"].append(result)

    # ---- experiment: size sweep ---------------------------------------
    if args.experiment in ("size", "all"):
        print("\n" + "#" * 100)
        print("# EXPERIMENT 3/3: occupancy and dispersion vs network size")
        print("#" * 100, flush=True)
        sizes = SIZE_SWEEP_QUICK if args.quick else SIZE_SWEEP
        for regime_name in cfg["size_regimes"]:
            regime = REGIMES[regime_name]
            print(f"\n=== regime {regime_name} ===", flush=True)
            for n_nodes in sizes:
                for seed in cfg["seeds"]:
                    unit = f"size|{regime_name}|{n_nodes}|{seed}"

                    def run(rn=regime_name, r=regime, s=seed, n=n_nodes):
                        return measure_size_unit(rn, r, cfg, s, device, n)

                    result = execute(unit, run, f"  n={n_nodes} seed {seed}")
                    output["size"].append(result)

    return output


# ----------------------------------------------------------------- output ---
def write_outputs(args, cfg: dict, prov: dict, results: dict) -> dict:
    os.makedirs(args.out, exist_ok=True)
    header = provenance_comment_lines(prov)
    paths = {}

    def open_csv(name, extra_comments=()):
        handle = open(os.path.join(args.out, name), "w", newline="")
        for line in header:
            handle.write(line + "\n")
        for line in extra_comments:
            handle.write("# " + line + "\n")
        return handle

    json_path = os.path.join(args.out, "adaptive_stepping_ablation.json")
    with open(json_path, "w") as handle:
        json.dump({"provenance": prov, "regimes": REGIMES, **results},
                  handle, indent=2, default=float)
    paths["json"] = json_path

    if results["grid"]:
        with open_csv("summary.csv") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "regime", "config_id", "adaptive_stepping", "adaptive_rtol",
                "adaptive_error_estimator", "reflection", "seed", "alpha",
                "estimate", "timed_level", "timed_wall_s", "work_units",
                "uniform_mask_step_fraction", "mixed_step_fraction",
                "sustained_mixed_fraction", "frac_half_mean",
                "err_spread_median", "alpha_mean", "alpha_sd",
                "estimate_mean", "estimate_sd", "timed_wall_mean_s",
                "timed_wall_sd_s",
            ])
            for entry in results["grid"]:
                params, agg = entry["params"], entry["aggregate"]
                for result in entry["seeds"]:
                    occ = result["occupancy_summary"]
                    writer.writerow([
                        entry["regime"], entry["config_id"],
                        params["adaptive_stepping"], params["adaptive_rtol"],
                        params["adaptive_error_estimator"], params["reflection"],
                        result["seed"], result["alpha"], result["estimate"],
                        result["timed_level"], result["timed_wall_s"],
                        result["work_units"],
                        result["uniform_mask_step_fraction"],
                        occ["mixed_step_fraction"],
                        occ["sustained_mixed_fraction"], occ["frac_half_mean"],
                        occ["err_spread_median"], agg["alpha"]["mean"],
                        agg["alpha"]["sd"], agg["estimate"]["mean"],
                        agg["estimate"]["sd"], agg["timed_wall_s"]["mean"],
                        agg["timed_wall_s"]["sd"],
                    ])
        paths["summary_csv"] = os.path.join(args.out, "summary.csv")

        with open_csv("bucket_occupancy.csv", [
                "every level and both path roles, one row per nominal step",
                f"n_samples={cfg['n_samples']}"]) as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "regime", "config_id", "adaptive_rtol",
                "adaptive_error_estimator", "reflection", "seed", "level",
                "role", "step", "dt", "n_full", "n_half", "frac_half",
                "err_p05", "err_p50", "err_p95", "err_spread",
            ])
            for entry in results["grid"]:
                params = entry["params"]
                for result in entry["seeds"]:
                    for record in result["occupancy_trace"]:
                        writer.writerow([
                            entry["regime"], entry["config_id"],
                            params["adaptive_rtol"],
                            params["adaptive_error_estimator"],
                            params["reflection"], result["seed"],
                            record["level"], record["role"], record["step"],
                            record["dt"], record["n_full"], record["n_half"],
                            record["frac_half"], record.get("err_p05"),
                            record.get("err_p50"), record.get("err_p95"),
                            record.get("err_spread"),
                        ])
        paths["occupancy_csv"] = os.path.join(args.out, "bucket_occupancy.csv")

        with open_csv("matmul_scaling.csv", [
                "one nominal step, buckets forced to a 50/50 split;",
                "mm_calls must stay constant as n_paths grows"]) as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "regime", "config_id", "adaptive_stepping", "adaptive_rtol",
                "adaptive_error_estimator", "reflection", "n_paths",
                "mm_calls", "mm_calls_constant_across_path_counts",
            ])
            for entry in results["grid"]:
                params = entry["params"]
                invariant = len({row["mm_calls"]
                                 for row in entry["matmul_scaling"]}) == 1
                for row in entry["matmul_scaling"]:
                    writer.writerow([
                        entry["regime"], entry["config_id"],
                        params["adaptive_stepping"], params["adaptive_rtol"],
                        params["adaptive_error_estimator"], params["reflection"],
                        row["n_paths"], row["mm_calls"], invariant,
                    ])
        paths["matmul_csv"] = os.path.join(args.out, "matmul_scaling.csv")

    if results["frontier"]:
        with open_csv("accuracy_work.csv", [
                "strong error at T vs a fine-grid reference on the same",
                "Brownian path; speedup > 1 means adaptive reaches the same",
                "accuracy for less work/wall clock than a uniform grid"]) as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "regime", "seed", "scheme", "rtol", "n_steps", "dt", "error",
                "work_units", "wall_s", "frac_half", "within_fixed_range",
                "fixed_work_at_same_error", "work_speedup_vs_fixed",
                "fixed_wall_at_same_error", "wall_speedup_vs_fixed",
            ])
            for entry in results["frontier"]:
                for point in entry["fixed_points"] + entry["adaptive_points"]:
                    writer.writerow([
                        entry["regime"], entry["seed"], point["scheme"],
                        point.get("rtol"), point["n_steps"], point["dt"],
                        point["error"], point["work_units"], point["wall_s"],
                        point["frac_half"], point.get("within_fixed_range"),
                        point.get("fixed_work_at_same_error"),
                        point.get("work_speedup_vs_fixed"),
                        point.get("fixed_wall_at_same_error"),
                        point.get("wall_speedup_vs_fixed"),
                    ])
        paths["accuracy_work_csv"] = os.path.join(args.out, "accuracy_work.csv")

        with open_csv("gate_verdict.csv", [
                "the gate: does adaptive stepping beat a uniform grid at",
                "matched accuracy, on work or on wall clock?"]) as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "regime", "seed", "n_comparable_points", "best_work_speedup",
                "best_wall_speedup", "best_point_scheme", "best_point_rtol",
                "best_point_frac_half", "win_margin", "adaptive_wins_on_work",
                "adaptive_wins_on_wall", "verdict", "fixed_work_slope",
                "fixed_wall_slope",
            ])
            for entry in results["frontier"]:
                writer.writerow([
                    entry["regime"], entry["seed"],
                    entry["n_comparable_points"], entry["best_work_speedup"],
                    entry["best_wall_speedup"], entry["best_point_scheme"],
                    entry["best_point_rtol"], entry["best_point_frac_half"],
                    entry["win_margin"], entry["adaptive_wins_on_work"],
                    entry["adaptive_wins_on_wall"], entry["verdict"],
                    entry["fixed_work_slope"], entry["fixed_wall_slope"],
                ])
        paths["gate_csv"] = os.path.join(args.out, "gate_verdict.csv")

    if results["mlmc"]:
        with open_csv("mlmc_levels.csv", [
                "per-level bias, variance and MEASURED cost per sample;",
                "cost is in timestep-path evaluations so adaptive is charged",
                "for the sub-steps refinement actually performed"]) as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "regime", "scheme", "seed", "level", "mean_diff", "variance",
                "cost_per_sample", "direct_mean", "frac_half",
                "telescope_gap", "telescope_mc_se",
            ])
            for entry in results["mlmc"]:
                for level in range(len(entry["level_means"])):
                    writer.writerow([
                        entry["regime"], entry["scheme"], entry["seed"], level,
                        entry["level_means"][level],
                        entry["level_variances"][level],
                        entry["level_costs"][level],
                        entry["level_direct_means"][level],
                        entry["level_frac_half"][level],
                        entry["telescope_gap"][level],
                        entry["telescope_mc_se"][level],
                    ])
        paths["mlmc_levels_csv"] = os.path.join(args.out, "mlmc_levels.csv")

        with open_csv("mlmc_exponents.csv", [
                "Giles (2008) convention, as used by the manuscript Theorem 1:",
                "|E[P_l - P_{l-1}]| ~ 2^-(alpha_weak l);  V_l ~ 2^-(beta_var l);",
                "C_l ~ 2^(gamma_cost l).  NOTE: elsewhere in this study 'alpha'",
                "has loosely meant the variance slope, i.e. beta_var here."]) as handle:
            writer = csv.writer(handle)
            writer.writerow(["regime", "scheme", "seed", "alpha_weak",
                             "beta_var", "gamma_cost", "beta_minus_gamma"])
            for entry in results["mlmc"]:
                writer.writerow([
                    entry["regime"], entry["scheme"], entry["seed"],
                    entry["alpha_weak"], entry["beta_var"],
                    entry["gamma_cost"], entry["beta_minus_gamma"],
                ])
        paths["mlmc_exponents_csv"] = os.path.join(args.out,
                                                   "mlmc_exponents.csv")

        with open_csv("mlmc_cost_to_epsilon.csv", [
                "total cost to reach MSE <= epsilon^2 under Giles's optimal",
                "allocation; this is the quantity Theorem 1 is about"]) as handle:
            writer = csv.writer(handle)
            writer.writerow(["regime", "scheme", "seed", "epsilon",
                             "levels_used", "total_cost", "attainable"])
            for entry in results["mlmc"]:
                for row in entry["cost_to_epsilon"]:
                    writer.writerow([
                        entry["regime"], entry["scheme"], entry["seed"],
                        row["epsilon"], row["levels_used"], row["total_cost"],
                        row["attainable"],
                    ])
        paths["mlmc_cost_csv"] = os.path.join(args.out,
                                              "mlmc_cost_to_epsilon.csv")

    if results["size"]:
        with open_csv("size_sweep.csv", [
                "the refine decision is per path but the error indicator is a",
                "max over nodes; if dispersion falls as n grows, a larger",
                "network makes mixed buckets less reachable, not more"]) as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "regime", "n_nodes", "seed", "mixed_step_fraction",
                "sustained_mixed_fraction", "frac_half_mean",
                "err_spread_median", "err_p05_median", "err_p95_median",
            ])
            for row in results["size"]:
                writer.writerow([
                    row["regime"], row["n_nodes"], row["seed"],
                    row["mixed_step_fraction"], row["sustained_mixed_fraction"],
                    row["frac_half_mean"], row["err_spread_median"],
                    row["err_p05_median"], row["err_p95_median"],
                ])
        paths["size_csv"] = os.path.join(args.out, "size_sweep.csv")

    return paths


WIDTH = 50


def print_report(results: dict, cfg: dict) -> None:
    if results["grid"]:
        print("\n" + "=" * 118)
        print("ABLATION GRID")
        print("=" * 118)
        print(f"{'regime':<20}{'configuration':<{WIDTH}}{'alpha':>14}"
              f"{'estimate':>17}{'wall/rep':>10}{'work':>11}")
        for entry in results["grid"]:
            agg = entry["aggregate"]
            print(f"{entry['regime']:<20}{entry['config_id']:<{WIDTH}}"
                  f"{agg['alpha']['mean']:>8.4f}+-{agg['alpha']['sd']:<5.3f}"
                  f"{agg['estimate']['mean']:>10.5f}+-{agg['estimate']['sd']:<6.5f}"
                  f"{agg['timed_wall_s']['mean']:>9.3f}s"
                  f"{agg['work_units']['mean']:>11.3e}")

        print("\nMIXED-BUCKET REACHABILITY  (the design premise under test)")
        print(f"{'regime':<20}{'configuration':<{WIDTH}}{'mixed':>9}"
              f"{'sustained':>11}{'frac_half':>11}{'err p95/p05':>13}")
        for entry in results["grid"]:
            if not entry["params"]["adaptive_stepping"]:
                continue
            occ = entry["occupancy_aggregate"]
            print(f"{entry['regime']:<20}{entry['config_id']:<{WIDTH}}"
                  f"{occ['mixed_step_fraction']['mean']:>9.3f}"
                  f"{occ['sustained_mixed_fraction']['mean']:>11.3f}"
                  f"{occ['frac_half_mean']['mean']:>11.3f}"
                  f"{occ['err_spread_median']['mean']:>13.3f}")

        print("\ntorch.mm calls per nominal step vs path count")
        seen = set()
        for entry in results["grid"]:
            if entry["config_id"] in seen:
                continue
            seen.add(entry["config_id"])
            rows = entry["matmul_scaling"]
            counts = {row["mm_calls"] for row in rows}
            verdict = ("INVARIANT" if len(counts) == 1
                       else "VARIES WITH N_PATHS -- claim unsupported")
            detail = " ".join(f"N={r['n_paths']}:{r['mm_calls']}" for r in rows)
            print(f"{entry['config_id']:<{WIDTH}}{verdict:<11} {detail}")

    if results["frontier"]:
        print("\n" + "=" * 118)
        print("THE GATE: does adaptive beat a uniform grid at matched accuracy?")
        print("=" * 118)
        print(f"{'regime':<22}{'seed':>5}{'best work x':>13}{'best wall x':>13}"
              f"{'at rtol':>10}{'f_half':>8}{'  verdict'}")
        wall_wins, work_wins = [], []
        for entry in results["frontier"]:
            work, wall = entry["best_work_speedup"], entry["best_wall_speedup"]
            wall_wins.append(entry["adaptive_wins_on_wall"])
            work_wins.append(entry["adaptive_wins_on_work"])
            rtol = entry.get("best_point_rtol")
            frac = entry.get("best_point_frac_half")
            work_s = "n/a" if work is None else f"{work:.3f}"
            wall_s = "n/a" if wall is None else f"{wall:.3f}"
            rtol_s = "n/a" if rtol is None else f"{rtol:g}"
            frac_s = "n/a" if frac is None else f"{frac:.3f}"
            print(f"{entry['regime']:<22}{entry['seed']:>5}{work_s:>13}"
                  f"{wall_s:>13}{rtol_s:>10}{frac_s:>8}  {entry['verdict']}")
        # The decisive cut.  If refinement were buying accuracy in proportion
        # to its cost, the work speed-up would RISE with half-step occupancy.
        # If it falls, the paths being refined are not the ones carrying the
        # error, and the scheme cannot be rescued by tuning the tolerance.
        bins = [("0.00 (degenerates to fixed)", 0.0, 0.01),
                ("0.01-0.25", 0.01, 0.25),
                ("0.25-0.75 (mixed buckets)", 0.25, 0.75),
                ("0.75-1.00", 0.75, 1.01)]
        points = [p for entry in results["frontier"]
                  for p in entry["adaptive_points"]
                  if p.get("work_speedup_vs_fixed")]
        if points:
            print("\n  Work and wall speed-up vs achieved half-step occupancy")
            print(f"  {'occupancy f_half':<30}{'points':>8}{'work x':>10}"
                  f"{'wall x':>10}")
            for label, lo, hi in bins:
                chosen = [p for p in points if lo <= p["frac_half"] < hi]
                if not chosen:
                    continue
                work = float(np.mean([p["work_speedup_vs_fixed"] for p in chosen]))
                walls = [p["wall_speedup_vs_fixed"] for p in chosen
                         if p["wall_speedup_vs_fixed"]]
                wall = float(np.mean(walls)) if walls else float("nan")
                print(f"  {label:<30}{len(chosen):>8}{work:>10.3f}{wall:>10.3f}")
            print("  (a speed-up that FALLS as occupancy rises means refinement")
            print("   is not being spent on the paths that carry the error)")

        print(f"\n  Win margin: a speed-up must exceed {WIN_MARGIN:.2f}x to count.")
        print("  Ceiling for reference: with two buckets the best possible")
        print("  work speed-up over a uniformly halved grid is 2/(1+f) <= 2.0x.")
        if any(wall_wins):
            overall = "adaptive stepping beats a uniform grid on wall clock"
        elif any(work_wins):
            overall = ("adaptive stepping saves work in some regime but is "
                       "SLOWER in wall clock everywhere")
        else:
            overall = ("no regime tested favours adaptive stepping on either "
                       "work or wall clock")
        print(f"\n  OVERALL: {overall}")

    if results["mlmc"]:
        print("\n" + "=" * 118)
        print("MLMC ESTIMATOR CRITERION: bias, variance and cost-to-epsilon")
        print("=" * 118)
        print("  Giles (2008) convention, as in the manuscript Theorem 1:")
        print("    |E[P_l - P_{l-1}]| ~ 2^-(alpha_weak.l)   V_l ~ 2^-(beta_var.l)"
              "   C_l ~ 2^(gamma_cost.l)")
        print("  (elsewhere in this study 'alpha' loosely meant the variance"
              " slope, i.e. beta_var)")

        grouped = {}
        for entry in results["mlmc"]:
            grouped.setdefault((entry["regime"], entry["scheme"]), []).append(entry)

        print(f"\n{'regime':<20}{'scheme':<30}{'alpha_weak':>16}"
              f"{'beta_var':>16}{'gamma_cost':>16}{'beta-gamma':>13}"
              f"{'f_half':>8}")
        for (regime_name, scheme), entries in sorted(grouped.items()):
            def stat(key):
                values = np.array([e[key] for e in entries], dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    return float("nan"), float("nan")
                return (float(values.mean()),
                        float(values.std(ddof=1)) if values.size > 1 else 0.0)
            a_m, a_s = stat("alpha_weak")
            b_m, b_s = stat("beta_var")
            g_m, g_s = stat("gamma_cost")
            bg_m, _ = stat("beta_minus_gamma")
            frac = float(np.mean([np.mean(e["level_frac_half"]) for e in entries]))
            print(f"{regime_name:<20}{scheme:<30}{a_m:>9.3f}+-{a_s:<5.3f}"
                  f"{b_m:>9.3f}+-{b_s:<5.3f}{g_m:>9.3f}+-{g_s:<5.3f}"
                  f"{bg_m:>13.3f}{frac:>8.3f}")

        # Telescoping validation: adaptivity must not have broken the identity.
        worst = 0.0
        for entry in results["mlmc"]:
            for gap, se in zip(entry["telescope_gap"], entry["telescope_mc_se"]):
                if se > 0:
                    worst = max(worst, gap / se)
        print(f"\n  Telescoping check: worst |sum of level differences - direct "
              f"E[P_l]| = {worst:.2f} Monte Carlo standard errors")
        print("  (a large value would mean the identity broke and every cost "
              "number below is void)")

        print("\n  TOTAL COST TO REACH MSE <= epsilon^2  (Theorem 1's quantity)")
        print(f"  {'regime':<20}{'epsilon':>10}{'uniform':>14}"
              f"{'best adaptive':>16}{'scheme':>30}{'cost ratio':>12}")
        for regime_name in sorted({e["regime"] for e in results["mlmc"]}):
            for eps in cfg["mlmc_epsilons"]:
                costs = {}
                for (r, scheme), entries in grouped.items():
                    if r != regime_name:
                        continue
                    values = [row["total_cost"] for e in entries
                              for row in e["cost_to_epsilon"]
                              if row["epsilon"] == eps and row["total_cost"]]
                    if values:
                        costs[scheme] = float(np.mean(values))
                if "uniform" not in costs or len(costs) < 2:
                    continue
                adaptive = {k: v for k, v in costs.items() if k != "uniform"}
                best_scheme = min(adaptive, key=adaptive.get)
                ratio = costs["uniform"] / adaptive[best_scheme]
                print(f"  {regime_name:<20}{eps:>10.4f}{costs['uniform']:>14.3e}"
                      f"{adaptive[best_scheme]:>16.3e}{best_scheme:>30}"
                      f"{ratio:>12.3f}")
        print("\n  cost ratio = uniform / best adaptive; > 1 means adaptive is"
              " cheaper at that accuracy")

    if results["size"]:
        print("\n" + "=" * 118)
        print("OCCUPANCY AND ERROR DISPERSION VS NETWORK SIZE")
        print("=" * 118)
        print(f"{'regime':<22}{'n_nodes':>9}{'mixed':>9}{'sustained':>11}"
              f"{'frac_half':>11}{'err p95/p05':>13}")
        grouped = {}
        for row in results["size"]:
            grouped.setdefault((row["regime"], row["n_nodes"]), []).append(row)
        for (regime_name, n_nodes), rows in sorted(grouped.items()):
            def mean(key):
                values = [r[key] for r in rows if np.isfinite(r[key])]
                return float(np.mean(values)) if values else float("nan")
            print(f"{regime_name:<22}{n_nodes:>9}{mean('mixed_step_fraction'):>9.3f}"
                  f"{mean('sustained_mixed_fraction'):>11.3f}"
                  f"{mean('frac_half_mean'):>11.3f}"
                  f"{mean('err_spread_median'):>13.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive stepping ablation and accuracy-work study")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default="results/adaptive_stepping")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="auto prefers cuda > mps > cpu")
    parser.add_argument("--quick", action="store_true",
                        help="Small smoke-test configuration")
    parser.add_argument("--experiment", default="all",
                        choices=["all", "grid", "frontier", "size", "mlmc"])
    parser.add_argument("--mlmc-samples", type=int, default=8192,
                        help="Paths per level for the bias/variance estimates")
    parser.add_argument("--mlmc-L-max", type=int, default=6)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--regimes", nargs="+", default=None,
                        choices=sorted(REGIMES), help="default: all regimes")
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--L-max", type=int, default=4)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--base-dt", type=float, default=0.1)
    parser.add_argument("--rtols", type=float, nargs="+", default=RTOL_DEFAULTS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timing-level", type=int, default=3)
    parser.add_argument("--alpha-skip", type=int, default=1)
    parser.add_argument("--frontier-paths", type=int, default=1024)
    parser.add_argument("--frontier-ref-steps", type=int, default=2048)
    parser.add_argument("--frontier-repeats", type=int, default=2)
    args = parser.parse_args()

    regimes = args.regimes or sorted(REGIMES)
    frontier_regimes = [r for r in FRONTIER_REGIMES if r in regimes] or regimes
    # Adaptivity's best case is where sustained mixed occupancy was observed
    # (heterogeneous_rho, bimodal_load), plus the original chain as a control.
    mlmc_regimes = [r for r in ("heterogeneous_rho", "bimodal_load",
                                "reflection_active", "homogeneous_chain")
                    if r in regimes] or regimes[:1]
    mlmc_rtols = [1e-2, 1e-3, 1e-4, 1e-5]
    mlmc_epsilons = [0.05, 0.03, 0.02, 0.01, 0.007, 0.005]
    step_counts = [16, 32, 64, 128, 256]
    base_step_counts = [16, 32, 64]
    frontier_rtols = FRONTIER_RTOLS
    frontier_estimators = ESTIMATORS

    if args.quick:
        args.n_samples = min(args.n_samples, 512)
        args.L_max = min(args.L_max, 2)
        args.repeats = 1
        args.timing_level = min(args.timing_level, 2)
        args.frontier_paths = min(args.frontier_paths, 256)
        args.frontier_ref_steps = min(args.frontier_ref_steps, 512)
        args.frontier_repeats = 1
        args.seeds = args.seeds[:2]
        regimes = [r for r in QUICK_REGIMES if r in regimes] or regimes[:2]
        frontier_regimes = regimes[:2]
        step_counts = [16, 32, 64]
        base_step_counts = [16, 32]
        frontier_rtols = FRONTIER_RTOLS_QUICK
        frontier_estimators = ["embedded"]
        args.mlmc_samples = min(args.mlmc_samples, 2048)
        args.mlmc_L_max = min(args.mlmc_L_max, 3)
        mlmc_regimes = mlmc_regimes[:2]
        mlmc_rtols = [1e-2, 1e-4]
        mlmc_epsilons = [0.02, 0.005]

    if args.timing_level > args.L_max:
        parser.error("--timing-level must be <= --L-max")

    device = select_device(args.device)
    cfg = {
        "regimes": regimes,
        "frontier_regimes": frontier_regimes,
        "size_regimes": [r for r in ("bimodal_load", "reflection_active")
                         if r in regimes] or regimes[:1],
        "n_samples": args.n_samples,
        "L_max": args.L_max,
        "T": args.T,
        "base_dt": args.base_dt,
        "rtols": list(args.rtols),
        "repeats": args.repeats,
        "timing_level": args.timing_level,
        "alpha_skip": args.alpha_skip,
        "seeds": list(args.seeds),
        "influence_strength": 0.2,
        "decay_rate": 0.5,
        "frontier_paths": args.frontier_paths,
        "frontier_ref_steps": args.frontier_ref_steps,
        "frontier_repeats": args.frontier_repeats,
        "frontier_step_counts": step_counts,
        "frontier_base_step_counts": base_step_counts,
        "frontier_rtols": frontier_rtols,
        "frontier_estimators": frontier_estimators,
        "mlmc_regimes": mlmc_regimes,
        "mlmc_samples": args.mlmc_samples,
        "mlmc_L_max": args.mlmc_L_max,
        "mlmc_skip": 1,
        "mlmc_rtols": mlmc_rtols,
        "mlmc_epsilons": mlmc_epsilons,
        "size_rtol": 1e-4,
        "size_level": 1,
        "size_paths": 1024 if not args.quick else 256,
        "experiment": args.experiment,
        "quick": args.quick,
    }

    os.makedirs(args.out, exist_ok=True)
    prov = build_provenance(device, cfg)

    print("=" * 100)
    print("Adaptive stepping ablation and accuracy-work study")
    print("=" * 100)
    print(f"  git sha    : {prov['git_sha']}")
    print(f"  device     : {prov['device']} ({prov['device_name']})")
    print(f"  torch      : {prov['torch_version']}")
    print(f"  seeds      : {cfg['seeds']}")
    print(f"  experiment : {args.experiment}")
    print(f"  regimes    : {', '.join(regimes)}")
    print(f"  out        : {os.path.abspath(args.out)}", flush=True)

    started = time.perf_counter()
    results = run_all(args, cfg, device)
    total = time.perf_counter() - started

    paths = write_outputs(args, cfg, prov, results)
    print_report(results, cfg)

    print(f"\nTotal wall clock: {total:.1f}s")
    for label, path in paths.items():
        print(f"  {label:<20} {path}")


if __name__ == "__main__":
    main()
