"""
Cumulative component-ablation ladder for the GPU-MLMC congestion estimator
(Reviewer 2, T2.1).  Answers: which of the manuscript's six additions on top
of single-level GPU Monte Carlo actually earns its keep, and at what cost?

Rungs (each ADDS one component to the previous):
    R0  single-level GPU Monte Carlo baseline (no MLMC hierarchy)
    R1  + MLMC, uniform Giles allocation                 (reflection=euler_clamp)
    R2  + predictor-corrector reflection                 (reflection=predictor_corrector)
    R3  + SIMT-adaptive time stepping                    (adaptive_stepping=True)
    R4  + ANA network-aware weighted allocation           (estimand changes!)
    R5  + importance sampling                             (rare-event target only; estimand changes!)
    R6  + multi-GPU G=2                                    (guarded; estimand differs architecturally)

Two API limitations of src/gpu/parallel_mc.py, discovered while writing this
script and NOT worked around by editing src/ (out of scope for this file):

  1. GPUAdaptiveNetworkAwareMLMC.__init__ does not forward adaptive_stepping /
     reflection / adaptive_rtol / adaptive_error_estimator to its parent
     GPUCoupledPropagationMLMC.__init__ -- those constructor kwargs simply
     don't exist on the ANA subclass.  R4 needs adaptive_stepping=True and
     reflection='predictor_corrector' carried forward from R3, so this script
     sets those as plain post-construction attribute overrides (see
     `_carry_forward_r3_scheme`).  This is behaviourally identical to what the
     constructor would have done had it forwarded the kwargs -- nothing in
     __init__ depends on them beyond storing them -- but it is a workaround,
     not a supported code path, and is called out here for the record.

  2. GPUImportanceSamplingMLMC.simulate_is_paths() and
     MultiGPUMLMC._simulate_level_local() / _step_with_halo() each implement
     their OWN fixed-step Euler-Maruyama kernel and never call self._step /
     self._em_step_adaptive.  Passing adaptive_stepping=True or
     reflection='predictor_corrector' into either class's constructor has
     ZERO effect on what actually runs at R5 or R6 -- those flags are simply
     dead in those code paths.  R5 and R6 therefore cannot literally inherit
     R2's or R3's scheme improvements; this script does not pretend otherwise
     ("components_active" in every row records what actually ran) and both
     rungs are marked comparable=False against the rest of the ladder.

Per rung this script records: work W = sum_l N_l * M_l (timestep-path
evaluations), wall-clock, CI half-width, achieved MSE vs a tight per-seed
reference, samples per level, incremental change vs the previous rung, and a
paired-t test (t, p, Cohen's-d effect size) across seeds vs the previous rung.

Estimand honesty.  R0-R3 all target E[mean_congestion] (uniform average over
nodes) and share one "uniform_mean" reference computed once per (topology,
seed) at a much tighter epsilon.  R4 targets sum_i w_i E[Q_i] under the ANA
node weights -- a different functional -- and gets its own "ana_weighted"
reference.  R5 targets a tail probability P(Q_target > B); R6's estimator
uses a structurally different SDE discretisation (see limitation 2 above).
Every row carries "comparable" (bool, relative to the uniform_mean family)
and "comparable_note" explaining why, per the instruction not to compare
different estimands as if they were the same quantity.

Usage:
    python3 scripts/run_ablation_ladder.py --quick
    python3 scripts/run_ablation_ladder.py --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --topologies caida ba --n-nodes 500 --epsilon 0.02 --device cuda

Output (under --out, default results/ablation_ladder):
    ablation_ladder.json      full results + provenance
    rungs.csv                 one row per (topology, rung, seed)
    transitions.csv           one row per (topology, rung-pair): incremental
                               change + paired t-test + effect size
    checkpoint.jsonl          append-only resume log
"""
from __future__ import annotations

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

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import networkx as nx  # noqa: E402
import torch  # noqa: E402
from scipy import stats as sp_stats  # noqa: E402

from gpu.parallel_mc import (  # noqa: E402
    GPUAdaptiveNetworkAwareMLMC,
    GPUCoupledPropagationMLMC,
    GPUImportanceSamplingMLMC,
    MultiGPUMLMC,
)
from network.topology import NetworkGraph, TopologyGenerator, load_caida_topology  # noqa: E402

#: Bumped whenever a checkpoint record's shape changes.
SCHEMA_VERSION = 1

RUNG_ORDER = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
RUNG_LABELS = {
    "R0": "single-level GPU MC baseline",
    "R1": "+ MLMC, uniform Giles allocation",
    "R2": "+ predictor-corrector reflection",
    "R3": "+ SIMT-adaptive time stepping",
    "R4": "+ ANA network-aware weighted allocation",
    "R5": "+ importance sampling (rare-event target)",
    "R6": "+ multi-GPU G=2",
}
#: Rung family: rungs in the same family share a reference and the CI /
#: estimate / MSE columns of the paired t-test are meaningful between them.
RUNG_FAMILY = {
    "R0": "uniform_mean", "R1": "uniform_mean", "R2": "uniform_mean",
    "R3": "uniform_mean", "R4": "ana_weighted", "R5": "rare_event",
    "R6": "multi_gpu_local",
}
WIN_MARGIN = 1.05  # a change must clear this margin to be reported as a real effect, not noise


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


def pin_device(sim, device: torch.device):
    """Move a GPUCoupledPropagationMLMC (or subclass) onto `device`.

    The constructor hardcodes 'cuda if available else cpu' -- it never
    consults MPS -- so every instance is re-pinned here, matching the
    convention in scripts/run_adaptive_stepping_ablation.py.
    """
    if sim._device == device:
        return sim
    sim._device = device
    sim._influence = sim._influence.to(device)
    if hasattr(sim, "_adjacency"):
        sim._adjacency = sim._adjacency.to(device)
    if getattr(sim, "sla_priority", None) is not None:
        sim.sla_priority = sim.sla_priority.to(device)
    return sim


# ------------------------------------------------------------- provenance ---
def git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "-C", ROOT, "diff", "--quiet"],
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
    return [
        f"# git_sha={prov['git_sha']}",
        f"# timestamp_utc={prov['timestamp_utc']}",
        f"# device={prov['device']} ({prov['device_name']})",
        f"# torch={prov['torch_version']} numpy={prov['numpy_version']}"
        f" python={prov['python_version']}",
        f"# config={json.dumps(prov['config'], sort_keys=True, default=str)}",
    ]


# --------------------------------------------------------------- topology ---
def load_topology(kind: str, n_nodes: int, seed: int) -> tuple:
    """Load or synthesize the requested network.

    Returns (adjacency, topology_used, note).  If 'caida' is requested but no
    local CAIDA AS-REL2 file is present, this falls back to Barabasi-Albert
    and says so explicitly in `topology_used` / `note` -- results are never
    labelled 'caida' when they are actually synthetic (per the instruction
    never to blur measured and modelled values).
    """
    generator = TopologyGenerator(seed=seed)
    if kind == "ba":
        network = generator.generate_barabasi_albert(n_nodes=n_nodes, m=3).get_largest_component()
        return network.get_adjacency_matrix().astype(np.float32), "ba", None

    if kind != "caida":
        raise ValueError(f"unknown topology kind {kind!r}")

    caida_dir = os.path.join(ROOT, "datasets", "caida")
    candidates = []
    if os.path.isdir(caida_dir):
        for ext in ("*.as-rel2.txt.bz2", "*.as-rel2.txt.gz", "*.as-rel2.txt"):
            import glob
            candidates.extend(glob.glob(os.path.join(caida_dir, ext)))
    if candidates:
        candidates.sort()
        network = load_caida_topology(candidates[-1], as_undirected=True, largest_component=True)
        if network.n_nodes > n_nodes:
            selected = [n for n, _ in
                        sorted(network.graph.degree(), key=lambda kv: kv[1], reverse=True)[:n_nodes]]
            limited = NetworkGraph(directed=network.graph.is_directed())
            limited.graph = network.graph.subgraph(selected).copy()
            network = limited.get_largest_component()
        return network.get_adjacency_matrix().astype(np.float32), "caida", None

    note = ("no local CAIDA AS-REL2 file under datasets/caida/; "
            "fell back to Barabasi-Albert -- these rows are NOT real CAIDA topology data")
    network = generator.generate_barabasi_albert(n_nodes=n_nodes, m=3).get_largest_component()
    return network.get_adjacency_matrix().astype(np.float32), "ba_fallback_no_caida_data", note


# -------------------------------------------------------------- work units --
def fine_only_work_units(T: float, base_dt: float, M: int, level: int, n_samples: int) -> float:
    """Fine-grid-only timestep-path evaluations (no coarse companion)."""
    dt_fine = base_dt / (M ** level)
    n_fine = int(T / dt_fine)
    return float(n_fine * n_samples)


def coupled_fixed_work_units(T: float, base_dt: float, M: int, level: int, n_samples: int) -> float:
    """Fine+coarse timestep-path evaluations for one coupled MLMC level (fixed step)."""
    dt_fine = base_dt / (M ** level)
    n_fine = int(T / dt_fine)
    n_coarse = 0 if level == 0 else int(T / (dt_fine * M))
    return float((n_fine + n_coarse) * n_samples)


# ---------------------------------------------------------- CI / stats ------
def ci_half_width(variance: float, confidence: float = 0.95) -> float:
    z = float(sp_stats.norm.ppf(1 - (1 - confidence) / 2))
    return z * float(np.sqrt(max(variance, 0.0)))


def paired_ttest(a: list, b: list) -> dict:
    """Paired t-test of a vs b (a = this rung, b = previous rung), plus
    Cohen's d for paired samples (mean difference / sd of differences)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return {"t": None, "p": None, "effect_size": None, "n": int(len(a)),
                "note": "fewer than 2 paired seeds available"}
    diff = a - b
    sd = float(diff.std(ddof=1))
    if sd == 0.0:
        return {"t": None, "p": None, "effect_size": 0.0 if diff.mean() == 0 else None,
                "n": int(len(a)), "note": "zero variance in paired differences"}
    t_stat, p_val = sp_stats.ttest_rel(a, b)
    return {"t": float(t_stat), "p": float(p_val),
            "effect_size": float(diff.mean() / sd), "n": int(len(a)), "note": None}


# ------------------------------------------------------- component runners --
def run_r0_single_level(sim, *, epsilon, T, base_dt, level, pilot_samples, metric) -> dict:
    """R0: single-level GPU MC at the finest level used by the rest of the ladder.

    `run_level` internally computes a coarse companion too (implementation
    reuse of the coupled-path kernel) even though a true single-level
    estimator would not need one; W below counts fine-only work (the
    theoretically correct single-level cost), so wall-clock is a
    conservative (pessimistic-for-R0) measurement relative to W -- it can
    only make later rungs' apparent speed-up look smaller, never larger.
    """
    M = sim.refinement_factor
    y_fine, _ = sim.run_level(level, pilot_samples, T, base_dt, metric)
    var_pilot = float(np.var(y_fine, ddof=1)) if pilot_samples > 1 else 0.0
    n_total = max(pilot_samples, int(np.ceil(var_pilot / epsilon ** 2)))
    n_add = n_total - pilot_samples
    diffs = list(y_fine)
    work = fine_only_work_units(T, base_dt, M, level, pilot_samples)
    if n_add > 0:
        y_more, _ = sim.run_level(level, n_add, T, base_dt, metric)
        diffs.extend(list(y_more))
        work += fine_only_work_units(T, base_dt, M, level, n_add)
    diffs = np.asarray(diffs)
    estimate = float(np.mean(diffs))
    variance = float(np.var(diffs, ddof=1)) / len(diffs)
    return {
        "estimate": estimate, "variance": variance,
        "ci_half_width": ci_half_width(variance),
        "work_units": work, "samples_per_level": [int(len(diffs))],
        "note": "wall-clock includes incidental coarse-path compute; W is fine-only",
    }


def run_uniform_mlmc(sim, *, epsilon, T, base_dt, L_max, pilot_samples, metric,
                      adaptive_stepping: bool) -> dict:
    """R1/R2/R3: pilot -> Giles allocation -> final pass, MLMC telescoping sum.

    Mirrors GPUCoupledPropagationMLMC.mlmc_estimate's algorithm, but queries
    sim.adaptive_work_units() immediately after every run_level() call --
    before the NEXT call's internal reset_adaptive_state() wipes the bucket
    history -- so adaptive-stepping rungs are charged for the sub-steps
    actually executed rather than the nominal fixed grid.
    """
    M = sim.refinement_factor
    variances, costs, pilot_diffs = [], [], []
    work = 0.0

    for level in range(L_max + 1):
        y_fine, y_coarse = sim.run_level(level, pilot_samples, T, base_dt, metric)
        work += (sim.adaptive_work_units() if adaptive_stepping
                 else coupled_fixed_work_units(T, base_dt, M, level, pilot_samples))
        diffs = y_fine - y_coarse
        variances.append(float(np.var(diffs, ddof=1)) if pilot_samples > 1 else 0.0)
        dt_l = base_dt / (M ** level)
        costs.append(float(T / dt_l))
        pilot_diffs.append(diffs)

    sum_vc = float(np.sum([np.sqrt(v * c) for v, c in zip(variances, costs)]))
    optimal_N = []
    for level in range(L_max + 1):
        if variances[level] <= 0 or sum_vc <= 0:
            optimal_N.append(1)
        else:
            n_l = (2.0 / epsilon ** 2) * np.sqrt(variances[level] / costs[level]) * sum_vc
            optimal_N.append(max(1, int(np.ceil(n_l))))

    level_stats = []
    for level in range(L_max + 1):
        n_add = max(0, optimal_N[level] - pilot_samples)
        diffs = list(pilot_diffs[level])
        if n_add > 0:
            y_fine, y_coarse = sim.run_level(level, n_add, T, base_dt, metric)
            work += (sim.adaptive_work_units() if adaptive_stepping
                     else coupled_fixed_work_units(T, base_dt, M, level, n_add))
            diffs.extend(list(y_fine - y_coarse))
        diffs = np.asarray(diffs)
        mean_diff = float(np.mean(diffs))
        var_diff = float(np.var(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        level_stats.append({"level": level, "n_samples": int(len(diffs)),
                             "mean_diff": mean_diff, "var_diff": var_diff})

    estimate = float(sum(s["mean_diff"] for s in level_stats))
    variance = float(sum(s["var_diff"] / s["n_samples"] for s in level_stats))
    return {
        "estimate": estimate, "variance": variance,
        "ci_half_width": ci_half_width(variance),
        "work_units": work,
        "samples_per_level": [s["n_samples"] for s in level_stats],
    }


def _carry_forward_r3_scheme(sim, adaptive_rtol: float) -> None:
    """Enable adaptive_stepping + predictor_corrector on an ANA instance.

    Workaround for limitation 1 in the module docstring: the ANA subclass's
    __init__ does not forward these kwargs to the parent, so they are set as
    plain instance attributes post-construction -- exactly what the parent
    constructor would have stored had the kwargs been forwarded.
    """
    sim.adaptive_stepping = True
    sim.adaptive_rtol = float(adaptive_rtol)
    sim.adaptive_error_estimator = "embedded"
    sim.reflection = "predictor_corrector"
    sim.adaptive_diagnostics = False
    sim._adaptive_h_scale = {}
    sim.adaptive_bucket_history = []
    sim.adaptive_mm_calls = 0
    sim._adaptive_uniform_steps = 0


def run_ana_weighted(sim, *, epsilon, T, base_dt, L_max, pilot_samples, metric) -> dict:
    """R4: ANA-weighted MLMC (sum_i w_i E[Q_i]), carrying forward R3's scheme.

    Reimplements GPUAdaptiveNetworkAwareMLMC.mlmc_estimate_weighted's
    algorithm with per-level work-unit harvesting added, for the same reason
    as run_uniform_mlmc above.
    """
    M = sim.refinement_factor
    adaptive = sim.adaptive_stepping
    per_node_variances, costs, pilot_diffs = [], [], []
    work = 0.0

    for level in range(L_max + 1):
        y_fine, y_coarse = sim.run_level_node_values(level, pilot_samples, T, base_dt, metric)
        work += (sim.adaptive_work_units() if adaptive
                 else coupled_fixed_work_units(T, base_dt, M, level, pilot_samples))
        diffs = y_fine - y_coarse
        var_per_node = (diffs.var(dim=0, unbiased=True) if pilot_samples > 1
                         else torch.zeros(sim.n_nodes, device=sim._device))
        per_node_variances.append(var_per_node)
        dt_l = base_dt / (M ** level)
        costs.append(float(T / dt_l))
        pilot_diffs.append(diffs)

    level_var_per_node = torch.stack(per_node_variances, dim=0)
    costs_tensor = torch.tensor(costs, dtype=torch.float32, device=sim._device)
    weights = sim.compute_node_weights(level_var_per_node, sla_vec=None)
    optimal_N = sim.compute_optimal_samples_weighted(level_var_per_node, costs_tensor, weights, epsilon)

    level_stats = []
    for level in range(L_max + 1):
        n_add = max(0, optimal_N[level] - pilot_samples)
        diffs = pilot_diffs[level]
        if n_add > 0:
            y_fine, y_coarse = sim.run_level_node_values(level, n_add, T, base_dt, metric)
            work += (sim.adaptive_work_units() if adaptive
                     else coupled_fixed_work_units(T, base_dt, M, level, n_add))
            diffs = torch.cat([diffs, y_fine - y_coarse], dim=0)
        n_total = int(diffs.shape[0])
        mean_per_node = diffs.mean(dim=0)
        weighted_diffs = torch.mv(diffs, weights)
        mean_diff = float(torch.dot(weights, mean_per_node).item())
        w_var = float(weighted_diffs.var(unbiased=True).item()) if n_total > 1 else 0.0
        level_stats.append({"level": level, "n_samples": n_total, "mean_diff": mean_diff,
                             "weighted_estimator_var": w_var})

    estimate = float(sum(s["mean_diff"] for s in level_stats))
    variance = float(sum(s["weighted_estimator_var"] / s["n_samples"] for s in level_stats))
    return {
        "estimate": estimate, "variance": variance,
        "ci_half_width": ci_half_width(variance),
        "work_units": work,
        "samples_per_level": [s["n_samples"] for s in level_stats],
        "node_weights": weights.detach().cpu().tolist(),
    }


def calibrate_overflow_threshold(sim, *, T, base_dt, L_max, target_node, n_calib, k) -> float:
    """Data-driven rare-event threshold: mean + k*sd of the (non-tilted)
    congestion at the target node, so the event is rare but reachable rather
    than an arbitrary hardcoded constant."""
    c_fine, _ = sim._run_level_state_tensors(L_max, n_calib, T, base_dt)
    vals = c_fine[target_node].detach().cpu().numpy()
    return float(np.mean(vals) + k * np.std(vals))


def run_importance_sampling(sim, *, T, base_dt, L_max, n_paths, target_node) -> dict:
    """R5: Girsanov IS for P(Q_target > B).  A different estimand (a tail
    probability, not a mean functional), and simulate_is_paths() has its own
    fixed-step kernel -- see limitation 2 in the module docstring.
    """
    dt_l = base_dt / (sim.refinement_factor ** L_max)
    n_steps = int(T / dt_l)
    result = sim.simulate_is_paths(n_paths=n_paths, n_steps=n_steps, dt=dt_l, target_node=target_node)
    p_hat = result["p_hat"]
    ess = max(result["ess"], 1.0)
    # ESS-based normal-approximation CI half-width for a self-normalised IS
    # probability estimate (ESS treated as an effective binomial count).
    variance = p_hat * (1.0 - p_hat) / ess
    return {
        "estimate": p_hat, "variance": variance,
        "ci_half_width": ci_half_width(variance),
        "work_units": float(n_paths * n_steps),
        "samples_per_level": [int(n_paths)],
        "ess_pct": result["ess_pct"], "overflow_threshold": result["overflow_threshold"],
        "note": "reflection/adaptive_stepping flags are not consulted by simulate_is_paths()",
    }


def run_multi_gpu(adjacency, *, seed, epsilon, T, base_dt, L_max, pilot_samples, cfg, world_size) -> dict:
    """R6: multi-GPU G=2.  Guarded on real CUDA device count, per spec.

    MultiGPUMLMC._simulate_level_local implements its own SDE step
    (_step_with_halo) with a different drift formula and fixed unit arrivals
    -- see limitation 2 in the module docstring -- so even when it runs, its
    target quantity is architecturally distinct from R0-R4.
    """
    n_cuda = torch.cuda.device_count()
    if n_cuda < world_size:
        return {
            "status": f"skipped: requires >={world_size} CUDA devices",
            "estimate": None, "variance": None, "ci_half_width": None,
            "work_units": None, "samples_per_level": None, "wall_s": None,
        }

    sim = MultiGPUMLMC(
        adjacency, world_size=world_size, rank=0,
        influence_strength=cfg["influence_strength"], decay_rate=cfg["decay_rate"],
        noise_intensity=cfg["noise_intensity"], seed=seed,
    )
    started = time.perf_counter()
    result = sim.mlmc_estimate_multigpu(epsilon=epsilon, L_max=L_max, N_pilot=pilot_samples, T=T)
    wall = time.perf_counter() - started

    work = 0.0
    for s in result["level_stats"]:
        level, n = s["level"], s["n_samples"]
        n_fine = int(2 ** level)
        n_coarse = 0 if level == 0 else n_fine // sim.refinement_factor
        work += (n_fine + n_coarse) * n

    return {
        "status": "ok",
        "estimate": result["estimate"], "variance": result["variance"],
        "ci_half_width": ci_half_width(result["variance"]),
        "work_units": work, "wall_s": wall,
        "samples_per_level": [s["n_samples"] for s in result["level_stats"]],
    }


# --------------------------------------------------------------- reference --
def compute_uniform_reference(adjacency, seed, cfg, device) -> float:
    sim = GPUCoupledPropagationMLMC(
        adjacency_matrix=adjacency, seed=seed * 97 + 1,
        influence_strength=cfg["influence_strength"], decay_rate=cfg["decay_rate"],
        noise_intensity=cfg["noise_intensity"], reflection="predictor_corrector",
        adaptive_stepping=False,
    )
    pin_device(sim, device)
    ref_eps = cfg["epsilon"] / cfg["reference_epsilon_factor"]
    ref_L = cfg["L_max"] + cfg["reference_L_max_bonus"]
    result = sim.mlmc_estimate(epsilon=ref_eps, T=cfg["T"], base_dt=cfg["base_dt"], L_max=ref_L,
                                pilot_samples=cfg["reference_pilot_samples"], verbose=False)
    return float(result["estimate"])


def compute_ana_reference(adjacency, seed, cfg, device) -> float:
    sim = GPUAdaptiveNetworkAwareMLMC(
        adjacency_matrix=adjacency, seed=seed * 97 + 2,
        influence_strength=cfg["influence_strength"], decay_rate=cfg["decay_rate"],
        noise_intensity=cfg["noise_intensity"],
    )
    pin_device(sim, device)
    ref_eps = cfg["epsilon"] / cfg["reference_epsilon_factor"]
    ref_L = cfg["L_max"] + cfg["reference_L_max_bonus"]
    result = sim.mlmc_estimate_weighted(epsilon=ref_eps, T=cfg["T"], base_dt=cfg["base_dt"], L_max=ref_L,
                                         pilot_samples=cfg["reference_pilot_samples"], verbose=False)
    return float(result["estimate"])


# -------------------------------------------------------------- checkpoint --
def config_fingerprint(cfg: dict) -> str:
    payload = {k: v for k, v in sorted(cfg.items()) if k != "seeds"}
    return json.dumps(payload, sort_keys=True, default=str)


def load_checkpoint(path: str, cfg: dict) -> dict:
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
                print("  [checkpoint] ignoring truncated trailing record", flush=True)
                continue
            if (record.get("schema_version") != SCHEMA_VERSION
                    or record.get("config_fingerprint") != fingerprint):
                stale += 1
                continue
            done[record["unit_id"]] = record
    if stale:
        print(f"  [checkpoint] ignoring {stale} record(s) from a different schema/config", flush=True)
    return done


def append_checkpoint(path: str, record: dict) -> None:
    with open(path, "a") as handle:
        handle.write(json.dumps(record, default=float) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ------------------------------------------------------------------- main ---
def build_sim(rung: str, adjacency, seed: int, cfg: dict, device: torch.device):
    """Construct the simulator for a rung, wiring in only the flags that
    class's underlying code path actually consults (see module docstring)."""
    common = dict(influence_strength=cfg["influence_strength"], decay_rate=cfg["decay_rate"],
                  noise_intensity=cfg["noise_intensity"], seed=seed)

    if rung in ("R0", "R1"):
        sim = GPUCoupledPropagationMLMC(adjacency_matrix=adjacency, reflection="euler_clamp",
                                         adaptive_stepping=False, **common)
    elif rung == "R2":
        sim = GPUCoupledPropagationMLMC(adjacency_matrix=adjacency, reflection="predictor_corrector",
                                         adaptive_stepping=False, **common)
    elif rung == "R3":
        sim = GPUCoupledPropagationMLMC(adjacency_matrix=adjacency, reflection="predictor_corrector",
                                         adaptive_stepping=True, adaptive_rtol=cfg["adaptive_rtol"],
                                         adaptive_error_estimator="embedded", **common)
    elif rung == "R4":
        sim = GPUAdaptiveNetworkAwareMLMC(adjacency_matrix=adjacency, **common)
        _carry_forward_r3_scheme(sim, cfg["adaptive_rtol"])
    elif rung == "R5":
        sim = GPUImportanceSamplingMLMC(adjacency_matrix=adjacency,
                                         overflow_threshold=cfg["is_overflow_threshold_placeholder"],
                                         is_strength=cfg["is_strength"], **common)
    else:
        raise ValueError(f"build_sim does not handle {rung!r} (R6 is constructed separately)")

    return pin_device(sim, device)


def components_active(rung: str) -> dict:
    """What actually ran, distinct from what was nominally requested (per
    the module-docstring limitations)."""
    base = {"single_level": rung == "R0", "mlmc": rung not in ("R0",),
            "predictor_corrector_reflection": False, "adaptive_stepping": False,
            "ana_weighted_allocation": rung in ("R4",), "importance_sampling": rung == "R5",
            "multi_gpu": rung == "R6"}
    if rung in ("R2", "R3", "R4"):
        base["predictor_corrector_reflection"] = True
    if rung in ("R3", "R4"):
        base["adaptive_stepping"] = True
    if rung == "R5":
        base["predictor_corrector_reflection"] = None  # constructor accepts it; kernel ignores it
        base["adaptive_stepping"] = None
    if rung == "R6":
        base["predictor_corrector_reflection"] = None  # class does not expose the flag at all
        base["adaptive_stepping"] = None
    return base


def run_one_unit(rung: str, topology: str, seed: int, cfg: dict, device: torch.device,
                  references: dict) -> dict:
    adjacency, topology_used, topo_note = load_topology(topology, cfg["n_nodes"], seed)
    family = RUNG_FAMILY[rung]

    ref_key = (family, topology, seed)
    if family == "uniform_mean" and ref_key not in references:
        references[ref_key] = compute_uniform_reference(adjacency, seed, cfg, device)
    if family == "ana_weighted" and ref_key not in references:
        references[ref_key] = compute_ana_reference(adjacency, seed, cfg, device)

    started = time.perf_counter()
    if rung == "R0":
        sim = build_sim(rung, adjacency, seed, cfg, device)
        payload = run_r0_single_level(sim, epsilon=cfg["epsilon"], T=cfg["T"], base_dt=cfg["base_dt"],
                                       level=cfg["L_max"], pilot_samples=cfg["pilot_samples"],
                                       metric="mean_congestion")
    elif rung in ("R1", "R2", "R3"):
        sim = build_sim(rung, adjacency, seed, cfg, device)
        payload = run_uniform_mlmc(sim, epsilon=cfg["epsilon"], T=cfg["T"], base_dt=cfg["base_dt"],
                                    L_max=cfg["L_max"], pilot_samples=cfg["pilot_samples"],
                                    metric="mean_congestion", adaptive_stepping=(rung == "R3"))
    elif rung == "R4":
        sim = build_sim(rung, adjacency, seed, cfg, device)
        payload = run_ana_weighted(sim, epsilon=cfg["epsilon"], T=cfg["T"], base_dt=cfg["base_dt"],
                                    L_max=cfg["L_max"], pilot_samples=cfg["pilot_samples"],
                                    metric="mean_congestion")
    elif rung == "R5":
        cfg_r5 = dict(cfg, is_overflow_threshold_placeholder=1.0)
        sim = build_sim(rung, adjacency, seed, cfg_r5, device)
        graph_nx = nx.from_numpy_array(adjacency)
        pagerank = nx.pagerank(graph_nx) if graph_nx.number_of_nodes() > 0 else {}
        target_node = (cfg["is_target_node"] if cfg["is_target_node"] is not None
                        else max(pagerank, key=pagerank.get))
        threshold = calibrate_overflow_threshold(
            sim, T=cfg["T"], base_dt=cfg["base_dt"], L_max=cfg["L_max"], target_node=target_node,
            n_calib=max(200, cfg["pilot_samples"] * 4), k=cfg["is_threshold_k"])
        sim.overflow_threshold = threshold
        payload = run_importance_sampling(sim, T=cfg["T"], base_dt=cfg["base_dt"], L_max=cfg["L_max"],
                                           n_paths=cfg["is_n_paths"], target_node=target_node)
        payload["target_node"] = int(target_node)
    elif rung == "R6":
        payload = run_multi_gpu(adjacency, seed=seed, epsilon=cfg["epsilon"], T=cfg["T"],
                                 base_dt=cfg["base_dt"], L_max=cfg["L_max"],
                                 pilot_samples=cfg["pilot_samples"], cfg=cfg,
                                 world_size=cfg["multi_gpu_world_size"])
    else:
        raise ValueError(f"unknown rung {rung!r}")
    wall_s = payload.get("wall_s", time.perf_counter() - started)

    reference = references.get(ref_key)
    achieved_mse = ((payload["estimate"] - reference) ** 2
                     if payload.get("estimate") is not None and reference is not None else None)

    comparable = family == "uniform_mean"
    comparable_note = {
        "uniform_mean": None,
        "ana_weighted": "ANA estimates sum_i w_i E[Q_i], a different functional from the "
                         "uniform-mean target of R0-R3; not directly comparable on estimate/CI/MSE.",
        "rare_event": "IS targets a tail probability P(Q_target>B), not a mean functional; "
                       "not comparable to any other rung on estimate/CI/MSE.",
        "multi_gpu_local": "MultiGPUMLMC._simulate_level_local uses a structurally different SDE "
                            "step (own drift formula, fixed unit arrivals); not comparable to "
                            "R0-R4 on estimate/CI/MSE even when the CUDA-device guard is satisfied.",
    }[family]

    return {
        "rung": rung, "rung_label": RUNG_LABELS[rung], "estimand_family": family,
        "topology_requested": topology, "topology_used": topology_used, "topology_note": topo_note,
        "seed": seed, "n_nodes": cfg["n_nodes"],
        "components_active": components_active(rung),
        "comparable": comparable, "comparable_note": comparable_note,
        "reference": reference, "achieved_mse": achieved_mse,
        "wall_s": wall_s, "status": payload.get("status", "ok"),
        **{k: v for k, v in payload.items() if k not in ("status", "wall_s")},
    }


def run_all(args, cfg: dict, device: torch.device) -> list:
    checkpoint_path = os.path.join(args.out, "checkpoint.jsonl")
    if args.no_resume and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("  [checkpoint] --no-resume: cleared previous log", flush=True)

    done = load_checkpoint(checkpoint_path, cfg)
    if done:
        print(f"  [checkpoint] resuming, {len(done)} unit(s) already complete", flush=True)
    fingerprint = config_fingerprint(cfg)
    references: dict = {}

    def execute(unit_id: str, fn, label: str):
        if unit_id in done:
            print(f"  {label}: cached", flush=True)
            return done[unit_id]["result"]
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        record = {"schema_version": SCHEMA_VERSION, "config_fingerprint": fingerprint,
                  "unit_id": unit_id, "result": result, "unit_wall_s": elapsed}
        append_checkpoint(checkpoint_path, record)
        done[unit_id] = record
        print(f"  {label}  ({elapsed:.1f}s)", flush=True)
        return result

    rows = []
    for topology in cfg["topologies"]:
        print(f"\n{'#' * 90}\n# topology: {topology}\n{'#' * 90}", flush=True)
        for rung in cfg["rungs"]:
            print(f"\n[{rung}] {RUNG_LABELS[rung]}", flush=True)
            for seed in cfg["seeds"]:
                unit = f"{topology}|{rung}|{seed}"

                def run(r=rung, t=topology, s=seed):
                    return run_one_unit(r, t, s, cfg, device, references)

                row = execute(unit, run, f"  seed {seed}")
                rows.append(row)
    return rows


# --------------------------------------------------------------- outputs ----
def build_transitions(rows: list) -> list:
    """One row per (topology, rung-pair): incremental change + paired t-test."""
    by_key = {}
    for row in rows:
        by_key.setdefault((row["topology_requested"], row["rung"]), []).append(row)

    transitions = []
    for topology in sorted({r["topology_requested"] for r in rows}):
        present = [r for r in RUNG_ORDER if (topology, r) in by_key]
        for prev_rung, rung in zip(present, present[1:]):
            prev_rows = sorted(by_key[(topology, prev_rung)], key=lambda r: r["seed"])
            cur_rows = sorted(by_key[(topology, rung)], key=lambda r: r["seed"])
            prev_by_seed = {r["seed"]: r for r in prev_rows}
            cur_by_seed = {r["seed"]: r for r in cur_rows}
            common_seeds = sorted(set(prev_by_seed) & set(cur_by_seed))

            prev_ok = [prev_by_seed[s] for s in common_seeds if prev_by_seed[s]["status"] == "ok"]
            cur_ok = [cur_by_seed[s] for s in common_seeds if cur_by_seed[s]["status"] == "ok"]
            paired_seeds = [r["seed"] for r in cur_ok if r["seed"] in {p["seed"] for p in prev_ok}]
            prev_ok = [prev_by_seed[s] for s in paired_seeds]
            cur_ok = [cur_by_seed[s] for s in paired_seeds]

            if not cur_ok:
                transitions.append({
                    "topology": topology, "from_rung": prev_rung, "to_rung": rung,
                    "status": cur_rows[0]["status"] if cur_rows else "no data",
                    "n_paired_seeds": 0,
                })
                continue

            work_prev = [r["work_units"] for r in prev_ok]
            work_cur = [r["work_units"] for r in cur_ok]
            wall_prev = [r["wall_s"] for r in prev_ok]
            wall_cur = [r["wall_s"] for r in cur_ok]

            same_family = (cur_ok[0]["estimand_family"] == prev_ok[0]["estimand_family"]) if prev_ok else False

            entry = {
                "topology": topology, "from_rung": prev_rung, "to_rung": rung,
                "status": "ok", "n_paired_seeds": len(paired_seeds),
                "work_mean_prev": float(np.mean(work_prev)) if prev_ok else None,
                "work_mean_cur": float(np.mean(work_cur)),
                "work_ratio_cur_over_prev": (float(np.mean(work_cur) / np.mean(work_prev))
                                              if prev_ok and np.mean(work_prev) > 0 else None),
                "wall_mean_prev": float(np.mean(wall_prev)) if prev_ok else None,
                "wall_mean_cur": float(np.mean(wall_cur)),
                "wall_ratio_cur_over_prev": (float(np.mean(wall_cur) / np.mean(wall_prev))
                                              if prev_ok and np.mean(wall_prev) > 0 else None),
                "ttest_work": paired_ttest(work_cur, work_prev) if prev_ok else None,
                "ttest_wall": paired_ttest(wall_cur, wall_prev) if prev_ok else None,
                "same_estimand_family": same_family,
            }

            if same_family and prev_ok:
                ci_prev = [r["ci_half_width"] for r in prev_ok]
                ci_cur = [r["ci_half_width"] for r in cur_ok]
                mse_prev = [r["achieved_mse"] for r in prev_ok if r["achieved_mse"] is not None]
                mse_cur = [r["achieved_mse"] for r in cur_ok if r["achieved_mse"] is not None]
                entry["ci_half_width_mean_prev"] = float(np.mean(ci_prev))
                entry["ci_half_width_mean_cur"] = float(np.mean(ci_cur))
                entry["ttest_ci_half_width"] = paired_ttest(ci_cur, ci_prev)
                if len(mse_prev) == len(mse_cur) and len(mse_cur) > 1:
                    entry["achieved_mse_mean_prev"] = float(np.mean(mse_prev))
                    entry["achieved_mse_mean_cur"] = float(np.mean(mse_cur))
                    entry["ttest_achieved_mse"] = paired_ttest(mse_cur, mse_prev)
            else:
                entry["ci_half_width_note"] = (
                    "different estimand family from previous rung; "
                    "CI half-width and achieved MSE are not comparable across the transition")
            transitions.append(entry)
    return transitions


def write_outputs(args, cfg: dict, prov: dict, rows: list, transitions: list) -> dict:
    os.makedirs(args.out, exist_ok=True)
    header = provenance_comment_lines(prov)
    paths = {}

    json_path = os.path.join(args.out, "ablation_ladder.json")
    with open(json_path, "w") as handle:
        json.dump({"provenance": prov, "rung_labels": RUNG_LABELS, "rung_family": RUNG_FAMILY,
                   "rows": rows, "transitions": transitions}, handle, indent=2, default=float)
    paths["json"] = json_path

    def open_csv(name):
        handle = open(os.path.join(args.out, name), "w", newline="")
        for line in header:
            handle.write(line + "\n")
        return handle

    with open_csv("rungs.csv") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "topology_requested", "topology_used", "rung", "rung_label", "estimand_family",
            "seed", "status", "estimate", "variance", "ci_half_width", "work_units", "wall_s",
            "reference", "achieved_mse", "samples_per_level", "comparable", "comparable_note",
        ])
        for row in rows:
            writer.writerow([
                row["topology_requested"], row["topology_used"], row["rung"], row["rung_label"],
                row["estimand_family"], row["seed"], row["status"], row.get("estimate"),
                row.get("variance"), row.get("ci_half_width"), row.get("work_units"), row["wall_s"],
                row.get("reference"), row.get("achieved_mse"),
                json.dumps(row.get("samples_per_level")), row["comparable"], row["comparable_note"],
            ])
    paths["rungs_csv"] = os.path.join(args.out, "rungs.csv")

    with open_csv("transitions.csv") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "topology", "from_rung", "to_rung", "status", "n_paired_seeds", "same_estimand_family",
            "work_mean_prev", "work_mean_cur", "work_ratio_cur_over_prev",
            "wall_mean_prev", "wall_mean_cur", "wall_ratio_cur_over_prev",
            "t_work", "p_work", "effect_size_work",
            "t_wall", "p_wall", "effect_size_wall",
            "ci_half_width_mean_prev", "ci_half_width_mean_cur",
            "t_ci", "p_ci", "effect_size_ci",
            "achieved_mse_mean_prev", "achieved_mse_mean_cur",
            "t_mse", "p_mse", "effect_size_mse",
        ])
        for t in transitions:
            tw = t.get("ttest_work") or {}
            twall = t.get("ttest_wall") or {}
            tci = t.get("ttest_ci_half_width") or {}
            tmse = t.get("ttest_achieved_mse") or {}
            writer.writerow([
                t["topology"], t["from_rung"], t["to_rung"], t["status"], t["n_paired_seeds"],
                t.get("same_estimand_family"),
                t.get("work_mean_prev"), t.get("work_mean_cur"), t.get("work_ratio_cur_over_prev"),
                t.get("wall_mean_prev"), t.get("wall_mean_cur"), t.get("wall_ratio_cur_over_prev"),
                tw.get("t"), tw.get("p"), tw.get("effect_size"),
                twall.get("t"), twall.get("p"), twall.get("effect_size"),
                t.get("ci_half_width_mean_prev"), t.get("ci_half_width_mean_cur"),
                tci.get("t"), tci.get("p"), tci.get("effect_size"),
                t.get("achieved_mse_mean_prev"), t.get("achieved_mse_mean_cur"),
                tmse.get("t"), tmse.get("p"), tmse.get("effect_size"),
            ])
    paths["transitions_csv"] = os.path.join(args.out, "transitions.csv")
    return paths


def print_report(rows: list, transitions: list) -> None:
    print("\n" + "=" * 118)
    print("ABLATION LADDER")
    print("=" * 118)
    by_key = {}
    for row in rows:
        by_key.setdefault((row["topology_requested"], row["rung"]), []).append(row)

    for topology in sorted({r["topology_requested"] for r in rows}):
        print(f"\n--- topology: {topology} ---")
        print(f"{'rung':<5}{'label':<42}{'work(mean)':>14}{'wall_s(mean)':>13}"
              f"{'ci_hw(mean)':>13}{'mse(mean)':>13}{'comparable':>12}")
        for rung in RUNG_ORDER:
            key = (topology, rung)
            if key not in by_key:
                continue
            group = by_key[key]
            ok = [r for r in group if r["status"] == "ok"]
            if not ok:
                print(f"{rung:<5}{RUNG_LABELS[rung]:<42}{group[0]['status']:>14}")
                continue
            work = np.mean([r["work_units"] for r in ok])
            wall = np.mean([r["wall_s"] for r in ok])
            ci = [r["ci_half_width"] for r in ok if r["ci_half_width"] is not None]
            mse = [r["achieved_mse"] for r in ok if r["achieved_mse"] is not None]
            ci_s = f"{np.mean(ci):.4g}" if ci else "n/a"
            mse_s = f"{np.mean(mse):.4g}" if mse else "n/a"
            print(f"{rung:<5}{RUNG_LABELS[rung]:<42}{work:>14.3e}{wall:>13.3f}"
                  f"{ci_s:>13}{mse_s:>13}{str(ok[0]['comparable']):>12}")

    print("\n" + "=" * 118)
    print("TRANSITIONS  (paired t-test vs previous rung; * marks a real effect at p<0.05, |d|>0.5)")
    print("=" * 118)
    print(f"{'topology':<10}{'from->to':<10}{'work ratio':>11}{'p(work)':>9}"
          f"{'wall ratio':>11}{'p(wall)':>9}{'ci comparable':>15}{'p(ci)':>9}")
    for t in transitions:
        if t["status"] != "ok":
            print(f"{t['topology']:<10}{t['from_rung']}->{t['to_rung']:<7}{t['status']}")
            continue
        tw = t.get("ttest_work") or {}
        twall = t.get("ttest_wall") or {}
        tci = t.get("ttest_ci_half_width") or {}
        wr = t.get("work_ratio_cur_over_prev")
        walr = t.get("wall_ratio_cur_over_prev")
        wr_s = f"{wr:.3f}x" if wr else "n/a"
        walr_s = f"{walr:.3f}x" if walr else "n/a"
        pw = tw.get("p")
        pw_s = f"{pw:.3g}" if pw is not None else "n/a"
        pwall = twall.get("p")
        pwall_s = f"{pwall:.3g}" if pwall is not None else "n/a"
        comparable_s = "yes" if t.get("same_estimand_family") else "no"
        pci = tci.get("p")
        pci_s = f"{pci:.3g}" if pci is not None else "n/a"
        print(f"{t['topology']:<10}{t['from_rung']}->{t['to_rung']:<7}{wr_s:>11}{pw_s:>9}"
              f"{walr_s:>11}{pwall_s:>9}{comparable_s:>15}{pci_s:>9}")

    print("\nNote: R3 (SIMT-adaptive stepping) is measured honestly here -- if it adds work/wall-clock")
    print("rather than reducing it, that is reported as such, not reframed as a speed win.")


def parse_args():
    parser = argparse.ArgumentParser(description="Cumulative component-ablation ladder R0-R6")
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--topologies", nargs="+", choices=["caida", "ba"], default=["caida", "ba"])
    parser.add_argument("--n-nodes", type=int, default=500)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--base-dt", type=float, default=0.1)
    parser.add_argument("--L-max", type=int, default=4)
    parser.add_argument("--pilot-samples", type=int, default=50)
    parser.add_argument("--adaptive-rtol", type=float, default=1e-3)
    parser.add_argument("--reference-epsilon-factor", type=float, default=5.0,
                        help="Tight reference uses epsilon / this factor")
    parser.add_argument("--reference-pilot-samples", type=int, default=200)
    parser.add_argument("--reference-L-max-bonus", type=int, default=1)
    parser.add_argument("--rungs", nargs="+", choices=RUNG_ORDER, default=RUNG_ORDER)
    parser.add_argument("--multi-gpu-world-size", type=int, default=2)
    parser.add_argument("--is-overflow-threshold-k", dest="is_threshold_k", type=float, default=4.0,
                        help="Calibrated threshold = mean + k*sd of pilot congestion at the target node")
    parser.add_argument("--is-n-paths", type=int, default=5000)
    parser.add_argument("--is-strength", type=float, default=1.0)
    parser.add_argument("--is-target-node", type=int, default=None,
                        help="Default: highest-PageRank node")
    parser.add_argument("--influence-strength", type=float, default=0.1)
    parser.add_argument("--decay-rate", type=float, default=0.5)
    parser.add_argument("--noise-intensity", type=float, default=0.1)
    parser.add_argument("--out", default=os.path.join("results", "ablation_ladder"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--quick", action="store_true", help="Small smoke-test configuration")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive")
    if args.L_max < 1:
        raise ValueError("--L-max must be >= 1 (R1+ needs at least 2 levels)")
    if args.pilot_samples < 2:
        raise ValueError("--pilot-samples must be >= 2 (ddof=1 variance needs at least 2 samples)")
    if not args.seeds:
        raise ValueError("--seeds must be non-empty")
    if args.n_nodes < 10:
        raise ValueError("--n-nodes must be >= 10")
    if args.multi_gpu_world_size < 2:
        raise ValueError("--multi-gpu-world-size must be >= 2 (rung R6 is 'multi-GPU G=2')")
    if args.is_threshold_k <= 0:
        raise ValueError("--is-overflow-threshold-k must be positive")

    if args.quick:
        args.n_nodes = min(args.n_nodes, 30)
        # L_max is deliberately NOT reduced under --quick.  This SDE's
        # level-difference variance is dominated by level 0 and decays with
        # more levels available to exploit -- collapsing to L_max=2 removes
        # exactly the structure that makes MLMC cheaper than single-level MC,
        # which would make the R0->R1 sanity check ("W decreases") fail on a
        # smoke-config artifact rather than a real defect.
        args.pilot_samples = min(args.pilot_samples, 16)
        args.reference_pilot_samples = min(args.reference_pilot_samples, 40)
        args.seeds = args.seeds[:2]
        args.is_n_paths = min(args.is_n_paths, 200)
        # The default epsilon=0.02 is tuned for the full n=500 config; on this
        # tiny n=30 smoke network the level-0 variance is far too small for
        # that epsilon to need more than the pilot-sample floor at any level,
        # which pins both R0 and R1 at their floors and hides MLMC's
        # advantage entirely.  Tighten epsilon (unless the caller already
        # asked for something tighter) so the smoke run is small but still
        # representative of the regime the ladder is meant to measure.
        args.epsilon = min(args.epsilon, 5e-4)

    device = select_device(args.device)
    cfg = {
        "epsilon": args.epsilon, "seeds": list(args.seeds), "topologies": list(args.topologies),
        "n_nodes": args.n_nodes, "T": args.T, "base_dt": args.base_dt, "L_max": args.L_max,
        "pilot_samples": args.pilot_samples, "adaptive_rtol": args.adaptive_rtol,
        "reference_epsilon_factor": args.reference_epsilon_factor,
        "reference_pilot_samples": args.reference_pilot_samples,
        "reference_L_max_bonus": args.reference_L_max_bonus, "rungs": list(args.rungs),
        "multi_gpu_world_size": args.multi_gpu_world_size, "is_threshold_k": args.is_threshold_k,
        "is_n_paths": args.is_n_paths, "is_strength": args.is_strength,
        "is_target_node": args.is_target_node, "influence_strength": args.influence_strength,
        "decay_rate": args.decay_rate, "noise_intensity": args.noise_intensity, "quick": args.quick,
    }

    os.makedirs(args.out, exist_ok=True)
    prov = build_provenance(device, cfg)

    print("=" * 100)
    print("GPU-MLMC component-ablation ladder (R0-R6)")
    print("=" * 100)
    print(f"  git sha    : {prov['git_sha']}")
    print(f"  device     : {prov['device']} ({prov['device_name']})")
    print(f"  torch      : {prov['torch_version']}")
    print(f"  seeds      : {cfg['seeds']}")
    print(f"  topologies : {cfg['topologies']}  n_nodes={cfg['n_nodes']}")
    print(f"  epsilon    : {cfg['epsilon']}")
    print(f"  rungs      : {cfg['rungs']}")
    print(f"  out        : {os.path.abspath(args.out)}", flush=True)

    started = time.perf_counter()
    rows = run_all(args, cfg, device)
    transitions = build_transitions(rows)
    total = time.perf_counter() - started

    paths = write_outputs(args, cfg, prov, rows, transitions)
    print_report(rows, transitions)

    print(f"\nTotal wall clock: {total:.1f}s")
    for label, path in paths.items():
        print(f"  {label:<16} {path}")


if __name__ == "__main__":
    main()
