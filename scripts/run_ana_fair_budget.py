"""
Fair fixed-work-budget comparison of ANA-weighted vs standard-Giles sample
allocation (Reviewer 1, concern 6, T2.2).

The problem this fixes.  The existing `run_ana_mlmc_experiment.py` comparison
pits ANA-MLMC (which estimates sum_i w_i E[Q_i], the network-risk-weighted
functional) against standard Giles allocation (which estimates the uniform
mean E[Q]) -- two DIFFERENT quantities, driven to the SAME target epsilon.
ANA can only look worse in that comparison: it is being charged for a harder
estimation problem (a weighted sum with unequal per-node variances) while
being judged against a rule optimised for a different, easier target.

The fair version implemented here:
  1. Fixes a total work budget B (timestep-path evaluations), identical for
     both arms.
  2. Both arms estimate the SAME target: sum_i w_i E[Q_i], using ONE fixed
     ANA weight vector w (estimated once from a shared pilot and then held
     fixed) as the shared definition of "the quantity of interest".
  3. Arm A (ana): allocates N_l proportional to sqrt(V_l^w / C_l), where
     V_l^w = w^T Var_l is the ANA-weighted per-level variance.
  4. Arm B (giles): allocates N_l proportional to sqrt(V_l^u / C_l), where
     V_l^u is the plain unweighted mean of per-node level variance -- the
     allocation rule that ignores node weighting -- but its samples are
     STILL combined with the same fixed w into the same target, so both
     arms are scored on the identical estimation problem.
  5. Compares CI half-width on the shared scalar target, and RMSE on the
     top-k highest-PageRank nodes (k in {10, 25, 50}) against a tight
     per-(topology, seed) reference, swept over several budgets, 10 seeds
     each, mean +/- SD and paired tests (Arm A vs Arm B).

This is the experiment that should show ANA winning, because for the first
time both arms are charged for the same problem.  If it does not win, that
is reported plainly (see the printed verdict and `--out`/summary JSON) --
this script does not search for a configuration that makes ANA win.

Usage:
    python3 scripts/run_ana_fair_budget.py --quick
    python3 scripts/run_ana_fair_budget.py --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --topologies ba caida --n-nodes 500 --device cuda

Output (under --out, default results/ana_fair_budget):
    ana_fair_budget.json     full results + provenance
    arms.csv                 one row per (topology, budget, arm, seed)
    comparisons.csv          one row per (topology, budget): CI + RMSE@k
                              paired t-tests, Arm A vs Arm B
    checkpoint.jsonl         append-only resume log
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

from gpu.parallel_mc import GPUAdaptiveNetworkAwareMLMC  # noqa: E402
from network.topology import NetworkGraph, TopologyGenerator, load_caida_topology  # noqa: E402

SCHEMA_VERSION = 1
WIN_MARGIN = 1.05  # an arm must beat the other by this margin to count as a real win, not noise
DEFAULT_BUDGETS = [2000.0, 5000.0, 10000.0, 20000.0, 50000.0]


# ----------------------------------------------------------------- device ---
def select_device(requested: str) -> torch.device:
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
    """Load or synthesize the requested network; honest CAIDA fallback (see
    run_ablation_ladder.py's load_topology for the same convention)."""
    generator = TopologyGenerator(seed=seed)
    if kind == "ba":
        network = generator.generate_barabasi_albert(n_nodes=n_nodes, m=3).get_largest_component()
        return network.get_adjacency_matrix().astype(np.float32), "ba", None

    if kind != "caida":
        raise ValueError(f"unknown topology kind {kind!r}")

    caida_dir = os.path.join(ROOT, "datasets", "caida")
    candidates = []
    if os.path.isdir(caida_dir):
        import glob
        for ext in ("*.as-rel2.txt.bz2", "*.as-rel2.txt.gz", "*.as-rel2.txt"):
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

    note = ("no local CAIDA AS-REL2 file under datasets/caida/; fell back to Barabasi-Albert -- "
            "these rows are NOT real CAIDA topology data")
    network = generator.generate_barabasi_albert(n_nodes=n_nodes, m=3).get_largest_component()
    return network.get_adjacency_matrix().astype(np.float32), "ba_fallback_no_caida_data", note


# -------------------------------------------------------------- work units --
def coupled_work_units(T: float, base_dt: float, M: int, level: int, n_samples: int) -> float:
    dt_fine = base_dt / (M ** level)
    n_fine = int(T / dt_fine)
    n_coarse = 0 if level == 0 else int(T / (dt_fine * M))
    return float((n_fine + n_coarse) * n_samples)


def level_costs(T: float, base_dt: float, M: int, L_max: int) -> list:
    return [float(T / (base_dt / (M ** level))) for level in range(L_max + 1)]


# ----------------------------------------------------------- allocation -----
def budget_allocation(variances: list, costs: list, budget: float) -> list:
    """Lagrangian-optimal N_l ~ sqrt(V_l/C_l) for a FIXED total-cost budget.

    Derivation: minimise sum_l V_l/N_l subject to sum_l N_l*C_l = B gives
    N_l = kappa*sqrt(V_l/C_l); substituting into the constraint gives
    kappa = B / sum_k sqrt(V_k*C_k), i.e.
        N_l = B * sqrt(V_l/C_l) / sum_k sqrt(V_k*C_k).
    This is the same Lagrangian as Giles's epsilon-driven formula, solved for
    a cost budget instead of a target epsilon -- the two are related by
    eps^2 = 2*(sum_k sqrt(V_k*C_k))^2 / B.
    """
    v = np.maximum(np.asarray(variances, dtype=float), 0.0)
    c = np.asarray(costs, dtype=float)
    sqrt_vc = np.sqrt(v * c)
    denom = float(np.sum(sqrt_vc))
    if denom <= 0.0:
        # No measurable pilot variance anywhere: split the budget evenly by
        # cost across levels rather than guessing a variance-driven skew.
        n_levels = len(v)
        equal_share = budget / n_levels
        n = np.maximum(1.0, np.ceil(equal_share / np.maximum(c, 1e-30)))
        return n.astype(int).tolist()
    raw = np.sqrt(v / np.maximum(c, 1e-30))
    n = budget * raw / denom
    n = np.maximum(1.0, np.ceil(n))
    return n.astype(int).tolist()


# --------------------------------------------------------------- pilot ------
def run_pilot(sim, *, T, base_dt, L_max, pilot_samples, metric):
    """One shared pilot pass: per-node level variances + diffs, reused by
    BOTH arms (this cost is not charged against the budget B; it is the
    same fixed overhead standard MLMC always pays before allocating)."""
    per_node_variances, pilot_diffs = [], []
    for level in range(L_max + 1):
        y_fine, y_coarse = sim.run_level_node_values(level, pilot_samples, T, base_dt, metric)
        diffs = y_fine - y_coarse
        var_per_node = (diffs.var(dim=0, unbiased=True) if pilot_samples > 1
                         else torch.zeros(sim.n_nodes, device=sim._device))
        per_node_variances.append(var_per_node)
        pilot_diffs.append(diffs)
    return torch.stack(per_node_variances, dim=0), pilot_diffs


def uniform_level_variances(level_var_per_node: "torch.Tensor") -> "torch.Tensor":
    """Unweighted per-level variance -- the standard-Giles allocation rule."""
    return level_var_per_node.mean(dim=1)


def weighted_level_variances(level_var_per_node: "torch.Tensor", weights: "torch.Tensor") -> "torch.Tensor":
    """ANA-weighted per-level variance V_l^w = w^T Var_l."""
    return torch.mv(level_var_per_node, weights)


# ---------------------------------------------------------------- arm run ---
def run_arm(sim, *, pilot_diffs, allocation, T, base_dt, L_max, weights, metric):
    """Run one arm at a fixed per-level allocation, combining into the
    SHARED weighted target sum_i w_i E[Q_i].  `allocation[l]` is used
    EXACTLY (truncating the pilot when it is smaller, extending it with
    fresh samples when it is larger), so the realised work matches the
    budget as closely as ceiling rounding allows.
    """
    total_work = 0.0
    per_node_cumulative = torch.zeros(sim.n_nodes, device=sim._device)
    level_weighted_vars, n_used_per_level = [], []

    for level in range(L_max + 1):
        target_n = int(allocation[level])
        diffs = pilot_diffs[level]
        pilot_n = diffs.shape[0]
        if target_n > pilot_n:
            n_add = target_n - pilot_n
            y_fine, y_coarse = sim.run_level_node_values(level, n_add, T, base_dt, metric)
            diffs = torch.cat([diffs, y_fine - y_coarse], dim=0)
        elif target_n < pilot_n:
            diffs = diffs[:target_n]
        n_used = int(diffs.shape[0])

        mean_per_node = diffs.mean(dim=0)
        per_node_cumulative = per_node_cumulative + mean_per_node
        weighted_diffs = torch.mv(diffs, weights)
        w_var = float(weighted_diffs.var(unbiased=True).item()) if n_used > 1 else 0.0
        level_weighted_vars.append(w_var)
        n_used_per_level.append(n_used)
        total_work += coupled_work_units(T, base_dt, sim.refinement_factor, level, n_used)

    estimate = float(torch.dot(weights, per_node_cumulative).item())
    variance = float(sum(v / n for v, n in zip(level_weighted_vars, n_used_per_level) if n > 0))
    z = float(sp_stats.norm.ppf(0.975))
    ci_half_width = z * float(np.sqrt(max(variance, 0.0)))

    return {
        "estimate": estimate, "variance": variance, "ci_half_width": ci_half_width,
        "work_units": total_work, "samples_per_level": n_used_per_level,
        "per_node_estimate": per_node_cumulative.detach().cpu().numpy(),
    }


# --------------------------------------------------------------- reference --
def compute_reference(sim, *, T, base_dt, L_max, reference_samples_per_level, metric):
    """Near-ground-truth per-node estimate: one large fixed-N pass per level,
    ONCE per (topology, seed), reused for every budget in the sweep."""
    per_node_cumulative = torch.zeros(sim.n_nodes, device=sim._device)
    for level in range(L_max + 1):
        y_fine, y_coarse = sim.run_level_node_values(level, reference_samples_per_level, T, base_dt, metric)
        per_node_cumulative = per_node_cumulative + (y_fine - y_coarse).mean(dim=0)
    return per_node_cumulative.detach().cpu().numpy()


def top_k_rmse(estimate: np.ndarray, reference: np.ndarray, node_order: list, k: int) -> float:
    idx = np.asarray(node_order[:k], dtype=int)
    return float(np.sqrt(np.mean((estimate[idx] - reference[idx]) ** 2)))


def rmse_lookup(rmse_topk: dict, k: int) -> float:
    """Look up a top-k RMSE by int key, falling back to the JSON string key.

    A round trip through the checkpoint file (json.dump/json.load) turns
    dict keys into strings, so a resumed run's cached `rmse_topk` has keys
    like "10" instead of 10 even though a freshly-computed one has int keys.
    """
    return rmse_topk[k] if k in rmse_topk else rmse_topk[str(k)]


# ---------------------------------------------------------------- stats -----
def paired_ttest(a: list, b: list) -> dict:
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
    return {"t": float(t_stat), "p": float(p_val), "effect_size": float(diff.mean() / sd),
            "n": int(len(a)), "note": None}


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


# ------------------------------------------------------------------- unit ---
def run_one_seed(topology: str, seed: int, cfg: dict, device: torch.device) -> dict:
    adjacency, topology_used, topo_note = load_topology(topology, cfg["n_nodes"], seed)

    sim = GPUAdaptiveNetworkAwareMLMC(
        adjacency_matrix=adjacency, seed=seed,
        influence_strength=cfg["influence_strength"], decay_rate=cfg["decay_rate"],
        noise_intensity=cfg["noise_intensity"],
    )
    pin_device(sim, device)

    level_var_per_node, pilot_diffs = run_pilot(
        sim, T=cfg["T"], base_dt=cfg["base_dt"], L_max=cfg["L_max"],
        pilot_samples=cfg["pilot_samples"], metric="mean_congestion")
    weights = sim.compute_node_weights(level_var_per_node, sla_vec=None)  # FIXED for both arms

    v_ana = weighted_level_variances(level_var_per_node, weights).detach().cpu().tolist()
    v_giles = uniform_level_variances(level_var_per_node).detach().cpu().tolist()
    costs = level_costs(cfg["T"], cfg["base_dt"], sim.refinement_factor, cfg["L_max"])

    reference = compute_reference(
        sim, T=cfg["T"], base_dt=cfg["base_dt"], L_max=cfg["L_max"],
        reference_samples_per_level=cfg["reference_samples_per_level"], metric="mean_congestion")
    reference_scalar = float(np.dot(weights.detach().cpu().numpy(), reference))

    graph_nx = nx.from_numpy_array(adjacency)
    pagerank = nx.pagerank(graph_nx) if graph_nx.number_of_nodes() > 0 else {}
    node_order = [n for n, _ in sorted(pagerank.items(), key=lambda kv: kv[1], reverse=True)]

    per_budget = []
    for budget in cfg["budgets"]:
        alloc_ana = budget_allocation(v_ana, costs, budget)
        alloc_giles = budget_allocation(v_giles, costs, budget)

        arm_ana = run_arm(sim, pilot_diffs=pilot_diffs, allocation=alloc_ana, T=cfg["T"],
                           base_dt=cfg["base_dt"], L_max=cfg["L_max"], weights=weights,
                           metric="mean_congestion")
        arm_giles = run_arm(sim, pilot_diffs=pilot_diffs, allocation=alloc_giles, T=cfg["T"],
                             base_dt=cfg["base_dt"], L_max=cfg["L_max"], weights=weights,
                             metric="mean_congestion")

        for arm_name, arm in (("ana", arm_ana), ("giles", arm_giles)):
            rmse_k = {k: top_k_rmse(arm["per_node_estimate"], reference, node_order, k)
                      for k in cfg["k_values"]}
            per_budget.append({
                "budget": budget, "arm": arm_name,
                "estimate": arm["estimate"], "reference": reference_scalar,
                "bias": arm["estimate"] - reference_scalar,
                "variance": arm["variance"], "ci_half_width": arm["ci_half_width"],
                "work_units": arm["work_units"], "samples_per_level": arm["samples_per_level"],
                "allocation": alloc_ana if arm_name == "ana" else alloc_giles,
                "rmse_topk": rmse_k,
            })

    return {
        "topology_requested": topology, "topology_used": topology_used, "topology_note": topo_note,
        "seed": seed, "n_nodes": cfg["n_nodes"], "node_weights": weights.detach().cpu().tolist(),
        "level_variance_ana": v_ana, "level_variance_giles": v_giles, "level_costs": costs,
        "per_budget": per_budget,
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

    seed_results = []
    for topology in cfg["topologies"]:
        print(f"\n{'#' * 90}\n# topology: {topology}\n{'#' * 90}", flush=True)
        for seed in cfg["seeds"]:
            unit = f"{topology}|{seed}"

            def run(t=topology, s=seed):
                return run_one_seed(t, s, cfg, device)

            seed_results.append(execute(unit, run, f"  seed {seed}"))
    return seed_results


# --------------------------------------------------------------- analysis ---
def build_comparisons(seed_results: list, cfg: dict) -> list:
    by_topo = {}
    for r in seed_results:
        by_topo.setdefault(r["topology_requested"], []).append(r)

    comparisons = []
    for topology, results in by_topo.items():
        for budget in cfg["budgets"]:
            ana_rows, giles_rows = [], []
            for r in results:
                pb = {(e["arm"]): e for e in r["per_budget"] if e["budget"] == budget}
                ana_rows.append(pb["ana"])
                giles_rows.append(pb["giles"])

            ci_ana = [e["ci_half_width"] for e in ana_rows]
            ci_giles = [e["ci_half_width"] for e in giles_rows]
            work_ana = [e["work_units"] for e in ana_rows]
            work_giles = [e["work_units"] for e in giles_rows]

            entry = {
                "topology": topology, "budget": budget, "n_seeds": len(results),
                "ci_half_width_mean_ana": float(np.mean(ci_ana)),
                "ci_half_width_sd_ana": float(np.std(ci_ana, ddof=1)) if len(ci_ana) > 1 else 0.0,
                "ci_half_width_mean_giles": float(np.mean(ci_giles)),
                "ci_half_width_sd_giles": float(np.std(ci_giles, ddof=1)) if len(ci_giles) > 1 else 0.0,
                "ttest_ci_half_width": paired_ttest(ci_ana, ci_giles),
                "work_mean_ana": float(np.mean(work_ana)), "work_mean_giles": float(np.mean(work_giles)),
                "ana_wins_ci": bool(np.mean(ci_giles) > np.mean(ci_ana) * WIN_MARGIN),
            }

            for k in cfg["k_values"]:
                rmse_ana = [rmse_lookup(e["rmse_topk"], k) for e in ana_rows]
                rmse_giles = [rmse_lookup(e["rmse_topk"], k) for e in giles_rows]
                entry[f"rmse_top{k}_mean_ana"] = float(np.mean(rmse_ana))
                entry[f"rmse_top{k}_sd_ana"] = float(np.std(rmse_ana, ddof=1)) if len(rmse_ana) > 1 else 0.0
                entry[f"rmse_top{k}_mean_giles"] = float(np.mean(rmse_giles))
                entry[f"rmse_top{k}_sd_giles"] = (float(np.std(rmse_giles, ddof=1))
                                                   if len(rmse_giles) > 1 else 0.0)
                entry[f"ttest_rmse_top{k}"] = paired_ttest(rmse_ana, rmse_giles)
                entry[f"ana_wins_rmse_top{k}"] = bool(
                    np.mean(rmse_giles) > np.mean(rmse_ana) * WIN_MARGIN)

            n_metrics = 1 + len(cfg["k_values"])
            n_ana_wins = int(entry["ana_wins_ci"]) + sum(
                int(entry[f"ana_wins_rmse_top{k}"]) for k in cfg["k_values"])
            entry["ana_wins_majority"] = n_ana_wins > n_metrics / 2
            comparisons.append(entry)
    return comparisons


# --------------------------------------------------------------- outputs ----
def write_outputs(args, cfg: dict, prov: dict, seed_results: list, comparisons: list) -> dict:
    os.makedirs(args.out, exist_ok=True)
    header = provenance_comment_lines(prov)
    paths = {}

    json_path = os.path.join(args.out, "ana_fair_budget.json")
    with open(json_path, "w") as handle:
        json.dump({"provenance": prov, "seed_results": seed_results, "comparisons": comparisons},
                  handle, indent=2, default=float)
    paths["json"] = json_path

    def open_csv(name):
        handle = open(os.path.join(args.out, name), "w", newline="")
        for line in header:
            handle.write(line + "\n")
        return handle

    with open_csv("arms.csv") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "topology_requested", "topology_used", "seed", "budget", "arm", "estimate", "reference",
            "bias", "variance", "ci_half_width", "work_units", "samples_per_level",
            *[f"rmse_top{k}" for k in cfg["k_values"]],
        ])
        for r in seed_results:
            for e in r["per_budget"]:
                writer.writerow([
                    r["topology_requested"], r["topology_used"], r["seed"], e["budget"], e["arm"],
                    e["estimate"], e["reference"], e["bias"], e["variance"], e["ci_half_width"],
                    e["work_units"], json.dumps(e["samples_per_level"]),
                    *[rmse_lookup(e["rmse_topk"], k) for k in cfg["k_values"]],
                ])
    paths["arms_csv"] = os.path.join(args.out, "arms.csv")

    with open_csv("comparisons.csv") as handle:
        writer = csv.writer(handle)
        base_cols = ["topology", "budget", "n_seeds",
                     "ci_half_width_mean_ana", "ci_half_width_sd_ana",
                     "ci_half_width_mean_giles", "ci_half_width_sd_giles",
                     "t_ci", "p_ci", "effect_size_ci", "ana_wins_ci",
                     "work_mean_ana", "work_mean_giles"]
        k_cols = []
        for k in cfg["k_values"]:
            k_cols += [f"rmse_top{k}_mean_ana", f"rmse_top{k}_sd_ana",
                       f"rmse_top{k}_mean_giles", f"rmse_top{k}_sd_giles",
                       f"t_rmse_top{k}", f"p_rmse_top{k}", f"effect_size_rmse_top{k}",
                       f"ana_wins_rmse_top{k}"]
        writer.writerow(base_cols + k_cols + ["ana_wins_majority"])
        for c in comparisons:
            tci = c["ttest_ci_half_width"]
            row = [c["topology"], c["budget"], c["n_seeds"],
                   c["ci_half_width_mean_ana"], c["ci_half_width_sd_ana"],
                   c["ci_half_width_mean_giles"], c["ci_half_width_sd_giles"],
                   tci["t"], tci["p"], tci["effect_size"], c["ana_wins_ci"],
                   c["work_mean_ana"], c["work_mean_giles"]]
            for k in cfg["k_values"]:
                tk = c[f"ttest_rmse_top{k}"]
                row += [c[f"rmse_top{k}_mean_ana"], c[f"rmse_top{k}_sd_ana"],
                        c[f"rmse_top{k}_mean_giles"], c[f"rmse_top{k}_sd_giles"],
                        tk["t"], tk["p"], tk["effect_size"], c[f"ana_wins_rmse_top{k}"]]
            row.append(c["ana_wins_majority"])
            writer.writerow(row)
    paths["comparisons_csv"] = os.path.join(args.out, "comparisons.csv")
    return paths


def print_report(comparisons: list, cfg: dict) -> None:
    print("\n" + "=" * 120)
    print("ANA FAIR-BUDGET COMPARISON  (both arms estimate the SAME sum_i w_i E[Q_i], budget-matched)")
    print("=" * 120)
    for topology in sorted({c["topology"] for c in comparisons}):
        print(f"\n--- topology: {topology} ---")
        print(f"{'budget':>12}{'CI_ana':>12}{'CI_giles':>12}{'p(CI)':>9}{'ana<giles CI':>13}"
              + "".join(f"{'RMSE@'+str(k)+'(a/g)':>20}" for k in cfg["k_values"]) + f"{'majority':>10}")
        for c in sorted([c for c in comparisons if c["topology"] == topology],
                        key=lambda c: c["budget"]):
            rmse_str = "".join(
                f"{c[f'rmse_top{k}_mean_ana']:.3g}/{c[f'rmse_top{k}_mean_giles']:.3g}".rjust(20)
                for k in cfg["k_values"])
            print(f"{c['budget']:>12.3g}{c['ci_half_width_mean_ana']:>12.4g}"
                  f"{c['ci_half_width_mean_giles']:>12.4g}"
                  f"{(c['ttest_ci_half_width']['p'] if c['ttest_ci_half_width']['p'] is not None else float('nan')):>9.3g}"
                  f"{str(c['ana_wins_ci']):>13}{rmse_str}{str(c['ana_wins_majority']):>10}")

    n_total = len(comparisons)
    n_ana_majority = sum(1 for c in comparisons if c["ana_wins_majority"])
    n_ci_wins = sum(1 for c in comparisons if c["ana_wins_ci"])
    print(f"\nVERDICT: ANA wins the majority of metrics in {n_ana_majority}/{n_total} "
          f"(topology, budget) cells; wins on CI half-width alone in {n_ci_wins}/{n_total}.")
    if n_ana_majority >= n_total * 0.5:
        print("  Under the fair budget-matched, same-target comparison, ANA allocation "
              "favours the estimator MORE OFTEN THAN NOT across the swept budgets.")
    else:
        print("  Under the fair budget-matched, same-target comparison, ANA allocation does "
              "NOT favour the estimator in most swept budgets. This is reported as measured, "
              "not tuned toward a positive result.")


def parse_args():
    parser = argparse.ArgumentParser(description="Fair fixed-budget ANA vs Giles comparison")
    parser.add_argument("--budgets", type=float, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--topologies", nargs="+", choices=["caida", "ba"], default=["ba"])
    parser.add_argument("--n-nodes", type=int, default=500)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--base-dt", type=float, default=0.1)
    parser.add_argument("--L-max", type=int, default=4)
    parser.add_argument("--pilot-samples", type=int, default=50)
    parser.add_argument("--reference-samples-per-level", type=int, default=20000)
    parser.add_argument("--k-values", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--influence-strength", type=float, default=0.1)
    parser.add_argument("--decay-rate", type=float, default=0.5)
    parser.add_argument("--noise-intensity", type=float, default=0.1)
    parser.add_argument("--out", default=os.path.join("results", "ana_fair_budget"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--quick", action="store_true", help="Small smoke-test configuration")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if any(b <= 0 for b in args.budgets):
        raise ValueError("--budgets must all be positive")
    if args.L_max < 1:
        raise ValueError("--L-max must be >= 1")
    if args.pilot_samples < 2:
        raise ValueError("--pilot-samples must be >= 2 (ddof=1 variance needs at least 2 samples)")
    if not args.seeds:
        raise ValueError("--seeds must be non-empty")
    if args.n_nodes < 10:
        raise ValueError("--n-nodes must be >= 10")
    if any(k <= 0 for k in args.k_values):
        raise ValueError("--k-values must all be positive")
    if any(k > args.n_nodes for k in args.k_values):
        raise ValueError(f"--k-values must all be <= --n-nodes ({args.n_nodes})")
    if args.reference_samples_per_level <= max(args.budgets) / (args.L_max + 1):
        raise ValueError(
            "--reference-samples-per-level is not comfortably larger than the largest budget's "
            "typical per-level allocation; the 'reference' would not be a trustworthy ground truth")

    if args.quick:
        args.n_nodes = min(args.n_nodes, 30)
        args.pilot_samples = min(args.pilot_samples, 16)
        args.reference_samples_per_level = min(args.reference_samples_per_level, 800)
        args.seeds = args.seeds[:2]
        # Small, strictly-increasing budgets rather than a capped/deduped
        # transform of the (much larger) full-config defaults, which could
        # otherwise collapse to duplicate values after clipping.
        if list(args.budgets) == DEFAULT_BUDGETS:
            args.budgets = [500.0, 1000.0, 2000.0]
        args.k_values = [k for k in args.k_values if k <= args.n_nodes] or [5]

    device = select_device(args.device)
    cfg = {
        "budgets": sorted(args.budgets), "seeds": list(args.seeds), "topologies": list(args.topologies),
        "n_nodes": args.n_nodes, "T": args.T, "base_dt": args.base_dt, "L_max": args.L_max,
        "pilot_samples": args.pilot_samples,
        "reference_samples_per_level": args.reference_samples_per_level,
        "k_values": sorted(args.k_values), "influence_strength": args.influence_strength,
        "decay_rate": args.decay_rate, "noise_intensity": args.noise_intensity, "quick": args.quick,
    }

    os.makedirs(args.out, exist_ok=True)
    prov = build_provenance(device, cfg)

    print("=" * 100)
    print("ANA fair fixed-work-budget comparison (Arm A: ANA-weighted allocation vs "
          "Arm B: standard Giles, SAME target)")
    print("=" * 100)
    print(f"  git sha    : {prov['git_sha']}")
    print(f"  device     : {prov['device']} ({prov['device_name']})")
    print(f"  torch      : {prov['torch_version']}")
    print(f"  seeds      : {cfg['seeds']}")
    print(f"  topologies : {cfg['topologies']}  n_nodes={cfg['n_nodes']}")
    print(f"  budgets    : {cfg['budgets']}")
    print(f"  k values   : {cfg['k_values']}")
    print(f"  out        : {os.path.abspath(args.out)}", flush=True)

    started = time.perf_counter()
    seed_results = run_all(args, cfg, device)
    comparisons = build_comparisons(seed_results, cfg)
    total = time.perf_counter() - started

    paths = write_outputs(args, cfg, prov, seed_results, comparisons)
    print_report(comparisons, cfg)

    print(f"\nTotal wall clock: {total:.1f}s")
    for label, path in paths.items():
        print(f"  {label:<16} {path}")


if __name__ == "__main__":
    main()
