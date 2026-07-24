"""ANA-MLMC weight and hyperparameter sensitivity sweep (Reviewer 2, T2.2).

The original version of this script covered 6 named corner/mix configurations
of (gamma_C, gamma_V, gamma_S) at n=300, eps=0.005, 3 seeds -- too thin to
cite. This extends it to:

  1. A ~21-point simplex grid over (gamma_C, gamma_V, gamma_S) (default
     resolution D=5, giving exactly (D+1)(D+2)/2 = 21 lattice points
     gamma = (i/D, j/D, k/D) with i+j+k=D), at the real operating point
     (n=500, eps=0.02 by default, both CLI-overridable).
  2. One-factor-at-a-time sweeps over L_max in {2,3,4,5}, refinement M in
     {2,4}, and pilot samples N_pilot in {20,50,100,200}, holding the ANA
     weights at the default mix (0.4, 0.4, 0.2) and everything else at the
     operating point.
  3. >=5 seeds everywhere (default 5, CLI-overridable upward).

Backward compatibility.  The original 6-corner-plus-baseline comparison
(`weight_sweep.csv` / `weight_sweep_summary.json`) is preserved with the
SAME columns / JSON keys as before -- only a provenance comment header is
now prepended to the CSV (read back with `pandas.read_csv(..., comment='#')`
or by skipping '#'-prefixed lines), which every output file in this script
now carries per the shared experiment-script convention. The new simplex and
one-factor-sweep results are written to NEW files alongside it.

Usage:
    python3 scripts/run_ana_weight_sweep.py --quick
    python3 scripts/run_ana_weight_sweep.py --n 500 --epsilon 0.02 --seeds 5 --device cuda

Output (under --out/--output, default results/ana_weight_sweep):
    weight_sweep.csv              legacy: 6 named configs + baseline, unchanged schema
    weight_sweep_summary.json     legacy: unchanged shape, provenance added
    weight_simplex.csv            NEW: one row per (seed, simplex point) --
                                   shaped directly for a ternary heatmap
                                   (gamma_C, gamma_V, gamma_S, metric columns)
    weight_simplex_summary.json   NEW: per-point seed-aggregated summary
    tornado_sweep.csv             NEW: one row per (factor, level, seed) --
                                   shaped for a tornado sensitivity plot
    tornado_summary.json          NEW: per-factor effect size, sorted by
                                   |max(metric) - min(metric)| descending
    checkpoint.jsonl              append-only resume log (all three
                                   experiments share one checkpoint, unit ids
                                   namespaced by experiment)
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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from gpu.parallel_mc import GPUAdaptiveNetworkAwareMLMC, GPUCoupledPropagationMLMC  # noqa: E402
from network.topology import TopologyGenerator  # noqa: E402

SCHEMA_VERSION = 1

#: Default ANA weight mix used as the fixed point for the one-factor sweeps
#: and as the "default_mix" legacy corner.
DEFAULT_MIX = (0.4, 0.4, 0.2)


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
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "-C", str(ROOT), "diff", "--quiet"],
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


def open_csv(path: Path, header_lines: list):
    handle = path.open("w", newline="")
    for line in header_lines:
        handle.write(line + "\n")
    return handle


# --------------------------------------------------------------- topology ---
def _load_ba(n_nodes: int, seed: int):
    return TopologyGenerator(seed=seed).generate_barabasi_albert(n_nodes=n_nodes, m=3).get_largest_component()


# ------------------------------------------------------------- simplex ------
def simplex_points(resolution: int) -> List[Tuple[float, float, float]]:
    """Lattice points on the 2-simplex: gamma=(i/D, j/D, k/D), i+j+k=D.

    Yields exactly (D+1)(D+2)/2 points (D=5 -> 21).
    """
    if resolution < 1:
        raise ValueError("--simplex-resolution must be >= 1")
    points = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            points.append((i / resolution, j / resolution, k / resolution))
    return points


# --------------------------------------------------------------- runners ----
def _run_ana_weighted(adjacency, *, weights: Tuple[float, float, float], epsilon: float,
                       T: float, base_dt: float, L_max: int, pilot_samples: int,
                       refinement_factor: int, seed: int, device: torch.device) -> Dict:
    sim = GPUAdaptiveNetworkAwareMLMC(
        adjacency_matrix=adjacency, seed=seed, refinement_factor=refinement_factor,
        weight_centrality=weights[0], weight_variance=weights[1], weight_sla=weights[2],
    )
    pin_device(sim, device)
    started = time.perf_counter()
    result = sim.mlmc_estimate_weighted(
        epsilon=epsilon, T=T, base_dt=base_dt, L_max=L_max,
        pilot_samples=pilot_samples, verbose=False,
    )
    result["runtime_s"] = time.perf_counter() - started
    return result


def _run_baseline_giles(adjacency, *, epsilon: float, T: float, base_dt: float, L_max: int,
                         pilot_samples: int, refinement_factor: int, seed: int,
                         device: torch.device) -> Dict:
    sim = GPUCoupledPropagationMLMC(
        adjacency_matrix=adjacency, seed=seed, refinement_factor=refinement_factor,
    )
    pin_device(sim, device)
    started = time.perf_counter()
    result = sim.mlmc_estimate(
        epsilon=epsilon, T=T, base_dt=base_dt, L_max=L_max,
        pilot_samples=pilot_samples, verbose=False,
    )
    result["runtime_s"] = time.perf_counter() - started
    return result


# -------------------------------------------------------------- checkpoint --
def config_fingerprint(cfg: dict) -> str:
    payload = {k: v for k, v in sorted(cfg.items()) if k not in ("seeds", "_provenance")}
    return json.dumps(payload, sort_keys=True, default=str)


def load_checkpoint(path: Path, cfg: dict) -> dict:
    done = {}
    if not path.exists():
        return done
    fingerprint = config_fingerprint(cfg)
    stale = 0
    with path.open("r") as handle:
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


def append_checkpoint(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=float) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def make_executor(checkpoint_path: Path, cfg: dict):
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

    return execute


# -------------------------------------------------------- experiment: legacy
def run_legacy_corners(execute, cfg: dict, device: torch.device) -> List[Dict]:
    """The original 6-named-config + baseline comparison, unchanged schema."""
    configs = [
        ("centrality_only", (1.0, 0.0, 0.0)),
        ("variance_only", (0.0, 1.0, 0.0)),
        ("sla_only", (0.0, 0.0, 1.0)),
        ("default_mix", DEFAULT_MIX),
        ("balanced_mix", (1 / 3, 1 / 3, 1 / 3)),
    ]
    rows: List[Dict] = []
    for seed in range(cfg["seeds"]):
        adjacency = _load_ba(cfg["n"], seed=seed).get_adjacency_matrix()

        def run_b(a=adjacency, s=seed):
            return _run_baseline_giles(a, epsilon=cfg["epsilon"], T=cfg["T"], base_dt=cfg["base_dt"],
                                        L_max=cfg["L_max"], pilot_samples=cfg["pilot_samples"],
                                        refinement_factor=cfg["refinement_factor"], seed=s, device=device)

        b = execute(f"legacy|baseline_giles|{seed}", run_b, f"  [legacy] baseline_giles seed {seed}")
        rows.append({
            "config": "baseline_giles", "seed": seed,
            "gamma_C": np.nan, "gamma_V": np.nan, "gamma_S": np.nan,
            "total_cost": b["total_cost"], "runtime_s": b["runtime_s"],
            "estimate": b["estimate"], "ci_lower": b["ci_lower"], "ci_upper": b["ci_upper"],
            "ci_width": b["ci_upper"] - b["ci_lower"],
        })

        for name, w in configs:
            def run_a(a=adjacency, ww=w, s=seed):
                return _run_ana_weighted(a, weights=ww, epsilon=cfg["epsilon"], T=cfg["T"],
                                          base_dt=cfg["base_dt"], L_max=cfg["L_max"],
                                          pilot_samples=cfg["pilot_samples"],
                                          refinement_factor=cfg["refinement_factor"], seed=s, device=device)

            r = execute(f"legacy|{name}|{seed}", run_a, f"  [legacy] {name} seed {seed}")
            rows.append({
                "config": name, "seed": seed,
                "gamma_C": w[0], "gamma_V": w[1], "gamma_S": w[2],
                "total_cost": r["total_cost"], "runtime_s": r["runtime_s"],
                "estimate": r["estimate"], "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"],
                "ci_width": r["ci_upper"] - r["ci_lower"],
            })
    return rows


def write_legacy_outputs(rows: List[Dict], out_dir: Path, cfg: dict, header: list) -> Tuple[Path, Path]:
    csv_path = out_dir / "weight_sweep.csv"
    with open_csv(csv_path, header) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    summary: Dict[str, Dict[str, float]] = {}
    for c in {r["config"] for r in rows}:
        sub = [r for r in rows if r["config"] == c]
        summary[c] = {
            "n_seeds": len(sub),
            "total_cost_mean": float(np.mean([r["total_cost"] for r in sub])),
            "total_cost_std": float(np.std([r["total_cost"] for r in sub], ddof=1) if len(sub) > 1 else 0.0),
            "runtime_mean": float(np.mean([r["runtime_s"] for r in sub])),
            "ci_width_mean": float(np.mean([r["ci_width"] for r in sub])),
            "ci_width_std": float(np.std([r["ci_width"] for r in sub], ddof=1) if len(sub) > 1 else 0.0),
        }
    json_path = out_dir / "weight_sweep_summary.json"
    json_path.write_text(json.dumps({
        "epsilon": cfg["epsilon"], "n_nodes": cfg["n"], "T": cfg["T"], "L_max": cfg["L_max"],
        "pilot_samples": cfg["pilot_samples"], "provenance": cfg.get("_provenance"),
        "summary": summary, "raw_rows": rows,
    }, indent=2, default=float))
    return csv_path, json_path


# ------------------------------------------------------- experiment: simplex
def run_simplex(execute, cfg: dict, device: torch.device) -> List[Dict]:
    points = simplex_points(cfg["simplex_resolution"])
    print(f"  [simplex] {len(points)} lattice points x {cfg['seeds']} seeds", flush=True)
    rows: List[Dict] = []
    for seed in range(cfg["seeds"]):
        adjacency = _load_ba(cfg["n"], seed=seed).get_adjacency_matrix()
        for idx, w in enumerate(points):
            def run(a=adjacency, ww=w, s=seed):
                return _run_ana_weighted(a, weights=ww, epsilon=cfg["epsilon"], T=cfg["T"],
                                          base_dt=cfg["base_dt"], L_max=cfg["L_max"],
                                          pilot_samples=cfg["pilot_samples"],
                                          refinement_factor=cfg["refinement_factor"], seed=s, device=device)

            r = execute(f"simplex|{idx}|{seed}", run, f"  [simplex] point {idx} seed {seed}")
            rows.append({
                "point_index": idx, "seed": seed,
                "gamma_C": w[0], "gamma_V": w[1], "gamma_S": w[2],
                "total_cost": r["total_cost"], "runtime_s": r["runtime_s"],
                "estimate": r["estimate"], "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"],
                "ci_width": r["ci_upper"] - r["ci_lower"],
            })
    return rows


def write_simplex_outputs(rows: List[Dict], out_dir: Path, cfg: dict, header: list) -> Tuple[Path, Path]:
    csv_path = out_dir / "weight_simplex.csv"
    with open_csv(csv_path, header + [
            "# one row per (seed, simplex point); shaped for a ternary heatmap:",
            "# plot (gamma_C, gamma_V, gamma_S) with color = ci_width or total_cost"]) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    by_point: Dict[int, List[Dict]] = {}
    for r in rows:
        by_point.setdefault(r["point_index"], []).append(r)
    summary = []
    for idx, sub in sorted(by_point.items()):
        gamma = (sub[0]["gamma_C"], sub[0]["gamma_V"], sub[0]["gamma_S"])
        summary.append({
            "point_index": idx, "gamma_C": gamma[0], "gamma_V": gamma[1], "gamma_S": gamma[2],
            "n_seeds": len(sub),
            "total_cost_mean": float(np.mean([r["total_cost"] for r in sub])),
            "total_cost_std": float(np.std([r["total_cost"] for r in sub], ddof=1) if len(sub) > 1 else 0.0),
            "ci_width_mean": float(np.mean([r["ci_width"] for r in sub])),
            "ci_width_std": float(np.std([r["ci_width"] for r in sub], ddof=1) if len(sub) > 1 else 0.0),
            "estimate_mean": float(np.mean([r["estimate"] for r in sub])),
        })
    json_path = out_dir / "weight_simplex_summary.json"
    json_path.write_text(json.dumps({
        "resolution": cfg["simplex_resolution"], "n_points": len(by_point),
        "epsilon": cfg["epsilon"], "n_nodes": cfg["n"], "provenance": cfg.get("_provenance"),
        "summary": summary,
    }, indent=2, default=float))
    return csv_path, json_path


# ------------------------------------------------------- experiment: tornado
def run_tornado(execute, cfg: dict, device: torch.device) -> List[Dict]:
    """One-factor-at-a-time sweep over L_max, refinement M, and N_pilot,
    holding the ANA weights at DEFAULT_MIX and everything else at the
    operating point."""
    factors = [
        ("L_max", cfg["L_max_sweep"], "L_max"),
        ("refinement_factor", cfg["refinement_sweep"], "refinement_factor"),
        ("pilot_samples", cfg["pilot_sweep"], "pilot_samples"),
    ]
    rows: List[Dict] = []
    for seed in range(cfg["seeds"]):
        adjacency = _load_ba(cfg["n"], seed=seed).get_adjacency_matrix()
        for factor_name, levels, kwarg in factors:
            for level in levels:
                params = dict(epsilon=cfg["epsilon"], T=cfg["T"], base_dt=cfg["base_dt"],
                              L_max=cfg["L_max"], pilot_samples=cfg["pilot_samples"],
                              refinement_factor=cfg["refinement_factor"])
                params[kwarg] = level

                def run(a=adjacency, p=params, s=seed):
                    return _run_ana_weighted(a, weights=DEFAULT_MIX, seed=s, device=device, **p)

                unit = f"tornado|{factor_name}|{level}|{seed}"
                r = execute(unit, run, f"  [tornado] {factor_name}={level} seed {seed}")
                rows.append({
                    "factor": factor_name, "level": level, "seed": seed,
                    "total_cost": r["total_cost"], "runtime_s": r["runtime_s"],
                    "estimate": r["estimate"], "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"],
                    "ci_width": r["ci_upper"] - r["ci_lower"],
                })
    return rows


def write_tornado_outputs(rows: List[Dict], out_dir: Path, cfg: dict, header: list) -> Tuple[Path, Path]:
    csv_path = out_dir / "tornado_sweep.csv"
    with open_csv(csv_path, header + [
            "# one row per (factor, level, seed); shaped for a tornado plot:",
            "# for each factor, plot [min(ci_width_mean), max(ci_width_mean)] across levels"]) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    by_factor: Dict[str, Dict] = {}
    for r in rows:
        by_factor.setdefault(r["factor"], {}).setdefault(r["level"], []).append(r)

    factor_summaries = []
    for factor, by_level in by_factor.items():
        level_stats = []
        for level, sub in sorted(by_level.items()):
            level_stats.append({
                "level": level, "n_seeds": len(sub),
                "ci_width_mean": float(np.mean([r["ci_width"] for r in sub])),
                "ci_width_std": float(np.std([r["ci_width"] for r in sub], ddof=1) if len(sub) > 1 else 0.0),
                "total_cost_mean": float(np.mean([r["total_cost"] for r in sub])),
                "runtime_mean": float(np.mean([r["runtime_s"] for r in sub])),
            })
        ci_means = [s["ci_width_mean"] for s in level_stats]
        low_idx, high_idx = int(np.argmin(ci_means)), int(np.argmax(ci_means))
        factor_summaries.append({
            "factor": factor, "levels": level_stats,
            "ci_width_low": level_stats[low_idx]["ci_width_mean"],
            "ci_width_low_level": level_stats[low_idx]["level"],
            "ci_width_high": level_stats[high_idx]["ci_width_mean"],
            "ci_width_high_level": level_stats[high_idx]["level"],
            "effect_size": level_stats[high_idx]["ci_width_mean"] - level_stats[low_idx]["ci_width_mean"],
        })
    factor_summaries.sort(key=lambda s: abs(s["effect_size"]), reverse=True)

    json_path = out_dir / "tornado_summary.json"
    json_path.write_text(json.dumps({
        "epsilon": cfg["epsilon"], "n_nodes": cfg["n"], "default_mix": DEFAULT_MIX,
        "provenance": cfg.get("_provenance"),
        "factors_sorted_by_effect_size": factor_summaries,
    }, indent=2, default=float))
    return csv_path, json_path


# ------------------------------------------------------------------- main ---
def print_legacy_table(rows: List[Dict]) -> None:
    summary: Dict[str, Dict[str, float]] = {}
    for cfg_name in {r["config"] for r in rows}:
        sub = [r for r in rows if r["config"] == cfg_name]
        summary[cfg_name] = {
            "total_cost_mean": float(np.mean([r["total_cost"] for r in sub])),
            "total_cost_std": float(np.std([r["total_cost"] for r in sub], ddof=1) if len(sub) > 1 else 0.0),
            "runtime_mean": float(np.mean([r["runtime_s"] for r in sub])),
            "ci_width_mean": float(np.mean([r["ci_width"] for r in sub])),
        }
    print(f"\n{'config':<20}{'cost_mean':>14}{'cost_sd':>10}{'rt_mean(s)':>12}{'ci_width':>12}")
    order = ["baseline_giles", "centrality_only", "variance_only", "sla_only", "balanced_mix", "default_mix"]
    for c in order:
        if c not in summary:
            continue
        s = summary[c]
        print(f"{c:<20}{s['total_cost_mean']:>14.0f}{s['total_cost_std']:>10.0f}"
              f"{s['runtime_mean']:>12.3f}{s['ci_width_mean']:>12.4f}")


def print_tornado_table(json_path: Path) -> None:
    data = json.loads(json_path.read_text())
    print(f"\n{'factor':<20}{'low(level)':>16}{'high(level)':>16}{'|effect|':>12}")
    for f in data["factors_sorted_by_effect_size"]:
        print(f"{f['factor']:<20}"
              f"{f['ci_width_low']:.4g} (@{f['ci_width_low_level']})".rjust(16)
              + f"{f['ci_width_high']:.4g} (@{f['ci_width_high_level']})".rjust(16)
              + f"{abs(f['effect_size']):>12.4g}")


def parse_args():
    parser = argparse.ArgumentParser(description="ANA-MLMC weight + hyperparameter sensitivity sweep")
    parser.add_argument("--n", type=int, default=500, help="Network size (was 300)")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Target accuracy (was 0.005)")
    parser.add_argument("--seeds", type=int, default=5, help="Seed count, >=5 (was 3)")
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--base-dt", type=float, default=0.1)
    parser.add_argument("--L-max", type=int, default=4, help="Baseline L_max for the legacy/simplex sweeps")
    parser.add_argument("--pilot-samples", type=int, default=50, help="Baseline N_pilot")
    parser.add_argument("--refinement-factor", type=int, default=2, help="Baseline refinement M")
    parser.add_argument("--simplex-resolution", type=int, default=5,
                        help="D such that the simplex has (D+1)(D+2)/2 points (D=5 -> 21)")
    parser.add_argument("--L-max-sweep", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--refinement-sweep", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--pilot-sweep", type=int, nargs="+", default=[20, 50, 100, 200])
    parser.add_argument("--experiments", nargs="+", choices=["legacy", "simplex", "tornado"],
                        default=["legacy", "simplex", "tornado"])
    parser.add_argument("--out", "--output", dest="output", type=Path,
                        default=ROOT / "results" / "ana_weight_sweep")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--quick", action="store_true", help="Small smoke-test configuration")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive")
    if args.L_max < 1:
        raise ValueError("--L-max must be >= 1")
    if args.pilot_samples < 2:
        raise ValueError("--pilot-samples must be >= 2 (ddof=1 variance needs at least 2 samples)")
    if args.seeds < 5:
        raise ValueError("--seeds must be >= 5 per the task spec (thin seed counts were the "
                          "reviewer complaint this script fixes)")
    if args.n < 10:
        raise ValueError("--n must be >= 10")
    if args.simplex_resolution < 1:
        raise ValueError("--simplex-resolution must be >= 1")
    if any(L < 1 for L in args.L_max_sweep):
        raise ValueError("--L-max-sweep values must all be >= 1")
    if any(m < 2 for m in args.refinement_sweep):
        raise ValueError("--refinement-sweep values must all be >= 2")
    if any(p < 2 for p in args.pilot_sweep):
        raise ValueError("--pilot-sweep values must all be >= 2")

    if args.quick:
        args.n = min(args.n, 30)
        args.seeds = 5  # keep the >=5 floor even in --quick; this is what the sanity check verifies
        args.simplex_resolution = min(args.simplex_resolution, 2)  # (2+1)(2+2)/2 = 6 points
        args.L_max_sweep = [2, 3]
        args.refinement_sweep = [2]
        args.pilot_sweep = [10, 20]
        args.L_max = min(args.L_max, 3)
        args.pilot_samples = min(args.pilot_samples, 16)

    device = select_device(args.device)
    cfg = {
        "n": args.n, "epsilon": args.epsilon, "seeds": args.seeds, "T": args.T,
        "base_dt": args.base_dt, "L_max": args.L_max, "pilot_samples": args.pilot_samples,
        "refinement_factor": args.refinement_factor, "simplex_resolution": args.simplex_resolution,
        "L_max_sweep": args.L_max_sweep, "refinement_sweep": args.refinement_sweep,
        "pilot_sweep": args.pilot_sweep, "experiments": args.experiments, "quick": args.quick,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    # Snapshot cfg (dict(cfg)) before embedding it in prov["config"] -- prov
    # is then stashed back onto cfg as cfg["_provenance"] for the JSON
    # outputs, and mutating the live cfg afterwards must not also mutate the
    # snapshot inside prov (that would be a circular reference).
    prov = build_provenance(device, dict(cfg))
    cfg["_provenance"] = prov
    header = provenance_comment_lines(prov)

    print("=" * 100)
    print("ANA-MLMC weight + hyperparameter sensitivity sweep")
    print("=" * 100)
    print(f"  git sha     : {prov['git_sha']}")
    print(f"  device      : {prov['device']} ({prov['device_name']})")
    print(f"  torch       : {prov['torch_version']}")
    print(f"  n_nodes     : {cfg['n']}   epsilon: {cfg['epsilon']}   seeds: {cfg['seeds']}")
    print(f"  simplex     : resolution={cfg['simplex_resolution']} "
          f"({len(simplex_points(cfg['simplex_resolution']))} points)")
    print(f"  experiments : {cfg['experiments']}")
    print(f"  out         : {args.output.resolve()}", flush=True)

    checkpoint_path = args.output / "checkpoint.jsonl"
    if args.no_resume and checkpoint_path.exists():
        checkpoint_path.unlink()
        print("  [checkpoint] --no-resume: cleared previous log", flush=True)
    execute = make_executor(checkpoint_path, cfg)

    started = time.perf_counter()
    written = {}

    if "legacy" in cfg["experiments"]:
        print("\n" + "#" * 90 + "\n# experiment: legacy 6-corner comparison\n" + "#" * 90, flush=True)
        legacy_rows = run_legacy_corners(execute, cfg, device)
        csv_p, json_p = write_legacy_outputs(legacy_rows, args.output, cfg, header)
        written["weight_sweep_csv"], written["weight_sweep_json"] = csv_p, json_p
        print(f"\n=== legacy corners (BA n={cfg['n']}, eps={cfg['epsilon']}, {cfg['seeds']} seeds) ===")
        print_legacy_table(legacy_rows)

    if "simplex" in cfg["experiments"]:
        print("\n" + "#" * 90 + "\n# experiment: 21-point weight simplex\n" + "#" * 90, flush=True)
        simplex_rows = run_simplex(execute, cfg, device)
        csv_p, json_p = write_simplex_outputs(simplex_rows, args.output, cfg, header)
        written["weight_simplex_csv"], written["weight_simplex_json"] = csv_p, json_p

    if "tornado" in cfg["experiments"]:
        print("\n" + "#" * 90 + "\n# experiment: one-factor tornado sweep (L_max, M, N_pilot)\n" + "#" * 90,
              flush=True)
        tornado_rows = run_tornado(execute, cfg, device)
        csv_p, json_p = write_tornado_outputs(tornado_rows, args.output, cfg, header)
        written["tornado_csv"], written["tornado_json"] = csv_p, json_p
        print("\n=== tornado sensitivity (sorted by |effect on ci_width|, descending) ===")
        print_tornado_table(json_p)

    total = time.perf_counter() - started
    print(f"\nTotal wall clock: {total:.1f}s")
    for label, path in written.items():
        print(f"  {label:<20} {path}")


if __name__ == "__main__":
    main()
