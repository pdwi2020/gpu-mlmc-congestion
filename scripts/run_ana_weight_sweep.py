"""ANA-MLMC weight sensitivity sweep.

For a fixed network and accuracy target, vary (gamma_C, gamma_V, gamma_S)
across a small set of named configurations and report sample budget,
runtime, and global CI half-width per configuration. This addresses the
reviewer-flagged sensitivity question in Limitations.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gpu.parallel_mc import GPUAdaptiveNetworkAwareMLMC, GPUCoupledPropagationMLMC
from network.topology import TopologyGenerator


def _load_ba(n_nodes: int, seed: int):
    return TopologyGenerator(seed=seed).generate_barabasi_albert(n_nodes=n_nodes, m=3).get_largest_component()


def _run_one(adjacency, *, weights: Tuple[float, float, float], epsilon: float,
             T: float, base_dt: float, L_max: int, pilot_samples: int, seed: int) -> Dict:
    sim = GPUAdaptiveNetworkAwareMLMC(
        adjacency_matrix=adjacency, seed=seed,
        weight_centrality=weights[0],
        weight_variance=weights[1],
        weight_sla=weights[2],
    )
    started = time.perf_counter()
    res = sim.mlmc_estimate_weighted(
        epsilon=epsilon, T=T, base_dt=base_dt, L_max=L_max,
        pilot_samples=pilot_samples, verbose=False,
    )
    res["runtime_s"] = time.perf_counter() - started
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--base-dt", type=float, default=0.1)
    parser.add_argument("--L-max", type=int, default=4)
    parser.add_argument("--pilot-samples", type=int, default=50)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "ana_weight_sweep")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Named configurations covering the simplex corners and the default mix.
    configs = [
        ("centrality_only", (1.0, 0.0, 0.0)),
        ("variance_only",   (0.0, 1.0, 0.0)),
        ("sla_only",        (0.0, 0.0, 1.0)),
        ("default_mix",     (0.4, 0.4, 0.2)),
        ("balanced_mix",    (1/3, 1/3, 1/3)),
    ]

    # Also run a baseline (standard Giles) for reference.
    rows: List[Dict] = []
    for seed in range(args.seeds):
        adjacency = _load_ba(args.n, seed=seed).get_adjacency_matrix()
        # baseline
        sim_b = GPUCoupledPropagationMLMC(adjacency_matrix=adjacency, seed=seed)
        t0 = time.perf_counter()
        b = sim_b.mlmc_estimate(
            epsilon=args.epsilon, T=args.T, base_dt=args.base_dt,
            L_max=args.L_max, pilot_samples=args.pilot_samples, verbose=False,
        )
        b["runtime_s"] = time.perf_counter() - t0
        rows.append({
            "config": "baseline_giles", "seed": seed,
            "gamma_C": np.nan, "gamma_V": np.nan, "gamma_S": np.nan,
            "total_cost": b["total_cost"], "runtime_s": b["runtime_s"],
            "estimate": b["estimate"], "ci_lower": b["ci_lower"], "ci_upper": b["ci_upper"],
            "ci_width": b["ci_upper"] - b["ci_lower"],
        })

        for name, w in configs:
            r = _run_one(adjacency, weights=w, epsilon=args.epsilon, T=args.T,
                         base_dt=args.base_dt, L_max=args.L_max,
                         pilot_samples=args.pilot_samples, seed=seed)
            rows.append({
                "config": name, "seed": seed,
                "gamma_C": w[0], "gamma_V": w[1], "gamma_S": w[2],
                "total_cost": r["total_cost"], "runtime_s": r["runtime_s"],
                "estimate": r["estimate"], "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"],
                "ci_width": r["ci_upper"] - r["ci_lower"],
            })

    csv_path = args.output / "weight_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Aggregate per-config means
    summary: Dict[str, Dict[str, float]] = {}
    for cfg in {r["config"] for r in rows}:
        sub = [r for r in rows if r["config"] == cfg]
        summary[cfg] = {
            "n_seeds": len(sub),
            "total_cost_mean": float(np.mean([r["total_cost"] for r in sub])),
            "total_cost_std":  float(np.std([r["total_cost"] for r in sub], ddof=1) if len(sub) > 1 else 0.0),
            "runtime_mean":    float(np.mean([r["runtime_s"] for r in sub])),
            "ci_width_mean":   float(np.mean([r["ci_width"] for r in sub])),
            "ci_width_std":    float(np.std([r["ci_width"] for r in sub], ddof=1) if len(sub) > 1 else 0.0),
        }
    json_path = args.output / "weight_sweep_summary.json"
    json_path.write_text(json.dumps({"epsilon": args.epsilon, "n_nodes": args.n,
                                     "T": args.T, "L_max": args.L_max,
                                     "pilot_samples": args.pilot_samples,
                                     "summary": summary}, indent=2))

    print(f"\n=== ANA weight-sensitivity sweep (BA n={args.n}, eps={args.epsilon}, {args.seeds} seeds) ===")
    print(f"{'config':<20}{'cost_mean':>14}{'cost_sd':>10}{'rt_mean(s)':>12}{'ci_width':>12}")
    order = ["baseline_giles", "centrality_only", "variance_only", "sla_only", "balanced_mix", "default_mix"]
    for cfg in order:
        if cfg not in summary: continue
        s = summary[cfg]
        print(f"{cfg:<20}{s['total_cost_mean']:>14.0f}{s['total_cost_std']:>10.0f}{s['runtime_mean']:>12.3f}{s['ci_width_mean']:>12.4f}")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
