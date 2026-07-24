#!/usr/bin/env python3
"""Benchmark rare-event / tail-probability estimators for the IEEE Access paper.

Answers the question that decides the paper's positioning: for a high-variance
estimand -- the tail overflow probability P(Q_max >= B) -- does the paper's
GPU-MLMC importance sampler genuinely reduce the *work* needed to reach a target
accuracy, unlike the near-deterministic mean queue occupancy (where accuracy-
matched single-level MC is cheaper, see results/pod_run/crossover_pod/)?

Estimators (all validated against closed-form answers in tests/test_rare_event.py):
  * plain_monte_carlo          -- single-level MC, the honest baseline
  * gpu_mlmc_is                -- the paper's exponential-twisting IS walk
  * adaptive_multilevel_splitting (Cerou-Guyader)
  * cross_entropy_is           (Rubinstein-Kroese)
  * fixed_level_splitting      -- classical SMC reference

Ground truth: chapman_kolmogorov_reference (exact solve of the discretised chain).

Scoring (the metric that actually decides which method is better):
  work-normalised efficiency = relative_RMSE^2 * work   (lower is better;
  budget-independent, the standard rare-event efficiency measure).
Also reports ESS, fraction of non-degenerate runs (plain MC returns 0 at large
B -- that IS the result), and wall-clock.

Run:  python scripts/run_rare_event_comparators.py --quick
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from simulation.rare_event import (  # noqa: E402
    QueueConfig,
    plain_monte_carlo,
    adaptive_multilevel_splitting,
    cross_entropy_is,
    fixed_level_splitting,
    gpu_mlmc_is,
    fw_optimal_delta,
    chapman_kolmogorov_reference,
)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def provenance(cfg: dict) -> dict:
    return {
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "config": cfg,
    }


def relative_rmse(estimates, ref):
    """RMSE of estimates about the exact reference, relative to the reference."""
    est = np.asarray([e for e in estimates if math.isfinite(e) and e > 0.0])
    if len(est) == 0 or ref <= 0:
        return float("inf"), 0
    rmse = math.sqrt(np.mean((est - ref) ** 2))
    return rmse / ref, len(est)


# each competitor tuned at least as carefully as the incumbent (documented)
def run_method(name, cfg, B, seed, budget):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    if name == "plain_mc":
        r = plain_monte_carlo(cfg, B, rng, n_paths=budget)
    elif name == "gpu_mlmc_is":
        # FW-optimal tilt so the incumbent is not hobbled (the published
        # delta=0.03 is far too weak at large B; fw_optimal_delta fixes that)
        delta = fw_optimal_delta(cfg, B)
        r = gpu_mlmc_is(cfg, B, n_paths=budget, delta=delta, device="cpu",
                        self_normalised=True, seed=seed)
    elif name == "ams":
        r = adaptive_multilevel_splitting(cfg, B, rng,
                                          n_particles=max(200, budget // 20),
                                          survival_frac=0.5, max_stages=200)
    elif name == "cross_entropy":
        r = cross_entropy_is(cfg, B, rng, n_pilot=max(500, budget // 10),
                             n_final=budget, elite_frac=0.1, max_iter=20,
                             refine_iters=3, self_normalised=True)
    elif name == "fixed_splitting":
        r = fixed_level_splitting(cfg, B, rng,
                                  n_particles=max(200, budget // 20),
                                  n_levels=8, levels=None, level_mode="linear")
    else:
        raise ValueError(name)
    r["wall_s"] = time.perf_counter() - t0
    return r


METHODS = ["plain_mc", "gpu_mlmc_is", "ams", "cross_entropy", "fixed_splitting"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--B", type=float, nargs="+", default=[10.0, 15.0, 20.0, 25.0])
    ap.add_argument("--rho", type=float, default=0.97)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--budget", type=int, default=50000,
                    help="nominal path budget per method per rep")
    ap.add_argument("--out", type=str, default=str(REPO / "results" / "rare_event_comparators"))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.seeds = [0, 1, 2]
        args.B = [10.0, 15.0]
        args.budget = 4000

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg_dict = vars(args).copy()
    prov = provenance(cfg_dict)

    print("=" * 90)
    print("Rare-event / tail-probability estimator comparison")
    print(f"  rho={args.rho} mu={args.mu} T={args.T} dt={args.dt} budget={args.budget}")
    print(f"  B={args.B}  seeds={len(args.seeds)}  methods={METHODS}")
    print("=" * 90)

    rows = []
    summary = []
    for B in args.B:
        cfg = QueueConfig(rho=args.rho, mu=args.mu, T=args.T, dt=args.dt)
        ref = chapman_kolmogorov_reference(cfg, B)
        ref_p = ref["probability"] if isinstance(ref, dict) else float(ref)
        print(f"\nB={B:g}  exact P(Q_max>={B:g}) = {ref_p:.3e}")
        print(f"  {'method':>16} {'mean_est':>11} {'relRMSE':>9} "
              f"{'work':>9} {'eff(RMSE^2*W)':>14} {'nondeg':>7} {'wall_s':>8}")
        for name in METHODS:
            ests, works, walls, ess_list, ndeg = [], [], [], [], 0
            for s in args.seeds:
                r = run_method(name, cfg, B, s, args.budget)
                ests.append(r["estimate"])
                works.append(r.get("work", args.budget))
                walls.append(r.get("wall_s", 0.0))
                if "ess" in r:
                    ess_list.append(r["ess"])
                if not r.get("degenerate", False):
                    ndeg += 1
                rows.append({
                    "B": B, "method": name, "seed": s,
                    "estimate": r["estimate"], "work": r.get("work", args.budget),
                    "degenerate": r.get("degenerate", False),
                    "ess": r.get("ess", ""), "wall_s": r.get("wall_s", 0.0),
                    "ref": ref_p,
                })
            rel, n_ok = relative_rmse(ests, ref_p)
            work_mean = float(np.mean(works))
            eff = (rel ** 2) * work_mean if math.isfinite(rel) else float("inf")
            mean_est = float(np.mean([e for e in ests if math.isfinite(e) and e > 0])) if n_ok else float("nan")
            frac_ndeg = ndeg / len(args.seeds)
            summary.append({
                "B": B, "method": name, "mean_estimate": mean_est,
                "rel_rmse": rel, "work_mean": work_mean,
                "work_norm_efficiency": eff, "frac_nondegenerate": frac_ndeg,
                "wall_s_mean": float(np.mean(walls)),
                "ess_mean": float(np.mean(ess_list)) if ess_list else "",
                "ref": ref_p,
            })
            print(f"  {name:>16} {mean_est:11.3e} {rel:9.3f} "
                  f"{work_mean:9.0f} {eff:14.3e} {frac_ndeg:7.2f} {np.mean(walls):8.3f}")

    # write outputs
    with open(out / "raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(out / "comparators.json", "w") as f:
        json.dump({"provenance": prov, "summary": summary}, f, indent=2, default=str)

    # verdict: does gpu_mlmc_is beat plain MC on work-normalised efficiency?
    print("\n" + "=" * 90)
    print("VERDICT (work-normalised efficiency, lower=better; does IS-MLMC beat plain MC?)")
    for B in args.B:
        srows = {s["method"]: s for s in summary if s["B"] == B}
        mc = srows["plain_mc"]["work_norm_efficiency"]
        mlmc = srows["gpu_mlmc_is"]["work_norm_efficiency"]
        best = min(summary, key=lambda s: s["work_norm_efficiency"] if s["B"] == B and math.isfinite(s["work_norm_efficiency"]) else float("inf"))
        winner = min((s for s in summary if s["B"] == B and math.isfinite(s["work_norm_efficiency"])),
                     key=lambda s: s["work_norm_efficiency"], default=None)
        ratio = (mc / mlmc) if (math.isfinite(mc) and math.isfinite(mlmc) and mlmc > 0) else float("inf")
        wname = winner["method"] if winner else "none"
        print(f"  B={B:g}: IS-MLMC/plain-MC efficiency gain = {ratio:.2e}x   "
              f"best method = {wname}")
    print(f"\nOutputs: {out}/summary.csv, comparators.json")


if __name__ == "__main__":
    main()
