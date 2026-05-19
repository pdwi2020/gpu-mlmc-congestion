"""
Python discrete-event simulation of a single bottleneck link (M/D/1 queue).

Matches the ns3_bottleneck.cc model exactly:
  - Poisson packet arrivals at rate λ pkt/s
  - Deterministic service time 1/μ s
  - Infinite buffer (DropTail)
  - Utilisation ρ = λ/μ ∈ {0.5, 0.7, 0.8, 0.9}
  - 10 seeds per ρ

Runs locally without ns-3 installation. Produces identical output CSV.
Used to generate the SDE vs. DES comparison table and CDF plot (ρ=0.8).

Usage:
    python validation/python_des_bottleneck.py
    python validation/python_des_bottleneck.py --rho 0.8 --seeds 10 --sim-time 300
"""

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from heapq import heappush, heappop

import numpy as np


# ---------------------------------------------------------------------------
# DES event types
# ---------------------------------------------------------------------------
ARRIVAL   = 0
DEPARTURE = 1


@dataclass(order=True)
class Event:
    time: float
    kind: int   # ARRIVAL or DEPARTURE


def simulate_md1(lam: float, mu: float, sim_time: float,
                 seed: int, sample_dt: float = 0.1) -> dict:
    """
    Simulate M/D/1 queue via discrete events.

    Returns dict with queue-length samples and summary statistics.
    """
    rng = np.random.default_rng(seed)
    service_time = 1.0 / mu

    heap = []
    queue_len = 0      # packets waiting + in service
    server_busy = False
    next_service_end = np.inf

    # Schedule first arrival
    t_arrival = rng.exponential(1.0 / lam)
    heappush(heap, Event(t_arrival, ARRIVAL))

    # Queue-length samples at fixed intervals
    t_sample = sample_dt
    samples = []

    t = 0.0
    while heap:
        ev = heappop(heap)
        t = ev.time
        if t > sim_time:
            break

        # Collect samples up to this event time
        while t_sample <= t:
            samples.append(queue_len)
            t_sample += sample_dt

        if ev.kind == ARRIVAL:
            queue_len += 1
            if not server_busy:
                server_busy = True
                next_service_end = t + service_time
                heappush(heap, Event(next_service_end, DEPARTURE))
            # Schedule next arrival
            next_arr = t + rng.exponential(1.0 / lam)
            if next_arr < sim_time:
                heappush(heap, Event(next_arr, ARRIVAL))

        else:  # DEPARTURE
            queue_len -= 1
            if queue_len > 0:
                next_service_end = t + service_time
                heappush(heap, Event(next_service_end, DEPARTURE))
            else:
                server_busy = False
                next_service_end = np.inf

    samples = np.array(samples, dtype=float)
    return {
        "mean":  float(samples.mean()) if len(samples) else 0.0,
        "var":   float(samples.var())  if len(samples) else 0.0,
        "p95":   float(np.percentile(samples, 95)) if len(samples) else 0.0,
        "p99":   float(np.percentile(samples, 99)) if len(samples) else 0.0,
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# TV-σ SDE comparison
# ---------------------------------------------------------------------------

def sde_queue_samples(lam: float, mu: float, T: float, n_paths: int,
                      dt: float, seed: int, warmup: float = 50.0) -> np.ndarray:
    """
    Sample the stationary queue-length distribution from the TV-σ SDE.

    Runs `n_paths` independent trajectories for T+warmup seconds, discards
    the warmup period, then pools all time-step samples across paths.
    This gives the stationary (ergodic) distribution, matching the DES
    instantaneous-queue-length samples.
    """
    rng = np.random.default_rng(seed + 99999)
    n_warmup = int(warmup / dt)
    n_steps = int(T / dt)
    sigma = np.sqrt(lam)
    d = lam - mu
    sqrt_dt = np.sqrt(dt)

    all_samples = []
    # Run in chunks to limit memory
    chunk = min(n_paths, 500)
    for start in range(0, n_paths, chunk):
        c = min(chunk, n_paths - start)
        Q = np.zeros(c)
        # Warmup
        for _ in range(n_warmup):
            Q = np.maximum(0.0, Q + d * dt + sigma * rng.normal(0.0, sqrt_dt, c))
        # Collect
        for _ in range(n_steps):
            Q = np.maximum(0.0, Q + d * dt + sigma * rng.normal(0.0, sqrt_dt, c))
            all_samples.extend(Q.tolist())

    return np.array(all_samples)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Python M/D/1 DES bottleneck")
    parser.add_argument("--mu", type=float, default=1000.0,
                        help="Service rate (pkt/s) — normalised to 1.0 for SDE comparison")
    parser.add_argument("--rhos", type=float, nargs="+",
                        default=[0.5, 0.7, 0.8, 0.9])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--sim-time", type=float, default=200.0,
                        help="DES simulation duration (s)")
    parser.add_argument("--sde-paths", type=int, default=5000)
    parser.add_argument("--sde-dt", type=float, default=0.01)
    parser.add_argument("--out-dir", default="results/ns3")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Normalise: work with μ=1.0 for SDE comparison
    mu_norm = 1.0

    csv_path = os.path.join(args.out_dir, "bottleneck_results.csv")
    summary_path = os.path.join(args.out_dir, "bottleneck_summary.json")
    cdf_path = os.path.join(args.out_dir, "cdf_rho08.png")

    all_rows = []
    summary = {}

    print(f"{'rho':>6} {'seed':>6} {'DES mean':>10} {'DES p95':>10} {'DES p99':>10}")
    print("-" * 50)

    for rho in args.rhos:
        lam = rho * mu_norm
        des_p95s, des_p99s, des_means = [], [], []

        for s in range(1, args.seeds + 1):
            r = simulate_md1(lam, mu_norm, args.sim_time, seed=s * 100 + int(rho * 100))
            des_p95s.append(r["p95"])
            des_p99s.append(r["p99"])
            des_means.append(r["mean"])
            all_rows.append({
                "rho": rho, "seed": s,
                "p95_queue": r["p95"], "p99_queue": r["p99"], "mean_queue": r["mean"],
            })
            print(f"{rho:6.2f} {s:6d} {r['mean']:10.3f} {r['p95']:10.3f} {r['p99']:10.3f}")

        # SDE comparison — instantaneous queue-length distribution (ergodic samples)
        sde_samples = sde_queue_samples(lam, mu_norm, args.sim_time,
                                        args.sde_paths, args.sde_dt, seed=42)
        sde_p95 = float(np.percentile(sde_samples, 95))
        sde_p99 = float(np.percentile(sde_samples, 99))
        des_p95_mean = float(np.mean(des_p95s))
        des_p99_mean = float(np.mean(des_p99s))

        err95 = abs(sde_p95 - des_p95_mean) / max(des_p95_mean, 1e-9) * 100
        err99 = abs(sde_p99 - des_p99_mean) / max(des_p99_mean, 1e-9) * 100
        summary[str(rho)] = {
            "des_p95_mean": des_p95_mean, "des_p99_mean": des_p99_mean,
            "sde_p95": sde_p95, "sde_p99": sde_p99,
            "err_p95_pct": err95, "err_p99_pct": err99,
        }
        print(f"  => ρ={rho}  DES P95={des_p95_mean:.2f}  SDE P95={sde_p95:.2f}  "
              f"err={err95:.1f}%  |  DES P99={des_p99_mean:.2f}  SDE P99={sde_p99:.2f}  "
              f"err={err99:.1f}%\n")

    # Write CSV (matches ns-3 output format)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rho", "seed", "p95_queue", "p99_queue", "mean_queue"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"Saved: {csv_path}")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # CDF plot at ρ=0.8
    try:
        import matplotlib
        matplotlib.rcParams.update({"font.size": 11})
        import matplotlib.pyplot as plt

        rho_plot = 0.8
        lam_plot = rho_plot * mu_norm
        rng_plot = np.random.default_rng(42)

        # DES CDF (pooled across seeds)
        des_samples = []
        for s in range(1, args.seeds + 1):
            r = simulate_md1(lam_plot, mu_norm, args.sim_time,
                             seed=s * 100 + int(rho_plot * 100))
            des_samples.extend(r["samples"].tolist())
        des_arr = np.sort(des_samples)
        des_cdf = np.arange(1, len(des_arr) + 1) / len(des_arr)

        # SDE peak distribution CDF
        sde_p = np.sort(sde_queue_samples(
            lam_plot, mu_norm, args.sim_time, args.sde_paths, args.sde_dt, seed=42))
        sde_cdf = np.arange(1, len(sde_p) + 1) / len(sde_p)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(des_arr, des_cdf, color="tab:blue", label="DES (M/D/1)", linewidth=1.5)
        ax.plot(sde_p, sde_cdf, color="tab:orange", linestyle="--",
                label=r"TV-$\sigma$ SDE", linewidth=1.5)
        ax.set_xlabel("Queue occupancy (packets)")
        ax.set_ylabel("CDF")
        ax.set_title(r"SDE vs. DES Queue CDF ($\rho=0.8$)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        fig.tight_layout()
        fig.savefig(cdf_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {cdf_path}")
    except ImportError:
        print("matplotlib not available — skipping CDF plot")


if __name__ == "__main__":
    main()
