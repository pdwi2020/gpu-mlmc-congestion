"""Real-trace validation: drive coupled SDE with MAWI (or Pareto surrogate) arrivals.

Pillar 2C of the JNCA/TOMACS revision. Demonstrates the ANA-MLMC pipeline on
realistic, non-Poisson traffic, comparing predicted P95/P99 queue length against
the burst structure observed in the trace.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def _try_mawi(date: str, max_mb: int, hhmm: str = "1400") -> Tuple[np.ndarray, str]:
    """Attempt MAWI download; return (arrival_series, source_label).

    Uses the canonical samplepoint-F archive layout
    (https://mawi.wide.ad.jp/mawi/samplepoint-F/<YYYY>/<YYYYMMDDHHMM>.pcap.gz),
    which differs from the legacy ~agurim mirror baked into the default loader.
    """
    try:
        from datasets.mawi.loader import MAWITraceProcessor
    except ImportError as e:
        return _pareto_surrogate(), f"surrogate-mawi-import-failed: {e}"
    try:
        # Override mirror to the templated form so the URL builder matches
        # samplepoint-F/<YYYY>/<YYYYMMDDHHMM>.pcap.gz (no YYYYMMDD subdirectory).
        mirror = "https://mawi.wide.ad.jp/mawi/samplepoint-F/{YYYY}/{filename}"
        proc = MAWITraceProcessor(
            data_dir=ROOT / "datasets" / "mawi",
            mirror_url=mirror,
            max_download_bytes=max_mb * 1024 * 1024,
        )
        path = proc.download_trace(date, sample_point=hhmm)
        arrivals_dict, metadata = proc.extract_arrival_series(path, bin_seconds=1.0, max_flows=5)
        if not arrivals_dict:
            return _pareto_surrogate(), f"surrogate-mawi-empty-trace-{date}{hhmm}"
        ts = proc.to_lambda_series(arrivals_dict, scale=1.0)
        n_kept_flows = ts.shape[1] if ts.ndim == 2 else 1
        n_bins = ts.shape[0]
        return ts, f"mawi-samplepoint-F-{date}{hhmm} ({metadata['total_packets']}pkts, {n_kept_flows}flows, {n_bins}bins)"
    except Exception as e:
        logger.warning(f"MAWI download/parse failed: {type(e).__name__}: {e}; using Pareto surrogate.")
        return _pareto_surrogate(), f"surrogate-mawi-unreachable: {type(e).__name__}: {str(e)[:120]}"


def _pareto_surrogate(n_steps: int = 600, n_flows: int = 5, seed: int = 7) -> np.ndarray:
    """Realistic surrogate: Pareto burst superposition mimicking MAWI tails."""
    rng = np.random.default_rng(seed)
    base = 1.0
    out = np.zeros((n_steps, n_flows))
    for k in range(n_flows):
        # Stationary Pareto-tailed arrival rate with periodic component
        baseline = base * (1.0 + 0.3 * np.sin(2 * np.pi * np.arange(n_steps) / 200 + k))
        bursts = rng.pareto(2.5, n_steps) * 0.15 * base
        bursts[bursts < 0] = 0.0
        out[:, k] = baseline + bursts
    return out


def _drive_sde(arrivals: np.ndarray, mu: float, sigma: float, T: float, dt: float, seed: int) -> np.ndarray:
    """Reflected SDE driven by lambda(t) per flow. Returns Q(t) shape (n_steps+1, n_flows)."""
    rng = np.random.default_rng(seed)
    n_steps = arrivals.shape[0]
    n_flows = arrivals.shape[1]
    Q = np.zeros((n_steps + 1, n_flows))
    sqrt_dt = np.sqrt(dt)
    for t in range(n_steps):
        noise = rng.normal(0.0, sqrt_dt, n_flows)
        Q[t + 1] = np.maximum(0.0, Q[t] + (arrivals[t] - mu) * dt + sigma * noise)
    return Q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mawi-date", default="20240101")
    parser.add_argument("--max-mb", type=int, default=20)
    parser.add_argument("--mu", type=float, default=1.5)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--T", type=float, default=600.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--n-mlmc-paths", type=int, default=200)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "real_trace")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    arrivals, source = _try_mawi(args.mawi_date, args.max_mb)
    logger.info(f"source: {source}, arrivals shape: {arrivals.shape}")
    arrivals = np.asarray(arrivals, dtype=np.float64)
    if arrivals.ndim == 1:
        arrivals = arrivals[:, None]
    # Scale-invariance: normalize so mean arrival rate per flow is 1.0; the SDE
    # operates in dimensionless time. This lets us use the same mu/sigma for
    # synthetic surrogates and real packet-count traces.
    flow_means = arrivals.mean(axis=0)
    flow_means[flow_means == 0.0] = 1.0
    arrivals = arrivals / flow_means
    n_steps, n_flows = arrivals.shape
    logger.info(f"flows: {n_flows}, steps: {n_steps}, normalized to unit-mean per flow")

    # Run an ensemble of paths to build the predicted CI envelope.
    ensemble: List[np.ndarray] = []
    started = time.perf_counter()
    for seed in range(args.n_mlmc_paths):
        Q = _drive_sde(arrivals, args.mu, args.sigma, args.T, args.dt, seed=seed)
        ensemble.append(Q.max(axis=0))  # peak queue per flow per path
    runtime = time.perf_counter() - started
    peak = np.array(ensemble)  # (n_paths, n_flows)

    p95 = np.percentile(peak, 95, axis=0)
    p99 = np.percentile(peak, 99, axis=0)
    mean = peak.mean(axis=0)
    ci_lo = np.percentile(peak, 2.5, axis=0)
    ci_hi = np.percentile(peak, 97.5, axis=0)

    # Burst statistics from the input trace itself, for narrative.
    arr_p95 = np.percentile(arrivals, 95, axis=0)
    arr_p99 = np.percentile(arrivals, 99, axis=0)

    summary = {
        "source": source,
        "n_flows": int(n_flows),
        "n_steps": int(n_steps),
        "n_mlmc_paths": int(args.n_mlmc_paths),
        "mu": args.mu,
        "sigma": args.sigma,
        "runtime_s": runtime,
        "per_flow": [
            {
                "flow": k,
                "arrivals_p95": float(arr_p95[k]),
                "arrivals_p99": float(arr_p99[k]),
                "queue_mean_peak": float(mean[k]),
                "queue_p95": float(p95[k]),
                "queue_p99": float(p99[k]),
                "queue_ci95": [float(ci_lo[k]), float(ci_hi[k])],
            }
            for k in range(n_flows)
        ],
    }
    out_json = args.output / "real_trace_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    np.savez(args.output / "real_trace_arrivals.npz",
             arrivals=arrivals, peak_queue=peak)
    logger.info(f"Wrote: {out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
