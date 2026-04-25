"""Deeper MAWI validation: framework SDE vs Poisson G/D/1 reference simulator.

Pillar 2C upgrade. Compares the reflected SDE used by the framework against a
discrete-event Poisson G/D/1 queue (the natural micro-scale analogue) on the
same per-flow arrival series, and reports relative error on P95/P99 queue
length per flow. Finer time bins (0.1 s) and longer windows than the v1 demo.
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


def _load_mawi(date: str, hhmm: str, max_mb: int, bin_seconds: float, max_flows: int) -> Tuple[np.ndarray, str]:
    """Load MAWI arrivals at fine bin resolution; return (n_bins, n_flows) array."""
    from datasets.mawi.loader import MAWITraceProcessor

    mirror = "https://mawi.wide.ad.jp/mawi/samplepoint-F/{YYYY}/{filename}"
    proc = MAWITraceProcessor(
        data_dir=ROOT / "datasets" / "mawi",
        mirror_url=mirror,
        max_download_bytes=max_mb * 1024 * 1024,
    )
    path = proc.download_trace(date, sample_point=hhmm)
    arrivals_dict, metadata = proc.extract_arrival_series(
        path, bin_seconds=bin_seconds, max_flows=max_flows
    )
    if not arrivals_dict:
        raise RuntimeError("No arrivals extracted from MAWI trace")
    ts = proc.to_lambda_series(arrivals_dict, scale=1.0)
    return ts, f"mawi-samplepoint-F-{date}{hhmm}-{metadata['total_packets']}pkts-{ts.shape[0]}bins-{ts.shape[1]}flows"


def reflected_sde_paths(arrivals: np.ndarray, mu: float, sigma: float, dt: float, n_paths: int, seed: int) -> np.ndarray:
    """Vectorised reflected-SDE ensemble. Returns peak queue per (path, flow)."""
    rng = np.random.default_rng(seed)
    n_steps, n_flows = arrivals.shape
    Q = np.zeros((n_paths, n_flows), dtype=np.float64)
    peak = np.zeros((n_paths, n_flows), dtype=np.float64)
    sqrt_dt = np.sqrt(dt)
    for t in range(n_steps):
        noise = rng.normal(0.0, sqrt_dt, size=(n_paths, n_flows))
        Q = np.maximum(0.0, Q + (arrivals[t][None, :] - mu) * dt + sigma * noise)
        peak = np.maximum(peak, Q)
    return peak


def des_poisson_paths(arrivals: np.ndarray, mu: float, dt: float, n_paths: int, seed: int) -> np.ndarray:
    """Reference DES: Poisson(lambda * dt) arrivals, deterministic serve at mu * dt.

    This is the discrete-event analogue: a G/D/1-style queue where arrivals
    are Poisson with time-varying rate read from the trace and service
    capacity is constant. Queue is integer-valued; return per-(path, flow)
    peak in the same units as the SDE (so we can compare directly we then
    rescale the integer queue back to 'arrival-per-bin' units, which is how
    the SDE measures Q).
    """
    rng = np.random.default_rng(seed + 10_000)  # different seed offset
    n_steps, n_flows = arrivals.shape
    Q = np.zeros((n_paths, n_flows), dtype=np.float64)
    peak = np.zeros((n_paths, n_flows), dtype=np.float64)
    service_per_step = mu * dt
    for t in range(n_steps):
        # Arrivals in this dt-window are Poisson with rate (lambda(t) * dt).
        arr = rng.poisson(np.maximum(arrivals[t][None, :] * dt, 0.0), size=(n_paths, n_flows)).astype(np.float64)
        Q = np.maximum(0.0, Q + arr - service_per_step)
        peak = np.maximum(peak, Q)
    return peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mawi-date", default="20240101")
    parser.add_argument("--mawi-hhmm", default="1400")
    parser.add_argument("--max-mb", type=int, default=200)
    parser.add_argument("--bin-seconds", type=float, default=0.1)
    parser.add_argument("--max-flows", type=int, default=10)
    parser.add_argument("--mu", type=float, default=1.5)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--n-paths", type=int, default=300)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "real_trace_deep")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    arrivals, source = _load_mawi(args.mawi_date, args.mawi_hhmm, args.max_mb, args.bin_seconds, args.max_flows)
    arrivals = np.asarray(arrivals, dtype=np.float64)
    if arrivals.ndim == 1:
        arrivals = arrivals[:, None]
    flow_means = arrivals.mean(axis=0)
    flow_means[flow_means == 0.0] = 1.0
    arrivals = arrivals / flow_means
    n_bins, n_flows = arrivals.shape
    logger.info(f"source: {source}; bins={n_bins}, flows={n_flows}")

    started = time.perf_counter()
    sde_peak = reflected_sde_paths(arrivals, args.mu, args.sigma, dt=args.bin_seconds,
                                   n_paths=args.n_paths, seed=42)
    sde_runtime = time.perf_counter() - started

    started = time.perf_counter()
    des_peak = des_poisson_paths(arrivals, args.mu, dt=args.bin_seconds,
                                 n_paths=args.n_paths, seed=42)
    des_runtime = time.perf_counter() - started

    per_flow = []
    for k in range(n_flows):
        sde_p95, sde_p99 = float(np.percentile(sde_peak[:, k], 95)), float(np.percentile(sde_peak[:, k], 99))
        des_p95, des_p99 = float(np.percentile(des_peak[:, k], 95)), float(np.percentile(des_peak[:, k], 99))
        rel95 = abs(sde_p95 - des_p95) / max(des_p95, 1e-9) * 100
        rel99 = abs(sde_p99 - des_p99) / max(des_p99, 1e-9) * 100
        arr_p95 = float(np.percentile(arrivals[:, k], 95))
        arr_p99 = float(np.percentile(arrivals[:, k], 99))
        per_flow.append({
            "flow": k,
            "arrivals_p95": arr_p95,
            "arrivals_p99": arr_p99,
            "sde_p95": sde_p95, "sde_p99": sde_p99,
            "des_p95": des_p95, "des_p99": des_p99,
            "rel_err_p95_pct": rel95,
            "rel_err_p99_pct": rel99,
        })

    summary = {
        "source": source,
        "n_bins": int(n_bins),
        "n_flows": int(n_flows),
        "n_paths": int(args.n_paths),
        "bin_seconds": args.bin_seconds,
        "mu": args.mu, "sigma": args.sigma,
        "sde_runtime_s": sde_runtime,
        "des_runtime_s": des_runtime,
        "per_flow": per_flow,
        "median_rel_err_p95_pct": float(np.median([f["rel_err_p95_pct"] for f in per_flow])),
        "median_rel_err_p99_pct": float(np.median([f["rel_err_p99_pct"] for f in per_flow])),
    }
    out_path = args.output / "deep_validation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    np.savez(args.output / "deep_arrays.npz", arrivals=arrivals, sde_peak=sde_peak, des_peak=des_peak)

    print(f"\n=== Deep MAWI validation summary ===")
    print(f"source: {source}")
    print(f"n_bins={n_bins} (window={n_bins*args.bin_seconds:.1f}s), n_flows={n_flows}, n_paths={args.n_paths}")
    print(f"SDE runtime: {sde_runtime:.2f}s   DES runtime: {des_runtime:.2f}s")
    print(f"\n{'flow':<6}{'arr_p99':>10}{'SDE_p95':>10}{'DES_p95':>10}{'err95(%)':>10}{'SDE_p99':>10}{'DES_p99':>10}{'err99(%)':>10}")
    for f in per_flow:
        print(f"{f['flow']:<6}{f['arrivals_p99']:>10.3f}{f['sde_p95']:>10.3f}{f['des_p95']:>10.3f}{f['rel_err_p95_pct']:>10.2f}{f['sde_p99']:>10.3f}{f['des_p99']:>10.3f}{f['rel_err_p99_pct']:>10.2f}")
    print(f"\nMedian rel.err P95: {summary['median_rel_err_p95_pct']:.2f}%   P99: {summary['median_rel_err_p99_pct']:.2f}%")


if __name__ == "__main__":
    main()
