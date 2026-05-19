"""
Hardware diversity benchmark for GPU-MLMC.

Runs the MAWI-trace benchmark (N=2000, 5 flows, 1s bins, TV-σ) on the current
machine and records runtime, throughput, peak GPU memory, and energy.

Run this script on each target device; results are appended to a shared JSON:
    results/hardware/diversity_results.json

Target devices (run separately on each):
    A100  — RunPod / Colab Pro+
    V100  — RunPod
    T4    — Colab free
    Jetson AGX Orin — edge device
    RPi4  — CPU-only baseline (no CUDA)

Usage:
    python scripts/run_hardware_diversity.py --device-label A100
    python scripts/run_hardware_diversity.py --device-label CPU_RPi4
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# Benchmark kernel: TV-σ SDE on synthetic MAWI-like arrivals
# ---------------------------------------------------------------------------

def run_benchmark(n_paths: int = 2000, n_flows: int = 5, n_bins: int = 24,
                  seed: int = 42) -> dict:
    """
    TV-σ SDE on synthetic MAWI-like traffic (N paths, 5 flows, 24 time bins).
    Returns timing and peak-queue P95/P99.
    """
    rng = np.random.default_rng(seed)
    # Synthetic arrivals: Poisson with time-varying rate
    t = np.linspace(0, 24, n_bins)
    lam_base = 1.0 + 0.5 * np.sin(2 * np.pi * t / 24)  # diurnal
    arrivals = rng.poisson(lam_base[:, None] * np.ones((1, n_flows))).astype(float)
    arrivals /= np.maximum(arrivals.mean(axis=0), 1e-9)

    mu = 1.5
    dt = 1.0

    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arr_t = torch.tensor(arrivals, dtype=torch.float32, device=device)
        Q = torch.zeros(n_paths, n_flows, device=device)
        peak = Q.clone()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for step in range(n_bins):
            sigma = torch.sqrt(torch.clamp(arr_t[step], min=1e-6))
            dQ = (arr_t[step] - mu) * dt + sigma * torch.randn_like(Q)
            Q = torch.clamp(Q + dQ, min=0.0)
            peak = torch.maximum(peak, Q)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak_np = peak.cpu().numpy()
        gpu_mem_mb = 0.0
        if device.type == "cuda":
            gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        backend = f"torch-{device.type}"

    except ImportError:
        # Pure NumPy fallback
        Q = np.zeros((n_paths, n_flows))
        peak = np.zeros_like(Q)
        t0 = time.perf_counter()
        for step in range(n_bins):
            sigma = np.sqrt(np.maximum(arrivals[step], 1e-6))
            dQ = (arrivals[step] - mu) * dt + sigma * rng.normal(0.0, 1.0, (n_paths, n_flows))
            Q = np.maximum(0.0, Q + dQ)
            peak = np.maximum(peak, Q)
        elapsed = time.perf_counter() - t0
        peak_np = peak
        gpu_mem_mb = 0.0
        backend = "numpy-cpu"

    p95 = float(np.percentile(peak_np, 95))
    p99 = float(np.percentile(peak_np, 99))
    throughput = n_paths * n_bins / elapsed  # path-steps per second

    return {
        "elapsed_s": elapsed,
        "throughput_paths_per_s": throughput,
        "peak_gpu_mem_mb": gpu_mem_mb,
        "p95": p95,
        "p99": p99,
        "backend": backend,
    }


def get_gpu_info() -> dict:
    """Query nvidia-smi for GPU name and power draw."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,power.draw,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip().split("\n")[0].split(", ")
        return {
            "gpu_name": out[0].strip(),
            "power_draw_w": float(out[1]) if out[1].strip() != "N/A" else None,
            "gpu_mem_used_mb": float(out[2]),
            "gpu_mem_total_mb": float(out[3]),
        }
    except Exception:
        return {"gpu_name": "N/A (no GPU)", "power_draw_w": None}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hardware diversity benchmark")
    parser.add_argument("--device-label", default=None,
                        help="Human label for this device (e.g. A100, T4, RPi4)")
    parser.add_argument("--n-paths", type=int, default=2000)
    parser.add_argument("--n-flows", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=24)
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Repeat runs to get stable timing")
    parser.add_argument("--out-dir", default="results/hardware")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Auto-detect device label
    if args.device_label is None:
        gpu_info = get_gpu_info()
        args.device_label = gpu_info.get("gpu_name", "CPU")

    print(f"Benchmarking on: {args.device_label}")
    print(f"  N_paths={args.n_paths}, N_flows={args.n_flows}, N_bins={args.n_bins}")
    print(f"  Platform: {platform.platform()}")

    gpu_info = get_gpu_info()

    # Warm-up run (not recorded)
    run_benchmark(args.n_paths, args.n_flows, args.n_bins)

    # Timed runs
    times, throughputs, mem_mbs = [], [], []
    for i in range(args.n_runs):
        r = run_benchmark(args.n_paths, args.n_flows, args.n_bins, seed=i)
        times.append(r["elapsed_s"])
        throughputs.append(r["throughput_paths_per_s"])
        mem_mbs.append(r["peak_gpu_mem_mb"])
        print(f"  Run {i+1}: {r['elapsed_s']:.3f}s  "
              f"{r['throughput_paths_per_s']:.0f} path-steps/s  "
              f"GPU-mem={r['peak_gpu_mem_mb']:.1f}MB  "
              f"P95={r['p95']:.2f}  P99={r['p99']:.2f}")

    elapsed_mean = float(np.mean(times))
    power_w = gpu_info.get("power_draw_w")
    energy_j = (elapsed_mean * power_w) if power_w else None

    result = {
        "device_label": args.device_label,
        "gpu_name": gpu_info.get("gpu_name"),
        "platform": platform.platform(),
        "n_paths": args.n_paths,
        "n_flows": args.n_flows,
        "n_bins": args.n_bins,
        "elapsed_mean_s": elapsed_mean,
        "elapsed_std_s": float(np.std(times)),
        "throughput_mean": float(np.mean(throughputs)),
        "peak_gpu_mem_mb": float(np.mean(mem_mbs)),
        "power_draw_w": power_w,
        "energy_j": energy_j,
        "backend": r["backend"],
    }

    print(f"\nSummary: {elapsed_mean:.3f}s ± {np.std(times):.3f}s  |  "
          f"throughput={np.mean(throughputs):.0f} path-steps/s  |  "
          f"power={power_w}W  |  energy={energy_j:.2f}J" if energy_j else
          f"\nSummary: {elapsed_mean:.3f}s  |  throughput={np.mean(throughputs):.0f} path-steps/s")

    # Append to shared results file
    out_path = os.path.join(args.out_dir, "diversity_results.json")
    existing = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    # Replace entry for this device if it already exists
    existing = [e for e in existing if e.get("device_label") != args.device_label]
    existing.append(result)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved: {out_path}")

    # Print LaTeX table row
    mem_str = f"{result['peak_gpu_mem_mb']:.0f}" if result['peak_gpu_mem_mb'] > 0 else "N/A"
    pw_str = f"{power_w:.0f}" if power_w else "N/A"
    ej_str = f"{energy_j:.1f}" if energy_j else "N/A"
    print(f"\nLaTeX row:")
    print(f"  {args.device_label} & {elapsed_mean:.2f} & "
          f"{np.mean(throughputs):.0f} & {mem_str} & {pw_str} & {ej_str} \\\\")


if __name__ == "__main__":
    main()
