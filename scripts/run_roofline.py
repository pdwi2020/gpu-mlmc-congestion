"""
Roofline analysis (C4).

Measures Operational Intensity (OI = FLOPs / Bytes) and Achieved Performance
(P = FLOPs / time) for the key GPU kernels in MultiGPUMLMC using the
PyTorch profiler. Identifies whether each kernel is memory-bound or compute-bound
relative to the device's roofline.

Device roofline (measured/spec):
  A100 80GB PCIe:  312 TFLOP/s FP32,  1935 GB/s HBM bandwidth  → ridge OI ≈ 161 FLOPs/B
  T4:              65 TFLOP/s FP32,   320 GB/s HBM bandwidth  → ridge OI ≈ 203 FLOPs/B

Usage (A100 RunPod):
    python scripts/run_roofline.py --device cuda --out-dir results/roofline

Usage (Colab T4):
    python scripts/run_roofline.py --device cuda --device-name T4 --out-dir results/roofline

Output:
    results/roofline/roofline_<device>.json
    results/roofline/roofline_<device>.png   (if matplotlib available)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# Roofline reference data
# ---------------------------------------------------------------------------
DEVICE_SPECS = {
    "A100": {"peak_tflops_fp32": 312.0, "bw_gbs": 1935.0},
    "T4":   {"peak_tflops_fp32":   8.1, "bw_gbs":  320.0},  # 8.1 TFLOPS FP32 (65 is FP16)
    "RTX3090": {"peak_tflops_fp32": 35.6, "bw_gbs": 936.0},
    "V100": {"peak_tflops_fp32":  14.0, "bw_gbs":  900.0},  # V100 PCIe FP32
}


def flops_em_step(n_nodes: int, n_paths: int, n_local: int) -> int:
    """Approximate FLOPs for one EM step (predictor-corrector on local partition).

    Dominant term: torch.mm(Q_global, influence_local.T)
      shape: (n_paths, n_nodes) × (n_nodes, n_local) → (n_paths, n_local)
      FLOPs: 2 * n_paths * n_nodes * n_local

    Plus: drift, noise, clamp — minor compared to matmul.
    """
    matmul = 2 * n_paths * n_nodes * n_local
    other = 6 * n_paths * n_local   # drift + noise + clamp (rough)
    return matmul + other


def bytes_em_step(n_nodes: int, n_paths: int, n_local: int) -> int:
    """Bytes read/written for one EM step (float32 = 4 bytes).

    Reads:  Q_global (n_paths × n_nodes), influence_local (n_local × n_nodes),
            Q (n_paths × n_local)
    Writes: Q_new (n_paths × n_local), Q_global update (n_paths × n_local)
    """
    bytes_per = 4  # float32
    reads = (n_paths * n_nodes + n_local * n_nodes + n_paths * n_local) * bytes_per
    writes = 2 * n_paths * n_local * bytes_per
    return reads + writes


def bench_kernel(n_nodes: int = 1000, n_paths: int = 512, n_steps: int = 50,
                 device: str = "cuda") -> dict:
    """Benchmark the core EM kernel using PyTorch profiler + manual FLOPs."""
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch not available")

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU")

    # Build representative tensors
    n_local = n_nodes  # single-GPU: local = full graph
    Q = torch.zeros(n_paths, n_local, device=device)
    Q_global = torch.zeros(n_paths, n_nodes, device=device)
    influence = torch.rand(n_local, n_nodes, device=device) * 0.1
    dt = 0.01
    sqrt_dt = dt ** 0.5
    arrivals = 1.0

    # Warmup
    for _ in range(5):
        it = torch.mm(Q_global, influence.T)
        drift = arrivals - 0.5 * Q + 0.1 * it
        noise = torch.randn_like(Q) * sqrt_dt
        Q = torch.clamp_min(Q + drift * dt + 0.1 * noise, 0.0)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed run
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        it = torch.mm(Q_global, influence.T)
        drift = arrivals - 0.5 * Q + 0.1 * it
        noise = torch.randn_like(Q) * sqrt_dt
        Q = torch.clamp_min(Q + drift * dt + 0.1 * noise, 0.0)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mb = 0.0
    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1e6

    total_flops = flops_em_step(n_nodes, n_paths, n_local) * n_steps
    total_bytes = bytes_em_step(n_nodes, n_paths, n_local) * n_steps
    oi = total_flops / total_bytes
    achieved_gflops = total_flops / elapsed / 1e9

    return {
        "kernel": "SDE_EM_step",
        "n_nodes": n_nodes,
        "n_paths": n_paths,
        "n_steps": n_steps,
        "elapsed_s": elapsed,
        "flops_total": total_flops,
        "bytes_total": total_bytes,
        "oi_flops_per_byte": oi,
        "achieved_gflops": achieved_gflops,
        "peak_mem_mb": peak_mb,
        "device": device,
    }


def classify_bound(oi: float, device_name: str) -> str:
    spec = DEVICE_SPECS.get(device_name, DEVICE_SPECS["A100"])
    ridge_oi = (spec["peak_tflops_fp32"] * 1e12) / (spec["bw_gbs"] * 1e9)
    if oi < ridge_oi * 0.5:
        return "memory-bound"
    elif oi > ridge_oi * 2.0:
        return "compute-bound"
    else:
        return "transitional"


def plot_roofline(results: list, device_name: str, out_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping roofline plot")
        return

    spec = DEVICE_SPECS.get(device_name, DEVICE_SPECS["A100"])
    peak_gflops = spec["peak_tflops_fp32"] * 1000
    bw_gbs = spec["bw_gbs"]
    ridge_oi = (spec["peak_tflops_fp32"] * 1e12) / (spec["bw_gbs"] * 1e9)

    oi_range = np.logspace(-1, 4, 200)
    roofline = np.minimum(bw_gbs * oi_range, peak_gflops)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(oi_range, roofline, 'k-', linewidth=2, label=f'{device_name} roofline')
    ax.axvline(ridge_oi, color='k', linestyle=':', alpha=0.4, label=f'Ridge OI={ridge_oi:.0f}')

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for r, c in zip(results, colors):
        oi = r["oi_flops_per_byte"]
        perf = r["achieved_gflops"]
        label = f"{r['kernel']} n={r['n_nodes']}"
        ax.scatter([oi], [perf], color=c, s=120, zorder=5, label=label)
        ax.annotate(label, (oi, perf), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Operational Intensity (FLOPs/Byte)", fontsize=12)
    ax.set_ylabel("Achieved Performance (GFLOPs/s)", fontsize=12)
    ax.set_title(f"Roofline Model — {device_name}", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Roofline analysis (C4)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-name", default="A100",
                        choices=list(DEVICE_SPECS.keys()),
                        help="Device spec for roofline reference")
    parser.add_argument("--n-nodes", type=int, nargs="+", default=[500, 1000, 5000])
    parser.add_argument("--n-paths", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--out-dir", default="results/roofline")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Auto-detect CUDA device name if possible
    device_name = args.device_name
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU: {gpu_name}")
            for k in DEVICE_SPECS:
                if k.lower() in gpu_name.lower():
                    device_name = k
                    print(f"Matched device spec: {device_name}")
                    break
    except Exception:
        pass

    print(f"Running roofline on {device_name} ({args.device})", flush=True)
    results = []
    for n in args.n_nodes:
        print(f"  n={n}...", flush=True)
        r = bench_kernel(n_nodes=n, n_paths=args.n_paths,
                         n_steps=args.n_steps, device=args.device)
        r["device_name"] = device_name
        r["bound"] = classify_bound(r["oi_flops_per_byte"], device_name)
        spec = DEVICE_SPECS.get(device_name, DEVICE_SPECS["A100"])
        r["ridge_oi"] = (spec["peak_tflops_fp32"] * 1e12) / (spec["bw_gbs"] * 1e9)
        r["peak_gflops"] = spec["peak_tflops_fp32"] * 1000
        r["efficiency_pct"] = 100 * r["achieved_gflops"] / r["peak_gflops"]
        results.append(r)
        print(f"    OI={r['oi_flops_per_byte']:.1f} FLOPs/B  "
              f"P={r['achieved_gflops']:.1f} GFLOPs/s  "
              f"bound={r['bound']}  eff={r['efficiency_pct']:.1f}%")

    out_json = os.path.join(args.out_dir, f"roofline_{device_name}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {out_json}")

    out_png = os.path.join(args.out_dir, f"roofline_{device_name}.png")
    plot_roofline(results, device_name, out_png)


if __name__ == "__main__":
    main()
