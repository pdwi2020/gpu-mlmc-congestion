"""
Scalability experiment: GPU-MLMC runtime vs network size n.

Measures wall-clock runtime for n in {50, 100, 200, 500} nodes using the
CongestionPropagationSDE (coupled model) and saves results + figures.

Run from project root:
    python scripts/run_scaling_experiment.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try GPU-coupled MLMC, fall back to CPU-only numpy simulation
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if not TORCH_AVAILABLE:
        print("[INFO] CUDA not available — running in CPU mode (torch.device('cpu'))")
except ImportError:
    TORCH_AVAILABLE = False

sys.path.insert(0, str(ROOT / "experiments"))

# ------------------------------------------------------------------
# Minimal coupled-SDE MLMC that works without the full experiment stack
# ------------------------------------------------------------------
def _coupled_mlmc_runtime(n_nodes: int, epsilon: float = 0.05,
                           T: float = 1.0, p: float = 0.15,
                           influence: float = 0.1, decay: float = 0.5,
                           noise: float = 0.1, L_max: int = 4,
                           pilot: int = 50, seed: int = 42) -> dict:
    """Run GPU (or CPU) coupled MLMC and return timing + cost info."""
    rng = np.random.default_rng(seed)

    # Erdos-Renyi adjacency
    adj = (rng.random((n_nodes, n_nodes)) < p).astype(np.float32)
    np.fill_diagonal(adj, 0)
    adj = np.maximum(adj, adj.T)           # symmetrise

    degrees = adj.sum(axis=1).clip(min=1)
    influence_mat = (adj / degrees[:, None]) * influence

    def em_step_np(c, dt, dw):
        drift = influence_mat @ c - decay * c
        return np.maximum(0.0, c + drift * dt + noise * dw)

    def run_level_np(level, n_samples, base_dt=0.1, refinement=2):
        dt_fine = base_dt / (refinement ** level)
        n_steps_fine = int(T / dt_fine)
        if level == 0:
            c = np.zeros((n_nodes, n_samples), dtype=np.float32)
            for _ in range(n_steps_fine):
                dw = rng.standard_normal((n_nodes, n_samples)).astype(np.float32) * (dt_fine ** 0.5)
                c = em_step_np(c, dt_fine, dw)
            return c.mean(axis=0), np.zeros(n_samples, np.float32)
        else:
            M = refinement
            dt_coarse = dt_fine * M
            n_steps_c = int(T / dt_coarse)
            c_f = np.zeros((n_nodes, n_samples), dtype=np.float32)
            c_c = np.zeros((n_nodes, n_samples), dtype=np.float32)
            for _ in range(n_steps_c):
                dw_sum = np.zeros((n_nodes, n_samples), dtype=np.float32)
                for _ in range(M):
                    dw = rng.standard_normal((n_nodes, n_samples)).astype(np.float32) * (dt_fine ** 0.5)
                    c_f = em_step_np(c_f, dt_fine, dw)
                    dw_sum += dw
                c_c = em_step_np(c_c, dt_coarse, dw_sum)
            return c_f.mean(axis=0), c_c.mean(axis=0)

    # Try PyTorch GPU path
    use_torch = False
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        influence_t = torch.tensor(influence_mat, dtype=torch.float32, device=device)

        def em_step_t(c, dt, dw):
            drift = torch.mm(influence_t, c) - decay * c
            return torch.clamp_min(c + drift * dt + noise * dw, 0.0)

        def run_level_t(level, n_samples, base_dt=0.1, refinement=2):
            dt_fine = base_dt / (refinement ** level)
            n_steps_fine = int(T / dt_fine)
            if level == 0:
                c = torch.zeros(n_nodes, n_samples, device=device, dtype=torch.float32)
                for _ in range(n_steps_fine):
                    dw = torch.randn(n_nodes, n_samples, device=device) * (dt_fine ** 0.5)
                    c = em_step_t(c, dt_fine, dw)
                return c.mean(0).cpu().numpy(), np.zeros(n_samples, np.float32)
            else:
                M = refinement
                dt_coarse = dt_fine * M
                n_steps_c = int(T / dt_coarse)
                c_f = torch.zeros(n_nodes, n_samples, device=device, dtype=torch.float32)
                c_c = torch.zeros(n_nodes, n_samples, device=device, dtype=torch.float32)
                for _ in range(n_steps_c):
                    dw_sum = torch.zeros(n_nodes, n_samples, device=device, dtype=torch.float32)
                    for _ in range(M):
                        dw = torch.randn(n_nodes, n_samples, device=device) * (dt_fine ** 0.5)
                        c_f = em_step_t(c_f, dt_fine, dw)
                        dw_sum += dw
                    c_c = em_step_t(c_c, dt_coarse, dw_sum)
                return c_f.mean(0).cpu().numpy(), c_c.mean(0).cpu().numpy()

        run_level = run_level_t
        use_torch = True
    except Exception:
        run_level = run_level_np

    base_dt = 0.1
    # Pilot phase
    variances, costs = [], []
    for l in range(L_max + 1):
        Yf, Yc = run_level(l, pilot)
        diffs = Yf - Yc
        variances.append(float(np.var(diffs, ddof=1)))
        dt_l = base_dt / (2 ** l)
        costs.append(T / dt_l)

    sum_vc = sum(np.sqrt(v * c) for v, c in zip(variances, costs))
    optimal_N = [max(1, int(np.ceil((2.0 / epsilon ** 2) *
                                     np.sqrt(variances[l] / costs[l]) * sum_vc)))
                 if variances[l] > 0 else 1
                 for l in range(L_max + 1)]

    t0 = time.perf_counter()
    total_cost = 0.0
    for l in range(L_max + 1):
        Yf, Yc = run_level(l, optimal_N[l])
        total_cost += costs[l] * optimal_N[l]
    runtime = time.perf_counter() - t0

    return {
        'n_nodes': n_nodes,
        'runtime_s': runtime,
        'total_cost': total_cost,
        'optimal_N': optimal_N,
        'backend': 'torch-' + ('cuda' if use_torch and
                                torch.cuda.is_available() else 'cpu')
        if use_torch else 'numpy',
    }


def main():
    node_sizes = [50, 100, 200, 500]
    n_repeats = 3
    epsilon = 0.05

    print(f"Running coupled-propagation GPU-MLMC scaling experiment")
    print(f"Node sizes: {node_sizes}, epsilon={epsilon}, repeats={n_repeats}")
    print()

    results = {}
    for n in node_sizes:
        runtimes = []
        costs = []
        for rep in range(n_repeats):
            print(f"  n={n:4d}, rep={rep+1}/{n_repeats} ... ", end="", flush=True)
            r = _coupled_mlmc_runtime(n, epsilon=epsilon, seed=42 + rep)
            runtimes.append(r['runtime_s'])
            costs.append(r['total_cost'])
            print(f"{r['runtime_s']:.3f}s  [{r['backend']}]")
        results[n] = {
            'runtimes': runtimes,
            'costs': costs,
            'median_runtime': float(np.median(runtimes)),
            'min_runtime': float(np.min(runtimes)),
            'max_runtime': float(np.max(runtimes)),
            'median_cost': float(np.median(costs)),
        }

    # Save raw results
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "scaling_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    fig_dir = ROOT / "paper" / "figures"
    fig_dir.mkdir(exist_ok=True)

    ns = node_sizes
    medians = [results[n]['median_runtime'] for n in ns]
    mins = [results[n]['min_runtime'] for n in ns]
    maxs = [results[n]['max_runtime'] for n in ns]
    err_lo = [m - mn for m, mn in zip(medians, mins)]
    err_hi = [mx - m for m, mx in zip(medians, maxs)]

    # Reference O(n) and O(n^2) lines anchored at n=50
    ref_n = np.array(ns, dtype=float)
    ref_on = medians[0] * (ref_n / ns[0])
    ref_on2 = medians[0] * (ref_n / ns[0]) ** 2

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.errorbar(ns, medians,
                yerr=[err_lo, err_hi],
                fmt='o-', color='tab:blue', capsize=3,
                linewidth=1.5, markersize=5, label='GPU-MLMC (coupled SDE)')
    ax.plot(ns, ref_on, '--', color='gray', linewidth=1, label='$O(n)$ ref.', alpha=0.7)
    ax.plot(ns, ref_on2, ':', color='gray', linewidth=1, label='$O(n^2)$ ref.', alpha=0.7)
    ax.set_xlabel('Network Size $n$ (nodes)', fontsize=8)
    ax.set_ylabel('Runtime (s)', fontsize=8)
    ax.set_title('GPU-MLMC Runtime Scaling vs Network Size', fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xticks(ns)
    plt.tight_layout()
    fig.savefig(fig_dir / "scaling_n_vs_runtime.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {fig_dir / 'scaling_n_vs_runtime.png'}")

    # Cost vs n
    med_costs = [results[n]['median_cost'] for n in ns]
    fig2, ax2 = plt.subplots(figsize=(3.5, 2.8))
    ax2.plot(ns, med_costs, 's-', color='tab:orange', linewidth=1.5, markersize=5)
    ax2.set_xlabel('Network Size $n$ (nodes)', fontsize=8)
    ax2.set_ylabel('Total Cost (timestep-paths)', fontsize=8)
    ax2.set_title('GPU-MLMC Computational Cost vs Network Size', fontsize=9)
    ax2.tick_params(labelsize=7)
    ax2.set_xticks(ns)
    plt.tight_layout()
    fig2.savefig(fig_dir / "cost_vs_n.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Figure saved: {fig_dir / 'cost_vs_n.png'}")

    # Summary
    print("\nSummary:")
    print(f"{'n':>6}  {'Median (s)':>12}  {'Min':>8}  {'Max':>8}")
    for n in ns:
        r = results[n]
        print(f"{n:6d}  {r['median_runtime']:12.3f}  "
              f"{r['min_runtime']:8.3f}  {r['max_runtime']:8.3f}")


if __name__ == "__main__":
    main()
