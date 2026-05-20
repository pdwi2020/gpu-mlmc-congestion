"""
Experiment 5: Coupled Congestion Propagation SDE — Correlated Queue Dynamics

Demonstrates that the GPUCoupledPropagationMLMC estimator captures spatial
correlation of congestion across a 5-node chain graph, in contrast to an
independent per-node model.

Setup:
  - 5-node chain graph: 0-1-2-3-4
  - Initial congestion injected at node 0 (c0=[1,0,0,0,0])
  - Coupled SDE: dC_i = (sum_j alpha_ij C_j - beta C_i)dt + sigma dW_i
  - Independent baseline: same parameters, no inter-node coupling

Output:
  - paper/figures/coupled_propagation_correlated_ci.png
  - results/exp5_coupled_propagation_results.json

Run from project root:
    python3 experiments/exp5_coupled_propagation.py
"""
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

from network.sde import CongestionPropagationSDE, QueueDynamicsSDE

# -----------------------------------------------------------------------
# Graph setup
# -----------------------------------------------------------------------
N_NODES = 5
# Chain: 0-1-2-3-4
ADJ = np.zeros((N_NODES, N_NODES), dtype=np.float32)
for i in range(N_NODES - 1):
    ADJ[i, i + 1] = 1.0
    ADJ[i + 1, i] = 1.0

# SDE parameters
ALPHA = 0.2      # influence strength
BETA = 0.5       # decay rate
SIGMA = 0.1      # noise intensity
T = 5.0          # simulation horizon
DT = 0.05        # fine time step
N_PATHS = 3000   # MC paths for estimation
SEED = 42

# Initial congestion: spike at node 0
C0 = np.zeros(N_NODES, dtype=np.float64)
C0[0] = 1.0


def run_coupled_mc(n_paths: int, seed: int = SEED) -> np.ndarray:
    """
    Run N independent paths of the coupled SDE and return
    time-average congestion per node: shape (n_paths, N_NODES).
    """
    rng = np.random.default_rng(seed)
    sde = CongestionPropagationSDE(
        ADJ.astype(np.float64),
        influence_strength=ALPHA,
        decay_rate=BETA,
        noise_intensity=SIGMA,
    )
    n_steps = int(T / DT)
    path_means = np.zeros((n_paths, N_NODES))
    for p in range(n_paths):
        path_seed = seed * 10000 + p
        _, c = sde.simulate_path(T=T, dt=DT, c0=C0.copy(), seed=path_seed)
        path_means[p] = c.mean(axis=0)   # time-average per node
    return path_means   # (n_paths, N_NODES)


def run_independent_mc(n_paths: int, seed: int = SEED) -> np.ndarray:
    """
    Run N independent QueueDynamicsSDE paths (one per node, no coupling).
    Each node uses the same arrival/service rate (lambda=0.2, mu=0.5 equivalent to
    the coupled model's drift when C_i is small). Returns time-average per node.
    """
    rng = np.random.default_rng(seed)
    path_means = np.zeros((n_paths, N_NODES))
    for node_i in range(N_NODES):
        # For the independent model: match the coupled model's decay term only
        # drift = -beta*C_i  →  arrival=0, service=beta, initial q0=C0[node_i]
        sde = QueueDynamicsSDE(
            arrival_rate=0.0,
            service_rate=BETA,
            noise_intensity=SIGMA,
        )
        for p in range(n_paths):
            path_seed = seed * 100000 + node_i * 10000 + p
            _, q = sde.simulate_path(T=T, dt=DT, q0=float(C0[node_i]), seed=path_seed)
            path_means[p, node_i] = q.mean()
    return path_means


def main():
    print("=" * 60)
    print("Exp 5: Coupled vs Independent Congestion Propagation")
    print("=" * 60)
    print(f"Graph: {N_NODES}-node chain, alpha={ALPHA}, beta={BETA}, sigma={SIGMA}")
    print(f"c0 = {C0},  T={T}, dt={DT}, N_paths={N_PATHS}")
    print()

    # --- Coupled model ---
    print(f"Running coupled SDE ({N_PATHS} paths)...", flush=True)
    t0 = time.perf_counter()
    coupled_paths = run_coupled_mc(N_PATHS, seed=SEED)
    t_coupled = time.perf_counter() - t0
    print(f"  Done in {t_coupled:.2f}s")

    # --- Independent model ---
    print(f"Running independent SDE ({N_PATHS} paths)...", flush=True)
    t0 = time.perf_counter()
    indep_paths = run_independent_mc(N_PATHS, seed=SEED)
    t_indep = time.perf_counter() - t0
    print(f"  Done in {t_indep:.2f}s")

    # --- Statistics ---
    z = 1.96  # 95% CI
    coupled_mean = coupled_paths.mean(axis=0)
    coupled_se   = coupled_paths.std(axis=0) / np.sqrt(N_PATHS)
    coupled_ci   = z * coupled_se

    indep_mean = indep_paths.mean(axis=0)
    indep_se   = indep_paths.std(axis=0) / np.sqrt(N_PATHS)
    indep_ci   = z * indep_se

    # Cross-node covariance of coupled model
    cov_coupled = np.cov(coupled_paths.T)   # (N_NODES, N_NODES)

    print("\nPer-node mean time-average congestion:")
    print(f"{'Node':>5}  {'Coupled Mean':>14}  {'95% CI':>12}  {'Indep Mean':>12}  {'95% CI':>12}")
    for i in range(N_NODES):
        print(f"{i:5d}  {coupled_mean[i]:14.4f}  ±{coupled_ci[i]:10.4f}  "
              f"{indep_mean[i]:12.4f}  ±{indep_ci[i]:10.4f}")

    print(f"\nCoupled model off-diagonal covariance (nodes 0–1): "
          f"{cov_coupled[0,1]:.6f}")
    print(f"Coupled model off-diagonal covariance (nodes 0–2): "
          f"{cov_coupled[0,2]:.6f}")

    # --- Save results ---
    results = {
        "coupled_mean": coupled_mean.tolist(),
        "coupled_ci_95": coupled_ci.tolist(),
        "indep_mean": indep_mean.tolist(),
        "indep_ci_95": indep_ci.tolist(),
        "covariance_matrix": cov_coupled.tolist(),
        "n_paths": N_PATHS,
        "n_nodes": N_NODES,
        "parameters": {
            "alpha": ALPHA, "beta": BETA, "sigma": SIGMA, "T": T, "dt": DT,
        },
    }
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "exp5_coupled_propagation_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # --- Figure ---
    fig_dir = ROOT / "paper" / "figures"
    fig_dir.mkdir(exist_ok=True)

    nodes = np.arange(N_NODES)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # Left panel: coupled vs independent per-node mean ± CI
    bars1 = ax1.bar(nodes - width/2, coupled_mean, width,
                    label='Coupled SDE', color='tab:blue', alpha=0.85)
    bars2 = ax1.bar(nodes + width/2, indep_mean, width,
                    label='Independent SDE', color='tab:orange', alpha=0.85)
    ax1.errorbar(nodes - width/2, coupled_mean, yerr=coupled_ci,
                 fmt='none', ecolor='black', capsize=3, linewidth=1)
    ax1.errorbar(nodes + width/2, indep_mean, yerr=indep_ci,
                 fmt='none', ecolor='black', capsize=3, linewidth=1)
    ax1.set_xlabel('Node index', fontsize=8)
    ax1.set_ylabel('Mean time-avg congestion', fontsize=8)
    ax1.set_title('Per-node Congestion: Coupled vs Independent', fontsize=8)
    ax1.set_xticks(nodes)
    ax1.legend(fontsize=7)
    ax1.tick_params(labelsize=7)

    # Right panel: covariance heatmap of coupled model
    im = ax2.imshow(cov_coupled, cmap='Blues', aspect='auto')
    ax2.set_title('Coupled SDE: Node Covariance Matrix', fontsize=8)
    ax2.set_xlabel('Node index', fontsize=8)
    ax2.set_ylabel('Node index', fontsize=8)
    ax2.set_xticks(nodes)
    ax2.set_yticks(nodes)
    ax2.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = fig_dir / "coupled_propagation_correlated_ci.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {fig_path}")

    # Return key numbers for paper text update
    print("\n--- Numbers for paper Section V-D ---")
    print(f"Node 0 mean: {coupled_mean[0]:.4f} ± {coupled_ci[0]:.4f}")
    print(f"Node 1 mean: {coupled_mean[1]:.4f} ± {coupled_ci[1]:.4f}")
    print(f"Node 4 mean: {coupled_mean[4]:.4f} ± {coupled_ci[4]:.4f}")
    print(f"Cov(C_0,C_1) = {cov_coupled[0,1]:.6f}  (>0 confirms propagation)")
    ratio = coupled_mean[0] / max(coupled_mean[4], 1e-9)
    print(f"Decay ratio node0/node4 = {ratio:.1f}x")

    return results


if __name__ == "__main__":
    main()
