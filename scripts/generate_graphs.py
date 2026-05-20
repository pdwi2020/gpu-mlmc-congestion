"""
Synthetic graph generator for GPU-MLMC experiments.

Generates Erdős-Rényi and Barabási-Albert random graphs and exports them as:
  - NumPy adjacency matrix (.npy)
  - Edge list (.csv)
  - JSON metadata

Usage:
    python scripts/generate_graphs.py --model er --n 500 --p 0.05 --out results/graphs/
    python scripts/generate_graphs.py --model ba --n 500 --m 2 --out results/graphs/
    python scripts/generate_graphs.py --model both --n 500 --out results/graphs/
"""

import argparse
import json
import os

import numpy as np

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------

def erdos_renyi(n: int, p: float, seed: int = 42) -> np.ndarray:
    """Adjacency matrix of an Erdős-Rényi G(n, p) graph."""
    if _HAS_NX:
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        return nx.to_numpy_array(G, dtype=float)
    rng = np.random.default_rng(seed)
    adj = (rng.random((n, n)) < p).astype(float)
    adj = np.triu(adj, k=1)
    adj = adj + adj.T
    np.fill_diagonal(adj, 0.0)
    return adj


def barabasi_albert(n: int, m: int, seed: int = 42) -> np.ndarray:
    """Adjacency matrix of a Barabási-Albert preferential-attachment graph."""
    if _HAS_NX:
        G = nx.barabasi_albert_graph(n, m, seed=seed)
        return nx.to_numpy_array(G, dtype=float)
    # Fallback: manual preferential attachment
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=float)
    # Start with a complete graph on m+1 nodes
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1.0
    degrees = adj.sum(axis=0)
    for new_node in range(m + 1, n):
        probs = degrees / degrees.sum()
        targets = rng.choice(new_node, size=m, replace=False, p=probs[:new_node])
        for t in targets:
            adj[new_node, t] = adj[t, new_node] = 1.0
            degrees[t] += 1
        degrees[new_node] = m
    return adj


# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------

def save_graph(adj: np.ndarray, name: str, out_dir: str, meta: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n = adj.shape[0]
    n_edges = int(adj.sum()) // 2

    # Adjacency matrix
    npy_path = os.path.join(out_dir, f"{name}.npy")
    np.save(npy_path, adj)

    # Edge list
    csv_path = os.path.join(out_dir, f"{name}_edges.csv")
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    with open(csv_path, "w") as f:
        f.write("src,dst,weight\n")
        for r, c in zip(rows, cols):
            f.write(f"{r},{c},{adj[r,c]:.4f}\n")

    # Metadata
    degrees = adj.sum(axis=0)
    meta.update({
        "n_nodes": n,
        "n_edges": n_edges,
        "density": float(n_edges / (n * (n - 1) / 2)),
        "mean_degree": float(degrees.mean()),
        "max_degree": float(degrees.max()),
        "min_degree": float(degrees.min()),
        "adjacency_matrix_file": npy_path,
        "edge_list_file": csv_path,
    })
    json_path = os.path.join(out_dir, f"{name}_meta.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {name}: {n} nodes, {n_edges} edges → {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic graphs for GPU-MLMC")
    parser.add_argument("--model", choices=["er", "ba", "both"], default="both",
                        help="Graph model: er=Erdős-Rényi, ba=Barabási-Albert, both")
    parser.add_argument("--n", type=int, default=500, help="Number of nodes")
    parser.add_argument("--p", type=float, default=0.05,
                        help="Edge probability (ER only)")
    parser.add_argument("--m", type=int, default=2,
                        help="Edges per new node in preferential attachment (BA only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/graphs",
                        help="Output directory")
    args = parser.parse_args()

    if not _HAS_NX:
        print("Warning: networkx not installed — using built-in fallback generators")

    if args.model in ("er", "both"):
        adj = erdos_renyi(args.n, args.p, args.seed)
        save_graph(adj, f"er_n{args.n}_p{args.p:.3f}_s{args.seed}", args.out,
                   {"model": "erdos_renyi", "n": args.n, "p": args.p, "seed": args.seed})

    if args.model in ("ba", "both"):
        adj = barabasi_albert(args.n, args.m, args.seed)
        save_graph(adj, f"ba_n{args.n}_m{args.m}_s{args.seed}", args.out,
                   {"model": "barabasi_albert", "n": args.n, "m": args.m, "seed": args.seed})


if __name__ == "__main__":
    main()
