"""
Extract a connected subgraph from the CAIDA AS-Relationships dataset.

Downloads (or reads a cached copy of) the CAIDA AS-relationships file and
extracts the top-N ASes by degree as a subgraph for use in GPU-MLMC experiments.

CAIDA AS-relationships format (one edge per line):
    <from_as>|<to_as>|<relationship>|<source>
    relationship: -1 = provider→customer, 0 = peer-peer, 1 = customer→provider

Dataset: https://www.caida.org/catalog/datasets/as-relationships/

Usage:
    # From a local CAIDA file
    python scripts/extract_caida_subgraph.py \
        --input datasets/caida/20240101.as-rel.txt.bz2 \
        --n 500 --out results/graphs/

    # Download most recent file (requires CAIDA account / HTTP access)
    python scripts/extract_caida_subgraph.py --download --n 500 --out results/graphs/

Outputs:
    results/graphs/caida_n500.npy      — adjacency matrix (0/1 float)
    results/graphs/caida_n500_edges.csv — edge list (src, dst, rel)
    results/graphs/caida_n500_meta.json — metadata (AS numbers, degree stats)
"""

import argparse
import bz2
import collections
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_caida_file(path: str) -> list:
    """Read CAIDA AS-relationships file; handles .bz2 and plain text."""
    edges = []
    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            try:
                src, dst, rel = int(parts[0]), int(parts[1]), int(parts[2])
                edges.append((src, dst, rel))
            except ValueError:
                continue
    return edges


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------

def extract_top_n_subgraph(edges: list, n: int) -> tuple:
    """
    Select the top-N ASes by degree and extract the induced subgraph.

    Returns:
        adj       : (n, n) float adjacency matrix (symmetric, 0/1)
        as_ids    : list[int] — original AS numbers (length n)
        node_map  : dict mapping original AS number → local index
    """
    # Count degree of each AS
    degree = collections.Counter()
    for src, dst, _ in edges:
        degree[src] += 1
        degree[dst] += 1

    # Select top-N by degree
    top_ases = {as_num for as_num, _ in degree.most_common(n)}
    as_ids = sorted(top_ases)
    node_map = {as_num: idx for idx, as_num in enumerate(as_ids)}

    # Build adjacency matrix on the induced subgraph
    adj = np.zeros((n, n), dtype=float)
    for src, dst, _ in edges:
        if src in node_map and dst in node_map:
            i, j = node_map[src], node_map[dst]
            adj[i, j] = 1.0
            adj[j, i] = 1.0  # undirected

    return adj, as_ids, node_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract CAIDA AS-relationships subgraph for GPU-MLMC"
    )
    parser.add_argument("--input", default=None,
                        help="Path to CAIDA .as-rel.txt or .as-rel.txt.bz2 file")
    parser.add_argument("--download", action="store_true",
                        help="Attempt to download the most recent CAIDA file")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of top-degree ASes to keep")
    parser.add_argument("--out", default="results/graphs",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- obtain the raw file ------------------------------------------------
    if args.download:
        print("Note: CAIDA data requires registration at https://www.caida.org")
        print("Please download the AS-relationships file manually and re-run with --input")
        raise SystemExit(1)

    if args.input is None:
        # Try default locations
        candidates = [
            "datasets/caida/as-rel.txt",
            "datasets/caida/as-rel.txt.bz2",
        ]
        for c in candidates:
            if os.path.exists(c):
                args.input = c
                break

    if args.input is None or not os.path.exists(args.input):
        print("No CAIDA file found. Provide --input <path> or place the file at "
              "datasets/caida/as-rel.txt[.bz2]")
        print("Generating a synthetic scale-free stand-in instead...")
        # Synthetic fallback: Barabási-Albert graph as AS-topology proxy
        try:
            import networkx as nx
            G = nx.barabasi_albert_graph(args.n, m=2, seed=42)
            adj = nx.to_numpy_array(G, dtype=float)
        except ImportError:
            rng = np.random.default_rng(42)
            adj = np.zeros((args.n, args.n), dtype=float)
            for i in range(args.n):
                for _ in range(2):
                    j = rng.integers(0, i) if i > 0 else 0
                    adj[i, j] = adj[j, i] = 1.0
        as_ids = list(range(args.n))
        name = f"caida_synthetic_n{args.n}"
        meta = {"source": "synthetic_ba_fallback", "n": args.n}
    else:
        print(f"Parsing {args.input} ...")
        edges = parse_caida_file(args.input)
        all_ases = set()
        for e in edges:
            all_ases.add(e[0])
            all_ases.add(e[1])
        print(f"  Loaded {len(edges):,} edges from {len(all_ases):,} ASes")
        adj, as_ids, _ = extract_top_n_subgraph(edges, args.n)
        name = f"caida_n{args.n}"
        meta = {
            "source": args.input,
            "n_ases_in_source": len(all_ases),
            "n_edges_in_source": len(edges),
        }

    # ---- save ----------------------------------------------------------------
    n = len(as_ids)
    degrees = adj.sum(axis=0)
    n_edges = int(adj.sum()) // 2

    npy_path = os.path.join(args.out, f"{name}.npy")
    np.save(npy_path, adj)

    csv_path = os.path.join(args.out, f"{name}_edges.csv")
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    with open(csv_path, "w") as f:
        f.write("local_src,local_dst,as_src,as_dst\n")
        for r, c in zip(rows, cols):
            f.write(f"{r},{c},{as_ids[r]},{as_ids[c]}\n")

    meta.update({
        "n_nodes": n,
        "n_edges": n_edges,
        "density": float(n_edges / max(n * (n - 1) / 2, 1)),
        "mean_degree": float(degrees.mean()),
        "max_degree": float(degrees.max()),
        "min_degree": float(degrees.min()),
        "as_numbers": as_ids[:20],  # first 20 for reference
    })
    json_path = os.path.join(args.out, f"{name}_meta.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {npy_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Subgraph: {n} nodes, {n_edges} edges, "
          f"mean degree {degrees.mean():.1f}, max degree {int(degrees.max())}")


if __name__ == "__main__":
    main()
