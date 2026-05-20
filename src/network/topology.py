"""
Network Topology Module

This module provides classes for network graph representation, dataset loading,
and synthetic topology generation for network modeling and simulation.

Classes:
    NetworkGraph: Network topology representation with link properties
    TopologyGenerator: Generate synthetic network topologies
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union
import logging
import bz2
import gzip


logger = logging.getLogger(__name__)


class NetworkGraph:
    """
    Network topology representation wrapper around NetworkX graph.

    Supports loading from SNAP and CAIDA datasets, and provides utility
    methods for network analysis and property management.

    Attributes:
        graph: NetworkX Graph or DiGraph object
        n_nodes: Number of nodes in the network
        n_edges: Number of edges in the network
    """

    def __init__(self, directed: bool = False):
        """
        Initialize NetworkGraph.

        Args:
            directed: If True, create directed graph; otherwise undirected
        """
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self._shortest_paths_cache = None

    @property
    def n_nodes(self) -> int:
        """Return number of nodes in the network."""
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        """Return number of edges in the network."""
        return self.graph.number_of_edges()

    @property
    def nodes(self):
        """Return node view."""
        return self.graph.nodes()

    @property
    def edges(self):
        """Return edge view."""
        return self.graph.edges()

    def add_node(self, node_id: int, **attributes):
        """
        Add a node to the network.

        Args:
            node_id: Unique node identifier
            **attributes: Node attributes (e.g., capacity, type)
        """
        self.graph.add_node(node_id, **attributes)
        self._shortest_paths_cache = None  # Invalidate cache

    def add_edge(self, source: int, target: int, **attributes):
        """
        Add an edge to the network.

        Args:
            source: Source node ID
            target: Target node ID
            **attributes: Edge attributes (e.g., bandwidth, delay, capacity)
        """
        self.graph.add_edge(source, target, **attributes)
        self._shortest_paths_cache = None  # Invalidate cache

    def get_node_attributes(self, node_id: int) -> Dict:
        """Get all attributes for a specific node."""
        return self.graph.nodes[node_id]

    def get_edge_attributes(self, source: int, target: int) -> Dict:
        """Get all attributes for a specific edge."""
        return self.graph.edges[source, target]

    def set_link_properties(self,
                           bandwidth_range: Tuple[float, float] = (100.0, 1000.0),
                           delay_range: Tuple[float, float] = (1.0, 10.0),
                           capacity_range: Tuple[float, float] = (100.0, 1000.0),
                           seed: Optional[int] = None):
        """
        Assign synthetic link properties to all edges.

        Useful for datasets that only provide topology without link characteristics.

        Args:
            bandwidth_range: (min, max) bandwidth in Mbps
            delay_range: (min, max) propagation delay in ms
            capacity_range: (min, max) queue capacity in packets
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        for u, v in self.graph.edges():
            # Assign random properties within specified ranges
            bandwidth = np.random.uniform(*bandwidth_range)
            delay = np.random.uniform(*delay_range)
            capacity = np.random.uniform(*capacity_range)

            self.graph.edges[u, v]['bandwidth'] = bandwidth
            self.graph.edges[u, v]['delay'] = delay
            self.graph.edges[u, v]['capacity'] = capacity
            # Service rate (packets/ms) based on bandwidth
            self.graph.edges[u, v]['service_rate'] = bandwidth / 10.0

        logger.info(f"Assigned link properties to {self.n_edges} edges")

    def get_link_properties(self) -> Dict[str, np.ndarray]:
        """
        Get all link properties as arrays.

        Returns:
            Dictionary with keys: 'bandwidth', 'delay', 'capacity', 'service_rate'
            Each value is a numpy array of length n_edges
        """
        properties = {
            'bandwidth': [],
            'delay': [],
            'capacity': [],
            'service_rate': []
        }

        for u, v, data in self.graph.edges(data=True):
            for key in properties:
                properties[key].append(data.get(key, 0.0))

        return {k: np.array(v) for k, v in properties.items()}

    def compute_shortest_paths(self, source: Optional[int] = None,
                              use_cache: bool = True) -> Dict:
        """
        Compute shortest paths in the network.

        Args:
            source: Source node for single-source shortest paths.
                   If None, compute all-pairs shortest paths.
            use_cache: Use cached results if available

        Returns:
            Dictionary of shortest paths (format depends on NetworkX version)
        """
        if source is None:
            # All-pairs shortest paths
            if use_cache and self._shortest_paths_cache is not None:
                return self._shortest_paths_cache

            paths = dict(nx.all_pairs_shortest_path(self.graph))
            self._shortest_paths_cache = paths
            return paths
        else:
            # Single-source shortest paths
            return nx.single_source_shortest_path(self.graph, source)

    def get_neighbors(self, node_id: int) -> List[int]:
        """Get all neighbors of a node."""
        return list(self.graph.neighbors(node_id))

    def get_degree(self, node_id: int) -> int:
        """Get degree of a node."""
        return self.graph.degree(node_id)

    def get_adjacency_matrix(self, sparse: bool = False) -> np.ndarray:
        """
        Get adjacency matrix representation.

        Args:
            sparse: If True, return scipy sparse matrix

        Returns:
            Adjacency matrix (dense or sparse)
        """
        if sparse:
            return nx.adjacency_matrix(self.graph)
        else:
            return nx.to_numpy_array(self.graph)

    def to_directed(self) -> 'NetworkGraph':
        """Convert to directed graph (if undirected)."""
        new_graph = NetworkGraph(directed=True)
        new_graph.graph = self.graph.to_directed()
        return new_graph

    def to_undirected(self) -> 'NetworkGraph':
        """Convert to undirected graph (if directed)."""
        new_graph = NetworkGraph(directed=False)
        new_graph.graph = self.graph.to_undirected()
        return new_graph

    def get_largest_component(self) -> 'NetworkGraph':
        """
        Extract largest connected component.

        Returns:
            New NetworkGraph containing only the largest component
        """
        if self.graph.is_directed():
            components = nx.weakly_connected_components(self.graph)
        else:
            components = nx.connected_components(self.graph)

        largest_cc = max(components, key=len)

        new_graph = NetworkGraph(directed=self.graph.is_directed())
        new_graph.graph = self.graph.subgraph(largest_cc).copy()

        logger.info(f"Extracted largest component: {new_graph.n_nodes} nodes, {new_graph.n_edges} edges")
        return new_graph

    def summary(self) -> Dict:
        """
        Get network summary statistics.

        Returns:
            Dictionary with network properties
        """
        stats = {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'directed': self.graph.is_directed(),
            'density': nx.density(self.graph),
        }

        # Add connectivity info
        if self.graph.number_of_nodes() == 0:
            stats['connected'] = True
            stats['n_components'] = 0
        elif self.graph.is_directed():
            stats['weakly_connected'] = nx.is_weakly_connected(self.graph)
            stats['n_components'] = nx.number_weakly_connected_components(self.graph)
        else:
            stats['connected'] = nx.is_connected(self.graph)
            stats['n_components'] = nx.number_connected_components(self.graph)

        # Degree statistics
        degrees = [d for n, d in self.graph.degree()]
        if degrees:
            stats['avg_degree'] = np.mean(degrees)
            stats['max_degree'] = np.max(degrees)
            stats['min_degree'] = np.min(degrees)
        else:
            stats['avg_degree'] = 0.0
            stats['max_degree'] = 0
            stats['min_degree'] = 0

        return stats

    def __repr__(self) -> str:
        return f"NetworkGraph(nodes={self.n_nodes}, edges={self.n_edges}, directed={self.graph.is_directed()})"


def _centrality_graph_with_self_loops(graph: nx.Graph) -> nx.Graph:
    """Return a copy with tiny self-loops when connectivity is incomplete."""
    if graph.number_of_nodes() == 0:
        return graph.copy()

    is_connected = (
        nx.is_weakly_connected(graph)
        if graph.is_directed()
        else nx.is_connected(graph)
    )
    if is_connected:
        return graph.copy()

    graph_with_loops = graph.copy()
    for node in graph_with_loops.nodes:
        if not graph_with_loops.has_edge(node, node):
            graph_with_loops.add_edge(node, node, weight=1.0e-12)
    return graph_with_loops


def _normalize_centrality(values: np.ndarray) -> np.ndarray:
    """Clip and normalize a centrality vector."""
    if values.size == 0:
        return values.astype(float)

    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.maximum(values, 0.0)
    total = float(np.sum(values))
    if total <= 0.0:
        return np.full(values.shape, 1.0 / values.size, dtype=float)
    return values / total


def centrality_weights(
    graph: NetworkGraph,
    kind: Literal['pagerank', 'betweenness', 'degree'] = 'pagerank',
    alpha: float = 0.85,
) -> np.ndarray:
    """Return non-negative node centrality weights in NetworkGraph node order."""
    nx_graph = _centrality_graph_with_self_loops(graph.graph)
    nodes = list(nx_graph.nodes)
    if not nodes:
        return np.array([], dtype=float)

    if kind == 'pagerank':
        pagerank_func = getattr(nx, 'pagerank_numpy', None)
        if pagerank_func is None:
            scores = nx.pagerank(nx_graph, alpha=alpha, weight='weight')
        else:
            scores = pagerank_func(nx_graph, alpha=alpha, weight='weight')
    elif kind == 'betweenness':
        scores = nx.betweenness_centrality(nx_graph, normalized=True, weight='weight')
    elif kind == 'degree':
        scores = nx.degree_centrality(nx_graph)
    else:
        raise ValueError(f"Unknown centrality kind: {kind}")

    return _normalize_centrality(np.array([scores[node] for node in nodes], dtype=float))


def load_snap_graph(filepath: Union[str, Path],
                    directed: bool = False,
                    largest_component: bool = True) -> NetworkGraph:
    """
    Load network graph from SNAP dataset.

    SNAP format: Edge list with each line as "source target"
    Comments start with '#'

    Args:
        filepath: Path to SNAP edge list file
        directed: If True, create directed graph
        largest_component: If True, extract largest connected component

    Returns:
        NetworkGraph object

    Example:
        >>> graph = load_snap_graph("datasets/snap/email-Eu-core.txt")
        >>> print(graph.summary())
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Loading SNAP graph from {filepath}")

    # Handle compressed files
    if filepath.suffix == '.gz':
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'

    network = NetworkGraph(directed=directed)

    with open_func(filepath, mode) as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse edge
            parts = line.split()
            if len(parts) >= 2:
                source = int(parts[0])
                target = int(parts[1])
                network.add_edge(source, target)

    logger.info(f"Loaded SNAP graph: {network.n_nodes} nodes, {network.n_edges} edges")

    # Extract largest component if requested
    if largest_component:
        network = network.get_largest_component()

    return network


def load_caida_topology(filepath: Union[str, Path],
                       as_undirected: bool = True,
                       largest_component: bool = True) -> NetworkGraph:
    """
    Load CAIDA AS relationships dataset (serial-2 format).

    CAIDA AS-REL2 format:
        <provider-as>|<customer-as>|-1  (provider-to-customer)
        <peer-as>|<peer-as>|0           (peer-to-peer)

    Args:
        filepath: Path to CAIDA AS relationships file (.txt or .bz2)
        as_undirected: If True, treat all relationships as undirected
        largest_component: If True, extract largest connected component

    Returns:
        NetworkGraph object

    Example:
        >>> graph = load_caida_topology("datasets/caida/20260101.as-rel2.txt.bz2")
        >>> print(graph.summary())
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Loading CAIDA topology from {filepath}")

    # Handle compressed files
    if filepath.suffix == '.bz2':
        open_func = bz2.open
        mode = 'rt'
    elif filepath.suffix == '.gz':
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'

    network = NetworkGraph(directed=not as_undirected)

    relationship_counts = {'provider_customer': 0, 'peer_peer': 0}

    with open_func(filepath, mode) as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse AS relationship: as1|as2|relationship_type
            parts = line.split('|')
            if len(parts) >= 3:
                as1 = int(parts[0])
                as2 = int(parts[1])
                rel_type = int(parts[2])

                if rel_type == -1:
                    # Provider-to-customer relationship
                    relationship_counts['provider_customer'] += 1
                    if as_undirected:
                        network.add_edge(as1, as2, relationship='provider_customer')
                    else:
                        network.add_edge(as1, as2, relationship='provider_customer')
                elif rel_type == 0:
                    # Peer-to-peer relationship
                    relationship_counts['peer_peer'] += 1
                    network.add_edge(as1, as2, relationship='peer_peer')

    logger.info(f"Loaded CAIDA topology: {network.n_nodes} AS nodes, {network.n_edges} edges")
    logger.info(f"Relationships: {relationship_counts['provider_customer']} provider-customer, "
                f"{relationship_counts['peer_peer']} peer-peer")

    # Extract largest component if requested
    if largest_component:
        network = network.get_largest_component()

    return network


class TopologyGenerator:
    """
    Generate synthetic network topologies for testing and benchmarking.

    Supports various random graph models commonly used in network research.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize TopologyGenerator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def generate_erdos_renyi(self, n_nodes: int, p: float,
                            directed: bool = False) -> NetworkGraph:
        """
        Generate Erdős-Rényi random graph.

        Each edge exists independently with probability p.

        Args:
            n_nodes: Number of nodes
            p: Edge probability (0 < p < 1)
            directed: If True, create directed graph

        Returns:
            NetworkGraph object
        """
        logger.info(f"Generating Erdős-Rényi graph: n={n_nodes}, p={p}")

        if directed:
            nx_graph = nx.erdos_renyi_graph(n_nodes, p, directed=True, seed=self.seed)
        else:
            nx_graph = nx.erdos_renyi_graph(n_nodes, p, seed=self.seed)

        network = NetworkGraph(directed=directed)
        network.graph = nx_graph

        logger.info(f"Generated graph: {network.n_nodes} nodes, {network.n_edges} edges")
        return network

    def generate_barabasi_albert(self, n_nodes: int, m: int) -> NetworkGraph:
        """
        Generate Barabási-Albert scale-free network.

        Preferential attachment: new nodes connect to m existing nodes
        with probability proportional to their degree.

        Args:
            n_nodes: Number of nodes
            m: Number of edges to attach from new node

        Returns:
            NetworkGraph object
        """
        logger.info(f"Generating Barabási-Albert graph: n={n_nodes}, m={m}")

        nx_graph = nx.barabasi_albert_graph(n_nodes, m, seed=self.seed)

        network = NetworkGraph(directed=False)
        network.graph = nx_graph

        logger.info(f"Generated graph: {network.n_nodes} nodes, {network.n_edges} edges")
        return network

    def generate_watts_strogatz(self, n_nodes: int, k: int, p: float) -> NetworkGraph:
        """
        Generate Watts-Strogatz small-world network.

        Start with ring lattice, then rewire edges with probability p.

        Args:
            n_nodes: Number of nodes
            k: Each node connected to k nearest neighbors in ring
            p: Rewiring probability

        Returns:
            NetworkGraph object
        """
        logger.info(f"Generating Watts-Strogatz graph: n={n_nodes}, k={k}, p={p}")

        nx_graph = nx.watts_strogatz_graph(n_nodes, k, p, seed=self.seed)

        network = NetworkGraph(directed=False)
        network.graph = nx_graph

        logger.info(f"Generated graph: {network.n_nodes} nodes, {network.n_edges} edges")
        return network

    def generate_random_regular(self, n_nodes: int, d: int) -> NetworkGraph:
        """
        Generate random regular graph.

        All nodes have exactly degree d.

        Args:
            n_nodes: Number of nodes
            d: Degree of each node (must be even for undirected graph)

        Returns:
            NetworkGraph object
        """
        logger.info(f"Generating random regular graph: n={n_nodes}, d={d}")

        nx_graph = nx.random_regular_graph(d, n_nodes, seed=self.seed)

        network = NetworkGraph(directed=False)
        network.graph = nx_graph

        logger.info(f"Generated graph: {network.n_nodes} nodes, {network.n_edges} edges")
        return network

    def generate_hierarchical(self, n_levels: int, branching_factor: int) -> NetworkGraph:
        """
        Generate hierarchical tree-like topology.

        Common in network architectures (e.g., data center networks).

        Args:
            n_levels: Number of hierarchy levels
            branching_factor: Children per node

        Returns:
            NetworkGraph object
        """
        logger.info(f"Generating hierarchical graph: levels={n_levels}, branching={branching_factor}")

        network = NetworkGraph(directed=False)

        # Root node
        node_id = 0
        network.add_node(node_id, level=0)

        # BFS construction
        queue = [(node_id, 0)]  # (node_id, level)
        node_id += 1

        while queue:
            parent, level = queue.pop(0)

            if level < n_levels - 1:
                for _ in range(branching_factor):
                    network.add_node(node_id, level=level + 1)
                    network.add_edge(parent, node_id)
                    queue.append((node_id, level + 1))
                    node_id += 1

        logger.info(f"Generated graph: {network.n_nodes} nodes, {network.n_edges} edges")
        return network

    def generate_line_graph(self, n_nodes: int) -> NetworkGraph:
        """
        Generate a line (path) graph: nodes 0-1-2-..-(n_nodes-1).

        Args:
            n_nodes: Number of nodes

        Returns:
            NetworkGraph object
        """
        logger.info(f"Generating line graph: n={n_nodes}")

        nx_graph = nx.path_graph(n_nodes)

        network = NetworkGraph(directed=False)
        network.graph = nx_graph

        logger.info(f"Generated graph: {network.n_nodes} nodes, {network.n_edges} edges")
        return network

    def generate_star_graph(self, n_nodes: int) -> NetworkGraph:
        """
        Generate a star graph: one central hub connected to all others.

        Args:
            n_nodes: Total number of nodes (including hub)

        Returns:
            NetworkGraph object
        """
        logger.info(f"Generating star graph: n={n_nodes}")

        nx_graph = nx.star_graph(n_nodes - 1)

        network = NetworkGraph(directed=False)
        network.graph = nx_graph

        logger.info(f"Generated graph: {network.n_nodes} nodes, {network.n_edges} edges")
        return network


# Convenience function for quick testing
def create_example_network(network_type: str = 'erdos_renyi',
                          size: str = 'small') -> NetworkGraph:
    """
    Create example networks for quick testing.

    Args:
        network_type: Type of network ('erdos_renyi', 'barabasi_albert', 'watts_strogatz')
        size: Network size ('small', 'medium', 'large')

    Returns:
        NetworkGraph object
    """
    size_params = {
        'small': 50,
        'medium': 500,
        'large': 5000
    }

    n_nodes = size_params.get(size, 50)
    gen = TopologyGenerator(seed=42)

    if network_type == 'erdos_renyi':
        return gen.generate_erdos_renyi(n_nodes, p=0.05)
    elif network_type == 'barabasi_albert':
        return gen.generate_barabasi_albert(n_nodes, m=3)
    elif network_type == 'watts_strogatz':
        return gen.generate_watts_strogatz(n_nodes, k=4, p=0.1)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Network Topology Module - Example Usage")
    print("=" * 60)

    # Generate synthetic network
    print("\n1. Generating synthetic Erdős-Rényi network...")
    gen = TopologyGenerator(seed=42)
    network = gen.generate_erdos_renyi(n_nodes=100, p=0.05)
    print(network)
    print(network.summary())

    # Add link properties
    print("\n2. Adding synthetic link properties...")
    network.set_link_properties(seed=42)
    props = network.get_link_properties()
    print(f"Average bandwidth: {np.mean(props['bandwidth']):.2f} Mbps")
    print(f"Average delay: {np.mean(props['delay']):.2f} ms")

    # Test shortest paths
    print("\n3. Computing shortest paths...")
    paths = network.compute_shortest_paths(source=0)
    print(f"Computed paths from node 0 to {len(paths)} reachable nodes")

    print("\n" + "=" * 60)
