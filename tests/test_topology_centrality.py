import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from network.topology import NetworkGraph, TopologyGenerator, centrality_weights


def test_degree_centrality_star_hub_dominates():
    """Degree centrality gives the star hub the largest normalized weight."""
    graph = TopologyGenerator(seed=1).generate_star_graph(n_nodes=6)
    weights = centrality_weights(graph, kind='degree')

    assert int(np.argmax(weights)) == 0
    np.testing.assert_allclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0.0)


def test_pagerank_disconnected_graph_is_normalized():
    """Disconnected graphs still produce finite PageRank weights."""
    graph = NetworkGraph()
    for node in range(5):
        graph.add_node(node)
    graph.add_edge(0, 1)
    graph.add_edge(2, 3)

    weights = centrality_weights(graph, kind='pagerank')

    assert weights.shape == (5,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(np.sum(weights), 1.0)


def test_betweenness_centrality_path_center_dominates():
    """Betweenness centrality ranks the center of a path graph highest."""
    graph = TopologyGenerator(seed=2).generate_line_graph(n_nodes=5)
    weights = centrality_weights(graph, kind='betweenness')

    assert int(np.argmax(weights)) == 2
    np.testing.assert_allclose(np.sum(weights), 1.0)
