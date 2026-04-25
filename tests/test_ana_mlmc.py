import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from network.topology import TopologyGenerator, centrality_weights
from simulation.mlmc import AdaptiveNetworkAwareMLMC, MLMCSimulator


def test_uniform_weights_match_giles():
    """Uniform ANA weights reduce exactly to standard Giles allocation."""
    ana = AdaptiveNetworkAwareMLMC(refinement_factor=2, seed=7)
    giles = MLMCSimulator(refinement_factor=2, seed=7)

    level_var_per_node = np.array([
        [2.0, 1.0, 0.5, 0.25],
        [0.5, 0.25, 0.125, 0.0625],
        [0.125, 0.0625, 0.03125, 0.015625],
    ])
    costs = [10.0, 20.0, 40.0]
    weights = np.full(level_var_per_node.shape[1], 1.0 / level_var_per_node.shape[1])
    epsilon = 0.02

    ana_samples = ana.compute_optimal_samples_weighted(
        level_var_per_node=level_var_per_node,
        costs=costs,
        weights=weights,
        epsilon=epsilon,
    )
    giles_samples = giles.compute_optimal_samples(
        variances=np.mean(level_var_per_node, axis=1).tolist(),
        costs=costs,
        epsilon=epsilon,
    )

    np.testing.assert_array_equal(ana_samples, giles_samples)


def test_centrality_weights_normalized():
    """Centrality weights are a valid probability vector on an ER graph."""
    graph = TopologyGenerator(seed=3).generate_erdos_renyi(n_nodes=12, p=0.25)
    weights = centrality_weights(graph, kind='pagerank')

    assert weights.shape == (graph.n_nodes,)
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(np.sum(weights), 1.0)


def test_weight_combination_sla():
    """High-SLA nodes with larger variance increase ANA sample allocation."""
    graph = TopologyGenerator(seed=4).generate_line_graph(n_nodes=4)
    ana = AdaptiveNetworkAwareMLMC(
        refinement_factor=2,
        seed=4,
        weight_centrality=0.0,
        weight_variance=0.0,
        weight_sla=1.0,
        centrality_kind='degree',
    )
    level_var_per_node = np.array([
        [16.0, 1.0, 1.0, 0.5],
        [4.0, 0.25, 0.25, 0.125],
        [1.0, 0.0625, 0.0625, 0.03125],
    ])
    costs = [5.0, 10.0, 20.0]
    epsilon = 0.05

    high_sla = ana.compute_node_weights(graph, level_var_per_node, sla_vec=np.array([10, 0, 0, 0]))
    low_sla = ana.compute_node_weights(graph, level_var_per_node, sla_vec=np.array([0, 0, 0, 10]))

    high_samples = ana.compute_optimal_samples_weighted(level_var_per_node, costs, high_sla, epsilon)
    low_samples = ana.compute_optimal_samples_weighted(level_var_per_node, costs, low_sla, epsilon)

    assert high_sla[0] > low_sla[0]
    assert sum(high_samples) > sum(low_samples)


def test_complexity_preservation():
    """For fixed weighted variances, ANA total cost scales like epsilon^-2."""
    ana = AdaptiveNetworkAwareMLMC(refinement_factor=2, seed=5)
    level_var_per_node = np.array([
        [8.0, 4.0, 2.0],
        [2.0, 1.0, 0.5],
        [0.5, 0.25, 0.125],
        [0.125, 0.0625, 0.03125],
    ])
    costs = np.array([4.0, 8.0, 16.0, 32.0])
    weights = np.array([0.5, 0.3, 0.2])

    samples_loose = ana.compute_optimal_samples_weighted(
        level_var_per_node, costs.tolist(), weights, epsilon=0.05
    )
    samples_tight = ana.compute_optimal_samples_weighted(
        level_var_per_node, costs.tolist(), weights, epsilon=0.025
    )

    cost_loose = float(np.dot(samples_loose, costs))
    cost_tight = float(np.dot(samples_tight, costs))
    ratio = cost_tight / cost_loose

    assert abs(ratio - 4.0) / 4.0 <= 0.10
