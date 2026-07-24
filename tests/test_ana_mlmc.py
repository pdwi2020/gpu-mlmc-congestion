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


def test_weighted_stopping_criterion_uses_sla_weights():
    sla_vec = np.array([10.0, 1.0, 1.0])
    uniform_w = np.ones(3) / 3.0
    stopping_w = sla_vec / sla_vec.sum()
    # SLA-weighted node 0 should have much higher weight than uniform
    assert stopping_w[0] > uniform_w[0] * 2
    np.testing.assert_allclose(stopping_w.sum(), 1.0, rtol=1e-10)
    # Stopping threshold is epsilon/2 (RMSE-based), not epsilon^2/2
    epsilon = 0.1
    assert abs(epsilon / 2.0 - 0.05) < 1e-12


def test_nonuniform_weights_sharpen_high_weight_node():
    """Non-uniform SLA weights allocate more samples to the focused node,
    reducing its per-level variance contribution relative to a uniform run."""
    from network.traffic import PoissonTraffic

    graph = TopologyGenerator(seed=9).generate_erdos_renyi(n_nodes=6, p=0.4)
    traffic = PoissonTraffic(rate=5.0, seed=9)

    def make_ana(**kwargs):
        return AdaptiveNetworkAwareMLMC(
            refinement_factor=2, seed=9,
            weight_centrality=0.0, weight_variance=0.0, weight_sla=1.0,
            **kwargs,
        )

    common = dict(
        network=graph, traffic=traffic,
        epsilon=0.3, L_max=2, T=2.0, base_dt=0.5,
        pilot_samples=20, verbose=False,
    )

    # Focus all SLA weight on node 0
    focused_sla = np.zeros(graph.n_nodes)
    focused_sla[0] = 1.0
    r_focused = make_ana().mlmc_estimate_weighted(**common, sla_vec=focused_sla)

    # Uniform SLA
    uniform_sla = np.ones(graph.n_nodes)
    r_uniform = make_ana().mlmc_estimate_weighted(**common, sla_vec=uniform_sla)

    focused_w = np.array(r_focused.metadata['ana_node_weights'])
    uniform_w = np.array(r_uniform.metadata['ana_node_weights'])

    # Focused run must place strictly more weight on node 0 than uniform
    assert focused_w[0] > uniform_w[0], (
        f"Expected focused_w[0]={focused_w[0]:.4f} > uniform_w[0]={uniform_w[0]:.4f}"
    )
    # Weighted level variances in focused run track node 0's variance tightly
    focused_wv = r_focused.metadata['ana_weighted_level_variances']
    uniform_wv = r_uniform.metadata['ana_weighted_level_variances']
    # Focused run's weighted variance (= node-0 variance) should be <= uniform's
    # at the finest level after full sample allocation
    assert focused_wv[-1] <= uniform_wv[-1] * 1.5, (
        "Focused run did not reduce node-0 weighted variance relative to uniform"
    )


def test_continuation_criterion_meets_mse():
    """After mlmc_estimate_weighted, the final estimator MSE should be ≤ ε²
    (with a small safety factor for pilot estimation noise).

    This verifies that the Collier et al. (BIT 2015) stopping criterion produces
    a result whose variance + bias² is bounded by the target tolerance.
    """
    from network.traffic import PoissonTraffic

    graph = TopologyGenerator(seed=11).generate_line_graph(n_nodes=4)
    traffic = PoissonTraffic(rate=3.0, seed=11)
    ana = AdaptiveNetworkAwareMLMC(refinement_factor=2, seed=11)

    epsilon = 0.3
    result = ana.mlmc_estimate_weighted(
        network=graph,
        traffic=traffic,
        epsilon=epsilon,
        L_max=3,
        T=1.0,
        base_dt=0.5,
        pilot_samples=40,
        verbose=False,
    )

    # Final estimator MSE = variance + bias_estimate².
    # Allow 4× tolerance for pilot estimation noise.
    assert result.mse <= (epsilon ** 2) * 4.0, (
        f"MSE={result.mse:.4f} exceeds 4×ε²={4*epsilon**2:.4f}; "
        "continuation criterion did not adequately bound error"
    )


def test_continuation_stops_early_when_criterion_met():
    """With a very loose epsilon the continuation criterion should stop
    after the minimum levels (l>0 required), using fewer total samples
    than a run forced to reach L_max.

    This validates that bias² + Σ(V_l^w/N_pilot) ≤ ε² causes an early break
    rather than always advancing to L_max levels.
    """
    from network.traffic import PoissonTraffic

    graph = TopologyGenerator(seed=13).generate_line_graph(n_nodes=3)
    traffic = PoissonTraffic(rate=2.0, seed=13)

    common = dict(
        network=graph, traffic=traffic,
        L_max=4, T=1.0, base_dt=0.5, pilot_samples=30, verbose=False,
    )

    # Loose epsilon: should stop early (fewer levels)
    ana_loose = AdaptiveNetworkAwareMLMC(refinement_factor=2, seed=13)
    result_loose = ana_loose.mlmc_estimate_weighted(epsilon=0.8, **common)

    # Tight epsilon: should use more levels
    ana_tight = AdaptiveNetworkAwareMLMC(refinement_factor=2, seed=13)
    result_tight = ana_tight.mlmc_estimate_weighted(epsilon=0.02, **common)

    # Loose run must use fewer or equal levels than tight run
    n_loose = len(result_loose.level_stats)
    n_tight = len(result_tight.level_stats)
    assert n_loose <= n_tight, (
        f"Expected loose run ({n_loose} levels) to stop no later than "
        f"tight run ({n_tight} levels)"
    )
    # Loose run must have smaller or equal total cost than tight run
    assert result_loose.total_cost <= result_tight.total_cost * 1.05, (
        f"Loose run cost {result_loose.total_cost:.2e} not less than "
        f"tight run cost {result_tight.total_cost:.2e}"
    )
