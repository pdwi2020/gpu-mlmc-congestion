"""Tests for scheduled dynamic link failures."""

import numpy as np

from network.link_failures import LinkFailureSchedule
from network.topology import NetworkGraph


def _line_graph(n_nodes: int) -> NetworkGraph:
    graph = NetworkGraph(directed=False)
    for node in range(n_nodes):
        graph.add_node(node)
    for node in range(n_nodes - 1):
        graph.add_edge(node, node + 1)
    return graph


def test_failure_removes_edge() -> None:
    """The active adjacency should contain fewer edges at a failure time."""
    graph = _line_graph(5)
    schedule = LinkFailureSchedule.schedule_failures(
        graph=graph,
        T=10.0,
        dt=0.1,
        n_failures=1,
        seed=3,
    )

    base_edges = int(schedule.get_adjacency_at(0.0).sum())
    failed_edges = int(schedule.get_adjacency_at(schedule.events[0].start_time).sum())

    assert failed_edges < base_edges


def test_recovery_restores_edge() -> None:
    """A recovered edge should be present again after the recovery delay."""
    graph = _line_graph(4)
    schedule = LinkFailureSchedule.schedule_failures(
        graph=graph,
        T=10.0,
        dt=0.1,
        n_failures=1,
        seed=4,
        recovery_time=1.0,
    )
    event = schedule.events[0]
    u, v = event.edge

    failed = schedule.get_adjacency_at(event.start_time)
    recovered = schedule.get_adjacency_at(event.start_time + 1.0 + 0.01)

    assert failed[u, v] == 0.0
    assert recovered[u, v] == schedule.base_adjacency[u, v]


def test_n_failures_obeyed() -> None:
    """The number of scheduled failure events should match the request."""
    graph = _line_graph(8)
    schedule = LinkFailureSchedule.schedule_failures(
        graph=graph,
        T=20.0,
        dt=0.1,
        n_failures=3,
        seed=5,
    )
    times = np.linspace(0.0, 20.0, 21)
    adjacency_t = schedule.as_time_series(times)

    assert len(schedule.events) == 3
    assert adjacency_t.shape == (21, 8, 8)
