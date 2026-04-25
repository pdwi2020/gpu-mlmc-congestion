"""Tests for dynamic traffic arrival-rate generation."""

import numpy as np

from network.dynamic_traffic import DynamicArrivalProcess


def test_diurnal_amplitude_obeyed() -> None:
    """A pure one-day sinusoid should hit the configured max/min ratio."""
    amplitude = 0.3
    process = DynamicArrivalProcess(
        lambda_base=2.0,
        amplitude=amplitude,
        burst_rate=0.0,
        jitter_std=0.0,
    )

    arrivals = process.generate(n_nodes=4, T=24.0, dt=0.01, seed=1)
    observed_ratio = float(arrivals.max() / arrivals.min())
    expected_ratio = (1.0 + amplitude) / (1.0 - amplitude)

    assert np.isclose(observed_ratio, expected_ratio, rtol=1.0e-3)


def test_bursts_increase_total_arrivals() -> None:
    """Burst-enabled traffic should integrate higher and produce tail spikes."""
    without_bursts = DynamicArrivalProcess(
        lambda_base=1.0,
        amplitude=0.0,
        burst_rate=0.0,
        jitter_std=0.0,
    ).generate(n_nodes=6, T=24.0, dt=0.05, seed=7)

    process = DynamicArrivalProcess(
        lambda_base=1.0,
        amplitude=0.0,
        burst_rate=3.0 / 24.0,
        burst_duration=0.2,
        burst_magnitude=5.0,
        jitter_std=0.0,
        n_bursts=3,
    )
    with_bursts = process.generate(n_nodes=6, T=24.0, dt=0.05, seed=7)

    total_without = float(without_bursts.sum())
    total_with = float(with_bursts.sum())
    total_series = with_bursts.sum(axis=1)

    assert total_with > total_without
    assert float(total_series.max()) > float(np.percentile(total_series, 95))
    assert len(process.last_burst_onsets) == 3


def test_multi_seed_independent() -> None:
    """Different seeds should produce different dynamic paths."""
    process = DynamicArrivalProcess(
        lambda_base=1.0,
        amplitude=0.2,
        burst_rate=2.0,
        burst_duration=0.2,
        burst_magnitude=3.0,
        jitter_std=0.05,
    )

    arrivals = process.generate_multi_seed(n_nodes=3, T=4.0, dt=0.05, seeds=[1, 2])

    assert arrivals.shape == (2, 80, 3)
    assert not np.allclose(arrivals[0], arrivals[1])
