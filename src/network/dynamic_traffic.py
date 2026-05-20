"""Dynamic arrival-rate processes for time-varying network experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np


ArrayLikeRate = Union[float, np.ndarray]


@dataclass
class DynamicArrivalProcess:
    """Generate per-node arrival-rate trajectories with diurnal and burst terms."""

    lambda_base: ArrayLikeRate = 1.0
    amplitude: ArrayLikeRate = 0.3
    period_hours: float = 24.0
    burst_rate: ArrayLikeRate = 0.0
    burst_duration: ArrayLikeRate = 0.25
    burst_magnitude: ArrayLikeRate = 2.0
    jitter_std: ArrayLikeRate = 0.0
    n_bursts: Optional[int] = None
    global_bursts: bool = True
    burst_node_fraction: float = 1.0
    last_burst_onsets: np.ndarray = field(default_factory=lambda: np.array([], dtype=float), init=False)

    def generate(self, n_nodes: int, T: float, dt: float, seed: int) -> np.ndarray:
        """Generate one dynamic arrival-rate tensor of shape ``(n_steps, n_nodes)``."""
        if n_nodes <= 0:
            raise ValueError("n_nodes must be positive")
        if T <= 0.0:
            raise ValueError("T must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.period_hours <= 0.0:
            raise ValueError("period_hours must be positive")
        if not 0.0 <= self.burst_node_fraction <= 1.0:
            raise ValueError("burst_node_fraction must be in [0, 1]")

        n_steps = int(T / dt)
        if n_steps <= 0:
            raise ValueError("T / dt must produce at least one step")

        rng = np.random.default_rng(seed)
        times = np.arange(n_steps, dtype=float) * dt
        base = self._node_values(self.lambda_base, n_nodes, "lambda_base")
        amplitude = self._node_values(self.amplitude, n_nodes, "amplitude")

        diurnal = base[None, :] * (
            1.0 + amplitude[None, :] * np.sin(2.0 * np.pi * times[:, None] / self.period_hours)
        )
        arrivals = diurnal.astype(float, copy=True)

        if self.global_bursts:
            self._add_global_bursts(arrivals, T, dt, rng, base)
        else:
            self._add_per_node_bursts(arrivals, T, dt, rng, base)

        jitter_std = self._node_values(self.jitter_std, n_nodes, "jitter_std")
        if np.any(jitter_std > 0.0):
            noise = rng.normal(0.0, jitter_std[None, :] * base[None, :], arrivals.shape)
            arrivals = arrivals + noise

        return np.maximum(arrivals, 0.0)

    def generate_multi_seed(
        self,
        n_nodes: int,
        T: float,
        dt: float,
        seeds: Sequence[int],
    ) -> np.ndarray:
        """Generate an ensemble with shape ``(len(seeds), n_steps, n_nodes)``."""
        return np.stack(
            [self.generate(n_nodes=n_nodes, T=T, dt=dt, seed=int(seed)) for seed in seeds],
            axis=0,
        )

    @staticmethod
    def _node_values(value: ArrayLikeRate, n_nodes: int, name: str) -> np.ndarray:
        """Broadcast a scalar or validate a per-node vector."""
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            return np.full(n_nodes, float(values), dtype=float)
        if values.shape != (n_nodes,):
            raise ValueError(f"{name} must be a scalar or have shape ({n_nodes},)")
        return values.astype(float, copy=False)

    def _event_count(self, T: float, rng: np.random.Generator, rate: float) -> int:
        """Return a deterministic or Poisson event count."""
        if self.n_bursts is not None:
            if self.n_bursts < 0:
                raise ValueError("n_bursts must be non-negative")
            return int(self.n_bursts)
        if rate < 0.0:
            raise ValueError("burst_rate must be non-negative")
        return int(rng.poisson(rate * T))

    def _add_global_bursts(
        self,
        arrivals: np.ndarray,
        T: float,
        dt: float,
        rng: np.random.Generator,
        base: np.ndarray,
    ) -> None:
        """Add network-wide burst events, optionally to a random node subset."""
        n_steps, n_nodes = arrivals.shape
        burst_rate = self._node_values(self.burst_rate, n_nodes, "burst_rate")
        burst_duration = self._node_values(self.burst_duration, n_nodes, "burst_duration")
        burst_magnitude = self._node_values(self.burst_magnitude, n_nodes, "burst_magnitude")
        n_events = self._event_count(T, rng, float(np.mean(burst_rate)))
        onsets = np.sort(rng.uniform(0.0, T, size=n_events)) if n_events else np.array([], dtype=float)
        self.last_burst_onsets = onsets.copy()

        for onset in onsets:
            start = min(int(onset / dt), n_steps - 1)
            if self.burst_node_fraction >= 1.0:
                nodes = np.arange(n_nodes)
            elif self.burst_node_fraction <= 0.0:
                nodes = np.array([], dtype=int)
            else:
                k = max(1, int(np.ceil(self.burst_node_fraction * n_nodes)))
                nodes = rng.choice(n_nodes, size=k, replace=False)

            if nodes.size == 0:
                continue

            durations = np.maximum(burst_duration[nodes], dt)
            end_offsets = np.maximum(1, np.ceil(durations / dt).astype(int))
            for node, offset in zip(nodes, end_offsets):
                end = min(n_steps, start + int(offset))
                arrivals[start:end, node] += burst_magnitude[node] * base[node]

    def _add_per_node_bursts(
        self,
        arrivals: np.ndarray,
        T: float,
        dt: float,
        rng: np.random.Generator,
        base: np.ndarray,
    ) -> None:
        """Add independent Poisson burst events for each node."""
        n_steps, n_nodes = arrivals.shape
        burst_rate = self._node_values(self.burst_rate, n_nodes, "burst_rate")
        burst_duration = self._node_values(self.burst_duration, n_nodes, "burst_duration")
        burst_magnitude = self._node_values(self.burst_magnitude, n_nodes, "burst_magnitude")
        onsets: list[float] = []

        for node in range(n_nodes):
            n_events = self._event_count(T, rng, float(burst_rate[node]))
            node_onsets = np.sort(rng.uniform(0.0, T, size=n_events)) if n_events else np.array([])
            for onset in node_onsets:
                start = min(int(onset / dt), n_steps - 1)
                duration_steps = max(1, int(np.ceil(max(burst_duration[node], dt) / dt)))
                end = min(n_steps, start + duration_steps)
                arrivals[start:end, node] += burst_magnitude[node] * base[node]
            onsets.extend(float(value) for value in node_onsets)

        self.last_burst_onsets = np.array(sorted(onsets), dtype=float)
