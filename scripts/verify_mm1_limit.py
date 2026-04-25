"""Verify a reflected SDE against the M/M/1 heavy-traffic mean.

The default horizon is intentionally modest for local reproducibility. For the
paper-quality Colab run, call this script with ``--t-final 10000``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_RHOS = (0.7, 0.8, 0.9)
DEFAULT_MU = 1.25
DEFAULT_DT = 1e-3
DEFAULT_T_FINAL = 1e3
DEFAULT_BURN_IN_FRAC = 0.2
DEFAULT_SEED = 42
DEFAULT_OUTPUT = Path("results/mm1_sanity.json")


@dataclass(frozen=True)
class MM1SanityResult:
    """JSON-serializable summary for one utilization value."""

    analytical: float
    sde_mean: float
    rel_error_pct: float
    T: float
    dt: float
    burn_in_frac: float


def analytical_mean_queue(rho: float) -> float:
    """Return the M/M/1 analytical mean queue length rho / (1 - rho)."""

    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    return rho / (1.0 - rho)


def simulate_reflected_sde_mean(
    rho: float,
    *,
    mu: float = DEFAULT_MU,
    dt: float = DEFAULT_DT,
    t_final: float = DEFAULT_T_FINAL,
    burn_in_frac: float = DEFAULT_BURN_IN_FRAC,
    rng: np.random.Generator | None = None,
) -> float:
    """Simulate reflected Euler-Maruyama paths and return the post-burn-in mean."""

    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    if dt <= 0.0 or t_final <= 0.0:
        raise ValueError("dt and t_final must be positive")
    if not 0.0 <= burn_in_frac < 1.0:
        raise ValueError("burn_in_frac must satisfy 0 <= burn_in_frac < 1")

    random = rng if rng is not None else np.random.default_rng(DEFAULT_SEED)
    steps = int(round(t_final / dt))
    if steps < 2:
        raise ValueError("t_final / dt must produce at least two steps")

    burn_steps = int(steps * burn_in_frac)
    lambda_rate = rho * mu
    drift_step = (lambda_rate - mu) * dt
    diffusion_step = math.sqrt(lambda_rate + mu) * math.sqrt(dt)

    q_value = 0.0
    mean_sum = 0.0
    mean_count = 0

    for step in range(steps):
        increment = drift_step + diffusion_step * float(random.normal())
        q_value = float(np.maximum(0.0, q_value + increment))
        if step >= burn_steps:
            mean_sum += q_value
            mean_count += 1

    return mean_sum / mean_count


def run_verification(
    rhos: Iterable[float] = DEFAULT_RHOS,
    *,
    mu: float = DEFAULT_MU,
    dt: float = DEFAULT_DT,
    t_final: float = DEFAULT_T_FINAL,
    burn_in_frac: float = DEFAULT_BURN_IN_FRAC,
    seed: int = DEFAULT_SEED,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, dict[str, float]]:
    """Run the M/M/1 sanity check and write the JSON results file."""

    rng = np.random.default_rng(seed)
    results: dict[str, dict[str, float]] = {}

    for rho in rhos:
        analytical = analytical_mean_queue(rho)
        sde_mean = simulate_reflected_sde_mean(
            rho,
            mu=mu,
            dt=dt,
            t_final=t_final,
            burn_in_frac=burn_in_frac,
            rng=rng,
        )
        rel_error_pct = abs(sde_mean - analytical) / analytical * 100.0
        key = f"{rho:.1f}"
        results[key] = asdict(
            MM1SanityResult(
                analytical=analytical,
                sde_mean=sde_mean,
                rel_error_pct=rel_error_pct,
                T=t_final,
                dt=dt,
                burn_in_frac=burn_in_frac,
            )
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the standalone verifier."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhos", nargs="+", type=float, default=list(DEFAULT_RHOS))
    parser.add_argument("--mu", type=float, default=DEFAULT_MU)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--t-final", type=float, default=DEFAULT_T_FINAL)
    parser.add_argument("--burn-in-frac", type=float, default=DEFAULT_BURN_IN_FRAC)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run the verifier from the command line."""

    args = parse_args()
    results = run_verification(
        args.rhos,
        mu=args.mu,
        dt=args.dt,
        t_final=args.t_final,
        burn_in_frac=args.burn_in_frac,
        seed=args.seed,
        output_path=args.output,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
