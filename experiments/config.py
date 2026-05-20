"""
Experiment Configuration Module

Provides centralized configuration for all experiments with:
- Default parameters
- Command-line argument parsing
- Validation
- Reproducibility settings
"""

from dataclasses import dataclass, field
from typing import List, Optional
import argparse
import logging
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for MLMC experiments.

    Attributes:
        T: Simulation time horizon
        dt: Base timestep for coarsest level
        seed: Random seed for reproducibility
        n_samples: Default sample count for standard MC
        L_max: Maximum number of MLMC levels
        refinement_factor: Timestep refinement factor between levels
        use_gpu: Whether to use GPU acceleration if available
        output_dir: Directory for results output
        target_epsilons: List of target accuracies for convergence tests
        verbose: Enable verbose logging
        gpu_only: Skip CPU benchmarks and run GPU-only comparison
    """
    T: float = 10.0
    dt: float = 0.1
    seed: int = 42
    n_samples: int = 1000
    L_max: int = 5
    refinement_factor: int = 2
    use_gpu: bool = True
    output_dir: str = "results"
    target_epsilons: List[float] = field(default_factory=lambda: [0.1, 0.05, 0.01])
    verbose: bool = False
    gpu_only: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        validate_config(self)


def validate_config(config: ExperimentConfig) -> None:
    """Validate configuration parameters.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If any parameter is invalid
    """
    if config.T <= 0:
        raise ValueError(f"T must be positive, got {config.T}")

    if config.dt <= 0:
        raise ValueError(f"dt must be positive, got {config.dt}")

    if config.dt > config.T:
        raise ValueError(f"dt ({config.dt}) must be <= T ({config.T})")

    if config.seed < 0:
        raise ValueError(f"seed must be non-negative, got {config.seed}")

    if config.n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {config.n_samples}")

    if config.L_max < 0:
        raise ValueError(f"L_max must be non-negative, got {config.L_max}")

    if config.refinement_factor < 2:
        raise ValueError(f"refinement_factor must be >= 2, got {config.refinement_factor}")

    if not config.target_epsilons:
        raise ValueError("target_epsilons cannot be empty")

    for eps in config.target_epsilons:
        if eps <= 0:
            raise ValueError(f"All target_epsilons must be positive, got {eps}")


def parse_args(description: str = "MLMC Experiment") -> ExperimentConfig:
    """Parse command-line arguments into configuration.

    Args:
        description: Description for argparse help message

    Returns:
        ExperimentConfig with parsed arguments
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--T", type=float, default=10.0,
        help="Simulation time horizon (default: 10.0)"
    )
    parser.add_argument(
        "--dt", type=float, default=0.1,
        help="Base timestep for coarsest level (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Default sample count for standard MC (default: 1000)"
    )
    parser.add_argument(
        "--L-max", type=int, default=5,
        help="Maximum number of MLMC levels (default: 5)"
    )
    parser.add_argument(
        "--refinement-factor", type=int, default=2,
        help="Timestep refinement factor between levels (default: 2)"
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Disable GPU acceleration"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for results output (default: results)"
    )
    parser.add_argument(
        "--epsilons", type=float, nargs="+", default=[0.1, 0.05, 0.01],
        help="Target accuracies for convergence tests (default: 0.1 0.05 0.01)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Skip CPU benchmarks and run GPU-only comparison"
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        T=args.T,
        dt=args.dt,
        seed=args.seed,
        n_samples=args.n_samples,
        L_max=args.L_max,
        refinement_factor=args.refinement_factor,
        use_gpu=not args.no_gpu,
        output_dir=args.output_dir,
        target_epsilons=args.epsilons,
        verbose=args.verbose,
        gpu_only=args.gpu_only,
    )

    return config


def setup_logging(config: ExperimentConfig) -> None:
    """Setup logging based on configuration.

    Args:
        config: Experiment configuration
    """
    level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def setup_output_dirs(config: ExperimentConfig) -> tuple:
    """Create output directories if they don't exist.

    Args:
        config: Experiment configuration

    Returns:
        Tuple of (results_dir, figures_dir, tables_dir)
    """
    results_dir = Path(config.output_dir)
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"

    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    return results_dir, figures_dir, tables_dir


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration.

    Returns:
        ExperimentConfig with default values
    """
    return ExperimentConfig()


def get_runpod_config() -> ExperimentConfig:
    """Get RunPod experiment configuration for large-scale GPU runs.

    Returns:
        ExperimentConfig optimised for RunPod GPU instances.
    """
    return ExperimentConfig(
        T=5.0,
        dt=0.1,
        seed=42,
        n_samples=1_000_000,
        L_max=8,
        refinement_factor=2,
        use_gpu=True,
        target_epsilons=[0.1, 0.05, 0.02, 0.01, 0.005],
        verbose=True,
        gpu_only=False,
    )


if __name__ == "__main__":
    # Test configuration parsing
    print("Testing configuration module...")

    # Test default config
    default = get_default_config()
    print(f"\nDefault config:")
    print(f"  T={default.T}, dt={default.dt}, seed={default.seed}")
    print(f"  L_max={default.L_max}, use_gpu={default.use_gpu}")
    print(f"  target_epsilons={default.target_epsilons}")

    # Test validation
    print("\nValidation tests:")
    try:
        ExperimentConfig(T=-1.0)
    except ValueError as e:
        print(f"  Caught expected error: {e}")

    try:
        ExperimentConfig(refinement_factor=1)
    except ValueError as e:
        print(f"  Caught expected error: {e}")

    print("\nConfiguration module working correctly!")
