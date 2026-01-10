"""
Discretization Module

This module provides time discretization strategies and coupling methods
for Multilevel Monte Carlo (MLMC) simulations.

Key concepts:
- Hierarchical time discretization with refinement factor M
- Brownian bridge coupling for variance reduction
- Consistent random number generation across levels

Classes:
    DiscretizationLevel: Represents a single discretization level
    MLMCHierarchy: Manages hierarchy of discretization levels
"""

import numpy as np
from typing import Optional, List, Tuple
import logging


logger = logging.getLogger(__name__)


class DiscretizationLevel:
    """
    Represents a single discretization level in MLMC hierarchy.

    Attributes:
        level: Level index (0 = coarsest)
        dt: Time step size for this level
        n_steps: Number of time steps for fixed duration
    """

    def __init__(self, level: int, dt: float, refinement_factor: int = 2):
        """
        Initialize discretization level.

        Args:
            level: Level index (l=0 is coarsest)
            dt: Time step size
            refinement_factor: Refinement factor M (dt_{l+1} = dt_l / M)
        """
        self.level = level
        self.dt = dt
        self.refinement_factor = refinement_factor

    def get_coarser_dt(self) -> float:
        """Get time step for one level coarser."""
        return self.dt * self.refinement_factor

    def get_finer_dt(self) -> float:
        """Get time step for one level finer."""
        return self.dt / self.refinement_factor

    def n_steps_for_duration(self, T: float) -> int:
        """Get number of time steps for given duration."""
        return int(T / self.dt)

    def __repr__(self) -> str:
        return f"DiscretizationLevel(level={self.level}, dt={self.dt})"


class MLMCHierarchy:
    """
    Manages hierarchy of discretization levels for MLMC.

    The hierarchy follows: dt_l = M^{-l} * dt_0
    where M is the refinement factor (typically M=2).
    """

    def __init__(self,
                 dt_coarsest: float,
                 L_max: int,
                 refinement_factor: int = 2):
        """
        Initialize MLMC discretization hierarchy.

        Args:
            dt_coarsest: Coarsest time step (level 0)
            L_max: Maximum level index
            refinement_factor: Refinement factor M
        """
        self.dt_coarsest = dt_coarsest
        self.L_max = L_max
        self.refinement_factor = refinement_factor

        # Create levels
        self.levels = []
        for l in range(L_max + 1):
            dt_l = dt_coarsest / (refinement_factor ** l)
            self.levels.append(DiscretizationLevel(l, dt_l, refinement_factor))

        logger.info(f"Created MLMC hierarchy: L_max={L_max}, M={refinement_factor}")
        logger.info(f"Time steps: {[f'{level.dt:.6f}' for level in self.levels]}")

    def get_level(self, l: int) -> DiscretizationLevel:
        """Get discretization level by index."""
        if l < 0 or l > self.L_max:
            raise ValueError(f"Level {l} out of range [0, {self.L_max}]")
        return self.levels[l]

    def get_timestep(self, level: int) -> float:
        """Get time step for given level."""
        return self.get_level(level).dt

    def get_refinement_factor_between(self, l_fine: int, l_coarse: int) -> int:
        """
        Get effective refinement factor between two levels.

        Args:
            l_fine: Fine level index
            l_coarse: Coarse level index

        Returns:
            M^{l_fine - l_coarse}
        """
        if l_fine < l_coarse:
            raise ValueError("Fine level must be >= coarse level")
        return self.refinement_factor ** (l_fine - l_coarse)

    def verify_coupling(self, l: int) -> bool:
        """
        Verify that level l can be coupled with level l-1.

        Coupling requires dt_{l-1} to be integer multiple of dt_l.

        Args:
            l: Level to verify

        Returns:
            True if coupling is valid
        """
        if l == 0:
            return True  # No coupling needed for level 0

        dt_coarse = self.get_timestep(l - 1)
        dt_fine = self.get_timestep(l)

        M = dt_coarse / dt_fine
        is_valid = np.isclose(M, round(M))

        if not is_valid:
            logger.warning(f"Level {l} coupling invalid: dt_coarse/dt_fine = {M}")

        return is_valid

    def __repr__(self) -> str:
        return (f"MLMCHierarchy(L_max={self.L_max}, "
                f"dt_0={self.dt_coarsest}, M={self.refinement_factor})")


def get_timestep(level: int, base_dt: float, refinement_factor: int = 2) -> float:
    """
    Get time step for given MLMC level.

    Args:
        level: Level index (0 = coarsest)
        base_dt: Base time step at level 0
        refinement_factor: Refinement factor M

    Returns:
        Time step: dt_l = base_dt / M^l
    """
    return base_dt / (refinement_factor ** level)


def generate_coupled_noise(dt_coarse: float,
                          dt_fine: float,
                          n_steps_fine: int,
                          dim: int = 1,
                          seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate coupled Brownian motion increments for MLMC.

    Uses Brownian bridge property: sum of fine increments = coarse increment.
    This ensures coupling between coarse and fine paths.

    Args:
        dt_coarse: Coarse time step
        dt_fine: Fine time step
        n_steps_fine: Number of fine time steps
        dim: Dimension (number of independent Wiener processes)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (dW_fine, dW_coarse)
        - dW_fine: shape (n_steps_fine, dim)
        - dW_coarse: shape (n_steps_coarse, dim)
    """
    if seed is not None:
        np.random.seed(seed)

    # Verify refinement relationship
    M = dt_coarse / dt_fine
    if not np.isclose(M, round(M)):
        raise ValueError(
            f"dt_coarse must be integer multiple of dt_fine. "
            f"Got ratio: {M}"
        )

    M = int(round(M))
    n_steps_coarse = n_steps_fine // M

    if n_steps_coarse * M != n_steps_fine:
        raise ValueError(
            f"n_steps_fine must be divisible by refinement factor M={M}. "
            f"Got n_steps_fine={n_steps_fine}"
        )

    # Generate fine increments
    dW_fine = np.random.normal(0, np.sqrt(dt_fine), (n_steps_fine, dim))

    # Aggregate to coarse increments (sum M consecutive fine increments)
    dW_coarse = np.zeros((n_steps_coarse, dim))
    for i in range(n_steps_coarse):
        dW_coarse[i] = np.sum(dW_fine[i * M:(i + 1) * M], axis=0)

    # Verify coupling property
    variance_coarse_expected = dt_coarse * dim
    variance_coarse_actual = np.var(dW_coarse) * n_steps_coarse

    logger.debug(f"Coupled noise: M={M}, n_fine={n_steps_fine}, n_coarse={n_steps_coarse}")
    logger.debug(f"Variance check: expected={variance_coarse_expected:.6f}, "
                f"actual={variance_coarse_actual:.6f}")

    return dW_fine, dW_coarse


def align_coarse_to_fine_grid(coarse_values: np.ndarray,
                               refinement_factor: int) -> np.ndarray:
    """
    Align coarse path values to fine time grid by repeating.

    Useful for computing differences Y_l - Y_{l-1} on same grid.

    Args:
        coarse_values: Coarse path values (n_coarse,)
        refinement_factor: Refinement factor M

    Returns:
        Aligned values on fine grid (n_fine,)
        where n_fine = n_coarse * M
    """
    n_coarse = len(coarse_values)
    n_fine = n_coarse * refinement_factor

    aligned = np.zeros(n_fine)

    for i in range(n_coarse):
        aligned[i * refinement_factor:(i + 1) * refinement_factor] = coarse_values[i]

    return aligned


def adaptive_timestep_selection(error_tolerance: float,
                                base_dt: float,
                                refinement_factor: int = 2,
                                max_levels: int = 10) -> int:
    """
    Adaptively select number of MLMC levels based on error tolerance.

    Uses theoretical weak convergence rate to estimate required levels.

    For SDEs with weak order α:
        Bias at level L ≈ C * dt_L^α = C * (base_dt / M^L)^α

    To achieve error tolerance ε, need:
        L ≥ log(C * base_dt^α / ε) / (α * log(M))

    Args:
        error_tolerance: Target error ε
        base_dt: Base time step at level 0
        refinement_factor: Refinement factor M
        max_levels: Maximum allowed levels

    Returns:
        Number of levels L
    """
    # Assume weak order α = 1 for Euler-Maruyama
    # and constant C = 1 (conservative estimate)
    alpha = 1.0
    C = 1.0

    bias_L0 = C * (base_dt ** alpha)

    # Solve for L
    if bias_L0 <= error_tolerance:
        return 0  # Level 0 already sufficient

    L = np.log(bias_L0 / error_tolerance) / (alpha * np.log(refinement_factor))
    L = int(np.ceil(L))

    # Clamp to max_levels
    L = min(L, max_levels)

    logger.info(f"Adaptive level selection: ε={error_tolerance}, L={L}")

    return L


class BrownianBridge:
    """
    Brownian bridge construction for path coupling in MLMC.

    Given coarse Brownian increments, constructs fine increments
    that are conditionally consistent.
    """

    @staticmethod
    def construct_fine_path(W_coarse: np.ndarray,
                           t_coarse: np.ndarray,
                           t_fine: np.ndarray,
                           seed: Optional[int] = None) -> np.ndarray:
        """
        Construct fine Brownian path conditioned on coarse path.

        Uses Brownian bridge interpolation.

        Args:
            W_coarse: Coarse Brownian path values (n_coarse,)
            t_coarse: Coarse time grid (n_coarse,)
            t_fine: Fine time grid (n_fine,)
            seed: Random seed

        Returns:
            Fine Brownian path (n_fine,)
        """
        if seed is not None:
            np.random.seed(seed)

        n_fine = len(t_fine)
        W_fine = np.zeros(n_fine)

        # Interpolate and add bridge noise
        for i in range(n_fine):
            t = t_fine[i]

            # Find bracketing coarse points
            idx_next = np.searchsorted(t_coarse, t, side='right')

            if idx_next == 0:
                # Before first coarse point
                W_fine[i] = 0.0
            elif idx_next >= len(t_coarse):
                # After last coarse point
                W_fine[i] = W_coarse[-1]
            else:
                # Between two coarse points
                idx_prev = idx_next - 1
                t_prev = t_coarse[idx_prev]
                t_next = t_coarse[idx_next]
                W_prev = W_coarse[idx_prev]
                W_next = W_coarse[idx_next]

                # Linear interpolation
                alpha = (t - t_prev) / (t_next - t_prev)
                W_interp = W_prev + alpha * (W_next - W_prev)

                # Bridge variance
                bridge_var = alpha * (1 - alpha) * (t_next - t_prev)

                # Add bridge noise
                W_fine[i] = W_interp + np.random.normal(0, np.sqrt(max(bridge_var, 0)))

        return W_fine


def estimate_discretization_error(values_coarse: np.ndarray,
                                  values_fine: np.ndarray) -> Dict:
    """
    Estimate discretization error between two levels.

    Args:
        values_coarse: Values at coarse level (N,)
        values_fine: Values at fine level (N,)

    Returns:
        Dictionary with error statistics
    """
    differences = values_fine - values_coarse

    error_stats = {
        'mean_difference': np.mean(differences),
        'std_difference': np.std(differences),
        'max_difference': np.max(np.abs(differences)),
        'rmse': np.sqrt(np.mean(differences ** 2)),
        'relative_error': np.std(differences) / (np.std(values_fine) + 1e-10)
    }

    return error_stats


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Discretization Module - Example Usage")
    print("=" * 60)

    # Example 1: Create MLMC hierarchy
    print("\n1. MLMC Hierarchy")
    print("-" * 60)

    hierarchy = MLMCHierarchy(dt_coarsest=0.1, L_max=5, refinement_factor=2)
    print(hierarchy)

    print("\nTime steps by level:")
    for l in range(hierarchy.L_max + 1):
        level = hierarchy.get_level(l)
        print(f"  Level {l}: dt = {level.dt:.6f}")

    # Verify coupling
    print("\nCoupling verification:")
    for l in range(1, hierarchy.L_max + 1):
        is_valid = hierarchy.verify_coupling(l)
        print(f"  Level {l}: {'Valid' if is_valid else 'Invalid'}")

    # Example 2: Generate coupled noise
    print("\n2. Coupled Brownian Motion")
    print("-" * 60)

    dW_fine, dW_coarse = generate_coupled_noise(
        dt_coarse=0.1,
        dt_fine=0.01,
        n_steps_fine=1000,
        dim=1,
        seed=42
    )

    print(f"Fine increments: shape={dW_fine.shape}, std={np.std(dW_fine):.6f}")
    print(f"Coarse increments: shape={dW_coarse.shape}, std={np.std(dW_coarse):.6f}")

    # Verify coupling
    for i in range(len(dW_coarse)):
        fine_sum = np.sum(dW_fine[i * 10:(i + 1) * 10])
        coarse_val = dW_coarse[i]
        assert np.abs(fine_sum - coarse_val) < 1e-10, "Coupling failed!"

    print("Coupling verification: PASSED")

    # Example 3: Adaptive level selection
    print("\n3. Adaptive Level Selection")
    print("-" * 60)

    for epsilon in [0.1, 0.05, 0.01, 0.005, 0.001]:
        L = adaptive_timestep_selection(
            error_tolerance=epsilon,
            base_dt=0.1,
            refinement_factor=2
        )
        dt_finest = get_timestep(L, base_dt=0.1, refinement_factor=2)
        print(f"  ε={epsilon:.4f} → L={L}, dt_finest={dt_finest:.6f}")

    print("\n" + "=" * 60)
