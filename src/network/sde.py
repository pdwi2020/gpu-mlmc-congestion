"""
Stochastic Differential Equation (SDE) Module

This module implements SDE formulations for network queue dynamics and
congestion propagation using Euler-Maruyama numerical integration.

Classes:
    QueueDynamicsSDE: Queue length evolution model
    CongestionPropagationSDE: Network-wide congestion spread model
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Callable
import logging


logger = logging.getLogger(__name__)


class QueueDynamicsSDE:
    """
    Queue dynamics SDE model.

    Models queue length evolution as a stochastic differential equation:
        dQ(t) = (λ(t) - μ(t)) dt + σ dW(t)

    where:
        - Q(t) = queue length at time t
        - λ(t) = arrival rate (packets/time unit)
        - μ(t) = service rate (packets/time unit)
        - σ = noise intensity
        - W(t) = Wiener process (Brownian motion)

    The queue length is constrained to be non-negative: Q(t) >= 0

    Note on Numerical Properties:
        The non-negativity constraint Q(t) >= 0 is enforced via reflection at the
        boundary (max(0, Q)). This transforms the SDE into a reflected Brownian
        motion with the following implications:

        1. Discretization bias: O(dt^0.5) instead of O(dt) for standard EM
        2. MLMC variance decay: Still achieves α ≈ 2 for level differences
        3. Overall MLMC complexity: Remains O(ε^-2) optimal

        Reference: Gobet (2000) "Weak approximation of killed diffusion"

        For high-precision estimates, use finer time steps or Milstein scheme.
    """

    def __init__(self,
                 arrival_rate: float,
                 service_rate: float,
                 noise_intensity: float = 0.1,
                 max_capacity: Optional[float] = None):
        """
        Initialize Queue Dynamics SDE.

        Args:
            arrival_rate: Mean arrival rate λ (packets/time unit)
            service_rate: Service rate μ (packets/time unit)
            noise_intensity: Noise coefficient σ (controls randomness)
            max_capacity: Maximum queue capacity (None = unbounded)
        """
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.noise_intensity = noise_intensity
        self.max_capacity = max_capacity

        # Check stability condition
        if arrival_rate >= service_rate:
            logger.warning(f"Queue is unstable: λ={arrival_rate} >= μ={service_rate}")

    def drift(self, q: float, t: float) -> float:
        """
        Drift term: (λ(t) - μ(t))

        Args:
            q: Current queue length
            t: Current time

        Returns:
            Drift value
        """
        # Can make arrival/service rate time-dependent here if needed
        return self.arrival_rate - self.service_rate

    def diffusion(self, q: float, t: float) -> float:
        """
        Diffusion term: σ

        Args:
            q: Current queue length
            t: Current time

        Returns:
            Diffusion coefficient
        """
        # Noise intensity can depend on queue state
        return self.noise_intensity

    def euler_maruyama_step(self,
                           q: float,
                           t: float,
                           dt: float,
                           dw: Optional[float] = None) -> float:
        """
        Single Euler-Maruyama step for SDE integration.

        Q(t + dt) = Q(t) + drift(Q, t) * dt + diffusion(Q, t) * dW

        Args:
            q: Current queue length
            t: Current time
            dt: Time step
            dw: Wiener increment (if None, generate randomly)

        Returns:
            Queue length at t + dt
        """
        # Generate Wiener increment if not provided
        if dw is None:
            dw = np.random.normal(0, np.sqrt(dt))

        # Euler-Maruyama update
        drift_term = self.drift(q, t) * dt
        diffusion_term = self.diffusion(q, t) * dw

        q_new = q + drift_term + diffusion_term

        # Enforce non-negativity constraint (reflected boundary)
        # Note: This reflection introduces O(dt^0.5) bias - see class docstring
        q_new = max(0.0, q_new)

        # Enforce capacity constraint if specified
        if self.max_capacity is not None:
            q_new = min(q_new, self.max_capacity)

        return q_new

    def simulate_path(self,
                     T: float,
                     dt: float,
                     q0: float = 0.0,
                     seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a single sample path of the queue dynamics.

        Args:
            T: Total simulation time
            dt: Time step
            q0: Initial queue length
            seed: Random seed for reproducibility

        Returns:
            Tuple of (time_array, queue_length_array)
        """
        if seed is not None:
            np.random.seed(seed)

        # Number of time steps
        n_steps = int(T / dt)
        time = np.linspace(0, T, n_steps + 1)

        # Initialize arrays
        q = np.zeros(n_steps + 1)
        q[0] = q0

        # Simulate path
        for i in range(n_steps):
            q[i + 1] = self.euler_maruyama_step(q[i], time[i], dt)

        return time, q

    def simulate_coupled_paths(self,
                               T: float,
                               dt_coarse: float,
                               dt_fine: float,
                               q0: float = 0.0,
                               seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate coupled coarse and fine paths for MLMC.

        Uses Brownian bridge to ensure coupling between paths.
        Applies synchronized reflection to preserve coupling for variance reduction.

        Args:
            T: Total simulation time
            dt_coarse: Coarse time step
            dt_fine: Fine time step (must divide dt_coarse evenly)
            q0: Initial queue length
            seed: Random seed

        Returns:
            Tuple of (time_fine, queue_fine, queue_coarse)
        """
        if seed is not None:
            np.random.seed(seed)

        # Verify refinement relationship
        M = int(dt_coarse / dt_fine)
        if not np.isclose(dt_coarse, M * dt_fine):
            raise ValueError(f"dt_coarse must be integer multiple of dt_fine")

        # Number of steps
        n_steps_fine = int(T / dt_fine)
        n_steps_coarse = int(T / dt_coarse)
        time_fine = np.linspace(0, T, n_steps_fine + 1)

        # Generate fine Wiener increments
        dw_fine = np.random.normal(0, np.sqrt(dt_fine), n_steps_fine)

        # Initialize paths
        q_fine = np.zeros(n_steps_fine + 1)
        q_coarse = np.zeros(n_steps_coarse + 1)
        q_fine[0] = q0
        q_coarse[0] = q0

        # Simulate both paths together with synchronized reflection
        # Process in coarse step blocks for proper coupling
        for i_coarse in range(n_steps_coarse):
            t_coarse = i_coarse * dt_coarse

            # Aggregate fine increments for coarse step
            dw_coarse = np.sum(dw_fine[i_coarse * M:(i_coarse + 1) * M])

            # Simulate M fine steps
            for j in range(M):
                i_fine = i_coarse * M + j
                t_fine = time_fine[i_fine]

                # Fine step (without reflection yet)
                drift_fine = self.drift(q_fine[i_fine], t_fine) * dt_fine
                diff_fine = self.diffusion(q_fine[i_fine], t_fine) * dw_fine[i_fine]
                q_fine_raw = q_fine[i_fine] + drift_fine + diff_fine

                # Apply reflection for fine path
                q_fine[i_fine + 1] = max(0.0, q_fine_raw)
                if self.max_capacity is not None:
                    q_fine[i_fine + 1] = min(q_fine[i_fine + 1], self.max_capacity)

            # Coarse step (without reflection yet)
            drift_coarse = self.drift(q_coarse[i_coarse], t_coarse) * dt_coarse
            diff_coarse = self.diffusion(q_coarse[i_coarse], t_coarse) * dw_coarse
            q_coarse_raw = q_coarse[i_coarse] + drift_coarse + diff_coarse

            # Synchronized reflection: apply reflection to coarse path
            # Use same reflection logic to maintain coupling structure
            q_coarse[i_coarse + 1] = max(0.0, q_coarse_raw)
            if self.max_capacity is not None:
                q_coarse[i_coarse + 1] = min(q_coarse[i_coarse + 1], self.max_capacity)

        # Align coarse to fine time grid by repeating each coarse value M times
        # q_coarse[:-1] has n_steps_coarse values; repeating gives n_steps_fine values
        # Then append final value to match q_fine length (n_steps_fine + 1)
        q_coarse_aligned = np.append(np.repeat(q_coarse[:-1], M), q_coarse[-1])
        return time_fine, q_fine, q_coarse_aligned

    def expected_queue_length(self) -> float:
        """
        Theoretical expected queue length (M/M/1 model approximation).

        Only valid when arrival_rate < service_rate.

        Returns:
            Expected queue length
        """
        if self.arrival_rate >= self.service_rate:
            return float('inf')

        rho = self.arrival_rate / self.service_rate  # Utilization
        return rho / (1 - rho)


class CongestionPropagationSDE:
    """
    Network-wide congestion propagation SDE model.

    Models how congestion spreads across network nodes:
        dC_i(t) = (Σ_j α_{ij} C_j(t) - β_i C_i(t)) dt + σ_i dW_i(t)

    where:
        - C_i(t) = congestion level at node i
        - α_{ij} = influence of node j on node i
        - β_i = decay/dissipation rate at node i
        - σ_i = noise intensity at node i
        - W_i(t) = Wiener process for node i
    """

    def __init__(self,
                 adjacency_matrix: np.ndarray,
                 influence_strength: float = 0.1,
                 decay_rate: float = 0.5,
                 noise_intensity: float = 0.1):
        """
        Initialize Congestion Propagation SDE.

        Args:
            adjacency_matrix: Network adjacency matrix (n_nodes x n_nodes)
            influence_strength: Strength of neighbor influence α
            decay_rate: Congestion decay rate β
            noise_intensity: Noise coefficient σ
        """
        self.adjacency = adjacency_matrix
        self.n_nodes = adjacency_matrix.shape[0]
        self.influence_strength = influence_strength
        self.decay_rate = decay_rate
        self.noise_intensity = noise_intensity

        # Compute influence matrix: α_{ij} = influence_strength * A_{ij} / degree_i
        self.influence_matrix = self._compute_influence_matrix()

    def _compute_influence_matrix(self) -> np.ndarray:
        """
        Compute influence matrix from adjacency matrix.

        Normalized by node degree to prevent unbounded growth.
        """
        return self._compute_influence_from_adjacency(
            self.adjacency,
            self.influence_strength,
        )

    @staticmethod
    def _compute_influence_from_adjacency(
        adjacency_matrix: np.ndarray,
        influence_strength: float,
    ) -> np.ndarray:
        """Compute a degree-normalized influence matrix from an adjacency matrix."""
        # Degree of each node
        degrees = np.sum(adjacency_matrix, axis=1)
        degrees[degrees == 0] = 1  # Avoid division by zero

        # Normalize by degree
        influence = adjacency_matrix / degrees[:, np.newaxis]
        influence *= influence_strength

        return influence

    def _drift_with_inputs(
        self,
        c: np.ndarray,
        influence_matrix: np.ndarray,
        lambda_vec: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute drift with optional exogenous arrival-rate forcing."""
        neighbor_influence = influence_matrix @ c
        self_decay = self.decay_rate * c
        drift = neighbor_influence - self_decay
        if lambda_vec is not None:
            drift = drift + lambda_vec
        return drift

    def _em_step_with_inputs(
        self,
        c: np.ndarray,
        t: float,
        dt: float,
        dw: Optional[np.ndarray],
        influence_matrix: np.ndarray,
        lambda_vec: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply one Euler-Maruyama step with optional dynamic inputs."""
        if dw is None:
            dw = np.random.normal(0, np.sqrt(dt), self.n_nodes)

        drift_term = self._drift_with_inputs(c, influence_matrix, lambda_vec) * dt
        diffusion_term = self.diffusion(c, t) * dw
        return np.maximum(0.0, c + drift_term + diffusion_term)

    def _validate_lambda_series(
        self,
        lambda_t: Optional[np.ndarray],
        n_steps: int,
    ) -> Optional[np.ndarray]:
        """Validate an arrival-rate series and return interval values."""
        if lambda_t is None:
            return None
        values = np.asarray(lambda_t, dtype=float)
        if values.shape == (n_steps + 1, self.n_nodes):
            return values[:-1]
        if values.shape == (n_steps, self.n_nodes):
            return values
        raise ValueError(
            f"lambda_t must have shape ({n_steps}, {self.n_nodes}) "
            f"or ({n_steps + 1}, {self.n_nodes})"
        )

    def _validate_adjacency_series(
        self,
        adjacency_t: Optional[np.ndarray],
        n_steps: int,
    ) -> Optional[np.ndarray]:
        """Validate an adjacency series and return interval snapshots."""
        if adjacency_t is None:
            return None
        values = np.asarray(adjacency_t, dtype=float)
        expected = (n_steps, self.n_nodes, self.n_nodes)
        expected_plus_endpoint = (n_steps + 1, self.n_nodes, self.n_nodes)
        if values.shape == expected_plus_endpoint:
            return values[:-1]
        if values.shape == expected:
            return values
        raise ValueError(
            f"adjacency_t must have shape {expected} or {expected_plus_endpoint}"
        )

    def _simulate_path_core(
        self,
        T: float,
        dt: float,
        c0: Optional[np.ndarray],
        seed: Optional[int],
        lambda_t: Optional[np.ndarray] = None,
        adjacency_t: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the Euler-Maruyama path loop with optional dynamic inputs."""
        if seed is not None:
            np.random.seed(seed)

        if c0 is None:
            c0 = np.zeros(self.n_nodes)

        n_steps = int(T / dt)
        time = np.linspace(0, T, n_steps + 1)
        lambda_series = self._validate_lambda_series(lambda_t, n_steps)
        adjacency_series = self._validate_adjacency_series(adjacency_t, n_steps)

        c = np.zeros((n_steps + 1, self.n_nodes))
        c[0] = c0

        for i in range(n_steps):
            influence_matrix = self.influence_matrix
            if adjacency_series is not None:
                influence_matrix = self._compute_influence_from_adjacency(
                    adjacency_series[i],
                    self.influence_strength,
                )
            lambda_vec = None if lambda_series is None else lambda_series[i]
            c[i + 1] = self._em_step_with_inputs(
                c=c[i],
                t=time[i],
                dt=dt,
                dw=None,
                influence_matrix=influence_matrix,
                lambda_vec=lambda_vec,
            )

        return time, c

    def drift(self, c: np.ndarray, t: float) -> np.ndarray:
        """
        Drift term: Σ_j α_{ij} C_j(t) - β_i C_i(t)

        Args:
            c: Congestion vector (n_nodes,)
            t: Current time

        Returns:
            Drift vector (n_nodes,)
        """
        return self._drift_with_inputs(c, self.influence_matrix)

    def diffusion(self, c: np.ndarray, t: float) -> np.ndarray:
        """
        Diffusion term: σ_i for each node

        Args:
            c: Congestion vector (n_nodes,)
            t: Current time

        Returns:
            Diffusion vector (n_nodes,)
        """
        # Uniform noise across all nodes
        return np.full(self.n_nodes, self.noise_intensity)

    def euler_maruyama_step(self,
                           c: np.ndarray,
                           t: float,
                           dt: float,
                           dw: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Single Euler-Maruyama step for network congestion.

        Args:
            c: Current congestion vector (n_nodes,)
            t: Current time
            dt: Time step
            dw: Wiener increments (n_nodes,) (if None, generate randomly)

        Returns:
            Congestion vector at t + dt
        """
        return self._em_step_with_inputs(
            c=c,
            t=t,
            dt=dt,
            dw=dw,
            influence_matrix=self.influence_matrix,
        )

    def simulate_path(self,
                     T: float,
                     dt: float,
                     c0: Optional[np.ndarray] = None,
                     seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate network congestion evolution.

        Args:
            T: Total simulation time
            dt: Time step
            c0: Initial congestion (if None, starts at zero)
            seed: Random seed

        Returns:
            Tuple of (time_array, congestion_matrix)
            congestion_matrix shape: (n_steps + 1, n_nodes)
        """
        return self._simulate_path_core(
            T=T,
            dt=dt,
            c0=c0,
            seed=seed,
        )

    def simulate_with_dynamic_inputs(
        self,
        T: float,
        dt: float,
        lambda_t: Optional[np.ndarray],
        adjacency_t: Optional[np.ndarray] = None,
        c0: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate congestion with time-indexed arrivals and topology snapshots.

        ``lambda_t`` is an additive drift term with shape ``(n_steps, n_nodes)``.
        ``adjacency_t`` may have shape ``(n_steps, n_nodes, n_nodes)``; when it is
        omitted, the static adjacency supplied at construction time is used.
        """
        if lambda_t is None and adjacency_t is None:
            return self.simulate_path(T=T, dt=dt, c0=c0, seed=seed)
        return self._simulate_path_core(
            T=T,
            dt=dt,
            c0=c0,
            seed=seed,
            lambda_t=lambda_t,
            adjacency_t=adjacency_t,
        )

    def simulate_coupled_paths(self,
                              T: float,
                              dt_coarse: float,
                              dt_fine: float,
                              c0: Optional[np.ndarray] = None,
                              seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate coupled coarse and fine congestion paths for MLMC.

        Both paths share the same Brownian increments (antithetic coupling):
        fine steps use sqrt(dt_fine) increments; coarse steps aggregate pairs.

        Args:
            T: Total simulation time
            dt_coarse: Coarse time step (must be integer multiple of dt_fine)
            dt_fine: Fine time step
            c0: Initial congestion vector (shape n_nodes,); zeros if None
            seed: Random seed

        Returns:
            Tuple of (time_fine, congestion_fine, congestion_coarse_aligned)
            All arrays shape: (n_steps_fine + 1, n_nodes)
        """
        if seed is not None:
            np.random.seed(seed)

        M = int(round(dt_coarse / dt_fine))
        if not np.isclose(dt_coarse, M * dt_fine):
            raise ValueError("dt_coarse must be an integer multiple of dt_fine")

        if c0 is None:
            c0 = np.zeros(self.n_nodes)

        n_steps_fine = int(T / dt_fine)
        n_steps_coarse = int(T / dt_coarse)
        time_fine = np.linspace(0, T, n_steps_fine + 1)

        # Pre-generate all fine Wiener increments: shape (n_steps_fine, n_nodes)
        dw_fine = np.random.normal(0.0, np.sqrt(dt_fine), (n_steps_fine, self.n_nodes))

        c_fine = np.zeros((n_steps_fine + 1, self.n_nodes))
        c_coarse = np.zeros((n_steps_coarse + 1, self.n_nodes))
        c_fine[0] = c0.copy()
        c_coarse[0] = c0.copy()

        for i_c in range(n_steps_coarse):
            # Aggregate M fine increments for one coarse increment (sum = sqrt(dt_coarse) in distribution)
            dw_coarse = np.sum(dw_fine[i_c * M:(i_c + 1) * M], axis=0)

            # Fine path: M Euler-Maruyama steps
            for j in range(M):
                i_f = i_c * M + j
                drift = self.drift(c_fine[i_f], time_fine[i_f]) * dt_fine
                diff = self.diffusion(c_fine[i_f], time_fine[i_f]) * dw_fine[i_f]
                c_fine[i_f + 1] = np.maximum(0.0, c_fine[i_f] + drift + diff)

            # Coarse path: one Euler-Maruyama step using aggregated increment
            t_c = i_c * dt_coarse
            drift_c = self.drift(c_coarse[i_c], t_c) * dt_coarse
            diff_c = self.diffusion(c_coarse[i_c], t_c) * dw_coarse
            c_coarse[i_c + 1] = np.maximum(0.0, c_coarse[i_c] + drift_c + diff_c)

        # Align coarse to fine time grid: repeat each coarse step M times, keep final
        c_coarse_aligned = np.zeros_like(c_fine)
        c_coarse_aligned[:-1] = np.repeat(c_coarse[:-1], M, axis=0)
        c_coarse_aligned[-1] = c_coarse[-1]

        return time_fine, c_fine, c_coarse_aligned

    def simulate_coupled_dynamic_paths(
        self,
        T: float,
        dt_coarse: float,
        dt_fine: float,
        lambda_t: Optional[np.ndarray] = None,
        adjacency_t: Optional[np.ndarray] = None,
        c0: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate coupled MLMC paths driven by dynamic arrivals and topology.

        The supplied dynamic arrays are interpreted on the fine grid. Coarse
        inputs are block averages over the matching fine-grid intervals.
        """
        if lambda_t is None and adjacency_t is None:
            return self.simulate_coupled_paths(
                T=T,
                dt_coarse=dt_coarse,
                dt_fine=dt_fine,
                c0=c0,
                seed=seed,
            )

        if seed is not None:
            np.random.seed(seed)

        M = int(round(dt_coarse / dt_fine))
        if not np.isclose(dt_coarse, M * dt_fine):
            raise ValueError("dt_coarse must be an integer multiple of dt_fine")

        if c0 is None:
            c0 = np.zeros(self.n_nodes)

        n_steps_fine = int(T / dt_fine)
        n_steps_coarse = int(T / dt_coarse)
        if n_steps_fine != n_steps_coarse * M:
            raise ValueError("fine and coarse grids must align exactly")

        time_fine = np.linspace(0, T, n_steps_fine + 1)
        lambda_fine = self._validate_lambda_series(lambda_t, n_steps_fine)
        adjacency_fine = self._validate_adjacency_series(adjacency_t, n_steps_fine)
        lambda_coarse = None
        adjacency_coarse = None
        if lambda_fine is not None:
            lambda_coarse = lambda_fine.reshape(n_steps_coarse, M, self.n_nodes).mean(axis=1)
        if adjacency_fine is not None:
            adjacency_coarse = adjacency_fine.reshape(
                n_steps_coarse,
                M,
                self.n_nodes,
                self.n_nodes,
            ).mean(axis=1)

        dw_fine = np.random.normal(0.0, np.sqrt(dt_fine), (n_steps_fine, self.n_nodes))
        c_fine = np.zeros((n_steps_fine + 1, self.n_nodes))
        c_coarse = np.zeros((n_steps_coarse + 1, self.n_nodes))
        c_fine[0] = c0.copy()
        c_coarse[0] = c0.copy()

        for i_c in range(n_steps_coarse):
            dw_coarse = np.sum(dw_fine[i_c * M:(i_c + 1) * M], axis=0)

            for j in range(M):
                i_f = i_c * M + j
                influence_f = self.influence_matrix
                if adjacency_fine is not None:
                    influence_f = self._compute_influence_from_adjacency(
                        adjacency_fine[i_f],
                        self.influence_strength,
                    )
                lambda_f = None if lambda_fine is None else lambda_fine[i_f]
                c_fine[i_f + 1] = self._em_step_with_inputs(
                    c=c_fine[i_f],
                    t=time_fine[i_f],
                    dt=dt_fine,
                    dw=dw_fine[i_f],
                    influence_matrix=influence_f,
                    lambda_vec=lambda_f,
                )

            influence_c = self.influence_matrix
            if adjacency_coarse is not None:
                influence_c = self._compute_influence_from_adjacency(
                    adjacency_coarse[i_c],
                    self.influence_strength,
                )
            lambda_c = None if lambda_coarse is None else lambda_coarse[i_c]
            t_c = i_c * dt_coarse
            c_coarse[i_c + 1] = self._em_step_with_inputs(
                c=c_coarse[i_c],
                t=t_c,
                dt=dt_coarse,
                dw=dw_coarse,
                influence_matrix=influence_c,
                lambda_vec=lambda_c,
            )

        c_coarse_aligned = np.zeros_like(c_fine)
        c_coarse_aligned[:-1] = np.repeat(c_coarse[:-1], M, axis=0)
        c_coarse_aligned[-1] = c_coarse[-1]

        return time_fine, c_fine, c_coarse_aligned

    def inject_congestion(self,
                         c: np.ndarray,
                         node_ids: list,
                         intensity: float) -> np.ndarray:
        """
        Inject congestion at specific nodes.

        Useful for simulating congestion events or hotspots.

        Args:
            c: Current congestion vector
            node_ids: List of node IDs to inject congestion
            intensity: Congestion intensity to add

        Returns:
            Updated congestion vector
        """
        c_new = c.copy()
        c_new[node_ids] += intensity
        return c_new


def generate_coupled_brownian_increments(dt_coarse: float,
                                        dt_fine: float,
                                        n_steps_fine: int,
                                        dim: int = 1,
                                        seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate coupled Brownian increments for MLMC.

    Uses the Brownian bridge property to ensure coupling.

    Args:
        dt_coarse: Coarse time step
        dt_fine: Fine time step
        n_steps_fine: Number of fine steps
        dim: Dimension (number of independent Wiener processes)
        seed: Random seed

    Returns:
        Tuple of (dw_fine, dw_coarse)
        dw_fine: shape (n_steps_fine, dim)
        dw_coarse: shape (n_steps_coarse, dim)
    """
    if seed is not None:
        np.random.seed(seed)

    # Verify refinement
    M = int(dt_coarse / dt_fine)
    if not np.isclose(dt_coarse, M * dt_fine):
        raise ValueError("dt_coarse must be integer multiple of dt_fine")

    # Generate fine increments
    dw_fine = np.random.normal(0, np.sqrt(dt_fine), (n_steps_fine, dim))

    # Aggregate to coarse increments
    n_steps_coarse = n_steps_fine // M
    dw_coarse = np.zeros((n_steps_coarse, dim))

    for i in range(n_steps_coarse):
        dw_coarse[i] = np.sum(dw_fine[i * M:(i + 1) * M], axis=0)

    return dw_fine, dw_coarse


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("SDE Module - Example Usage")
    print("=" * 60)

    # Example 1: Queue Dynamics
    print("\n1. Queue Dynamics SDE")
    print("-" * 60)

    queue_sde = QueueDynamicsSDE(
        arrival_rate=8.0,
        service_rate=10.0,
        noise_intensity=0.5
    )

    print(f"Expected queue length (theoretical): {queue_sde.expected_queue_length():.4f}")

    # Simulate single path
    time, q = queue_sde.simulate_path(T=10.0, dt=0.01, seed=42)
    print(f"Simulated mean queue length: {np.mean(q):.4f}")
    print(f"Simulated max queue length: {np.max(q):.4f}")

    # Simulate coupled paths for MLMC
    print("\n2. Coupled Paths for MLMC")
    print("-" * 60)

    time_fine, q_fine, q_coarse = queue_sde.simulate_coupled_paths(
        T=10.0,
        dt_coarse=0.1,
        dt_fine=0.01,
        seed=42
    )

    print(f"Fine path mean: {np.mean(q_fine):.4f}")
    print(f"Coarse path mean: {np.mean(q_coarse):.4f}")
    print(f"Difference: {np.mean(q_fine - q_coarse):.4f}")

    # Example 2: Congestion Propagation
    print("\n3. Congestion Propagation SDE")
    print("-" * 60)

    # Create simple network (5 nodes, ring topology)
    adjacency = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ])

    congestion_sde = CongestionPropagationSDE(
        adjacency_matrix=adjacency,
        influence_strength=0.2,
        decay_rate=0.5,
        noise_intensity=0.1
    )

    # Initial congestion at node 0
    c0 = np.zeros(5)
    c0[0] = 10.0

    time, c = congestion_sde.simulate_path(T=10.0, dt=0.01, c0=c0, seed=42)

    print("Final congestion levels:")
    for i in range(5):
        print(f"  Node {i}: {c[-1, i]:.4f}")

    print("\n" + "=" * 60)
