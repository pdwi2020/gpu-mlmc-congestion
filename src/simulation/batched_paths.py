"""Batched (many-paths-at-once) integration of the coupled congestion SDE.

`src.simulation.qmc` integrates one path at a time, which is the right shape for
its MLMC allocation code but roughly two orders of magnitude too slow for the
variance sweeps the reviewer response needs (tens of thousands of paths per
cell).  This module is the vectorised companion: identical recursion, `(P, n)`
state instead of `(n,)`.

The two implementations must not drift apart, so `assert_matches_reference`
checks the batched step against `qmc.coupled_em_step` for P=1 and every runner
here calls it at startup.  A silent divergence between them would corrupt every
number in the baseline tables while leaving both code paths individually
plausible, so the check is not optional.

Sign conventions, the predictor-corrector structure and the max(0, .) reflection
all follow `qmc.coupled_em_step` exactly; see that function for the derivation
and its relationship to `src/network/sde.py`.
"""
from __future__ import annotations

import math

import numpy as np


def em_step_batch(c, influence, beta, sigma, dt, dw, lam, clamp=True):
    """One predictor-corrector Euler-Maruyama step for P paths at once.

    `c` is (P, n); the single-path `influence @ c` becomes `c @ influence.T`.
    `beta`, `sigma` and `lam` may be scalars or (n,) arrays, which is what makes
    the heterogeneous-load sweep possible.

    clamp=False drops the reflection, giving the linear recursion whose mean is
    exactly `qmc.fluid_limit_mean` -- used as an unbiased control variate.
    """
    diffusion = sigma * dw
    drift_n = c @ influence.T - beta * c + lam
    c_pred = c + drift_n * dt + diffusion
    if clamp:
        c_pred = np.maximum(0.0, c_pred)
    drift_p = c_pred @ influence.T - beta * c_pred + lam
    out = c + drift_p * dt + diffusion
    return np.maximum(0.0, out) if clamp else out


def run_paths(dw_paths, influence, beta, sigma, dt, lam, want_linear=False):
    """Integrate a batch of Brownian paths; return the terminal functional.

    `dw_paths` is (P, n_steps, n), already sqrt(dt)-scaled.  Returns
    (Y, Y_linear, clamp_frac) where Y = mean_i C_i(T), Y_linear is the same
    functional under the unreflected recursion driven by the identical
    increments (None unless want_linear), and clamp_frac is the fraction of
    updates on which the reflection actually bound -- the diagnostic that says
    whether the reflected and linear models are even distinguishable.
    """
    n_paths, n_steps, n_nodes = dw_paths.shape
    c = np.zeros((n_paths, n_nodes))
    c_lin = np.zeros((n_paths, n_nodes)) if want_linear else None
    n_clamped = 0
    for k in range(n_steps):
        dw = dw_paths[:, k, :]
        c = em_step_batch(c, influence, beta, sigma, dt, dw, lam, clamp=True)
        n_clamped += int(np.count_nonzero(c <= 0.0))
        if want_linear:
            c_lin = em_step_batch(c_lin, influence, beta, sigma, dt, dw, lam,
                                  clamp=False)
    clamp_frac = n_clamped / float(n_paths * n_steps * n_nodes)
    return (c.mean(axis=1),
            c_lin.mean(axis=1) if want_linear else None,
            clamp_frac)


def run_level_pair(dw_fine, influence, beta, sigma, dt_fine, lam, refinement):
    """MLMC level difference: fine and coarse paths on ONE Brownian path.

    `dw_fine` is (P, n_steps_fine, n).  The coarse path takes steps of
    dt_coarse = refinement * dt_fine driven by the SUM of the matching fine
    increments, which is what makes the telescoping identity hold -- exactly the
    coupling in `qmc.simulate_coupled_paths_with_lambda`, batched.  Returns
    (Y_fine, Y_coarse).
    """
    n_paths, n_steps_fine, n_nodes = dw_fine.shape
    m = int(refinement)
    if n_steps_fine % m:
        raise ValueError(f"fine step count {n_steps_fine} not divisible by {m}")
    dt_coarse = dt_fine * m
    c_f = np.zeros((n_paths, n_nodes))
    c_c = np.zeros((n_paths, n_nodes))
    for i_c in range(n_steps_fine // m):
        block = dw_fine[:, i_c * m:(i_c + 1) * m, :]
        for j in range(m):
            c_f = em_step_batch(c_f, influence, beta, sigma, dt_fine,
                                block[:, j, :], lam)
        c_c = em_step_batch(c_c, influence, beta, sigma, dt_coarse,
                            block.sum(axis=1), lam)
    return c_f.mean(axis=1), c_c.mean(axis=1)


# ---------------------------------------------------------------------------
# Brownian bridge (dimension ordering for QMC)
# ---------------------------------------------------------------------------
def bridge_schedule(n_steps):
    """Bridge fill order over grid points 1..n_steps, point 0 pinned at zero.

    Returns (idx, i_left, i_right) triples: the terminal point first, then
    successive midpoint refinements -- the Caflisch-Morokoff-Owen ordering that
    concentrates path variance in the leading dimensions, which is the only
    reason Sobol points help on a path space at all.  i_left = -1 marks the
    terminal point.
    """
    order = [(n_steps, -1, -1)]
    queue = [(0, n_steps)]
    while queue:
        nxt = []
        for lo, hi in queue:
            if hi - lo < 2:
                continue
            mid = (lo + hi) // 2
            order.append((mid, lo, hi))
            nxt.append((lo, mid))
            nxt.append((mid, hi))
        queue = nxt
    return order


def paths_from_bridge(z, schedule, n_steps, dt):
    """Brownian increments from standard normals `z` (P, n_bridge, n)."""
    n_paths, _, n_nodes = z.shape
    w = np.zeros((n_paths, n_steps + 1, n_nodes))
    for b, (idx, lo, hi) in enumerate(schedule):
        if lo < 0:
            w[:, idx, :] = math.sqrt(idx * dt) * z[:, b, :]
            continue
        t_lo, t_mid, t_hi = lo * dt, idx * dt, hi * dt
        wl, wh = w[:, lo, :], w[:, hi, :]
        frac = (t_mid - t_lo) / (t_hi - t_lo)
        sd = math.sqrt((t_mid - t_lo) * (t_hi - t_mid) / (t_hi - t_lo))
        w[:, idx, :] = wl + frac * (wh - wl) + sd * z[:, b, :]
    return np.diff(w, axis=1)


# ---------------------------------------------------------------------------
def assert_matches_reference(influence, beta, sigma, dt, lam, rng):
    """Assert the batched step reproduces qmc.coupled_em_step for P=1."""
    from simulation.qmc import coupled_em_step

    n = influence.shape[0]
    c = rng.normal(0.5, 0.1, n)
    dw = rng.normal(0.0, math.sqrt(dt), n)
    ref = coupled_em_step(c, influence, beta, sigma, dt, dw, lam)
    got = em_step_batch(c[None, :], influence, beta, sigma, dt, dw[None, :],
                        lam)[0]
    if not np.allclose(ref, got, rtol=1e-12, atol=1e-14):
        raise AssertionError(
            "batched EM step disagrees with simulation.qmc.coupled_em_step "
            f"(max abs diff {np.max(np.abs(ref - got)):.3e})")
