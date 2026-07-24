"""Tests for SIMT two-bucket adaptive pathwise stepping in GPUCoupledPropagationMLMC.

These tests exist because `_em_step_adaptive` was previously dead code: it was
gated behind a constructor flag that no code path ever consulted, so every
published number was produced with the feature inert.  They pin down the
properties the manuscript claims for the scheme, so that a regression that
silently disconnects it again fails the suite instead of reaching a reviewer:

  * the adaptive stepper is really entered when `adaptive_stepping=True`
    (asserted with a call counter, never by reading the source);
  * each bucket is advanced by a single batched ``torch.mm`` per drift
    evaluation, with a matmul count that does not depend on the number of
    paths -- the property that actually avoids SIMT thread divergence;
  * the default flags reproduce the pre-change integrator bit for bit;
  * fine/coarse common-random-number coupling survives refinement;
  * strong order 1/2 is retained empirically;
  * queue occupancy never goes negative through refinement.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

torch = pytest.importorskip("torch")

from gpu.parallel_mc import GPUCoupledPropagationMLMC  # noqa: E402


# ----------------------------------------------------------------- helpers
def chain_adj(n: int) -> np.ndarray:
    """Undirected chain graph adjacency, the topology used by the GPU tests."""
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


def make_sim(n_nodes: int = 6, seed: int = 42, noise_intensity: float = 0.1,
             **kwargs) -> GPUCoupledPropagationMLMC:
    return GPUCoupledPropagationMLMC(
        chain_adj(n_nodes), influence_strength=0.2, decay_rate=0.5,
        noise_intensity=noise_intensity, seed=seed, **kwargs)


def count_mm(fn):
    """Run `fn` with torch.mm instrumented; return (result, n_calls, arg_shapes)."""
    real_mm = torch.mm
    record = []

    def counting_mm(a, b, *args, **kwargs):
        record.append(tuple(b.shape))
        return real_mm(a, b, *args, **kwargs)

    torch.mm = counting_mm
    try:
        result = fn()
    finally:
        torch.mm = real_mm
    return result, len(record), record


def reference_em_step(sim, c_batch, dt, dw, influence=None, lambda_vec=None):
    """Frozen copy of `_em_step` as it stood before adaptive stepping was wired in.

    Reproduced verbatim from the pre-change source so that bitwise identity of
    the default configuration is checked against an independent implementation
    rather than against the code under test.
    """
    t = torch
    if influence is None:
        influence = sim._influence
    noise = sim.noise_intensity * dw

    drift_n = t.mm(influence, c_batch) - sim.decay_rate * c_batch
    if lambda_vec is not None:
        drift_n = drift_n + lambda_vec[:, None]
    c_pred = t.clamp_min(c_batch + drift_n * dt + noise, 0.0)

    drift_pred = t.mm(influence, c_pred) - sim.decay_rate * c_pred
    if lambda_vec is not None:
        drift_pred = drift_pred + lambda_vec[:, None]
    return t.clamp_min(c_batch + drift_pred * dt + noise, 0.0)


def reference_level_states(sim, level, n_samples, T, base_dt):
    """Frozen copy of the pre-change `_run_level_state_tensors` path loop.

    Consumes the global RNG in exactly the original order, so a mismatch would
    catch a change in randomness consumption as well as in arithmetic.
    """
    dev = sim._device
    dt_fine = base_dt / (sim.refinement_factor ** level)
    n_steps_fine = int(T / dt_fine)

    if level == 0:
        c_fine = torch.zeros(sim.n_nodes, n_samples, device=dev, dtype=torch.float32)
        for _ in range(n_steps_fine):
            dw = torch.randn(sim.n_nodes, n_samples, device=dev) * (dt_fine ** 0.5)
            c_fine = reference_em_step(sim, c_fine, dt_fine, dw)
        return c_fine, torch.zeros_like(c_fine)

    M = sim.refinement_factor
    dt_coarse = dt_fine * M
    n_steps_coarse = int(T / dt_coarse)
    c_fine = torch.zeros(sim.n_nodes, n_samples, device=dev, dtype=torch.float32)
    c_coarse = torch.zeros(sim.n_nodes, n_samples, device=dev, dtype=torch.float32)

    for _ in range(n_steps_coarse):
        dw_sum = torch.zeros(sim.n_nodes, n_samples, device=dev, dtype=torch.float32)
        for _ in range(M):
            dw_f = torch.randn(sim.n_nodes, n_samples, device=dev) * (dt_fine ** 0.5)
            c_fine = reference_em_step(sim, c_fine, dt_fine, dw_f)
            dw_sum = dw_sum + dw_f
        c_coarse = reference_em_step(sim, c_coarse, dt_coarse, dw_sum)

    return c_fine, c_coarse


# ============================================================ invocation ===
class TestAdaptiveStepperIsReachable:
    """The flag must reach the path loop -- this is the defect being fixed."""

    def _instrument(self, sim):
        """Wrap the adaptive stepper on the instance with a call counter."""
        calls = {"n": 0, "roles": []}
        original = sim._em_step_adaptive

        def counting(*args, **kwargs):
            calls["n"] += 1
            calls["roles"].append(kwargs.get("role", "fine"))
            return original(*args, **kwargs)

        sim._em_step_adaptive = counting
        return calls

    def test_adaptive_stepper_is_called_when_flag_on(self):
        """run_level must route every nominal step through `_em_step_adaptive`."""
        sim = make_sim(adaptive_stepping=True, adaptive_rtol=1e-5)
        calls = self._instrument(sim)

        sim.run_level(level=0, n_samples=32, T=1.0, base_dt=0.1)

        assert calls["n"] == 10, (
            f"level 0 with T=1.0, dt=0.1 has 10 nominal steps but the adaptive "
            f"stepper was entered {calls['n']} times -- the flag is not wired in"
        )

    def test_adaptive_stepper_is_not_called_when_flag_off(self):
        """The default configuration must not pay for machinery it disabled."""
        sim = make_sim(adaptive_stepping=False)
        calls = self._instrument(sim)

        sim.run_level(level=0, n_samples=32, T=1.0, base_dt=0.1)

        assert calls["n"] == 0
        assert sim.adaptive_bucket_history == []

    def test_both_fine_and_coarse_paths_are_adapted(self):
        """Level l > 0 must adapt fine and coarse paths with separate state."""
        sim = make_sim(adaptive_stepping=True, adaptive_rtol=1e-5)
        calls = self._instrument(sim)

        sim.run_level(level=1, n_samples=32, T=1.0, base_dt=0.1)

        roles = calls["roles"]
        assert roles.count("fine") == 20, "level 1 has 20 fine steps"
        assert roles.count("coarse") == 10, "level 1 has 10 coarse steps"
        assert set(sim._adaptive_h_scale) == {"fine", "coarse"}, (
            "fine and coarse must carry independent step-size controller state"
        )

    def test_bucket_history_records_occupancy(self):
        """Occupancy diagnostics must be populated for the ablation experiment."""
        sim = make_sim(adaptive_stepping=True, adaptive_rtol=1e-5)
        sim.run_level(level=0, n_samples=64, T=1.0, base_dt=0.1)

        hist = sim.adaptive_bucket_history
        assert len(hist) == 10
        for rec in hist:
            assert rec["n_full"] + rec["n_half"] == 64, (
                "every path must land in exactly one of the two buckets"
            )
            assert 0.0 <= rec["frac_half"] <= 1.0


# ================================================================ matmuls ===
class TestBucketMatmulAccounting:
    """One batched torch.mm per bucket per drift evaluation, never per path."""

    def _mixed_scale(self, sim, n_paths, role="fine"):
        """Force a genuinely mixed ensemble: half full-step, half half-step."""
        scale = torch.where(
            torch.arange(n_paths) % 2 == 0,
            torch.ones(n_paths), torch.full((n_paths,), 0.5))
        sim._adaptive_h_scale[role] = scale
        return scale

    def _run_one_step(self, sim, n_paths):
        c = torch.rand(sim.n_nodes, n_paths)
        dw = torch.randn(sim.n_nodes, n_paths) * 0.1
        return count_mm(lambda: sim._em_step_adaptive(c, 0.05, dw, role="fine"))

    @pytest.mark.parametrize("n_paths", [64, 4096])
    def test_mixed_buckets_use_one_matmul_per_bucket_per_drift_eval(self, n_paths):
        """Predictor-corrector: full bucket 2 matmuls, half bucket 2x2 = 4."""
        sim = make_sim(adaptive_stepping=True, adaptive_rtol=1e-5)
        self._mixed_scale(sim, n_paths)
        _, n_mm, shapes = self._run_one_step(sim, n_paths)

        assert n_mm == 6, (
            f"expected 6 matmuls (full bucket: predictor+corrector; half bucket: "
            f"2 sub-steps x predictor+corrector), got {n_mm}"
        )
        # Two matmuls over the full bucket, four over the half bucket, and every
        # one of them spans the whole bucket -- no per-path or per-chunk work.
        half = n_paths // 2
        assert sorted(s[1] for s in shapes) == [half] * 6
        assert all(s[0] == sim.n_nodes for s in shapes)

    @pytest.mark.parametrize("n_paths", [64, 4096])
    def test_euler_clamp_uses_one_matmul_per_bucket_per_substep(self, n_paths):
        """Plain Euler has a single drift stage: full bucket 1, half bucket 2."""
        sim = make_sim(adaptive_stepping=True, adaptive_rtol=1e-5,
                       reflection="euler_clamp")
        self._mixed_scale(sim, n_paths)
        _, n_mm, shapes = self._run_one_step(sim, n_paths)

        assert n_mm == 3, f"expected 1 (full) + 2 (half sub-steps) matmuls, got {n_mm}"
        assert sorted(s[1] for s in shapes) == [n_paths // 2] * 3

    def test_matmul_count_is_independent_of_path_count(self):
        """The SIMT claim: work per step scales with buckets, not with threads."""
        counts = []
        for n_paths in (32, 256, 2048, 16384):
            sim = make_sim(adaptive_stepping=True, adaptive_rtol=1e-5)
            self._mixed_scale(sim, n_paths)
            _, n_mm, _ = self._run_one_step(sim, n_paths)
            counts.append(n_mm)

        assert len(set(counts)) == 1, (
            f"matmul count varies with the number of paths ({counts}); the "
            f"implementation is doing per-path work and divergence is not avoided"
        )

    def test_single_bucket_costs_no_more_than_fixed_stepping(self):
        """With every path full-step, the adaptive stepper issues the same 2 matmuls."""
        n_paths = 512
        sim = make_sim(adaptive_stepping=True, adaptive_rtol=1e9)
        sim._adaptive_h_scale["fine"] = torch.ones(n_paths)
        _, n_adaptive, _ = self._run_one_step(sim, n_paths)

        fixed = make_sim()
        c = torch.rand(fixed.n_nodes, n_paths)
        dw = torch.randn(fixed.n_nodes, n_paths) * 0.1
        _, n_fixed, _ = count_mm(lambda: fixed._em_step(c, 0.05, dw))

        assert n_adaptive == n_fixed == 2

    def test_internal_matmul_counter_matches_instrumented_count(self):
        """`adaptive_mm_calls` is the hook experiments use; keep it truthful."""
        n_paths = 128
        sim = make_sim(adaptive_stepping=True, adaptive_rtol=1e-5)
        self._mixed_scale(sim, n_paths)
        _, n_mm, _ = self._run_one_step(sim, n_paths)

        assert sim.adaptive_mm_calls == n_mm


# ============================================================== bitwise ====
class TestDefaultFlagsAreBitwiseUnchanged:
    """Defaults must reproduce every published number exactly."""

    def test_em_step_matches_frozen_reference(self):
        """The step primitive is unchanged for the default reflection scheme."""
        sim = make_sim(n_nodes=6, seed=11)
        torch.manual_seed(1234)
        c = torch.rand(6, 32)
        dw = torch.randn(6, 32) * 0.1
        lam = torch.rand(6) * 0.05

        assert torch.equal(sim._em_step(c, 0.05, dw),
                           reference_em_step(sim, c, 0.05, dw))
        assert torch.equal(sim._em_step(c, 0.05, dw, None, lam),
                           reference_em_step(sim, c, 0.05, dw, None, lam))

    @pytest.mark.parametrize("level", [0, 1, 2])
    def test_level_trajectories_match_frozen_reference(self, level):
        """Whole-loop identity, including the order of RNG consumption."""
        sim_ref = make_sim(n_nodes=5, seed=42)
        ref_fine, ref_coarse = reference_level_states(
            sim_ref, level, n_samples=48, T=1.0, base_dt=0.1)

        sim_new = make_sim(n_nodes=5, seed=42)
        new_fine, new_coarse = sim_new._run_level_state_tensors(
            level=level, n_samples=48, T=1.0, base_dt=0.1)

        assert torch.equal(new_fine, ref_fine), (
            f"level {level} fine trajectory drifted from the pre-change result"
        )
        assert torch.equal(new_coarse, ref_coarse), (
            f"level {level} coarse trajectory drifted from the pre-change result"
        )

    def test_adaptive_with_unreachable_tolerance_is_bitwise_fixed_stepping(self):
        """No refinement means every path stays full-step, so nothing may change."""
        fixed = make_sim(n_nodes=5, seed=17)
        f_fine, f_coarse = fixed._run_level_state_tensors(
            level=2, n_samples=64, T=1.0, base_dt=0.1)

        never = make_sim(n_nodes=5, seed=17, adaptive_stepping=True,
                         adaptive_rtol=1e9)
        a_fine, a_coarse = never._run_level_state_tensors(
            level=2, n_samples=64, T=1.0, base_dt=0.1)

        assert all(r["n_half"] == 0 for r in never.adaptive_bucket_history)
        assert torch.equal(a_fine, f_fine)
        assert torch.equal(a_coarse, f_coarse)


# =============================================================== fastpath ==
class TestUniformMaskFastPath:
    """Measured occupancy is nearly always all-full or all-half, so the
    uniform-mask case skips the bucket gather/scatter.  It must be a pure cost
    saving: identical numbers, identical matmul count, same diagnostics."""

    def _step_once(self, sim, scale_values, n_paths=256, seed=5):
        torch.manual_seed(seed)
        c = torch.rand(sim.n_nodes, n_paths)
        dw = torch.randn(sim.n_nodes, n_paths) * 0.1
        sim._adaptive_h_scale["fine"] = scale_values
        return sim._em_step_adaptive(c, 0.05, dw, role="fine"), c, dw

    def test_mixed_ensemble_agrees_with_uniform_ensembles(self):
        """A path's update must not depend on which other paths share its batch.

        The full-step columns of a mixed ensemble must equal what those paths
        get in an all-full ensemble (the fast path), and likewise for the
        half-step columns.  If the fast path and the bucketed path disagreed,
        results would silently depend on bucket occupancy.
        """
        n_paths = 256
        mixed_scale = torch.where(
            torch.arange(n_paths) % 2 == 0,
            torch.ones(n_paths), torch.full((n_paths,), 0.5))

        sim_mixed = make_sim(adaptive_stepping=True, adaptive_rtol=1e-5)
        out_mixed, c, dw = self._step_once(sim_mixed, mixed_scale, n_paths)

        idx_full = (torch.arange(n_paths) % 2 == 0).nonzero(as_tuple=True)[0]
        idx_half = (torch.arange(n_paths) % 2 == 1).nonzero(as_tuple=True)[0]

        sim_full = make_sim(adaptive_stepping=True, adaptive_rtol=1e-5)
        sim_full._adaptive_h_scale["fine"] = torch.ones(idx_full.numel())
        out_full = sim_full._em_step_adaptive(
            c.index_select(1, idx_full), 0.05, dw.index_select(1, idx_full),
            role="fine")

        sim_half = make_sim(adaptive_stepping=True, adaptive_rtol=1e-5)
        sim_half._adaptive_h_scale["fine"] = torch.full((idx_half.numel(),), 0.5)
        out_half = sim_half._em_step_adaptive(
            c.index_select(1, idx_half), 0.05, dw.index_select(1, idx_half),
            role="fine")

        assert torch.equal(out_mixed.index_select(1, idx_full), out_full)
        assert torch.equal(out_mixed.index_select(1, idx_half), out_half)

    def test_uniform_steps_are_counted(self):
        """The fast-path counter drives the overhead analysis; keep it honest."""
        n_paths = 128
        sim = make_sim(adaptive_stepping=True, adaptive_rtol=1e-5)
        self._step_once(sim, torch.ones(n_paths), n_paths)
        assert sim._adaptive_uniform_steps == 1

        mixed = torch.where(torch.arange(n_paths) % 2 == 0,
                            torch.ones(n_paths), torch.full((n_paths,), 0.5))
        sim_mixed = make_sim(adaptive_stepping=True, adaptive_rtol=1e-5)
        self._step_once(sim_mixed, mixed, n_paths)
        assert sim_mixed._adaptive_uniform_steps == 0

    @pytest.mark.parametrize("uniform_scale,expected_mm", [
        (1.0, 2),   # all full-step: predictor + corrector
        (0.5, 4),   # all half-step: two sub-steps x predictor + corrector
    ])
    def test_uniform_fast_path_matmul_count(self, uniform_scale, expected_mm):
        n_paths = 128
        sim = make_sim(adaptive_stepping=True, adaptive_rtol=1e-5)
        scale = torch.full((n_paths,), uniform_scale)
        torch.manual_seed(5)
        c = torch.rand(sim.n_nodes, n_paths)
        dw = torch.randn(sim.n_nodes, n_paths) * 0.1
        sim._adaptive_h_scale["fine"] = scale

        _, n_mm, _ = count_mm(
            lambda: sim._em_step_adaptive(c, 0.05, dw, role="fine"))
        assert n_mm == expected_mm

    def test_occupancy_recorded_without_diagnostics(self):
        """Occupancy is free and always on; error statistics are opt-in."""
        sim = make_sim(adaptive_stepping=True, adaptive_rtol=1e-5)
        assert sim.adaptive_diagnostics is False
        sim.run_level(level=0, n_samples=64, T=1.0, base_dt=0.1)

        record = sim.adaptive_bucket_history[0]
        assert {"role", "dt", "n_full", "n_half", "frac_half"} <= set(record)
        assert "err_p50" not in record, (
            "error quantiles cost a sort and a sync; they must stay opt-in so "
            "that timing runs are not contaminated"
        )

    def test_diagnostics_add_error_distribution(self):
        sim = make_sim(adaptive_stepping=True, adaptive_rtol=1e-5,
                       adaptive_diagnostics=True)
        sim.run_level(level=0, n_samples=64, T=1.0, base_dt=0.1)

        record = sim.adaptive_bucket_history[0]
        for key in ("err_min", "err_p05", "err_p50", "err_p95", "err_max",
                    "err_spread"):
            assert key in record
        assert record["err_min"] <= record["err_p50"] <= record["err_max"]

    def test_diagnostics_do_not_change_trajectories(self):
        """Turning diagnostics on must be observation-only."""
        # Each simulator must be constructed immediately before its own run:
        # the constructor reseeds the global RNG, so building both up front
        # would leave the second run reading a different stream.
        a, _ = make_sim(n_nodes=5, seed=61, adaptive_stepping=True,
                        adaptive_rtol=1e-5)._run_level_state_tensors(
            level=1, n_samples=64, T=1.0, base_dt=0.1)
        b, _ = make_sim(n_nodes=5, seed=61, adaptive_stepping=True,
                        adaptive_rtol=1e-5, adaptive_diagnostics=True
                        )._run_level_state_tensors(
            level=1, n_samples=64, T=1.0, base_dt=0.1)
        assert torch.equal(a, b)


# =============================================================== coupling ==
class TestMLMCCouplingSurvivesRefinement:
    """Adaptive refinement must not consume randomness asymmetrically."""

    @staticmethod
    def _coupled_variance(level, **kwargs):
        sim = make_sim(n_nodes=8, seed=5, **kwargs)
        yf, yc = sim.run_level(level, 2048, T=1.0, base_dt=0.1)
        return float(np.var(yf - yc, ddof=1))

    @staticmethod
    def _independent_variance(level, **kwargs):
        a = make_sim(n_nodes=8, seed=101, **kwargs)
        b = make_sim(n_nodes=8, seed=202, **kwargs)
        yf, _ = a.run_level(level, 2048, T=1.0, base_dt=0.1)
        _, yc = b.run_level(level, 2048, T=1.0, base_dt=0.1)
        return float(np.var(yf - yc, ddof=1))

    @pytest.mark.parametrize("kwargs", [
        pytest.param({}, id="fixed"),
        pytest.param({"adaptive_stepping": True, "adaptive_rtol": 1e-5},
                     id="adaptive"),
    ])
    def test_coupled_variance_far_below_independent(self, kwargs):
        v_coupled = self._coupled_variance(2, **kwargs)
        v_independent = self._independent_variance(2, **kwargs)

        assert v_coupled < v_independent / 50.0, (
            f"common-random-number coupling lost: V_coupled={v_coupled:.3e} vs "
            f"V_independent={v_independent:.3e}; MLMC variance reduction is gone"
        )

    def test_adaptive_coupling_as_strong_as_fixed(self):
        """Refinement must not degrade the coupling relative to fixed stepping."""
        v_fixed = self._coupled_variance(2)
        v_adaptive = self._coupled_variance(
            2, adaptive_stepping=True, adaptive_rtol=1e-5)

        assert v_adaptive < 2.0 * v_fixed, (
            f"adaptive level-difference variance {v_adaptive:.3e} inflated over "
            f"fixed {v_fixed:.3e}"
        )

    def test_level_variance_still_decays_with_level(self):
        """Variance decay across levels must survive with the flag on."""
        sim = make_sim(n_nodes=8, seed=5, adaptive_stepping=True,
                       adaptive_rtol=1e-5)
        variances = []
        for level in (1, 2, 3):
            yf, yc = sim.run_level(level, 2048, T=1.0, base_dt=0.1)
            variances.append(float(np.var(yf - yc, ddof=1)))

        assert variances[1] < variances[0]
        assert variances[2] < variances[1]

    def test_randomness_consumption_is_independent_of_refinement(self):
        """Refined and unrefined runs must leave the RNG in the same state.

        This is the mechanism that protects the coupling: the half-step bucket
        splits the outer Wiener increment deterministically instead of drawing
        a fresh Brownian bridge sample.
        """
        states = []
        for rtol in (1e9, 1e-9):
            torch.manual_seed(2024)
            sim = make_sim(n_nodes=6, adaptive_stepping=True, adaptive_rtol=rtol)
            torch.manual_seed(2024)
            sim.run_level(level=1, n_samples=64, T=1.0, base_dt=0.1)
            states.append(torch.random.get_rng_state().clone())

        assert torch.equal(states[0], states[1]), (
            "refinement changed how much randomness was drawn; fine/coarse "
            "increments would decouple"
        )


# ========================================================= strong order ====
class TestStrongConvergence:
    """Strong order 1/2 must be retained, measured rather than assumed."""

    @staticmethod
    def _scalar_sim(**kwargs):
        """Scalar reflected OU: dc = (lam - k c) dt + sigma dW, c >= 0."""
        return GPUCoupledPropagationMLMC(
            np.zeros((1, 1), dtype=np.float32), influence_strength=0.0,
            decay_rate=1.0, noise_intensity=0.3, seed=1, **kwargs)

    @staticmethod
    def _drive(sim, dz, n_steps, T, lam):
        """Advance an ensemble on `n_steps` using block sums of the fine increments."""
        block = dz.shape[2] // n_steps
        dt = T / n_steps
        c = torch.zeros(1, dz.shape[1])
        sim.reset_adaptive_state()
        for i in range(n_steps):
            dw = dz[:, :, i * block:(i + 1) * block].sum(dim=2)
            c = sim._step(c, dt, dw, None, lam, role="fine")
        return c

    def test_ou_moments_match_closed_form(self):
        """Anchor the integrator on a problem with a known solution.

        With lam/k well above the reflecting barrier the clamp never binds, so
        the exact law at T is Gaussian with
            mean = (lam/k)(1 - e^{-kT}),  var = sigma^2 (1 - e^{-2kT}) / (2k).
        """
        T, dt, lam_val, k, sigma = 2.0, 0.002, 2.0, 1.0, 0.1
        sim = GPUCoupledPropagationMLMC(
            np.zeros((1, 1), dtype=np.float32), influence_strength=0.0,
            decay_rate=k, noise_intensity=sigma, seed=3)
        lam = torch.full((1,), lam_val)

        torch.manual_seed(7)
        n_paths, n_steps = 4096, int(T / dt)
        c = torch.zeros(1, n_paths)
        for _ in range(n_steps):
            dw = torch.randn(1, n_paths) * (dt ** 0.5)
            c = sim._em_step(c, dt, dw, None, lam)

        mean_exact = (lam_val / k) * (1.0 - np.exp(-k * T))
        var_exact = sigma ** 2 * (1.0 - np.exp(-2 * k * T)) / (2 * k)
        assert float(c.mean()) == pytest.approx(mean_exact, abs=0.01)
        assert float(c.var()) == pytest.approx(var_exact, rel=0.15)
        assert float(c.min()) > 0.0, "clamp must not bind in this regime"

    @pytest.mark.parametrize("kwargs", [
        pytest.param({}, id="fixed"),
        pytest.param({"adaptive_stepping": True, "adaptive_rtol": 1e-5},
                     id="adaptive"),
    ])
    def test_strong_order_at_least_one_half(self, kwargs):
        """Strong error at T against a Brownian-refined reference path."""
        T, n_ref, n_paths = 1.0, 1024, 1024
        lam = torch.full((1,), 0.8)
        torch.manual_seed(9)
        dz = torch.randn(1, n_paths, n_ref) * ((T / n_ref) ** 0.5)

        reference = self._drive(self._scalar_sim(**kwargs), dz, n_ref, T, lam)

        step_counts = (16, 32, 64, 128)
        errors, dts = [], []
        for n_steps in step_counts:
            c = self._drive(self._scalar_sim(**kwargs), dz, n_steps, T, lam)
            errors.append(float(torch.sqrt(((c - reference) ** 2).mean())))
            dts.append(T / n_steps)

        design = np.vstack([np.log(dts), np.ones(len(dts))]).T
        slope = float(np.linalg.lstsq(design, np.log(errors), rcond=None)[0][0])

        assert slope >= 0.4, (
            f"measured strong order {slope:.3f} is below the Euler-Maruyama "
            f"rate of 1/2 (errors {errors} at dt {dts})"
        )
        assert errors == sorted(errors, reverse=True), (
            "strong error must decrease monotonically as dt shrinks"
        )


# ============================================================ invariants ===
class TestPhysicalInvariants:
    """Queue occupancy is a physical quantity and cannot go negative."""

    @pytest.mark.parametrize("rtol", [1e-2, 1e-5, 1e-9])
    def test_occupancy_non_negative_through_refinement(self, rtol):
        sim = make_sim(n_nodes=8, seed=13, adaptive_stepping=True,
                       adaptive_rtol=rtol)
        c_fine, c_coarse = sim._run_level_state_tensors(
            level=2, n_samples=256, T=1.0, base_dt=0.1)

        assert float(c_fine.min()) >= 0.0
        assert float(c_coarse.min()) >= 0.0
        assert torch.isfinite(c_fine).all()
        assert torch.isfinite(c_coarse).all()

    def test_occupancy_non_negative_with_strong_arrivals(self):
        """A driven, heavily refined ensemble still respects the barrier."""
        n_nodes = 6
        sim = make_sim(n_nodes=n_nodes, seed=21, noise_intensity=0.5,
                       adaptive_stepping=True, adaptive_rtol=1e-9)
        lam_t = np.full((10, n_nodes), 0.4, dtype=np.float32)

        yf, yc = sim.run_level(level=1, n_samples=256, T=1.0, base_dt=0.1,
                               lambda_t=lam_t)

        assert np.all(yf >= 0.0) and np.all(yc >= 0.0)
        assert np.all(np.isfinite(yf)) and np.all(np.isfinite(yc))

    def test_bucket_assignment_stays_binary(self):
        """The manuscript claims exactly two buckets; the state must honour that."""
        sim = make_sim(n_nodes=6, seed=31, adaptive_stepping=True,
                       adaptive_rtol=1e-4)
        sim._run_level_state_tensors(level=1, n_samples=128, T=1.0, base_dt=0.1)

        for role, scale in sim._adaptive_h_scale.items():
            unique = set(np.unique(scale.cpu().numpy()).tolist())
            assert unique <= {1.0, 0.5}, (
                f"{role} step scales {unique} are not a two-bucket partition"
            )

    def test_coarse_nominal_step_stays_at_two_h(self):
        """Telescoping needs the coarse nominal step pinned at 2*h_l."""
        sim = make_sim(n_nodes=5, seed=41, adaptive_stepping=True,
                       adaptive_rtol=1e-5)
        base_dt, level = 0.1, 2
        sim._run_level_state_tensors(level=level, n_samples=32, T=1.0,
                                     base_dt=base_dt)

        h_l = base_dt / (sim.refinement_factor ** level)
        fine_dts = {r["dt"] for r in sim.adaptive_bucket_history if r["role"] == "fine"}
        coarse_dts = {r["dt"] for r in sim.adaptive_bucket_history
                      if r["role"] == "coarse"}

        assert sorted(fine_dts) == pytest.approx([h_l])
        assert sorted(coarse_dts) == pytest.approx([2.0 * h_l])


# ================================================= independent ablation ====
class TestFlagsAreIndependentlySwitchable:
    """The ablation ladder needs each component toggled on its own."""

    def test_reflection_default_is_predictor_corrector(self):
        assert make_sim().reflection == "predictor_corrector"
        assert make_sim().adaptive_stepping is False
        assert make_sim().adaptive_error_estimator == "embedded"

    def test_euler_clamp_changes_the_trajectory(self):
        """The corrector must be a real ablation, not a no-op rename."""
        pc = make_sim(n_nodes=5, seed=55)
        pc_fine, _ = pc._run_level_state_tensors(level=1, n_samples=64, T=1.0,
                                                 base_dt=0.1)
        eu = make_sim(n_nodes=5, seed=55, reflection="euler_clamp")
        eu_fine, _ = eu._run_level_state_tensors(level=1, n_samples=64, T=1.0,
                                                 base_dt=0.1)

        assert not torch.equal(pc_fine, eu_fine)
        assert float(eu_fine.min()) >= 0.0, "clamping must still enforce the barrier"

    def test_euler_clamp_halves_the_drift_evaluations(self):
        sim = make_sim(n_nodes=5, seed=55, reflection="euler_clamp")
        c = torch.rand(5, 128)
        dw = torch.randn(5, 128) * 0.1
        _, n_mm, _ = count_mm(lambda: sim._em_step(c, 0.05, dw))
        assert n_mm == 1

    @pytest.mark.parametrize("adaptive", [False, True])
    @pytest.mark.parametrize("reflection", ["predictor_corrector", "euler_clamp"])
    def test_full_flag_matrix_runs(self, adaptive, reflection):
        """All four corners of the ablation must produce a usable estimate."""
        sim = make_sim(n_nodes=5, seed=77, adaptive_stepping=adaptive,
                       adaptive_rtol=1e-5, reflection=reflection)
        yf, yc = sim.run_level(level=1, n_samples=128, T=1.0, base_dt=0.1)

        assert yf.shape == (128,) and yc.shape == (128,)
        assert np.all(np.isfinite(yf)) and np.all(yf >= 0.0)
        assert sim.adaptive_stepping is adaptive
        assert sim.reflection == reflection
        assert bool(sim.adaptive_bucket_history) is adaptive

    def test_error_estimator_is_switchable(self):
        """The manuscript's full-vs-two-half indicator stays available."""
        embedded = make_sim(n_nodes=5, seed=88, adaptive_stepping=True,
                            adaptive_rtol=1e-5)
        half_step = make_sim(n_nodes=5, seed=88, adaptive_stepping=True,
                             adaptive_rtol=1e-5,
                             adaptive_error_estimator="half_step")
        c = torch.rand(5, 128)
        dw = torch.randn(5, 128) * 0.1

        _, mm_embedded, _ = count_mm(
            lambda: embedded._em_step_adaptive(c, 0.05, dw, role="fine"))
        _, mm_half, _ = count_mm(
            lambda: half_step._em_step_adaptive(c, 0.05, dw, role="fine"))

        assert mm_embedded == 2, "embedded indicator costs no extra matmul"
        assert mm_half == 6, (
            "the manuscript indicator must evaluate a full step and two half "
            "steps for every path"
        )

    @pytest.mark.parametrize("kwargs", [
        {"reflection": "midpoint"},
        {"adaptive_error_estimator": "richardson"},
    ])
    def test_unknown_flag_values_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            make_sim(**kwargs)

    def test_reset_clears_controller_state(self):
        """Step-size history must not leak between runs."""
        sim = make_sim(n_nodes=5, seed=99, adaptive_stepping=True,
                       adaptive_rtol=1e-5)
        sim.run_level(level=0, n_samples=32, T=1.0, base_dt=0.1)
        assert sim.adaptive_bucket_history

        sim.reset_adaptive_state()
        assert sim._adaptive_h_scale == {}
        assert sim.adaptive_bucket_history == []
        assert sim.adaptive_mm_calls == 0

    def test_repeated_runs_are_reproducible(self):
        """A fresh simulator with the same seed must repeat its run exactly."""
        first = make_sim(n_nodes=5, seed=123, adaptive_stepping=True,
                         adaptive_rtol=1e-5)._run_level_state_tensors(
            level=1, n_samples=64, T=1.0, base_dt=0.1)[0]
        second = make_sim(n_nodes=5, seed=123, adaptive_stepping=True,
                          adaptive_rtol=1e-5)._run_level_state_tensors(
            level=1, n_samples=64, T=1.0, base_dt=0.1)[0]

        assert torch.equal(first, second)

    def test_work_units_report_the_refinement_cost(self):
        """`adaptive_work_units` must charge the half-step bucket twice."""
        sim = make_sim(n_nodes=5, seed=131, adaptive_stepping=True,
                       adaptive_rtol=1e-9)
        n_paths, n_steps = 64, 10
        sim.run_level(level=0, n_samples=n_paths, T=1.0, base_dt=0.1)

        fixed_work = n_steps * n_paths
        work = sim.adaptive_work_units()
        assert fixed_work < work <= 2 * fixed_work
