"""Tests for SDE calibration helpers and time-varying sigma model."""
from __future__ import annotations

import sys
import pathlib
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.run_real_trace_deep import (
    calibrate_params,
    sde_paths,
    reflected_sde_paths,
    jump_diffusion_sde_paths,
    des_poisson_paths,
    crn_coupled_paths,
    crn_nrmse,
)
from src.network.sde import QueueDynamicsSDE


@pytest.fixture
def unit_arrivals():
    rng = np.random.default_rng(0)
    arr = rng.exponential(1.0, (200, 3))
    arr /= arr.mean(axis=0)
    return arr


@pytest.fixture
def flat_arrivals():
    return np.ones((100, 2), dtype=float)


class TestCalibrateParams:
    def test_unit_mean_gives_sigma_one(self, flat_arrivals):
        cal = calibrate_params(flat_arrivals, dt=1.0)
        assert abs(cal["sigma_cal"] - 1.0) < 1e-9

    def test_lam_bar_correct(self, unit_arrivals):
        cal = calibrate_params(unit_arrivals, dt=1.0)
        assert abs(cal["lam_bar"] - float(np.mean(unit_arrivals))) < 1e-9

    def test_sigma_cal_is_sqrt_lam_bar(self, unit_arrivals):
        cal = calibrate_params(unit_arrivals, dt=1.0)
        assert abs(cal["sigma_cal"] - np.sqrt(cal["lam_bar"])) < 1e-9

    def test_jump_params(self, flat_arrivals):
        cal = calibrate_params(flat_arrivals, dt=1.0)
        assert abs(cal["jump_intensity_cal"] - 2.0) < 1e-9
        assert abs(cal["jump_size_mean_cal"] - 0.5) < 1e-9


class TestBackwardCompat:
    def test_reflected_alias_matches(self, unit_arrivals):
        a = sde_paths(unit_arrivals, mu=1.5, sigma=0.2, dt=1.0, n_paths=50, seed=7)
        b = reflected_sde_paths(unit_arrivals, mu=1.5, sigma=0.2, dt=1.0, n_paths=50, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_jump_alias_shape(self, unit_arrivals):
        b = jump_diffusion_sde_paths(unit_arrivals, mu=1.5, sigma=0.2,
                                     jump_intensity=0.5, jump_size_mean=0.3,
                                     dt=1.0, n_paths=50, seed=7)
        assert b.shape == (50, unit_arrivals.shape[1])


class TestNonNegativity:
    @pytest.mark.parametrize("tv", [False, True])
    def test_sde_non_negative(self, unit_arrivals, tv):
        peak = sde_paths(unit_arrivals, mu=1.5, sigma=0.2, dt=1.0, n_paths=100,
                         seed=42, time_varying_sigma=tv)
        assert np.all(peak >= 0.0)

    def test_jump_diffusion_non_negative(self, unit_arrivals):
        peak = sde_paths(unit_arrivals, mu=1.5, sigma=0.1, dt=1.0, n_paths=100,
                         seed=42, jump_intensity=1.0, jump_size_mean=0.5)
        assert np.all(peak >= 0.0)

    def test_des_non_negative(self, unit_arrivals):
        peak = des_poisson_paths(unit_arrivals, mu=1.5, dt=1.0, n_paths=100, seed=42)
        assert np.all(peak >= 0.0)


class TestSeedReproducibility:
    def test_same_seed(self, unit_arrivals):
        a = sde_paths(unit_arrivals, mu=1.5, sigma=1.0, dt=1.0, n_paths=50, seed=99)
        b = sde_paths(unit_arrivals, mu=1.5, sigma=1.0, dt=1.0, n_paths=50, seed=99)
        np.testing.assert_array_equal(a, b)

    def test_diff_seeds(self, unit_arrivals):
        a = sde_paths(unit_arrivals, mu=1.5, sigma=1.0, dt=1.0, n_paths=50, seed=1)
        b = sde_paths(unit_arrivals, mu=1.5, sigma=1.0, dt=1.0, n_paths=50, seed=2)
        assert not np.allclose(a, b)

    def test_tv_sigma_reproducible(self, unit_arrivals):
        a = sde_paths(unit_arrivals, mu=1.5, sigma=0.0, dt=1.0, n_paths=50,
                      seed=42, time_varying_sigma=True)
        b = sde_paths(unit_arrivals, mu=1.5, sigma=0.0, dt=1.0, n_paths=50,
                      seed=42, time_varying_sigma=True)
        np.testing.assert_array_equal(a, b)


class TestStatisticalImprovement:
    def _p95_err(self, sde_peak, des_peak):
        errs = []
        for k in range(sde_peak.shape[1]):
            s = np.percentile(sde_peak[:, k], 95)
            d = np.percentile(des_peak[:, k], 95)
            errs.append(abs(s - d) / max(d, 1e-9))
        return float(np.median(errs))

    def test_calibrated_better_than_plain(self, unit_arrivals):
        N, dt, mu = 200, 1.0, 1.5
        cal = calibrate_params(unit_arrivals, dt)
        des = des_poisson_paths(unit_arrivals, mu, dt, n_paths=N, seed=0)
        plain = sde_paths(unit_arrivals, mu, sigma=0.2, dt=dt, n_paths=N, seed=0)
        calib = sde_paths(unit_arrivals, mu, sigma=cal["sigma_cal"], dt=dt, n_paths=N, seed=0)
        assert self._p95_err(calib, des) < self._p95_err(plain, des)

    def test_tv_sigma_p95_p99_beats_plain_and_calibrated(self, unit_arrivals):
        """TV-sigma also improves the original P95/P99 relative-error metric."""
        N, dt, mu = 300, 1.0, 1.5
        cal = calibrate_params(unit_arrivals, dt)
        des = des_poisson_paths(unit_arrivals, mu, dt, n_paths=N, seed=0)
        plain = sde_paths(unit_arrivals, mu, sigma=0.2, dt=dt, n_paths=N, seed=0)
        calib = sde_paths(unit_arrivals, mu, sigma=cal["sigma_cal"], dt=dt, n_paths=N, seed=0)
        tv = sde_paths(unit_arrivals, mu, sigma=0.0, dt=dt, n_paths=N, seed=0,
                       time_varying_sigma=True)
        assert self._p95_err(tv, des) < self._p95_err(plain, des)
        assert self._p95_err(tv, des) < self._p95_err(calib, des)

    def test_tv_sigma_snrmse_beats_plain_and_calibrated(self, unit_arrivals):
        N, dt, mu = 300, 1.0, 1.5
        cal = calibrate_params(unit_arrivals, dt)
        des = des_poisson_paths(unit_arrivals, mu, dt, n_paths=N, seed=0)
        plain = sde_paths(unit_arrivals, mu, sigma=0.2, dt=dt, n_paths=N, seed=0)
        calib = sde_paths(unit_arrivals, mu, sigma=cal["sigma_cal"], dt=dt, n_paths=N, seed=0)
        tv = sde_paths(unit_arrivals, mu, sigma=0.0, dt=dt, n_paths=N, seed=0,
                       time_varying_sigma=True)
        def snrmse(a, b):
            return float(np.median([
                np.sqrt(np.mean((np.sort(a[:, k]) - np.sort(b[:, k]))**2))
                / max(b[:, k].mean(), 1e-9)
                for k in range(a.shape[1])
            ]))
        assert snrmse(tv, des) < snrmse(plain, des)
        assert snrmse(tv, des) < snrmse(calib, des)

    def test_snrmse_lower_than_raw_nrmse_for_same_dist(self, unit_arrivals):
        N, dt, mu = 300, 1.0, 1.5
        des1 = des_poisson_paths(unit_arrivals, mu, dt, n_paths=N, seed=1)
        des2 = des_poisson_paths(unit_arrivals, mu, dt, n_paths=N, seed=2)
        def raw_nrmse(a, b):
            return float(np.median([
                np.sqrt(np.mean((a[:, k] - b[:, k])**2)) / max(b[:, k].mean(), 1e-9)
                for k in range(a.shape[1])
            ]))
        def snrmse(a, b):
            return float(np.median([
                np.sqrt(np.mean((np.sort(a[:, k]) - np.sort(b[:, k]))**2))
                / max(b[:, k].mean(), 1e-9)
                for k in range(a.shape[1])
            ]))
        assert snrmse(des1, des2) < raw_nrmse(des1, des2)
        assert snrmse(des1, des2) < 0.20

    def test_moment_matched_params(self, flat_arrivals):
        sde = QueueDynamicsSDE.moment_matched(flat_arrivals, service_rate=2.0, dt=1.0)
        assert abs(sde.jump_intensity - 2.0) < 1e-9
        assert abs(sde.jump_size_mean - 0.5) < 1e-9
        assert sde.noise_intensity == 0.0

    def test_cir_calibrated_sets_flag(self, unit_arrivals):
        sde = QueueDynamicsSDE.cir_calibrated(unit_arrivals, service_rate=2.0)
        assert sde.time_varying_sigma is True
        assert sde.noise_intensity == 0.0

    def test_euler_maruyama_non_negative(self, unit_arrivals):
        sde = QueueDynamicsSDE(arrival_rate=1.0, service_rate=1.5, noise_intensity=0.5)
        q = 0.0
        for _ in range(500):
            q = sde.euler_maruyama_step(q, t=0.0, dt=0.1)
            assert q >= 0.0

    def test_euler_maruyama_tv_sigma(self, flat_arrivals):
        sde = QueueDynamicsSDE.cir_calibrated(flat_arrivals, service_rate=2.0)
        q = 0.0
        for _ in range(100):
            q = sde.euler_maruyama_step(q, t=0.0, dt=1.0, current_arrival_rate=1.0)
            assert q >= 0.0

    def test_default_is_non_tv(self):
        sde = QueueDynamicsSDE(arrival_rate=1.0, service_rate=1.5)
        assert sde.time_varying_sigma is False


class TestCRNNRMSE:
    """CRN coupling eliminates permutation noise from raw NRMSE."""

    def _uncoupled_des_floor(self, arrivals, mu=1.5, dt=1.0, n=300):
        """Raw NRMSE (%) between two independent DES runs — the permutation-noise floor."""
        des1 = des_poisson_paths(arrivals, mu, dt, n_paths=n, seed=1)
        des2 = des_poisson_paths(arrivals, mu, dt, n_paths=n, seed=2)
        vals = [
            np.sqrt(np.mean((des1[:, k] - des2[:, k]) ** 2))
            / max(des2[:, k].mean(), 1e-9) * 100
            for k in range(arrivals.shape[1])
        ]
        return float(np.median(vals))

    def test_crn_nrmse_below_uncoupled_floor(self, unit_arrivals):
        """CRN NRMSE must be lower than the DES-vs-DES uncoupled floor."""
        floor = self._uncoupled_des_floor(unit_arrivals)
        sde_pk, des_pk = crn_coupled_paths(unit_arrivals, mu=1.5, dt=1.0, n_paths=300, seed=42)
        coupled = crn_nrmse(sde_pk, des_pk)
        assert coupled < floor, (
            f"CRN NRMSE {coupled:.1f}% should beat uncoupled floor {floor:.1f}%"
        )

    def test_crn_nrmse_below_50pct(self, unit_arrivals):
        """TV-sigma CRN NRMSE should be well below 50% — it is ~14% on MAWI data."""
        sde_pk, des_pk = crn_coupled_paths(unit_arrivals, mu=1.5, dt=1.0, n_paths=300, seed=42)
        assert crn_nrmse(sde_pk, des_pk) < 50.0

    def test_crn_preserves_marginal_p95(self, unit_arrivals):
        """CRN coupling must not change the SDE marginal distribution (P95 unchanged)."""
        N, mu, dt = 400, 1.5, 1.0
        tv_peak = sde_paths(unit_arrivals, mu, sigma=0.0, dt=dt, n_paths=N,
                            seed=42, time_varying_sigma=True)
        crn_sde, _ = crn_coupled_paths(unit_arrivals, mu=mu, dt=dt, n_paths=N, seed=42)
        for k in range(unit_arrivals.shape[1]):
            p95_tv  = float(np.percentile(tv_peak[:, k], 95))
            p95_crn = float(np.percentile(crn_sde[:, k], 95))
            rel = abs(p95_tv - p95_crn) / max(p95_tv, 1e-9)
            assert rel < 0.15, (
                f"flow {k}: CRN P95={p95_crn:.3f} vs TV-sigma P95={p95_tv:.3f} "
                f"({rel:.1%} diff) — coupling should not shift marginals"
            )

    def test_crn_outputs_non_negative(self, unit_arrivals):
        sde_pk, des_pk = crn_coupled_paths(unit_arrivals, mu=1.5, dt=1.0, n_paths=100, seed=7)
        assert np.all(sde_pk >= 0.0)
        assert np.all(des_pk >= 0.0)

    def test_crn_reproducible(self, unit_arrivals):
        a_sde, a_des = crn_coupled_paths(unit_arrivals, mu=1.5, dt=1.0, n_paths=50, seed=5)
        b_sde, b_des = crn_coupled_paths(unit_arrivals, mu=1.5, dt=1.0, n_paths=50, seed=5)
        np.testing.assert_array_equal(a_sde, b_sde)
        np.testing.assert_array_equal(a_des, b_des)
