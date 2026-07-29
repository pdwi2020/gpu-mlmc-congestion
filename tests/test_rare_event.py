"""Tests for rare-event simulation estimators (src/simulation/rare_event.py).

Each estimator is validated against a closed-form problem where the answer is
known analytically:

* the exponential-twisting IS and CE method are checked on the terminal
  Gaussian tail P(Z >= gamma) = Phi_bar(gamma) and the CE-optimal tilt (the
  Mills ratio);
* AMS and fixed-level splitting are checked on the drifted-Brownian-motion
  running-maximum probability (reflection-principle closed form), compared to
  a long Monte Carlo on the *same* discrete chain to isolate estimator error
  from time-discretisation bias;
* the deterministic Chapman-Kolmogorov reference is checked for stability under
  grid refinement and against long Monte Carlo.

Structural invariants are also asserted: AMS stage survival fractions multiply
to the reported estimate, and the Girsanov weight is unbiased (E_tilt[w] = 1).
"""
import math
import os
import sys

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import simulation.rare_event as re  # noqa: E402
from simulation.rare_event import (  # noqa: E402
    QueueConfig,
    adaptive_multilevel_splitting,
    chapman_kolmogorov_reference,
    cross_entropy_is,
    drifted_bm_max_exceedance,
    fixed_level_splitting,
    freidlin_wentzell_logprob,
    fw_optimal_delta,
    gaussian_optimal_tilt,
    gpu_mlmc_is,
    make_result,
    plain_monte_carlo,
)


# ---------------------------------------------------------------------------
# Configuration and bookkeeping
# ---------------------------------------------------------------------------
def test_config_derived_quantities():
    cfg = QueueConfig(rho=0.97, mu=1.0, T=20.0, dt=0.05)
    assert cfg.lam == pytest.approx(0.97)
    assert cfg.sigma == pytest.approx(math.sqrt(0.97))
    assert cfg.d == pytest.approx(-0.03)
    assert cfg.n_steps == 400
    assert cfg.sigma_step == pytest.approx(math.sqrt(0.97) * math.sqrt(0.05))


def test_make_result_flags_degenerate():
    ok = make_result("m", 5.0, 1e-6, work=10, n_paths=10)
    assert not ok["degenerate"]
    assert ok["log_estimate"] == pytest.approx(math.log(1e-6))
    zero = make_result("m", 5.0, 0.0, work=10, n_paths=10)
    assert zero["degenerate"]
    assert zero["log_estimate"] == float("-inf")
    nan = make_result("m", 5.0, float("nan"), work=10, n_paths=10)
    assert nan["degenerate"]


# ---------------------------------------------------------------------------
# Girsanov weight is unbiased: E_tilt[w] = 1
# ---------------------------------------------------------------------------
def test_is_weight_unbiased():
    cfg = QueueConfig(lam_override=1.0, mu=1.0, T=1.0, dt=0.02, reflect=False)
    n = 200_000
    mu_shift = 0.8
    _, sum_z = re._simulate_tilted(cfg, n, mu_shift, np.random.default_rng(0))
    w = np.exp(re._log_weight_to_nominal(mu_shift, sum_z, cfg.n_steps))
    mean_w = float(w.mean())
    se = float(w.std(ddof=1) / math.sqrt(n))
    assert abs(mean_w - 1.0) < 5.0 * se + 0.01


# ---------------------------------------------------------------------------
# Exponential-twisting IS recovers the Gaussian tail
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gamma", [1.0, 1.5, 2.0])
def test_exptwist_is_recovers_gaussian_tail(gamma):
    # 1-step chain with d=0, sigma_step=1  ->  Q_1 = Z ~ N(0,1); event {Z>=gamma}.
    cfg = QueueConfig(lam_override=1.0, mu=1.0, T=1.0, dt=1.0, reflect=False)
    exact = float(norm.sf(gamma))
    res = gpu_mlmc_is(cfg, gamma, n_paths=200_000, delta="fw_optimal",
                      seed=1, self_normalised=False)
    assert not res["degenerate"]
    assert res["estimate"] == pytest.approx(exact, rel=0.06)


# ---------------------------------------------------------------------------
# Cross-entropy converges to the analytic optimal tilt (Mills ratio)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gamma", [1.0, 1.5])
def test_ce_converges_to_mills_ratio(gamma):
    cfg = QueueConfig(lam_override=1.0, mu=1.0, T=1.0, dt=1.0, reflect=False)
    optimal = gaussian_optimal_tilt(gamma)          # phi(g)/Phi_bar(g)
    res = cross_entropy_is(cfg, gamma, np.random.default_rng(7),
                           n_pilot=40_000, n_final=40_000, elite_frac=0.1,
                           max_iter=40)
    assert res["reached_B"]
    assert res["fitted_mu"] == pytest.approx(optimal, rel=0.05)
    assert res["estimate"] == pytest.approx(float(norm.sf(gamma)), rel=0.08)


# ---------------------------------------------------------------------------
# AMS on the drifted-BM running maximum (closed form)
# ---------------------------------------------------------------------------
def _drifted_bm_config():
    # lam=1, mu=1.5  ->  sigma=1, drift d=-0.5; T=5, dt=0.02 (250 steps).
    return QueueConfig(lam_override=1.0, mu=1.5, T=5.0, dt=0.02, reflect=False)


def _discrete_chain_truth(cfg, b, n_paths=200_000, seed=123):
    return plain_monte_carlo(cfg, b, np.random.default_rng(seed),
                             n_paths=n_paths)["estimate"]


def test_ams_recovers_drifted_bm_max():
    cfg = _drifted_bm_config()
    b = 2.0
    # Continuous closed form: sanity that the discrete chain approximates it.
    cont = drifted_bm_max_exceedance(cfg.d, cfg.sigma, cfg.T, b)
    truth = _discrete_chain_truth(cfg, b)
    assert truth == pytest.approx(cont, rel=0.15)   # discretisation gap
    res = adaptive_multilevel_splitting(cfg, b, np.random.default_rng(11),
                                        n_particles=5000, survival_frac=0.75)
    assert not res["degenerate"]
    assert res["estimate"] == pytest.approx(truth, rel=0.15)


def test_ams_stage_product_identity():
    cfg = _drifted_bm_config()
    res = adaptive_multilevel_splitting(cfg, 2.5, np.random.default_rng(12),
                                        n_particles=4000, survival_frac=0.75)
    prod = float(np.prod(res["stage_fractions"]))
    assert prod == pytest.approx(res["estimate"], abs=1e-12)


def test_ams_extinction_is_degenerate():
    # A target unreachable within the horizon must return a degenerate 0,
    # not a silently wrong positive number.
    cfg = QueueConfig(lam_override=1.0, mu=1.5, T=1.0, dt=0.05, reflect=False)
    res = adaptive_multilevel_splitting(cfg, 1e6, np.random.default_rng(13),
                                        n_particles=500, survival_frac=0.75,
                                        max_stages=30)
    assert res["degenerate"]
    assert res["extinct"]
    assert res["estimate"] == 0.0


# ---------------------------------------------------------------------------
# Fixed-level splitting on the same closed-form problem
# ---------------------------------------------------------------------------
def test_splitting_recovers_drifted_bm_max():
    cfg = _drifted_bm_config()
    b = 2.0
    truth = _discrete_chain_truth(cfg, b)
    res = fixed_level_splitting(cfg, b, np.random.default_rng(21),
                                n_particles=5000, n_levels=6)
    assert not res["degenerate"]
    assert res["estimate"] == pytest.approx(truth, rel=0.15)
    prod = float(np.prod(res["stage_fractions"]))
    assert prod == pytest.approx(res["estimate"], abs=1e-12)


# ---------------------------------------------------------------------------
# Chapman-Kolmogorov exact reference
# ---------------------------------------------------------------------------
def test_ck_stable_under_refinement():
    cfg = QueueConfig(rho=0.9, mu=1.0, T=10.0, dt=0.05)
    coarse = chapman_kolmogorov_reference(cfg, 4.0, bins_per_sigma=8)
    fine = chapman_kolmogorov_reference(cfg, 4.0, bins_per_sigma=24)
    assert coarse["probability"] == pytest.approx(fine["probability"], rel=0.01)


def test_ck_matches_long_mc_small_B():
    cfg = QueueConfig(rho=0.9, mu=1.0, T=10.0, dt=0.05)
    B = 4.0
    ck = chapman_kolmogorov_reference(cfg, B, bins_per_sigma=16)
    mc = plain_monte_carlo(cfg, B, np.random.default_rng(31), n_paths=400_000)
    se = mc["standard_error"]
    assert abs(ck["probability"] - mc["estimate"]) < 5.0 * se + 0.002


def test_ck_matches_ams_on_reflected_chain():
    # Cross-validate the CK reference on the reflected chain against AMS at a
    # moderately rare B where AMS is reliable.
    cfg = QueueConfig(rho=0.9, mu=1.0, T=10.0, dt=0.05)
    B = 6.0
    ck = chapman_kolmogorov_reference(cfg, B, bins_per_sigma=16)["probability"]
    ams = adaptive_multilevel_splitting(cfg, B, np.random.default_rng(32),
                                        n_particles=6000, survival_frac=0.8)["estimate"]
    assert ams == pytest.approx(ck, rel=0.20)


# ---------------------------------------------------------------------------
# All methods agree at a moderate B against the exact reference
# ---------------------------------------------------------------------------
def test_all_methods_agree_moderate_B():
    cfg = QueueConfig(rho=0.9, mu=1.0, T=10.0, dt=0.05)
    B = 5.0
    ref = chapman_kolmogorov_reference(cfg, B, bins_per_sigma=16)["probability"]
    ams = adaptive_multilevel_splitting(cfg, B, np.random.default_rng(41),
                                        n_particles=5000, survival_frac=0.8)
    ce = cross_entropy_is(cfg, B, np.random.default_rng(42),
                          n_pilot=4000, n_final=30_000)
    spl = fixed_level_splitting(cfg, B, np.random.default_rng(43),
                                n_particles=5000, n_levels=7)
    is_opt = gpu_mlmc_is(cfg, B, n_paths=40_000, delta="fw_optimal",
                         seed=44, self_normalised=False)
    for res in (ams, ce, spl, is_opt):
        assert not res["degenerate"], res["method"]
        assert res["estimate"] == pytest.approx(ref, rel=0.20), res["method"]


# ---------------------------------------------------------------------------
# Freidlin-Wentzell analytic reference
# ---------------------------------------------------------------------------
def test_fw_logprob_monotone_and_negative():
    cfg = QueueConfig(rho=0.97, mu=1.0, T=20.0, dt=0.05)
    lp = [freidlin_wentzell_logprob(cfg, B)["log_probability"]
          for B in (10.0, 15.0, 20.0, 25.0)]
    assert all(x < 0 for x in lp)
    assert lp == sorted(lp, reverse=True)           # decreasing in B


def test_fw_optimal_delta_grows_with_B():
    cfg = QueueConfig(rho=0.97, mu=1.0, T=20.0, dt=0.05)
    d10 = fw_optimal_delta(cfg, 10.0)
    d25 = fw_optimal_delta(cfg, 25.0)
    assert d25 > d10 > 0.03                          # published 0.03 is far too weak
    assert d25 == pytest.approx(25.0 / 20.0 + 0.03, rel=1e-6)


# ---------------------------------------------------------------------------
# The honest finding, encoded as a regression guard: the published constant
# tilt delta=0.03 degenerates (returns 0) at large B, exactly like plain MC.
# ---------------------------------------------------------------------------
def test_published_tilt_degenerates_at_large_B():
    cfg = QueueConfig(rho=0.97, mu=1.0, T=20.0, dt=0.05)
    weak = gpu_mlmc_is(cfg, 25.0, n_paths=8000, delta=0.03, seed=1)
    assert weak["degenerate"]                        # no tilted path reaches B=25
    # The same family at its Freidlin-Wentzell optimum does reach B.
    strong = gpu_mlmc_is(cfg, 25.0, n_paths=8000, delta="fw_optimal", seed=1)
    assert strong["estimate"] > 0.0


def test_plain_mc_degenerates_but_ams_does_not_at_large_B():
    cfg = QueueConfig(rho=0.97, mu=1.0, T=20.0, dt=0.05)
    mc = plain_monte_carlo(cfg, 25.0, np.random.default_rng(2), n_paths=20_000)
    assert mc["degenerate"]                          # zero events
    ams = adaptive_multilevel_splitting(cfg, 25.0, np.random.default_rng(3),
                                        n_particles=2000, survival_frac=0.75)
    ck = chapman_kolmogorov_reference(cfg, 25.0)["probability"]
    assert not ams["degenerate"]
    assert ams["estimate"] == pytest.approx(ck, rel=0.5)  # rare, single rep
