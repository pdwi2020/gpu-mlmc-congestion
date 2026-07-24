"""
Validation tests for `src/utils/stats.py`, the statistical-significance
primitives behind `scripts/compute_significance_tests.py`.

These are the tests that let anyone trust the paper's confidence intervals,
so each one is checked against a known answer rather than merely "does not
crash":

  * BCa bootstrap CI empirical coverage near 95% on synthetic data with a
    known population mean (both a symmetric and a right-skewed generator).
  * Paired t-test numerically agrees with `scipy.stats.ttest_rel` called
    directly, on a fixed, hand-inspectable dataset.
  * Wilcoxon signed-rank agrees with `scipy.stats.wilcoxon` called directly.
  * Cohen's d against a hand-computed value (arithmetic shown in a comment).
  * Ratio-of-means CI correctness on synthetic paired data with a known true
    ratio: (a) coverage of the true ratio near 95%, and (b) a direct
    demonstration that the paired bootstrap is NOT equivalent to bootstrapping
    numerator and denominator separately and dividing the CI endpoints -- the
    exact shortcut the harness is required to avoid.
"""

import os
import sys

import numpy as np
import pytest
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.stats import (  # noqa: E402
    MIN_N_HARD,
    MIN_N_RECOMMENDED,
    bca_bootstrap_ci,
    cohens_d_independent,
    cohens_d_paired,
    flag_insufficient_n,
    matched_pairs_rank_biserial,
    mean_sd_n,
    paired_ttest,
    ratio_ci_bootstrap,
    wilcoxon_signed_rank,
)


# ===========================================================================
# mean_sd_n / flag_insufficient_n -- basic sanity
# ===========================================================================
class TestDescriptive:
    def test_mean_sd_n_basic(self):
        out = mean_sd_n([1.0, 2.0, 3.0, 4.0, 5.0])
        assert out["n"] == 5
        assert out["mean"] == pytest.approx(3.0)
        # sample sd (ddof=1) of 1..5 is sqrt(2.5) = 1.5811388300841898
        assert out["sd"] == pytest.approx(1.5811388300841898, rel=1e-12)
        assert out["min"] == 1.0 and out["max"] == 5.0

    def test_mean_sd_n_single_value_sd_is_zero(self):
        out = mean_sd_n([7.0])
        assert out["n"] == 1
        assert out["sd"] == 0.0

    def test_mean_sd_n_drops_non_finite(self):
        out = mean_sd_n([1.0, float("nan"), 3.0, float("inf")])
        assert out["n"] == 2
        assert out["mean"] == pytest.approx(2.0)

    def test_flag_insufficient_n(self):
        assert flag_insufficient_n(3) is True
        assert flag_insufficient_n(MIN_N_RECOMMENDED - 1) is True
        assert flag_insufficient_n(MIN_N_RECOMMENDED) is False
        assert flag_insufficient_n(20) is False


# ===========================================================================
# BCa bootstrap coverage -- the load-bearing test
# ===========================================================================
class TestBCaCoverage:
    """Empirical coverage: run many independent trials, each drawing a fresh
    sample from a distribution with a KNOWN population mean, build a 95% BCa
    CI, and check the true mean falls inside it close to 95% of the time.

    Nominal 95% coverage from ~500 Monte Carlo trials has a binomial standard
    error of sqrt(0.95*0.05/500) ~= 0.0097, so a 3-SE band is roughly
    [0.92, 0.98]; we use a slightly wider [0.90, 0.99] to keep the test from
    flaking while still catching a badly miscalibrated interval (e.g. one
    that were actually a 68% or 99.9% interval).
    """

    N_TRIALS = 500
    N_PER_TRIAL = 25
    N_RESAMPLES = 999

    def _coverage(self, sampler, true_mean, seed0=0):
        hits = 0
        for trial in range(self.N_TRIALS):
            rng = np.random.default_rng(seed0 * 1_000_003 + trial)
            sample = sampler(rng, self.N_PER_TRIAL)
            ci = bca_bootstrap_ci(sample, n_resamples=self.N_RESAMPLES, seed=trial)
            assert ci["ci_lower"] is not None, f"trial {trial}: CI failed to compute ({ci['note']})"
            if ci["ci_lower"] <= true_mean <= ci["ci_upper"]:
                hits += 1
        return hits / self.N_TRIALS

    def test_coverage_normal(self):
        mu, sigma = 5.0, 2.0
        coverage = self._coverage(
            lambda rng, n: rng.normal(mu, sigma, size=n), true_mean=mu, seed0=1)
        assert 0.90 <= coverage <= 0.99, f"BCa coverage {coverage:.3f} far from nominal 0.95"

    def test_coverage_skewed_exponential(self):
        # Exponential(scale=3) has mean 3; this is the case naive percentile
        # bootstrap notoriously under-covers on, which is why BCa exists.
        scale = 3.0
        coverage = self._coverage(
            lambda rng, n: rng.exponential(scale, size=n), true_mean=scale, seed0=2)
        assert 0.88 <= coverage <= 0.99, f"BCa coverage {coverage:.3f} far from nominal 0.95 (skewed case)"

    def test_point_estimate_matches_statistic(self):
        data = [1.0, 2.0, 3.0, 4.0, 10.0]
        ci = bca_bootstrap_ci(data, n_resamples=2000, seed=0)
        assert ci["point_estimate"] == pytest.approx(np.mean(data))
        assert ci["ci_lower"] < ci["point_estimate"] < ci["ci_upper"]
        assert ci["method"] == "BCa"
        assert ci["insufficient_n"] is True  # n=5 < MIN_N_RECOMMENDED=10

    def test_insufficient_n_below_hard_minimum(self):
        ci = bca_bootstrap_ci([1.0], n_resamples=100, seed=0)
        assert ci["ci_lower"] is None and ci["ci_upper"] is None
        assert ci["n"] == 1
        assert "cannot compute" in ci["note"]

    def test_degenerate_constant_data_reports_reason_not_crash(self):
        ci = bca_bootstrap_ci([5.0, 5.0, 5.0, 5.0, 5.0], n_resamples=500, seed=0)
        # constant data => zero variance => BCa (and percentile) degenerate;
        # must not raise, must explain itself.
        assert ci["n"] == 5
        assert ci["point_estimate"] == pytest.approx(5.0)
        if ci["ci_lower"] is not None:
            # percentile fallback on constant data collapses to a point
            assert ci["ci_lower"] == pytest.approx(5.0)
            assert ci["ci_upper"] == pytest.approx(5.0)
        else:
            assert ci["note"] is not None


# ===========================================================================
# Paired t-test agreement with scipy
# ===========================================================================
class TestPairedTTest:
    # Fixed, hand-inspectable dataset: "method" is faster than "baseline" by
    # a roughly constant margin plus noise, across 8 paired seeds.
    BASELINE = np.array([10.2, 9.8, 10.5, 10.1, 9.9, 10.3, 10.0, 10.4])
    METHOD = np.array([9.1, 8.9, 9.4, 9.0, 8.8, 9.3, 9.0, 9.2])

    def test_matches_scipy_ttest_rel_exactly(self):
        ours = paired_ttest(self.BASELINE, self.METHOD)
        ref = sp_stats.ttest_rel(self.BASELINE, self.METHOD)
        assert ours["statistic"] == pytest.approx(ref.statistic, rel=1e-12)
        assert ours["pvalue"] == pytest.approx(ref.pvalue, rel=1e-10)
        assert ours["n"] == 8
        assert ours["df"] == 7
        assert ours["insufficient_n"] is True  # n=8 < 10

    def test_mean_and_sd_diff_hand_checkable(self):
        ours = paired_ttest(self.BASELINE, self.METHOD)
        diff = self.BASELINE - self.METHOD
        assert ours["mean_diff"] == pytest.approx(float(np.mean(diff)), rel=1e-12)
        assert ours["sd_diff"] == pytest.approx(float(np.std(diff, ddof=1)), rel=1e-12)

    def test_identical_arrays_give_p_one(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        ours = paired_ttest(a, a.copy())
        assert ours["statistic"] == 0.0 or np.isnan(ours["statistic"])
        # scipy returns nan statistic/pvalue for zero-variance differences;
        # confirm we pass that through rather than fabricating a value.
        ref = sp_stats.ttest_rel(a, a.copy())
        if np.isnan(ref.pvalue):
            assert np.isnan(ours["pvalue"])
        else:
            assert ours["pvalue"] == pytest.approx(ref.pvalue)

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            paired_ttest([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_below_hard_minimum(self):
        ours = paired_ttest([1.0], [2.0])
        assert np.isnan(ours["statistic"])
        assert "requires at least 2 pairs" in ours["note"]


# ===========================================================================
# Wilcoxon signed-rank agreement with scipy
# ===========================================================================
class TestWilcoxon:
    A = np.array([12.0, 15.0, 9.0, 14.0, 11.0, 13.0, 10.0, 16.0, 8.0, 12.5])
    B = np.array([10.0, 13.0, 9.5, 12.0, 10.5, 11.0, 9.0, 14.0, 8.5, 11.5])

    def test_matches_scipy_wilcoxon_exactly(self):
        ours = wilcoxon_signed_rank(self.A, self.B)
        ref = sp_stats.wilcoxon(self.A, self.B, zero_method="wilcox", mode="auto")
        assert ours["statistic"] == pytest.approx(ref.statistic, rel=1e-12)
        assert ours["pvalue"] == pytest.approx(ref.pvalue, rel=1e-10)
        assert ours["n"] == 10
        assert ours["insufficient_n"] is False  # n=10 == MIN_N_RECOMMENDED

    def test_all_zero_differences_handled(self):
        a = np.array([1.0, 2.0, 3.0])
        ours = wilcoxon_signed_rank(a, a.copy())
        # scipy 1.16's default zero_method handles the fully-degenerate case
        # without raising; if a future scipy raises instead, we must catch it
        # and explain rather than propagate.
        assert not np.isnan(ours["statistic"]) or ours["note"] is not None

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            wilcoxon_signed_rank([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_below_hard_minimum(self):
        ours = wilcoxon_signed_rank([1.0], [2.0])
        assert np.isnan(ours["statistic"])
        assert "requires at least 2 pairs" in ours["note"]


# ===========================================================================
# Cohen's d -- hand-computed examples
# ===========================================================================
class TestCohensD:
    def test_paired_hand_computed(self):
        # a - b = [2, 4, 6, 8] for every i (a = [12,14,16,18], b = [10]*4)
        # mean(diff) = 5
        # sample variance (ddof=1) = ((2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2)/3
        #                          = (9+1+1+9)/3 = 20/3
        # sd(diff) = sqrt(20/3) = 2.581988897471611
        # d_z = mean/sd = 5 / 2.581988897471611 = 1.9364916731037085
        a = np.array([12.0, 14.0, 16.0, 18.0])
        b = np.array([10.0, 10.0, 10.0, 10.0])
        d = cohens_d_paired(a, b)
        assert d == pytest.approx(1.9364916731037085, rel=1e-12)

    def test_paired_zero_effect(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert cohens_d_paired(a, a.copy()) == 0.0

    def test_paired_zero_variance_nonzero_mean_is_signed_inf(self):
        # every pair has EXACTLY the same difference -> sd(diff) = 0, a
        # perfectly consistent effect, which is the correct d -> +/-inf edge
        # case rather than a divide-by-zero crash.
        a = np.array([3.0, 3.0, 3.0])
        b = np.array([1.0, 1.0, 1.0])
        assert cohens_d_paired(a, b) == float("inf")
        assert cohens_d_paired(b, a) == float("-inf")

    def test_independent_hand_computed(self):
        # Two groups, textbook pooled-SD example.
        # a: mean=10, var(ddof=1) computed below; b: mean=8
        a = np.array([8.0, 10.0, 12.0])   # mean=10, var=(4+0+4)/2=4
        b = np.array([6.0, 8.0, 10.0])    # mean=8,  var=(4+0+4)/2=4
        # pooled_sd = sqrt(((2*4)+(2*4))/(3+3-2)) = sqrt(16/4) = 2
        # d = (10-8)/2 = 1.0
        d = cohens_d_independent(a, b)
        assert d == pytest.approx(1.0, rel=1e-12)

    def test_independent_too_few_samples_is_nan(self):
        assert np.isnan(cohens_d_independent([1.0], [1.0, 2.0, 3.0]))


class TestRankBiserial:
    def test_all_positive_differences_gives_plus_one(self):
        a = np.array([5.0, 6.0, 7.0, 8.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        assert matched_pairs_rank_biserial(a, b) == pytest.approx(1.0)

    def test_all_negative_differences_gives_minus_one(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([5.0, 6.0, 7.0, 8.0])
        assert matched_pairs_rank_biserial(a, b) == pytest.approx(-1.0)

    def test_hand_computed_mixed_signs(self):
        # diffs = a - b = [3, -1, 2, -4] -> |diffs| = [3, 1, 2, 4]
        # ranks of |diffs| = [3, 1, 2, 4] (ties absent)
        # positive diffs at rank 3 (value 3) and rank 2 (value 2) -> sum = 5
        # negative diffs at rank 1 (value 1) and rank 4 (value 4) -> sum = 5
        # r = (5 - 5) / 10 = 0.0
        a = np.array([10.0, 5.0, 10.0, 5.0])
        b = np.array([7.0, 6.0, 8.0, 9.0])
        assert matched_pairs_rank_biserial(a, b) == pytest.approx(0.0)

    def test_all_ties_is_nan(self):
        a = np.array([1.0, 2.0, 3.0])
        assert np.isnan(matched_pairs_rank_biserial(a, a.copy()))


# ===========================================================================
# Ratio-of-means bootstrap CI -- correctness on a KNOWN true ratio
# ===========================================================================
class TestRatioCI:
    def test_point_estimate_noiseless_ratio(self):
        denom = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        numer = 3.5 * denom  # exact ratio, every seed
        out = ratio_ci_bootstrap(numer, denom, n_resamples=2000, seed=0)
        assert out["point_estimate"] == pytest.approx(3.5, rel=1e-12)
        # every bootstrap resample reproduces the same exact ratio -> CI collapses
        assert out["ci_upper"] - out["ci_lower"] < 1e-9

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            ratio_ci_bootstrap([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_zero_denominator_reports_reason_not_crash(self):
        out = ratio_ci_bootstrap([1.0, 2.0, 3.0], [1.0, 0.0, 3.0])
        assert out["ci_lower"] is None
        assert "denominator" in out["note"]

    def test_insufficient_n_flagged(self):
        out = ratio_ci_bootstrap([10.0, 11.0, 9.0], [5.0, 5.5, 4.5], n_resamples=500, seed=0)
        assert out["n"] == 3
        assert out["insufficient_n"] is True

    def test_coverage_of_known_true_ratio(self):
        """Coverage simulation: numerator = true_ratio * denominator + noise,
        where denominator itself varies across seeds (simulating heterogeneous
        per-seed problem difficulty, as in this codebase's runtime/cost data).
        Checks that the seed-paired bootstrap CI on the ratio covers the known
        true ratio close to 95% of the time.
        """
        true_ratio = 2.4
        n_per_trial = 15
        n_trials = 300
        hits = 0
        for trial in range(n_trials):
            rng = np.random.default_rng(9_000_000 + trial)
            denom = rng.uniform(4.0, 6.0, size=n_per_trial)
            noise = rng.normal(0.0, 0.05, size=n_per_trial)  # small, zero-mean
            numer = true_ratio * denom + noise
            ci = ratio_ci_bootstrap(numer, denom, n_resamples=999, seed=trial)
            assert ci["ci_lower"] is not None, f"trial {trial}: {ci['note']}"
            if ci["ci_lower"] <= true_ratio <= ci["ci_upper"]:
                hits += 1
        coverage = hits / n_trials
        assert 0.88 <= coverage <= 0.99, f"ratio CI coverage {coverage:.3f} far from nominal 0.95"

    def test_paired_ratio_ci_is_not_equivalent_to_dividing_separate_cis(self):
        """The decisive correctness check: when numerator and denominator move
        together across seeds (shared per-seed scale factor, e.g. a network
        that happens to be busy that run inflates both the baseline and the
        method's cost equally), the TRUE ratio is constant across every seed
        even though each quantity individually has huge seed-to-seed spread.
        The paired bootstrap must detect that the ratio is essentially
        noise-free; naively bootstrapping each mean separately and dividing
        the CI endpoints ignores the shared scale factor and produces a far
        (and wrongly) wider interval.
        """
        rng = np.random.default_rng(123)
        n = 30
        shared_scale = rng.uniform(0.5, 5.0, size=n)  # large, shared seed-to-seed variation
        k_true = 1.8
        denom = 10.0 * shared_scale
        numer = k_true * denom  # ratio is EXACTLY k_true for every seed

        ratio_out = ratio_ci_bootstrap(numer, denom, seed=0, n_resamples=2000)
        correct_width = ratio_out["ci_upper"] - ratio_out["ci_lower"]
        assert ratio_out["point_estimate"] == pytest.approx(k_true, rel=1e-9)
        assert correct_width < 1e-6, "paired ratio CI should collapse when the ratio is seed-invariant"

        # The (incorrect) shortcut a harness must NOT take: bootstrap the
        # numerator and denominator means independently and divide the CI
        # endpoints. This ignores the pairing and inflates the interval.
        num_ci = bca_bootstrap_ci(numer, seed=1, n_resamples=2000)
        den_ci = bca_bootstrap_ci(denom, seed=2, n_resamples=2000)
        naive_lo = num_ci["ci_lower"] / den_ci["ci_upper"]
        naive_hi = num_ci["ci_upper"] / den_ci["ci_lower"]
        naive_width = naive_hi - naive_lo

        assert naive_width > 1000 * (correct_width + 1e-12), (
            "naive division-of-separate-CIs should be dramatically wider than "
            "the correct paired-ratio CI on perfectly-correlated data")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
