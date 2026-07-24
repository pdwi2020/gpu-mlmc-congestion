"""
Statistical-significance primitives for the IEEE Access resubmission.

Reviewer 2 asked for confidence intervals on every reported improvement, and
several headline numbers in the manuscript were single-run.  This module is
the pure-computation layer: given raw per-seed arrays it computes descriptive
statistics, a bias-corrected-and-accelerated (BCa) bootstrap confidence
interval, paired significance tests (t-test and Wilcoxon signed-rank), and
effect sizes (Cohen's d and matched-pairs rank-biserial correlation).

Design choices, stated up front because they are exactly what a reviewer
would probe:

* BCa, not naive percentile.  The percentile bootstrap is only first-order
  accurate and can be badly biased for skewed statistics (runtime and cost
  distributions in this codebase are right-skewed).  BCa corrects for both
  bias and skewness-induced non-constant variance.  We lean on
  ``scipy.stats.bootstrap(..., method="BCa")`` rather than a hand-rolled
  implementation because it is the well-tested reference implementation
  (Efron & Tibshirani 1993, ch. 14); this module adds the fallback and
  reporting logic scipy does not provide out of the box.

* Ratio-of-means claims (speedup, work reduction) are bootstrapped on the
  RATIO directly, resampling seed-paired (numerator, denominator) pairs
  together (``paired=True``) so that within-seed correlation between the two
  quantities is preserved.  Dividing two independently-computed CIs on the
  numerator and denominator means would ignore that correlation and produce
  an interval that is systematically too wide (or, if the two quantities are
  negatively correlated across seeds, misleadingly narrow).

* Every function returns machine-readable dicts with an explicit
  ``insufficient_n`` flag rather than silently producing a number.  The
  threshold (``MIN_N_RECOMMENDED = 10``) is the bar this project's own task
  list sets for citable claims (RESUBMISSION_TASKS.md, T2.8: "n>=10 seeds
  everywhere"); ``MIN_N_HARD = 2`` is the point below which a dispersion
  estimate is not merely underpowered but mathematically undefined.

No plotting, no I/O, no file-format assumptions -- those live in the calling
scripts (``scripts/compute_significance_tests.py``).
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats as sp_stats

#: Below this many paired observations, a paired t-test / Wilcoxon / bootstrap
#: SE is not computable (variance needs >=2 points; a signed-rank test needs
#: >=1 non-zero difference and scipy's normal approximation needs more).
MIN_N_HARD = 2

#: Below this many seeds, treat a CI as present-but-not-citable.  This is the
#: bar the project's own revision plan sets ("n>=10 seeds everywhere" --
#: RESUBMISSION_TASKS.md T2.8), not a universal statistical constant: BCa's
#: asymptotic coverage guarantees only kick in with reasonably large n, and a
#: 3-seed CI on an expensive GPU sweep is exactly the kind of number a
#: reviewer flagged as "too thin to cite".
MIN_N_RECOMMENDED = 10

DEFAULT_CONFIDENCE = 0.95
DEFAULT_N_RESAMPLES = 10000


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------
def mean_sd_n(values: Sequence[float]) -> dict:
    """Mean, sample SD (ddof=1), n, min, max over the finite entries of `values`."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    return {
        "n": n,
        "mean": float(np.mean(arr)) if n else float("nan"),
        "sd": float(np.std(arr, ddof=1)) if n > 1 else (0.0 if n == 1 else float("nan")),
        "min": float(np.min(arr)) if n else float("nan"),
        "max": float(np.max(arr)) if n else float("nan"),
    }


def flag_insufficient_n(n: int, min_n: int = MIN_N_RECOMMENDED) -> bool:
    """True when `n` is below the project's citability bar."""
    return n < min_n


# --------------------------------------------------------------------------
# BCa bootstrap confidence intervals
# --------------------------------------------------------------------------
def _run_scipy_bootstrap(samples: tuple, statistic: Callable, *, paired: bool,
                          confidence: float, n_resamples: int, method: str,
                          rng: np.random.Generator):
    """One attempt at scipy.stats.bootstrap, warnings captured not raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        boot = sp_stats.bootstrap(
            samples, statistic, paired=paired, confidence_level=confidence,
            n_resamples=n_resamples, method=method, random_state=rng,
            vectorized=True,
        )
    lo, hi = float(boot.confidence_interval.low), float(boot.confidence_interval.high)
    se = float(boot.standard_error)
    warning_msgs = [str(w.message) for w in caught]
    return lo, hi, se, warning_msgs


def _bootstrap_with_fallback(samples: tuple, statistic: Callable, *, paired: bool,
                              point_estimate: float, n: int, confidence: float,
                              n_resamples: int, seed) -> dict:
    """BCa first; on failure or a degenerate (NaN) result, fall back to the
    percentile method; if that is also degenerate, report why rather than a
    number. This is the shared machinery behind `bca_bootstrap_ci` and
    `ratio_ci_bootstrap`."""
    result = {
        "point_estimate": float(point_estimate) if n else float("nan"),
        "ci_lower": None,
        "ci_upper": None,
        "standard_error": None,
        "confidence": confidence,
        "n": int(n),
        "n_resamples": int(n_resamples),
        "method": "BCa",
        "insufficient_n": flag_insufficient_n(n),
        "note": None,
    }
    if n < MIN_N_HARD:
        result["note"] = f"n={n} < {MIN_N_HARD}: cannot compute a bootstrap CI"
        return result

    rng = np.random.default_rng(seed)
    notes = []
    for method in ("BCa", "percentile"):
        try:
            lo, hi, se, warns = _run_scipy_bootstrap(
                samples, statistic, paired=paired, confidence=confidence,
                n_resamples=n_resamples, method=method, rng=rng)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the harness
            notes.append(f"{method} raised {type(exc).__name__}: {exc}")
            continue
        if warns:
            notes.append(f"{method} warnings: {'; '.join(warns)}")
        if np.isfinite(lo) and np.isfinite(hi):
            result["ci_lower"], result["ci_upper"] = lo, hi
            result["standard_error"] = se
            result["method"] = method
            if method != "BCa":
                notes.append("BCa was degenerate for this sample; used percentile bootstrap instead")
            result["note"] = "; ".join(notes) if notes else None
            return result
        notes.append(f"{method} produced a non-finite CI (degenerate statistic, e.g. zero variance)")

    result["note"] = "; ".join(notes) if notes else "bootstrap CI could not be computed"
    return result


def bca_bootstrap_ci(data: Sequence[float], statistic: Callable = np.mean,
                      confidence: float = DEFAULT_CONFIDENCE,
                      n_resamples: int = DEFAULT_N_RESAMPLES,
                      seed: Optional[int] = None) -> dict:
    """Bias-corrected-and-accelerated bootstrap CI for `statistic(data)`.

    `statistic` must accept an `axis` keyword (as `np.mean`/`np.median` do)
    since scipy's vectorized bootstrap calls it on a 2-D array of resamples.

    Returns a dict with point_estimate, ci_lower/ci_upper (None if the CI
    could not be computed), standard_error, n, method actually used ("BCa" or
    a documented fallback), and `insufficient_n` per MIN_N_RECOMMENDED.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    point = statistic(arr) if n else float("nan")
    return _bootstrap_with_fallback(
        (arr,), statistic, paired=False, point_estimate=point, n=n,
        confidence=confidence, n_resamples=n_resamples, seed=seed)


def ratio_ci_bootstrap(numerator: Sequence[float], denominator: Sequence[float],
                        confidence: float = DEFAULT_CONFIDENCE,
                        n_resamples: int = DEFAULT_N_RESAMPLES,
                        seed: Optional[int] = None) -> dict:
    """BCa bootstrap CI on the RATIO OF MEANS mean(numerator)/mean(denominator).

    `numerator` and `denominator` must be seed-paired (same length, index i in
    both arrays comes from the same seed/run).  The bootstrap resamples seed
    INDICES once per replicate and applies that resampling to both arrays
    (`paired=True` in scipy's bootstrap), so within-seed correlation between
    numerator and denominator is preserved.  This is deliberately NOT the
    same as bootstrapping the numerator and denominator means separately and
    dividing their CI endpoints -- that approach ignores the pairing and is
    what this function exists to avoid.

    Speedup and work-reduction claims are ratio-of-means claims: e.g.
    numerator = per-seed baseline runtime, denominator = per-seed method
    runtime, ratio = mean(baseline)/mean(method) = mean speedup.
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if num.shape != den.shape:
        raise ValueError(
            f"numerator and denominator must be paired (equal-length) arrays, "
            f"got shapes {num.shape} and {den.shape}")

    finite = np.isfinite(num) & np.isfinite(den)
    num, den = num[finite], den[finite]
    n = int(num.size)

    result_shell = {
        "point_estimate": None, "ci_lower": None, "ci_upper": None,
        "standard_error": None, "confidence": confidence, "n": n,
        "n_resamples": int(n_resamples), "method": "BCa (paired ratio-of-means)",
        "insufficient_n": flag_insufficient_n(n), "paired": True, "note": None,
    }
    if n == 0:
        result_shell["note"] = "no finite paired observations"
        return result_shell
    if np.any(den == 0):
        result_shell["note"] = (
            "denominator contains zero for at least one seed; ratio-of-means "
            "is undefined")
        result_shell["point_estimate"] = float("nan")
        return result_shell

    def ratio_stat(n_arr, d_arr, axis=-1):
        return np.mean(n_arr, axis=axis) / np.mean(d_arr, axis=axis)

    point = float(np.mean(num) / np.mean(den))
    out = _bootstrap_with_fallback(
        (num, den), ratio_stat, paired=True, point_estimate=point, n=n,
        confidence=confidence, n_resamples=n_resamples, seed=seed)
    out["method"] = f"{out['method']} (paired ratio-of-means)"
    out["paired"] = True
    return out


# --------------------------------------------------------------------------
# Paired significance tests
# --------------------------------------------------------------------------
def paired_ttest(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired (dependent-samples) t-test, thin wrapper around scipy.stats.ttest_rel."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must have equal length, got {a.shape} vs {b.shape}")
    n = int(a.size)
    diff = a - b
    out = {
        "n": n,
        "df": max(n - 1, 0),
        "mean_diff": float(np.mean(diff)) if n else float("nan"),
        "sd_diff": float(np.std(diff, ddof=1)) if n > 1 else float("nan"),
        "statistic": float("nan"),
        "pvalue": float("nan"),
        "insufficient_n": flag_insufficient_n(n),
        "note": None,
    }
    if n < MIN_N_HARD:
        out["note"] = f"n={n} < {MIN_N_HARD}: paired t-test requires at least 2 pairs"
        return out
    res = sp_stats.ttest_rel(a, b)
    out["statistic"] = float(res.statistic)
    out["pvalue"] = float(res.pvalue)
    return out


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> dict:
    """Wilcoxon signed-rank test on paired arrays, thin wrapper around scipy.stats.wilcoxon."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must have equal length, got {a.shape} vs {b.shape}")
    n = int(a.size)
    out = {
        "n": n, "statistic": float("nan"), "pvalue": float("nan"),
        "insufficient_n": flag_insufficient_n(n), "note": None,
    }
    if n < MIN_N_HARD:
        out["note"] = f"n={n} < {MIN_N_HARD}: Wilcoxon requires at least 2 pairs"
        return out
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = sp_stats.wilcoxon(a, b, zero_method="wilcox", mode="auto")
        out["statistic"] = float(res.statistic)
        out["pvalue"] = float(res.pvalue)
        if caught:
            out["note"] = "; ".join(str(w.message) for w in caught)
    except ValueError as exc:
        out["note"] = f"scipy.stats.wilcoxon failed: {exc}"
    return out


# --------------------------------------------------------------------------
# Effect sizes
# --------------------------------------------------------------------------
def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d_z for paired data: mean(a-b) / sd(a-b).

    This is the standard paired-samples effect size (Cohen 1988, eq. 2.3.7),
    distinct from the independent-samples d that uses a pooled SD.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    if diff.size < 2:
        return float("nan")
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        mean_diff = float(np.mean(diff))
        if mean_diff == 0.0:
            return 0.0
        return float("inf") if mean_diff > 0 else float("-inf")
    return float(np.mean(diff) / sd)


def cohens_d_independent(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d for independent (unpaired) samples, using the pooled SD."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = int(a.size), int(b.size)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = float(np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)))
    if pooled_sd == 0.0:
        mean_diff = float(np.mean(a) - np.mean(b))
        return 0.0 if mean_diff == 0.0 else (float("inf") if mean_diff > 0 else float("-inf"))
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def matched_pairs_rank_biserial(a: Sequence[float], b: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation, the standard effect size for
    the Wilcoxon signed-rank test: r = (sum of positive ranks - sum of
    negative ranks) / (sum of all ranks), computed over non-zero differences.

    r in [-1, 1]; r=1 means every non-tied pair favours `a`, r=-1 every pair
    favours `b`. Ties (a_i == b_i) are excluded, matching scipy's default
    'wilcox' zero-handling for the paired test above.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    nz = diff[diff != 0]
    if nz.size < 1:
        return float("nan")
    ranks = sp_stats.rankdata(np.abs(nz))
    pos = float(ranks[nz > 0].sum())
    neg = float(ranks[nz < 0].sum())
    total = pos + neg
    if total == 0:
        return float("nan")
    return (pos - neg) / total
