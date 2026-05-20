"""
Uncertainty Quantification Module

Tools for quantifying and visualizing uncertainty in network simulations:
- Bootstrap confidence intervals
- Uncertainty bands for time series
- Variance reduction analysis (MC vs MLMC)
- Prediction intervals
- Statistical convergence diagnostics

Mathematical Background:
- Bootstrap: Resample with replacement to estimate sampling distribution
- Confidence Interval: [Q_α/2, Q_1-α/2] where Q is quantile function
- Variance Reduction Ratio: VRR = Var(MC) / Var(MLMC)
- Effective Sample Size: ESS = n / (1 + 2Σρ_k) for autocorrelated samples
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import logging
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyBand:
    """Container for uncertainty band data.

    Attributes:
        times: Time points
        mean: Mean trajectory
        lower: Lower confidence bound
        upper: Upper confidence bound
        confidence_level: Confidence level (e.g., 0.95)
        method: Method used (e.g., 'bootstrap', 'analytical')
    """
    times: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    confidence_level: float
    method: str

    def width(self) -> np.ndarray:
        """Compute width of uncertainty band at each time point."""
        return self.upper - self.lower

    def relative_width(self) -> np.ndarray:
        """Compute relative width (width / mean) at each time point."""
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_width = self.width() / np.abs(self.mean)
            rel_width = np.where(np.isfinite(rel_width), rel_width, np.nan)
        return rel_width


@dataclass
class VarianceReductionMetrics:
    """Container for variance reduction analysis results.

    Attributes:
        mc_variance: Standard MC variance
        mlmc_variance: MLMC variance
        reduction_factor: Variance reduction factor (MC/MLMC)
        reduction_percent: Percentage reduction
        mc_cost: Computational cost for MC
        mlmc_cost: Computational cost for MLMC
        efficiency_gain: Overall efficiency gain
    """
    mc_variance: float
    mlmc_variance: float
    reduction_factor: float
    reduction_percent: float
    mc_cost: float
    mlmc_cost: float
    efficiency_gain: float

    def summary(self) -> Dict:
        """Return summary dictionary."""
        return {
            'mc_variance': self.mc_variance,
            'mlmc_variance': self.mlmc_variance,
            'reduction_factor': self.reduction_factor,
            'reduction_percent': self.reduction_percent,
            'mc_cost': self.mc_cost,
            'mlmc_cost': self.mlmc_cost,
            'efficiency_gain': self.efficiency_gain
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"VarianceReductionMetrics(\n"
            f"  MC Variance: {self.mc_variance:.6e}\n"
            f"  MLMC Variance: {self.mlmc_variance:.6e}\n"
            f"  Reduction Factor: {self.reduction_factor:.2f}x\n"
            f"  Reduction: {self.reduction_percent:.1f}%\n"
            f"  Efficiency Gain: {self.efficiency_gain:.2f}x\n"
            f")"
        )


class UncertaintyQuantifier:
    """Quantify uncertainty in Monte Carlo simulations.

    Provides methods for computing confidence intervals, prediction intervals,
    and uncertainty bands using various techniques including bootstrap and
    analytical approaches.

    Args:
        confidence_level: Default confidence level (default: 0.95)
        n_bootstrap: Number of bootstrap samples (default: 1000)
        random_seed: Random seed for reproducibility
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        random_seed: Optional[int] = None
    ):
        """Initialize uncertainty quantifier.

        Args:
            confidence_level: Default confidence level (0 < α < 1)
            n_bootstrap: Number of bootstrap resamples
            random_seed: Random seed
        """
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(random_seed)

        logger.info(
            f"Initialized UncertaintyQuantifier with {n_bootstrap} bootstrap samples"
        )

    def bootstrap_confidence_interval(
        self,
        samples: np.ndarray,
        statistic: Callable = np.mean,
        confidence_level: Optional[float] = None,
        method: str = 'percentile'
    ) -> Tuple[float, float, np.ndarray]:
        """Compute bootstrap confidence interval.

        Args:
            samples: Original data samples (shape: [n_samples])
            statistic: Function to compute statistic (default: mean)
            confidence_level: Confidence level (uses default if None)
            method: Bootstrap method ('percentile', 'bca', 'basic')

        Returns:
            Tuple of (lower_bound, upper_bound, bootstrap_distribution)
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        n = len(samples)
        bootstrap_stats = np.zeros(self.n_bootstrap)

        # Generate bootstrap samples
        for i in range(self.n_bootstrap):
            bootstrap_sample = self.rng.choice(samples, size=n, replace=True)
            bootstrap_stats[i] = statistic(bootstrap_sample)

        # Compute confidence interval based on method
        if method == 'percentile':
            alpha = 1 - confidence_level
            lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        elif method == 'basic':
            # Basic bootstrap (reflection method)
            theta_hat = statistic(samples)
            alpha = 1 - confidence_level
            lower_percentile = np.percentile(bootstrap_stats, 100 * alpha / 2)
            upper_percentile = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            lower = 2 * theta_hat - upper_percentile
            upper = 2 * theta_hat - lower_percentile

        elif method == 'bca':
            # Bias-corrected and accelerated (BCa) bootstrap
            # Simplified version - full BCa requires jackknife
            logger.warning("BCa not fully implemented, using percentile method")
            alpha = 1 - confidence_level
            lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        else:
            raise ValueError(f"Unknown bootstrap method: {method}")

        return (lower, upper, bootstrap_stats)

    def compute_prediction_interval(
        self,
        samples: np.ndarray,
        confidence_level: Optional[float] = None,
        method: str = 'quantile'
    ) -> Tuple[float, float]:
        """Compute prediction interval for future observations.

        Prediction intervals are wider than confidence intervals as they
        account for both sampling uncertainty and inherent variability.

        Args:
            samples: Historical samples
            confidence_level: Confidence level
            method: 'quantile' or 'normal'

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        alpha = 1 - confidence_level

        if method == 'quantile':
            # Non-parametric quantile-based prediction interval
            lower = np.percentile(samples, 100 * alpha / 2)
            upper = np.percentile(samples, 100 * (1 - alpha / 2))

        elif method == 'normal':
            # Parametric prediction interval assuming normality
            mean = np.mean(samples)
            std = np.std(samples, ddof=1)
            n = len(samples)

            # Prediction interval accounts for both estimation and future variance
            # PI = mean ± t * std * sqrt(1 + 1/n)
            t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n - 1)
            margin = t_critical * std * np.sqrt(1 + 1/n)

            lower = mean - margin
            upper = mean + margin

        else:
            raise ValueError(f"Unknown method: {method}")

        return (lower, upper)

    def compute_uncertainty_band(
        self,
        time_series_samples: np.ndarray,
        confidence_level: Optional[float] = None,
        method: str = 'quantile'
    ) -> UncertaintyBand:
        """Compute uncertainty band for time series.

        Args:
            time_series_samples: Samples of trajectories
                Shape: [n_samples, n_timesteps]
            confidence_level: Confidence level
            method: 'quantile' or 'bootstrap'

        Returns:
            UncertaintyBand object
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        n_samples, n_timesteps = time_series_samples.shape
        alpha = 1 - confidence_level

        # Compute mean trajectory
        mean = np.mean(time_series_samples, axis=0)
        times = np.arange(n_timesteps)

        if method == 'quantile':
            # Point-wise quantiles
            lower = np.percentile(time_series_samples, 100 * alpha / 2, axis=0)
            upper = np.percentile(time_series_samples, 100 * (1 - alpha / 2), axis=0)

        elif method == 'bootstrap':
            # Bootstrap at each time point
            lower = np.zeros(n_timesteps)
            upper = np.zeros(n_timesteps)

            for t in range(n_timesteps):
                samples_t = time_series_samples[:, t]
                ci_lower, ci_upper, _ = self.bootstrap_confidence_interval(
                    samples_t, confidence_level=confidence_level
                )
                lower[t] = ci_lower
                upper[t] = ci_upper

        else:
            raise ValueError(f"Unknown method: {method}")

        return UncertaintyBand(
            times=times,
            mean=mean,
            lower=lower,
            upper=upper,
            confidence_level=confidence_level,
            method=method
        )

    def plot_uncertainty_bands(
        self,
        uncertainty_band: UncertaintyBand,
        xlabel: str = 'Time',
        ylabel: str = 'Value',
        title: str = 'Uncertainty Band',
        filepath: Optional[Union[str, Path]] = None,
        show_samples: Optional[np.ndarray] = None,
        max_samples_plot: int = 100
    ) -> plt.Figure:
        """Plot uncertainty band with shaded region.

        Args:
            uncertainty_band: UncertaintyBand object
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            filepath: Save path (optional)
            show_samples: Optional sample trajectories to overlay
            max_samples_plot: Maximum number of sample trajectories to plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        times = uncertainty_band.times
        mean = uncertainty_band.mean
        lower = uncertainty_band.lower
        upper = uncertainty_band.upper

        # Plot sample trajectories (if provided)
        if show_samples is not None:
            n_samples = min(show_samples.shape[0], max_samples_plot)
            indices = self.rng.choice(show_samples.shape[0], size=n_samples, replace=False)
            for idx in indices:
                ax.plot(times, show_samples[idx], alpha=0.1, color='gray', linewidth=0.5)

        # Plot uncertainty band
        ax.fill_between(
            times, lower, upper,
            alpha=0.3, color='blue',
            label=f'{uncertainty_band.confidence_level*100:.0f}% CI'
        )

        # Plot mean
        ax.plot(times, mean, color='blue', linewidth=2, label='Mean')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if filepath is not None:
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved uncertainty band plot to {filepath}")

        return fig

    def analyze_variance_reduction(
        self,
        mc_samples: np.ndarray,
        mlmc_samples: np.ndarray,
        mc_cost: float,
        mlmc_cost: float
    ) -> VarianceReductionMetrics:
        """Analyze variance reduction achieved by MLMC vs standard MC.

        Args:
            mc_samples: Standard MC samples
            mlmc_samples: MLMC samples
            mc_cost: Computational cost for MC (e.g., seconds, FLOPs)
            mlmc_cost: Computational cost for MLMC

        Returns:
            VarianceReductionMetrics object
        """
        # Compute variances
        mc_var = np.var(mc_samples, ddof=1)
        mlmc_var = np.var(mlmc_samples, ddof=1)

        # Variance reduction factor
        if mlmc_var > 0:
            reduction_factor = mc_var / mlmc_var
        else:
            reduction_factor = np.inf

        reduction_percent = 100 * (1 - mlmc_var / mc_var) if mc_var > 0 else 0.0

        # Efficiency gain: (Var_MC / Cost_MC) / (Var_MLMC / Cost_MLMC)
        # = (Var_MC / Var_MLMC) * (Cost_MLMC / Cost_MC)
        if mlmc_cost > 0 and mlmc_var > 0:
            efficiency_gain = reduction_factor * (mc_cost / mlmc_cost)
        else:
            efficiency_gain = np.inf

        return VarianceReductionMetrics(
            mc_variance=mc_var,
            mlmc_variance=mlmc_var,
            reduction_factor=reduction_factor,
            reduction_percent=reduction_percent,
            mc_cost=mc_cost,
            mlmc_cost=mlmc_cost,
            efficiency_gain=efficiency_gain
        )

    def compare_estimator_variance(
        self,
        estimators: Dict[str, np.ndarray],
        reference: Optional[str] = None
    ) -> Dict:
        """Compare variance across multiple estimators.

        Args:
            estimators: Dictionary mapping estimator names to sample arrays
            reference: Name of reference estimator for comparison (optional)

        Returns:
            Comparison dictionary
        """
        results = {}

        for name, samples in estimators.items():
            var = np.var(samples, ddof=1)
            std = np.std(samples, ddof=1)
            mean = np.mean(samples)
            ci_lower, ci_upper, _ = self.bootstrap_confidence_interval(samples)

            results[name] = {
                'mean': mean,
                'variance': var,
                'std': std,
                'ci': (ci_lower, ci_upper),
                'n_samples': len(samples)
            }

        # Compute relative metrics if reference provided
        if reference is not None and reference in results:
            ref_var = results[reference]['variance']

            for name in results:
                if name != reference:
                    var = results[name]['variance']
                    results[name]['variance_reduction'] = ref_var / var if var > 0 else np.inf
                    results[name]['variance_reduction_pct'] = (
                        100 * (1 - var / ref_var) if ref_var > 0 else 0.0
                    )

        return results

    def compute_effective_sample_size(
        self,
        samples: np.ndarray,
        max_lag: Optional[int] = None
    ) -> float:
        """Compute effective sample size accounting for autocorrelation.

        ESS = n / (1 + 2 * Σ_{k=1}^{max_lag} ρ_k)

        where ρ_k is the autocorrelation at lag k.

        Args:
            samples: Time series samples
            max_lag: Maximum lag to consider (default: n/10)

        Returns:
            Effective sample size
        """
        n = len(samples)

        if max_lag is None:
            max_lag = min(n // 10, 100)

        # Compute autocorrelation
        mean = np.mean(samples)
        var = np.var(samples, ddof=1)

        if var == 0:
            return n

        autocorr_sum = 0.0
        for k in range(1, max_lag + 1):
            if k >= n:
                break

            # Autocorrelation at lag k
            numerator = np.sum(
                (samples[:-k] - mean) * (samples[k:] - mean)
            )
            # Unbiased autocorrelation estimator: divide by (n-k) for the number of pairs
            autocorr_k = numerator / ((n - k) * var)

            # Stop if autocorrelation becomes negligible
            if abs(autocorr_k) < 0.05:
                break

            autocorr_sum += autocorr_k

        ess = n / (1 + 2 * autocorr_sum)
        return max(1.0, ess)  # ESS >= 1

    def diagnose_convergence(
        self,
        samples: np.ndarray,
        batch_size: Optional[int] = None
    ) -> Dict:
        """Diagnose convergence of Monte Carlo estimation.

        Uses batch means method to assess if estimation has converged.

        Args:
            samples: Monte Carlo samples
            batch_size: Batch size for batch means (default: sqrt(n))

        Returns:
            Dictionary with convergence diagnostics
        """
        n = len(samples)

        if batch_size is None:
            batch_size = int(np.sqrt(n))

        n_batches = n // batch_size

        if n_batches < 2:
            logger.warning("Too few samples for convergence diagnostics")
            return {
                'converged': False,
                'reason': 'insufficient_samples',
                'n_samples': n
            }

        # Compute batch means
        batch_means = np.zeros(n_batches)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batch_means[i] = np.mean(samples[start:end])

        # Variance of batch means
        batch_var = np.var(batch_means, ddof=1)

        # Overall variance
        total_var = np.var(samples, ddof=1)

        # Relative variance (should decrease as 1/n)
        relative_var = batch_var * batch_size / total_var if total_var > 0 else np.inf

        # Convergence test: relative variance should be close to 1
        converged = abs(relative_var - 1.0) < 0.2

        return {
            'converged': converged,
            'relative_variance': relative_var,
            'batch_variance': batch_var,
            'total_variance': total_var,
            'n_batches': n_batches,
            'batch_size': batch_size,
            'effective_sample_size': self.compute_effective_sample_size(samples)
        }

    def export_uncertainty_data(
        self,
        uncertainty_band: UncertaintyBand,
        filepath: Union[str, Path],
        metadata: Optional[Dict] = None
    ):
        """Export uncertainty band data to file.

        Args:
            uncertainty_band: UncertaintyBand object
            filepath: Output file path (.npz format)
            metadata: Optional metadata
        """
        filepath = Path(filepath)

        np.savez(
            filepath,
            times=uncertainty_band.times,
            mean=uncertainty_band.mean,
            lower=uncertainty_band.lower,
            upper=uncertainty_band.upper,
            confidence_level=uncertainty_band.confidence_level,
            method=uncertainty_band.method,
            metadata=metadata if metadata is not None else {}
        )

        logger.info(f"Exported uncertainty band to {filepath}")


def compute_coverage_probability(
    true_values: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray
) -> float:
    """Compute empirical coverage probability of confidence intervals.

    For well-calibrated intervals at confidence level α, coverage should be ≈ α.

    Args:
        true_values: Ground truth values
        ci_lower: Lower confidence bounds
        ci_upper: Upper confidence bounds

    Returns:
        Coverage probability (fraction of intervals containing true value)
    """
    coverage = np.mean((true_values >= ci_lower) & (true_values <= ci_upper))
    return coverage


def plot_variance_reduction_comparison(
    estimators: Dict[str, np.ndarray],
    labels: Optional[Dict[str, str]] = None,
    title: str = 'Estimator Variance Comparison',
    filepath: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Create box plot comparing variance across estimators.

    Args:
        estimators: Dictionary mapping names to sample arrays
        labels: Optional pretty labels for estimators
        title: Plot title
        filepath: Save path (optional)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(estimators.keys())
    data = [estimators[name] for name in names]

    if labels is not None:
        display_names = [labels.get(name, name) for name in names]
    else:
        display_names = names

    bp = ax.boxplot(data, labels=display_names, patch_artist=True)

    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if filepath is not None:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved variance comparison plot to {filepath}")

    return fig
