"""
Performance Metrics Package

This package provides comprehensive tools for analyzing network simulation results:

- delay: End-to-end delay computation and distribution analysis
- congestion: Queue length, utilization, and congestion event detection
- uncertainty: Uncertainty quantification, confidence intervals, variance reduction

Example usage:
    from metrics.delay import DelayCalculator
    from metrics.congestion import CongestionAnalyzer
    from metrics.uncertainty import UncertaintyQuantifier

    delay_calc = DelayCalculator(network)
    delay_metrics = delay_calc.estimate_delay_distribution(delay_samples)

    congestion_analyzer = CongestionAnalyzer(network, threshold=0.8)
    congestion_metrics = congestion_analyzer.analyze_simulation_congestion(queue_states)

    uq = UncertaintyQuantifier(confidence_level=0.95)
    uncertainty_band = uq.compute_uncertainty_band(time_series_samples)
"""

from .delay import (
    DelayCalculator,
    DelayMetrics,
    compute_delay_variance_reduction
)

from .congestion import (
    CongestionAnalyzer,
    CongestionMetrics,
    CongestionEvent,
    compute_congestion_probability,
    compare_congestion_scenarios
)

from .uncertainty import (
    UncertaintyQuantifier,
    UncertaintyBand,
    VarianceReductionMetrics,
    compute_coverage_probability,
    plot_variance_reduction_comparison
)

__all__ = [
    # Delay
    'DelayCalculator',
    'DelayMetrics',
    'compute_delay_variance_reduction',

    # Congestion
    'CongestionAnalyzer',
    'CongestionMetrics',
    'CongestionEvent',
    'compute_congestion_probability',
    'compare_congestion_scenarios',

    # Uncertainty
    'UncertaintyQuantifier',
    'UncertaintyBand',
    'VarianceReductionMetrics',
    'compute_coverage_probability',
    'plot_variance_reduction_comparison',
]
