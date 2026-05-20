"""
Experiments Package

Four comprehensive experiments demonstrating the GPU-accelerated MLMC framework:

1. exp1_mlmc_convergence.py:
   - Validates MLMC convergence rates (O(ε^-2) vs MC O(ε^-3))
   - Measures computational cost savings
   - Analyzes level-wise variance decay

2. exp2_gpu_speedup.py:
   - Benchmarks GPU acceleration vs CPU
   - Evaluates scaling with sample size and network size
   - Measures speedup efficiency (100x-500x expected)

3. exp3_uncertainty_quantification.py:
   - Demonstrates uncertainty-aware metrics
   - Generates confidence intervals and uncertainty bands
   - Compares stochastic vs deterministic predictions

4. exp4_realworld_validation.py:
   - Validates on CAIDA AS topology (~70K nodes)
   - Uses MAWI-based traffic models
   - Identifies congestion hotspots
   - Quantifies prediction uncertainty

Usage:
    python experiments/exp1_mlmc_convergence.py
    python experiments/exp2_gpu_speedup.py
    python experiments/exp3_uncertainty_quantification.py
    python experiments/exp4_realworld_validation.py

Results are saved to:
    results/tables/ - JSON and CSV files
    results/figures/ - Plots and visualizations
"""

__all__ = []
