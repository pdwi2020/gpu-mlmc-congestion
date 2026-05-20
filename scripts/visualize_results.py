"""
Visualization Script for Experiment Results

Generates publication-quality plots for all four experiments:
- Experiment 1: MLMC convergence plots
- Experiment 2: GPU speedup charts
- Experiment 3: Uncertainty bands and distributions
- Experiment 4: Real-world validation visualizations

Usage:
    python scripts/visualize_results.py --experiment all
    python scripts/visualize_results.py --experiment 1
    python scripts/visualize_results.py --experiment 2 3 4
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

# Directories
RESULTS_DIR = Path(__file__).parent.parent / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


def plot_exp1_convergence(results_file: Path):
    """Plot MLMC convergence analysis results.

    Generates:
    - Fig 1a: MSE vs Cost (log-log) for MC and MLMC
    - Fig 1b: Variance decay across levels
    - Fig 1c: Sample allocation per level
    """
    print("\n" + "=" * 80)
    print("Visualizing Experiment 1: MLMC Convergence")
    print("=" * 80)

    # Load results
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run: python experiments/exp1_mlmc_convergence.py")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    mc_results = data['mc_results']
    mlmc_results = data['mlmc_results']
    convergence = data['convergence_comparison']

    # Figure 1a: MSE vs Cost
    fig, ax = plt.subplots(figsize=(10, 7))

    # Extract data
    mc_costs = np.array(convergence['mc']['costs'])
    mc_mses = np.array(convergence['mc']['mses'])
    mlmc_costs = np.array(convergence['mlmc']['costs'])
    mlmc_mses = np.array(convergence['mlmc']['mses'])

    # Plot data points
    ax.loglog(mc_costs, mc_mses, 'o-', label='Standard MC', markersize=8, linewidth=2)
    ax.loglog(mlmc_costs, mlmc_mses, 's-', label='MLMC', markersize=8, linewidth=2)

    # Add theoretical reference lines
    ref_cost = np.logspace(np.log10(mc_costs.min()), np.log10(mc_costs.max()), 100)
    mc_ref = mc_mses[0] * (ref_cost / mc_costs[0])**(-0.5)  # O(Cost^-0.5)
    mlmc_ref = mlmc_mses[0] * (ref_cost / mlmc_costs[0])**(-1.0)  # O(Cost^-1)

    ax.loglog(ref_cost, mc_ref, '--', color='gray', alpha=0.5, label='O(Cost⁻⁰·⁵) reference')
    ax.loglog(ref_cost, mlmc_ref, '--', color='black', alpha=0.5, label='O(Cost⁻¹) reference')

    ax.set_xlabel('Computational Cost (timesteps)', fontsize=14)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=14)
    ax.set_title('MLMC Convergence: MSE vs Cost', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, which='both', alpha=0.3)

    # Add convergence rates as text
    mc_beta = convergence['mc']['beta']
    mlmc_beta = convergence['mlmc']['beta']
    textstr = f'MC: β = {mc_beta:.3f}\nMLMC: β = {mlmc_beta:.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = FIGURES_DIR / "exp1_convergence_rate.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)

    # Figure 1b: Variance Decay
    if 'variance_decay' in data:
        variance_decay = data['variance_decay']
        levels = np.array(variance_decay['levels'])
        variances = np.array(variance_decay['variances'])

        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter out zero/invalid variances
        valid = (variances > 0) & np.isfinite(variances)
        if np.sum(valid) > 0:
            ax.semilogy(levels[valid], variances[valid], 'o-', markersize=8, linewidth=2,
                       label='Empirical Variance')

            # Fitted line
            alpha = variance_decay.get('alpha', np.nan)
            C = variance_decay.get('C', np.nan)
            if not np.isnan(alpha) and not np.isnan(C):
                fit_variance = C * 2**(-alpha * levels)
                ax.semilogy(levels, fit_variance, '--', linewidth=2,
                           label=f'Fit: V_l ∝ 2^(-{alpha:.2f}·l)')

                # Add text
                textstr = f'α = {alpha:.2f}\n(theory: α ≈ 2)'
                ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                       fontsize=12, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

            ax.set_xlabel('Level l', fontsize=14)
            ax.set_ylabel('Variance V_l', fontsize=14)
            ax.set_title('MLMC Variance Decay Across Levels', fontsize=16, fontweight='bold')
            ax.legend(loc='best', fontsize=12)
            ax.grid(True, which='both', alpha=0.3)

            plt.tight_layout()
            output_path = FIGURES_DIR / "exp1_variance_decay.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close(fig)

    print(f"\nExperiment 1 visualizations complete: {FIGURES_DIR}")


def plot_exp2_speedup(results_file: Path):
    """Plot GPU speedup results.

    Generates:
    - Fig 2a: Speedup vs Sample Size
    - Fig 2b: Speedup vs Network Size
    - Fig 2c: Throughput comparison
    """
    print("\n" + "=" * 80)
    print("Visualizing Experiment 2: GPU Speedup")
    print("=" * 80)

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run: python experiments/exp2_gpu_speedup.py")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    sample_size_results = data['sample_size_scaling']
    network_size_results = data.get('network_size_scaling', {})

    # Figure 2a: Speedup vs Sample Size
    fig, ax = plt.subplots(figsize=(10, 6))

    sample_sizes = []
    speedups = []

    for n_samples_str, result in sample_size_results.items():
        n_samples = int(n_samples_str)
        speedup = result.get('speedup', np.nan)
        if not np.isnan(speedup):
            sample_sizes.append(n_samples)
            speedups.append(speedup)

    if sample_sizes:
        sample_sizes = np.array(sample_sizes)
        speedups = np.array(speedups)

        # Sort
        idx = np.argsort(sample_sizes)
        sample_sizes = sample_sizes[idx]
        speedups = speedups[idx]

        ax.semilogx(sample_sizes, speedups, 'o-', markersize=10, linewidth=2.5, color='#2E86AB')

        ax.set_xlabel('Number of Samples', fontsize=14)
        ax.set_ylabel('Speedup (CPU/GPU)', fontsize=14)
        ax.set_title('GPU Speedup vs Sample Size', fontsize=16, fontweight='bold')
        ax.grid(True, which='both', alpha=0.3)

        # Add max speedup annotation
        max_speedup = np.max(speedups)
        max_idx = np.argmax(speedups)
        ax.annotate(f'Max: {max_speedup:.1f}x',
                   xy=(sample_sizes[max_idx], speedups[max_idx]),
                   xytext=(20, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()
        output_path = FIGURES_DIR / "exp2_speedup_vs_samples.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)

    # Figure 2b: Speedup vs Network Size
    if network_size_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        network_sizes = []
        speedups_net = []

        for n_nodes_str, result in network_size_results.items():
            n_nodes = int(n_nodes_str)
            speedup = result.get('speedup', np.nan)
            if not np.isnan(speedup):
                network_sizes.append(n_nodes)
                speedups_net.append(speedup)

        if network_sizes:
            network_sizes = np.array(network_sizes)
            speedups_net = np.array(speedups_net)

            # Sort
            idx = np.argsort(network_sizes)
            network_sizes = network_sizes[idx]
            speedups_net = speedups_net[idx]

            ax.plot(network_sizes, speedups_net, 's-', markersize=10, linewidth=2.5, color='#A23B72')

            ax.set_xlabel('Network Size (nodes)', fontsize=14)
            ax.set_ylabel('Speedup (CPU/GPU)', fontsize=14)
            ax.set_title('GPU Speedup vs Network Size', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = FIGURES_DIR / "exp2_speedup_vs_network_size.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close(fig)

    print(f"\nExperiment 2 visualizations complete: {FIGURES_DIR}")


def plot_exp3_uncertainty(results_file: Path, band_file: Path):
    """Plot uncertainty quantification results.

    Generates:
    - Fig 3a: Delay distribution histogram with CI
    - Fig 3b: Uncertainty band for queue evolution
    - Fig 3c: Stochastic vs deterministic comparison
    """
    print("\n" + "=" * 80)
    print("Visualizing Experiment 3: Uncertainty Quantification")
    print("=" * 80)

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run: python experiments/exp3_uncertainty_quantification.py")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    # Figure 3a: Uncertainty Band
    if band_file.exists():
        band_data = np.load(band_file)
        times = band_data['times']
        mean = band_data['mean']
        lower = band_data['lower']
        upper = band_data['upper']

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot uncertainty band
        ax.fill_between(times, lower, upper, alpha=0.3, color='blue', label='95% CI')
        ax.plot(times, mean, 'b-', linewidth=2, label='Mean')

        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel('Queue Length (packets)', fontsize=14)
        ax.set_title('Uncertainty Band for Queue Evolution', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = FIGURES_DIR / "exp3_uncertainty_band.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)

    # Figure 3b: Stochastic vs Deterministic
    if 'comparison' in data:
        comparison = data['comparison']

        fig, ax = plt.subplots(figsize=(8, 6))

        stochastic_mean = comparison['stochastic_mean']
        deterministic_value = comparison['deterministic_value']

        categories = ['Stochastic\n(with uncertainty)', 'Deterministic\n(no uncertainty)']
        values = [stochastic_mean, deterministic_value]
        colors = ['#3498db', '#e74c3c']

        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.2f}',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')

        ax.set_ylabel('Mean Queue Length', fontsize=14)
        ax.set_title('Stochastic vs Deterministic Prediction', fontsize=16, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.2)

        # Add difference annotation
        diff_pct = comparison.get('relative_difference_percent', 0)
        ax.text(0.5, max(values) * 1.1, f'Difference: {diff_pct:.1f}%',
               ha='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        plt.tight_layout()
        output_path = FIGURES_DIR / "exp3_stochastic_vs_deterministic.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)

    print(f"\nExperiment 3 visualizations complete: {FIGURES_DIR}")


def plot_exp4_realworld(results_file: Path):
    """Plot real-world validation results.

    Generates:
    - Fig 4a: Delay distribution across AS paths
    - Fig 4b: Top bottleneck ASes
    - Fig 4c: Prediction uncertainty visualization
    """
    print("\n" + "=" * 80)
    print("Visualizing Experiment 4: Real-World Validation")
    print("=" * 80)

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run: python experiments/exp4_realworld_validation.py")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    # Figure 4a: Network Info
    network_info = data.get('network', {})
    mlmc_info = data.get('mlmc', {})
    uncertainty = data.get('uncertainty', {})

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create summary visualization
    metrics = ['Network\nSize', 'MLMC\nLevels', 'Total\nSamples', 'Runtime\n(sec)']
    values = [
        network_info.get('n_nodes', 0),
        mlmc_info.get('L', 0),
        sum(mlmc_info.get('N_samples', [])),
        mlmc_info.get('runtime', 0)
    ]

    colors = ['#1abc9c', '#3498db', '#9b59b6', '#e74c3c']
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if isinstance(value, float):
            label = f'{value:.1f}'
        else:
            label = f'{value:,}'

        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
               label, ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Value', fontsize=14)
    ax.set_title('Real-World Validation: Network Scale Summary', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()
    output_path = FIGURES_DIR / "exp4_network_scale.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)

    # Figure 4b: Prediction Uncertainty
    if uncertainty:
        fig, ax = plt.subplots(figsize=(8, 6))

        mean = uncertainty.get('mean', 0)
        ci_lower = uncertainty.get('ci_lower', 0)
        ci_upper = uncertainty.get('ci_upper', 0)

        ax.errorbar([1], [mean], yerr=[[mean - ci_lower], [ci_upper - mean]],
                   fmt='o', markersize=15, capsize=10, capthick=3,
                   color='#2980b9', ecolor='#34495e', linewidth=3)

        ax.set_xlim(0.5, 1.5)
        ax.set_ylim(0, ci_upper * 1.2)
        ax.set_xticks([1])
        ax.set_xticklabels(['Mean Queue Length'], fontsize=14)
        ax.set_ylabel('Queue Length (packets)', fontsize=14)
        ax.set_title('Prediction with 95% Confidence Interval', fontsize=16, fontweight='bold')

        # Add values as text
        ax.text(1.15, mean, f'Mean: {mean:.2f}', fontsize=12, va='center')
        ax.text(1.15, ci_upper, f'Upper: {ci_upper:.2f}', fontsize=12, va='center')
        ax.text(1.15, ci_lower, f'Lower: {ci_lower:.2f}', fontsize=12, va='center')

        # Add relative uncertainty
        rel_unc = uncertainty.get('relative_uncertainty_percent', 0)
        ax.text(0.5, ci_upper * 1.1, f'Relative Uncertainty: ±{rel_unc:.1f}%',
               fontsize=12, ha='left',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        plt.tight_layout()
        output_path = FIGURES_DIR / "exp4_prediction_uncertainty.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)

    print(f"\nExperiment 4 visualizations complete: {FIGURES_DIR}")


def main():
    """Main visualization runner."""
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument('--experiment', nargs='+', choices=['1', '2', '3', '4', 'all'],
                       default=['all'], help="Which experiments to visualize")

    args = parser.parse_args()

    experiments = args.experiment
    if 'all' in experiments:
        experiments = ['1', '2', '3', '4']

    print("=" * 80)
    print("EXPERIMENT RESULTS VISUALIZATION")
    print("=" * 80)
    print(f"Experiments to visualize: {', '.join(experiments)}")
    print(f"Output directory: {FIGURES_DIR}")

    # Experiment 1
    if '1' in experiments:
        results_file = TABLES_DIR / "exp1_mlmc_convergence_results.json"
        plot_exp1_convergence(results_file)

    # Experiment 2
    if '2' in experiments:
        results_file = TABLES_DIR / "exp2_gpu_speedup_results.json"
        plot_exp2_speedup(results_file)

    # Experiment 3
    if '3' in experiments:
        results_file = TABLES_DIR / "exp3_uncertainty_quantification_results.json"
        band_file = TABLES_DIR / "exp3_uncertainty_band.npz"
        plot_exp3_uncertainty(results_file, band_file)

    # Experiment 4
    if '4' in experiments:
        results_file = TABLES_DIR / "exp4_realworld_validation_results.json"
        plot_exp4_realworld(results_file)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("\nGenerated figures:")
    for fig_file in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  - {fig_file.name}")


if __name__ == "__main__":
    main()
