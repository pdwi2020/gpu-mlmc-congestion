"""
Dataset Download Script

Download all required datasets for the project:
- SNAP network graphs
- CAIDA AS topology
- MAWI traffic traces (optional)
- Generate synthetic benchmarks
"""

import sys
from pathlib import Path
import logging
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "datasets"))

from snap.loader import SNAPDatasetLoader
from caida.loader import CAIDATopologyLoader
from mawi.loader import MAWITraceProcessor
from synthetic.generator import SyntheticBenchmarkGenerator


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_snap_datasets(loader: SNAPDatasetLoader, datasets: list):
    """
    Download SNAP datasets.

    Args:
        loader: SNAPDatasetLoader instance
        datasets: List of dataset names to download
    """
    print("\n" + "=" * 80)
    print("DOWNLOADING SNAP DATASETS")
    print("=" * 80)

    for dataset_name in datasets:
        try:
            print(f"\n[{dataset_name}]")
            print("-" * 80)

            filepath = loader.download_dataset(dataset_name, force=False)
            print(f"✓ Downloaded: {filepath}")

        except Exception as e:
            print(f"✗ Failed to download {dataset_name}: {e}")
            logger.error(f"Error downloading {dataset_name}: {e}")


def download_caida_topology(loader: CAIDATopologyLoader, dates: list):
    """
    Download CAIDA AS topologies.

    Args:
        loader: CAIDATopologyLoader instance
        dates: List of dates to download (YYYYMMDD format)
    """
    print("\n" + "=" * 80)
    print("DOWNLOADING CAIDA AS TOPOLOGY")
    print("=" * 80)

    for date in dates:
        try:
            print(f"\n[{date}]")
            print("-" * 80)

            filepath = loader.download_topology(date, force=False)
            print(f"✓ Downloaded: {filepath}")

        except Exception as e:
            print(f"✗ Failed to download CAIDA {date}: {e}")
            logger.error(f"Error downloading CAIDA {date}: {e}")


def download_mawi_traces(processor: MAWITraceProcessor, traces: list):
    """
    Download MAWI traffic traces.

    Args:
        processor: MAWITraceProcessor instance
        traces: List of (date, time) tuples to download
    """
    print("\n" + "=" * 80)
    print("DOWNLOADING MAWI TRAFFIC TRACES (OPTIONAL)")
    print("=" * 80)
    print("\nNote: MAWI traces are large (100-500 MB each).")
    print("Skipping by default. Set --include-mawi to download.")
    print("=" * 80)

    # Skip by default
    return


def generate_synthetic_benchmarks(generator: SyntheticBenchmarkGenerator):
    """
    Generate synthetic benchmark scenarios.

    Args:
        generator: SyntheticBenchmarkGenerator instance
    """
    print("\n" + "=" * 80)
    print("GENERATING SYNTHETIC BENCHMARKS")
    print("=" * 80)

    try:
        # Generate convergence test suite
        print("\n[Convergence Test Suite]")
        print("-" * 80)

        suite = generator.generate_convergence_test_suite()
        print(f"✓ Generated {len(suite)} test scenarios:")
        for name in suite.keys():
            print(f"  - {name}")

        # Save scenarios
        for name, scenario in suite.items():
            filename = f"{name}.json"
            generator.save_scenario(scenario, filename)
            print(f"  ✓ Saved: {filename}")

        # Generate parameter sweeps
        print("\n[Parameter Sweeps]")
        print("-" * 80)

        utilization_sweep = generator.generate_parameter_sweep(
            'utilization',
            [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        )
        print(f"✓ Generated utilization sweep: {len(utilization_sweep)} scenarios")

        print("\n✓ Synthetic benchmark generation complete!")

    except Exception as e:
        print(f"✗ Failed to generate synthetic benchmarks: {e}")
        logger.error(f"Error generating benchmarks: {e}")


def main():
    """Main download script."""

    parser = argparse.ArgumentParser(description="Download datasets for MLMC network project")
    parser.add_argument('--snap', action='store_true', help="Download SNAP datasets")
    parser.add_argument('--caida', action='store_true', help="Download CAIDA topologies")
    parser.add_argument('--mawi', action='store_true', help="Download MAWI traces (large)")
    parser.add_argument('--synthetic', action='store_true', help="Generate synthetic benchmarks")
    parser.add_argument('--all', action='store_true', help="Download all (except MAWI)")
    parser.add_argument('--data-dir', type=str, default=None, help="Data directory path")

    args = parser.parse_args()

    # If no options, default to --all
    if not any([args.snap, args.caida, args.mawi, args.synthetic, args.all]):
        args.all = True

    print("=" * 80)
    print("GPU-ACCELERATED MLMC NETWORK SIMULATION - DATASET DOWNLOADER")
    print("=" * 80)

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent / "datasets"

    print(f"\nData directory: {data_dir}")
    print(f"Downloads will be saved to respective subdirectories.")

    # Initialize loaders
    snap_loader = SNAPDatasetLoader(data_dir=data_dir / "snap")
    caida_loader = CAIDATopologyLoader(data_dir=data_dir / "caida")
    mawi_processor = MAWITraceProcessor(data_dir=data_dir / "mawi")
    synthetic_gen = SyntheticBenchmarkGenerator(data_dir=data_dir / "synthetic", seed=42)

    # Download SNAP datasets
    if args.snap or args.all:
        snap_datasets = [
            'email-Eu-core',  # Primary benchmark (~1000 nodes)
            'CA-GrQc',        # Optional scalability test (~5000 nodes)
        ]
        download_snap_datasets(snap_loader, snap_datasets)

    # Download CAIDA topology
    if args.caida or args.all:
        caida_dates = [
            '20260101',  # January 2026 (latest)
        ]
        download_caida_topology(caida_loader, caida_dates)

    # Download MAWI traces (optional, large)
    if args.mawi:
        mawi_traces = [
            ('20240619', '1400'),  # June 2024 sample
        ]
        download_mawi_traces(mawi_processor, mawi_traces)

    # Generate synthetic benchmarks
    if args.synthetic or args.all:
        generate_synthetic_benchmarks(synthetic_gen)

    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print("\nDatasets are ready for use!")
    print("\nNext steps:")
    print("  1. Load networks:")
    print("     from snap.loader import SNAPDatasetLoader")
    print("     loader = SNAPDatasetLoader()")
    print("     network = loader.load_dataset('email-Eu-core')")
    print("\n  2. Run simulations:")
    print("     python examples/basic_simulation.py")
    print("\n  3. Run GPU benchmarks:")
    print("     python examples/gpu_benchmark.py")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
