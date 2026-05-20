"""
SNAP Dataset Loader

Download and load network graphs from the Stanford Network Analysis Project (SNAP).

Supported datasets:
- Email-Eu-core: European research institution email network (~1,000 nodes)
- CA-GrQc: Collaboration network (~5,000 nodes)
- And more...

Dataset source: https://snap.stanford.edu/data/
"""
from __future__ import annotations

import urllib.request
import gzip
import shutil
from pathlib import Path
from typing import Optional, Dict, List
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from network.topology import NetworkGraph

logger = logging.getLogger(__name__)


# SNAP dataset URLs and metadata
SNAP_DATASETS = {
    'email-Eu-core': {
        'url': 'https://snap.stanford.edu/data/email-Eu-core.txt.gz',
        'filename': 'email-Eu-core.txt.gz',
        'description': 'Email network of a European research institution',
        'nodes': 1005,
        'edges': 25571,
        'directed': True,
        'type': 'communication'
    },
    'CA-GrQc': {
        'url': 'https://snap.stanford.edu/data/ca-GrQc.txt.gz',
        'filename': 'ca-GrQc.txt.gz',
        'description': 'General Relativity and Quantum Cosmology collaboration network',
        'nodes': 5242,
        'edges': 14496,
        'directed': False,
        'type': 'collaboration'
    },
    'CA-HepTh': {
        'url': 'https://snap.stanford.edu/data/ca-HepTh.txt.gz',
        'filename': 'ca-HepTh.txt.gz',
        'description': 'High Energy Physics Theory collaboration network',
        'nodes': 9877,
        'edges': 25998,
        'directed': False,
        'type': 'collaboration'
    },
    'p2p-Gnutella04': {
        'url': 'https://snap.stanford.edu/data/p2p-Gnutella04.txt.gz',
        'filename': 'p2p-Gnutella04.txt.gz',
        'description': 'Gnutella peer-to-peer network snapshot',
        'nodes': 10876,
        'edges': 39994,
        'directed': True,
        'type': 'p2p'
    }
}


class SNAPDatasetLoader:
    """
    Load and manage SNAP network datasets.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize SNAP dataset loader.

        Args:
            data_dir: Directory to store datasets (default: datasets/snap/)
        """
        if data_dir is None:
            data_dir = Path(__file__).parent

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SNAP dataset directory: {self.data_dir}")

    def download_dataset(self,
                        dataset_name: str,
                        force: bool = False) -> Path:
        """
        Download SNAP dataset.

        Args:
            dataset_name: Name of dataset (e.g., 'email-Eu-core')
            force: Force re-download even if file exists

        Returns:
            Path to downloaded file
        """
        if dataset_name not in SNAP_DATASETS:
            available = ', '.join(SNAP_DATASETS.keys())
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {available}"
            )

        dataset_info = SNAP_DATASETS[dataset_name]
        url = dataset_info['url']
        filename = dataset_info['filename']
        filepath = self.data_dir / filename

        # Check if already exists
        if filepath.exists() and not force:
            logger.info(f"Dataset already exists: {filepath}")
            return filepath

        # Download
        logger.info(f"Downloading {dataset_name} from {url}")
        logger.info(f"Description: {dataset_info['description']}")
        logger.info(f"Expected: {dataset_info['nodes']} nodes, {dataset_info['edges']} edges")

        try:
            with urllib.request.urlopen(url) as response:
                with open(filepath, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)

            logger.info(f"Downloaded to: {filepath}")

        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            raise

        return filepath

    def load_dataset(self,
                    dataset_name: str,
                    download_if_missing: bool = True,
                    directed: Optional[bool] = None,
                    largest_component: bool = True,
                    add_link_properties: bool = True,
                    seed: int = 42) -> NetworkGraph:
        """
        Load SNAP dataset as NetworkGraph.

        Args:
            dataset_name: Name of dataset
            download_if_missing: Download if not found locally
            directed: Override graph direction (None = use dataset default)
            largest_component: Extract largest connected component
            add_link_properties: Add synthetic link properties
            seed: Random seed for link properties

        Returns:
            NetworkGraph object
        """
        # Download if needed
        dataset_info = SNAP_DATASETS[dataset_name]
        filepath = self.data_dir / dataset_info['filename']

        if not filepath.exists():
            if download_if_missing:
                filepath = self.download_dataset(dataset_name)
            else:
                raise FileNotFoundError(
                    f"Dataset not found: {filepath}. "
                    f"Set download_if_missing=True to download."
                )

        # Determine if directed
        if directed is None:
            directed = dataset_info['directed']

        logger.info(f"Loading {dataset_name} from {filepath}")

        # Load graph
        network = self._load_edge_list(
            filepath=filepath,
            directed=directed,
            largest_component=largest_component
        )

        # Add link properties if requested
        if add_link_properties:
            network.set_link_properties(
                bandwidth_range=(100.0, 1000.0),
                delay_range=(1.0, 10.0),
                capacity_range=(100.0, 1000.0),
                seed=seed
            )
            logger.info("Added synthetic link properties")

        logger.info(f"Loaded {dataset_name}: {network}")

        return network

    def _load_edge_list(self,
                       filepath: Path,
                       directed: bool,
                       largest_component: bool) -> NetworkGraph:
        """
        Load edge list file.

        Args:
            filepath: Path to edge list file (.txt or .txt.gz)
            directed: Create directed graph
            largest_component: Extract largest component

        Returns:
            NetworkGraph
        """
        network = NetworkGraph(directed=directed)

        # Open file (handle gzip)
        if filepath.suffix == '.gz':
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'

        with open_func(filepath, mode) as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Parse edge
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        source = int(parts[0])
                        target = int(parts[1])
                        network.add_edge(source, target)
                    except ValueError:
                        continue  # Skip non-integer edges

        logger.info(f"Loaded graph: {network.n_nodes} nodes, {network.n_edges} edges")

        # Extract largest component
        if largest_component:
            network = network.get_largest_component()

        return network

    def list_available_datasets(self) -> List[Dict]:
        """
        List all available SNAP datasets.

        Returns:
            List of dataset info dictionaries
        """
        datasets = []
        for name, info in SNAP_DATASETS.items():
            datasets.append({
                'name': name,
                'description': info['description'],
                'nodes': info['nodes'],
                'edges': info['edges'],
                'directed': info['directed'],
                'type': info['type']
            })
        return datasets

    def print_dataset_info(self, dataset_name: str):
        """Print detailed information about a dataset."""
        if dataset_name not in SNAP_DATASETS:
            print(f"Unknown dataset: {dataset_name}")
            return

        info = SNAP_DATASETS[dataset_name]
        print("=" * 60)
        print(f"SNAP Dataset: {dataset_name}")
        print("=" * 60)
        print(f"Description: {info['description']}")
        print(f"Nodes: {info['nodes']}")
        print(f"Edges: {info['edges']}")
        print(f"Directed: {info['directed']}")
        print(f"Type: {info['type']}")
        print(f"URL: {info['url']}")
        print("=" * 60)


def download_all_datasets(data_dir: Optional[Path] = None, force: bool = False):
    """
    Download all SNAP datasets.

    Args:
        data_dir: Directory to store datasets
        force: Force re-download
    """
    loader = SNAPDatasetLoader(data_dir)

    for dataset_name in SNAP_DATASETS.keys():
        try:
            loader.download_dataset(dataset_name, force=force)
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("SNAP Dataset Loader - Example Usage")
    print("=" * 60)

    loader = SNAPDatasetLoader()

    # List available datasets
    print("\n1. Available SNAP Datasets")
    print("-" * 60)
    datasets = loader.list_available_datasets()
    for ds in datasets:
        print(f"{ds['name']:20s} - {ds['nodes']:5d} nodes, {ds['edges']:6d} edges ({ds['type']})")

    # Download and load Email-Eu-core
    print("\n2. Loading Email-Eu-core Dataset")
    print("-" * 60)

    network = loader.load_dataset(
        'email-Eu-core',
        download_if_missing=True,
        largest_component=True,
        add_link_properties=True
    )

    print(f"\nLoaded network: {network}")
    summary = network.summary()
    print(f"Network summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Print dataset info
    print("\n3. Dataset Information")
    print("-" * 60)
    loader.print_dataset_info('CA-GrQc')

    print("\n" + "=" * 60)
