"""
CAIDA AS Relationships Dataset Loader

Download and load Internet AS-level topology from CAIDA.

Dataset: AS Relationships (serial-2)
Source: http://data.caida.org/datasets/as-relationships/serial-2/
Format: <AS1>|<AS2>|<relationship>
  -1 = provider-to-customer
   0 = peer-to-peer
"""

import urllib.request
import bz2
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from network.topology import NetworkGraph

logger = logging.getLogger(__name__)


class CAIDATopologyLoader:
    """
    Load CAIDA AS-level Internet topology datasets.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize CAIDA topology loader.

        Args:
            data_dir: Directory to store datasets (default: datasets/caida/)
        """
        if data_dir is None:
            data_dir = Path(__file__).parent

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.base_url = "http://data.caida.org/datasets/as-relationships/serial-2/"

        logger.info(f"CAIDA dataset directory: {self.data_dir}")

    def get_available_dates(self) -> list:
        """
        Get list of commonly available dates.

        Note: Full list requires parsing CAIDA website.
        Here we provide recent dates as of January 2026.

        Returns:
            List of date strings (YYYYMMDD format)
        """
        # Recent dates (as of Jan 2026)
        recent_dates = [
            '20260101',  # January 2026
            '20251201',  # December 2025
            '20251101',  # November 2025
            '20251001',  # October 2025
            '20240101',  # January 2024
        ]

        return recent_dates

    def download_topology(self,
                         date: str = '20260101',
                         force: bool = False) -> Path:
        """
        Download CAIDA AS relationships dataset.

        Args:
            date: Date in YYYYMMDD format
            force: Force re-download if file exists

        Returns:
            Path to downloaded file
        """
        filename = f"{date}.as-rel2.txt.bz2"
        filepath = self.data_dir / filename
        url = self.base_url + filename

        # Check if exists
        if filepath.exists() and not force:
            logger.info(f"Dataset already exists: {filepath}")
            return filepath

        # Download
        logger.info(f"Downloading CAIDA AS topology from {url}")
        logger.info(f"This may take a few minutes...")

        try:
            with urllib.request.urlopen(url) as response:
                with open(filepath, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)

            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"Downloaded to: {filepath} ({file_size_mb:.1f} MB)")

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.error(
                    f"Dataset not found for date {date}. "
                    f"Available dates: {self.get_available_dates()}"
                )
            raise

        except Exception as e:
            logger.error(f"Failed to download CAIDA topology: {e}")
            raise

        return filepath

    def load_topology(self,
                     date: str = '20260101',
                     download_if_missing: bool = True,
                     as_undirected: bool = True,
                     largest_component: bool = True,
                     add_link_properties: bool = False,
                     seed: int = 42) -> NetworkGraph:
        """
        Load CAIDA AS topology as NetworkGraph.

        Args:
            date: Date in YYYYMMDD format
            download_if_missing: Download if not found
            as_undirected: Convert to undirected graph
            largest_component: Extract largest component
            add_link_properties: Add synthetic link properties
            seed: Random seed for link properties

        Returns:
            NetworkGraph object
        """
        filename = f"{date}.as-rel2.txt.bz2"
        filepath = self.data_dir / filename

        # Download if needed
        if not filepath.exists():
            if download_if_missing:
                filepath = self.download_topology(date)
            else:
                raise FileNotFoundError(
                    f"Dataset not found: {filepath}. "
                    f"Set download_if_missing=True to download."
                )

        logger.info(f"Loading CAIDA AS topology from {filepath}")

        # Parse topology
        network = self._parse_as_relationships(
            filepath=filepath,
            as_undirected=as_undirected
        )

        # Extract largest component
        if largest_component:
            network = network.get_largest_component()

        # Add link properties if requested
        if add_link_properties:
            network.set_link_properties(
                bandwidth_range=(100.0, 10000.0),  # Higher for backbone
                delay_range=(1.0, 100.0),          # Longer for Internet
                capacity_range=(1000.0, 100000.0), # Larger buffers
                seed=seed
            )
            logger.info("Added synthetic link properties")

        logger.info(f"Loaded CAIDA topology: {network}")

        return network

    def _parse_as_relationships(self,
                               filepath: Path,
                               as_undirected: bool) -> NetworkGraph:
        """
        Parse CAIDA AS relationships file.

        Format:
            <provider-as>|<customer-as>|-1  (provider-to-customer)
            <peer-as>|<peer-as>|0           (peer-to-peer)

        Args:
            filepath: Path to .bz2 file
            as_undirected: Create undirected graph

        Returns:
            NetworkGraph
        """
        network = NetworkGraph(directed=not as_undirected)

        relationship_counts = {
            'provider_customer': 0,
            'peer_peer': 0,
            'unknown': 0
        }

        # Parse compressed file
        with bz2.open(filepath, 'rt') as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Parse: AS1|AS2|relationship
                parts = line.split('|')
                if len(parts) < 3:
                    continue

                try:
                    as1 = int(parts[0])
                    as2 = int(parts[1])
                    rel_type = int(parts[2])

                    # Add edge based on relationship type
                    if rel_type == -1:
                        # Provider-to-customer
                        relationship_counts['provider_customer'] += 1
                        if as_undirected:
                            network.add_edge(as1, as2, relationship='provider_customer')
                        else:
                            # Directed: provider → customer
                            network.add_edge(as1, as2, relationship='provider_customer')

                    elif rel_type == 0:
                        # Peer-to-peer (always undirected in practice)
                        relationship_counts['peer_peer'] += 1
                        network.add_edge(as1, as2, relationship='peer_peer')

                    else:
                        relationship_counts['unknown'] += 1

                except (ValueError, IndexError):
                    continue

        logger.info(f"Parsed AS relationships:")
        logger.info(f"  Provider-Customer: {relationship_counts['provider_customer']}")
        logger.info(f"  Peer-Peer: {relationship_counts['peer_peer']}")
        logger.info(f"  Total edges: {network.n_edges}")
        logger.info(f"  Total ASes: {network.n_nodes}")

        return network

    def get_topology_statistics(self, network: NetworkGraph) -> Dict:
        """
        Compute statistics for AS topology.

        Args:
            network: NetworkGraph of AS topology

        Returns:
            Dictionary with statistics
        """
        import numpy as np

        stats = network.summary()

        # Additional AS-specific stats
        degrees = [network.get_degree(node) for node in network.nodes]
        stats['degree_distribution'] = {
            'mean': np.mean(degrees),
            'median': np.median(degrees),
            'max': np.max(degrees),
            'min': np.min(degrees),
            'std': np.std(degrees)
        }

        # Count relationship types if available
        relationship_counts = {'provider_customer': 0, 'peer_peer': 0, 'unknown': 0}
        for u, v, data in network.graph.edges(data=True):
            rel = data.get('relationship', 'unknown')
            if rel in relationship_counts:
                relationship_counts[rel] += 1
            else:
                relationship_counts['unknown'] += 1

        stats['relationships'] = relationship_counts

        return stats

    def print_topology_info(self, date: str = '20260101'):
        """Print information about a CAIDA topology dataset."""
        print("=" * 60)
        print(f"CAIDA AS Topology: {date}")
        print("=" * 60)

        # Parse date
        try:
            dt = datetime.strptime(date, '%Y%m%d')
            print(f"Date: {dt.strftime('%B %d, %Y')}")
        except:
            print(f"Date: {date}")

        print(f"URL: {self.base_url}{date}.as-rel2.txt.bz2")
        print(f"Format: AS Relationships (serial-2)")
        print(f"\nRelationship types:")
        print(f"  -1: Provider-to-Customer")
        print(f"   0: Peer-to-Peer")
        print("=" * 60)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("CAIDA AS Topology Loader - Example Usage")
    print("=" * 60)

    loader = CAIDATopologyLoader()

    # Print available dates
    print("\n1. Available Dates")
    print("-" * 60)
    dates = loader.get_available_dates()
    print("Recent available snapshots:")
    for date in dates:
        dt = datetime.strptime(date, '%Y%m%d')
        print(f"  {date} - {dt.strftime('%B %Y')}")

    # Print dataset info
    print("\n2. Dataset Information")
    print("-" * 60)
    loader.print_topology_info('20260101')

    # Download and load topology
    print("\n3. Loading AS Topology")
    print("-" * 60)

    try:
        network = loader.load_topology(
            date='20260101',
            download_if_missing=True,
            as_undirected=True,
            largest_component=True
        )

        print(f"\nLoaded network: {network}")

        # Statistics
        stats = loader.get_topology_statistics(network)
        print(f"\nTopology Statistics:")
        print(f"  Nodes (ASes): {stats['n_nodes']}")
        print(f"  Edges: {stats['n_edges']}")
        print(f"  Density: {stats['density']:.6f}")
        print(f"  Average degree: {stats['avg_degree']:.2f}")
        print(f"  Max degree: {stats['max_degree']}")
        print(f"\nDegree distribution:")
        print(f"  Mean: {stats['degree_distribution']['mean']:.2f}")
        print(f"  Median: {stats['degree_distribution']['median']:.2f}")
        print(f"  Std: {stats['degree_distribution']['std']:.2f}")

    except Exception as e:
        print(f"\nNote: Download may require network access.")
        print(f"Error: {e}")

    print("\n" + "=" * 60)
