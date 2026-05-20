"""
CAIDA AS Relationships Dataset Loader

Download and load Internet AS-level topology from CAIDA.

Dataset: AS Relationships (serial-2)
Source: http://data.caida.org/datasets/as-relationships/serial-2/
Format: <AS1>|<AS2>|<relationship>
  -1 = provider-to-customer
   0 = peer-to-peer
"""
from __future__ import annotations

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


class CAIDAPassiveTraceLoader:
    """Download and process CAIDA Anonymized Internet Traces (passive backbone PCAP).

    Companion to ``CAIDATopologyLoader``. Targets the gated
    ``passive-<year>-pcap`` archives. After CAIDA approves a dataset request
    (typically 2-3 business days), users receive HTTP Basic credentials that
    must be supplied here either directly or via environment variables
    ``CAIDA_USER`` / ``CAIDA_PASSWORD``.

    The loader applies the same Range-capped bounded-download strategy as
    ``MAWITraceProcessor`` so the downloaded prefix stays well below typical
    laptop / Colab disk budgets, and reuses the MAWI extractor for per-flow
    arrival binning. Anonymization is prefix-preserving at the IP layer; the
    pcap timestamps and flow tuples are exposed normally.
    """

    DEFAULT_BASE_URL = "https://data.caida.org/datasets"

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        base_url: Optional[str] = None,
        max_download_bytes: int = 100 * 1024 * 1024,
        retries: int = 3,
    ) -> None:
        """Initialize the passive trace loader.

        Args:
            data_dir: Directory to store passive trace caches (defaults to
                ``datasets/caida/passive``).
            username: CAIDA-issued HTTP Basic Auth username (or read from
                ``CAIDA_USER`` env var).
            password: CAIDA-issued HTTP Basic Auth password (or read from
                ``CAIDA_PASSWORD`` env var).
            base_url: Override the dataset root URL.
            max_download_bytes: Maximum compressed bytes to fetch per trace
                via HTTP Range. Defaults to 100 MiB; sufficient for tens of
                seconds of 10 G backbone traffic.
            retries: Download retry count.
        """
        import os

        if data_dir is None:
            data_dir = Path(__file__).parent / "passive"
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.username = username or os.environ.get("CAIDA_USER")
        self.password = password or os.environ.get("CAIDA_PASSWORD")
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.max_download_bytes = int(max_download_bytes)
        self.retries = int(retries)

        logger.info("CAIDA passive trace cache: %s", self.data_dir)

    def get_trace_url(self, year: int, filename: str) -> str:
        """Construct the HTTPS URL for a specific passive PCAP filename.

        CAIDA stores anonymized traces under
        ``<base>/passive-<year>-pcap/<filename>``; consult the
        approval email for the exact filename you want (e.g.
        ``equinix-nyc.dirA.20180315-130000.UTC.anon.pcap.gz``).
        """
        return f"{self.base_url}/passive-{year}-pcap/{filename}"

    def download_trace(self, year: int, filename: str) -> Path:
        """Range-capped download of one anonymized PCAP gzip.

        Behaves like ``MAWITraceProcessor.download_trace`` but adds HTTP
        Basic Auth headers using the loader credentials. Falls back to
        a full download if the server does not honor Range.
        """
        if not self.username or not self.password:
            raise RuntimeError(
                "CAIDA passive credentials missing. Set CAIDA_USER + "
                "CAIDA_PASSWORD env vars, or pass username/password kwargs."
            )

        out_path = self.raw_dir / filename
        if out_path.exists():
            logger.info("Cached passive trace already present: %s", out_path)
            return out_path

        url = self.get_trace_url(year, filename)
        logger.info("Downloading passive trace from %s", url)
        logger.info(
            "Caching at most %.1f MB of compressed trace data",
            self.max_download_bytes / (1024 * 1024),
        )

        import urllib.request as _ur

        # HTTP Basic Auth handler.
        pwd_mgr = _ur.HTTPPasswordMgrWithDefaultRealm()
        pwd_mgr.add_password(None, url, self.username, self.password)
        auth_handler = _ur.HTTPBasicAuthHandler(pwd_mgr)
        opener = _ur.build_opener(auth_handler)

        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                req = _ur.Request(
                    url,
                    headers={"Range": f"bytes=0-{self.max_download_bytes - 1}"},
                )
                with opener.open(req, timeout=120) as resp, open(tmp_path, "wb") as fh:
                    written = 0
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        written += len(chunk)
                        if written >= self.max_download_bytes:
                            break
                    if written == 0:
                        raise RuntimeError("Empty response from CAIDA")
                tmp_path.rename(out_path)
                logger.info(
                    "Downloaded %.1f MB of %s",
                    out_path.stat().st_size / (1024 * 1024),
                    filename,
                )
                return out_path
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "CAIDA passive download attempt %d/%d failed: %s",
                    attempt, self.retries, exc,
                )
        raise RuntimeError(
            f"Failed to download CAIDA passive trace {url}: {last_err}"
        )

    def extract_arrival_series(
        self,
        pcap_path: Path,
        bin_seconds: float = 1.0,
        max_flows: int = 500,
    ) -> Tuple[Dict[str, "object"], Dict[str, "object"]]:
        """Delegate to the MAWI extractor; identical PCAP semantics."""
        # Lazy import to avoid hard dependency at module-load time.
        from datasets.mawi.loader import MAWITraceProcessor  # type: ignore

        proxy = MAWITraceProcessor(data_dir=self.data_dir)
        return proxy.extract_arrival_series(
            pcap_path, bin_seconds=bin_seconds, max_flows=max_flows
        )

    def to_lambda_series(
        self,
        arrival_counts: Dict[str, "object"],
        scale: float = 1.0,
    ) -> "object":
        """Stack per-flow counts into a (T, n_flows) array (delegated)."""
        from datasets.mawi.loader import MAWITraceProcessor  # type: ignore

        proxy = MAWITraceProcessor(data_dir=self.data_dir)
        return proxy.to_lambda_series(arrival_counts, scale=scale)


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
