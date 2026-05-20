"""
MAWI Traffic Trace Processor

Extract traffic statistics from MAWI Working Group traffic traces.

Dataset: MAWI backbone traffic traces
Source: http://mawi.wide.ad.jp/~agurim/dataset/
Format: PCAP files from samplepoint-F (backbone link)

Note: This module extracts statistical properties only (not full packet-level simulation).
Requires scapy for PCAP processing (optional dependency).
"""
from __future__ import annotations

import gzip
import json
import time as time_module
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Optional
from datetime import datetime
import logging
import socket
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from network.traffic import MAWIBasedTraffic

logger = logging.getLogger(__name__)

# Check if scapy is available
try:
    from scapy.all import IP, IPv6, TCP, UDP, PcapReader, rdpcap
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logger.warning("Scapy not available. PCAP processing will be limited.")


class MAWITraceProcessor:
    """Process MAWI traffic traces and extract validation-ready series."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        mirror_url: Optional[str] = None,
        max_download_bytes: int = 100 * 1024 * 1024,
        retries: int = 3,
    ) -> None:
        """Initialize the MAWI trace processor.

        Args:
            data_dir: Directory to store datasets.
            mirror_url: Optional mirror URL for MAWI trace downloads.
            max_download_bytes: Maximum compressed bytes to cache per trace.
            retries: Number of download attempts before failing.
        """
        if data_dir is None:
            data_dir = Path(__file__).parent

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.base_url = (mirror_url or "http://mawi.wide.ad.jp/~agurim/dataset/").rstrip("/")
        self.max_download_bytes = max_download_bytes
        self.retries = retries

        logger.info(f"MAWI dataset directory: {self.data_dir}")

    def get_trace_url(self, date: str, sample_point: str = "F", time: str = "1400") -> str:
        """Construct a URL for a MAWI sample-point trace.

        Args:
            date: Date in YYYYMMDD format
            sample_point: MAWI sample point label.
            time: Time in HHMM format (default: 1400 = 2:00 PM)

        Returns:
            URL string
        """
        if sample_point.isdigit() and len(sample_point) == 4:
            time = sample_point
            sample_point = "F"
        self._validate_date(date)

        # Default archive layout requested for validation:
        #   <mirror>/<YYYY>/<YYYYMMDD>/<YYYYMMDDHHMM>.pcap.gz
        # Mirrors can opt into richer formatting with named placeholders.
        sample_point = sample_point.upper()
        filename = f"{date}{time}.pcap.gz"
        if "{" in self.base_url:
            return self.base_url.format(
                YYYY=date[:4],
                YYYYMMDD=date,
                date=date,
                time=time,
                sample_point=sample_point,
                filename=filename,
            )
        return f"{self.base_url}/{date[:4]}/{date}/{filename}"

    def download_trace(self, date: str, sample_point: str = "F") -> Path:
        """Download and cache a bounded MAWI PCAP gzip trace.

        Args:
            date: Date in YYYYMMDD format
            sample_point: MAWI sample point label, defaulting to F.

        Returns:
            Path to downloaded file
        """
        self._validate_date(date)
        filepath = self.raw_dir / f"{date}.pcap.gz"
        if filepath.exists():
            logger.info(f"Trace already exists: {filepath}")
            return filepath

        url = self.get_trace_url(date=date, sample_point=sample_point)
        tmp_path = filepath.with_suffix(filepath.suffix + ".part")
        headers = {"Range": f"bytes=0-{self.max_download_bytes - 1}"}

        logger.info(f"Downloading MAWI trace from {url}")
        logger.info(
            "Caching at most %.1f MB of compressed trace data",
            self.max_download_bytes / (1024 * 1024),
        )

        last_error: Optional[BaseException] = None
        for attempt in range(1, self.retries + 1):
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    if getattr(response, "status", None) != 206:
                        logger.warning(
                            "MAWI mirror did not honor HTTP Range; stopping after %.1f MB",
                            self.max_download_bytes / (1024 * 1024),
                        )
                    self._copy_limited(response, tmp_path, self.max_download_bytes)

                tmp_path.replace(filepath)
                file_size_mb = filepath.stat().st_size / (1024 * 1024)
                logger.info(f"Downloaded to: {filepath} ({file_size_mb:.1f} MB)")
                return filepath

            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 404:
                    logger.error(f"Trace not found for {date}. Check MAWI archive URL: {url}")
                    raise FileNotFoundError(f"MAWI trace not found for date {date}: {url}") from exc
                self._handle_download_retry(exc, attempt)

            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                self._handle_download_retry(exc, attempt)

            finally:
                if tmp_path.exists() and not filepath.exists():
                    tmp_path.unlink(missing_ok=True)

        message = f"Failed to download MAWI trace {date} after {self.retries} attempts: {last_error}"
        logger.error(message)
        raise RuntimeError(message) from last_error

    def extract_arrival_series(
        self,
        pcap_path: Path,
        bin_seconds: float = 1.0,
        max_flows: int = 500,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Extract per-flow packet arrivals from a PCAP file.

        Args:
            pcap_path: Path to a `.pcap` or `.pcap.gz` file.
            bin_seconds: Width of each time bin in seconds.
            max_flows: Maximum number of highest-volume flows to keep.

        Returns:
            Tuple of flow arrival arrays and metadata.
        """
        if bin_seconds <= 0:
            raise ValueError("bin_seconds must be positive")
        if max_flows <= 0:
            raise ValueError("max_flows must be positive")

        pcap_path = Path(pcap_path)
        logger.info(f"Extracting arrival series from {pcap_path}")

        try:
            if SCAPY_AVAILABLE:
                packet_iter = self._iter_ip_packets_scapy(pcap_path)
            else:
                packet_iter = self._iter_ip_packets_dpkt(pcap_path)

            flow_times: dict[tuple[str, str], list[float]] = defaultdict(list)
            flow_counts: Counter[tuple[str, str]] = Counter()
            total_packets = 0
            start_time: Optional[float] = None
            end_time: Optional[float] = None

            for timestamp, src_ip, dst_ip in packet_iter:
                total_packets += 1
                flow_key = (src_ip, dst_ip)
                flow_times[flow_key].append(timestamp)
                flow_counts[flow_key] += 1
                start_time = timestamp if start_time is None else min(start_time, timestamp)
                end_time = timestamp if end_time is None else max(end_time, timestamp)

        except ImportError:
            raise
        except Exception as exc:
            logger.error(f"Failed to parse PCAP {pcap_path}: {exc}")
            raise RuntimeError(f"Failed to parse PCAP {pcap_path}: {exc}") from exc

        if total_packets == 0 or start_time is None or end_time is None:
            metadata = {
                "bin_seconds": bin_seconds,
                "start_time": None,
                "total_packets": 0,
                "flow_ids": [],
                "pcap_path": str(pcap_path),
            }
            return {}, metadata

        n_bins = int(np.floor((end_time - start_time) / bin_seconds)) + 1
        top_flows = flow_counts.most_common(max_flows)
        arrivals: dict[str, np.ndarray] = {}

        for flow_key, _ in top_flows:
            timestamps = np.asarray(flow_times[flow_key], dtype=float)
            bin_indices = np.floor((timestamps - start_time) / bin_seconds).astype(int)
            counts = np.zeros(n_bins, dtype=np.int64)
            np.add.at(counts, np.clip(bin_indices, 0, n_bins - 1), 1)
            arrivals[self._flow_id(flow_key)] = counts

        flow_ids = list(arrivals.keys())
        metadata = {
            "bin_seconds": bin_seconds,
            "start_time": start_time,
            "end_time": end_time,
            "total_packets": total_packets,
            "flow_ids": flow_ids,
            "pcap_path": str(pcap_path),
        }
        logger.info(
            "Extracted %d packets across %d retained flows and %d bins",
            total_packets,
            len(arrivals),
            n_bins,
        )
        return arrivals, metadata

    def to_lambda_series(
        self,
        arrival_counts: dict[str, np.ndarray],
        scale: float = 1.0,
    ) -> np.ndarray:
        """Stack per-flow counts into a time-varying lambda matrix.

        Args:
            arrival_counts: Mapping of flow ID to per-bin packet counts.
            scale: Multiplicative scale, typically `1 / bin_seconds`.

        Returns:
            Array of shape `(T, n_flows)`.
        """
        if scale <= 0:
            raise ValueError("scale must be positive")
        if not arrival_counts:
            return np.zeros((0, 0), dtype=float)

        n_bins = max(len(values) for values in arrival_counts.values())
        series = np.zeros((n_bins, len(arrival_counts)), dtype=float)
        for column, counts in enumerate(arrival_counts.values()):
            values = np.asarray(counts, dtype=float)
            series[: len(values), column] = values * scale
        return series

    def save_processed(self, series: np.ndarray, metadata: dict[str, Any], out_path: Path) -> None:
        """Save processed MAWI arrivals as a compressed `.npz` file.

        Args:
            series: Arrival or lambda matrix of shape `(T, n_flows)`.
            metadata: Processing metadata to store as JSON.
            out_path: Output path, relative to `processed/` if not absolute.
        """
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = self.processed_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        bin_seconds = float(metadata.get("bin_seconds", 1.0))
        start_time = metadata.get("start_time")
        if start_time is None:
            times = np.arange(series.shape[0], dtype=float) * bin_seconds
        else:
            times = float(start_time) + np.arange(series.shape[0], dtype=float) * bin_seconds

        flow_ids = np.asarray(metadata.get("flow_ids", []), dtype=object)
        meta_json = json.dumps(metadata, default=self._json_default)
        np.savez_compressed(out_path, arrivals=series, times=times, flow_ids=flow_ids, meta=meta_json)
        logger.info(f"Saved processed MAWI arrivals to {out_path}")

    @staticmethod
    def _validate_date(date: str) -> None:
        if len(date) != 8 or not date.isdigit():
            raise ValueError("date must be in YYYYMMDD format")

    @staticmethod
    def _copy_limited(response: BinaryIO, tmp_path: Path, max_bytes: int) -> None:
        remaining = max_bytes
        with tmp_path.open("wb") as out_file:
            while remaining > 0:
                chunk = response.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                out_file.write(chunk)
                remaining -= len(chunk)

    def _handle_download_retry(self, exc: BaseException, attempt: int) -> None:
        if attempt >= self.retries:
            return
        delay_seconds = min(2 ** (attempt - 1), 30)
        logger.warning(
            "MAWI download attempt %d/%d failed: %s; retrying in %d seconds",
            attempt,
            self.retries,
            exc,
            delay_seconds,
        )
        time_module.sleep(delay_seconds)

    @staticmethod
    def _flow_id(flow_key: tuple[str, str]) -> str:
        return f"{flow_key[0]}->{flow_key[1]}"

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        return str(value)

    @staticmethod
    def _open_binary(pcap_path: Path) -> BinaryIO:
        if pcap_path.suffix == ".gz":
            return gzip.open(pcap_path, "rb")
        return pcap_path.open("rb")

    def _iter_ip_packets_scapy(self, pcap_path: Path) -> Iterator[tuple[float, str, str]]:
        source: str | BinaryIO
        handle: Optional[BinaryIO] = None
        if pcap_path.suffix == ".gz":
            handle = self._open_binary(pcap_path)
            source = handle
        else:
            source = str(pcap_path)

        try:
            reader = PcapReader(source)
            try:
                for packet in reader:
                    if IP in packet:
                        ip_layer = packet[IP]
                        yield float(packet.time), str(ip_layer.src), str(ip_layer.dst)
                    elif IPv6 in packet:
                        ip_layer = packet[IPv6]
                        yield float(packet.time), str(ip_layer.src), str(ip_layer.dst)
            except EOFError as exc:
                logger.warning(f"Stopped reading truncated PCAP {pcap_path}: {exc}")
            finally:
                reader.close()
        finally:
            if handle is not None:
                handle.close()

    def _iter_ip_packets_dpkt(self, pcap_path: Path) -> Iterator[tuple[float, str, str]]:
        try:
            import dpkt
        except ImportError as exc:
            raise ImportError(
                "PCAP parsing requires scapy or dpkt. Install one with: "
                "pip install scapy  # preferred, or pip install dpkt"
            ) from exc

        handle = self._open_binary(pcap_path)
        try:
            reader = dpkt.pcap.Reader(handle)
            try:
                for timestamp, packet_bytes in reader:
                    ethernet = dpkt.ethernet.Ethernet(packet_bytes)
                    ip_packet = ethernet.data
                    if isinstance(ip_packet, dpkt.ip.IP):
                        src_ip = socket.inet_ntoa(ip_packet.src)
                        dst_ip = socket.inet_ntoa(ip_packet.dst)
                    elif isinstance(ip_packet, dpkt.ip6.IP6):
                        src_ip = socket.inet_ntop(socket.AF_INET6, ip_packet.src)
                        dst_ip = socket.inet_ntop(socket.AF_INET6, ip_packet.dst)
                    else:
                        continue
                    yield float(timestamp), src_ip, dst_ip
            except (EOFError, dpkt.dpkt.NeedData) as exc:
                logger.warning(f"Stopped reading truncated PCAP {pcap_path}: {exc}")
        finally:
            handle.close()

    def extract_statistics_fast(self, pcap_path: Path, max_packets: int = 100000) -> dict[str, Any]:
        """
        Extract traffic statistics using fast method (no full PCAP parsing).

        This provides estimated statistics based on file metadata and sampling.

        Args:
            pcap_path: Path to PCAP file (.pcap or .pcap.gz)
            max_packets: Maximum packets to sample

        Returns:
            Dictionary with traffic statistics
        """
        logger.info(f"Extracting statistics (fast mode) from {pcap_path}")

        # Get file size
        file_size = pcap_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        # Estimate from file size (rough approximation)
        # Typical: 1 MB ≈ 700-1000 packets (avg 1400 bytes/packet)
        estimated_packets = int(file_size_mb * 800)

        # Use typical MAWI trace characteristics
        # Based on published MAWI statistics
        stats = {
            'source': 'estimated',
            'pcap_file': str(pcap_path),
            'file_size_mb': file_size_mb,
            'estimated_packets': estimated_packets,
            'duration': 900.0,  # 15 minutes (typical MAWI trace)

            # Estimated statistics (typical MAWI values)
            'arrival_rate': estimated_packets / 900.0,  # packets/sec
            'mean_packet_size': 800.0,  # bytes (typical for backbone)
            'packet_size_std': 400.0,

            # Burstiness (MAWI traces are typically bursty)
            'burstiness': 2.5,  # Coefficient of variation

            # Protocol mix (typical)
            'tcp_fraction': 0.85,
            'udp_fraction': 0.12,
            'other_fraction': 0.03
        }

        logger.info(f"Estimated statistics: {stats['arrival_rate']:.1f} packets/sec")

        return stats

    def extract_statistics_full(
        self,
        pcap_path: Path,
        max_packets: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Extract detailed traffic statistics from PCAP file.

        Requires scapy package.

        Args:
            pcap_path: Path to PCAP file
            max_packets: Maximum packets to process (None = all)

        Returns:
            Dictionary with detailed traffic statistics
        """
        if not SCAPY_AVAILABLE:
            raise ImportError(
                "Scapy required for full PCAP processing. "
                "Install with: pip install scapy\n"
                "Or use extract_statistics_fast() for estimation."
            )

        logger.info(f"Extracting full statistics from {pcap_path}")
        logger.info("This may take several minutes for large traces...")

        # Read packets
        packets = rdpcap(str(pcap_path), count=max_packets)
        n_packets = len(packets)

        logger.info(f"Read {n_packets} packets from PCAP")

        # Extract timestamps and sizes
        timestamps = []
        sizes = []
        protocols = {'TCP': 0, 'UDP': 0, 'Other': 0}

        for pkt in packets:
            if IP in pkt:
                timestamps.append(float(pkt.time))
                sizes.append(len(pkt))

                # Protocol classification
                if TCP in pkt:
                    protocols['TCP'] += 1
                elif UDP in pkt:
                    protocols['UDP'] += 1
                else:
                    protocols['Other'] += 1

        timestamps = np.array(timestamps)
        sizes = np.array(sizes)

        # Compute statistics
        duration = timestamps[-1] - timestamps[0]
        arrival_rate = n_packets / duration

        # Inter-arrival times
        inter_arrivals = np.diff(timestamps)
        mean_inter = np.mean(inter_arrivals)
        std_inter = np.std(inter_arrivals)
        burstiness = std_inter / mean_inter if mean_inter > 0 else 1.0

        # Packet sizes
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)

        # Protocol fractions
        tcp_frac = protocols['TCP'] / n_packets
        udp_frac = protocols['UDP'] / n_packets
        other_frac = protocols['Other'] / n_packets

        stats = {
            'source': 'pcap_full',
            'pcap_file': str(pcap_path),
            'n_packets': n_packets,
            'duration': duration,

            # Arrival statistics
            'arrival_rate': arrival_rate,
            'mean_inter_arrival': mean_inter,
            'std_inter_arrival': std_inter,
            'burstiness': burstiness,

            # Packet size statistics
            'mean_packet_size': mean_size,
            'packet_size_std': std_size,
            'min_packet_size': float(np.min(sizes)),
            'max_packet_size': float(np.max(sizes)),

            # Protocol distribution
            'tcp_fraction': tcp_frac,
            'udp_fraction': udp_frac,
            'other_fraction': other_frac
        }

        logger.info(f"Extracted statistics: {arrival_rate:.1f} packets/sec, "
                   f"burstiness={burstiness:.2f}")

        return stats

    def create_traffic_model(
        self,
        pcap_path: Optional[Path] = None,
        stats: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> MAWIBasedTraffic:
        """
        Create MAWIBasedTraffic model from trace statistics.

        Args:
            pcap_path: Path to PCAP file (will extract stats)
            stats: Pre-computed statistics dictionary
            seed: Random seed for traffic model

        Returns:
            MAWIBasedTraffic object
        """
        if stats is None:
            if pcap_path is None:
                raise ValueError("Must provide either pcap_path or stats")

            # Extract statistics (use fast mode by default)
            stats = self.extract_statistics_fast(pcap_path)

        # Create traffic model
        traffic_model = MAWIBasedTraffic(
            arrival_rate=stats['arrival_rate'],
            burstiness=stats['burstiness'],
            mean_packet_size=stats['mean_packet_size'],
            packet_size_std=stats['packet_size_std'],
            seed=seed
        )

        logger.info(f"Created MAWI-based traffic model: {traffic_model}")

        return traffic_model

    def print_trace_info(self, date: str = '20240619', time: str = '1400') -> None:
        """Print information about a MAWI trace."""
        print("=" * 60)
        print(f"MAWI Traffic Trace: {date} {time}")
        print("=" * 60)

        # Parse date
        try:
            dt = datetime.strptime(f"{date}{time}", '%Y%m%d%H%M')
            print(f"Date/Time: {dt.strftime('%B %d, %Y at %H:%M')}")
        except ValueError:
            print(f"Date/Time: {date} {time}")

        url = self.get_trace_url(date, time)
        print(f"URL: {url}")
        print(f"Samplepoint: F (backbone link)")
        print(f"Duration: ~15 minutes")
        print(f"Typical size: 100-500 MB compressed")
        print("=" * 60)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("MAWI Traffic Trace Processor - Example Usage")
    print("=" * 60)

    processor = MAWITraceProcessor()

    # Print trace info
    print("\n1. Trace Information")
    print("-" * 60)
    processor.print_trace_info('20240619', '1400')

    # Demonstrate fast statistics extraction
    print("\n2. Fast Statistics Extraction (No PCAP needed)")
    print("-" * 60)

    # Create a dummy path for demonstration
    dummy_path = Path("example_trace.pcap.gz")

    print("\nNote: This example demonstrates the API.")
    print("To actually download and process traces:")
    print("  1. Download trace: processor.download_trace('20240619', sample_point='F')")
    print("  2. Extract stats: stats = processor.extract_statistics_fast(pcap_path)")
    print("  3. Create model: traffic = processor.create_traffic_model(stats=stats)")

    # Show what statistics would look like
    print("\nExample statistics structure:")
    example_stats = {
        'arrival_rate': 1200.5,
        'burstiness': 2.3,
        'mean_packet_size': 850.0,
        'packet_size_std': 420.0
    }

    print("  Statistics extracted from MAWI trace:")
    for key, value in example_stats.items():
        print(f"    {key}: {value}")

    print("\nThese statistics can be used to create realistic traffic models")
    print("for network simulation without full packet-level processing.")

    print("\n" + "=" * 60)
