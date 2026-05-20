"""Tests for real-trace validation dataset loaders."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def test_mawi_loader_imports() -> None:
    """Verify the MAWI loader imports cleanly."""
    from datasets.mawi.loader import MAWITraceProcessor

    assert MAWITraceProcessor is not None


def test_mawi_extract_arrival_series_synthetic(tmp_path: Path) -> None:
    """Extract flow arrivals from a tiny synthetic PCAP."""
    scapy_all = pytest.importorskip("scapy.all")
    from datasets.mawi.loader import MAWITraceProcessor

    packets = []
    base_time = 1_700_000_000.0
    flow_specs = [
        ("10.0.0.1", "10.0.0.2", 20),
        ("10.0.0.3", "10.0.0.4", 20),
        ("10.0.0.5", "10.0.0.6", 10),
    ]
    offset = 0
    for flow_index, (src_ip, dst_ip, packet_count) in enumerate(flow_specs):
        for packet_index in range(packet_count):
            packet = (
                scapy_all.Ether()
                / scapy_all.IP(src=src_ip, dst=dst_ip)
                / scapy_all.UDP(sport=10_000 + flow_index, dport=20_000 + flow_index)
                / scapy_all.Raw(load=b"x")
            )
            packet.time = base_time + (offset + packet_index) * 0.05
            packets.append(packet)
        offset += packet_count

    pcap_path = tmp_path / "synthetic.pcap"
    scapy_all.wrpcap(str(pcap_path), packets)

    processor = MAWITraceProcessor(data_dir=tmp_path / "mawi")
    arrivals, metadata = processor.extract_arrival_series(
        pcap_path,
        bin_seconds=1.0,
        max_flows=3,
    )
    lambda_series = processor.to_lambda_series(arrivals)

    assert len(arrivals) == 3
    assert metadata["total_packets"] == 50
    assert sum(int(counts.sum()) for counts in arrivals.values()) == 50
    assert all(counts.ndim == 1 for counts in arrivals.values())
    assert lambda_series.shape[1] == 3
    assert lambda_series.shape[0] == next(iter(arrivals.values())).shape[0]


def test_kaggle_loader_imports() -> None:
    """Verify the Kaggle loader imports cleanly."""
    from datasets.kaggle.loader import KaggleNetworkDatasetLoader

    assert KaggleNetworkDatasetLoader is not None


def test_kaggle_to_flow_arrival_series_on_fake_df(tmp_path: Path) -> None:
    """Bin a small CICIDS-like DataFrame into flow arrival series."""
    pd = pytest.importorskip("pandas")
    from datasets.kaggle.loader import KaggleNetworkDatasetLoader

    dataframe = pd.DataFrame(
        {
            " Timestamp ": [
                "2024-01-01 00:00:00",
                "2024-01-01 00:00:01",
                "2024-01-01 00:00:02",
            ],
            " Flow Duration ": [2_000_000, 0, 1_000_000],
            " Total Fwd Packets ": [4, 3, 2],
            " Flow ID ": ["long-flow", "burst-flow", "small-flow"],
        }
    )

    loader = KaggleNetworkDatasetLoader(data_dir=tmp_path / "kaggle")
    series = loader.to_flow_arrival_series(dataframe, bin_seconds=1.0, max_flows=2)

    assert series.shape == (3, 2)
    assert series.sum() == 7
    np.testing.assert_array_equal(series[:, 0], np.array([2.0, 2.0, 0.0]))
    np.testing.assert_array_equal(series[:, 1], np.array([0.0, 3.0, 0.0]))
    assert loader.last_metadata["flow_ids"] == ["long-flow", "burst-flow"]


@pytest.mark.skipif(
    os.environ.get("CLAUDE_ALLOW_NETWORK") != "1",
    reason="Set CLAUDE_ALLOW_NETWORK=1 to exercise network-aware download checks.",
)
def test_download_functions_skipped_if_no_network(tmp_path: Path) -> None:
    """Keep download tests gated behind an explicit network opt-in."""
    from datasets.kaggle.loader import KaggleNetworkDatasetLoader
    from datasets.mawi.loader import MAWITraceProcessor

    processor = MAWITraceProcessor(data_dir=tmp_path / "mawi")
    cached_trace = processor.raw_dir / "20240101.pcap.gz"
    cached_trace.write_bytes(b"cached")
    assert processor.download_trace("20240101") == cached_trace

    loader = KaggleNetworkDatasetLoader(data_dir=tmp_path / "kaggle")
    assert callable(loader.download_dataset)
