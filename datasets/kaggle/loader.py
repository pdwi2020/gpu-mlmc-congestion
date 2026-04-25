"""Load public Kaggle network traces for validation experiments."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False


class KaggleNetworkDatasetLoader:
    """Download and transform Kaggle network-flow datasets."""

    CICIDS2017_FILES = {
        "friday-portscan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    }

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        """Initialize the Kaggle dataset loader.

        Args:
            data_dir: Directory to store Kaggle datasets.
        """
        if data_dir is None:
            data_dir = Path(__file__).parent

        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.last_metadata: dict[str, Any] = {}

        logger.info(f"Kaggle dataset directory: {self.data_dir}")

    def download_dataset(self, dataset_slug: str, dest_subdir: str = "raw") -> Path:
        """Download and unzip a Kaggle dataset with the Kaggle CLI.

        Args:
            dataset_slug: Kaggle dataset slug, such as `cicdataset/cicids2017`.
            dest_subdir: Destination subdirectory under `data_dir`.

        Returns:
            Path to the destination directory.
        """
        credentials_path = Path.home() / ".kaggle" / "kaggle.json"
        if not credentials_path.exists():
            raise FileNotFoundError(
                "Kaggle credentials not found at ~/.kaggle/kaggle.json. "
                "Create an API token in Kaggle account settings and place it there."
            )
        if shutil.which("kaggle") is None:
            raise RuntimeError(
                "Kaggle CLI is not installed or not on PATH. Install it with: pip install kaggle"
            )

        dest_dir = self.data_dir / dest_subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_slug,
            "-p",
            str(dest_dir),
            "--unzip",
        ]

        logger.info(f"Downloading Kaggle dataset {dataset_slug} to {dest_dir}")
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            logger.error(f"Kaggle dataset download failed: {message}")
            raise RuntimeError(f"Kaggle dataset download failed for {dataset_slug}: {message}") from exc

        logger.info(f"Downloaded Kaggle dataset {dataset_slug} to {dest_dir}")
        return dest_dir

    def load_cicids2017(self, which: str = "friday-portscan") -> pd.DataFrame:
        """Load a CICIDS2017 CSV and normalize column names.

        Args:
            which: Named CICIDS2017 subset to load.

        Returns:
            Loaded DataFrame with stripped column names.
        """
        self._require_pandas()
        if which not in self.CICIDS2017_FILES:
            available = ", ".join(sorted(self.CICIDS2017_FILES))
            raise ValueError(f"Unknown CICIDS2017 subset {which!r}. Available: {available}")

        filename = self.CICIDS2017_FILES[which]
        matches = list(self.raw_dir.rglob(filename))
        if not matches:
            matches = [
                path
                for path in self.raw_dir.rglob("*.csv")
                if "portscan" in path.name.lower() and "friday" in path.name.lower()
            ]
        if not matches:
            raise FileNotFoundError(
                f"Could not find {filename} under {self.raw_dir}. "
                "Run download_dataset('cicdataset/cicids2017') first."
            )

        csv_path = matches[0]
        logger.info(f"Loading CICIDS2017 subset {which} from {csv_path}")
        dataframe = pd.read_csv(csv_path)
        return dataframe.rename(columns=self._strip_column_name)

    def to_flow_arrival_series(
        self,
        df: pd.DataFrame,
        bin_seconds: float = 1.0,
        max_flows: int = 500,
    ) -> np.ndarray:
        """Convert CICIDS flow rows into per-bin forward-packet arrivals.

        Args:
            df: CICIDS-like DataFrame.
            bin_seconds: Width of each time bin in seconds.
            max_flows: Maximum number of highest-packet flows to keep.

        Returns:
            Array of shape `(T, n_flows)`.
        """
        self._require_pandas()
        if bin_seconds <= 0:
            raise ValueError("bin_seconds must be positive")
        if max_flows <= 0:
            raise ValueError("max_flows must be positive")

        dataframe = df.rename(columns=self._strip_column_name).copy()
        required_columns = ["Flow Duration", "Total Fwd Packets", "Timestamp"]
        missing = [column for column in required_columns if column not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing required CICIDS columns: {missing}")

        packet_counts = pd.to_numeric(dataframe["Total Fwd Packets"], errors="coerce").fillna(0)
        durations = pd.to_numeric(dataframe["Flow Duration"], errors="coerce").fillna(0)
        start_seconds = self._timestamp_seconds(dataframe["Timestamp"])

        dataframe["_packet_count"] = packet_counts.astype(int)
        dataframe["_duration_seconds"] = np.maximum(durations.to_numpy(dtype=float), 0.0) / 1_000_000.0
        dataframe["_start_seconds"] = start_seconds
        dataframe["_flow_id"] = self._flow_ids(dataframe)
        dataframe = dataframe[
            (dataframe["_packet_count"] > 0) & np.isfinite(dataframe["_start_seconds"])
        ].copy()

        if dataframe.empty:
            self.last_metadata = {
                "bin_seconds": bin_seconds,
                "start_time": None,
                "total_packets": 0,
                "flow_ids": [],
            }
            return np.zeros((0, 0), dtype=float)

        dataframe = dataframe.sort_values("_packet_count", ascending=False).head(max_flows)
        origin = float(dataframe["_start_seconds"].min())
        end_offsets = (
            dataframe["_start_seconds"].to_numpy(dtype=float)
            - origin
            + dataframe["_duration_seconds"].to_numpy(dtype=float)
        )
        n_bins = int(np.floor(float(np.max(end_offsets)) / bin_seconds)) + 1
        arrivals = np.zeros((n_bins, len(dataframe)), dtype=float)

        for column, (_, row) in enumerate(dataframe.iterrows()):
            packet_count = int(row["_packet_count"])
            start_offset = float(row["_start_seconds"]) - origin
            duration_seconds = float(row["_duration_seconds"])
            if duration_seconds <= 0 or packet_count == 1:
                packet_times = np.full(packet_count, start_offset, dtype=float)
            else:
                packet_times = start_offset + np.linspace(
                    0.0,
                    duration_seconds,
                    num=packet_count,
                    endpoint=False,
                    dtype=float,
                )
            bin_indices = np.floor(packet_times / bin_seconds).astype(int)
            np.add.at(arrivals[:, column], np.clip(bin_indices, 0, n_bins - 1), 1.0)

        flow_ids = dataframe["_flow_id"].astype(str).tolist()
        self.last_metadata = {
            "bin_seconds": bin_seconds,
            "start_time": origin,
            "total_packets": int(dataframe["_packet_count"].sum()),
            "flow_ids": flow_ids,
            "source": "cicids2017",
        }
        logger.info(
            "Converted CICIDS rows into %d bins and %d flows",
            arrivals.shape[0],
            arrivals.shape[1],
        )
        return arrivals

    def save_processed(self, series: np.ndarray, metadata: dict[str, Any], out_path: Path) -> None:
        """Save processed Kaggle arrivals as a compressed `.npz` file.

        Args:
            series: Arrival matrix of shape `(T, n_flows)`.
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
        logger.info(f"Saved processed Kaggle arrivals to {out_path}")

    @staticmethod
    def _require_pandas() -> None:
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for Kaggle CSV loading. Install with: pip install pandas"
            )

    @staticmethod
    def _strip_column_name(column: Any) -> Any:
        if isinstance(column, str):
            return column.strip()
        return column

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
    def _timestamp_seconds(series: pd.Series) -> np.ndarray:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

        try:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        except TypeError:
            parsed = pd.to_datetime(series, errors="coerce")

        seconds = np.full(len(parsed), np.nan, dtype=float)
        valid = parsed.notna().to_numpy()
        if valid.any():
            seconds[valid] = parsed[valid].astype("int64").to_numpy(dtype=float) / 1_000_000_000.0
        return seconds

    @staticmethod
    def _flow_ids(dataframe: pd.DataFrame) -> pd.Series:
        if "Flow ID" in dataframe.columns:
            return dataframe["Flow ID"].astype(str)

        endpoint_columns = ["Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol"]
        if all(column in dataframe.columns for column in endpoint_columns):
            return (
                dataframe["Source IP"].astype(str)
                + ":"
                + dataframe["Source Port"].astype(str)
                + "->"
                + dataframe["Destination IP"].astype(str)
                + ":"
                + dataframe["Destination Port"].astype(str)
                + "/"
                + dataframe["Protocol"].astype(str)
            )

        return pd.Series([f"flow-{index}" for index in dataframe.index], index=dataframe.index)
