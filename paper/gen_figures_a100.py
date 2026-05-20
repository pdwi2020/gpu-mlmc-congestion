"""Generate six publication-quality A100 figures from merged GPU simulation CSVs.

Run 1 (`prior_work_gpu_mc_vs_gpu_mlmc_colab.csv`) is treated as the primary
source. The script supplements only `epsilon == 0.01` rows from
`run3_caida_final.csv`, filters out rows where
`equal_accuracy_ci_targeted == False`, and saves six PNG figures under
`paper/figures/`.
"""

from __future__ import annotations

import ast
import io
import json
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_CACHE_ROOT = Path(tempfile.gettempdir()) / "gpu_acc_net_prop_cache"
_MPL_CACHE = _CACHE_ROOT / "matplotlib"
_XDG_CACHE = _CACHE_ROOT / "xdg"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
_XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TITLE_SIZE = 12
LABEL_SIZE = 10
LEGEND_SIZE = 8
TICK_SIZE = 9
DPI = 150

SCENARIO_DISPLAY_NAMES: Dict[str, str] = {
    "synthetic_n100": "Synth ER n=100",
    "synthetic_n500": "Synth ER n=500",
    "real_caida_asrel2_20260101_n500": "CAIDA n=500",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PAPER_DIR / "figures"
RUN1_CSV = (
    PROJECT_ROOT
    / "results"
    / "results"
    / "runpod_a100_20260314"
    / "prior_work_gpu_mc_vs_gpu_mlmc_colab.csv"
)
RUN3_CSV = (
    PROJECT_ROOT
    / "results"
    / "results"
    / "runpod_a100_20260314"
    / "run3_caida_final.csv"
)
RUN_EXT_CSV = (
    PROJECT_ROOT
    / "results"
    / "results"
    / "runpod_a100_extended"
    / "extended_epsilon_results.csv"
)
SEED_RUN_DIR = PROJECT_ROOT / "results" / "results" / "runs_5seed_colab"

PALETTE = sns.color_palette("tab10", n_colors=10)
METHOD_COLORS = {"GPU-MC": PALETTE[0], "GPU-MLMC": PALETTE[1]}
SCENARIO_COLORS = {
    "synthetic_n100": PALETTE[2],
    "synthetic_n500": PALETTE[3],
    "real_caida_asrel2_20260101_n500": PALETTE[4],
}

FIGURE_FILENAMES = [
    "loglog_cost_vs_epsilon_a100.png",
    "runtime_vs_epsilon_a100.png",
    "cost_ratio_vs_epsilon_a100.png",
    "mlmc_level_allocation_a100.png",
    "empirical_slope_bars_a100.png",
    "runtime_scaling_tightest_epsilon_a100.png",
]

NUMERIC_COLUMNS = [
    "nodes",
    "epsilon",
    "h_finest",
    "mc_paths",
    "mlmc_levels",
    "mc_runtime_s",
    "mlmc_runtime_s",
    "speedup_runtime",
    "mc_cost",
    "mlmc_cost",
    "cost_ratio_mc_over_mlmc",
    "mc_estimate",
    "mlmc_estimate",
    "ci_target_half",
    "mc_ci_half",
    "mlmc_ci_half",
    "error_proxy_mc_ci2_plus_hL",
    "error_proxy_mlmc_ci2_plus_hL",
]

CSV_BLOCK_START_RE = re.compile(r"^===\s*CSV_START:(?P<filename>.+?)\s*===\s*$")
CSV_BLOCK_END_RE = re.compile(r"^===\s*CSV_END:(?P<filename>.+?)\s*===\s*$")


sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update(
    {
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
    }
)


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _scenario_order(df: pd.DataFrame) -> List[str]:
    preferred = [key for key in SCENARIO_DISPLAY_NAMES if key in set(df["scenario"])]
    extras = sorted(set(df["scenario"]) - set(preferred))
    return preferred + extras


def _scenario_label(scenario: str) -> str:
    return SCENARIO_DISPLAY_NAMES.get(scenario, scenario)


def _parse_level_samples(raw: object) -> List[int]:
    try:
        parsed = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [int(value) for value in parsed]


def _fit_log10_slope(subset: pd.DataFrame, value_col: str) -> Optional[float]:
    valid = subset[["epsilon", value_col]].dropna().copy()
    valid = valid[(valid["epsilon"] > 0) & (valid[value_col] > 0)]
    if len(valid) < 2:
        return None
    slope, _ = np.polyfit(np.log10(valid["epsilon"]), np.log10(valid[value_col]), 1)
    return float(slope)


def _tightest_reliable_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in _scenario_order(df):
        subset = df[df["scenario"] == scenario].sort_values("epsilon")
        if not subset.empty:
            rows.append(subset.iloc[0])
    if not rows:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(rows).reset_index(drop=True)


def _save_figure(fig: plt.Figure, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _std_or_zero(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    if len(valid) == 1:
        return 0.0
    return float(valid.std(ddof=0))


def load_results() -> pd.DataFrame:
    if not RUN1_CSV.exists():
        raise FileNotFoundError(f"Missing Run 1 CSV: {RUN1_CSV}")
    if not RUN3_CSV.exists():
        raise FileNotFoundError(f"Missing Run 3 CSV: {RUN3_CSV}")

    run1 = pd.read_csv(RUN1_CSV)
    run3 = pd.read_csv(RUN3_CSV)

    run1["source_run"] = "run1"
    run3_eps001 = run3[
        np.isclose(pd.to_numeric(run3["epsilon"], errors="coerce"), 0.01)
    ].copy()
    run3_eps001["source_run"] = "run3_eps001"

    merged = pd.concat([run1, run3_eps001], ignore_index=True, sort=False)
    merged = merged.drop_duplicates(
        subset=["scenario", "nodes", "epsilon"], keep="first"
    )

    # Merge extended-ε rows if available (Chunk 4.1)
    if RUN_EXT_CSV.exists():
        run_ext = pd.read_csv(RUN_EXT_CSV)
        run_ext["source_run"] = "run_extended"
        merged = pd.concat([merged, run_ext], ignore_index=True, sort=False)
        merged = merged.drop_duplicates(
            subset=["scenario", "nodes", "epsilon"], keep="first"
        )

    merged = _coerce_numeric_columns(merged)
    merged["equal_accuracy_ci_targeted"] = _coerce_bool(
        merged["equal_accuracy_ci_targeted"]
    )
    merged["scenario_display"] = (
        merged["scenario"].map(SCENARIO_DISPLAY_NAMES).fillna(merged["scenario"])
    )

    reliable = merged[merged["equal_accuracy_ci_targeted"]].copy()
    reliable = reliable.sort_values(["scenario_display", "epsilon"]).reset_index(
        drop=True
    )
    import logging as _logging

    _log = _logging.getLogger(__name__)
    _log.debug(f"Reliable rows after filter: {len(reliable)}")
    return reliable


def load_seed_runs() -> Optional[pd.DataFrame]:
    if not SEED_RUN_DIR.exists():
        return None

    seed_frames: List[pd.DataFrame] = []
    for raw_path in sorted(SEED_RUN_DIR.glob("colab_seed*_raw.txt")):
        try:
            raw_text = raw_path.read_text(encoding="utf-8")
        except OSError as exc:
            warnings.warn(f"Skipping unreadable seed run file {raw_path}: {exc}")
            continue

        seed_run = raw_path.stem.removesuffix("_raw")
        active_filename: Optional[str] = None
        active_lines: List[str] = []

        for line_number, line in enumerate(raw_text.splitlines(keepends=True), start=1):
            start_match = CSV_BLOCK_START_RE.match(line)
            if start_match:
                if active_filename is not None:
                    warnings.warn(
                        f"Skipping unterminated CSV block {active_filename!r} in {raw_path} before line {line_number}."
                    )
                active_filename = start_match.group("filename").strip()
                active_lines = []
                continue

            end_match = CSV_BLOCK_END_RE.match(line)
            if end_match:
                end_filename = end_match.group("filename").strip()
                if active_filename is None:
                    warnings.warn(
                        f"Found CSV_END for {end_filename!r} without CSV_START in {raw_path}:{line_number}."
                    )
                    continue
                if end_filename != active_filename:
                    warnings.warn(
                        f"Skipping mismatched CSV block in {raw_path}:{line_number} "
                        f"(start={active_filename!r}, end={end_filename!r})."
                    )
                    active_filename = None
                    active_lines = []
                    continue

                csv_payload = "".join(active_lines).strip()
                if not csv_payload:
                    warnings.warn(
                        f"Skipping empty CSV block {active_filename!r} in {raw_path}."
                    )
                else:
                    try:
                        block_df = pd.read_csv(io.StringIO(csv_payload))
                    except Exception as exc:
                        warnings.warn(
                            f"Skipping unparsable CSV block {active_filename!r} in {raw_path}: {exc}"
                        )
                    else:
                        if block_df.empty:
                            warnings.warn(
                                f"Skipping empty parsed CSV block {active_filename!r} in {raw_path}."
                            )
                        else:
                            block_df["seed_run"] = seed_run
                            seed_frames.append(block_df)

                active_filename = None
                active_lines = []
                continue

            if active_filename is not None:
                active_lines.append(line)

        if active_filename is not None:
            warnings.warn(
                f"Skipping unterminated CSV block {active_filename!r} at end of {raw_path}."
            )

    if not seed_frames:
        return None

    seed_df = pd.concat(seed_frames, ignore_index=True, sort=False)
    seed_df = _coerce_numeric_columns(seed_df)
    if "equal_accuracy_ci_targeted" in seed_df.columns:
        seed_df["equal_accuracy_ci_targeted"] = _coerce_bool(
            seed_df["equal_accuracy_ci_targeted"]
        )
    return seed_df


def compute_multi_run_stats(seed_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if seed_df is None or seed_df.empty:
        return None

    required_columns = {
        "scenario",
        "epsilon",
        "cost_ratio_mc_over_mlmc",
        "speedup_runtime",
    }
    if not required_columns.issubset(seed_df.columns):
        missing = ", ".join(sorted(required_columns - set(seed_df.columns)))
        warnings.warn(
            f"Skipping multi-run stats because required seed columns are missing: {missing}"
        )
        return None

    stats_source = seed_df.copy()
    if "equal_accuracy_ci_targeted" in stats_source.columns:
        stats_source = stats_source[
            _coerce_bool(stats_source["equal_accuracy_ci_targeted"])
        ].copy()

    stats_source = _coerce_numeric_columns(stats_source)
    stats_source = stats_source.dropna(subset=["scenario", "epsilon"])
    if stats_source.empty:
        return None

    stats_df = (
        stats_source.groupby(["scenario", "epsilon"], as_index=False)
        .agg(
            cost_ratio_mean=("cost_ratio_mc_over_mlmc", "mean"),
            cost_ratio_std=("cost_ratio_mc_over_mlmc", _std_or_zero),
            speedup_runtime_mean=("speedup_runtime", "mean"),
            speedup_runtime_std=("speedup_runtime", _std_or_zero),
            n_seed_runs=("seed_run", "nunique")
            if "seed_run" in stats_source.columns
            else ("scenario", "size"),
        )
        .sort_values(["scenario", "epsilon"])
        .reset_index(drop=True)
    )
    stats_df["scenario_display"] = (
        stats_df["scenario"].map(SCENARIO_DISPLAY_NAMES).fillna(stats_df["scenario"])
    )
    return stats_df


def compute_slopes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in _scenario_order(df):
        subset = df[df["scenario"] == scenario].sort_values("epsilon")
        for method, value_col, theory in (
            ("GPU-MC", "mc_cost", -3.0),
            ("GPU-MLMC", "mlmc_cost", -2.0),
        ):
            slope = _fit_log10_slope(subset, value_col)
            if slope is None:
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_display": _scenario_label(scenario),
                    "method": method,
                    "slope": slope,
                    "theory": theory,
                }
            )
    return pd.DataFrame(rows)


def plot_loglog_cost_vs_epsilon(df: pd.DataFrame) -> Path:
    scenarios = _scenario_order(df)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(8, 5), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    for index, (ax, scenario) in enumerate(zip(axes, scenarios)):
        subset = df[df["scenario"] == scenario].sort_values("epsilon")
        label = _scenario_label(scenario)
        mc_slope = _fit_log10_slope(subset, "mc_cost")
        mlmc_slope = _fit_log10_slope(subset, "mlmc_cost")

        mc_label = "GPU-MC" if mc_slope is None else f"GPU-MC (fit={mc_slope:.2f})"
        mlmc_label = (
            "GPU-MLMC" if mlmc_slope is None else f"GPU-MLMC (fit={mlmc_slope:.2f})"
        )

        ax.plot(
            subset["epsilon"],
            subset["mc_cost"],
            color=METHOD_COLORS["GPU-MC"],
            marker="o",
            linewidth=1.8,
            label=mc_label,
        )
        ax.plot(
            subset["epsilon"],
            subset["mlmc_cost"],
            color=METHOD_COLORS["GPU-MLMC"],
            marker="s",
            linestyle="--",
            linewidth=1.8,
            label=mlmc_label,
        )

        if not subset.empty:
            x_ref = np.array(
                [subset["epsilon"].min(), subset["epsilon"].max()], dtype=float
            )
            eps_anchor = float(subset["epsilon"].max())
            mc_anchor = float(subset.loc[subset["epsilon"].idxmax(), "mc_cost"])
            mlmc_anchor = float(subset.loc[subset["epsilon"].idxmax(), "mlmc_cost"])
            ax.plot(
                x_ref,
                mc_anchor * (x_ref / eps_anchor) ** (-3.0),
                color=PALETTE[7],
                linestyle=":",
                linewidth=1.2,
                label="Ref slope -3",
            )
            ax.plot(
                x_ref,
                mlmc_anchor * (x_ref / eps_anchor) ** (-2.0),
                color=PALETTE[8],
                linestyle="-.",
                linewidth=1.2,
                label="Ref slope -2",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xlabel("Epsilon")
        if index == 0:
            ax.set_ylabel("Cost proxy")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(fontsize=LEGEND_SIZE, loc="best")

    fig.suptitle("A100 Cost Scaling vs Epsilon", fontsize=TITLE_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_figure(fig, "loglog_cost_vs_epsilon_a100.png")


def plot_runtime_vs_epsilon(
    df: pd.DataFrame, stats_df: Optional[pd.DataFrame] = None
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    speedup_ax = None
    for scenario in _scenario_order(df):
        subset = df[df["scenario"] == scenario].sort_values("epsilon")
        color = SCENARIO_COLORS.get(scenario, PALETTE[0])
        label = _scenario_label(scenario)
        ax.plot(
            subset["epsilon"],
            subset["mc_runtime_s"],
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1.8,
            label=f"{label} | GPU-MC",
        )
        ax.plot(
            subset["epsilon"],
            subset["mlmc_runtime_s"],
            color=color,
            marker="s",
            linestyle="--",
            linewidth=1.8,
            label=f"{label} | GPU-MLMC",
        )

        if stats_df is not None:
            stats_subset = stats_df[stats_df["scenario"] == scenario].sort_values(
                "epsilon"
            )
            if not stats_subset.empty:
                if speedup_ax is None:
                    speedup_ax = ax.twinx()
                speedup_mean = stats_subset["speedup_runtime_mean"].to_numpy(
                    dtype=float
                )
                speedup_std = (
                    stats_subset["speedup_runtime_std"]
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                speedup_ax.plot(
                    stats_subset["epsilon"],
                    speedup_mean,
                    color=color,
                    linestyle=":",
                    linewidth=1.3,
                    alpha=0.9,
                )
                speedup_ax.fill_between(
                    stats_subset["epsilon"],
                    np.clip(speedup_mean - speedup_std, a_min=0.0, a_max=None),
                    speedup_mean + speedup_std,
                    color=color,
                    alpha=0.2,
                )

    ax.set_xscale("log")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("A100 Runtime vs Epsilon")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=LEGEND_SIZE, ncol=2)
    if speedup_ax is not None:
        speedup_ax.set_ylabel("Speedup (MC / MLMC)")
        speedup_ax.grid(False)
    fig.tight_layout()
    return _save_figure(fig, "runtime_vs_epsilon_a100.png")


def plot_cost_ratio_vs_epsilon(
    df: pd.DataFrame, stats_df: Optional[pd.DataFrame] = None
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for scenario in _scenario_order(df):
        subset = df[df["scenario"] == scenario].sort_values("epsilon")
        color = SCENARIO_COLORS.get(scenario, PALETTE[0])
        label = _scenario_label(scenario)
        if stats_df is not None:
            stats_subset = stats_df[stats_df["scenario"] == scenario].sort_values(
                "epsilon"
            )
            if not stats_subset.empty:
                cost_ratio_mean = stats_subset["cost_ratio_mean"].to_numpy(dtype=float)
                cost_ratio_std = (
                    stats_subset["cost_ratio_std"].fillna(0.0).to_numpy(dtype=float)
                )
                ax.fill_between(
                    stats_subset["epsilon"],
                    np.clip(cost_ratio_mean - cost_ratio_std, a_min=0.0, a_max=None),
                    cost_ratio_mean + cost_ratio_std,
                    color=color,
                    alpha=0.2,
                )
        ax.plot(
            subset["epsilon"],
            subset["cost_ratio_mc_over_mlmc"],
            color=color,
            marker="o",
            linewidth=1.8,
            label=label,
        )
        peak_row = subset.loc[subset["cost_ratio_mc_over_mlmc"].idxmax()]
        ax.annotate(
            f"{peak_row['cost_ratio_mc_over_mlmc']:.1f}x",
            xy=(peak_row["epsilon"], peak_row["cost_ratio_mc_over_mlmc"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=LEGEND_SIZE,
            color=color,
        )

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("MC cost / MLMC cost")
    ax.set_title("A100 Cost Ratio vs Epsilon")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=LEGEND_SIZE)
    fig.tight_layout()
    return _save_figure(fig, "cost_ratio_vs_epsilon_a100.png")


def plot_mlmc_level_allocation(df: pd.DataFrame) -> Path:
    tightest = _tightest_reliable_rows(df)
    allocation_rows: List[Tuple[str, float, List[int]]] = []
    for _, row in tightest.iterrows():
        samples = _parse_level_samples(row["mlmc_N_l"])
        if samples:
            allocation_rows.append(
                (str(row["scenario"]), float(row["epsilon"]), samples)
            )

    fig, ax = plt.subplots(figsize=(6, 4))
    if not allocation_rows:
        ax.text(
            0.5, 0.5, "No MLMC level allocations available.", ha="center", va="center"
        )
        ax.set_axis_off()
    else:
        max_levels = max(len(samples) for _, _, samples in allocation_rows)
        x = np.arange(max_levels, dtype=float)
        width = 0.8 / len(allocation_rows)

        for index, (scenario, epsilon, samples) in enumerate(allocation_rows):
            offsets = x + (index - (len(allocation_rows) - 1) / 2.0) * width
            heights = np.full(max_levels, np.nan)
            heights[: len(samples)] = samples
            ax.bar(
                offsets,
                heights,
                width=width,
                color=SCENARIO_COLORS.get(scenario, PALETTE[index]),
                label=f"{_scenario_label(scenario)} (eps={epsilon:g})",
            )

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(level)) for level in x])
        ax.set_xlabel("MLMC level index")
        ax.set_ylabel("Allocated samples N_l")
        ax.set_title("A100 MLMC Level Allocation at Tightest Reliable Epsilon")
        ax.grid(True, axis="y", which="both", alpha=0.35)
        ax.legend(fontsize=LEGEND_SIZE)

    fig.tight_layout()
    return _save_figure(fig, "mlmc_level_allocation_a100.png")


def plot_empirical_slope_bars(
    slopes_df: pd.DataFrame, scenario_order: Sequence[str]
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    if slopes_df.empty:
        ax.text(
            0.5, 0.5, "Insufficient data for slope fitting.", ha="center", va="center"
        )
        ax.set_axis_off()
        fig.tight_layout()
        return _save_figure(fig, "empirical_slope_bars_a100.png")

    pivot = slopes_df.pivot(index="scenario", columns="method", values="slope")
    pivot = pivot.reindex(
        [scenario for scenario in scenario_order if scenario in pivot.index]
    )

    y = np.arange(len(pivot), dtype=float)
    height = 0.36

    mc_values = (
        pivot["GPU-MC"].to_numpy(dtype=float)
        if "GPU-MC" in pivot.columns
        else np.full(len(pivot), np.nan)
    )
    mlmc_values = (
        pivot["GPU-MLMC"].to_numpy(dtype=float)
        if "GPU-MLMC" in pivot.columns
        else np.full(len(pivot), np.nan)
    )

    ax.barh(
        y + height / 2,
        mc_values,
        height=height,
        color=METHOD_COLORS["GPU-MC"],
        label="GPU-MC",
    )
    ax.barh(
        y - height / 2,
        mlmc_values,
        height=height,
        color=METHOD_COLORS["GPU-MLMC"],
        label="GPU-MLMC",
    )

    for y_pos, value in zip(y + height / 2, mc_values):
        if np.isfinite(value):
            ax.text(
                value - 0.05,
                y_pos,
                f"{value:.2f}",
                va="center",
                ha="right",
                fontsize=LEGEND_SIZE,
            )
    for y_pos, value in zip(y - height / 2, mlmc_values):
        if np.isfinite(value):
            ax.text(
                value - 0.05,
                y_pos,
                f"{value:.2f}",
                va="center",
                ha="right",
                fontsize=LEGEND_SIZE,
            )

    ax.axvline(-3.0, color=PALETTE[7], linestyle=":", linewidth=1.2, label="Ref -3")
    ax.axvline(-2.0, color=PALETTE[8], linestyle="--", linewidth=1.2, label="Ref -2")
    ax.set_yticks(y)
    ax.set_yticklabels([_scenario_label(scenario) for scenario in pivot.index])
    ax.set_xlabel("Fitted slope of log10(cost) vs log10(epsilon)")
    ax.set_title("Empirical A100 Cost Slopes")
    ax.grid(True, axis="x", alpha=0.35)
    ax.legend(fontsize=LEGEND_SIZE, ncol=2)
    ax.invert_yaxis()
    fig.tight_layout()
    return _save_figure(fig, "empirical_slope_bars_a100.png")


def plot_runtime_scaling_tightest_epsilon(df: pd.DataFrame) -> Path:
    tightest = _tightest_reliable_rows(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    if tightest.empty:
        ax.text(
            0.5, 0.5, "No reliable runtime rows available.", ha="center", va="center"
        )
        ax.set_axis_off()
        fig.tight_layout()
        return _save_figure(fig, "runtime_scaling_tightest_epsilon_a100.png")

    x = np.arange(len(tightest), dtype=float)
    width = 0.36

    mc_bars = ax.bar(
        x - width / 2,
        tightest["mc_runtime_s"],
        width=width,
        color=METHOD_COLORS["GPU-MC"],
        label="GPU-MC",
    )
    mlmc_bars = ax.bar(
        x + width / 2,
        tightest["mlmc_runtime_s"],
        width=width,
        color=METHOD_COLORS["GPU-MLMC"],
        label="GPU-MLMC",
    )

    for bars in (mc_bars, mlmc_bars):
        for bar in bars:
            height = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=LEGEND_SIZE,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_scenario_label(scenario) for scenario in tightest["scenario"]],
        rotation=10,
        ha="right",
    )
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("A100 Runtime at Tightest Reliable Epsilon")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=LEGEND_SIZE)
    fig.tight_layout()
    return _save_figure(fig, "runtime_scaling_tightest_epsilon_a100.png")


def plot_memory_bandwidth_comparison(bw_log_path: Path) -> Optional[Path]:
    """Bar chart: legacy vs transposed CUDA memory layout bandwidth (GB/s).

    Parses log lines of the form:
        Memory layout benchmark  |  legacy=X.X ms (Y.Y GB/s)  |  transposed=...
    and produces a two-bar figure comparing the two layouts.
    """
    import re

    if not bw_log_path.exists():
        warnings.warn(f"Memory benchmark log not found: {bw_log_path}")
        return None

    text = bw_log_path.read_text(encoding="utf-8", errors="replace")

    legacy_match = re.search(r"legacy=[\d.]+ ms \(([\d.]+) GB/s\)", text)
    new_match = re.search(r"transposed=[\d.]+ ms \(([\d.]+) GB/s\)", text)

    if not legacy_match or not new_match:
        warnings.warn(
            f"Could not parse bandwidth values from log: {bw_log_path}\n"
            "Expected lines like: legacy=X.X ms (Y.Y GB/s) | transposed=..."
        )
        return None

    legacy_bw = float(legacy_match.group(1))
    new_bw = float(new_match.group(1))

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(
        ["Legacy\n[paths\u00d7time]", "Transposed\n[time\u00d7paths]"],
        [legacy_bw, new_bw],
        color=[METHOD_COLORS["GPU-MC"], METHOD_COLORS["GPU-MLMC"]],
    )
    ax.set_ylabel("Memory Bandwidth (GB/s)")
    ax.set_title("CUDA Memory Layout Bandwidth")
    for bar, val in zip(bars, [legacy_bw, new_bw]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max(legacy_bw, new_bw) * 1.25)
    fig.tight_layout()
    return _save_figure(fig, "memory_layout_bandwidth_a100.png")


# Candidate paths for the memory-layout benchmark log, in priority order.
# exp2_gpu_speedup.py writes this log when benchmark_memory_layout() is called.
_BW_LOG_CANDIDATES: List[Path] = [
    PROJECT_ROOT
    / "results"
    / "results"
    / "runpod_a100_extended"
    / "memory_layout_benchmark.log",
    PROJECT_ROOT
    / "results"
    / "results"
    / "runpod_a100_20260314_105715"
    / "memory_layout_benchmark.log",
    PROJECT_ROOT
    / "results"
    / "results"
    / "runpod_a100_20260314"
    / "memory_layout_benchmark.log",
    PROJECT_ROOT / "results" / "memory_layout_benchmark.log",
]


def _find_bw_log() -> Optional[Path]:
    """Return the first existing memory-layout benchmark log, or None."""
    for candidate in _BW_LOG_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


EXP1_JSON = (
    PROJECT_ROOT / "results" / "results" / "tables" / "exp1_mlmc_convergence_results.json"
)


def plot_variance_decay() -> Optional[Path]:
    """Semi-log plot of V_l vs MLMC level from exp1 convergence data.

    Reads the variance_decay block from exp1_mlmc_convergence_results.json and
    plots empirical V_l alongside the fitted decay V_l ~ C * 2^{-alpha * l}.
    This diagnostic confirms the theoretical MLMC variance decay assumption.
    """
    if not EXP1_JSON.exists():
        return None

    with open(EXP1_JSON) as f:
        data = json.load(f)

    vd = data.get("variance_decay", {})
    levels = vd.get("levels")
    variances = vd.get("variances")
    alpha = vd.get("alpha")

    if not levels or not variances:
        warnings.warn("variance_decay block missing levels/variances in exp1 JSON.")
        return None

    levels_arr = np.array(levels, dtype=float)
    var_arr = np.array(variances, dtype=float)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.semilogy(
        levels_arr,
        var_arr,
        "o-",
        color=METHOD_COLORS["GPU-MLMC"],
        linewidth=1.8,
        markersize=6,
        label=r"Empirical $V_l$",
    )

    if alpha is not None:
        C = float(var_arr[0])
        fit = C * (2.0 ** (-float(alpha) * levels_arr))
        ax.semilogy(
            levels_arr,
            fit,
            "--",
            color="gray",
            linewidth=1.2,
            label=rf"Fit: $V_l \propto 2^{{-{alpha:.2f}\,l}}$",
        )

    ax.set_xlabel("MLMC Level $l$")
    ax.set_ylabel(r"Sample Variance $V_l$")
    ax.set_title("MLMC Variance Decay Across Levels")
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, "mlmc_variance_decay_a100.png")


def main() -> None:
    results = load_results()
    seed_runs = load_seed_runs()
    stats_df = compute_multi_run_stats(seed_runs)
    scenario_order = _scenario_order(results)
    slopes_df = compute_slopes(results)

    print(f"Loaded {len(results)} reliable rows after merge/filter.")
    for scenario in scenario_order:
        subset = results[results["scenario"] == scenario].sort_values("epsilon")
        epsilons = ", ".join(f"{eps:g}" for eps in subset["epsilon"])
        print(f"  {_scenario_label(scenario)}: epsilons = [{epsilons}]")
    if stats_df is None:
        print("No multi-run seed CSV blocks found; generating single-run figures only.")
    else:
        print(
            f"Loaded {len(seed_runs)} seed-run rows across {int(stats_df['n_seed_runs'].max())} repeated runs."
        )

    created_paths: List[Optional[Path]] = [
        plot_loglog_cost_vs_epsilon(results),
        plot_runtime_vs_epsilon(results, stats_df=stats_df),
        plot_cost_ratio_vs_epsilon(results, stats_df=stats_df),
        plot_mlmc_level_allocation(results),
        plot_empirical_slope_bars(slopes_df, scenario_order),
        plot_runtime_scaling_tightest_epsilon(results),
    ]

    # Variance decay figure (optional — requires exp1 convergence results JSON)
    vd_path = plot_variance_decay()
    if vd_path is not None:
        created_paths.append(vd_path)
    else:
        print(
            "  exp1 convergence JSON not found — skipping mlmc_variance_decay_a100.png\n"
            f"  (expected at {EXP1_JSON.relative_to(PROJECT_ROOT)})"
        )

    # Memory-layout bandwidth figure (optional — requires benchmark log)
    bw_log = _find_bw_log()
    if bw_log is not None:
        print(f"  Memory bandwidth log found: {bw_log.relative_to(PROJECT_ROOT)}")
        bw_path = plot_memory_bandwidth_comparison(bw_log)
        created_paths.append(bw_path)
    else:
        print(
            "  Memory bandwidth log not found — skipping memory_layout_bandwidth_a100.png\n"
            "  (run exp2_gpu_speedup.py on the pod and save its output to one of:\n"
            + "\n".join(f"    {p}" for p in _BW_LOG_CANDIDATES[:2])
            + ")"
        )

    print("\nCreated figure files:")
    for path in created_paths:
        if path is not None:
            print(f"  {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
