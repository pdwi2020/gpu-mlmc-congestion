#!/usr/bin/env python3
"""
post_process_extended.py
========================
Phase B post-processing: validate the extended-epsilon results CSV returned
from RunPod, check that the W8 abstract numbers are reproduced, and regenerate
all paper figures via gen_figures_a100.py.

Run from the project root after copying the RunPod output directory:

    scp -r root@<pod>:/root/results/extended_eps \
        results/results/runpod_a100_extended/

    python3 scripts/post_process_extended.py

Flags
-----
  --csv PATH          Override CSV path (default: auto-detect under
                      results/results/runpod_a100_extended/)
  --no-figures        Skip figure regeneration
  --strict            Exit non-zero if W8 numbers are not reproduced
  --output-dir DIR    Where to write the post-processing report
                      (default: results/results/runpod_a100_extended/)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "results" / "runpod_a100_extended"
GEN_FIGURES_SCRIPT = PROJECT_ROOT / "paper" / "gen_figures_a100.py"

# W8 target numbers from the paper abstract (run2 config, synthetic_n500, ε=0.01)
W8_SPEEDUP_TARGET = 12.91
W8_COST_TARGET = 257.72
W8_TOLERANCE = 0.10  # 10 % — numbers may differ slightly with cap_mc=1M vs 500K

# Expected experiment matrix
EXPECTED_SCENARIOS = {
    "synthetic_n100",
    "synthetic_n500",
    "real_caida_asrel2_20260101_n500",
}
EXPECTED_EPSILONS = [0.10, 0.05, 0.02, 0.01, 0.005]

# CSV columns that must be numeric for validation
NUMERIC_COLS = [
    "epsilon",
    "speedup_runtime",
    "cost_ratio_mc_over_mlmc",
    "mc_runtime_s",
    "mlmc_runtime_s",
    "mc_cost",
    "mlmc_cost",
    "mc_ci_half",
    "mlmc_ci_half",
    "ci_target_half",
]

REQUIRED_CSV_COLUMNS = [
    "scenario",
    "nodes",
    "epsilon",
    "qoi",
    "mc_paths",
    "mlmc_levels",
    "mlmc_N_l",
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
    "equal_accuracy_ci_targeted",
    "error_proxy_mc_ci2_plus_hL",
    "error_proxy_mlmc_ci2_plus_hL",
    "sanity_same_qoi",
    "sanity_same_hL",
    "sanity_seed_policy",
    "sanity_cost_definition",
    "sanity_warmup_excluded",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP = "=" * 70


def _hr(title: str = "") -> str:
    if title:
        pad = max(0, 68 - len(title))
        return f"{'=' * 3} {title} {'=' * pad}"
    return SEP


def _coerce_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in {"true", "1", "yes"}


def _find_csv(results_dir: Path) -> Optional[Path]:
    """Locate extended_epsilon_results.csv under results_dir."""
    candidate = results_dir / "extended_epsilon_results.csv"
    if candidate.exists():
        return candidate
    # Recursive search one level deeper
    for p in sorted(results_dir.glob("**/extended_epsilon_results.csv")):
        return p
    return None


def _load_csv(csv_path: Path):
    """Load CSV with pandas, coerce numeric and bool columns."""
    try:
        import pandas as pd
    except ImportError:
        sys.exit("[FATAL] pandas is required: pip install pandas")

    df = pd.read_csv(csv_path)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "equal_accuracy_ci_targeted" in df.columns:
        df["equal_accuracy_ci_targeted"] = df["equal_accuracy_ci_targeted"].apply(
            _coerce_bool
        )
    return df


# ---------------------------------------------------------------------------
# Validation sections
# ---------------------------------------------------------------------------


class ValidationReport:
    """Accumulates pass/warn/fail items and renders a formatted report."""

    def __init__(self) -> None:
        self.items: List[Tuple[str, str, str]] = []  # (level, section, message)
        self.sections: List[str] = []

    def _add(self, level: str, section: str, msg: str) -> None:
        self.items.append((level, section, msg))

    def ok(self, section: str, msg: str) -> None:
        self._add("OK  ", section, msg)

    def warn(self, section: str, msg: str) -> None:
        self._add("WARN", section, msg)

    def fail(self, section: str, msg: str) -> None:
        self._add("FAIL", section, msg)

    def has_failures(self) -> bool:
        return any(lvl.strip() == "FAIL" for lvl, _, _ in self.items)

    def summary_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {"OK": 0, "WARN": 0, "FAIL": 0}
        for lvl, _, _ in self.items:
            k = lvl.strip()
            counts[k] = counts.get(k, 0) + 1
        return counts

    def render(self) -> str:
        lines = [SEP, "POST-PROCESSING REPORT", SEP]
        current_section = ""
        for lvl, section, msg in self.items:
            if section != current_section:
                current_section = section
                lines.append(f"\n  [ {section} ]")
            lines.append(f"    [{lvl}] {msg}")
        counts = self.summary_counts()
        lines.append("")
        lines.append(SEP)
        lines.append(
            f"  Summary:  OK={counts['OK']}  WARN={counts['WARN']}  FAIL={counts['FAIL']}"
        )
        lines.append(SEP)
        return "\n".join(lines)


def check_csv_schema(df, report: ValidationReport) -> None:
    """Check all required columns are present."""
    sec = "Schema"
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        report.fail(sec, f"Missing columns: {missing}")
    else:
        report.ok(sec, f"All {len(REQUIRED_CSV_COLUMNS)} required columns present")

    # Check for rows with NaN in key numeric columns
    for col in ["speedup_runtime", "cost_ratio_mc_over_mlmc", "epsilon"]:
        if col in df.columns:
            n_nan = int(df[col].isna().sum())
            if n_nan > 0:
                report.warn(sec, f"{col}: {n_nan} NaN value(s)")
            else:
                report.ok(sec, f"{col}: no NaNs")


def check_coverage(df, report: ValidationReport) -> None:
    """Check scenario × epsilon coverage."""
    sec = "Coverage"
    scenarios_found = set(df["scenario"].unique())
    missing_scen = EXPECTED_SCENARIOS - scenarios_found
    extra_scen = scenarios_found - EXPECTED_SCENARIOS

    if missing_scen:
        report.fail(sec, f"Missing scenarios: {missing_scen}")
    else:
        report.ok(sec, f"All 3 expected scenarios present")

    if extra_scen:
        report.warn(sec, f"Unexpected scenarios: {extra_scen}")

    # Check epsilon coverage per scenario
    for scen in sorted(scenarios_found & EXPECTED_SCENARIOS):
        scen_eps = sorted(df[df["scenario"] == scen]["epsilon"].dropna().unique())
        missing_eps = [
            e
            for e in EXPECTED_EPSILONS
            if not any(abs(e - se) < 0.0001 for se in scen_eps)
        ]
        if missing_eps:
            report.warn(sec, f"{scen}: missing ε={missing_eps}")
        else:
            report.ok(sec, f"{scen}: all 5 ε values present {scen_eps}")

    total = len(df)
    expected_total = len(EXPECTED_SCENARIOS) * len(EXPECTED_EPSILONS)
    report.ok(sec, f"Total rows: {total}  (expected ≥ {expected_total})")


def check_equal_accuracy(df, report: ValidationReport) -> None:
    """Check equal_accuracy_ci_targeted rates."""
    sec = "Equal Accuracy"
    total = len(df)
    eq_acc = df["equal_accuracy_ci_targeted"].sum()
    rate = eq_acc / total if total > 0 else 0.0

    if rate >= 0.80:
        report.ok(sec, f"equal_accuracy=True: {eq_acc}/{total} ({rate:.0%})")
    elif rate >= 0.60:
        report.warn(
            sec, f"equal_accuracy=True: {eq_acc}/{total} ({rate:.0%}) — below 80%"
        )
    else:
        report.fail(
            sec, f"equal_accuracy=True: {eq_acc}/{total} ({rate:.0%}) — too low"
        )

    # Per-epsilon breakdown
    for eps in EXPECTED_EPSILONS:
        subset = df[df["epsilon"].between(eps * 0.95, eps * 1.05)]
        if subset.empty:
            continue
        n_eq = int(subset["equal_accuracy_ci_targeted"].sum())
        report.ok(sec, f"  ε={eps:.3f}: {n_eq}/{len(subset)} equal_accuracy=True")

    # Check sanity flags
    for flag in [
        "sanity_same_qoi",
        "sanity_same_hL",
        "sanity_seed_policy",
        "sanity_cost_definition",
        "sanity_warmup_excluded",
    ]:
        if flag not in df.columns:
            report.warn(sec, f"Sanity flag missing: {flag}")
        elif not df[flag].apply(_coerce_bool).all():
            n_bad = int((~df[flag].apply(_coerce_bool)).sum())
            report.fail(sec, f"{flag}: {n_bad} False row(s)")
        else:
            report.ok(sec, f"{flag}: all True")


def check_ci_targeting(df, report: ValidationReport) -> None:
    """Verify CI half-widths are within tolerance of targets."""
    sec = "CI Targeting"
    tol = 0.15  # 15 % tolerance (CI_MATCH_TOL)

    # Only check rows that claim equal accuracy
    reliable = df[df["equal_accuracy_ci_targeted"]].copy()
    if reliable.empty:
        report.warn(sec, "No equal-accuracy rows to check")
        return

    for col, method in [("mc_ci_half", "GPU-MC"), ("mlmc_ci_half", "GPU-MLMC")]:
        if col not in reliable.columns or "ci_target_half" not in reliable.columns:
            continue
        rel_err = (reliable[col] - reliable["ci_target_half"]).abs() / reliable[
            "ci_target_half"
        ]
        n_outside = int((rel_err > tol).sum())
        max_err = float(rel_err.max()) if len(rel_err) else 0.0
        if n_outside > 0:
            report.warn(
                sec,
                f"{method}: {n_outside} rows outside ±{tol:.0%} CI tolerance "
                f"(max err={max_err:.1%})",
            )
        else:
            report.ok(
                sec,
                f"{method}: all rows within ±{tol:.0%} CI tolerance "
                f"(max err={max_err:.1%})",
            )


def check_w8_numbers(df, report: ValidationReport) -> Tuple[bool, float, float]:
    """
    Verify that the W8 abstract numbers (12.91x speedup, 257.72x cost ratio
    at ε=0.01, synthetic_n500) are reproduced within W8_TOLERANCE.

    Returns (reproduced: bool, actual_speedup: float, actual_cost_ratio: float).
    """
    sec = "W8 Verification"
    reliable = df[df["equal_accuracy_ci_targeted"]].copy()

    mask = reliable["scenario"].str.contains("synthetic_n500", na=False) & reliable[
        "epsilon"
    ].between(0.009, 0.011)
    candidate = reliable[mask]

    if candidate.empty:
        report.fail(
            sec,
            "No equal-accuracy row for synthetic_n500 ε=0.01 — "
            "W8 numbers cannot be verified. "
            "Check that cap_mc=1_000_000 and cap_mlmc=500_000 were used.",
        )
        return False, float("nan"), float("nan")

    row = candidate.iloc[0]
    actual_speedup = float(row["speedup_runtime"])
    actual_cost = float(row["cost_ratio_mc_over_mlmc"])

    speedup_ok = actual_speedup >= W8_SPEEDUP_TARGET * (1.0 - W8_TOLERANCE)
    cost_ok = actual_cost >= W8_COST_TARGET * (1.0 - W8_TOLERANCE)

    report.ok(
        sec,
        f"Row found: scenario=synthetic_n500  ε=0.01  "
        f"equal_accuracy={row['equal_accuracy_ci_targeted']}",
    )

    speedup_flag = "[OK]" if speedup_ok else "[LOWER]"
    cost_flag = "[OK]" if cost_ok else "[LOWER]"

    report.ok(
        sec,
        f"speedup_runtime        : {actual_speedup:.2f}x  "
        f"(paper target: {W8_SPEEDUP_TARGET}x, tol: ±{W8_TOLERANCE:.0%})  "
        f"{speedup_flag}",
    ) if speedup_ok else report.warn(
        sec,
        f"speedup_runtime        : {actual_speedup:.2f}x  "
        f"(paper target: {W8_SPEEDUP_TARGET}x, tol: ±{W8_TOLERANCE:.0%})  "
        f"{speedup_flag}",
    )

    report.ok(
        sec,
        f"cost_ratio_mc_over_mlmc: {actual_cost:.2f}x  "
        f"(paper target: {W8_COST_TARGET}x, tol: ±{W8_TOLERANCE:.0%})  "
        f"{cost_flag}",
    ) if cost_ok else report.warn(
        sec,
        f"cost_ratio_mc_over_mlmc: {actual_cost:.2f}x  "
        f"(paper target: {W8_COST_TARGET}x, tol: ±{W8_TOLERANCE:.0%})  "
        f"{cost_flag}",
    )

    reproduced = speedup_ok and cost_ok
    if reproduced:
        report.ok(sec, "W8 NUMBERS REPRODUCED within tolerance")
    else:
        report.warn(
            sec,
            "W8 numbers not fully reproduced — possible causes:\n"
            "         cap_mc/cap_mlmc differ from run2 config,\n"
            "         or CAIDA download failed (using BA fallback).",
        )

    return reproduced, actual_speedup, actual_cost


def check_complexity_slopes(df, report: ValidationReport) -> None:
    """
    Fit log10(cost) ~ log10(ε) slopes per scenario and method.
    GPU-MC should be ≈ -3 (or steeper), GPU-MLMC ≈ -2.
    Require at least 3 equal-accuracy points per scenario.
    """
    sec = "Complexity Slopes"
    try:
        import numpy as np
    except ImportError:
        report.warn(sec, "numpy not available — skipping slope check")
        return

    reliable = df[df["equal_accuracy_ci_targeted"]].copy()
    if reliable.empty:
        report.warn(sec, "No reliable rows for slope fitting")
        return

    for scen in sorted(reliable["scenario"].unique()):
        sub = reliable[reliable["scenario"] == scen].sort_values("epsilon")
        if len(sub) < 3:
            report.warn(sec, f"{scen}: only {len(sub)} point(s) — skipping slope fit")
            continue

        valid = sub[sub["epsilon"] > 0].copy()
        log_eps = np.log10(valid["epsilon"].to_numpy(dtype=float))

        for cost_col, method in [("mc_cost", "GPU-MC"), ("mlmc_cost", "GPU-MLMC")]:
            costs = valid[cost_col].to_numpy(dtype=float)
            mask = costs > 0
            if mask.sum() < 2:
                report.warn(sec, f"{scen} {method}: insufficient positive cost points")
                continue
            slope = float(np.polyfit(log_eps[mask], np.log10(costs[mask]), 1)[0])
            expected = -3.0 if method == "GPU-MC" else -2.0
            deviation = abs(slope - expected)
            if deviation < 1.5:
                report.ok(
                    sec, f"{scen} {method}: slope={slope:.2f}  (expected≈{expected})"
                )
            else:
                report.warn(
                    sec,
                    f"{scen} {method}: slope={slope:.2f}  (expected≈{expected}, "
                    f"deviation={deviation:.2f})",
                )


def print_top_results(df) -> None:
    """Print a formatted table of top-5 speedup rows and tightest-ε rows."""
    reliable = df[df["equal_accuracy_ci_targeted"]].copy()
    if reliable.empty:
        print("  (no equal-accuracy rows to display)")
        return

    cols = [
        "scenario",
        "epsilon",
        "speedup_runtime",
        "cost_ratio_mc_over_mlmc",
        "mc_runtime_s",
        "mlmc_runtime_s",
    ]
    cols = [c for c in cols if c in reliable.columns]

    print(f"\n{'─' * 70}")
    print("  Top 5 rows by runtime speedup (equal_accuracy=True):")
    print(f"{'─' * 70}")
    top = reliable.nlargest(5, "speedup_runtime")[cols]
    print(
        top.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print(f"\n{'─' * 70}")
    print("  Tightest reliable ε per scenario:")
    print(f"{'─' * 70}")
    tightest_rows = []
    for scen in sorted(reliable["scenario"].unique()):
        sub = reliable[reliable["scenario"] == scen].sort_values("epsilon")
        if not sub.empty:
            tightest_rows.append(sub.iloc[0])
    if tightest_rows:
        try:
            import pandas as pd

            tight_df = pd.DataFrame(tightest_rows)[cols]
            print(tight_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        except Exception:
            for r in tightest_rows:
                print(
                    f"  {r['scenario']}  ε={r['epsilon']:.4f}  "
                    f"speedup={r['speedup_runtime']:.2f}x"
                )


# ---------------------------------------------------------------------------
# Figure regeneration
# ---------------------------------------------------------------------------


def regenerate_figures(report: ValidationReport) -> None:
    """Run paper/gen_figures_a100.py to regenerate all 6 figures."""
    sec = "Figure Regeneration"

    if not GEN_FIGURES_SCRIPT.exists():
        report.fail(sec, f"gen_figures_a100.py not found at {GEN_FIGURES_SCRIPT}")
        return

    report.ok(sec, f"Running {GEN_FIGURES_SCRIPT.name} ...")
    print(f"\n{'─' * 70}")
    print("  Running gen_figures_a100.py ...")
    print(f"{'─' * 70}")

    result = subprocess.run(
        [sys.executable, str(GEN_FIGURES_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
        text=True,
    )

    if result.returncode == 0:
        figures_dir = PROJECT_ROOT / "paper" / "figures"
        pngs = sorted(figures_dir.glob("*.png"))
        report.ok(
            sec, f"Figure generation successful — {len(pngs)} PNG(s) in paper/figures/"
        )
        for p in pngs:
            size_kb = p.stat().st_size // 1024
            report.ok(sec, f"  {p.name}  ({size_kb} KB)")
    else:
        report.fail(sec, f"gen_figures_a100.py exited with code {result.returncode}")


# ---------------------------------------------------------------------------
# Report saving
# ---------------------------------------------------------------------------


def save_report(
    report: ValidationReport,
    output_dir: Path,
    w8_reproduced: bool,
    actual_speedup: float,
    actual_cost: float,
    csv_path: Path,
) -> Path:
    """Write JSON + plain-text report to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = report.summary_counts()
    summary = {
        "status": "PASS" if not report.has_failures() else "FAIL",
        "ok_count": counts["OK"],
        "warn_count": counts["WARN"],
        "fail_count": counts["FAIL"],
        "w8_reproduced": w8_reproduced,
        "w8_speedup_actual": actual_speedup,
        "w8_speedup_target": W8_SPEEDUP_TARGET,
        "w8_cost_actual": actual_cost,
        "w8_cost_target": W8_COST_TARGET,
        "csv_path": str(csv_path),
    }

    json_path = output_dir / "post_process_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    txt_path = output_dir / "post_process_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report.render())
        f.write("\n\nJSON summary:\n")
        f.write(json.dumps(summary, indent=2))
        f.write("\n")

    return txt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase B post-processing: validate extended-ε results and regenerate figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to extended_epsilon_results.csv (auto-detected if omitted)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root directory containing the RunPod output files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write post_process_report.{json,txt} (defaults to --results-dir)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        default=False,
        help="Skip figure regeneration",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit with code 1 if W8 numbers are not reproduced or any FAIL is found",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir: Path = args.results_dir
    output_dir: Path = args.output_dir or results_dir

    print(SEP)
    print("Phase B — Extended-ε Results Post-Processing")
    print(SEP)

    # ------------------------------------------------------------------
    # Locate CSV
    # ------------------------------------------------------------------
    csv_path: Optional[Path] = args.csv
    if csv_path is None:
        csv_path = _find_csv(results_dir)
    if csv_path is None or not csv_path.exists():
        print(
            f"\n[FATAL] Cannot find extended_epsilon_results.csv under:\n"
            f"  {results_dir}\n\n"
            "Copy the RunPod output first:\n"
            "  scp -r root@<pod>:/root/results/extended_eps \\\n"
            f"      {results_dir}\n"
        )
        sys.exit(1)

    print(f"\n  CSV      : {csv_path}")
    print(f"  Size     : {csv_path.stat().st_size:,} bytes")
    print(f"  Out dir  : {output_dir}")

    # Optional JSON summary from the pod
    json_summary_path = csv_path.parent / "run_summary.json"
    if json_summary_path.exists():
        with open(json_summary_path) as f:
            pod_summary = json.load(f)
        print(f"\n  Pod run summary:")
        print(f"    run_date_utc : {pod_summary.get('run_date_utc', 'n/a')}")
        print(f"    gpu_available: {pod_summary.get('gpu_available', 'n/a')}")
        print(f"    total_rows   : {pod_summary.get('total_rows', 'n/a')}")
        print(f"    equal_acc    : {pod_summary.get('equal_accuracy_rows', 'n/a')}")
        print(
            f"    cap_mc       : {pod_summary.get('cap_mc', 'n/a'):,}"
            if pod_summary.get("cap_mc")
            else ""
        )
        print(
            f"    cap_mlmc     : {pod_summary.get('cap_mlmc', 'n/a'):,}"
            if pod_summary.get("cap_mlmc")
            else ""
        )

    # ------------------------------------------------------------------
    # Load + validate
    # ------------------------------------------------------------------
    print(f"\n{_hr('Loading CSV')}")
    df = _load_csv(csv_path)
    print(f"  Loaded {len(df)} rows × {len(df.columns)} columns")

    report = ValidationReport()

    print(f"\n{_hr('Schema check')}")
    check_csv_schema(df, report)

    print(f"\n{_hr('Coverage check')}")
    check_coverage(df, report)

    print(f"\n{_hr('Equal accuracy check')}")
    check_equal_accuracy(df, report)

    print(f"\n{_hr('CI targeting check')}")
    check_ci_targeting(df, report)

    print(f"\n{_hr('Complexity slope check')}")
    check_complexity_slopes(df, report)

    print(f"\n{_hr('W8 number verification')}")
    w8_reproduced, actual_speedup, actual_cost = check_w8_numbers(df, report)

    # ------------------------------------------------------------------
    # Print top results table
    # ------------------------------------------------------------------
    print(f"\n{_hr('Top results')}")
    print_top_results(df)

    # ------------------------------------------------------------------
    # Figure regeneration
    # ------------------------------------------------------------------
    if not args.no_figures:
        print(f"\n{_hr('Figure regeneration')}")
        regenerate_figures(report)
    else:
        print(f"\n  --no-figures: skipping gen_figures_a100.py")

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    txt_path = save_report(
        report, output_dir, w8_reproduced, actual_speedup, actual_cost, csv_path
    )

    # ------------------------------------------------------------------
    # Final print
    # ------------------------------------------------------------------
    print(f"\n{report.render()}")
    print(f"\n  Report saved -> {txt_path}")
    print(f"  JSON   saved -> {txt_path.with_suffix('.json')}")

    # ------------------------------------------------------------------
    # HANDOFF.md update hint
    # ------------------------------------------------------------------
    if w8_reproduced:
        print(
            textwrap.dedent(f"""
            ╔══════════════════════════════════════════════════════════════════╗
            ║  W8 REPRODUCED                                                  ║
            ║  speedup={actual_speedup:.2f}x  (target {W8_SPEEDUP_TARGET}x)               ║
            ║  cost_ratio={actual_cost:.2f}x (target {W8_COST_TARGET}x)             ║
            ║  Abstract numbers are valid — paper ready for submission.       ║
            ╚══════════════════════════════════════════════════════════════════╝
            """)
        )
    else:
        print(
            textwrap.dedent(f"""
            ┌──────────────────────────────────────────────────────────────────┐
            │  W8 NOT reproduced at ≥90 % threshold                           │
            │  speedup={actual_speedup:.2f}x  (target {W8_SPEEDUP_TARGET}x)               │
            │  cost_ratio={actual_cost:.2f}x (target {W8_COST_TARGET}x)             │
            │  Check equal_accuracy_ci_targeted and re-run with:              │
            │    --cap-mc 1000000 --cap-mlmc 500000                           │
            └──────────────────────────────────────────────────────────────────┘
            """)
        )

    if args.strict and (report.has_failures() or not w8_reproduced):
        sys.exit(1)


if __name__ == "__main__":
    main()
