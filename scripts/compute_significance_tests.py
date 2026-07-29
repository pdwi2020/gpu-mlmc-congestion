"""
Statistical-significance harness for the IEEE Access resubmission (Reviewer 2:
"put a confidence interval on every reported improvement").

Ingests per-seed raw values that OTHER experiment scripts already wrote under
results/, and for every claim it can support from raw data emits:
  * mean / SD / n
  * a bias-corrected-and-accelerated (BCa) bootstrap 95% CI (see src/utils/stats.py
    for why BCa rather than naive percentile)
  * a paired t-test AND a Wilcoxon signed-rank test, seed-paired
  * an effect size: Cohen's d (paired) and the matched-pairs rank-biserial r
  * an explicit `insufficient_n` flag when n is too small to be a citable claim
    (project bar: n>=10 seeds, RESUBMISSION_TASKS.md T2.8)

Ratio-of-means claims (speedup, work/cost reduction) get a CI on the RATIO,
bootstrapped jointly across seed-paired (numerator, denominator) values -- see
`ratio_ci_bootstrap` in src/utils/stats.py for why this is not the same as
dividing two independently-bootstrapped means' CIs.

Some existing result files under results/ contain only PRE-AGGREGATED summary
statistics (mean/std/n_seeds) with no raw per-seed array retained. Those are
recorded as `kind: "aggregate_only"` claims: the harness reports what the
source file itself claims (with `insufficient_n` still evaluated against its
stated n) and explicitly does not attempt to bootstrap a CI it has no raw data
to bootstrap from.

Output: one consolidated results/significance/all_claims.json, keyed by a
stable claim id, plus results/significance/all_claims.csv (one row per claim)
and a `registry` section mapping claim id -> description -> source file(s).
A legacy results/significance/significance_tests.json is also written,
preserving the schema of the pre-extension version of this script (nothing in
this repository currently reads it, but keeping the format costs nothing and
avoids surprising anyone who does).

Usage:
    python scripts/compute_significance_tests.py
    python scripts/compute_significance_tests.py --quick
    python scripts/compute_significance_tests.py --seeds 0 1 2 --n-resamples 20000

Output (under --out, default results/significance):
    all_claims.json       full results + registry + provenance
    all_claims.csv         one row per claim, flattened for quick scanning
    significance_tests.json  legacy schema, preserved for continuity
"""

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.stats import (  # noqa: E402
    MIN_N_HARD,
    MIN_N_RECOMMENDED,
    bca_bootstrap_ci,
    cohens_d_paired,
    flag_insufficient_n,
    matched_pairs_rank_biserial,
    mean_sd_n,
    paired_ttest,
    ratio_ci_bootstrap,
    wilcoxon_signed_rank,
)

try:
    from scipy import stats as sp_stats  # noqa: F401  (availability probe only)
    _SCIPY = True
except ImportError:
    _SCIPY = False
    print("WARNING: scipy not available -- BCa bootstrap and Wilcoxon are unavailable; "
          "claims will be recorded with a skip reason instead of crashing.")

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')


# ===========================================================================
# Provenance (same conventions as scripts/run_adaptive_stepping_ablation.py)
# ===========================================================================
def git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(
            ["git", "-C", REPO_ROOT, "diff", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def select_device_name(requested: str) -> str:
    """Resolve --device to a human-readable label. This script's own
    computation (numpy/scipy bootstrap) is CPU-only regardless of --device;
    the flag exists for interface consistency with run_profiling.py and is
    recorded in provenance, not used to dispatch any tensor op here."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda (available; unused -- this script is CPU-only)"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps (available; unused -- this script is CPU-only)"
    except ImportError:
        pass
    return "cpu"


def build_provenance(device_label: str, config: dict) -> dict:
    torch_version = None
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        pass
    try:
        import scipy
        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = None
    return {
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": device_label,
        "compute_device": "cpu (numpy/scipy bootstrap; --device is informational only)",
        "torch_version": torch_version,
        "scipy_version": scipy_version,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "config": config,
    }


# ===========================================================================
# JSON sanitisation: guarantee strictly-valid JSON (no bare NaN/Infinity)
# ===========================================================================
def _sanitize(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


# ===========================================================================
# Claim registry + analysis wrappers
# ===========================================================================
class Harness:
    """Accumulates claims and the source-file registry; owns the RNG seed and
    resample count so every claim in a run is reproducible end to end."""

    def __init__(self, n_resamples: int, rng_seed: int, min_n: int,
                 seeds_filter, results_root: str):
        self.n_resamples = n_resamples
        self.rng_seed = rng_seed
        self.min_n = min_n
        self.seeds_filter = set(seeds_filter) if seeds_filter else None
        self.results_root = results_root
        self.claims = {}
        self.registry = {}
        self.skips = []
        self._next_seed_offset = 0

    def _rng_seed_for(self, claim_id: str) -> int:
        # Deterministic per-claim seed derived from the run seed and claim id,
        # so re-running with the same --rng-seed reproduces every CI exactly,
        # and no two claims share a bootstrap stream by accident.
        self._next_seed_offset += 1
        return (hash((self.rng_seed, claim_id)) % (2 ** 31)) ^ self._next_seed_offset

    def restrict_seeds(self, seed_to_value: dict) -> dict:
        if self.seeds_filter is None:
            return seed_to_value
        return {s: v for s, v in seed_to_value.items() if s in self.seeds_filter}

    def skip(self, source: str, reason: str):
        self.skips.append({"source": source, "reason": reason})
        print(f"  [skip] {source}: {reason}")

    def _register(self, claim_id: str, description: str, source, kind: str):
        if claim_id in self.claims:
            raise ValueError(f"duplicate claim id {claim_id!r}")
        self.registry[claim_id] = {
            "description": description,
            "source": source if isinstance(source, list) else [source],
            "kind": kind,
        }

    # ---- single-sample claim: one raw array, e.g. "runtime at n=500" -------
    def add_single_sample(self, claim_id: str, description: str, source,
                           values, unit: str = ""):
        self._register(claim_id, description, source, "single_sample")
        desc = mean_sd_n(values)
        ci = bca_bootstrap_ci(values, n_resamples=self.n_resamples,
                               seed=self._rng_seed_for(claim_id))
        entry = {
            "claim_id": claim_id, "description": description, "source": source,
            "kind": "single_sample", "unit": unit,
            "n": desc["n"], "mean": desc["mean"], "sd": desc["sd"],
            "min": desc["min"], "max": desc["max"],
            "bca_ci_95": ci,
            "insufficient_n": flag_insufficient_n(desc["n"], self.min_n),
        }
        self.claims[claim_id] = entry
        return entry

    # ---- paired-difference claim: two raw arrays, same seeds ---------------
    def add_paired_difference(self, claim_id: str, description: str, source,
                               a, b, label_a: str, label_b: str, unit: str = ""):
        self._register(claim_id, description, source, "paired_difference")
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        n = int(a.size)
        diff_ci = bca_bootstrap_ci(a - b, n_resamples=self.n_resamples,
                                    seed=self._rng_seed_for(claim_id))
        entry = {
            "claim_id": claim_id, "description": description, "source": source,
            "kind": "paired_difference", "unit": unit,
            "label_a": label_a, "label_b": label_b, "n": n,
            "mean_a": float(np.mean(a)) if n else float("nan"),
            "mean_b": float(np.mean(b)) if n else float("nan"),
            "mean_diff_a_minus_b": float(np.mean(a - b)) if n else float("nan"),
            "diff_bca_ci_95": diff_ci,
            "paired_ttest": paired_ttest(a, b),
            "wilcoxon_signed_rank": wilcoxon_signed_rank(a, b),
            "cohens_d_paired": cohens_d_paired(a, b),
            "rank_biserial_r": matched_pairs_rank_biserial(a, b),
            "insufficient_n": flag_insufficient_n(n, self.min_n),
        }
        self.claims[claim_id] = entry
        return entry

    # ---- paired-ratio claim: speedup / work-reduction, CI on the RATIO -----
    def add_paired_ratio(self, claim_id: str, description: str, source,
                          numerator, denominator, numerator_label: str,
                          denominator_label: str, higher_is_better: bool = True):
        self._register(claim_id, description, source, "paired_ratio")
        numerator = np.asarray(numerator, dtype=float)
        denominator = np.asarray(denominator, dtype=float)
        n = int(numerator.size)
        ratio_ci = ratio_ci_bootstrap(numerator, denominator,
                                       n_resamples=self.n_resamples,
                                       seed=self._rng_seed_for(claim_id))
        entry = {
            "claim_id": claim_id, "description": description, "source": source,
            "kind": "paired_ratio",
            "numerator_label": numerator_label, "denominator_label": denominator_label,
            "higher_is_better": higher_is_better, "n": n,
            "mean_numerator": float(np.mean(numerator)) if n else float("nan"),
            "mean_denominator": float(np.mean(denominator)) if n else float("nan"),
            "ratio_bca_ci_95": ratio_ci,
            # Diagnostics on the raw (unratioed) values -- useful context, not
            # the headline number, since numerator and denominator are
            # different quantities (e.g. seconds vs seconds is fine, but a
            # cost-vs-runtime ratio would make this section not meaningful;
            # each call site documents which case it is in its `description`).
            "paired_ttest_raw_values": paired_ttest(numerator, denominator),
            "wilcoxon_raw_values": wilcoxon_signed_rank(numerator, denominator),
            "cohens_d_paired_raw_values": cohens_d_paired(numerator, denominator),
            "insufficient_n": flag_insufficient_n(n, self.min_n),
        }
        self.claims[claim_id] = entry
        return entry

    # ---- known-gap claim: a headline paper number with NO backing per-seed
    #      file at all (worse than aggregate_only: there is nothing to ingest,
    #      only the number hardcoded in the manuscript's LaTeX source). This is
    #      the class of claim that most urgently determines the A100 re-run
    #      list, so it gets its own explicit kind rather than being silently
    #      dropped because no extractor matched a file.
    def add_known_gap(self, claim_id: str, description: str, paper_location: str,
                       reported_value, n_available: int, gap_reason: str,
                       required_action: str):
        self._register(claim_id, description, paper_location, "known_gap")
        entry = {
            "claim_id": claim_id, "description": description, "kind": "known_gap",
            "source": [paper_location],
            "reported_value": reported_value, "n": n_available,
            "gap_reason": gap_reason, "required_action": required_action,
            "insufficient_n": True,
        }
        self.claims[claim_id] = entry
        return entry

    # ---- aggregate-only claim: source file has no raw per-seed array -------
    def add_aggregate_only(self, claim_id: str, description: str, source,
                            mean, n, sd=None, reported_ci=None, extra=None):
        self._register(claim_id, description, source, "aggregate_only")
        entry = {
            "claim_id": claim_id, "description": description, "source": source,
            "kind": "aggregate_only", "n": int(n) if n is not None else None,
            "mean": mean, "sd": sd,
            "reported_ci": reported_ci,
            "note": ("source file provides only pre-aggregated summary statistics; "
                     "no raw per-seed array is retained, so a BCa bootstrap CI "
                     "cannot be (re)computed here -- this is exactly the class of "
                     "claim that needs a raw-data re-run"),
            "insufficient_n": flag_insufficient_n(n, self.min_n) if n is not None else True,
        }
        if extra:
            entry.update(extra)
        self.claims[claim_id] = entry
        return entry


# ===========================================================================
# I/O helpers
# ===========================================================================
def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_csv_rows(path, comment_prefix="#"):
    with open(path, newline="") as f:
        lines = [ln for ln in f if not ln.startswith(comment_prefix)]
    return list(csv.DictReader(lines))


def relpath(path):
    return os.path.relpath(path, REPO_ROOT)


# ===========================================================================
# Extractors -- one per existing result-file family under results/
# ===========================================================================
def extract_ana_mlmc(h: Harness):
    """results/ana_mlmc/ana_mlmc_results.csv: ANA-MLMC vs Giles-baseline
    runtime, paired by seed within each (network, n_nodes, epsilon) cell."""
    path = os.path.join(h.results_root, "ana_mlmc", "ana_mlmc_results.csv")
    if not os.path.exists(path):
        h.skip(relpath(path), "file not found")
        return
    cells = defaultdict(dict)  # (network, n_nodes, epsilon) -> {seed: {method: runtime}}
    for row in load_csv_rows(path):
        key = (row["network"], row["n_nodes"], row["epsilon"])
        seed = int(row["seed"])
        if h.seeds_filter is not None and seed not in h.seeds_filter:
            continue
        cells[key].setdefault(seed, {})[row["method"]] = float(row["runtime_s"])

    all_baseline, all_ana = [], []
    for (network, n_nodes, epsilon), by_seed in sorted(cells.items()):
        seeds = sorted(s for s, m in by_seed.items() if {"baseline", "ana"} <= m.keys())
        if not seeds:
            continue
        baseline = [by_seed[s]["baseline"] for s in seeds]
        ana = [by_seed[s]["ana"] for s in seeds]
        all_baseline.extend(baseline)
        all_ana.extend(ana)
        claim_id = f"ana_vs_baseline_speedup__{network}_n{n_nodes}_eps{epsilon}"
        h.add_paired_ratio(
            claim_id,
            f"ANA-MLMC vs Giles-baseline runtime speedup, {network.upper()} "
            f"n={n_nodes}, eps={epsilon} (ratio=baseline/ANA runtime, seed-paired, "
            f"n={len(seeds)} seeds {seeds})",
            relpath(path), baseline, ana, "baseline_runtime_s", "ana_runtime_s")

    if all_baseline:
        # Legacy-equivalent pooled claim (matches the pre-extension script's
        # single "ana_vs_baseline_runtime" test, which pooled every
        # network/n_nodes/epsilon cell together). Kept for continuity and for
        # the reproducibility cross-check against the old significance_tests.json.
        h.add_paired_ratio(
            "ana_vs_baseline_speedup__pooled_all_conditions",
            "ANA-MLMC vs Giles-baseline runtime speedup, POOLED across every "
            "network/n_nodes/epsilon cell (heterogeneous conditions -- kept only "
            "for continuity with the pre-extension significance_tests.json; the "
            "per-condition claims above are the statistically appropriate ones)",
            relpath(path), all_baseline, all_ana,
            "baseline_runtime_s", "ana_runtime_s")


def extract_scaling_results(h: Harness):
    """results/scaling_results.json: MLMC-only wall-clock runtimes per n_nodes
    (no baseline pairing available in this file -- single-sample claims)."""
    path = os.path.join(h.results_root, "scaling_results.json")
    if not os.path.exists(path):
        h.skip(relpath(path), "file not found")
        return
    d = load_json(path)
    for n_str, v in sorted(d.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
        if not isinstance(v, dict) or "runtimes" not in v:
            continue
        runtimes = v["runtimes"]
        if not runtimes:
            continue
        h.add_single_sample(
            f"mlmc_runtime__n{n_str}", f"GPU-MLMC wall-clock runtime, n={n_str} nodes",
            relpath(path), runtimes, unit="seconds")


def extract_ana_weight_sweep(h: Harness):
    """results/ana_weight_sweep/weight_sweep.csv: raw per-(config, seed) rows.
    Ratio claims: each weighting scheme's total_cost and runtime_s vs the
    baseline_giles config, paired by seed. Flagged: only 3 seeds are present
    (RESUBMISSION_TASKS.md T2.2 already calls this data "too thin to cite")."""
    path = os.path.join(h.results_root, "ana_weight_sweep", "weight_sweep.csv")
    if not os.path.exists(path):
        h.skip(relpath(path), "file not found")
        return
    by_config = defaultdict(dict)  # config -> {seed: {cost, runtime}}
    for row in load_csv_rows(path):
        seed = int(row["seed"])
        if h.seeds_filter is not None and seed not in h.seeds_filter:
            continue
        by_config[row["config"]][seed] = {
            "total_cost": float(row["total_cost"]),
            "runtime_s": float(row["runtime_s"]),
        }
    if "baseline_giles" not in by_config:
        h.skip(relpath(path), "no baseline_giles rows found")
        return
    baseline = by_config["baseline_giles"]
    for config, by_seed in sorted(by_config.items()):
        if config == "baseline_giles":
            continue
        seeds = sorted(set(by_seed) & set(baseline))
        if not seeds:
            continue
        base_cost = [baseline[s]["total_cost"] for s in seeds]
        cfg_cost = [by_seed[s]["total_cost"] for s in seeds]
        base_rt = [baseline[s]["runtime_s"] for s in seeds]
        cfg_rt = [by_seed[s]["runtime_s"] for s in seeds]
        h.add_paired_ratio(
            f"ana_weight_sweep_cost_reduction__{config}",
            f"ANA weight config '{config}' vs baseline_giles: total-cost reduction "
            f"factor (ratio=baseline_cost/config_cost, seed-paired, n={len(seeds)})",
            relpath(path), base_cost, cfg_cost, "baseline_giles_total_cost", "config_total_cost")
        h.add_paired_ratio(
            f"ana_weight_sweep_runtime_speedup__{config}",
            f"ANA weight config '{config}' vs baseline_giles: runtime speedup "
            f"(ratio=baseline_runtime/config_runtime, seed-paired, n={len(seeds)})",
            relpath(path), base_rt, cfg_rt, "baseline_giles_runtime_s", "config_runtime_s")


def extract_per_topology(h: Harness):
    """results/per_topology/per_topology_speedup.json: aggregate-only (mean +
    reported CI95 + n_seeds), no raw per-seed array retained in the file."""
    path = os.path.join(h.results_root, "per_topology", "per_topology_speedup.json")
    if not os.path.exists(path):
        h.skip(relpath(path), "file not found")
        return
    d = load_json(path)
    for topology, v in sorted(d.items()):
        if topology == "summary" or not isinstance(v, dict):
            continue
        n_seeds = v.get("n_seeds")
        h.add_aggregate_only(
            f"per_topology_runtime_speedup__{topology}",
            f"GPU-MC vs GPU-MLMC runtime speedup, topology={topology}, "
            f"n_nodes={v.get('n_nodes')}, eps={v.get('epsilon')} "
            f"(as reported by source; not recomputed)",
            relpath(path), mean=v.get("metric1_runtime_speedup_mean"), n=n_seeds,
            reported_ci={"half_width_95": v.get("metric1_runtime_speedup_ci95")},
            extra={"geomean_reported": v.get("metric1_geomean")})
        h.add_aggregate_only(
            f"per_topology_work_reduction__{topology}",
            f"GPU-MLMC work reduction factor, topology={topology}, "
            f"n_nodes={v.get('n_nodes')}, eps={v.get('epsilon')} "
            f"(as reported by source; not recomputed)",
            relpath(path), mean=v.get("metric2_work_reduction_mean"), n=n_seeds,
            reported_ci={"half_width_95": v.get("metric2_work_reduction_ci95")})


def extract_halo_sweep(h: Harness):
    """results/halo_sweep/halo_k_sweep.json: aggregate-only (mean/std/n_seeds
    per halo depth K), no raw per-seed array retained in the file."""
    path = os.path.join(h.results_root, "halo_sweep", "halo_k_sweep.json")
    if not os.path.exists(path):
        h.skip(relpath(path), "file not found")
        return
    rows = load_json(path)
    if not isinstance(rows, list):
        h.skip(relpath(path), "unexpected schema (expected a list of per-K rows)")
        return
    for row in rows:
        k = row.get("K")
        h.add_aggregate_only(
            f"halo_sweep_elapsed__K{k}",
            f"Multi-GPU halo-exchange wall-clock, K={k} (halo depth), "
            f"world_size={row.get('world_size')}, eps={row.get('epsilon')} "
            f"(as reported by source; not recomputed)",
            relpath(path), mean=row.get("elapsed_mean_s"), sd=row.get("elapsed_std_s"),
            n=row.get("n_seeds"),
            extra={"phases_mean_reported": row.get("phases_mean"),
                   "runtime_reduction_pct_reported": row.get("runtime_reduction_pct")})


def extract_multi_gpu(h: Harness):
    """results/multi_gpu/*.json: three files with three different shapes.

    * scaling_results.json           -- has raw `times` repeats per world_size
                                         (local CPU/MPS smoke measurements)
    * distributed_scaling_results.json -- single real-hardware (4xRTX3090) run
                                         per world_size, n=1, no repeats
    * memory_scaling.json            -- measured-vs-analytical-model memory per
                                         (n, G); single measurement, n=1
    """
    base = os.path.join(h.results_root, "multi_gpu")

    p1 = os.path.join(base, "scaling_results.json")
    if os.path.exists(p1):
        d = load_json(p1)
        for scaling_type in ("weak_scaling", "strong_scaling"):
            for row in d.get(scaling_type, []):
                times = row.get("times")
                if not times:
                    continue
                h.add_single_sample(
                    f"multi_gpu_{scaling_type}_time__ws{row['world_size']}_n{row.get('n_nodes')}",
                    f"Multi-GPU {scaling_type.replace('_', ' ')} wall-clock, "
                    f"world_size={row['world_size']}, n_nodes={row.get('n_nodes')} "
                    f"(local CPU/MPS smoke run, repeats={len(times)})",
                    relpath(p1), times, unit="seconds")
    else:
        h.skip(relpath(p1), "file not found")

    p2 = os.path.join(base, "distributed_scaling_results.json")
    if os.path.exists(p2):
        d = load_json(p2)
        hardware = d.get("hardware", "unknown hardware")
        for section in ("strong_scaling", "weak_scaling"):
            sect = d.get(section, {})
            for row in sect.get("results", []):
                ws = row.get("world_size")
                time_s = row.get("time_s")
                if time_s is None:
                    continue
                h.add_aggregate_only(
                    f"multi_gpu_distributed_{section}_time__ws{ws}",
                    f"Multi-GPU {section.replace('_', ' ')} wall-clock, world_size={ws}, "
                    f"real hardware ({hardware}); n=1, no repeated measurement in source",
                    relpath(p2), mean=time_s, n=1)
    else:
        h.skip(relpath(p2), "file not found")

    p3 = os.path.join(base, "memory_scaling.json")
    if os.path.exists(p3):
        d = load_json(p3)
        for row in d.get("rows", []):
            n, g = row.get("n"), row.get("G")
            measured = row.get("measured_mem_mb")
            model = row.get("model_mem_mb")
            ratio = (measured / model) if (measured is not None and model) else None
            h.add_aggregate_only(
                f"multi_gpu_memory_measured_vs_model__n{n}_G{g}",
                f"Multi-GPU peak memory, n={n} nodes, G={g} ranks: MEASURED vs the "
                f"paper's analytical model n*(n/G)*4 bytes/rank (adjacency tensor "
                f"only); single measurement, n=1",
                relpath(p3), mean=measured, n=1,
                extra={"unit": "MB", "analytical_model_mb": model,
                       "measured_over_model_ratio": ratio,
                       "partitioner": row.get("partitioner"),
                       "halo_pct": row.get("halo_pct")})
    else:
        h.skip(relpath(p3), "file not found")


def extract_adaptive_stepping_full(h: Harness):
    """results/adaptive_stepping_full/: the S1b.5 decision-gate evidence.

    mlmc_cost_to_epsilon.csv -- ratio claims: uniform-scheme cost / adaptive-
    scheme cost at each regime's tightest common attainable epsilon, paired by
    seed (this is the exact "cost reduction under Giles's optimal allocation"
    quantity Theorem 1 is about).

    mlmc_exponents.csv -- paired-difference claims on beta_var (the MLMC
    level-difference-variance decay exponent): this is the number the S1b.5
    demotion decision in RESUBMISSION_TASKS.md was made from ("beta_var
    unchanged vs uniform across all 4 regimes"); this harness gives that
    decision a real CI instead of an eyeballed delta.
    """
    base = os.path.join(h.results_root, "adaptive_stepping_full")

    p_cost = os.path.join(base, "mlmc_cost_to_epsilon.csv")
    if os.path.exists(p_cost):
        rows = load_csv_rows(p_cost)
        # (regime, scheme) -> {seed: {epsilon: (total_cost, attainable)}}
        by_rs = defaultdict(lambda: defaultdict(dict))
        for row in rows:
            seed = int(row["seed"])
            if h.seeds_filter is not None and seed not in h.seeds_filter:
                continue
            attainable = row["attainable"] == "True"
            cost = float(row["total_cost"]) if row["total_cost"] not in ("", None) else None
            by_rs[(row["regime"], row["scheme"])][seed][float(row["epsilon"])] = (cost, attainable)

        regimes = sorted({r for r, s in by_rs if s == "uniform"})
        for regime in regimes:
            uniform = by_rs[(regime, "uniform")]
            schemes = sorted({s for (r, s) in by_rs if r == regime and s != "uniform"})
            for scheme in schemes:
                adaptive = by_rs[(regime, scheme)]
                seeds = sorted(set(uniform) & set(adaptive))
                if not seeds:
                    continue
                # Tightest (smallest) epsilon attainable by BOTH schemes on
                # EVERY shared seed: iterate ascending and keep the last hit.
                common_eps = None
                for eps in sorted({e for s in seeds for e in uniform[s]}):
                    if all(uniform[s].get(eps, (None, False))[1]
                           and adaptive[s].get(eps, (None, False))[1] for s in seeds):
                        common_eps = eps  # last assignment = tightest (smallest) attainable
                if common_eps is None:
                    continue
                uniform_cost = [uniform[s][common_eps][0] for s in seeds]
                adaptive_cost = [adaptive[s][common_eps][0] for s in seeds]
                h.add_paired_ratio(
                    f"adaptive_stepping_cost_reduction__{regime}__{scheme}",
                    f"MLMC total cost to reach MSE<=eps^2 (Giles allocation), "
                    f"regime={regime}, uniform vs {scheme}, at the tightest "
                    f"epsilon={common_eps} attainable by both on every shared seed "
                    f"(ratio=uniform_cost/{scheme}_cost, seed-paired, n={len(seeds)}); "
                    f"S1b.5 evidence file",
                    relpath(p_cost), uniform_cost, adaptive_cost,
                    "uniform_total_cost", f"{scheme}_total_cost")
    else:
        h.skip(relpath(p_cost), "file not found")

    p_exp = os.path.join(base, "mlmc_exponents.csv")
    if os.path.exists(p_exp):
        rows = load_csv_rows(p_exp)
        by_rs = defaultdict(dict)  # (regime, scheme) -> {seed: beta_var}
        for row in rows:
            seed = int(row["seed"])
            if h.seeds_filter is not None and seed not in h.seeds_filter:
                continue
            try:
                beta = float(row["beta_var"])
            except (KeyError, ValueError):
                continue
            by_rs[(row["regime"], row["scheme"])][seed] = beta

        regimes = sorted({r for r, s in by_rs if s == "uniform"})
        for regime in regimes:
            uniform = by_rs[(regime, "uniform")]
            schemes = sorted({s for (r, s) in by_rs if r == regime and s != "uniform"})
            for scheme in schemes:
                adaptive = by_rs[(regime, scheme)]
                seeds = sorted(set(uniform) & set(adaptive))
                if not seeds:
                    continue
                u = [uniform[s] for s in seeds]
                a = [adaptive[s] for s in seeds]
                h.add_paired_difference(
                    f"adaptive_stepping_beta_var_shift__{regime}__{scheme}",
                    f"MLMC level-difference-variance decay exponent beta_var "
                    f"(Giles 2008 convention), regime={regime}, adaptive-minus-uniform "
                    f"({scheme} minus uniform, seed-paired, n={len(seeds)}); if this CI "
                    f"excludes 0 it contradicts the S1b.5 'beta_var unchanged' finding, "
                    f"if it includes 0 it confirms it with a real interval instead of "
                    f"an eyeballed delta",
                    relpath(p_exp), a, u, f"{scheme}_beta_var", "uniform_beta_var")
    else:
        h.skip(relpath(p_exp), "file not found")


def extract_paper_headline_gaps(h: Harness):
    """Headline numbers in paper/ieee_access/main.tex that have NO backing
    per-seed file anywhere under results/ -- worse than `aggregate_only`, there
    is nothing to ingest, only the literal value typed into the LaTeX table.
    Recorded read-only (this script does not edit paper/*); the point is to
    make the gap machine-readable so it feeds the A100 re-run priority list.

    Verified by inspection of paper/ieee_access/main.tex (lines ~889-996) and
    by grep across results/ finding no JSON/CSV with per-seed GPU-MC / GPU-MLMC
    work counts or runtimes for these six (topology, epsilon) cells.
    """
    paper_loc = "paper/ieee_access/main.tex"

    h.add_known_gap(
        "paper_headline_work_reduction_geomean_57x",
        "Table 'work_units' + Sec. Empirical Validation (2): computational work "
        "reduction GPU-MC -> GPU-MLMC, geometric mean ~57x (range 28-120x across "
        "6 A100 configurations: ER/CAIDA n=500 at eps in {0.05, 0.02, 0.01}); "
        "text separately claims 'rising above 250x at the tightest target when "
        "the single-level baseline is capped at 5e5 paths'.",
        paper_loc,
        reported_value={
            "geomean": "~57x", "range": "28-120x",
            "capped_baseline_claim": ">250x at tightest target, single-level "
                                      "baseline capped at 5e5 paths",
            "table_cost_ratios": {
                "ER eps=0.05": "31.10x", "ER eps=0.02": "77.69x",
                "ER eps=0.01 (dagger, MC capped at 1e6 samples)": ">=64.33x",
                "CAIDA eps=0.05": "27.68x", "CAIDA eps=0.02": "120.35x",
                "CAIDA eps=0.01 (dagger, MC capped at 1e6 samples)": ">=64.33x",
            },
        },
        n_available=1,
        gap_reason=(
            "Each of the 6 (topology, epsilon) cells is a SINGLE run -- no "
            "results/ file (checked: exhaustive scan of this repo's results/ "
            "tree) contains repeated-seed work counts W_MC or W_MLMC for these "
            "specific cells, so there is nothing to bootstrap a CI from. "
            "Separately and more importantly (flagged by a parallel workstream): "
            "the MC baseline's path budget N_MC appears to be a FIXED/CAPPED "
            "count (a 1e6-sample cap is hit and explicitly flagged at eps=0.01; "
            "a 5e5-path cap is invoked for the >250x figure) rather than a "
            "budget chosen to match GPU-MLMC's achieved CI half-width. A "
            "work-reduction ratio computed against an over- or under-provisioned "
            "fixed MC budget is not the same claim as an accuracy-matched "
            "comparison, and can move substantially once the comparison is "
            "redone honestly."
        ),
        required_action=(
            "On the A100: for each of the 6 cells, run >=10 seeds of BOTH "
            "GPU-MC and GPU-MLMC; size the MC baseline's N_MC per seed so its "
            "achieved CI half-width matches GPU-MLMC's (not a fixed/capped "
            "count); write the resulting per-seed (W_MC, W_MLMC) pairs to a "
            "results/ file; route them through Harness.add_paired_ratio here "
            "(ratio = W_MC/W_MLMC, BCa-bootstrapped across seeds, not a naive "
            "division of separately-bootstrapped means). Only then does the "
            "headline number get a CI at all."
        ),
    )

    h.add_known_gap(
        "paper_gpu_mc_vs_mlmc_speedup_tab_gpu_gpu_a100",
        "Table 'gpu_gpu_a100': GPU-MC vs GPU-MLMC wall-clock speedup, same 6 "
        "(topology, epsilon) cells as the work-reduction table; text reports "
        "geometric mean ~3.0x runtime speedup where both methods meet the CI "
        "target (eps>=0.02).",
        paper_loc,
        reported_value={
            "speedups": {"ER eps=0.05": "0.49x", "ER eps=0.02": "2.41x",
                         "ER eps=0.01 (dagger)": ">=1.75x",
                         "CAIDA eps=0.05": "0.59x", "CAIDA eps=0.02": "3.79x",
                         "CAIDA eps=0.01 (dagger)": ">=1.73x"},
            "geomean_where_both_meet_ci_target": "~3.0x",
        },
        n_available=1,
        gap_reason=(
            "The table's own footnote states: 'GPU-MLMC vs. GPU-MC speedup is "
            "from a single controlled run; per-seed replication data in "
            "results/significance/significance_tests.json.' This harness "
            "extracted that exact file (see extract_ana_mlmc / the legacy "
            "cross-check above): it contains only the ANA-vs-Giles-baseline "
            "runtime test and MLMC-only scaling runtimes -- ZERO per-seed "
            "GPU-MC-vs-GPU-MLMC data for these 6 cells. The footnote's citation "
            "does not point to data that exists in this repository."
        ),
        required_action=(
            "Either (a) produce the per-seed replication file the footnote "
            "already claims to exist (>=10 seeds per cell, both methods, at "
            "ER/CAIDA n=500 x eps in {0.05, 0.02, 0.01}) and route it through "
            "Harness.add_paired_ratio, or (b) correct the footnote to state "
            "accurately that these are single-run values."
        ),
    )


def cross_check_legacy(h: Harness):
    """Reproducibility check: does the new pipeline's pooled ANA-vs-baseline
    ratio point estimate match the pre-extension significance_tests.json's
    `mean_speedup`, computed from the SAME underlying CSV? A mismatch would
    mean the extension silently changed what is being measured."""
    path = os.path.join(h.results_root, "significance", "significance_tests.json")
    check = {"source": relpath(path), "checked": False}
    if not os.path.exists(path):
        check["note"] = "no prior significance_tests.json to cross-check against (first run)"
        return check
    try:
        legacy = load_json(path)
    except Exception as exc:
        check["note"] = f"could not parse prior significance_tests.json: {exc}"
        return check
    legacy_entry = legacy.get("ana_vs_baseline_runtime")
    new_entry = h.claims.get("ana_vs_baseline_speedup__pooled_all_conditions")
    if not legacy_entry or not new_entry:
        check["note"] = "one or both sides of the comparison are unavailable this run"
        return check
    legacy_speedup = legacy_entry.get("mean_speedup")
    new_ratio = new_entry["ratio_bca_ci_95"]["point_estimate"]
    if legacy_speedup is None or new_ratio is None:
        check["note"] = "missing point estimate on one side"
        return check
    rel_diff = abs(new_ratio - legacy_speedup) / max(abs(legacy_speedup), 1e-12)
    check.update({
        "checked": True,
        "legacy_mean_speedup": legacy_speedup,
        "new_ratio_of_means_point_estimate": new_ratio,
        "relative_difference": rel_diff,
        "matches_within_1pct": rel_diff < 0.01,
        "note": ("legacy used a naive mean-of-per-pair-ratios; new pipeline uses a "
                 "ratio-of-means, so a small difference is EXPECTED and is not a bug "
                 "-- see docstring of ratio_ci_bootstrap in src/utils/stats.py"),
    })
    return check


EXTRACTORS = [
    extract_ana_mlmc,
    extract_scaling_results,
    extract_ana_weight_sweep,
    extract_per_topology,
    extract_halo_sweep,
    extract_multi_gpu,
    extract_adaptive_stepping_full,
    extract_paper_headline_gaps,
]


# ===========================================================================
# Legacy output (schema of the pre-extension script), for continuity
# ===========================================================================
def write_legacy_output(h: Harness, out_dir: str):
    results = {}
    pooled = h.claims.get("ana_vs_baseline_speedup__pooled_all_conditions")
    if pooled:
        ratio = pooled["ratio_bca_ci_95"]["point_estimate"]
        ttest = pooled["paired_ttest_raw_values"]
        p_val = ttest["pvalue"]
        results["ana_vs_baseline_runtime"] = {
            "n_pairs": pooled["n"],
            "t_statistic": round(ttest["statistic"], 4) if ttest["statistic"] == ttest["statistic"] else None,
            "p_value": round(p_val, 6) if p_val == p_val else None,
            "significant_p05": bool(p_val < 0.05) if p_val == p_val else False,
            "mean_speedup": round(ratio, 3) if ratio == ratio else None,
            "interpretation": (
                f"ANA-MLMC is {'significantly' if (p_val == p_val and p_val < 0.05) else 'NOT significantly'} "
                f"faster than baseline (p={p_val:.4f}, n={pooled['n']}); NOTE: this legacy field is "
                f"kept for continuity -- see all_claims.json for per-condition claims with BCa CIs, "
                f"paired t + Wilcoxon, and effect sizes."
            ) if p_val == p_val else "insufficient data",
        }

    size_results = {}
    for claim_id, entry in h.claims.items():
        if not claim_id.startswith("mlmc_runtime__n"):
            continue
        n_str = claim_id.split("mlmc_runtime__n", 1)[1]
        size_results[int(n_str)] = {
            "mlmc_runtimes": None,  # raw values not retained in the claim dict
            "n_runs": entry["n"],
            "mean_runtime": entry["mean"],
            "std_runtime": entry["sd"],
        }
    if size_results:
        results["mlmc_scaling_runtimes"] = size_results

    out_path = os.path.join(out_dir, "significance_tests.json")
    with open(out_path, "w") as f:
        json.dump(_sanitize(results), f, indent=2)
    return out_path


# ===========================================================================
# CSV output: one row per claim, flattened
# ===========================================================================
CSV_COLUMNS = [
    "claim_id", "kind", "description", "source", "n", "insufficient_n",
    "point_estimate", "point_estimate_kind", "ci_lower", "ci_upper", "ci_method",
    "paired_ttest_pvalue", "wilcoxon_pvalue", "cohens_d", "rank_biserial_r",
]


def claim_to_csv_row(entry: dict) -> dict:
    kind = entry["kind"]
    row = {c: "" for c in CSV_COLUMNS}
    row["claim_id"] = entry["claim_id"]
    row["kind"] = kind
    row["description"] = entry["description"]
    row["source"] = ";".join(entry["source"]) if isinstance(entry["source"], list) else entry["source"]
    row["n"] = entry.get("n", "")
    row["insufficient_n"] = entry.get("insufficient_n", "")

    if kind == "single_sample":
        row["point_estimate"] = entry["mean"]
        row["point_estimate_kind"] = "mean"
        ci = entry["bca_ci_95"]
        row["ci_lower"], row["ci_upper"], row["ci_method"] = ci["ci_lower"], ci["ci_upper"], ci["method"]
    elif kind == "paired_difference":
        row["point_estimate"] = entry["mean_diff_a_minus_b"]
        row["point_estimate_kind"] = "mean_diff"
        ci = entry["diff_bca_ci_95"]
        row["ci_lower"], row["ci_upper"], row["ci_method"] = ci["ci_lower"], ci["ci_upper"], ci["method"]
        row["paired_ttest_pvalue"] = entry["paired_ttest"]["pvalue"]
        row["wilcoxon_pvalue"] = entry["wilcoxon_signed_rank"]["pvalue"]
        row["cohens_d"] = entry["cohens_d_paired"]
        row["rank_biserial_r"] = entry["rank_biserial_r"]
    elif kind == "paired_ratio":
        ci = entry["ratio_bca_ci_95"]
        row["point_estimate"] = ci["point_estimate"]
        row["point_estimate_kind"] = "ratio_of_means"
        row["ci_lower"], row["ci_upper"], row["ci_method"] = ci["ci_lower"], ci["ci_upper"], ci["method"]
        row["paired_ttest_pvalue"] = entry["paired_ttest_raw_values"]["pvalue"]
        row["wilcoxon_pvalue"] = entry["wilcoxon_raw_values"]["pvalue"]
        row["cohens_d"] = entry["cohens_d_paired_raw_values"]
    elif kind == "aggregate_only":
        row["point_estimate"] = entry.get("mean")
        row["point_estimate_kind"] = "mean (as reported by source, not recomputed)"
        reported = entry.get("reported_ci") or {}
        row["ci_method"] = "reported by source (no raw data to recompute)"
        if "half_width_95" in reported and reported["half_width_95"] is not None and entry.get("mean") is not None:
            row["ci_lower"] = entry["mean"] - reported["half_width_95"]
            row["ci_upper"] = entry["mean"] + reported["half_width_95"]
    elif kind == "known_gap":
        row["point_estimate"] = json.dumps(entry.get("reported_value"))
        row["point_estimate_kind"] = "reported_in_paper_no_backing_file"
        row["ci_method"] = "N/A -- " + entry.get("gap_reason", "")[:200]
    return row


def write_csv(claims: dict, out_dir: str) -> str:
    out_path = os.path.join(out_dir, "all_claims.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for claim_id in sorted(claims):
            writer.writerow(claim_to_csv_row(claims[claim_id]))
    return out_path


# ===========================================================================
# Report
# ===========================================================================
def print_report(h: Harness, cross_check: dict):
    print("\n" + "=" * 100)
    print("SIGNIFICANCE HARNESS -- SUMMARY")
    print("=" * 100)

    gaps = {cid: e for cid, e in h.claims.items() if e["kind"] == "known_gap"}
    if gaps:
        print("\n  *** KNOWN GAPS: headline paper numbers with NO backing per-seed "
              "file at all ***")
        for cid, e in sorted(gaps.items()):
            print(f"\n    [{cid}]")
            print(f"      {e['description']}")
            print(f"      gap: {e['gap_reason'][:220]}{'...' if len(e['gap_reason']) > 220 else ''}")
            print(f"      -> required action: {e['required_action'][:200]}"
                  f"{'...' if len(e['required_action']) > 200 else ''}")

    by_kind = defaultdict(list)
    for claim_id, entry in h.claims.items():
        by_kind[entry["kind"]].append(claim_id)
    for kind, ids in sorted(by_kind.items()):
        print(f"\n  {kind}: {len(ids)} claim(s)")

    insufficient = [cid for cid, e in h.claims.items() if e.get("insufficient_n")]
    print(f"\n  TOTAL claims: {len(h.claims)}")
    print(f"  Claims flagged insufficient_n (< {h.min_n} seeds): {len(insufficient)}")
    if insufficient:
        print("\n  Flagged claims (need more seeds before citing in the paper):")
        for cid in sorted(insufficient):
            n = h.claims[cid].get("n")
            print(f"    - {cid}  (n={n})")

    if h.skips:
        print(f"\n  Sources skipped (file not found or unusable): {len(h.skips)}")
        for s in h.skips:
            print(f"    - {s['source']}: {s['reason']}")

    if cross_check.get("checked"):
        status = "MATCH" if cross_check["matches_within_1pct"] else "DIVERGED"
        print(f"\n  Legacy cross-check vs prior significance_tests.json: {status} "
              f"(legacy={cross_check['legacy_mean_speedup']:.4f}, "
              f"new={cross_check['new_ratio_of_means_point_estimate']:.4f}, "
              f"rel_diff={cross_check['relative_difference']:.4%})")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Statistical-significance harness: BCa CIs, paired tests, "
                     "effect sizes, ratio-of-means CIs, over every raw per-seed "
                     "result file this repo has written under results/.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="restrict ingestion to these seed values where a "
                             "source distinguishes seeds (default: use all "
                             "seeds present in each source file)")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "significance"))
    parser.add_argument("--results-root", default=os.path.join(REPO_ROOT, "results"),
                        help="base directory to scan for input result files")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="informational only -- this script's bootstrap runs on CPU "
                             "regardless (numpy/scipy); recorded in provenance for "
                             "interface consistency with run_profiling.py")
    parser.add_argument("--quick", action="store_true",
                        help="fewer bootstrap resamples, for a fast smoke test")
    parser.add_argument("--n-resamples", type=int, default=None,
                        help="bootstrap resamples per CI (default 10000, or 500 with --quick)")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="master RNG seed; every claim's bootstrap stream is "
                             "derived deterministically from this + the claim id")
    parser.add_argument("--min-n", type=int, default=MIN_N_RECOMMENDED,
                        help=f"seeds below this count are flagged insufficient_n "
                             f"(default {MIN_N_RECOMMENDED}, the project's own bar; "
                             f"hard floor for any dispersion estimate is {MIN_N_HARD})")
    args = parser.parse_args()

    n_resamples = args.n_resamples
    if n_resamples is None:
        n_resamples = 500 if args.quick else 10000

    device_label = select_device_name(args.device)
    config = {
        "seeds_filter": args.seeds, "n_resamples": n_resamples,
        "rng_seed": args.rng_seed, "min_n": args.min_n, "quick": args.quick,
        "results_root": os.path.relpath(args.results_root, REPO_ROOT),
    }
    prov = build_provenance(device_label, config)

    print("=" * 100)
    print("Statistical-significance harness")
    print("=" * 100)
    print(f"  git sha       : {prov['git_sha']}")
    print(f"  device        : {prov['device']}  ({prov['compute_device']})")
    print(f"  n_resamples   : {n_resamples}")
    print(f"  min_n         : {args.min_n}")
    print(f"  seeds filter  : {args.seeds or 'all available'}")
    print(f"  results_root  : {os.path.abspath(args.results_root)}")
    print(f"  out           : {os.path.abspath(args.out)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    h = Harness(n_resamples=n_resamples, rng_seed=args.rng_seed, min_n=args.min_n,
                seeds_filter=args.seeds, results_root=args.results_root)

    print("\nIngesting result files:")
    for extractor in EXTRACTORS:
        print(f"  running {extractor.__name__} ...", flush=True)
        try:
            extractor(h)
        except Exception as exc:  # noqa: BLE001 - one bad source must not sink the run
            h.skip(extractor.__name__, f"extractor raised {type(exc).__name__}: {exc}")

    cross_check = cross_check_legacy(h)

    output = {
        "provenance": prov,
        "registry": h.registry,
        "claims": h.claims,
        "skipped_sources": h.skips,
        "legacy_cross_check": cross_check,
        "summary": {
            "n_claims": len(h.claims),
            "n_claims_by_kind": {k: len(v) for k, v in
                                  defaultdict(list, {e["kind"]: [c for c, e2 in h.claims.items()
                                                                  if e2["kind"] == e["kind"]]
                                                      for e in h.claims.values()}).items()},
            "n_insufficient_n": sum(1 for e in h.claims.values() if e.get("insufficient_n")),
            "insufficient_n_claim_ids": sorted(cid for cid, e in h.claims.items()
                                                if e.get("insufficient_n")),
            "min_n_threshold": args.min_n,
        },
    }

    json_path = os.path.join(args.out, "all_claims.json")
    with open(json_path, "w") as f:
        json.dump(_sanitize(output), f, indent=2)

    csv_path = write_csv(h.claims, args.out)
    legacy_path = write_legacy_output(h, args.out)

    print_report(h, cross_check)

    print(f"\nOutput:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {legacy_path}  (legacy schema, kept for continuity)")


if __name__ == "__main__":
    main()
