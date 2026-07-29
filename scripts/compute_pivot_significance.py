#!/usr/bin/env python3
"""Confidence intervals and significance for the repositioned paper's headline claims.

Addresses Reviewer 2's request for statistical significance / confidence intervals on
reported improvements, applied to the two claims the revised manuscript actually rests on:

  1. The MLMC crossover: the accuracy-matched work ratio W_MC / W_MLMC per accuracy target,
     with a bootstrap CI on the RATIO (not a ratio of two independently-CI'd means).
  2. The rare-event comparison: work-normalised efficiency per method per threshold B,
     with BCa bootstrap CIs, and the AMS-vs-IS efficiency gain.

Uses the validated helpers in src/utils/stats.py (33 passing tests).

Run:  python scripts/compute_pivot_significance.py
Out:  results/significance/pivot_claims.json  (+ console table)
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from utils.stats import (  # noqa: E402
    mean_sd_n,
    bca_bootstrap_ci,
    ratio_ci_bootstrap,
)

OUT = REPO / "results" / "significance"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def crossover_claims():
    """Claim 1: accuracy-matched work ratio per epsilon, with a CI on the ratio."""
    out = []
    dirs = sorted(glob.glob(str(REPO / "results/pod_run/crossover_pod/eps_*")),
                  key=lambda p: -float(os.path.basename(p).replace("eps_", "")))
    for d in dirs:
        eps = float(os.path.basename(d).replace("eps_", ""))
        f = os.path.join(d, "rungs.csv")
        if not os.path.exists(f):
            continue
        # pair R0 and R1 by seed so the ratio is bootstrapped on matched pairs
        by = defaultdict(dict)
        for r in read_rows(f):
            if r["rung"] in ("R0", "R1") and r["status"] == "ok":
                by[r["seed"]][r["rung"]] = float(r["work_units"])
        seeds = sorted(s for s in by if "R0" in by[s] and "R1" in by[s])
        if len(seeds) < 2:
            continue
        mc = [by[s]["R0"] for s in seeds]
        ml = [by[s]["R1"] for s in seeds]
        ratio = ratio_ci_bootstrap(mc, ml)
        s_mc, s_ml = mean_sd_n(mc), mean_sd_n(ml)
        out.append({
            "claim_id": f"crossover_eps_{eps:g}",
            "epsilon": eps, "n_seeds": s_mc["n"],
            "W_MC_mean": s_mc["mean"], "W_MC_sd": s_mc["sd"],
            "W_MLMC_mean": s_ml["mean"], "W_MLMC_sd": s_ml["sd"],
            "ratio_MC_over_MLMC": ratio,
            "mlmc_cheaper": (ratio["point_estimate"] > 1.0),
        })
    return out


def rare_event_claims():
    """Claim 2: work-normalised efficiency per (method, B), with BCa CIs."""
    f = REPO / "results/rare_event_comparators/raw.csv"
    if not f.exists():
        return []
    rows = read_rows(f)
    by = defaultdict(lambda: defaultdict(list))
    ref = {}
    for r in rows:
        B = float(r["B"])
        ref[B] = float(r["ref"])
        est = float(r["estimate"]) if r["estimate"] not in ("", "nan") else float("nan")
        by[B][r["method"]].append((est, float(r["work"])))
    out = []
    for B in sorted(by):
        R = ref[B]
        per_method = {}
        for m, vals in by[B].items():
            ests = np.array([v[0] for v in vals], dtype=float)
            works = np.array([v[1] for v in vals], dtype=float)
            good = np.isfinite(ests) & (ests > 0)
            frac_ok = float(good.mean())
            if good.sum() < 2:
                per_method[m] = {"frac_nondegenerate": frac_ok, "degenerate": True}
                continue
            # efficiency statistic: relRMSE^2 * mean work, bootstrapped over seeds.
            # Must accept `axis` so scipy's vectorised bootstrap can call it.
            wmean = float(works.mean())

            def eff_stat(sample, axis=-1, _R=R, _w=wmean):
                s = np.asarray(sample, dtype=float)
                rel2 = np.mean((s - _R) ** 2, axis=axis) / (_R ** 2)
                return rel2 * _w
            ci = bca_bootstrap_ci(ests[good].tolist(), eff_stat)
            per_method[m] = {
                "frac_nondegenerate": frac_ok,
                "mean_estimate": float(ests[good].mean()),
                "work_mean": float(works.mean()),
                "efficiency": ci,
                "degenerate": False,
            }
        # headline: AMS vs the paper's IS
        gain = None
        if not per_method.get("ams", {}).get("degenerate", True) and \
           not per_method.get("gpu_mlmc_is", {}).get("degenerate", True):
            a = per_method["ams"]["efficiency"].get("point_estimate")
            i = per_method["gpu_mlmc_is"]["efficiency"].get("point_estimate")
            if a and i and a > 0:
                gain = i / a
        out.append({"claim_id": f"rare_event_B{B:g}", "B": B, "exact_P": R,
                    "methods": per_method, "ams_efficiency_gain_over_is": gain})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cross = crossover_claims()
    rare = rare_event_claims()

    print("=" * 92)
    print("CLAIM 1 — MLMC crossover: accuracy-matched work ratio (bootstrap CI on the ratio)")
    print(f"  {'epsilon':>10} {'W_MC':>12} {'W_MLMC':>12} {'ratio':>8} {'95% CI':>22} {'verdict':>12}")
    for c in cross:
        r = c["ratio_MC_over_MLMC"]
        pt = r.get("point_estimate", float("nan"))
        lo, hi = r.get("ci_lower") or float("nan"), r.get("ci_upper") or float("nan")
        verdict = "MLMC cheaper" if pt > 1 else "MC cheaper"
        print(f"  {c['epsilon']:10.5f} {c['W_MC_mean']:12.0f} {c['W_MLMC_mean']:12.0f} "
              f"{pt:8.2f} [{lo:8.2f},{hi:8.2f}] {verdict:>12}")

    print("\n" + "=" * 92)
    print("CLAIM 2 — rare-event work-normalised efficiency (lower=better), BCa CI")
    for c in rare:
        print(f"\n  B={c['B']:g}  exact P={c['exact_P']:.3e}"
              f"   AMS efficiency gain over IS: "
              f"{('%.0fx' % c['ams_efficiency_gain_over_is']) if c['ams_efficiency_gain_over_is'] else 'n/a'}")
        print(f"    {'method':>16} {'efficiency':>13} {'95% CI':>26} {'nondeg':>7}")
        for m, d in sorted(c["methods"].items(),
                           key=lambda kv: kv[1].get("efficiency", {}).get("point_estimate", float("inf")) or float("inf")
                           if not kv[1].get("degenerate") else float("inf")):
            if d.get("degenerate"):
                print(f"    {m:>16} {'degenerate':>13} {'--':>26} {d['frac_nondegenerate']:7.2f}")
                continue
            e = d["efficiency"]
            print(f"    {m:>16} {e.get("point_estimate", float("nan")):13.3e} "
                  f"[{e.get("ci_lower") or float("nan"):11.3e},{e.get("ci_upper") or float("nan"):11.3e}] "
                  f"{d['frac_nondegenerate']:7.2f}")

    payload = {
        "provenance": {"git_sha": git_sha(),
                       "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                       "numpy": np.__version__},
        "crossover_claims": cross,
        "rare_event_claims": rare,
    }
    with open(OUT / "pivot_claims.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {OUT/'pivot_claims.json'}")


if __name__ == "__main__":
    main()
