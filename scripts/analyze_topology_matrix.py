#!/usr/bin/env python3
"""Aggregate a topology-matrix run into the paper table.

Reads `matrix.csv` (per-cell variance quantities) and `levels.csv` (per-level
V_l, C_l) from a completed run and re-derives the accuracy-matched MC/MLMC
comparison at any accuracy target.  No simulation is re-run: work at a given
epsilon is a closed-form function of the measured V_l, C_l and per-path
variance, so re-deriving costs milliseconds where re-measuring costs minutes.

The reason this exists: a first pass reported the MLMC/MC work ratio at
epsilon = 10^-2 and got *exactly* 0.364 in all sixteen cells.  That is not a
finding, it is the ratio of the two arms' pilot costs -- at a target that loose,
both estimators are floor-limited and neither is doing any work proportional to
the target.  Ratios are therefore reported here only where at least one arm has
left its floor, and floored cells are labelled as such.

Run:
    python scripts/analyze_topology_matrix.py
    python scripts/analyze_topology_matrix.py --dir results/topology_matrix_stress
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def giles_work(v_l, c_l, eps_abs, pilot):
    s = sum(math.sqrt(v * c) for v, c in zip(v_l, c_l) if v > 0)
    if s <= 0:
        return float("nan"), True
    total, floored = 0.0, True
    for v, c in zip(v_l, c_l):
        if v <= 0:
            continue
        n_opt = 2.0 * eps_abs ** -2 * math.sqrt(v / c) * s
        if n_opt > pilot:
            floored = False
        total += max(float(pilot), n_opt) * c
    return total, floored


def mc_work(var_mc, eps_abs, n_steps_fine, pilot):
    n_opt = 2.0 * var_mc / eps_abs ** 2
    return max(float(pilot), n_opt) * n_steps_fine, n_opt <= pilot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(REPO / "results" / "topology_matrix"))
    ap.add_argument("--rel-eps", type=float, nargs="+",
                    default=[1e-2, 5e-3, 2e-3, 1e-3, 5e-4])
    ap.add_argument("--pilot-samples", type=int, default=50)
    ap.add_argument("--n-steps-fine", type=int, default=400)
    args = ap.parse_args()

    d = Path(args.dir)
    rows = read_csv(d / "matrix.csv")
    levels = read_csv(d / "levels.csv")

    lv = defaultdict(list)
    for r in levels:
        key = (r["topology"], r["traffic"], r["seed"])
        lv[key].append((int(r["level"]), float(r["V_l"]), float(r["C_l"])))

    out = []
    for r in rows:
        key = (r["topology"], r["traffic"], r["seed"])
        ls = sorted(lv.get(key, []))
        if not ls:
            continue
        v_l = [x[1] for x in ls]
        c_l = [x[2] for x in ls]
        var_mc = float(r["var_mc_per_path"])
        mean = abs(float(r["ref_mean"]))
        rec = {k: r[k] for k in ("topology", "traffic", "seed")}
        rec.update({
            "ref_mean": mean,
            "var_mc_per_path": var_mc,
            "antithetic_work_ratio_vs_mc": float(r["antithetic_work_ratio_vs_mc"]),
            "cv_work_ratio_vs_mc": float(r["cv_work_ratio_vs_mc"]),
            "beta_var": float(r["beta_var"]),
            "min_V_l": float(r["min_V_l"]),
            "crossover_rel_eps": float(r["crossover_rel_eps"]),
        })
        for rel in args.rel_eps:
            e = rel * mean
            wm, fm = mc_work(var_mc, e, args.n_steps_fine, args.pilot_samples)
            wl, fl = giles_work(v_l, c_l, e, args.pilot_samples)
            both_floored = fm and fl
            rec[f"W_MC_{rel:g}"] = wm
            rec[f"W_MLMC_{rel:g}"] = wl
            rec[f"both_floored_{rel:g}"] = both_floored
            rec[f"ratio_{rel:g}"] = float("nan") if both_floored else wm / wl
        out.append(rec)

    if not out:
        print("no rows")
        return

    with open(d / "aggregated.csv", "w", newline="") as f:
        keys = sorted({k for r in out for k in r})
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(out)

    groups = defaultdict(list)
    for r in out:
        groups[(r["topology"], r["traffic"])].append(r)

    def mu(cells, k):
        v = [c[k] for c in cells if isinstance(c[k], float) and np.isfinite(c[k])]
        return float(np.mean(v)) if v else float("nan")

    print("=" * 104)
    print(f"TOPOLOGY x TRAFFIC -- aggregated over seeds  ({d.name})")
    hdr = (f"  {'topo':>5} {'traffic':>8} {'Var_MC':>10} {'anti x':>7} "
           f"{'cv x':>6} {'beta_var':>8} {'min V_l':>10} {'cross eps':>10}")
    for rel in args.rel_eps:
        hdr += f" {'MLMC@%g' % rel:>10}"
    print(hdr)
    for (topo, traffic), cells in groups.items():
        line = (f"  {topo:>5} {traffic:>8} {mu(cells,'var_mc_per_path'):10.3e} "
                f"{mu(cells,'antithetic_work_ratio_vs_mc'):7.1f} "
                f"{mu(cells,'cv_work_ratio_vs_mc'):6.1f} "
                f"{mu(cells,'beta_var'):+8.2f} {mu(cells,'min_V_l'):10.3e} "
                f"{mu(cells,'crossover_rel_eps'):10.2e}")
        for rel in args.rel_eps:
            if all(c[f"both_floored_{rel:g}"] for c in cells):
                line += f" {'floor':>10}"
            else:
                line += f" {mu(cells, f'ratio_{rel:g}'):10.2f}"
        print(line)

    # stability summary -- the actual claim the matrix supports
    anti = [r["antithetic_work_ratio_vs_mc"] for r in out]
    bvar = [r["beta_var"] for r in out]
    cross = [r["crossover_rel_eps"] for r in out]
    minv = [r["min_V_l"] for r in out]
    by_traffic = defaultdict(list)
    for r in out:
        by_traffic[r["traffic"]].append(r["antithetic_work_ratio_vs_mc"])

    print("\nSTABILITY ACROSS THE MATRIX")
    print(f"  antithetic gain      {min(anti):.1f}x - {max(anti):.1f}x")
    for t, v in sorted(by_traffic.items()):
        print(f"      {t:>8}: {np.mean(v):.1f}x")
    print(f"  beta_var             {min(bvar):+.2f} to {max(bvar):+.2f}")
    print(f"  crossover rel eps    {min(cross):.2e} to {max(cross):.2e} "
          f"({max(cross)/min(cross):.2f}x spread)")
    print(f"  min_l V_l            {min(minv):.3e}  "
          f"(Lemma 2 positive-variance hypothesis: "
          f"{'HOLDS in all cells' if min(minv) > 0 else 'VIOLATED'})")
    with open(d / "aggregated_summary.json", "w") as f:
        json.dump({"antithetic_gain_range": [min(anti), max(anti)],
                   "antithetic_by_traffic": {k: float(np.mean(v))
                                             for k, v in by_traffic.items()},
                   "beta_var_range": [min(bvar), max(bvar)],
                   "crossover_rel_eps_range": [min(cross), max(cross)],
                   "min_V_l": min(minv),
                   "lemma2_holds": bool(min(minv) > 0),
                   "n_cells": len(out)}, f, indent=2)
    print(f"\nWrote {d}/aggregated.csv, aggregated_summary.json")


if __name__ == "__main__":
    main()
