#!/usr/bin/env python3
"""Topology x traffic matrix, and the heterogeneous-load stress sweep.

Reviewer 2: "the evaluation could be strengthened by considering more diverse
network topologies and traffic patterns", and separately "numerical stability
under heterogeneous or extreme load ... interaction with the positive-variance
assumption of Lemma 2".  Both are answered here, because both are questions
about the SAME object: how the estimand's variance structure changes with the
network and the offered load.

What is measured, per (topology, traffic) cell
----------------------------------------------
The revised manuscript's claims are about *which estimator to use when*, so the
matrix reports the quantities that decide that, not a speedup number:

  * reference mean and the reflection bind fraction -- is the estimand even
    stochastic in this cell, or effectively deterministic?
  * per-path variance under plain MC, and the accuracy-matched work ratios for
    antithetic and control variates (same convention as the variance-reduction
    baseline table, scripts/run_qmc_baseline.py)
  * MLMC level variances V_l, their decay rate beta_var, and the Giles-allocated
    work -- from which the measured MC/MLMC crossover epsilon* is located
  * min_l V_l > 0, which is exactly the positive-variance hypothesis of Lemma 2

A cell where beta_var collapses or min V_l reaches zero is a cell where the
multilevel machinery has nothing to stand on; reporting that is the point.

Traffic patterns
----------------
The coupled SDE takes an exogenous per-node offered load lambda_i, so a "traffic
pattern" here is the SPATIAL distribution of that load across nodes, with every
pattern normalised to the SAME mean offered load so cells stay comparable:

    uniform   lambda_i = lambda_bar                      (homogeneous Poisson)
    mmpp2     two-state bursty: a fraction of nodes at a high rate
    pareto    heavy-tailed per-node rates
    degree    hot-spot: lambda_i proportional to node degree

Time-varying arrival processes (diurnal MAWI, on-off) are NOT covered by this
sweep -- the CPU model carries a constant lambda vector -- and the manuscript
says so rather than implying broader coverage than was measured.

Heterogeneous load (--stress)
-----------------------------
Per-node utilisation rho_i = lambda_i / beta_i is spread across a requested band
(default [0.3, 0.99]) by scaling beta_i per node, holding the offered load
fixed.  This is the R2 stress case: near-saturated nodes coexisting with lightly
loaded ones in one network.

Run:
    python scripts/run_topology_matrix.py --quick
    python scripts/run_topology_matrix.py
    python scripts/run_topology_matrix.py --stress

Out:
    results/topology_matrix/matrix.csv
    results/topology_matrix/levels.csv        per-level V_l, C_l
    results/topology_matrix/topology_matrix.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from simulation.batched_paths import (  # noqa: E402
    assert_matches_reference,
    run_level_pair,
    run_paths,
)
from simulation.qmc import (  # noqa: E402
    build_provenance,
    influence_matrix_from_adjacency,
    provenance_comment_lines,
    set_blas_threads,
)

OUT = REPO / "results" / "topology_matrix"
TRAFFIC = ("uniform", "mmpp2", "pareto", "degree")
TOPOLOGIES = ("er", "ba", "ws", "rr", "caida")


# ---------------------------------------------------------------------------
def build_topology(kind, n_nodes, seed):
    """Return (adjacency, label, note).  A synthetic fallback is NEVER labelled
    as real data -- the reviewers objected specifically to blurring measured and
    modelled values."""
    import glob

    from network.topology import (NetworkGraph, TopologyGenerator,
                                  load_caida_topology)

    gen = TopologyGenerator(seed=seed)
    if kind == "er":
        g = gen.generate_erdos_renyi(n_nodes=n_nodes,
                                     p=min(1.0, 10.0 / max(n_nodes - 1, 1)))
    elif kind == "ba":
        g = gen.generate_barabasi_albert(n_nodes=n_nodes, m=3)
    elif kind == "ws":
        g = gen.generate_watts_strogatz(n_nodes=n_nodes, k=10, p=0.1)
    elif kind == "rr":
        g = gen.generate_random_regular(n_nodes=n_nodes, d=10)
    elif kind == "caida":
        cands = []
        cdir = REPO / "datasets" / "caida"
        if cdir.is_dir():
            for ext in ("*.as-rel2.txt.bz2", "*.as-rel2.txt.gz", "*.as-rel2.txt"):
                cands.extend(glob.glob(str(cdir / ext)))
        if not cands:
            return None, "caida", ("no local CAIDA AS-REL2 file under "
                                   "datasets/caida/; cell skipped rather than "
                                   "silently substituted with synthetic data")
        cands.sort()
        net = load_caida_topology(cands[-1], as_undirected=True,
                                  largest_component=True)
        if net.n_nodes > n_nodes:
            keep = [i for i, _ in sorted(net.graph.degree(),
                                         key=lambda kv: kv[1],
                                         reverse=True)[:n_nodes]]
            lim = NetworkGraph(directed=net.graph.is_directed())
            lim.graph = net.graph.subgraph(keep).copy()
            net = lim.get_largest_component()
        return net.get_adjacency_matrix().astype(np.float64), "caida", None
    else:
        raise ValueError(f"unknown topology {kind!r}")
    g = g.get_largest_component()
    return g.get_adjacency_matrix().astype(np.float64), kind, None


def build_lambda(pattern, adjacency, lam_bar, seed):
    """Per-node offered load, normalised so every pattern has mean lam_bar."""
    n = adjacency.shape[0]
    rng = np.random.default_rng(seed + 4242)
    if pattern == "uniform":
        lam = np.full(n, 1.0)
    elif pattern == "mmpp2":
        # 20% of nodes in the high state at 4x the low rate
        high = rng.random(n) < 0.2
        lam = np.where(high, 4.0, 1.0)
    elif pattern == "pareto":
        lam = (rng.pareto(2.0, n) + 1.0)  # finite mean, infinite variance-ish
    elif pattern == "degree":
        deg = adjacency.sum(axis=1)
        lam = np.maximum(deg, 1.0)
    else:
        raise ValueError(f"unknown traffic pattern {pattern!r}")
    return lam * (lam_bar / lam.mean())


def build_beta(args, lam, seed):
    """Per-node decay rate.  In stress mode beta_i is set so utilisation
    rho_i = lambda_i / beta_i spans the requested band."""
    n = lam.size
    if not args.stress:
        return np.full(n, args.beta), None
    rng = np.random.default_rng(seed + 777)
    rho = rng.uniform(args.rho_low, args.rho_high, n)
    beta = lam / rho
    return beta, {"rho_min": float(rho.min()), "rho_max": float(rho.max()),
                  "rho_mean": float(rho.mean())}


# ---------------------------------------------------------------------------
def measure_cell(adjacency, lam, beta, args, seed):
    """All per-cell quantities.  Returns (summary_dict, level_rows)."""
    influence = influence_matrix_from_adjacency(adjacency, args.alpha)
    n_nodes = adjacency.shape[0]
    dt0 = args.base_dt
    n_steps0 = int(round(args.T / dt0))
    m = args.refinement
    rng = np.random.default_rng(seed)
    assert_matches_reference(influence, beta, args.sigma, dt0, lam, rng)

    # finest level used for the single-level MC arm and the MLMC telescope
    dt_fine = dt0 / (m ** args.L_max)
    n_steps_fine = int(round(args.T / dt_fine))

    def draw(p, n_steps, dt):
        return rng.normal(0.0, math.sqrt(dt), (p, n_steps, n_nodes))

    # Every batch below is chunked.  A (4000, 400, 200) float64 increment array
    # is 2.6 GB in one allocation -- enough to push this machine into swap and
    # make a 30-second cell take minutes.  The chunk size bounds the working set
    # to roughly chunk * n_steps_fine * n_nodes * 8 bytes.
    def batched(total, fn):
        out, clamp = [], 0.0
        done = 0
        while done < total:
            k = min(args.chunk, total - done)
            res, cf = fn(k)
            out.append(res)
            clamp = max(clamp, cf)
            done += k
        return out, clamp

    # --- reference + plain-MC variance at the finest level -----------------
    def _mc(k):
        dw = draw(k, n_steps_fine, dt_fine)
        y, _, cf = run_paths(dw, influence, beta, args.sigma, dt_fine, lam)
        return y, cf

    ys, clamp = batched(args.var_samples, _mc)
    y_mc = np.concatenate(ys)
    ref_mean = float(y_mc.mean())
    var_mc = float(np.var(y_mc, ddof=1))

    # --- antithetic and control-variate variance ---------------------------
    def _anti(k):
        dw = draw(k, n_steps_fine, dt_fine)
        y2, x2, cf = run_paths(np.concatenate([dw, -dw], axis=0), influence,
                               beta, args.sigma, dt_fine, lam, want_linear=True)
        return (0.5 * (y2[:k] + y2[k:]), y2[:k], x2[:k]), cf

    parts, _ = batched(max(2, args.var_samples // 2), _anti)
    anti = np.concatenate([p[0] for p in parts])
    yv = np.concatenate([p[1] for p in parts])
    xv = np.concatenate([p[2] for p in parts])
    var_anti = float(np.var(anti, ddof=1))
    vx = float(np.var(xv, ddof=1))
    coef = float(np.cov(yv, xv, ddof=1)[0, 1] / vx) if vx > 0 else 0.0
    var_cv = float(np.var(yv - coef * xv, ddof=1))

    # --- MLMC level variances ---------------------------------------------
    level_rows, v_l, c_l = [], [], []
    def _lev0(k):
        dw0 = draw(k, n_steps0, dt0)
        y0, _, cf = run_paths(dw0, influence, beta, args.sigma, dt0, lam)
        return y0, cf

    y0parts, _ = batched(args.level_samples, _lev0)
    y0 = np.concatenate(y0parts)
    v_l.append(float(np.var(y0, ddof=1)))
    c_l.append(float(n_steps0))
    level_rows.append({"level": 0, "dt": dt0, "V_l": v_l[0], "C_l": c_l[0],
                       "mean_Y": float(y0.mean())})
    for lev in range(1, args.L_max + 1):
        dt_f = dt0 / (m ** lev)
        n_f = int(round(args.T / dt_f))

        def _pair(k, _dt_f=dt_f, _n_f=n_f):
            dwf = draw(k, _n_f, _dt_f)
            yf, yc = run_level_pair(dwf, influence, beta, args.sigma, _dt_f,
                                    lam, m)
            return yf - yc, 0.0

        diffs, _ = batched(args.level_samples, _pair)
        d = np.concatenate(diffs)
        v_l.append(float(np.var(d, ddof=1)))
        c_l.append(float(n_f + n_f // m))
        level_rows.append({"level": lev, "dt": dt_f, "V_l": v_l[-1],
                           "C_l": c_l[-1], "mean_Y": float(d.mean())})

    # beta_var: V_l ~ h_l^beta, i.e. slope of log V_l vs log dt over l>=1
    if len(v_l) > 2 and all(v > 0 for v in v_l[1:]):
        logs = np.log([r["dt"] for r in level_rows[1:]])
        logv = np.log(v_l[1:])
        beta_var = float(np.polyfit(logs, logv, 1)[0])
    else:
        beta_var = float("nan")

    # --- accuracy-matched work and the crossover ---------------------------
    # Both arms carry a pilot floor: no estimator can run fewer than
    # `pilot_samples` paths per level, because the variances that drive the
    # allocation have to be estimated first.  Omitting it makes MLMC look
    # cheaper than MC at every tolerance -- and that floor, spread over L+1
    # levels, is exactly why MLMC loses at loose targets in the measured GPU
    # protocol.  Leaving it out would have contradicted Table 3.
    def work_mlmc(eps_abs):
        s = sum(math.sqrt(v * c) for v, c in zip(v_l, c_l) if v > 0)
        if s <= 0:
            return float("nan")
        total = 0.0
        for v, c in zip(v_l, c_l):
            if v <= 0:
                continue
            n_lev = max(float(args.pilot_samples),
                        2.0 * eps_abs ** -2 * math.sqrt(v / c) * s)
            total += n_lev * c
        return total

    def work_mc(eps_abs):
        return max(float(args.pilot_samples),
                   2.0 * var_mc / eps_abs ** 2) * n_steps_fine

    # A cell where BOTH arms sit on the pilot floor yields a ratio that is just
    # the ratio of pilot costs -- identical in every cell, and a measurement of
    # nothing.  Flag it rather than report it.
    floor_mc = float(args.pilot_samples) * n_steps_fine
    floor_mlmc = sum(float(args.pilot_samples) * c for v, c in zip(v_l, c_l) if v > 0)

    works = {}
    for rel in args.rel_eps:
        e = rel * abs(ref_mean)
        wmc, wml = work_mc(e), work_mlmc(e)
        works[rel] = (wmc, wml)
        summary_floor = (wmc <= floor_mc * 1.0000001
                         and wml <= floor_mlmc * 1.0000001)
        works[rel] = (wmc, wml, summary_floor)
    # locate the crossover on a fine relative-epsilon grid
    grid = np.logspace(math.log10(args.cross_hi), math.log10(args.cross_lo), 200)
    cross = float("nan")
    for rel in grid:
        e = rel * abs(ref_mean)
        if work_mlmc(e) < work_mc(e):
            cross = float(rel)
            break

    summary = {
        "n_nodes": n_nodes, "ref_mean": ref_mean,
        "V_l": list(v_l), "C_l": list(c_l), "n_steps_fine": n_steps_fine,
        "pilot_samples": args.pilot_samples,
        "reflection_bind_fraction": clamp,
        "var_mc_per_path": var_mc,
        "var_antithetic_per_pair": var_anti,
        "var_cv_per_path": var_cv,
        "cv_coefficient": coef,
        # work ratios: antithetic and CV each cost 2 paths per sample
        "antithetic_work_ratio_vs_mc": (var_mc / (var_anti * 2.0)
                                        if var_anti > 0 else float("inf")),
        "cv_work_ratio_vs_mc": (var_mc / (var_cv * 2.0)
                                if var_cv > 0 else float("inf")),
        "beta_var": beta_var,
        "min_V_l": float(min(v_l)),
        "lemma2_positive_variance_holds": bool(min(v_l) > 0),
        "crossover_rel_eps": cross,
    }
    for rel, (wmc, wml, at_floor) in works.items():
        summary[f"W_MC_eps_{rel:g}"] = wmc
        summary[f"W_MLMC_eps_{rel:g}"] = wml
        summary[f"both_at_pilot_floor_eps_{rel:g}"] = at_floor
        summary[f"ratio_eps_{rel:g}"] = (
            float("nan") if at_floor
            else (wmc / wml if wml and wml > 0 else float("nan")))
    return summary, level_rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topologies", nargs="+", default=list(TOPOLOGIES))
    ap.add_argument("--traffic", nargs="+", default=list(TRAFFIC))
    ap.add_argument("--n-nodes", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.433,
                    help="default is the variance-calibrated value of Table 3")
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--base-dt", type=float, default=0.4)
    ap.add_argument("--L-max", type=int, default=3)
    ap.add_argument("--refinement", type=int, default=2)
    ap.add_argument("--pilot-samples", type=int, default=50,
                    help="per-level pilot floor; matches the measured GPU protocol")
    ap.add_argument("--var-samples", type=int, default=4000)
    ap.add_argument("--level-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=250,
                    help="paths per batch; bounds peak memory")
    ap.add_argument("--rel-eps", type=float, nargs="+", default=[0.02, 0.01, 0.005])
    ap.add_argument("--cross-hi", type=float, default=0.05)
    ap.add_argument("--cross-lo", type=float, default=1e-5)
    ap.add_argument("--stress", action="store_true",
                    help="heterogeneous per-node utilisation (Reviewer 2)")
    ap.add_argument("--rho-low", type=float, default=0.3)
    ap.add_argument("--rho-high", type=float, default=0.99)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.quick:
        given = {a.split("=")[0] for a in sys.argv[1:] if a.startswith("--")}
        for flag, attr, val in (("--n-nodes", "n_nodes", 60),
                                ("--seeds", "seeds", [0]),
                                ("--var-samples", "var_samples", 1000),
                                ("--level-samples", "level_samples", 500),
                                ("--chunk", "chunk", 500),
                                ("--topologies", "topologies", ["er", "ba"]),
                                ("--traffic", "traffic", ["uniform", "pareto"])):
            if flag not in given:
                setattr(args, attr, val)

    thread_info = set_blas_threads(args.threads)
    out_dir = Path(args.out) if args.out else (
        OUT.parent / "topology_matrix_stress" if args.stress else OUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = vars(args).copy()
    prov = build_provenance(cfg, thread_info, device="cpu")
    print("\n".join(provenance_comment_lines(prov)), flush=True)
    print(f"# mode: {'heterogeneous-load stress' if args.stress else 'topology x traffic'}",
          flush=True)

    rows, level_rows, notes = [], [], []
    for topo in args.topologies:
        for traffic in args.traffic:
            for seed in args.seeds:
                adjacency, label, note = build_topology(topo, args.n_nodes, seed)
                if adjacency is None:
                    if note and note not in notes:
                        notes.append(note)
                        print(f"  SKIP {topo}: {note}", flush=True)
                    continue
                lam = build_lambda(traffic, adjacency, args.lam, seed)
                beta, rho_info = build_beta(args, lam, seed)
                t0 = time.time()
                summary, lv = measure_cell(adjacency, lam, beta, args, seed)
                summary.update({"topology": label, "traffic": traffic,
                                "seed": seed, "elapsed_s": time.time() - t0})
                if rho_info:
                    summary.update(rho_info)
                rows.append(summary)
                for r in lv:
                    r.update({"topology": label, "traffic": traffic, "seed": seed})
                level_rows.extend(r for r in lv)
                print(f"  {label:>6}/{traffic:<8} seed={seed}  "
                      f"mean={summary['ref_mean']:.4f}  "
                      f"Var_MC={summary['var_mc_per_path']:.3e}  "
                      f"anti={summary['antithetic_work_ratio_vs_mc']:.3g}x  "
                      f"beta_var={summary['beta_var']:+.2f}  "
                      f"cross_eps={summary['crossover_rel_eps']:.2e}  "
                      f"[{summary['elapsed_s']:.1f}s]", flush=True)

    if not rows:
        print("No cells ran.")
        return

    def write_csv(path, data):
        keys = sorted({k for r in data for k in r})
        with open(path, "w", newline="") as f:
            for line in provenance_comment_lines(prov):
                f.write(line + "\n")
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(data)

    write_csv(out_dir / "matrix.csv", rows)
    write_csv(out_dir / "levels.csv", level_rows)
    with open(out_dir / "topology_matrix.json", "w") as f:
        json.dump({"provenance": prov, "rows": rows, "levels": level_rows,
                   "notes": notes}, f, indent=2, default=float)

    # ---- aggregate across seeds ------------------------------------------
    print("\n" + "=" * 96)
    title = ("HETEROGENEOUS-LOAD STRESS" if args.stress
             else "TOPOLOGY x TRAFFIC MATRIX")
    print(f"{title}  (mean over {len(args.seeds)} seed(s), sigma={args.sigma})")
    # The antithetic/CV ratios and the MLMC ratio are printed side by side and
    # computed in the SAME cell, against the same MC arm and the same work
    # convention.  That is the only way the comparison means anything: quoting a
    # variance-reduction gain measured here against an MLMC gain measured in a
    # different model and network would be comparing two unrelated numbers.
    eps_ref = args.rel_eps[len(args.rel_eps) // 2]
    print(f"  {'topology':>8} {'traffic':>8} {'mean':>8} {'Var_MC':>10} "
          f"{'anti x':>9} {'cv x':>8} {'MLMC x':>8} {'beta_var':>9} "
          f"{'min V_l':>10} {'cross eps':>10}   (MLMC x at rel eps=%g)" % eps_ref)
    for topo in args.topologies:
        for traffic in args.traffic:
            cells = [r for r in rows
                     if r["topology"] == topo and r["traffic"] == traffic]
            if not cells:
                continue
            def mu(k):
                v = [c[k] for c in cells if np.isfinite(c[k])]
                return float(np.mean(v)) if v else float("nan")
            ok = all(c["lemma2_positive_variance_holds"] for c in cells)
            print(f"  {topo:>8} {traffic:>8} {mu('ref_mean'):8.4f} "
                  f"{mu('var_mc_per_path'):10.3e} "
                  f"{mu('antithetic_work_ratio_vs_mc'):9.3g} "
                  f"{mu('cv_work_ratio_vs_mc'):8.3g} "
                  f"{mu(f'ratio_eps_{eps_ref:g}'):8.3g} {mu('beta_var'):+9.2f} "
                  f"{mu('min_V_l'):10.3e} {mu('crossover_rel_eps'):10.2e}"
                  f"{'' if ok else '  [Lemma 2 violated: V_l = 0]'}")
    if notes:
        print("\nNotes:")
        for n in notes:
            print(f"  - {n}")
    print(f"\nWrote {out_dir}/matrix.csv, levels.csv, topology_matrix.json")


if __name__ == "__main__":
    main()
