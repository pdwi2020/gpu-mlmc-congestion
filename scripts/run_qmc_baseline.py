#!/usr/bin/env python3
"""Variance-reduction baseline suite for the coupled congestion SDE.

Reviewer 1, concern 5: "More baselines should be added, such as CPU-MLMC,
optimized discrete-event simulation, existing GPU network simulators, or other
uncertainty quantification methods."

The optimised discrete-event baseline is scripts/run_des_baseline.py and the
CPU-MLMC baseline is scripts/run_cpu_mlmc_baseline.py.  This script supplies the
remaining arm -- the classical variance-reduction / QMC alternatives that a
referee would expect to see before accepting that a multilevel method is
warranted:

    mc          plain Monte Carlo (the reference arm)
    antithetic  antithetic variates, (dW, -dW) pairs
    cv          control variates, using the UNREFLECTED linear recursion driven
                by the same Brownian path as the control.  Its mean is known in
                closed form (qmc.fluid_limit_mean), so the estimator is
                unbiased -- no simulated control mean, no pilot bias in E[X].
    qmc         randomised quasi-Monte Carlo: Owen-scrambled Sobol points mapped
                through a Brownian-bridge time ordering, so the low-index (high
                quality) Sobol coordinates carry the coarsest, highest-variance
                bridge levels.  R independent scramblings give an unbiased
                estimate and an honest variance estimate.

All arms share ONE estimand and ONE time grid: Y = mean_i C_i(T) under the
predictor-corrector Euler-Maruyama recursion of src/simulation/qmc.py, at the
same (lambda, alpha, beta, sigma, T, base_dt) as the DES baseline.  Sharing the
grid is deliberate: it removes discretisation bias from the comparison so what
is measured is variance-reduction efficiency alone, which is the quantity the
multilevel claim has to beat.

Scoring is accuracy-matched, matching the convention used for the crossover
table (Table 3).  Every arm is summarised as Var(N) = const * N^rate, from which
the work to reach a relative-RMSE target eps is

    N(eps) = (eps * |mu| )^2 / const )^(1/rate),   W = N * n_steps * paths/sample

For the three i.i.d. arms (mc, antithetic, cv) the rate is -1 exactly -- these
are sample means of i.i.d. quantities, so the rate is not something to estimate,
and `const` is the per-sample (or per-antithetic-pair) variance measured
directly from one large batch.  That is both cheaper and far more precise than
inferring a slope from replication-to-replication scatter: an early version of
this script fitted plain MC's slope from 4 replications and got +1.63 instead of
-1, which would have made every ratio in the table meaningless.

Only randomised QMC has a genuinely unknown rate -- that is the whole question
about QMC -- so it alone is fitted, from R independent Owen scramblings at each
of several budgets, with the fitted rate, its R^2 and the number of points
reported so a reader can see when the fit is trustworthy.

`paths_per_sample` charges each arm for what it actually costs: antithetic pairs
cost two paths, and the control-variate arm costs two path-equivalents because
it integrates the linear companion alongside.  Without that, variance reduction
bought with extra work would look free.

Both sigma parameterisations of Table 3 are run by default: the manuscript's
sigma=0.1 and the variance-calibrated sigma=0.433.

Run:
    python scripts/run_qmc_baseline.py --quick          # smoke test, ~1 min
    python scripts/run_qmc_baseline.py                  # full grid

Out:
    results/baselines/qmc/raw.csv        one row per (sigma, method, N, rep)
    results/baselines/qmc/summary.csv    one row per (sigma, method): rate, W(eps)
    results/baselines/qmc/qmc_baseline.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import zlib
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from simulation.batched_paths import (  # noqa: E402
    assert_matches_reference,
    bridge_schedule,
    paths_from_bridge,
    run_paths,
)
from simulation.qmc import (  # noqa: E402
    build_provenance,
    fluid_limit_mean,
    influence_matrix_from_adjacency,
    load_topology,
    provenance_comment_lines,
    set_blas_threads,
    sobol_normals,
)

OUT = REPO / "results" / "baselines" / "qmc"
SOBOL_MAXDIM = 21201  # scipy.stats.qmc.Sobol hard limit


# ---------------------------------------------------------------------------
# Estimator arms
#
# The i.i.d. arms are characterised by ONE number each -- the variance of a
# single sample unit -- measured from a large batch.  The QMC arm is
# characterised by a fitted (rate, const) pair.  Both are returned in the common
# form Var(N) = const * N^rate so a single work formula scores every arm.
# ---------------------------------------------------------------------------
def draw_normal_paths(n, ctx, rng):
    return rng.normal(0.0, math.sqrt(ctx["dt"]),
                      (n, ctx["n_steps"], ctx["n_nodes"]))


def _chunks(total, size):
    done = 0
    while done < total:
        m = min(size, total - done)
        yield m
        done += m


def arm_mc(ctx, rng, args):
    """Plain MC: const = per-path variance of Y, rate = -1, 1 path per sample."""
    ys, cf = [], 0.0
    for m in _chunks(args.var_samples, args.chunk):
        dw = draw_normal_paths(m, ctx, rng)
        y, _, c = run_paths(dw, ctx["influence"], ctx["beta"], ctx["sigma"],
                            ctx["dt"], ctx["lam"])
        ys.append(y)
        cf = max(cf, c)
    y = np.concatenate(ys)
    return {"rate": -1.0, "const": float(np.var(y, ddof=1)), "rate_r2": None,
            "rate_points": None, "paths_per_sample": 1.0,
            "sample_unit": "path", "mean_estimate": float(y.mean()),
            "n_var_samples": int(y.size), "clamp_frac": cf}


def arm_antithetic(ctx, rng, args):
    """Antithetic variates: the sample unit is the PAIR mean (Y+ + Y-)/2.

    const is the per-pair variance and each pair costs two paths, so a pair that
    merely halves the variance buys nothing -- the work charge makes that
    visible rather than hiding it behind a per-sample variance comparison.
    """
    n_pairs = max(2, args.var_samples // 2)
    means, cf = [], 0.0
    for m in _chunks(n_pairs, max(1, args.chunk // 2)):
        dw = draw_normal_paths(m, ctx, rng)
        both = np.concatenate([dw, -dw], axis=0)
        y, _, c = run_paths(both, ctx["influence"], ctx["beta"], ctx["sigma"],
                            ctx["dt"], ctx["lam"])
        means.append(0.5 * (y[:m] + y[m:]))
        cf = max(cf, c)
    a = np.concatenate(means)
    return {"rate": -1.0, "const": float(np.var(a, ddof=1)), "rate_r2": None,
            "rate_points": None, "paths_per_sample": 2.0,
            "sample_unit": "antithetic pair", "mean_estimate": float(a.mean()),
            "n_var_samples": int(a.size), "clamp_frac": cf}


def arm_cv(ctx, rng, args):
    """Control variates against the unreflected linear recursion.

    E[X] is known in closed form (fluid_limit_mean), so the estimator is
    unbiased for any coefficient; the optimal coefficient is estimated from an
    independent pilot.  Integrating the linear companion alongside the reflected
    path roughly doubles the per-step arithmetic, hence 2 paths per sample.
    """
    n_pilot = ctx["cv_pilot"]
    dwp = draw_normal_paths(n_pilot, ctx, rng)
    yp, xp, _ = run_paths(dwp, ctx["influence"], ctx["beta"], ctx["sigma"],
                          ctx["dt"], ctx["lam"], want_linear=True)
    var_x = float(np.var(xp, ddof=1))
    coef = float(np.cov(yp, xp, ddof=1)[0, 1] / var_x) if var_x > 0 else 0.0
    corr = float(np.corrcoef(yp, xp)[0, 1]) if var_x > 0 else float("nan")

    zs, cf = [], 0.0
    for m in _chunks(args.var_samples, args.chunk):
        dw = draw_normal_paths(m, ctx, rng)
        y, x, c = run_paths(dw, ctx["influence"], ctx["beta"], ctx["sigma"],
                            ctx["dt"], ctx["lam"], want_linear=True)
        zs.append(y - coef * (x - ctx["control_mean"]))
        cf = max(cf, c)
    z = np.concatenate(zs)
    return {"rate": -1.0, "const": float(np.var(z, ddof=1)), "rate_r2": None,
            "rate_points": None, "paths_per_sample": 2.0,
            "sample_unit": "path (+ linear companion)",
            "mean_estimate": float(z.mean()), "n_var_samples": int(z.size),
            "clamp_frac": cf, "cv_coef": coef, "cv_corr": corr}


def qmc_estimate(n, ctx, rng):
    """One randomised-QMC estimate from an Owen-scrambled Sobol point set.

    The Sobol dimension budget is capped at scipy's 21201.  Bridge levels beyond
    the cap are filled with pseudo-random normals -- a hybrid, which is what any
    honest application of QMC to an 80,000-dimensional path space has to do.
    The realised quasi-random fraction is recorded so the reader can see how much
    of the path was actually QMC.
    """
    n_steps, n_nodes = ctx["n_steps"], ctx["n_nodes"]
    schedule = ctx["schedule"]
    n_bridge = len(schedule)
    total_dim = n_bridge * n_nodes
    d_qmc = min(SOBOL_MAXDIM, total_dim)
    d_qmc -= d_qmc % n_nodes  # keep whole bridge levels quasi-random
    n_bridge_qmc = d_qmc // n_nodes

    n_pow2 = 1 << int(round(math.log2(n)))
    seed = int(rng.integers(0, 2**31 - 1))
    z = np.empty((n_pow2, n_bridge, n_nodes))
    if n_bridge_qmc > 0:
        z[:, :n_bridge_qmc, :] = sobol_normals(n_pow2, d_qmc, seed).reshape(
            n_pow2, n_bridge_qmc, n_nodes)
    if n_bridge_qmc < n_bridge:
        z[:, n_bridge_qmc:, :] = rng.normal(
            0.0, 1.0, (n_pow2, n_bridge - n_bridge_qmc, n_nodes))

    dw = paths_from_bridge(z, schedule, n_steps, ctx["dt"])
    y, _, cf = run_paths(dw, ctx["influence"], ctx["beta"], ctx["sigma"],
                         ctx["dt"], ctx["lam"])
    return float(y.mean()), n_pow2, cf, d_qmc / float(total_dim)


def bootstrap_rate_ci(ns, per_n_estimates, n_boot, seed, level=0.95):
    """Percentile bootstrap CI for the fitted QMC convergence rate.

    The rate is the entire QMC claim, and it is fitted from variances that are
    themselves estimated from only R scramblings each -- so it carries real
    uncertainty.  Resampling scramblings within each budget and refitting shows
    how much.  Reporting the point estimate alone would overstate what R
    randomisations can establish.
    """
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_boot):
        variances = []
        for est in per_n_estimates:
            e = np.asarray(est)
            variances.append(float(np.var(rng.choice(e, size=e.size,
                                                     replace=True), ddof=1)))
        r, _, _ = fit_rate(ns, variances)
        if np.isfinite(r):
            rates.append(r)
    if len(rates) < 10:
        return float("nan"), float("nan")
    lo = float(np.percentile(rates, 100 * (1 - level) / 2))
    hi = float(np.percentile(rates, 100 * (1 + level) / 2))
    return lo, hi


def arm_qmc(ctx, rng, args, raw_rows, sigma, ref_mean):
    """Randomised QMC: fit Var(N) = const * N^rate over the budget ladder."""
    ns, variances, per_n, cf, dim_frac = [], [], [], 0.0, float("nan")
    last_mean = float("nan")
    for n in args.budgets:
        ests = []
        for rep in range(args.reps):
            est, charged, c, df = qmc_estimate(n, ctx, rng)
            ests.append(est)
            cf, dim_frac = max(cf, c), df
            raw_rows.append({"sigma": sigma, "method": "qmc", "N": n, "rep": rep,
                             "estimate": est, "paths_charged": charged,
                             "work_units": charged * ctx["n_steps"],
                             "ref_mean": ref_mean,
                             "abs_err": abs(est - ref_mean)})
        ests = np.asarray(ests)
        ns.append(n)
        per_n.append(ests)
        variances.append(float(np.var(ests, ddof=1)))
        last_mean = float(ests.mean())
        print(f"  {'qmc':>10} N={n:>5}  var={variances[-1]:.3e}  "
              f"bias={last_mean - ref_mean:+.2e}  (R={args.reps} scramblings)",
              flush=True)
    rate, const, r2 = fit_rate(ns, variances)
    lo, hi = bootstrap_rate_ci(ns, per_n, args.rate_boot, args.seed + 1)
    return {"rate": rate, "const": const, "rate_r2": r2, "rate_points": len(ns),
            "rate_ci_lower": lo, "rate_ci_upper": hi,
            "paths_per_sample": 1.0, "sample_unit": "Sobol point",
            "mean_estimate": last_mean, "n_var_samples": args.reps,
            "clamp_frac": cf, "qmc_dim_fraction": dim_frac}


IID_ARMS = {"mc": arm_mc, "antithetic": arm_antithetic, "cv": arm_cv}
ARMS = list(IID_ARMS) + ["qmc"]


# ---------------------------------------------------------------------------
# Rate fitting and accuracy-matched work
# ---------------------------------------------------------------------------
def fit_rate(ns, variances):
    """Least-squares fit of log Var = a + r log N.  Returns (r, exp(a), R^2).

    R^2 is reported as NaN with fewer than three points, where a straight line
    fits exactly by construction and an R^2 of 1.0 would be pure self-flattery.
    """
    x = np.log(np.asarray(ns, float))
    y = np.log(np.asarray(variances, float))
    good = np.isfinite(x) & np.isfinite(y) & (np.asarray(variances, float) > 0)
    if good.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    x, y = x[good], y[good]
    r, a = np.polyfit(x, y, 1)
    if good.sum() < 3:
        return float(r), float(np.exp(a)), float("nan")
    resid = y - (r * x + a)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return float(r), float(np.exp(a)), r2


def work_for_epsilon(rate, const, rel_eps, mean, n_steps, paths_per_sample,
                     n_floor=1.0):
    """Work (timestep-path evaluations) to reach relative RMSE `rel_eps`.

    Var(N) = const * N^rate, target Var = (rel_eps * |mean|)^2, so
    N = (target / const)^(1/rate).  Same work convention as the crossover
    table: W = N * n_steps, scaled by the path-equivalents each sample costs.

    N is floored at ONE sample unit.  Without the floor the antithetic arm at
    sigma=0.1 reports W = 2e-7 timestep-path evaluations -- a fifth of a
    millionth of a single path -- and a "540,000,000x cheaper than MC" ratio.
    Both are artifacts of extrapolating Var = const/N below one sample, where it
    has no operational meaning: you cannot buy less than one path.  The honest
    statement in that regime is that the very first sample already overshoots
    the target, which `floored` records and which the caller reports instead of
    the extrapolated ratio.
    """
    if not (np.isfinite(rate) and np.isfinite(const)) or rate >= 0 or const <= 0:
        return float("nan"), float("nan"), False
    target = (rel_eps * abs(mean)) ** 2
    n_req = (target / const) ** (1.0 / rate)
    floored = n_req < n_floor
    n_req = max(n_floor, n_req)
    return n_req, n_req * n_steps * paths_per_sample, floored


def achieved_rel_rmse(const, mean):
    """Relative RMSE delivered by ONE sample unit -- the accuracy floor of an
    arm, and the only meaningful summary when the target is already met by the
    first sample."""
    if not np.isfinite(const) or const <= 0 or not mean:
        return float("nan")
    return math.sqrt(const) / abs(mean)


# ---------------------------------------------------------------------------
def recompute_from_json(out_dir, args):
    """Re-derive the work table from a completed run without re-simulating.

    Everything downstream of the measurement -- work at each accuracy target,
    ratios against MC, floor flags -- is a pure function of the measured
    (rate, const, paths_per_sample) triple and the reference mean, all of which
    are stored.  Re-deriving them costs milliseconds where re-running the
    simulation costs twenty minutes, and because the run is deterministic the
    result is identical to what a re-run would produce.  Used when the REPORTING
    rule changes (e.g. how the sample floor is applied), not when the model does.
    """
    path = Path(out_dir) / "qmc_baseline.json"
    with open(path) as f:
        payload = json.load(f)
    cfg = payload["provenance"]["config"]
    n_steps = int(cfg["n_steps"])
    budgets = list(cfg["budgets"])
    rel_eps = list(args.rel_eps) if args.rel_eps else list(cfg["rel_eps"])
    rows = []
    for r in payload["summary"]:
        ref_mean = r["ref_mean"]
        arm = r["method"]
        n_floor = float(min(budgets)) if arm == "qmc" else 1.0
        row = dict(r)
        row["n_floor"] = n_floor
        row["rel_rmse_one_sample"] = (
            float("nan") if arm == "qmc"
            else achieved_rel_rmse(r["const"], ref_mean))
        for eps in rel_eps:
            n_req, w_req, floored = work_for_epsilon(
                r["rate"], r["const"], eps, ref_mean, n_steps,
                r["paths_per_sample"], n_floor=n_floor)
            row[f"N_eps_{eps:g}"] = n_req
            row[f"W_eps_{eps:g}"] = w_req
            row[f"floored_eps_{eps:g}"] = floored
        rows.append(row)
    for sigma in sorted({r["sigma"] for r in rows}):
        base = next((r for r in rows if r["sigma"] == sigma and r["method"] == "mc"), None)
        for r in rows:
            if r["sigma"] != sigma:
                continue
            for eps in rel_eps:
                b = base.get(f"W_eps_{eps:g}") if base else float("nan")
                w = r.get(f"W_eps_{eps:g}")
                r[f"work_ratio_vs_mc_eps_{eps:g}"] = (
                    b / w if base and w and w > 0 and np.isfinite(b) else float("nan"))
    payload["summary"] = rows
    payload["recomputed_utc"] = datetime.now(timezone.utc).isoformat()
    payload["recompute_note"] = ("work columns re-derived from the stored measured "
                                 "(rate, const); no simulation was re-run")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    keys = sorted({k for r in rows for k in r})
    with open(Path(out_dir) / "summary.csv", "w", newline="") as f:
        for line in provenance_comment_lines(payload["provenance"]):
            f.write(line + "\n")
        f.write("# work columns re-derived from stored measurements; no re-simulation\n")
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print("=" * 92)
    print("RECOMPUTED from stored measurements (no simulation re-run)")
    for sigma in sorted({r["sigma"] for r in rows}):
        print(f"\n  sigma = {sigma}")
        hdr = f"    {'method':>10} {'rate':>8} {'const':>11}"
        for eps in rel_eps:
            hdr += f" {'W(e=%g)' % eps:>12} {'x MC':>7}"
        print(hdr)
        for r in rows:
            if r["sigma"] != sigma:
                continue
            line = f"    {r['method']:>10} {r['rate']:+8.3f} {r['const']:11.3e}"
            for eps in rel_eps:
                ratio = ("  floor" if r.get(f"floored_eps_{eps:g}")
                         else f"{r[f'work_ratio_vs_mc_eps_{eps:g}']:7.2f}")
                line += f" {r[f'W_eps_{eps:g}']:12.3e} {ratio:>7}"
            print(line)
        for r in rows:
            if r["sigma"] == sigma and np.isfinite(r["rel_rmse_one_sample"]):
                print(f"      {r['method']:>10}  rel RMSE from one sample unit: "
                      f"{r['rel_rmse_one_sample']:.2e}")
    print(f"\nRewrote {out_dir}/summary.csv and qmc_baseline.json")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-nodes", type=int, default=200)
    ap.add_argument("--topology", default="er")
    ap.add_argument("--p-er", type=float, default=None,
                    help="ER edge probability; default gives avg degree 10")
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.433],
                    help="manuscript default and the variance-calibrated value")
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--base-dt", type=float, default=0.05)
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048],
                    help="QMC budgets N (powers of two, for Sobol balance)")
    ap.add_argument("--reps", type=int, default=24,
                    help="independent Owen scramblings per QMC budget")
    ap.add_argument("--var-samples", type=int, default=20000,
                    help="batch size for the i.i.d. arms' per-sample variance")
    ap.add_argument("--cv-pilot", type=int, default=512)
    ap.add_argument("--rate-boot", type=int, default=2000,
                    help="bootstrap resamples for the QMC rate CI")
    ap.add_argument("--rel-eps", type=float, nargs="+", default=[0.02, 0.01, 0.005])
    ap.add_argument("--ref-samples", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=2000,
                    help="max paths simulated at once (memory bound)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--quick", action="store_true",
                    help="tiny grid for smoke-testing")
    ap.add_argument("--recompute", action="store_true",
                    help="re-derive the work table from a stored run; no simulation")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if args.recompute:
        recompute_from_json(args.out, args)
        return

    if args.quick:
        # Fill in a small grid ONLY where the caller did not say otherwise --
        # blanket overriding meant `--quick --reps 24` silently ran 8 reps.
        given = {a.split("=")[0] for a in sys.argv[1:] if a.startswith("--")}
        for flag, attr, value in (("--n-nodes", "n_nodes", 40),
                                  ("--budgets", "budgets", [128, 256, 512]),
                                  ("--reps", "reps", 8),
                                  ("--ref-samples", "ref_samples", 2000),
                                  ("--sigmas", "sigmas", [0.433]),
                                  ("--cv-pilot", "cv_pilot", 128),
                                  ("--var-samples", "var_samples", 2000)):
            if flag not in given:
                setattr(args, attr, value)

    thread_info = set_blas_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_er = args.p_er if args.p_er is not None else min(1.0, 10.0 / max(args.n_nodes - 1, 1))
    adjacency, topo_used, topo_note = load_topology(args.topology, args.n_nodes,
                                                    args.seed, p_er=p_er)
    n_nodes = adjacency.shape[0]
    influence = influence_matrix_from_adjacency(adjacency, args.alpha)
    lam = np.full(n_nodes, args.lam)
    dt = args.base_dt
    n_steps = int(round(args.T / dt))
    schedule = bridge_schedule(n_steps)

    cfg = {"n_nodes": n_nodes, "topology": topo_used, "p_er": p_er,
           "lam": args.lam, "alpha": args.alpha, "beta": args.beta,
           "sigmas": args.sigmas, "T": args.T, "base_dt": dt, "n_steps": n_steps,
           "budgets": args.budgets, "reps": args.reps, "cv_pilot": args.cv_pilot,
           "ref_samples": args.ref_samples, "rel_eps": args.rel_eps,
           "arms": args.arms, "seed": args.seed, "quick": args.quick,
           "estimand": "terminal mean_i C_i(T)"}
    prov = build_provenance(cfg, thread_info, device="cpu")
    print("\n".join(provenance_comment_lines(prov)), flush=True)
    if topo_note:
        print(f"# topology note: {topo_note}", flush=True)
    print(f"# path dimension = n_steps * n_nodes = {n_steps * n_nodes}"
          f"  (Sobol cap {SOBOL_MAXDIM})", flush=True)

    rng_master = np.random.default_rng(args.seed)
    assert_matches_reference(influence, args.beta, args.sigmas[0], dt, lam,
                             rng_master)
    print("# batched EM step verified against simulation.qmc.coupled_em_step",
          flush=True)

    raw_rows, summary_rows = [], []
    refs = {}

    for sigma in args.sigmas:
        ctx = {"influence": influence, "beta": args.beta, "sigma": sigma,
               "dt": dt, "lam": lam, "n_steps": n_steps, "n_nodes": n_nodes,
               "schedule": schedule, "cv_pilot": args.cv_pilot,
               "control_mean": fluid_limit_mean(influence, args.beta, lam,
                                                args.T, dt, terminal=True)}

        # High-precision reference on the SAME grid: isolates variance-reduction
        # efficiency from discretisation bias, which every arm shares.
        # Built with ANTITHETIC pairs: the reference must be much tighter than
        # the biases it is used to judge, and pairing buys that at no extra cost.
        # Its own standard error is reported alongside, because a "bias" smaller
        # than the reference's standard error is a statement about the reference,
        # not about the arm being scored.
        t0 = time.time()
        ys, clamp_acc, seen = [], 0.0, 0
        rng_ref = np.random.default_rng(args.seed + 9973)
        n_ref_pairs = max(1, args.ref_samples // 2)
        for m in _chunks(n_ref_pairs, max(1, args.chunk // 2)):
            dw = rng_ref.normal(0.0, math.sqrt(dt), (m, n_steps, n_nodes))
            y, _, cf = run_paths(np.concatenate([dw, -dw], axis=0), influence,
                                 args.beta, sigma, dt, lam)
            ys.append(0.5 * (y[:m] + y[m:]))
            clamp_acc += cf * m
            seen += m
        y_ref = np.concatenate(ys)
        ref_mean = float(y_ref.mean())
        ref_stderr = float(np.std(y_ref, ddof=1) / math.sqrt(y_ref.size))
        ref_clamp = clamp_acc / seen
        refs[sigma] = {"mean": ref_mean, "stderr": ref_stderr,
                       "n_pairs": int(y_ref.size),
                       "reflection_bind_fraction": ref_clamp}
        print(f"\nsigma={sigma}: reference mean {ref_mean:.6f} "
              f"+/- {ref_stderr:.2e} (antithetic, {y_ref.size} pairs, "
              f"{time.time()-t0:.1f}s), control mean {ctx['control_mean']:.6f}, "
              f"reflection bind fraction {ref_clamp:.2e}", flush=True)

        for arm in args.arms:
            t1 = time.time()
            # zlib.crc32, not hash(): str.__hash__ is salted per process, so
            # seeding from it makes the whole run irreproducible between
            # invocations -- which it silently was until this was caught.
            rng = np.random.default_rng(
                [args.seed, int(sigma * 1e6), zlib.crc32(arm.encode())])
            if arm == "qmc":
                res = arm_qmc(ctx, rng, args, raw_rows, sigma, ref_mean)
            else:
                res = IID_ARMS[arm](ctx, rng, args)
                raw_rows.append({
                    "sigma": sigma, "method": arm, "N": res["n_var_samples"],
                    "rep": 0, "estimate": res["mean_estimate"],
                    "paths_charged": res["n_var_samples"] * res["paths_per_sample"],
                    "work_units": res["n_var_samples"] * res["paths_per_sample"] * n_steps,
                    "ref_mean": ref_mean,
                    "abs_err": abs(res["mean_estimate"] - ref_mean)})
                print(f"  {arm:>10} per-{res['sample_unit']} variance "
                      f"{res['const']:.3e}  (batch {res['n_var_samples']}, "
                      f"{res['paths_per_sample']:.0f} path(s)/sample)", flush=True)

            row = {"sigma": sigma, "method": arm, "ref_mean": ref_mean,
                   "ref_stderr": ref_stderr,
                   "bias_vs_ref": res["mean_estimate"] - ref_mean,
                   "elapsed_s": time.time() - t1, **res}
            n_floor = float(min(args.budgets)) if arm == "qmc" else 1.0
            row["n_floor"] = n_floor
            row["rel_rmse_one_sample"] = (
                float("nan") if arm == "qmc"
                else achieved_rel_rmse(res["const"], ref_mean))
            for eps in args.rel_eps:
                n_req, w_req, floored = work_for_epsilon(
                    res["rate"], res["const"], eps, ref_mean, n_steps,
                    res["paths_per_sample"], n_floor=n_floor)
                row[f"N_eps_{eps:g}"] = n_req
                row[f"W_eps_{eps:g}"] = w_req
                row[f"floored_eps_{eps:g}"] = floored
            summary_rows.append(row)
            r2 = res["rate_r2"]
            ci = ("" if res.get("rate_ci_lower") is None else
                  " 95%% CI [%+.3f, %+.3f]" % (res["rate_ci_lower"],
                                               res["rate_ci_upper"]))
            print(f"  {arm:>10} rate r={res['rate']:+.3f}"
                  f"{'' if r2 is None else ' (fitted, R^2=%.3f)' % r2}{ci}"
                  f"  const={res['const']:.3e}  [{time.time()-t1:.1f}s]", flush=True)

    # ---- work ratios vs plain MC at each epsilon -------------------------
    for sigma in args.sigmas:
        base = next((r for r in summary_rows
                     if r["sigma"] == sigma and r["method"] == "mc"), None)
        for r in summary_rows:
            if r["sigma"] != sigma:
                continue
            for eps in args.rel_eps:
                key = f"W_eps_{eps:g}"
                b = base.get(key) if base else float("nan")
                r[f"work_ratio_vs_mc_eps_{eps:g}"] = (
                    b / r[key] if base and r.get(key) and r[key] > 0
                    and np.isfinite(b) else float("nan"))

    # ---- write ------------------------------------------------------------
    with open(out_dir / "raw.csv", "w", newline="") as f:
        for line in provenance_comment_lines(prov):
            f.write(line + "\n")
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_rows)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        for line in provenance_comment_lines(prov):
            f.write(line + "\n")
        keys = sorted({k for r in summary_rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(summary_rows)
    with open(out_dir / "qmc_baseline.json", "w") as f:
        json.dump({"provenance": prov, "reference_means": refs,
                   "summary": summary_rows}, f, indent=2, default=float)

    print("\n" + "=" * 88)
    print("VARIANCE-REDUCTION BASELINES -- accuracy-matched work (>1 = better than MC)")
    for sigma in args.sigmas:
        print(f"\n  sigma = {sigma}   (reference mean {refs[sigma]['mean']:.6f}"
              f" +/- {refs[sigma]['stderr']:.1e})")
        hdr = f"    {'method':>10} {'rate':>7} {'paths/s':>8} {'const':>11}"
        for eps in args.rel_eps:
            hdr += f" {'W(e=%g)' % eps:>12} {'x MC':>7}"
        print(hdr)
        for r in summary_rows:
            if r["sigma"] != sigma:
                continue
            rate = f"{r['rate']:+.3f}" + ("" if r["rate_r2"] is None else "*")
            line = (f"    {r['method']:>10} {rate:>7} "
                    f"{r['paths_per_sample']:8.0f} {r['const']:11.3e}")
            for eps in args.rel_eps:
                ratio = ("  floor" if r.get(f"floored_eps_{eps:g}")
                         else f"{r[f'work_ratio_vs_mc_eps_{eps:g}']:7.2f}")
                line += f" {r[f'W_eps_{eps:g}']:12.3e} {ratio:>7}"
            print(line)
    print("\n  * = empirically fitted rate; unstarred rates are -1 exactly "
          "(sample mean of i.i.d. units).")
    print("  'floor' = one sample unit already beats the target, so no ratio is "
          "quoted; see rel_rmse_one_sample.")
    for sigma in args.sigmas:
        for r in summary_rows:
            if r["sigma"] == sigma:
                if np.isfinite(r["rel_rmse_one_sample"]):
                    print(f"    sigma={sigma:<6} {r['method']:>10}  relative RMSE from a "
                          f"single sample unit: {r['rel_rmse_one_sample']:.2e}")
                else:
                    print(f"    sigma={sigma:<6} {r['method']:>10}  fitted over N in "
                          f"[{min(args.budgets)}, {max(args.budgets)}]; not "
                          f"extrapolated below N={r['n_floor']:.0f}")
    print(f"\nWrote {out_dir}/raw.csv, summary.csv, qmc_baseline.json")


if __name__ == "__main__":
    main()
