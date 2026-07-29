#!/usr/bin/env python3
"""Validity envelope of the diffusion approximation (Reviewer 1, concern 4).

"The applicable regime of the coupled SDE model should be clarified. The ns-3
validation shows relatively large errors under some light-load and tail-risk
conditions."

The reviewer is right that the diffusion approximation is a heavy-traffic limit
and must not be presented as uniformly valid.  The original plan answered this
with an extended ns-3 sweep.  We answer it against a stronger reference instead:
the discrete-event baseline (scripts/run_des_baseline.py) simulates the
continuous-time Markov jump process whose diffusion limit *is* the paper's SDE,
so it is the exact model the approximation is an approximation OF -- not a
second, independently-calibrated simulator whose disagreements confound
approximation error with configuration mismatch.

Method.  Utilisation is rho = lambda / beta per node.  Holding beta fixed and
sweeping lambda walks the model from light load to near-saturation, and at each
point we compare the SDE against the exact jump process on:

    mean error      |E_SDE - E_DES| / E_DES
    variance ratio  Var_SDE / Var_DES     (the quantity that exposed the
                                           sigma miscalibration at rho = 0.6)
    Welch z         standardised mean gap, as a significance check

The output is a recommended operating band: the range of rho over which the
approximation's mean error stays inside a stated tolerance.

This wraps the DES script rather than reimplementing it, so both the exact model
and the SDE come from already-tested code.

Run:
    python scripts/run_validity_envelope.py --quick
    python scripts/run_validity_envelope.py

Out:
    results/validity_envelope/envelope.csv
    results/validity_envelope/validity_envelope.json
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DES = REPO / "scripts" / "run_des_baseline.py"
OUT = REPO / "results" / "validity_envelope"


def run_one_rho(rho, args, workdir):
    """Run the DES-vs-SDE comparison at one utilisation and return its rows."""
    lam = rho * args.beta
    out_dir = workdir / f"rho_{rho:.3f}"
    cmd = [
        sys.executable, str(DES),
        "--mode", "sde-compare",
        "--seeds", *[str(s) for s in args.seeds],
        "--n-nodes", str(args.n_nodes),
        "--topologies", "er",
        "--lam", f"{lam:.6f}",
        "--alpha", str(args.alpha),
        "--beta", str(args.beta),
        "--sigma", str(args.sigma),
        "--T", str(args.T),
        "--base-dt", str(args.base_dt),
        "--sde-compare-reps", str(args.reps),
        "--out", str(out_dir),
        "--no-resume",
    ]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    payload_path = out_dir / "des_baseline.json"
    if proc.returncode != 0 or not payload_path.exists():
        return [], (proc.stderr or proc.stdout)[-600:]
    with open(payload_path) as f:
        payload = json.load(f)
    return payload.get("sde_compare", []), None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rhos", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    ap.add_argument("--n-nodes", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.1,
                    help="manuscript default; the sweep is what shows where it fails")
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--base-dt", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--tolerance-pct", type=float, default=10.0,
                    help="mean-error tolerance defining the recommended band")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if args.quick:
        given = {a.split("=")[0] for a in sys.argv[1:] if a.startswith("--")}
        for flag, attr, val in (("--rhos", "rhos", [0.4, 0.8]),
                                ("--n-nodes", "n_nodes", 60),
                                ("--seeds", "seeds", [0]),
                                ("--reps", "reps", 40)):
            if flag not in given:
                setattr(args, attr, val)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    with tempfile.TemporaryDirectory(prefix="envelope_") as tmp:
        workdir = Path(tmp)
        for rho in args.rhos:
            cells, err = run_one_rho(rho, args, workdir)
            if err:
                errors.append({"rho": rho, "error": err})
                print(f"  rho={rho:.2f}  FAILED: {err.splitlines()[-1][:120]}",
                      flush=True)
                continue
            for c in cells:
                r = dict(c)
                r["rho"] = rho
                r["variance_ratio_sde_over_des"] = (
                    c["sde_var"] / c["des_var"] if c["des_var"] else float("nan"))
                rows.append(r)
            errs = [abs(c["rel_diff_mean_pct"]) for c in cells]
            vr = [c["sde_var"] / c["des_var"] for c in cells if c["des_var"]]
            print(f"  rho={rho:.2f}  mean err {np.mean(errs):6.2f}%   "
                  f"Var_SDE/Var_DES {np.mean(vr):7.4f}   "
                  f"(n={len(cells)} seeds)", flush=True)

    if not rows:
        print("No cells succeeded.")
        if errors:
            print(errors[0]["error"])
        return

    keys = sorted({k for r in rows for k in r})
    with open(out_dir / "envelope.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)

    # ---- recommended operating band --------------------------------------
    per_rho = {}
    for rho in sorted({r["rho"] for r in rows}):
        cells = [r for r in rows if r["rho"] == rho]
        per_rho[rho] = {
            "mean_abs_err_pct": float(np.mean([abs(c["rel_diff_mean_pct"]) for c in cells])),
            "max_abs_err_pct": float(np.max([abs(c["rel_diff_mean_pct"]) for c in cells])),
            "variance_ratio": float(np.mean([c["variance_ratio_sde_over_des"] for c in cells])),
            "n_seeds": len(cells),
        }
    inside = [r for r, d in per_rho.items()
              if d["mean_abs_err_pct"] <= args.tolerance_pct]
    band = (min(inside), max(inside)) if inside else None

    payload = {"config": vars(args), "rows": rows, "per_rho": per_rho,
               "recommended_band": band, "tolerance_pct": args.tolerance_pct,
               "errors": errors}
    with open(out_dir / "validity_envelope.json", "w") as f:
        json.dump(payload, f, indent=2, default=float)

    print("\n" + "=" * 72)
    print(f"VALIDITY ENVELOPE (sigma={args.sigma}, n={args.n_nodes}, "
          f"exact CTMC reference)")
    print(f"  {'rho':>6} {'mean err %':>11} {'max err %':>10} {'Var ratio':>11}")
    for rho, d in per_rho.items():
        print(f"  {rho:6.2f} {d['mean_abs_err_pct']:11.2f} "
              f"{d['max_abs_err_pct']:10.2f} {d['variance_ratio']:11.4f}")
    if band:
        print(f"\n  Recommended operating band (mean error <= "
              f"{args.tolerance_pct:g}%): rho in [{band[0]:.2f}, {band[1]:.2f}]")
    else:
        print(f"\n  No rho met the {args.tolerance_pct:g}% tolerance.")
    print(f"\nWrote {out_dir}/envelope.csv, validity_envelope.json")


if __name__ == "__main__":
    main()
