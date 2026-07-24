#!/usr/bin/env python3
"""Device-matched CPU vs GPU MLMC baseline (Reviewer 1, concern 5).

The reviewer asked for a CPU-MLMC baseline.  The repository already contained
one (`results/results/tables/cpu_vs_gpu_mlmc.json`), and it does not survive
inspection: its CPU and GPU arms report 0.0088 and 0.0638 for the same estimand,
a sevenfold disagreement, so the "speedup" it quotes compares two different
quantities -- and at tighter accuracy targets it shows the CPU beating the GPU.
The cause is that the two arms were different implementations (`MLMCSimulator`
vs `GPUCoupledPropagationMLMC`) estimating different functionals.

This script replaces it with the only comparison that isolates the accelerator:
ONE implementation, ONE estimand, ONE seed policy, run twice with
`GPUCoupledPropagationMLMC(..., device='cpu')` and `device='cuda'`.

Because both arms are the same code, the estimates must agree to within Monte
Carlo error.  That agreement is CHECKED, not assumed: `estimate_agreement_ok`
is reported per cell, and a run in which the arms disagree beyond the reported
tolerance is a broken measurement, not a speedup.

Run:
    python scripts/run_device_matched_mlmc.py --devices cpu          # local
    python scripts/run_device_matched_mlmc.py --devices cpu cuda     # on a GPU box

Out:
    results/baselines/device_matched/device_matched.csv
    results/baselines/device_matched/device_matched.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from simulation.qmc import (  # noqa: E402
    build_provenance,
    load_topology,
    provenance_comment_lines,
    set_blas_threads,
)

OUT = REPO / "results" / "baselines" / "device_matched"


def run_cell(adjacency, device, args, seed):
    """One (device, n, epsilon, seed) measurement."""
    from gpu.parallel_mc import GPUCoupledPropagationMLMC

    sim = GPUCoupledPropagationMLMC(
        adjacency,
        influence_strength=args.alpha,
        decay_rate=args.beta,
        noise_intensity=args.sigma,
        seed=seed,
        device=device,
    )
    if str(sim._device) != device:
        raise RuntimeError(f"requested device {device!r}, got {sim._device!r}")

    t0 = time.perf_counter()
    res = sim.mlmc_estimate(
        epsilon=args.epsilon_placeholder,
        T=args.T,
        base_dt=args.base_dt,
        L_max=args.L_max,
        pilot_samples=args.pilot_samples,
        metric=args.metric,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    work = res.get("total_cost") or res.get("total_work") or float("nan")
    return {
        "device": device,
        "wall_s": elapsed,
        "estimate": float(res.get("estimate", res.get("mean", float("nan")))),
        "work_units": float(work),
        "throughput_work_per_s": float(work) / elapsed if elapsed > 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", default=["cpu"],
                    choices=["cpu", "cuda"])
    ap.add_argument("--n-nodes", type=int, nargs="+", default=[200, 500, 1000])
    ap.add_argument("--epsilons", type=float, nargs="+", default=[0.02, 0.01, 0.005])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--topology", default="ba", choices=["ba", "er", "caida"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.433,
                    help="variance-calibrated value, matching Table 3")
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--base-dt", type=float, default=0.1)
    ap.add_argument("--L-max", type=int, default=4)
    ap.add_argument("--pilot-samples", type=int, default=50)
    ap.add_argument("--metric", default="mean_congestion")
    ap.add_argument("--agreement-rtol", type=float, default=0.05,
                    help="max relative gap between device estimates before the "
                         "cell is flagged as a broken measurement")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if args.quick:
        given = {a.split("=")[0] for a in sys.argv[1:] if a.startswith("--")}
        for flag, attr, val in (("--n-nodes", "n_nodes", [100]),
                                ("--epsilons", "epsilons", [0.02]),
                                ("--seeds", "seeds", [0])):
            if flag not in given:
                setattr(args, attr, val)

    thread_info = set_blas_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = build_provenance(vars(args).copy(), thread_info,
                            device="+".join(args.devices))
    print("\n".join(provenance_comment_lines(prov)), flush=True)

    rows = []
    for n in args.n_nodes:
        for eps in args.epsilons:
            args.epsilon_placeholder = eps
            for seed in args.seeds:
                adjacency, label, note = load_topology(args.topology, n, seed)
                if note:
                    print(f"  note: {note}", flush=True)
                per_dev = {}
                for dev in args.devices:
                    try:
                        per_dev[dev] = run_cell(adjacency, dev, args, seed)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  {label} n={n} eps={eps} seed={seed} "
                              f"{dev}: FAILED {exc}", flush=True)
                        per_dev[dev] = {"device": dev, "error": str(exc)}
                row = {"topology": label, "n_nodes": n, "epsilon": eps,
                       "seed": seed}
                for dev, d in per_dev.items():
                    for k, v in d.items():
                        if k != "device":
                            row[f"{dev}_{k}"] = v
                if ("cpu" in per_dev and "cuda" in per_dev
                        and "error" not in per_dev["cpu"]
                        and "error" not in per_dev["cuda"]):
                    a, b = per_dev["cpu"]["estimate"], per_dev["cuda"]["estimate"]
                    denom = max(abs(a), abs(b), 1e-30)
                    rel = abs(a - b) / denom
                    row["estimate_rel_gap"] = rel
                    row["estimate_agreement_ok"] = bool(rel <= args.agreement_rtol)
                    row["speedup_cpu_over_gpu"] = (
                        per_dev["cpu"]["wall_s"] / per_dev["cuda"]["wall_s"]
                        if per_dev["cuda"]["wall_s"] > 0 else float("nan"))
                rows.append(row)
                msg = f"  {label} n={n:>5} eps={eps:<6g} seed={seed}"
                for dev in args.devices:
                    d = per_dev[dev]
                    if "error" not in d:
                        msg += f"  {dev}: {d['wall_s']:7.2f}s est={d['estimate']:.5f}"
                if "speedup_cpu_over_gpu" in row:
                    flag = "" if row["estimate_agreement_ok"] else "  [ESTIMATES DISAGREE]"
                    msg += f"  speedup={row['speedup_cpu_over_gpu']:.1f}x{flag}"
                print(msg, flush=True)

    if not rows:
        print("no cells ran")
        return
    keys = sorted({k for r in rows for k in r})
    with open(out_dir / "device_matched.csv", "w", newline="") as f:
        for line in provenance_comment_lines(prov):
            f.write(line + "\n")
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "device_matched.json", "w") as f:
        json.dump({"provenance": prov, "rows": rows}, f, indent=2, default=float)

    ok = [r for r in rows if r.get("estimate_agreement_ok") is True]
    bad = [r for r in rows if r.get("estimate_agreement_ok") is False]
    if ok:
        sp = np.array([r["speedup_cpu_over_gpu"] for r in ok], dtype=float)
        sp = sp[np.isfinite(sp)]
        print(f"\nGPU speedup over CPU on identical code: "
              f"median {np.median(sp):.1f}x, range {sp.min():.1f}-{sp.max():.1f}x "
              f"over {len(sp)} agreeing cells")
    if bad:
        print(f"WARNING: {len(bad)} cell(s) had disagreeing estimates and are "
              f"excluded from the speedup summary -- these are broken "
              f"measurements, not results")
    print(f"\nWrote {out_dir}/device_matched.csv, device_matched.json")


if __name__ == "__main__":
    main()
