"""
Optimized network discrete-event simulation baseline for the coupled
congestion-propagation SDE (paper Eq. 6-7, "Coupled Congestion Propagation
SDE" section) -- Reviewer 1, concern 5: "More baselines should be added,
such as CPU-MLMC, optimized discrete-event simulation, existing GPU network
simulators, or other uncertainty quantification methods."

Model.
The coupled SDE is
    dC_i(t) = ( sum_j alpha_ij C_j(t) - beta_i C_i(t) ) dt + sigma_i dW_i(t),
    alpha_ij = alpha * A_ij / deg(i)
Standard density-dependent Markov-chain theory (Kurtz 1970/1978 -- the same
"diffusion limit of a jump process" construction behind the M/M/1
heavy-traffic diffusion approximation already used elsewhere in this repo,
see validation/python_des_bottleneck.py) identifies the SDE as the diffusion
limit of the following continuous-time Markov jump process on integer
congestion counts C_i(t) in {0,1,2,...}:

  exogenous arrival at i        rate lambda_i              C_i -> C_i + 1
  coupling arrival at i (from
    neighbour j's congestion)   rate alpha_ij * C_j(t)      C_i -> C_i + 1
  decay / departure at i        rate beta_i * C_i(t)        C_i -> C_i - 1
                                 [--service-dist exponential, default]
                             OR: each arrived unit departs exactly
                                 1/beta_i time units after it arrived
                                 [--service-dist deterministic]

lambda_i supplies the task's required "Poisson arrivals rate lambda_i".
beta_i*C_i (exponential mode) is the discrete analogue of "service rate mu":
a state-dependent, memoryless per-unit departure -- the network generalises
python_des_bottleneck.py's single-queue M/D/1-vs-SDE comparison to n coupled
queues. Deterministic mode gives each unit a FIXED 1/beta_i sojourn instead
(an M/D/inf-style decay), matching "deterministic ... service rate mu"
literally.

What makes this optimized (as required by the task):
  1. Calendar queue for the exogenous-arrival stream: a heapq of plain
     (time, node) tuples, no custom Event objects. Rate lambda_i never
     changes, so channels are simply redrawn (not rescheduled) when they
     fire -- O(log n) push, zero rescaling logic.
  2. State-dependent coupling/decay rates are maintained incrementally in
     numpy arrays: a single node's congestion change updates only its
     O(deg(node)) neighbours' coupling-arrival rates (fancy indexing on the
     adjacency's sparse neighbour list) rather than recomputing an O(n)
     rate vector from scratch every event, and rather than the O(n) per
     packet object churn a naive per-packet DES would incur.
  3. Channel selection for the state-dependent (Gillespie) step is one
     vectorised cumsum+searchsorted call over the current rate arrays, not
     a per-event Python comparison loop.
  4. Exponential/uniform random draws are pulled from the RNG in batches
     (chunks of 2^13) and consumed from a buffer, amortising per-call
     overhead instead of calling the RNG once per event.
  5. Deterministic-service departures use a SECOND plain-tuple calendar
     queue (no per-token Python objects beyond a (time, node) tuple).
This is EXACT stochastic simulation (Gillespie's Direct Method interleaved
with two calendar queues) -- no time-discretisation bias, which is the
point of using it as ground truth against the Euler-Maruyama SDE.

Two experiments:
  --mode accuracy   Time-to-target-accuracy: replicate the DES until the
                     across-replication CI half-width on the target
                     functional (time-and-node-averaged congestion, see
                     src/simulation/qmc.py:simulate_functional) reaches each
                     requested epsilon. Reports wall-clock, events processed,
                     replications needed, alongside the lambda-forced
                     coupled-model MLMC reference from
                     src/simulation/qmc.py:coupled_mlmc_estimate (same
                     topology/seed/functional/epsilon).
  --mode sde-compare At a fixed "moderate load" operating point, runs DES
                     replications and Euler-Maruyama SDE replications side
                     by side and reports mean/variance for both -- directly
                     answers whether DES and SDE queue statistics agree at
                     moderate load, or diverge (a finding about the SDE's
                     validity regime).
  --mode all         both (default).

Usage:
    python scripts/run_des_baseline.py --quick
    python scripts/run_des_baseline.py --seeds 0 1 2 3 4 --n-nodes 100 500 \\
        --epsilons 0.05 0.02 0.01 --topologies er caida

Output (under --out, default results/baselines/des):
    des_baseline.json     full results + provenance
    accuracy.csv          time-to-epsilon rows (DES + coupled-MLMC reference)
    sde_compare.csv        DES vs SDE moderate-load comparison rows
    checkpoint.jsonl       append-only resume log
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import sys
import time

# Set BLAS/OMP thread env vars BEFORE importing numpy/scipy, so the pin is
# effective for backends that read them only at first initialisation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from simulation.qmc import prescan_threads_arg, set_blas_threads  # noqa: E402

_THREADS_PRESCAN = prescan_threads_arg(sys.argv)
_THREAD_INFO = set_blas_threads(_THREADS_PRESCAN)

import numpy as np  # noqa: E402
from scipy import stats as sp_stats  # noqa: E402

from simulation import qmc as qmc_lib  # noqa: E402

SCHEMA_VERSION = 1

DEFAULT_EPSILONS = [0.05, 0.02, 0.01]
DEFAULT_N_NODES = [30, 100, 500]
DEFAULT_N_NODES_QUICK = [12]
DEFAULT_TOPOLOGIES = ["er"]
DEFAULT_P_ER = None  # None -> derived from --avg-degree


# ============================================================================
# Batched RNG draws (optimization item 4 in the module docstring)
# ============================================================================
class _BatchDraws:
    """Pull exponential(1)/uniform[0,1) draws from a refillable buffer to
    amortise per-call RNG overhead across millions of Gillespie events."""

    def __init__(self, rng: np.random.Generator, kind: str, chunk: int = 8192):
        self.rng, self.kind, self.chunk = rng, kind, chunk
        self._buf = np.empty(0)
        self._i = 0

    def next(self) -> float:
        if self._i >= len(self._buf):
            self._buf = (self.rng.exponential(1.0, self.chunk) if self.kind == "exp"
                         else self.rng.random(self.chunk))
            self._i = 0
        v = self._buf[self._i]
        self._i += 1
        return float(v)


# ============================================================================
# Exact network DES (Gillespie + calendar queues)
# ============================================================================
class NetworkDES:
    """Exact discrete-event simulation of the CTMC whose diffusion limit is
    the coupled congestion-propagation SDE (see module docstring)."""

    def __init__(self, adjacency: np.ndarray, lambda_vec: np.ndarray,
                 alpha: float, beta, seed: int,
                 service_dist: str = "exponential"):
        self.n = adjacency.shape[0]
        self.adjacency = adjacency
        self.influence = qmc_lib.influence_matrix_from_adjacency(adjacency, alpha)
        self.lambda_vec = np.asarray(lambda_vec, dtype=float)
        self.beta = (np.full(self.n, float(beta)) if np.isscalar(beta)
                     else np.asarray(beta, dtype=float))
        if service_dist not in ("exponential", "deterministic"):
            raise ValueError("service_dist must be 'exponential' or 'deterministic'")
        self.service_dist = service_dist
        self.rng = np.random.default_rng(seed)
        # Sparse neighbour lists for O(deg) incremental rate updates.
        self._neighbors = [np.nonzero(adjacency[:, i])[0] for i in range(self.n)]
        self._exp_draws = _BatchDraws(self.rng, "exp")
        self._u_draws = _BatchDraws(self.rng, "uniform")
        self.reset()

    def reset(self) -> None:
        n = self.n
        self.C = np.zeros(n, dtype=np.int64)
        self.birth_state_rate = np.zeros(n, dtype=float)   # coupling-arrival rate per node
        self.death_rate = np.zeros(n, dtype=float)         # decay rate per node (exponential mode)
        self.t = 0.0
        self.events_processed = 0
        self._exo_heap = []
        for i in range(n):
            if self.lambda_vec[i] > 0:
                dt0 = self._exp_draws.next() / self.lambda_vec[i]
                heapq.heappush(self._exo_heap, (dt0, i))
        self._dep_heap = []  # only used in deterministic mode

    def _apply_delta(self, i: int, delta: int) -> None:
        """C[i] += delta; O(deg(i)) incremental rate-array update."""
        self.C[i] += delta
        nbrs = self._neighbors[i]
        if nbrs.size:
            self.birth_state_rate[nbrs] += self.influence[nbrs, i] * delta
            # Clip float noise from repeated +/- increments; rates cannot be negative.
            np.maximum(self.birth_state_rate[nbrs], 0.0, out=self.birth_state_rate[nbrs])
        if self.service_dist == "exponential":
            self.death_rate[i] = self.beta[i] * self.C[i]

    def _on_arrival(self, i: int, t: float) -> None:
        self._apply_delta(i, +1)
        if self.service_dist == "deterministic":
            heapq.heappush(self._dep_heap, (t + 1.0 / self.beta[i], i))

    def step(self, T: float) -> bool:
        """Advance exactly one event. Returns False if none remain before T."""
        total_state_rate = float(self.birth_state_rate.sum() + self.death_rate.sum())
        t_state = (self.t + self._exp_draws.next() / total_state_rate
                   if total_state_rate > 0 else np.inf)
        t_exo = self._exo_heap[0][0] if self._exo_heap else np.inf
        t_det = self._dep_heap[0][0] if self._dep_heap else np.inf

        candidates = (t_state, t_exo, t_det)
        which = int(np.argmin(candidates))
        t_next = candidates[which]
        if t_next > T or not np.isfinite(t_next):
            return False

        self.t = t_next
        self.events_processed += 1

        if which == 1:
            _, i = heapq.heappop(self._exo_heap)
            self._on_arrival(i, self.t)
            dt_next = self._exp_draws.next() / self.lambda_vec[i]
            heapq.heappush(self._exo_heap, (self.t + dt_next, i))
        elif which == 2:
            _, i = heapq.heappop(self._dep_heap)
            self._apply_delta(i, -1)
        else:
            u = self._u_draws.next() * total_state_rate
            birth_total = float(self.birth_state_rate.sum())
            if u < birth_total:
                idx = int(np.searchsorted(np.cumsum(self.birth_state_rate), u))
                self._on_arrival(min(idx, self.n - 1), self.t)
            else:
                idx = int(np.searchsorted(np.cumsum(self.death_rate), u - birth_total))
                self._apply_delta(min(idx, self.n - 1), -1)
        return True

    def run_and_sample(self, T: float, sample_dt: float, warmup: float) -> float:
        """Run from t=0 to T, periodically sampling mean-node congestion
        (python_des_bottleneck.py's sampling convention: value held constant
        since the last event is recorded for every sample time it spans),
        and return the time-averaged post-warmup functional (a single
        scalar, matching src/simulation/qmc.py:simulate_functional)."""
        acc, count = 0.0, 0
        t_sample = sample_dt
        while self.t < T:
            prev_mean = float(np.mean(self.C))
            if not self.step(T):
                break
            while t_sample <= self.t and t_sample <= T:
                if t_sample >= warmup:
                    acc += prev_mean
                    count += 1
                t_sample += sample_dt
        return acc / max(count, 1)


# ============================================================================
# Replication-based "time to target accuracy"
# ============================================================================
def run_des_to_epsilon(adjacency: np.ndarray, lambda_vec: np.ndarray,
                        alpha: float, beta: float, T: float, sample_dt: float,
                        warmup: float, service_dist: str, epsilon: float,
                        base_seed: int, confidence_level: float = 0.95,
                        min_reps: int = 8, max_reps: int = 20000) -> dict:
    """Replicate NetworkDES until the across-replication CI half-width on
    the target functional reaches `epsilon`. Returns wall-clock, events
    processed, replications used, and per-replication raw values."""
    z = float(sp_stats.norm.ppf(1 - (1 - confidence_level) / 2))
    values, events = [], []
    t0 = time.perf_counter()
    converged = False
    rep = 0
    while rep < max_reps:
        des = NetworkDES(adjacency, lambda_vec, alpha, beta,
                         seed=base_seed + rep, service_dist=service_dist)
        y = des.run_and_sample(T, sample_dt, warmup)
        values.append(y)
        events.append(des.events_processed)
        rep += 1
        if rep >= min_reps:
            se = float(np.std(values, ddof=1)) / np.sqrt(rep)
            half_width = z * se
            if half_width <= epsilon:
                converged = True
                break
    wall_s = time.perf_counter() - t0
    return {
        "epsilon": epsilon, "converged": converged, "replications": rep,
        "total_events": int(sum(events)), "wall_s": wall_s,
        "estimate": float(np.mean(values)),
        "ci_halfwidth": float(z * np.std(values, ddof=1) / np.sqrt(rep)) if rep > 1 else float("nan"),
        "sample_values": values, "sample_events": events,
    }


# ============================================================================
# DES vs SDE moderate-load comparison
# ============================================================================
def run_sde_compare(adjacency: np.ndarray, lambda_vec: np.ndarray, alpha: float,
                     beta: float, sigma: float, T: float, dt: float,
                     sample_dt: float, warmup: float, service_dist: str,
                     n_reps: int, base_seed: int) -> dict:
    """Independent DES and Euler-Maruyama-SDE replications at the same
    (lambda, alpha, beta, sigma, topology) operating point; returns
    per-method mean/variance of the target functional for direct
    comparison."""
    influence = qmc_lib.influence_matrix_from_adjacency(adjacency, alpha)

    des_values, sde_values = [], []
    t0 = time.perf_counter()
    for r in range(n_reps):
        des = NetworkDES(adjacency, lambda_vec, alpha, beta,
                         seed=base_seed + r, service_dist=service_dist)
        des_values.append(des.run_and_sample(T, sample_dt, warmup))
    des_wall_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_steps = int(T / dt)
    for r in range(n_reps):
        rng = np.random.default_rng(base_seed + 1_000_000 + r)
        dw = rng.normal(0.0, np.sqrt(dt), (n_steps, adjacency.shape[0]))
        sde_values.append(qmc_lib.simulate_functional(
            influence, beta, sigma, lambda_vec, T, dt, dw, warmup / T))
    sde_wall_s = time.perf_counter() - t0

    des_mean, des_var = float(np.mean(des_values)), float(np.var(des_values, ddof=1))
    sde_mean, sde_var = float(np.mean(sde_values)), float(np.var(sde_values, ddof=1))
    rel_diff_mean = abs(des_mean - sde_mean) / max(abs(des_mean), 1e-9)
    pooled_se = float(np.sqrt(des_var / n_reps + sde_var / n_reps))
    z_stat = (des_mean - sde_mean) / pooled_se if pooled_se > 0 else float("nan")

    return {
        "n_reps": n_reps, "des_mean": des_mean, "des_var": des_var,
        "des_wall_s": des_wall_s, "sde_mean": sde_mean, "sde_var": sde_var,
        "sde_wall_s": sde_wall_s, "rel_diff_mean_pct": rel_diff_mean * 100.0,
        "welch_z_stat": z_stat,
        "diverges_wildly": bool(rel_diff_mean > 0.25 or abs(z_stat) > 5),
        "des_values": des_values, "sde_values": sde_values,
    }


# ============================================================================
# CLI plumbing
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["accuracy", "sde-compare", "all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-nodes", type=int, nargs="+", default=None)
    p.add_argument("--topologies", nargs="+", choices=["er", "caida"], default=None)
    p.add_argument("--epsilons", type=float, nargs="+", default=DEFAULT_EPSILONS)
    p.add_argument("--avg-degree", type=float, default=10.0,
                   help="target ER average degree (p_er = avg_degree/(n_nodes-1)); "
                        "controls DES's O(deg) per-event cost -- keep this modest for "
                        "large n_nodes unless explicitly probing dense-topology cost")
    p.add_argument("--p-er", type=float, default=None,
                   help="override ER edge probability directly (takes precedence over --avg-degree)")
    p.add_argument("--lam", type=float, default=0.3, help="uniform per-node exogenous arrival rate")
    p.add_argument("--alpha", type=float, default=0.1, help="coupling strength (paper's influence_strength)")
    p.add_argument("--beta", type=float, default=0.5, help="decay rate (paper's decay_rate); must exceed alpha for stability")
    p.add_argument("--sigma", type=float, default=0.1, help="SDE noise intensity (constant, as in CongestionPropagationSDE)")
    p.add_argument("--service-dist", choices=["exponential", "deterministic"], default="exponential")
    p.add_argument("--T", type=float, default=None, help="simulation horizon")
    p.add_argument("--base-dt", type=float, default=0.05, help="MLMC base_dt / SDE Euler-Maruyama dt")
    p.add_argument("--sample-dt", type=float, default=0.1, help="DES periodic-sampling interval")
    p.add_argument("--warmup-frac", type=float, default=0.2)
    p.add_argument("--L-max", type=int, default=3, help="coupled-MLMC reference max level")
    p.add_argument("--pilot-samples", type=int, default=30, help="coupled-MLMC reference pilot samples/level")
    p.add_argument("--sde-compare-reps", type=int, default=200)
    p.add_argument("--min-reps", type=int, default=8)
    p.add_argument("--max-reps", type=int, default=20000)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default="cpu",
                   help="informational only -- this baseline is pure-CPU exact "
                        "stochastic simulation (no tensor ops to place on a GPU)")
    p.add_argument("--out", default="results/baselines/des")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def build_config(args) -> dict:
    n_nodes = args.n_nodes or (DEFAULT_N_NODES_QUICK if args.quick else DEFAULT_N_NODES)
    topologies = args.topologies or DEFAULT_TOPOLOGIES
    epsilons = [0.1] if args.quick else args.epsilons
    T = args.T if args.T is not None else (2.0 if args.quick else 20.0)
    sde_reps = 10 if args.quick else args.sde_compare_reps
    max_reps = 50 if args.quick else args.max_reps
    return {
        "mode": args.mode, "seeds": args.seeds, "n_nodes": n_nodes,
        "topologies": topologies, "epsilons": epsilons, "lam": args.lam,
        "alpha": args.alpha, "beta": args.beta, "sigma": args.sigma,
        "service_dist": args.service_dist, "T": T, "base_dt": args.base_dt,
        "sample_dt": args.sample_dt, "warmup_frac": args.warmup_frac,
        "L_max": args.L_max, "pilot_samples": args.pilot_samples,
        "sde_compare_reps": sde_reps, "min_reps": args.min_reps,
        "max_reps": max_reps, "avg_degree": args.avg_degree, "p_er": args.p_er,
        "quick": args.quick,
    }


def p_er_for(cfg: dict, n_nodes: int) -> float:
    if cfg["p_er"] is not None:
        return cfg["p_er"]
    return min(1.0, cfg["avg_degree"] / max(n_nodes - 1, 1))


def main():
    args = parse_args()
    if args.alpha >= args.beta:
        raise SystemExit(f"--alpha ({args.alpha}) must be < --beta ({args.beta}) "
                          f"for the coupled process to be stable (else congestion diverges)")

    cfg = build_config(args)
    prov = qmc_lib.build_provenance(cfg, _THREAD_INFO, device=args.device)
    os.makedirs(args.out, exist_ok=True)
    checkpoint_path = os.path.join(args.out, "checkpoint.jsonl")
    if args.no_resume and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("  [checkpoint] --no-resume: cleared previous log", flush=True)
    fingerprint = qmc_lib.config_fingerprint(cfg)
    done = qmc_lib.load_checkpoint(checkpoint_path, fingerprint, SCHEMA_VERSION)
    if done:
        print(f"  [checkpoint] resuming, {len(done)} unit(s) already complete", flush=True)

    def execute(unit_id, fn, label):
        if unit_id in done:
            print(f"  {label}: cached", flush=True)
            return done[unit_id]["result"]
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        record = {"schema_version": SCHEMA_VERSION, "config_fingerprint": fingerprint,
                  "unit_id": unit_id, "result": result, "unit_wall_s": elapsed}
        qmc_lib.append_checkpoint(checkpoint_path, record)
        done[unit_id] = record
        print(f"  {label}  ({elapsed:.1f}s)", flush=True)
        return result

    accuracy_rows, compare_rows = [], []

    if cfg["mode"] in ("accuracy", "all"):
        print("#" * 100)
        print("# DES time-to-target-accuracy vs. coupled-model MLMC reference")
        print("#" * 100, flush=True)
        for topo in cfg["topologies"]:
            for n_nodes in cfg["n_nodes"]:
                for seed in cfg["seeds"]:
                    adjacency, topo_used, note = qmc_lib.load_topology(
                        topo, n_nodes, seed, p_er=p_er_for(cfg, n_nodes))
                    lambda_vec = np.full(adjacency.shape[0], cfg["lam"])
                    for epsilon in cfg["epsilons"]:
                        unit = f"acc|{topo}|{n_nodes}|{seed}|{epsilon}"

                        def run(adj=adjacency, lam=lambda_vec, eps=epsilon, s=seed):
                            des_result = run_des_to_epsilon(
                                adj, lam, cfg["alpha"], cfg["beta"], cfg["T"],
                                cfg["sample_dt"], cfg["warmup_frac"] * cfg["T"],
                                cfg["service_dist"], eps, base_seed=s * 100_000,
                                min_reps=cfg["min_reps"], max_reps=cfg["max_reps"])
                            t0 = time.perf_counter()
                            mlmc_result = qmc_lib.coupled_mlmc_estimate(
                                adj, lam, cfg["alpha"], cfg["beta"], cfg["sigma"],
                                cfg["T"], cfg["base_dt"], cfg["L_max"], eps,
                                cfg["pilot_samples"], seed=s)
                            mlmc_wall_s = time.perf_counter() - t0
                            return {"des": des_result,
                                    "mlmc_reference": {**mlmc_result, "wall_s": mlmc_wall_s}}

                        result = execute(unit, run,
                                         f"[{topo_used}] n={n_nodes} seed={seed} eps={epsilon}")
                        des_r, mlmc_r = result["des"], result["mlmc_reference"]
                        accuracy_rows.append({
                            "topology": topo_used, "topology_note": note, "n_nodes": n_nodes,
                            "seed": seed, "epsilon": epsilon,
                            "des_estimate": des_r["estimate"], "des_ci_halfwidth": des_r["ci_halfwidth"],
                            "des_converged": des_r["converged"], "des_replications": des_r["replications"],
                            "des_total_events": des_r["total_events"], "des_wall_s": des_r["wall_s"],
                            "mlmc_estimate": mlmc_r["estimate"], "mlmc_ci_halfwidth": mlmc_r["ci_halfwidth"],
                            "mlmc_total_work_w": mlmc_r["total_cost"], "mlmc_wall_s": mlmc_r["wall_s"],
                            "des_to_mlmc_wall_ratio": (des_r["wall_s"] / mlmc_r["wall_s"]
                                                        if mlmc_r["wall_s"] > 0 else None),
                        })

    if cfg["mode"] in ("sde-compare", "all"):
        print("\n" + "#" * 100)
        print("# DES vs Euler-Maruyama SDE at moderate load (fixed operating point)")
        print("#" * 100, flush=True)
        for topo in cfg["topologies"]:
            for n_nodes in cfg["n_nodes"]:
                for seed in cfg["seeds"]:
                    adjacency, topo_used, note = qmc_lib.load_topology(
                        topo, n_nodes, seed, p_er=p_er_for(cfg, n_nodes))
                    lambda_vec = np.full(adjacency.shape[0], cfg["lam"])
                    unit = f"cmp|{topo}|{n_nodes}|{seed}"

                    def run(adj=adjacency, lam=lambda_vec, s=seed):
                        return run_sde_compare(
                            adj, lam, cfg["alpha"], cfg["beta"], cfg["sigma"],
                            cfg["T"], cfg["base_dt"], cfg["sample_dt"],
                            cfg["warmup_frac"] * cfg["T"], cfg["service_dist"],
                            cfg["sde_compare_reps"], base_seed=s * 200_000)

                    result = execute(unit, run, f"[{topo_used}] n={n_nodes} seed={seed}")
                    compare_rows.append({
                        "topology": topo_used, "topology_note": note, "n_nodes": n_nodes,
                        "seed": seed, "n_reps": result["n_reps"],
                        "des_mean": result["des_mean"], "des_var": result["des_var"],
                        "des_wall_s": result["des_wall_s"], "sde_mean": result["sde_mean"],
                        "sde_var": result["sde_var"], "sde_wall_s": result["sde_wall_s"],
                        "rel_diff_mean_pct": result["rel_diff_mean_pct"],
                        "welch_z_stat": result["welch_z_stat"],
                        "diverges_wildly": result["diverges_wildly"],
                    })
                    flag = " *** DIVERGES ***" if result["diverges_wildly"] else ""
                    print(f"    DES mean={result['des_mean']:.4f} var={result['des_var']:.4e}  |  "
                          f"SDE mean={result['sde_mean']:.4f} var={result['sde_var']:.4e}  |  "
                          f"rel.diff={result['rel_diff_mean_pct']:.1f}%{flag}", flush=True)

    write_outputs(args, cfg, prov, accuracy_rows, compare_rows)


def write_outputs(args, cfg, prov, accuracy_rows, compare_rows):
    header = qmc_lib.provenance_comment_lines(prov)

    def open_csv(name):
        handle = open(os.path.join(args.out, name), "w", newline="")
        for line in header:
            handle.write(line + "\n")
        return handle

    json_path = os.path.join(args.out, "des_baseline.json")
    with open(json_path, "w") as f:
        json.dump({"provenance": prov, "accuracy": accuracy_rows,
                   "sde_compare": compare_rows}, f, indent=2, default=float)
    print(f"\nWrote {json_path}")

    if accuracy_rows:
        path = os.path.join(args.out, "accuracy.csv")
        with open_csv("accuracy.csv") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(accuracy_rows[0].keys()))
            writer.writeheader()
            writer.writerows(accuracy_rows)
        print(f"Wrote {path}")

    if compare_rows:
        path = os.path.join(args.out, "sde_compare.csv")
        with open_csv("sde_compare.csv") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(compare_rows[0].keys()))
            writer.writeheader()
            writer.writerows(compare_rows)
        print(f"Wrote {path}")

    if accuracy_rows:
        print("\nCPU-only exact DES vs. coupled-model MLMC, wall-clock:")
        print(f"{'topology':<8} {'n':>5} {'eps':>6} {'DES(s)':>10} {'MLMC(s)':>10} {'DES/MLMC':>10}")
        for r in accuracy_rows:
            ratio = f"{r['des_to_mlmc_wall_ratio']:.1f}x" if r["des_to_mlmc_wall_ratio"] else "N/A"
            print(f"{r['topology']:<8} {r['n_nodes']:>5} {r['epsilon']:>6} "
                  f"{r['des_wall_s']:>10.3f} {r['mlmc_wall_s']:>10.3f} {ratio:>10}")

    if compare_rows:
        n_diverge = sum(1 for r in compare_rows if r["diverges_wildly"])
        print(f"\nDES vs SDE moderate-load comparison: {n_diverge}/{len(compare_rows)} "
              f"row(s) flagged as diverging wildly (|rel diff| > 25% or |Welch z| > 5)")


if __name__ == "__main__":
    main()
