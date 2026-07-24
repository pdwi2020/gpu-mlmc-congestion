"""
GPU profiling harness for the coupled-SDE MLMC kernel (Reviewer 2: bottleneck /
memory / parallel-efficiency analysis).

Profiles `GPUCoupledPropagationMLMC` (src/gpu/parallel_mc.py, imported
read-only -- this script does not modify that file) at n_nodes in
{500, 5000, 50000} and reports, per size:

  * Phase breakdown of one MLMC fine-level path loop: degree-normalised
    matmul (the drift term t.mm(influence, c)), RNG (the dW draw), Skorokhod
    reflection (the clamp_min boundary), MLMC bookkeeping/reduction
    (level-difference mean/variance), and all-reduce (only meaningful when
    distributed; see below). Measured via torch.profiler with CUDA activities
    on GPU, CPU activities otherwise (MPS exposes no separate device-side
    profiler activity in this torch version -- CPU dispatch/launch time is
    reported instead, and that limitation is recorded in the output, not
    hidden).

  * Measured peak memory vs n. The manuscript's Table `max_graph_size` gives
    only the analytical adjacency-tensor model n*(n/G)*4 bytes/rank; this
    script's measured curve is what validates or replaces it (see
    `analytical_adjacency_memory_bytes` below, and the large gap it already
    predicts against the paper's own Table `large_scale` measured numbers).

  * Kernel-level stats where available (CUDA only: call counts + durations
    per kernel name from the profiler trace). Achieved SM occupancy needs
    Nsight Compute (`ncu`), which torch.profiler does not expose; this is
    recorded as a skip reason, never fabricated.

  * An optional nsys (Nsight Systems) trace, attempted only if `nsys` is on
    PATH; never required, and its absence never fails the run.

  * A best-effort, explicitly-degenerate torch.distributed all_reduce
    timing (single-rank gloo backend) as the "all-reduce" phase when no real
    multi-GPU process group is available -- this is NOT a substitute for the
    real multi-GPU NCCL measurement in scripts/run_multi_gpu_scaling.py-style
    harnesses; it is labelled as such.

Every number in the output is tagged MEASURED or DERIVED (config/formula), so
this feeds directly into a Measured-vs-Modelled table without ambiguity.

To keep n=50000's dense n x n influence matrix (10 GB in float32) from
thrashing a laptop, this script auto-scales the path-ensemble size (n_samples)
down as n_nodes grows, holding total matmul FLOPs roughly constant across
sizes (see `auto_scale_n_samples`); on non-CUDA devices it ALSO refuses to
allocate a dense adjacency tensor above --memory-safety-cap-gb (default 2 GB)
and reports a skip reason with the DERIVED memory estimate instead of OOMing
the machine. Neither limit applies on CUDA (--device cuda), where the point is
to actually run n=50000.

Usage:
    python scripts/run_profiling.py --quick
    python scripts/run_profiling.py --seeds 0 1 2 --device cuda --no-sample-autoscale --n-samples-base 8192

Output (under --out, default results/profiling):
    profiling_results.json   full results + provenance, MEASURED/DERIVED tagged
    phase_breakdown.csv      one row per (n_nodes, seed, phase)
    memory_scaling.csv       one row per n_nodes: measured vs analytical model
"""

import argparse
import csv
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch  # noqa: E402
from gpu.parallel_mc import GPUCoupledPropagationMLMC  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')

DEFAULT_SIZES = [500, 5000, 50000]
BASE_N_FOR_SCALING = 500  # the size at which n_samples_base applies unscaled
MIN_N_SAMPLES = 4
#: Refuse to allocate a dense n x n adjacency tensor above this on non-CUDA
#: devices (a laptop CPU/MPS run has no business allocating 10 GB). Bypassed
#: entirely on CUDA, where running the full size is the point.
DEFAULT_MEMORY_SAFETY_CAP_GB = 2.0


# =============================================================================
# Device / provenance (same conventions as scripts/run_adaptive_stepping_ablation.py)
# =============================================================================
def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple MPS ({platform.processor() or platform.machine()})"
    return platform.processor() or platform.machine() or "cpu"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


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


def build_provenance(device: torch.device, config: dict) -> dict:
    return {
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_name": device_name(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "config": config,
    }


def provenance_comment_lines(prov: dict) -> list:
    return [
        f"# git_sha={prov['git_sha']}",
        f"# timestamp_utc={prov['timestamp_utc']}",
        f"# device={prov['device']} ({prov['device_name']})",
        f"# torch={prov['torch_version']} numpy={prov['numpy_version']} python={prov['python_version']}",
        f"# config={json.dumps(prov['config'], sort_keys=True, default=str)}",
    ]


def _sanitize(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


# =============================================================================
# Graph generation (self-contained, matching run_adaptive_stepping_ablation.py's
# convention of not importing src/network/topology.py for a simple ER graph)
# =============================================================================
def er_adjacency(n: int, p: float = None, seed: int = 42) -> np.ndarray:
    """Erdos-Renyi adjacency, connected up by a spanning chain so no node is
    isolated. `p` defaults to a mean-degree-~8 target, capped so large n stays
    sparse-ish (irrelevant to this script's dense-tensor memory model, but
    keeps graph construction itself fast)."""
    if p is None:
        p = min(0.08, 8.0 / max(n - 1, 1))
    rng = np.random.default_rng(seed)
    adj = (rng.random((n, n)) < p).astype(np.float32)
    adj = ((adj + adj.T) > 0).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


# =============================================================================
# Auto-scaling: keep total matmul FLOPs roughly constant across n_nodes so a
# CPU/MPS smoke test completes in comparable time at every size, while the
# n x n adjacency allocation (the memory-scaling quantity of interest) is
# always built at FULL size.
# =============================================================================
def auto_scale_n_samples(n_nodes: int, n_samples_base: int,
                          autoscale: bool) -> int:
    if not autoscale:
        return n_samples_base
    scaled = n_samples_base * (BASE_N_FOR_SCALING / float(n_nodes)) ** 2
    return max(MIN_N_SAMPLES, int(round(scaled)))


def analytical_adjacency_memory_bytes(n: int, world_size: int = 1,
                                       dtype_bytes: int = 4) -> float:
    """The manuscript's own model (Table `max_graph_size`, paper/ieee_access/
    main.tex): n * (n/G) * dtype_bytes per GPU rank, adjacency tensor only.
    G=1 (world_size=1) is this script's single-GPU-kernel case. This is
    DERIVED, not measured, and the manuscript's own text notes it excludes
    "GPU framework overhead" (~200-400 MB/rank) and, as this harness's
    measured numbers show, the path-ensemble state tensor and CUDA caching
    allocator overhead as well -- which is exactly why a measured curve
    matters."""
    return float(n) * (float(n) / world_size) * dtype_bytes


def extended_analytical_memory_bytes(n: int, n_samples: int,
                                      dtype_bytes: int = 4) -> float:
    """This harness's OWN extended estimate (not the paper's), adding the
    dominant per-step ensemble-state allocations the adjacency-only model
    omits: the state tensor c (n x n_samples), plus predictor/corrector
    intermediates and the noise draw dw (each n x n_samples), on top of the
    adjacency tensor. Reported as a second DERIVED reference point, kept
    clearly distinct from the paper's published formula."""
    adjacency = float(n) * float(n) * dtype_bytes
    # c, c_pred, drift_n, drift_pred, noise, dw ~ up to 6 live (n, n_samples) buffers
    # during the predictor-corrector step; a generous but explicit constant.
    ensemble_buffers = 6.0 * float(n) * float(n_samples) * dtype_bytes
    return adjacency + ensemble_buffers


# =============================================================================
# Memory measurement -- MEASURED, method depends on device
# =============================================================================
def measure_peak_memory(device: torch.device, fn):
    """Run `fn()` and report peak memory MEASURED during it. Method differs by
    device since PyTorch exposes a true allocator-peak counter only for CUDA:

      * CUDA: torch.cuda.reset_peak_memory_stats() + max_memory_allocated()
        after a sync -- the standard, accurate approach.
      * MPS: no true peak-tracking API exists in this torch version (only
        `current_allocated_memory()`); `fn` is called with a `sample_peak`
        callback threaded through so THIS script can sample-and-max it at
        every simulation step. Reported as a step-sampled approximation, not
        a true allocator peak, and labelled as such.
      * CPU: `resource.getrusage(RUSAGE_SELF).ru_maxrss`, a coarse whole-process
        high-water mark (not kernel-specific), before vs after. macOS reports
        bytes, Linux reports KB -- handled explicitly.

    Returns (result_of_fn, {"measured_peak_mb": ..., "method": ..., "note": ...}).
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        result = fn(peak_sampler=None)
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        return result, {"measured_peak_mb": peak_bytes / 1e6,
                         "method": "torch.cuda.max_memory_allocated (true allocator peak)",
                         "note": None}

    if device.type == "mps":
        peak = {"bytes": torch.mps.current_allocated_memory()}

        def sampler():
            cur = torch.mps.current_allocated_memory()
            if cur > peak["bytes"]:
                peak["bytes"] = cur

        result = fn(peak_sampler=sampler)
        torch.mps.synchronize()
        sampler()
        return result, {"measured_peak_mb": peak["bytes"] / 1e6,
                         "method": "torch.mps.current_allocated_memory, sampled every "
                                    "simulation step and maxed (MPS backend exposes no "
                                    "true allocator-peak counter in this torch version)",
                         "note": "approximation, not a guaranteed peak"}

    # CPU
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = fn(peak_sampler=None)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    units = "bytes" if sys.platform == "darwin" else "KB"
    scale = 1.0 if units == "bytes" else 1024.0
    return result, {"measured_peak_mb": (after * scale) / 1e6,
                     "measured_delta_mb": ((after - before) * scale) / 1e6,
                     "method": f"resource.getrusage(RUSAGE_SELF).ru_maxrss "
                               f"(whole-process high-water mark, {units} on this platform; "
                               f"NOT kernel-specific -- includes the whole Python process)",
                     "note": "coarse, whole-process; delta vs pre-run baseline given "
                             "alongside the absolute high-water mark"}


# =============================================================================
# Phase-by-phase instrumented kernel loop
# =============================================================================
def profiled_level_loop(sim, n_samples: int, n_steps: int, dt: float,
                         device: torch.device, peak_sampler=None):
    """One MLMC fine-level (level-0) path loop, phase-instrumented.

    Mirrors GPUCoupledPropagationMLMC._run_level_state_tensors(level=0)
    exactly: same dw draw, same dt, same dispatch through the REAL sim._step.
    Reflection and matmul phases wrap the actual t.mm / t.clamp_min calls the
    production kernel makes, via a temporary monkeypatch -- the same
    instrumentation technique scripts/run_adaptive_stepping_ablation.py uses
    for call counting (see its `measure_matmul_scaling`), applied here for
    phase-time attribution instead. This measures the production kernel, not
    a re-implementation of its math.
    """
    t = torch
    sim.reset_adaptive_state()
    c = t.zeros(sim.n_nodes, n_samples, device=device, dtype=t.float32)

    real_mm = t.mm
    real_clamp_min = t.clamp_min

    def traced_mm(*a, **kw):
        with torch.profiler.record_function("phase/matmul_drift"):
            return real_mm(*a, **kw)

    def traced_clamp_min(*a, **kw):
        with torch.profiler.record_function("phase/reflection"):
            return real_clamp_min(*a, **kw)

    t.mm = traced_mm
    t.clamp_min = traced_clamp_min
    try:
        for _ in range(n_steps):
            with torch.profiler.record_function("phase/rng"):
                dw = t.randn(sim.n_nodes, n_samples, device=device) * (dt ** 0.5)
            with torch.profiler.record_function("phase/step_dispatch"):
                c = sim._step(c, dt, dw, role="fine")
            if peak_sampler is not None:
                peak_sampler()
    finally:
        t.mm = real_mm
        t.clamp_min = real_clamp_min
    return c


def mlmc_bookkeeping_phase(c_fine, device: torch.device):
    """The post-loop MLMC level-difference bookkeeping/reduction: extract the
    scalar metric, form Y_fine - Y_coarse (level 0's coarse path is
    identically zero, matching GPUCoupledPropagationMLMC.run_level(level=0)),
    and compute mean/variance -- mirrors mlmc_estimate's per-level reduction."""
    with torch.profiler.record_function("phase/mlmc_bookkeeping"):
        y_fine = c_fine.mean(dim=0).cpu().numpy()
        y_coarse = np.zeros_like(y_fine)
        diffs = y_fine - y_coarse
        mean_diff = float(np.mean(diffs))
        var_diff = float(np.var(diffs, ddof=1)) if diffs.size > 1 else 0.0
    return {"mean_diff": mean_diff, "var_diff": var_diff, "n_samples": int(diffs.size)}


# =============================================================================
# All-reduce phase: best-effort degenerate single-rank baseline
# =============================================================================
def _pick_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def profile_all_reduce_phase(device: torch.device, quick: bool) -> dict:
    """Best-effort MEASURED all-reduce timing via a single-rank torch.distributed
    process group (gloo). This is a DEGENERATE baseline -- one rank has
    nothing to communicate with -- so it measures API/dispatch overhead only,
    NOT real multi-GPU NCCL communication cost. Real multi-GPU all-reduce
    timing lives in the multi-GPU scaling scripts (see comm_compute_ratio in
    src/gpu/parallel_mc.py's MultiGPUMLMC); that is what the paper's
    multi-GPU figures should keep using. This phase exists so the profiler
    trace always has SOMETHING under "phase/all_reduce" rather than a bare
    skip, while being explicit about what it does and does not measure.
    """
    try:
        import torch.distributed as dist
        already_initialized = dist.is_initialized()
        if dist.is_available() and not already_initialized:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(_pick_free_port()))
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

        tensor = torch.zeros(1024, dtype=torch.float32)
        with torch.profiler.record_function("phase/all_reduce"):
            dist.all_reduce(tensor)  # warm-up, inside the trace deliberately

        reps = 2 if quick else 5
        t0 = time.perf_counter()
        for _ in range(reps):
            with torch.profiler.record_function("phase/all_reduce"):
                dist.all_reduce(tensor)
        elapsed = (time.perf_counter() - t0) / reps

        if not already_initialized:
            dist.destroy_process_group()

        return {
            "measured": True, "world_size": 1, "backend": "gloo",
            "mean_wall_s_per_call": elapsed, "reps": reps,
            "note": ("DEGENERATE single-rank baseline (nothing to communicate with); "
                     "measures torch.distributed API/dispatch overhead only, NOT real "
                     "multi-GPU NCCL communication cost. Real multi-GPU all-reduce "
                     "timing must come from a >1-world-size CUDA run (see "
                     "MultiGPUMLMC.comm_compute_ratio in src/gpu/parallel_mc.py)."),
        }
    except Exception as exc:  # noqa: BLE001 - never let this sink the whole run
        return {"measured": False,
                "skip_reason": f"torch.distributed all_reduce unavailable/failed: "
                                f"{type(exc).__name__}: {exc}"}


# =============================================================================
# Phase-breakdown summarisation from a torch.profiler trace
# =============================================================================
def summarize_phases(prof, device: torch.device) -> dict:
    avgs = prof.key_averages()
    phases = {}
    total_self_s = 0.0
    use_device_time = device.type == "cuda"
    for e in avgs:
        if not e.key.startswith("phase/"):
            continue
        name = e.key[len("phase/"):]
        if use_device_time:
            self_us = getattr(e, "self_device_time_total", None)
            if not self_us:
                self_us = getattr(e, "self_cuda_time_total", 0.0)
        else:
            self_us = e.self_cpu_time_total
        self_s = float(self_us) / 1e6
        slot = phases.setdefault(name, {"self_time_s": 0.0, "calls": 0})
        slot["self_time_s"] += self_s
        slot["calls"] += int(e.count)
        total_self_s += self_s
    for v in phases.values():
        v["pct_of_traced_total"] = (v["self_time_s"] / total_self_s * 100.0
                                     if total_self_s > 0 else None)
    return {
        "device_activity_kind": ("cuda_device_time (true device-side kernel time)"
                                  if use_device_time else
                                  "cpu_self_time (dispatch/launch time; MPS backend exposes "
                                  "no separate device-side profiler activity in this torch "
                                  "version, so this is CPU-observed latency, not GPU kernel "
                                  "time -- stated explicitly, not hidden)"),
        "total_traced_self_time_s": total_self_s,
        "phases": phases,
    }


def summarize_cuda_kernels(prof, top_k: int = 15) -> dict:
    """CUDA-only kernel-level stats (call counts + durations) from the trace.
    This is NOT achieved SM occupancy -- torch.profiler does not expose that;
    it requires Nsight Compute (`ncu`), reported separately as a skip reason.
    """
    avgs = prof.key_averages()
    rows = []
    for e in avgs:
        if e.key.startswith("phase/"):
            continue  # our own labels, not device kernels
        device_us = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0.0)
        if not device_us:
            continue
        rows.append({"kernel": e.key, "calls": int(e.count),
                     "self_device_time_s": float(device_us) / 1e6})
    rows.sort(key=lambda r: r["self_device_time_s"], reverse=True)
    return {"measured": True, "top_kernels": rows[:top_k], "n_distinct_kernels": len(rows)}


def occupancy_stats() -> dict:
    return {
        "measured": False,
        "skip_reason": ("achieved SM occupancy requires NVIDIA Nsight Compute (ncu); "
                         "torch.profiler's CUPTI-based trace reports kernel call counts "
                         "and durations (see kernel_stats) but not occupancy/warp metrics. "
                         "Not attempted here -- run `ncu --metrics sm__warps_active.avg."
                         "pct_of_peak_sustained_active <cmd>` separately on the A100 if "
                         "occupancy is needed for the paper."),
    }


# =============================================================================
# Optional nsys trace (never required)
# =============================================================================
def try_nsys_trace(out_dir: str, n_nodes: int, n_samples: int, n_steps: int,
                    device_arg: str, timeout_s: int = 180) -> dict:
    nsys_bin = shutil.which("nsys")
    if not nsys_bin:
        return {"attempted": False, "available": False,
                "skip_reason": "nsys (NVIDIA Nsight Systems) not found on PATH; "
                                "optional, never required for this harness"}
    trace_stub = os.path.join(out_dir, f"nsys_trace_n{n_nodes}")
    cmd = [nsys_bin, "profile", "-o", trace_stub, "--force-overwrite=true",
           sys.executable, os.path.abspath(__file__), "--nsys-child",
           "--nsys-child-n", str(n_nodes), "--nsys-child-samples", str(n_samples),
           "--nsys-child-steps", str(n_steps), "--device", device_arg]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        rep_path = trace_stub + ".nsys-rep"
        ok = proc.returncode == 0 and os.path.exists(rep_path)
        return {"attempted": True, "available": True, "success": ok,
                "trace_path": rep_path if ok else None,
                "returncode": proc.returncode,
                "stderr_tail": (proc.stderr or "")[-2000:] if not ok else None}
    except Exception as exc:  # noqa: BLE001
        return {"attempted": True, "available": True, "success": False,
                "error": f"{type(exc).__name__}: {exc}"}


def run_nsys_child(args) -> None:
    """Minimal profiled pass for `nsys profile -- python run_profiling.py
    --nsys-child ...` to wrap. Not part of the normal harness output."""
    device = select_device(args.device)
    adjacency = er_adjacency(args.nsys_child_n, seed=42)
    sim = GPUCoupledPropagationMLMC(adjacency, influence_strength=0.2, decay_rate=0.5,
                                     noise_intensity=0.1, seed=0,
                                     adaptive_stepping=False,
                                     reflection="predictor_corrector")
    if sim._device != device:
        sim._device = device
        sim._influence = sim._influence.to(device)
    profiled_level_loop(sim, args.nsys_child_samples, args.nsys_child_steps,
                        0.1, device)
    synchronize(device)
    print(f"[nsys-child] completed n={args.nsys_child_n} "
          f"n_samples={args.nsys_child_samples} n_steps={args.nsys_child_steps}")


# =============================================================================
# Per-size profiling driver
# =============================================================================
def profile_one_size(n_nodes: int, n_samples_base: int, autoscale: bool,
                      T: float, base_dt: float, device: torch.device,
                      seed: int, repeats: int, quick: bool,
                      memory_safety_cap_gb: float, attempt_all_reduce: bool) -> dict:
    n_samples = auto_scale_n_samples(n_nodes, n_samples_base, autoscale)
    n_steps = max(1, int(T / base_dt))
    analytical_bytes = analytical_adjacency_memory_bytes(n_nodes)
    extended_bytes = extended_analytical_memory_bytes(n_nodes, n_samples)

    record = {
        "n_nodes": n_nodes, "n_samples": n_samples, "n_steps": n_steps,
        "seed": seed, "T": T, "base_dt": base_dt,
        "derived": {
            "paper_adjacency_only_model_mb": analytical_bytes / 1e6,
            "extended_model_incl_ensemble_buffers_mb": extended_bytes / 1e6,
            "note": ("paper_adjacency_only_model_mb is the manuscript's own published "
                     "formula (n*(n/G)*4 bytes/rank, G=1); "
                     "extended_model_incl_ensemble_buffers_mb is THIS harness's own "
                     "estimate, adding the ensemble state/predictor-corrector buffers "
                     "the published formula omits -- both DERIVED, neither MEASURED"),
        },
    }

    if device.type != "cuda" and analytical_bytes > memory_safety_cap_gb * 1024**3:
        record["skipped"] = True
        record["skip_reason"] = (
            f"dense {n_nodes}x{n_nodes} float32 adjacency tensor would need "
            f"{analytical_bytes / 1e9:.2f} GB, above the "
            f"--memory-safety-cap-gb={memory_safety_cap_gb} GB guard for non-CUDA "
            f"devices (this machine: {device.type}). This size must be profiled on "
            f"the A100 (--device cuda), where the cap does not apply.")
        return record

    adjacency = er_adjacency(n_nodes, seed=seed)
    torch.manual_seed(seed)
    sim = GPUCoupledPropagationMLMC(adjacency, influence_strength=0.2, decay_rate=0.5,
                                     noise_intensity=0.1, seed=seed,
                                     adaptive_stepping=False,
                                     reflection="predictor_corrector")
    if sim._device != device:
        sim._device = device
        sim._influence = sim._influence.to(device)

    # Untraced warm-up: excludes first-call allocator/JIT overhead from the
    # timed and profiled measurement below.
    warm_samples = min(n_samples, 8)
    profiled_level_loop(sim, warm_samples, min(n_steps, 2), base_dt, device)
    synchronize(device)

    def _run(peak_sampler=None):
        return profiled_level_loop(sim, n_samples, n_steps, base_dt, device,
                                   peak_sampler=peak_sampler)

    # Separate, clean pass purely for the memory measurement (not profiled --
    # torch.profiler's own bookkeeping perturbs allocator behaviour slightly,
    # so memory and phase-timing are measured in independent passes).
    _, mem = measure_peak_memory(device, _run)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    wall_times = []
    prof = None
    bookkeeping = None
    all_reduce_result = None
    for rep in range(repeats):
        sim.reset_adaptive_state()
        synchronize(device)
        t0 = time.perf_counter()
        with torch.profiler.profile(activities=activities, record_shapes=False,
                                    profile_memory=True, with_stack=False) as p:
            c_fine = profiled_level_loop(sim, n_samples, n_steps, base_dt, device)
            bookkeeping = mlmc_bookkeeping_phase(c_fine, device)
            if attempt_all_reduce and rep == repeats - 1:
                all_reduce_result = profile_all_reduce_phase(device, quick)
        synchronize(device)
        wall_times.append(time.perf_counter() - t0)
        prof = p  # keep the last rep's trace for phase/kernel summarisation

    if all_reduce_result is None:
        all_reduce_result = {"measured": False,
                              "skip_reason": "disabled via --no-all-reduce"}

    phase_summary = summarize_phases(prof, device)
    phase_summary["phases"]["all_reduce"] = (
        {"self_time_s": all_reduce_result.get("mean_wall_s_per_call", 0.0)
                        * all_reduce_result.get("reps", 0),
         "calls": all_reduce_result.get("reps", 0),
         "pct_of_traced_total": None,
         "measured": all_reduce_result["measured"],
         "note": all_reduce_result.get("note") or all_reduce_result.get("skip_reason")}
    )

    if device.type == "cuda":
        kernel_stats = summarize_cuda_kernels(prof)
    else:
        kernel_stats = {"measured": False,
                         "skip_reason": f"kernel-level device stats require CUDA "
                                        f"activities; device is {device.type}"}

    record.update({
        "skipped": False,
        "measured": {
            "wall_clock_s_per_rep": wall_times,
            "wall_clock_s_mean": float(np.mean(wall_times)),
            "wall_clock_s_sd": float(np.std(wall_times, ddof=1)) if len(wall_times) > 1 else 0.0,
            "peak_memory": mem,
            "phase_breakdown": phase_summary,
            "kernel_stats": kernel_stats,
            "mlmc_bookkeeping_result": bookkeeping,
            "all_reduce": all_reduce_result,
        },
        "occupancy": occupancy_stats(),
        "memory_ratio_measured_over_paper_model": (
            (mem["measured_peak_mb"] / record["derived"]["paper_adjacency_only_model_mb"])
            if record["derived"]["paper_adjacency_only_model_mb"] > 0 else None),
    })
    return record


# =============================================================================
# Output
# =============================================================================
def write_outputs(args, prov: dict, per_size_records: dict, nsys_result: dict) -> dict:
    os.makedirs(args.out, exist_ok=True)
    header = provenance_comment_lines(prov)
    paths = {}

    json_path = os.path.join(args.out, "profiling_results.json")
    with open(json_path, "w") as f:
        json.dump(_sanitize({"provenance": prov, "nsys": nsys_result,
                             "results": per_size_records}), f, indent=2)
    paths["json"] = json_path

    phase_csv = os.path.join(args.out, "phase_breakdown.csv")
    with open(phase_csv, "w", newline="") as f:
        for line in header:
            f.write(line + "\n")
        writer = csv.writer(f)
        writer.writerow(["n_nodes", "seed", "phase", "self_time_s",
                         "pct_of_traced_total", "calls", "device_activity_kind"])
        for n_nodes, seeds in sorted(per_size_records.items()):
            for seed, rec in sorted(seeds.items()):
                if rec.get("skipped"):
                    continue
                pb = rec["measured"]["phase_breakdown"]
                for phase, v in pb["phases"].items():
                    writer.writerow([n_nodes, seed, phase, v.get("self_time_s"),
                                     v.get("pct_of_traced_total"), v.get("calls"),
                                     pb["device_activity_kind"]])
    paths["phase_csv"] = phase_csv

    mem_csv = os.path.join(args.out, "memory_scaling.csv")
    with open(mem_csv, "w", newline="") as f:
        for line in header:
            f.write(line + "\n")
        f.write("# MEASURED columns come from this run; DERIVED columns are formulas,\n")
        f.write("# never measurements. See profiling_results.json for full detail.\n")
        writer = csv.writer(f)
        writer.writerow(["n_nodes", "seed", "n_samples", "skipped", "skip_reason",
                         "measured_peak_mb", "measured_method",
                         "derived_paper_adjacency_model_mb",
                         "derived_extended_model_mb",
                         "measured_over_paper_model_ratio"])
        for n_nodes, seeds in sorted(per_size_records.items()):
            for seed, rec in sorted(seeds.items()):
                if rec.get("skipped"):
                    writer.writerow([n_nodes, seed, rec["n_samples"], True,
                                     rec["skip_reason"], "", "",
                                     rec["derived"]["paper_adjacency_only_model_mb"],
                                     rec["derived"]["extended_model_incl_ensemble_buffers_mb"],
                                     ""])
                else:
                    mem = rec["measured"]["peak_memory"]
                    writer.writerow([n_nodes, seed, rec["n_samples"], False, "",
                                     mem["measured_peak_mb"], mem["method"],
                                     rec["derived"]["paper_adjacency_only_model_mb"],
                                     rec["derived"]["extended_model_incl_ensemble_buffers_mb"],
                                     rec["memory_ratio_measured_over_paper_model"]])
    paths["memory_csv"] = mem_csv
    return paths


def print_report(per_size_records: dict, nsys_result: dict) -> None:
    print("\n" + "=" * 110)
    print("PROFILING RESULTS")
    print("=" * 110)
    print(f"\n{'n_nodes':>9}{'seed':>6}{'n_samples':>11}{'status':>10}"
          f"{'wall_s':>10}{'peak_MB':>12}{'paper_model_MB':>16}{'ratio':>9}")
    for n_nodes, seeds in sorted(per_size_records.items()):
        for seed, rec in sorted(seeds.items()):
            model_mb = rec["derived"]["paper_adjacency_only_model_mb"]
            if rec.get("skipped"):
                print(f"{n_nodes:>9}{seed:>6}{rec['n_samples']:>11}{'SKIPPED':>10}"
                      f"{'--':>10}{'--':>12}{model_mb:>16.2f}{'--':>9}")
                continue
            m = rec["measured"]
            ratio = rec["memory_ratio_measured_over_paper_model"]
            print(f"{n_nodes:>9}{seed:>6}{rec['n_samples']:>11}{'ok':>10}"
                  f"{m['wall_clock_s_mean']:>10.4f}"
                  f"{m['peak_memory']['measured_peak_mb']:>12.3f}"
                  f"{model_mb:>16.2f}"
                  f"{(f'{ratio:.3f}x' if ratio else '--'):>9}")

    print("\nPhase breakdown (self time, one representative seed per size):")
    for n_nodes, seeds in sorted(per_size_records.items()):
        seed0 = sorted(seeds)[0]
        rec = seeds[seed0]
        if rec.get("skipped"):
            continue
        pb = rec["measured"]["phase_breakdown"]
        print(f"\n  n_nodes={n_nodes}  (device_activity: {pb['device_activity_kind']})")
        for phase, v in sorted(pb["phases"].items(), key=lambda kv: -kv[1]["self_time_s"]):
            pct = f"{v['pct_of_traced_total']:.1f}%" if v.get("pct_of_traced_total") is not None else "n/a"
            print(f"    {phase:<20}{v['self_time_s']:>12.6f}s  ({pct:>7}, calls={v['calls']})")

    print(f"\nnsys: {'attempted' if nsys_result.get('attempted') else 'not attempted'} "
          f"({nsys_result.get('skip_reason') or ('success' if nsys_result.get('success') else 'see profiling_results.json')})")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Profile the coupled-SDE MLMC kernel: phase breakdown, "
                     "measured-vs-modelled memory, kernel stats, optional nsys.")
    # --nsys-child is a hidden re-entry point used by try_nsys_trace(); not
    # part of the normal CLI surface.
    parser.add_argument("--nsys-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--nsys-child-n", type=int, default=500, help=argparse.SUPPRESS)
    parser.add_argument("--nsys-child-samples", type=int, default=64, help=argparse.SUPPRESS)
    parser.add_argument("--nsys-child-steps", type=int, default=10, help=argparse.SUPPRESS)

    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "profiling"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="auto prefers cuda > mps > cpu")
    parser.add_argument("--quick", action="store_true", help="fast CPU/MPS smoke test")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--n-samples-base", type=int, default=256,
                        help="path-ensemble size at n_nodes=500; auto-scaled down "
                             "at larger n_nodes to hold FLOPs roughly constant "
                             "(disable with --no-sample-autoscale)")
    parser.add_argument("--no-sample-autoscale", action="store_true",
                         help="use --n-samples-base unscaled at every size "
                              "(what you want on the A100 for realistic numbers)")
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--base-dt", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=3,
                        help="profiled repetitions per (size, seed), for a wall-clock mean/SD")
    parser.add_argument("--memory-safety-cap-gb", type=float,
                        default=DEFAULT_MEMORY_SAFETY_CAP_GB,
                        help="refuse to allocate a dense adjacency tensor above this "
                             "on non-CUDA devices (always bypassed on --device cuda)")
    parser.add_argument("--no-all-reduce", action="store_true",
                         help="skip the degenerate single-rank all_reduce phase entirely")
    parser.add_argument("--no-nsys", action="store_true",
                         help="never attempt an nsys trace even if nsys is on PATH")
    args = parser.parse_args()

    if args.nsys_child:
        run_nsys_child(args)
        return

    if args.quick:
        args.n_samples_base = min(args.n_samples_base, 64)
        args.repeats = 1
        args.seeds = args.seeds[:1]
        args.T = min(args.T, 0.3)
        args.base_dt = max(args.base_dt, 0.1)  # keep n_steps small

    device = select_device(args.device)
    config = {
        "seeds": args.seeds, "sizes": args.sizes,
        "n_samples_base": args.n_samples_base,
        "sample_autoscale": not args.no_sample_autoscale,
        "T": args.T, "base_dt": args.base_dt, "repeats": args.repeats,
        "memory_safety_cap_gb": args.memory_safety_cap_gb,
        "attempt_all_reduce": not args.no_all_reduce, "quick": args.quick,
    }
    prov = build_provenance(device, config)

    print("=" * 110)
    print("GPU profiling harness -- coupled-SDE MLMC kernel")
    print("=" * 110)
    print(f"  git sha    : {prov['git_sha']}")
    print(f"  device     : {prov['device']} ({prov['device_name']})")
    print(f"  torch      : {prov['torch_version']}")
    print(f"  sizes      : {args.sizes}")
    print(f"  seeds      : {args.seeds}")
    print(f"  autoscale  : {not args.no_sample_autoscale} (n_samples_base={args.n_samples_base})")
    print(f"  out        : {os.path.abspath(args.out)}", flush=True)

    os.makedirs(args.out, exist_ok=True)

    per_size_records = {}
    started = time.perf_counter()
    for n_nodes in args.sizes:
        per_size_records[n_nodes] = {}
        for seed in args.seeds:
            print(f"\n[n_nodes={n_nodes} seed={seed}] profiling ...", flush=True)
            try:
                rec = profile_one_size(
                    n_nodes, args.n_samples_base, not args.no_sample_autoscale,
                    args.T, args.base_dt, device, seed, args.repeats, args.quick,
                    args.memory_safety_cap_gb, not args.no_all_reduce)
            except Exception as exc:  # noqa: BLE001 - one size must not sink the whole run
                rec = {"n_nodes": n_nodes, "seed": seed, "skipped": True,
                       "skip_reason": f"profiling raised {type(exc).__name__}: {exc}",
                       "derived": {
                           "paper_adjacency_only_model_mb": analytical_adjacency_memory_bytes(n_nodes) / 1e6,
                           "extended_model_incl_ensemble_buffers_mb":
                               extended_analytical_memory_bytes(
                                   n_nodes, auto_scale_n_samples(
                                       n_nodes, args.n_samples_base, not args.no_sample_autoscale)) / 1e6,
                           "note": "profiling failed; see skip_reason",
                       },
                       "n_samples": auto_scale_n_samples(n_nodes, args.n_samples_base,
                                                          not args.no_sample_autoscale)}
                print(f"  FAILED: {rec['skip_reason']}")
            per_size_records[n_nodes][seed] = rec
            if rec.get("skipped"):
                print(f"  skipped: {rec['skip_reason']}")
            else:
                print(f"  wall={rec['measured']['wall_clock_s_mean']:.4f}s  "
                      f"peak_mem={rec['measured']['peak_memory']['measured_peak_mb']:.3f}MB")

    nsys_result = {"attempted": False, "available": False,
                   "skip_reason": "disabled via --no-nsys"}
    if not args.no_nsys and args.sizes:
        n0 = args.sizes[0]
        n_samples0 = auto_scale_n_samples(n0, args.n_samples_base, not args.no_sample_autoscale)
        n_steps0 = max(1, int(args.T / args.base_dt))
        print(f"\nAttempting optional nsys trace on n_nodes={n0} ...", flush=True)
        nsys_result = try_nsys_trace(args.out, n0, n_samples0, n_steps0, args.device)
        print(f"  nsys: {nsys_result}")

    total = time.perf_counter() - started
    paths = write_outputs(args, prov, per_size_records, nsys_result)
    print_report(per_size_records, nsys_result)

    print(f"\nTotal wall clock: {total:.1f}s")
    for label, path in paths.items():
        print(f"  {label:<12} {path}")


if __name__ == "__main__":
    main()
