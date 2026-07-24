#!/usr/bin/env python3
"""
RunPod live dashboard — redraws in-place, no blanking.

Usage:
    python scripts/runpod_dashboard.py \
        --log    <path/to/task.output>   \
        --pod-ip <host> --pod-port <port> \
        --ssh-key ~/.runpod/ssh/RunPod-Key-Go \
        --cost-per-hour 0.88 \
        --refresh 5

All flags optional — sensible defaults for the current 4×RTX3090 job.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta

# ── ANSI helpers ──────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
CYAN    = "\033[36m"
RED     = "\033[31m"
WHITE   = "\033[97m"
BG_DARK = "\033[48;5;234m"

def clr():
    """Move cursor to top-left and clear screen (no flicker vs system clear)."""
    sys.stdout.write("\033[H\033[2J")
    sys.stdout.flush()

def width():
    return shutil.get_terminal_size((100, 40)).columns

def hline(char="─", color=DIM):
    return f"{color}{char * width()}{RESET}"

def header(title: str, ts: str) -> str:
    pad = width() - len(title) - len(ts) - 4
    return (f"{BOLD}{CYAN}  {title}{RESET}"
            f"{DIM}{' ' * max(pad, 1)}{ts}  {RESET}")

# ── SSH GPU query ─────────────────────────────────────────────────────────────
_GPU_CACHE: list[dict] = []
_GPU_CACHE_TS: float = 0.0

def fetch_gpu_stats(host, port, key, timeout=6) -> list[dict]:
    global _GPU_CACHE, _GPU_CACHE_TS
    cmd = [
        "ssh", "-i", key,
        "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={timeout}",
        "-o", "BatchMode=yes",
        f"root@{host}", "-p", str(port),
        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,"
        "memory.total,power.draw,temperature.gpu "
        "--format=csv,noheader,nounits"
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=timeout+2)
        rows = []
        for line in out.decode().strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            rows.append({
                "idx":   parts[0],
                "name":  parts[1].replace("NVIDIA ", "").replace("GeForce ", ""),
                "util":  parts[2],
                "mem_used": parts[3],
                "mem_total": parts[4],
                "power": parts[5],
                "temp":  parts[6],
            })
        _GPU_CACHE = rows
        _GPU_CACHE_TS = time.time()
        return rows
    except Exception:
        return _GPU_CACHE  # return last known on failure

def render_gpu_section(gpus: list[dict], stale: bool) -> list[str]:
    lines = []
    stalemark = f"  {YELLOW}(cached){RESET}" if stale else ""
    lines.append(f"{BOLD}  GPU Utilisation{RESET}{stalemark}")
    if not gpus:
        lines.append(f"  {DIM}No data — SSH unreachable{RESET}")
        return lines
    lines.append(f"  {DIM}{'GPU':<4}{'Name':<22}{'Util':>6}{'Mem':>12}{'Power':>9}{'Temp':>7}{RESET}")
    for g in gpus:
        util = int(g["util"]) if g["util"].isdigit() else 0
        bar_len = 20
        filled = int(bar_len * util / 100)
        color = GREEN if util > 70 else (YELLOW if util > 20 else DIM)
        bar = f"{color}{'█' * filled}{'░' * (bar_len - filled)}{RESET}"
        mem_str = f"{g['mem_used']}/{g['mem_total']} MB"
        pwr = g["power"] if g["power"] != "[N/A]" else "N/A"
        tmp = g["temp"] if g["temp"] != "[N/A]" else "N/A"
        lines.append(
            f"  {BOLD}{g['idx']:<4}{RESET}{g['name']:<22}"
            f"{bar} {util:>3}%"
            f"  {mem_str:>15}"
            f"  {pwr:>6}W"
            f"  {tmp:>4}°C"
        )
    return lines

# ── Job log tail ──────────────────────────────────────────────────────────────
def tail_file(path: str, n: int = 14) -> list[str]:
    if not os.path.exists(path):
        return [f"  {DIM}Waiting for log: {path}{RESET}"]
    try:
        with open(path, "rb") as f:
            # Fast tail: seek from end
            f.seek(0, 2)
            size = f.tell()
            chunk = min(size, 8192)
            f.seek(-chunk, 2)
            raw = f.read().decode(errors="replace")
        lines = raw.splitlines()[-n:]
        return [f"  {DIM}{l}{RESET}" if l.startswith("W0") or l.startswith("E0")
                else f"  {l}" for l in lines]
    except Exception as e:
        return [f"  {RED}Error reading log: {e}{RESET}"]

# ── Pod list ─────────────────────────────────────────────────────────────────
_POD_CACHE: str = ""
_POD_CACHE_TS: float = 0.0

def fetch_pod_status(max_age=15) -> str:
    global _POD_CACHE, _POD_CACHE_TS
    if time.time() - _POD_CACHE_TS < max_age:
        return _POD_CACHE
    try:
        out = subprocess.check_output(
            ["runpodctl", "get", "pod"], stderr=subprocess.DEVNULL, timeout=8
        ).decode().strip()
        _POD_CACHE = out
        _POD_CACHE_TS = time.time()
    except Exception:
        pass
    return _POD_CACHE

def render_pod_section(raw: str) -> list[str]:
    lines = [f"{BOLD}  Active Pods{RESET}"]
    if not raw:
        lines.append(f"  {DIM}runpodctl unavailable{RESET}")
        return lines
    for i, row in enumerate(raw.splitlines()):
        color = DIM if i == 0 else ""
        # Highlight RUNNING green, anything else yellow
        row_colored = row.replace("RUNNING", f"{GREEN}RUNNING{RESET}")
        lines.append(f"  {color}{row_colored}{RESET}")
    return lines

# ── Job ETA ──────────────────────────────────────────────────────────────────
def parse_job_start(log_path: str) -> float | None:
    """Return job start time from the log file's local creation timestamp.
    Uses st_birthtime on macOS, ctime fallback on Linux — both timezone-correct,
    avoiding the UTC/localtime mismatch when parsing server-side log timestamps."""
    if not os.path.exists(log_path):
        return None
    st = os.stat(log_path)
    # st_birthtime is available on macOS; fall back to st_ctime on Linux
    return getattr(st, "st_birthtime", st.st_ctime)

def job_is_done(log_path: str) -> bool:
    """True if any rank printed its final timing line."""
    if not os.path.exists(log_path):
        return False
    try:
        with open(log_path, "r", errors="replace") as f:
            return any("[rank" in l and "time=" in l for l in f)
    except Exception:
        return False

def render_eta(log_path: str, dashboard_start: float, expected_s: float = 300.0) -> list[str]:
    """
    Show job elapsed time and a simple ETA bar.
    expected_s: rough expected runtime in seconds (conservative 5 min default).
    """
    lines = [f"{BOLD}  Job Progress / ETA{RESET}"]

    job_start = parse_job_start(log_path)
    if job_start is None:
        lines.append(f"  {DIM}Waiting for job to start...{RESET}")
        return lines

    elapsed = time.time() - job_start
    td_elapsed = timedelta(seconds=int(elapsed))

    if job_is_done(log_path):
        lines.append(f"  {GREEN}{BOLD}✓ DONE{RESET}  — job finished after {td_elapsed}")
        return lines

    # Progress bar based on elapsed / expected
    pct = min(elapsed / expected_s, 0.99)
    bar_len = 30
    filled = int(bar_len * pct)
    color = GREEN if pct < 0.6 else (YELLOW if pct < 0.9 else RED)
    bar = f"{color}{'█' * filled}{'░' * (bar_len - filled)}{RESET}"

    remaining = max(expected_s - elapsed, 0)
    td_eta = timedelta(seconds=int(remaining))
    eta_str = f"~{td_eta} remaining" if remaining > 5 else f"{YELLOW}any moment now{RESET}"

    lines.append(f"  Elapsed: {BOLD}{str(td_elapsed):>8}{RESET}   {bar}   ETA: {BOLD}{eta_str}{RESET}")
    lines.append(f"  {DIM}(estimate based on {expected_s:.0f}s expected; MLMC adapts — may finish sooner){RESET}")
    return lines


# ── Cost tracker ─────────────────────────────────────────────────────────────
def render_cost(start_ts: float, cost_per_hour: float) -> str:
    elapsed = time.time() - start_ts
    td = timedelta(seconds=int(elapsed))
    cost = elapsed / 3600 * cost_per_hour
    bar_max = 10.0
    bar_len = 20
    filled = min(int(bar_len * cost / bar_max), bar_len)
    color = RED if cost > 5 else (YELLOW if cost > 2 else GREEN)
    bar = f"{color}{'█' * filled}{'░' * (bar_len - filled)}{RESET}"
    return (f"  Elapsed: {BOLD}{str(td):>8}{RESET}   "
            f"Cost: {color}{BOLD}${cost:.4f}{RESET}  {bar}  "
            f"{DIM}@ ${cost_per_hour}/hr{RESET}")

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="RunPod live dashboard")
    ap.add_argument("--log", default=(
        "/private/tmp/claude-501/"
        "-Users-paritoshdwivedi-projects-GPU-Acc-Net-Prop-Congestion-Multi-Monte-Carlo-paper/"
        "cb38ea42-23af-4abc-829c-5713a8f2504a/tasks/boi3fqtck.output"
    ))
    ap.add_argument("--pod-ip",   default="99.69.17.69")
    ap.add_argument("--pod-port", type=int, default=11408)
    ap.add_argument("--ssh-key",  default=os.path.expanduser("~/.runpod/ssh/RunPod-Key-Go"))
    ap.add_argument("--cost-per-hour", type=float, default=0.88)
    ap.add_argument("--refresh",    type=int,   default=5)
    ap.add_argument("--log-lines",  type=int,   default=14)
    ap.add_argument("--expected-s", type=float, default=300.0,
                    help="Expected job duration in seconds (for ETA bar)")
    args = ap.parse_args()

    start_ts = time.time()
    gpu_fetch_interval = max(args.refresh, 5)
    last_gpu_fetch = 0.0

    print(f"\033[?25l", end="")  # hide cursor
    try:
        while True:
            now_str = datetime.now().strftime("%H:%M:%S")

            # Fetch GPU stats (rate-limited to avoid SSH hammering)
            if time.time() - last_gpu_fetch >= gpu_fetch_interval:
                gpus = fetch_gpu_stats(args.pod_ip, args.pod_port, args.ssh_key)
                last_gpu_fetch = time.time()
            else:
                gpus = _GPU_CACHE
            gpu_stale = (time.time() - _GPU_CACHE_TS) > gpu_fetch_interval * 2

            pod_raw  = fetch_pod_status()
            log_tail = tail_file(args.log, args.log_lines)

            # ── Render ────────────────────────────────────────────────────────
            out = []
            out.append(hline("═", CYAN))
            out.append(header("RunPod GPU-MLMC Dashboard", now_str))
            out.append(hline("═", CYAN))
            out.append("")

            # GPU section
            out.extend(render_gpu_section(gpus, gpu_stale))
            out.append("")
            out.append(hline())

            # Pod section
            out.extend(render_pod_section(pod_raw))
            out.append("")
            out.append(hline())

            # ETA section
            out.extend(render_eta(args.log, start_ts, expected_s=args.expected_s))
            out.append("")
            out.append(hline())

            # Cost section
            out.append(f"{BOLD}  Cost Tracker{RESET}")
            out.append(render_cost(start_ts, args.cost_per_hour))
            out.append("")
            out.append(hline())

            # Log tail
            out.append(f"{BOLD}  Job Output  {DIM}{args.log[-60:]}{RESET}")
            out.extend(log_tail)
            out.append("")
            out.append(hline("─", DIM))
            out.append(f"  {DIM}Refresh every {args.refresh}s — Ctrl+C to exit{RESET}")

            # Write all at once to minimise flicker
            clr()
            sys.stdout.write("\n".join(out) + "\n")
            sys.stdout.flush()

            time.sleep(args.refresh)

    except KeyboardInterrupt:
        pass
    finally:
        print("\033[?25h", end="")  # restore cursor
        print()


if __name__ == "__main__":
    main()
