#!/usr/bin/env python3
"""Regenerate the multi-GPU scaling figure as a vertically-stacked (3x1) panel
layout for single-column placement, per co-author (AK) review request.

Data source: results/multi_gpu/distributed_scaling_results.json -- the real
distributed NCCL run on 4xRTX 3090 whose numbers match Table 16 / the figure
caption exactly (weak 100/99/43%, strong 1.00x/0.82x/0.65x, estimates
12.902/12.923/12.845, mean 12.890). No re-running; plot-only, CPU-only.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "results", "multi_gpu", "distributed_scaling_results.json")
OUT = os.path.join(ROOT, "paper", "ieee_access", "multi_gpu_scaling.png")

with open(DATA) as f:
    d = json.load(f)

weak = d["weak_scaling"]["results"]
strong = d["strong_scaling"]["results"]

ws_w = [r["world_size"] for r in weak]
eff = [r["efficiency_pct"] for r in weak]

ws_s = [r["world_size"] for r in strong]
speedup = [r["speedup"] for r in strong]
est = [r["estimate"] for r in strong]
mean_est = float(np.mean(est))

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.35,
})

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.5, 7.4))

# --- Panel 1: weak-scaling efficiency (bars) ---
xpos = np.arange(len(ws_w))
bar_colors = ["#1f77b4", "#4c9bd6", "#aacbe8"]
ax1.bar(xpos, eff, color=bar_colors[: len(ws_w)], edgecolor="#12496b", width=0.62)
ax1.axhline(100, ls="--", color="0.4", lw=1.2)
ax1.text(xpos[-1], 112, "Ideal (100%)", color="0.4", ha="center", va="center",
         fontsize=8)
for x, v in zip(xpos, eff):
    ax1.text(x, v + 2.5, f"{v}%", ha="center", va="bottom", fontweight="bold",
             fontsize=8.5)
ax1.set_xticks(xpos)
ax1.set_xticklabels(ws_w)
ax1.set_ylim(0, 125)
ax1.set_xlabel("GPUs (world size)")
ax1.set_ylabel("Weak-scaling eff. (%)")
ax1.set_title("Weak Scaling (PCIe RTX 3090, NCCL)")

# --- Panel 2: strong-scaling speedup (lines) ---
ax2.plot(ws_s, ws_s, ls="--", color="0.45", marker="", label="Ideal linear")
ax2.plot(ws_s, speedup, color="#e6550d", marker="o", lw=2,
         label="Actual (NCCL)")
for x, v in zip(ws_s, speedup):
    xytext = (-6, 9) if x == ws_s[-1] else (6, -11)
    ax2.annotate(f"{v:.2f}×", (x, v), textcoords="offset points",
                 xytext=xytext, fontsize=8)
ax2.set_xticks(ws_s)
ax2.set_xlim(0.7, 4.45)
ax2.set_ylim(0, max(ws_s) + 0.5)
ax2.set_xlabel("GPUs (world size)")
ax2.set_ylabel("Speedup")
ax2.set_title("Strong Scaling ($n{=}1000$, post-warmup)")
ax2.legend(loc="upper left", framealpha=0.9)

# --- Panel 3: estimate consistency (line) ---
ax3.plot(ws_s, est, color="#2ca02c", marker="s", lw=2, label="Estimate")
ax3.axhline(mean_est, ls="--", color="0.45", lw=1.2,
            label=f"Mean = {mean_est:.3f}")
for x, v in zip(ws_s, est):
    xytext = (-8, 7) if x == ws_s[-1] else (6, 5)
    ax3.annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                 xytext=xytext, fontsize=8)
ax3.set_xticks(ws_s)
ax3.set_xlim(0.7, 4.45)
lo, hi = min(est), max(est)
pad = (hi - lo) * 0.6 + 1e-3
ax3.set_ylim(lo - pad, hi + pad)
ax3.set_xlabel("GPUs (world size)")
ax3.set_ylabel("Queue-length estimate")
ax3.set_title("Estimate Consistency ($n{=}1000$)")
ax3.legend(loc="lower left", framealpha=0.9)

fig.tight_layout(h_pad=1.4)
fig.savefig(OUT, dpi=200)
print("wrote", OUT)
print("weak eff:", eff, "| strong speedup:", speedup, "| est:", est,
      "| mean:", round(mean_est, 4))
