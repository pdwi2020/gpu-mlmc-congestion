#!/usr/bin/env python3
"""Generate Figure 1: the GPU-MLMC workflow diagram for the IEEE Access paper.

Reviewer 2 asked for "a workflow or implementation diagram summarizing the
interaction between the MLMC hierarchy, GPU execution, and weighted sampling
strategy."  The layout is three horizontal bands:

  * HOST (top)   : input -> network-risk weighting -> ANA sample allocation
  * DEVICE (mid) : MLMC level hierarchy -> coupled-SDE EM kernel -> multi-GPU halo
  * HOST (bottom): uncertainty-quantified outputs (CI, tail quantiles, rare-event P)

A dashed feedback edge carries pilot-run per-node variance back into both the
weighting and the allocation blocks -- the loop reviewers need to see.

Design constraints (IEEE Access):
  * double-column width, vector PDF, no Type-3 fonts (fonttype 42)
  * greyscale-legible: host vs device distinguished by fill + hatch, not colour
  * no usetex, no network access -- reproducible on a bare matplotlib install

Run:  python scripts/gen_workflow_figure.py
Out:  paper/ieee_access/fig_workflow.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- embed real (Type-1/TrueType) fonts, never Type-3 -----------------------
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]

OUT = Path(__file__).resolve().parents[1] / "paper" / "ieee_access" / "fig_workflow.pdf"

# greyscale-safe fills
HOST_FILL = "#e9e9e9"     # light grey, solid border  -> host / CPU
DEV_FILL = "#ffffff"      # white, hatched            -> device / GPU
ACCENT = "#c7c7c7"

FIG_W, FIG_H = 7.16, 3.05   # inches: IEEE Access double column


def box(ax, x, y, w, h, title, lines, *, device=False):
    """Draw a rounded process box.

    Text always sits on a clean fill.  Device (GPU) boxes are distinguished
    from host (CPU) boxes by a heavier border and a small hatched corner tab
    labelled 'GPU' -- greyscale-legible without hatching over the text.
    """
    fc = DEV_FILL if device else HOST_FILL
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.014",
        linewidth=1.6 if device else 1.0, edgecolor="black", facecolor=fc,
        zorder=3,
    )
    ax.add_patch(patch)
    if device:
        # hatched corner tab in the top-right, kept clear of the text column
        tabw, tabh = 0.052, 0.05
        ax.add_patch(FancyBboxPatch(
            (x + w - tabw - 0.012, y + h - tabh - 0.012), tabw, tabh,
            boxstyle="round,pad=0.002,rounding_size=0.008",
            linewidth=0.8, edgecolor="black", facecolor="#ffffff",
            hatch="////", zorder=5))
        ax.text(x + w - tabw / 2 - 0.012, y + h - tabh / 2 - 0.012, "GPU",
                ha="center", va="center", fontsize=4.6, fontweight="bold",
                zorder=6)
    # keep the centred title clear of the GPU corner tab
    title_cx = (x + w / 2 - 0.03) if device else (x + w / 2)
    ax.text(title_cx, y + h - 0.052, title, ha="center", va="top",
            fontsize=7.4, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h - 0.135, "\n".join(lines), ha="center", va="top",
            fontsize=6.0, zorder=4, linespacing=1.28)
    return patch


def arrow(ax, p0, p1, *, dashed=False, rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=11, linewidth=lw,
        linestyle=(0, (4, 2)) if dashed else "solid",
        color="black", zorder=2,
    ))


def band_label(ax, y, text):
    ax.text(0.012, y, text, ha="left", va="center", fontsize=6.4,
            fontweight="bold", rotation=90, color="#404040")


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # band backgrounds
    for y0, y1 in [(0.70, 0.985), (0.365, 0.655), (0.03, 0.315)]:
        ax.add_patch(FancyBboxPatch(
            (0.045, y0), 0.94, y1 - y0,
            boxstyle="round,pad=0.002,rounding_size=0.01",
            linewidth=0, facecolor=ACCENT, alpha=0.16, zorder=0))
    band_label(ax, 0.83, "HOST (CPU)")
    band_label(ax, 0.51, "DEVICE (GPU)")
    band_label(ax, 0.17, "HOST (CPU)")

    bw, bh = 0.265, 0.235
    xL, xM, xR = 0.075, 0.3675, 0.66

    # --- top band: host preprocessing --------------------------------------
    yT = 0.725
    b_in = box(ax, xL, yT, bw, bh, "Input",
               ["CAIDA / ER / BA topology  ->  A",
                "MAWI trace  ->  lambda_i(t)"])
    b_w = box(ax, xM, yT, bw, bh, "Network-risk weighting",
              ["PageRank c_i,  pilot var v_i,",
               "SLA s_i  ->  w_i  (Eq. 10)"])
    b_alloc = box(ax, xR, yT, bw, bh, "ANA allocation",
                  ["N_l ~ sqrt( V_l^w / C_l )",
                   "weighted variance (Eq. 8-9)"])

    # --- middle band: GPU execution (read right-to-left underneath) --------
    yM = 0.39
    b_mlmc = box(ax, xR, yM, bw, bh, "MLMC hierarchy",
                 ["levels l = 0..L,  h_l = h_0/2^l",
                  "coupled fine/coarse, CRN"], device=True)
    b_kern = box(ax, xM, yM, bw, bh, "Coupled-SDE kernel",
                 ["batched Philox randn,  A C",
                  "Skorokhod reflect, 2-bucket EM step"], device=True)
    b_multi = box(ax, xL, yM, bw, bh, "Multi-GPU halo",
                  ["METIS partition x G ranks",
                   "NCCL all-reduce every K steps"], device=True)

    # --- bottom band: outputs ----------------------------------------------
    yB = 0.055
    b_out = box(ax, xM, yB, bw, bh, "Uncertainty-quantified output",
                ["mean queue occupancy + 95% CI",
                 "P95 / P99 tail delay",
                 "rare-event overflow P (IS)"])

    # --- forward edges ------------------------------------------------------
    arrow(ax, (xL + bw, yT + bh / 2), (xM, yT + bh / 2))            # in -> weight
    arrow(ax, (xM + bw, yT + bh / 2), (xR, yT + bh / 2))           # weight -> alloc
    arrow(ax, (xR + bw / 2, yT), (xR + bw / 2, yM + bh))          # alloc -> mlmc
    arrow(ax, (xR, yM + bh / 2), (xM + bw, yM + bh / 2))          # mlmc -> kernel
    arrow(ax, (xM, yM + bh / 2), (xL + bw, yM + bh / 2))          # kernel -> multi
    # multi-GPU + kernel converge to output
    arrow(ax, (xM + bw / 2, yM), (xM + bw / 2, yB + bh))          # kernel -> output

    # --- pilot-variance feedback loop (the part reviewers want to see) -----
    # from the GPU MLMC hierarchy back up into weighting and allocation
    arrow(ax, (xR + bw * 0.5, yM + bh), (xR + bw * 0.86, yT), dashed=True, rad=-0.28)
    arrow(ax, (xR + bw * 0.14, yM + bh), (xM + bw * 0.5, yT), dashed=True, rad=0.30)
    ax.text(xR + bw + 0.008, (yT + yM + bh) / 2 + 0.02,
            "pilot per-node\nvariance v_i",
            ha="left", va="center", fontsize=5.7, style="italic", color="#333333")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
