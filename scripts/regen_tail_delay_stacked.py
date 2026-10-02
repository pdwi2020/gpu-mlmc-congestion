#!/usr/bin/env python3
"""Restack the 2-panel tail-delay validation figure vertically for single-column
placement.

The underlying ECDF sample arrays for this A100 run are not cached, so we do NOT
re-run (which would change the reported P95=0.133 / P99=0.217 numbers). Instead we
split the EXISTING published figure (backed up in figs_prestack_backup/) into its
two panels along the white gutter and stack them vertically. Exact same pixels,
just ~2x larger per panel at column width.
"""
import os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "paper", "ieee_access", "figs_prestack_backup",
                   "tail_delay_validation_a100.png")
OUT = os.path.join(ROOT, "paper", "ieee_access", "tail_delay_validation_a100.png")

img = Image.open(SRC).convert("RGB")
arr = np.asarray(img)
h, w, _ = arr.shape

# white-column detection: a column is "white" if all its pixels are near-white
col_is_white = (arr.min(axis=0) > 245).all(axis=1)  # per column (min over rows)

# find the widest white run whose centre is in the central 35-65% band
runs = []
i = 0
while i < w:
    if col_is_white[i]:
        j = i
        while j < w and col_is_white[j]:
            j += 1
        runs.append((i, j))
        i = j
    else:
        i += 1
central = [(a, b) for (a, b) in runs if 0.33 * w < (a + b) / 2 < 0.67 * w]
if central:
    a, b = max(central, key=lambda r: r[1] - r[0])
    split = (a + b) // 2
else:
    split = w // 2  # fallback

def trim(a):
    """trim surrounding all-white border of a panel array."""
    g = a.min(axis=2)
    rows = np.where((g < 245).any(axis=1))[0]
    cols = np.where((g < 245).any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return a
    pad = 6
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, a.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, a.shape[1])
    return a[r0:r1, c0:c1]

left = trim(arr[:, :split])
right = trim(arr[:, split:])

# match widths (pad narrower panel to the wider one, centred, white)
W = max(left.shape[1], right.shape[1])
def padw(a):
    if a.shape[1] == W:
        return a
    extra = W - a.shape[1]
    l = extra // 2
    r = extra - l
    return np.pad(a, ((0, 0), (l, r), (0, 0)), constant_values=255)

left, right = padw(left), padw(right)
gap = np.full((28, W, 3), 255, np.uint8)  # vertical spacer between panels
stacked = np.vstack([left, gap, right])
Image.fromarray(stacked).save(OUT)
print("split at x =", split, "| out size =", stacked.shape[1], "x", stacked.shape[0])
