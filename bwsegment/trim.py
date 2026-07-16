"""
trim.py — safe paper trimming inside a region (the "rect + safe-trim" recipe).

    mask = (union image_boxes) MINUS (union text_boxes)     <- region, never looks at pixels
           MINUS safe_trim(flat AND paper-toned AND connected-to-outside)
           MINUS specks

Why this exists
---------------
On a scanned page, parts of the artwork are genuinely PAPER-TONED — a light-grey
statue photo and beige temple stone read at the same tone as the aged paper
around them. Every appearance-based detector (local contrast, a paper/content
CNN, a magic-wand colour flood) therefore EATS INTO the art: there is nothing to
separate, tone-wise.

So don't judge art. Start from the region locate already gives us (image_boxes),
where the default is KEEP, and only carve away paper we are *confident* about.
A pixel is confidently paper only if ALL THREE hold:

  1. FLAT              — local std < `flat`. Paper is featureless; paper-toned ART
                         still carries photographic micro-texture. This is the cue
                         that separates the two when tone cannot.
  2. PAPER-TONED       — CIELAB distance to the measured paper colour < `tone`.
                         Uses colour, not brightness: beige stone and blue sky are
                         far from cream paper in Lab even at equal brightness.
  3. CONNECTED-TO-OUTSIDE — reachable by a flood from the page border through
                         other candidate-paper pixels. Paper *enclosed inside* the
                         artwork is never trimmed.

INVARIANT: a textured pixel inside an image_box and outside every text_box is
always kept. The recipe cannot eat art; its only failure mode is keeping some
paper — the conservative, recoverable direction.
"""

from __future__ import annotations

import cv2
import numpy as np

# a pixel must be flatter than this (local std, 9x9) to be trim candidate paper
DEFAULT_FLAT = 6.0
# ...and within this CIELAB distance of the measured paper colour
DEFAULT_TONE = 12.0


def paper_lab(image_bgr: np.ndarray, outside: np.ndarray):
    """Measure the page's paper colour in CIELAB from pixels OUTSIDE the region.

    Takes the brighter tail (>=85th percentile - 8) of the outside area so ink
    and furniture don't drag the reference dark. Returns None if unmeasurable.
    """
    if not outside.any():
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tone = np.percentile(gray[outside], 85)
    sel = outside & (gray > tone - 8)
    if not sel.any():
        return None
    return np.array([lab[:, :, c][sel].mean() for c in range(3)], np.float32)


def local_std(image_bgr: np.ndarray, win: int = 9) -> np.ndarray:
    """Local standard deviation of luminance — the 'texture' cue."""
    g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.boxFilter(g, -1, (win, win))
    mu2 = cv2.boxFilter(g * g, -1, (win, win))
    return np.sqrt(np.maximum(mu2 - mu * mu, 0))


def safe_trim(image_bgr: np.ndarray, region: np.ndarray,
              flat: float = DEFAULT_FLAT, tone: float = DEFAULT_TONE):
    """Carve confidently-paper pixels out of `region`. Cannot remove textured art.

    Args:
        image_bgr: HxWx3 uint8 page.
        region   : HxW uint8/bool, non-zero = candidate artwork region.
        flat     : local-std ceiling for "featureless".
        tone     : CIELAB distance ceiling for "paper-coloured".

    Returns:
        (trimmed_bool, sd) — trimmed region, and the local-std map (reusable).
    """
    h, w = image_bgr.shape[:2]
    region_b = np.asarray(region) > 0
    outside = ~region_b
    sd = local_std(image_bgr)

    ref = paper_lab(image_bgr, outside)
    if ref is None:                      # no outside to measure paper from
        return region_b, sd

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # (1) flat AND (2) paper-toned
    cand = (sd < flat) & (np.linalg.norm(lab - ref[None, None, :], axis=2) < tone)

    # (3) connected-to-outside: flood from the page border through candidate paper
    ff = (cand | outside).astype(np.uint8)
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(0, w, 25):
        for y in (0, h - 1):
            if ff[y, x] == 1:
                cv2.floodFill(ff, ffmask, (x, y), 2)
    for y in range(0, h, 25):
        for x in (0, w - 1):
            if ff[y, x] == 1:
                cv2.floodFill(ff, ffmask, (x, y), 2)

    return region_b & ~(ff == 2), sd


def despeck(mask: np.ndarray, min_area: int | None = None) -> np.ndarray:
    """Drop connected components smaller than `min_area` (default: max(2000, 0.03%))."""
    m = np.asarray(mask)
    if min_area is None:
        min_area = max(2000, int(0.0003 * m.size))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8) * 255, 8)
    keep = np.zeros(m.shape, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep |= (lbl == i)
    return keep
