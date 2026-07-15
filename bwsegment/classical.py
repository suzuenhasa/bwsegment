"""
classical.py — GPU-accelerated CLASSICAL content detector (no model, no weights).

This is the "old classical" figure/content finder from
`segment/invert/invert_seg.py` (function `content_masks`), ported to run on the
GPU with torch so it is fast enough to ship as a first-class, per-page selectable
detector alongside the CNN.

Why keep it? Classical and CNN win on *different* pages:
  * CLASSICAL captures the WHOLE illustrated page (full-bleed art + border)
    better — it keys on contrast/ink-density, so a busy page reads as one big
    content region (e.g. the `tm_p007` illustrated page).
  * CNN is TIGHTER on clean-paper pages where the artwork is a well-separated
    island on white.

Algorithm (identical logic to the CPU original, computed on GPU):
  1. PAPER BASELINE via a large grayscale morphological CLOSE
     (dilate = max_pool2d ; erode = -max_pool2d(-x) ; CLOSE = erode(dilate)),
     then smoothed. The CPU original smoothed the CLOSE with a cv2.medianBlur;
     on GPU we approximate that with a large-kernel average blur (a slowly
     varying illumination field is all the baseline needs to be).
  2. CONTRAST: a pixel is CONTENT when it is meaningfully DARKER than its local
     paper baseline:
         rel = (baseline - gray) / baseline  > REL_THRESH     (illumination-robust)
       AND diff = (baseline - gray)          > ABS_MIN         (absolute floor)
  3. TEXTURE (halftone backstop): high local std + at least slightly darker:
         local_std > STD_THRESH  AND  rel > STD_REL_MIN
     local_std via avg_pool2d(x^2) - avg_pool2d(x)^2.
  4. content = (contrast | texture), de-speckled with a small morphological OPEN.

`classical_content()` returns the strict content mask (HxW uint8, 255 = content).
It needs NO weights. On CUDA it runs in a few ms; with no CUDA it transparently
falls back to the EXACT cv2/numpy path from the original `content_masks`.

Tuned defaults are exposed as documented params (same values as the original).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

try:  # cv2 is only needed for the CPU fallback path
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------------------
# Tuned defaults (identical to segment/invert/invert_seg.py)
# ---------------------------------------------------------------------------
K_BASE       = 41     # px, kernel for the paper-baseline grayscale CLOSE
BASE_SMOOTH  = 15     # px, smoothing of the baseline (median on CPU / avg on GPU)
REL_THRESH   = 0.14   # strict: pixel must be >=14% darker than its local paper
ABS_MIN      = 10     # strict: and at least 10 gray-levels darker (noise floor)
STD_WIN      = 15     # px, window for the local-texture std
STD_THRESH   = 22.0   # texture std above which a halftone pixel counts as content
STD_REL_MIN  = 0.03   # texture pixels must also be >=3% darker than paper
OPEN_K       = 3      # px, small morphological OPEN to de-speckle the raw mask


# ---------------------------------------------------------------------------
# GPU morphology / pooling primitives (operate on (1,1,H,W) float tensors)
#   reflect padding is used so borders behave like cv2's BORDER_DEFAULT rather
#   than being pulled toward 0 by zero-padding.
# ---------------------------------------------------------------------------

def _dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    p = k // 2
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    return F.max_pool2d(xp, kernel_size=k, stride=1, padding=0)


def _erode(x: torch.Tensor, k: int) -> torch.Tensor:
    return -_dilate(-x, k)


def _close(x: torch.Tensor, k: int) -> torch.Tensor:
    """Grayscale morphological CLOSE with a (k x k) square SE (k odd)."""
    return _erode(_dilate(x, k), k)


def _open(x: torch.Tensor, k: int) -> torch.Tensor:
    """Morphological OPEN with a (k x k) square SE (k odd)."""
    return _dilate(_erode(x, k), k)


def _boxmean(x: torch.Tensor, k: int) -> torch.Tensor:
    """Local mean via avg_pool2d with reflect padding (== cv2.boxFilter normed)."""
    p = k // 2
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    return F.avg_pool2d(xp, kernel_size=k, stride=1, padding=0)


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

def _classical_content_gpu(gray: np.ndarray, device: str,
                           k_base: int, base_smooth: int,
                           rel_thresh: float, abs_min: float,
                           std_win: int, std_thresh: float, std_rel_min: float,
                           open_k: int) -> np.ndarray:
    dev = torch.device(device)
    g = torch.from_numpy(gray.astype(np.float32)).to(dev)[None, None]  # (1,1,H,W)

    # 1) paper baseline: large grayscale CLOSE, then smooth to an illumination field
    base = _close(g, _odd(k_base))
    if base_smooth and base_smooth >= 3:
        base = _boxmean(base, _odd(base_smooth))

    # 2) contrast (illumination-robust darkness relative to local paper)
    diff = base - g
    rel = diff / base.clamp(min=1.0)

    # 3) local-std texture: var = E[x^2] - E[x]^2
    m = _boxmean(g, _odd(std_win))
    m2 = _boxmean(g * g, _odd(std_win))
    std = (m2 - m * m).clamp(min=0.0).sqrt()

    ink = (rel > rel_thresh) & (diff > abs_min)
    texture = (std > std_thresh) & (rel > std_rel_min)
    strict = (ink | texture).to(torch.float32)

    # 4) de-speckle with a small OPEN
    if open_k and open_k >= 3:
        strict = _open(strict, _odd(open_k))

    mask = (strict[0, 0] > 0.5).to(torch.uint8) * 255
    return mask.cpu().numpy()


# ---------------------------------------------------------------------------
# EXACT CPU fallback (the original cv2 / numpy content_masks path)
# ---------------------------------------------------------------------------

def _classical_content_cpu(gray: np.ndarray,
                           k_base: int, base_smooth: int,
                           rel_thresh: float, abs_min: float,
                           std_win: int, std_thresh: float, std_rel_min: float,
                           open_k: int) -> np.ndarray:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("cv2 is required for the classical CPU fallback")

    # paper baseline: grayscale CLOSE then median smooth (as in invert_seg.py)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k_base), _odd(k_base)))
    base = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    if base_smooth and base_smooth >= 3:
        base = cv2.medianBlur(base, _odd(base_smooth))
    base = base.astype(np.float32)

    g = gray.astype(np.float32)
    diff = base - g
    rel = diff / np.clip(base, 1.0, None)

    # local std via box filters
    win = _odd(std_win)
    mean = cv2.boxFilter(g, ddepth=-1, ksize=(win, win), normalize=True)
    mean_sq = cv2.boxFilter(g * g, ddepth=-1, ksize=(win, win), normalize=True)
    std = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))

    ink = (rel > rel_thresh) & (diff > abs_min)
    texture = (std > std_thresh) & (rel > std_rel_min)
    strict = (ink | texture).astype(np.uint8)

    if open_k and open_k >= 3:
        ok = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(open_k), _odd(open_k)))
        strict = cv2.morphologyEx(strict, cv2.MORPH_OPEN, ok)

    return (strict > 0).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def classical_content(image_bgr: np.ndarray, device: str = "cuda:0",
                      *, k_base: int = K_BASE, base_smooth: int = BASE_SMOOTH,
                      rel_thresh: float = REL_THRESH, abs_min: float = ABS_MIN,
                      std_win: int = STD_WIN, std_thresh: float = STD_THRESH,
                      std_rel_min: float = STD_REL_MIN,
                      open_k: int = OPEN_K) -> np.ndarray:
    """Classical (model-free) content mask via local contrast + halftone texture.

    Args:
        image_bgr  : HxWx3 uint8 BGR page (also accepts a HxW grayscale array).
        device     : torch device string, e.g. "cuda:0". Falls back to an exact
                     cv2/numpy CPU path when CUDA is unavailable.
        k_base     : px kernel for the paper-baseline grayscale CLOSE.
        base_smooth: px smoothing of the baseline illumination field.
        rel_thresh : relative darkness threshold ((base-gray)/base) for ink.
        abs_min    : absolute darkness floor (base-gray) for ink (noise gate).
        std_win    : window (px) for the local-texture std.
        std_thresh : local-std above which a halftone pixel counts as content.
        std_rel_min: texture pixels must also be at least this much darker.
        open_k     : px kernel for the small de-speckle OPEN (0/<3 disables).

    Returns:
        content_mask : HxW uint8, 255 = content, 0 = paper. Same size as input.
    """
    if image_bgr.ndim == 3:
        if cv2 is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:  # pragma: no cover - luminance fallback if cv2 missing
            b, g, r = (image_bgr[..., i].astype(np.float32) for i in range(3))
            gray = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
    else:
        gray = image_bgr
    gray = np.ascontiguousarray(gray, dtype=np.uint8)

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        return _classical_content_gpu(
            gray, device, k_base, base_smooth, rel_thresh, abs_min,
            std_win, std_thresh, std_rel_min, open_k)
    return _classical_content_cpu(
        gray, k_base, base_smooth, rel_thresh, abs_min,
        std_win, std_thresh, std_rel_min, open_k)
