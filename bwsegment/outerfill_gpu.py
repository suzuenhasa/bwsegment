"""
outerfill_gpu.py — GPU (torch + cupy) drop-in for the CPU `close_fill` path of
outerfill.consolidate().

Why: outerfill.consolidate() is the segment pipeline's bottleneck (1.3-2.8 s/page)
because it does, per connected component, a full-image `np.where` scan plus a
native-resolution cv2 MORPH_CLOSE + scipy.binary_fill_holes. That is
O(components x pixels).

This module keeps the SAME public API and the SAME semantics as the CPU
`close_fill` variant, but moves the heavy work onto the GPU:

  * morphology (bridge dilation + CLOSE)  -> torch F.max_pool2d
        dilate = max_pool2d(x, k, 1, k//2)
        erode  = -max_pool2d(-x, k, 1, k//2)
        CLOSE  = erode(dilate(x))            (square SE ~= elliptical SE)
  * fill-holes                            -> cupyx.scipy.ndimage.binary_fill_holes
        the connected-component (CCL) based fill (structure=None, the fast
        non-iterative path), reached zero-copy from the torch mask via DLPack.
        This replaces the old GPU morphological-reconstruction flood that
        iterated ~1000x and was the module's own bottleneck.
  * connected components + drop_area + boxes -> cupyx.scipy.ndimage.label
        run ON the filled mask. Components are used ONLY to (a) drop tiny
        regions by area and (b) report per-region boxes. They are NEVER used to
        re-open filled interior: the cupy fill result is authoritative for which
        pixels are foreground.

Semantic notes vs the per-group CPU path:
  Groups produced by `bridge` are separated by gaps > merge_dist. The CLOSE
  kernel (~merge_dist) bridges components within merge_dist, so the connected
  components OF THE CLOSED+FILLED mask are the same "groups" the CPU code forms
  by dilating with a merge_dist kernel. A single GLOBAL close+fill over the
  whole mask is therefore equivalent to the CPU's per-group close+fill, unioned.
  The one intentional approximation is a square structuring element instead of
  cv2's ellipse for the close.

  Dense-page fix: the old code gated the output on a downsampled group-label map
  and so re-dropped just-filled interior pixels on dense pages. Here the cupy
  fill is the output foreground directly; CCL only removes whole tiny components
  and never carves the interior back out.

Public API (identical to outerfill.consolidate):
    consolidate(binary_mask, variant, merge_dist, drop_area) -> (out_mask, boxes)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# reuse the exact CPU helpers so binarisation / kernel sizing match bit-for-bit
from .outerfill import _to_binary, _odd
from . import outerfill as _cpu


# ---------------------------------------------------------------------------
# GPU morphology primitives (operate on float {0,1} tensors, shape (1,1,H,W))
# ---------------------------------------------------------------------------

def _dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


def _erode(x: torch.Tensor, k: int) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)


def _close(x: torch.Tensor, k: int) -> torch.Tensor:
    """Morphological CLOSE with a (k x k) square SE, k odd."""
    return _erode(_dilate(x, k), k)


# ---------------------------------------------------------------------------
# fill-holes: cupyx CCL-based binary_fill_holes (non-iterative), DLPack in/out
# ---------------------------------------------------------------------------

def _fill_holes_gpu(mask: torch.Tensor) -> torch.Tensor:
    """Fill interior holes of a 2-D binary torch tensor (bool/uint8, on cuda).

    Uses cupyx.scipy.ndimage.binary_fill_holes with the DEFAULT structure
    (structure=None), which is the connected-component-based fill -- a single
    label + border-touch test, not the old ~1000-iteration geodesic flood.

    The torch mask is handed to cupy zero-copy via DLPack, and the filled result
    is handed back to torch zero-copy via DLPack, staying on the same cuda
    device throughout. Returns a bool torch tensor with the same shape.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import binary_fill_holes

    cm = cp.from_dlpack(torch.utils.dlpack.to_dlpack(mask.to(torch.uint8)))
    # structure=None -> default (4-connected) CCL fill; do NOT pass a structure.
    filled = binary_fill_holes(cm.astype(cp.bool_))
    return torch.from_dlpack(filled.astype(cp.uint8)).to(torch.bool)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def consolidate(binary_mask, variant="close_fill", merge_dist=40, drop_area=150,
                *, device="cuda", ds=4, return_debug=False):
    """GPU drop-in for outerfill.consolidate(..., variant="close_fill").

    Same return contract:
        out_mask : uint8 {0,1} np.ndarray, same HxW as the input.
        boxes    : list of (x0, y0, x1, y1), x1/y1 exclusive, one per surviving
                   region.

    Extra (keyword-only) knobs that do not change the contract:
        device : torch device for the morphology ("cuda").
        ds     : accepted for backward compatibility; unused by the CCL fill.
    """
    if variant not in ("hull", "close_fill"):
        raise ValueError(f"unknown variant {variant!r}")
    if variant == "hull":
        # hull path is cheap/rare; defer to the exact CPU implementation.
        return _cpu.consolidate(binary_mask, variant="hull",
                                merge_dist=merge_dist, drop_area=drop_area)

    fg_np = _to_binary(binary_mask)                 # uint8 {0,1}, HxW
    h, w = fg_np.shape
    out = np.zeros((h, w), np.uint8)
    boxes = []
    if fg_np.sum() == 0:
        return (out, boxes) if not return_debug else (out, boxes, {})

    # cupy needs a real GPU; if there is none, defer to the exact CPU path.
    if not torch.cuda.is_available():
        return _cpu.consolidate(binary_mask, variant="close_fill",
                                merge_dist=merge_dist, drop_area=drop_area)

    import cupy as cp
    from cupyx.scipy.ndimage import label, find_objects

    dev = torch.device(device)
    fg = torch.from_numpy(fg_np).to(dev, torch.float32)[None, None]  # (1,1,H,W)

    # close kernel size: identical to the CPU code
    ck = _odd(max(3, merge_dist))

    # --- GPU morphology: merge_dist CLOSE (global == per-group union) ---------
    closed = (_close(fg, ck) > 0.5)[0, 0]           # (H,W) bool on device

    # --- cupy CCL-based fill-holes: this is the AUTHORITATIVE foreground ------
    filled = _fill_holes_gpu(closed)                # (H,W) bool on device

    # --- connected components ON the filled mask (boxes + drop tiny only) -----
    filled_cp = cp.from_dlpack(
        torch.utils.dlpack.to_dlpack(filled.to(torch.uint8)))
    struct8 = cp.ones((3, 3), cp.int32)             # 8-connectivity grouping
    labels, n_comp = label(filled_cp, structure=struct8)

    # area per label (index 0 == background)
    areas = cp.bincount(labels.ravel(), minlength=n_comp + 1)

    # a component survives iff its (filled) area >= drop_area
    survive = areas >= int(drop_area)
    survive[0] = False

    # out mask = filled foreground restricted to surviving components. Interior
    # is never re-opened: we only zero out whole tiny components.
    keep_cp = survive[labels]                       # bool (H,W) on GPU
    out = cp.asnumpy(keep_cp.astype(cp.uint8))

    # --- per-region boxes from tight per-label slices (cupy find_objects) -----
    slices = find_objects(labels)                   # list length n_comp
    survive_host = cp.asnumpy(survive)
    for i, sl in enumerate(slices, start=1):
        if sl is None or not survive_host[i]:
            continue
        ys, xs = sl                                 # tight to label i's pixels
        boxes.append((int(xs.start), int(ys.start),
                      int(xs.stop), int(ys.stop)))   # x1/y1 exclusive

    if return_debug:
        n_survive = int(survive_host[1:].sum()) if n_comp else 0
        dbg = dict(n_comp=int(n_comp), n_survive=n_survive,
                   fill="cupyx.binary_fill_holes(structure=None)")
        return out, boxes, dbg
    return out, boxes


# ---------------------------------------------------------------------------
# smoke test / parity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, time, glob

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        png = args[0]
    else:
        cands = (glob.glob("/workspace/*_A_figurekeep.png")
                 + glob.glob("out/*_A_figurekeep.png")
                 + glob.glob("*_A_figurekeep.png"))
        png = cands[0] if cands else None

    MD, DA = 55, 600
    print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    if png is None:
        print("no mask png found; running a tiny synthetic instead")
        fg = _cpu.build_synthetic()
    else:
        from PIL import Image
        fg = _to_binary(np.array(Image.open(png).convert("RGBA")))
        print(f"mask: {png}  shape={fg.shape}  fg%={fg.mean()*100:.2f}")

    # warmup (cuda init / kernel compile) then time
    _ = consolidate(fg, "close_fill", MD, DA)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    out_g, boxes_g, dbg = consolidate(fg, "close_fill", MD, DA, return_debug=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_gpu = time.time() - t0

    t0 = time.time()
    out_c, boxes_c = _cpu.consolidate(fg, "close_fill", MD, DA)
    t_cpu = time.time() - t0

    inter = int(np.logical_and(out_g > 0, out_c > 0).sum())
    union = int(np.logical_or(out_g > 0, out_c > 0).sum())
    iou = inter / union if union else 1.0

    print(f"GPU: {t_gpu*1000:7.1f} ms   boxes={len(boxes_g)}   debug={dbg}")
    print(f"CPU: {t_cpu*1000:7.1f} ms   boxes={len(boxes_c)}")
    print(f"speedup x{t_cpu/t_gpu:.1f}   mask IoU(GPU,CPU)={iou:.4f}")
    print(f"GPU boxes: {boxes_g}")
    print(f"CPU boxes: {boxes_c}")
