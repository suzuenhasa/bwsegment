"""
outerfill.py — mask post-processing for the typesetter `segment` replacement.

Fixes the behaviour of the old `_clean_cutout`, which DROPS small components
(specks) and thereby deletes dots that legitimately belong to a figure (e.g. a
spray of dots scattering off the main blob).

This module does the opposite: it takes the OUTER boundary of everything that
belongs to a figure and FILLS it into one solid shape, so scattered near-dots
are KEPT and internal fragments/holes are FILLED. Only genuinely isolated far
specks are dropped as noise.

Pure OpenCV / numpy / scipy. No model, no network.

Public API:
    consolidate(binary_mask, variant, merge_dist, drop_area) -> (out_mask, boxes)
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def _to_binary(mask: np.ndarray) -> np.ndarray:
    """Coerce any 2D array / alpha channel to a uint8 {0,1} foreground mask."""
    m = np.asarray(mask)
    if m.ndim == 3:
        # take the alpha channel if RGBA, else any nonzero channel
        m = m[:, :, 3] if m.shape[2] == 4 else m.max(axis=2)
    # Accept both {0,1} masks and 0-255 alpha: threshold at the midpoint only
    # when the data actually spans that range.
    thr = 127 if m.max() > 1 else 0
    return (m > thr).astype(np.uint8)


def _odd(n: int) -> int:
    n = int(round(n))
    if n < 1:
        n = 1
    return n if n % 2 == 1 else n + 1


def consolidate(binary_mask, variant="close_fill", merge_dist=40, drop_area=150):
    """
    Consolidate a binary figure mask into solid per-figure shapes.

    Steps:
      1. GROUP components by proximity: dilate the mask so components within
         ~merge_dist of one another bridge together, then connected-component
         label the bridged mask. Each label is a figure GROUP. Map each group
         back to the ORIGINAL foreground pixels it covers.
      2. Per group, produce a solid shape according to `variant`:
           - "hull"       : convex hull over the group's original fg points,
                            filled -> solid convex blob.
           - "close_fill" : morphological CLOSE (elliptical, ~merge_dist) on the
                            group region, then binary_fill_holes -> a solid shape
                            that follows the outer contour but fills interior
                            holes and bridges the dots.
      3. ATTACH-THEN-DROP: a component that joined a multi-component group is
         KEPT (merged in). A component that is alone in its group AND whose
         original fg area < drop_area is dropped as real noise.
      4. Return the consolidated uint8 {0,1} mask + one bbox per surviving group.

    Args:
        binary_mask : 2D array, RGBA image, or alpha channel. Foreground = >127.
        variant     : "hull" or "close_fill".
        merge_dist  : proximity (px) within which components bridge into a group.
        drop_area   : original-fg area (px) below which an ISOLATED component is
                      dropped as noise.

    Returns:
        (out_mask, boxes)
          out_mask : uint8 {0,1} array, same HxW as input.
          boxes    : list of (x0, y0, x1, y1) — one per surviving group,
                     x1/y1 exclusive.
    """
    if variant not in ("hull", "close_fill"):
        raise ValueError(f"unknown variant {variant!r}")

    fg = _to_binary(binary_mask)
    h, w = fg.shape
    out = np.zeros((h, w), np.uint8)
    boxes = []

    if fg.sum() == 0:
        return out, boxes

    # --- Step 1: bridge nearby components, then label GROUPS -----------------
    # Dilate by ~merge_dist/2 on all sides so two blobs whose gap is <= merge_dist
    # touch after dilation (each grows merge_dist/2 toward the other).
    k = _odd(max(1, merge_dist))  # full-width kernel ~merge_dist
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bridged = cv2.dilate(fg, bridge_kernel, iterations=1)

    n_groups, group_lab = cv2.connectedComponents(bridged, connectivity=8)

    # Original per-component labels (to reason about isolation / area).
    n_comp, comp_lab, comp_stats, _ = cv2.connectedComponentsWithStats(
        fg, connectivity=8
    )

    # For each ORIGINAL component, which group does it fall in, and how many
    # original components share that group?
    comps_per_group = np.zeros(n_groups, dtype=np.int64)
    comp_group = np.zeros(n_comp, dtype=np.int64)
    for c in range(1, n_comp):
        ys, xs = np.where(comp_lab == c)
        g = int(group_lab[ys[0], xs[0]])  # group is a superset -> any pixel works
        comp_group[c] = g
        comps_per_group[g] += 1

    # --- Steps 2 & 3: per group, decide keep/drop and build a solid shape ----
    for g in range(1, n_groups):
        # original foreground belonging to this group
        group_fg = (fg > 0) & (group_lab == g)
        if not group_fg.any():
            continue

        # component ids in this group
        member_comps = [c for c in range(1, n_comp) if comp_group[c] == g]

        # ATTACH-THEN-DROP: if this group is a single isolated component below
        # drop_area, it is real noise -> drop.
        if len(member_comps) == 1:
            c = member_comps[0]
            area = int(comp_stats[c, cv2.CC_STAT_AREA])
            if area < drop_area:
                continue  # dropped

        ys, xs = np.where(group_fg)
        pts = np.stack([xs, ys], axis=1)  # (N,2) x,y for cv2

        if variant == "hull":
            hull = cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.int32))
            shape = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(shape, hull, 1)
        else:  # close_fill
            shape = group_fg.astype(np.uint8)
            ck = _odd(max(3, merge_dist))
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
            shape = cv2.morphologyEx(shape, cv2.MORPH_CLOSE, close_kernel)
            shape = ndimage.binary_fill_holes(shape).astype(np.uint8)

        out[shape > 0] = 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        # bbox of the produced shape (may extend past raw fg for close_fill)
        sys_, sxs_ = np.where(shape > 0)
        if sxs_.size:
            x0 = min(x0, int(sxs_.min())); x1 = max(x1, int(sxs_.max()) + 1)
            y0 = min(y0, int(sys_.min())); y1 = max(y1, int(sys_.max()) + 1)
        boxes.append((x0, y0, x1, y1))

    return out, boxes


# ---------------------------------------------------------------------------
# Demo / visualisation helpers
# ---------------------------------------------------------------------------

import os
from PIL import Image, ImageDraw, ImageFont

PAPER = (245, 243, 236)     # light paper background
INK = (30, 30, 34)          # figure ink
PANEL_BG = (255, 255, 255)


def _font(size=22):
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(cand):
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def _colorize_mask(fg, boxes=None, base=None):
    """Render a {0,1} mask (or supply base RGB) as an RGB panel on paper."""
    h, w = fg.shape
    if base is None:
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = PAPER
        img[fg > 0] = INK
    else:
        img = base.copy()
    pil = Image.fromarray(img)
    if boxes:
        d = ImageDraw.Draw(pil)
        for (x0, y0, x1, y1) in boxes:
            d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(220, 40, 40), width=4)
    return pil


def _checkerboard(h, w, sq=24):
    board = np.zeros((h, w, 3), np.uint8)
    c1, c2 = (210, 210, 210), (170, 170, 170)
    for yy in range(0, h, sq):
        for xx in range(0, w, sq):
            board[yy:yy + sq, xx:xx + sq] = c1 if ((yy // sq + xx // sq) % 2 == 0) else c2
    return board


def _label_panel(pil, title, subtitle=""):
    """Add a title bar above a panel; returns new PIL image."""
    w, h = pil.size
    bar = 78
    canvas = Image.new("RGB", (w, h + bar), PANEL_BG)
    canvas.paste(pil, (0, bar))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, w - 1, bar - 1], fill=(24, 26, 32))
    d.text((14, 10), title, fill=(255, 255, 255), font=_font(26))
    if subtitle:
        d.text((14, 44), subtitle, fill=(180, 200, 230), font=_font(18))
    return canvas


def _row(panels, pad=12):
    """Horizontally concatenate equal-height PIL panels with padding."""
    h = max(p.size[1] for p in panels)
    w = sum(p.size[0] for p in panels) + pad * (len(panels) + 1)
    canvas = Image.new("RGB", (w, h + 2 * pad), (60, 62, 70))
    x = pad
    for p in panels:
        canvas.paste(p, (x, pad))
        x += p.size[0] + pad
    return canvas


# ---------------------------------------------------------------------------
# DEMO A — synthetic
# ---------------------------------------------------------------------------

def build_synthetic(size=900, seed=7):
    """Return a {0,1} mask illustrating the exact user case."""
    rng = np.random.default_rng(seed)
    m = np.zeros((size, size), np.uint8)

    # (i) MAIN irregular figure blob (union of a few ellipses) with a HOLE
    cx, cy = 300, 380
    cv2.ellipse(m, (cx, cy), (150, 110), 20, 0, 360, 1, -1)
    cv2.ellipse(m, (cx - 70, cy + 60), (90, 80), 0, 0, 360, 1, -1)
    cv2.ellipse(m, (cx + 60, cy - 70), (80, 70), 0, 0, 360, 1, -1)
    # internal HOLE
    cv2.circle(m, (cx + 10, cy + 5), 45, 0, -1)

    # (ii) cluster of ~12 scattered DOTS trailing off the RIGHT edge, each
    #      within merge_dist of the previous -> should be KEPT / merged.
    x, y = cx + 150, cy - 40
    for i in range(12):
        x += int(rng.integers(28, 40))          # step < merge_dist
        y += int(rng.integers(-22, 22))
        r = int(rng.integers(5, 11))
        cv2.circle(m, (x, y), r, 1, -1)

    # (iii) one FAR isolated speck, well beyond merge_dist -> real noise, DROP
    cv2.circle(m, (760, 130), 7, 1, -1)

    # (iv) a SEPARATE second figure blob, far from the first -> own shape
    cv2.ellipse(m, (660, 700), (120, 90), -15, 0, 360, 1, -1)
    cv2.circle(m, (700, 640), 40, 0, -1)         # give it a small hole too
    return m


def demo_synthetic(out_path, merge_dist, drop_area):
    fg = build_synthetic()
    raw_panel = _label_panel(
        _colorize_mask(fg), "raw mask",
        "main blob + hole, 12 near dots, 1 far speck, 2nd figure")

    panels = [raw_panel]
    notes = {}
    for variant in ("hull", "close_fill"):
        out, boxes = consolidate(fg, variant=variant,
                                 merge_dist=merge_dist, drop_area=drop_area)
        sub = f"{len(boxes)} figure group(s)  md={merge_dist} da={drop_area}"
        panels.append(_label_panel(_colorize_mask(out, boxes), variant, sub))
        notes[variant] = (out, boxes)
    row = _row(panels)
    row.save(out_path)
    return notes, fg


# ---------------------------------------------------------------------------
# DEMO B — real masks
# ---------------------------------------------------------------------------

def _composite_rgba(rgba, mask=None):
    """Composite an RGBA image over a checkerboard. If mask given, use it as
    the alpha (values {0,1}) keeping original RGB where available."""
    h, w = rgba.shape[:2]
    board = _checkerboard(h, w)
    rgb = rgba[:, :, :3].astype(np.float32)
    if mask is None:
        a = (rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    else:
        a = mask.astype(np.float32)[:, :, None]
    comp = rgb * a + board.astype(np.float32) * (1 - a)
    return comp.astype(np.uint8)


def demo_real(png_path, out_path, merge_dist, drop_area, max_w=820):
    im = Image.open(png_path).convert("RGBA")
    arr = np.array(im)
    fg = _to_binary(arr)

    # scale down for the montage (process at full res, then resize panels)
    def _panel(base_rgb, boxes, title, sub):
        pil = Image.fromarray(base_rgb)
        if boxes:
            d = ImageDraw.Draw(pil)
            for (x0, y0, x1, y1) in boxes:
                d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(220, 40, 40), width=6)
        w0, h0 = pil.size
        if w0 > max_w:
            pil = pil.resize((max_w, int(h0 * max_w / w0)), Image.LANCZOS)
        return _label_panel(pil, title, sub)

    raw_comp = _composite_rgba(arr)  # original alpha over checkerboard
    panels = [_panel(raw_comp, None, "raw alpha",
                     f"{(fg>0).mean()*100:.1f}% fg")]

    results = {}
    for variant in ("hull", "close_fill"):
        out, boxes = consolidate(fg, variant=variant,
                                 merge_dist=merge_dist, drop_area=drop_area)
        comp = _composite_rgba(arr, mask=out)
        panels.append(_panel(comp, boxes, variant,
                             f"{len(boxes)} group(s)  md={merge_dist} da={drop_area}"))
        results[variant] = (out, boxes)
    row = _row(panels)
    row.save(out_path)
    return results


def run_demos():
    seg = "/home/suzunik/boundarywatch/segment"
    outdir = os.path.join(seg, "postproc")
    os.makedirs(outdir, exist_ok=True)

    # ---- synthetic ----
    S_MD, S_DA = 45, 400
    notes, fg = demo_synthetic(os.path.join(outdir, "synthetic_demo.png"),
                               merge_dist=S_MD, drop_area=S_DA)

    # analyse synthetic outcome
    def report_syn(variant):
        out, boxes = notes[variant]
        # far speck region (around 760,130): was it dropped?
        far = out[100:160, 730:790].sum()
        # dots region (right of main blob) present?
        dots = out[300:400, 480:720].sum()
        # hole in main blob filled? sample hole center (~310,385)
        hole = out[380:392, 300:322].mean()
        # two separate groups?
        return dict(groups=len(boxes),
                    far_speck_px=int(far),
                    dots_px=int(dots),
                    hole_filled=bool(hole > 0.5))

    syn = {v: report_syn(v) for v in ("hull", "close_fill")}

    # ---- real ----
    R_MD, R_DA = 55, 600
    demo_real(os.path.join(seg, "out/page11_A_figurekeep.png"),
              os.path.join(outdir, "page11_postproc.png"),
              merge_dist=R_MD, drop_area=R_DA)
    demo_real(os.path.join(seg, "out/page3_A_figurekeep.png"),
              os.path.join(outdir, "page3_postproc.png"),
              merge_dist=R_MD, drop_area=R_DA)

    print("=== SYNTHETIC (merge_dist=%d drop_area=%d) ===" % (S_MD, S_DA))
    for v in ("hull", "close_fill"):
        print(" ", v, syn[v])
    print("=== REAL (merge_dist=%d drop_area=%d) ===" % (R_MD, R_DA))
    print("outputs:")
    for f in ("synthetic_demo.png", "page11_postproc.png", "page3_postproc.png"):
        print("  ", os.path.join(outdir, f))
    return syn


if __name__ == "__main__":
    run_demos()
