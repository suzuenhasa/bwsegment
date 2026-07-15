"""
strip.py — remove text before the paper/content CNN sees the page.

`locate` already knows where every text block is. Whiting out those regions
(slightly dilated so glyph edges and antialiasing go too) means the CNN only has
to separate artwork from blank paper, never artwork from text. This is the
`striptext` step of the pipeline, written clean and minimal.
"""

from __future__ import annotations

from typing import Sequence, Union, Mapping

import numpy as np

TextBox = Union[Sequence[float], Mapping[str, float]]


def _box_to_xyxy_norm(box: TextBox):
    """Normalise one text box to (x1, y1, x2, y2) in [0,1] fractional coords.

    Accepts either:
      - a 4-sequence [x1, y1, x2, y2]                (normalised corners), or
      - a mapping {x, y, w, h}                        (locate's bbox format).
    Returns None if the box cannot be parsed.
    """
    if isinstance(box, Mapping):
        # locate format: some blocks wrap the coords in a "bbox" sub-dict.
        b = box.get("bbox", box)
        x, y, w, h = b.get("x"), b.get("y"), b.get("w"), b.get("h")
        if None in (x, y, w, h):
            return None
        return float(x), float(y), float(x) + float(w), float(y) + float(h)
    if len(box) >= 4:
        x1, y1, x2, y2 = box[:4]
        return float(x1), float(y1), float(x2), float(y2)
    return None


def strip_text(image_bgr: np.ndarray, text_boxes: Sequence[TextBox],
               dilate_px: int = 4) -> np.ndarray:
    """Return a copy of `image_bgr` with every text box painted white.

    Args:
        image_bgr : HxWx3 uint8 BGR page.
        text_boxes: list of normalised [x1,y1,x2,y2] OR {x,y,w,h} boxes.
        dilate_px : grow each box by this many pixels on every side before
                    whiting it out, to catch glyph edges / antialiasing.

    Returns:
        A new HxWx3 uint8 image with text regions set to white (255).
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]
    for box in text_boxes or ():
        parsed = _box_to_xyxy_norm(box)
        if parsed is None:
            continue
        x1, y1, x2, y2 = parsed
        px1 = int(round(x1 * w)) - dilate_px
        py1 = int(round(y1 * h)) - dilate_px
        px2 = int(round(x2 * w)) + dilate_px
        py2 = int(round(y2 * h)) + dilate_px
        px1 = max(0, px1); py1 = max(0, py1)
        px2 = min(w, px2); py2 = min(h, py2)
        if px2 > px1 and py2 > py1:
            out[py1:py2, px1:px2] = 255
    return out
