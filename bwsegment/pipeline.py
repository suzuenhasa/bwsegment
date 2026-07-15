"""
pipeline.py — prompt-free document artwork/figure segmenter (SAM3 replacement).

Chain per page:
    strip_text (whiten locate's text boxes)
      -> resize to the CNN's training size (W x H)  [INTER_AREA, RGB, /255]
      -> TinyUNet forward -> sigmoid -> content probability
      -> resize probability back to native resolution [INTER_LINEAR]
      -> threshold at `conf`
      -> outerfill_gpu.consolidate(close_fill, merge_dist)  [GPU fill-holes]
      -> solid mask + per-figure boxes
      -> build an RGBA (artwork RGB kept, everything else transparent).

`PaperSegmenter` loads the CNN once so a long-lived server can reuse it. A
module-level `segment(...)` convenience is provided for one-off calls.
"""

from __future__ import annotations

import os
from typing import Sequence

import cv2
import numpy as np
import torch

from .model import load_paper_cnn
from .strip import strip_text, TextBox
from . import outerfill_gpu

# native-resolution area (px) below which an ISOLATED component is dropped as
# noise. Tuned on native-res scans; overridable per call.
DEFAULT_DROP_AREA = 600

# default weights shipped with the package
_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "paper_cnn.pt")


class PaperSegmenter:
    """Reusable segmenter: loads the paper/content CNN once, segments many pages.

    Args:
        weights_path : path to paper_cnn.pt (defaults to the packaged weights).
        device       : torch device string, e.g. "cuda:0" (falls back to cpu).
    """

    def __init__(self, weights_path: str = _DEFAULT_WEIGHTS, device: str = "cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.weights_path = weights_path
        self.model, self.W, self.H = load_paper_cnn(weights_path, self.device)

    @torch.no_grad()
    def segment(self, page_bgr: np.ndarray, text_boxes: Sequence[TextBox],
                conf: float = 0.5, merge_dist: int = 55,
                drop_area: int = DEFAULT_DROP_AREA) -> dict:
        """Segment artwork/figures on one page.

        Args:
            page_bgr  : HxWx3 uint8 BGR page.
            text_boxes: locate text boxes, normalised [x1,y1,x2,y2] or {x,y,w,h}.
            conf      : probability threshold for content.
            merge_dist: proximity (px) within which figure fragments bridge.
            drop_area : isolated-component area (px) dropped as noise.

        Returns dict:
            mask  : HxW uint8, 255 = artwork.
            boxes : list of [x1, y1, x2, y2] pixel boxes (x2/y2 exclusive).
            rgba  : HxWx4 uint8, original RGB kept where artwork, else transparent.
        """
        nh, nw = page_bgr.shape[:2]

        # 1) strip text so the CNN only separates artwork from blank paper
        stripped = strip_text(page_bgr, text_boxes, dilate_px=4)

        # 2) preprocess EXACTLY as training/inference did: BGR->RGB, resize to
        #    (W,H) with INTER_AREA, float /255, CHW.
        rgb = cv2.cvtColor(stripped, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)
        inp = inp.astype(np.float32) / 255.0
        t = torch.from_numpy(inp.transpose(2, 0, 1))[None].to(self.device)

        # 3) forward -> sigmoid -> content probability at the CNN size
        prob = torch.sigmoid(self.model(t))[0, 0].cpu().numpy()  # HxW in [0,1]

        # 4) resize probability back to NATIVE resolution, then threshold
        prob_native = cv2.resize(prob, (nw, nh), interpolation=cv2.INTER_LINEAR)
        binary = (prob_native >= conf).astype(np.uint8)

        # 5) GPU consolidation: close + fill-holes + drop tiny -> solid figures
        cons, raw_boxes = outerfill_gpu.consolidate(
            binary, variant="close_fill", merge_dist=merge_dist, drop_area=drop_area,
            device=(self.device if str(self.device).startswith("cuda") else "cuda"),
        )

        mask = (cons > 0).astype(np.uint8) * 255
        boxes = [[int(x0), int(y0), int(x1), int(y1)] for (x0, y0, x1, y1) in raw_boxes]

        # 6) RGBA: keep the ORIGINAL page RGB where artwork, transparent elsewhere
        rgba = np.zeros((nh, nw, 4), np.uint8)
        rgba[:, :, :3] = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
        rgba[:, :, 3] = mask

        return dict(mask=mask, boxes=boxes, rgba=rgba)


# module-level convenience: caches one segmenter per (weights, device)
_CACHE: dict = {}


def segment(page_bgr: np.ndarray, text_boxes: Sequence[TextBox],
            weights_path: str = _DEFAULT_WEIGHTS, device: str = "cuda:0",
            conf: float = 0.5, merge_dist: int = 55,
            drop_area: int = DEFAULT_DROP_AREA) -> dict:
    """One-off segment(); builds (and caches) a PaperSegmenter under the hood."""
    key = (weights_path, device)
    seg = _CACHE.get(key)
    if seg is None:
        seg = _CACHE[key] = PaperSegmenter(weights_path=weights_path, device=device)
    return seg.segment(page_bgr, text_boxes, conf=conf,
                       merge_dist=merge_dist, drop_area=drop_area)
