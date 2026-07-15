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
from .classical import classical_content
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

    # -- CNN content probability (internal) ---------------------------------
    def _cnn_binary(self, stripped_bgr: np.ndarray, nw: int, nh: int,
                    conf: float) -> np.ndarray:
        """Native-resolution CNN content mask (uint8 {0,1}) for a stripped page."""
        # preprocess EXACTLY as training/inference did: BGR->RGB, resize to
        # (W,H) with INTER_AREA, float /255, CHW.
        rgb = cv2.cvtColor(stripped_bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)
        inp = inp.astype(np.float32) / 255.0
        t = torch.from_numpy(inp.transpose(2, 0, 1))[None].to(self.device)

        prob = torch.sigmoid(self.model(t))[0, 0].cpu().numpy()  # HxW in [0,1]
        prob_native = cv2.resize(prob, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return (prob_native >= conf).astype(np.uint8)

    def _consolidate(self, binary: np.ndarray, merge_dist: int,
                     drop_area: int):
        """GPU close + fill-holes + drop-tiny -> (mask255, boxes)."""
        cons, raw_boxes = outerfill_gpu.consolidate(
            binary, variant="close_fill", merge_dist=merge_dist,
            drop_area=drop_area,
            device=(self.device if str(self.device).startswith("cuda") else "cuda"),
        )
        mask = (cons > 0).astype(np.uint8) * 255
        boxes = [[int(x0), int(y0), int(x1), int(y1)]
                 for (x0, y0, x1, y1) in raw_boxes]
        return mask, boxes

    @torch.no_grad()
    def segment(self, page_bgr: np.ndarray, text_boxes: Sequence[TextBox],
                conf: float = 0.5, merge_dist: int = 55,
                drop_area: int = DEFAULT_DROP_AREA,
                detector: str = "cnn", return_all: bool = False) -> dict:
        """Segment artwork/figures on one page.

        Args:
            page_bgr  : HxWx3 uint8 BGR page.
            text_boxes: locate text boxes, normalised [x1,y1,x2,y2] or {x,y,w,h}.
            conf      : probability threshold for content (CNN path only).
            merge_dist: proximity (px) within which figure fragments bridge.
            drop_area : isolated-component area (px) dropped as noise.
            detector  : which content detector to return, one of:
                          "cnn"       (DEFAULT) — TinyUNet content probability;
                                      tighter on clean-paper pages.
                          "classical" — model-free contrast/texture detector;
                                      captures full illustrated pages + borders.
                          "union"     — OR of both content maps (max recall).
                        All three feed the SAME GPU outerfill consolidation.
            return_all: if True, also compute and return every detector's result
                        (post-outerfill) under cnn_mask / classical_mask /
                        union_mask, so the caller can compare/choose per page.

        Returns dict:
            mask  : HxW uint8, 255 = artwork (from the selected `detector`).
            boxes : list of [x1, y1, x2, y2] pixel boxes (x2/y2 exclusive).
            rgba  : HxWx4 uint8, original RGB kept where artwork, else transparent.
          plus, when return_all=True:
            cnn_mask, classical_mask, union_mask : HxW uint8 255-masks, each the
                post-outerfill result of that detector.
        """
        if detector not in ("cnn", "classical", "union"):
            raise ValueError(
                f"unknown detector {detector!r}; expected 'cnn', 'classical' or 'union'")

        nh, nw = page_bgr.shape[:2]

        # strip text so the detectors separate artwork from blank paper only
        stripped = strip_text(page_bgr, text_boxes, dilate_px=4)

        # decide which native-resolution content maps we need to compute
        need_cnn = return_all or detector in ("cnn", "union")
        need_classical = return_all or detector in ("classical", "union")

        cnn_binary = self._cnn_binary(stripped, nw, nh, conf) if need_cnn else None
        classical_binary = None
        if need_classical:
            cls255 = classical_content(
                stripped,
                device=(self.device if str(self.device).startswith("cuda") else "cuda"),
            )
            classical_binary = (cls255 > 0).astype(np.uint8)

        # consolidate exactly the detectors we need (cache to avoid double work)
        results: dict = {}

        def get(name: str):
            if name in results:
                return results[name]
            if name == "cnn":
                out = self._consolidate(cnn_binary, merge_dist, drop_area)
            elif name == "classical":
                out = self._consolidate(classical_binary, merge_dist, drop_area)
            else:  # union
                union_binary = (
                    (cnn_binary.astype(bool) | classical_binary.astype(bool))
                    .astype(np.uint8))
                out = self._consolidate(union_binary, merge_dist, drop_area)
            results[name] = out
            return out

        mask, boxes = get(detector)

        # RGBA: keep the ORIGINAL page RGB where artwork, transparent elsewhere
        rgba = np.zeros((nh, nw, 4), np.uint8)
        rgba[:, :, :3] = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
        rgba[:, :, 3] = mask

        result = dict(mask=mask, boxes=boxes, rgba=rgba)
        if return_all:
            result["cnn_mask"] = get("cnn")[0]
            result["classical_mask"] = get("classical")[0]
            result["union_mask"] = get("union")[0]
        return result


# module-level convenience: caches one segmenter per (weights, device)
_CACHE: dict = {}


def segment(page_bgr: np.ndarray, text_boxes: Sequence[TextBox],
            weights_path: str = _DEFAULT_WEIGHTS, device: str = "cuda:0",
            conf: float = 0.5, merge_dist: int = 55,
            drop_area: int = DEFAULT_DROP_AREA,
            detector: str = "cnn", return_all: bool = False) -> dict:
    """One-off segment(); builds (and caches) a PaperSegmenter under the hood."""
    key = (weights_path, device)
    seg = _CACHE.get(key)
    if seg is None:
        seg = _CACHE[key] = PaperSegmenter(weights_path=weights_path, device=device)
    return seg.segment(page_bgr, text_boxes, conf=conf,
                       merge_dist=merge_dist, drop_area=drop_area,
                       detector=detector, return_all=return_all)
