"""
bwsegment — prompt-free document artwork/figure segmenter (SAM3 replacement).

Public API:
    PaperSegmenter(weights_path=..., device="cuda:0")   load once, reuse
    segment(page_bgr, text_boxes, ...)                   one-off convenience
    classical_content(image_bgr, device="cuda:0", ...)   model-free content mask
"""

from .pipeline import PaperSegmenter, segment
from .classical import classical_content

__all__ = ["PaperSegmenter", "segment", "classical_content"]
