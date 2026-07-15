"""
bwsegment — prompt-free document artwork/figure segmenter (SAM3 replacement).

Public API:
    PaperSegmenter(weights_path=..., device="cuda:0")   load once, reuse
    segment(page_bgr, text_boxes, ...)                   one-off convenience
"""

from .pipeline import PaperSegmenter, segment

__all__ = ["PaperSegmenter", "segment"]
