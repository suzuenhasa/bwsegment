#!/usr/bin/env python3
"""Minimal bwsegment usage on a single page.

    python example.py page.png [locate_blocks.json] [out_prefix]

- page.png            : the document page (any format cv2 can read).
- locate_blocks.json  : optional locate output. Either a list of text boxes
                        ([x1,y1,x2,y2] normalised, or {x,y,w,h}), or the raw
                        locate blocks list (items with kind=="text" and a bbox).
                        Omit to run with no text stripping.
- out_prefix          : optional output prefix (default: derived from page.png).

Writes <prefix>_mask.png and <prefix>_rgba.png and prints the figure boxes.
"""

import json
import os
import sys

import cv2

from bwsegment import PaperSegmenter


def load_text_boxes(path):
    if not path or not os.path.exists(path):
        return []
    data = json.load(open(path))
    if isinstance(data, dict):
        data = data.get("blocks") or data.get("locate") or []
    boxes = []
    for item in data:
        if isinstance(item, dict) and item.get("kind") not in (None, "text"):
            continue
        boxes.append(item)
    return boxes


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    page_path = sys.argv[1]
    locate_path = sys.argv[2] if len(sys.argv) > 2 else None
    prefix = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(page_path)[0]

    page_bgr = cv2.imread(page_path, cv2.IMREAD_COLOR)
    if page_bgr is None:
        raise SystemExit(f"could not read {page_path}")
    text_boxes = load_text_boxes(locate_path)

    seg = PaperSegmenter(device="cuda:0")            # loads the CNN once
    res = seg.segment(page_bgr, text_boxes, conf=0.5, merge_dist=55)

    cv2.imwrite(f"{prefix}_mask.png", res["mask"])
    cv2.imwrite(f"{prefix}_rgba.png", cv2.cvtColor(res["rgba"], cv2.COLOR_RGBA2BGRA))
    print(f"figures: {len(res['boxes'])}")
    for b in res["boxes"]:
        print("  box", b)
    print(f"wrote {prefix}_mask.png and {prefix}_rgba.png")


if __name__ == "__main__":
    main()
