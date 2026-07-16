# bwsegment

A **prompt-free document artwork / figure segmenter**. Given a scanned or
rendered document page (plus the text boxes `locate` already found), it returns a
solid mask of the non-text artwork — photos, figures, diagrams, halftones — with
per-figure bounding boxes and a ready-to-composite RGBA cutout.

It is the **drop-in replacement for SAM3** in the typesetter's `segment()` step.

## Pipeline

```
page (BGR) + locate text boxes
   │
   ├─ strip_text        whiten locate's text regions (dilated), so the CNN only
   │                     has to separate artwork from blank paper
   ├─ paper CNN          TinyUNet (~0.5M params) at its training size (680x1024),
   │                     sigmoid -> per-pixel content probability
   ├─ resize + threshold prob map back to native res, threshold at `conf`
   └─ GPU outerfill      close + cupy binary_fill_holes + drop-tiny consolidation
                         -> solid per-figure shapes + boxes
```

Output: `mask` (HxW uint8, 255 = artwork), `boxes` (pixel `[x1,y1,x2,y2]`), and
`rgba` (original RGB kept where artwork, transparent elsewhere).

## API

```python
import cv2
from bwsegment import PaperSegmenter

seg = PaperSegmenter(device="cuda:0")          # loads the CNN once; reuse in a server
page = cv2.imread("page.png", cv2.IMREAD_COLOR)

# text_boxes: locate's boxes, either normalised [x1,y1,x2,y2] or {x,y,w,h}
res = seg.segment(page, text_boxes, conf=0.5, merge_dist=55)

res["mask"]    # HxW uint8, 255 = artwork
res["boxes"]   # [[x1,y1,x2,y2], ...] pixel coords
res["rgba"]    # HxWx4 uint8 cutout
```

A module-level `segment(page_bgr, text_boxes, ...)` convenience is also exported
for one-off calls (it caches a `PaperSegmenter` per weights/device).

## Detector modes (CNN, classical, or union)

`segment()` takes a `detector` argument selecting which content detector drives
the result. All three feed the **same** GPU `outerfill` consolidation, so only
the raw content map differs:

| `detector`    | how it finds content                                   | wins on |
|---------------|--------------------------------------------------------|---------|
| `"cnn"` (default) | TinyUNet content probability                        | **clean-paper** pages — tighter, well-separated artwork islands on white |
| `"classical"` | model-free local contrast + halftone texture (no weights) | **full illustrated** pages — captures the whole page + border better (e.g. `tm_p007`) |
| `"union"`     | OR of the CNN and classical content maps               | **max recall** — when you'd rather over- than under-detect |
| `"box"`       | no pixel classification at all — trusts `locate`'s `image_boxes` as the region (text boxes punched back out) | **pages where an appearance detector eats the art** — greyscale photos and line-art especially. `"box"` structurally **cannot** eat the art; it's coarser (paper inside the rect is kept). Requires `image_boxes=`. |
| `"box_trim"`  | `"box"`, then carve out only pixels that are **flat AND paper-toned AND connected-to-outside**, then despeck | **RECOMMENDED when `image_boxes` exist.** Keeps the box's safety guarantee while removing the paper the plain rect leaves behind. Requires `image_boxes=`. |

### Why `box_trim` is the safe one

Appearance detectors eat greyscale artwork on these pages — and **not because the
art and paper share a tone**. Measured on a real page, the art sits at tone
**32–150** while clean paper's 1st percentile is **221**: a ~70-level gap with
nothing in it. The real cause is **contaminated background statistics**. A
detector that learns "background" from everything outside the region is learning
the *text column* — white paper **and black ink** — so its model spans the whole
grey axis and explains any greyscale artwork about as well as it explains paper.
Chromatic art (blue sky, beige stone) survives on saturation alone; a neutral B&W
photo is a coin flip. Box-init GrabCut fails this way by construction, and it's
the most likely reason a content CNN or a border flood drops a statue.

`box_trim` sidesteps it twice over. It **never models art** — it starts from
locate's region, where the default is **keep** — and its paper reference is
**one-class, measured from a clean sample**: the bright tail of the outside, so
ink can't drag it dark. A pixel is carved only if **all three** hold:

1. **flat** — local std below `flat`. Paper is featureless; art carries
   photographic micro-texture down to its lightest highlights.
2. **paper-toned** — CIELAB distance to the *measured* paper colour below `tone`.
   Colour, not brightness: beige stone and blue sky are far from cream paper in Lab
   even at equal brightness — a brightness-only rule misses exactly this.
3. **connected-to-outside** — reachable by a flood from the page border. Paper
   *enclosed inside* the artwork is never trimmed.

**Invariant: a textured pixel inside an image_box and outside every text_box is always
kept.** The recipe cannot eat art — its only failure mode is keeping some paper, the
conservative and recoverable direction. Measured art-kept across our pages: 95–100%.

They genuinely win on **different pages**, which is why both ship: keep both
available and pick per page. The `"classical"` path needs **no weights** and is
now **GPU-accelerated** (torch `max_pool2d` morphology for the paper baseline +
`avg_pool2d` for the local-std texture — a few ms/page; it transparently falls
back to the exact cv2/numpy path with no CUDA).

```python
res = seg.segment(page, text_boxes, detector="classical")   # or "cnn" / "union"

# region mode — pass locate's image_boxes (layout.json -> image_boxes)
res = seg.segment(page, text_boxes, detector="box", image_boxes=image_boxes)

# Compare all three on a page and choose per-page:
res = seg.segment(page, text_boxes, return_all=True)
res["cnn_mask"], res["classical_mask"], res["union_mask"]   # each post-outerfill
res["mask"]     # == the selected `detector`'s mask (still "cnn" by default)
```

`detector="cnn"` is the default and preserves the exact prior behavior, so
existing callers are unaffected.

The classical detector is also exported directly as
`classical_content(image_bgr, device="cuda:0", ...) -> HxW uint8 (255=content)`
if you want just the raw content mask.

## Performance

- **~25 ms / page** end-to-end on GPU.
- **~0.6 GB VRAM** resident (vs ~17 GB for SAM3).
- **~370-490x faster** than the SAM3 path it replaces.
- **val IoU ~0.99** for the GPU consolidation vs the reference CPU `close_fill`.

The old bottleneck was the CPU `outerfill.consolidate` (1.3-2.8 s/page, a per-
component full-image scan + native-res morphology + scipy fill). `outerfill_gpu`
keeps the same public API and semantics but moves morphology to
`torch.max_pool2d` and hole-filling to `cupyx.scipy.ndimage.binary_fill_holes`,
handed between torch and cupy **zero-copy via DLPack**.

## Install

```bash
pip install -r requirements.txt
```

- Needs **`cupy-cuda12x`** for the GPU fill; it shares the CUDA context with
  torch and exchanges tensors zero-copy through DLPack.
- Use the **cu128** torch build; **Blackwell** GPUs are fine on cu128.
- With no CUDA available, `outerfill_gpu.consolidate` transparently falls back to
  the exact CPU `outerfill` path, so the package still runs (slower) on CPU.
- The trained weights ship in `weights/paper_cnn.pt`; `PaperSegmenter()` loads
  them by default.

## Integration (typesetter `segment()`)

- **Drop-in**: build one `PaperSegmenter` at startup, call `.segment(page_bgr,
  text_boxes, ...)` per page. `text_boxes` is exactly `locate`'s output format
  (normalised `[x1,y1,x2,y2]` or `{x,y,w,h}` — both accepted).
- **Delete the SAM3 / Locate VRAM load-unload swap machinery**: because this
  resident footprint is ~0.6 GB (not ~17 GB), there is no need to unload Locate
  to fit SAM3 and reload it afterwards. The model stays hot.

## Known caveats

- **2-vs-3 box merge on dense pages**: on very dense pages the `merge_dist`
  bridge can occasionally consolidate what should be 3 figures into 2 (or split
  the other way). Tune `merge_dist` / `drop_area` per corpus if needed.
- **Real-scan robustness**: a v2 `bgasset` weight set (trained with more real-
  scan background/asset variety) is coming to improve robustness on noisy scans.
