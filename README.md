# bwsegment

A **prompt-free document artwork / figure segmenter**. Given a scanned or
rendered document page (plus the text boxes `locate` already found), it returns a
solid mask of the non-text artwork — photos, figures, diagrams, halftones — with
per-figure bounding boxes and a ready-to-composite RGBA cutout.

It is the **drop-in replacement for SAM3** in the typesetter's `segment()` step:
no prompts, no 17 GB model swap, ~370-490x faster.

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
