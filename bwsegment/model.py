"""
model.py — the tiny paper/content U-Net used by the segmenter.

A small fully-convolutional U-Net (~100K-500K params). It takes a 3-channel RGB
page (resized to the training size) and emits a single content logit per pixel
(sigmoid -> content probability, where "content" = artwork/figure vs blank
paper). Extracted verbatim from the training / inference scripts and paired with
a single `load_paper_cnn` helper so a server can load the weights once and reuse
the module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    """Fully-convolutional. Encoder 16->32->64->128 (bottleneck), light upsample head."""

    def __init__(self, base: int = 16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8  # 16,32,64,128
        self.enc1 = conv_block(3, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.bott = conv_block(c3, c4)
        # decoder: bilinear upsample + concat skip + conv
        self.dec3 = conv_block(c4 + c3, c3)
        self.dec2 = conv_block(c3 + c2, c2)
        self.dec1 = conv_block(c2 + c1, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    @staticmethod
    def up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                 # H
        e2 = self.enc2(self.pool(e1))     # H/2
        e3 = self.enc3(self.pool(e2))     # H/4
        b = self.bott(self.pool(e3))      # H/8
        d3 = self.dec3(torch.cat([self.up(b, e3), e3], 1))
        d2 = self.dec2(torch.cat([self.up(d3, e2), e2], 1))
        d1 = self.dec1(torch.cat([self.up(d2, e1), e1], 1))
        return self.head(d1)              # logit, full res


def load_paper_cnn(weights_path: str, device: str = "cuda:0"):
    """Load a trained TinyUNet checkpoint.

    Returns (model, W, H):
      model : TinyUNet in eval mode on `device` (falls back to cpu if no cuda).
      W, H  : the exact input width/height the model was trained at; callers must
              resize pages to (W, H) before forwarding so preprocessing matches
              training bit-for-bit.

    The checkpoint is the dict saved by train_paper.py:
      {"model": state_dict, "base": int, "W": int, "H": int, ...}
    """
    dev = device if (isinstance(device, str) and torch.cuda.is_available()) else \
        (device if not isinstance(device, str) else "cpu")
    ck = torch.load(weights_path, map_location="cpu")
    W, H, base = ck["W"], ck["H"], ck["base"]
    model = TinyUNet(base=base).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, int(W), int(H)
