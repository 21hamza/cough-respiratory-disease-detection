"""M2 - Standalone 2D-CNN on log-Mel spectrograms (Report Section V).

Two conv blocks, then global pooling over time AND frequency, so all temporal order is discarded.
It shares its front-end with M3; the only difference is the head, so M2 vs. M3 measures the value
of temporal modelling.

Input (B, 1, 128, 401) -> conv block 1 -> (B, 32, 64, 200) -> conv block 2 -> (B, 64, 32, 100)
-> global pool -> (B, 64) -> dropout 0.5 -> linear -> (B, 2)

pool="avg" (Phase 2): average over the whole map, which mixes the short cough burst with silence/padding.
pool="max" (final):   keep each channel's strongest response, i.e. the most cough-like region.
"""
import torch.nn as nn


def conv_block(c_in, c_out):
    """3x3 conv -> ReLU -> 2x2 max pool -> batch norm (same block in M2 and M3)."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(c_out))


class CNN2D(nn.Module):
    def __init__(self, n_classes=2, pool="avg"):
        super().__init__()
        self.f = nn.Sequential(conv_block(1, 32), conv_block(32, 64))
        gp = nn.AdaptiveAvgPool2d(1) if pool == "avg" else nn.AdaptiveMaxPool2d(1)
        self.h = nn.Sequential(gp, nn.Flatten(), nn.Dropout(0.5), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.h(self.f(x))
