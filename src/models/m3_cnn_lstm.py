"""M3 - Hybrid CNN-LSTM (the proposed model, Report Section V).

Same conv front-end as M2, but the time axis is kept: each of the 100 remaining time steps becomes
one 2,048-dim vector (64 channels x 32 Mel bins), read in order by a stacked LSTM.

Input (B, 1, 128, 401) -> 2 conv blocks -> (B, 64, 32, 100)
-> permute/reshape -> (B, 100, 2048) -> LSTM(128) -> dropout 0.5 -> LSTM(64)
-> last hidden state (B, 64) -> linear -> (B, 2)
"""
import torch.nn as nn
from src.models.m2_cnn2d import conv_block


class CNNLSTM(nn.Module):
    def __init__(self, n_classes=2, n_mels=128):
        super().__init__()
        self.f = nn.Sequential(conv_block(1, 32), conv_block(32, 64))
        feat = 64 * (n_mels // 4)                      # channels x Mel bins after two 2x2 pools
        self.l1 = nn.LSTM(feat, 128, batch_first=True)
        self.dp = nn.Dropout(0.5)
        self.l2 = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        z = self.f(x)                                  # (B, C, F', T')
        b, c, fr, t = z.shape
        z = z.permute(0, 3, 1, 2).reshape(b, t, c * fr)   # time-major sequence, order preserved
        z, _ = self.l1(z)
        z = self.dp(z)
        z, _ = self.l2(z)
        return self.fc(z[:, -1, :])
