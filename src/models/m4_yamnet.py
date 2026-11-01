"""M4 - Transfer learning: frozen pretrained YAMNet + trainable BiLSTM head (Report Section V).

YAMNet (Google) is a MobileNet-v1 CNN trained on AudioSet (2M YouTube clips, 521 sound classes, one of
which is "Cough"). We use it as a frozen feature extractor: it turns each 0.96 s window (hop 0.48 s)
of the 4 s waveform into a 1,024-dim embedding, giving a sequence of 8 embeddings per recording.
Only a small head is trained on our 229 training recordings, which is the point of transfer learning
when labelled data is scarce.

Input (B, 8, 1024) -> dropout -> BiLSTM(64 per direction) -> mean over time (B, 128) -> linear -> (B, 2)
or, for the "mean"/"max" heads: pool over time (B, 1024) -> linear(128) -> ReLU -> dropout -> linear -> (B, 2)
"""
import shutil, subprocess, tempfile
import numpy as np, torch, torch.nn as nn
from src import config as C


# ---------------------------------------------------------------- pretrained feature extractor
def download_yamnet():
    """Fetch the official TF Hub YAMNet (.tflite) and its class labels from tensorflow/tflite-support."""
    C.PRETRAINED.mkdir(parents=True, exist_ok=True)
    model, labels = C.PRETRAINED / "yamnet.tflite", C.PRETRAINED / "yamnet_labels.txt"
    if model.exists() and labels.exists():
        return model, labels
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout", C.YAMNET_REPO, tmp], check=True)
        subprocess.run(["git", "checkout", "HEAD", "--", C.YAMNET_PATH_IN_REPO, C.YAMNET_LABELS_IN_REPO], cwd=tmp, check=True)
        shutil.copy(f"{tmp}/{C.YAMNET_PATH_IN_REPO}", model)
        shutil.copy(f"{tmp}/{C.YAMNET_LABELS_IN_REPO}", labels)
    return model, labels


class YAMNet:
    """Thin wrapper around the TFLite interpreter. embed(waveform) -> (frames, 1024), class scores (frames, 521)."""

    def __init__(self):
        from ai_edge_litert.interpreter import Interpreter
        model, labels = download_yamnet()
        self.labels = [l.strip() for l in open(labels)]
        self.it = Interpreter(str(model))
        self.inp = self.it.get_input_details()[0]["index"]
        out = self.it.get_output_details()
        self.scores_idx, self.emb_idx = out[0]["index"], out[1]["index"]

    def embed(self, waveform):
        w = np.asarray(waveform, dtype=np.float32)
        self.it.resize_tensor_input(self.inp, [len(w)])
        self.it.allocate_tensors()
        self.it.set_tensor(self.inp, w)
        self.it.invoke()
        return self.it.get_tensor(self.emb_idx).copy(), self.it.get_tensor(self.scores_idx).copy()


# ---------------------------------------------------------------- trainable head
class YAMNetHead(nn.Module):
    def __init__(self, kind="bilstm", n_classes=2, emb=1024, hidden=64, dropout=0.5):
        super().__init__()
        self.kind = kind
        self.drop = nn.Dropout(dropout)
        if kind == "bilstm":
            self.rnn = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(2 * hidden, n_classes)
        else:                                            # "mean" / "max": order-free pooling heads
            self.fc = nn.Sequential(nn.Linear(emb, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):                                # x: (B, T, 1024)
        x = self.drop(x)
        if self.kind == "bilstm":
            h, _ = self.rnn(x)                           # (B, T, 128): forward and backward states per frame
            return self.fc(h.mean(1))
        if self.kind == "max":                           # strongest response over frames, ignores silent padding
            return self.fc(x.max(1).values)
        return self.fc(x.mean(1))
