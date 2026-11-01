"""Inference: classify cough recordings with all four trained models.

    python predict.py samples/healthy_1.wav                 # one file
    python predict.py samples/*.wav                         # several files

Uses the trained fixed-split models in results/checkpoints/ (created by `python -m src.run_experiment`).
For each recording it prints P(COVID) from every model, the decision at threshold 0.5, and YAMNet's own top
AudioSet labels as a sanity check. This is a course project, not a medical device.
"""
import sys, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from src import config as C
from src.data.preprocessing import load_audio, clean, clean_peak, log_mel, mfcc_stats
from src.models.m2_cnn2d import CNN2D
from src.models.m3_cnn_lstm import CNNLSTM
from src.models.m4_yamnet import YAMNet, YAMNetHead
from src.training import predict as predict_deep

NAMES = {"m1": "M1 MFCC + SVM", "m2": "M2 2D-CNN", "m3": "M3 CNN-LSTM", "m4": "M4 YAMNet + BiLSTM"}


def load_models():
    ck = C.RESULTS / "checkpoints"
    models = {"m1": pickle.load(open(ck / "m1_svm.pkl", "rb"))}
    for m in ("m2", "m3", "m4"):
        d = torch.load(ck / f"{m}.pt", weights_only=False)
        cfg = d["config"]
        net = {"m2": lambda: CNN2D(pool=cfg["pool"]), "m3": CNNLSTM, "m4": lambda: YAMNetHead(kind=cfg["head"])}[m]()
        net.load_state_dict(d["state_dict"]); net.eval()
        models[m] = (net, d["norm"])
    return models


def classify(path, models, yam):
    raw = load_audio(path)
    if raw is None:
        raise ValueError(f"{path}: unreadable or shorter than 0.3 s")
    y = clean(raw)                                                    # same pipeline as training
    mel = log_mel(y).astype(np.float16).astype(np.float32)[None]      # (1, 128, 401)
    mfcc = mfcc_stats(y).astype(np.float32)[None]                     # (1, 78)
    emb, scores = yam.embed(clean_peak(raw))                          # (8, 1024), (8, 521)
    p = {"m1": float(models["m1"].predict_proba(mfcc)[0, 1])}
    for m, X, kind in (("m2", mel, "mel"), ("m3", mel, "mel"), ("m4", emb[None].astype(np.float32), "emb")):
        net, (mu, sd) = models[m]
        p[m] = float(predict_deep(net, ((X - mu) / sd).astype(np.float32), kind)[0])
    top = [yam.labels[i] for i in np.argsort(scores.max(0))[::-1][:3]]
    return p, top, len(raw) / C.SR


if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        sys.exit(__doc__)
    models, yam = load_models(), YAMNet()
    for f in files:
        p, top, dur = classify(f, models, yam)
        avg = np.mean(list(p.values()))
        print(f"\n{f}  ({dur:.1f} s)   YAMNet hears: {', '.join(top)}")
        for m, v in p.items():
            bar = "#" * int(round(v * 20))
            print(f"  {NAMES[m]:20s} P(COVID) = {v:.3f}  {bar:<20s} -> {'COVID' if v >= 0.5 else 'healthy'}")
        print(f"  {'Average of 4 models':20s} P(COVID) = {avg:.3f}  {'#' * int(round(avg * 20)):<20s} -> "
              f"{'COVID' if avg >= 0.5 else 'healthy'}")
