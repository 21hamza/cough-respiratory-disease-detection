"""Noise-robustness test (Objective 4): add white Gaussian noise to the RAW test recordings at decreasing
signal-to-noise ratios, run the full preprocessing again, and score the trained fixed-split models.

    python -m src.robustness            -> results/robustness.json
Requires results/checkpoints/ from `python -m src.run_experiment`.
"""
import json, pickle
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from src import config as C
from src.data.preprocessing import load_audio, add_noise, clean, clean_peak, log_mel, mfcc_stats
from src.models.m2_cnn2d import CNN2D
from src.models.m3_cnn_lstm import CNNLSTM
from src.models.m4_yamnet import YAMNet, YAMNetHead
from src.training import predict

SNRS = [None, 20, 10, 5, 0]          # None = clean


def load_deep(m):
    ck = torch.load(C.RESULTS / "checkpoints" / f"{m}.pt", weights_only=False)
    cfg = ck["config"]
    model = {"m2": lambda: CNN2D(pool=cfg["pool"]), "m3": CNNLSTM, "m4": lambda: YAMNetHead(kind=cfg["head"])}[m]()
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, ck["norm"]


if __name__ == "__main__":
    df = pd.read_csv(C.PROCESSED / "splits.csv", dtype={"batch": str})
    te = df[df.split == "test"]
    y = te.label.to_numpy()
    svm = pickle.load(open(C.RESULTS / "checkpoints" / "m1_svm.pkl", "rb"))
    deep = {m: load_deep(m) for m in ("m2", "m3", "m4")}
    yam = YAMNet()
    raw = [load_audio(C.RAW / r.batch / r.pid / "cough-heavy.wav") for _, r in te.iterrows()]
    out = {}
    for snr in SNRS:
        rng = np.random.default_rng(C.SEED)
        waves = raw if snr is None else [add_noise(w, snr, rng) for w in raw]
        cleaned = [clean(w) for w in waves]
        mel = np.stack([log_mel(w) for w in cleaned]).astype(np.float16).astype(np.float32)
        mfcc = np.stack([mfcc_stats(w) for w in cleaned]).astype(np.float32)
        emb = np.stack([yam.embed(clean_peak(w))[0] for w in waves]).astype(np.float32)
        p = {"m1": svm.predict_proba(mfcc)[:, 1]}
        for m, X, kind in (("m2", mel, "mel"), ("m3", mel, "mel"), ("m4", emb, "emb")):
            model, (mu, sd) = deep[m]
            p[m] = predict(model, ((X - mu) / sd).astype(np.float32), kind)
        key = "clean" if snr is None else f"{snr} dB"
        out[key] = {m: round(float(roc_auc_score(y, v)), 3) for m, v in p.items()}
        print(key, out[key], flush=True)
    json.dump(out, open(C.RESULTS / "robustness.json", "w"), indent=1)
