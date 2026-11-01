"""Cache frozen YAMNet embeddings for every recording in data/processed/splits.csv (input for M4).

    python -m src.data.extract_yamnet
Outputs: data/processed/yamnet_emb.npy (N, 8, 1024), yamnet_scores.npy (N, 521, mean over frames)
"""
import numpy as np, pandas as pd
from src import config as C
from src.data.preprocessing import load_audio, clean_peak
from src.models.m4_yamnet import YAMNet


def recording_path(row):
    return C.RAW / str(row.batch) / row.pid / "cough-heavy.wav"


if __name__ == "__main__":
    df = pd.read_csv(C.PROCESSED / "splits.csv", dtype={"batch": str})
    yam = YAMNet()
    embs, scores = [], []
    for i, r in df.iterrows():
        e, s = yam.embed(clean_peak(load_audio(recording_path(r))))
        embs.append(e); scores.append(s.mean(0))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(df)}", flush=True)
    np.save(C.PROCESSED / "yamnet_emb.npy", np.stack(embs).astype(np.float32))
    np.save(C.PROCESSED / "yamnet_scores.npy", np.stack(scores).astype(np.float32))
    top = pd.Series([yam.labels[s.argmax()] for s in scores]).value_counts().head(5)
    print("embeddings:", np.stack(embs).shape, "\nmost frequent YAMNet top class:\n" + top.to_string())
