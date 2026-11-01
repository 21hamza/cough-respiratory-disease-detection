"""Build the labelled, batch-matched dataset and the fixed train/val/test split (Report Section IV-A).

Labels: 'healthy' -> 0, 'positive_*' (mild, moderate, asymptomatic) -> 1; everything else excluded.
Balancing: inside each collection batch, keep equally many healthy and positive participants
(random, seeded), so the class says nothing about when or with which app version it was recorded.

    python -m src.data.build_dataset
Outputs (data/processed/): splits.csv, mels.npy (N,128,401 float16), mfcc.npy (N,78), summary.json
"""
import json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from src import config as C
from src.data.preprocessing import load_audio, clean, log_mel, mfcc_stats


def collect():
    meta = pd.read_csv(C.metadata_csv()).set_index("id")
    rows = []
    for f in sorted(C.RAW.glob("*/*/cough-heavy.wav")):
        batch, pid = f.parts[-3], f.parts[-2]
        if batch not in C.BATCHES or pid not in meta.index:
            continue
        status = meta.loc[pid, "covid_status"]
        if status == "healthy":
            label = 0
        elif str(status).startswith("positive"):
            label = 1
        else:
            continue
        rows.append((str(f), batch, pid, label, status))
    return pd.DataFrame(rows, columns=["path", "batch", "pid", "label", "status"])


def featurise(df):
    keep, mels, mfccs, durs = [], [], [], []
    for i, r in df.iterrows():
        y = load_audio(r.path)
        if y is None or np.abs(y).max() < 1e-4:       # unreadable, too short or silent
            continue
        durs.append(len(y) / C.SR)
        y = clean(y)
        mels.append(log_mel(y).astype(np.float16))
        mfccs.append(mfcc_stats(y))
        keep.append(i)
    print(f"usable recordings: {len(keep)} / {len(df)}")
    return df.loc[keep].reset_index(drop=True), np.stack(mels), np.stack(mfccs), np.array(durs)


def balance_within_batch(df):
    rng = np.random.default_rng(C.SEED)
    sel = []
    for _, g in df.groupby("batch"):
        k = min((g.label == 0).sum(), (g.label == 1).sum())
        for lab in (0, 1):
            sel += list(rng.choice(g.index[g.label == lab].to_numpy(), size=k, replace=False))
    return sorted(sel)


def split(df):
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=df.label, random_state=C.SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=df.label.iloc[tmp], random_state=C.SEED)
    df["split"] = ""
    df.loc[tr, "split"], df.loc[va, "split"], df.loc[te, "split"] = "train", "val", "test"
    assert df.pid.is_unique, "one recording per participant -> no speaker overlap between splits"
    return df


if __name__ == "__main__":
    df = collect()
    df, mels, mfccs, durs = featurise(df)
    sel = balance_within_batch(df)
    df, mels, mfccs, durs = df.loc[sel].reset_index(drop=True), mels[sel], mfccs[sel], durs[sel]
    df = split(df)

    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.drop(columns="path").to_csv(C.PROCESSED / "splits.csv", index=False)
    np.save(C.PROCESSED / "mels.npy", mels)
    np.save(C.PROCESSED / "mfcc.npy", mfccs)
    summary = {s: {"healthy": int(((df.split == s) & (df.label == 0)).sum()),
                   "covid": int(((df.split == s) & (df.label == 1)).sum())} for s in ("train", "val", "test")}
    summary.update(total=len(df), status_counts=df.status.value_counts().to_dict(),
                   median_duration_s=round(float(np.median(durs)), 2), mel_shape=list(mels.shape))
    json.dump(summary, open(C.PROCESSED / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
