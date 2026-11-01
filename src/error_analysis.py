"""Error analysis on out-of-fold (OOF) predictions: every one of the 328 recordings is predicted exactly
once by a model that never saw it (5-fold CV, selected configuration).

    python -m src.error_analysis        -> results/error_analysis.json (+ printed summary)

1. OOF AUC with 95% bootstrap CI, and paired bootstrap of the AUC difference vs. M1
2. Sensitivity by COVID severity, specificity/sensitivity by gender and age group
3. Error overlap between models and pairwise agreement (Cohen's kappa)
4. Recording properties (active cough duration, loudness) for correct vs. wrong predictions
5. Simple ensemble: average of the four models' probabilities
"""
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from src import config as C
from src.evaluation import metrics
from src.experiments import NAMES
from src.data.preprocessing import load_audio, energy_vad

MODELS = ["m1", "m2", "m3", "m4"]
rng = np.random.default_rng(C.SEED)


def oof():
    P = {}
    for m in MODELS:
        d = json.load(open(C.RESULTS / "cv" / f"{m}.json"))
        P[m] = np.array(d["configs"][d["selected"]]["oof"])
        y = np.array(d["y"])
    return P, y


def boot_auc(y, p, q=None, n=2000):
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n)]
    idx = [i for i in idx if 0 < y[i].sum() < len(i)]
    a = np.array([roc_auc_score(y[i], p[i]) for i in idx])
    if q is None:
        return np.percentile(a, [2.5, 97.5]).round(3).tolist()
    d = a - np.array([roc_auc_score(y[i], q[i]) for i in idx])
    return dict(diff=round(float(roc_auc_score(y, p) - roc_auc_score(y, q)), 3),
                ci=np.percentile(d, [2.5, 97.5]).round(3).tolist(), p_le_0=round(float((d <= 0).mean()), 3))


def recording_props(df):
    dur, rms = [], []
    for _, r in df.iterrows():
        w = load_audio(C.RAW / str(r.batch) / r.pid / "cough-heavy.wav")
        seg = energy_vad(w)
        dur.append(len(seg) / C.SR); rms.append(float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)))
    return np.array(dur), np.array(rms)


if __name__ == "__main__":
    df = pd.read_csv(C.PROCESSED / "splits.csv", dtype={"batch": str})
    meta = pd.read_csv(C.metadata_csv()).set_index("id")
    df["gender"] = df.pid.map(meta["g"]).fillna("unknown")
    df["age"] = pd.to_numeric(df.pid.map(meta["a"]), errors="coerce")
    df["age_group"] = pd.cut(df.age, [0, 30, 45, 120], labels=["<30", "30-45", ">45"]).astype(str)
    P, y = oof()
    P["ens"] = np.mean([P[m] for m in MODELS], axis=0)
    names = {**NAMES, "ens": "Average of M1-M4"}
    pred = {m: (p >= 0.5).astype(int) for m, p in P.items()}
    out = {}

    # 1. overall OOF metrics with CIs
    out["overall"] = {m: {**metrics(y, P[m]), "auc_ci": boot_auc(y, P[m]),
                          **({"vs_m1": boot_auc(y, P[m], P["m1"])} if m != "m1" else {})} for m in P}

    # 2. subgroups
    def rate(mask, m, label):
        k = mask & (y == label)
        return dict(n=int(k.sum()), rate=round(float((pred[m][k] == label).mean()), 2) if k.sum() else None)
    out["sensitivity_by_severity"] = {s: {m: rate((df.status == s).to_numpy(), m, 1) for m in P}
                                      for s in ["positive_asymp", "positive_mild", "positive_moderate"]}
    out["by_gender"] = {g: {m: dict(sens=rate((df.gender == g).to_numpy(), m, 1), spec=rate((df.gender == g).to_numpy(), m, 0))
                            for m in P} for g in ["male", "female"]}
    out["by_age"] = {a: {m: dict(sens=rate((df.age_group == a).to_numpy(), m, 1), spec=rate((df.age_group == a).to_numpy(), m, 0))
                         for m in P} for a in ["<30", "30-45", ">45"]}

    # 3. error overlap
    wrong = np.stack([pred[m] != y for m in MODELS])
    n_wrong = wrong.sum(0)
    out["overlap"] = dict(all_correct=int((n_wrong == 0).sum()), all_wrong=int((n_wrong == 4).sum()),
                          histogram={k: int((n_wrong == k).sum()) for k in range(5)},
                          kappa={f"{a}-{b}": round(float(cohen_kappa_score(pred[a], pred[b])), 2)
                                 for i, a in enumerate(MODELS) for b in MODELS[i + 1:]})

    # 4. recording properties of hard cases
    dur, rms = recording_props(df)
    hard, easy = n_wrong >= 3, n_wrong <= 1
    out["hard_vs_easy"] = dict(n_hard=int(hard.sum()), n_easy=int(easy.sum()),
                               active_duration_s=dict(hard=round(float(np.median(dur[hard])), 2), easy=round(float(np.median(dur[easy])), 2)),
                               level_db=dict(hard=round(float(np.median(rms[hard])), 1), easy=round(float(np.median(rms[easy])), 1)),
                               covid_share=dict(hard=round(float(y[hard].mean()), 2), easy=round(float(y[easy].mean()), 2)))
    json.dump(out, open(C.RESULTS / "error_analysis.json", "w"), indent=1)

    for m in P:
        o = out["overall"][m]
        extra = f"  dAUC vs M1 {o['vs_m1']['diff']:+.3f} CI {o['vs_m1']['ci']}" if "vs_m1" in o else ""
        print(f"{names[m]:22s} OOF AUC {o['auc']:.3f} CI {o['auc_ci']}  acc {o['accuracy']}%  sens {o['recall'][1]}  spec {o['recall'][0]}{extra}")
    print("sensitivity by severity:", json.dumps(out["sensitivity_by_severity"]))
    print("overlap:", out["overlap"])
    print("hard vs easy:", out["hard_vs_easy"])
