"""Run the common evaluation protocol.

    python -m src.run_experiment --cv               # step 1: hyperparameter selection + 5-fold CV  -> results/cv/<m>.json
    python -m src.run_experiment                    # step 2: fixed split with the selected configs -> results/main/<m>.json
    python -m src.run_experiment --phase2           # reproduce the Phase 2 (untuned) numbers       -> results/phase2/
    add  --models m3 m4  to run only some models.
"""
import argparse, json, pickle
import numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from src import config as C
from src.evaluation import metrics
from src.experiments import GRID, PHASE2, NAMES, FEATURE, run, describe

ALL = ["m1", "m2", "m3", "m4"]


def load():
    df = pd.read_csv(C.PROCESSED / "splits.csv", dtype={"batch": str})
    feats = {"mel": np.load(C.PROCESSED / "mels.npy").astype(np.float32),
             "mfcc": np.load(C.PROCESSED / "mfcc.npy").astype(np.float32)}
    if (C.PROCESSED / "yamnet_emb.npy").exists():
        feats["yamnet"] = np.load(C.PROCESSED / "yamnet_emb.npy").astype(np.float32)
    return df, feats, df.label.to_numpy()


def cv_folds(y):
    skf = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    out = []
    for trva, te in skf.split(np.zeros(len(y)), y):
        tr, va = train_test_split(trva, test_size=0.15, stratify=y[trva], random_state=C.SEED)
        out.append((tr, va, te))
    return out


def do_cv(models, feats, y):
    folds = cv_folds(y)
    (C.RESULTS / "cv").mkdir(parents=True, exist_ok=True)
    for m in models:
        X, configs = feats[FEATURE[m]], []
        for k, cfg in enumerate(GRID[m]):
            fold_res, oof = [], np.full(len(y), np.nan)
            for tr, va, te in folds:
                r = run(m, cfg, X, y, tr, va, te)
                fold_res.append({**metrics(y[te], r["p_test"]), **r["info"]})
                oof[te] = r["p_test"]
            inner = float(np.mean([f["val_auc"] for f in fold_res]))
            configs.append(dict(config=cfg, label=describe(m, cfg), inner_val_auc=round(inner, 3), folds=fold_res, oof=oof.tolist()))
            print(f"{NAMES[m]} [{describe(m, cfg)}]  inner-val AUC {inner:.3f}  "
                  f"test AUC {np.mean([f['auc'] for f in fold_res]):.3f}", flush=True)
        best = int(np.argmax([c["inner_val_auc"] for c in configs]))
        json.dump(dict(model=NAMES[m], selected=best, configs=configs, y=y.tolist()),
                  open(C.RESULTS / "cv" / f"{m}.json", "w"), indent=1)
        f = configs[best]["folds"]
        print(f"==> {NAMES[m]} selected [{configs[best]['label']}]: AUC {np.mean([x['auc'] for x in f]):.2f} "
              f"+/- {np.std([x['auc'] for x in f]):.2f}", flush=True)


def selected_config(m):
    d = json.load(open(C.RESULTS / "cv" / f"{m}.json"))
    return d["configs"][d["selected"]]["config"]


def do_fixed(models, df, feats, y, phase2=False):
    tr, va, te = (np.where(df.split == s)[0] for s in ("train", "val", "test"))
    out_dir = C.RESULTS / ("phase2" if phase2 else "main")
    ckpt = C.RESULTS / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    for m in models:
        cfg = PHASE2[m] if phase2 else selected_config(m)
        print(f"== {NAMES[m]} [{describe(m, cfg)}] | train {len(tr)}  val {len(va)}  test {len(te)}", flush=True)
        r = run(m, cfg, feats[FEATURE[m]], y, tr, va, te, verbose=True)
        res = {**metrics(y[te], r["p_test"]), **r["info"]}
        json.dump(dict(model=NAMES[m], config=cfg, label=describe(m, cfg), results=res, history=r["history"],
                       p_test=r["p_test"].tolist(), y_test=y[te].tolist(), test_index=te.tolist()),
                  open(out_dir / f"{m}.json", "w"), indent=1)
        if not phase2:                                   # keep trained models for the robustness test and predict.py
            if m == "m1":
                pickle.dump(r["model"], open(ckpt / "m1_svm.pkl", "wb"))
            else:
                torch.save(dict(state_dict=r["model"].state_dict(), config=cfg, norm=r["norm"]), ckpt / f"{m}.pt")
        print(f"   accuracy {res['accuracy']}%  AUC {res['auc']}  sens {res['recall'][1]}  spec {res['recall'][0]}"
              f"  macro-F1 {res['macro_f1']}  cm {res['cm']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--cv", action="store_true", help="hyperparameter selection + 5-fold CV")
    ap.add_argument("--phase2", action="store_true", help="fixed split with the untuned Phase 2 settings")
    a = ap.parse_args()
    df, feats, y = load()
    if a.cv:
        do_cv(a.models, feats, y)
    else:
        do_fixed([m for m in a.models if not a.phase2 or m in PHASE2], df, feats, y, phase2=a.phase2)


if __name__ == "__main__":
    main()
