"""Model registry and the hyperparameter grid searched for the final comparison.

Selection rule (same for every model): for each configuration, run 5-fold stratified CV; inside each
fold, 15% of the training portion is held out as an inner validation set. The configuration with the
highest mean INNER-validation AUC is selected, so test folds are never used for selection. The
selected configuration's outer test-fold results are the reported CV results.
"""
from src.models import m1_svm
from src.models.m2_cnn2d import CNN2D
from src.models.m3_cnn_lstm import CNNLSTM
from src.models.m4_yamnet import YAMNetHead
from src.training import train_and_predict

NAMES = {"m1": "M1 MFCC + SVM", "m2": "M2 2D-CNN", "m3": "M3 CNN–LSTM", "m4": "M4 YAMNet + BiLSTM"}
FEATURE = {"m1": "mfcc", "m2": "mel", "m3": "mel", "m4": "yamnet"}

GRID = {
    "m1": [dict(c=c, gamma=g) for c in (0.1, 1.0, 10.0) for g in ("scale", 0.001, 0.01)],
    "m2": [dict(pool="avg", lr=1e-3), dict(pool="max", lr=1e-3), dict(pool="max", lr=3e-4)],
    "m3": [dict(lr=1e-3, specaug=False), dict(lr=3e-4, specaug=False), dict(lr=3e-4, specaug=True)],
    "m4": [dict(head=h, lr=lr) for h in ("bilstm", "mean", "max") for lr in (1e-3, 3e-4)],
}
PHASE2 = {"m1": dict(c=1.0, gamma="scale"), "m2": dict(pool="avg", lr=1e-3), "m3": dict(lr=1e-3, specaug=False)}


def run(model, cfg, X, y, tr, va, te, verbose=False):
    """Train one configuration of one model; returns dict(p_val, p_test, info, history, model, norm)."""
    if model == "m1":
        return m1_svm.fit_predict(X, y, tr, va, te, c=cfg["c"], gamma=cfg["gamma"])
    if model == "m2":
        return train_and_predict(lambda: CNN2D(pool=cfg["pool"]), X, y, tr, va, te, kind="mel", lr=cfg["lr"], verbose=verbose)
    if model == "m3":
        return train_and_predict(CNNLSTM, X, y, tr, va, te, kind="mel", lr=cfg["lr"], specaug=cfg["specaug"], verbose=verbose)
    if model == "m4":
        return train_and_predict(lambda: YAMNetHead(kind=cfg["head"]), X, y, tr, va, te, kind="emb", lr=cfg["lr"], verbose=verbose)
    raise ValueError(model)


def describe(model, cfg):
    if model == "m1":
        return f"C={cfg['c']}, gamma={cfg['gamma']}"
    if model == "m2":
        return f"{cfg['pool']} pooling, lr={cfg['lr']:g}"
    if model == "m3":
        return f"lr={cfg['lr']:g}, SpecAugment={'on' if cfg['specaug'] else 'off'}"
    return f"{cfg['head']} head, lr={cfg['lr']:g}"
