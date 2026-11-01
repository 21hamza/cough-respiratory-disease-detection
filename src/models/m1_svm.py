"""M1 - Classical baseline: RBF-kernel SVM on 78-dim MFCC statistics (Report Section V).

Tests whether handcrafted features + a classical classifier are enough. Hyperparameters C (margin
softness) and gamma (kernel width) are chosen on validation data. For the final test prediction the
SVM is refit on train + val, since it needs no early stopping.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from src import config as C


def build(c=1.0, gamma="scale"):
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c, gamma=gamma, probability=True, random_state=C.SEED))


def fit_predict(X, y, tr, va, te, c=1.0, gamma="scale"):
    """X: (N, 78) MFCC stats. Returns dict with P(COVID) on val (model fit on tr) and test (model refit on tr+va)."""
    clf_val = build(c, gamma).fit(X[tr], y[tr])
    p_val = clf_val.predict_proba(X[va])[:, 1]
    fit = np.concatenate([tr, va])
    clf = build(c, gamma).fit(X[fit], y[fit])
    svc = clf[-1]
    info = dict(params=int(svc.support_vectors_.size + svc.dual_coef_.size), support_vectors=int(len(svc.support_)),
                val_auc=round(float(roc_auc_score(y[va], p_val)), 3))
    return dict(p_val=p_val, p_test=clf.predict_proba(X[te])[:, 1], info=info, history=None, model=clf, norm=None)
