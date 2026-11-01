"""Common evaluation protocol: same metrics for every model (Report Section VI)."""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


def metrics(y_true, p_covid, threshold=0.5):
    """Class 0 = healthy, class 1 = COVID. recall[1] = sensitivity, recall[0] = specificity."""
    pred = (p_covid >= threshold).astype(int)
    pr, rc, f1, sup = precision_recall_fscore_support(y_true, pred, zero_division=0)
    return dict(accuracy=round(accuracy_score(y_true, pred) * 100, 2),
                auc=round(float(roc_auc_score(y_true, p_covid)), 3),
                precision=[round(float(v), 2) for v in pr],
                recall=[round(float(v), 2) for v in rc],
                f1=[round(float(v), 2) for v in f1],
                macro_f1=round(float(f1.mean()), 2),
                support=sup.tolist(),
                cm=confusion_matrix(y_true, pred).tolist())
