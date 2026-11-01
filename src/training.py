"""Training loop shared by the deep models (M2, M3, M4 head) so all are trained identically (Report Section V-B).

Adam + cross-entropy, early stopping on validation loss (patience C.PATIENCE), best checkpoint restored.
"""
import time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from src import config as C


def seed_all(seed=C.SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def augment_mel(x, specaug=False):
    """Training-only augmentation on a standardised log-Mel (1, 128, 401)."""
    x = torch.roll(x, shifts=int(np.random.randint(-40, 41)), dims=-1)   # time shift, +/-0.4 s
    x = torch.roll(x, shifts=int(np.random.randint(-2, 3)), dims=-2)     # +/-2 Mel bins ~ small pitch shift
    x = x + 0.01 * torch.randn_like(x)                                   # Gaussian noise
    if specaug:                                                          # SpecAugment: 1 time + 1 frequency mask
        t0, tw = np.random.randint(0, x.shape[-1] - 40), np.random.randint(0, 41)
        f0, fw = np.random.randint(0, x.shape[-2] - 16), np.random.randint(0, 17)
        x = x.clone()
        x[..., t0:t0 + tw] = 0.0
        x[..., f0:f0 + fw, :] = 0.0
    return x


def augment_emb(x):
    """Training-only augmentation on standardised YAMNet embeddings (8, 1024): small Gaussian noise."""
    return x + 0.1 * torch.randn_like(x)


def batches(X, y, train, kind, specaug):
    idx = np.random.permutation(len(X)) if train else np.arange(len(X))
    for i in range(0, len(idx), C.BATCH):
        b = idx[i:i + C.BATCH]
        xb = torch.from_numpy(X[b])
        if kind == "mel":
            xb = xb.unsqueeze(1)                                         # add channel axis for Conv2d
        if train:
            xb = torch.stack([augment_mel(v, specaug) if kind == "mel" else augment_emb(v) for v in xb])
        yield xb, torch.from_numpy(y[b])


def normaliser(X, tr, kind):
    """Statistics from the TRAINING indices only (no leakage). Mel: one global mean/std; embeddings: per dimension."""
    if kind == "mel":
        return X[tr].mean(), X[tr].std()
    return X[tr].mean((0, 1), keepdims=True), X[tr].std((0, 1), keepdims=True) + 1e-6


def predict(model, X, kind):
    model.eval()
    with torch.no_grad():
        return torch.cat([torch.softmax(model(xb), 1)[:, 1] for xb, _ in
                          batches(X, np.zeros(len(X), dtype=np.int64), False, kind, False)]).numpy()


def train_and_predict(model_fn, X, y, tr, va, te, kind="mel", lr=C.LR, specaug=False, verbose=True):
    """Train on tr, early-stop on va, return P(COVID) on va and te plus training info.

    model_fn: callable returning a fresh nn.Module. kind: "mel" (M2, M3) or "emb" (M4).
    """
    seed_all()
    mu, sd = normaliser(X, tr, kind)
    Xn = ((X - mu) / sd).astype(np.float32)
    model = model_fn()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    best, best_ep, wait, history, state = float("inf"), 0, 0, [], None
    t0 = time.time()
    for ep in range(1, C.EPOCHS + 1):
        model.train()
        tl = 0.0
        for xb, yb in batches(Xn[tr], y[tr], True, kind, specaug):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(yb)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in batches(Xn[va], y[va], False, kind, False):
                vl += loss_fn(model(xb), yb).item() * len(yb)
        tl, vl = tl / len(tr), vl / len(va)
        history.append([ep, tl, vl])
        if verbose:
            print(f"  epoch {ep:2d}  train {tl:.3f}  val {vl:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if vl < best - 1e-4:
            best, best_ep, wait = vl, ep, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= C.PATIENCE:
                break
    model.load_state_dict(state)
    p_val, p_te = predict(model, Xn[va], kind), predict(model, Xn[te], kind)
    info = dict(params=sum(q.numel() for q in model.parameters()), best_epoch=best_ep, best_val_loss=round(best, 4),
                val_auc=round(float(roc_auc_score(y[va], p_val)), 3), epochs_run=len(history),
                train_time_s=round(time.time() - t0))
    return dict(p_val=p_val, p_test=p_te, info=info, history=history, model=model,
                norm=(np.asarray(mu, dtype=np.float32), np.asarray(sd, dtype=np.float32)))
