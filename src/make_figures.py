"""Comparison tables and figures from results/ (final report and slides).

    python -m src.make_figures
Writes results/comparison_table.md, results/tuning_table.md and figures/*.png
"""
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, roc_auc_score
from src import config as C

MODELS = ["m1", "m2", "m3", "m4"]
NAMES = {"m1": "M1 MFCC + SVM", "m2": "M2 2D-CNN", "m3": "M3 CNN–LSTM", "m4": "M4 YAMNet + BiLSTM"}
COLOR = {"m1": "#2a78d6", "m2": "#eb6834", "m3": "#1baf7a", "m4": "#eda100"}
STYLE = {"m1": "-", "m2": "--", "m3": "-.", "m4": ":"}
MARK = {"m1": "o", "m2": "s", "m3": "^", "m4": "D"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5df"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def jl(path):
    return json.load(open(path)) if path.exists() else None


def load():
    main = {m: jl(C.RESULTS / "main" / f"{m}.json") for m in MODELS}
    cv = {m: jl(C.RESULTS / "cv" / f"{m}.json") for m in MODELS}
    return {k: v for k, v in main.items() if v}, {k: v for k, v in cv.items() if v}


def sel(cv, m):
    return cv[m]["configs"][cv[m]["selected"]]


def tables(main, cv):
    L = ["| Model | Selected setting | Test acc. | Test AUC | Sens. | Spec. | Macro F1 | 5-fold AUC | 5-fold acc. | Params |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        if m not in main or m not in cv:
            continue
        r, f = main[m]["results"], sel(cv, m)["folds"]
        auc, acc = [x["auc"] for x in f], [x["accuracy"] for x in f]
        L.append(f"| {NAMES[m]} | {main[m]['label']} | {r['accuracy']:.1f}% | {r['auc']:.2f} | {r['recall'][1]:.2f} | "
                 f"{r['recall'][0]:.2f} | {r['macro_f1']:.2f} | {np.mean(auc):.2f} ± {np.std(auc):.2f} | "
                 f"{np.mean(acc):.1f} ± {np.std(acc):.1f}% | {r['params']:,} |")
    (C.RESULTS / "comparison_table.md").write_text("\n".join(L) + "\n")
    T = ["| Model | Setting | Inner-val AUC | Outer 5-fold AUC | Selected |", "|---|---|---|---|---|"]
    for m in MODELS:
        if m in cv:
            for k, c in enumerate(cv[m]["configs"]):
                a = [x["auc"] for x in c["folds"]]
                T.append(f"| {NAMES[m]} | {c['label']} | {c['inner_val_auc']:.3f} | {np.mean(a):.3f} ± {np.std(a):.3f} | "
                         f"{'yes' if k == cv[m]['selected'] else ''} |")
    (C.RESULTS / "tuning_table.md").write_text("\n".join(T) + "\n")
    print("\n".join(L))


def loss_curves(main):
    deep = [m for m in ("m2", "m3", "m4") if m in main]
    fig, axs = plt.subplots(1, len(deep), figsize=(2.6 * len(deep), 2.5), squeeze=False)
    for ax, m in zip(axs[0], deep):
        h, r = np.array(main[m]["history"]), main[m]["results"]
        ax.plot(h[:, 0], h[:, 1], color="#2a78d6", lw=2, label="Train loss")
        ax.plot(h[:, 0], h[:, 2], color="#eb6834", lw=2, ls="--", label="Validation loss")
        ax.axvline(r["best_epoch"], color=MUTED, lw=1, ls=":")
        ax.annotate(f"best epoch {r['best_epoch']}", (r["best_epoch"], 1.0), xycoords=("data", "axes fraction"), xytext=(-3, -2),
                    textcoords="offset points", fontsize=7.5, color=INK, va="top", ha="right")
        ax.set_title(NAMES[m], fontsize=9, color=INK); ax.set_xlabel("Epoch"); ax.grid(axis="y", color=GRID, lw=0.6)
    axs[0][0].set_ylabel("Cross-entropy loss")
    fig.legend(*axs[0][0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(C.FIGURES / "loss_curves.png", dpi=220); plt.close(fig)


def confusion(main):
    ms = [m for m in MODELS if m in main]
    cmap = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#86b6ef", "#1c5cab"])
    fig, axs = plt.subplots(1, len(ms), figsize=(1.95 * len(ms), 2.3), squeeze=False)
    for ax, m in zip(axs[0], ms):
        cm = np.array(main[m]["results"]["cm"]); top = cm.sum(1).max()
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=top)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11, color="white" if cm[i, j] > 0.55 * top else INK)
        ax.set_xticks([0, 1], ["Healthy", "COVID"]); ax.set_yticks([0, 1], ["Healthy", "COVID"])
        ax.set_xlabel("Predicted"); ax.set_title(NAMES[m], fontsize=8.5, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0][0].set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(C.FIGURES / "confusion.png", dpi=220); plt.close(fig)


def roc_oof(cv):
    fig, ax = plt.subplots(figsize=(3.7, 4.0))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    for m in MODELS:
        if m in cv:
            y, p = np.array(cv[m]["y"]), np.array(sel(cv, m)["oof"])
            f, t, _ = roc_curve(y, p)
            ax.plot(f, t, color=COLOR[m], lw=2, ls=STYLE[m], label=f"{NAMES[m]} (AUC {roc_auc_score(y, p):.2f})")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2)); ax.grid(color=GRID, lw=0.6)
    fig.tight_layout(); fig.savefig(C.FIGURES / "roc_oof.png", dpi=220); plt.close(fig)


def cv_bars(cv):
    ms = [m for m in MODELS if m in cv]
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    for i, m in enumerate(ms):
        a = [x["auc"] for x in sel(cv, m)["folds"]]
        ax.bar(i, np.mean(a), color=COLOR[m], width=0.6, edgecolor="white", linewidth=2)
        ax.errorbar(i, np.mean(a), yerr=np.std(a), color=INK, capsize=4, lw=1)
        ax.scatter([i] * len(a), a, s=12, color=INK, zorder=3, alpha=0.6)
        ax.text(i, 0.32, f"{np.mean(a):.2f}", ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")
    ax.axhline(0.5, color=MUTED, lw=1, ls=":"); ax.text(len(ms) - 0.62, 0.505, "chance", fontsize=7.5, color=MUTED, ha="left", va="bottom"); ax.set_xlim(-0.5, len(ms) - 0.05)
    ax.set_xticks(range(len(ms)), [NAMES[m].replace(" ", "\n", 1) for m in ms], fontsize=8)
    ax.set_ylabel("AUC (5-fold CV)"); ax.set_ylim(0.3, 0.9); ax.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout(); fig.savefig(C.FIGURES / "cv_auc.png", dpi=220); plt.close(fig)


def robustness():
    r = jl(C.RESULTS / "robustness.json")
    if not r:
        return
    keys = list(r.keys())
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for m in MODELS:
        ax.plot(range(len(keys)), [r[k][m] for k in keys], color=COLOR[m], ls=STYLE[m], marker=MARK[m], lw=2, ms=5, label=NAMES[m])
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.set_xticks(range(len(keys)), ["clean" if k == "clean" else k.replace(" dB", "") for k in keys])
    ax.set_xlabel("Signal-to-noise ratio of added white noise (dB)"); ax.set_ylabel("Test AUC")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout(); fig.savefig(C.FIGURES / "robustness.png", dpi=220); plt.close(fig)


def severity():
    e = jl(C.RESULTS / "error_analysis.json")
    if not e:
        return
    groups = ["positive_asymp", "positive_mild", "positive_moderate"]
    lab = {"positive_asymp": "Asymptomatic", "positive_mild": "Mild", "positive_moderate": "Moderate"}
    fig, ax = plt.subplots(figsize=(4.6, 2.6)); w = 0.2
    for i, m in enumerate(MODELS):
        v = [e["sensitivity_by_severity"][g][m]["rate"] for g in groups]
        ax.bar(np.arange(3) + (i - 1.5) * w, v, w, color=COLOR[m], label=NAMES[m], edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(3), [f"{lab[g]}\n(n={e['sensitivity_by_severity'][g]['m1']['n']})" for g in groups])
    ax.set_ylabel("Sensitivity (OOF)"); ax.set_ylim(0, 1); ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout(); fig.savefig(C.FIGURES / "severity.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    C.FIGURES.mkdir(exist_ok=True)
    main, cv = load()
    tables(main, cv)
    loss_curves(main); confusion(main); roc_oof(cv); cv_bars(cv); robustness(); severity()
    print("figures written to", C.FIGURES)
