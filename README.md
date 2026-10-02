# Cough Sound-Based Respiratory Disease Detection

ICT 4442 Deep Learning mini project. The project compares models of increasing complexity for detecting COVID-19 from cough
recordings in the public [Coswara](https://github.com/iiscleap/Coswara-Data) database. Every model uses the same data,
preprocessing, splits and metrics.

| ID | Model | Architecture family | Status |
|---|---|---|---|
| M1 | MFCC statistics + RBF SVM | Classical baseline | Implemented |
| M2 | 2D-CNN on log-Mel spectrograms | CNN | Implemented |
| M3 | CNN–LSTM on log-Mel spectrograms | CNN + RNN | Implemented |
| M4 | Pretrained audio CNN + BiLSTM head | Transfer learning | Planned (final phase) |

## Results (Phase 2, unoptimised)

Test set: 50 recordings (25 healthy, 25 COVID). Decision threshold 0.5.

| Model | Test acc. | Test AUC | Sensitivity | Specificity | Macro F1 | 5-fold AUC |
|---|---|---|---|---|---|---|
| M1 MFCC + SVM | 66.00% | 0.74 | 0.60 | 0.72 | 0.66 | 0.64 ± 0.08 |
| M2 2D-CNN | 40.00% | 0.47 | 0.12 | 0.68 | 0.35 | 0.50 ± 0.05 |
| M3 CNN–LSTM | 70.00% | 0.78 | 0.60 | 0.80 | 0.70 | 0.56 ± 0.06 |

M3 is best on the fixed split, but cross-validation shows that lead is not stable: the SVM baseline is the most
consistent model, and M2 stays at chance. See the report's Discussion section for why.
`python -m src.make_figures` regenerates this table (`results/comparison_table.md`) and the figures.

## Setup

```bash
pip install -r requirements.txt          # Python 3.10+, CPU is enough
```

## Reproduce everything

```bash
python -m src.data.download_coswara      # ~0.6 GB, 15 Coswara batches -> data/raw/   (~5 min)
python -m src.data.build_dataset         # preprocessing + balanced splits -> data/processed/   (~3 min)
python -m src.run_experiment --models m1 m2 m3          # fixed split -> results/main/   (~10 min on CPU)
python -m src.run_experiment --models m1 m2 m3 --cv     # 5-fold CV   -> results/cv/     (~25 min on CPU)
python -m src.make_figures               # table + figures -> results/, figures/
```

To run a single model, pass just that model, for example `python -m src.run_experiment --models m3`. Seeds are fixed (42),
so reruns give the same numbers on the same machine.

## Data

- **Source:** Coswara (IISc Bangalore), `cough-heavy.wav` per participant, with labels from `combined_data.csv`.
- **Labels:** `healthy` → 0; `positive_mild` / `positive_moderate` / `positive_asymp` → 1. Others are excluded.
- **Confound control:** In Coswara, most healthy recordings are from 2020 and most positives from 2021–22. To keep the models
  from learning recording period instead of cough, only 2021–22 batches are used, with equal healthy and positive counts
  sampled **within each batch**.
- **Final set:** 328 recordings from 328 participants (164 / 164). Splits are stratified 70/15/15:
  229 train, 49 val, 50 test. Each participant has exactly one recording, so no speaker appears in two splits.

## Preprocessing (`src/data/preprocessing.py`)

The pipeline runs: mono → 16 kHz → spectral gating (noisereduce) → energy-based VAD (frames with RMS > 1.5 × median RMS)
→ zero-mean/unit-variance → pad/crop to 4 s. Then it computes either a 128-band log-Mel spectrogram (n_fft 512, hop 160 →
128 × 401) for M2/M3, or 13 MFCC + Δ + ΔΔ mean/std (78-dim) for M1. Training-only augmentation:
time shift ±0.4 s, Mel-bin shift ±2, and Gaussian noise σ = 0.01.

## Repository layout

```
src/
  config.py                  shared paths and hyperparameters
  data/download_coswara.py   fetch + extract Coswara batches
  data/preprocessing.py      audio cleaning and features
  data/build_dataset.py      labels, batch-matched balancing, splits, feature cache
  models/m1_svm.py           M1
  models/m2_cnn2d.py         M2
  models/m3_cnn_lstm.py      M3
  training.py                shared training loop for M2/M3 (Adam, early stopping)
  evaluation.py              common metrics
  run_experiment.py          fixed-split and cross-validation runs
  make_figures.py            comparison table, loss curves, confusion matrices, ROC
results/                     JSON results (per model), comparison table
figures/                     generated figures
docs/                        literature review, model notes for the viva
```

## Model ownership

Edit this section to match your team's real split. The task-to-member split comes from the report's contribution log.

| Part | Files | Owner |
|---|---|---|
| Data acquisition & preprocessing | `src/data/*` | |
| M1 + literature review | `src/models/m1_svm.py`, `docs/literature_review.md` | |
| M2 | `src/models/m2_cnn2d.py` | |
| M3 + cross-validation & figures | `src/models/m3_cnn_lstm.py`, `src/make_figures.py` | |

## Use of AI tools

The course guidelines require disclosing LLM use. State here exactly where an AI assistant was used (for example, code
scaffolding, debugging or text drafting) and what the team wrote or changed itself.

## References

See `docs/literature_review.md` and the project report.
