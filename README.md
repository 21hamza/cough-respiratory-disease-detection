# Cough Sound-Based Respiratory Disease Detection

ICT 4442 Deep Learning mini project. We compare four models from different architecture families for detecting
COVID-19 from cough recordings in the public [Coswara](https://github.com/iiscleap/Coswara-Data) database. All four use the
same confound-controlled data, preprocessing, splits and metrics, with hyperparameters chosen by nested cross-validation.

| ID | Model | Family | Owner |
|---|---|---|---|
| M1 | MFCC statistics + RBF SVM | Classical baseline | Mohammad Hamza Alam |
| M2 | 2D-CNN on log-Mel spectrograms (global max pooling) | CNN | Kummarasani Varun |
| M3 | CNN–LSTM on log-Mel spectrograms | CNN + RNN | Kanwarpal Singh Sachdeva |
| M4 | Frozen pretrained YAMNet (AudioSet) + BiLSTM head | Transfer learning | Devesh Gautam |


## Results

5-fold = mean ± std over outer folds (primary measure). OOF = out-of-fold predictions for all 328 recordings, with 95%
bootstrap CI. Test = fixed split, 50 recordings (25 / 25).

| Model | Selected setting | 5-fold AUC | OOF AUC [95% CI] | Test AUC | Test acc. | Sens. | Spec. |
|---|---|---|---|---|---|---|---|
| M1 MFCC + SVM | C=10, γ=0.001 | **0.62 ± 0.08** | 0.62 [0.55, 0.68] | 0.70 | 64% | 0.52 | 0.76 |
| M2 2D-CNN | max pooling, lr 1e-3 | 0.61 ± 0.06 | 0.59 [0.53, 0.65] | 0.67 | 62% | 0.52 | 0.72 |
| M3 CNN–LSTM | lr 3e-4, SpecAugment | 0.56 ± 0.07 | 0.57 [0.51, 0.63] | 0.64 | 58% | 0.48 | 0.68 |
| M4 YAMNet + BiLSTM | BiLSTM head, lr 3e-4 | 0.56 ± 0.09 | 0.56 [0.50, 0.62] | 0.49 | 52% | 0.48 | 0.56 |
| Average of M1–M4 | – | – | **0.65 [0.59, 0.71]** | – | – | – | – |

Main findings:
- **All four models are above chance, and no deep model significantly beats the baseline.** Paired bootstrap ΔAUC vs. M1
  ranges from −0.03 to −0.06, and every interval contains 0.
- **Max pooling fixed M2.** The 5-fold AUC goes from 0.50 with average pooling (Phase 2) to 0.62.
- **The models make different errors** (Cohen's κ 0.04–0.23), so averaging their probabilities gives the best OOF AUC.
- **Main failure: missed positives.** Classes are batch-matched but differ in age and gender, a residual confound.
- **Noise robustness:** M1 degrades steadily with added noise, while M2 is the most stable.

Details are in `results/`, `figures/` and the final report.

## How to run (VS Code)

1. Open the `cough-dl` folder in VS Code (**File → Open Folder**). Select a Python 3.10+ interpreter
   (**Ctrl+Shift+P → Python: Select Interpreter**).
2. Open a terminal (**Terminal → New Terminal**) and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline from the terminal, or from **Run and Debug** (Ctrl+Shift+D): pick a configuration and press F5.

```bash
python main.py                                   # complete pipeline, raw audio -> figures (~2.5 h on CPU)
python main.py --stage train                     # train + test all four models on the fixed split (~15 min)
python main.py --stage train --models m2         # one model
python main.py --from-stage tune                 # tuning and everything after it
python predict.py samples/covid_1.wav            # classify a recording with all four trained models
```

| Stage | What it does | Code | Time (2-core CPU) |
|---|---|---|---|
| `download` | fetch 15 Coswara batches, extract heavy-cough recordings | `src/data/download_coswara.py` | ~5 min |
| `dataset` | labels, cleaning, batch-matched balancing, split, MFCC + log-Mel cache | `src/data/build_dataset.py`, `preprocessing.py` | ~3 min |
| `yamnet` | download pretrained YAMNet, cache embeddings for M4 | `src/data/extract_yamnet.py` | ~7 min |
| `tune` | hyperparameter grid + nested 5-fold CV for all models | `src/run_experiment.py --cv` | ~2 h |
| `train` | train selected settings on the fixed split, test, save models | `src/run_experiment.py` | ~15 min |
| `analyse` | out-of-fold error analysis, bootstrap CIs | `src/error_analysis.py` | ~2 min |
| `robust` | noise-robustness test | `src/robustness.py` | ~1 min |
| `figures` | comparison tables and report figures | `src/make_figures.py` | <1 min |

The repository already contains the outputs of the runs used in the report: cached features (`data/processed/`), trained
models (`results/checkpoints/`), results (`results/`), figures (`figures/`) and the console logs of every stage (`logs/`).
So any stage can be rerun on its own. Seeds are fixed (42), and reruns reproduce the reported numbers exactly.
`data/processed/mels.npy` (33 MB) is excluded from git; `python main.py --from-stage download` or
`--stage dataset` (after `download`) recreates it.

## Data

- **Source:** Coswara (IISc Bangalore), one `cough-heavy.wav` per participant, labels from `combined_data.csv`.
- **Labels:** `healthy` → 0; `positive_asymp` / `positive_mild` / `positive_moderate` → 1. Other statuses are excluded.
- **Confound control:** In Coswara, most healthy recordings are from 2020 and most positives from 2021–22. We use only
  the 2021–22 batches and sample equal healthy and positive counts **within each batch**.
- **Final set:** 328 recordings from 328 participants (164 / 164). The fixed split is stratified 70/15/15: 229 train,
  49 val, 50 test.

## Method in brief

- **Preprocessing** (`src/data/preprocessing.py`): mono, 16 kHz → spectral gating → energy VAD → normalise → 4 s.
- **Features:** M1 uses 78-dim MFCC statistics; M2/M3 use a 128 × 401 log-Mel spectrogram; M4 uses 8 × 1024 YAMNet
  embeddings.
- **Training** (`src/training.py`): Adam, weight decay 1e-4, batch 16, ≤40 epochs, early stopping (patience 8).
- **Selection** (`src/experiments.py`): every grid setting is run in 5-fold CV with an inner 15% validation split. The
  setting with the best mean **inner-validation** AUC is selected, so test folds are never used for choices.
- **YAMNet:** the official TF Hub model in TFLite form, from Google's
  [tensorflow/tflite-support](https://github.com/tensorflow/tflite-support) repository, run with `ai-edge-litert`.

## Repository layout

```
src/
  config.py                  paths and shared hyperparameters
  data/download_coswara.py   fetch + extract Coswara batches
  data/preprocessing.py      audio cleaning, features, noise injection
  data/build_dataset.py      labels, batch-matched balancing, splits, feature cache
  data/extract_yamnet.py     frozen YAMNet embeddings for M4
  models/m1_svm.py           M1
  models/m2_cnn2d.py         M2
  models/m3_cnn_lstm.py      M3
  models/m4_yamnet.py        M4 (YAMNet wrapper + heads)
  training.py                shared training loop for M2-M4
  evaluation.py              common metrics
  experiments.py             model registry + hyperparameter grids
  run_experiment.py          nested CV, fixed split, Phase 2 reproduction
  error_analysis.py          OOF error analysis, bootstrap CIs
  robustness.py              noise-robustness test
  make_figures.py            tables and figures
results/                     JSON results, tables, checkpoints
figures/                     generated figures
docs/                        literature review, viva notes, final report
logs/                        console output of every pipeline stage
samples/                     8 held-out test recordings for predict.py (Coswara, CC BY 4.0)
main.py                      complete pipeline (all stages)
predict.py                   classify any .wav with all four trained models
.vscode/                     run configurations for VS Code
data/metadata/               Coswara labels (combined_data.csv, CC BY 4.0)
```

## Use of AI tools

The course guidelines require disclosing LLM use. An AI assistant (Claude, Anthropic) was used to draft and refactor parts of
the code, to draft report text and to format the report and slides. _(Team: edit this so it states exactly where AI was and
was not used.)_

## References

See `docs/literature_review.md` and the final report.
