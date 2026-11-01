# Viva Notes: What Each Owner Should Be Able to Explain

The final viva questions each member on their own part. Each owner should read their files and rerun their part.

## Data & preprocessing (`src/data/`)
- Why use **heavy cough** recordings, and why are exposed and recovered participants excluded?
- **Batch-matched balancing:** what the recording-period confound is, and why a naive sample would let a model "cheat".
- Why does splitting need to happen per participant? (Here each participant has one recording, so it is automatic.)
- What each step does:
  - Spectral gating estimates a noise profile and subtracts it.
  - The VAD keeps frames above 1.5 × median RMS.
  - Normalisation removes loudness differences.
  - Padding to 4 s gives fixed-size inputs.
- How log-Mel works: STFT with 32 ms windows and a 10 ms hop, then 128 Mel bands, then log, giving 128 × 401.
- Why augmentation is training-only, and why standardisation uses training statistics only (no test leakage).

## M1: MFCC + SVM (`src/models/m1_svm.py`)
- What MFCCs are (log-Mel → DCT, keeping 13 coefficients), and what Δ / ΔΔ add.
- Why use mean + std pooling (fixed length, 78-dim), and what is lost (time order).
- RBF kernel, C and γ. Why StandardScaler matters for an SVM.
- Why it wins on 5-fold CV (0.62 ± 0.08): few, compact features give low variance on 229 samples.
- Tuning: grid C ∈ {0.1, 1, 10} × γ ∈ {scale, 0.001, 0.01}; selected C = 10, γ = 0.001. Why C = 0.1 gives AUC < 0.5
  (an almost constant SVM, so its Platt-scaled probabilities are unreliable).
- Weak points: loses most under noise (0.70 → 0.47 at 0 dB); its sensitivity is highest for women and over-45s, where
  positives are over-represented, so it partly learns age/gender cues. But it is the only model whose sensitivity rises
  with severity.

## M2: 2D-CNN (`src/models/m2_cnn2d.py`)
- Trace the shapes: 1×128×401 → 32×64×200 → 64×32×100 → global pool → 64 → 2. The parameter count is 19,138.
- What each layer does: conv, ReLU, max pool and batch norm.
- **Why it failed (AUC ≈ 0.5):** validation loss never improved after epoch 1. Global average pooling over 4 s mixes
  short cough bursts with silence and padding. There is very little data, and the model has no temporal model.
- **The fix (final):** global MAX pooling keeps each channel's strongest response ("does this pattern occur anywhere?").
  5-fold AUC 0.50 → 0.62 ± 0.06, tied with M1, and the most robust model under added noise.
- Why the high initial training loss with max pooling (large activations pass straight to the classifier).

## M3: CNN–LSTM (`src/models/m3_cnn_lstm.py`)
- The reshape: 64×32×100 → 100 time steps × 2,048 features. Why the time order must be kept.
- LSTM gates (input, forget, output), and why the stacked 128 → 64 LSTM uses the last hidden state.
- Why it has 1.18 M parameters (mostly the first LSTM: 4 × (2048 + 128 + 1) × 128).
- Tuning: lr 1e-3 / 3e-4 and SpecAugment (time + frequency masks). Selected lr 3e-4 + SpecAugment; inner-val AUC rose
  0.61 → 0.67 but test-fold AUC did not (0.565 → 0.558): the classic winner's curse of selecting on small sets.
- Best epochs vary from 4 to 38 across runs: high variance, too much capacity (1.16 M extra parameters) for 229 clips.
- Why the Phase 2 single-split result (AUC 0.78) and the CV result (0.56 ± 0.07) differ: a small test set and high variance.

## M4: YAMNet + BiLSTM (`src/models/m4_yamnet.py`, `src/data/extract_yamnet.py`)
- What transfer learning is: reuse features learned on a big dataset (AudioSet, ~2M clips, 521 classes incl. Cough),
  and train only a small head on our 229 recordings.
- YAMNet = MobileNet-v1 (depthwise-separable convolutions). It takes the raw 16 kHz waveform, computes its own
  log-Mel patches (0.96 s, hop 0.48 s), and outputs 1,024-d embeddings -> 8 embeddings per 4 s clip.
- Why frozen (not fine-tuned): too little data and CPU only; also it is the cleanest test of "are generic audio
  features enough?".
- Head: BiLSTM(64 each direction) over the 8 embeddings, mean over time, then linear. The BiLSTM reads the
  sequence forwards and backwards. Alternatives tried: mean-pool and max-pool MLP heads.
- Why peak-normalisation instead of z-score: YAMNet expects waveforms in [-1, 1].
- Where the weights come from: the official TF Hub YAMNet (.tflite) in Google's tensorflow/tflite-support repo,
  run with the LiteRT interpreter.
- Sanity check: YAMNet's own most frequent top class on our recordings is "Cough" (114 of 328).
- Result: 5-fold AUC 0.56 ± 0.09. Why pretraining did not help: YAMNet encodes THAT a sound is a cough, not the
  subtle within-class differences; and many of the 8 windows are silent padding.

## Final-phase protocol (everyone should know this)
- **Nested selection**: each setting in the grid is run with 5-fold CV; inside each fold 15% of the training part is
  an inner validation set. Pick the setting with the best mean inner-val AUC, so test folds are never used to choose.
- Why the selected M1 setting is not the one with the best test AUC: selection must not look at test data, and
  inner-val sets are small (~40 recordings), so selection is noisy. That is honest, not a bug.
- **OOF predictions**: every recording gets one prediction from a model that never saw it -> 328 predictions per
  model for error analysis and bootstrap confidence intervals.
- **Robustness test**: white noise added to the raw test audio at 20/10/5/0 dB SNR, full preprocessing rerun.

## Shared
- Metrics: sensitivity = COVID recall, specificity = healthy recall, AUC is threshold-free. Why accuracy alone is not
  enough for screening.
- Why 5-fold CV is more trustworthy than one 50-sample test set (one error = 2 percentage points).
- Why averaging the four models helps (OOF AUC 0.65): they make different errors (Cohen's κ 0.04–0.23).
- The residual confound: positives are older and more often female; batch matching does not remove that.
