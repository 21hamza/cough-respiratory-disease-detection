# Literature Review Summary

| # | Paper | Method | Dataset | Key result | Relevance |
|---|---|---|---|---|---|
| 1 | Balamurali et al., *Sensors* 21(16):5555, 2021 | MFCC (+Δ, ΔΔ) with a 2-layer BiLSTM | Hospital smartphone coughs of children: healthy, asthma, URTI, LRTI | Healthy vs. pathological AUC 0.91, 91.2% at subject level | Recurrent models work on noisy clinical cough audio; MFCC reference |
| 2 | Imran et al., AI4COVID-19, *Inform. Med. Unlocked* 20:100378, 2020 | CNN on Mel spectrograms, SVM on MFCC+PCA, binary CNN; mediator with veto | 543 coughs: 70 COVID, 96 bronchitis, 130 pertussis, 247 normal | Binary CNN 92.85%; MFCC+SVM 88.76% (multi-class) | Spectrogram-CNN vs. MFCC-SVM, i.e. our M1 vs. M2 comparison |
| 3 | Laguarta et al., *IEEE OJEMB* 1:275–281, 2020 | Three pretrained ResNet50 biomarker branches on MFCC input | 5,320 subjects from MIT Open Voice | AUC 0.97; sensitivity 98.5%, specificity 94.2% | Pretrained extractors transfer to cough audio (M4) |
| 4 | Brown et al., *ACM KDD*, 2020 | Handcrafted + VGGish features, LR / SVM | Crowdsourced COVID-19 Sounds (cough, breathing) | AUC > 0.80 on all binary tasks | Realistic performance on crowdsourced data; VGGish embeddings |
| 5 | Sharma et al., Coswara, *Interspeech*, 2020 | Dataset paper + baseline | Crowdsourced breathing, cough, vowel, counting | Open multi-modality dataset | **Dataset used in this project** |
| 6 | Orlandic et al., COUGHVID, *Scientific Data* 8:156, 2021 | Dataset + XGBoost cough detector, expert labels | 25,000+ recordings; 2,800+ expert-annotated | Cough detector AUC 0.964 | Candidate for cross-dataset testing |
| 7 | Pahar et al., *Comput. Biol. Med.* 135:104572, 2021 | LR, KNN, SVM, MLP, CNN, LSTM, ResNet50 on the same features | Coswara (92 pos / 1,079 healthy) + lab-confirmed SA set | ResNet50 AUC 0.98 (Coswara); LSTM AUC 0.94 (lab set) | Closest design: several models compared on one pipeline |
| 8 | Pahar et al., *Comput. Biol. Med.* 141:105153, 2022 | Pretrain on unlabelled audio, fine-tune CNN / LSTM / ResNet50 | COVID-labelled cough, breath, speech sets | AUC 0.98 cough, 0.94 breath, 0.92 speech | Supports transfer learning for small data (M4) |
| 9 | Mohammed et al., *Scientific Reports* 11, 2021 | Six feature types; shallow models + CNN/VGG16; majority-vote ensemble | 1,502 participants (114 positive) | Ensemble AUC 0.77, F1 0.75 | Performance drops on larger, imbalanced data |
| 10 | Ghrabli et al., *Diagnostics* 12(9):2142, 2022 | Scoping review of 35 DL studies | 16 to 30,000+ samples, often ~1:10 imbalance | Reported accuracy 73.1%–98.5%; inconsistent metrics | Sets our evaluation rules: per-class recall, F1, AUC |

**Comparison.** Learned spectrogram features beat MFCC + classical classifiers when data is plentiful [2], [7].
Recurrent models are competitive on small, clean sets [1], [7]. Results on small or uncontrolled datasets are
optimistic and fall on larger, controlled data [4], [9], [10]. Pretraining is the most consistent remedy [3], [8].
Our Phase 2 results fit this picture: with 229 training recordings, the MFCC + SVM baseline is the most stable model.
