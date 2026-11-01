| Model | Selected setting | Test acc. | Test AUC | Sens. | Spec. | Macro F1 | 5-fold AUC | 5-fold acc. | Params |
|---|---|---|---|---|---|---|---|---|---|
| M1 MFCC + SVM | C=10.0, gamma=0.001 | 62.0% | 0.68 | 0.72 | 0.52 | 0.62 | 0.60 ± 0.06 | 59.4 ± 4.9% | 18,960 |
| M2 2D-CNN | avg pooling, lr=0.001 | 48.0% | 0.48 | 0.64 | 0.32 | 0.47 | 0.49 ± 0.05 | 47.8 ± 4.2% | 19,138 |
| M3 CNN–LSTM | lr=0.001, SpecAugment=off | 66.0% | 0.67 | 0.52 | 0.80 | 0.65 | 0.56 ± 0.07 | 53.0 ± 6.8% | 1,183,938 |
| M4 YAMNet + BiLSTM | bilstm head, lr=0.001 | 50.0% | 0.43 | 0.52 | 0.48 | 0.50 | 0.55 ± 0.04 | 54.6 ± 2.3% | 558,338 |
