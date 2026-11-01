| Model | Setting | Inner-val AUC | Outer 5-fold AUC | Selected |
|---|---|---|---|---|
| M1 MFCC + SVM | C=0.1, gamma=scale | 0.395 | 0.463 ± 0.091 |  |
| M1 MFCC + SVM | C=0.1, gamma=0.001 | 0.413 | 0.390 ± 0.046 |  |
| M1 MFCC + SVM | C=0.1, gamma=0.01 | 0.408 | 0.465 ± 0.092 |  |
| M1 MFCC + SVM | C=1.0, gamma=scale | 0.613 | 0.628 ± 0.050 |  |
| M1 MFCC + SVM | C=1.0, gamma=0.001 | 0.598 | 0.534 ± 0.129 |  |
| M1 MFCC + SVM | C=1.0, gamma=0.01 | 0.606 | 0.625 ± 0.046 |  |
| M1 MFCC + SVM | C=10.0, gamma=scale | 0.615 | 0.637 ± 0.083 |  |
| M1 MFCC + SVM | C=10.0, gamma=0.001 | 0.619 | 0.605 ± 0.057 | yes |
| M1 MFCC + SVM | C=10.0, gamma=0.01 | 0.613 | 0.641 ± 0.077 |  |
| M2 2D-CNN | avg pooling, lr=0.001 | 0.587 | 0.491 ± 0.047 | yes |
| M2 2D-CNN | max pooling, lr=0.001 | 0.573 | 0.508 ± 0.062 |  |
| M2 2D-CNN | max pooling, lr=0.0003 | 0.519 | 0.505 ± 0.078 |  |
| M3 CNN–LSTM | lr=0.001, SpecAugment=off | 0.705 | 0.561 ± 0.074 | yes |
| M3 CNN–LSTM | lr=0.0003, SpecAugment=off | 0.682 | 0.574 ± 0.049 |  |
| M3 CNN–LSTM | lr=0.0003, SpecAugment=on | 0.664 | 0.523 ± 0.066 |  |
| M4 YAMNet + BiLSTM | bilstm head, lr=0.001 | 0.602 | 0.555 ± 0.037 | yes |
| M4 YAMNet + BiLSTM | bilstm head, lr=0.0003 | 0.587 | 0.547 ± 0.058 |  |
| M4 YAMNet + BiLSTM | mean head, lr=0.001 | 0.554 | 0.570 ± 0.059 |  |
| M4 YAMNet + BiLSTM | mean head, lr=0.0003 | 0.556 | 0.581 ± 0.082 |  |
| M4 YAMNet + BiLSTM | max head, lr=0.001 | 0.543 | 0.537 ± 0.062 |  |
| M4 YAMNet + BiLSTM | max head, lr=0.0003 | 0.545 | 0.548 ± 0.067 |  |
