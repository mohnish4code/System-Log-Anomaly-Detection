| Model | Precision | Recall | F1 | PR-AUC | False alarms (test) | False alarms / day |
|---|---|---|---|---|---|---|
| Keyword rule (ERROR/FATAL/fail/exception) | 1.000 | 0.487 | 0.655 | 0.495 | 0 | 0.0 |
| PCA | 0.739 | 0.857 | 0.793 | 0.735 | 509 | 3,429.9 |
| Isolation Forest | 0.352 | 0.865 | 0.501 | 0.764 | 2,673 | 18,011.8 |
| Logistic Regression (supervised) | 0.955 | 0.990 | 0.972 | 0.996 | 79 | 532.3 |
| Decision Tree (supervised) | 1.000 | 0.989 | 0.995 | 0.996 | 0 | 0.0 |
| LSTM (DeepLog-style) | 0.616 | 0.702 | 0.656 | 0.657 | 737 | 4,966.2 |
| Transformer | 0.552 | 0.715 | 0.623 | 0.671 | 973 | 6,556.5 |
