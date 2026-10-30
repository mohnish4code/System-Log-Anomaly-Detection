| Model | Precision | Recall | F1 | PR-AUC | False alarms (test) | False alarms / day |
|---|---|---|---|---|---|---|
| Keyword rule (ERROR/FATAL/fail/exception) | 0.209 | 1.000 | 0.346 | 0.489 | 515 | 10.6 |
| PCA | 0.219 | 0.993 | 0.359 | 0.521 | 481 | 9.9 |
| Isolation Forest | 0.870 | 0.441 | 0.585 | 0.757 | 9 | 0.2 |
| Logistic Regression (supervised) | 0.971 | 0.500 | 0.660 | 0.760 | 2 | 0.0 |
| Decision Tree (supervised) | 1.000 | 0.434 | 0.605 | 0.555 | 0 | 0.0 |
| LSTM (DeepLog-style) | 0.277 | 0.956 | 0.429 | 0.592 | 340 | 7.0 |
| Transformer | 0.280 | 0.934 | 0.431 | 0.594 | 326 | 6.7 |
