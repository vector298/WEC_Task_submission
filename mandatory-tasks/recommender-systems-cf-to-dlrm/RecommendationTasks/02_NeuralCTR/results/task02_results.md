# Task 02 results

Selected: F  4d (32,) dropout 0.0 | params 43457 | weight memory 0.17 MB | train time 7 s | best epoch 15 | threshold 0.100

## Architecture sweep (mean over seeds 42, 43, 44; validation)

| config | params | best epoch | PR-AUC | PR std | AUC | AUC std | log loss |
|---|---|---|---|---|---|---|---|
| A  8d (256,128) dropout 0.2 [baseline] | 169153 | 3.7 | 0.0994 | 0.0042 | 0.7148 | 0.0245 | 0.1327 |
| B  4d (64,) dropout 0.2 | 47265 | 10.7 | 0.1025 | 0.0006 | 0.7189 | 0.0071 | 0.1316 |
| C  8d (128,64) dropout 0.3 | 116033 | 3.7 | 0.1000 | 0.0054 | 0.7228 | 0.0091 | 0.1318 |
| D  16d (256,128) dropout 0.5 | 301697 | 3.3 | 0.1017 | 0.0068 | 0.7213 | 0.0026 | 0.1318 |
| E  8d (512,256,128) dropout 0.3 | 357313 | 1.7 | 0.0980 | 0.0093 | 0.7257 | 0.0136 | 0.1313 |
| F  4d (32,) dropout 0.0 | 43457 | 13.3 | 0.1036 | 0.0075 | 0.7187 | 0.0077 | 0.1315 |

## Selected model, val and test (test: one run)

| split | ROC-AUC (95% CI) | PR-AUC (95% CI) | log loss (95% CI) | acc | F1 | precision | recall |
|---|---|---|---|---|---|---|---|
| val | 0.720 [0.688, 0.752] | 0.1014 [0.0790, 0.1396] | 0.1315 [0.1188, 0.1442] | 0.9416 | 0.1616 | 0.1495 | 0.1758 |
| test | 0.668 [0.635, 0.694] | 0.0647 [0.0540, 0.0814] | 0.1441 [0.1310, 0.1575] | 0.9415 | 0.0845 | 0.0888 | 0.0806 |