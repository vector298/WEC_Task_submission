# Part C results (val)

| config | threshold | val F1 | precision | recall |
|---|---|---|---|---|
| ResNet50 raw | 0.79 | 0.6917 | 0.938 | 0.626 |
| ResNet50 centred | 0.735 | 0.69 | 0.95 | 0.615 |
| CLIP raw | 0.83 | 0.6616 | 0.918 | 0.602 |
| CLIP centred | 0.635 | 0.6765 | 0.894 | 0.641 |

Floor (val): 0.4629. Floor (test): 0.4401.
Selected on val: ResNet50 raw, threshold 0.79. Test F1 (one run): 0.6800