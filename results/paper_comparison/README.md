# Model Comparison

## Local real-HSI engineering validation

Dataset: Salinas corrected hyperspectral scene.

- 54,129 labelled pixels
- 16 agricultural classes
- 204 original bands
- 162 ANOVA-selected bands
- stratified 72/8/20 effective train/validation/test split
- ANOVA and StandardScaler fitted on training samples only
- NVIDIA GeForce RTX 4070 CUDA execution
- 15 training epochs

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Time (s) | Best Epoch |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 92.67% | 96.13% | 96.18% | 96.14% | 17.44 | 15 |
| 1D-CNN | 91.89% | 95.53% | 95.40% | 95.43% | 19.83 | 15 |
| LSTM | 71.85% | 66.43% | 69.91% | 66.14% | 19.61 | 15 |
| CNN-LSTM | 86.80% | 91.76% | 92.90% | 92.21% | 17.25 | 13 |

## Target-paper reported results

Dataset: private nine-class beverage-stain hyperspectral dataset.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| MLP | 95.58% | 95.54% | 95.55% | 95.53% |
| 1D-CNN | 95.24% | 95.19% | 95.18% | 95.17% |
| LSTM | 94.51% | 94.97% | 94.97% | 94.96% |
| CNN-LSTM | 90.33% | 90.49% | 90.47% | 90.46% |

## Interpretation boundary

These tables are contextual rather than a direct numerical comparison.
The datasets, classes, acquisition settings, sample structures, software
frameworks and training durations differ.

The Salinas results demonstrate that the reconstructed workflow operates on
genuine 204-band hyperspectral data. They are not beverage-stain results.

The synthetic beverage experiment verifies the nine-class software path.
Its accuracy is not forensic evidence.

## Local model ranking

1. MLP
2. 1D-CNN
3. CNN-LSTM
4. LSTM
