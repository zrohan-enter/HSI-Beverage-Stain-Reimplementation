# Faculty Demonstration Guide

## Objective

Reimplement the paper's main engineering workflow:

1. load 204-band hyperspectral spectra;
2. split labelled pixels into training, validation and testing sets;
3. fit ANOVA band selection on training data only;
4. retain 162 spectral bands;
5. fit StandardScaler on training data only;
6. train MLP, 1D-CNN, LSTM and CNN-LSTM models;
7. produce metrics, confusion matrices and learning curves.

## Environment

- Python 3.10
- PyTorch 2.11
- CUDA runtime 12.8
- NVIDIA GeForce RTX 4070
- Native Windows execution

## Verified real-HSI results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Time (s) | Best Epoch |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 92.67% | 96.13% | 96.18% | 96.14% | 17.44 | 15 |
| 1D-CNN | 91.89% | 95.53% | 95.40% | 95.43% | 19.83 | 15 |
| LSTM | 71.85% | 66.43% | 69.91% | 66.14% | 19.61 | 15 |
| CNN-LSTM | 86.80% | 91.76% | 92.90% | 92.21% | 17.25 | 13 |

## Reproduced components

- 204-band input structure
- ANOVA-based feature selection
- 162 retained bands
- training-only feature fitting
- standardized spectral features
- four neural-network architecture families
- Adam optimization
- learning-rate reduction
- early-stopping support
- macro-averaged evaluation metrics
- learning curves and confusion matrices

## Important limitations

- The paper's beverage-stain dataset is private.
- Salinas contains 16 agricultural classes rather than nine beverages.
- Synthetic beverage spectra are used only for software validation.
- The paper used TensorFlow/Keras; this project uses PyTorch CUDA.
- The faculty-demo experiments use 15 epochs.
- Exact unpublished implementation details may differ.

## Windows PowerShell demonstration commands

    Set-Location "D:\Academic\Research paper\HSI-Beverage-Stain-Reimplementation"
    & ".\.venv\Scripts\Activate.ps1"
    python scripts\verify_gpu.py
    python scripts\inspect_salinas.py
    python -m pytest -q
    python scripts\verify_results.py
    explorer.exe results\paper_comparison

## Recommended explanation

The original target dataset is private, so exact numerical reproduction is
not claimed. The project separates validation into a synthetic nine-beverage
software test and a real Salinas hyperspectral experiment. All feature
selection and scaling operations are fitted only on training data.
