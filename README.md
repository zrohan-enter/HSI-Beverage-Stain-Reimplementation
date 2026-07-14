# HSI Beverage-Stain Classification Reimplementation

A transparent engineering reimplementation of the workflow presented in the
Scientific Reports paper:

**A Hyperspectral Imaging Framework Integrating Band Selection and Deep
Learning for Beverage Stain Classification in Forensic Analysis**

## Status

The end-to-end pipeline is operational and GPU verified.

- Python 3.10
- PyTorch 2.11 with CUDA 12.8
- NVIDIA GeForce RTX 4070
- 204-band hyperspectral input
- ANOVA selection of 162 bands
- MLP, 1D-CNN, LSTM and CNN-LSTM
- automated metrics and result verification
- confusion matrices and training curves

## Scope

The original beverage-stain dataset is not publicly downloadable. This
repository therefore does not claim exact numerical reproduction.

Two validation paths are included:

1. A nine-class synthetic beverage dataset validates the paper-shaped
   software workflow.
2. The public Salinas dataset validates the workflow on genuine 204-band
   hyperspectral data.

Synthetic accuracy is not forensic evidence. Salinas accuracy is not
beverage-stain accuracy.

## Real Salinas HSI results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| MLP | 92.67% | 96.14% |
| 1D-CNN | 91.89% | 95.43% |
| CNN-LSTM | 86.80% | 92.21% |
| LSTM | 71.85% | 66.14% |

These are 15-epoch engineering-validation experiments.

## Pipeline

    204-band spectra
          |
    stratified train/validation/test split
          |
    training-only ANOVA fitting
          |
    162 selected spectral bands
          |
    training-only StandardScaler fitting
          |
    MLP / 1D-CNN / LSTM / CNN-LSTM
          |
    accuracy, macro metrics and confusion matrices

## Windows PowerShell setup

    Set-Location "D:\Academic\Research paper\HSI-Beverage-Stain-Reimplementation"
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
    & ".\.venv\Scripts\Activate.ps1"

## Verify the environment

    python scripts\verify_gpu.py
    python scripts\inspect_salinas.py
    python -m pytest -q

## Run an experiment

    python scripts\run_experiment.py --dataset salinas --model mlp --epochs 15 --batch-size 256 --selected-bands 162

## Verify generated results

    python scripts\verify_results.py

## Generate comparison reports

    python scripts\build_comparison.py

## Documentation

- `docs/METHODOLOGY.md`
- `docs/LIMITATIONS.md`
- `docs/FACULTY_DEMO.md`
- `docs/environment_report.txt`
- `docs/salinas_dataset_report.txt`
- `docs/four_model_verification.txt`

## Data policy

Raw datasets, model checkpoints, predictions and probability arrays are
excluded from Git. The repository stores code, configuration, documentation
and lightweight result summaries.

## Reproducibility boundary

The target paper used TensorFlow/Keras and a private dataset. This
implementation uses PyTorch for reliable native-Windows CUDA execution.
Architecture families and methodological stages are reconstructed, but exact
framework-level equivalence is not claimed.
