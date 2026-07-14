from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "salinas_real_hsi"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "paper_comparison"
DOCS_ROOT = PROJECT_ROOT / "docs"

RUNS = (
    ("MLP", "faculty_demo_mlp"),
    ("1D-CNN", "faculty_demo_cnn1d"),
    ("LSTM", "faculty_demo_lstm"),
    ("CNN-LSTM", "faculty_demo_cnn_lstm"),
)

PAPER_RESULTS = {
    "MLP": {
        "accuracy": 0.9558,
        "macro_precision": 0.9554,
        "macro_recall": 0.9555,
        "macro_f1": 0.9553,
    },
    "1D-CNN": {
        "accuracy": 0.9524,
        "macro_precision": 0.9519,
        "macro_recall": 0.9518,
        "macro_f1": 0.9517,
    },
    "LSTM": {
        "accuracy": 0.9451,
        "macro_precision": 0.9497,
        "macro_recall": 0.9497,
        "macro_f1": 0.9496,
    },
    "CNN-LSTM": {
        "accuracy": 0.9033,
        "macro_precision": 0.9049,
        "macro_recall": 0.9047,
        "macro_f1": 0.9046,
    },
}


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def make_local_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Model | Accuracy | Macro Precision | Macro Recall | "
        "Macro F1 | Time (s) | Best Epoch |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['model']} "
            f"| {percent(float(row['accuracy']))} "
            f"| {percent(float(row['macro_precision']))} "
            f"| {percent(float(row['macro_recall']))} "
            f"| {percent(float(row['macro_f1']))} "
            f"| {float(row['training_seconds']):.2f} "
            f"| {int(row['best_epoch'])} |"
        )

    return "\n".join(lines)


def make_paper_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['model']} "
            f"| {percent(float(row['accuracy']))} "
            f"| {percent(float(row['macro_precision']))} "
            f"| {percent(float(row['macro_recall']))} "
            f"| {percent(float(row['macro_f1']))} |"
        )

    return "\n".join(lines)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)

    local_rows: list[dict[str, object]] = []

    for model_name, run_name in RUNS:
        run_directory = RESULT_ROOT / run_name
        metrics_path = run_directory / "metrics.json"
        metadata_path = run_directory / "run_metadata.json"

        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics: {metrics_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata: {metadata_path}")

        metrics = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        training = metadata["training"]

        if metadata["device"] != "cuda:0":
            raise ValueError(
                f"{model_name} was not recorded as a CUDA experiment."
            )

        local_rows.append(
            {
                "model": model_name,
                "accuracy": float(metrics["accuracy"]),
                "macro_precision": float(
                    metrics["macro_precision"]
                ),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
                "training_seconds": float(
                    training["elapsed_seconds"]
                ),
                "best_epoch": int(training["best_epoch"]),
                "parameters": int(
                    metadata["trainable_parameters"]
                ),
                "device": metadata["device"],
                "gpu": metadata["gpu"],
            }
        )

    paper_rows = [
        {"model": model_name, **metrics}
        for model_name, metrics in PAPER_RESULTS.items()
    ]

    local_frame = pd.DataFrame(local_rows)
    paper_frame = pd.DataFrame(paper_rows)

    local_frame.to_csv(
        OUTPUT_ROOT / "salinas_model_comparison.csv",
        index=False,
    )

    paper_frame.to_csv(
        OUTPUT_ROOT / "paper_reported_metrics.csv",
        index=False,
    )

    context_frame = local_frame[
        [
            "model",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        ]
    ].rename(
        columns={
            "accuracy": "salinas_accuracy",
            "macro_precision": "salinas_macro_precision",
            "macro_recall": "salinas_macro_recall",
            "macro_f1": "salinas_macro_f1",
        }
    )

    paper_context = paper_frame.rename(
        columns={
            "accuracy": "paper_beverage_accuracy",
            "macro_precision": "paper_beverage_macro_precision",
            "macro_recall": "paper_beverage_macro_recall",
            "macro_f1": "paper_beverage_macro_f1",
        }
    )

    context_frame = context_frame.merge(
        paper_context,
        on="model",
        how="left",
    )

    context_frame["comparison_warning"] = (
        "Different datasets and class definitions; "
        "not a direct numerical reproduction."
    )

    context_frame.to_csv(
        OUTPUT_ROOT / "paper_vs_local_context.csv",
        index=False,
    )

    positions = np.arange(len(local_frame))
    width = 0.36

    plt.figure(figsize=(10, 6))
    plt.bar(
        positions - width / 2,
        local_frame["accuracy"] * 100,
        width=width,
        label="Accuracy",
    )
    plt.bar(
        positions + width / 2,
        local_frame["macro_f1"] * 100,
        width=width,
        label="Macro F1",
    )
    plt.xticks(positions, local_frame["model"])
    plt.ylabel("Score (%)")
    plt.ylim(0, 100)
    plt.title("Salinas Real-HSI Four-Model Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_ROOT / "salinas_model_comparison.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(
        local_frame["model"],
        local_frame["training_seconds"],
    )
    plt.ylabel("Training time (seconds)")
    plt.title("RTX 4070 Training Time for 15 Epochs")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_ROOT / "gpu_training_time_comparison.png",
        dpi=200,
    )
    plt.close()

    local_table = make_local_table(local_rows)
    paper_table = make_paper_table(paper_rows)

    comparison_document = f"""# Model Comparison

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

{local_table}

## Target-paper reported results

Dataset: private nine-class beverage-stain hyperspectral dataset.

{paper_table}

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
"""

    (OUTPUT_ROOT / "README.md").write_text(
        comparison_document,
        encoding="utf-8",
    )

    faculty_document = f"""# Faculty Demonstration Guide

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

{local_table}

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

    Set-Location "D:\\Academic\\Research paper\\HSI-Beverage-Stain-Reimplementation"
    & ".\\.venv\\Scripts\\Activate.ps1"
    python scripts\\verify_gpu.py
    python scripts\\inspect_salinas.py
    python -m pytest -q
    python scripts\\verify_results.py
    explorer.exe results\\paper_comparison

## Recommended explanation

The original target dataset is private, so exact numerical reproduction is
not claimed. The project separates validation into a synthetic nine-beverage
software test and a real Salinas hyperspectral experiment. All feature
selection and scaling operations are fitted only on training data.
"""

    (DOCS_ROOT / "FACULTY_DEMO.md").write_text(
        faculty_document,
        encoding="utf-8",
    )

    print("=" * 72)
    print("COMPARISON REPORT GENERATION COMPLETED")
    print("=" * 72)

    for row in local_rows:
        print(
            f"{row['model']:10s} | "
            f"accuracy={float(row['accuracy']):.6f} | "
            f"macro_f1={float(row['macro_f1']):.6f} | "
            f"time={float(row['training_seconds']):.2f}s"
        )

    print("=" * 72)
    print(f"Comparison directory: {OUTPUT_ROOT}")
    print(f"Faculty guide: {DOCS_ROOT / 'FACULTY_DEMO.md'}")


if __name__ == "__main__":
    main()
