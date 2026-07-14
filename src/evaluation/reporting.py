from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.training.engine import TrainingHistory


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def save_training_history(history: TrainingHistory, output_dir: Path) -> None:
    frame = pd.DataFrame(
        {
            "epoch": history.epoch,
            "train_loss": history.train_loss,
            "train_accuracy": history.train_accuracy,
            "val_loss": history.val_loss,
            "val_accuracy": history.val_accuracy,
            "learning_rate": history.learning_rate,
        }
    )
    frame.to_csv(output_dir / "training_history.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(history.epoch, history.train_loss, label="Train loss")
    plt.plot(history.epoch, history.val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.epoch, history.train_accuracy, label="Train accuracy")
    plt.plot(history.epoch, history.val_accuracy, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=180)
    plt.close()


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    ).to_csv(output_dir / "confusion_matrix.csv")

    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure_size = max(8, min(15, len(class_names) * 0.75))
    plt.figure(figsize=(figure_size, figure_size))
    image = plt.imshow(normalized, interpolation="nearest", cmap="Blues")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(
        np.arange(len(class_names)),
        class_names,
        rotation=90,
        fontsize=8,
    )
    plt.yticks(np.arange(len(class_names)), class_names, fontsize=8)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Normalized confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def save_classification_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path,
) -> dict[str, float]:
    metrics = calculate_metrics(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        output_dir / "classification_report.csv"
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics


def save_selected_bands(
    indices: np.ndarray,
    scores: np.ndarray,
    output_dir: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "selected_rank": np.arange(1, len(indices) + 1),
            "zero_based_band_index": indices,
            "one_based_band_number": indices + 1,
            "anova_f_score": scores,
        }
    ).sort_values("anova_f_score", ascending=False)
    frame.to_csv(output_dir / "selected_bands.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.scatter(indices + 1, scores, s=14)
    plt.xlabel("Original band number")
    plt.ylabel("ANOVA F-score")
    plt.title("Selected spectral bands")
    plt.tight_layout()
    plt.savefig(output_dir / "selected_bands.png", dpi=180)
    plt.close()


def save_run_metadata(
    metadata: dict[str, object],
    history: TrainingHistory,
    output_dir: Path,
) -> None:
    serializable = dict(metadata)
    serializable["training"] = {
        "best_epoch": history.best_epoch,
        "elapsed_seconds": history.elapsed_seconds,
        "epochs_completed": len(history.epoch),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2, default=str)
