from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "salinas_real_hsi"
RUNS = {
    "MLP": "faculty_demo_mlp",
    "1D-CNN": "faculty_demo_cnn1d",
    "LSTM": "faculty_demo_lstm",
    "CNN-LSTM": "faculty_demo_cnn_lstm",
}
def main() -> None:
    print("=" * 78)
    print("FOUR-MODEL RESULT INTEGRITY VERIFICATION")
    print("=" * 78)
    for model_name, run_name in RUNS.items():
        run_directory = RESULT_ROOT / run_name
        required_files = (
            run_directory / "metrics.json",
            run_directory / "run_metadata.json",
            run_directory / "predictions.npy",
            run_directory / "test_labels.npy",
            run_directory / "classification_report.csv",
            run_directory / "confusion_matrix.csv",
        )
        for required_file in required_files:
            if not required_file.exists():
                raise FileNotFoundError(
                    f"Missing result file: {required_file}"
                )
        saved_metrics = json.loads(
            (run_directory / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        metadata = json.loads(
            (run_directory / "run_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        labels = np.load(run_directory / "test_labels.npy")
        predictions = np.load(run_directory / "predictions.npy")
        if labels.shape != predictions.shape:
            raise ValueError(
                f"{model_name}: prediction and label shapes differ."
            )
        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        )
        calculated_metrics = {
            "accuracy": float(
                accuracy_score(labels, predictions)
            ),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
        }
        for metric_name, calculated_value in calculated_metrics.items():
            saved_value = float(saved_metrics[metric_name])
            if not np.isclose(
                saved_value,
                calculated_value,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    f"{model_name}: {metric_name} mismatch: "
                    f"{saved_value} != {calculated_value}"
                )
        if metadata["device"] != "cuda:0":
            raise ValueError(
                f"{model_name} was not recorded as a CUDA run."
            )
        if metadata["gpu"] != "NVIDIA GeForce RTX 4070":
            raise ValueError(
                f"{model_name} has an unexpected GPU record."
            )
        if int(metadata["original_bands"]) != 204:
            raise ValueError(
                f"{model_name} did not begin with 204 bands."
            )
        if int(metadata["selected_bands"]) != 162:
            raise ValueError(
                f"{model_name} did not use 162 selected bands."
            )
        print(f"\nModel:           {model_name}")
        print(f"Device:          {metadata['device']}")
        print(f"GPU:             {metadata['gpu']}")
        print(f"Test samples:    {len(labels):,}")
        print(
            f"Accuracy:        "
            f"{calculated_metrics['accuracy']:.6f}"
        )
        print(
            f"Macro precision: "
            f"{calculated_metrics['macro_precision']:.6f}"
        )
        print(
            f"Macro recall:    "
            f"{calculated_metrics['macro_recall']:.6f}"
        )
        print(
            f"Macro F1:        "
            f"{calculated_metrics['macro_f1']:.6f}"
        )
        print("STATUS: VERIFIED")
    print("\n" + "=" * 78)
    print("ALL FOUR GPU RESULTS VERIFIED SUCCESSFULLY")
    print("=" * 78)
if __name__ == "__main__":
    main()
