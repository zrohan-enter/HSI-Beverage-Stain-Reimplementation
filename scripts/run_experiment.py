from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.datasets import (  # noqa: E402
    generate_synthetic_beverage,
    load_salinas,
)
from src.evaluation.reporting import (  # noqa: E402
    save_classification_outputs,
    save_confusion_matrix,
    save_run_metadata,
    save_selected_bands,
    save_training_history,
)
from src.features.preprocessing import prepare_splits  # noqa: E402
from src.models.networks import build_model  # noqa: E402
from src.training.engine import (  # noqa: E402
    make_loader,
    predict,
    set_reproducible_seed,
    train_model,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the hyperspectral band-selection and deep-learning pipeline."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=("salinas", "synthetic"),
        default="salinas",
    )
    parser.add_argument(
        "--model",
        choices=("mlp", "cnn1d", "lstm", "cnn_lstm"),
        default="mlp",
    )
    parser.add_argument("--selected-bands", type=int, default=162)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--reduce-lr-patience", type=int, default=2)
    parser.add_argument("--reduce-lr-factor", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=2500)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution even when CUDA is available.",
    )

    return parser.parse_args()


def select_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


def load_dataset(arguments: argparse.Namespace):
    if arguments.dataset == "salinas":
        return load_salinas(PROJECT_ROOT)

    return generate_synthetic_beverage(
        samples_per_class=arguments.samples_per_class,
        seed=arguments.seed,
    )


def create_output_directory(
    dataset_name: str,
    model_name: str,
    run_name: str | None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    folder_name = (
        run_name.strip()
        if run_name and run_name.strip()
        else f"{model_name}_{timestamp}"
    )

    output_directory = (
        PROJECT_ROOT
        / "results"
        / dataset_name
        / folder_name
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "checkpoints").mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


def save_summary(
    output_directory: Path,
    dataset_name: str,
    model_name: str,
    device: torch.device,
    metrics: dict[str, float],
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    parameters: int,
) -> None:
    summary = f"""# Experiment Summary

## Experiment

- Dataset: `{dataset_name}`
- Model: `{model_name}`
- Device: `{device}`
- Train samples: {train_samples:,}
- Validation samples: {validation_samples:,}
- Test samples: {test_samples:,}
- Trainable parameters: {parameters:,}

## Test Results

- Accuracy: {metrics["accuracy"]:.6f}
- Macro precision: {metrics["macro_precision"]:.6f}
- Macro recall: {metrics["macro_recall"]:.6f}
- Macro F1-score: {metrics["macro_f1"]:.6f}

## Interpretation

For `salinas_real_hsi`, these values demonstrate that the pipeline
works on a genuine public hyperspectral dataset. They are not beverage-stain
reproduction results.

For `synthetic_beverage`, the values demonstrate software and architecture
functionality only. They are not valid forensic-performance claims.
"""

    (output_directory / "summary.md").write_text(
        summary,
        encoding="utf-8",
    )


def main() -> None:
    arguments = parse_arguments()
    set_reproducible_seed(arguments.seed)

    device = select_device(arguments.cpu)
    dataset = load_dataset(arguments)

    print("=" * 72)
    print("HSI BAND-SELECTION AND DEEP-LEARNING EXPERIMENT")
    print("=" * 72)
    print(f"Dataset:          {dataset.name}")
    print(f"Samples:          {len(dataset.y):,}")
    print(f"Original bands:   {dataset.X.shape[1]}")
    print(f"Classes:          {len(dataset.class_names)}")
    print(f"Selected bands:   {arguments.selected_bands}")
    print(f"Model:            {arguments.model}")
    print(f"Device:           {device}")

    if device.type == "cuda":
        print(f"GPU:              {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime:     {torch.version.cuda}")

    print("=" * 72)

    prepared = prepare_splits(
        dataset.X,
        dataset.y,
        selected_bands=arguments.selected_bands,
        test_size=0.20,
        validation_fraction_of_training=0.10,
        seed=arguments.seed,
    )

    print(f"Training samples:   {len(prepared.y_train):,}")
    print(f"Validation samples: {len(prepared.y_val):,}")
    print(f"Test samples:       {len(prepared.y_test):,}")

    train_loader = make_loader(
        prepared.X_train,
        prepared.y_train,
        batch_size=arguments.batch_size,
        shuffle=True,
        seed=arguments.seed,
    )

    validation_loader = make_loader(
        prepared.X_val,
        prepared.y_val,
        batch_size=arguments.batch_size,
        shuffle=False,
        seed=arguments.seed,
    )

    test_loader = make_loader(
        prepared.X_test,
        prepared.y_test,
        batch_size=arguments.batch_size,
        shuffle=False,
        seed=arguments.seed,
    )

    model = build_model(
        model_name=arguments.model,
        input_bands=prepared.X_train.shape[1],
        classes=len(dataset.class_names),
        dropout=arguments.dropout,
    ).to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable parameters: {trainable_parameters:,}")

    output_directory = create_output_directory(
        dataset_name=dataset.name,
        model_name=arguments.model,
        run_name=arguments.run_name,
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        device=device,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        patience=arguments.early_stopping_patience,
        lr_patience=arguments.reduce_lr_patience,
        lr_factor=arguments.reduce_lr_factor,
    )

    predictions, probabilities = predict(
        model=model,
        loader=test_loader,
        device=device,
    )

    metrics = save_classification_outputs(
        y_true=prepared.y_test,
        y_pred=predictions,
        class_names=dataset.class_names,
        output_dir=output_directory,
    )

    save_confusion_matrix(
        y_true=prepared.y_test,
        y_pred=predictions,
        class_names=dataset.class_names,
        output_dir=output_directory,
    )

    save_training_history(
        history=history,
        output_dir=output_directory,
    )

    save_selected_bands(
        indices=prepared.selected_indices,
        scores=prepared.selected_scores,
        output_dir=output_directory,
    )

    metadata = {
        "dataset": dataset.name,
        "dataset_metadata": dataset.metadata,
        "model": arguments.model,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "cuda_runtime": torch.version.cuda,
        "selected_bands": arguments.selected_bands,
        "original_bands": int(dataset.X.shape[1]),
        "classes": len(dataset.class_names),
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "dropout": arguments.dropout,
        "seed": arguments.seed,
        "train_samples": len(prepared.y_train),
        "validation_samples": len(prepared.y_val),
        "test_samples": len(prepared.y_test),
        "trainable_parameters": trainable_parameters,
    }

    save_run_metadata(
        metadata=metadata,
        history=history,
        output_dir=output_directory,
    )

    np.save(
        output_directory / "predictions.npy",
        predictions,
    )
    np.save(
        output_directory / "probabilities.npy",
        probabilities,
    )
    np.save(
        output_directory / "test_labels.npy",
        prepared.y_test,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": arguments.model,
            "input_bands": prepared.X_train.shape[1],
            "classes": len(dataset.class_names),
            "selected_indices": prepared.selected_indices,
            "class_names": list(dataset.class_names),
        },
        output_directory / "checkpoints" / "best_model.pt",
    )

    with (output_directory / "class_names.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(list(dataset.class_names), file, indent=2)

    save_summary(
        output_directory=output_directory,
        dataset_name=dataset.name,
        model_name=arguments.model,
        device=device,
        metrics=metrics,
        train_samples=len(prepared.y_train),
        validation_samples=len(prepared.y_val),
        test_samples=len(prepared.y_test),
        parameters=trainable_parameters,
    )

    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETED")
    print("=" * 72)
    print(f"Accuracy:         {metrics['accuracy']:.6f}")
    print(f"Macro precision:  {metrics['macro_precision']:.6f}")
    print(f"Macro recall:     {metrics['macro_recall']:.6f}")
    print(f"Macro F1-score:   {metrics['macro_f1']:.6f}")
    print(f"Best epoch:       {history.best_epoch}")
    print(f"Training time:    {history.elapsed_seconds:.2f} seconds")
    print(f"Results directory:{output_directory}")
    print("=" * 72)


if __name__ == "__main__":
    main()
