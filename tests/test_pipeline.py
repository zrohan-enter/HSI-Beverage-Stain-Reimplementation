from pathlib import Path

import numpy as np
import torch

from src.data.datasets import generate_synthetic_beverage
from src.features.preprocessing import prepare_splits
from src.models.networks import build_model


def test_synthetic_dataset_shape() -> None:
    dataset = generate_synthetic_beverage(
        samples_per_class=100,
        seed=42,
    )

    assert dataset.X.shape == (900, 204)
    assert dataset.y.shape == (900,)
    assert len(dataset.class_names) == 9
    assert dataset.X.dtype == np.float32
    assert dataset.y.dtype == np.int64


def test_leakage_safe_preprocessing() -> None:
    dataset = generate_synthetic_beverage(
        samples_per_class=100,
        seed=42,
    )

    prepared = prepare_splits(
        dataset.X,
        dataset.y,
        selected_bands=162,
        seed=42,
    )

    assert prepared.X_train.shape[1] == 162
    assert prepared.X_val.shape[1] == 162
    assert prepared.X_test.shape[1] == 162
    assert len(prepared.selected_indices) == 162

    assert np.allclose(
        prepared.X_train.mean(axis=0),
        0.0,
        atol=1e-5,
    )


def test_all_model_output_shapes() -> None:
    batch = torch.randn(4, 162)

    for model_name in ("mlp", "cnn1d", "lstm", "cnn_lstm"):
        model = build_model(
            model_name=model_name,
            input_bands=162,
            classes=9,
            dropout=0.30,
        )

        output = model(batch)

        assert output.shape == (4, 9)
        assert torch.isfinite(output).all()
