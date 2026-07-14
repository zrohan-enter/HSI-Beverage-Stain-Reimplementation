from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    selector: SelectKBest
    scaler: StandardScaler
    selected_indices: np.ndarray
    selected_scores: np.ndarray


def prepare_splits(
    X: np.ndarray,
    y: np.ndarray,
    selected_bands: int = 162,
    test_size: float = 0.20,
    validation_fraction_of_training: float = 0.10,
    seed: int = 42,
) -> PreparedData:
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}.")
    if len(X) != len(y):
        raise ValueError("X and y contain different sample counts.")
    if not 1 <= selected_bands <= X.shape[1]:
        raise ValueError(
            f"selected_bands must be within [1, {X.shape[1]}]."
        )

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_fraction_of_training,
        random_state=seed,
        stratify=y_train_full,
    )

    # Leakage prevention: ANOVA is fitted only on the effective training set.
    selector = SelectKBest(score_func=f_classif, k=selected_bands)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    selected_indices = selector.get_support(indices=True)
    all_scores = np.asarray(selector.scores_, dtype=np.float64)
    selected_scores = all_scores[selected_indices]

    # Leakage prevention: scaling is fitted only on the training samples.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    return PreparedData(
        X_train=np.asarray(X_train_scaled, dtype=np.float32),
        X_val=np.asarray(X_val_scaled, dtype=np.float32),
        X_test=np.asarray(X_test_scaled, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.int64),
        y_val=np.asarray(y_val, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
        selector=selector,
        scaler=scaler,
        selected_indices=np.asarray(selected_indices, dtype=np.int64),
        selected_scores=np.asarray(selected_scores, dtype=np.float64),
    )
