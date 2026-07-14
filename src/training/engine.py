from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingHistory:
    epoch: list[int]
    train_loss: list[float]
    train_accuracy: list[float]
    val_loss: list[float]
    val_accuracy: list[float]
    learning_rate: list[float]
    best_epoch: int
    elapsed_seconds: float


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int = 0,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(features)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            current_batch = labels.shape[0]
            total_loss += float(loss.item()) * current_batch
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_samples += current_batch

    return total_loss / total_samples, total_correct / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience: int,
    lr_patience: int,
    lr_factor: float,
) -> TrainingHistory:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = TrainingHistory(
        epoch=[],
        train_loss=[],
        train_accuracy=[],
        val_loss=[],
        val_accuracy=[],
        learning_rate=[],
        best_epoch=0,
        elapsed_seconds=0.0,
    )

    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = _run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        scheduler.step(val_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.epoch.append(epoch)
        history.train_loss.append(train_loss)
        history.train_accuracy.append(train_accuracy)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_accuracy)
        history.learning_rate.append(current_lr)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f} acc {val_accuracy:.4f} | "
            f"lr {current_lr:.2e}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best validation loss was at epoch {best_epoch}."
            )
            break

    model.load_state_dict(best_state)
    history.best_epoch = best_epoch
    history.elapsed_seconds = time.perf_counter() - started
    return history


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    with torch.inference_mode():
        for features, _ in loader:
            features = features.to(device, non_blocking=True)
            logits = model(features)
            batch_probabilities = torch.softmax(logits, dim=1)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            probabilities.append(batch_probabilities.cpu().numpy())

    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(probabilities, axis=0),
    )
