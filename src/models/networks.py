from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, input_bands: int, classes: int, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_bands, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CNN1DClassifier(nn.Module):
    def __init__(self, input_bands: int, classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x.unsqueeze(1)))


class LSTMClassifier(nn.Module):
    def __init__(self, input_bands: int, classes: int, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.unsqueeze(-1)
        output, _ = self.lstm(sequence)
        return self.classifier(output[:, -1, :])


class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_bands: int, classes: int, dropout: float = 0.3):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.convolution(x.unsqueeze(1)).transpose(1, 2)
        output, _ = self.lstm(features)
        return self.classifier(output[:, -1, :])


def build_model(
    model_name: str,
    input_bands: int,
    classes: int,
    dropout: float = 0.3,
) -> nn.Module:
    normalized = model_name.lower().strip()
    factories = {
        "mlp": MLPClassifier,
        "cnn1d": CNN1DClassifier,
        "lstm": LSTMClassifier,
        "cnn_lstm": CNNLSTMClassifier,
    }
    if normalized not in factories:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {sorted(factories)}."
        )
    return factories[normalized](input_bands, classes, dropout)
