"""A scikit-learn compatible PyTorch binary classifier for tabular data."""

from __future__ import annotations

import random

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """PyTorch MLP with early stopping and scikit-learn-style probabilities."""

    def __init__(
        self,
        hidden_layer_sizes=(128,),
        alpha=1e-4,
        learning_rate_init=1e-3,
        batch_size=256,
        epochs=100,
        patience=12,
        random_state=42,
        device="cpu",
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def _set_seed(self):
        import torch

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _build_network(self, input_size):
        import torch.nn as nn

        layers = []
        previous_size = input_size
        for hidden_size in self.hidden_layer_sizes:
            layers.extend([nn.Linear(previous_size, hidden_size), nn.ReLU()])
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        self._set_seed()
        X_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.classes_ = np.array([0, 1])

        generator = torch.Generator().manual_seed(self.random_state)
        dataset = TensorDataset(torch.from_numpy(X_array), torch.from_numpy(y_array))
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )

        self.model_ = self._build_network(X_array.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate_init,
            weight_decay=self.alpha,
        )
        loss_function = nn.BCEWithLogitsLoss()
        best_loss = float("inf")
        best_state = None
        waiting = 0

        for _ in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                loss = loss_function(self.model_(features), labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(features)

            epoch_loss /= len(dataset)
            if epoch_loss < best_loss - 1e-5:
                best_loss = epoch_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }
                waiting = 0
            else:
                waiting += 1
                if waiting >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        import torch

        X_array = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X_array).to(self.device))
            positive_probability = torch.sigmoid(logits).cpu().numpy().ravel()
        return np.column_stack([1.0 - positive_probability, positive_probability])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
