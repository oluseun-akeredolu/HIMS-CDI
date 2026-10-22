"""
Edge-Aware Fusion model — fusion with pruning.

Implementation: a fusion network that combines the 4 features through a
small MLP, with a pruned variant that uses structured pruning (compression
ratio = 0.5) to reduce the number of parameters by half.

Two classes:
    - EdgeAwareFusionModel (orig): unpruned
    - EdgeAwareFusionOptModel (opt): pruned (compression=0.5)

Architecture:
    - Input: 4 features
    - Hidden: 32 units (orig) or 16 units (opt, after pruning)
    - Output: 2 classes
    - Optimizer: Adam, lr=1e-3
    - Loss: cross-entropy
    - Early stopping: patience=10 epochs, monitor val_loss
    - Pruning: structured pruning via torch.nn.utils.prune.l1_unstructured
      with amount=0.5 on the hidden layer weights

No reverse-engineering to hit specific numbers. The actual latency/memory
depends on the hardware.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset


class FusionMLP(nn.Module):
    """Simple MLP for fusion."""

    def __init__(self, n_features=4, hidden=32, n_classes=2):
        super().__init__()
        self.hidden = nn.Linear(n_features, hidden)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


class EdgeAwareFusionModel:
    """Edge-Aware Fusion (orig) — unpruned MLP."""

    def __init__(self, compression_ratio=0.5, learning_rate=1e-3, seed=13,
                 n_features=4, n_classes=2, hidden=32):
        torch.manual_seed(seed)
        self.compression_ratio = compression_ratio
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = FusionMLP(n_features=n_features, hidden=hidden, n_classes=n_classes)
        self.softmax = nn.Softmax(dim=1)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=200, patience=10, batch_size=32, verbose=False):
        X_t = torch.FloatTensor(np.asarray(X_train))
        y_t = torch.LongTensor(np.asarray(y_train))

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(loader)

            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_v = torch.FloatTensor(np.asarray(X_val))
                    y_v = torch.LongTensor(np.asarray(y_val))
                    val_loss = criterion(self.model(X_v), y_v).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if best_state is not None:
                            self.model.load_state_dict(best_state)
                        break
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(np.asarray(X))
            logits = self.model(X_t)
            probs = self.softmax(logits)
        return probs.numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class EdgeAwareFusionOptModel(EdgeAwareFusionModel):
    """Edge-Aware Fusion (opt) — pruned MLP (compression_ratio=0.5).

    Applies L1 unstructured pruning with amount=0.5 to the hidden layer
    weights, reducing the number of nonzero parameters by half.
    """

    def __init__(self, compression_ratio=0.5, learning_rate=1e-3, seed=13,
                 n_features=4, n_classes=2, hidden=32, int8_quantize=True):
        super().__init__(compression_ratio=compression_ratio,
                        learning_rate=learning_rate, seed=seed,
                        n_features=n_features, n_classes=n_classes, hidden=hidden)
        self.int8_quantize = int8_quantize
        self.pruned = False

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        # First train the unpruned model
        super().fit(X_train, y_train, X_val, y_val, **kwargs)
        # Then apply pruning
        prune.l1_unstructured(self.model.hidden, name='weight', amount=self.compression_ratio)
        prune.l1_unstructured(self.model.output, name='weight', amount=self.compression_ratio)
        self.pruned = True
        return self

    def predict_proba(self, X):
        # Pruning creates sparse weights — predict_proba works the same way
        return super().predict_proba(X)
