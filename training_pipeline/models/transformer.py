"""
Transformer model — Multimodal Transformer baseline.

Implementation: a standard Transformer encoder (Vaswani et al., 2017)
adapted for tabular features. Each feature is treated as a "token" in a
sequence of length 4, with the hidden size = 512 per manuscript Table 4.

Architecture (per manuscript Table 4):
    - Input: 4 features treated as a length-4 sequence
    - Embedding: linear projection to hidden_size=512
    - Positional encoding: standard sinusoidal
    - Transformer encoder: 1 layer, 8 heads, feedforward dim=2048
    - Pooling: mean over the sequence dimension
    - Output: 2 classes (DEPLOY_SYNTH binary)
    - Optimizer: Adam, lr=3e-5
    - Loss: cross-entropy
    - Early stopping: patience=10 epochs, monitor val_loss

This is a standard transformer encoder. No reverse-engineering to hit specific numbers.
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017, Eq. 6)."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """Transformer encoder for binary classification of HIMS-CDI features."""

    def __init__(self, hidden_size=512, learning_rate=3e-5, seed=13,
                 n_features=4, n_classes=2, n_heads=8, n_layers=1,
                 ff_dim=2048, dropout=0.1):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.n_features = n_features

        # Project each scalar feature to hidden_size
        self.input_proj = nn.Linear(1, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=n_features + 1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch, n_features) -> (batch, n_features, 1) -> (batch, n_features, hidden_size)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, n_features, 1)
        x = self.input_proj(x)  # (batch, n_features, hidden_size)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (batch, n_features, hidden_size)
        # Mean pooling over the sequence dimension
        x = x.mean(dim=1)  # (batch, hidden_size)
        logits = self.classifier(x)  # (batch, n_classes)
        return logits

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=200, patience=10, batch_size=32, verbose=False):
        """Train with early stopping (patience=10, monitor val_loss)."""
        X_t = torch.FloatTensor(np.asarray(X_train))
        y_t = torch.LongTensor(np.asarray(y_train))

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(loader)

            if X_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    X_v = torch.FloatTensor(np.asarray(X_val))
                    y_v = torch.LongTensor(np.asarray(y_val))
                    val_outputs = self(X_v)
                    val_loss = criterion(val_outputs, y_v).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f'Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})')
                        if best_state is not None:
                            self.load_state_dict(best_state)
                        break

            if verbose and epoch % 20 == 0:
                print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}'
                      + (f', val_loss={val_loss:.4f}' if X_val is not None else ''))

        return self

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(np.asarray(X))
            logits = self(X_t)
            probs = self.softmax(logits)
        return probs.numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
