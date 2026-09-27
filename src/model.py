"""
model.py
========
Lightweight 1D CNN for 5-class sleep stage classification.

Architecture (174,597 trainable parameters, ~0.67 MB):
    Input (1, 3000)
    -> Conv1d(1->64,  k=7, pad=3)  -> BatchNorm1d(64)  -> ReLU
    -> Conv1d(64->128, k=5, pad=2) -> BatchNorm1d(128) -> ReLU
    -> Conv1d(128->256, k=3, pad=1) -> BatchNorm1d(256) -> ReLU
    -> MaxPool1d(k=2)              — halves temporal dim: 3000 -> 1500
    -> AdaptiveAvgPool1d(1)        — Global Average Pooling: 1500 -> 1
    -> Linear(256, 128) -> ReLU -> Dropout(0.5)
    -> Linear(128, 5)

Design notes:
  - "Same" padding on Conv1d layers preserves the temporal dimension
    through the convolutional blocks (3000 throughout).
  - Global Average Pooling collapses (B, 256, 1500) -> (B, 256, 1),
    avoiding a 384K-dim Flatten that would inflate the model to ~49M
    parameters. This keeps the model truly lightweight.
  - Softmax is omitted from forward() because PyTorch's CrossEntropyLoss
    applies log-softmax internally.
"""

import torch
import torch.nn as nn


class SleepCNN(nn.Module):
    """Lightweight 1D CNN for EEG sleep stage classification."""

    def __init__(self, n_classes: int = 5):
        super().__init__()

        # ----- Feature extractor -----
        self.features = nn.Sequential(
            # Block 1: Conv1d 64 filters, kernel 7
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # Block 2: Conv1d 128 filters, kernel 5
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Block 3: Conv1d 256 filters, kernel 3
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            # Pooling: MaxPool then Global Average Pool
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(1),          # (B, 256, 1)
        )

        # ----- Classifier -----
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, 1, 3000)
            Single-channel 30-s EEG epoch at 100 Hz.

        Returns
        -------
        logits : Tensor of shape (batch, n_classes)
        """
        x = self.features(x)       # (B, 256, 1)
        x = x.squeeze(-1)          # (B, 256)
        x = self.classifier(x)     # (B, n_classes)
        return x


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Approximate model size in MB (parameters only)."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 * 1024)


if __name__ == "__main__":
    # Quick sanity check
    model = SleepCNN(n_classes=5)
    print(model)
    print(f"\nTrainable parameters : {count_parameters(model):,}")
    print(f"Model size           : {model_size_mb(model):.2f} MB")

    # Test forward pass
    dummy = torch.randn(4, 1, 3000)
    out = model(dummy)
    print(f"Input shape          : {dummy.shape}")
    print(f"Output shape         : {out.shape}")
