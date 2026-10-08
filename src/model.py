"""
model.py
========
Lightweight single-channel 1D CNN for 5-class sleep stage classification.

Architecture:
    Input (1, 3000)                                         30 s @ 100 Hz
    -> Conv1d(1->32,  k=50, stride=6)  -> BN -> ReLU        ~0.5 s filters
    -> MaxPool1d(8) -> Dropout(0.25)                        3000 -> 62
    -> Conv1d(32->64, k=7, same) -> BN -> ReLU
    -> Conv1d(64->64, k=7, same) -> BN -> ReLU
    -> MaxPool1d(4) -> Dropout(0.25)                        62 -> 15
    -> Conv1d(64->128, k=7, same) -> BN -> ReLU
    -> AdaptiveAvgPool1d(1)                                 Global Average Pooling
    -> Dropout(0.5) -> Linear(128, 5)

Design notes:
  - The wide, strided first layer (kernel = fs/2, stride = fs/16, as in the
    small-filter branch of DeepSleepNet) lets each filter see ~0.5 s of EEG,
    enough to capture delta waves, spindles and K-complexes, instead of the
    ~0.13 s receptive field of stacked stride-1 kernels.
  - Pooling between blocks grows the receptive field to ~17 s before Global
    Average Pooling and cuts compute to ~5 M multiply-accumulates per epoch,
    which is what matters for low-power / wearable deployment.
  - Softmax is omitted from forward() because CrossEntropyLoss applies
    log-softmax internally.
"""

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, kernel: int, stride: int = 1, padding="same"):
    return [
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    ]


class SleepCNN(nn.Module):
    """Lightweight 1D CNN for EEG sleep stage classification."""

    def __init__(self, n_classes: int = 5):
        super().__init__()

        self.features = nn.Sequential(
            *_conv_block(1, 32, kernel=50, stride=6, padding=22),
            nn.MaxPool1d(8),
            nn.Dropout(0.25),

            *_conv_block(32, 64, kernel=7),
            *_conv_block(64, 64, kernel=7),
            nn.MaxPool1d(4),
            nn.Dropout(0.25),

            *_conv_block(64, 128, kernel=7),
            nn.AdaptiveAvgPool1d(1),          # (B, 128, 1)
        )

        self.classifier = nn.Sequential(
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
        x = self.features(x).squeeze(-1)   # (B, 128)
        return self.classifier(x)          # (B, n_classes)


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


def count_macs(model: nn.Module, n_samples: int = 3000) -> int:
    """Multiply-accumulate operations for one epoch (Conv1d + Linear layers)."""
    macs = 0

    def conv_hook(module, _inp, out):
        nonlocal macs
        macs += out.numel() * (module.in_channels // module.groups) * module.kernel_size[0]

    def linear_hook(module, _inp, out):
        nonlocal macs
        macs += out.numel() * module.in_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, n_samples))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs


def load_checkpoint(path: str, device="cpu"):
    """Load a checkpoint saved by train.py; returns (model, checkpoint dict)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = SleepCNN(n_classes=checkpoint.get("n_classes", 5)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


if __name__ == "__main__":
    model = SleepCNN(n_classes=5)
    print(model)
    print(f"\nTrainable parameters : {count_parameters(model):,}")
    print(f"Model size           : {model_size_mb(model):.2f} MB")
    print(f"MACs per epoch       : {count_macs(model) / 1e6:.2f} M")

    dummy = torch.randn(4, 1, 3000)
    out = model(dummy)
    print(f"Input shape          : {dummy.shape}")
    print(f"Output shape         : {out.shape}")
