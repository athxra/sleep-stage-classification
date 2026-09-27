"""
train.py
========
Train the lightweight 1D CNN on the preprocessed Sleep-EDF data using
subject-independent train / val splits.

Features:
  - Class-weighted CrossEntropyLoss (weights from training data only).
  - Best model selected by validation macro-F1.
  - Saves best checkpoint to models/best_model.pth.

Usage:
    python src/train.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import SleepCNN, count_parameters, model_size_mb


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
N_CLASSES = 5

INT_TO_CLASS = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_split_data():
    """Load preprocessed data and split indices."""

    print("Loading preprocessed data...")
    X = np.load(os.path.join(PROC_DIR, "X_epochs.npy"))       # (N, 3000)
    y = np.load(os.path.join(PROC_DIR, "y_labels.npy"))        # (N,)

    train_idx = np.load(os.path.join(PROC_DIR, "train_indices.npy"))
    val_idx = np.load(os.path.join(PROC_DIR, "val_indices.npy"))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"  Train : {X_train.shape[0]:5d} epochs")
    print(f"  Val   : {X_val.shape[0]:5d} epochs")

    return X_train, y_train, X_val, y_val


def make_dataloader(X, y, batch_size, shuffle=True):
    """Create a PyTorch DataLoader from numpy arrays."""
    # Reshape X from (N, 3000) -> (N, 1, 3000) for Conv1d
    X_t = torch.from_numpy(X).unsqueeze(1).float()   # (N, 1, 3000)
    y_t = torch.from_numpy(y).long()                  # (N,)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def compute_class_weights(y_train):
    """Compute inverse-frequency class weights from training labels only."""
    class_counts = np.bincount(y_train, minlength=N_CLASSES).astype(np.float64)
    total = class_counts.sum()
    weights = total / (N_CLASSES * class_counts)
    return weights


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    X_train, y_train, X_val, y_val = load_split_data()

    train_loader = make_dataloader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, BATCH_SIZE, shuffle=False)

    # Class weights (training data only)
    weights = compute_class_weights(y_train)
    print(f"\nClass weights (from training data):")
    for i in range(N_CLASSES):
        print(f"  {i} - {INT_TO_CLASS[i]:5s} : {weights[i]:.4f}")

    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    # Model
    model = SleepCNN(n_classes=N_CLASSES).to(device)
    print(f"\nModel summary:")
    print(f"  Trainable parameters : {count_parameters(model):,}")
    print(f"  Model size           : {model_size_mb(model):.2f} MB")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Tracking
    best_val_f1 = 0.0
    best_epoch = -1
    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
    }

    print()
    print("=" * 75)
    print(f"{'Epoch':>5s}  {'Tr Loss':>8s}  {'Tr Acc':>7s}  {'Tr F1':>6s}  "
          f"{'Val Loss':>8s}  {'Val Acc':>7s}  {'Val F1':>7s}  {'Best':>4s}")
    print("=" * 75)

    total_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

        train_loss = running_loss / len(all_labels)
        train_acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
        train_f1 = f1_score(all_labels, all_preds, average="macro")

        # ---- Validate ----
        model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                running_loss += loss.item() * X_batch.size(0)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())

        val_loss = running_loss / len(all_labels)
        val_acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        # Track history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # Save best model by val macro-F1
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            save_path = os.path.join(MODEL_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "class_weights": weights.tolist(),
            }, save_path)

        marker = " *" if is_best else ""
        print(f"{epoch:5d}  {train_loss:8.4f}  {train_acc:6.2f}%  {train_f1:.4f}  "
              f"{val_loss:8.4f}  {val_acc:6.2f}%  {val_f1:7.4f}{marker}")

    elapsed = time.time() - total_start
    print("=" * 75)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best val macro-F1 : {best_val_f1:.4f}  (epoch {best_epoch})")
    print(f"Model saved       : {os.path.abspath(os.path.join(MODEL_DIR, 'best_model.pth'))}")

    # Save training history
    np.savez(
        os.path.join(MODEL_DIR, "training_history.npz"),
        **{k: np.array(v) for k, v in history.items()},
    )
    print(f"History saved     : training_history.npz")

    return model, history


if __name__ == "__main__":
    train()
