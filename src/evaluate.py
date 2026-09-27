"""
evaluate.py
===========
Evaluate the best saved model on the held-out TEST set.

Metrics:
  - Overall accuracy
  - Per-class precision, recall, F1
  - Macro-averaged precision, recall, F1
  - Confusion matrix

Usage:
    python src/evaluate.py
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import SleepCNN, count_parameters, model_size_mb


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

N_CLASSES = 5
BATCH_SIZE = 256
CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    print("\nLoading test data...")
    X = np.load(os.path.join(PROC_DIR, "X_epochs.npy"))
    y = np.load(os.path.join(PROC_DIR, "y_labels.npy"))
    test_idx = np.load(os.path.join(PROC_DIR, "test_indices.npy"))

    X_test = X[test_idx]
    y_test = y[test_idx]
    print(f"  Test epochs : {len(X_test)}")

    X_t = torch.from_numpy(X_test).unsqueeze(1).float()
    y_t = torch.from_numpy(y_test).long()
    test_loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=False,
    )

    # ------------------------------------------------------------------
    # Load best model
    # ------------------------------------------------------------------
    print("\nLoading best model...")
    model = SleepCNN(n_classes=N_CLASSES).to(device)

    ckpt_path = os.path.join(MODEL_DIR, "best_model.pth")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  Checkpoint epoch   : {checkpoint['epoch']}")
    print(f"  Val macro-F1 (ckpt): {checkpoint['val_f1']:.4f}")
    print(f"  Trainable params   : {count_parameters(model):,}")
    print(f"  Model size         : {model_size_mb(model):.2f} MB")

    # Saved checkpoint file size
    ckpt_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"  Checkpoint file    : {ckpt_mb:.2f} MB")

    # ------------------------------------------------------------------
    # Inference on test set
    # ------------------------------------------------------------------
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    print()
    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)

    print(f"\n  Overall Accuracy : {accuracy * 100:.2f}%")
    print(f"  Macro F1-Score   : {macro_f1:.4f}")

    # Per-class report
    print()
    print("  Classification Report:")
    print("  " + "-" * 56)
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        digits=4,
    )
    for line in report.split("\n"):
        print(f"  {line}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    print()
    print("  Confusion Matrix:")
    print(f"  {'':>8s}", end="")
    for name in CLASS_NAMES:
        print(f"  {name:>6s}", end="")
    print("   <-- Predicted")
    print(f"  {'':>8s}", end="")
    for _ in CLASS_NAMES:
        print(f"  {'------':>6s}", end="")
    print()

    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:>8s}", end="")
        for j in range(N_CLASSES):
            print(f"  {cm[i][j]:>6d}", end="")
        print()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
        "predictions": all_preds,
        "true_labels": all_labels,
    }

    np.savez(
        os.path.join(OUTPUT_DIR, "test_results.npz"),
        accuracy=accuracy,
        macro_f1=macro_f1,
        confusion_matrix=cm,
        predictions=all_preds,
        true_labels=all_labels,
    )

    print()
    print(f"  Results saved to: outputs/test_results.npz")
    print()
    print("=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
