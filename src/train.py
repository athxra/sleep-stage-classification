"""
train.py
========
Train the lightweight 1D CNN on preprocessed Sleep-EDF data with
subject-grouped splits.

Features:
  - Class-weighted CrossEntropyLoss (weights from training subjects only).
  - Light data augmentation (time shift, amplitude scaling, Gaussian noise).
  - ReduceLROnPlateau + early stopping on validation macro-F1.
  - Fixed random seeds for reproducibility.
  - --cv trains every fold and reports mean +/- std test metrics.

Usage:
    python src/train.py              # train fold 0 -> models/best_model.pth
    python src/train.py --cv         # k-fold CV    -> outputs/cv_metrics.json
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (CLASS_NAMES, CV_METRICS_PATH, MODEL_DIR, MODEL_PATH,
                        N_CLASSES, OUTPUT_DIR, PROC_DIR, SEED)
from src.metrics import compute_metrics
from src.model import SleepCNN, count_macs, count_parameters, model_size_mb
from src.splits import fold_indices


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    """Load preprocessed arrays and fold definitions."""
    data = {name: np.load(os.path.join(PROC_DIR, f"{name}.npy"))
            for name in ("X_epochs", "y_labels", "subject_ids", "recording_ids", "epoch_index")}
    with open(os.path.join(PROC_DIR, "split_info.json")) as f:
        data["split_info"] = json.load(f)
    return data


def compute_class_weights(y_train, mode: str = "inverse"):
    """Class weights from training labels only (normalized to mean 1)."""
    counts = np.bincount(y_train, minlength=N_CLASSES).astype(np.float64)
    if mode == "none":
        return np.ones(N_CLASSES)
    weights = counts.sum() / (N_CLASSES * np.maximum(counts, 1))
    if mode == "sqrt":
        weights = np.sqrt(weights)
    return weights / weights.mean()


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random time shift (up to +/-3 s), amplitude scaling and Gaussian noise."""
    shift = int(torch.randint(-300, 301, (1,)))
    x = torch.roll(x, shifts=shift, dims=-1)
    scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(0.9, 1.1)
    return x * scale + 0.05 * torch.randn_like(x)


@torch.no_grad()
def predict(model, X, device, batch_size: int = 1024):
    """Predicted class per epoch for an (N, 3000) array."""
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device).unsqueeze(1)
        preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.int64)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_fold(data, fold_id: int, args, save_path: str, verbose: bool = True):
    """Train on one fold; save the best-val-F1 checkpoint to save_path."""
    set_seed(args.seed + fold_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold = data["split_info"]["folds"][fold_id]
    X, y = data["X_epochs"], data["y_labels"]
    train_idx, val_idx, _ = fold_indices(data["subject_ids"], fold)

    X_train = torch.as_tensor(X[train_idx], device=device).unsqueeze(1)
    y_train = torch.as_tensor(y[train_idx], device=device)
    X_val, y_val = X[val_idx], y[val_idx]

    weights = compute_class_weights(y[train_idx], args.class_weight)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    model = SleepCNN(n_classes=N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=3)

    if verbose:
        print(f"\nFold {fold_id} | device {device} | train subjects {fold['train']}")
        print(f"         val subjects {fold['val']} | test subjects {fold['test']}")
        print(f"  Train {len(train_idx)} epochs, Val {len(val_idx)} epochs")
        print("  Class weights: " + ", ".join(f"{n} {w:.2f}" for n, w in zip(CLASS_NAMES, weights)))
        print(f"  {'Epoch':>5s}  {'Tr Loss':>8s}  {'Tr F1':>6s}  {'Val Acc':>7s}  {'Val F1':>6s}  {'LR':>8s}")

    best_f1, best_epoch, stale = -1.0, 0, 0
    history = {"train_loss": [], "train_f1": [], "val_acc": [], "val_f1": []}
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(y_train), device=device)
        total_loss, preds = 0.0, torch.empty_like(y_train)
        for i in range(0, len(perm), args.batch_size):
            batch = perm[i:i + args.batch_size]
            xb, yb = X_train[batch], y_train[batch]
            if args.augment:
                xb = augment(xb)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
            preds[batch] = logits.argmax(dim=1).detach()

        train_loss = total_loss / len(y_train)
        train_f1 = f1_score(y_train.cpu().numpy(), preds.cpu().numpy(), average="macro")
        val_pred = predict(model, X_val, device)
        val_acc = float((val_pred == y_val).mean())
        val_f1 = float(f1_score(y_val, val_pred, average="macro"))
        scheduler.step(val_f1)

        for key, val in zip(history, (train_loss, train_f1, val_acc, val_f1)):
            history[key].append(float(val))

        improved = val_f1 > best_f1
        if improved:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_classes": N_CLASSES,
                "epoch": epoch,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "fold": fold_id,
                "train_subjects": fold["train"],
                "val_subjects": fold["val"],
                "test_subjects": fold["test"],
                "class_weights": [float(w) for w in weights],
                "hyperparameters": {k: v for k, v in vars(args).items()
                                    if isinstance(v, (int, float, str, bool))},
            }, save_path)
        else:
            stale += 1

        if verbose:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  {epoch:5d}  {train_loss:8.4f}  {train_f1:6.4f}  {100 * val_acc:6.2f}%  "
                  f"{val_f1:6.4f}  {lr:8.1e}{'  *' if improved else ''}")
        if stale >= args.patience:
            if verbose:
                print(f"  Early stopping (no val-F1 gain for {args.patience} epochs).")
            break

    if verbose:
        print(f"  Best val macro-F1 {best_f1:.4f} at epoch {best_epoch} "
              f"({time.time() - start:.0f}s) -> {os.path.relpath(save_path)}")
    return history, best_epoch, best_f1


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def run_single(data, args):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model = SleepCNN(n_classes=N_CLASSES)
    print(f"Model: {count_parameters(model):,} params, {model_size_mb(model):.2f} MB, "
          f"{count_macs(model) / 1e6:.1f} M MACs/epoch")
    history, _, _ = train_fold(data, args.fold, args, MODEL_PATH)
    np.savez(os.path.join(MODEL_DIR, "training_history.npz"),
             **{k: np.array(v) for k, v in history.items()})
    print("Run `python src/evaluate.py` to score this model on its held-out test subjects.")


def run_cv(data, args):
    from src.model import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv_dir = os.path.join(MODEL_DIR, "cv")
    os.makedirs(cv_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    folds = data["split_info"]["folds"]
    all_true, all_pred, per_fold = [], [], []
    for k in range(len(folds)):
        path = os.path.join(cv_dir, f"fold{k}.pth")
        _, best_epoch, best_val_f1 = train_fold(data, k, args, path)
        model, _ = load_checkpoint(path, device)
        _, _, test_idx = fold_indices(data["subject_ids"], folds[k])
        y_true = data["y_labels"][test_idx]
        y_pred = predict(model, data["X_epochs"][test_idx], device)
        m = compute_metrics(y_true, y_pred, data["recording_ids"][test_idx],
                            data["epoch_index"][test_idx])
        print(f"  Fold {k} TEST: acc {100 * m['accuracy']:.2f}%  macro-F1 {m['macro_f1']:.4f}  "
              f"kappa {m['cohen_kappa']:.4f}")
        per_fold.append({"fold": k, "test_subjects": folds[k]["test"], "best_epoch": best_epoch,
                         "best_val_f1": best_val_f1, **{key: m[key] for key in
                         ("accuracy", "macro_f1", "cohen_kappa", "n_epochs")},
                         "per_class_f1": {c: m["per_class"][c]["f1"] for c in CLASS_NAMES}})
        all_true.append(y_true)
        all_pred.append(y_pred)

    def mean_std(key):
        vals = [f[key] for f in per_fold]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    pooled = compute_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    summary = {
        "n_folds": len(folds),
        "n_subjects": data["split_info"]["n_subjects"],
        "accuracy": mean_std("accuracy"),
        "macro_f1": mean_std("macro_f1"),
        "cohen_kappa": mean_std("cohen_kappa"),
        "per_class_f1": {c: {"mean": float(np.mean([f["per_class_f1"][c] for f in per_fold])),
                             "std": float(np.std([f["per_class_f1"][c] for f in per_fold]))}
                         for c in CLASS_NAMES},
        "pooled": pooled,
        "folds": per_fold,
    }
    with open(CV_METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{len(folds)}-FOLD SUBJECT-GROUPED CV  ({summary['n_subjects']} subjects)")
    print("=" * 70)
    for key in ("accuracy", "macro_f1", "cohen_kappa"):
        print(f"  {key:12s}: {summary[key]['mean']:.4f} +/- {summary[key]['std']:.4f}")
    for c in CLASS_NAMES:
        s = summary["per_class_f1"][c]
        print(f"  F1 {c:5s}    : {s['mean']:.4f} +/- {s['std']:.4f}")
    print(f"Saved {os.path.relpath(CV_METRICS_PATH)}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the lightweight sleep-stage CNN.")
    parser.add_argument("--cv", action="store_true", help="run all subject-grouped folds")
    parser.add_argument("--fold", type=int, default=0, help="fold to train (single-run mode)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", choices=["inverse", "sqrt", "none"], default="inverse")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    data = load_data()
    if args.cv:
        run_cv(data, args)
    else:
        run_single(data, args)
