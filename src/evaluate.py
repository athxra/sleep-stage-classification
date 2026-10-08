"""
evaluate.py
===========
Evaluate the saved model on its held-out TEST subjects (never used for
training or checkpoint selection).

Metrics:
  - Accuracy, macro-F1, Cohen's kappa (and the majority-class baseline)
  - Per-class precision, recall, F1 and confusion matrix
  - N1 / REM confusion focus
  - Stage-transition-aware analysis: performance on epochs next to an
    expert-scored stage change vs. epochs inside stable stage runs

Writes outputs/test_metrics.json (read by the dashboard) and
outputs/test_predictions.npz.

Usage:
    python src/evaluate.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (CLASS_NAMES, MODEL_PATH, OUTPUT_DIR, TEST_METRICS_PATH)
from src.metrics import compute_metrics
from src.model import count_macs, count_parameters, load_checkpoint, model_size_mb
from src.train import load_data, predict


def print_report(m: dict) -> None:
    print("=" * 64)
    print("TEST SET RESULTS")
    print("=" * 64)
    print(f"  Test subjects        : {m['test_subjects']}  ({m['n_epochs']} epochs)")
    print(f"  Accuracy             : {100 * m['accuracy']:.2f}%  "
          f"(majority-class baseline {100 * m['majority_class_baseline']:.2f}%)")
    print(f"  Macro F1             : {m['macro_f1']:.4f}")
    print(f"  Cohen's kappa        : {m['cohen_kappa']:.4f}")

    print(f"\n  {'Class':<6s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'Support':>8s}")
    for name in CLASS_NAMES:
        c = m["per_class"][name]
        print(f"  {name:<6s} {c['precision']:7.4f} {c['recall']:7.4f} {c['f1']:7.4f} {c['support']:8d}")

    print("\n  Confusion matrix (rows = expert, cols = predicted)")
    print("  " + " " * 6 + "".join(f"{n:>7s}" for n in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, m["confusion_matrix"]):
        print(f"  {name:<6s}" + "".join(f"{v:7d}" for v in row))

    print("\n  Top confusions")
    for c in m["top_confusions"]:
        print(f"    {c['true']:>5s} -> {c['predicted']:<5s} {c['count']:5d}  "
              f"({100 * c['rate']:.1f}% of {c['true']})")

    t = m["transition_analysis"]
    print("\n  Stage-transition analysis")
    for key, label in (("stable", "Stable epochs"), ("near_transition", "Near a transition")):
        s = t[key]
        if s["n_epochs"]:
            print(f"    {label:<18s} n={s['n_epochs']:5d}  acc {100 * s['accuracy']:.2f}%  "
                  f"macro-F1 {s['macro_f1']:.4f}")


def evaluate(model_path: str = MODEL_PATH) -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_checkpoint(model_path, device)
    data = load_data()
    test_mask = np.isin(data["subject_ids"], ckpt["test_subjects"])
    assert not set(ckpt["test_subjects"]) & set(ckpt["train_subjects"] + ckpt["val_subjects"]), \
        "Test subjects overlap with training/validation subjects"

    y_true = data["y_labels"][test_mask]
    y_pred = predict(model, data["X_epochs"][test_mask], device)
    rec_ids = data["recording_ids"][test_mask]
    epoch_idx = data["epoch_index"][test_mask]

    metrics = compute_metrics(y_true, y_pred, rec_ids, epoch_idx)
    recordings = {r["id"]: r["psg"] for r in data["split_info"]["recordings"]}
    metrics.update({
        "test_subjects": ckpt["test_subjects"],
        "val_subjects": ckpt["val_subjects"],
        "train_subjects": ckpt["train_subjects"],
        "test_recordings": sorted({recordings[int(r)] for r in np.unique(rec_ids)}),
        "fold": ckpt["fold"],
        "checkpoint_epoch": ckpt["epoch"],
        "val_macro_f1": ckpt["val_f1"],
        "model": {
            "parameters": count_parameters(model),
            "size_mb": round(model_size_mb(model), 3),
            "macs_per_epoch": count_macs(model.cpu()),
            "checkpoint_mb": round(os.path.getsize(model_path) / 2**20, 3),
        },
        "dataset": {
            "n_subjects": data["split_info"]["n_subjects"],
            "n_recordings": len(data["split_info"]["recordings"]),
            "n_epochs": data["split_info"]["n_epochs"],
            "wake_edge_min": data["split_info"]["wake_edge_min"],
        },
    })

    print_report(metrics)
    with open(TEST_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(os.path.join(OUTPUT_DIR, "test_predictions.npz"), y_true=y_true, y_pred=y_pred,
             recording_ids=rec_ids, epoch_index=epoch_idx)
    print(f"\n  Saved {os.path.relpath(TEST_METRICS_PATH)} and test_predictions.npz")
    return metrics


if __name__ == "__main__":
    evaluate()
