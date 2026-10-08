"""
preprocess.py
=============
Preprocess every Sleep-EDF (PSG, Hypnogram) pair in data/raw/ into labelled
30-second EEG epochs with subject tracking and subject-grouped splits.

Steps:
  1. Discover every SC4xxx PSG file in data/raw/ and pair it with its hypnogram.
  2. Extract the Fpz-Cz EEG channel (100 Hz) and cut it into 30-s epochs.
  3. Label each epoch from the hypnogram; merge S3+S4 -> N3 (AASM).
  4. Discard unscored epochs ("Sleep stage ?", "Movement time").
  5. Keep the sleep period plus 30 min of Wake on each side.
  6. Apply per-epoch Z-score normalization.
  7. Save epochs, labels, subject / recording IDs and epoch positions.
  8. Build subject-grouped cross-validation folds (both nights of a subject
     always stay in the same split).

Usage:
    python src/preprocess.py [--folds 5] [--val-subjects 2]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (CLASS_NAMES, PROC_DIR, RAW_DIR, WAKE_EDGE_MIN,
                        parse_recording, recording_key)
from src.signals import (UNSCORED, labels_on_grid, load_fpz_cz, segment_epochs,
                         sleep_period_mask, zscore_epochs)
from src.splits import make_folds


def discover_recordings(raw_dir: str = RAW_DIR):
    """Return a sorted list of (psg_path, hypnogram_path) pairs."""
    hypnos = {recording_key(p): p for p in glob.glob(os.path.join(raw_dir, "SC4*-Hypnogram.edf"))}
    pairs = []
    for psg in sorted(glob.glob(os.path.join(raw_dir, "SC4*-PSG.edf"))):
        hyp = hypnos.get(recording_key(psg))
        if hyp is None:
            print(f"  WARNING: no hypnogram for {os.path.basename(psg)}, skipping.")
            continue
        pairs.append((psg, hyp))
    return pairs


def process_recording(psg_path: str, hypno_path: str):
    """Return (X, y, epoch_index, stats) for one recording."""
    import mne

    signal = load_fpz_cz(psg_path)
    epochs = segment_epochs(signal)
    labels = labels_on_grid(mne.read_annotations(hypno_path), len(epochs))

    keep = sleep_period_mask(labels) & (labels != UNSCORED)
    stats = {
        "grid_epochs": int(len(labels)),
        "unscored": int((labels == UNSCORED).sum()),
        "trimmed_wake": int(((labels == 0) & ~sleep_period_mask(labels)).sum()),
        "kept": int(keep.sum()),
    }
    epoch_index = np.flatnonzero(keep)
    return zscore_epochs(epochs[keep]), labels[keep], epoch_index, stats


def print_distribution(title: str, y: np.ndarray) -> None:
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    parts = [f"{name} {c:5d} ({100 * c / max(len(y), 1):4.1f}%)"
             for name, c in zip(CLASS_NAMES, counts)]
    print(f"  {title:<8s} n={len(y):6d} | " + " | ".join(parts))


def preprocess(n_folds: int = 5, n_val_subjects: int = 2) -> None:
    os.makedirs(PROC_DIR, exist_ok=True)
    pairs = discover_recordings()
    if not pairs:
        raise SystemExit(f"No Sleep-EDF files found in {RAW_DIR}. "
                         f"Run `python src/download_data.py` first.")

    print("=" * 70)
    print(f"PROCESSING {len(pairs)} RECORDINGS  (Wake edge = {WAKE_EDGE_MIN} min)")
    print("=" * 70)

    all_X, all_y, all_sub, all_rec, all_idx = [], [], [], [], []
    recordings = []
    for rec_id, (psg, hyp) in enumerate(pairs):
        subject, night = parse_recording(psg)
        X, y, idx, stats = process_recording(psg, hyp)
        print(f"  {os.path.basename(psg):<20s} subject {subject:2d} night {night} | "
              f"grid {stats['grid_epochs']:5d}  unscored {stats['unscored']:4d}  "
              f"trimmed-wake {stats['trimmed_wake']:5d}  kept {stats['kept']:5d}")
        all_X.append(X)
        all_y.append(y)
        all_sub.append(np.full(len(y), subject, dtype=np.int64))
        all_rec.append(np.full(len(y), rec_id, dtype=np.int64))
        all_idx.append(idx.astype(np.int64))
        recordings.append({
            "id": rec_id, "psg": os.path.basename(psg), "hypnogram": os.path.basename(hyp),
            "subject": subject, "night": night, **stats,
        })

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    subject_ids = np.concatenate(all_sub)
    recording_ids = np.concatenate(all_rec)
    epoch_index = np.concatenate(all_idx)

    print()
    print_distribution("Overall", y)

    np.save(os.path.join(PROC_DIR, "X_epochs.npy"), X)
    np.save(os.path.join(PROC_DIR, "y_labels.npy"), y)
    np.save(os.path.join(PROC_DIR, "subject_ids.npy"), subject_ids)
    np.save(os.path.join(PROC_DIR, "recording_ids.npy"), recording_ids)
    np.save(os.path.join(PROC_DIR, "epoch_index.npy"), epoch_index)

    # ------------------------------------------------------------------
    # Subject-grouped folds
    # ------------------------------------------------------------------
    folds = make_folds(subject_ids, n_folds=n_folds, n_val_subjects=n_val_subjects)
    print()
    print("=" * 70)
    print(f"SUBJECT-GROUPED FOLDS  ({len(folds)} folds, {len(set(subject_ids))} subjects)")
    print("=" * 70)
    for k, fold in enumerate(folds):
        parts = [set(fold["train"]), set(fold["val"]), set(fold["test"])]
        assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]), \
            f"Subject overlap in fold {k}"
        print(f"  Fold {k}: train {fold['train']}\n          val {fold['val']}  test {fold['test']}")
        for name in ("train", "val", "test"):
            fold[f"n_{name}_epochs"] = int(np.isin(subject_ids, fold[name]).sum())

    split_info = {
        "strategy": "subject-grouped k-fold (both nights of a subject in the same split)",
        "wake_edge_min": WAKE_EDGE_MIN,
        "n_epochs": int(len(y)),
        "n_subjects": int(len(set(subject_ids))),
        "class_counts": {name: int(c) for name, c in
                         zip(CLASS_NAMES, np.bincount(y, minlength=len(CLASS_NAMES)))},
        "recordings": recordings,
        "folds": folds,
    }
    with open(os.path.join(PROC_DIR, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    print()
    print(f"  Saved X_epochs.npy {X.shape}, labels, subject/recording IDs and split_info.json")
    print("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-subjects", type=int, default=2)
    args = parser.parse_args()
    preprocess(n_folds=args.folds, n_val_subjects=args.val_subjects)
