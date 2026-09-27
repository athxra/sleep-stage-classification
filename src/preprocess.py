"""
preprocess.py
=============
Preprocess ALL Sleep-EDF PSG + Hypnogram file pairs into labelled 30-second
EEG epochs with subject tracking and subject-independent data splitting.

Steps:
  1. Iterate over every (PSG, Hypnogram) pair in data/raw/.
  2. Extract the Fpz-Cz EEG channel (100 Hz).
  3. Load hypnogram annotations and expand into 30-second epochs.
  4. Map sleep-stage labels to 5 classes (Wake, N1, N2, N3, REM).
  5. Discard unknown ("Sleep stage ?") and "Movement time" epochs.
  6. Apply per-epoch Z-score normalization.
  7. Save epochs (X), labels (y), and subject IDs as .npy files.
  8. Create a subject-independent train / val / test split.

Usage:
    python src/preprocess.py
"""

import os
import json
import numpy as np
import mne


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Each entry: (subject_id, PSG filename, Hypnogram filename)
SUBJECTS = [
    (0, "SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf"),
    (1, "SC4002E0-PSG.edf", "SC4002EC-Hypnogram.edf"),
    (2, "SC4011E0-PSG.edf", "SC4011EH-Hypnogram.edf"),
    (3, "SC4012E0-PSG.edf", "SC4012EC-Hypnogram.edf"),
    (4, "SC4021E0-PSG.edf", "SC4021EH-Hypnogram.edf"),
    (5, "SC4022E0-PSG.edf", "SC4022EJ-Hypnogram.edf"),
]

CHANNEL = "EEG Fpz-Cz"
SFREQ = 100          # Hz
EPOCH_SEC = 30       # seconds
EPOCH_SAMPLES = SFREQ * EPOCH_SEC   # 3000

# Label mapping: annotation description -> class name
LABEL_MAP = {
    "Sleep stage W": "Wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",   # AASM: merge S3+S4 -> N3
    "Sleep stage R": "REM",
}

# Integer encoding
CLASS_TO_INT = {
    "Wake": 0,
    "N1":   1,
    "N2":   2,
    "N3":   3,
    "REM":  4,
}

INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}

# Subject-independent split assignment
#   Train : 4 subjects
#   Val   : 1 subject (completely held out)
#   Test  : 1 subject (completely held out)
SPLIT_ASSIGNMENT = {
    0: "train",    # SC4001
    1: "train",    # SC4002
    2: "train",    # SC4011
    3: "train",    # SC4012
    4: "val",      # SC4021 — validation subject
    5: "test",     # SC4022 — test subject (unseen)
}


# ---------------------------------------------------------------------------
# Process a single subject
# ---------------------------------------------------------------------------
def process_subject(subject_id: int, psg_path: str, hypno_path: str):
    """Return (epochs, labels, subject_ids, discard_info) for one subject."""

    print(f"  Loading PSG  : {os.path.basename(psg_path)}")
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    raw.pick([CHANNEL])

    sfreq = raw.info["sfreq"]
    assert sfreq == SFREQ, f"Expected {SFREQ} Hz, got {sfreq} Hz"

    eeg_data = raw.get_data()[0]          # (n_samples,)
    total_samples = len(eeg_data)

    print(f"  Loading Hyp  : {os.path.basename(hypno_path)}")
    annotations = mne.read_annotations(hypno_path)

    epochs_list = []
    labels_list = []
    discard_info = {}   # description -> epoch count
    skipped_boundary = 0

    for ann_idx in range(len(annotations)):
        onset = annotations.onset[ann_idx]
        duration = annotations.duration[ann_idx]
        description = annotations.description[ann_idx]
        n_epochs_in_ann = int(duration // EPOCH_SEC)

        if description not in LABEL_MAP:
            discard_info[description] = discard_info.get(description, 0) + n_epochs_in_ann
            continue

        label_int = CLASS_TO_INT[LABEL_MAP[description]]

        for i in range(n_epochs_in_ann):
            epoch_start_sec = onset + i * EPOCH_SEC
            start_sample = int(epoch_start_sec * sfreq)
            end_sample = start_sample + EPOCH_SAMPLES

            if end_sample > total_samples:
                skipped_boundary += 1
                continue

            epochs_list.append(eeg_data[start_sample:end_sample])
            labels_list.append(label_int)

    X = np.array(epochs_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    sids = np.full(len(y), subject_id, dtype=np.int64)

    print(f"  Epochs       : {len(X)}  "
          f"(discarded={sum(discard_info.values())}, boundary={skipped_boundary})")

    return X, y, sids, discard_info


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def preprocess() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    all_X, all_y, all_sids = [], [], []
    all_discard = {}   # subject_id -> {desc: count}

    print("=" * 60)
    print("PROCESSING ALL SUBJECTS")
    print("=" * 60)

    for subject_id, psg_name, hypno_name in SUBJECTS:
        psg_path = os.path.join(DATA_DIR, psg_name)
        hypno_path = os.path.join(DATA_DIR, hypno_name)

        # Verify files exist
        missing = False
        for tag, path in [("PSG", psg_path), ("Hypnogram", hypno_path)]:
            if not os.path.isfile(path):
                print(f"\n  ERROR: {tag} file not found: {os.path.abspath(path)}")
                print(f"  Skipping subject {subject_id}.\n")
                missing = True
                break
        if missing:
            continue

        print(f"\n--- Subject {subject_id} ({psg_name}) ---")
        X, y, sids, disc = process_subject(subject_id, psg_path, hypno_path)
        all_X.append(X)
        all_y.append(y)
        all_sids.append(sids)
        all_discard[subject_id] = disc

    # Concatenate across subjects
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subject_ids = np.concatenate(all_sids, axis=0)

    # ------------------------------------------------------------------
    # Discarded annotations summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("DISCARDED ANNOTATIONS SUMMARY")
    print("=" * 60)

    for subject_id, psg_name, _ in SUBJECTS:
        if subject_id not in all_discard:
            continue
        disc = all_discard[subject_id]
        if disc:
            for desc, count in sorted(disc.items()):
                print(f"  Subject {subject_id} ({psg_name}):  {desc:<20s}  {count} epochs")
        else:
            print(f"  Subject {subject_id} ({psg_name}):  (none discarded)")

    # ------------------------------------------------------------------
    # Z-score normalization
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Z-SCORE NORMALIZATION")
    print("=" * 60)

    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    X = (X - means) / stds

    print(f"Post-normalization -- mean range : [{X.mean(axis=1).min():.6f}, {X.mean(axis=1).max():.6f}]")
    print(f"Post-normalization -- std  range : [{X.std(axis=1).min():.4f}, {X.std(axis=1).max():.4f}]")

    # ------------------------------------------------------------------
    # Per-subject class distribution
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("PER-SUBJECT CLASS DISTRIBUTION")
    print("=" * 60)

    for subject_id, psg_name, _ in SUBJECTS:
        mask = subject_ids == subject_id
        if mask.sum() == 0:
            continue
        sub_y = y[mask]
        print(f"\n  Subject {subject_id}  ({psg_name})")
        for cls_int in sorted(INT_TO_CLASS.keys()):
            cls_name = INT_TO_CLASS[cls_int]
            count = int((sub_y == cls_int).sum())
            print(f"    {cls_int} - {cls_name:5s} : {count:5d}")
        print(f"    {'':7s} Total : {int(mask.sum()):5d}")

    # ------------------------------------------------------------------
    # Overall class distribution
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("OVERALL CLASS DISTRIBUTION")
    print("=" * 60)

    for cls_int in sorted(INT_TO_CLASS.keys()):
        cls_name = INT_TO_CLASS[cls_int]
        count = int((y == cls_int).sum())
        pct = 100.0 * count / len(y)
        print(f"  {cls_int} - {cls_name:5s} : {count:5d} epochs  ({pct:5.1f}%)")
    print(f"  {'':7s} Total : {len(y):5d} epochs")

    # ------------------------------------------------------------------
    # Save preprocessed data
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("SAVING PREPROCESSED DATA")
    print("=" * 60)

    x_path = os.path.join(OUT_DIR, "X_epochs.npy")
    y_path = os.path.join(OUT_DIR, "y_labels.npy")
    s_path = os.path.join(OUT_DIR, "subject_ids.npy")

    np.save(x_path, X)
    np.save(y_path, y)
    np.save(s_path, subject_ids)

    x_mb = os.path.getsize(x_path) / (1024 * 1024)
    y_kb = os.path.getsize(y_path) / 1024
    s_kb = os.path.getsize(s_path) / 1024

    print(f"  X_epochs.npy    : {X.shape}  ({x_mb:.1f} MB)")
    print(f"  y_labels.npy    : {y.shape}  ({y_kb:.1f} KB)")
    print(f"  subject_ids.npy : {subject_ids.shape}  ({s_kb:.1f} KB)")

    # ==================================================================
    # SUBJECT-INDEPENDENT SPLIT
    # ==================================================================
    print()
    print("=" * 60)
    print("SUBJECT-INDEPENDENT SPLIT")
    print("=" * 60)

    split_map = {
        "train_subjects": [],
        "val_subjects": [],
        "test_subjects": [],
    }

    train_idx, val_idx, test_idx = [], [], []

    for subject_id, psg_name, _ in SUBJECTS:
        role = SPLIT_ASSIGNMENT.get(subject_id)
        if role is None:
            continue
        mask = np.where(subject_ids == subject_id)[0]
        if len(mask) == 0:
            continue

        if role == "train":
            train_idx.extend(mask.tolist())
            split_map["train_subjects"].append(
                {"id": subject_id, "file": psg_name, "epochs": int(len(mask))}
            )
        elif role == "val":
            val_idx.extend(mask.tolist())
            split_map["val_subjects"].append(
                {"id": subject_id, "file": psg_name, "epochs": int(len(mask))}
            )
        elif role == "test":
            test_idx.extend(mask.tolist())
            split_map["test_subjects"].append(
                {"id": subject_id, "file": psg_name, "epochs": int(len(mask))}
            )

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)

    total = len(train_idx) + len(val_idx) + len(test_idx)

    # Print split summary
    print()
    print("  Split assignment:")
    print(f"    Train ({len(split_map['train_subjects'])} subjects):")
    for s in split_map["train_subjects"]:
        print(f"      Subject {s['id']}  ({s['file']})  — {s['epochs']} epochs")

    print(f"    Val   ({len(split_map['val_subjects'])} subject):")
    for s in split_map["val_subjects"]:
        print(f"      Subject {s['id']}  ({s['file']})  — {s['epochs']} epochs")

    print(f"    Test  ({len(split_map['test_subjects'])} subject):")
    for s in split_map["test_subjects"]:
        print(f"      Subject {s['id']}  ({s['file']})  — {s['epochs']} epochs")

    print()
    print(f"  Split sizes:")
    print(f"    Train : {len(train_idx):5d} epochs  ({100*len(train_idx)/total:.1f}%)")
    print(f"    Val   : {len(val_idx):5d} epochs  ({100*len(val_idx)/total:.1f}%)")
    print(f"    Test  : {len(test_idx):5d} epochs  ({100*len(test_idx)/total:.1f}%)")
    print(f"    Total : {total:5d} epochs")

    # Verify no overlap
    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())

    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set

    print()
    if not overlap_tv and not overlap_tt and not overlap_vt:
        print("  [PASS] No overlap between train / val / test splits.")
    else:
        print(f"  [FAIL] Overlaps found: train&val={len(overlap_tv)}, "
              f"train&test={len(overlap_tt)}, val&test={len(overlap_vt)}")

    # Verify subject independence
    train_subs = set(subject_ids[train_idx].tolist())
    val_subs = set(subject_ids[val_idx].tolist())
    test_subs = set(subject_ids[test_idx].tolist())

    if not (train_subs & val_subs) and not (train_subs & test_subs) and not (val_subs & test_subs):
        print("  [PASS] Splits are fully subject-independent (no subject in multiple splits).")
    else:
        print("  [FAIL] Subject appears in multiple splits!")

    # Per-class distribution in each split
    for split_name, indices in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        split_y = y[indices]
        print(f"\n  {split_name} class distribution:")
        for cls_int in sorted(INT_TO_CLASS.keys()):
            cls_name = INT_TO_CLASS[cls_int]
            count = int((split_y == cls_int).sum())
            pct = 100.0 * count / len(split_y)
            print(f"    {cls_int} - {cls_name:5s} : {count:5d}  ({pct:5.1f}%)")

    # Save split indices and metadata
    np.save(os.path.join(OUT_DIR, "train_indices.npy"), train_idx)
    np.save(os.path.join(OUT_DIR, "val_indices.npy"), val_idx)
    np.save(os.path.join(OUT_DIR, "test_indices.npy"), test_idx)

    split_meta = {
        "strategy": "subject-independent (LOSO)",
        "train": {
            "subjects": [s["id"] for s in split_map["train_subjects"]],
            "files": [s["file"] for s in split_map["train_subjects"]],
            "n_epochs": int(len(train_idx)),
        },
        "val": {
            "subjects": [s["id"] for s in split_map["val_subjects"]],
            "files": [s["file"] for s in split_map["val_subjects"]],
            "n_epochs": int(len(val_idx)),
        },
        "test": {
            "subjects": [s["id"] for s in split_map["test_subjects"]],
            "files": [s["file"] for s in split_map["test_subjects"]],
            "n_epochs": int(len(test_idx)),
        },
    }

    meta_path = os.path.join(OUT_DIR, "split_info.json")
    with open(meta_path, "w") as f:
        json.dump(split_meta, f, indent=2)

    print()
    print("=" * 60)
    print("SAVED SPLIT FILES")
    print("=" * 60)
    print(f"  train_indices.npy : {train_idx.shape}")
    print(f"  val_indices.npy   : {val_idx.shape}")
    print(f"  test_indices.npy  : {test_idx.shape}")
    print(f"  split_info.json   : split metadata")

    print()
    print("=" * 60)
    print("Preprocessing and splitting complete.")
    print("=" * 60)


if __name__ == "__main__":
    preprocess()
