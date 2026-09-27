"""
verify_preprocessing.py
=======================
Verify the preprocessed data and plan a subject-independent train/val/test split.

Usage:
    python src/verify_preprocessing.py
"""

import os
import numpy as np
import mne


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

SUBJECTS = [
    (0, "SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf"),
    (1, "SC4002E0-PSG.edf", "SC4002EC-Hypnogram.edf"),
    (2, "SC4011E0-PSG.edf", "SC4011EH-Hypnogram.edf"),
]

VALID_DESCRIPTIONS = {
    "Sleep stage W",
    "Sleep stage 1",
    "Sleep stage 2",
    "Sleep stage 3",
    "Sleep stage 4",
    "Sleep stage R",
}

INT_TO_CLASS = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
EPOCH_SEC = 30


# ===================================================================
# CHECK 1 — Discarded annotations per subject
# ===================================================================
def check_discarded_annotations():
    print("=" * 60)
    print("CHECK 1: DISCARDED HYPNOGRAM ANNOTATIONS")
    print("=" * 60)

    for subject_id, _, hypno_name in SUBJECTS:
        hypno_path = os.path.join(DATA_DIR, hypno_name)
        annotations = mne.read_annotations(hypno_path)

        print(f"\n  Subject {subject_id}  ({hypno_name})")
        print(f"  {'Description':<30s}  {'Epochs':>8s}  {'Status':<10s}")
        print(f"  {'-'*30}  {'-'*8}  {'-'*10}")

        total_kept = 0
        total_discarded = 0

        # Collect all unique descriptions in this file
        seen = {}
        for idx in range(len(annotations)):
            desc = annotations.description[idx]
            dur = annotations.duration[idx]
            n_epochs = int(dur // EPOCH_SEC)
            if desc not in seen:
                seen[desc] = 0
            seen[desc] += n_epochs

        for desc in sorted(seen.keys()):
            count = seen[desc]
            if desc in VALID_DESCRIPTIONS:
                status = "KEPT"
                total_kept += count
            else:
                status = "DISCARDED"
                total_discarded += count
            print(f"  {desc:<30s}  {count:>8d}  {status:<10s}")

        print(f"  {'':30s}  --------")
        print(f"  {'Kept':<30s}  {total_kept:>8d}")
        print(f"  {'Discarded':<30s}  {total_discarded:>8d}")

    print()


# ===================================================================
# CHECK 2 — Confirm only valid labels in saved data
# ===================================================================
def check_valid_labels():
    print("=" * 60)
    print("CHECK 2: LABEL VALIDITY")
    print("=" * 60)

    y = np.load(os.path.join(PROC_DIR, "y_labels.npy"))
    unique_labels = sorted(np.unique(y))
    expected = sorted(INT_TO_CLASS.keys())

    print(f"  Unique labels in y_labels.npy : {unique_labels}")
    print(f"  Expected labels               : {expected}")

    if unique_labels == expected:
        print("  [PASS] All labels are valid (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM).")
    else:
        print("  [FAIL] Unexpected labels detected!")

    # Extra: confirm no label outside 0-4
    if y.min() >= 0 and y.max() <= 4:
        print("  [PASS] No out-of-range labels.")
    else:
        print(f"  [FAIL] Label range [{y.min()}, {y.max()}] is outside [0, 4].")

    print()
    return y


# ===================================================================
# CHECK 3 — Confirm X_epochs shape
# ===================================================================
def check_shape():
    print("=" * 60)
    print("CHECK 3: EPOCH ARRAY SHAPE")
    print("=" * 60)

    X = np.load(os.path.join(PROC_DIR, "X_epochs.npy"), mmap_mode="r")
    print(f"  X_epochs.npy shape : {X.shape}")
    print(f"  dtype              : {X.dtype}")

    if X.shape == (8281, 3000):
        print("  [PASS] Shape matches expected (8281, 3000).")
    else:
        print(f"  [FAIL] Expected (8281, 3000), got {X.shape}.")

    # Quick sanity: Z-score normalization
    # Check a few random epochs
    rng = np.random.default_rng(42)
    idxs = rng.choice(X.shape[0], size=5, replace=False)
    all_ok = True
    for i in idxs:
        row = np.array(X[i])  # load from mmap
        m, s = row.mean(), row.std()
        if abs(m) > 1e-4 or abs(s - 1.0) > 1e-2:
            all_ok = False
            print(f"  [FAIL] Epoch {i}: mean={m:.6f}, std={s:.6f}")

    if all_ok:
        print("  [PASS] Z-score normalization verified on sample epochs.")

    print()
    return X


# ===================================================================
# CHECK 4 — Confirm subject IDs
# ===================================================================
def check_subject_ids():
    print("=" * 60)
    print("CHECK 4: SUBJECT IDS")
    print("=" * 60)

    sids = np.load(os.path.join(PROC_DIR, "subject_ids.npy"))
    unique_subs = sorted(np.unique(sids))
    print(f"  subject_ids.npy shape   : {sids.shape}")
    print(f"  Unique subject IDs      : {unique_subs}")
    print(f"  Number of subjects      : {len(unique_subs)}")

    for sid in unique_subs:
        count = int((sids == sid).sum())
        print(f"    Subject {sid} : {count} epochs")

    if len(unique_subs) == 3 and unique_subs == [0, 1, 2]:
        print("  [PASS] Exactly 3 subjects (0, 1, 2).")
    else:
        print("  [FAIL] Subject count or IDs do not match expected.")

    print()
    return sids


# ===================================================================
# CHECK 5 — Subject-independent split plan
# ===================================================================
def plan_split(y, sids):
    print("=" * 60)
    print("CHECK 5: SUBJECT-INDEPENDENT SPLIT PLAN")
    print("=" * 60)

    print()
    print("  Strategy: Leave-One-Subject-Out (LOSO)")
    print("  - Training uses 2 subjects, testing uses 1 subject.")
    print("  - Validation is carved from the training subjects.")
    print()
    print("  Recommended assignment (balanced test size):")
    print()
    print("  +----------+-------------------------------------------+")
    print("  |  Split   |  Subjects                                 |")
    print("  +----------+-------------------------------------------+")
    print("  |  Train   |  Subject 0 (SC4001)  +  Subject 1 (SC4002)|")
    print("  |  Val     |  10-15% held out from Train (random)      |")
    print("  |  Test    |  Subject 2 (SC4011)                       |")
    print("  +----------+-------------------------------------------+")
    print()

    # Compute sizes
    train_mask = (sids == 0) | (sids == 1)
    test_mask = sids == 2

    train_y = y[train_mask]
    test_y = y[test_mask]

    # Simulate an 85/15 train/val split from the train pool
    n_train_total = int(train_mask.sum())
    n_val = int(round(0.15 * n_train_total))
    n_train = n_train_total - n_val
    n_test = int(test_mask.sum())

    print(f"  Split sizes (with 85/15 train/val from training pool):")
    print(f"    Train : {n_train:5d} epochs  ({100*n_train/(n_train+n_val+n_test):.1f}%)")
    print(f"    Val   : {n_val:5d} epochs  ({100*n_val/(n_train+n_val+n_test):.1f}%)")
    print(f"    Test  : {n_test:5d} epochs  ({100*n_test/(n_train+n_val+n_test):.1f}%)")
    print(f"    Total : {n_train+n_val+n_test:5d} epochs")
    print()

    # Per-class in each split (approximate for train/val since val is random)
    print("  Test set class distribution (Subject 2 — SC4011):")
    for cls_int in sorted(INT_TO_CLASS.keys()):
        count = int((test_y == cls_int).sum())
        pct = 100.0 * count / len(test_y)
        print(f"    {cls_int} - {INT_TO_CLASS[cls_int]:5s} : {count:5d}  ({pct:5.1f}%)")
    print()

    print("  Training pool class distribution (Subject 0 + 1):")
    for cls_int in sorted(INT_TO_CLASS.keys()):
        count = int((train_y == cls_int).sum())
        pct = 100.0 * count / len(train_y)
        print(f"    {cls_int} - {INT_TO_CLASS[cls_int]:5s} : {count:5d}  ({pct:5.1f}%)")
    print()

    print("  This split ensures NO data leakage between subjects.")
    print("  The model is evaluated on a completely unseen subject.")
    print()


# ===================================================================
# Main
# ===================================================================
def main():
    check_discarded_annotations()
    y = check_valid_labels()
    check_shape()
    sids = check_subject_ids()
    plan_split(y, sids)

    print("=" * 60)
    print("ALL VERIFICATION CHECKS COMPLETE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
