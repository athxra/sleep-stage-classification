"""
splits.py
=========
Subject-grouped cross-validation folds.

Every Sleep-EDF subject contributes up to two nights; both nights always land
in the same split, so the test subjects are genuinely unseen individuals and
model selection (validation) never touches a test subject.
"""

import numpy as np

from src.config import SEED


def make_folds(subjects, n_folds: int = 5, n_val_subjects: int = 2, seed: int = SEED):
    """
    Return a list of dicts {"train", "val", "test"} of sorted subject IDs.

    Subjects are shuffled once, divided into `n_folds` disjoint test groups,
    and for each fold `n_val_subjects` of the remaining subjects are held out
    for validation / checkpoint selection.
    """
    subjects = np.array(sorted(set(int(s) for s in subjects)))
    if len(subjects) < 3:
        raise ValueError("Need at least 3 subjects for train/val/test splits.")
    n_folds = min(n_folds, len(subjects) - 2)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(subjects)
    test_groups = np.array_split(shuffled, n_folds)

    folds = []
    for k, test in enumerate(test_groups):
        rest = np.setdiff1d(shuffled, test)
        n_val = max(1, min(n_val_subjects, len(rest) - 1))
        val = np.random.default_rng(seed + k).choice(rest, size=n_val, replace=False)
        train = np.setdiff1d(rest, val)
        folds.append({
            "train": sorted(int(s) for s in train),
            "val": sorted(int(s) for s in val),
            "test": sorted(int(s) for s in test),
        })
    return folds


def fold_indices(subject_ids: np.ndarray, fold: dict):
    """Epoch indices for the train / val / test subjects of one fold."""
    return tuple(np.flatnonzero(np.isin(subject_ids, fold[part]))
                 for part in ("train", "val", "test"))
