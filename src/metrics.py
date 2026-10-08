"""
metrics.py
==========
Evaluation metrics, stage-transition-aware error analysis, and clinical
sleep summary statistics.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from src.config import CLASS_NAMES, CLASS_TO_INT, EPOCH_SEC, N_CLASSES

LABELS = list(range(N_CLASSES))


def transition_mask(y_true, recording_ids, epoch_index):
    """
    True for epochs adjacent to an expert-scored stage change: the epoch's label
    differs from the previous or next epoch of the same recording. Neighbours
    must be contiguous on the epoch grid (no discarded epoch in between).
    """
    y_true = np.asarray(y_true)
    rec = np.asarray(recording_ids)
    idx = np.asarray(epoch_index)
    order = np.lexsort((idx, rec))
    y, r, e = y_true[order], rec[order], idx[order]

    contiguous = (r[1:] == r[:-1]) & (e[1:] == e[:-1] + 1)
    changed = contiguous & (y[1:] != y[:-1])

    sorted_mask = np.zeros(len(y), dtype=bool)
    sorted_mask[1:] |= changed     # differs from previous epoch
    sorted_mask[:-1] |= changed    # differs from next epoch

    mask = np.empty_like(sorted_mask)
    mask[order] = sorted_mask
    return mask


def _summary(y_true, y_pred):
    if len(y_true) == 0:
        return {"n_epochs": 0}
    return {
        "n_epochs": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro",
                                   zero_division=0)),
    }


def compute_metrics(y_true, y_pred, recording_ids=None, epoch_index=None) -> dict:
    """All evaluation metrics as a JSON-serialisable dict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)

    results = {
        **_summary(y_true, y_pred),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=LABELS)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.round(4).tolist(),
        "majority_class_baseline": float(np.bincount(y_true, minlength=N_CLASSES).max()
                                         / max(len(y_true), 1)),
    }

    # Most frequent off-diagonal confusions, as a share of the true class.
    confusions = [
        {"true": CLASS_NAMES[i], "predicted": CLASS_NAMES[j],
         "count": int(cm[i, j]), "rate": float(cm_norm[i, j])}
        for i in LABELS for j in LABELS if i != j and cm[i, j] > 0
    ]
    results["top_confusions"] = sorted(confusions, key=lambda c: -c["count"])[:6]

    # N1 / REM focus: the adjacent-stage confusions highlighted in the project brief.
    n1, rem = CLASS_TO_INT["N1"], CLASS_TO_INT["REM"]
    results["n1_rem_focus"] = {
        "n1_recall": float(recall[n1]),
        "rem_recall": float(recall[rem]),
        "n1_predicted_as": {CLASS_NAMES[j]: float(cm_norm[n1, j]) for j in LABELS if j != n1},
        "rem_predicted_as": {CLASS_NAMES[j]: float(cm_norm[rem, j]) for j in LABELS if j != rem},
    }

    if recording_ids is not None and epoch_index is not None:
        near = transition_mask(y_true, recording_ids, epoch_index)
        results["transition_analysis"] = {
            "near_transition": _summary(y_true[near], y_pred[near]),
            "stable": _summary(y_true[~near], y_pred[~near]),
        }

    return results


# ---------------------------------------------------------------------------
# Clinical sleep summary (hypnogram page)
# ---------------------------------------------------------------------------
def sleep_summary(stages, epoch_sec: int = EPOCH_SEC) -> dict:
    """
    Standard sleep-architecture statistics from a sequence of stage labels.
    Times are in minutes; efficiency and stage shares in percent.
    """
    stages = np.asarray(stages)
    wake = CLASS_TO_INT["Wake"]
    minutes = epoch_sec / 60.0
    asleep = np.flatnonzero(stages != wake)

    summary = {
        "time_in_bed_min": len(stages) * minutes,
        "total_sleep_time_min": len(asleep) * minutes,
        "sleep_efficiency_pct": 100.0 * len(asleep) / max(len(stages), 1),
        "sleep_onset_latency_min": None,
        "rem_latency_min": None,
        "waso_min": None,
        "stage_minutes": {name: float((stages == i).sum() * minutes)
                          for i, name in enumerate(CLASS_NAMES)},
    }
    if len(asleep):
        onset, final = asleep[0], asleep[-1]
        summary["sleep_onset_latency_min"] = onset * minutes
        summary["waso_min"] = float((stages[onset:final + 1] == wake).sum() * minutes)
        rem_idx = np.flatnonzero(stages[onset:] == CLASS_TO_INT["REM"])
        if len(rem_idx):
            summary["rem_latency_min"] = rem_idx[0] * minutes
        sleep_total = max(len(asleep), 1)
        summary["stage_pct_of_sleep"] = {
            name: 100.0 * float((stages == i).sum()) / sleep_total
            for i, name in enumerate(CLASS_NAMES) if i != wake
        }
    return summary
