"""
signals.py
==========
EEG epoching, labelling, and normalization shared by preprocessing and the
dashboard, so training-time and inference-time inputs are built identically.

Epochs live on a fixed grid: epoch k covers samples [k*3000, (k+1)*3000) of the
recording. Expert labels are attached to that same grid by looking up which
hypnogram annotation covers each epoch's midpoint.
"""

import numpy as np

from src.config import (CHANNEL, CLASS_TO_INT, EPOCH_SAMPLES, EPOCH_SEC,
                        LABEL_MAP, SFREQ, WAKE_EDGE_MIN)

UNSCORED = -1


def segment_epochs(signal: np.ndarray) -> np.ndarray:
    """Split a 1-D signal into (n_epochs, 3000); a trailing partial epoch is dropped."""
    n_epochs = len(signal) // EPOCH_SAMPLES
    return signal[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)


def zscore_epochs(epochs: np.ndarray) -> np.ndarray:
    """Per-epoch Z-score normalization, returned as float32."""
    epochs = np.asarray(epochs, dtype=np.float64)
    means = epochs.mean(axis=-1, keepdims=True)
    stds = epochs.std(axis=-1, keepdims=True)
    stds[stds < 1e-12] = 1.0
    return ((epochs - means) / stds).astype(np.float32)


def labels_on_grid(annotations, n_epochs: int) -> np.ndarray:
    """
    Integer stage label for each grid epoch, or UNSCORED (-1) where the
    hypnogram says "Sleep stage ?", "Movement time", or has no annotation.
    """
    labels = np.full(n_epochs, UNSCORED, dtype=np.int64)
    midpoints = (np.arange(n_epochs) + 0.5) * EPOCH_SEC
    for onset, duration, desc in zip(annotations.onset, annotations.duration,
                                     annotations.description):
        if desc not in LABEL_MAP:
            continue
        covered = (midpoints >= onset) & (midpoints < onset + duration)
        labels[covered] = CLASS_TO_INT[LABEL_MAP[desc]]
    return labels


def sleep_period_mask(labels: np.ndarray, edge_min: int = WAKE_EDGE_MIN) -> np.ndarray:
    """
    Mask keeping the sleep period plus `edge_min` minutes of Wake on each side.
    Sleep-EDF cassette recordings run ~20 h, mostly daytime Wake; without this
    trim, accuracy is dominated by trivially easy Wake epochs.
    """
    mask = np.zeros(len(labels), dtype=bool)
    sleep_idx = np.flatnonzero((labels != UNSCORED) & (labels != CLASS_TO_INT["Wake"]))
    if len(sleep_idx) == 0:
        return mask
    edge = edge_min * 60 // EPOCH_SEC
    start = max(0, sleep_idx[0] - edge)
    stop = min(len(labels), sleep_idx[-1] + edge + 1)
    mask[start:stop] = True
    return mask


def load_fpz_cz(psg_path: str) -> np.ndarray:
    """Load the Fpz-Cz channel of a Sleep-EDF PSG file (volts, 100 Hz)."""
    import mne

    raw = mne.io.read_raw_edf(psg_path, include=[CHANNEL], preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    if sfreq != SFREQ:
        raise ValueError(f"Expected {SFREQ} Hz, got {sfreq} Hz in {psg_path}")
    return raw.get_data()[0]
