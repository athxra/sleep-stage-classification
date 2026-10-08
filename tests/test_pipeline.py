"""Unit tests for the preprocessing, splitting, model and metrics code."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CLASS_TO_INT, EPOCH_SAMPLES, parse_recording, recording_key
from src.metrics import compute_metrics, sleep_summary, transition_mask
from src.model import SleepCNN, count_macs, count_parameters
from src.signals import (UNSCORED, labels_on_grid, segment_epochs, sleep_period_mask,
                         zscore_epochs)
from src.splits import fold_indices, make_folds

W, N1, N2, N3, R = (CLASS_TO_INT[c] for c in ("Wake", "N1", "N2", "N3", "REM"))


def annotations(*rows):
    onset, duration, desc = zip(*rows)
    return SimpleNamespace(onset=np.array(onset, float), duration=np.array(duration, float),
                           description=np.array(desc))


# ---------------------------------------------------------------------------
# File naming
# ---------------------------------------------------------------------------
def test_parse_recording_uses_subject_and_night():
    assert parse_recording("SC4021E0-PSG.edf") == (2, 1)
    assert parse_recording("SC4022EJ-Hypnogram.edf") == (2, 2)
    assert parse_recording("/any/dir/SC4192E0-PSG.edf") == (19, 2)
    assert recording_key("SC4012E0-PSG.edf") == recording_key("SC4012EC-Hypnogram.edf")


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def test_segment_drops_partial_epoch():
    epochs = segment_epochs(np.arange(EPOCH_SAMPLES * 3 + 100, dtype=float))
    assert epochs.shape == (3, EPOCH_SAMPLES)
    assert epochs[1, 0] == EPOCH_SAMPLES


def test_zscore_per_epoch_and_flat_epochs():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(5, 3, EPOCH_SAMPLES), np.full(EPOCH_SAMPLES, 7.0)])
    z = zscore_epochs(x)
    assert z.dtype == np.float32
    assert abs(z[0].mean()) < 1e-5 and abs(z[0].std() - 1) < 1e-4
    assert np.all(z[1] == 0)


def test_labels_on_grid_maps_merges_and_discards():
    ann = annotations(
        (0, 60, "Sleep stage W"),
        (60, 30, "Sleep stage 3"),
        (90, 30, "Sleep stage 4"),        # S4 merges into N3
        (120, 30, "Movement time"),       # discarded
        (150, 30, "Sleep stage R"),
        (180, 30, "Sleep stage ?"),       # discarded
    )
    labels = labels_on_grid(ann, n_epochs=8)
    assert labels.tolist() == [W, W, N3, N3, UNSCORED, R, UNSCORED, UNSCORED]


def test_sleep_period_mask_trims_distant_wake():
    labels = np.array([W] * 200 + [N2] * 10 + [W] * 200)
    mask = sleep_period_mask(labels, edge_min=30)   # 60 epochs of Wake each side
    assert mask.sum() == 60 + 10 + 60
    assert mask[200 - 60] and not mask[200 - 61]
    assert mask[209 + 60] and not mask[209 + 61]


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def test_folds_are_subject_disjoint_and_cover_every_subject_once():
    subjects = list(range(20))
    folds = make_folds(subjects, n_folds=5, n_val_subjects=2)
    tested = []
    for fold in folds:
        train, val, test = map(set, (fold["train"], fold["val"], fold["test"]))
        assert not (train & val or train & test or val & test)
        assert train | val | test == set(subjects)
        assert len(val) == 2
        tested += fold["test"]
    assert sorted(tested) == subjects


def test_both_nights_of_a_subject_share_a_split():
    # Two nights per subject (e.g. SC4021 + SC4022 -> subject 2).
    subject_ids = np.repeat([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], 50)
    for fold in make_folds(subject_ids, n_folds=5, n_val_subjects=1):
        train_idx, val_idx, test_idx = fold_indices(subject_ids, fold)
        assert not set(subject_ids[test_idx]) & set(subject_ids[val_idx])
        assert not set(subject_ids[test_idx]) & set(subject_ids[train_idx])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def test_model_output_shape_and_is_lightweight():
    model = SleepCNN(n_classes=5).eval()
    with torch.no_grad():
        out = model(torch.randn(3, 1, EPOCH_SAMPLES))
    assert out.shape == (3, 5)
    assert count_parameters(model) < 200_000
    assert count_macs(model) < 10_000_000


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_transition_mask_respects_recordings_and_gaps():
    y = np.array([N2, N2, N3, N3, N3, N3])
    rec = np.array([0, 0, 0, 1, 1, 1])
    idx = np.array([0, 1, 2, 0, 1, 5])   # gap between the last two epochs
    assert transition_mask(y, rec, idx).tolist() == [False, True, True, False, False, False]


def test_compute_metrics_perfect_prediction():
    y = np.array([W, N1, N2, N3, R, N2, N2])
    m = compute_metrics(y, y, np.zeros(len(y)), np.arange(len(y)))
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["cohen_kappa"] == 1.0
    assert m["top_confusions"] == []
    t = m["transition_analysis"]
    assert t["near_transition"]["n_epochs"] + t["stable"]["n_epochs"] == len(y)


def test_sleep_summary():
    stages = [W, W, N1, N2, W, N2, R, N3, W]      # 30-s epochs
    s = sleep_summary(stages)
    assert s["time_in_bed_min"] == pytest.approx(4.5)
    assert s["total_sleep_time_min"] == pytest.approx(2.5)
    assert s["sleep_onset_latency_min"] == pytest.approx(1.0)
    assert s["waso_min"] == pytest.approx(0.5)
    assert s["rem_latency_min"] == pytest.approx(2.0)
