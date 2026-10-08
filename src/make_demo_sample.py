"""
make_demo_sample.py
===================
Build the dashboard's demo sample from a held-out TEST subject of the
deployed model, so demo-mode predictions are honest unseen-subject results.

Writes:
  data/sample/demo_eeg.npy     Fpz-Cz signal in volts (float32, 100 Hz)
  data/sample/demo_labels.npy  expert stage per 30-s epoch (-1 = unscored)
  data/sample/demo_info.json   source recording and time offset

Usage:
    python src/make_demo_sample.py [--hours 4]
"""

import argparse
import json
import os
import sys

import mne
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import EPOCH_SAMPLES, EPOCH_SEC, MODEL_PATH, SAMPLE_DIR, parse_recording
from src.model import load_checkpoint
from src.preprocess import discover_recordings
from src.signals import labels_on_grid, load_fpz_cz, sleep_period_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=4.0, help="length of the excerpt")
    args = parser.parse_args()

    _, ckpt = load_checkpoint(MODEL_PATH)
    test_subjects = ckpt["test_subjects"]
    pairs = [p for p in discover_recordings() if parse_recording(p[0])[0] in test_subjects]
    if not pairs:
        raise SystemExit(f"No raw recordings found for test subjects {test_subjects}.")
    psg, hyp = pairs[0]

    signal = load_fpz_cz(psg)
    labels = labels_on_grid(mne.read_annotations(hyp), len(signal) // EPOCH_SAMPLES)
    start = int(np.flatnonzero(sleep_period_mask(labels))[0])   # 30 min before sleep onset
    n_epochs = int(args.hours * 3600 // EPOCH_SEC)
    stop = min(start + n_epochs, len(labels))

    os.makedirs(SAMPLE_DIR, exist_ok=True)
    np.save(os.path.join(SAMPLE_DIR, "demo_eeg.npy"),
            signal[start * EPOCH_SAMPLES:stop * EPOCH_SAMPLES].astype(np.float32))
    np.save(os.path.join(SAMPLE_DIR, "demo_labels.npy"), labels[start:stop])
    info = {
        "source": f"{os.path.basename(psg)} (test subject {parse_recording(psg)[0]}), "
                  f"epochs {start}-{stop - 1}, starting 30 min before sleep onset",
        "recording": os.path.basename(psg),
        "subject": parse_recording(psg)[0],
        "start_epoch": start,
        "n_epochs": stop - start,
    }
    with open(os.path.join(SAMPLE_DIR, "demo_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"Demo sample written: {info['source']}")


if __name__ == "__main__":
    main()
