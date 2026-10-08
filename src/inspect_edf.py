"""
inspect_edf.py
==============
Inspect the raw Sleep-EDF PSG and Hypnogram EDF files.

Usage:
    python src/inspect_edf.py [RECORDING]      e.g. SC4001 (default)
"""

import glob
import os
import sys

import mne


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import CHANNEL as PREFERRED_CHANNEL
from src.config import RAW_DIR


def find_files(key: str):
    """PSG and hypnogram paths for a recording key such as 'SC4001'."""
    psg = glob.glob(os.path.join(RAW_DIR, f"{key}*-PSG.edf"))
    hyp = glob.glob(os.path.join(RAW_DIR, f"{key}*-Hypnogram.edf"))
    return (psg[0] if psg else os.path.join(RAW_DIR, f"{key}E0-PSG.edf"),
            hyp[0] if hyp else os.path.join(RAW_DIR, f"{key}-Hypnogram.edf"))


def inspect_psg(filepath: str) -> None:
    """Load the PSG EDF and print channel / recording metadata."""

    print("=" * 60)
    print("PSG FILE INSPECTION")
    print("=" * 60)
    print(f"File : {os.path.basename(filepath)}")
    print()

    raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)

    # 1. Channel names
    ch_names = raw.ch_names
    print(f"Number of channels : {len(ch_names)}")
    print("Channel names:")
    for i, name in enumerate(ch_names, 1):
        print(f"  {i}. {name}")
    print()

    # 2. Sampling frequency
    sfreq = raw.info["sfreq"]
    print(f"Sampling frequency : {sfreq} Hz")

    # 3. Recording duration
    duration_sec = raw.n_times / sfreq
    duration_min = duration_sec / 60
    duration_hr = duration_min / 60
    print(f"Total samples      : {raw.n_times}")
    print(f"Recording duration : {duration_sec:.1f} s  "
          f"({duration_min:.1f} min / {duration_hr:.2f} hr)")
    print()

    # 4. Check for preferred channel
    if PREFERRED_CHANNEL in ch_names:
        print(f"[OK] Preferred channel '{PREFERRED_CHANNEL}' is AVAILABLE.")
    else:
        print(f"[MISSING] Preferred channel '{PREFERRED_CHANNEL}' NOT found.")
        # Show partial matches as a hint
        matches = [ch for ch in ch_names if "Fpz" in ch or "fpz" in ch.lower()]
        if matches:
            print(f"  Possible matches: {matches}")

    print()


def inspect_hypnogram(filepath: str) -> None:
    """Load the Hypnogram EDF and print annotation metadata."""

    print("=" * 60)
    print("HYPNOGRAM FILE INSPECTION")
    print("=" * 60)
    print(f"File : {os.path.basename(filepath)}")
    print()

    annotations = mne.read_annotations(filepath)

    # 5. Total number of annotations
    print(f"Total annotations : {len(annotations)}")
    print()

    # 6. Unique sleep-stage labels
    unique_labels = sorted(set(annotations.description))
    print(f"Unique labels ({len(unique_labels)}):")
    for label in unique_labels:
        count = list(annotations.description).count(label)
        print(f"  • {label:25s} — {count} epochs")
    print()

    # 7. Show the first few annotations as a sanity check
    n_preview = min(10, len(annotations))
    print(f"First {n_preview} annotations (onset, duration, label):")
    for i in range(n_preview):
        onset = annotations.onset[i]
        duration = annotations.duration[i]
        desc = annotations.description[i]
        print(f"  {i+1:3d}.  onset={onset:>10.1f}s  "
              f"duration={duration:>6.1f}s  label={desc}")
    print()


def main() -> None:
    PSG_FILE, HYPNO_FILE = find_files(sys.argv[1] if len(sys.argv) > 1 else "SC4001")
    # Verify files exist before attempting to load
    for tag, path in [("PSG", PSG_FILE), ("Hypnogram", HYPNO_FILE)]:
        if not os.path.isfile(path):
            print(f"ERROR: {tag} file not found at:\n  {os.path.abspath(path)}")
            print("Run `python src/download_data.py` to fetch the Sleep-EDF files.")
            return

    inspect_psg(PSG_FILE)
    inspect_hypnogram(HYPNO_FILE)

    print("=" * 60)
    print("Inspection complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
