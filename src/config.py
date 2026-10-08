"""
config.py
=========
Single source of truth for paths, signal constants, and label definitions
shared by preprocessing, training, evaluation, and the dashboard.
"""

import os
import re

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROC_DIR = os.path.join(ROOT_DIR, "data", "processed")
SAMPLE_DIR = os.path.join(ROOT_DIR, "data", "sample")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
TEST_METRICS_PATH = os.path.join(OUTPUT_DIR, "test_metrics.json")
CV_METRICS_PATH = os.path.join(OUTPUT_DIR, "cv_metrics.json")

# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
CHANNEL = "EEG Fpz-Cz"
SFREQ = 100                          # Hz
EPOCH_SEC = 30                       # seconds (AASM scoring unit)
EPOCH_SAMPLES = SFREQ * EPOCH_SEC    # 3000

# Wake epochs kept on either side of the sleep period. Sleep-EDF cassette
# recordings span ~20 h, so untrimmed data is dominated by daytime Wake.
# 30 min is the convention used by DeepSleepNet / TinySleepNet / AttnSleep.
WAKE_EDGE_MIN = 30

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "Sleep stage W": "Wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",   # AASM: merge S3+S4 -> N3
    "Sleep stage R": "REM",
}

CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
CLASS_TO_INT = {name: i for i, name in enumerate(CLASS_NAMES)}
INT_TO_CLASS = dict(enumerate(CLASS_NAMES))
N_CLASSES = len(CLASS_NAMES)

SEED = 42

# ---------------------------------------------------------------------------
# Sleep-EDF file naming
# ---------------------------------------------------------------------------
# Sleep-Cassette files are named SC4<ss><n>...: <ss> is the SUBJECT number and
# <n> the NIGHT (1 or 2). SC4021 and SC4022 are two nights of the same person,
# so splits must group by subject, not by recording.
_SC_PATTERN = re.compile(r"^SC4(\d{2})(\d)")


def parse_recording(filename: str):
    """Return (subject, night) parsed from a Sleep-Cassette filename."""
    match = _SC_PATTERN.match(os.path.basename(filename))
    if match is None:
        raise ValueError(f"Not a Sleep-Cassette filename: {filename}")
    return int(match.group(1)), int(match.group(2))


def recording_key(filename: str) -> str:
    """'SC4012E0-PSG.edf' -> 'SC4012' (shared by a PSG and its hypnogram)."""
    return os.path.basename(filename)[:6]
