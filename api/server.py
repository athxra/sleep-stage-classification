"""
server.py
=========
FastAPI backend for the React dashboard (web/). It serves the pipeline's
results and runs the trained CNN on Sleep-EDF recordings, reusing the same
epoching / normalization code as training.

Usage:
    python -m uvicorn api.server:app --port 8000
    (after `npm run build` in web/, the dashboard is served at http://localhost:8000)
"""

import glob
import json
import os
import re
import sys
import uuid
import warnings
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (CHANNEL, CLASS_NAMES, CV_METRICS_PATH, EPOCH_SAMPLES, EPOCH_SEC,
                        MODEL_PATH, OUTPUT_DIR, PROC_DIR, RAW_DIR, SAMPLE_DIR, SFREQ,
                        TEST_METRICS_PATH, WAKE_EDGE_MIN, parse_recording, recording_key)
from src.metrics import compute_metrics, sleep_summary
from src.model import count_macs, count_parameters, load_checkpoint, model_size_mb
from src.signals import (UNSCORED, labels_on_grid, load_fpz_cz, segment_epochs,
                         sleep_period_mask, zscore_epochs)

WEB_DIST = os.path.join(os.path.dirname(__file__), "..", "web", "dist")
DEMO_NAME = "demo"
# Compute of the project's first model (3 stride-1 convs at full resolution),
# measured with src.model.count_macs on that architecture.
BASELINE_MACS = 419_200_000
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12), "Sigma": (12, 15), "Beta": (15, 30)}
UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")
UPLOADS = {}   # key -> {"psg", "hypnogram", "name", "channel", "hours"}

app = FastAPI(title="Sleep Stage Classification API")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def read_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_model():
    if not os.path.isfile(MODEL_PATH):
        raise HTTPException(503, "Model not trained yet: run `python src/train.py`.")
    return load_checkpoint(MODEL_PATH, "cpu")


def subject_role(subject):
    _, ckpt = get_model()
    for role in ("test", "val", "train"):
        if subject in ckpt.get(f"{role}_subjects", []):
            return role
    return "unseen"


def raw_recordings():
    """{key: (psg_path, hypnogram_path)} for every PSG in data/raw/."""
    hypnos = {recording_key(p): p for p in glob.glob(os.path.join(RAW_DIR, "SC4*-Hypnogram.edf"))}
    return {recording_key(p): (p, hypnos.get(recording_key(p)))
            for p in sorted(glob.glob(os.path.join(RAW_DIR, "SC4*-PSG.edf")))}


def load_any_eeg(path):
    """
    Load one EEG channel from an arbitrary EDF: Fpz-Cz if present, otherwise
    the first EEG channel, resampled to 100 Hz when needed. Returns (volts, channel).
    """
    import mne

    with warnings.catch_warnings():          # arbitrary EDF headers often trip harmless MNE warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    eeg = [ch for ch in raw.ch_names if "EEG" in ch.upper()] or raw.ch_names
    channel = CHANNEL if CHANNEL in raw.ch_names else eeg[0]
    raw.pick([channel]).load_data(verbose=False)
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)
    return raw.get_data()[0], channel


@lru_cache(maxsize=6)
def load_recording(key):
    """(signal_volts, expert_labels_or_None, info) for a recording key or the demo."""
    if key == DEMO_NAME:
        signal = np.load(os.path.join(SAMPLE_DIR, "demo_eeg.npy")).astype(np.float64)
        labels_path = os.path.join(SAMPLE_DIR, "demo_labels.npy")
        labels = np.load(labels_path) if os.path.isfile(labels_path) else None
        info = read_json(os.path.join(SAMPLE_DIR, "demo_info.json")) or {}
        return signal, labels, info

    import mne

    if key in UPLOADS:
        up = UPLOADS[key]
        signal, channel = load_any_eeg(up["psg"])
        labels = None
        if up["hypnogram"]:
            labels = labels_on_grid(mne.read_annotations(up["hypnogram"]), len(signal) // EPOCH_SAMPLES)
        return signal, labels, {"recording": up["name"], "channel": channel, "uploaded": True}

    recs = raw_recordings()
    if key not in recs:
        raise HTTPException(404, f"Unknown recording {key}")

    psg, hyp = recs[key]
    signal = load_fpz_cz(psg)
    labels = None
    if hyp:
        labels = labels_on_grid(mne.read_annotations(hyp), len(signal) // EPOCH_SAMPLES)
    subject, night = parse_recording(psg)
    return signal, labels, {"subject": subject, "night": night, "recording": os.path.basename(psg)}


@lru_cache(maxsize=6)
def predict_recording(key):
    """Softmax probabilities (n_epochs, 5) for every epoch of a recording."""
    signal, _, _ = load_recording(key)
    model, _ = get_model()
    epochs = zscore_epochs(segment_epochs(signal))
    out = []
    with torch.no_grad():
        for i in range(0, len(epochs), 512):
            out.append(F.softmax(model(torch.from_numpy(epochs[i:i + 512]).unsqueeze(1)), 1).numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASS_NAMES)))


def band_powers(epoch_volts):
    """Relative power (%) in the classic EEG bands for one 30-s epoch."""
    x = epoch_volts - epoch_volts.mean()
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SFREQ)
    power = {name: float(spectrum[(freqs >= lo) & (freqs < hi)].sum()) for name, (lo, hi) in BANDS.items()}
    total = sum(power.values()) or 1.0
    return {name: round(100 * p / total, 2) for name, p in power.items()}


def spectrum_db(epoch_volts, fmax=30.0):
    """Welch power spectral density (dB re 1 µV²/Hz) of one epoch, 0.5–fmax Hz."""
    seg = 4 * SFREQ                                   # 4-s Hann windows, 50% overlap
    x = (epoch_volts - epoch_volts.mean()) * 1e6
    win = np.hanning(seg)
    starts = range(0, len(x) - seg + 1, seg // 2)
    psd = np.mean([np.abs(np.fft.rfft(x[s:s + seg] * win)) ** 2 for s in starts], axis=0)
    psd /= SFREQ * (win ** 2).sum()
    freqs = np.fft.rfftfreq(seg, 1 / SFREQ)
    keep = (freqs >= 0.5) & (freqs <= fmax)
    return freqs[keep].round(2).tolist(), (10 * np.log10(psd[keep] + 1e-12)).round(2).tolist()


def jsonable_summary(summary):
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary.items()}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/summary")
def summary():
    model, ckpt = get_model()
    split = read_json(os.path.join(PROC_DIR, "split_info.json"))
    return {
        "classes": CLASS_NAMES,
        "test": read_json(TEST_METRICS_PATH),
        "cv": read_json(CV_METRICS_PATH),
        "split": None if split is None else {
            "n_epochs": split["n_epochs"],
            "n_subjects": split["n_subjects"],
            "n_recordings": len(split["recordings"]),
            "class_counts": split["class_counts"],
            "wake_edge_min": split["wake_edge_min"],
            "folds": split["folds"],
        },
        "model": {
            "parameters": count_parameters(model),
            "size_mb": round(model_size_mb(model), 3),
            "macs_per_epoch": count_macs(model),
            "baseline_macs_per_epoch": BASELINE_MACS,
            "checkpoint_epoch": ckpt["epoch"],
            "val_f1": ckpt["val_f1"],
            "train_subjects": ckpt["train_subjects"],
            "val_subjects": ckpt["val_subjects"],
            "test_subjects": ckpt["test_subjects"],
            "hyperparameters": ckpt.get("hyperparameters", {}),
        },
    }


@app.get("/api/recordings")
def recordings():
    split = read_json(os.path.join(PROC_DIR, "split_info.json")) or {"recordings": []}
    grid = {recording_key(r["psg"]): r["grid_epochs"] for r in split["recordings"]}
    items = []
    for key, (psg, hyp) in raw_recordings().items():
        subject, night = parse_recording(psg)
        items.append({
            "key": key, "label": key, "subject": subject, "night": night,
            "role": subject_role(subject), "has_labels": hyp is not None,
            "hours": round(grid[key] * EPOCH_SEC / 3600, 1) if key in grid else None,
        })
    for key, up in UPLOADS.items():
        items.append({"key": key, "label": up["name"], "subject": None, "night": None,
                      "role": "unseen", "has_labels": up["hypnogram"] is not None,
                      "hours": up.get("hours"), "uploaded": True, "channel": up.get("channel")})
    if not any(not it.get("uploaded") for it in items):
        info = read_json(os.path.join(SAMPLE_DIR, "demo_info.json")) or {}
        signal, labels, _ = load_recording(DEMO_NAME)
        items.append({
            "key": DEMO_NAME, "label": "Demo sample", "subject": info.get("subject"),
            "night": None, "role": subject_role(info.get("subject")),
            "has_labels": labels is not None, "demo": True, "source": info.get("source"),
            "hours": round(len(signal) / SFREQ / 3600, 1),
        })
    return {"demo_mode": any(it.get("demo") for it in items), "recordings": items}


@app.get("/api/recordings/{key}/night")
def night(key: str):
    signal, labels, info = load_recording(key)
    probs = predict_recording(key)
    predicted = probs.argmax(axis=1)
    n = len(predicted)
    expert = labels[:n] if labels is not None else None

    window = sleep_period_mask(expert) if expert is not None else np.ones(n, dtype=bool)
    if not window.any():
        window = np.ones(n, dtype=bool)
    idx = np.flatnonzero(window)
    start, stop = int(idx[0]), int(idx[-1]) + 1

    result = {
        "key": key,
        "info": info,
        "role": subject_role(info.get("subject")) if info.get("subject") is not None else None,
        "epoch_sec": EPOCH_SEC,
        "n_epochs": n,
        "sleep_window": [start, stop],
        "wake_edge_min": WAKE_EDGE_MIN,
        "predicted": predicted.tolist(),
        "confidence": probs.max(axis=1).round(3).tolist(),
        "probs": probs.round(3).tolist(),
        "expert": expert.tolist() if expert is not None else None,
        "summary_predicted": jsonable_summary(sleep_summary(predicted[start:stop])),
        "summary_expert": None,
        "agreement": None,
    }
    if expert is not None:
        exp_w, pred_w = expert[start:stop], predicted[start:stop]
        scored = exp_w != UNSCORED
        m = compute_metrics(exp_w[scored], pred_w[scored])
        result["agreement"] = {k: m[k] for k in ("accuracy", "macro_f1", "cohen_kappa", "n_epochs")}
        result["agreement"]["per_class_f1"] = {c: m["per_class"][c]["f1"] for c in CLASS_NAMES}
        result["summary_expert"] = jsonable_summary(sleep_summary(np.where(exp_w == UNSCORED, 0, exp_w)))
    return result


@app.get("/api/recordings/{key}/epoch/{index}")
def epoch(key: str, index: int):
    signal, labels, _ = load_recording(key)
    n = len(signal) // EPOCH_SAMPLES
    if not 0 <= index < n:
        raise HTTPException(404, f"Epoch {index} out of range (0-{n - 1})")
    segment = signal[index * EPOCH_SAMPLES:(index + 1) * EPOCH_SAMPLES]
    probs = predict_recording(key)[index]
    return {
        "index": index,
        "start_sec": index * EPOCH_SEC,
        "sfreq": SFREQ,
        "signal_uv": (segment * 1e6).round(2).tolist(),
        "probs": probs.round(4).tolist(),
        "predicted": int(probs.argmax()),
        "expert": int(labels[index]) if labels is not None and index < len(labels) else None,
        "bands": band_powers(segment),
        "psd": dict(zip(("freqs", "db"), spectrum_db(segment))),
    }


@app.get("/api/examples")
def examples(true: str, pred: str, limit: int = 24):
    """
    Test-set epochs (deployed model) where the expert said `true` and the CNN
    said `pred`, spread across recordings: drill-down from the confusion matrix.
    """
    path = os.path.join(OUTPUT_DIR, "test_predictions.npz")
    split = read_json(os.path.join(PROC_DIR, "split_info.json"))
    if not os.path.isfile(path) or split is None:
        raise HTTPException(404, "Run `python src/evaluate.py` first.")
    if true not in CLASS_NAMES or pred not in CLASS_NAMES:
        raise HTTPException(400, "Unknown stage name.")
    d = np.load(path)
    hit = np.flatnonzero((d["y_true"] == CLASS_NAMES.index(true)) & (d["y_pred"] == CLASS_NAMES.index(pred)))
    keys = {r["id"]: recording_key(r["psg"]) for r in split["recordings"]}
    pick = hit[np.linspace(0, len(hit) - 1, min(limit, len(hit))).astype(int)] if len(hit) else hit
    return {
        "true": true, "pred": pred, "total": int(len(hit)),
        "examples": [{"key": keys[int(d["recording_ids"][i])], "epoch": int(d["epoch_index"][i])}
                     for i in pick],
    }


def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name or "file.edf"))


@app.post("/api/upload")
async def upload(psg: UploadFile = File(...), hypnogram: UploadFile | None = File(None)):
    """Stage a user-supplied EDF recording (optionally with a Sleep-EDF style hypnogram)."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    key = f"upload-{uuid.uuid4().hex[:8]}"
    paths = {}
    for field, f in (("psg", psg), ("hypnogram", hypnogram)):
        if f is None or not f.filename:
            paths[field] = None
            continue
        if not f.filename.lower().endswith(".edf"):
            raise HTTPException(400, f"{f.filename} is not an .edf file.")
        dest = os.path.join(UPLOAD_DIR, f"{key}-{_safe_name(f.filename)}")
        with open(dest, "wb") as out:
            while chunk := await f.read(1 << 20):
                out.write(chunk)
        paths[field] = dest

    UPLOADS[key] = {"psg": paths["psg"], "hypnogram": paths["hypnogram"], "name": psg.filename}
    try:
        signal, labels, info = load_recording(key)
        if len(signal) < EPOCH_SAMPLES:
            raise ValueError("recording is shorter than one 30-second epoch")
        if paths["hypnogram"] and not (labels != UNSCORED).any():
            raise ValueError("the hypnogram has no Sleep-EDF stage annotations")
    except Exception as exc:
        UPLOADS.pop(key, None)
        load_recording.cache_clear()
        for path in paths.values():
            if path and os.path.isfile(path):
                os.remove(path)
        raise HTTPException(400, f"Could not read this EDF: {exc}")
    UPLOADS[key].update(hours=round(len(signal) / SFREQ / 3600, 1), channel=info.get("channel"))
    return {"key": key, "name": psg.filename, "channel": info.get("channel"),
            "hours": UPLOADS[key]["hours"], "has_labels": labels is not None}


# ---------------------------------------------------------------------------
# Static frontend (production build)
# ---------------------------------------------------------------------------
if os.path.isdir(WEB_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIST, "assets")), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        target = os.path.join(WEB_DIST, path)
        if path and os.path.isfile(target):
            return FileResponse(target)
        return FileResponse(os.path.join(WEB_DIST, "index.html"))
