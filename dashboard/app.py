"""
Sleep Stage Classification Dashboard
=====================================
Streamlit dashboard for the Sleep Stage Classification from EEG Signals
project using a Lightweight 1D CNN.

All numbers shown are read from the files written by the pipeline
(outputs/test_metrics.json, outputs/cv_metrics.json, models/best_model.pth),
and inference uses the same epoching / normalization code as training.

Usage:
    streamlit run dashboard/app.py
"""

import glob
import json
import os
import sys
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (CHANNEL, CLASS_NAMES, CV_METRICS_PATH, EPOCH_SAMPLES, EPOCH_SEC,
                        MODEL_PATH, RAW_DIR, SAMPLE_DIR, SFREQ, TEST_METRICS_PATH,
                        parse_recording, recording_key)
from src.metrics import compute_metrics, sleep_summary
from src.model import count_macs, count_parameters, load_checkpoint, model_size_mb
from src.signals import (UNSCORED, labels_on_grid, segment_epochs, sleep_period_mask,
                         zscore_epochs)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sleep Stage Classification from EEG Signals",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---------- sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }

    /* ---------- metric cards ---------- */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.35);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #7c83ff;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0a0b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ---------- stage badges ---------- */
    .stage-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.4rem;
    }
    .stage-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #fff;
    }
    .badge-wake { background: #4fc3f7; }
    .badge-n1   { background: #7c83ff; }
    .badge-n2   { background: #ab47bc; }
    .badge-n3   { background: #5c6bc0; }
    .badge-rem  { background: #ef5350; }

    /* ---------- section header ---------- */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #7c83ff;
        margin-bottom: 0.6rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid rgba(124,131,255,0.3);
    }

    /* ---------- hero ---------- */
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #7c83ff, #ab47bc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #9e9eb8;
        margin-bottom: 1.5rem;
    }

    /* ---------- pipeline step ---------- */
    .pipeline-step {
        background: rgba(124,131,255,0.08);
        border-left: 3px solid #7c83ff;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 SleepNet")
    st.caption("Lightweight CNN for EEG Sleep Staging")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        options=[
            "Overview",
            "EEG Signal Viewer",
            "Sleep Stage Prediction",
            "Overnight Hypnogram",
            "Model Performance",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Minor Project • 2026")


CLASS_COLORS = ["#4fc3f7", "#7c83ff", "#ab47bc", "#5c6bc0", "#ef5350"]
# Hypnogram y-axis order: Wake on top, then REM, N1, N2, N3 (clinical convention).
HYPNO_LEVEL = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
HYPNO_TICKS = dict(tickmode="array", tickvals=[4, 3, 2, 1, 0],
                   ticktext=["Wake", "REM", "N1", "N2", "N3"])
DEMO_NAME = "Demo sample"
PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(30,30,47,0.6)",
    margin=dict(l=60, r=20, t=30, b=50),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    hoverlabel=dict(bgcolor="#2d2d44"),
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def metric_cards(items, value_size=None):
    style = f' style="font-size:{value_size}"' if value_size else ""
    for col, (value, label) in zip(st.columns(len(items)), items):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value"{style}>{value}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )


def header(title, subtitle):
    st.markdown(f'<div class="hero-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt_min(value):
    return "—" if value is None else f"{value:.0f} min"


@st.cache_resource(show_spinner="Loading PyTorch model...")
def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return load_checkpoint(MODEL_PATH, "cpu")


def list_recordings():
    """Available recordings: raw PSG files if present, else the bundled demo sample."""
    hypnos = {recording_key(p): p for p in glob.glob(os.path.join(RAW_DIR, "SC4*-Hypnogram.edf"))}
    recs = {os.path.basename(p): (p, hypnos.get(recording_key(p)))
            for p in sorted(glob.glob(os.path.join(RAW_DIR, "SC4*-PSG.edf")))}
    return recs or {DEMO_NAME: (None, None)}


@st.cache_data(show_spinner="Loading EEG recording...", max_entries=4)
def load_recording(name, psg_path, hyp_path):
    """
    Return (signal_volts, expert_labels_or_None, info). Expert labels are on
    the same 30-s grid used for inference, so they align epoch-for-epoch.
    """
    if psg_path is None:
        signal = np.load(os.path.join(SAMPLE_DIR, "demo_eeg.npy")).astype(np.float64)
        labels_path = os.path.join(SAMPLE_DIR, "demo_labels.npy")
        labels = np.load(labels_path) if os.path.isfile(labels_path) else None
        info = load_json(os.path.join(SAMPLE_DIR, "demo_info.json")) or {}
        return signal, labels, info

    import mne
    from src.signals import load_fpz_cz

    signal = load_fpz_cz(psg_path)
    labels = None
    if hyp_path:
        labels = labels_on_grid(mne.read_annotations(hyp_path), len(signal) // EPOCH_SAMPLES)
    subject, night = parse_recording(name)
    return signal, labels, {"subject": subject, "night": night}


@st.cache_data(show_spinner="Running CNN inference...", max_entries=4)
def predict_probs(name, _signal):
    """Softmax probabilities (n_epochs, 5) for every 30-s epoch (cached per recording name)."""
    model, _ = get_model()
    epochs = zscore_epochs(segment_epochs(_signal))
    probs = []
    with torch.no_grad():
        for i in range(0, len(epochs), 512):
            x = torch.from_numpy(epochs[i:i + 512]).unsqueeze(1)
            probs.append(F.softmax(model(x), dim=1).numpy())
    return np.concatenate(probs) if probs else np.zeros((0, len(CLASS_NAMES)))


def subject_role(subject):
    """Which split the deployed model used this subject for (train / val / test)."""
    try:
        _, ckpt = get_model()
    except Exception:
        return None
    for role in ("test", "val", "train"):
        if subject in ckpt.get(f"{role}_subjects", []):
            return role
    return "unseen"


def recording_picker(key):
    recs = list_recordings()
    if DEMO_NAME in recs:
        st.warning("⚠️ **Demo Mode Active**: no raw EDF files found in `data/raw/`. "
                   "Using the bundled sample EEG (100 Hz, Fpz-Cz).")
    name = st.selectbox("PSG recording", options=list(recs), index=0, key=key)
    psg_path, hyp_path = recs[name]
    try:
        signal, labels, info = load_recording(name, psg_path, hyp_path)
    except Exception as exc:
        st.error(f"Failed to load recording: {exc}")
        st.stop()

    role = subject_role(info.get("subject")) if "subject" in info else None
    if role == "test":
        st.success(f"Subject {info['subject']} is a held-out **test** subject: "
                   "predictions here reflect genuine unseen-subject performance.")
    elif role in ("train", "val"):
        st.info(f"Subject {info['subject']} was used for model **{role}ing**, so "
                "agreement with the expert here is optimistic. Pick a test subject "
                "for an honest view.")
    if info.get("source"):
        st.caption(f"Sample source: {info['source']}")
    return name, signal, labels, info


def hypnogram_figure(hours, predicted, expert=None, x_max=None, height=380):
    fig = go.Figure()
    if expert is not None:
        scored = expert != UNSCORED
        y_exp = np.where(scored, [HYPNO_LEVEL.get(int(s), np.nan) for s in expert], np.nan)
        fig.add_trace(go.Scatter(x=hours, y=y_exp, mode="lines", name="Expert",
                                 line=dict(color="rgba(255,255,255,0.55)", shape="hv", width=2)))
    fig.add_trace(go.Scatter(x=hours, y=[HYPNO_LEVEL[int(p)] for p in predicted],
                             mode="lines", name="CNN prediction",
                             line=dict(color="#7c83ff", shape="hv", width=1.5)))
    layout = dict(PLOT_LAYOUT)
    layout["yaxis"] = dict(HYPNO_TICKS, range=[-0.4, 4.4], gridcolor="rgba(255,255,255,0.06)")
    layout["xaxis"] = dict(title="Time (hours)", gridcolor="rgba(255,255,255,0.06)",
                           range=[hours[0] if len(hours) else 0, x_max] if x_max else None)
    fig.update_layout(**layout, height=height,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
def page_overview():
    header("Sleep Stage Classification from EEG Signals",
           "Automated sleep staging using a lightweight 1D CNN on single-channel EEG")

    section("🌙 Sleep Stages")
    st.markdown("This system classifies 30-second EEG epochs into **five** standard sleep "
                "stages based on the AASM scoring standard:")
    st.markdown("""
    <div class="stage-row">
        <span class="stage-badge badge-wake">Wake</span>
        <span class="stage-badge badge-n1">N1 — Light Sleep</span>
        <span class="stage-badge badge-n2">N2 — Intermediate Sleep</span>
        <span class="stage-badge badge-n3">N3 — Deep Sleep</span>
        <span class="stage-badge badge-rem">REM — Rapid Eye Movement</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    metrics = load_json(TEST_METRICS_PATH)
    cv = load_json(CV_METRICS_PATH)
    section("📊 Project at a Glance")
    if metrics is None:
        st.warning("No evaluation results yet. Run `python src/evaluate.py` to generate "
                   "`outputs/test_metrics.json`.")
    else:
        ds, mdl = metrics["dataset"], metrics["model"]
        f1_value = (f"{cv['macro_f1']['mean']:.3f} ± {cv['macro_f1']['std']:.3f}" if cv
                    else f"{metrics['macro_f1']:.3f}")
        metric_cards([
            (f"{ds['n_subjects']} / {ds['n_recordings']}", "Subjects / Recordings"),
            (f"{ds['n_epochs']:,}", "Scored Epochs"),
            (f"{mdl['parameters']:,}", "Model Parameters"),
            (f1_value, "Macro-F1" + (" (CV)" if cv else " (Test)")),
        ])
    st.markdown("")

    left_col, right_col = st.columns(2)
    with left_col:
        section("📂 Dataset Details")
        details = {
            "Source": "Sleep-EDF Expanded, Sleep-Cassette (PhysioNet)",
            "EEG Channel": "Fpz-Cz (single channel)",
            "Sampling Rate": f"{SFREQ} Hz",
            "Epoch Length": f"{EPOCH_SEC} seconds ({EPOCH_SAMPLES:,} samples)",
            "Sleep Stages": "5 (Wake, N1, N2, N3, REM)",
        }
        if metrics:
            details["Wake kept"] = (f"{metrics['dataset']['wake_edge_min']} min before sleep "
                                    "onset / after final awakening")
        for key, val in details.items():
            st.markdown(f"**{key}:** {val}")
    with right_col:
        section("🤖 Model Details")
        try:
            model, _ = get_model()
            params, size, macs = count_parameters(model), model_size_mb(model), count_macs(model)
        except Exception:
            params = size = macs = None
        details = {
            "Architecture": "Lightweight single-channel 1D CNN",
            "Front end": "Wide strided conv (0.5 s filters) + max-pooling",
            "Feature blocks": "32 → 64 → 64 → 128 filters, ~17 s receptive field",
            "Head": "Global Average Pooling → Dropout(0.5) → Dense(5)",
        }
        if params:
            details["Size"] = f"{params:,} parameters (~{size:.2f} MB)"
            details["Compute"] = f"{macs / 1e6:.1f} M multiply-accumulates per 30-s epoch"
        for key, val in details.items():
            st.markdown(f"**{key}:** {val}")
    st.markdown("")

    section("⚙️ Processing Pipeline")
    steps = [
        "1️⃣  Load raw PSG (.edf) and Hypnogram files using MNE-Python",
        "2️⃣  Extract single-channel EEG (Fpz-Cz) at 100 Hz and cut 30-second epochs",
        "3️⃣  Map annotations to 5 classes — merge S3+S4 → N3, discard unscored epochs",
        "4️⃣  Keep the sleep period plus 30 min of Wake on either side",
        "5️⃣  Apply per-epoch Z-score normalization",
        "6️⃣  Split by subject — both nights of a person always stay in the same split",
        "7️⃣  Train the lightweight CNN with class-weighted loss and augmentation",
        "8️⃣  Evaluate on unseen subjects: F1, Cohen's κ, N1/REM and transition analysis",
    ]
    for step in steps:
        st.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)

    if metrics:
        st.markdown("")
        section("📋 Data Split (deployed model)")
        metric_cards([
            (", ".join(map(str, metrics["train_subjects"])), "Train Subjects"),
            (", ".join(map(str, metrics["val_subjects"])), "Validation Subjects"),
            (", ".join(map(str, metrics["test_subjects"])), "Test Subjects (unseen)"),
        ], value_size="1.1rem")


# ---------------------------------------------------------------------------
# Page: EEG Signal Viewer
# ---------------------------------------------------------------------------
def page_eeg_viewer():
    header("EEG Signal Viewer",
           "Browse and visualise raw Fpz-Cz EEG recordings from the Sleep-EDF dataset")
    section("📂 Select Recording")
    name, signal, labels, _ = recording_picker("viewer_rec")

    total_seconds = len(signal) / SFREQ
    metric_cards([
        (name, "Recording"),
        (f"{SFREQ} Hz", "Sampling Rate"),
        (f"{total_seconds / 3600:.2f} hr ({total_seconds / 60:.0f} min)", "Duration"),
    ], value_size="1.2rem")
    st.markdown("")

    section("🕐 Time Window")
    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col1:
        start_time = st.slider("Start time (seconds)", 0.0, max(0.0, total_seconds - 5.0),
                               0.0, step=1.0, format="%.0f s")
    with ctrl_col2:
        max_duration = max(5.0, min(30.0, total_seconds - start_time))
        duration = st.slider("Display duration (seconds)", 5.0, max_duration,
                             min(30.0, max_duration), step=1.0, format="%.0f s")
    end_time = start_time + duration

    start_sample, stop_sample = int(start_time * SFREQ), int(end_time * SFREQ)
    eeg_uv = signal[start_sample:stop_sample] * 1e6
    time_axis = start_time + np.arange(len(eeg_uv)) / SFREQ

    stage_note = ""
    if labels is not None:
        epoch = int(start_time // EPOCH_SEC)
        if epoch < len(labels) and labels[epoch] != UNSCORED:
            stage_note = f" — expert stage at window start: **{CLASS_NAMES[labels[epoch]]}**"

    section("📈 EEG Waveform (Fpz-Cz)")
    if stage_note:
        st.markdown(stage_note)
    fig = go.Figure(go.Scatter(
        x=time_axis, y=eeg_uv, mode="lines", line=dict(color="#7c83ff", width=1), name="Fpz-Cz",
        hovertemplate="Time: %{x:.2f} s<br>Amplitude: %{y:.2f} µV<extra></extra>",
    ))
    fig.update_layout(**PLOT_LAYOUT, height=420,
                      xaxis_title="Time (seconds)", yaxis_title="EEG Amplitude (µV)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Showing {duration:.0f}s of EEG  |  Samples: {len(eeg_uv):,}  |  "
               f"Range: {start_time:.0f}s - {end_time:.0f}s  |  "
               f"Min: {eeg_uv.min():.2f} µV  |  Max: {eeg_uv.max():.2f} µV")


# ---------------------------------------------------------------------------
# Page: Sleep Stage Prediction
# ---------------------------------------------------------------------------
def page_prediction():
    header("Sleep Stage Prediction",
           "Run the lightweight 1D CNN on a selected 30-second EEG epoch.")
    try:
        get_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return

    section("📂 Select Recording & Epoch")
    col_file, col_epoch = st.columns([2, 1])
    with col_file:
        name, signal, labels, _ = recording_picker("pred_rec")
    n_epochs = len(signal) // EPOCH_SAMPLES
    if n_epochs == 0:
        st.error("Recording is shorter than one 30-second epoch.")
        return

    default_epoch = 0
    if labels is not None:
        sleep_idx = np.flatnonzero(sleep_period_mask(labels))
        default_epoch = int(sleep_idx[0]) if len(sleep_idx) else 0
    with col_epoch:
        epoch_idx = int(st.number_input("Epoch Number (30s)", min_value=0,
                                        max_value=n_epochs - 1, value=default_epoch))

    probs = predict_probs(name, signal)[epoch_idx]
    pred = int(np.argmax(probs))
    expert = int(labels[epoch_idx]) if labels is not None and epoch_idx < len(labels) else UNSCORED

    section("🎯 Prediction Results")
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        color = CLASS_COLORS[pred]
        st.markdown(
            f'<div class="metric-card" style="border-top: 4px solid {color}">'
            f'<div class="metric-value" style="color: {color}">{CLASS_NAMES[pred]}</div>'
            f'<div class="metric-label">Predicted Stage · {probs[pred] * 100:.1f}% confidence</div>'
            f'</div>', unsafe_allow_html=True)
        if expert != UNSCORED:
            verdict = "✅ matches" if expert == pred else "❌ differs from"
            st.markdown(
                f'<div class="metric-card" style="margin-top:1rem;">'
                f'<div class="metric-value">{CLASS_NAMES[expert]}</div>'
                f'<div class="metric-label">Expert label — prediction {verdict} it</div>'
                f'</div>', unsafe_allow_html=True)
    with res_col2:
        fig_bar = go.Figure(go.Bar(
            x=probs * 100, y=CLASS_NAMES, orientation="h", marker_color=CLASS_COLORS,
            text=[f"{p * 100:.1f}%" for p in probs], textposition="auto",
        ))
        fig_bar.update_layout(**PLOT_LAYOUT, title="Class Probabilities", height=240,
                              xaxis_title="Probability (%)")
        fig_bar.update_xaxes(range=[0, 100])
        st.plotly_chart(fig_bar, use_container_width=True)

    start_time = epoch_idx * EPOCH_SEC
    epoch_signal = signal[epoch_idx * EPOCH_SAMPLES:(epoch_idx + 1) * EPOCH_SAMPLES]
    section(f"📈 Epoch {epoch_idx} Waveform ({start_time}s - {start_time + EPOCH_SEC}s)")
    fig_wave = go.Figure(go.Scatter(
        x=start_time + np.arange(EPOCH_SAMPLES) / SFREQ, y=epoch_signal * 1e6,
        mode="lines", line=dict(color="#7c83ff", width=1), name="Fpz-Cz"))
    fig_wave.update_layout(**PLOT_LAYOUT, height=300,
                           xaxis_title="Time (seconds)", yaxis_title="EEG Amplitude (µV)")
    st.plotly_chart(fig_wave, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Overnight Hypnogram
# ---------------------------------------------------------------------------
def summary_table(pred_summary, expert_summary=None):
    rows = [
        ("Time in bed", "time_in_bed_min"),
        ("Total sleep time", "total_sleep_time_min"),
        ("Sleep onset latency", "sleep_onset_latency_min"),
        ("Wake after sleep onset (WASO)", "waso_min"),
        ("REM latency", "rem_latency_min"),
    ]
    header_row = "| Measure | CNN prediction |" + (" Expert |" if expert_summary else "")
    lines = [header_row, "|---|---:|" + ("---:|" if expert_summary else "")]
    for label, key in rows:
        line = f"| {label} | {fmt_min(pred_summary[key])} |"
        if expert_summary:
            line += f" {fmt_min(expert_summary[key])} |"
        lines.append(line)
    line = f"| Sleep efficiency | {pred_summary['sleep_efficiency_pct']:.1f}% |"
    if expert_summary:
        line += f" {expert_summary['sleep_efficiency_pct']:.1f}% |"
    lines.append(line)
    for i, stage in enumerate(CLASS_NAMES[1:], start=1):
        line = f"| {stage} time | {pred_summary['stage_minutes'][stage]:.0f} min |"
        if expert_summary:
            line += f" {expert_summary['stage_minutes'][stage]:.0f} min |"
        lines.append(line)
    st.markdown("\n".join(lines))


def page_hypnogram():
    header("Overnight Hypnogram",
           "Full-night sleep staging by the lightweight CNN, compared with the expert scoring.")
    try:
        get_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return

    section("📂 Select Recording")
    name, signal, labels, _ = recording_picker("hyp_rec")
    probs = predict_probs(name, signal)
    predicted = probs.argmax(axis=1)
    if len(predicted) == 0:
        st.error("Recording is shorter than one 30-second epoch.")
        return

    expert = labels[:len(predicted)] if labels is not None else None
    window = np.ones(len(predicted), dtype=bool)
    if expert is not None:
        only_sleep = st.checkbox("Show sleep period only (±30 min of Wake around sleep)",
                                 value=True)
        if only_sleep and sleep_period_mask(expert).any():
            window = sleep_period_mask(expert)
    idx = np.flatnonzero(window)
    pred_w = predicted[idx]
    exp_w = expert[idx] if expert is not None else None
    hours = idx * EPOCH_SEC / 3600.0

    if exp_w is not None:
        scored = exp_w != UNSCORED
        m = compute_metrics(exp_w[scored], pred_w[scored])
        section("🎯 Agreement with Expert Scoring")
        metric_cards([
            (f"{100 * m['accuracy']:.1f}%", "Epoch Agreement"),
            (f"{m['cohen_kappa']:.3f}", "Cohen's κ"),
            (f"{m['macro_f1']:.3f}", "Macro-F1"),
            (f"{int(scored.sum()):,}", "Scored Epochs"),
        ])
        st.markdown("")

    section("🌙 Hypnogram")
    st.caption("Grey: expert hypnogram. Purple: CNN prediction, one 30-second epoch at a time.")
    placeholder = st.empty()
    placeholder.plotly_chart(hypnogram_figure(hours, pred_w, exp_w), use_container_width=True)

    with st.expander("▶ Overnight streaming simulation"):
        st.markdown("Replays the night epoch by epoch, as an overnight monitor would see it.")
        speed = st.select_slider("Replay speed (epochs per frame)", options=[5, 10, 20, 40, 80],
                                 value=20)
        if st.button("Start simulation", type="primary"):
            x_max = hours[-1] if len(hours) else 1
            status = st.empty()
            for end in range(speed, len(idx) + speed, speed):
                end = min(end, len(idx))
                placeholder.plotly_chart(
                    hypnogram_figure(hours[:end], pred_w[:end],
                                     exp_w[:end] if exp_w is not None else None, x_max=x_max),
                    use_container_width=True, key=f"stream_{end}")
                current = CLASS_NAMES[int(pred_w[end - 1])]
                status.markdown(f"**t = {hours[end - 1]:.2f} h** · epoch {int(idx[end - 1])} "
                                f"· predicted stage **{current}**")
                time.sleep(0.05)

    section("🛏️ Sleep Architecture")
    pred_summary = sleep_summary(pred_w)
    expert_summary = None
    if exp_w is not None:
        expert_summary = sleep_summary(np.where(exp_w == UNSCORED, 0, exp_w))
    summary_table(pred_summary, expert_summary)


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------
def page_performance():
    header("Model Performance", "Evaluation on subjects never seen during training "
                                "or model selection.")
    m = load_json(TEST_METRICS_PATH)
    if m is None:
        st.warning("No evaluation results found. Run `python src/evaluate.py` first.")
        return

    section("📊 Held-out Test Subjects")
    metric_cards([
        (f"{m['macro_f1']:.3f}", "Macro-F1"),
        (f"{m['cohen_kappa']:.3f}", "Cohen's κ"),
        (f"{100 * m['accuracy']:.1f}%", f"Accuracy (baseline {100 * m['majority_class_baseline']:.0f}%)"),
        (", ".join(map(str, m["test_subjects"])), f"Test Subjects · {m['n_epochs']:,} epochs"),
    ])
    st.caption("The baseline is the accuracy of always predicting the most common stage; "
               "macro-F1 and Cohen's κ are the fairer measures for imbalanced sleep data.")
    st.markdown("")

    col_cm, col_table = st.columns([1.2, 1])
    with col_cm:
        section("🔲 Confusion Matrix")
        normalized = st.toggle("Show row percentages (recall per stage)", value=True)
        z = np.array(m["confusion_matrix_normalized"]) * 100 if normalized \
            else np.array(m["confusion_matrix"])
        text = [[f"{v:.1f}%" if normalized else f"{int(v)}" for v in row] for row in z]
        fig_cm = go.Figure(go.Heatmap(z=z, x=CLASS_NAMES, y=CLASS_NAMES, colorscale="Blues",
                                      text=text, texttemplate="%{text}",
                                      textfont={"size": 14}, hoverongaps=False))
        fig_cm.update_layout(**PLOT_LAYOUT, height=420)
        fig_cm.update_xaxes(title="Predicted Stage")
        fig_cm.update_yaxes(title="Expert Stage", autorange="reversed")
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_table:
        section("📈 Per-Class Performance")
        lines = ["| Stage | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        for name in CLASS_NAMES:
            c = m["per_class"][name]
            lines.append(f"| **{name}** | {100 * c['precision']:.1f}% | {100 * c['recall']:.1f}% "
                         f"| {100 * c['f1']:.1f}% | {c['support']:,} |")
        st.markdown("\n".join(lines))

        section("🔀 Most Frequent Confusions")
        for c in m["top_confusions"][:5]:
            st.markdown(f"- **{c['true']} → {c['predicted']}**: {c['count']} epochs "
                        f"({100 * c['rate']:.1f}% of {c['true']})")

    st.markdown("")
    col_n1, col_tr = st.columns(2)
    with col_n1:
        section("🧩 N1 / REM Focus")
        focus = m["n1_rem_focus"]
        fig = go.Figure()
        for stage, key in (("N1", "n1_predicted_as"), ("REM", "rem_predicted_as")):
            targets = list(focus[key])
            fig.add_trace(go.Bar(name=f"True {stage}", x=targets,
                                 y=[100 * focus[key][t] for t in targets]))
        fig.update_layout(**PLOT_LAYOUT, barmode="group", height=320,
                          yaxis_title="% of true-stage epochs", xaxis_title="Predicted as")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"N1 recall {100 * focus['n1_recall']:.1f}% · REM recall "
                   f"{100 * focus['rem_recall']:.1f}%. N1 is the hardest stage for human "
                   "scorers too; it is mostly confused with its neighbours on the hypnogram.")
    with col_tr:
        section("↕️ Stage-Transition Analysis")
        t = m["transition_analysis"]
        groups = [("Stable epochs", t["stable"]), ("Near a stage change", t["near_transition"])]
        fig = go.Figure()
        for metric, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
            fig.add_trace(go.Bar(name=label, x=[g[0] for g in groups],
                                 y=[100 * g[1].get(metric, 0) for g in groups]))
        fig.update_layout(**PLOT_LAYOUT, barmode="group", height=320, yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{t['near_transition']['n_epochs']:,} of {m['n_epochs']:,} test epochs sit "
                   "next to an expert-scored stage change, where even human scorers disagree most.")

    cv = load_json(CV_METRICS_PATH)
    if cv:
        st.markdown("")
        section(f"🔁 {cv['n_folds']}-Fold Subject-Grouped Cross-Validation")
        metric_cards([
            (f"{cv['macro_f1']['mean']:.3f} ± {cv['macro_f1']['std']:.3f}", "Macro-F1"),
            (f"{cv['cohen_kappa']['mean']:.3f} ± {cv['cohen_kappa']['std']:.3f}", "Cohen's κ"),
            (f"{100 * cv['accuracy']['mean']:.1f} ± {100 * cv['accuracy']['std']:.1f}%", "Accuracy"),
            (str(cv["n_subjects"]), "Subjects (each tested once)"),
        ], value_size="1.3rem")
        lines = ["| Fold | Test subjects | Accuracy | Macro-F1 | κ |", "|---|---|---:|---:|---:|"]
        for f in cv["folds"]:
            lines.append(f"| {f['fold']} | {', '.join(map(str, f['test_subjects']))} | "
                         f"{100 * f['accuracy']:.1f}% | {f['macro_f1']:.3f} | {f['cohen_kappa']:.3f} |")
        st.markdown("\n".join(lines))

    section("🤖 Model Info")
    mdl = m["model"]
    st.markdown(f"- **{mdl['parameters']:,}** trainable parameters (~{mdl['size_mb']:.2f} MB)\n"
                f"- **{mdl['macs_per_epoch'] / 1e6:.1f} M** multiply-accumulates per 30-s epoch\n"
                f"- Checkpoint selected at epoch {m['checkpoint_epoch']} by validation macro-F1 "
                f"({m['val_macro_f1']:.3f}) on subjects {', '.join(map(str, m['val_subjects']))}")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
PAGES = {
    "Overview": page_overview,
    "EEG Signal Viewer": page_eeg_viewer,
    "Sleep Stage Prediction": page_prediction,
    "Overnight Hypnogram": page_hypnogram,
    "Model Performance": page_performance,
}
PAGES[page]()
