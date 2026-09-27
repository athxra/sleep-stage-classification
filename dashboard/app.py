"""
Sleep Stage Classification Dashboard
=====================================
Streamlit dashboard for the Sleep Stage Classification from EEG Signals
project using a Lightweight 1D CNN.

Usage:
    streamlit run dashboard/app.py
"""

import os
import glob
import sys

import streamlit as st
import numpy as np
import mne
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

# Add project root to sys.path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import SleepCNN


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


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
def page_overview():
    # Hero
    st.markdown('<div class="hero-title">Sleep Stage Classification from EEG Signals</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        'Automated sleep staging using a lightweight 1D CNN on single-channel EEG'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Sleep stages explanation ---
    st.markdown('<div class="section-header">🌙 Sleep Stages</div>', unsafe_allow_html=True)
    st.markdown(
        "This system classifies 30-second EEG epochs into **five** standard sleep stages "
        "based on the AASM scoring standard:"
    )

    st.markdown("""
    <div class="stage-row">
        <span class="stage-badge badge-wake">Wake</span>
        <span class="stage-badge badge-n1">N1 — Light Sleep</span>
        <span class="stage-badge badge-n2">N2 — Intermediate Sleep</span>
        <span class="stage-badge badge-n3">N3 — Deep Sleep</span>
        <span class="stage-badge badge-rem">REM — Rapid Eye Movement</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")  # spacer

    # --- Key metrics ---
    st.markdown('<div class="section-header">📊 Project at a Glance</div>', unsafe_allow_html=True)

    cols = st.columns(4)

    metrics = [
        ("16,688", "Total Valid Epochs"),
        ("6", "EDF Recordings"),
        ("174,597", "Model Parameters"),
        ("89.00%", "Test Accuracy"),
    ]

    for col, (value, label) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'  <div class="metric-value">{value}</div>'
                f'  <div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")  # spacer

    # --- Dataset & Model details ---
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown('<div class="section-header">📂 Dataset Details</div>', unsafe_allow_html=True)

        details_data = {
            "Source": "Sleep-EDF Expanded (PhysioNet)",
            "Recordings Used": "6 subjects",
            "EEG Channel": "Fpz-Cz",
            "Sampling Rate": "100 Hz",
            "Epoch Length": "30 seconds (3,000 samples)",
            "Sleep Stages": "5 (Wake, N1, N2, N3, REM)",
        }

        for key, val in details_data.items():
            st.markdown(f"**{key}:** {val}")

    with right_col:
        st.markdown('<div class="section-header">🤖 Model Details</div>', unsafe_allow_html=True)

        model_details = {
            "Architecture": "Lightweight 1D CNN",
            "Conv Blocks": "3 (64 → 128 → 256 filters)",
            "Pooling": "MaxPool + Global Average Pooling",
            "Classifier": "Dense(128) → Dropout(0.5) → Dense(5)",
            "Parameters": "174,597 (~0.67 MB)",
            "Framework": "PyTorch",
        }

        for key, val in model_details.items():
            st.markdown(f"**{key}:** {val}")

    st.markdown("")  # spacer

    # --- Processing pipeline ---
    st.markdown('<div class="section-header">⚙️ Processing Pipeline</div>', unsafe_allow_html=True)

    steps = [
        "1️⃣  Load raw PSG (.edf) and Hypnogram files using MNE-Python",
        "2️⃣  Extract single-channel EEG (Fpz-Cz) at 100 Hz",
        "3️⃣  Segment into 30-second epochs (3,000 samples each)",
        "4️⃣  Map annotations to 5 classes — merge S3+S4 → N3, discard unknowns",
        "5️⃣  Apply per-epoch Z-score normalization",
        "6️⃣  Subject-independent train / val / test split (no data leakage)",
        "7️⃣  Train lightweight CNN with class-weighted loss",
        "8️⃣  Evaluate on completely unseen test subject",
    ]

    for step in steps:
        st.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)

    st.markdown("")  # spacer

    # --- Split info ---
    st.markdown('<div class="section-header">📋 Data Split</div>', unsafe_allow_html=True)

    split_col1, split_col2, split_col3 = st.columns(3)

    with split_col1:
        st.markdown(
            '<div class="metric-card">'
            '  <div class="metric-value">11,129</div>'
            '  <div class="metric-label">Train Epochs (4 subjects)</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with split_col2:
        st.markdown(
            '<div class="metric-card">'
            '  <div class="metric-value">2,804</div>'
            '  <div class="metric-label">Validation Epochs (1 subject)</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with split_col3:
        st.markdown(
            '<div class="metric-card">'
            '  <div class="metric-value">2,755</div>'
            '  <div class="metric-label">Test Epochs (1 subject)</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page: EEG Signal Viewer
# ---------------------------------------------------------------------------
DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PREFERRED_CHANNEL = "EEG Fpz-Cz"


@st.cache_resource(show_spinner="Loading EDF file...")
def load_raw_edf(filepath: str):
    """Load a PSG EDF lazily (preload=False) and return the Raw object."""
    raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
    return raw


def page_eeg_viewer():
    st.markdown('<div class="hero-title">EEG Signal Viewer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        'Browse and visualise raw Fpz-Cz EEG recordings from the Sleep-EDF dataset'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Discover PSG files ---
    psg_pattern = os.path.join(DATA_RAW_DIR, "*-PSG.edf")
    psg_files = sorted(glob.glob(psg_pattern))

    if not psg_files:
        st.error(
            f"No PSG EDF files found in `data/raw/`.\n\n"
            f"Looked for: `{psg_pattern}`"
        )
        return

    psg_basenames = [os.path.basename(f) for f in psg_files]

    # --- File selector ---
    st.markdown('<div class="section-header">📂 Select Recording</div>', unsafe_allow_html=True)
    selected_name = st.selectbox(
        "PSG recording",
        options=psg_basenames,
        index=0,
        label_visibility="collapsed",
    )
    selected_path = psg_files[psg_basenames.index(selected_name)]

    # --- Load EDF ---
    try:
        raw = load_raw_edf(selected_path)
    except Exception as exc:
        st.error(f"Failed to read EDF file: {exc}")
        return

    # --- Verify Fpz-Cz channel ---
    if PREFERRED_CHANNEL not in raw.ch_names:
        st.error(
            f"Channel **{PREFERRED_CHANNEL}** not found in this recording.\n\n"
            f"Available channels: {', '.join(raw.ch_names)}"
        )
        return

    sfreq = raw.info["sfreq"]
    total_seconds = raw.n_times / sfreq
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60

    # --- Recording info cards ---
    info_cols = st.columns(3)
    info_items = [
        (selected_name, "Recording"),
        (f"{sfreq:.0f} Hz", "Sampling Rate"),
        (f"{total_hours:.2f} hr ({total_minutes:.0f} min)", "Duration"),
    ]
    for col, (val, lbl) in zip(info_cols, info_items):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'  <div class="metric-value" style="font-size:1.2rem">{val}</div>'
                f'  <div class="metric-label">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")  # spacer

    # --- Time controls ---
    st.markdown('<div class="section-header">🕐 Time Window</div>', unsafe_allow_html=True)

    ctrl_col1, ctrl_col2 = st.columns([3, 1])

    with ctrl_col1:
        max_start = max(0.0, total_seconds - 5.0)
        start_time = st.slider(
            "Start time (seconds)",
            min_value=0.0,
            max_value=max_start,
            value=0.0,
            step=1.0,
            format="%.0f s",
        )

    with ctrl_col2:
        remaining = total_seconds - start_time
        max_duration = min(30.0, remaining)
        duration = st.slider(
            "Display duration (seconds)",
            min_value=5.0,
            max_value=max_duration,
            value=min(30.0, max_duration),
            step=1.0,
            format="%.0f s",
        )

    end_time = start_time + duration

    # --- Read only the selected window (memory efficient) ---
    start_sample = int(start_time * sfreq)
    stop_sample = int(end_time * sfreq)

    try:
        # Pick the Fpz-Cz channel and read only the needed segment
        raw_pick = raw.copy().pick([PREFERRED_CHANNEL])
        data, times = raw_pick[:, start_sample:stop_sample]
        eeg_signal = data[0] * 1e6  # Convert V -> uV for readability
        time_axis = times
    except Exception as exc:
        st.error(f"Error reading EEG data: {exc}")
        return

    # --- Plotly chart ---
    st.markdown('<div class="section-header">📈 EEG Waveform (Fpz-Cz)</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=eeg_signal,
        mode="lines",
        line=dict(color="#7c83ff", width=1),
        name="Fpz-Cz",
        hovertemplate="Time: %{x:.2f} s<br>Amplitude: %{y:.2f} uV<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="EEG Amplitude (uV)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,30,47,0.6)",
        height=420,
        margin=dict(l=60, r=20, t=30, b=50),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.15)",
        ),
        hoverlabel=dict(bgcolor="#2d2d44"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Epoch info ---
    n_samples_shown = len(eeg_signal)
    st.caption(
        f"Showing {duration:.0f}s of EEG  |  "
        f"Samples: {n_samples_shown:,}  |  "
        f"Range: {start_time:.0f}s - {end_time:.0f}s  |  "
        f"Min: {eeg_signal.min():.2f} uV  |  "
        f"Max: {eeg_signal.max():.2f} uV"
    )

# ---------------------------------------------------------------------------
# Page: Sleep Stage Prediction
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth")
CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
CLASS_COLORS = ["#4fc3f7", "#7c83ff", "#ab47bc", "#5c6bc0", "#ef5350"]

@st.cache_resource(show_spinner="Loading PyTorch Model...")
def load_pytorch_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = SleepCNN(n_classes=5)
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def page_prediction():
    st.markdown('<div class="hero-title">Sleep Stage Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        'Run the lightweight 1D CNN inference on a selected 30-second EEG epoch.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Discover PSG files ---
    psg_pattern = os.path.join(DATA_RAW_DIR, "*-PSG.edf")
    psg_files = sorted(glob.glob(psg_pattern))

    if not psg_files:
        st.error(f"No PSG EDF files found in `data/raw/`.\n\nLooked for: `{psg_pattern}`")
        return

    psg_basenames = [os.path.basename(f) for f in psg_files]

    # --- Load Model ---
    try:
        model = load_pytorch_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return

    st.markdown('<div class="section-header">📂 Select Recording & Epoch</div>', unsafe_allow_html=True)
    col_file, col_epoch = st.columns([2, 1])
    
    with col_file:
        selected_name = st.selectbox("PSG recording", options=psg_basenames, index=0)
        selected_path = psg_files[psg_basenames.index(selected_name)]

    try:
        raw = load_raw_edf(selected_path)
    except Exception as exc:
        st.error(f"Failed to read EDF file: {exc}")
        return

    if PREFERRED_CHANNEL not in raw.ch_names:
        st.error(f"Channel **{PREFERRED_CHANNEL}** not found in this recording.")
        return

    sfreq = raw.info["sfreq"]
    if sfreq != 100:
        st.error(f"Expected 100 Hz sampling rate, got {sfreq} Hz.")
        return
        
    total_seconds = raw.n_times / sfreq
    total_epochs = int(total_seconds // 30)

    with col_epoch:
        epoch_idx = st.number_input("Epoch Number (30s)", min_value=0, max_value=max(0, total_epochs - 1), value=0)

    start_time = epoch_idx * 30.0
    end_time = start_time + 30.0
    start_sample = int(start_time * sfreq)
    stop_sample = int(end_time * sfreq)

    # Read exactly one 30-second epoch
    try:
        raw_pick = raw.copy().pick([PREFERRED_CHANNEL])
        data, times = raw_pick[:, start_sample:stop_sample]
        eeg_signal = data[0]
    except Exception as exc:
        st.error(f"Error reading EEG data: {exc}")
        return

    if len(eeg_signal) != 3000:
        st.error(f"Incomplete epoch: got {len(eeg_signal)} samples, expected 3000.")
        return

    # Preprocessing: Z-score normalization
    mean_val = np.mean(eeg_signal)
    std_val = np.std(eeg_signal)
    if std_val > 1e-8:
        eeg_norm = (eeg_signal - mean_val) / std_val
    else:
        eeg_norm = eeg_signal - mean_val

    # Inference
    x_tensor = torch.tensor(eeg_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3000)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
    
    pred_class_idx = int(np.argmax(probs))
    pred_class_name = CLASS_NAMES[pred_class_idx]
    pred_confidence = probs[pred_class_idx] * 100

    # Display Prediction Results
    st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
    
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.markdown(
            f'<div class="metric-card" style="border-top: 4px solid {CLASS_COLORS[pred_class_idx]}">'
            f'  <div class="metric-value" style="color: {CLASS_COLORS[pred_class_idx]}">{pred_class_name}</div>'
            f'  <div class="metric-label">Predicted Stage</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-card" style="margin-top:1rem;">'
            f'  <div class="metric-value">{pred_confidence:.1f}%</div>'
            f'  <div class="metric-label">Confidence</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        
    with res_col2:
        # Probability Bar Chart
        fig_bar = go.Figure(go.Bar(
            x=probs * 100,
            y=CLASS_NAMES,
            orientation='h',
            marker_color=CLASS_COLORS,
            text=[f"{p*100:.1f}%" for p in probs],
            textposition='auto',
        ))
        fig_bar.update_layout(
            title="Class Probabilities",
            xaxis_title="Probability (%)",
            yaxis_title="Stage",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,30,47,0.6)",
            height=200,
            margin=dict(l=60, r=20, t=30, b=30),
            xaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Display Waveform
    st.markdown(f'<div class="section-header">📈 Epoch {epoch_idx} Waveform ({start_time:.0f}s - {end_time:.0f}s)</div>', unsafe_allow_html=True)
    
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(
        x=np.linspace(start_time, end_time, 3000),
        y=eeg_signal * 1e6, # Plot in uV
        mode="lines",
        line=dict(color="#7c83ff", width=1),
        name="Fpz-Cz",
    ))

    fig_wave.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="EEG Amplitude (uV)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,30,47,0.6)",
        height=300,
        margin=dict(l=60, r=20, t=30, b=50),
    )
    st.plotly_chart(fig_wave, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Overnight Hypnogram
# ---------------------------------------------------------------------------
def page_hypnogram():
    st.markdown('<div class="hero-title">Overnight Hypnogram</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        'Full night sleep stage prediction using the trained lightweight CNN.'
        '</div>',
        unsafe_allow_html=True,
    )
    
    st.info("Predictions are generated by the trained CNN from EEG signals. They are not the expert annotation labels.")

    # --- Discover PSG files ---
    psg_pattern = os.path.join(DATA_RAW_DIR, "*-PSG.edf")
    psg_files = sorted(glob.glob(psg_pattern))

    if not psg_files:
        st.error(f"No PSG EDF files found in `data/raw/`.\n\nLooked for: `{psg_pattern}`")
        return

    psg_basenames = [os.path.basename(f) for f in psg_files]

    # --- Load Model ---
    try:
        model = load_pytorch_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return

    st.markdown('<div class="section-header">📂 Select Recording</div>', unsafe_allow_html=True)
    selected_name = st.selectbox("PSG recording", options=psg_basenames, index=0)
    selected_path = psg_files[psg_basenames.index(selected_name)]
    
    if st.button("Generate Hypnogram", type="primary"):
        with st.spinner("Processing recording and running inference..."):
            try:
                raw = load_raw_edf(selected_path)
                if PREFERRED_CHANNEL not in raw.ch_names:
                    st.error(f"Channel **{PREFERRED_CHANNEL}** not found.")
                    return
                
                sfreq = raw.info["sfreq"]
                if sfreq != 100:
                    st.error(f"Expected 100 Hz sampling rate, got {sfreq} Hz.")
                    return
                
                # Load data
                raw_pick = raw.copy().pick([PREFERRED_CHANNEL])
                raw_pick.load_data()
                data = raw_pick.get_data()[0]
                total_samples = len(data)
                
                epoch_samples = 3000
                n_epochs = total_samples // epoch_samples
                
                if n_epochs == 0:
                    st.error("Recording is too short for a single epoch.")
                    return
                    
                # Reshape data into epochs
                epochs_data = data[:n_epochs * epoch_samples].reshape(n_epochs, epoch_samples)
                
                # Z-score normalization
                means = np.mean(epochs_data, axis=1, keepdims=True)
                stds = np.std(epochs_data, axis=1, keepdims=True)
                stds[stds < 1e-8] = 1.0  # Avoid division by zero
                epochs_norm = (epochs_data - means) / stds
                
                # Inference in batches to avoid OOM
                batch_size = 128
                all_preds = []
                
                with torch.no_grad():
                    for i in range(0, n_epochs, batch_size):
                        batch = epochs_norm[i:i+batch_size]
                        x_tensor = torch.tensor(batch, dtype=torch.float32).unsqueeze(1) # (B, 1, 3000)
                        logits = model(x_tensor)
                        probs = F.softmax(logits, dim=1)
                        preds = torch.argmax(probs, dim=1).numpy()
                        all_preds.extend(preds)
                        
                all_preds = np.array(all_preds)
                time_hours = np.arange(n_epochs) * 30 / 3600.0
                
                # Display summary info
                st.markdown('<div class="section-header">📊 Recording Summary</div>', unsafe_allow_html=True)
                info_cols = st.columns(3)
                duration_hours = n_epochs * 30 / 3600.0
                
                with info_cols[0]:
                    st.metric("Recording", selected_name)
                with info_cols[1]:
                    st.metric("Duration", f"{duration_hours:.2f} hours")
                with info_cols[2]:
                    st.metric("Analyzed Epochs", f"{n_epochs:,}")
                
                # Calculate class distributions
                counts = np.bincount(all_preds, minlength=5)
                percentages = counts / n_epochs * 100
                
                dist_cols = st.columns(5)
                for i in range(5):
                    with dist_cols[i]:
                        st.metric(CLASS_NAMES[i], f"{counts[i]} ({percentages[i]:.1f}%)")
                
                # Plotly Hypnogram
                st.markdown('<div class="section-header">🌙 Hypnogram</div>', unsafe_allow_html=True)
                
                # Y-axis mapping to achieve Wake, REM, N1, N2, N3 order (Wake=4, REM=3, N1=2, N2=1, N3=0)
                stage_map = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
                y_mapped = np.array([stage_map[p] for p in all_preds])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_hours,
                    y=y_mapped,
                    mode='lines',
                    line=dict(color="#7c83ff", shape='hv'),
                    name='Sleep Stage'
                ))
                
                fig.update_layout(
                    xaxis_title="Time (hours)",
                    yaxis_title="Sleep Stage",
                    yaxis=dict(
                        tickmode='array',
                        tickvals=[4, 3, 2, 1, 0],
                        ticktext=['Wake', 'REM', 'N1', 'N2', 'N3'],
                    ),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(30,30,47,0.6)",
                    height=400,
                    margin=dict(l=60, r=20, t=30, b=50),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as exc:
                st.error(f"Error processing recording: {exc}")


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------
def page_performance():
    st.markdown('<div class="hero-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        'Evaluation results on the completely unseen test subject (SC4022).'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Summary Metrics ---
    st.markdown('<div class="section-header">📊 Overall Test Metrics</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    metrics = [
        ("89.00%", "Test Accuracy"),
        ("69.22%", "Macro-F1 Score"),
        ("SC4022", "Test Subject"),
        ("2,755", "Test Epochs"),
    ]
    for col, (value, label) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'  <div class="metric-value">{value}</div>'
                f'  <div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")  # spacer

    col_cm, col_table = st.columns([1.2, 1])

    with col_cm:
        st.markdown('<div class="section-header">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
        st.markdown(
            "The confusion matrix compares the actual expert-annotated sleep stages (Y-axis) "
            "with the stages predicted by the CNN model (X-axis)."
        )

        cm_data = [
            [1858,    5,    0,    0,    8],
            [   7,   64,   66,   18,   29],
            [   0,    1,  327,   74,    0],
            [   0,    0,    0,  119,    0],
            [   7,   38,   27,   23,   84]
        ]
        classes = ["Wake", "N1", "N2", "N3", "REM"]

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=classes,
            y=classes,
            hoverongaps=False,
            colorscale='Blues',
            text=cm_data,
            texttemplate="%{text}",
            textfont={"size": 14}
        ))
        
        # Reverse y-axis to match typical matrix representation
        fig_cm.update_layout(
            xaxis_title="Predicted Stage",
            yaxis_title="Actual Stage",
            yaxis=dict(autorange="reversed"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,30,47,0.6)",
            height=400,
            margin=dict(l=50, r=20, t=30, b=50),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_table:
        st.markdown('<div class="section-header">📈 Per-Class Performance</div>', unsafe_allow_html=True)
        
        # Create a markdown table
        st.markdown("""
        | Stage | Precision | Recall | F1-Score | Support |
        |-------|-----------|--------|----------|---------|
        | **Wake** | 99.25% | 99.31% | 99.28% | 1,871 |
        | **N1** | 59.26% | 34.78% | 43.84% | 184 |
        | **N2** | 77.86% | 81.34% | 79.56% | 402 |
        | **N3** | 50.85% | 100.00% | 67.42% | 119 |
        | **REM** | 69.42% | 46.93% | 56.00% | 179 |
        """)
        
        st.markdown('<div class="section-header" style="margin-top:2rem;">🤖 Model Info</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Lightweight 1D CNN** architecture
        - **174,597** trainable parameters
        - Approximately **0.67 MB** parameter size
        - 3 convolutional blocks
        - MaxPool + **Global Average Pooling**
        - Dense(128) → Dropout(0.5) → Dense(5)
        """)


# ---------------------------------------------------------------------------
# Page: Placeholder pages

# ---------------------------------------------------------------------------
def page_placeholder(name: str):
    st.markdown(
        f'<div class="hero-title">{name}</div>',
        unsafe_allow_html=True,
    )
    st.info(
        f"🚧 The {name} page is under construction and will be implemented soon."
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "Overview":
    page_overview()
elif page == "EEG Signal Viewer":
    page_eeg_viewer()
elif page == "Sleep Stage Prediction":
    page_prediction()
elif page == "Overnight Hypnogram":
    page_hypnogram()
elif page == "Model Performance":
    page_performance()

