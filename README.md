# Sleep Stage Classification from EEG Signals Using a Lightweight CNN

A college minor project that classifies sleep stages from single-channel EEG using a lightweight 1D Convolutional Neural Network (CNN), evaluated on subjects the model has never seen.

## Overview

The model scores each 30-second EEG epoch as one of five standard sleep stages:

| Label | Stage | Description |
|-------|-------|-------------|
| W     | Wake  | Awake state |
| N1    | NREM 1 | Light sleep |
| N2    | NREM 2 | Intermediate sleep |
| N3    | NREM 3 | Deep / slow-wave sleep |
| R     | REM   | Rapid eye movement sleep |

What the project adds beyond the standard CNN-on-EEG setup:

- **Single-channel, lightweight design.** One raw Fpz-Cz channel and a compact 1D CNN (about 0.1 M parameters, about 4 M multiply-accumulates per epoch), sized for low-power, wearable-style devices.
- **Subject-independent evaluation.** Both nights of a subject always go in the same split, and every result is on unseen people.
- **Stage-transition-aware error analysis.** The analysis focuses on N1/REM confusion and compares performance on epochs next to an expert-scored stage change with performance on stable stretches.
- **Overnight monitoring simulation.** A hypnogram that replays the night epoch by epoch, with the expert hypnogram overlaid.

## Dataset

- **Source:** [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/), Sleep-Cassette study (PhysioNet)
- **Subset:** Sleep-EDF-20: subjects 0–19, 39 overnight recordings (subject 13 has one night). This is the standard benchmark subset, so results can be compared with published work.
- **Channel:** Fpz-Cz, 100 Hz
- **Epochs:** 30 s (3,000 samples)
- **Sleep period:** Cassette recordings last about 20 h and are mostly daytime Wake. Each recording is trimmed to the sleep period plus 30 min of Wake on either side, the convention used by DeepSleepNet, TinySleepNet and AttnSleep. Without this trim, accuracy mostly reflects easy Wake epochs.

### Label Mapping

| Annotation | Mapped Class | Integer Label |
|------------|-------------|:------------:|
| Sleep stage W | Wake | 0 |
| Sleep stage 1 | N1 | 1 |
| Sleep stage 2 | N2 | 2 |
| Sleep stage 3 | N3 | 3 |
| Sleep stage 4 | N3 | 3 |
| Sleep stage R | REM | 4 |
| Sleep stage ? | Discarded | — |
| Movement time | Discarded | — |

### Subject-Independent Splits

Sleep-Cassette files are named `SC4<ss><n>`, where `<ss>` is the **subject** and `<n>` is the **night**. For example, `SC4021` and `SC4022` are two nights of subject 2. Splits therefore group recordings by subject:

- **5-fold subject-grouped cross-validation.** Each subject is tested exactly once.
- In every fold, 2 of the non-test subjects are held out for validation (checkpoint selection, LR schedule and early stopping), so no test subject influences training in any way.
- **Fold 0** is the deployed model used by the dashboard.

Fold assignments are written to `data/processed/split_info.json`.

## Model Architecture

```
Input: (batch, 1, 3000)                          30 s @ 100 Hz
    │
    ├── Conv1D(1 → 32, kernel=50, stride=6) → BN → ReLU      ~0.5 s filters
    ├── MaxPool1D(8) → Dropout(0.25)                          3000 → 62
    │
    ├── Conv1D(32 → 64, kernel=7) → BN → ReLU
    ├── Conv1D(64 → 64, kernel=7) → BN → ReLU
    ├── MaxPool1D(4) → Dropout(0.25)                          62 → 15
    │
    ├── Conv1D(64 → 128, kernel=7) → BN → ReLU
    ├── AdaptiveAvgPool1D(1)                                  Global Average Pooling
    │
    ├── Dropout(0.5)
    └── Dense(128 → 5)

Output: (batch, 5)  →  class logits
```

| Property | Value |
|----------|-------|
| Trainable parameters | **103,173** |
| Model size (params) | **~0.39 MB** |
| Compute | **~4.3 M MACs per 30-s epoch** |
| Receptive field | ~17 s |

**Design notes**

- The wide, strided first layer has kernel = fs/2 and stride = fs/16, as in the small-filter branch of DeepSleepNet. Each filter sees about 0.5 s of EEG, long enough to capture delta waves (N3), spindles and K-complexes (N2), and sawtooth waves (REM).
- Pooling between the blocks grows the receptive field to about 17 s before Global Average Pooling.
- An earlier version used three stride-1 convolutions at full resolution. It had a receptive field of only 0.13 s and needed about 420 M MACs per epoch. The current design is about 97× cheaper and runs about 50× faster on CPU.

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (weight decay 1e-4) |
| Learning rate | 1e-3, halved when validation macro-F1 plateaus (patience 3) |
| Batch size | 128 |
| Epochs | up to 60, early stopping with patience 10 |
| Loss | CrossEntropyLoss, inverse-frequency class weights from training subjects only |
| Augmentation | random time shift (±3 s), amplitude scaling (0.9–1.1), Gaussian noise |
| Model selection | best validation macro-F1 |
| Seed | 42 |

## Results

All numbers come from Sleep-EDF-20: 20 subjects, 39 recordings, 40,274 scored epochs. `outputs/*.json` is the source of truth, and the dashboard reads it directly.

### 5-Fold Subject-Grouped Cross-Validation (every subject tested once)

| Metric | Mean ± std across folds |
|--------|------:|
| Macro F1-Score | **0.709 ± 0.041** |
| Cohen's κ | **0.688 ± 0.063** |
| Accuracy | 76.4 ± 4.9% |

| Class | F1 (mean ± std) |
|-------|----------------:|
| Wake  | 0.816 ± 0.083 |
| N1    | 0.366 ± 0.028 |
| N2    | 0.822 ± 0.060 |
| N3    | 0.807 ± 0.077 |
| REM   | 0.732 ± 0.047 |

Fold macro-F1 ranges from 0.65 to 0.77 depending on which four subjects are held out. That spread is why a single test night is not a reliable estimate.

**N1 / REM focus (pooled over all folds).** N1 recall is 50%. When N1 is missed, it is most often called REM (27%), then N2 (11%) or Wake (10%). In turn, 16% of REM epochs are called N1. N1 and REM look similar on a single frontal EEG channel: both have low-amplitude, mixed-frequency activity, and the eye-movement and muscle-tone signals that separate them are not in the Fpz-Cz channel.

### Deployed Model (fold 0, test subjects 7, 9, 14 and 15)

| Metric | Value |
|--------|------:|
| Macro F1-Score | **0.776** |
| Cohen's κ | **0.790** |
| Accuracy | 84.2% (majority-class baseline 37.5%) |

| Class | Precision | Recall | F1-Score | Support |
|-------|----------:|-------:|---------:|--------:|
| Wake  | 0.948 | 0.939 | 0.944 | 2,105 |
| N1    | 0.260 | 0.675 | 0.375 | 397 |
| N2    | 0.906 | 0.874 | 0.889 | 3,343 |
| N3    | 0.916 | 0.882 | 0.898 | 1,291 |
| REM   | 0.903 | 0.676 | 0.773 | 1,786 |

**Stage-transition analysis.** On epochs inside stable stage runs, the model reaches 87.3% accuracy (macro-F1 0.775). On epochs next to an expert-scored stage change, it reaches only 63.7% (macro-F1 0.614). Most remaining errors sit at stage boundaries, which is also where human scorers disagree most.

### Why these numbers differ from the first version

The first version reported 89.0% accuracy / 0.692 macro-F1, but:

1. **Validation and test were the same person.** SC4021 and SC4022 are two nights of subject 2, so model selection had already seen the test subject.
2. **68% of test epochs were daytime Wake.** Always predicting Wake would already score about 68% accuracy.
3. **The model was evaluated on a single night**, with no estimate of how much the score varies.

The current protocol fixes all three problems. Its numbers are lower on paper but hold up for unseen people. For context, published single-channel Fpz-Cz models on Sleep-EDF-20 report roughly 0.77–0.80 macro-F1 (DeepSleepNet ≈0.77, AttnSleep ≈0.78, TinySleepNet ≈0.80). Those models are larger, and most also use the context of neighbouring epochs (LSTM or attention over the sequence), whereas this model scores each epoch on its own.

## Tech Stack

| Component | Tool |
|-----------|------|
| Deep Learning | PyTorch |
| EEG Processing | MNE-Python |
| Evaluation | Scikit-learn |
| Web API | FastAPI + Uvicorn |
| Dashboard | React 19 + TypeScript (Vite), hand-built SVG charts |
| Legacy dashboard | Streamlit + Plotly |
| Testing | pytest |

## Project Structure

```
sleep-stage-classification/
├── data/
│   ├── raw/                # Sleep-EDF .edf files (downloaded, git-ignored)
│   ├── processed/          # Preprocessed arrays + split_info.json
│   └── sample/             # Demo excerpt of a test subject, with expert labels
├── models/best_model.pth   # Deployed model (fold 0)
├── outputs/                # test_metrics.json, cv_metrics.json (read by dashboard)
├── src/
│   ├── config.py           # Paths, constants, labels, file-name parsing
│   ├── download_data.py    # Fetch Sleep-EDF-20 from PhysioNet
│   ├── inspect_edf.py      # Inspect one PSG / hypnogram pair
│   ├── signals.py          # Epoching, labelling, trimming, normalization (shared)
│   ├── preprocess.py       # Build the dataset and subject-grouped folds
│   ├── splits.py           # Subject-grouped k-fold splitting
│   ├── model.py            # Lightweight 1D CNN
│   ├── train.py            # Training (single fold or full CV)
│   ├── metrics.py          # Metrics, transition analysis, sleep statistics
│   ├── evaluate.py         # Test-set evaluation of the deployed model
│   └── make_demo_sample.py # Build the dashboard demo sample
├── api/server.py           # FastAPI backend: results + live CNN inference for the web UI
├── web/                    # React + TypeScript dashboard (Vite)
├── dashboard/app.py        # Legacy Streamlit dashboard
├── tests/                  # Unit tests (pytest)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

## Usage

```bash
# 1. Download Sleep-EDF-20 (~1.9 GB, resumable) into data/raw/
python src/download_data.py

# 2. Preprocess and create subject-grouped folds
python src/preprocess.py

# 3. Cross-validate (all folds) and train the deployed model (fold 0)
python src/train.py --cv
python src/train.py

# 4. Evaluate the deployed model on its unseen test subjects
python src/evaluate.py

# 5. (Optional) rebuild the dashboard demo sample from a test subject
python src/make_demo_sample.py

# Tests
python -m pytest tests
```

## Dashboard

The main dashboard is a React app served by a small FastAPI backend. The backend reuses the pipeline's own code (`src/signals.py`, `src/model.py`, `src/metrics.py`), so the UI shows exactly what the model computes.

```bash
# One-time: build the frontend (needs Node.js 18+)
cd web
npm install
npm run build
cd ..

# Serve the API and the built dashboard at http://localhost:8000
python -m uvicorn api.server:app --port 8000
```

For frontend development with hot reload, run `python -m uvicorn api.server:app --port 8000` and `npm run dev` (inside `web/`) side by side, then open http://localhost:5173. Vite proxies `/api` to the backend.

The dashboard has four pages. Each has a collapsible **How to read this page** guide, and every metric has an ⓘ tooltip explaining it in plain language.

- **Overview**: headline cross-validation results, quick-start cards, dataset and model at a glance, the five stages, and the pipeline.
- **Night explorer**: pick, search or filter any of the 39 nights (each labelled as a test, validation or training subject).
  - Hypnogram: the expert and CNN hypnograms, a disagreement strip, and a hypnodensity lane (the model's per-epoch stage probabilities).
  - Navigation: **drag to zoom**, a seekable timeline, and **previous/next disagreement** buttons.
  - **Overnight replay** plays the night back as a live monitor would see it.
  - Epoch inspector: a strip of neighbouring epochs, the raw EEG, stage probabilities, band power, and the **power spectrum**.
  - Sleep measures, CNN vs expert: total sleep time, efficiency, latencies, wake after sleep onset, and time in each stage.
  - **Export CSV** of per-epoch predictions, and **Copy link** to share the exact night and epoch.
  - **Analyze your own EDF**: upload a recording (plus an optional hypnogram) and it is staged on the spot. Fpz-Cz is used if present, otherwise the first EEG channel, resampled to 100 Hz.
- **Performance**:
  - Cross-validation or deployed-model results, per-stage F1 with fold error bars, the N1/REM focus, the stage-transition analysis, and per-fold results.
  - **Click any confusion-matrix cell** (or a row of the confusion list) to see real test epochs with that mistake, and open them in the Night explorer.
- **Model**: architecture, compute budget, training setup, and the subject-grouped fold assignment.

Keyboard shortcuts (press `?` in the app):

| Key | Action |
|---|---|
| ← / → | Previous / next epoch (hold Shift to jump 10) |
| D / Shift+D | Next / previous disagreement |
| Space | Play / pause the replay |
| Z | Reset zoom |

The dashboard supports light and dark themes and works on phones. Without raw data it runs in demo mode on the bundled test-subject excerpt.

The previous Streamlit dashboard still works: `streamlit run dashboard/app.py`.

## License

This project is for educational purposes.
