# Sleep Stage Classification from EEG Signals Using a Lightweight CNN

A college minor project that classifies sleep stages from single-channel EEG signals using a lightweight 1D Convolutional Neural Network (CNN).

## Overview

This project performs automated sleep stage scoring by classifying 30-second EEG epochs into five standard sleep stages:

| Label | Stage | Description |
|-------|-------|-------------|
| W     | Wake  | Awake state |
| N1    | NREM 1 | Light sleep |
| N2    | NREM 2 | Intermediate sleep |
| N3    | NREM 3 | Deep / slow-wave sleep |
| R     | REM   | Rapid eye movement sleep |

## Dataset

- **Source:** [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/) (PhysioNet)
- **Subjects:** 6 recordings (SC4001, SC4002, SC4011, SC4012, SC4021, SC4022)
- **Channel:** Fpz-Cz (single-channel EEG)
- **Sampling Rate:** 100 Hz
- **Epoch Length:** 30 seconds (3000 samples per epoch)
- **Total Epochs:** 16,688

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

### Subject-Independent Split

| Split | Subjects | Epochs |
|-------|----------|-------:|
| Train | SC4001, SC4002, SC4011, SC4012 | 11,129 |
| Val   | SC4021 | 2,804 |
| Test  | SC4022 | 2,755 |

No subject appears in more than one split, ensuring zero data leakage.

## Model Architecture

A lightweight 1D CNN with Global Average Pooling to minimize the parameter count.

```
Input: (batch, 1, 3000)
    │
    ├── Conv1D(1 → 64,  kernel=7, padding=3)
    ├── BatchNorm1D(64)
    ├── ReLU
    │
    ├── Conv1D(64 → 128, kernel=5, padding=2)
    ├── BatchNorm1D(128)
    ├── ReLU
    │
    ├── Conv1D(128 → 256, kernel=3, padding=1)
    ├── BatchNorm1D(256)
    ├── ReLU
    │
    ├── MaxPool1D(kernel=2)
    ├── AdaptiveAvgPool1D(1)          ← Global Average Pooling
    │
    ├── Dense(256 → 128)
    ├── ReLU
    ├── Dropout(0.5)
    └── Dense(128 → 5)

Output: (batch, 5)  →  class logits
```

| Property | Value |
|----------|-------|
| Trainable parameters | **174,597** |
| Model size (params) | **~0.67 MB** |
| Checkpoint file | ~2.0 MB |

**Design note:** Global Average Pooling (`AdaptiveAvgPool1d(1)`) is used after MaxPool to collapse the temporal dimension from 1500 → 1 before the dense layers. Without it, a naive Flatten would produce a 384,000-dimensional vector, inflating the first dense layer to ~49 million parameters. Global Average Pooling keeps the model truly lightweight while acting as a spatial regularizer that reduces overfitting.

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 128 |
| Epochs | 30 |
| Loss | CrossEntropyLoss (class-weighted) |
| Model selection | Best validation macro-F1 |
| Best epoch | 19 |

Class weights are computed only from the training data using inverse-frequency weighting to handle class imbalance.

## Results

### Test Set Performance (Subject SC4022 — Unseen)

| Metric | Value |
|--------|------:|
| Overall Accuracy | **89.00%** |
| Macro F1-Score | **0.6922** |

### Per-Class Results

| Class | Precision | Recall | F1-Score | Support |
|-------|----------:|-------:|---------:|--------:|
| Wake  | 0.9925 | 0.9931 | 0.9928 | 1,871 |
| N1    | 0.5926 | 0.3478 | 0.4384 | 184 |
| N2    | 0.7786 | 0.8134 | 0.7956 | 402 |
| N3    | 0.5085 | 1.0000 | 0.6742 | 119 |
| REM   | 0.6942 | 0.4693 | 0.5600 | 179 |

## Tech Stack

| Component | Tool |
|-----------|------|
| Deep Learning | PyTorch |
| EEG Processing | MNE-Python |
| Evaluation | Scikit-learn |
| Dashboard | Streamlit |
| Visualization | Plotly |

## Project Structure

```
minor_project/
├── data/
│   ├── raw/            # Raw EDF files from PhysioNet
│   ├── processed/      # Preprocessed arrays and split indices
│   └── sample/         # Small sample files for quick testing
├── models/             # Saved model checkpoints
├── src/
│   ├── inspect_edf.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── dashboard/          # Streamlit dashboard app
├── outputs/            # Training logs, plots and test results
├── tests/              # Unit tests
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# 1. Inspect raw EDF files
python src/inspect_edf.py

# 2. Preprocess and split data
python src/preprocess.py

# 3. Train the model
python src/train.py

# 4. Evaluate on the test set
python src/evaluate.py
```

## License

This project is for educational purposes.
