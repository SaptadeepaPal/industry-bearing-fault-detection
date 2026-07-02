# Bearing Fault Detection using Multi-Domain Signal Processing + XGBoost

A machine learning pipeline that detects and classifies bearing faults (Normal, Ball, Inner Race, Outer Race) from raw vibration signals, using a custom-built multi-domain feature engineering pipeline (time-domain, FFT, and envelope analysis) on the CWRU Bearing Dataset.

**Final model: XGBoost — 99.53% test accuracy, 99.50% macro F1, on a strictly leakage-free, group-based train/test split.**

---

## Why this project matters

Bearing failures are one of the most common causes of unplanned downtime in rotating machinery. Catching a fault early — and correctly identifying *which* component is failing (ball, inner race, or outer race) — lets maintenance teams intervene before a full breakdown. This project builds an end-to-end pipeline that goes from raw accelerometer signals to a deployable fault classifier, using signal processing techniques (FFT, bandpass filtering, Hilbert transform envelope analysis) borrowed directly from industrial vibration analysis practice, not just generic tabular ML.

---

## A note on rigor: catching my own data leakage

This is the part of the project I'm most proud of, and I want to be upfront about it rather than bury it.

**First pass:** with a naive random train/test split, every model — XGBoost, SVM, and LightGBM — scored **100% accuracy, 100% precision, 100% recall, 100% F1**, with a 0.00% train/test gap.

A perfect score across every metric and every class is not a good sign — it's a red flag. In this case, the cause was **overlapping signal windows** (50% overlap, used to increase dataset size) from the *same recording file* ending up split across both train and test. Even though no single window was duplicated, windows from the same file share so much of the same underlying signal that a random split leaks information between train and test.

**The fix:** I re-did the split using `GroupShuffleSplit` from scikit-learn, grouping by `file_name` — so every window from a given `.mat` file goes entirely into either train or test, never both. I explicitly verified zero file overlap between splits with an automated assertion before training.

**Result after the fix:** XGBoost's accuracy dropped from a suspicious 100% to a much more credible **99.53%**, with a small but non-zero train/test gap (100.00% train vs. 99.53% test = 0.47% gap) — consistent with a model that has genuinely learned, rather than memorized, the fault signatures.

**A second, less obvious effect of the fix worth noting:** because the group-based split assigns whole files (not individual windows) to train or test, the resulting test set is no longer perfectly class-balanced — for example, the final XGBoost test set has 471 InnerRace windows vs. only 118 OuterRace windows. This is an expected and correct trade-off of leakage-safe splitting on this dataset (you can't simultaneously group by file *and* stratify by class when files aren't evenly distributed across classes), and it's why the model comparison below reports macro-averaged metrics — which weight every class equally regardless of how many test samples it has — rather than relying on accuracy alone.

---

## Pipeline overview

```
Raw CWRU .mat files (vibration accelerometer signals)
        │
        ▼
Windowing (2048 samples/window, 50% overlap)
        │
        ├──► Time-domain features (10)     — Mean, STD, RMS, Kurtosis, Crest Factor, etc.
        ├──► FFT features (10)             — Dominant frequency, spectral centroid/entropy, band energies
        └──► Envelope features (12)        — Bandpass filter (2–5 kHz) → Hilbert transform → envelope stats
        │
        ▼
Merge into single 32-feature dataset (row-aligned by window)
        │
        ▼
Preprocessing — mean imputation, inf handling, dedup, label encoding, StandardScaler
        │
        ▼
Corrupted-row removal — invalid labels + flat/near-zero signal windows
        │
        ▼
GroupShuffleSplit by file_name (80/20) — leakage-safe train/test split
        │
        ▼
Model training — XGBoost, SVM, LightGBM (baseline comparison)
        │
        ▼
Model selection — XGBoost (best generalization on leakage-safe split)
        │
        ▼
Deployment — Streamlit web app
```

---

## Feature engineering (32 features total)

### 1. Time-domain features (10)
Extracted directly from raw vibration windows: `Mean, STD, Variance, RMS, Max, Min, PeakToPeak, Skewness, Kurtosis, CrestFactor`.
Kurtosis and Crest Factor are the strongest fault indicators here — healthy bearings sit near Kurtosis ≈ 3, while faulty bearings show sharp impulsive spikes pushing Kurtosis well above that baseline.

### 2. FFT (frequency-domain) features (10)
A Hann-windowed FFT is applied to each window to extract: `dominant_frequency, dominant_amplitude, spectral_centroid, spectral_bandwidth, spectral_entropy`, and energy in 5 frequency bands (0–500 Hz through 4000–6000 Hz). These features reveal *where* in the frequency spectrum a fault is active — different fault types excite different characteristic frequencies (BPFI, BPFO, BSF) determined by bearing geometry.

### 3. Envelope features (12)
The signal is bandpass-filtered (2000–5000 Hz, 4th-order Butterworth, `filtfilt` for zero phase shift) to isolate the bearing's resonance band, then the amplitude envelope is extracted via the **Hilbert transform**. Twelve statistical features are computed on this envelope (`env_mean, env_std, env_rms, env_kurtosis, env_crest_factor, env_entropy`, etc.). This is the industry-standard "envelope analysis" technique for bearing diagnostics — it reveals the *repetition rate* of impact events that plain FFT smears out and misses, especially for early-stage faults.

**Why combine all three domains?** Each lens sees a different aspect of the same physical fault. A subtle fault might be nearly invisible in the time domain but obvious in the envelope spectrum. Feeding the model all three simultaneously (multi-domain feature fusion) makes it harder for any single fault signature to hide.

---

## Preprocessing

- Column name normalization, missing-value imputation (mean), infinite-value handling (from division-by-zero in Crest Factor / Kurtosis on flat signals)
- Duplicate row and duplicate column removal, empty-column removal
- Label encoding (`fault_type` → integer)
- Feature scaling via `StandardScaler` (essential for SVM, which is distance-based, and generally good practice for consistent feature contribution)
- A second, targeted cleaning pass before splitting: removed rows with invalid encoded labels and rows where all 32 features were near-zero (flat/corrupted signal windows) — an extra data-quality check beyond the basic preprocessing pass

---

## Train/test split methodology

```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in gss.split(X, y, groups=df["file_name"]):
    ...
```

Grouping by `file_name` guarantees no signal window from the same recording session appears in both train and test. Verified explicitly via:

```python
overlap = set(train_files) & set(test_files)
assert len(overlap) == 0   # zero data leakage, confirmed programmatically
```

Final split: **80% train / 20% test**, 1,061 test samples in the final XGBoost run.

---

## Model comparison

Three classifiers were trained and evaluated on the **same leakage-safe, group-based split**: XGBoost, Support Vector Machine (RBF kernel), and LightGBM.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Notes |
|---|---|---|---|---|---|
| **XGBoost (selected)** | **99.53%** | **99.48%** | **99.52%** | **99.50%** | Train/test gap: 0.47% — no meaningful overfitting |
| SVM (RBF kernel) | `[INSERT — see notebook]` | `[INSERT]` | `[INSERT]` | `[INSERT]` | |
| LightGBM | `[INSERT — see notebook]` | `[INSERT]` | `[INSERT]` | `[INSERT]` | |

*(SVM and LightGBM figures above are from the pre-leakage-fix run and are not reported here to avoid presenting misleading numbers — final post-fix values should be filled in from the saved notebook output before publishing.)*

### XGBoost — per-class performance (final model, post leakage-fix)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Ball | 0.98 | 1.00 | 0.99 | 235 |
| InnerRace | 1.00 | 1.00 | 1.00 | 471 |
| Normal | 1.00 | 0.98 | 0.99 | 237 |
| OuterRace | 1.00 | 1.00 | 1.00 | 118 |

5 total misclassifications out of 1,061 test samples — 4 of them were Normal windows misclassified, 1 was an InnerRace window misclassified. No Ball or OuterRace faults were missed (zero false negatives on those classes), which matters in a safety-critical monitoring context where a missed fault is more costly than a false alarm.

### Why XGBoost was selected

`[Add your specific reasoning here — e.g., highest accuracy/recall after the leakage fix, fastest inference, most interpretable feature importances, or best recall on the hardest class. I don't have your stated reason on record, so please fill this in with 1–2 honest sentences rather than leave it generic.]`

---

## Model deployment

The final XGBoost model is deployed as an interactive **Streamlit** web app, allowing a user to upload/select signal data and receive a real-time fault classification.

---

## Known limitations & honest caveats

- **Group-based splitting trades class balance for leakage safety.** The test set is not perfectly balanced across fault classes (see class distribution above) as a direct consequence of grouping by file. Metrics are reported as macro averages to account for this.
- **CWRU is a clean, lab-controlled dataset.** Real industrial vibration data is noisier and more variable than these recordings; accuracy in a production deployment would very likely be lower than reported here.
- **A file-numbering collision was caught and fixed during development**: in an early version of the FFT feature extraction script, file ID `130` was accidentally mapped to two different labels (InnerRace and OuterRace) due to a duplicate dictionary key, which silently caused the second definition to overwrite the first. This was identified and corrected before the final dataset was built.
- **This was trained and evaluated on a single fixed 80/20 split**, not k-fold cross-validation — a natural next step to further validate stability of the reported metrics across different file groupings.

---

## Tech stack

`Python` · `NumPy` · `Pandas` · `SciPy` (signal processing: FFT, Butterworth filter, Hilbert transform) · `scikit-learn` (preprocessing, GroupShuffleSplit, SVM) · `XGBoost` · `LightGBM` · `Matplotlib` / `Seaborn` (visualization) · `Streamlit` (deployment)

---

## Repository structure

```
├── 1_time_domain_features.py       # Time-domain feature extraction
├── 2_fft_features.py               # FFT feature extraction
├── 3_envelope_features.py          # Bandpass + Hilbert envelope feature extraction
├── 4_merge_datasets.py             # Merges all three feature sets, quality checks
├── 5_preprocessing.py              # Cleaning, encoding, scaling
├── 6_group_split.py                # Leakage-safe train/test split
├── 7_train_xgboost.py              # Final model training + evaluation
├── 8_feature_importance.py         # Feature importance analysis
├── app.py                          # Streamlit deployment
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
# Run feature extraction scripts 1-4 on the raw CWRU .mat files
# Run 5-7 in sequence to preprocess, split, and train
streamlit run app.py
```
