# Industrial Bearing Fault Diagnosis using Multi-Domain Signal Processing and Leakage-Safe Machine Learning

A machine learning pipeline that detects and classifies bearing faults (Normal, Ball, Inner Race, Outer Race) from raw vibration signals, using a custom-built multi-domain feature engineering pipeline (time-domain, FFT, and envelope analysis) on the CWRU Bearing Dataset.

## Project Highlights

- Built an end-to-end predictive maintenance pipeline from raw vibration signals to deployment.
- Engineered 32 features across time, frequency, and envelope domains.
- Identified and eliminated data leakage caused by overlapping signal windows using `GroupShuffleSplit`.
- Achieved **99.53% test accuracy** and **99.50% macro F1** on a strict leakage-safe evaluation.
- Deployed the selected XGBoost model as an interactive Streamlit application.

**Streamlit Demo Link**
https://industry-bearing-fault-detection-nwm9exqgcb3hibericinjm.streamlit.app/

<img width="1892" height="907" alt="image" src="https://github.com/user-attachments/assets/e71fe26b-184d-4fcd-ac2a-4eea03f20f9c" />

<img width="1896" height="922" alt="image" src="https://github.com/user-attachments/assets/5bdd72d7-0d03-4104-b5fa-dd53a9b19585" />

<img width="1900" height="596" alt="image" src="https://github.com/user-attachments/assets/68f5c9ae-d61e-4631-b27d-f481d55d2a7b" />


## Why this project matters

Bearing failures are one of the most common causes of unplanned downtime in rotating machinery. Catching a fault early — and correctly identifying *which* component is failing (ball, inner race, or outer race) — lets maintenance teams intervene before a full breakdown. This project builds an end-to-end pipeline that goes from raw accelerometer signals to a deployable fault classifier, using signal processing techniques (FFT, bandpass filtering, Hilbert transform envelope analysis) borrowed directly from industrial vibration analysis practice, not just generic tabular ML.

---

## A note on rigor: catching my own data leakage


**First pass:** with a normal random train/test split, every model — XGBoost, SVM, and LightGBM — scored **100% accuracy, 100% precision, 100% recall, 100% F1**, with a 0.00% train/test gap.

A perfect score across every metric and every class is not a good sign — it's a red flag. In this case, the cause was overlapping signal windows. The signals were segmented using 50% overlapping windows, a common practice in vibration analysis to avoid missing transient fault impulses that may occur near the boundary of a window. Without overlap, an important spike could be split or truncated during segmentation; with 50% overlap, that same event is fully captured in at least one of the adjacent windows while also increasing the number of training samples. However, this introduced a subtle issue: windows extracted from the same recording file share much of the same underlying signal. When a random train/test split was used, overlapping windows from the same file ended up in both the training and testing sets, allowing information to leak between them even though no individual window was duplicated.

**The fix:** I re-did the split using `GroupShuffleSplit` from scikit-learn, grouping by `file_name` — so every window from a given `.mat` file goes entirely into either train or test, never both. I explicitly verified zero file overlap between splits with an automated assertion before training.

**Result after the fix:** XGBoost's accuracy dropped from a suspicious 100% to a much more credible **99.53%**, with a small but non-zero train/test gap (100.00% train vs. 99.53% test = 0.47% gap) — consistent with a model that has genuinely learned, rather than memorized, the fault signatures.

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

- Column name normalization, missing-value imputation (mean), and infinite-value handling (from division-by-zero in features such as Crest Factor and Kurtosis on flat signals)
- Duplicate row, duplicate column, and empty-column removal
- Label encoding of the target variable (`fault_type`) into integer classes: **Ball → 0, InnerRace → 1, Normal → 2, OuterRace → 3**
- Feature scaling using `StandardScaler` (essential for SVM, which is distance-based, and good practice for ensuring consistent feature contribution across all models)
- A second, targeted data-quality check before splitting: removed rows with invalid encoded labels and rows where all 32 features were near-zero (flat/corrupted signal windows), preventing poor-quality samples from reaching the training pipeline

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
| SVM (RBF kernel) | **99** | **97.82** | **99.41** | **99.10** | |
| LightGBM | **97** | **100** | **100** | **99.23** | |

### XGBoost — per-class performance (final model, post leakage-fix)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Ball | 0.98 | 1.00 | 0.99 | 235 |
| InnerRace | 1.00 | 1.00 | 1.00 | 471 |
| Normal | 1.00 | 0.98 | 0.99 | 237 |
| OuterRace | 1.00 | 1.00 | 1.00 | 118 |

5 total misclassifications out of 1,061 test samples — 4 of them were Normal windows misclassified, 1 was an InnerRace window misclassified. No Ball or OuterRace faults were missed (zero false negatives on those classes), which matters in a safety-critical monitoring context where a missed fault is more costly than a false alarm.

### Why XGBoost was selected

XGBoost was selected after benchmarking it against SVM (RBF kernel) and LightGBM on the same leakage-safe, group-based train/test split. It achieved the strongest overall performance, delivering **99.53% test accuracy**, **99.50% macro F1**, and a small **0.47% train/test gap**, indicating good generalization with no meaningful overfitting. In addition to its predictive performance, XGBoost provides feature importance scores, making it easier to interpret which time-domain, frequency-domain, and envelope features contributed most to the final predictions.

<img width="737" height="555" alt="image" src="https://github.com/user-attachments/assets/39370ec0-f4b3-4f92-b872-b793d3e8abf4" />

<img width="1337" height="725" alt="image" src="https://github.com/user-attachments/assets/da1c0e58-833e-4047-91e1-5b2f9bde3f44" />

## Model deployment

The final XGBoost model is deployed as an interactive **Streamlit** web app, allowing a user to upload/select signal data and receive a real-time fault classification.

---

## Known limitations & honest caveats

- **Group-based splitting trades class balance for leakage safety.** The test set is not perfectly balanced across fault classes as a direct consequence of grouping by file. Metrics are reported as macro averages to account for this.
- **CWRU is a clean, lab-controlled dataset.** Real industrial vibration data is noisier and more variable than these recordings; accuracy in a production deployment would very likely be lower than reported here.
- **A file-numbering collision was caught and fixed during development**: in an early version of the FFT feature extraction script, file ID `130` was accidentally mapped to two different labels (InnerRace and OuterRace) due to a duplicate dictionary key, which silently caused the second definition to overwrite the first. This was identified and corrected before the final dataset was built.
- **This was trained and evaluated on a single fixed 80/20 split**, not k-fold cross-validation — a natural next step to further validate stability of the reported metrics across different file groupings.

---

## Tech stack

`Python` · `NumPy` · `Pandas` · `SciPy` (signal processing: FFT, Bandpass filter, Hilbert transform) · `scikit-learn` (preprocessing, GroupShuffleSplit, SVM) · `XGBoost` · `LightGBM` · `Matplotlib` / `Seaborn` (visualization) · `Streamlit` (deployment)

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
