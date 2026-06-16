# =============================================================================
# INDUSTRIAL BEARING FAULT DETECTION - INFERENCE PIPELINE
# =============================================================================
# Reads N .mat files, extracts all 32 features, scales, predicts fault type.
#
# ALL 32 FEATURES USED (matching training exactly):
#   Time Domain : mean, std, variance, rms, max, min, peaktopeak,
#                 skewness, kurtosis, crestfactor
#   FFT         : dominant_frequency, dominant_amplitude, spectral_centroid,
#                 spectral_bandwidth, spectral_entropy, band_energy_0_500,
#                 band_energy_500_1000, band_energy_1000_2000,
#                 band_energy_2000_4000, band_energy_4000_6000
#   Envelope    : env_mean, env_std, env_variance, env_rms, env_max, env_min,
#                 env_peak_to_peak, env_skewness, env_kurtosis,
#                 env_crest_factor, env_energy, env_entropy
# =============================================================================

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
import pickle
import warnings

from scipy.signal import butter, filtfilt
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION — Adjust these paths before running
# =============================================================================

INPUT_DIR          = "."
SCALER_PATH        = "scaler.pkl"
MODEL_PATH         = "xgboost_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

WINDOW_SIZE   = 2048
STEP_SIZE     = 1024
SAMPLING_FREQ = 12000

BANDPASS_LOW  = 2000
BANDPASS_HIGH = 5000
FILTER_ORDER  = 4

# =============================================================================
# ALL 32 FEATURES — in the EXACT ORDER used during training
# This order MUST match X_train column order exactly
# =============================================================================

ALL_32_FEATURES = [
    'mean', 'std', 'variance', 'rms', 'max', 'min', 'skewness', 'kurtosis',
    'dominant_frequency', 'dominant_amplitude', 'spectral_centroid',
    'spectral_bandwidth', 'spectral_entropy', 'env_mean', 'env_std',
    'env_variance', 'env_rms', 'env_max', 'env_min', 'env_peak_to_peak',
    'env_skewness', 'env_kurtosis', 'env_crest_factor', 'peaktopeak',
    'crestfactor', 'band_energy_0_500', 'band_energy_500_1000',
    'band_energy_1000_2000', 'band_energy_2000_4000', 'band_energy_4000_6000',
    'env_energy', 'env_entropy'
]

# =============================================================================
# STEP 1 — SIGNAL LOADING
# =============================================================================

def load_signal(filepath):
    try:
        mat_data = sio.loadmat(filepath)
    except Exception as e:
        return None, f"Cannot open file: {e}"

    preferred_suffixes = ["_DE_time", "_BA_time", "_FE_time"]

    for suffix in preferred_suffixes:
        for key in mat_data.keys():
            if key.startswith("__"):
                continue
            if key.endswith(suffix):
                raw = mat_data[key].flatten().astype(np.float64)
                if len(raw) == 0:
                    return None, "Signal is empty"
                if np.any(np.isnan(raw)):
                    return None, f"Signal has {np.sum(np.isnan(raw))} NaN values"
                if np.any(np.isinf(raw)):
                    return None, f"Signal has {np.sum(np.isinf(raw))} infinite values"
                return raw, key

    available = [k for k in mat_data.keys() if not k.startswith("__")]
    return None, f"No recognisable signal variable. Available keys: {available}"


# =============================================================================
# STEP 2 — WINDOWING
# =============================================================================

def create_windows(signal, window_size, step_size):
    windows = []
    start = 0
    while start + window_size <= len(signal):
        windows.append(signal[start : start + window_size])
        start += step_size
    return windows


# =============================================================================
# STEP 3 — TIME DOMAIN FEATURE EXTRACTION (10 features)
# =============================================================================

def extract_time_features(window):
    feats = {}

    feats['mean']       = np.mean(window)
    feats['std']        = np.std(window)
    feats['variance']   = np.var(window)
    feats['rms']        = np.sqrt(np.mean(window ** 2))
    feats['max']        = np.max(window)
    feats['min']        = np.min(window)
    feats['peaktopeak'] = np.max(window) - np.min(window)
    feats['skewness']   = stats.skew(window)
    feats['kurtosis']   = stats.kurtosis(window, fisher=False)

    # Crest factor = max absolute value / RMS
    rms = feats['rms']
    feats['crestfactor'] = np.max(np.abs(window)) / rms if rms > 0 else 0.0

    return feats


# =============================================================================
# STEP 4 — BANDPASS FILTER
# =============================================================================

def apply_bandpass_filter(window, low_hz, high_hz, order, fs):
    nyquist = fs / 2.0
    low  = low_hz  / nyquist
    high = high_hz / nyquist

    if low <= 0 or high >= 1 or low >= high:
        return window

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, window)


# =============================================================================
# STEP 5 — ENVELOPE EXTRACTION (Hilbert Transform)
# =============================================================================

def extract_envelope(filtered_window):
    analytic = hilbert(filtered_window)
    return np.abs(analytic)


# =============================================================================
# STEP 6 — ENVELOPE FEATURE EXTRACTION (12 features)
# =============================================================================

def extract_envelope_features(envelope):
    feats = {}

    feats['env_mean']         = np.mean(envelope)
    feats['env_std']          = np.std(envelope)
    feats['env_variance']     = np.var(envelope)
    feats['env_rms']          = np.sqrt(np.mean(envelope ** 2))
    feats['env_max']          = np.max(envelope)
    feats['env_min']          = np.min(envelope)
    feats['env_peak_to_peak'] = np.max(envelope) - np.min(envelope)
    feats['env_skewness']     = stats.skew(envelope)
    feats['env_kurtosis']     = stats.kurtosis(envelope, fisher=False)

    # Envelope crest factor
    env_rms = feats['env_rms']
    feats['env_crest_factor'] = np.max(np.abs(envelope)) / env_rms if env_rms > 0 else 0.0

    # Envelope energy
    feats['env_energy'] = np.sum(envelope ** 2)

    # Envelope entropy
    env_prob = envelope / np.sum(envelope) if np.sum(envelope) > 0 else envelope
    env_prob = env_prob[env_prob > 0]
    feats['env_entropy'] = scipy_entropy(env_prob)

    return feats


# =============================================================================
# STEP 7 — FFT FEATURE EXTRACTION (10 features)
# =============================================================================

def extract_fft_features(window, fs):
    n = len(window)

    # Apply Hann window
    hann     = np.hanning(n)
    windowed = window * hann

    # FFT — single sided
    fft_vals   = fft(windowed)
    freqs      = fftfreq(n, d=1.0 / fs)
    pos_mask   = freqs >= 0
    freqs      = freqs[pos_mask]
    amplitudes = np.abs(fft_vals[pos_mask]) / n
    amplitudes[1:-1] *= 2

    total_amp = np.sum(amplitudes)

    feats = {}

    # Dominant frequency and amplitude
    if total_amp > 0:
        dominant_idx               = np.argmax(amplitudes)
        feats['dominant_frequency'] = freqs[dominant_idx]
        feats['dominant_amplitude'] = amplitudes[dominant_idx]
    else:
        feats['dominant_frequency'] = 0.0
        feats['dominant_amplitude'] = 0.0

    # Spectral centroid
    feats['spectral_centroid'] = (
        np.sum(freqs * amplitudes) / total_amp if total_amp > 0 else 0.0
    )

    # Spectral bandwidth
    centroid = feats['spectral_centroid']
    feats['spectral_bandwidth'] = (
        np.sqrt(np.sum(((freqs - centroid) ** 2) * amplitudes) / total_amp)
        if total_amp > 0 else 0.0
    )

    # Spectral entropy
    if total_amp > 0:
        prob = amplitudes / total_amp
        prob = prob[prob > 0]
        feats['spectral_entropy'] = scipy_entropy(prob)
    else:
        feats['spectral_entropy'] = 0.0

    # Band energies — matching training bands exactly
    def band_energy(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return np.sum(amplitudes[mask] ** 2)

    feats['band_energy_0_500']     = band_energy(0,    500)
    feats['band_energy_500_1000']  = band_energy(500,  1000)
    feats['band_energy_1000_2000'] = band_energy(1000, 2000)
    feats['band_energy_2000_4000'] = band_energy(2000, 4000)
    feats['band_energy_4000_6000'] = band_energy(4000, 6000)

    return feats


# =============================================================================
# STEP 8 — FULL FEATURE VECTOR FOR ONE WINDOW (all 32 features)
# =============================================================================

def extract_all_features(window, fs, bp_low, bp_high, bp_order):
    time_feats = extract_time_features(window)

    filtered  = apply_bandpass_filter(window, bp_low, bp_high, bp_order, fs)
    envelope  = extract_envelope(filtered)
    env_feats = extract_envelope_features(envelope)

    fft_feats = extract_fft_features(window, fs)

    all_feats = {}
    all_feats.update(time_feats)
    all_feats.update(env_feats)
    all_feats.update(fft_feats)

    return all_feats


# =============================================================================
# STEP 9 — PROCESS ALL .mat FILES AND BUILD FEATURE MATRIX
# =============================================================================

def process_files(input_dir, window_size, step_size, fs, bp_low, bp_high, bp_order):
    print("=" * 65)
    print("SCANNING FOR .mat FILES")
    print("=" * 65)

    mat_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mat')])

    if not mat_files:
        print(f"  No .mat files found in: {input_dir}")
        print("  Please upload your .mat files and re-run.")
        return None

    print(f"  Found {len(mat_files)} .mat file(s):")
    for f in mat_files:
        print(f"    • {f}")

    all_rows         = []
    files_successful = []
    files_failed     = []

    for filename in mat_files:
        filepath = os.path.join(input_dir, filename)
        print(f"\n{'─' * 55}")
        print(f"  Processing: {filename}")

        signal, info = load_signal(filepath)
        if signal is None:
            print(f"  ✗ Load failed: {info}")
            files_failed.append((filename, info))
            continue

        print(f"  ✓ Signal loaded via key: {info}")
        print(f"    Length: {len(signal):,} samples  "
              f"({len(signal)/fs:.2f} sec at {fs} Hz)")

        if len(signal) < window_size:
            reason = (f"Signal too short ({len(signal)} samples) "
                      f"for window size {window_size}")
            print(f"  ✗ {reason}")
            files_failed.append((filename, reason))
            continue

        windows   = create_windows(signal, window_size, step_size)
        n_windows = len(windows)
        print(f"    Windows: {n_windows}  "
              f"(size={window_size}, step={step_size}, 50% overlap)")

        file_rows = 0
        for win_idx, window in enumerate(windows):
            try:
                feats = extract_all_features(window, fs, bp_low, bp_high, bp_order)
                feats['file_name']    = filename
                feats['window_index'] = win_idx
                all_rows.append(feats)
                file_rows += 1
            except Exception as e:
                print(f"    ⚠ Window {win_idx} skipped: {e}")

        print(f"  ✓ {file_rows} feature vectors extracted")
        files_successful.append(filename)

    print(f"\n{'=' * 65}")
    print(f"  Files processed : {len(files_successful)}")
    print(f"  Files failed    : {len(files_failed)}")
    for fname, reason in files_failed:
        print(f"    ✗ {fname}: {reason}")
    print(f"  Total windows   : {len(all_rows)}")
    print(f"{'=' * 65}")

    if not all_rows:
        print("  No windows extracted. Cannot proceed.")
        return None

    return pd.DataFrame(all_rows)


# =============================================================================
# STEP 10 — LOAD ARTIFACTS, SCALE AND PREDICT
# =============================================================================

def load_artifacts(scaler_path, model_path, label_encoder_path):
    print("\n" + "=" * 65)
    print("LOADING MODEL ARTIFACTS")
    print("=" * 65)

    artifacts = {}
    paths = {
        'scaler':        scaler_path,
        'model':         model_path,
        'label_encoder': label_encoder_path,
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"  ✗ {name} not found at: {path}\n"
                f"    Please upload the file and check the path."
            )
        with open(path, 'rb') as f:
            artifacts[name] = pickle.load(f)
        print(f"  ✓ {name} loaded from: {path}")

    return artifacts['scaler'], artifacts['model'], artifacts['label_encoder']


def scale_and_predict(df_features, scaler, model, label_encoder):
    print("\n" + "=" * 65)
    print("SCALING FEATURES AND RUNNING PREDICTIONS")
    print("=" * 65)

    # Verify all 32 features are present
    missing = [f for f in ALL_32_FEATURES if f not in df_features.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Select all 32 features in exact training order
    X = df_features[ALL_32_FEATURES].values
    print(f"  Feature matrix shape: {X.shape}  (should be [n_windows, 32])")

    X_scaled = scaler.transform(X)
    print("  ✓ Features scaled with loaded scaler")

    y_pred_numeric = model.predict(X_scaled)
    y_pred_proba   = model.predict_proba(X_scaled)
    y_pred_labels  = label_encoder.inverse_transform(y_pred_numeric)

    print("  ✓ Predictions complete")

    result_df = df_features[['file_name', 'window_index']].copy()
    result_df['predicted_label']      = y_pred_numeric
    result_df['predicted_fault_type'] = y_pred_labels
    result_df['confidence_%']         = (np.max(y_pred_proba, axis=1) * 100).round(2)

    class_names = label_encoder.classes_
    for i, cls in enumerate(class_names):
        result_df[f'prob_{cls}'] = (y_pred_proba[:, i] * 100).round(2)

    return result_df


# =============================================================================
# STEP 11 — DISPLAY RESULTS
# =============================================================================

def display_results(result_df):
    print("\n" + "=" * 65)
    print("PREDICTION RESULTS")
    print("=" * 65)

    files = result_df['file_name'].unique()

    print("\n📋 FILE-LEVEL SUMMARY (Majority Vote across all windows)")
    print("─" * 65)

    file_summaries = []
    for filename in sorted(files):
        file_rows      = result_df[result_df['file_name'] == filename]
        n_windows      = len(file_rows)
        vote_counts    = file_rows['predicted_fault_type'].value_counts()
        majority_fault = vote_counts.index[0]
        majority_count = vote_counts.iloc[0]
        confidence_pct = (majority_count / n_windows) * 100
        avg_confidence = file_rows['confidence_%'].mean()

        file_summaries.append({
            'File':            filename,
            'Windows':         n_windows,
            'Predicted Fault': majority_fault,
            'Vote Confidence': f"{confidence_pct:.1f}%",
            'Avg Model Conf.': f"{avg_confidence:.1f}%",
        })

        emoji = {'Normal': '✅', 'Ball': '🔴', 'InnerRace': '🟡', 'OuterRace': '🟠'}
        icon  = emoji.get(majority_fault, '❓')

        print(f"\n  {icon}  File      : {filename}")
        print(f"     Fault     : {majority_fault}")
        print(f"     Windows   : {n_windows}")
        print(f"     Vote      : {majority_count}/{n_windows} windows agreed "
              f"({confidence_pct:.1f}%)")
        print(f"     Avg Conf. : {avg_confidence:.1f}%")

        if len(vote_counts) > 1:
            print(f"     Breakdown :")
            for fault, count in vote_counts.items():
                pct = (count / n_windows) * 100
                print(f"       {fault:<12}: {count:>4} windows ({pct:.1f}%)")

    print("\n\n📊 WINDOW-LEVEL DETAIL TABLE")
    print("─" * 65)
    display_cols = ['file_name', 'window_index', 'predicted_fault_type', 'confidence_%']
    print(result_df[display_cols].to_string(index=False))

    print("\n\n📈 OVERALL STATISTICS")
    print("─" * 65)

    fault_counts = result_df['predicted_fault_type'].value_counts()
    total_windows = len(result_df)
    print(f"\n  Total windows analysed : {total_windows}")
    for fault, count in fault_counts.items():
        pct = (count / total_windows) * 100
        print(f"  {fault:<15}: {count:>5} windows ({pct:.1f}%)")

    return pd.DataFrame(file_summaries)


# =============================================================================
# MAIN
# =============================================================================

def main():
    df_features = process_files(
        INPUT_DIR, WINDOW_SIZE, STEP_SIZE, SAMPLING_FREQ,
        BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER
    )
    if df_features is None:
        return

    scaler, model, label_encoder = load_artifacts(
        SCALER_PATH, MODEL_PATH, LABEL_ENCODER_PATH
    )

    result_df = scale_and_predict(df_features, scaler, model, label_encoder)
    display_results(result_df)

    print("\n✅ Done.")

if __name__ == "__main__":
  main()