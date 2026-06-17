# =============================================================================
# INDUSTRIAL BEARING FAULT DETECTION SYSTEM — STREAMLIT FRONTEND
# =============================================================================
# The inference pipeline (signal loading, windowing, feature extraction,
# scaling, prediction) below is REUSED EXACTLY from the original pipeline.
# Only Streamlit UI code has been added around it. No ML logic is modified.
# =============================================================================

import os
import io
import pickle
import warnings
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
import matplotlib.pyplot as plt
import streamlit as st

from scipy.signal import butter, filtfilt
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION — Adjust these paths before running
# =============================================================================

SCALER_PATH        = "scaler.pkl"
MODEL_PATH          = "xgboost_model.pkl"
LABEL_ENCODER_PATH  = "label_encoder.pkl"

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
# STEP 1 — SIGNAL LOADING  (UNCHANGED)
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
# STEP 2 — WINDOWING  (UNCHANGED)
# =============================================================================

def create_windows(signal, window_size, step_size):
    windows = []
    start = 0
    while start + window_size <= len(signal):
        windows.append(signal[start: start + window_size])
        start += step_size
    return windows


# =============================================================================
# STEP 3 — TIME DOMAIN FEATURE EXTRACTION (10 features)  (UNCHANGED)
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

    rms = feats['rms']
    feats['crestfactor'] = np.max(np.abs(window)) / rms if rms > 0 else 0.0

    return feats


# =============================================================================
# STEP 4 — BANDPASS FILTER  (UNCHANGED)
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
# STEP 5 — ENVELOPE EXTRACTION (Hilbert Transform)  (UNCHANGED)
# =============================================================================

def extract_envelope(filtered_window):
    analytic = hilbert(filtered_window)
    return np.abs(analytic)


# =============================================================================
# STEP 6 — ENVELOPE FEATURE EXTRACTION (12 features)  (UNCHANGED)
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

    env_rms = feats['env_rms']
    feats['env_crest_factor'] = np.max(np.abs(envelope)) / env_rms if env_rms > 0 else 0.0

    feats['env_energy'] = np.sum(envelope ** 2)

    env_prob = envelope / np.sum(envelope) if np.sum(envelope) > 0 else envelope
    env_prob = env_prob[env_prob > 0]
    feats['env_entropy'] = scipy_entropy(env_prob)

    return feats


# =============================================================================
# STEP 7 — FFT FEATURE EXTRACTION (10 features)  (UNCHANGED)
# =============================================================================

def extract_fft_features(window, fs):
    n = len(window)

    hann     = np.hanning(n)
    windowed = window * hann

    fft_vals   = fft(windowed)
    freqs      = fftfreq(n, d=1.0 / fs)
    pos_mask   = freqs >= 0
    freqs      = freqs[pos_mask]
    amplitudes = np.abs(fft_vals[pos_mask]) / n
    amplitudes[1:-1] *= 2

    total_amp = np.sum(amplitudes)

    feats = {}

    if total_amp > 0:
        dominant_idx                = np.argmax(amplitudes)
        feats['dominant_frequency'] = freqs[dominant_idx]
        feats['dominant_amplitude'] = amplitudes[dominant_idx]
    else:
        feats['dominant_frequency'] = 0.0
        feats['dominant_amplitude'] = 0.0

    feats['spectral_centroid'] = (
        np.sum(freqs * amplitudes) / total_amp if total_amp > 0 else 0.0
    )

    centroid = feats['spectral_centroid']
    feats['spectral_bandwidth'] = (
        np.sqrt(np.sum(((freqs - centroid) ** 2) * amplitudes) / total_amp)
        if total_amp > 0 else 0.0
    )

    if total_amp > 0:
        prob = amplitudes / total_amp
        prob = prob[prob > 0]
        feats['spectral_entropy'] = scipy_entropy(prob)
    else:
        feats['spectral_entropy'] = 0.0

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
# STEP 8 — FULL FEATURE VECTOR FOR ONE WINDOW (all 32 features)  (UNCHANGED)
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
# STEP 9 — PROCESS ALL .mat FILES AND BUILD FEATURE MATRIX  (UNCHANGED LOGIC)
# =============================================================================
# Original process_files() scanned a directory. To integrate with Streamlit's
# file uploader (in-memory file, not a directory), we keep an unmodified
# directory-based version AND a thin wrapper that processes a single
# filepath using the exact same internal steps. No feature/window/scaling
# logic is altered.
# =============================================================================

def process_files(input_dir, window_size, step_size, fs, bp_low, bp_high, bp_order):
    """UNCHANGED original batch-directory version (kept for parity/back-compat)."""
    mat_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mat')])

    if not mat_files:
        return None

    all_rows = []

    for filename in mat_files:
        filepath = os.path.join(input_dir, filename)

        signal, info = load_signal(filepath)
        if signal is None:
            continue

        if len(signal) < window_size:
            continue

        windows = create_windows(signal, window_size, step_size)

        for win_idx, window in enumerate(windows):
            try:
                feats = extract_all_features(window, fs, bp_low, bp_high, bp_order)
                feats['file_name']    = filename
                feats['window_index'] = win_idx
                all_rows.append(feats)
            except Exception:
                continue

    if not all_rows:
        return None

    return pd.DataFrame(all_rows)


def process_single_file(filepath, filename, window_size, step_size, fs,
                         bp_low, bp_high, bp_order):
    """
    Thin wrapper around the unchanged per-window feature extraction steps,
    for processing exactly one uploaded file. Internally calls the same
    load_signal / create_windows / extract_all_features functions above
    with no modification to their logic.
    """
    signal, info = load_signal(filepath)
    if signal is None:
        return None, info

    if len(signal) < window_size:
        return None, f"Signal too short ({len(signal)} samples) for window size {window_size}"

    windows = create_windows(signal, window_size, step_size)

    all_rows = []
    for win_idx, window in enumerate(windows):
        try:
            feats = extract_all_features(window, fs, bp_low, bp_high, bp_order)
            feats['file_name']    = filename
            feats['window_index'] = win_idx
            all_rows.append(feats)
        except Exception:
            continue

    if not all_rows:
        return None, "No feature vectors could be extracted from this signal."

    return pd.DataFrame(all_rows), info


# =============================================================================
# STEP 10 — LOAD ARTIFACTS, SCALE AND PREDICT  (UNCHANGED)
# =============================================================================

def load_artifacts(scaler_path, model_path, label_encoder_path):
    artifacts = {}
    paths = {
        'scaler':        scaler_path,
        'model':         model_path,
        'label_encoder': label_encoder_path,
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found at: {path}. Please place the file alongside app.py."
            )
        with open(path, 'rb') as f:
            artifacts[name] = pickle.load(f)

    return artifacts['scaler'], artifacts['model'], artifacts['label_encoder']


def scale_and_predict(df_features, scaler, model, label_encoder):
    missing = [f for f in ALL_32_FEATURES if f not in df_features.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df_features[ALL_32_FEATURES].values
    X_scaled = scaler.transform(X)

    y_pred_numeric = model.predict(X_scaled)
    y_pred_proba   = model.predict_proba(X_scaled)
    y_pred_labels  = label_encoder.inverse_transform(y_pred_numeric)

    result_df = df_features[['file_name', 'window_index']].copy()
    result_df['predicted_label']      = y_pred_numeric
    result_df['predicted_fault_type'] = y_pred_labels
    result_df['confidence_%']         = (np.max(y_pred_proba, axis=1) * 100).round(2)

    class_names = label_encoder.classes_
    for i, cls in enumerate(class_names):
        result_df[f'prob_{cls}'] = (y_pred_proba[:, i] * 100).round(2)

    return result_df


# =============================================================================
# STREAMLIT APPLICATION — UI LAYER ONLY (no ML logic below this point)
# =============================================================================

st.set_page_config(
    page_title="Industrial Bearing Fault Detection System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------
# Theme / styling
# -----------------------------------------------------------------------

PRIMARY_DARK   = "#0F1B2D"   # deep industrial navy (banner)
ACCENT_STEEL   = "#4A8FB5"   # steel blue accent
ACCENT_AMBER   = "#D98E04"   # caution amber for alerts/highlights
BG_LIGHT       = "#1C2733"   # dark slate background (replaces bright white)
CARD_BG        = "#26323F"   # slightly lighter slate for cards
TEXT_PRIMARY   = "#E8EDF2"   # main text on dark background
TEXT_MUTED     = "#9AAAB8"
BORDER_COLOR   = "#3A4856"

FAULT_COLORS = {
    "Normal":    "#2E8B57",
    "Ball":      "#C0392B",
    "InnerRace": "#D98E04",
    "OuterRace": "#E07B00",
}
FAULT_ICONS = {
    "Normal":    "✅",
    "Ball":      "🔴",
    "InnerRace": "🟡",
    "OuterRace": "🟠",
}

st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG_LIGHT};
    }}
    [data-testid="stHeader"] {{
        background-color: {BG_LIGHT};
    }}
    [data-testid="stSidebar"] {{
        background-color: {PRIMARY_DARK};
    }}
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {{
        color: {TEXT_PRIMARY};
    }}
    .title-banner {{
        background-color: {PRIMARY_DARK};
        padding: 2.2rem 2rem 1.6rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid {BORDER_COLOR};
    }}
    .title-banner h1 {{
        color: #F5C24B;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: 0.5px;
    }}
    .title-banner p {{
        color: #C7D5E0;
        font-style: italic;
        font-size: 1.02rem;
        max-width: 900px;
        line-height: 1.5;
        margin: 0;
    }}
    .section-heading {{
        color: {TEXT_PRIMARY};
        font-weight: 700;
        font-size: 1.3rem;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        border-left: 5px solid {ACCENT_STEEL};
        padding-left: 0.7rem;
    }}
    div.stButton > button {{
        background-color: {ACCENT_STEEL};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: background-color 0.2s ease;
    }}
    div.stButton > button:hover {{
        background-color: {ACCENT_AMBER};
        color: white;
    }}
    [data-testid="stMetricValue"] {{
        font-weight: 800;
        color: {TEXT_PRIMARY};
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED};
    }}
    [data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border-radius: 10px;
        padding: 1rem 1rem 0.6rem 1rem;
        border: 1px solid {BORDER_COLOR};
        box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    }}
    .fault-card {{
        background-color: {CARD_BG};
        border-radius: 10px;
        border: 1px solid {BORDER_COLOR};
        padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    }}
    [data-testid="stFileUploader"] {{
        background-color: {CARD_BG};
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid {BORDER_COLOR};
    }}
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
    }}
    .footer-note {{
        color: {TEXT_MUTED};
        font-size: 0.8rem;
        margin-top: 2rem;
    }}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------
# Session state initialisation
# -----------------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts

if "last_result_df" not in st.session_state:
    st.session_state.last_result_df = None

if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None


def go_to(page_name):
    st.session_state.page = page_name


# -----------------------------------------------------------------------
# Cached artifact loading (scaler / model / label encoder loaded once)
# -----------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts(SCALER_PATH, MODEL_PATH, LABEL_ENCODER_PATH)


# =========================================================================
# PAGE 1 — HOME
# =========================================================================

def show_home_page():
    st.markdown(f"""
    <div class="title-banner">
        <h1>⚙️ Industrial Bearing Fault Detection System</h1>
        <p>An end-to-end bearing fault diagnosis system that combines signal processing,
        feature engineering, and machine learning to detect bearing health conditions
        from vibration data. The application automatically analyzes uploaded sensor
        signals and provides fault predictions with confidence scores.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Start Analysis</div>', unsafe_allow_html=True)
    st.write(
        "Upload a vibration signal recording (.mat file) and run the diagnostic "
        "pipeline to classify bearing health condition — Normal, Ball, Inner Race, "
        "or Outer Race fault — with full confidence breakdowns."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Make Prediction", use_container_width=True):
            go_to("predict")
            st.rerun()
    with col2:
        if st.button("📜 Prediction History", use_container_width=True):
            go_to("history")
            st.rerun()

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🧩 32 Engineered Features**")
        st.caption("Time-domain, FFT spectral, and envelope features extracted per window.")
    with c2:
        st.markdown("**🤖 XGBoost Classifier**")
        st.caption("Trained model with calibrated class probabilities.")
    with c3:
        st.markdown("**📊 Window-Level Voting**")
        st.caption("Majority vote across overlapping windows for robust file-level diagnosis.")

    st.markdown(
        '<div class="footer-note">Industrial Bearing Fault Detection System &middot; '
        'Signal Processing &amp; Machine Learning Pipeline</div>',
        unsafe_allow_html=True
    )


# =========================================================================
# PAGE 2 — PREDICTION
# =========================================================================

def show_prediction_page():
    st.markdown('<div class="section-heading">Upload Vibration Signal File</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a .mat file", type=["mat"])

    if uploaded_file is not None:
        st.success(f"📄 File selected: **{uploaded_file.name}**")

        run_clicked = st.button("▶️ Run Analysis", use_container_width=False)

        if run_clicked:
            with st.spinner("Running signal processing and inference pipeline..."):
                try:
                    # Write uploaded bytes to a temp file because load_signal()
                    # expects a filepath (uses scipy.io.loadmat under the hood).
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    df_features, info = process_single_file(
                        tmp_path, uploaded_file.name,
                        WINDOW_SIZE, STEP_SIZE, SAMPLING_FREQ,
                        BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER
                    )

                    os.unlink(tmp_path)

                    if df_features is None:
                        st.error(f"❌ Could not process file: {info}")
                        return

                    scaler, model, label_encoder = get_artifacts()
                    result_df = scale_and_predict(df_features, scaler, model, label_encoder)

                except FileNotFoundError as e:
                    st.error(f"❌ {e}")
                    return
                except Exception as e:
                    st.error(f"❌ Pipeline error: {e}")
                    return

            st.session_state.last_result_df = result_df
            st.session_state.last_file_name = uploaded_file.name

            # ---- Save to history ----
            vote_counts    = result_df['predicted_fault_type'].value_counts()
            majority_fault = vote_counts.index[0]
            majority_count = vote_counts.iloc[0]
            n_windows      = len(result_df)
            avg_confidence = result_df['confidence_%'].mean()

            st.session_state.history.append({
                "Timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "File Name":         uploaded_file.name,
                "Predicted Fault":   majority_fault,
                "Average Confidence": round(avg_confidence, 2),
                "Total Windows":     n_windows,
            })

    # ---- Display results if available ----
    if st.session_state.last_result_df is not None:
        result_df = st.session_state.last_result_df
        file_name = st.session_state.last_file_name

        vote_counts    = result_df['predicted_fault_type'].value_counts()
        majority_fault = vote_counts.index[0]
        majority_count = vote_counts.iloc[0]
        n_windows      = len(result_df)
        vote_pct       = (majority_count / n_windows) * 100
        avg_confidence = result_df['confidence_%'].mean()
        icon           = FAULT_ICONS.get(majority_fault, "❓")

        st.markdown("---")
        st.markdown(f'<div class="section-heading">Diagnosis Result — {file_name}</div>',
                    unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Predicted Fault Type", f"{icon} {majority_fault}")
        with m2:
            st.metric("Confidence Score", f"{avg_confidence:.1f}%")
        with m3:
            st.metric("Total Windows Analysed", f"{n_windows}")
        with m4:
            st.metric("Majority Vote", f"{majority_count}/{n_windows}", f"{vote_pct:.1f}% agreed")

        # ---- Charts ----
        st.markdown('<div class="section-heading">Diagnostic Charts</div>',
                    unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**Fault Distribution Across Windows**")
            fig1, ax1 = plt.subplots(figsize=(4, 2.6))
            fig1.patch.set_facecolor(CARD_BG)
            ax1.set_facecolor(CARD_BG)
            counts = result_df['predicted_fault_type'].value_counts()
            colors = [FAULT_COLORS.get(c, "#888888") for c in counts.index]
            ax1.bar(counts.index, counts.values, color=colors)
            ax1.set_ylabel("Window Count", color=TEXT_PRIMARY)
            ax1.set_xlabel("Fault Class", color=TEXT_PRIMARY)
            ax1.tick_params(colors=TEXT_PRIMARY)
            for spine in ax1.spines.values():
                spine.set_color(BORDER_COLOR)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            st.pyplot(fig1, use_container_width=True)

        with chart_col2:
            st.markdown("**Confidence Score Distribution**")
            fig2, ax2 = plt.subplots(figsize=(4, 2.8))
            fig2.patch.set_facecolor(CARD_BG)
            ax2.set_facecolor(CARD_BG)
            ax2.hist(result_df['confidence_%'], bins=15, color=ACCENT_STEEL, edgecolor=CARD_BG)
            ax2.set_xlabel("Confidence (%)", color=TEXT_PRIMARY)
            ax2.set_ylabel("Number of Windows", color=TEXT_PRIMARY)
            ax2.tick_params(colors=TEXT_PRIMARY)
            for spine in ax2.spines.values():
                spine.set_color(BORDER_COLOR)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            st.pyplot(fig2, use_container_width=True)

        st.markdown("**Predicted Class Share (Pie Chart)**")
        fig3, ax3 = plt.subplots(figsize=(3.1, 3.1))
        fig3.patch.set_facecolor(CARD_BG)
        ax3.set_facecolor(CARD_BG)
        counts = result_df['predicted_fault_type'].value_counts()
        colors = [FAULT_COLORS.get(c, "#888888") for c in counts.index]
        wedges, texts, autotexts = ax3.pie(
            counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90,
            wedgeprops={"edgecolor": CARD_BG, "linewidth": 1.5},
            textprops={"color": TEXT_PRIMARY}
        )
        for autotext in autotexts:
            autotext.set_color("#FFFFFF")
        ax3.axis('equal')
        st.pyplot(fig3, use_container_width=False)

        # ---- Window-level table ----
        st.markdown('<div class="section-heading">Window-Level Prediction Table</div>',
                    unsafe_allow_html=True)
        display_cols = ['file_name', 'window_index', 'predicted_fault_type', 'confidence_%']
        st.dataframe(result_df[display_cols], use_container_width=True, height=350)

        csv_buffer = io.StringIO()
        result_df[display_cols].to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{file_name}_predictions.csv",
            mime="text/csv",
        )

    st.markdown("---")
    if st.button("⬅️ Back to Home"):
        go_to("home")
        st.rerun()


# =========================================================================
# PAGE 3 — PREDICTION HISTORY
# =========================================================================

def show_history_page():
    st.markdown('<div class="section-heading">Prediction History</div>',
                unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No predictions have been made yet.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, height=400)

        csv_buffer = io.StringIO()
        history_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download History as CSV",
            data=csv_buffer.getvalue(),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

    st.markdown("---")
    if st.button("⬅️ Back to Home"):
        go_to("home")
        st.rerun()


# =========================================================================
# ROUTER
# =========================================================================

if st.session_state.page == "home":
    show_home_page()
elif st.session_state.page == "predict":
    show_prediction_page()
elif st.session_state.page == "history":
    show_history_page()