import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="ECG AI Clinical System",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 ECG AI Clinical System")
st.caption("AI-assisted ECG Analysis Prototype")

# -----------------------------
# MODEL LOADING (SAFE)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "ecg_model.pkl")

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error("❌ Model file not found. Please upload model.")
            st.stop()

        if os.path.getsize(MODEL_PATH) == 0:
            st.error("❌ Model file is empty/corrupted.")
            st.stop()

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        return model

    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

model = load_model()

# -----------------------------
# PREPROCESS
# -----------------------------
EXPECTED_LEN = 187

def prepare(signal):
    signal = np.array(signal).astype(float)

    # normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

    # resize
    if len(signal) < EXPECTED_LEN:
        signal = np.pad(signal, (0, EXPECTED_LEN - len(signal)))
    else:
        signal = signal[:EXPECTED_LEN]

    return signal

# -----------------------------
# SIMPLE CLASSIFICATION
# -----------------------------
def get_status(risk):
    if risk < 0.4:
        return "Normal Sinus Rhythm"
    elif risk < 0.75:
        return "Possible Arrhythmia"
    else:
        return "High Risk (Review Required)"

# -----------------------------
# UI INPUT
# -----------------------------
st.subheader("📂 Upload ECG CSV")

file = st.file_uploader("Upload ECG signal file", type=["csv"])

# -----------------------------
# MAIN LOGIC
# -----------------------------
if file:

    try:
        df = pd.read_csv(file)
        signal = df.iloc[:, 0].values

    except:
        st.error("Invalid CSV format. Must contain ECG signal column.")
        st.stop()

    processed = prepare(signal)
    X = processed.reshape(1, -1)

    # prediction
    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        risk = model.predict_proba(X)[0][1]
    else:
        risk = float(pred)

    status = get_status(risk)

    # peaks detection
    peaks, _ = find_peaks(processed, distance=20)

    # -----------------------------
    # RESULTS
    # -----------------------------
    st.subheader("🧠 Analysis Result")

    col1, col2 = st.columns(2)

    col1.metric("Risk Score", f"{risk:.2f}")
    col2.metric("Status", status)

    # -----------------------------
    # ECG PLOT
    # -----------------------------
    st.subheader("📈 ECG Waveform")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=processed,
        mode='lines',
        name='ECG Signal'
    ))

    fig.add_trace(go.Scatter(
        x=peaks,
        y=processed[peaks],
        mode='markers',
        marker=dict(color='red', size=6),
        name='R-Peaks'
    ))

    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Prototype System • Developed by Aditya Rawal")
