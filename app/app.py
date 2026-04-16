import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import plotly.graph_objects as go
from scipy.signal import find_peaks

# PDF
import tempfile
from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="ECG AI Clinical System", page_icon="🫀", layout="wide")

# =========================================================
# DATABASE
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ecg_clinic.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            age INTEGER,
            patient_id TEXT,
            risk REAL,
            status TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_record(name, age, pid, risk, status):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO records (patient_name, age, patient_id, risk, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (name, age, pid, float(risk), status, str(datetime.now())))
    conn.commit()
    conn.close()

# =========================================================
# MODEL
# =========================================================
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "ecg_model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ ECG Model not found.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# =========================================================
# UI
# =========================================================
st.markdown("<h1 style='color:#ff2d2d;'>🫀 CLINICAL ECG SYSTEM</h1>", unsafe_allow_html=True)
st.caption("AI-assisted Clinical Decision Support System")

mode = st.sidebar.selectbox("Mode", ["Single Patient", "Batch Research"])

EXPECTED_LEN = 187

# =========================================================
# PREPROCESS
# =========================================================
def prepare(signal):
    signal = np.array(signal).astype(float)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
    return signal

# =========================================================
# CLINICAL LABELS
# =========================================================
def clinical_status_class(pred_class):
    mapping = {
        0: "Normal Sinus Rhythm",
        1: "Supraventricular Arrhythmia",
        2: "Ventricular Arrhythmia",
        3: "Fusion Beat",
        4: "Unknown / Artifact"
    }
    return mapping.get(pred_class, "Unknown")

# =========================================================
# ECG PLOT
# =========================================================
def plot_ecg(signal, peaks=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=signal, mode='lines', name='ECG'))

    if peaks is not None:
        fig.add_trace(go.Scatter(
            x=peaks,
            y=signal[peaks],
            mode='markers',
            marker=dict(color='red', size=6),
            name='R-peaks'
        ))

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 🔥 CORE FIX: BEAT-BASED PREDICTION
# =========================================================
def predict_signal(signal):
    peaks, _ = find_peaks(signal, distance=20)

    predictions = []

    for p in peaks:
        start = max(0, p - 90)
        end = min(len(signal), p + 97)

        beat = signal[start:end]

        if len(beat) < 187:
            beat = np.pad(beat, (0, 187 - len(beat)))
        else:
            beat = beat[:187]

        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-6)

        X = beat.reshape(1, -1)

        pred = model.predict(X)[0]
        predictions.append(pred)

    if len(predictions) == 0:
        return 0, 0.0, []

    final_class = max(set(predictions), key=predictions.count)
    confidence = predictions.count(final_class) / len(predictions)

    return final_class, confidence, peaks

# =========================================================
# PDF
# =========================================================
def generate_pdf(df):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    path = tmp.name
    tmp.close()

    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("ECG Clinical Report", styles["Title"]))
    content.append(Spacer(1, 12))

    table = Table([df.columns.tolist()] + df.values.tolist())
    table.setStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.black)
    ])

    content.append(table)
    doc.build(content)

    return path

# =========================================================
# SINGLE PATIENT
# =========================================================
if mode == "Single Patient":

    st.subheader("Patient Info")

    name = st.text_input("Name")
    age = st.number_input("Age", 0, 120)
    pid = st.text_input("Patient ID")

    file = st.file_uploader("Upload ECG CSV", type=["csv"])

    if file:
        try:
            df = pd.read_csv(file)
            signal = df.iloc[:, 0].values
        except:
            st.error("Invalid CSV")
            st.stop()

        signal = prepare(signal)

        pred, confidence, peaks = predict_signal(signal)
        status = clinical_status_class(pred)

        st.metric("Status", status)
        st.metric("Confidence", f"{confidence:.2f}")

        plot_ecg(signal, peaks)

        save_record(name, age, pid, confidence, status)

# =========================================================
# BATCH MODE
# =========================================================
else:

    st.subheader("Batch Analysis")

    files = st.file_uploader("Upload multiple ECG CSV", type=["csv"], accept_multiple_files=True)

    results = []
    signals = {}

    if files:
        for f in files:
            try:
                df = pd.read_csv(f)
                signal = prepare(df.iloc[:, 0].values)

                pred, confidence, peaks = predict_signal(signal)
                status = clinical_status_class(pred)

                results.append({
                    "File": f.name,
                    "Status": status,
                    "Confidence": round(confidence, 3),
                    "Beats": len(peaks)
                })

                signals[f.name] = (signal, peaks)

            except:
                continue

        df_res = pd.DataFrame(results)
        st.dataframe(df_res)

        sel = st.selectbox("Select ECG", list(signals.keys()))

        if sel:
            sig, pk = signals[sel]
            plot_ecg(sig, pk)

        if st.button("Generate PDF"):
            path = generate_pdf(df_res)
            with open(path, "rb") as f:
                st.download_button("Download Report", f, "report.pdf")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Developed by Aditya Rawal • AI-assisted system")
