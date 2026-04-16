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
    try:
        if not os.path.exists(MODEL_PATH):
            st.error("❌ Model not found.")
            st.stop()

        if os.path.getsize(MODEL_PATH) == 0:
            st.error("❌ Model file is corrupted.")
            st.stop()

        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    except Exception as e:
        st.error(f"❌ Model load error: {str(e)}")
        st.stop()

model = load_model()

# =========================================================
# UI
# =========================================================
st.markdown("<h1 style='color:red;'>🫀 ECG Clinical System</h1>", unsafe_allow_html=True)

mode = st.sidebar.selectbox("Mode", ["Single Patient", "Batch Research"])

EXPECTED_LEN = 187

# =========================================================
# PREPROCESS
# =========================================================
def prepare(signal):
    signal = np.array(signal).astype(float)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

    if len(signal) < EXPECTED_LEN:
        signal = np.pad(signal, (0, EXPECTED_LEN - len(signal)))
    else:
        signal = signal[:EXPECTED_LEN]

    return signal

# =========================================================
# CLASSIFICATION
# =========================================================
def clinical_status_class(pred):
    mapping = {
        0: "Normal Sinus Rhythm",
        1: "Supraventricular Arrhythmia",
        2: "Ventricular Arrhythmia",
        3: "Fusion Beat",
        4: "Artifact / Unknown"
    }
    return mapping.get(int(pred), "Unknown")

def predict_signal(signal):
    X = signal.reshape(1, -1)
    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        confidence = np.max(probs)
    else:
        confidence = float(pred)

    return pred, confidence

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
            name='R-Peaks'
        ))

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PDF
# =========================================================
def generate_pdf(df, name="ECG_Report"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    path = tmp.name
    tmp.close()

    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("🫀 ECG Clinical Report", styles["Title"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Patient / Batch: {name}", styles["Normal"]))
    content.append(Spacer(1, 12))

    table = Table([df.columns.tolist()] + df.values.tolist())
    table.setStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
    ])

    content.append(table)
    doc.build(content)

    return path

# =========================================================
# SINGLE PATIENT MODE
# =========================================================
if mode == "Single Patient":

    st.subheader("🧾 Patient Details")

    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Name")
    age = c2.number_input("Age", 0, 120, 30)
    pid = c3.text_input("Patient ID")

    file = st.file_uploader("Upload ECG CSV", type=["csv"])

    if file:
        try:
            df = pd.read_csv(file)
            signal = df.iloc[:, 0].values
        except:
            st.error("Invalid CSV")
            st.stop()

        processed = prepare(signal)
        pred, confidence = predict_signal(processed)
        status = clinical_status_class(pred)

        peaks, _ = find_peaks(processed, distance=20)

        st.subheader("🧠 Results")

        col1, col2 = st.columns(2)
        col1.metric("Confidence", f"{confidence:.2f}")
        col2.metric("Diagnosis", status)

        save_record(name, age, pid, confidence, status)

        st.subheader("📈 ECG Waveform")
        plot_ecg(processed, peaks)

# =========================================================
# BATCH MODE
# =========================================================
else:

    st.subheader("📊 Batch ECG Analysis")

    files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

    results = []
    signal_map = {}

    if files:
        for file in files:
            try:
                df = pd.read_csv(file)
                signal = df.iloc[:, 0].values
            except:
                continue

            processed = prepare(signal)
            pred, confidence = predict_signal(processed)
            status = clinical_status_class(pred)

            peaks, _ = find_peaks(processed, distance=20)

            results.append({
                "File": file.name,
                "Confidence": round(confidence, 3),
                "Status": status,
                "Heartbeats": len(peaks)
            })

            signal_map[file.name] = processed

        df_res = pd.DataFrame(results)

        st.dataframe(df_res)

        selected = st.selectbox("Select ECG", list(signal_map.keys()))

        if selected:
            plot_ecg(signal_map[selected])

        if not df_res.empty and st.button("📄 Generate PDF"):
            pdf_path = generate_pdf(df_res, "Batch Report")

            with open(pdf_path, "rb") as f:
                st.download_button("⬇ Download", f, "report.pdf")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Developed by Aditya Rawal • AI-assisted")
