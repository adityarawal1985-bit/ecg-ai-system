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
st.set_page_config(
    page_title="ECG AI Clinical System",
    page_icon="🫀",
    layout="wide"
)

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
        st.error("❌ ECG Model not found. Check models folder.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# =========================================================
# UI
# =========================================================
st.markdown("""
<style>
.title{font-size:34px;font-weight:900;color:#ff2d2d;}
.sub{color:#9ca3af;font-size:14px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🫀 CLINICAL ECG SYSTEM</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>AI-assisted Clinical Decision Support System</div>", unsafe_allow_html=True)

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
# CLINICAL LABELS (FIXED)
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

    fig.add_trace(go.Scatter(y=signal, mode='lines', name='ECG Signal'))

    if peaks is not None:
        fig.add_trace(go.Scatter(
            x=peaks,
            y=signal[peaks],
            mode='markers',
            marker=dict(color='red', size=8),
            name='R-Peaks'
        ))

    fig.update_layout(title="ECG Waveform", height=400)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PDF GENERATOR
# =========================================================
def generate_pdf(df, patient_name="ECG_Report"):

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    file_path = tmp_file.name
    tmp_file.close()

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("🫀 Clinical ECG Report", styles["Title"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Patient / Batch: {patient_name}", styles["Normal"]))
    content.append(Spacer(1, 12))

    table_data = [df.columns.to_list()] + df.values.tolist()
    table = Table(table_data)

    table.setStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
    ])

    content.append(table)
    doc.build(content)

    return file_path

# =========================================================
# SINGLE PATIENT MODE
# =========================================================
pred, confidence = predict_signal(processed)
status = clinical_status_class(pred)

# =========================================================
# BATCH MODE
# =========================================================
else;

    st.subheader("📊 Batch ECG Analysis")

    files = st.file_uploader("Upload multiple ECG CSV", type=["csv"], accept_multiple_files=True)

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
            X = processed.reshape(1, -1)

            pred = model.predict(X)[0]

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                confidence = np.max(probs)
            else:
                confidence = float(pred)

            status = clinical_status_class(pred)
            peaks, _ = find_peaks(processed, distance=20)

            results.append({
                "File": file.name,
                "Confidence": round(confidence, 3),
                "Status": status,
                "HeartBeats": len(peaks)
            })

            signal_map[file.name] = processed

        df_res = pd.DataFrame(results)

        st.subheader("📊 Report Table")
        st.dataframe(df_res)

        selected = st.selectbox("Select ECG", list(signal_map.keys()))

        if selected:
            st.subheader("📈 ECG Waveform")
            plot_ecg(signal_map[selected])

        st.subheader("📄 Export Clinical Report")

        if not df_res.empty:

            if st.button("📥 Generate PDF Report"):

                pdf_path = generate_pdf(df_res, "AIIMS_ECG_Batch")

                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="⬇ Download PDF Report",
                        data=f,
                        file_name="AIIMS_ECG_Clinical_Report.pdf",
                        mime="application/pdf"
                    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("ECG Clinical System • Developed by Aditya Rawal (AI-assisted)")
