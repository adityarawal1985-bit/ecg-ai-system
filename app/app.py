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
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="ECG Decision Support System",
    page_icon="assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None,
    layout="wide"
)

# =========================================================
# CLINICAL DISCLAIMER — must appear before everything
# =========================================================
st.warning(
    "**IMPORTANT NOTICE:** This software is a **research and decision support tool only**. "
    "It is NOT a certified medical device and has NOT been validated for clinical diagnosis. "
    "All outputs must be reviewed and confirmed by a qualified medical professional. "
    "Do NOT make clinical decisions based solely on this system's output."
)

# =========================================================
# DATABASE
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ecg_clinic.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check existing columns
    c.execute("PRAGMA table_info(records)")
    existing_cols = {row[1] for row in c.fetchall()}

    if not existing_cols:
        # Table doesn't exist yet — create fresh
        c.execute("""
            CREATE TABLE records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT,
                age INTEGER,
                patient_id TEXT,
                ai_class INTEGER,
                predicted_label TEXT,
                model_confidence REAL,
                heartbeat_count INTEGER,
                timestamp TEXT
            )
        """)
    else:
        # Table exists — add any missing columns from the new schema
        migrations = {
            "ai_class":         "ALTER TABLE records ADD COLUMN ai_class INTEGER",
            "predicted_label":  "ALTER TABLE records ADD COLUMN predicted_label TEXT",
            "model_confidence": "ALTER TABLE records ADD COLUMN model_confidence REAL",
            "heartbeat_count":  "ALTER TABLE records ADD COLUMN heartbeat_count INTEGER",
        }
        for col, sql in migrations.items():
            if col not in existing_cols:
                c.execute(sql)

    conn.commit()
    conn.close()

init_db()

def save_record(name, age, pid, ai_class, label, confidence, hb_count):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO records
            (patient_name, age, patient_id, ai_class, predicted_label,
             model_confidence, heartbeat_count, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, age, pid, int(ai_class), label,
          float(confidence), int(hb_count), str(datetime.now())))
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
            st.error("Model file not found at: " + MODEL_PATH)
            st.stop()
        if os.path.getsize(MODEL_PATH) == 0:
            st.error("Model file is empty / corrupted.")
            st.stop()
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model load error: {str(e)}")
        st.stop()

model = load_model()

# =========================================================
# MODEL PERFORMANCE CARD (MIT-BIH benchmark — update with yours)
# =========================================================
MODEL_METRICS = {
    "Dataset": "MIT-BIH Arrhythmia Database",
    "Overall Accuracy": "~95% (reported on held-out test set)",
    "Sensitivity (Normal)": "~97%",
    "Specificity (Arrhythmia)": "~93%",
    "Note": "Metrics on MIT-BIH test split. Not validated on clinical hospital data."
}

# =========================================================
# LABEL MAP
# =========================================================
LABEL_MAP = {
    0: "Normal Sinus Rhythm",
    1: "Supraventricular Arrhythmia",
    2: "Ventricular Arrhythmia",
    3: "Fusion Beat",
    4: "Artifact / Unclassifiable"
}

URGENCY_MAP = {
    0: ("Routine", "green"),
    1: ("Review Recommended", "orange"),
    2: ("Urgent Review", "red"),
    3: ("Review Recommended", "orange"),
    4: ("Signal Quality Issue – Repeat ECG", "gray")
}

# =========================================================
# PREPROCESS
# =========================================================
EXPECTED_LEN = 187

def prepare(signal):
    signal = np.array(signal).astype(float)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
    if len(signal) < EXPECTED_LEN:
        signal = np.pad(signal, (0, EXPECTED_LEN - len(signal)))
    else:
        signal = signal[:EXPECTED_LEN]
    return signal

# =========================================================
# PREDICT
# =========================================================
def predict_signal(signal):
    X = signal.reshape(1, -1)
    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        confidence = float(np.max(probs))
        class_probs = probs
    else:
        confidence = None
        class_probs = None

    return int(pred), confidence, class_probs

# =========================================================
# ECG PLOT
# =========================================================
def plot_ecg(signal, peaks=None, title="ECG Waveform"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=signal, mode='lines', name='ECG Signal',
        line=dict(color='#1a73e8', width=1.5)
    ))
    if peaks is not None and len(peaks) > 0:
        fig.add_trace(go.Scatter(
            x=peaks, y=signal[peaks],
            mode='markers',
            marker=dict(color='red', size=7, symbol='triangle-up'),
            name='Detected R-Peaks'
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Sample Index",
        yaxis_title="Normalized Amplitude",
        height=380,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PROBABILITY BAR CHART
# =========================================================
def plot_class_probs(class_probs):
    labels = list(LABEL_MAP.values())
    bar_colors = ['green', 'orange', 'red', 'orange', 'gray']
    fig = go.Figure(go.Bar(
        x=labels,
        y=class_probs,
        marker_color=bar_colors,
        text=[f"{p:.1%}" for p in class_probs],
        textposition='outside'
    ))
    fig.update_layout(
        title="Model Class Probability Distribution",
        yaxis_title="Probability",
        yaxis_range=[0, 1.1],
        height=320,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PDF REPORT
# =========================================================
def generate_pdf(df, patient_info=None, filename="ECG_Report"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    path = tmp.name
    tmp.close()

    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=40, rightMargin=40,
                            topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Title'],
        fontSize=16, spaceAfter=6, alignment=TA_CENTER
    )
    header_style = ParagraphStyle(
        'SubHeader', parent=styles['Heading2'],
        fontSize=11, spaceAfter=4
    )
    normal_style = ParagraphStyle(
        'CustomNormal', parent=styles['Normal'],
        fontSize=9, spaceAfter=2
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer', parent=styles['Normal'],
        fontSize=8, textColor=colors.red,
        spaceAfter=4, borderPad=4
    )

    content = []

    # Header
    content.append(Paragraph("ECG Clinical Decision Support System", title_style))
    content.append(Paragraph("AI-Assisted Arrhythmia Classification Report", styles["Heading3"]))
    content.append(Spacer(1, 8))

    # Disclaimer box
    content.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI-based decision support tool and is NOT "
        "a certified diagnostic report. It must be reviewed and validated by a qualified "
        "cardiologist or physician before any clinical action is taken.",
        disclaimer_style
    ))
    content.append(Spacer(1, 8))

    # Report metadata
    content.append(Paragraph(f"Report Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", normal_style))
    content.append(Paragraph(f"Model Trained On: MIT-BIH Arrhythmia Database", normal_style))
    content.append(Spacer(1, 10))

    # Patient info if available
    if patient_info:
        content.append(Paragraph("Patient Information", header_style))
        pi_data = [["Field", "Value"]] + [[k, str(v)] for k, v in patient_info.items()]
        pi_table = Table(pi_data, colWidths=[150, 300])
        pi_table.setStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d0d8e4")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
        content.append(pi_table)
        content.append(Spacer(1, 10))

    # Results table
    content.append(Paragraph("Analysis Results", header_style))
    col_names = df.columns.tolist()
    table_data = [col_names] + df.astype(str).values.tolist()
    result_table = Table(table_data, repeatRows=1)
    result_table.setStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a73e8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
    ])
    content.append(result_table)
    content.append(Spacer(1, 12))

    # Model performance note
    content.append(Paragraph("Model Performance Reference (MIT-BIH Test Set)", header_style))
    mp_data = [["Metric", "Value"]] + [[k, str(v)] for k, v in MODEL_METRICS.items()]
    mp_table = Table(mp_data, colWidths=[200, 260])
    mp_table.setStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d0d8e4")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ])
    content.append(mp_table)
    content.append(Spacer(1, 10))

    # Footer disclaimer
    content.append(Paragraph(
        "This report was generated using the MIT-BIH Arrhythmia Database-trained model. "
        "Performance on data from other populations or recording equipment may differ. "
        "This tool is intended to assist, not replace, clinical judgement.",
        disclaimer_style
    ))

    doc.build(content)
    return path

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("ECG Decision Support System")
st.sidebar.caption("Version 1.0 — Research Use Only")
mode = st.sidebar.selectbox("Analysis Mode", ["Single Patient", "Batch Research"])

with st.sidebar.expander("Model Information"):
    for k, v in MODEL_METRICS.items():
        st.markdown(f"**{k}:** {v}")

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Aditya Rawal\nAI-Assisted Clinical Research Tool")

# =========================================================
# SINGLE PATIENT MODE
# =========================================================
if mode == "Single Patient":

    st.subheader("Patient Details")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Patient Name")
    age = c2.number_input("Age (years)", 0, 120, 30)
    pid = c3.text_input("Patient ID / MR Number")

    st.markdown("---")
    st.subheader("ECG Signal Upload")
    st.caption("Upload a single-column CSV containing 187 amplitude values (MIT-BIH format).")
    file = st.file_uploader("Upload ECG CSV", type=["csv"])

    if file:
        try:
            df_raw = pd.read_csv(file)
            signal = df_raw.iloc[:, 0].values
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        processed = prepare(signal)
        pred_class, confidence, class_probs = predict_signal(processed)
        label = LABEL_MAP.get(pred_class, "Unknown")
        urgency_text, urgency_color = URGENCY_MAP.get(pred_class, ("Unknown", "gray"))
        peaks, _ = find_peaks(processed, distance=20)
        hb_count = len(peaks)

        st.markdown("---")
        st.subheader("AI Analysis Results")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Class", label)
        col2.metric("Model Confidence", f"{confidence:.1%}" if confidence else "N/A")
        col3.metric("R-Peaks Detected", hb_count)
        col4.metric("Recommended Action", urgency_text)

        if pred_class == 2:
            st.error("Ventricular Arrhythmia detected by model. Requires urgent physician review.")
        elif pred_class == 4:
            st.warning("Signal quality issue detected. Consider repeating ECG recording.")
        elif pred_class in [1, 3]:
            st.warning(f"{label} detected. Physician review recommended.")
        else:
            st.success("AI classification: Normal Sinus Rhythm. Verify clinically.")

        if class_probs is not None:
            st.markdown("---")
            st.subheader("Class Probability Distribution")
            plot_class_probs(class_probs)

        st.markdown("---")
        st.subheader("ECG Waveform with R-Peak Detection")
        plot_ecg(processed, peaks, title=f"ECG — {name or 'Patient'} | {label}")

        # Save to DB
        if name or pid:
            save_record(name, age, pid, pred_class, label, confidence or 0.0, hb_count)
            st.caption("Record saved to database.")

        # PDF download
        st.markdown("---")
        if st.button("Generate Clinical Report (PDF)"):
            patient_info = {
                "Name": name or "Not provided",
                "Age": age,
                "Patient ID": pid or "Not provided",
                "Recording File": file.name
            }
            result_df = pd.DataFrame([{
                "Predicted Label": label,
                "Model Confidence": f"{confidence:.1%}" if confidence else "N/A",
                "R-Peaks": hb_count,
                "Recommended Action": urgency_text
            }])
            pdf_path = generate_pdf(result_df, patient_info=patient_info, filename=f"{name or 'patient'}_ecg")
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF Report", f, file_name="ECG_Report.pdf", mime="application/pdf")

# =========================================================
# BATCH RESEARCH MODE
# =========================================================
else:
    st.subheader("Batch ECG Analysis — Research Mode")
    st.caption("Upload multiple ECG CSV files. Results are for research and audit purposes only.")

    files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

    results = []
    signal_map = {}

    if files:
        progress = st.progress(0)
        for i, file in enumerate(files):
            try:
                df_raw = pd.read_csv(file)
                signal = df_raw.iloc[:, 0].values
            except Exception:
                results.append({
                    "File": file.name,
                    "Status": "Read Error",
                    "Predicted Label": "N/A",
                    "Model Confidence": "N/A",
                    "R-Peaks": "N/A",
                    "Action": "N/A"
                })
                continue

            processed = prepare(signal)
            pred_class, confidence, _ = predict_signal(processed)
            label = LABEL_MAP.get(pred_class, "Unknown")
            urgency_text, _ = URGENCY_MAP.get(pred_class, ("Unknown", "gray"))
            peaks, _ = find_peaks(processed, distance=20)

            results.append({
                "File": file.name,
                "Predicted Label": label,
                "Model Confidence": f"{confidence:.1%}" if confidence else "N/A",
                "R-Peaks": len(peaks),
                "Action": urgency_text
            })

            signal_map[file.name] = processed
            progress.progress((i + 1) / len(files))

        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True)

        # Summary stats
        if not df_res.empty:
            st.markdown("---")
            st.subheader("Batch Summary")
            label_counts = df_res["Predicted Label"].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Class Distribution:**")
                st.dataframe(label_counts.reset_index().rename(
                    columns={"index": "Label", "Predicted Label": "Count"}
                ))
            with col2:
                fig = go.Figure(go.Pie(
                    labels=label_counts.index,
                    values=label_counts.values,
                    hole=0.4
                ))
                fig.update_layout(height=280, margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

        # ECG viewer
        if signal_map:
            st.markdown("---")
            selected = st.selectbox("View individual ECG waveform", list(signal_map.keys()))
            if selected:
                s = signal_map[selected]
                peaks, _ = find_peaks(s, distance=20)
                plot_ecg(s, peaks, title=f"ECG — {selected}")

        # PDF
        if not df_res.empty and st.button("Generate Batch PDF Report"):
            pdf_path = generate_pdf(df_res, filename="Batch_ECG_Report")
            with open(pdf_path, "rb") as f:
                st.download_button("Download Batch Report", f,
                                   file_name="Batch_ECG_Report.pdf", mime="application/pdf")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "ECG Decision Support System v1.0 | Developed by Aditya Rawal | "
    "Research Use Only — Not for Clinical Diagnosis | "
    f"Model trained on MIT-BIH Arrhythmia Database"
)


# Add to sidebar, right after mode selection
st.sidebar.markdown("---")
st.sidebar.caption("**Research Framework v1.0** — Model requires local fine-tuning for clinical use.")
