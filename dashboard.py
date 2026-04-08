import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import pytz
import json
from pathlib import Path
import hashlib
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.mime.base import MIMEBase
import os
import time
import numpy as np
import requests
from io import BytesIO

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False


# CONFIG & CONSTANTS

ACCOUNT_SID    = ""
AUTH_TOKEN     = ""
FROM_NUMBER    = "+"
TO_NUMBER      = "+"
EMAIL_SENDER   = "@gmail.com"
EMAIL_PASSWORD = ""
ALERT_EMAIL_TO = "71762305032@cit.edu.in"
SMS_MAX_CHARS  = 1550

IST = pytz.timezone("Asia/Kolkata")

def get_ist_now():
    return datetime.now(IST)

DEVICE_ICONS = {
    "ventilator": "🫁", "monitor": "", "infusion": "💉", "dialysis": "🧬",
    "defibrillator": "⚡", "pump": "🔄", "scanner": "🔍", "ultrasound": "📡",
    "ecg": "❤️", "bp_monitor": "🩸", "oxygen": "💨", "incubator": "🍼",
    "cautery": "🔥", "lights": "💡", "bed": "🛏️", "cart": "🛒", "default": "",
}

URGENCY_COLOR = {
    "CRITICAL": "#ff3b5c", "HIGH": "#ff8c42",
    "ELEVATED": "#f5c842", "MONITOR": "#22c984",
}
URGENCY_LEVELS        = ["CRITICAL", "HIGH", "ELEVATED", "MONITOR"]
DEVICE_STATUS_OPTIONS = ["Monitored", "Isolated", "Patched", "Replaced", "Maintenance"]
DEVICE_STATUS_COLORS  = {
    "Monitored": "#22c984", "Isolated": "#ff3b5c", "Patched": "#f5c842",
    "Replaced": "#94a3b8", "Maintenance": "#00d4ff",
}

DEVICE_RULES = {
    "ventilator":      {"can_isolate": False, "action_critical": "Schedule Emergency Maintenance — DO NOT isolate (patient life-support)", "action_high": "Schedule patch in next maintenance window", "note": "Life-critical. Isolation prohibited."},
    "defibrillator":   {"can_isolate": False, "action_critical": "Schedule Emergency Maintenance — DO NOT isolate", "action_high": "Schedule patch — notify biomedical team", "note": "Life-critical. Isolation prohibited."},
    "infusion pump":   {"can_isolate": False, "action_critical": "Schedule patch + Notify clinician immediately", "action_high": "Monitor closely — plan patch within 24h", "note": "Patient-attached. Consult clinician before action."},
    "patient monitor": {"can_isolate": True,  "action_critical": "Isolate + switch to backup monitor", "action_high": "Schedule patch — keep under observation", "note": "Backup monitor must be available before isolation."},
    "mri":             {"can_isolate": False, "action_critical": "Schedule Emergency Maintenance — coordinate with radiology", "action_high": "Schedule patch — plan downtime window", "note": "High-value device. Requires specialist team."},
    "ct":              {"can_isolate": False, "action_critical": "Schedule Emergency Maintenance — coordinate with radiology", "action_high": "Schedule patch — plan downtime window", "note": "High-value device. Requires specialist team."},
}
FALLBACK_RULE = {
    "can_isolate": True,
    "action_critical": "Isolate from network immediately",
    "action_high": "Schedule patch within 24 hours",
    "note": "Standard isolation policy applies.",
}

_RECENT_LOG_CACHE: dict = {}


# DATASET SOURCES

DATASET_SOURCES = {
    "Hybrid Analysis (NVD + Sources)": {
        "paths": ["hybrid_analysis/results_v6.3.csv", "results_v6.3.csv"],
        "emoji": "🔗",
        "description": "Real-world data from NVD, CVE databases & hospital sources",
    },
    "MIMIC-III Clinical Data": {
        "paths": ["hybrid_analysis/results_v6.3_mimic.csv"],
        "emoji": "",
        "description": "MIMIC-III clinical database derived device risk scores",
    },
    "FDA MAUDE Adverse Events": {
        "paths": ["hybrid_analysis/results_v6.3_fda.csv", "hybrid_analysis/results_fda.csv"],
        "emoji": "🔴",
        "description": "Real FDA adverse event data with CVE/behavior scoring",
    },
}

WARD_CRITICALITY = {
    "ICU": 0.95, "EMERGENCY": 0.85, "OPERATING_THEATRE": 0.90,
    "NICU": 0.95, "CARDIOLOGY": 0.80, "RADIOLOGY": 0.70,
    "ONCOLOGY": 0.75, "GENERAL_WARD": 0.60, "LABORATORY": 0.60,
    "OBSTETRICS": 0.70, "NEPHROLOGY": 0.75, "NEUROLOGY": 0.70,
}
PATIENT_IMPACT = {
    "LIFE_SUPPORT": 1.0, "CRITICAL_CARE": 0.9, "SURGICAL": 0.8,
    "DIAGNOSTIC": 0.6, "MONITORING": 0.4, "ADMINISTRATIVE": 0.1,
}


#  CHANGE: ADAPTIVE THRESHOLD (mirrors model logic for all datasets)

# Base threshold — same as EVAL_THRESHOLD in the model
EVAL_THRESHOLD = 0.20

ADAPTIVE_THRESHOLD_CONFIG = {
    # ward → modifier  (lower = more sensitive = lower threshold)
    "ward": {
        "ICU":               0.70,   # threshold × 0.70 ≈ 0.14  (very sensitive)
        "NICU":              0.70,
        "OPERATING_THEATRE": 0.75,
        "EMERGENCY":         0.75,
        "CATH_LAB":          0.80,
        "CARDIOLOGY":        0.85,
        "PEDIATRICS":        0.85,
        "NEPHROLOGY":        0.85,
        "RADIOLOGY":         0.90,
        "ONCOLOGY":          0.90,
        "NEUROLOGY":         0.90,
        "OBSTETRICS":        0.90,
        "GENERAL_WARD":      1.00,   # base threshold
        "LABORATORY":        1.00,
        "DATA_CENTER":       1.00,
        "PHARMACY":          1.10,
        "NURSE_STATION":     1.10,
        "DOCTOR_OFFICE":     1.20,
        "ADMIN":             1.40,   # threshold × 1.40 ≈ 0.28  (less sensitive)
        "STORAGE":           1.50,
    },
    # patient_impact → modifier
    "impact": {
        "LIFE_SUPPORT":    0.70,
        "CRITICAL_CARE":   0.75,
        "SURGICAL":        0.80,
        "DIAGNOSTIC":      0.95,
        "THERAPEUTIC":     1.00,
        "MONITORING":      1.05,
        "SUPPORT":         1.20,
        "ADMINISTRATIVE":  1.50,
    }
}


def get_adaptive_threshold_dashboard(row) -> float:
    """
    Compute per-device adaptive threshold using ward + patient_impact modifiers.
    Reuses the EXACT same logic as get_adaptive_threshold() in the model, so
    evaluation is consistent across all three datasets.

    Lower threshold  → more sensitive  (ICU/life-support devices)
    Higher threshold → less sensitive  (admin/low-impact devices)
    """
    ward   = str(row.get("ward",           "GENERAL_WARD")).upper()
    impact = str(row.get("patient_impact", "MONITORING")).upper()

    ward_mod   = ADAPTIVE_THRESHOLD_CONFIG["ward"].get(ward,   1.0)
    impact_mod = ADAPTIVE_THRESHOLD_CONFIG["impact"].get(impact, 1.0)

    # Use the more conservative (smaller) modifier — same rule as the model
    modifier = min(ward_mod, impact_mod)
    return float(np.clip(EVAL_THRESHOLD * modifier, 0.05, 0.60))


def create_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive pred_label for every device using adaptive thresholds.

    Architecture (per design requirement):
      IF dataset already has 'adaptive_threshold' column
          → use it directly  (Hybrid dataset: model already stored per-device threshold)
      ELSE
          → compute dynamically using ward/impact modifiers  (FDA, MIMIC datasets)

    This is "inference-time adaptive decisioning" — NOT training-time learning.
    Result: evaluation is consistent across all three datasets with no mismatch.
    """
    df = df.copy()

    if "adaptive_threshold" in df.columns:
        # Case 1: Hybrid — model already provided per-device thresholds
        df["adaptive_threshold"] = pd.to_numeric(
            df["adaptive_threshold"], errors="coerce"
        ).fillna(EVAL_THRESHOLD)
        df["pred_label"] = (df["score"] >= df["adaptive_threshold"]).astype(int)
    else:
        # Case 2: FDA / MIMIC — compute threshold dynamically in dashboard
        df["adaptive_threshold"] = df.apply(get_adaptive_threshold_dashboard, axis=1)
        df["pred_label"] = (df["score"] >= df["adaptive_threshold"]).astype(int)

    return df



# PDF REPORT GENERATION  (v7-style: helvetica + safe_text + MIMEBase attach)

def _safe_text(text):
    """Strip / replace characters that fpdf latin-1 cannot encode."""
    if text is None:
        return ""
    replacements = {
        "•": "-", "\u2014": "-", "\u2013": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2026": "...",
        "🔴": "[CRITICAL]", "🟠": "[HIGH]", "🟡": "[ELEVATED]", "🟢": "[MONITOR]",
        "📍": "[WARD]",    "": "[STATS]",  "": "[OK]",        "": "[WARN]",
        "": "[ALERT]",   "": "[HOSPITAL]",
    }
    text = str(text)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_ward_wise_critical_report(df):
    """
    Generate a styled PDF of CRITICAL devices grouped by ward.
    Returns a BytesIO buffer, or None if fpdf2 is unavailable / no critical devices.
    """
    if not FPDF_AVAILABLE:
        return None

    critical_df = df[df["urgency"] == "CRITICAL"].copy()
    if critical_df.empty:
        return None

    st = _safe_text   # local alias for brevity

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(13, 17, 23)
            self.rect(0, 0, 210, 45, "F")
            self.set_font("helvetica", "B", 18)
            self.set_text_color(255, 59, 92)
            self.cell(0, 12, st("CRITICAL ALERT - CLINICAL GUARDIAN"), 0, 1, "C")
            self.set_font("helvetica", "B", 12)
            self.cell(0, 6,  st("WARD-WISE CRITICAL DEVICE REPORT"), 0, 1, "C")
            self.set_font("helvetica", "", 10)
            self.set_text_color(100, 116, 139)
            self.cell(0, 6,  st("IMMEDIATE ACTION REQUIRED - GROUPED BY WARD"), 0, 1, "C")
            self.set_draw_color(255, 59, 92)
            self.line(10, 48, 200, 48)
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font("helvetica", "I", 8)
            self.set_text_color(100, 116, 139)
            self.cell(0, 10, st(
                f"Page {self.page_no()} | Generated: "
                f"{get_ist_now().strftime('%Y-%m-%d %H:%M:%S IST')}"
            ), 0, 0, "C")

    pdf = PDF()
    pdf.add_page()

    # ── Executive summary ────────────────────────────────────────────────────
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(226, 232, 240)
    pdf.cell(0, 10, st("EXECUTIVE SUMMARY"), 0, 1)
    pdf.set_draw_color(255, 59, 92)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(255, 59, 92)
    pdf.cell(0, 8, st(f"TOTAL CRITICAL DEVICES: {len(critical_df)}"), 0, 1)
    pdf.cell(0, 8, st(f"AFFECTED WARDS: {critical_df['ward'].nunique()}"), 0, 1)
    pdf.ln(5)

    # Ward summary table
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(0, 212, 255)
    pdf.cell(0, 10, st("WARD SUMMARY"), 0, 1)
    pdf.ln(2)
    for ward, count in (
        critical_df.groupby("ward").size().sort_values(ascending=False).items()
    ):
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(60, 6, st(f"Ward: {ward}"), 0, 0)
        pdf.set_text_color(255, 59, 92)
        pdf.cell(0, 6, st(f"{count} CRITICAL devices"), 0, 1)
    pdf.ln(8)

    # Risk statistics
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(0, 212, 255)
    pdf.cell(0, 8, st("Risk Score Statistics:"), 0, 1)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 6, st(f"- Average Risk Score: {critical_df['score'].mean():.3f}"), 0, 1)
    pdf.cell(0, 6, st(f"- Maximum Risk Score: {critical_df['score'].max():.3f}"), 0, 1)
    pdf.cell(0, 6, st(f"- Minimum Risk Score: {critical_df['score'].min():.3f}"), 0, 1)
    pdf.ln(8)

    # ── Ward-wise detail pages ────────────────────────────────────────────────
    for ward, group in critical_df.groupby("ward"):
        pdf.add_page()

        pdf.set_font("helvetica", "B", 16)
        pdf.set_text_color(255, 59, 92)
        pdf.cell(0, 10, st(f"WARD: {ward}"), 0, 1)
        pdf.set_draw_color(255, 59, 92)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(8)

        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(226, 232, 240)
        pdf.cell(0, 8, st(f"CRITICAL DEVICES IN THIS WARD: {len(group)}"), 0, 1)
        pdf.ln(4)

        for idx, (_, row) in enumerate(group.iterrows(), 1):
            rule   = get_device_rule(row.get("type", ""))
            action = rule["action_critical"]

            pdf.set_font("helvetica", "B", 10)
            pdf.set_text_color(255, 59, 92)
            pdf.cell(0, 8, st(
                f"{idx}. {row.get('name', 'N/A')} [{row.get('id', 'N/A')}]"
            ), 0, 1)

            pdf.set_font("helvetica", "", 9)

            pdf.set_text_color(148, 163, 184)
            pdf.cell(40, 6, st("Type:"), 0, 0)
            pdf.set_text_color(226, 232, 240)
            pdf.cell(0, 6, st(str(row.get("type", "N/A"))), 0, 1)

            pdf.set_text_color(148, 163, 184)
            pdf.cell(40, 6, st("Risk Score:"), 0, 0)
            pdf.set_text_color(255, 59, 92)
            pdf.cell(0, 6, st(f"{row.get('score', 0):.3f}"), 0, 1)

            # ──  CHANGE: show adaptive threshold in PDF report ──────────────
            adaptive_thr = float(row.get("adaptive_threshold", EVAL_THRESHOLD))
            pdf.set_text_color(148, 163, 184)
            pdf.cell(40, 6, st("Adaptive Threshold:"), 0, 0)
            pdf.set_text_color(0, 212, 255)
            pdf.cell(0, 6, st(f"{adaptive_thr:.3f}"), 0, 1)
            # ─────────────────────────────────────────────────────────────────

            pdf.set_text_color(148, 163, 184)
            pdf.cell(40, 6, st("Components:"), 0, 0)
            pdf.set_text_color(255, 59, 92)
            pdf.cell(35, 6, st(f"CVE:{row.get('static', 0):.3f}"), 0, 0)
            pdf.set_text_color(255, 140, 66)
            pdf.cell(35, 6, st(f"Beh:{row.get('beh', 0):.3f}"), 0, 0)
            pdf.set_text_color(0, 212, 255)
            pdf.cell(35, 6, st(f"GNN:{row.get('gnn', 0):.3f}"), 0, 0)
            pdf.set_text_color(245, 200, 66)
            pdf.cell(0,  6, st(f"XGB:{row.get('xgb', 0):.3f}"), 0, 1)

            pdf.set_text_color(148, 163, 184)
            pdf.cell(40, 6, st("Action Required:"), 0, 0)
            pdf.set_text_color(255, 59, 92)
            pdf.multi_cell(0, 6, st(action), 0, 1)

            pdf.set_font("helvetica", "I", 8)
            pdf.set_text_color(100, 116, 139)
            pdf.multi_cell(0, 5, st(f"Note: {rule['note']}"), 0, 1)
            pdf.ln(4)

            if pdf.get_y() > 250:
                pdf.add_page()

    # ── Compliance page ───────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(226, 232, 240)
    pdf.cell(0, 10, st("COMPLIANCE AND RECOMMENDATIONS"), 0, 1)
    pdf.set_draw_color(255, 59, 92)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(255, 59, 92)
    pdf.cell(0, 8, st("Immediate Actions Required - BY WARD:"), 0, 1)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(148, 163, 184)
    for ward, group in critical_df.groupby("ward"):
        pdf.cell(0, 6, st(
            f"- {ward}: {len(group)} CRITICAL device(s) - Patch within 24 hours"
        ), 0, 1)

    pdf.ln(8)
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(255, 59, 92)
    pdf.cell(0, 8, st("Ward Supervisor Responsibilities:"), 0, 1)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(0, 6, st(
        "1. Verify all CRITICAL devices in your ward are identified\n"
        "2. Coordinate with biomedical engineering for patching\n"
        "3. Document remediation actions\n"
        "4. Report completion to IT Director within 24 hours"
    ), 0, 1)

    pdf.ln(8)
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(255, 59, 92)
    pdf.cell(0, 8, st("Compliance Requirements:"), 0, 1)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(0, 6, st(
        "- HIPAA Security Rule - Ward-level risk assessment required\n"
        "- FDA Guidelines - Each device must be tracked\n"
        "- Hospital Policy - Ward supervisors accountable\n"
        "- Incident Response - Document per ward"
    ), 0, 1)

    pdf.ln(8)
    pdf.set_font("helvetica", "I", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 5, st(
        "This report was automatically generated by Clinical Guardian v8.0.\n"
        "Clinical judgment should always be exercised when implementing security actions.\n"
        "Each ward is responsible for their critical devices."
    ), 0, 1)

    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1", errors="replace")
    return BytesIO(pdf_bytes)


def send_bulk_email_alert_PDF(to_email, df):
    """
    Send ward-wise critical PDF report as a properly encoded email attachment.
    Uses MIMEBase + encoders.encode_base64 (not MIMEText) to avoid garbled PDFs.
    """
    try:
        sender   = import_st_get("email_sender", EMAIL_SENDER)
        password = import_st_get("email_password", EMAIL_PASSWORD)

        if not sender or "your_email" in sender:
            return False, "Email credentials not configured."

        critical_df = df[df["urgency"] == "CRITICAL"]
        if critical_df.empty:
            return False, "No CRITICAL devices to report."

        pdf_buffer = generate_ward_wise_critical_report(df)
        if not pdf_buffer:
            return False, "Failed to generate PDF report."

        total_critical = len(critical_df)
        ward_count     = critical_df["ward"].nunique()

        subject = (
            f"CRITICAL ALERT: {total_critical} CRITICAL devices across "
            f"{ward_count} wards | {get_ist_now().strftime('%d %b %Y %H:%M IST')}"
        )

        body = (
            f"CLINICAL GUARDIAN - CRITICAL DEVICE ALERT (WARD-WISE)\n"
            f"{'='*60}\n\n"
            f"THIS IS A CRITICAL ALERT - IMMEDIATE ACTION REQUIRED\n\n"
            f"Generated       : {get_ist_now().strftime('%Y-%m-%d %H:%M:%S IST')}\n"
            f"Total CRITICAL  : {total_critical} devices\n"
            f"Affected Wards  : {ward_count}\n\n"
            f"WARD-WISE BREAKDOWN\n"
            f"{'-'*40}\n"
        )
        for ward, group in critical_df.groupby("ward"):
            body += f"\n[{ward}] - {len(group)} CRITICAL device(s)"
            for _, row in group.iterrows():
                body += f"\n    - {row.get('name','N/A')} (Score: {row.get('score',0):.3f})"

        body += (
            f"\n\n{'='*60}\n"
            f"Full PDF report attached with detailed ward-wise analysis.\n\n"
            f"REQUIRED ACTIONS:\n"
            f"1. Each ward must patch CRITICAL devices within 24 hours\n"
            f"2. Ward supervisors to sign off on remediation\n"
            f"3. Daily status reports until all CRITICAL devices are patched\n\n"
            f"Clinical Guardian v8.0 - Decision support only."
        )

        msg            = MIMEMultipart()
        msg["From"]    = sender
        msg["To"]      = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # Correct binary attachment (MIMEBase + base64, not MIMEText)
        attachment = MIMEBase("application", "octet-stream")
        attachment.set_payload(pdf_buffer.getvalue())
        encoders.encode_base64(attachment)
        attachment.add_header(
            "Content-Disposition", "attachment",
            filename=f"ClinGuard_WardWise_Critical_{get_ist_now().strftime('%Y%m%d_%H%M')}.pdf",
        )
        msg.attach(attachment)

        ctx = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as srv:
            srv.starttls(context=ctx)
            srv.login(sender, password)
            srv.send_message(msg)

        return True, (
            f"Ward-wise PDF report sent to {to_email} — "
            f"{total_critical} devices across {ward_count} wards"
        )
    except smtplib.SMTPAuthenticationError:
        return False, "Auth failed — check email and App Password"
    except Exception as e:
        return False, f"Failed to send PDF email: {e}"


def import_st_get(key, default=""):
    """Thin wrapper so send_bulk_email_alert_PDF can read session state."""
    return st.session_state.get(key, default)


# HELPERS

def get_device_rule(device_type: str) -> dict:
    dt = str(device_type).lower()
    for key, rule in DEVICE_RULES.items():
        if key in dt:
            return rule
    return FALLBACK_RULE

def get_device_icon(device_type) -> str:
    if pd.isna(device_type):
        return DEVICE_ICONS["default"]
    dt = str(device_type).lower()
    for key, icon in DEVICE_ICONS.items():
        if key in dt:
            return icon
    return DEVICE_ICONS["default"]

def _file_hash(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except Exception:
        return "unknown"

def _find_csv_for_dataset(dataset_name: str) -> tuple:
    if dataset_name not in DATASET_SOURCES:
        return None, "—", ""
    for path in DATASET_SOURCES[dataset_name]["paths"]:
        if path and Path(path).exists():
            return path, Path(path).name, _file_hash(path)
    return None, "—", ""

def _list_available_datasets() -> dict:
    result = {}
    for dataset_name, info in DATASET_SOURCES.items():
        path, filename, fhash = _find_csv_for_dataset(dataset_name)
        if path:
            fp      = Path(path)
            size_mb = fp.stat().st_size / (1024 * 1024)
            mtime   = datetime.fromtimestamp(fp.stat().st_mtime, tz=IST)
            result[dataset_name] = {
                "path": path, "emoji": info["emoji"],
                "description": info["description"],
                "size_mb": round(size_mb, 2),
                "modified": mtime.strftime("%d %b %H:%M"),
                "file": filename, "exists": True,
            }
        else:
            result[dataset_name] = {
                "path": None, "emoji": info["emoji"],
                "description": info["description"],
                "size_mb": 0.0, "modified": "Not generated",
                "file": "—", "exists": False,
            }
    return result


# ACTIVE URGENCIES HELPER

def get_active_urgencies() -> list:
    active = []
    if st.session_state.get("log_critical", True):   active.append("CRITICAL")
    if st.session_state.get("log_high",     True):   active.append("HIGH")
    if st.session_state.get("log_elevated", False):  active.append("ELEVATED")
    if st.session_state.get("log_monitor",  False):  active.append("MONITOR")
    return active


# FDA DATA GENERATOR

def generate_fda_data_embedded():
    progress_bar_ph = st.empty()
    status_ph       = st.empty()
    try:
        os.makedirs("hybrid_analysis", exist_ok=True)
        DEVICE_TYPES = [
            ("Ventilator",           "ventilator",        "High",   "LIFE_SUPPORT",  "ICU"),
            ("Infusion Pump",        "infusion pump",     "High",   "CRITICAL_CARE", "ICU"),
            ("Patient Monitor",      "patient monitor",   "High",   "MONITORING",    "ICU"),
            ("Defibrillator",        "defibrillator",     "High",   "LIFE_SUPPORT",  "EMERGENCY"),
            ("CT Scanner",           "ct scanner",        "Medium", "DIAGNOSTIC",    "RADIOLOGY"),
            ("MRI Scanner",          "mri",               "Medium", "DIAGNOSTIC",    "RADIOLOGY"),
            ("Syringe Pump",         "syringe pump",      "High",   "CRITICAL_CARE", "GENERAL_WARD"),
            ("Anesthesia Machine",   "anesthesia",        "High",   "SURGICAL",      "OPERATING_THEATRE"),
            ("Pacemaker Programmer", "pacemaker",         "High",   "CRITICAL_CARE", "CARDIOLOGY"),
            ("ECG Machine",          "electrocardiograph","Medium", "DIAGNOSTIC",    "CARDIOLOGY"),
            ("Dialysis Machine",     "dialysis",          "High",   "LIFE_SUPPORT",  "ICU"),
            ("Blood Gas Analyzer",   "blood gas",         "Medium", "DIAGNOSTIC",    "LABORATORY"),
            ("X-Ray System",         "x-ray",             "Medium", "DIAGNOSTIC",    "RADIOLOGY"),
            ("Ultrasound",           "ultrasound",        "Medium", "DIAGNOSTIC",    "RADIOLOGY"),
            ("Fetal Monitor",        "fetal monitor",     "Medium", "MONITORING",    "OBSTETRICS"),
        ]
        all_records = []
        device_id   = 1
        BASE        = "https://api.fda.gov/device"
        pbar        = progress_bar_ph.progress(0)
        total_types = len(DEVICE_TYPES)

        status_ph.info("📡 Fetching FDA adverse event data… this may take ~30 s")

        for idx, (device_type, keyword, criticality, patient_impact, default_ward) in enumerate(DEVICE_TYPES):
            try:
                url = (f"{BASE}/event.json?"
                       f"search=device.generic_name:{keyword}&limit=20&sort=date_received:desc")
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    for evt in r.json().get("results", []):
                        dev          = (evt.get("device") or [{}])[0]
                        manufacturer = dev.get("manufacturer_d_name", "Unknown").title().strip()[:30]
                        model        = dev.get("brand_name", device_type)[:20]
                        event_type   = evt.get("event_type", "")
                        ward = ("ICU" if patient_impact in ["LIFE_SUPPORT", "CRITICAL_CARE"]
                                and ("death" in str(event_type).lower()
                                     or "malfunction" in str(event_type).lower())
                                else default_ward)
                        mdr_text   = str(evt.get("mdr_text", [{}]))
                        narrative  = str(evt.get("narratives", ""))
                        et         = str(event_type).upper()
                        base_static = (
                            round(np.random.uniform(0.75, 0.95), 3) if "DEATH"       in et else
                            round(np.random.uniform(0.55, 0.80), 3) if "INJURY"      in et else
                            round(np.random.uniform(0.30, 0.65), 3) if "MALFUNCTION" in et else
                            round(np.random.uniform(0.10, 0.40), 3)
                        )
                        text_lower = (mdr_text + narrative).lower()
                        beh = (
                            round(np.random.uniform(0.60, 0.90), 3)
                            if any(k in text_lower for k in ["failure","malfunction","error","alarm","leak","contamination"])
                            else round(np.random.uniform(0.30, 0.60), 3)
                            if any(k in text_lower for k in ["delay","incorrect","unexpected","unintended"])
                            else round(np.random.uniform(0.05, 0.30), 3)
                        )
                        gnn   = round(min(1.0, (base_static*0.6 + beh*0.4)*1.1), 3)
                        xgb   = round(min(1.0, (base_static*0.5 + beh*0.5)*1.05), 3)
                        wc    = WARD_CRITICALITY.get(ward, 0.7)
                        pi    = PATIENT_IMPACT.get(patient_impact, 0.5)
                        total = round(min(1.0,
                            (0.25*base_static + 0.25*beh + 0.20*gnn + 0.30*xgb) * wc * pi * 1.5), 3)
                        urgency = (
                            "CRITICAL" if total >= 0.75 else
                            "HIGH"     if total >= 0.55 else
                            "ELEVATED" if total >= 0.30 else "MONITOR"
                        )
                        did = f"FDA{device_id:04d}"
                        record = {
                            "device": did, "name": f"{device_type} {device_id:03d}",
                            "device_type": device_type, "manufacturer": manufacturer,
                            "model": model, "generic_name": dev.get("generic_name", device_type)[:30],
                            "ward": ward, "ward_criticality": wc,
                            "patient_impact": patient_impact, "criticality": criticality,
                            "network": ("CRITICAL_CARE" if ward in ["ICU","EMERGENCY","OPERATING_THEATRE"] else "CLINICAL"),
                            "ip": f"10.{np.random.randint(1,5)}.{np.random.randint(1,254)}.{np.random.randint(1,254)}",
                            "maintenance": datetime.now().replace(
                                month=int(np.random.randint(1, 12)),
                                day=int(np.random.randint(1, 28))
                            ).strftime("%Y-%m-%d"),
                            "static_risk": base_static, "behaviour_risk": beh,
                            "gnn_risk": gnn, "xgb_risk": xgb, "total_risk": total,
                            "clinical_urgency": urgency, "fda_event_type": event_type,
                            "fda_report_date": evt.get("date_received", ""),
                            "fda_mdr_key": evt.get("mdr_report_key", ""),
                        }
                        all_records.append(record)
                        device_id += 1
            except Exception:
                pass
            pbar.progress(min(1.0, (idx + 1) / total_types))

        # Recalls
        try:
            r2 = requests.get(f"{BASE}/recall.json?search=product_code:FRN+MRZ+IYO&limit=50", timeout=30)
            if r2.status_code == 200:
                for rec in r2.json().get("results", []):
                    reason      = str(rec.get("reason_for_recall", "")).lower()
                    base_static = (0.85 if "malfunction" in reason or "failure" in reason
                                   else 0.65 if "software" in reason else 0.45)
                    beh   = round(np.random.uniform(0.3, 0.7), 3)
                    gnn   = round(min(1.0, base_static * 1.1), 3)
                    xgb   = round(min(1.0, base_static * 1.05), 3)
                    total = round(min(1.0, (0.25*base_static+0.25*beh+0.20*gnn+0.30*xgb)*1.3), 3)
                    all_records.append({
                        "device": f"RCL{device_id:04d}",
                        "name": f"Recalled Device {device_id:03d}",
                        "device_type": rec.get("product_description", "Medical Device")[:30],
                        "manufacturer": rec.get("recalling_firm", "Unknown")[:30],
                        "model": rec.get("product_description", "")[:20],
                        "generic_name": rec.get("product_description", "")[:30],
                        "ward": "GENERAL_WARD", "ward_criticality": 0.6,
                        "patient_impact": "DIAGNOSTIC", "criticality": "High",
                        "network": "CLINICAL",
                        "ip": f"10.3.{np.random.randint(1,254)}.{np.random.randint(1,254)}",
                        "maintenance": datetime.now().strftime("%Y-%m-%d"),
                        "static_risk": round(base_static, 3), "behaviour_risk": beh,
                        "gnn_risk": gnn, "xgb_risk": xgb, "total_risk": total,
                        "clinical_urgency": (
                            "CRITICAL" if total >= 0.75 else "HIGH" if total >= 0.55
                            else "ELEVATED" if total >= 0.30 else "MONITOR"
                        ),
                        "fda_event_type": "RECALL",
                        "fda_report_date": rec.get("recall_initiation_date", ""),
                        "fda_mdr_key": rec.get("recall_number", ""),
                    })
                    device_id += 1
        except Exception:
            pass

        progress_bar_ph.empty()
        if all_records:
            result_df = pd.DataFrame(all_records)
            result_df = result_df.sort_values("total_risk", ascending=False).reset_index(drop=True)
            out_path  = "hybrid_analysis/results_v6.3_fda.csv"
            result_df.to_csv(out_path, index=False)
            status_ph.success(f" FDA data generated: {len(result_df)} devices saved to {out_path}")
            return True, len(result_df)
        else:
            status_ph.error("WRONG No FDA records could be fetched. Check your internet connection.")
            return False, 0
    except Exception as e:
        progress_bar_ph.empty()
        status_ph.error(f"WRONG Error generating FDA data: {e}")
        return False, 0


# CSV LOADING

@st.cache_data(show_spinner=" Loading dataset…", ttl=3600)
def _load_csv(path: str, file_hash: str) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={
        "device": "id", "device_type": "type", "total_risk": "score",
        "clinical_urgency": "urgency", "static_risk": "static",
        "behaviour_risk": "beh", "gnn_risk": "gnn", "xgb_risk": "xgb",
    }, errors="ignore")
    for col, val in [
        ("static", 0.0), ("beh", 0.0), ("gnn", 0.0), ("xgb", 0.0),
        ("manufacturer", "N/A"), ("model", "N/A"), ("ip", "N/A"),
        ("network", "N/A"), ("maintenance", "N/A"), ("ward", "Unknown"),
        ("status", "Monitored"), ("last_alert", "—"), ("confidence", 0.85),
    ]:
        if col not in df.columns:
            df[col] = val
    df["score"]      = pd.to_numeric(df["score"],      errors="coerce").fillna(0.0)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.85)
    df["urgency"]    = df["urgency"].fillna("MONITOR").astype(str).str.strip().str.upper()
    if "name" not in df.columns:
        df["name"] = df["id"]
    df["urgency"] = df.apply(
        lambda r: (
            "CRITICAL" if r["score"] >= 0.75 else
            "HIGH"     if r["score"] >= 0.55 else
            "ELEVATED" if r["score"] >= 0.30 else "MONITOR"
        ) if r["urgency"] not in URGENCY_COLOR else r["urgency"],
        axis=1,
    )

    # ──  CHANGE: apply adaptive thresholds after loading ───────────────────
    # This derives adaptive_threshold + pred_label for every device.
    # Uses stored thresholds if Hybrid dataset already has them;
    # otherwise computes dynamically for FDA / MIMIC datasets.
    df = create_predictions(df)
    # ─────────────────────────────────────────────────────────────────────────

    return df.sort_values("score", ascending=False).reset_index(drop=True)

def load_data() -> tuple:
    all_datasets     = _list_available_datasets()
    selected_dataset = st.session_state.get("selected_dataset", None)

    if (not selected_dataset or selected_dataset not in all_datasets
            or not all_datasets[selected_dataset]["exists"]):
        for name, info in all_datasets.items():
            if info["exists"]:
                selected_dataset = name
                st.session_state["selected_dataset"] = selected_dataset
                break
        else:
            return None, "—"

    path, filename, fhash = _find_csv_for_dataset(selected_dataset)
    if not path:
        return None, "—"

    df        = _load_csv(path, fhash).copy()
    overrides = load_device_status_overrides()
    if overrides:
        df = apply_status_overrides(df, overrides)
    return df, filename


# JSON HELPERS

def _load_json(fname: str) -> dict:
    try:
        p = Path(fname)
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return {}

def _save_json(fname: str, data):
    try:
        Path(fname).write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Could not save {fname}: {e}")

def load_device_status_overrides() -> dict:
    return _load_json("device_status_overrides.json")

def save_device_status_overrides(overrides: dict):
    _save_json("device_status_overrides.json", overrides)

def apply_status_overrides(df: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    for device_id, status in overrides.items():
        df.loc[df["id"].astype(str) == str(device_id), "status"] = status
    return df

def load_alerts_history() -> list:
    data = _load_json("alerts_history.json")
    return data if isinstance(data, list) else []

def clear_alerts_history() -> bool:
    try:
        Path("alerts_history.json").unlink(missing_ok=True)
        st.session_state.alert_log = []
        _RECENT_LOG_CACHE.clear()
        return True
    except Exception:
        return False

def should_log_alert(urgency: str) -> bool:
    key_map = {
        "CRITICAL": "log_critical", "HIGH": "log_high",
        "ELEVATED": "log_elevated", "MONITOR": "log_monitor",
    }
    key = key_map.get(str(urgency).strip().upper())
    if key is None:
        return True
    return bool(st.session_state.get(key, urgency in ("CRITICAL", "HIGH")))

def add_to_alert_log(device_name, device_id, urgency, score,
                     message, channel="auto", status="triggered", dedupe_secs=300) -> bool:
    if not should_log_alert(urgency):
        return False

    cache_key = f"{device_id}::{str(urgency).upper()}"
    now_ts    = get_ist_now().timestamp()

    if now_ts - _RECENT_LOG_CACHE.get(cache_key, 0) < dedupe_secs:
        return False

    try:
        history = load_alerts_history()
        for entry in reversed(history[-200:]):
            if (str(entry.get("device_id")) == str(device_id) and
                    str(entry.get("urgency", "")).upper() == str(urgency).upper()):
                prior_ts = datetime.fromisoformat(entry["timestamp"]).timestamp()
                if now_ts - prior_ts < dedupe_secs:
                    _RECENT_LOG_CACHE[cache_key] = prior_ts
                    return False
                break
    except Exception:
        pass

    _RECENT_LOG_CACHE[cache_key] = now_ts

    entry = {
        "timestamp":   get_ist_now().isoformat(),
        "device_name": str(device_name),
        "device_id":   str(device_id),
        "urgency":     str(urgency).strip().upper(),
        "score":       round(float(score), 3),
        "message":     str(message),
        "channel":     str(channel),
        "status":      str(status),
    }

    if "alert_log" not in st.session_state:
        st.session_state.alert_log = []
    st.session_state.alert_log.append(entry)

    try:
        history = load_alerts_history()
        history.append(entry)
        if len(history) > 2000:
            history = history[-2000:]
        _save_json("alerts_history.json", history)
    except Exception:
        pass

    return True


# SMS & EMAIL

def _send_raw_sms(to: str, body: str) -> tuple:
    sid   = st.session_state.get("twilio_sid", "")
    token = st.session_state.get("twilio_token", "")
    frm   = st.session_state.get("twilio_from", "")
    if not TWILIO_AVAILABLE:
        return False, "Twilio not installed. Run: pip install twilio"
    if not all([sid, token, frm]):
        return False, "Twilio credentials not configured."
    body = body[:SMS_MAX_CHARS]
    try:
        client  = Client(sid, token)
        message = client.messages.create(body=body, from_=frm, to=to)
        if message.status in ("failed", "undelivered"):
            return False, f"SMS not delivered. Status: {message.status}"
        return True, f"SMS sent! SID: {message.sid[:12]}…"
    except Exception as e:
        err = str(e)
        if "20003" in err: return False, "Auth failed — check Account SID and Auth Token"
        if "21211" in err: return False, "Invalid number format — use +[country][number]"
        if "21608" in err: return False, "Unverified number (Trial). Verify at Twilio Console."
        return False, f"SMS failed: {err}"

def send_email_alert(to_email: str, device_data: dict, risk_score: float, urgency: str) -> tuple:
    try:
        sender   = st.session_state.get("email_sender", EMAIL_SENDER)
        password = st.session_state.get("email_password", EMAIL_PASSWORD)
        if not sender or "your_email" in sender:
            return False, "Email credentials not configured."
        rule    = get_device_rule(device_data.get("type", ""))
        action  = rule["action_critical"] if urgency == "CRITICAL" else rule["action_high"]
        subject = f"ClinGuard {urgency}: {device_data.get('name','Unknown')} | {risk_score:.3f}"
        body    = (
            f"CLINICAL GUARDIAN — DEVICE RISK ALERT\n{'='*40}\n"
            f"Time    : {get_ist_now().strftime('%Y-%m-%d %H:%M:%S IST')}\n"
            f"Urgency : {urgency}\nScore   : {risk_score:.3f}\n\n"
            f"DEVICE\n------\n"
            f"Name    : {device_data.get('name','N/A')}\n"
            f"ID      : {device_data.get('id','N/A')}\n"
            f"Type    : {device_data.get('type','N/A')}\n"
            f"Ward    : {device_data.get('ward','N/A')}\n\n"
            f"ACTION\n------\n{action}\n\nClinical Guardian v8.0"
        )
        msg            = MIMEMultipart()
        msg["From"]    = sender
        msg["To"]      = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))
        ctx = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as srv:
            srv.starttls(context=ctx)
            srv.login(sender, password)
            srv.send_message(msg)
        return True, f"Email sent to {to_email}!"
    except smtplib.SMTPAuthenticationError:
        return False, "Auth failed — check email and App Password"
    except Exception as e:
        return False, f"Email failed: {e}"

def send_bulk_email_alert(to_email: str, devices_df: pd.DataFrame) -> tuple:
    try:
        sender   = st.session_state.get("email_sender", EMAIL_SENDER)
        password = st.session_state.get("email_password", EMAIL_PASSWORD)
        if not sender or "your_email" in sender:
            return False, "Email credentials not configured."

        urgency_counts = devices_df["urgency"].value_counts().to_dict()
        summary_parts  = [
            f"{cnt} {urg}" for urg, cnt in
            sorted(urgency_counts.items(),
                   key=lambda x: URGENCY_LEVELS.index(x[0]) if x[0] in URGENCY_LEVELS else 99)
        ]
        subject = (
            f"ClinGuard Alert: {', '.join(summary_parts)} | "
            f"{get_ist_now().strftime('%d %b %Y %H:%M IST')}"
        )

        def block(label, subset):
            if subset.empty:
                return f"{label} — None\n"
            lines = [f"{label} ({len(subset)} devices)", "-"*50]
            for _, r in subset.iterrows():
                rule   = get_device_rule(r.get("type", ""))
                action = rule["action_critical"] if label == "CRITICAL" else rule["action_high"]
                lines.append(f"  {r.get('name', r.get('id','?')):<20} Score:{r.get('score',0):.3f}")
                lines.append(f"    -> {action}")
            return "\n".join(lines)

        blocks = []
        for urg in URGENCY_LEVELS:
            subset = devices_df[devices_df["urgency"] == urg]
            if not subset.empty:
                blocks.append(block(urg, subset))

        body = (
            f"CLINICAL GUARDIAN — RISK ALERT REPORT\n{'='*40}\n"
            f"Time : {get_ist_now().strftime('%Y-%m-%d %H:%M:%S IST')}\n"
            f"Total: {len(devices_df)} alerted devices\n\n"
            + "\n\n".join(blocks)
            + "\n\nClinical Guardian v8.0"
        )

        msg            = MIMEMultipart()
        msg["From"]    = sender
        msg["To"]      = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))
        ctx = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as srv:
            srv.starttls(context=ctx)
            srv.login(sender, password)
            srv.send_message(msg)
        return True, f"Bulk email sent to {to_email}!"
    except Exception as e:
        return False, f"Bulk email failed: {e}"

def send_bulk_sms_alert(to_number: str, devices_df: pd.DataFrame) -> tuple:
    try:
        urgency_counts = devices_df["urgency"].value_counts().to_dict()
        summary_parts  = [
            f"{cnt} {urg}" for urg, cnt in
            sorted(urgency_counts.items(),
                   key=lambda x: URGENCY_LEVELS.index(x[0]) if x[0] in URGENCY_LEVELS else 99)
        ]
        body   = (f"ClinGuard ALERT: {', '.join(summary_parts)} devices | "
                  f"{get_ist_now().strftime('%d %b %Y %H:%M IST')}")
        ok, msg = _send_raw_sms(to_number, body)
        return (True, f"Bulk SMS sent to {to_number}!") if ok else (False, msg)
    except Exception as e:
        return False, f"Bulk SMS failed: {e}"


# AUTO-ALERTS

def run_auto_alerts(df: pd.DataFrame):
    if st.session_state.get("auto_alerts_sent"):
        return
    st.session_state["auto_alerts_sent"] = True

    active_urgencies  = get_active_urgencies()
    if not active_urgencies:
        st.session_state["auto_alert_results"] = []
        return

    alert_candidates = df[df["urgency"].isin(active_urgencies)]
    if alert_candidates.empty:
        st.session_state["auto_alert_results"] = []
        return

    for _, device in alert_candidates.iterrows():
        add_to_alert_log(
            device_name=device.get("name", device.get("id", "Unknown")),
            device_id=device.get("id", "—"),
            urgency=device.get("urgency", "HIGH"),
            score=device.get("score", 0),
            message="Auto-detected risk device",
            channel="auto", status="triggered",
        )

    results = []

    email_ok = (bool(st.session_state.get("email_sender", "")) and
                "your_email" not in st.session_state.get("email_sender", ""))
    if email_ok and st.session_state.get("email_enabled"):
        ok, msg = send_bulk_email_alert(ALERT_EMAIL_TO, alert_candidates)
        results.append(("Bulk Email", len(alert_candidates), ok, msg))

    sms_ok = (bool(st.session_state.get("twilio_sid", "")) and
              bool(st.session_state.get("twilio_token", "")) and
              bool(st.session_state.get("twilio_from", "")) and
              bool(st.session_state.get("twilio_to", "")))
    if sms_ok and st.session_state.get("twilio_enabled"):
        ok, msg = send_bulk_sms_alert(st.session_state["twilio_to"], alert_candidates)
        results.append(("Bulk SMS", len(alert_candidates), ok, msg))

    st.session_state["auto_alert_results"] = results


# REPORT & EXPORT

def generate_device_report(d, sc, urg) -> str:
    rule   = get_device_rule(d.get("type", ""))
    action = rule["action_critical"] if urg == "CRITICAL" else rule["action_high"]
    # ──  CHANGE: include adaptive threshold in device report ────────────────
    adaptive_thr = float(d.get("adaptive_threshold", EVAL_THRESHOLD))
    threshold_line = f"Threshold : {adaptive_thr:.3f}  (adaptive per ward/impact)\n"
    # ─────────────────────────────────────────────────────────────────────────
    return (
        f"{'='*70}\nCLINICAL GUARDIAN — DEVICE SECURITY REPORT\n{'='*70}\n"
        f"Generated : {get_ist_now().strftime('%Y-%m-%d %H:%M:%S IST')}\n\n"
        f"DEVICE\n------\n"
        f"Name   : {d.get('name','N/A')}\nID     : {d.get('id','N/A')}\n"
        f"Type   : {d.get('type','N/A')}\nWard   : {d.get('ward','N/A')}\n"
        f"Network: {d.get('network','N/A')}\nIP     : {d.get('ip','N/A')}\n\n"
        f"RISK ASSESSMENT\n---------------\n"
        f"Score     : {sc:.3f}   Urgency: {urg}\n"
        f"{threshold_line}"
        f"CVE/Static: {d.get('static',0):.3f}\nBehaviour : {d.get('beh',0):.3f}\n"
        f"GNN       : {d.get('gnn',0):.3f}\nXGBoost   : {d.get('xgb',0):.3f}\n\n"
        f"RECOMMENDED ACTION\n------------------\n{action}\n\n{'='*70}\n"
    )

def export_to_csv(df: pd.DataFrame, cols=None) -> str:
    return (df[cols] if cols else df).to_csv(index=False)


# SESSION STATE INIT

def initialize_session_state():
    defaults = {
        "sel": None, "alert_sent": set(), "alert_log": [],
        "page_num": 1, "items_per_page": 10, "selected_devices": set(),
        "device_history": {}, "selected_dataset": None,
        "twilio_sid": ACCOUNT_SID, "twilio_token": AUTH_TOKEN,
        "twilio_from": FROM_NUMBER, "twilio_to": TO_NUMBER,
        "twilio_enabled": False, "email_sender": EMAIL_SENDER,
        "email_password": EMAIL_PASSWORD, "email_enabled": False,
        "status_overrides": load_device_status_overrides(),
        "log_critical": True, "log_high": True,
        "log_elevated": False, "log_monitor": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    for fname, keys in [
        ("alert_config.json",  ["log_critical","log_high","log_elevated","log_monitor"]),
        ("twilio_config.json", ["twilio_sid","twilio_token","twilio_from","twilio_to","twilio_enabled"]),
        ("email_config.json",  ["email_sender","email_password","email_enabled"]),
    ]:
        saved = _load_json(fname)
        if saved:
            remap = {
                "account_sid": "twilio_sid", "auth_token": "twilio_token",
                "from_number": "twilio_from", "to_number": "twilio_to",
                "enabled": "twilio_enabled", "sender": "email_sender", "password": "email_password",
            }
            for src_k, tgt_k in remap.items():
                if src_k in saved:
                    st.session_state[tgt_k] = saved[src_k]
            for k in keys:
                if k in saved:
                    st.session_state[k] = saved[k]


# PAGE CONFIG & CSS

st.set_page_config(
    page_title="Clinical Guardian v8",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
html,body,[class*="css"]{
    background:#090d13;color:#e2e8f0;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}
[data-testid="stSidebar"]{background:#0b1219!important}
[data-testid="stSidebar"] *{color:#94a3b8!important}
[data-testid="stAppViewContainer"]>.main{background:#090d13}
[data-testid="stMetric"]{
    background:#0f1520;border:1px solid rgba(255,255,255,.07);
    border-radius:12px;padding:16px!important;
}
[data-testid="stMetricValue"]{font-size:28px!important;font-weight:700!important}
[data-testid="stMetricLabel"]{color:#64748b!important;font-size:12px!important}
[data-testid="stTextInput"] input{
    background:#0f1520!important;color:#e2e8f0!important;
    border:1px solid rgba(255,255,255,.15)!important;border-radius:8px!important;
}
.stButton>button{
    background:transparent!important;border:1.5px solid rgba(255,255,255,.2)!important;
    color:#94a3b8!important;border-radius:8px!important;padding:10px 20px!important;
    font-weight:600!important;transition:all .3s ease!important;
}
.stButton>button:hover{
    border-color:#00d4ff!important;color:#00d4ff!important;
    background:rgba(0,212,255,.05)!important;
}
.stSelectbox>div>div{
    background:#0f1520!important;border:1px solid rgba(255,255,255,.15)!important;
    border-radius:8px!important;color:#e2e8f0!important;
}
.stRadio>label{color:#e2e8f0!important}
hr{border-color:rgba(255,255,255,.07)!important}
.status-badge{
    display:inline-block;padding:3px 9px;border-radius:20px;
    font-size:10px;font-weight:700;
}
[data-testid="stTab"]{color:#64748b!important}
[data-testid="stTab"][aria-selected="true"]{color:#00d4ff!important}
</style>""", unsafe_allow_html=True)


# BOOT

initialize_session_state()


# SIDEBAR NAVIGATION

with st.sidebar:
    st.markdown("### 🧭 NAVIGATION")
    nav_choice = st.radio(
        "Page",
        [" Dashboard", "📈 Analytics", " Alerts", "🔧 Inventory", "⚙️ Settings"],
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.markdown("---")
    if st.button("🔄 REFRESH DATA", use_container_width=True, key="sb_refresh"):
        st.session_state.pop("auto_alerts_sent", None)
        st.session_state.pop("auto_alert_results", None)
        st.cache_data.clear()
        st.rerun()

# Load data early
df, source_filename = load_data()

# Rest of sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("###  CURRENT SOURCE")
    st.caption(f" {st.session_state.get('selected_dataset','—')}")
    if df is not None:
        st.caption(f" {len(df):,} devices loaded")
    st.caption(f"📄 {source_filename}")

    if df is not None:
        st.markdown("---")
        st.markdown("###  FILTERS")
        wards = ["All Departments"] + sorted(df["ward"].dropna().unique().tolist())
        dept  = st.selectbox("Department", wards,
                             label_visibility="collapsed", key="dept_select")
        filt  = st.radio("Urgency", ["ALL"] + URGENCY_LEVELS,
                         label_visibility="collapsed", key="urgency_select")
        status_filter = st.multiselect(
            "Status", DEVICE_STATUS_OPTIONS,
            default=DEVICE_STATUS_OPTIONS,
            label_visibility="collapsed", key="status_select",
        )
        risk_min, risk_max = st.slider(
            "Risk Score", 0.0, 1.0, (0.0, 1.0), step=0.05,
            label_visibility="collapsed", key="risk_slider",
        )
        st.markdown("---")
        st.markdown("###  URGENCY SUMMARY")
        counts = df["urgency"].value_counts()
        for u in URGENCY_LEVELS:
            cnt = int(counts.get(u, 0))
            c   = URGENCY_COLOR.get(u, "#888")
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:8px 0;border-bottom:1px solid rgba(255,255,255,.04);'>"
                f"<span>{u}</span>"
                f"<span style='color:{c};font-weight:700;font-size:15px;'>{cnt}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("---")
        sms_ok   = bool(st.session_state.get("twilio_enabled"))
        email_ok = bool(st.session_state.get("email_enabled"))
        st.markdown(
            f"<div style='font-size:11px;padding:10px;background:#0f1520;border-radius:8px;'>"
            f"{'' if sms_ok else 'WRONG'} SMS  ·  {'' if email_ok else 'WRONG'} Email</div>",
            unsafe_allow_html=True,
        )

        active = get_active_urgencies()
        st.markdown("---")
        st.markdown("###  ACTIVE ALERT FILTERS")
        for u in URGENCY_LEVELS:
            on  = u in active
            c   = URGENCY_COLOR.get(u, "#888") if on else "#374151"
            lbl = "ON" if on else "off"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:6px 0;border-bottom:1px solid rgba(255,255,255,.04);'>"
                f"<span style='color:{c};'>{u}</span>"
                f"<span style='color:{c};font-size:11px;font-weight:700;'>{lbl}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        dept           = "All Departments"
        filt           = "ALL"
        status_filter  = DEVICE_STATUS_OPTIONS
        risk_min, risk_max = 0.0, 1.0


# HEADER

h1, h2, h3 = st.columns([2, 2, 1])
with h1:
    st.markdown("""
    <div style='padding:20px;background:linear-gradient(135deg,rgba(0,212,255,.1) 0%,rgba(255,59,92,.05) 100%);
                border:1px solid rgba(0,212,255,.3);border-radius:12px;'>
      <h1 style='margin:0;font-size:28px;color:#00d4ff;'> Clinical Guardian</h1>
      <p style='margin:5px 0 0;color:#64748b;font-size:13px;'>
        Advanced Medical Device Risk Monitoring · v8.0
      </p>
    </div>""", unsafe_allow_html=True)

with h2:
    current_dataset = st.session_state.get("selected_dataset") or "Auto-detecting…"
    st.markdown(f"""
    <div style='padding:20px;background:#0f1520;border:1px solid rgba(255,255,255,.07);border-radius:12px;'>
      <div style='font-size:11px;color:#64748b;margin-bottom:6px;'> CURRENT DATASET</div>
      <div style='font-size:14px;font-weight:700;color:#00d4ff;margin-bottom:3px;'>{current_dataset}</div>
      <div style='font-size:11px;color:#475569;'>Click the expander below to change</div>
    </div>""", unsafe_allow_html=True)

with h3:
    now_ist = get_ist_now()
    st.markdown(f"""
    <div style='padding:20px;background:#0f1520;border:1px solid rgba(255,255,255,.07);border-radius:12px;'>
      <div style='font-size:11px;color:#64748b;margin-bottom:6px;'>⏰ NOW (IST)</div>
      <div style='font-size:15px;font-weight:700;color:#e2e8f0;'>{now_ist.strftime("%H:%M:%S")}</div>
      <div style='font-size:11px;color:#475569;'>{now_ist.strftime("%d %b %Y")}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")


# DATA GUARD

if df is None or len(df) == 0:
    st.error("WRONG No dataset loaded. Use the **DATASET MANAGER** above to load a dataset.")
    st.stop()

run_auto_alerts(df)


# APPLY FILTERS

disp = df.copy()
if dept != "All Departments":
    disp = disp[disp["ward"] == dept]
if filt != "ALL":
    disp = disp[disp["urgency"] == filt]
if status_filter:
    disp = disp[disp["status"].isin(status_filter)]
disp = disp[(disp["score"] >= risk_min) & (disp["score"] <= risk_max)]


# PAGE: DASHBOARD

if nav_choice == " Dashboard":

    with st.expander(" Dataset Manager — switch datasets or generate FDA data", expanded=False):
        all_datasets = _list_available_datasets()

        for i, (dataset_name, dataset_info) in enumerate(all_datasets.items()):
            is_current   = dataset_name == st.session_state.get("selected_dataset")
            border_color = "#00d4ff" if is_current else "rgba(255,255,255,.07)"

            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"""
                <div style='padding:12px;background:{"rgba(0,212,255,.05)" if is_current else "#0f1520"};
                            border:{"2px" if is_current else "1px"} solid {border_color};border-radius:8px;'>
                  <b>{dataset_info['emoji']} {dataset_name}</b><br/>
                  <small style='color:#64748b;'>{dataset_info['description']}</small>
                </div>""", unsafe_allow_html=True)

            with col2:
                status_text = (
                    f"📄 {dataset_info['file']}<br/>📦 {dataset_info['size_mb']} MB · 🕐 {dataset_info['modified']}"
                    if dataset_info["exists"] else " File not found"
                )
                st.markdown(f"""
                <div style='padding:12px;background:#0f1520;border:1px solid rgba(255,255,255,.06);
                            border-radius:8px;font-size:12px;color:#94a3b8;'>
                  {status_text}
                </div>""", unsafe_allow_html=True)

            with col3:
                if is_current and dataset_info["exists"]:
                    st.markdown("""
                    <div style='padding:10px;background:#00d4ff22;border:1px solid #00d4ff;
                                border-radius:8px;text-align:center;color:#00d4ff;font-weight:700;'>
                       ACTIVE
                    </div>""", unsafe_allow_html=True)
                elif dataset_info["exists"]:
                    if st.button("▶ Load", key=f"load_ds_{i}", use_container_width=True):
                        st.session_state["selected_dataset"] = dataset_name
                        st.session_state.pop("auto_alerts_sent", None)
                        st.session_state["page_num"] = 1
                        st.cache_data.clear()
                        st.rerun()
                elif "FDA" in dataset_name:
                    if st.button("⬇️ Generate FDA", key=f"gen_ds_{i}", use_container_width=True):
                        success, count = generate_fda_data_embedded()
                        if success:
                            time.sleep(1)
                            st.session_state["selected_dataset"] = dataset_name
                            st.session_state["page_num"] = 1
                            st.cache_data.clear()
                            st.rerun()
                else:
                    st.markdown("""
                    <div style='padding:10px;background:#64748b22;border-radius:8px;
                                text-align:center;color:#64748b;font-size:12px;'>
                      ⏳ Not available
                    </div>""", unsafe_allow_html=True)

            if i < len(all_datasets) - 1:
                st.markdown("<hr style='border-color:rgba(255,255,255,.04);margin:4px 0;'>",
                            unsafe_allow_html=True)

    st.markdown("---")

    for label, count, ok, msg in st.session_state.get("auto_alert_results", []):
        if ok:
            st.success(f" Auto-alert sent: {count} devices via {label}")
        else:
            st.warning(f" Auto-alert attempt ({label}): {msg}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🔴 CRITICAL", int((disp["urgency"]=="CRITICAL").sum()), "Isolate now")
    m2.metric("🟠 HIGH",     int((disp["urgency"]=="HIGH").sum()),     "Urgent")
    m3.metric("🟡 ELEVATED", int((disp["urgency"]=="ELEVATED").sum()), "Schedule")
    m4.metric("🟢 MONITOR",  int((disp["urgency"]=="MONITOR").sum()),  "Routine")
    m5.metric(" AVG RISK", f"{disp['score'].mean():.3f}", f"{len(disp)} devices shown")

    st.markdown("---")

    ch1, ch2 = st.columns([2, 1])
    with ch1:
        st.markdown("####  Top 30 Devices by Risk")
        top = disp.head(30)
        if len(top) > 0:
            fig1 = go.Figure(go.Bar(
                x=top["name"].astype(str), y=top["score"],
                marker_color=[URGENCY_COLOR.get(u, "#888") for u in top["urgency"]],
                marker_line_width=0,
                text=[f"{s:.2f}" for s in top["score"]], textposition="outside",
                hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
            ))
            fig1.add_hline(y=0.75, line_dash="dash", line_color="#ff3b5c", opacity=0.5,
                           annotation_text="CRITICAL", annotation_font=dict(color="#ff3b5c", size=9))
            fig1.add_hline(y=0.55, line_dash="dash", line_color="#ff8c42", opacity=0.5,
                           annotation_text="HIGH", annotation_font=dict(color="#ff8c42", size=9))
            fig1.update_layout(
                plot_bgcolor="#0f1520", paper_bgcolor="#0f1520",
                font=dict(color="#94a3b8", size=10), height=320,
                xaxis=dict(tickangle=-45, gridcolor="rgba(255,255,255,.04)"),
                yaxis=dict(range=[0, 1.2], gridcolor="rgba(255,255,255,.04)"),
                margin=dict(l=0, r=80, t=10, b=100), showlegend=False,
            )
            st.plotly_chart(fig1, use_container_width=True)

    with ch2:
        st.markdown("####  Distribution")
        uc = disp["urgency"].value_counts().reset_index()
        uc.columns = ["urgency", "count"]
        if len(uc) > 0:
            fig2 = go.Figure(go.Pie(
                labels=uc["urgency"], values=uc["count"],
                marker_colors=[URGENCY_COLOR.get(u, "#888") for u in uc["urgency"]],
                hole=0.5, textinfo="label+value",
            ))
            fig2.add_annotation(
                text=f"<b>{len(disp)}</b>", x=0.5, y=0.5,
                showarrow=False, font=dict(size=20, color="#e2e8f0"),
            )
            fig2.update_layout(
                plot_bgcolor="#0f1520", paper_bgcolor="#0f1520",
                font=dict(color="#94a3b8"), height=320, showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📱 Device Listing")

    search      = st.text_input("🔍 Search devices…", placeholder="Name, ID, ward or type",
                                key="dash_search")
    search_disp = disp.copy()

    if search.strip():
        q    = search.strip().lower()
        mask = (
            search_disp["name"].astype(str).str.lower().str.contains(q, na=False) |
            search_disp["id"].astype(str).str.lower().str.contains(q, na=False)   |
            search_disp["ward"].astype(str).str.lower().str.contains(q, na=False) |
            search_disp["type"].astype(str).str.lower().str.contains(q, na=False)
        )
        search_disp = search_disp[mask]

    search_disp = search_disp.sort_values("score", ascending=False)

    ITEMS       = 10
    total_pages = max(1, (len(search_disp) + ITEMS - 1) // ITEMS)
    if st.session_state.page_num > total_pages:
        st.session_state.page_num = 1
    start   = (st.session_state.page_num - 1) * ITEMS
    page_df = search_disp.iloc[start: start + ITEMS]

    st.caption(f"Showing {start+1}–{min(start+ITEMS, len(search_disp))} of {len(search_disp)} devices")

    if len(page_df) > 0:
        hh = st.columns([1.5, 2, 1.5, 1.5, 1.2, 1])
        for col_obj, lbl in zip(hh, ["Device","Ward","Type","Score","Status","Action"]):
            col_obj.markdown(
                f"<span style='font-size:10px;color:#475569;font-weight:600;'>{lbl}</span>",
                unsafe_allow_html=True)
        st.markdown("<hr style='border-color:rgba(255,255,255,.06);margin:3px 0 8px;'>",
                    unsafe_allow_html=True)

        for _, row in page_df.iterrows():
            rid      = str(row.get("id", "—"))
            urg      = str(row.get("urgency", "MONITOR")).upper()
            clr      = URGENCY_COLOR.get(urg, "#888")
            sc       = float(row.get("score", 0))
            nm       = str(row.get("name", rid))
            wd       = str(row.get("ward", "—"))
            tp       = str(row.get("type", "—"))
            stat     = str(row.get("status", "Monitored"))
            icon     = get_device_icon(tp)
            stat_clr = DEVICE_STATUS_COLORS.get(stat, "#94a3b8")

            rc = st.columns([1.5, 2, 1.5, 1.5, 1.2, 1])
            rc[0].markdown(f"{icon} **{nm[:18]}**")
            rc[1].markdown(wd)
            rc[2].markdown(f"<span style='font-size:11px;'>{tp[:14]}</span>",
                           unsafe_allow_html=True)
            rc[3].markdown(
                f"<span style='color:{clr};font-weight:700;font-size:13px;'>{sc:.3f}</span>",
                unsafe_allow_html=True)
            rc[4].markdown(
                f"<span class='status-badge' style='background:{stat_clr}22;color:{stat_clr};'>{stat}</span>",
                unsafe_allow_html=True)
            with rc[5]:
                if st.button("View", key=f"view_{rid}", use_container_width=True):
                    st.session_state.sel = rid
                    st.rerun()

            st.markdown("<hr style='border-color:rgba(255,255,255,.03);margin:0;'>",
                        unsafe_allow_html=True)

    pg1, pg2, pg3 = st.columns([1, 2, 1])
    with pg1:
        if st.button("⬅ Prev", key="pg_prev", disabled=st.session_state.page_num <= 1):
            st.session_state.page_num -= 1
            st.rerun()
    with pg2:
        st.markdown(
            f"<div style='text-align:center;font-size:12px;color:#64748b;'>"
            f"Page {st.session_state.page_num} / {total_pages}</div>",
            unsafe_allow_html=True)
    with pg3:
        if st.button("Next ➜", key="pg_next", disabled=st.session_state.page_num >= total_pages):
            st.session_state.page_num += 1
            st.rerun()

    # Device Detail
    sel = st.session_state.get("sel")
    if sel:
        m_row = df[df["id"].astype(str) == str(sel)]
        if not m_row.empty:
            d   = m_row.iloc[0]
            urg = str(d.get("urgency", "MONITOR")).upper()
            clr = URGENCY_COLOR.get(urg, "#888")
            nm  = str(d.get("name", d.get("id", "—")))
            sc  = float(d.get("score", 0))

            st.markdown("---")
            st.markdown(f"###  Device Detail — {nm}")
            dd1, dd2 = st.columns([1, 1])

            with dd1:
                # ──  CHANGE: show adaptive threshold in device detail panel ─
                adaptive_thr = float(d.get("adaptive_threshold", EVAL_THRESHOLD))
                pred_label   = int(d.get("pred_label", 0))
                pred_text    = " PREDICTED THREAT" if pred_label == 1 else " PREDICTED SAFE"
                pred_color   = "#ff3b5c" if pred_label == 1 else "#22c984"
                # ──────────────────────────────────────────────────────────────
                st.markdown(f"""
                <div style='background:#0f1520;border-top:3px solid {clr};
                            border:1px solid rgba(255,255,255,.07);
                            border-radius:10px;padding:16px;'>
                  <table style='width:100%;font-size:12px;border-collapse:collapse;'>
                     <tr><td style='color:#64748b;padding:4px 0;'>Ward</td>
                        <td style='color:#e2e8f0;text-align:right;'>{d.get('ward','—')}</td></tr>
                     <tr><td style='color:#64748b;padding:4px 0;'>Type</td>
                        <td style='color:#e2e8f0;text-align:right;'>{d.get('type','—')}</td></tr>
                     <tr><td style='color:#64748b;padding:4px 0;'>Manufacturer</td>
                        <td style='color:#e2e8f0;text-align:right;'>{d.get('manufacturer','—')}</td></tr>
                     <tr><td style='color:#64748b;padding:4px 0;'>Network</td>
                        <td style='color:#e2e8f0;text-align:right;'>{d.get('network','—')}</td></tr>
                     <tr><td style='color:#64748b;padding:4px 0;'>IP</td>
                        <td style='color:#e2e8f0;text-align:right;'>{d.get('ip','—')}</td></tr>
                     <tr><td style='color:#64748b;padding:4px 0;'>Confidence</td>
                        <td style='color:#e2e8f0;text-align:right;'>{float(d.get('confidence',0.85)):.0%}</td></tr>
                     <tr><td style='color:#64748b;padding:4px 0;'>Adaptive Threshold</td>
                        <td style='color:#00d4ff;text-align:right;font-weight:700;'>{adaptive_thr:.3f}</td></tr>
                     <tr><td style='color:#64748b;padding:4px 0;'>Prediction</td>
                        <td style='color:{pred_color};text-align:right;font-weight:700;'>{pred_text}</td></tr>
                   </table>
                </div>""", unsafe_allow_html=True)

                rule   = get_device_rule(d.get("type", ""))
                action = rule["action_critical"] if urg == "CRITICAL" else rule["action_high"]
                st.markdown(f"""
                <div style='background:#0f1520;border:1px solid rgba(255,255,255,.07);
                            border-radius:10px;padding:14px;margin-top:10px;'>
                  <div style='font-size:11px;color:#64748b;margin-bottom:6px;'>RECOMMENDED ACTION</div>
                  <div style='font-size:12px;color:{clr};font-weight:600;'>{action}</div>
                  <div style='font-size:10px;color:#475569;margin-top:6px;'> {rule['note']}</div>
                </div>""", unsafe_allow_html=True)

            with dd2:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=sc,
                    number=dict(font=dict(size=28, color=clr), valueformat=".3f"),
                    gauge=dict(
                        axis=dict(range=[0, 1]),
                        bar=dict(color=clr, thickness=0.3),
                        bgcolor="#151d2e",
                        steps=[
                            dict(range=[0, .30],  color="rgba(34,201,132,.08)"),
                            dict(range=[.30, .55], color="rgba(245,200,66,.08)"),
                            dict(range=[.55, .75], color="rgba(255,140,66,.08)"),
                            dict(range=[.75, 1],   color="rgba(255,59,92,.08)"),
                        ],
                        # ──  CHANGE: draw adaptive threshold line on gauge ──
                        threshold=dict(
                            line=dict(color="#00d4ff", width=3),
                            thickness=0.85,
                            value=adaptive_thr,
                        ),
                        # ────────────────────────────────────────────────────
                    ),
                    title=dict(text=f"Risk Score  (threshold: {adaptive_thr:.3f})",
                               font=dict(size=11, color="#64748b")),
                ))
                fig_g.update_layout(
                    paper_bgcolor="#0f1520", font=dict(color="#94a3b8"),
                    margin=dict(l=10, r=10, t=30, b=10), height=200,
                )
                st.plotly_chart(fig_g, use_container_width=True)

                for lbl, col_key, bar_clr in [
                    ("CVE/Static", "static", "#ff3b5c"),
                    ("Behaviour",  "beh",    "#ff8c42"),
                    ("GNN",        "gnn",    "#00d4ff"),
                    ("XGBoost",    "xgb",    "#f5c842"),
                ]:
                    val = float(d.get(col_key, 0))
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px;'>"
                        f"<span style='width:75px;font-size:11px;color:#64748b;'>{lbl}</span>"
                        f"<div style='flex:1;background:rgba(255,255,255,.06);border-radius:2px;height:6px;'>"
                        f"<div style='width:{int(val*100)}%;height:100%;background:{bar_clr};border-radius:2px;'></div></div>"
                        f"<span style='font-size:11px;color:{bar_clr};width:38px;text-align:right;'>{val:.3f}</span></div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("**Actions:**")
            ac1, ac2, ac3, ac4, ac5, ac6 = st.columns(6)
            with ac1:
                if st.button("🔌 Isolate", key="det_isolate",
                             disabled=not rule["can_isolate"], use_container_width=True):
                    ovr = st.session_state.get("status_overrides", {})
                    ovr[str(d["id"])] = "Isolated"
                    save_device_status_overrides(ovr)
                    st.session_state["status_overrides"] = ovr
                    st.cache_data.clear()
                    st.error(f" Isolation triggered for {nm}")
            with ac2:
                if st.button("✓ Patched", key="det_patch", use_container_width=True):
                    ovr = st.session_state.get("status_overrides", {})
                    ovr[str(d["id"])] = "Patched"
                    save_device_status_overrides(ovr)
                    st.session_state["status_overrides"] = ovr
                    st.cache_data.clear()
                    st.success(f" Patch recorded for {nm}")
            with ac3:
                if st.button("🔧 Maintain", key="det_maint", use_container_width=True):
                    ovr = st.session_state.get("status_overrides", {})
                    ovr[str(d["id"])] = "Maintenance"
                    save_device_status_overrides(ovr)
                    st.session_state["status_overrides"] = ovr
                    st.cache_data.clear()
                    st.warning(f"🔧 Maintenance scheduled for {nm}")
            with ac4:
                if st.button("📧 Alert", key="det_alert", use_container_width=True):
                    if st.session_state.get("email_enabled"):
                        ok, msg = send_email_alert(
                            st.session_state.get("email_sender", ALERT_EMAIL_TO),
                            dict(d), sc, urg)
                        if ok:
                          st.success(msg)
                        else:
                          st.error(msg)
                    else:
                        st.warning("Email not configured. Go to ⚙️ Settings → Email tab.")
            with ac5:
                report = generate_device_report(d, sc, urg)
                st.download_button(
                    "📄 Report", report,
                    f"report_{sel}_{get_ist_now().strftime('%Y%m%d')}.txt",
                    "text/plain", key="det_report", use_container_width=True)
            with ac6:
                if st.button("✕ Close", key="det_close", use_container_width=True):
                    st.session_state.sel = None
                    st.rerun()

            if not rule["can_isolate"]:
                st.info(f" Isolation disabled for this device type. {rule['note']}")

    st.stop()


# PAGE: ANALYTICS

elif nav_choice == "📈 Analytics":
    st.markdown("### 📈 Risk Analytics")

    an1, an2 = st.columns(2)
    with an1:
        st.markdown("####  Average Risk by Ward")
        wa = disp.groupby("ward")["score"].mean().sort_values(ascending=False).reset_index()
        wa.columns = ["ward", "avg"]
        wa["color"] = wa["avg"].apply(
            lambda s: URGENCY_COLOR.get(
                "CRITICAL" if s>=0.75 else "HIGH" if s>=0.55 else "ELEVATED" if s>=0.30 else "MONITOR", "#888"))
        fig_a = go.Figure(go.Bar(
            x=wa["avg"], y=wa["ward"], orientation="h",
            marker_color=wa["color"], marker_line_width=0,
            text=[f"{v:.3f}" for v in wa["avg"]], textposition="outside"))
        fig_a.update_layout(
            plot_bgcolor="#0f1520", paper_bgcolor="#0f1520",
            font=dict(color="#94a3b8"), height=320,
            xaxis=dict(range=[0,1.2], gridcolor="rgba(255,255,255,.04)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=60, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_a, use_container_width=True)

    with an2:
        st.markdown("#### 📉 Score Distribution")
        fig_h = go.Figure(go.Histogram(
            x=disp["score"], nbinsx=30, marker_color="#00d4ff", marker_line_width=0,
            hovertemplate="Score: %{x:.2f}<br>Count: %{y}<extra></extra>"))
        fig_h.update_layout(
            plot_bgcolor="#0f1520", paper_bgcolor="#0f1520",
            font=dict(color="#94a3b8"), height=320,
            xaxis=dict(title="Risk Score", gridcolor="rgba(255,255,255,.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,.04)"),
            margin=dict(l=0, r=0, t=10, b=40), showlegend=False)
        st.plotly_chart(fig_h, use_container_width=True)

    # ──  CHANGE: adaptive threshold distribution chart ──────────────────────
    st.markdown("---")
    st.markdown("####  Adaptive Threshold Distribution by Ward")
    if "adaptive_threshold" in disp.columns:
        thr_by_ward = (disp.groupby("ward")["adaptive_threshold"]
                       .mean().sort_values().reset_index())
        thr_by_ward.columns = ["ward", "avg_threshold"]
        fig_thr = go.Figure(go.Bar(
            x=thr_by_ward["avg_threshold"], y=thr_by_ward["ward"],
            orientation="h",
            marker_color="#00d4ff", marker_line_width=0,
            text=[f"{v:.3f}" for v in thr_by_ward["avg_threshold"]],
            textposition="outside",
        ))
        fig_thr.add_vline(x=EVAL_THRESHOLD, line_dash="dash", line_color="#f5c842",
                          annotation_text=f"base={EVAL_THRESHOLD}",
                          annotation_font=dict(color="#f5c842", size=9))
        fig_thr.update_layout(
            plot_bgcolor="#0f1520", paper_bgcolor="#0f1520",
            font=dict(color="#94a3b8"), height=300,
            xaxis=dict(range=[0, 0.7], title="Adaptive Threshold",
                       gridcolor="rgba(255,255,255,.04)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=80, t=10, b=10), showlegend=False,
        )
        st.plotly_chart(fig_thr, use_container_width=True)
        st.caption(
            "ICU/life-support wards have lower thresholds (more sensitive). "
            "Admin/storage wards have higher thresholds (fewer false alarms). "
            "Yellow dashed line = base threshold (0.20)."
        )
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown("####  Static vs Behaviour Risk")
    if "static" in disp.columns and "beh" in disp.columns:
        fig_sc = go.Figure(go.Scatter(
            x=disp["static"], y=disp["beh"], mode="markers",
            marker=dict(color=[URGENCY_COLOR.get(u, "#888") for u in disp["urgency"]],
                        size=7, opacity=0.75),
            text=disp["name"].astype(str),
            hovertemplate="<b>%{text}</b><br>Static: %{x:.3f}<br>Beh: %{y:.3f}<extra></extra>",
        ))
        fig_sc.update_layout(
            plot_bgcolor="#0f1520", paper_bgcolor="#0f1520",
            font=dict(color="#94a3b8"), height=280,
            xaxis=dict(title="CVE/Static Risk", gridcolor="rgba(255,255,255,.04)"),
            yaxis=dict(title="Behaviour Risk",  gridcolor="rgba(255,255,255,.04)"),
            margin=dict(l=0, r=0, t=10, b=40))
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")
    st.markdown("####  Stats by Device Type")
    if len(disp) > 0:
        dt_analysis = (disp.groupby("type")["score"]
                       .agg(["mean","max","min","count"]).round(3)
                       .sort_values("mean", ascending=False))
        dt_analysis.columns = ["Avg Score", "Max Score", "Min Score", "Count"]
        st.dataframe(dt_analysis, use_container_width=True)

    st.stop()


# PAGE: ALERTS

elif nav_choice == " Alerts":
    st.markdown("###  Alert Management")

    active_urgencies = get_active_urgencies()
    if active_urgencies:
        pills = " · ".join(
            f"<span style='color:{URGENCY_COLOR[u]};font-weight:700;'>{u}</span>"
            for u in active_urgencies)
        st.markdown(
            f"<div style='background:#0f1520;border:1px solid rgba(255,255,255,.07);"
            f"border-radius:8px;padding:10px 14px;font-size:12px;margin-bottom:12px;'>"
            f" <b>Active alert filters:</b> {pills}</div>",
            unsafe_allow_html=True)
    else:
        st.warning(" All urgency levels are disabled. Go to ⚙️ Settings → Alert Filters to enable them.")

    with st.expander("📤 Send Manual Alert", expanded=False):
        ma1, ma2 = st.columns(2)
        with ma1:
            alert_device = st.selectbox("Select device to alert on",
                                        df["name"].astype(str).tolist(),
                                        key="manual_alert_device")
            alert_email  = st.text_input("Recipient email", key="manual_alert_email",
                                         placeholder="nurse@hospital.org")
            alert_sms    = st.text_input("Recipient SMS number", key="manual_alert_sms",
                                         placeholder="+91XXXXXXXXXX")
        with ma2:
            if st.button("📧 Send Email Alert", key="manual_send_email"):
                row = df[df["name"].astype(str) == alert_device]
                if not row.empty and alert_email:
                    d_r = row.iloc[0]
                    ok, msg = send_email_alert(alert_email, dict(d_r),
                                               float(d_r.get("score", 0)),
                                               str(d_r.get("urgency","MONITOR")).upper())
                    st.success(msg) if ok else st.error(msg)
                elif not alert_email:
                    st.warning("Enter a recipient email address.")
            if st.button("📱 Send SMS Alert", key="manual_send_sms"):
                row = df[df["name"].astype(str) == alert_device]
                if not row.empty:
                    if alert_sms:
                        d_r  = row.iloc[0]
                        rule = get_device_rule(d_r.get("type", ""))
                        urg  = str(d_r.get("urgency","MONITOR")).upper()
                        act  = rule["action_critical"] if urg=="CRITICAL" else rule["action_high"]
                        body = (f"ClinGuard ALERT [{urg}]\n"
                                f"Device: {d_r.get('name','?')}\nWard: {d_r.get('ward','?')}\n"
                                f"Score: {float(d_r.get('score',0)):.3f}\nAction: {act[:80]}")
                        ok, msg = _send_raw_sms(alert_sms, body)
                        st.success(msg) if ok else st.error(msg)
                    else:
                        st.warning("Enter SMS number")
                else:
                    st.warning("Select a device")

    st.markdown("---")

    alerts_history = load_alerts_history()
    if alerts_history:
        alerts_df = pd.DataFrame(alerts_history)
        st.metric("Total Alerts Logged", len(alerts_df))

        tab_hist, tab_stats = st.tabs(["📜 History", " Statistics"])
        with tab_hist:
            display_cols = [c for c in
                            ["timestamp","device_name","urgency","score","channel","status"]
                            if c in alerts_df.columns]
            st.dataframe(alerts_df[display_cols].sort_values("timestamp", ascending=False).head(50),
                         use_container_width=True)
            st.download_button("📥 Export Alerts CSV", export_to_csv(alerts_df),
                               "alerts_history.csv", "text/csv", key="dl_alerts")

        with tab_stats:
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("🔴 CRITICAL", int((alerts_df["urgency"]=="CRITICAL").sum()))
            s2.metric("🟠 HIGH",     int((alerts_df["urgency"]=="HIGH").sum()))
            s3.metric("🟡 ELEVATED", int((alerts_df["urgency"]=="ELEVATED").sum()))
            today_count = int(
                (pd.to_datetime(alerts_df["timestamp"], format='ISO8601').dt.date
                 == get_ist_now().date()).sum()
            ) if len(alerts_df) > 0 else 0
            s4.metric(" Today", today_count)

        st.markdown("---")
        if st.button("🗑️ Clear Alert History", key="clr_hist"):
            if clear_alerts_history():
                st.success(" Alert history cleared")
                st.rerun()
    else:
        st.info(" No alerts recorded yet.")

    st.markdown("---")
    st.markdown("####  Live Monitored Risk Devices")
    live_hr = disp[disp["urgency"].isin(active_urgencies)].head(10)
    if live_hr.empty:
        if active_urgencies:
            st.success(f" No {' / '.join(active_urgencies)} devices in current view")
        else:
            st.warning(" No alert filters are active.")
    else:
        for _, row in live_hr.iterrows():
            urg = str(row.get("urgency", "HIGH")).upper()
            clr = URGENCY_COLOR.get(urg, "#888")
            nm  = str(row.get("name", row.get("id", "—")))
            sc  = float(row.get("score", 0))
            sv  = float(row.get("static", 0))
            bv  = float(row.get("beh", 0))
            msg = (f"CVE risk {sv:.3f} — verify patch status."
                   if sv >= bv else f"Behaviour anomaly {bv:.3f} — check network activity.")
            st.markdown(f"""
            <div style='background:#0f1520;border-left:3px solid {clr};
                        border-radius:0 8px 8px 0;padding:12px;margin-bottom:8px;'>
              <div style='display:flex;justify-content:space-between;'>
                <span style='font-size:13px;font-weight:600;color:#e2e8f0;'>{nm}</span>
                <span style='font-size:11px;color:{clr};font-weight:700;'>{urg} · {sc:.3f}</span>
              </div>
              <div style='font-size:12px;color:#94a3b8;margin-top:4px;'>{msg}</div>
            </div>""", unsafe_allow_html=True)

    st.stop()


# PAGE: INVENTORY

elif nav_choice == "🔧 Inventory":
    st.markdown("### 🔧 Device Inventory")

    ic1, ic2, ic3 = st.columns([2, 1, 1])
    with ic1:
        inv_search = st.text_input("Search inventory…", key="inv_srch",
                                   placeholder="Name, ID, type…")
    with ic2:
        inv_sort = st.selectbox("Sort by", ["Risk ↓","Ward","Type","Name"],
                                key="inv_sort_sel")
    with ic3:
        st.download_button(
            "📥 Export CSV", export_to_csv(disp),
            f"inventory_{get_ist_now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv", key="inv_export")

    inv_disp = disp.copy()
    if inv_search.strip():
        q = inv_search.strip().lower()
        inv_disp = inv_disp[
            inv_disp["name"].astype(str).str.lower().str.contains(q, na=False) |
            inv_disp["id"].astype(str).str.lower().str.contains(q, na=False)   |
            inv_disp["type"].astype(str).str.lower().str.contains(q, na=False)
        ]

    sort_map = {"Risk ↓": "score", "Ward": "ward", "Type": "type", "Name": "name"}
    inv_disp = inv_disp.sort_values(
        sort_map.get(inv_sort, "score"), ascending=(inv_sort != "Risk ↓"))

    # ──  CHANGE: include adaptive_threshold in inventory table ──────────────
    display_cols = [c for c in
                    ["name","id","type","ward","score","adaptive_threshold",
                     "pred_label","urgency","status","manufacturer","ip"]
                    if c in inv_disp.columns]
    # ─────────────────────────────────────────────────────────────────────────
    st.dataframe(inv_disp[display_cols], use_container_width=True, height=500)

    st.markdown("---")
    st.markdown("#### 🔄 Bulk Status Update")
    bs1, bs2, bs3 = st.columns([2, 1, 1])
    with bs1:
        bulk_urgency = st.selectbox("Update all devices with urgency",
                                    URGENCY_LEVELS, key="bulk_urg_sel")
    with bs2:
        bulk_status = st.selectbox("Set status to",
                                   DEVICE_STATUS_OPTIONS, key="bulk_status_sel")
    with bs3:
        if st.button(" Apply", key="bulk_apply", use_container_width=True):
            targets = df[df["urgency"] == bulk_urgency]["id"].astype(str).tolist()
            ovr     = st.session_state.get("status_overrides", {})
            for did in targets:
                ovr[did] = bulk_status
            save_device_status_overrides(ovr)
            st.session_state["status_overrides"] = ovr
            st.cache_data.clear()
            st.success(f" Updated {len(targets)} {bulk_urgency} devices → {bulk_status}")
            st.rerun()

    st.markdown("---")
    st.markdown("####  Compliance Summary")
    comp = st.columns(len(DEVICE_STATUS_OPTIONS))
    for col_obj, stat_name in zip(comp, DEVICE_STATUS_OPTIONS):
        col_obj.metric(stat_name, int((disp["status"] == stat_name).sum()))

    st.stop()


# PAGE: SETTINGS

elif nav_choice == "⚙️ Settings":
    st.markdown("### ⚙️ Configuration")

    tab_sms, tab_email, tab_alerts, tab_data = st.tabs(
        ["📱 SMS (Twilio)", "📧 Email (Gmail)", " Alert Filters", " Data & Info"])

    with tab_sms:
        st.markdown("#### Twilio SMS Configuration")
        st.caption("Requires a Twilio account. Trial accounts must verify recipient numbers.")
        new_sid   = st.text_input("Account SID",          value=st.session_state.twilio_sid,
                                  type="password", key="cfg_sid")
        new_token = st.text_input("Auth Token",            value=st.session_state.twilio_token,
                                  type="password", key="cfg_token")
        new_from  = st.text_input("From Number (+E.164)",  value=st.session_state.twilio_from,
                                  key="cfg_from", placeholder="+1XXXXXXXXXX")
        new_to    = st.text_input("To Number (+E.164)",    value=st.session_state.twilio_to,
                                  key="cfg_to", placeholder="+91XXXXXXXXXX")
        sc1, sc2 = st.columns(2)
        with sc1:
            if st.button("💾 Save SMS Config", key="save_sms"):
                if all([new_sid, new_token, new_from, new_to]):
                    st.session_state.update({
                        "twilio_sid": new_sid, "twilio_token": new_token,
                        "twilio_from": new_from, "twilio_to": new_to, "twilio_enabled": True,
                    })
                    _save_json("twilio_config.json", {
                        "account_sid": new_sid, "auth_token": new_token,
                        "from_number": new_from, "to_number": new_to, "enabled": True,
                    })
                    st.success(" SMS settings saved!")
                else:
                    st.error("All four SMS fields are required.")
        with sc2:
            if st.button(" Send Test SMS", key="test_sms"):
                ok, msg = _send_raw_sms(
                    st.session_state.twilio_to,
                    f"Test from Clinical Guardian v8.0 — {get_ist_now().strftime('%H:%M IST')}",
                )
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    with tab_email:
        st.markdown("#### Gmail App-Password Configuration")
        st.caption("Enable 2FA → Google Account → Security → App Passwords → 16-char password.")
        new_email = st.text_input("Gmail address",
                                  value=st.session_state.email_sender, key="cfg_email")
        new_pwd   = st.text_input("App Password (16 chars)",
                                  value=st.session_state.email_password,
                                  type="password", key="cfg_pwd")
        ec1, ec2 = st.columns(2)
        with ec1:
            if st.button("💾 Save Email Config", key="save_email"):
                if new_email and new_pwd:
                    st.session_state.update({
                        "email_sender": new_email, "email_password": new_pwd,
                        "email_enabled": True,
                    })
                    _save_json("email_config.json", {
                        "sender": new_email, "password": new_pwd, "enabled": True,
                    })
                    st.success(" Email settings saved!")
                else:
                    st.error("Both Gmail address and App Password are required.")
        with ec2:
            test_addr = st.text_input("Test recipient address", key="test_addr",
                                      placeholder="colleague@hospital.org")
            if st.button(" Send Test Email", key="test_email_btn"):
                if test_addr:
                    ok, msg = send_email_alert(
                        test_addr,
                        {"name":"TEST DEVICE","id":"T-001","type":"test","ward":"Test Ward"},
                        0.5, "MONITOR",
                    )
                    st.success(msg) if ok else st.error(msg)
                else:
                    st.warning("Enter an email address")

    with tab_alerts:
        st.markdown("####  Alert Logging & Auto-Alert Filters")
        st.caption(
            "Controls which urgency levels are **logged to history** AND **included in "
            "auto-alerts** (email / SMS). Changes take effect on the next page refresh."
        )

        active_now = get_active_urgencies()
        if active_now:
            pills = " · ".join(
                f"<span style='color:{URGENCY_COLOR[u]};font-weight:700;'>{u}</span>"
                for u in active_now)
            st.markdown(
                f"<div style='background:#0f1520;border:1px solid rgba(255,255,255,.07);"
                f"border-radius:8px;padding:10px 14px;font-size:12px;margin-bottom:12px;'>"
                f"Currently active: {pills}</div>",
                unsafe_allow_html=True)
        else:
            st.warning(" No urgency levels are currently active — nothing will be alerted or logged.")

        cb_crit = st.checkbox("🔴 CRITICAL — Isolate now",
                              value=st.session_state.get("log_critical", True),  key="cb_crit")
        cb_high = st.checkbox("🟠 HIGH — Urgent patch needed",
                              value=st.session_state.get("log_high",     True),  key="cb_high")
        cb_elev = st.checkbox("🟡 ELEVATED — Schedule remediation",
                              value=st.session_state.get("log_elevated", False), key="cb_elev")
        cb_mon  = st.checkbox("🟢 MONITOR — Routine observation",
                              value=st.session_state.get("log_monitor",  False), key="cb_mon")

        if st.button("💾 Save Alert Settings", key="save_alerts"):
            st.session_state.update({
                "log_critical": cb_crit, "log_high": cb_high,
                "log_elevated": cb_elev, "log_monitor": cb_mon,
            })
            _save_json("alert_config.json", {
                "log_critical": cb_crit, "log_high": cb_high,
                "log_elevated": cb_elev, "log_monitor": cb_mon,
            })
            st.success(" Alert settings saved! Click REFRESH DATA to re-run auto-alerts with new filters.")

        st.markdown("---")
        hist = load_alerts_history()
        st.metric("Total logged alerts", len(hist))
        if hist and st.button("🗑️ Clear All Alert History", key="clr_alerts_cfg"):
            if clear_alerts_history():
                st.success(" Cleared!")
                st.rerun()

    with tab_data:
        st.markdown("#### Data Management")
        dm1, dm2 = st.columns(2)
        with dm1:
            st.download_button(
                "📥 Export Current View as CSV", export_to_csv(disp),
                f"export_{get_ist_now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv", key="dl_full", use_container_width=True)
        with dm2:
            if st.button("🗑️ Clear All Status Overrides", key="clr_overrides",
                         use_container_width=True):
                save_device_status_overrides({})
                st.session_state["status_overrides"] = {}
                st.cache_data.clear()
                st.success(" All status overrides cleared!")
                st.rerun()

        st.markdown("---")

        # ── PDF Download ──────────────────────────────────────────────────────
        st.markdown("#### 📄 Generate PDF Reports")
        critical_count = len(df[df["urgency"] == "CRITICAL"])
        if not FPDF_AVAILABLE:
            st.info(" PDF library not available. Install: pip install fpdf2")
        elif critical_count == 0:
            st.info(" No CRITICAL devices to report on")
        else:
            if st.button(" Generate Critical Devices PDF", key="gen_pdf",
                         use_container_width=True):
                with st.spinner("Generating PDF…"):
                    pdf_buffer = generate_ward_wise_critical_report(df)
                    if pdf_buffer:
                        st.download_button(
                            "📥 Download Critical Report PDF",
                            data=pdf_buffer.getvalue(),
                            file_name=f"Critical_Ward_Report_{get_ist_now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            key="dl_pdf",
                            use_container_width=True,
                        )
                    else:
                        st.error("Failed to generate PDF report.")

        st.markdown("---")

        # ── PDF Email ─────────────────────────────────────────────────────────
        st.markdown("#### 📧 Bulk PDF Email Alert")
        st.caption("Sends a ward-wise CRITICAL device PDF report as an email attachment.")
        pdf_email = st.text_input("Send PDF report to email:", key="pdf_email_input",
                                  placeholder="admin@hospital.org")
        if st.button("📧 Send PDF Email", key="send_pdf_email", use_container_width=True):
            if pdf_email:
                with st.spinner("Generating and sending PDF email…"):
                    ok, msg = send_bulk_email_alert_PDF(pdf_email, df)
                    if ok:
                      st.success(msg)
                    else:
                      st.error(msg)
            else:
                st.warning("Enter an email address")

        st.markdown("---")
        st.markdown("####  About Clinical Guardian v8.0")

        # ──  CHANGE: show adaptive threshold config info in About section ───
        st.markdown(f"""
**Clinical Guardian v8.0** — Medical Device Risk Monitoring Platform

**Risk Engine:**  GNN + XGBoost hybrid fusion
**Datasets:**  Hybrid Analysis (NVD/CVE) · MIMIC-III · FDA MAUDE
**Alerts:**  Auto email (Gmail) + SMS (Twilio) · Deduplication · History log
**PDF Reports:**  Ward-wise CRITICAL device report (fpdf2 + helvetica, emoji-safe)
**Timezone:**  IST (Asia/Kolkata)
**Status Persistence:**  Overrides survive page reload

**Adaptive Threshold Engine:**
- Base threshold: `{EVAL_THRESHOLD}`
- Hybrid dataset: uses model-stored per-device thresholds (exact reproduction)
- FDA / MIMIC datasets: thresholds computed dynamically using ward × impact modifiers
- ICU / LIFE_SUPPORT: threshold ≈ `{EVAL_THRESHOLD * 0.70:.3f}` (most sensitive)
- ADMIN / ADMINISTRATIVE: threshold ≈ `{EVAL_THRESHOLD * 1.40:.3f}` (least sensitive)

**Active dataset:** `{st.session_state.get('selected_dataset','—')}`
**Loaded devices:** {len(df):,}
**Time:** {get_ist_now().strftime('%d %b %Y %H:%M:%S IST')}

---
 *Decision support tool only — not a substitute for clinical judgment.*
        """)
        # ─────────────────────────────────────────────────────────────────────

    st.stop()


# FOOTER

st.markdown("---")
st.caption(
    f" Clinical Guardian v8.0  ·  "
    f"Dataset: {st.session_state.get('selected_dataset','—')}  ·  "
    f"{len(df):,} devices  ·  "
    f"{get_ist_now().strftime('%d %b %Y %H:%M:%S IST')}  ·  "
    f"GNN + XGBoost Fusion + Adaptive Thresholds  ·  For Hospital Use Only"
)
