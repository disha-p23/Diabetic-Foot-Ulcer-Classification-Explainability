#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
DFU Detect — Streamlit App
Run: streamlit run app.py
"""

import re
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from groq import Groq

import numpy as np
import cv2
import os
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# CONFIG  ← update these 2 lines only
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Using a relative path makes your code work on ANY computer once pushed to GitHub
MODEL_PATH = "dfu_model_v1.pth" 

# This specifically loads your "dfu.env" file
load_dotenv("dfu.env") 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Safety check: This will tell you EXACTLY what is wrong in the Streamlit UI
if not GROQ_API_KEY:
    st.error(" API Key Not Found! Check if 'dfu.env' exists in the folder and contains GROQ_API_KEY=gsk_...")

CLASSES = ["healthy", "ulcer"]
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="DFU Detect", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');
    .block-container { padding-top: 2rem; }
    .pred-box { padding: 16px 20px; border-radius: 10px; margin-bottom: 12px; }
    .severity-badge {
        display: inline-block; padding: 4px 14px; border-radius: 6px;
        font-family: 'DM Mono', monospace; font-weight: 800;
        font-size: 13px; letter-spacing: 0.08em;
    }
    .section-title {
        font-family: 'DM Mono', monospace; font-size: 11px;
        letter-spacing: 0.1em; text-transform: uppercase;
        color: #94a3b8; border-bottom: 1px solid #1e293b;
        padding-bottom: 4px; margin-bottom: 8px;
    }
    div[data-testid="stImage"] img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL (cached so it loads only once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    model.eval()
    return model.to(DEVICE)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def preprocess(img_rgb):
    return transform(img_rgb).unsqueeze(0).to(DEVICE)

def overlay_heatmap(img_rgb, heatmap, alpha=0.45):
    h, w = img_rgb.shape[:2]
    hm   = cv2.resize(heatmap, (w, h))
    hm   = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm   = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return (hm * alpha + img_rgb * (1 - alpha)).astype(np.uint8)

def run_model(img_rgb):
    model = load_model()
    inp   = preprocess(img_rgb)

    with torch.no_grad():
        out   = model(inp)
        probs = torch.softmax(out, dim=1)[0]
        pred  = int(probs.argmax())
        conf  = float(probs[pred])

    cam_obj      = GradCAM(model=model,        target_layers=[model.features[-1]])
    cam_plus_obj = GradCAMPlusPlus(model=model, target_layers=[model.features[-1]])

    gc      = overlay_heatmap(img_rgb, cam_obj(input_tensor=inp)[0])
    gc_plus = overlay_heatmap(img_rgb, cam_plus_obj(input_tensor=inp)[0])

    return {
        "prediction":   CLASSES[pred],
        "confidence":   round(conf * 100, 2),
        "healthy_prob": round(float(probs[0]) * 100, 2),
        "ulcer_prob":   round(float(probs[1]) * 100, 2),
        "gradcam":      gc,
        "gradcam_plus": gc_plus,
    }

def get_severity_report(prediction, confidence):
    client = Groq(api_key=GROQ_API_KEY)
    resp   = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=900,
        messages=[{
            "role": "user",
            "content": f"""You are a clinical AI assistant for diabetic foot ulcer assessment.

The deep learning model classified the foot image as: {prediction.upper()} ({confidence:.1f}% confidence)

Provide a structured clinical severity report with exactly these sections:

**VISUAL OBSERVATIONS**
Describe likely wound appearance based on the classification.

**SEVERITY LEVEL**
Rate as: None / Mild / Moderate / Severe (Wagner DFU Grade 0-5)

**AFFECTED REGIONS**
Common regions affected for this classification.

**CLINICAL FLAGS**
Urgent findings to watch for. Write "None identified" if none.

**RECOMMENDED NEXT STEPS**
What a clinician should consider.

Note: This is clinical decision support only, not a diagnosis."""
        }]
    )
    return resp.choices[0].message.content

def severity_color(text):
    t = (text or "").lower()
    if "severe"   in t: return "#ff5252", "#3b0a0a"
    if "moderate" in t: return "#ff9100", "#2a1500"
    if "mild"     in t: return "#b2ff59", "#1a2500"
    return "#80deea", "#0a1a1a"

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.markdown("# DFU**detect**")
st.caption("Diabetic Foot Ulcer · AI-Assisted Assessment")
st.markdown("---")

uploaded = st.file_uploader("Upload a foot image", type=["jpg", "jpeg", "png"])

if uploaded:
    img_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_rgb   = cv2.cvtColor(cv2.imdecode(img_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    with st.spinner("Running model + generating heatmaps…"):
        res = run_model(img_rgb)

    pred       = res["prediction"]
    conf       = res["confidence"]
    pred_color = "#f87171" if pred == "ulcer" else "#4ade80"

    # ── Row 1: prediction + images ──
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        st.markdown(f"""
        <div class="pred-box" style="border: 1px solid {pred_color}44; background: #080d14;">
            <div style="font-size:10px; color:#475569; font-family:'DM Mono',monospace; margin-bottom:4px;">PREDICTION</div>
            <div style="font-size:30px; font-weight:900; color:{pred_color}; font-family:'DM Mono',monospace;">{pred.upper()}</div>
            <div style="font-size:12px; color:#475569;">{conf}% confidence</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(res["ulcer_prob"]   / 100, text=f"Ulcer: {res['ulcer_prob']}%")
        st.progress(res["healthy_prob"] / 100, text=f"Healthy: {res['healthy_prob']}%")

    with c2:
        st.image(img_rgb, caption="Original", use_container_width=True)

    with c3:
        col_a, col_b = st.columns(2)
        col_a.image(res["gradcam"],      caption="GradCAM",   use_container_width=True)
        col_b.image(res["gradcam_plus"], caption="GradCAM++", use_container_width=True)

    # ── Row 2: AI severity report ──
    st.markdown("---")
    st.markdown("<div class='section-title'>AI Clinical Analysis · Powered by Groq / Llama 3.3</div>", unsafe_allow_html=True)

    if st.button("Generate Severity Report"):
        with st.spinner("Analyzing with Llama 3.3…"):
            try:
                report = get_severity_report(pred, conf)
                st.session_state["report"] = report
            except Exception as e:
                st.error(f"{e}")

    report = st.session_state.get("report")
    if report:
        match = re.search(r"\*\*SEVERITY LEVEL\*\*[\s\S]*?(None|Mild|Moderate|Severe)", report, re.I)
        level = match.group(1) if match else None
        if level:
            fg, bg = severity_color(level)
            st.markdown(
                f'<span class="severity-badge" style="background:{bg}; color:{fg}; border:1px solid {fg}44;">'
                f'{level.upper()}</span>', unsafe_allow_html=True
            )
            st.markdown("")
        st.markdown(report)


# In[ ]:




