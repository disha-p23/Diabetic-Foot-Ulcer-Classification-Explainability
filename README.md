# Diabetic-Foot-Ulcer-Classification-Explainability

# DFU Detect: Explainable AI for Diabetic Foot Ulcer Assessment 🩺

**A high-performance Computer Vision pipeline that combines Deep Learning classification with LLM-driven clinical reasoning.**

---

##  Summary
This project addresses the "Black Box" problem in medical AI. Instead of providing just a diagnosis, **DFU Detect** provides a three-layered assessment:
1.  **Classification:** Binary detection (Healthy vs. Ulcer) using EfficientNet-B0.
2.  **Visual Evidence:** Spatial localization of pathology using Grad-CAM++.
3.  **Clinical Reasoning:** Structured medical reporting via Llama 3.3 (Groq API).

---

## 📸 System Screenshots

### 1. The Diagnostic Dashboard
> *The main interface showing the prediction, confidence scores, and the original input.*


https://github.com/user-attachments/assets/f0a73db3-082b-42de-a7ca-322f7b652196



### 2. Explainable AI (XAI) Heatmaps
> *Using Grad-CAM and Grad-CAM++ to visualize the model's focus areas, ensuring the AI is looking at the wound and not background noise.*
<img width="1128" height="321" alt="image" src="https://github.com/user-attachments/assets/af59b395-9c6a-45b4-a3a2-ebeb928ebf5b" />

---

##  Core Highlights
* **Model:** EfficientNet-B0 (Transfer Learning) optimized for medical imaging.
* **Interpretability:** Implementation of `pytorch-grad-cam` for diagnostic transparency.
* **LLM Integration:** Real-time inference using **Groq’s LPU technology** for sub-second report generation.
* **Deployment:** Fully functional Streamlit web application.

---

##  Tech Stack
**Python** | **PyTorch** | **Streamlit** | **OpenCV** | **Groq API** | **Grad-CAM**

---

*Note: For full installation steps and source code analysis, please refer to the `ui.py` file or contact me directly.*
