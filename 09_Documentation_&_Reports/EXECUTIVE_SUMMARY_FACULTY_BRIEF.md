# 📑 AGROVISION AI: EXECUTIVE SUMMARY & FACULTY EVALUATION BRIEF

### Project Overview
**AgroVision AI** is an end-to-end Deep Learning Computer Vision platform engineered to diagnose foliar pathogens across **5 high-value agricultural crops** (**Potato, Tomato, Apple, Corn, Grape**) spanning **15 pathological conditions**. 

- **GitHub Repository:** https://github.com/REHANKHANN20/agrovision-plant-disease-detection
- **24/7 Live Permanent Web App:** https://agrovision--ai.streamlit.app

---

### 👥 5-Member Team Allocation Summary
- **Member 1 (Project Lead & Principal CV Architect - 45%):** End-to-end architecture design, Custom Dual-Pool CNN, MobileNetV2 transfer learning, optimization regimes.
- **Member 2 (Senior Data Engineer & Forensic Audit Specialist - 25%):** Multi-source data ingestion (PlantVillage + GVLiD Mendeley), SHA-256 deduplication, corrupt file purging.
- **Member 3 (CV Preprocessing & Feature Transformation Lead - 10%):** EXIF orientation correction, high-fidelity LANCZOS resizing (224x224x3), stratified 80/10/10 splitting, NPZ packaging.
- **Member 4 (Model Evaluation & Empirical Benchmark Analyst - 10%):** Confusion matrices, precision/recall analysis, statistical validation, laboratory vs. real-world field benchmarks.
- **Member 5 (Full-Stack MLOps & Agronomic Intelligence Lead - 10%):** Production Streamlit web application, agronomic treatment knowledge base, PDF report generator, 24/7 cloud deployment.

---

### 📊 Empirical Performance Summary Across 5 Crops

| Host Crop | Architecture | Parameters | Held-Out Test Accuracy | Real-World Field Accuracy | Key Engineering Highlight |
|---|---|:---:|:---:|:---:|---|
| 🥔 **Potato** | Custom Dual-Pool CNN | 11.1 M | **98.21%** | **92.40%** | Parallel MaxPooling + AvgPooling preserves lesion boundaries. |
| 🍅 **Tomato** | MobileNetV2 Fine-Tuned | 2.8 M | **97.78%** | **60.00%** | Ultra-lightweight footprint (11.1 MB), ideal for edge inference. |
| 🍎 **Apple** | MobileNetV2 Fine-Tuned | 2.9 M | **97.75%** | **56.67%** | High sensitivity on rust pycnidia and scabby foliar textures. |
| 🌽 **Corn** | MobileNetV2 Fine-Tuned | 2.9 M | **97.41%** | **83.33%** | Outstanding zero-shot generalization on wild Indian maize fields. |
| 🍇 **Grape** | MobileNetV2 Dual-Dense | 2.9 M | **95.56%** | **76.67%** | Trained on Mendeley GVLiD field data; 100% precision on Black Rot. |

---

### 🏆 Key Competitive Distinctions (vs. Standard Projects)
1. **Full 5-Crop Active Coverage:** Unlike standard prototypes supporting only 2-3 crops, all 5 crop pipelines are active with verified weights.
2. **Laboratory vs. Field Transparency:** Transparently reports performance drop when transitioning from controlled lab data to wild agricultural fields.
3. **Agronomic Clinical Prescriptions:** Beyond prediction, provides exact chemical dosages (Mancozeb, Azoxystrobin) and organic remedies (Neem, Trichoderma).
4. **Production 24/7 Web App:** Deployed on Streamlit Community Cloud with 42 ms latency and downloadable PDF pathology reports.
