# 🌿 AgroVision AI: Multi-Plant Foliar Disease Detection & Pathology Intelligence Suite

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-Yes-blue.svg?style=for-the-badge)]()

> **An enterprise-grade Deep Learning Computer Vision framework engineered for autonomous foliar disease identification across 5 agricultural crop species (Potato, Tomato, Apple, Corn, Grape), supporting 15 pathological conditions with laboratory held-out benchmarking, zero-shot real-world field validation, and clinical agronomic advisory reporting.**

---

## 📌 Executive Summary & Motivation
Crop diseases represent a persistent threat to global agricultural food security, responsible for an estimated **20% to 40% annual yield loss worldwide**. Conventional disease diagnosis relies on manual visual scouting by agronomists, which is labor-intensive, subjective, and inaccessible to smallholder farming communities.

While existing computer vision literature achieves high benchmark accuracy on laboratory datasets (e.g., PlantVillage), most models experience severe **domain-shift breakdown** when deployed in real-world agricultural conditions due to variable sunlight, cluttered soil/weed backgrounds, and complex lesion geometries.

**AgroVision AI addresses this gap through:**
1. **Dual-Source Multi-Host Pipelines:** Integrates controlled laboratory baselines with authentic, field-collected vineyard and crop imagery from **GVLiD (Mendeley Data DOI: 10.17632/wkymf8bhcg.5)** and real-world agricultural repositories.
2. **Specialized Hybrid Neural Architectures:** Combines custom dual-pathway CNNs (parallel MaxPooling and AveragePooling) for textured fungal blights with fine-tuned MobileNetV2 transfer learning for resource-constrained edge execution.
3. **Rigorous Dual-Suite Evaluation:** Models are assessed against both controlled held-out test splits and unconstrained wild field imagery.
4. **Actionable Agronomic Advisory:** Generates instant clinical diagnoses accompanied by organic biocontrols, chemical fungicide schedules (active ingredients and dosages), and downloadable pathology reports.

---

## 🌾 Supported Crops & Pathological Classification Matrix

The suite delivers active diagnostic intelligence across **5 major host species** spanning **15 pathological classes**:

| Host Crop | Botanical Taxonomy | Evaluated Classes | Model Architecture | Weight Footprint | Laboratory Held-Out | Real-World Field Generalization |
|---|---|---|---|---|---|---|
| 🥔 **Potato** | *Solanum tuberosum* | Healthy, Early Blight, Late Blight | Custom Dual-Pool CNN | 127.9 MB | **98.21%** | **92.40%** |
| 🍅 **Tomato** | *Solanum lycopersicum* | Healthy, Early Blight, Late Blight | MobileNetV2 Fine-Tuned | 11.1 MB | **97.78%** | **60.00%** |
| 🍎 **Apple** | *Malus domestica* | Healthy, Apple Scab, Cedar Apple Rust | MobileNetV2 Fine-Tuned | 25.3 MB | **97.75%** | **56.67%** |
| 🌽 **Corn** | *Zea mays* | Healthy, Common Rust, Northern Leaf Blight | MobileNetV2 Fine-Tuned | 20.8 MB | **97.41%** | **83.33%** |
| 🍇 **Grape** | *Vitis vinifera* | Healthy, Black Rot, Leaf Blight | MobileNetV2 Dual-Dense Head | 25.3 MB | **95.56%** | **76.67%** |

---

## 🏛️ End-to-End System Architecture

```
                                  [ RAW FIELD / LAB INPUT ]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
        [ Controlled PlantVillage Data ]               [ Authentic Field Imagery (GVLiD) ]
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                           [ STAGE 1: CRYPTOGRAPHIC AUDIT ]
                             • SHA-256 Hash Deduplication
                             • PIL/CV2 Header & Channel Integrity
                                              ▼
                           [ STAGE 2: ADAPTIVE PREPROCESSING ]
                             • EXIF Orientation Normalization
                             • High-Fidelity LANCZOS Interpolation (224x224x3)
                             • Structured 80/10/10 Stratified Split
                                              ▼
                           [ STAGE 3: DEEP LEARNING MODELING ]
                      ┌───────────────────────────────────────┐
                      │  Potato: Custom Dual-Pool ConvNet     │
                      │  Tomato/Apple/Corn/Grape: MobileNetV2 │
                      └───────────────────┬───────────────────┘
                                          ▼
                         [ STAGE 4: EMPIRICAL BENCHMARKING ]
                      ┌───────────────────┴───────────────────┐
                      ▼                                       ▼
           [ Laboratory Split (Held-Out) ]          [ Real-World Wild Field Test ]
           • 95.56% - 98.21% Accuracy               • Up to 92.4% Field Zero-Shot
                                          │
                                          ▼
                         [ STAGE 5: AGROVISION WEB ENGINE ]
                      ┌───────────────────────────────────────┐
                      │ • Streamlit Web Dashboard             │
                      │ • Dynamic Model Caching (@st.cache)   │
                      │ • Plotly Probability Spectrum         │
                      │ • Shannon Entropy Uncertainty Scoring │
                      │ • Organic & Chemical Prescriptions    │
                      │ • PDF & Markdown Pathology Reports    │
                      └───────────────────────────────────────┘
```

---

## 📂 Repository Organization

```
agrovision-plant-disease-detection/
├── 01_Problem_Statement_&_Specs/       # Formal technical requirements & project charter
├── 02_Data_Ingestion_&_Verification/   # Verification notebooks & SHA-256 deduplication manifests
├── 03_Data_Preprocessing/              # Stratified split scripts, preprocessing pipelines, dataset manifests
├── 04_Model_Architectures_&_Training/  # End-to-end model training notebooks & training history logs
├── 05_Model_Evaluation_&_Benchmarks/   # Confusion matrices, classification reports, error analyses
├── 06_Trained_Models/                  # Production model weights (.keras) & release catalog
├── 07_Real_World_Field_Testing/        # Dual-pool real-world evaluation notebooks
├── 08_Web_Application/                 # Production Streamlit Web Suite
│   ├── app.py                          # Master application entrypoint
│   ├── disease_knowledge.py            # Agronomic pathology encyclopedia (15 conditions)
│   ├── model_utils.py                  # Cached inference engine & EXIF preprocessor
│   ├── report_generator.py             # Diagnostic pathology report generator (PDF & Markdown)
│   ├── requirements.txt                # Python environment specifications
│   ├── run_web_application.ipynb       # 1-Click Google Colab cloud runner with Cloudflare tunnel
│   └── test_samples/                   # 15 Curated 1-click test samples (3 per crop)
├── requirements.txt                    # Project-level dependencies
├── .gitignore                          # Clean git ignore manifest
└── README.md                           # Master project documentation
```

---

## 🔬 Key Empirical Results & Scientific Insights

### 1. The Laboratory vs. Field Generalization Gap
Our experiments uncovered that laboratory-trained neural networks experience a marked drop in accuracy when exposed to unconstrained agricultural fields without domain adaptation:
- **Corn MobileNetV2**: Achieved **83.33% field zero-shot accuracy**, demonstrating the robust generalization of depthwise separable convolutions on parallel venation leaf structures.
- **Potato Custom CNN**: Maintained **92.40% accuracy** on natural field imagery, validating the advantage of combining **MaxPooling2D** (preserving sharp lesion contours) and **AveragePooling2D** (preserving overall leaf textural context).
- **Grape MobileNetV2**: Leveraged **Mendeley GVLiD vineyard imagery** to attain **100% precision on Black Rot** and **90% on Leaf Blight** under real Indian vineyard lighting conditions.

---

## 🚀 Quick Start Guide

### Option 1: Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/REHANKHANN20/agrovision-plant-disease-detection.git
   cd agrovision-plant-disease-detection/08_Web_Application
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the web application:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

### Option 2: 1-Click Google Colab Run
1. Open [`08_Web_Application/run_web_application.ipynb`](08_Web_Application/run_web_application.ipynb) in Google Colab.
2. Select **Runtime → Run all**.
3. Click the generated **Cloudflare Tunnel URL** to access the live web application on any device with zero setup.

---

## 📜 Citations & Dataset Acknowledgements
1. **PlantVillage Dataset:** Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.
2. **GVLiD Dataset (Grape Vineyard Leaves Dataset):** Mendeley Data, DOI: `10.17632/wkymf8bhcg.5`.
3. **MobileNetV2 Architecture:** Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.

---

## 👨‍💻 Author & Project Credits
- **Developer:** Rehan Khan
- **Project Domain:** Deep Learning, Computer Vision, Precision Agriculture
- **Portfolio Repository:** [agrovision-plant-disease-detection](https://github.com/REHANKHANN20/agrovision-plant-disease-detection)
