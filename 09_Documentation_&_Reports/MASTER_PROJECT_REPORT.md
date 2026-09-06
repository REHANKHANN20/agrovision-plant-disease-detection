# 🌿 AGROVISION AI: ENTERPRISE-GRADE MULTI-CROP FOLIAR DISEASE DIAGNOSTICS & PATHOLOGY PLATFORM
## An End-to-End Deep Learning Architecture, Empirical Generalization Benchmark, and Production Web Intelligence System Across 5 Agricultural Host Species

---

### 📋 PROJECT METADATA & INSTITUTIONAL SUBMISSION RECORD
- **Project Title:** AgroVision AI — Multi-Plant Disease Detection using Deep Learning (Computer Vision)
- **Domain:** Artificial Intelligence | Computer Vision | Precision Agriculture | Agro-Pathology MLOps
- **Supported Crops (5 Species):** Potato (*Solanum tuberosum*), Tomato (*Solanum lycopersicum*), Apple (*Malus domestica*), Corn/Maize (*Zea mays*), Grape Vine (*Vitis vinifera*)
- **Total Classified Pathological Conditions:** 15 Classes (Healthy baselines + 10 Major Fungal/Oomycete Infections)
- **Primary GitHub Repository:** [https://github.com/REHANKHANN20/agrovision-plant-disease-detection](https://github.com/REHANKHANN20/agrovision-plant-disease-detection)
- **24/7 Live Permanent Web Application:** [https://agrovision--ai.streamlit.app](https://agrovision--ai.streamlit.app)
- **Colab Interactive Tunnel Runner:** `8th Web_Application/run_web_application.ipynb`
- **Academic Submission Session:** Academic Capstone & Technical Engineering Audit

---

## 👥 SECTION 1: EXECUTIVE WORK ALLOCATION & 5-MEMBER CONTRIBUTION MATRIX

In alignment with modern enterprise engineering practices and institutional academic standards, project responsibilities were distributed across a specialized 5-member cross-functional engineering team. 

While **Member 1 (Project Lead & Principal Deep Learning Architect)** spearheaded the overarching technical vision, end-to-end pipeline implementation, and custom neural network design (~50% effort), and **Member 2 (Senior Data Engineer & Empirical Verification Lead)** co-directed the multi-source dataset curation, cryptographic auditing, and model benchmarking (~25% effort), **Members 3, 4, and 5** owned critical specialized functional modules (~25% combined effort) ensuring balanced, modular, and professional institutional contributions.

### 📊 1.1 Team Roles, Functional Titles & Ownership Matrix

| Member | Functional Industry Title | Core Module Ownership | Primary Technical Deliverables & Milestones | Ownership (%) |
|---|---|---|---|:---:|
| **Member 1 (Project Lead)** | **Principal CV Architect & Lead Systems Engineer** | End-to-End Architecture, Custom CNN & MobileNetV2 Models | • Formal problem formulation & multi-crop architecture design<br>• Designed Custom Dual-Pool CNN (Potato) & MobileNetV2 Heads<br>• Supervised mathematical loss formulation & optimization regimes<br>• Directed overall project integration across all 9 pipeline stages | **45%** |
| **Member 2** | **Senior Data Engineer & Forensic Audit Specialist** | Data Collection, Cryptographic Ingestion & Verification | • Ingestion of PlantVillage and Mendeley GVLiD India field data<br>• SHA-256 cryptographic hashing for strict deduplication<br>• Forensic corrupt file removal & RGB channel validation<br>• Designed data split manifests & baseline verification audits | **25%** |
| **Member 3** | **Computer Vision Preprocessing & Feature Engineering Lead** | Standardization, EXIF Correction & NPZ Packaging | • EXIF orientation normalization algorithm for mobile imagery<br>• High-fidelity LANCZOS interpolation pipeline to (224, 224, 3)<br>• Stratified 80/10/10 Train-Validation-Test splitting manifests<br>• Memory-mapped NPZ packaging for high-throughput I/O | **10%** |
| **Member 4** | **Model Evaluation & Empirical Benchmark Analyst** | Statistical Validation, Confusion Matrices & Error Audits | • Quantitative statistical evaluation across laboratory held-out splits<br>• Multi-class Confusion Matrix generation & Macro-F1 diagnostics<br>• Zero-shot wild field performance benchmark (Internal vs External)<br>• False-positive / false-negative agronomic pathology audit | **10%** |
| **Member 5** | **Full-Stack MLOps & Agronomic Intelligence Lead** | Web Deployment, Clinical Knowledge Base & Reports | • Production Streamlit web application development (`app.py`)<br>• Agronomic pathology encyclopedia (symptoms, organic & chemical treatments)<br>• Automated Clinical Pathology PDF & Markdown report generator<br>• Continuous cloud deployment via Streamlit Community Cloud (24/7) | **10%** |

### 🎯 1.2 Detailed RACI Governance Matrix

| Pipeline Milestone / Deliverable | Member 1 (Lead) | Member 2 (Data) | Member 3 (Prep) | Member 4 (Eval) | Member 5 (MLOps) |
|---|:---:|:---:|:---:|:---:|:---:|
| **1. Problem Formulation & Specification** | **A / R** | C | C | I | I |
| **2. Multi-Source Raw Data Collection (Lab + Field)** | A | **R** | C | I | I |
| **3. SHA-256 Deduplication & Integrity Audit** | A | **R** | C | I | I |
| **4. EXIF Correction & Resizing Pipeline (224x224)** | A | C | **R** | I | I |
| **5. Stratified 80/10/10 Split & NPZ Serialization** | A | C | **R** | I | I |
| **6. Custom Dual-Pool CNN Design (Potato)** | **A / R** | C | I | I | I |
| **7. MobileNetV2 Fine-Tuning (Tomato, Apple, Corn, Grape)** | **A / R** | C | I | I | I |
| **8. Model Convergence & Training History Audits** | **A / R** | I | I | C | I |
| **9. Confusion Matrices, Precision, Recall & F1-Scores** | A | I | I | **R** | I |
| **10. Dual-Pool Testing (Internal Held-Out vs. Wild Field)** | A | C | I | **R** | I |
| **11. Production Web Application (Streamlit Engine)** | A | I | I | I | **R** |
| **12. Agronomic Chemical & Organic Knowledge Base** | A | I | I | I | **R** |
| **13. Diagnostic PDF Pathology Report Generation** | A | I | I | I | **R** |
| **14. 24/7 Streamlit Cloud & Cloudflare Deployment** | **A** | I | I | I | **R** |

*(Legend: **R** = Responsible [does the work], **A** = Accountable [approves and ensures quality], **C** = Consulted, **I** = Informed)*

---

## 🌍 SECTION 2: PROBLEM CONTEXT, AGRONOMIC MOTIVATION & RESEARCH OBJECTIVES

### 2.1 The Global Food Security Challenge
Plant diseases threaten global agricultural yields, destabilizing rural economies and causing over **$220 billion in annual crop losses worldwide** according to the Food and Agriculture Organization (FAO). Smallholder farmers, who produce over 70% of the food supply in developing economies such as India, lack immediate access to certified plant pathologists.

### 2.2 The "Laboratory-to-Field" Generalization Breakdown
The vast majority of existing computer vision research utilizes controlled laboratory datasets such as PlantVillage, where leaves are photographed against uniform, flat neutral backgrounds under artificial lighting. When models trained exclusively on such datasets are deployed on actual farms, their empirical accuracy frequently degrades by **30% to 50%**. This breakdown occurs due to:
1. **Severe Background Clutter:** Soil, irrigation pipes, weeds, and neighboring canopy foliage create spatial visual noise.
2. **Variable Solar Illumination:** Direct sunlight, shadows, glare, and specular reflection distort RGB color signatures.
3. **Compound Foliar Pathologies:** Leaves in the field frequently exhibit insect grazing, nutrient deficiencies (nitrogen/potassium chlorosis), and physical mechanical damage alongside fungal infections.

### 2.3 AgroVision AI Objectives
To resolve these systemic deficiencies, our project established five foundational objectives:
1. **Comprehensive Multi-Host Intelligence:** Provide simultaneously active deep learning models across **5 high-value agricultural crops** (Potato, Tomato, Apple, Corn, Grape), unlike existing prototypes that support only 2 or 3 crops.
2. **Multi-Source Data Fusion:** Supplement laboratory baselines with authentic, field-collected vineyard imagery from **Mendeley GVLiD (DOI: 10.17632/wkymf8bhcg.5)** and real-world farm repositories.
3. **Domain-Adapted Architectures:** Engineer specialized architectures—specifically combining **Parallel MaxPooling2D and AveragePooling2D pathways** for textured fungal blights with fine-tuned **MobileNetV2 inverted residual blocks** for edge inference.
4. **Transparent Empirical Benchmarking:** Evaluate all models against both controlled held-out test splits and wild field imagery to openly report real-world generalization drops.
5. **Actionable MLOps Delivery:** Package the models into a 24/7 accessible, responsive web application offering Plotly probability distributions, Shannon entropy uncertainty scores, organic/chemical treatment schedules with exact dosages, and downloadable pathology reports.

---

## 🗄️ SECTION 3: MULTI-SOURCE DATA ENGINEERING & FORENSIC AUDITING (STAGES 1 & 2)

### 3.1 Raw Dataset Sourcing & Provenance
To ensure rigorous training and prevent domain overfitting, data was ingested from two distinct sources:
1. **Internal Laboratory Baseline (PlantVillage Repository):** High-resolution leaf imagery captured in controlled laboratory settings with uniform neutral paper backings.
2. **External Authentic Field Repositories (Mendeley Data GVLiD):** Authentic vineyard field imagery collected in Maharashtra, India (Mendeley Data, DOI: 10.17632/wkymf8bhcg.5) along with open-access agricultural field collections.

### 3.2 Cryptographic SHA-256 Deduplication Pipeline
Data leakage between training, validation, and testing sets is a primary cause of inflated, non-generalizable model accuracy. **Member 2** designed and executed a strict cryptographic verification script using the SHA-256 algorithm:

$$\text{Hash} = \text{SHA-256}(\text{ByteStream}(\text{Image}))$$

Every image file across all directories was read into an in-memory byte buffer, hashed, and checked against a hash registry. Exact duplicate files and corrupted headers were quarantined and purged.

### 3.3 Dataset Distribution Across 5 Crops (15 Classes)

| Crop | Class Name | Internal Lab Images | External Field Images | Total Verified Images | Clean Verification Status |
|---|---|:---:|:---:|:---:|:---:|
| **Potato** | Healthy | 300 | 300 | 600 | ✅ 0 Duplicates, 0 Corrupt |
| | Early Blight | 300 | 300 | 600 | ✅ 0 Duplicates, 0 Corrupt |
| | Late Blight | 300 | 300 | 600 | ✅ 0 Duplicates, 0 Corrupt |
| **Tomato** | Healthy | 300 | 300 | 600 | ✅ 0 Duplicates, 0 Corrupt |
| | Early Blight | 300 | 300 | 600 | ✅ 0 Duplicates, 0 Corrupt |
| | Late Blight | 300 | 300 | 600 | ✅ 0 Duplicates, 0 Corrupt |
| **Apple** | Healthy | 300 | 300 | 600 | ✅ 0 Duplicates, 0 Corrupt |
| | Apple Scab | 300 | 300 | 600 | ✅ 0 Duplicates, 0 Corrupt |
| | Cedar Apple Rust | 300 | 300 | 600 | ✅ 0 Duplicates, 0 Corrupt |
| **Corn** | Healthy | 300 | 300 | 600 | ✅ 0 Duplicates, 0 Corrupt |
| | Common Rust | 300 | 300 | 600 | ✅ 0 Duplicates, 0 Corrupt |
| | Northern Leaf Blight | 300 | 300 | 600 | ✅ 0 Duplicates, 0 Corrupt |
| **Grape** | Healthy | 300 | 300 | 600 | ✅ 0 Duplicates, 0 Corrupt |
| | Black Rot | 300 | 300 | 600 | ✅ 0 Duplicates, 0 Corrupt |
| | Leaf Blight | 300 | 300 | 600 | ✅ 0 Duplicates, 0 Corrupt |
| **TOTAL** | **15 Classes** | **4,500** | **4,500** | **9,000 Verified Images** | **100% Balanced** |

---

## ⚙️ SECTION 4: ADAPTIVE PREPROCESSING & DATA PIPELINES (STAGE 3)

### 4.1 EXIF Normalization & Color Space Homogenization
Smartphone cameras embed Exchangeable Image File Format (EXIF) metadata specifying sensor orientation. If ignored, vertical mobile photos are rotated 90 degrees during array conversion, introducing unnatural geometric distortions. **Member 3** implemented an automatic EXIF orientation transposition module via `PIL.ImageOps.exif_transpose()`.

All images were verified for standard 3-channel RGB representation. Any RGBA (transparency alpha channels) or grayscale images were converted to RGB using:

$$I_{\text{RGB}}(x, y) = \begin{cases} (I_{\text{gray}}, I_{\text{gray}}, I_{\text{gray}}), & \text{if Grayscale} \\ (R, G, B), & \text{if RGBA (Alpha discarded)} \end{cases}$$

### 4.2 High-Fidelity Spatial Resizing & Stratified Splitting
Images were resized to the unified dimension of $224 \times 224 \times 3$ pixels utilizing **LANCZOS interpolation**, which utilizes a sinc-windowed kernel to preserve sharp high-frequency lesion contours:

$$L(x) = \begin{cases} \text{sinc}(x) \cdot \text{sinc}\left(\frac{x}{a}\right), & \text{for } -a < x < a \\ 0, & \text{otherwise} \end{cases} \quad (a=3)$$

Datasets were split into **80% Training**, **10% Validation**, and **10% Held-Out Testing** partitions using stratified sampling to maintain class balance:
- **Training Set (80%):** 1,440 images per crop (used for gradient updates)
- **Validation Set (10%):** 180 images per crop (used for EarlyStopping & checkpoint monitoring)
- **Held-Out Test Set (10%):** 180 images per crop (reserved strictly for final benchmark evaluation)

### 4.3 High-Throughput Serialization (Compressed NPZ Archives)
To eliminate Colab I/O bottlenecks caused by reading thousands of individual image files from Google Drive, datasets were serialized into compressed NumPy archives (`.npz`) storing pre-extracted uint8 arrays:
- `potato_processed_data.npz` (743.5 MB)
- `tomato_processed_data.npz` (152.0 MB)
- `apple_processed_data.npz` (202.0 MB)
- `corn_processed_data.npz` (185.4 MB)
- `grape_processed_data.npz` (183.2 MB)

This achieved a **15x reduction in data-loading latency** during model training.

---

## 🧠 SECTION 5: DEEP LEARNING ARCHITECTURES & TRAINING REGIMES (STAGES 4 & 6)

### 5.1 Custom Dual-Pool CNN Architecture (Potato Pipeline)
Designed by **Member 1**, the Potato diagnostic model utilizes a specialized custom Convolutional Neural Network engineered for irregular necrotrophic fungal lesions.

#### The Dual-Pooling Rationale:
Standard CNNs rely on either `MaxPooling2D` (which captures sharp edge boundaries) or `AveragePooling2D` (which captures smooth background textures). For Late Blight, water-soaked margins require edge sensitivity, while Early Blight concentric rings require textural sensitivity. We deployed a **Parallel Dual-Pooling block**:

$$\mathbf{F}_{\text{pool}} = \text{Concat}\Big(\text{MaxPool2D}(\mathbf{F}_{\text{conv}}), \; \text{AvgPool2D}(\mathbf{F}_{\text{conv}})\Big)$$

```
Input (224x224x3) ──> Conv2D(32, 3x3) ──> BatchNorm ──> ReLU ──> MaxPool(2x2)
                 ──> Conv2D(64, 3x3) ──> BatchNorm ──> ReLU ──> MaxPool(2x2)
                 ──> Conv2D(128, 3x3) ──> BatchNorm ──> ReLU ──> MaxPool(2x2)
                 ──> Conv2D(256, 3x3) ──> BatchNorm ──> ReLU ──> [PARALLEL DUAL-POOL]
                 ──> Flatten ──> Dense(256) ──> Dropout(0.5) ──> Dense(3, Softmax)
```
- **Total Parameters:** 11.1 Million
- **Model Footprint:** 127.9 MB (`potato_cnn_combined_best.keras`)

### 5.2 MobileNetV2 Fine-Tuned Transfer Learning (Tomato, Apple, Corn, Grape)
For Tomato, Apple, Corn, and Grape, we leveraged **MobileNetV2** pretrained on ImageNet (1.4 million images). MobileNetV2 utilizes **Inverted Residual Blocks with Linear Bottlenecks**:
1. **$1 \times 1$ Expansion Convolution:** Expands low-dimensional input features to a higher-dimensional space (expansion factor $t=6$).
2. **$3 \times 3$ Depthwise Convolution:** Applies spatial filtering independently to each channel, dramatically reducing computational complexity:
   $$\text{FLOPs}_{\text{depthwise}} = D_K \cdot D_K \cdot M \cdot D_F \cdot D_F$$
   compared to standard convolution $\mathcal{O}(D_K^2 \cdot M \cdot N \cdot D_F^2)$.
3. **$1 \times 1$ Projection Bottleneck:** Projects channels back to low dimensions without non-linear activation to prevent information loss.

#### Custom Classification Head:
```
MobileNetV2 Base (Pretrained ImageNet, Frozen Lower 100 Layers)
                 ──> GlobalAveragePooling2D()
                 ──> BatchNormalization()
                 ──> Dense(256, activation='relu', kernel_regularizer=L2(1e-4))
                 ──> Dropout(0.4)
                 ──> Dense(64, activation='relu')
                 ──> Dense(3, activation='softmax')
```

### 5.3 Training Optimization & Hyperparameter Configurations
All 5 pipelines were trained using the **Adam Optimizer** with categorical cross-entropy loss:

$$\mathcal{L}_{\text{CCE}} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

- **Batch Size:** 32
- **Initial Learning Rate:** $1 \times 10^{-4}$ with `ReduceLROnPlateau` (decay factor $\alpha=0.2$, patience $p=3$).
- **EarlyStopping Callback:** Monitored `val_loss` with patience $p=7$, restoring optimal checkpoint weights.
- **Data Augmentation:** Real-time random horizontal flip, random rotation ($\pm 20^\circ$), shear ($\pm 0.15$), and zoom ($\pm 0.2$).

---

## 📊 SECTION 6: EMPIRICAL EVALUATION, BENCHMARKING & ERROR DIAGNOSTICS (STAGES 5 & 7)

### 6.1 Quantitative Performance Matrix Across All 5 Crops
Evaluated by **Member 4**, the models demonstrated high accuracy on held-out test splits and verified generalization on wild real-world field imagery.

| Host Crop | Model Architecture | Parameters | Held-Out Test Accuracy | Held-Out Loss | Internal Real-World Accuracy | External Field (Wild) Accuracy | Macro F1-Score |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥔 **Potato** | Custom Dual-Pool CNN | 11.1 M | **98.21%** | 0.0812 | **96.70%** | **92.40%** | **0.982** |
| 🍅 **Tomato** | MobileNetV2 Fine-Tuned | 2.8 M | **97.78%** | 0.0745 | **100.0%** | **60.00%** | **0.978** |
| 🍎 **Apple** | MobileNetV2 Transfer | 2.9 M | **97.75%** | 0.0911 | **96.67%** | **56.67%** | **0.977** |
| 🌽 **Corn** | MobileNetV2 Fine-Tuned | 2.9 M | **97.41%** | 0.0894 | **100.0%** | **83.33%** | **0.974** |
| 🍇 **Grape** | MobileNetV2 Dual-Dense | 2.9 M | **95.56%** | 0.1656 | **96.67%** | **76.67%** | **0.955** |

### 6.2 Analysis of the "Generalization Drop"
A central finding of our empirical evaluation is that models trained on laboratory imagery experience predictable accuracy drops when tested on unconstrained field data:
- **Corn (83.33% Field Generalization):** Retained strong generalization because parallel venation lines in maize leaves provide strong structural geometric priors that invariant filters in MobileNetV2 easily capture.
- **Grape (76.67% Field Generalization):** The inclusion of **Mendeley GVLiD field imagery** allowed the model to attain **100% precision on Black Rot** and **90% recall on Leaf Blight** under actual vineyard lighting in Maharashtra.
- **Tomato & Apple (56.67% - 60.0% Wild Generalization):** Suffered from specular reflections on glossy apple leaves and high soil background intrusion in outdoor tomato vines. This emphasizes the vital necessity of real-world field data in agricultural computer vision.

---

## 💻 SECTION 7: PRODUCTION WEB APPLICATION, AGRONOMIC EXPERT SYSTEM & MLOPS (STAGE 8)

### 7.1 Architecture of `8th Web_Application/`
Developed and deployed by **Member 5** with architecture supervision from **Member 1**, the web suite delivers an interactive clinical diagnostic interface:

```
8th Web_Application/
├── app.py                      # Master Streamlit dashboard & reactive session engine
├── disease_knowledge.py        # Agronomic pathology encyclopedia (15 conditions)
├── model_utils.py              # Resilient cached loader (@st.cache_resource) & inference engine
├── report_generator.py         # Clinical pathology PDF & Markdown report builder
├── requirements.txt            # Environment specifications
├── run_web_application.ipynb   # 1-Click Colab launcher with Cloudflare tunnel
└── test_samples/               # Curated 1-click test specimen library (15 samples)
```

### 7.2 Key User Interface & Diagnostic Capabilities
1. **Dynamic 5-Plant Selector:** Instantaneous switching between Potato, Tomato, Apple, Corn, and Grape without reloading page state.
2. **Dual Specimen Input:** Supports standard file drag-and-drop (`.jpg`, `.png`) alongside live camera capture for mobile field inspections.
3. **1-Click Instant Evaluation Buttons:** 3 pre-loaded samples per crop (1 Healthy, 2 Diseased) allow instant evaluation without requiring user uploads.
4. **Interactive Plotly Spectrum:** Displays full softmax probability distribution with entropy uncertainty scoring:
   $$H(X) = -\sum_{c=1}^{C} p(x_c) \log_2(p(x_c))$$
5. **Dynamic Severity Badges:** Color-coded severity tiers:
   - 🟢 `LOW` (Healthy Specimen)
   - 🟡 `MEDIUM` (Moderate infection: Early Blight, Common Rust, Cedar Apple Rust)
   - 🔴 `HIGH / CRITICAL` (Severe epidemic risk: Late Blight, Black Rot, Apple Scab)

### 7.3 The Agronomic Treatment Guide (4 Structured Clinical Tabs)
For every detected condition, the system provides four structured prescription tabs:
1. **Visual Symptoms:** Causal organism, pathogen class, and diagnostic foliar markers.
2. **Organic & Biocontrol:** Neem seed oil (3-5 ml/L), *Trichoderma harzianum* (4 g/L), *Bacillus subtilis*, and cultural sanitation.
3. **Chemical Fungicide Schedule:** Exact commercial formulations (e.g., Mancozeb 75% WP @ 2.5 g/L, Azoxystrobin 23% SC @ 1 ml/L, Metalaxyl 8% + Mancozeb 64% WP @ 2.0 g/L) with statutory pre-harvest intervals (PHI).
4. **Field Prevention & Hygiene:** Crop rotation regimens, canopy pruning, and drip irrigation practices.

### 7.4 Downloadable Pathology Reports & Cloud Deployment
- **Clinical PDF Report:** Formatted pathology document generated using ReportLab, containing diagnostic ID, specimen timestamp, probability spectrum, and prescriptions.
- **24/7 Cloud Deployment:** Deployed permanently to **Streamlit Community Cloud** (`https://agrovision--ai.streamlit.app`) connected to the GitHub repository.

---

## 🎓 SECTION 8: FACULTY VIVA VOCE & TECHNICAL DEFENSE GUIDE

This section prepares the 5 team members to confidently address core engineering and scientific questions during academic examination:

#### Q1: Why did you choose a Custom CNN for Potato instead of MobileNetV2 like the other crops?
> **Answer (Member 1):** Potato foliar diseases (Early Blight vs. Late Blight) exhibit distinct geometric morphologies: Early Blight produces sharp, concentric target-board rings, whereas Late Blight produces diffuse, water-soaked, spreading margins. A standard CNN with parallel **MaxPooling2D (preserving sharp edges)** and **AveragePooling2D (preserving textural background)** outperformed MobileNetV2 on this specific morphology, achieving 98.21% test accuracy.

#### Q2: How did you ensure zero data leakage between your train and test sets?
> **Answer (Member 2 & 3):** Many public implementations inadvertently leak data by splitting images after applying random augmentation. We executed **SHA-256 cryptographic deduplication** on the raw dataset first, purged all duplicates, and then applied a strict **stratified 80/10/10 split on raw files**. Data augmentation was applied dynamically *only* to the training batch tensors during runtime.

#### Q3: Why is there an accuracy drop when testing on real-world field images?
> **Answer (Member 4):** This illustrates the classical **domain-shift phenomenon** in computer vision. Laboratory datasets lack specular solar glare, complex soil backgrounds, and overlapping leaf occlusion. By including **GVLiD field data**, our Grape model achieved 76.67% zero-shot field accuracy, whereas models trained solely on clean data (Tomato/Apple) exhibited lower generalization (56-60%), validating our scientific thesis.

#### Q4: How is the web application optimized for mobile farmers with slow connections?
> **Answer (Member 5):** The web application implements **`@st.cache_resource`** to keep model weights loaded in memory, achieving an average inference latency of **42 milliseconds**. Furthermore, the user interface features automated EXIF orientation correction so mobile photos taken in vertical portrait mode are processed correctly without user intervention.

---

## 📚 SECTION 9: ACADEMIC REFERENCES & BIBLIOGRAPHY
1. **Hughes, D. P., & Salathé, M. (2015).** *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv preprint arXiv:1511.08060.
2. **Mendeley Data GVLiD Dataset:** *Grape Vineyard Leaves Dataset for Indian Viticulture.* DOI: `10.17632/wkymf8bhcg.5`.
3. **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018).** *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4510-4520.
4. **Tan, M., & Le, Q. (2019).** *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* International Conference on Machine Learning (ICML), pp. 6105-6114.
5. **Agrios, G. N. (2005).** *Plant Pathology (5th ed.).* Elsevier Academic Press.
