"""
AgroVision AI: Multi-Plant Foliar Disease Diagnostics & Pathology Suite
State-of-the-Art Deep Learning Computer Vision Platform
Supports All 5 Crops: Potato, Tomato, Apple, Corn, Grape
"""

import os
import io
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.graph_objects as go

# Import internal modules
from disease_knowledge import DISEASE_KNOWLEDGE, EMPIRICAL_EVALUATION_METRICS
from model_utils import CROPS_METADATA, predict_crop_disease, get_model_path
from report_generator import generate_pdf_report, generate_markdown_report

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="AgroVision AI — Multi-Plant Disease Diagnostic Suite",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# APP HEADER
# ==============================================================================
st.title("🌿 AgroVision AI — Multi-Plant Disease Intelligence Suite")
st.caption("Enterprise-grade Computer Vision platform for foliar pathogen diagnosis, empirical benchmarking, and actionable agronomic prescriptions.")

col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
col_b1.info("🥔 **Potato** (Solanaceae)")
col_b2.info("🍅 **Tomato** (Solanaceae)")
col_b3.info("🍎 **Apple** (Rosaceae)")
col_b4.info("🌽 **Corn** (Poaceae)")
col_b5.info("🍇 **Grape** (Vitaceae)")

st.divider()

# ==============================================================================
# SIDEBAR CONFIGURATION & CROP SELECTOR
# ==============================================================================
with st.sidebar:
    st.header("🌾 Select Target Crop")
    
    crop_options = list(CROPS_METADATA.keys())
    crop_labels = [f"{CROPS_METADATA[c]['icon']} {c}" for c in crop_options]
    
    selected_idx = st.selectbox(
        "Active Diagnostic Pipeline:",
        range(len(crop_options)),
        format_func=lambda i: crop_labels[i],
        index=0
    )
    selected_crop = crop_options[selected_idx]
    crop_meta = CROPS_METADATA[selected_crop]
    
    st.markdown(f"**Species:** `{crop_meta['display_name']}`")
    st.markdown(f"**Architecture:** `{crop_meta['architecture']}`")
    
    model_path = get_model_path(selected_crop)
    if model_path:
        st.success(f"✅ Model Loaded: `{os.path.basename(model_path)}`")
    else:
        st.warning("⚠️ Model weights not found in standard Drive path.")

    st.divider()
    st.subheader("⚙️ Diagnostic Settings")
    conf_threshold = st.slider(
        "Confidence Alert Threshold (%)",
        min_value=50,
        max_value=95,
        value=70,
        help="Diagnoses below this threshold flag an advisory note."
    )
    
    st.divider()
    st.caption("**AgroVision CV v2.4 (Production)**")
    st.caption("Includes PlantVillage + Mendeley GVLiD India field benchmarks.")

# ==============================================================================
# NAVIGATION TABS
# ==============================================================================
tab_diag, tab_bench, tab_lib, tab_about = st.tabs([
    "🔬 Leaf Diagnosis & Prescription",
    "📊 Empirical Benchmarks & Analytics",
    "🌿 Agronomic Disease Library",
    "ℹ️ Architecture & Documentation"
])

# ==============================================================================
# TAB 1: LEAF DIAGNOSIS & PRESCRIPTION
# ==============================================================================
with tab_diag:
    st.subheader(f"Step 1: Provide {selected_crop} Leaf Specimen")
    st.write(
        f"Provide a clear close-up photograph of the suspected **{selected_crop}** leaf. "
        "You can upload a photo, use your camera, or test instantly using the 1-Click sample buttons below."
    )
    
    col_input, col_samples = st.columns([1.1, 0.9])
    
    uploaded_file = None
    camera_file = None
    selected_demo_path = None
    
    with col_input:
        input_choice = st.radio(
            "Image Input Source:",
            ["📁 Upload File (JPG / PNG)", "📷 Live Camera Capture"],
            horizontal=True
        )
        if input_choice == "📁 Upload File (JPG / PNG)":
            uploaded_file = st.file_uploader(
                f"Choose a {selected_crop} leaf image",
                type=["jpg", "jpeg", "png"]
            )
        else:
            camera_file = st.camera_input(f"Capture live {selected_crop} leaf")

    with col_samples:
        st.markdown("**⚡ 1-Click Instant Test Samples:**")
        st.caption("Test the model immediately with pre-loaded samples:")
        
        sample_dir = os.path.join(os.path.dirname(__file__), "test_samples", selected_crop)
        if not os.path.isdir(sample_dir):
            sample_dir = os.path.join(r"G:\My Drive\Plant Disease Detection (Computer Vision)\8th Web_Application	est_samples", selected_crop)
        
        demo_cols = st.columns(3)
        if os.path.isdir(sample_dir):
            sample_files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            for i, f in enumerate(sample_files[:3]):
                name_clean = f.replace("_sample.jpg", "").replace("_", " ")
                btn_col = demo_cols[i % 3]
                with btn_col:
                    if st.button(f"🔎 {name_clean}", key=f"btn_{selected_crop}_{f}", use_container_width=True):
                        selected_demo_path = os.path.join(sample_dir, f)
        else:
            st.info("Demo samples directory ready.")

    active_image = None
    if selected_demo_path and os.path.isfile(selected_demo_path):
        active_image = selected_demo_path
    elif uploaded_file is not None:
        active_image = uploaded_file
    elif camera_file is not None:
        active_image = camera_file

    st.divider()

    # ==========================================================================
    # INFERENCE & DIAGNOSTIC RESULTS
    # ==========================================================================
    if active_image is not None:
        with st.spinner(f"Analyzing {selected_crop} specimen with deep neural network..."):
            try:
                start_time = time.time()
                result = predict_crop_disease(selected_crop, active_image)
                latency = (time.time() - start_time) * 1000.0
            except Exception as e:
                st.error(f"Inference Error: {str(e)}")
                result = None

        if result:
            pred_class = result["predicted_class"]
            conf = result["confidence"]
            class_probs = result["class_probs"]
            entropy = result["entropy"]
            
            crop_dict = DISEASE_KNOWLEDGE.get(selected_crop, {})
            diag_info = crop_dict.get(pred_class, {
                "scientific_name": "N/A",
                "pathogen_type": "N/A",
                "causal_agent": "N/A",
                "severity_level": "Undetermined",
                "severity_color": "#10B981",
                "visual_symptoms": [],
                "organic_management": [],
                "chemical_management": [],
                "prevention_guidelines": [],
            })

            col_view, col_diag = st.columns([1, 1.2])

            with col_view:
                st.subheader("📸 Specimen Image")
                st.image(result["pil_image"], use_container_width=True, caption=f"Analyzed {selected_crop} Specimen")
                
                w, h = result["pil_image"].size
                m1, m2, m3 = st.columns(3)
                m1.metric("Resolution", f"{w}x{h}")
                m2.metric("Inference Time", f"{latency:.1f} ms")
                m3.metric("Entropy", f"{entropy:.2f}")

            with col_diag:
                st.subheader("📋 Diagnostic Assessment")
                
                # Big Result Card using st.container
                is_healthy = (pred_class.lower() == "healthy")
                if is_healthy:
                    st.success(f"### Result: {pred_class.replace('_', ' ')} (Healthy)")
                else:
                    st.error(f"### Detected: {pred_class.replace('_', ' ')}")
                
                r_col1, r_col2 = st.columns(2)
                r_col1.metric("Confidence Score", f"{conf:.2f}%")
                r_col1.write(f"**Pathogen Type:** `{diag_info.get('pathogen_type', 'N/A')}`")
                
                r_col2.metric("Severity Rating", diag_info.get("severity_level", "Unknown"))
                r_col2.write(f"**Scientific Name:** *{diag_info.get('scientific_name', 'N/A')}*")

                if conf < conf_threshold:
                    st.warning(
                        f"⚠️ **Advisory Notice**: Confidence ({conf:.1f}%) is below {conf_threshold}%. "
                        "Secondary inspection or alternate photo under uniform lighting is recommended."
                    )

                # Probability spectrum chart
                st.markdown("**Diagnostic Probability Spectrum:**")
                df_probs = pd.DataFrame([
                    {"Condition": k.replace("_", " "), "Probability (%)": v}
                    for k, v in class_probs.items()
                ]).sort_values("Probability (%)", ascending=True)

                fig_bar = go.Figure(go.Bar(
                    x=df_probs["Probability (%)"],
                    y=df_probs["Condition"],
                    orientation='h',
                    marker=dict(
                        color=df_probs["Probability (%)"],
                        colorscale="Viridis",
                    ),
                    text=[f"{p:.1f}%" for p in df_probs["Probability (%)"]],
                    textposition='outside'
                ))
                fig_bar.update_layout(
                    height=200,
                    margin=dict(l=10, r=40, t=10, b=10),
                    xaxis=dict(range=[0, 115], title="Confidence (%)"),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # Prescription Tabs
            st.divider()
            st.subheader("💊 Agronomic Pathology & Treatment Guide")

            p_tab1, p_tab2, p_tab3, p_tab4 = st.tabs([
                "🔍 Visual Symptoms",
                "🌱 Organic & Biocontrol",
                "🧪 Chemical & Fungicides",
                "🛡️ Prevention & Field Hygiene"
            ])

            with p_tab1:
                st.markdown(f"#### Primary Symptoms of {pred_class.replace('_', ' ')}")
                for sym in diag_info.get("visual_symptoms", []):
                    st.markdown(f"- 🔎 **{sym}**")
                st.info(f"**Causal Organism:** `{diag_info.get('causal_agent', 'N/A')}`")

            with p_tab2:
                st.markdown("#### Eco-Friendly & Biological Interventions")
                for org in diag_info.get("organic_management", []):
                    st.markdown(f"- 🌿 {org}")
                st.caption("Suitable for IPM programs and organic farming protocols.")

            with p_tab3:
                st.markdown("#### Conventional Chemical Treatments & Dosages")
                for chem in diag_info.get("chemical_management", []):
                    st.markdown(f"- 🔬 **{chem}**")
                st.warning("⚠️ Always follow label guidelines and observe statutory pre-harvest intervals (PHI).")

            with p_tab4:
                st.markdown("#### Long-Term Cultural & Preventive Strategies")
                for prev in diag_info.get("prevention_guidelines", []):
                    st.markdown(f"- 🛡️ {prev}")

            # Download Reports
            st.divider()
            st.subheader("📥 Export Official Pathology Report")
            col_pdf, col_md = st.columns(2)

            with col_pdf:
                try:
                    pdf_bytes = generate_pdf_report(selected_crop, result, diag_info)
                    st.download_button(
                        label="📄 Download Official PDF Report",
                        data=pdf_bytes,
                        file_name=f"AgroVision_Pathology_{selected_crop}_{pred_class}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF Notice: {e}")

            with col_md:
                md_text = generate_markdown_report(selected_crop, result, diag_info)
                st.download_button(
                    label="📝 Download Markdown Text Report",
                    data=md_text,
                    file_name=f"AgroVision_Report_{selected_crop}_{pred_class}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
    else:
        st.info("👆 Please upload a photo, take a picture, or select one of the 1-Click test samples above.")

# ==============================================================================
# TAB 2: EMPIRICAL BENCHMARKS & ANALYTICS
# ==============================================================================
with tab_bench:
    st.subheader("📊 Empirical Performance Across All 5 Crops")
    st.write(
        "Our models were evaluated across dual test suites: controlled laboratory splits and "
        "unconstrained **real-world agricultural field imagery**."
    )

    bench_data = []
    for c_name, metrics in EMPIRICAL_EVALUATION_METRICS.items():
        bench_data.append({
            "Crop": f"{CROPS_METADATA[c_name]['icon']} {c_name}",
            "Architecture": metrics["model_architecture"],
            "Laboratory Held-Out": f"{metrics['heldout_accuracy']:.2f}%",
            "Internal Real-World": f"{metrics['internal_real_world_accuracy']:.2f}%",
            "External Field (Wild)": f"{metrics['external_field_accuracy']:.2f}%",
            "Parameters": metrics["parameter_count"],
            "Strength": metrics["key_strength"],
        })
    
    df_benchmark = pd.DataFrame(bench_data)
    st.dataframe(df_benchmark, use_container_width=True, hide_index=True)

    st.divider()
    crops_list = list(EMPIRICAL_EVALUATION_METRICS.keys())
    heldout_scores = [EMPIRICAL_EVALUATION_METRICS[c]["heldout_accuracy"] for c in crops_list]
    internal_scores = [EMPIRICAL_EVALUATION_METRICS[c]["internal_real_world_accuracy"] for c in crops_list]
    external_scores = [EMPIRICAL_EVALUATION_METRICS[c]["external_field_accuracy"] for c in crops_list]

    fig_bench = go.Figure()
    fig_bench.add_trace(go.Bar(name="Controlled Held-Out Split", x=crops_list, y=heldout_scores, marker_color="#10B981"))
    fig_bench.add_trace(go.Bar(name="Internal Real-World Pool", x=crops_list, y=internal_scores, marker_color="#3B82F6"))
    fig_bench.add_trace(go.Bar(name="External Real-World Field (Wild)", x=crops_list, y=external_scores, marker_color="#F59E0B"))
    fig_bench.update_layout(
        title="Accuracy Generalization Across Test Environments",
        barmode='group',
        yaxis=dict(title="Accuracy (%)", range=[40, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_bench, use_container_width=True)

# ==============================================================================
# TAB 3: AGRONOMIC DISEASE LIBRARY
# ==============================================================================
with tab_lib:
    st.subheader("📚 Comprehensive Plant Pathology Encyclopedia")
    st.write("Browse symptoms, etiology, and recommended controls for all 15 conditions across our 5 host species.")

    lib_crop = st.selectbox(
        "Select Crop to Explore:",
        list(DISEASE_KNOWLEDGE.keys()),
        format_func=lambda c: f"{CROPS_METADATA[c]['icon']} {c} ({CROPS_METADATA[c]['display_name']})",
        key="lib_crop_select"
    )

    crop_lib = DISEASE_KNOWLEDGE[lib_crop]
    for condition_name, info in crop_lib.items():
        with st.expander(f"{'🟢' if condition_name == 'Healthy' else '🔴'} {condition_name.replace('_', ' ')} — {info.get('scientific_name', '')}", expanded=(condition_name != "Healthy")):
            l_col1, l_col2 = st.columns([1, 1.2])
            with l_col1:
                st.write(f"**Pathogen Type:** `{info.get('pathogen_type', 'N/A')}`")
                st.write(f"**Causal Agent:** `{info.get('causal_agent', 'N/A')}`")
                st.write(f"**Severity Level:** **{info.get('severity_level', 'N/A')}**")
                st.markdown("##### 🔍 Symptoms:")
                for sym in info.get("visual_symptoms", []):
                    st.markdown(f"- {sym}")

            with l_col2:
                st.markdown("##### 🌱 Organic Treatments:")
                for org in info.get("organic_management", []):
                    st.markdown(f"- {org}")
                st.markdown("##### 🧪 Chemical Controls:")
                for chem in info.get("chemical_management", []):
                    st.markdown(f"- {chem}")

# ==============================================================================
# TAB 4: ARCHITECTURE & DOCUMENTATION
# ==============================================================================
with tab_about:
    st.subheader("🏛️ System Architecture & Data Methodology")
    st.markdown("""
    #### 1. Data Provenance & Preprocessing
    - **Dual-Source Ingestion:** Laboratory samples from PlantVillage combined with authentic field-collected vineyard imagery from **GVLiD (Mendeley Data DOI: 10.17632/wkymf8bhcg.5)** and real-world farm repositories.
    - **Rigorous Audit:** Strict cryptographic SHA-256 deduplication and RGB channel verification across all 5 crops.
    - **Standardized Preprocessing:** Unified image dimension `(224, 224, 3)` with EXIF orientation correction.

    #### 2. Deep Learning Modeling
    - **MobileNetV2 Transfer Learning:** Pretrained on ImageNet with fine-tuned depthwise separable convolutions for high inference efficiency and compact model weight footprint (~11-26 MB).
    - **Custom Dual-Pool CNN:** Integrated for potato foliar pathologies combining parallel MaxPooling2D and AveragePooling2D pathways.

    #### 3. Inference Engine & Deployment
    - Streamlit application optimized with `@st.cache_resource` for zero redundant disk I/O.
    - Dual platform compatibility: Runs seamlessly on local workstations or in Google Colab via high-speed tunnels.

    ---
    *Built with Python 3, TensorFlow, Streamlit, Plotly, and ReportLab.*
    """)
