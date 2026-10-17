"""
AI Image Classification Studio - Web Application
CodeOrbit Tech AI Internship - Task 3: Image Classification Using a Pretrained Model.

Interactive Streamlit interface featuring:
- Live ImageNet-1K pretrained model inference (MobileNetV3-Large, ResNet-50, EfficientNet-B0)
- Drag-and-drop image upload and curated sample presets
- Dynamic Top-K predictions with interactive confidence visualization
- Comprehensive image metadata extraction
- Model architecture deep dive, evaluation metrics, and technical interview viva prep
"""

import io
import os
import time
from typing import Any, Dict, List, Optional

from PIL import Image
import streamlit as st
import torch

from config_task3 import (
    ALLOWED_EXTENSIONS,
    AVAILABLE_MODELS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TOP_K,
    INTERNSHIP_ORGANIZATION,
    INTERNSHIP_TASK_NAME,
    MAX_FILE_SIZE_MB,
    MAX_TOP_K,
    MIN_TOP_K,
    PROJECT_TITLE,
)
from model import clean_label_name, load_classifier, run_inference
from preprocess import (
    ImageValidationError,
    load_and_validate_image,
    preprocess_image_tensor,
)
from utils import (
    build_export_csv,
    build_export_json,
    create_probability_chart,
    ensure_sample_images_exist,
    get_models_comparison_df,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title=f"{PROJECT_TITLE} | {INTERNSHIP_ORGANIZATION}",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling (Professional UI/UX Aesthetics)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Typography & Font Weights */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Badge & Card Components */
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-primary {
        background-color: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
    }
    .badge-success {
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }
    .badge-neutral {
        background-color: #F3F4F6;
        color: #374151;
        border: 1px solid #E5E7EB;
    }

    /* Info and Stat Cards */
    .info-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .dark .info-card {
        background: #1F2937;
        border-color: #374151;
    }

    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        color: white;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #FFFFFF !important;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #C7D2FE;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Prediction Result Cards */
    .prediction-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-left: 5px solid #4F46E5;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .dark .prediction-card {
        background: #111827;
        border-color: #374151;
        border-left-color: #6366F1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Cached Model Loader (Streamlit Resource Cache)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading pretrained model weights into memory...")
def get_cached_model(model_name: str):
    """Loads and caches the PyTorch model and weights for instant repeat inference."""
    return load_classifier(model_name)


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏢 CodeOrbit Tech")
    st.markdown(
        f'<span class="metric-badge badge-primary">{INTERNSHIP_TASK_NAME}</span>'
        f'<span class="metric-badge badge-success">Task Verified</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.subheader("⚙️ Model Configuration")
    selected_model_name = st.selectbox(
        "Choose Pretrained Backbone:",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Select between lightweight mobile models and deep residual networks.",
    )
    spec = AVAILABLE_MODELS[selected_model_name]

    # Model specs widget
    st.markdown(
        f"""
        <div class="info-card">
            <div style="font-weight: 600; font-size: 0.85rem; color: #4F46E5;">{spec.name}</div>
            <div style="font-size: 0.75rem; color: #6B7280; margin-top: 4px;"><b>Architecture:</b> {spec.architecture}</div>
            <div style="font-size: 0.75rem; color: #6B7280; margin-top: 4px;"><b>Model Size:</b> ~{spec.approx_size_mb:.1f} MB</div>
            <div style="font-size: 0.75rem; color: #6B7280; margin-top: 4px;"><b>ImageNet Top-1 Acc:</b> {spec.top1_accuracy:.1f}%</div>
            <div style="font-size: 0.75rem; color: #6B7280; margin-top: 4px;"><b>ImageNet Top-5 Acc:</b> {spec.top5_accuracy:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_k_val = st.slider(
        "Top-K Predictions to Display:",
        min_value=MIN_TOP_K,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
        step=1,
        help="Number of highest-ranking class probabilities to extract from the 1,000 ImageNet classes.",
    )

    st.markdown("---")
    st.subheader("🖼️ Image Input Source")
    input_source = st.radio(
        "Select Image Input Method:",
        options=["Upload Custom Image", "Use Curated Sample Presets"],
        index=0,
    )

    sample_images = ensure_sample_images_exist()
    selected_sample_label = None

    if input_source == "Use Curated Sample Presets":
        selected_sample_label = st.selectbox(
            "Choose Sample Image:",
            options=list(sample_images.keys()),
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #9CA3AF; text-align: center;">
            Two-Month AI Internship | CodeOrbit Tech<br>
            Developed by AI/ML Engineering Intern
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-banner">
        <h1 class="hero-title">🔬 {PROJECT_TITLE}</h1>
        <p class="hero-subtitle">
            Autonomous Deep Learning Vision Studio • Pretrained Torchvision Models • ImageNet-1K Benchmark
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Tabbed Navigation
# ---------------------------------------------------------
tab_studio, tab_architecture, tab_eval, tab_internship = st.tabs([
    "📸 Live Classification Studio",
    "🧠 Model Architecture & Pipeline",
    "📊 Evaluation Metrics & Limitations",
    "🎓 Internship Task Mapping & Viva Q&A",
])


# =========================================================
# TAB 1: Live Classification Studio
# =========================================================
with tab_studio:
    col_input, col_results = st.columns([1, 1], gap="large")

    active_image: Optional[Image.Image] = None
    active_metadata: Optional[Dict[str, Any]] = None
    active_filename: str = ""

    # Image Acquisition Section
    with col_input:
        st.subheader("1. Source Image Preview & Inspection")

        if input_source == "Upload Custom Image":
            uploaded_file = st.file_uploader(
                f"Drag & drop image (Formats: {', '.join(ALLOWED_EXTENSIONS)} | Max {MAX_FILE_SIZE_MB}MB)",
                type=[ext.replace(".", "") for ext in ALLOWED_EXTENSIONS],
                help=f"Supported file extensions: {', '.join(ALLOWED_EXTENSIONS)}. Max size: {MAX_FILE_SIZE_MB} MB.",
            )

            if uploaded_file is not None:
                try:
                    file_bytes = uploaded_file.read()
                    active_image, active_metadata = load_and_validate_image(
                        file_bytes, filename=uploaded_file.name
                    )
                    active_filename = uploaded_file.name
                except ImageValidationError as err:
                    st.error(f"❌ Input Validation Error: {err}")
            else:
                st.info(
                    "👆 Upload an image above, or switch the sidebar toggle to **'Use Curated Sample Presets'** to test immediately."
                )

        else:
            # Curated Preset Mode
            if selected_sample_label and selected_sample_label in sample_images:
                sample_path = sample_images[selected_sample_label]
                try:
                    active_image, active_metadata = load_and_validate_image(sample_path)
                    active_filename = os.path.basename(sample_path)
                except ImageValidationError as err:
                    st.error(f"❌ Error loading sample image: {err}")

        # Display preview and metadata card if image is valid
        if active_image is not None and active_metadata is not None:
            st.image(
                active_image,
                caption=f"Preview: {active_filename} ({active_metadata['width']}x{active_metadata['height']} px)",
                use_container_width=True,
            )

            # Metadata Table
            with st.expander("🔍 Image Technical Metadata", expanded=False):
                meta_cols = st.columns(2)
                with meta_cols[0]:
                    st.write(f"**Dimensions:** {active_metadata['width']} × {active_metadata['height']} px")
                    st.write(f"**Aspect Ratio:** {active_metadata['aspect_ratio']}")
                    st.write(f"**Channels:** 3 (Normalized RGB)")
                with meta_cols[1]:
                    st.write(f"**Original Mode:** `{active_metadata['original_mode']}`")
                    st.write(f"**Original Format:** `{active_metadata['original_format']}`")
                    st.write(f"**File Size:** {active_metadata['size_kb']}")

    # Results & Inference Section
    with col_results:
        st.subheader("2. Model Predictions & Softmax Probabilities")

        if active_image is not None:
            classify_btn = st.button("🚀 Run Neural Network Classification", type="primary", use_container_width=True)

            if classify_btn or "last_prediction" in st.session_state:
                with st.spinner(f"Executing feedforward inference with {selected_model_name}..."):
                    try:
                        # 1. Load model and transforms from cache
                        model, weights_transforms, categories = get_cached_model(selected_model_name)

                        # 2. Preprocess into standard ImageNet normalized tensor (1, 3, 224, 224)
                        input_tensor = preprocess_image_tensor(active_image, weights_transforms)

                        # 3. Execute inference and benchmark latency
                        inference_result = run_inference(
                            model, input_tensor, categories, top_k=top_k_val
                        )
                        st.session_state["last_prediction"] = inference_result
                        st.session_state["last_model"] = selected_model_name
                        st.session_state["last_meta"] = active_metadata

                    except Exception as exc:
                        st.error(f"❌ Inference Failure: {exc}")
                        inference_result = None

                if inference_result:
                    top_pred = inference_result["top_predictions"][0]
                    latency = inference_result["inference_time_ms"]

                    # Primary Result Callout
                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
                                    border: 1px solid #10B981; border-radius: 10px; padding: 16px; margin: 16px 0;">
                            <div style="font-size: 0.8rem; font-weight: 700; color: #065F46; text-transform: uppercase; letter-spacing: 0.05em;">
                                🏆 Primary Predicted Class (Rank #1)
                            </div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #064E3B; margin: 4px 0;">
                                {top_pred['label']}
                            </div>
                            <div style="display: flex; gap: 12px; font-size: 0.85rem; color: #047857;">
                                <span><b>Confidence:</b> {top_pred['percentage']}</span>
                                <span>•</span>
                                <span><b>Class ID:</b> #{top_pred['class_id']}</span>
                                <span>•</span>
                                <span><b>Inference Latency:</b> {latency:.2f} ms</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Interactive Plotly Chart
                    chart = create_probability_chart(inference_result["top_predictions"])
                    st.plotly_chart(chart, use_container_width=True)

                    # Detailed Top-K Breakdown Table
                    st.markdown("##### 📋 Complete Top-K Prediction Rankings")
                    for pred in inference_result["top_predictions"]:
                        st.markdown(
                            f"""
                            <div class="prediction-card">
                                <div>
                                    <span style="font-weight: 700; color: #4F46E5; margin-right: 8px;">#{pred['rank']}</span>
                                    <span style="font-weight: 600;">{pred['label']}</span>
                                    <span style="color: #6B7280; font-size: 0.8rem; margin-left: 6px;">(ImageNet #{pred['class_id']})</span>
                                </div>
                                <div style="font-weight: 700; color: #111827;">
                                    {pred['percentage']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Export Section for Task Submission Evidence
                    st.markdown("---")
                    st.markdown("##### 💾 Download Prediction Evidence for Internship Submission")
                    exp_cols = st.columns(2)
                    with exp_cols[0]:
                        json_str = build_export_json(
                            active_metadata, inference_result, selected_model_name
                        )
                        st.download_button(
                            label="📥 Download JSON Report",
                            data=json_str,
                            file_name=f"classification_{active_metadata['filename']}.json",
                            mime="application/json",
                            use_container_width=True,
                        )
                    with exp_cols[1]:
                        csv_str = build_export_csv(inference_result, selected_model_name)
                        st.download_button(
                            label="📥 Download CSV Summary",
                            data=csv_str,
                            file_name=f"predictions_{active_metadata['filename']}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
        else:
            st.info("👈 Please select or upload an image on the left to begin classification.")


# =========================================================
# TAB 2: Model Architecture & Preprocessing Pipeline
# =========================================================
with tab_architecture:
    st.subheader("🧠 Deep Learning Architecture & Mathematical Pipeline")

    st.markdown(
        """
        This project uses genuine PyTorch torchvision neural networks trained on the **ImageNet Large Scale Visual Recognition Challenge (ILSVRC-2012)** dataset, containing 1.28 million training images across 1,000 diverse categories.
        """
    )

    st.markdown("### 1. Preprocessing Pipeline (Tensor Standardization)")
    st.markdown(
        """
        Before an arbitrary pixel array can enter the convolutional backbone, it must undergo a rigorous 4-step transformation pipeline:

        1. **Aspect-Aware Resize**: The shortest side of the input image is resized to `256` pixels using bilinear interpolation.
        2. **Center Crop**: A central window of `224 × 224` pixels is cropped. This preserves high-density central features while standardizing input dimensions to match the fully connected classification head.
        3. **Tensor Scaling**: Pixel integers in range `[0, 255]` are converted to a floating-point PyTorch Tensor $\in [0.0, 1.0]$.
        4. **Z-Score Channel Normalization**: Channels are normalized using the official ImageNet empirical population statistics:
        """
    )

    st.latex(r"X_{norm} = \frac{X - \mu}{\sigma}")

    st.markdown(
        """
        - **Mean ($\mu$):** `Red: 0.485, Green: 0.456, Blue: 0.406`
        - **Std Dev ($\sigma$):** `Red: 0.229, Green: 0.224, Blue: 0.225`

        This zero-centering ensures gradients propagate stably without saturation during both training and feedforward inference.
        """
    )

    st.markdown("---")
    st.markdown("### 2. Supported Pretrained Backbones Comparison")
    st.dataframe(get_models_comparison_df(), use_container_width=True)

    st.markdown("---")
    st.markdown("### 3. Softmax Activation Function")
    st.markdown(
        """
        The final dense layer produces an unnormalized 1000-dimensional logit vector $\mathbf{z} = [z_1, z_2, \dots, z_{1000}]$.
        To transform these unbounded real values into a valid probability distribution where $\sum_{i=1}^{1000} p_i = 1$, the model applies the **Softmax function**:
        """
    )
    st.latex(r"P(Y = i \mid \mathbf{x}) = \frac{e^{z_i}}{\sum_{j=1}^{1000} e^{z_j}}")


# =========================================================
# TAB 3: Evaluation Metrics & Limitations
# =========================================================
with tab_eval:
    st.subheader("📊 Evaluation Methodology, Benchmark Metrics & Limitations")

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric(label="ImageNet Classes", value="1,000", help="ILSVRC 1K class ontology")
    with metric_cols[1]:
        st.metric(label="Validation Benchmark", value="50,000 Images", help="Standard ILSVRC val split")
    with metric_cols[2]:
        st.metric(label="Inference Mode", value="torch.no_grad()", help="Zero backprop memory overhead")

    st.markdown("### 1. Evaluation Metrics: Top-1 vs. Top-5 Accuracy")
    st.markdown(
        """
        In multi-class computer vision benchmarks with thousands of fine-grained categories, two primary metrics are standard:

        - **Top-1 Accuracy:** The prediction is counted as correct **only** if the single class with the highest probability matches the ground-truth label.
        - **Top-5 Accuracy:** The prediction is counted as correct if the ground-truth label is included anywhere among the model's top 5 highest probability predictions.

        > **Why Top-5 is essential for ImageNet:**
        > ImageNet contains over 120 distinct breeds of dogs (e.g., *Golden Retriever*, *Labrador Retriever*, *Chesapeake Bay Retriever*). Even human domain experts frequently confuse closely related sub-species. Top-5 accuracy reflects whether the model has learned the correct visual semantic family.
        """
    )

    st.markdown("---")
    st.markdown("### 2. Critical Real-World Limitations & Honest Assessment")
    st.warning(
        """
        **1. Closed-World Assumption (The 1,000-Class Ceiling):**
        The model can **only** predict classes that exist within the 1,000 ImageNet categories. If you upload an object outside ImageNet (e.g., a smartphone, quantum computer, or abstract painting), the model will force-map it to whichever of the 1,000 classes has the closest visual texture or color profile.

        **2. Softmax Overconfidence:**
        The Softmax function normalizes logits via exponents. When presented with out-of-distribution (OOD) noise or unrecognizable inputs, it can still assign a high confidence score to an arbitrary category. High confidence does **not** guarantee ground-truth correctness.

        **3. Center-Crop Information Loss:**
        Objects situated at extreme edges or corners of an uploaded image may be cropped out during the $224 \\times 224$ center-crop stage, resulting in misclassification.

        **4. Domain Shift & Lighting Variations:**
        ImageNet primarily comprises high-quality web photographs. Severe motion blur, thermal imagery, low-light night vision, or heavy digital compression will degrade prediction fidelity.
        """
    )


# =========================================================
# TAB 4: Internship Task Mapping & Viva Q&A
# =========================================================
with tab_internship:
    st.subheader("🎓 CodeOrbit Tech Internship Task 3 Guide & Viva Q&A")

    st.markdown(
        """
        <div class="info-card">
            <b>Internship Task Mapping:</b> Task 3 — Image Classification Using a Pretrained Model.<br>
            <b>Internship Organization:</b> CodeOrbit Tech Two-Month AI Internship Program.<br>
            <b>Core Deliverables Completed:</b> Pretrained PyTorch backbone, input validation, Top-K probability extraction, Streamlit UI, automated unit tests, and reproducible setup.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎯 Likely Technical Interview Questions & Model Answers")

    with st.expander("Q1: Why did you use a pretrained model instead of training a CNN from scratch?"):
        st.markdown(
            """
            **Answer:**
            Training a deep convolutional network like ResNet-50 from scratch requires:
            1. Millions of labeled training images (ImageNet is ~150 GB).
            2. High-performance GPU clusters training for days or weeks.
            3. Immense compute costs.

            By utilizing **Transfer Learning** with pretrained ImageNet weights, we leverage feature extractors that have already learned universal visual primitives (Gabor edges, textures, shapes, and complex object parts). This yields state-of-the-art inference accuracy instantly with zero training overhead.
            """
        )

    with st.expander("Q2: Why must the model be set to `model.eval()` during inference?"):
        st.markdown(
            """
            **Answer:**
            Certain neural network layers behave differently during training versus inference:
            - **Dropout:** During training, Dropout randomly zeroes out neurons to prevent co-adaptation. During inference (`eval()`), all neurons must remain active and their outputs scaled appropriately.
            - **Batch Normalization:** During training, BatchNorm computes mini-batch mean and variance. In `eval()` mode, it freezes the mini-batch calculation and uses the running population statistics learned during training.
            """
        )

    with st.expander("Q3: What does `torch.no_grad()` do and why is it essential?"):
        st.markdown(
            """
            **Answer:**
            `torch.no_grad()` deactivates the PyTorch autograd engine. Because we are only evaluating the model (forward pass) and not computing loss gradients for backpropagation, disabling gradient calculation reduces RAM usage by over 50% and dramatically speeds up feedforward inference.
            """
        )

    with st.expander("Q4: Explain the difference between Top-1 and Top-5 accuracy."):
        st.markdown(
            """
            **Answer:**
            - **Top-1 accuracy:** Measures the percentage of test samples where the single highest-probability predicted class matches the actual ground-truth label.
            - **Top-5 accuracy:** Measures the percentage where the correct label is anywhere among the model's top 5 predictions. In ImageNet with 1,000 fine-grained classes, Top-5 accuracy accounts for semantic ambiguities between closely related categories (e.g., Terrier breeds).
            """
        )

    with st.expander("Q5: தமிழ் / Tanglish Explanation (For Local Technical Clarifications)"):
        st.markdown(
            """
            **Simple Tanglish Summary for Interviews:**
            > *"Intha project-la naanga ImageNet dataset-la pre-trained aana deep learning models (MobileNetV3 / ResNet-50) use pannirukom. User upload panra image-ah standard $224 \\times 224$ size-ku convert panni, ImageNet mean and standard deviation vachu normalize pandrom.*
            > *Model forward pass mudinjathum, 1000 categories-ku logits kidaikkum. Athukku Softmax apply panni Top-5 probabilities extract pannuvom. Prediction results, inference speed (latency), and interactive Plotly visualization ellaame clean-ah Streamlit UI-la kaatrom."*
            """
        )
