"""
Utility functions for Project 1: AI Image Classification Studio.
Provides sample image management, interactive visual charts, and export generators.
"""

import io
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go

from config_task3 import AVAILABLE_MODELS


SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def ensure_sample_images_exist() -> Dict[str, str]:
    """
    Ensures sample images exist in the samples/ directory for instant 1-click testing.
    If default sample files are missing, synthesizes clear, labeled test images.

    Returns:
        Dictionary mapping display name to absolute file path.
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    samples_map = {
        "Golden Retriever (Domestic Dog)": os.path.join(SAMPLES_DIR, "golden_retriever.jpg"),
        "Sports Car (Automobile)": os.path.join(SAMPLES_DIR, "sports_car.jpg"),
        "Coffee Cup (Tableware)": os.path.join(SAMPLES_DIR, "coffee_cup.jpg"),
        "Airliner (Aircraft)": os.path.join(SAMPLES_DIR, "airliner.jpg"),
    }

    # Generate synthetic photographic proxy if file doesn't exist yet
    for label, path in samples_map.items():
        if not os.path.exists(path):
            _create_default_sample_image(path, label)

    return samples_map


def _create_default_sample_image(filepath: str, label_hint: str) -> None:
    """Creates an illustrative high-contrast test image with geometric patterns and text."""
    width, height = 400, 400

    # Color palette based on sample type
    if "Retriever" in label_hint:
        bg_color = (220, 160, 90)
        fg_color = (139, 69, 19)
        shape_type = "dog"
    elif "Sports Car" in label_hint:
        bg_color = (30, 30, 40)
        fg_color = (220, 40, 40)
        shape_type = "car"
    elif "Coffee Cup" in label_hint:
        bg_color = (235, 230, 220)
        fg_color = (101, 67, 33)
        shape_type = "cup"
    else:
        bg_color = (70, 130, 180)
        fg_color = (240, 248, 255)
        shape_type = "plane"

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw simple representative geometry
    if shape_type == "dog":
        draw.ellipse([100, 100, 300, 300], fill=fg_color, outline=(90, 40, 10), width=4)
        draw.ellipse([80, 80, 140, 170], fill=(110, 50, 15))  # Ear left
        draw.ellipse([260, 80, 320, 170], fill=(110, 50, 15))  # Ear right
        draw.ellipse([150, 170, 175, 195], fill=(20, 20, 20))  # Eye
        draw.ellipse([225, 170, 250, 195], fill=(20, 20, 20))  # Eye
        draw.ellipse([185, 220, 215, 250], fill=(10, 10, 10))  # Nose
    elif shape_type == "car":
        draw.rectangle([60, 200, 340, 280], fill=fg_color, outline=(255, 255, 255), width=3)
        draw.polygon([(110, 200), (160, 130), (280, 130), (320, 200)], fill=(80, 10, 10))
        draw.ellipse([90, 260, 150, 320], fill=(10, 10, 10), outline=(180, 180, 180), width=4)
        draw.ellipse([250, 260, 310, 320], fill=(10, 10, 10), outline=(180, 180, 180), width=4)
    elif shape_type == "cup":
        draw.rectangle([120, 160, 260, 310], fill=fg_color, outline=(70, 40, 20), width=4)
        draw.arc([230, 190, 310, 280], start=270, end=90, fill=(70, 40, 20), width=10)
        draw.ellipse([100, 300, 280, 330], fill=(180, 180, 180))  # Saucer
    else:  # plane
        draw.polygon([(200, 70), (230, 320), (170, 320)], fill=fg_color)  # Fuselage
        draw.polygon([(200, 160), (360, 260), (40, 260)], fill=(210, 225, 240))  # Wings
        draw.polygon([(200, 300), (260, 340), (140, 340)], fill=(210, 225, 240))  # Tail

    # Text caption
    draw.rectangle([10, 350, 390, 390], fill=(0, 0, 0, 180))
    draw.text((25, 362), f"Sample: {label_hint}", fill=(255, 255, 255))

    img.save(filepath, format="JPEG", quality=90)


def create_probability_chart(predictions: List[Dict[str, Any]]) -> go.Figure:
    """
    Generates a horizontal Plotly bar chart of Top-K prediction probabilities.
    Uses dynamic color grading (green for high confidence, blue for moderate).
    """
    # Reverse so top rank appears at the top of the horizontal bar chart
    labels = [p["label"] for p in reversed(predictions)]
    probs = [p["probability"] * 100 for p in reversed(predictions)]
    percentages = [p["percentage"] for p in reversed(predictions)]

    colors = []
    for val in probs:
        if val >= 70.0:
            colors.append("#10B981")  # Emerald Green
        elif val >= 30.0:
            colors.append("#3B82F6")  # Royal Blue
        elif val >= 10.0:
            colors.append("#F59E0B")  # Amber
        else:
            colors.append("#6B7280")  # Slate Gray

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=labels,
            orientation="h",
            text=percentages,
            textposition="auto",
            marker=dict(
                color=colors,
                line=dict(color="#1F2937", width=1),
            ),
            hovertemplate="<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Top Predicted Classes & Softmax Probabilities</b>",
        xaxis_title="Confidence Probability (%)",
        xaxis=dict(range=[0, 105], gridcolor="#E5E7EB"),
        yaxis=dict(autorange="reversed" if False else True),
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=13),
    )

    return fig


def get_models_comparison_df() -> pd.DataFrame:
    """Returns a pandas DataFrame comparing the available models for documentation."""
    rows = []
    for name, spec in AVAILABLE_MODELS.items():
        rows.append({
            "Model Name": spec.name,
            "Architecture": spec.architecture,
            "Size (MB)": f"{spec.approx_size_mb:.1f} MB",
            "Top-1 Acc (%)": f"{spec.top1_accuracy:.1f}%",
            "Top-5 Acc (%)": f"{spec.top5_accuracy:.1f}%",
            "Target Use Case": spec.target_use_case,
        })
    return pd.DataFrame(rows)


def build_export_json(
    metadata: Dict[str, Any],
    inference_result: Dict[str, Any],
    model_name: str,
) -> str:
    """Formats full prediction session metadata and results into a JSON string."""
    payload = {
        "internship": {
            "organization": "CodeOrbit Tech",
            "task": "Task 3: Image Classification Using a Pretrained Model",
            "project": "AI Image Classification Studio",
        },
        "model_selected": model_name,
        "image_metadata": metadata,
        "inference_latency_ms": inference_result.get("inference_time_ms"),
        "top_predictions": inference_result.get("top_predictions"),
    }
    return json.dumps(payload, indent=2)


def build_export_csv(inference_result: Dict[str, Any], model_name: str) -> str:
    """Formats prediction results into a CSV string for tabular analysis."""
    rows = []
    for p in inference_result.get("top_predictions", []):
        rows.append({
            "Rank": p["rank"],
            "Model": model_name,
            "ImageNet_Class_ID": p["class_id"],
            "Label": p["label"],
            "Probability": p["probability"],
            "Percentage": p["percentage"],
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)
