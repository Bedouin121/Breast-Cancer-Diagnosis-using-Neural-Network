"""
Model loading and inference utilities.

The saved model (model.keras) is a U-Net with two outputs:
  - seg_output : (256, 256, 1) sigmoid mask – highlights the tumour region
  - cls_output : (3,) softmax probabilities – Normal / Benign / Malignant
"""

import io
import base64
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe for Flask
import matplotlib.pyplot as plt

# ── constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = "model.keras"
INPUT_SIZE = (256, 256)
CLASS_LABELS = ["Normal", "Benign", "Malignant"]
CLASS_COLORS = {
    "Normal":    "#22c55e",   # green
    "Benign":    "#f59e0b",   # amber
    "Malignant": "#ef4444",   # red
}
SEG_THRESHOLD = 0.5           # pixel threshold for the binary mask overlay

# ── singleton model ─────────────────────────────────────────────────────────────
_model = None


def load_model():
    """Load the Keras model once and cache it."""
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


# ── preprocessing ───────────────────────────────────────────────────────────────

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Convert uploaded image bytes → (1, 256, 256, 1) float32 array
    suitable for the model.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("L")   # grayscale
    img = img.resize(INPUT_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)   # (256, 256, 1)
    arr = np.expand_dims(arr, axis=0)    # (1, 256, 256, 1)
    return arr


# ── inference ───────────────────────────────────────────────────────────────────

def run_inference(img_array: np.ndarray) -> dict:
    """
    Run the model and return a dict with:
      - label        : predicted class string
      - confidence   : float 0-100
      - probabilities: dict {label: pct}
      - overlay_b64  : base64-encoded PNG of original + segmentation overlay
      - mask_b64     : base64-encoded PNG of raw mask heatmap
    """
    model = load_model()
    predictions = model.predict(img_array, verbose=0)

    seg_map = predictions["seg_output"][0, :, :, 0]   # (256, 256)
    cls_probs = predictions["cls_output"][0]           # (3,)

    pred_idx = int(np.argmax(cls_probs))
    label = CLASS_LABELS[pred_idx]
    confidence = float(cls_probs[pred_idx]) * 100

    probabilities = {CLASS_LABELS[i]: round(float(cls_probs[i]) * 100, 1)
                     for i in range(len(CLASS_LABELS))}

    original_gray = (img_array[0, :, :, 0] * 255).astype(np.uint8)
    overlay_b64 = _build_overlay(original_gray, seg_map, label)
    mask_b64 = _build_heatmap(seg_map)

    return {
        "label":         label,
        "confidence":    round(confidence, 1),
        "probabilities": probabilities,
        "overlay_b64":   overlay_b64,
        "mask_b64":      mask_b64,
        "color":         CLASS_COLORS[label],
    }


# ── visualisation helpers ────────────────────────────────────────────────────────

def _build_overlay(gray: np.ndarray, seg_map: np.ndarray, label: str) -> str:
    """
    Blend the original grayscale image with a coloured segmentation mask.
    Returns a base64-encoded PNG string.
    """
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    mask_bin = (seg_map >= SEG_THRESHOLD).astype(np.uint8)

    # colour for the overlay depends on the class
    overlay_rgb = np.zeros_like(rgb)
    if label == "Normal":
        overlay_rgb[:, :, 1] = mask_bin * 220   # green
    elif label == "Benign":
        overlay_rgb[:, :, 0] = mask_bin * 245   # orange-ish (red channel)
        overlay_rgb[:, :, 1] = mask_bin * 158
    else:                                        # Malignant → red
        overlay_rgb[:, :, 0] = mask_bin * 239

    blended = cv2.addWeighted(rgb, 0.75, overlay_rgb, 0.45, 0)

    # draw contour for clarity
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_color = (34, 197, 94) if label == "Normal" else \
                    (245, 158, 11) if label == "Benign" else \
                    (239, 68, 68)
    cv2.drawContours(blended, contours, -1, contour_color, 2)

    return _ndarray_to_b64(blended)


def _build_heatmap(seg_map: np.ndarray) -> str:
    """Return a JET colourmap heatmap of the raw segmentation probabilities."""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(seg_map, cmap="jet", vmin=0, vmax=1)
    ax.set_title("Tumour Probability Map", fontsize=10, color="white")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.patch.set_facecolor("#1e293b")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _ndarray_to_b64(arr: np.ndarray) -> str:
    """Encode a numpy uint8 RGB array as a base64 PNG string."""
    img_pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
