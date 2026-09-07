"""
predict.py
----------
Standalone prediction module used by app.py.

Usage (standalone test):
    python predict.py path/to/leaf.jpg
"""

import os
import json
import numpy as np
from PIL import Image

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE    = 128
MODEL_PATH  = os.path.join("model", "crop_disease_model.h5")
CLASSES_PATH= os.path.join("model", "class_names.json")

# Disease metadata shown in the UI
DISEASE_INFO = {
    "Healthy": {
        "emoji": "🌿",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "treatment": "No treatment needed. Continue regular care and monitoring.",
        "severity": "None",
        "color": "#22c55e",
    },
    "Powdery_Mildew": {
        "emoji": "🌫️",
        "description": "White or grey powdery spots on leaf surfaces caused by fungal spores.",
        "treatment": "Apply neem oil or sulfur-based fungicide. Improve air circulation.",
        "severity": "Moderate",
        "color": "#f59e0b",
    },
    "Leaf_Blight": {
        "emoji": "🍂",
        "description": "Large irregular brown or tan lesions that cause premature leaf drop.",
        "treatment": "Remove infected leaves. Apply copper-based fungicide spray.",
        "severity": "High",
        "color": "#ef4444",
    },
    "Rust": {
        "emoji": "🔴",
        "description": "Orange-red pustules on leaf undersides caused by rust fungi.",
        "treatment": "Apply fungicide at first sign. Avoid overhead watering.",
        "severity": "High",
        "color": "#f97316",
    },
    "Bacterial_Spot": {
        "emoji": "🔵",
        "description": "Small dark angular water-soaked spots caused by bacterial infection.",
        "treatment": "Use copper bactericide sprays. Remove and destroy infected plant tissue.",
        "severity": "Moderate",
        "color": "#8b5cf6",
    },
}
# ──────────────────────────────────────────────────────────────────────────────


# ─── Model singleton (lazy-loaded) ───────────────────────────────────────────
_model       = None
_class_names = None


def _load_model():
    """Load model and class names once, cache them."""
    global _model, _class_names
    if _model is not None:
        return _model, _class_names

    import tensorflow as tf  # import here so the module is importable even without TF

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Please run 'python train.py' first."
        )
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(
            f"Class names file not found at '{CLASSES_PATH}'."
        )

    _model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH) as f:
        raw = json.load(f)
    # Stored as {index_str: class_name}; convert keys to int
    _class_names = {int(k): v for k, v in raw.items()}

    print(f"[predict] Model loaded. Classes: {list(_class_names.values())}")
    return _model, _class_names


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk, resize to (IMG_SIZE × IMG_SIZE),
    normalise to [0, 1], and add the batch dimension.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 128, 128, 3)


def predict(image_path: str) -> dict:
    """
    Predict the disease class for a single leaf image.

    Returns
    -------
    dict with keys:
        class_name   : str   — predicted disease label
        confidence   : float — confidence in [0, 100]
        all_probs    : dict  — {class_name: confidence%} for all classes
        emoji        : str
        description  : str
        treatment    : str
        severity     : str
        color        : str   — hex color for the UI badge
    """
    model, class_names = _load_model()

    img_array = preprocess_image(image_path)
    preds     = model.predict(img_array, verbose=0)[0]   # shape: (num_classes,)

    top_idx       = int(np.argmax(preds))
    top_class     = class_names[top_idx]
    top_conf      = float(preds[top_idx]) * 100

    all_probs = {
        class_names[i]: round(float(preds[i]) * 100, 2)
        for i in range(len(preds))
    }

    info = DISEASE_INFO.get(top_class, {
        "emoji": "🔍",
        "description": "Unknown disease detected.",
        "treatment": "Consult an agricultural expert.",
        "severity": "Unknown",
        "color": "#6b7280",
    })

    return {
        "class_name": top_class,
        "confidence": round(top_conf, 2),
        "all_probs":  all_probs,
        **info,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    result = predict(sys.argv[1])
    print("\n" + "=" * 40)
    print(f"  Prediction : {result['emoji']}  {result['class_name']}")
    print(f"  Confidence : {result['confidence']:.2f}%")
    print(f"  Severity   : {result['severity']}")
    print(f"  Treatment  : {result['treatment']}")
    print("=" * 40)
    print("\nAll class probabilities:")
    for cls, prob in sorted(result["all_probs"].items(),
                            key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob / 5)
        print(f"  {cls:<20} {prob:6.2f}%  {bar}")
