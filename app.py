"""
app.py
------
Flask backend for the Crop Disease Detection web app.

Routes:
  GET  /           → renders index.html (upload form)
  POST /predict    → handles image upload, returns JSON prediction
  GET  /health     → simple health-check endpoint

Run:
    python app.py
Then open http://localhost:5000
"""

import os
import uuid
import json
import traceback
from pathlib import Path

from flask import (Flask, render_template, request,
                   jsonify, send_from_directory)
from werkzeug.utils import secure_filename
from PIL import Image

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER   = os.path.join("static", "uploads")
ALLOWED_EXT     = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_MB  = 10
MODEL_AVAILABLE = False   # flipped to True once the model loads

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"]    = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
# ──────────────────────────────────────────────────────────────────────────────


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def validate_image(path: str) -> bool:
    """Verify the uploaded file is a real image (not just a renamed file)."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


# ─── Try importing the prediction module at startup ───────────────────────────
try:
    from predict import predict as run_predict, DISEASE_INFO
    # Warm-up: attempt to load model now so first request is fast
    # (will silently fail if model not yet trained)
    try:
        from predict import _load_model
        _load_model()
        MODEL_AVAILABLE = True
        print("[OK] ML model loaded and ready.")
    except FileNotFoundError as e:
        print(f"⚠️  Model not found: {e}")
        print("    Run 'python train.py' to train the model first.")
except ImportError as e:
    print(f"❌ Could not import predict.py: {e}")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the main upload page."""
    return render_template("index.html", model_ready=MODEL_AVAILABLE)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_ready": MODEL_AVAILABLE})


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Accepts a multipart POST with field 'file'.
    Returns JSON:
      {
        "success": true,
        "class_name": "Leaf_Blight",
        "confidence": 94.32,
        "all_probs": {...},
        "emoji": "🍂",
        "description": "...",
        "treatment": "...",
        "severity": "High",
        "color": "#ef4444",
        "image_url": "/static/uploads/abc123.jpg"
      }
    """
    # ── Validation ──
    if "file" not in request.files:
        return jsonify({"success": False,
                        "error": "No file part in request."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False,
                        "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False,
                        "error": f"File type not allowed. "
                                 f"Please upload: {', '.join(ALLOWED_EXT).upper()}"}), 400

    # ── Save to disk ──
    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # ── Validate image ──
    if not validate_image(filepath):
        os.remove(filepath)
        return jsonify({"success": False,
                        "error": "Uploaded file is not a valid image."}), 422

    # ── Run prediction ──
    if not MODEL_AVAILABLE:
        return jsonify({
            "success": False,
            "error": ("Model is not trained yet. "
                      "Run 'python train.py' first, then restart the server.")
        }), 503

    try:
        result = run_predict(filepath)
        result["success"]   = True
        result["image_url"] = f"/static/uploads/{filename}"
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False,
                        "error": f"Prediction failed: {str(e)}"}), 500


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ─── Error handlers ───────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False,
                    "error": f"File too large. Maximum size is {MAX_CONTENT_MB} MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", model_ready=MODEL_AVAILABLE), 404


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  🌿  Crop Disease Detection System")
    print("  📡  http://localhost:5000")
    print("═" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
