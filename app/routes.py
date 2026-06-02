"""
Flask route definitions.
"""

from flask import Blueprint, render_template, request, jsonify
from .model import preprocess_image, run_inference

main = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@main.route("/")
def index():
    return render_template("index.html")


@main.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not _allowed(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a PNG, JPG, BMP or TIFF image."}), 400

    try:
        file_bytes = file.read()
        img_array = preprocess_image(file_bytes)
        result = run_inference(img_array)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500
