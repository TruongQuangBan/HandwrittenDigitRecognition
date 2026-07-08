import base64
import os
from io import BytesIO

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from PIL import Image, UnidentifiedImageError


app = Flask(__name__)
model_params = np.load("model_params.npz")
scaler_mean = model_params["mean"]
scaler_scale = model_params["scale"]
model_coef = model_params["coef"]
model_intercept = model_params["intercept"]
model_classes = model_params["classes"]


def predict_digit(image_array):
    image_scaled = (image_array - scaler_mean) / scaler_scale
    scores = model_coef @ image_scaled + model_intercept
    return int(model_classes[np.argmax(scores)])


def decode_base64_image(image_base64):
    if not isinstance(image_base64, str) or not image_base64.strip():
        raise ValueError("Field 'image_base64' must be a non-empty base64 string.")

    encoded_image = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64

    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (ValueError, UnidentifiedImageError, OSError):
        raise ValueError("Field 'image_base64' must contain a valid base64 image.")

    return image


def image_to_model_input(image):
    image = image.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=float).reshape(-1)


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    image = data.get("image")

    if not isinstance(image, list) or len(image) != 784:
        return jsonify({"error": "Field 'image' must be an array of 784 numbers."}), 400

    try:
        image_array = np.array(image, dtype=float)
    except (TypeError, ValueError):
        return jsonify({"error": "Field 'image' must contain only numbers."}), 400

    prediction = predict_digit(image_array)

    return jsonify({"prediction": prediction})


@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    data = request.get_json(silent=True) or {}

    if "image_base64" not in data:
        return jsonify({"error": "Field 'image_base64' is required."}), 400

    try:
        image = decode_base64_image(data["image_base64"])
        image_array = image_to_model_input(image)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    prediction = predict_digit(image_array)

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
