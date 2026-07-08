import base64
import os
from io import BytesIO

from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError


app = Flask(__name__)
MODEL_FILE = "model.pkl"
scaler = None
model = None


def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"Canh bao: Khong tim thay file {MODEL_FILE}. API du doan se tam thoi khong kha dung.")
        return None, None

    try:
        model_bundle = joblib.load(MODEL_FILE)
    except Exception as error:
        print(f"Canh bao: Khong the load {MODEL_FILE}: {error}")
        return None, None

    loaded_scaler = model_bundle.get("scaler") if isinstance(model_bundle, dict) else None
    loaded_model = model_bundle.get("model") if isinstance(model_bundle, dict) else model_bundle

    if loaded_scaler is None or loaded_model is None:
        print(f"Canh bao: {MODEL_FILE} khong co du scaler va model.")
        return None, None

    return loaded_scaler, loaded_model


scaler, model = load_model()


def predict_digit(image_array):
    if scaler is None or model is None:
        raise RuntimeError("Model is not available.")

    image_scaled = scaler.transform(image_array.reshape(1, -1))
    prediction = model.predict(image_scaled)[0]
    return int(prediction)


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

    try:
        prediction = predict_digit(image_array)
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503

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

    try:
        prediction = predict_digit(image_array)
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
