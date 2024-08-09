"""Flask script for Intel image classification using TensorFlow model."""

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify
from pathlib import Path
from werkzeug.utils import secure_filename

# Flask app configuration
app = Flask(__name__)
app.debug = False

# Directory paths and model configuration
ROOT_DIR = Path(__file__).parent
IMAGE_DIR_PATH = os.path.join(ROOT_DIR, "images")
MODEL_PATH = os.path.join(ROOT_DIR, "model", "best_model.h5")
LABELS_PATH = os.path.join(ROOT_DIR, "class_id_label.npy")
IMAGE_SIZE = (150, 150)
MODEL = None

# Create the images directory if it doesn't exist
if not os.path.exists(IMAGE_DIR_PATH):
    os.makedirs(IMAGE_DIR_PATH)

def load_model():
    """Load the trained TensorFlow model."""
    global MODEL
    if MODEL is None:
        try:
            MODEL = keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

def get_labels() -> list:
    """Load class labels from a saved file or directory."""
    try:
        class_id_label = np.load(LABELS_PATH)
    except Exception:
        class_id_label = os.listdir(os.path.join(ROOT_DIR, "intel_images", "seg_train"))
        np.save(LABELS_PATH, class_id_label)
    return class_id_label

def predict_class(img_path: str) -> str:
    """
    Predict the output class for a given image.

    Args:
        img_path (str): Path to the image file.

    Returns:
        str: Predicted class label.
    """
    class_id_label = get_labels()
    load_model()

    image = load_img(img_path, color_mode="rgb", target_size=IMAGE_SIZE)
    input_arr = img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch

    predictions = MODEL.predict(input_arr)
    predicted_class = np.argmax(predictions)

    return class_id_label[predicted_class]

@app.route("/predict", methods=["POST"])
def get_result():
    """
    Handle image upload and return predicted class.
    
    Returns:
        JSON: A JSON response containing the predicted class label.
    """
    if 'image' not in request.files:
        return jsonify({
            "status": 400,
            "message": "No image file provided."
        }), 400
    
    upload_img = request.files["image"]
    filename = secure_filename(upload_img.filename)
    file_path = os.path.join(IMAGE_DIR_PATH, filename)
    
    try:
        upload_img.save(file_path)
        output = predict_class(file_path)
    except Exception as e:
        return jsonify({
            "status": 500,
            "message": f"Prediction failed: {e}"
        }), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({
        "status": 200,
        "predicted_class": output
    })

if __name__ == '__main__':
    app.run()
