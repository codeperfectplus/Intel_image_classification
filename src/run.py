""" Flask script for Intel image classification using tensorflow model """

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify
from pathlib import Path
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.debug = False

root_dir = Path(__file__).parent
image_dir_path = os.path.join(root_dir, "images")

if not os.path.isdir(image_dir_path):
    os.mkdir(image_dir_path)

def get_labels() -> list:
    try:
        class_id_label = np.load("class_id_label.npy")
    except Exception:
        class_id_label = os.listdir(r"intel_images/seg_train")
        np.save("class_id_label.npy", class_id_label)

    return class_id_label


def predict_class(img_path: str) -> str:
    """
    Predict the output class

    args:
        img_path:
            str like image path for classification
    
    output:
        output class
     """
    class_id_label = get_labels()
    model = keras.models.load_model("tmp")

    image = load_img(img_path, color_mode="rgb", target_size=(150, 150))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr]) # converting single image to batch

    predictions = model.predict(input_arr)
    predictions = np.argmax(predictions)

    return class_id_label[predictions]

@app.route("/predict", methods=["POST"])
def get_result():
    upload_img = request.files["image"]
    filename = secure_filename(upload_img.filename)

    file_path = os.path.join(image_dir_path, filename)
    upload_img.save(file_path)

    output = predict_class(file_path)
    os.remove(file_path)

    return jsonify({
        "status": 200,
        "Predicted class": output})

if __name__ == '__main__':
    app.run()