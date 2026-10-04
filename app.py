"""Plant Leaf Disease Classifier: Gradio demo.

Upload a tomato, potato, or pepper leaf image and the model predicts the
crop and disease, with a confidence score for every class.

Run with:  python app.py
"""

import glob
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter TensorFlow logs

import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import load_model

# ---------- Configuration ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMG_SIZE = (224, 224)  # MobileNetV3Large input size

# Class order used in training (alphabetical PlantVillage folder names).
# If a models/class_names.json file exists, it overrides this list.
DEFAULT_CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]


def find_model_path():
    """Look for the trained model in models/ (.keras or .h5)."""
    env_path = os.environ.get("MODEL_PATH")  # optional override
    if env_path and os.path.isfile(env_path):
        return env_path
    preferred = [
        os.path.join(MODELS_DIR, "single_step_model.keras"),
        os.path.join(MODELS_DIR, "single_step_model.h5"),
    ]
    for path in preferred:
        if os.path.isfile(path):
            return path
    found = sorted(
        glob.glob(os.path.join(MODELS_DIR, "*.keras"))
        + glob.glob(os.path.join(MODELS_DIR, "*.h5"))
    )
    return found[0] if found else None


def load_class_names():
    path = os.path.join(MODELS_DIR, "class_names.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CLASS_NAMES


# ---------- Load model once at startup ----------
MODEL_PATH = find_model_path()
if MODEL_PATH is None:
    sys.exit(
        "Trained model not found.\n"
        "Place your model file (e.g. single_step_model.keras) inside the "
        f"'models' folder:\n  {MODELS_DIR}"
    )

model = load_model(MODEL_PATH, compile=False)
class_names = load_class_names()

if model.output_shape[-1] != len(class_names):
    sys.exit(
        f"Class mismatch: the model has {model.output_shape[-1]} outputs but "
        f"{len(class_names)} class names were provided."
    )


# ---------- Prediction ----------
def predict(image):
    """Return {class_name: confidence} for the uploaded PIL image."""
    if image is None:
        return None

    # Nearest-neighbour resize matches Keras load_img, used during training
    img = image.convert("RGB").resize(IMG_SIZE, Image.NEAREST)
    img_array = preprocess_input(np.array(img, dtype="float32"))
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    probs = model.predict(img_array, verbose=0)[0]
    return {name: float(p) for name, p in zip(class_names, probs)}


# ---------- Web interface ----------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Plant Leaf Image"),
    outputs=gr.Label(num_top_classes=len(class_names), label="Predicted Disease"),
    title="Single-Step Plant Disease Classifier",
    description=(
        "Upload a leaf image (tomato, potato, or pepper). "
        "The model predicts the crop and disease in one step."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()