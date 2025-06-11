from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import tensorflow as tf

# Debug print
print(f"TensorFlow version: {tf.__version__}")

# Load model
try:
    model_path = os.path.join("weights", "keras_Model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path}")
    if not os.path.exists(os.path.join(model_path, "saved_model.pb")):
        raise FileNotFoundError(f"saved_model.pb not found in {model_path}")
    if not os.path.exists(os.path.join(model_path, "variables")):
        raise FileNotFoundError(f"variables directory not found in {model_path}")
    
    print(f"Loading model from {model_path}")
    model = load_model(model_path, compile=False)
    print("Model loaded successfully")

except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Load labels
with open(os.path.join("weights", "labels.txt"), "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def predict_image(img_pil):
    size = (224, 224)
    image = ImageOps.fit(img_pil, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = round(float(prediction[0][index]) * 100, 2)

    if class_name.strip().lower() in ["good/perfect", "perfect", "good"]:
        return f"✅ This is a **Good** product. (Confidence: {confidence_score}%)"
    else:
        return f"⚠️ Detected an **Abnormal** product. (Confidence: {confidence_score}%)"
