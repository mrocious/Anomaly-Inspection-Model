import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Debug print (will show in console when run locally)
print(f"TensorFlow version: {tf.__version__}")

# --- Model and Label Loading ---
# Paths are relative to this script's location in the repository.
# Ensure your repository has the following structure:
# your_repo/
# ├── predict_tf.py
# ├── Weights/  <--- Note the capital 'W' as provided
# │   └── converted_savedmodel/
# │       ├── model.savedmodel/ (contains saved_model.pb and variables/)
# │       └── labels.txt         <--- labels.txt is directly here
# └── sample_images/ (for local testing)
#     └── default_test_image.jpg

# Corrected paths based on your input:
MODEL_PATH_DIR = os.path.join("Weights", "converted_savedmodel", "model.savedmodel")
LABELS_FILE_PATH = os.path.join("Weights", "converted_savedmodel", "labels.txt")


# Load model
try:
    if not os.path.exists(MODEL_PATH_DIR):
        raise FileNotFoundError(f"Model directory not found at {MODEL_PATH_DIR}")
    if not os.path.exists(os.path.join(MODEL_PATH_DIR, "saved_model.pb")):
        raise FileNotFoundError(f"saved_model.pb not found in {MODEL_PATH_DIR}")
    if not os.path.exists(os.path.join(MODEL_PATH_DIR, "variables")):
        # Check for 'variables' subdirectory typical of a SavedModel
        # This check might need adjustment if your SavedModel has a different structure
        # or if it's a frozen graph where variables are embedded.
        if not os.path.isdir(os.path.join(MODEL_PATH_DIR, "variables")):
            print(f"Warning: 'variables' directory not found in {MODEL_PATH_DIR}. "
                  "This might be a frozen graph or a different SavedModel format.")
    
    print(f"Loading model from {MODEL_PATH_DIR}")
    # Using load_model as your original script uses it for Keras SavedModels
    model = load_model(MODEL_PATH_DIR, compile=False)
    print("Model loaded successfully")

except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Load labels
try:
    with open(LABELS_FILE_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(class_names)} class labels from {LABELS_FILE_PATH}")
except Exception as e:
    print(f"Error loading labels file: {str(e)}")
    raise

# --- Prediction Function ---
def predict_image(img_pil: Image.Image) -> str:
    """
    Performs image preprocessing and prediction using the loaded TensorFlow/Keras model.

    Args:
        img_pil (PIL.Image.Image): The input image as a PIL Image object.

    Returns:
        str: A formatted message indicating "Good" or "Abnormal" product status,
             along with the confidence score.
    """
    size = (224, 224)
    # Resize and crop the image to the target size (224x224)
    image = ImageOps.fit(img_pil, size, Image.Resampling.LANCZOS)
    
    # Convert image to numpy array
    image_array = np.asarray(image)

    # Normalize image from [0, 255] to [-1, 1] as per model's training
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Add batch dimension: (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    # The model's raw output (logits)
    raw_prediction = model.predict(data)
    
    # Convert raw prediction to probabilities using softmax
    probabilities = tf.nn.softmax(raw_prediction).numpy()[0] # Get probabilities for the single image
    
    # Determine predicted class and confidence
    predicted_index = np.argmax(probabilities)
    predicted_class_name = class_names[predicted_index]
    confidence_score = round(float(probabilities[predicted_index]) * 100, 2)

    # Format result message based on your specific logic
    if predicted_class_name.strip().lower() in ["good/perfect", "perfect", "good"]:
        return f"✅ This is a **Good** product. (Confidence: {confidence_score}%)"
    else:
        # Include the raw class name in the "Abnormal" message for more detail
        return f"⚠️ Detected an **Abnormal** product (Class: {predicted_class_name}). (Confidence: {confidence_score}%)"

# --- Local Testing Example ---
# This block runs only when the script is executed directly (e.g., `python predict_tf.py`)
# It will NOT run when this script is imported as a module by another script (e.g., app.py)
if __name__ == "__main__":
    print("\n--- Running Local Test Example ---")
    
    # Define a path to a test image relative to this script
    # Make sure to place a test image in 'sample_images/' if this path is used
    TEST_IMAGE_LOCAL_PATH = "sample_images/default_test_image.jpg"

    # --- Create a dummy image if no test image exists for demonstration ---
    if not os.path.exists(TEST_IMAGE_LOCAL_PATH):
        print(f"WARNING: Test image not found at '{TEST_IMAGE_LOCAL_PATH}'. Creating a dummy image for demonstration.")
        os.makedirs(os.path.dirname(TEST_IMAGE_LOCAL_PATH), exist_ok=True) # Ensure directory exists
        dummy_img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_img_array)
        dummy_image.save(TEST_IMAGE_LOCAL_PATH)
        print(f"Dummy image saved to {TEST_IMAGE_LOCAL_PATH}")
    
    try:
        # Load the test image using PIL
        test_image_pil = Image.open(TEST_IMAGE_LOCAL_PATH).convert('RGB')
        print(f"Processing test image: {TEST_IMAGE_LOCAL_PATH}")

        # Call the prediction function
        result = predict_image(test_image_pil)
        
        # Print the result
        print("\nPrediction Result:")
        print(result)

    except Exception as e:
        print(f"\nERROR during local test prediction: {e}")
        print("Please ensure your TEST_IMAGE_LOCAL_PATH is correct and the image is valid.")