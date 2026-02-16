# src/model.py
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Global variable to hold the loaded model
MODEL = None
# Define target image size based on your model's input requirements
IMAGE_SIZE = (64, 64) # Example for CIFAR-10, adjust as per your model
CLASS_LABELS = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9'] # Example for CIFAR-10

def load_model(model_path: str = None):
    global MODEL
    if MODEL is None:
        # Default model path, can be overridden by environment variable
        effective_model_path = model_path if model_path else os.environ.get("MODEL_PATH", "models/my_classifier_model.h5")
        if not os.path.exists(effective_model_path):
            raise FileNotFoundError(f"Model file not found at {effective_model_path}")
        MODEL = tf.keras.models.load_model(effective_model_path)
        # It's good practice to compile the model after loading, especially if you plan further training, or just for consistency.
        # MODEL.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return MODEL

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    # Decode image from bytes, convert to RGB, resize, and normalize pixel values.
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
        image_array = np.expand_dims(image_array, axis=0) # Add batch dimension (batch_size, height, width, channels)
        return image_array
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

def predict_image(preprocessed_image: np.ndarray):
    model = load_model() # Ensure model is loaded
    predictions = model.predict(preprocessed_image)
    # Convert raw predictions (e.g., softmax outputs) into meaningful class labels and probabilities.
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    probabilities = predictions[0].tolist() # Convert numpy array to list for JSON serialization
    
    return {
        "class_label": CLASS_LABELS[predicted_class_idx],
        "probabilities": probabilities
    }
