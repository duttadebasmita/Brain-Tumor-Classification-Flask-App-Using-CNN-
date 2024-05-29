import cv2
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('C:\\repo2\\my_BrainTumor10EpochsCategorical.keras')

def preprocess_image(image_path, target_size=(64, 64)):
    """Load and preprocess the image."""
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Convert the image to RGB (OpenCV loads images in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the target size
    image = Image.fromarray(image).resize(target_size)
    
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Normalize the image array (scale pixel values to [0, 1])
    image_array = image_array / 255.0
    
    # Expand dimensions to match the model input shape (1, 64, 64, 3)
    input_img = np.expand_dims(image_array, axis=0)
    
    return input_img

def get_class_name(image_path):
    """Predict if the image contains a tumor."""
    # Preprocess the image
    input_img = preprocess_image(image_path)
    
    # Predict using the model
    result = model.predict(input_img)
    
    # Assuming the model output is a single probability for the "tumor" class
    tumor_probability = result[0][1]  # Adjust index based on the model's output shape
    
    # Define a threshold for classification
    threshold = 0.5
    
    if tumor_probability >= threshold:
        return "Tumor detected."
    else:
        return "No tumor detected."

# Example usage
image_path = 'C:\\repo2\\Brain-Tumor-Classification\\pred\\pred20.jpg'
result = get_class_name(image_path)
print(result)
