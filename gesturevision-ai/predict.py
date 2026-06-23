import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os

# Load model lazily
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'saved_model', 'gesture_model.h5')
            MODEL = tf.keras.models.load_model(model_path)
        except Exception as e:
            print("Error loading model:", e)
            return str(e)
    return MODEL

LABELS = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb', '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']

def preprocess_image(image):
    is_bgr = True
    if isinstance(image, Image.Image):
        image = np.array(image)
        is_bgr = False
        
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to HSV based on whether it's BGR (webcam) or RGB (uploaded image)
        if is_bgr:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        # Detect skin color to isolate the hand
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Smooth the mask to create a pure silhouette of the hand
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        image = mask
        
    elif len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(image, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def predict_gesture(image):
    model = get_model()
    if isinstance(model, str):
        return f"Error: {model}", 0.0, None
    if model is None:
        return "Model not found", 0.0, None
    img = preprocess_image(image)
    preds = model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    return LABELS[idx], float(preds[idx]) * 100, img
