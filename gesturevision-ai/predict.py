import numpy as np
import cv2
import os
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
from PIL import Image

# Load model lazily
INTERPRETER = None
INPUT_DETAILS = None
OUTPUT_DETAILS = None

def get_interpreter():
    global INTERPRETER, INPUT_DETAILS, OUTPUT_DETAILS
    if INTERPRETER is None:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'saved_model', 'gesture_model.tflite')
            INTERPRETER = tflite.Interpreter(model_path=model_path)
            INTERPRETER.allocate_tensors()
            INPUT_DETAILS = INTERPRETER.get_input_details()
            OUTPUT_DETAILS = INTERPRETER.get_output_details()
        except Exception as e:
            print("Error loading TFLite model:", e)
            return str(e)
    return INTERPRETER

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
    interpreter = get_interpreter()
    if isinstance(interpreter, str):
        return f"Error: {interpreter}", 0.0, None
    if interpreter is None:
        return "Model not found", 0.0, None
        
    img = preprocess_image(image)
    
    # Run TFLite inference
    interpreter.set_tensor(INPUT_DETAILS[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(OUTPUT_DETAILS[0]['index'])[0]
    
    idx = np.argmax(preds)
    return LABELS[idx], float(preds[idx]) * 100, img
