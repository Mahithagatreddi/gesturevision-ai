import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from predict import predict_gesture, LABELS
from utils import set_custom_css

st.set_page_config(page_title="GestureVision AI", layout="wide")
set_custom_css()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Gesture Predictor", "Live Webcam", "Analytics", "About Project"])

if page == "Dashboard":
    st.markdown("<h1 style='text-align: center; font-size: 3em;'>GestureVision AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #00E5FF;'>AI-Powered Hand Gesture Recognition</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='glass-card'><h3>Total Gestures</h3><p style='font-size: 2em; color: #00FF95;'>10</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'><h3>Dataset Size</h3><p style='font-size: 2em; color: #00FF95;'>20,000</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='glass-card'><h3>Model Accuracy</h3><p style='font-size: 2em; color: #00FF95;'>~98%</p></div>", unsafe_allow_html=True)

elif page == "Gesture Predictor":
    st.title("Gesture Predictor")
    st.markdown("<div class='glass-card'><p>Upload an image to predict the hand gesture.</p></div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                gesture, conf = predict_gesture(image)
                st.success(f"Predicted Gesture: {gesture}")
                st.info(f"Confidence: {conf:.2f}%")

elif page == "Live Webcam":
    st.title("Live Webcam Prediction")
    st.markdown("<div class='glass-card'><p>Real-time predictions using your webcam.<br><b>IMPORTANT: Place your hand inside the green square!</b></p></div>", unsafe_allow_html=True)
    
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video from webcam.")
                break
            
            # Flip frame horizontally for natural mirror effect
            frame = cv2.flip(frame, 1)
            
            # Define a Region of Interest (ROI) bounding box
            h, w, _ = frame.shape
            x1, y1 = int(w/2) - 150, int(h/2) - 150
            x2, y2 = x1 + 300, y1 + 300
            
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                # Predict ONLY on the cropped ROI
                gesture, conf, proc_img = predict_gesture(roi)
                
                # Draw the green ROI box on the main frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 149), 2)
                
                # Draw the prediction text above the box
                cv2.putText(frame, f"{gesture} ({conf:.1f}%)", (x1, y1 - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 149), 2)
                            
                # Overlay AI Vision (What the CNN actually sees)
                if proc_img is not None:
                    vis = (proc_img[0, :, :, 0] * 255).astype(np.uint8)
                    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
                    vis_rgb = cv2.resize(vis_rgb, (150, 150))
                    
                    # Place in top right corner
                    frame[10:160, frame.shape[1]-160:frame.shape[1]-10] = vis_rgb
                    cv2.putText(frame, "AI Vision", (frame.shape[1]-150, 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 149), 2)
            
            # Convert BGR to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            
        cap.release()
    else:
        st.write("Check the box to start the webcam.")

elif page == "Analytics":
    st.title("Model Analytics")
    st.markdown("<div class='glass-card'>Here you can view the training accuracy and loss graphs.</div>", unsafe_allow_html=True)
    
    try:
        history = np.load('saved_model/history.npy', allow_pickle=True).item()
        st.subheader("Training History")
        st.line_chart({"Accuracy": history.get('accuracy', []), "Val Accuracy": history.get('val_accuracy', [])})
        st.line_chart({"Loss": history.get('loss', []), "Val Loss": history.get('val_loss', [])})
    except:
        st.info("Train the model first to see analytics. 'history.npy' not found in 'saved_model' directory.")

elif page == "About Project":
    st.title("About GestureVision AI")
    st.markdown("""
    <div class='glass-card'>
    <h3>Problem Statement</h3>
    <p>Hand gesture recognition is crucial for human-computer interaction, sign language translation, and virtual reality.</p>
    
    <h3>Dataset</h3>
    <p>Using the Kaggle LeapGestRecog dataset featuring 10 different hand gestures.</p>
    
    <h3>Technology Used</h3>
    <ul>
        <li>Frontend: Streamlit</li>
        <li>Backend: Python, TensorFlow/Keras</li>
        <li>Computer Vision: OpenCV</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
