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
    st.markdown("<div class='glass-card'><p>Choose your webcam mode below. <b>True Live Video</b> only works when running on your local PC!</p></div>", unsafe_allow_html=True)
    
    mode = st.radio("Select Webcam Mode:", ["Take a Photo (Works on Cloud)", "True Live Video (Local PC Only)"])
    
    if mode == "Take a Photo (Works on Cloud)":
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            with st.spinner("Analyzing gesture..."):
                gesture, conf, proc_img = predict_gesture(image)
                st.markdown(f"<div class='glass-card'><h2 style='color: #00FF95; text-align: center;'>Prediction: {gesture}</h2><h4 style='text-align: center;'>Confidence: {conf:.1f}%</h4></div>", unsafe_allow_html=True)
                if proc_img is not None:
                    vis = (proc_img[0, :, :, 0] * 255).astype(np.uint8)
                    st.image(vis, width=200, caption="AI Vision")
    else:
        st.info("💡 To use this mode, open your computer's terminal and run: `streamlit run app.py`")
        run = st.checkbox('Start Live Webcam')
        FRAME_WINDOW = st.image([])
        
        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("🚨 CRITICAL ERROR: Could not access a physical webcam! You are running this on Streamlit Cloud (which has no webcam). Please run this app locally on your PC to use Live Video!")
            else:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture video.")
                        break
                    
                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape
                    x1, y1 = int(w/2) - 150, int(h/2) - 150
                    x2, y2 = x1 + 300, y1 + 300
                    
                    roi = frame[y1:y2, x1:x2]
                    if roi.shape[0] > 0 and roi.shape[1] > 0:
                        gesture, conf, _ = predict_gesture(roi)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 149), 2)
                        cv2.putText(frame, f"{gesture} ({conf:.1f}%)", (x1, y1 - 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 149), 2)
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frame)
                cap.release()

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
