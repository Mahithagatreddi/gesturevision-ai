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
    st.title("Webcam Prediction")
    st.markdown("<div class='glass-card'><p>Take a picture using your webcam to predict the hand gesture. Make sure your hand is well-lit and centered!</p></div>", unsafe_allow_html=True)
    
    # Use Streamlit's native camera input which securely accesses the user's browser webcam!
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer is not None:
        # Read the image from the browser webcam
        image = Image.open(img_file_buffer)
        
        with st.spinner("Analyzing gesture..."):
            gesture, conf, proc_img = predict_gesture(image)
            
            # Display results beautifully
            st.markdown(f"<div class='glass-card'><h2 style='color: #00FF95; text-align: center;'>Prediction: {gesture}</h2><h4 style='text-align: center;'>Confidence: {conf:.1f}%</h4></div>", unsafe_allow_html=True)
            
            # Show the AI Vision (What the CNN actually sees)
            if proc_img is not None:
                st.write("**AI Vision (What the neural network processed):**")
                vis = (proc_img[0, :, :, 0] * 255).astype(np.uint8)
                st.image(vis, width=200, caption="Preprocessed Silhouette")

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
