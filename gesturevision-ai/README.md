# GestureVision AI

## Project Overview
GestureVision AI is a complete AI-powered Hand Gesture Recognition web application. It uses a lightweight Convolutional Neural Network (CNN) to classify 10 different hand gestures from the Kaggle LeapGestRecog dataset. 

## Features
- Dashboard with dataset statistics
- Image Upload Predictor
- Live Webcam Real-time Prediction
- Training Analytics Dashboard
- Custom Cyber Neon Dark Theme

## Installation
```bash
git clone <repository_url>
cd gesturevision-ai
pip install -r requirements.txt
```

## Run Locally
1. Download the LeapGestRecog dataset from Kaggle and place it in the `dataset/` folder within the project root.
2. Train the model (optional if a pre-trained model is already in `saved_model/`):
```bash
python train_model.py
```
3. Run the Streamlit application:
```bash
streamlit run app.py
```

## Deployment Steps
### Streamlit Cloud / Hugging Face Spaces
1. Push this repository to your GitHub account.
2. Connect your GitHub repository to Streamlit Cloud or Hugging Face Spaces.
3. Ensure `requirements.txt` and `packages.txt` are in the root directory.
4. Set `app.py` as the entrypoint.

## Future Improvements
- Expand dataset to include more complex gestures.
- Optimize model using TFLite for faster edge predictions.
- Add multi-hand support.
