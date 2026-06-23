import os
import streamlit as st

def set_custom_css():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1A0B2E 0%, #0B0014 100%);
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #D946EF;
            text-shadow: 0 0 10px rgba(217, 70, 239, 0.3);
        }
        .stButton>button {
            background: linear-gradient(45deg, #9333EA, #D946EF);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(147, 51, 234, 0.4);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(217, 70, 239, 0.6);
            color: white;
        }
        div[data-testid="stMetricValue"] {
            color: #C084FC;
            text-shadow: 0 0 10px rgba(192, 132, 252, 0.3);
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        .glass-card:hover {
            border-color: #D946EF;
            box-shadow: 0 0 15px rgba(217, 70, 239, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)
