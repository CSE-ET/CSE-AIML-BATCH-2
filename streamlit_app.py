import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import io
import base64
from rice_disease_model import RiceDiseasePredictor
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Rice Leaf Disease Prediction",
    page_icon="��",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #2E8B57;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        predictor = RiceDiseasePredictor()
        if os.path.exists("rice_disease_model.h5"):
            model = predictor.load_model("rice_disease_model.h5")
            return predictor, model
        else:
            st.error("Model file not found. Please run the training script first.")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def get_disease_info(disease_name):
    """Get information about the predicted disease"""
    disease_info = {
        "Bacterial leaf blight": {
            "description": "A deadly bacterial disease caused by Xanthomonas oryzae",
            "symptoms": ["Water-soaked streaks", "Grayish white lesions", "Milky ooze"],
            "management": ["Use resistant varieties", "Field hygiene", "Early detection"]
        },
        "Brown spot": {
            "description": "A fungal disease caused by Cochliobolus miyabeanus",
            "symptoms": ["Big spots on leaves", "Unfilled grains", "Discolored seeds"],
            "management": ["Resistant varieties", "Proper fertilization", "Crop rotation"]
        },
        "Leaf smut": {
            "description": "A fungal disease caused by Entyloma oryzae",
            "symptoms": ["Black smut sori", "Yellow areas", "Reduced photosynthesis"],
            "management": ["Resistant varieties", "Proper spacing", "Balanced fertilization"]
        }
    }
    return disease_info.get(disease_name, {})

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Paddy Leaf Disease Prediction</h1>', unsafe_allow_html=True)
    
    # Load model
    predictor, model = load_model()
    
    if predictor is None or model is None:
        st.error("Model not loaded. Please ensure the model file exists.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a rice leaf image...",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("Prediction Results")
            
            # Save uploaded file temporarily
            temp_path = "temp_image.jpg"
            image.save(temp_path)
            
            try:
                # Make prediction
                result = predictor.predict_image(temp_path, model)
                
                # Display results
                st.markdown(f"""
                <div class="prediction-box">
                <h3>Predicted Disease: {result['class']}</h3>
                <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show disease information
                disease_info = get_disease_info(result['class'])
                if disease_info:
                    st.markdown(f"**Description:** {disease_info['description']}")
                    st.markdown("**Symptoms:**")
                    for symptom in disease_info['symptoms']:
                        st.markdown(f"• {symptom}")
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    
    # Sidebar with information
    st.sidebar.markdown("## About")
    st.sidebar.markdown("""
    This application uses AI to classify rice leaf diseases:
    - Bacterial Leaf Blight
    - Brown Spot  
    - Leaf Smut
    """)
    
    st.sidebar.markdown("## Model Performance")
    st.sidebar.metric("Training Accuracy", "99.7%")
    st.sidebar.metric("Validation Accuracy", "70%")

if __name__ == "__main__":
    main() 