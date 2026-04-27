"""About Page"""

import streamlit as st

def show():
    """Display about page"""
    
    st.markdown("""
    ## ℹ️ About This System
    
    **🤖 AI-Powered Disease Prediction System**
    
    This system uses machine learning to predict diseases based on patient symptoms, organized into medical categories.
    
    ---
    
    ### ✨ Features
    
    - 🎯 **Real-time Disease Prediction** - Get Top 7 possible diseases with confidence scores
    - 🔬 **Symptom Pattern Analyzer** - Compare symptom patterns between two diseases
    - 🩺 **12 Symptom Categories** - Organized by medical specialty
    - 📊 **Interactive Visualizations** - Charts and heatmaps for better understanding
    - 🤖 **Random Forest ML Model** - 82.2% accuracy on test data
    
    ---
    
    ### 🩺 Symptom Categories
    
    | Category | Symptoms |
    |----------|----------|
    | 🧠 Mental & Emotional | 13 symptoms |
    | ❤️ Cardiovascular | 8 symptoms |
    | 🫁 Respiratory | 11 symptoms |
    | 🍽️ Digestive | 13 symptoms |
    | 🧠 Neurological | 8 symptoms |
    | 🚽 Genitourinary | 12 symptoms |
    | 🦴 Musculoskeletal | 13 symptoms |
    | 🩻 Skin & Appearance | 9 symptoms |
    | 👁️ Eye & Vision | 8 symptoms |
    | 🦻 Ear, Nose & Throat | 8 symptoms |
    | 🩸 Systemic & General | 10 symptoms |
    | 👶 Pregnancy & Reproductive | 10 symptoms |
    
    ---
    
    ### 📊 Model Performance
    
    - **Accuracy:** 82.2% on test data
    - **Training Data:** 96,088 patient records
    - **Symptoms:** 230 features
    - **Diseases:** 100 conditions
    - **Algorithm:** Random Forest Classifier
    
    ---
    
    ### 🛠️ Technology Stack
    
    - **Frontend:** Streamlit
    - **Machine Learning:** Scikit-learn
    - **Visualization:** Matplotlib, Plotly
    - **Data Source:** Kaggle API
    
    ---
    
    ### ⚠️ Important Disclaimer
    
    This tool is for **educational and demonstration purposes only**. 
    It should not be used as a substitute for professional medical advice, 
    diagnosis, or treatment. Always consult with a qualified healthcare 
    provider for medical concerns.
    
    ---
    
    **Version:** 1.0.0 | **Last Updated:** April 2026
    """)