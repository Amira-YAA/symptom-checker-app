"""Data loading utilities"""

import os
import pandas as pd
import streamlit as st
import kagglehub

@st.cache_data(ttl=86400)
def load_kaggle_data():
    """Load dataset from Kaggle"""
    
    # Get API token
    api_token = None
    try:
        api_token = st.secrets.get("KAGGLE_API_TOKEN")
    except:
        pass
    
    if not api_token:
        api_token = os.environ.get("KAGGLE_API_TOKEN")
    
    if not api_token:
        st.error("❌ Kaggle API token not found!")
        st.stop()
    
    os.environ['KAGGLE_API_TOKEN'] = api_token
    
    dataset_path = "rajawatprateek/symptomchecker-multi-disease-diagnostic-data"
    
    with st.spinner('📥 Loading 96,088 patient records from Kaggle...'):
        path = kagglehub.dataset_download(dataset_path)
        
        # Find CSV file
        csv_file = None
        for file in os.listdir(path):
            if file.endswith('.csv'):
                csv_file = os.path.join(path, file)
                break
        
        if csv_file:
            df = pd.read_csv(csv_file)
            return df
        else:
            st.error("CSV file not found")
            st.stop()

def get_data_info(df):
    """Get basic information about the dataset"""
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'diseases': df['diseases'].nunique(),
        'symptoms': len(df.columns) - 1,
        'memory': df.memory_usage(deep=True).sum() / 1024**2
    }