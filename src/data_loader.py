import os
import pandas as pd
import streamlit as st
from kagglehub import dataset_download
from dotenv import load_dotenv
import tempfile
import shutil
from tqdm import tqdm

load_dotenv()

class KaggleDataLoader:
    def __init__(self):
        self.dataset_path = os.getenv('KAGGLE_DATASET')
        self.filename = os.getenv('DATASET_FILE')
        self.cache_path = os.path.join(tempfile.gettempdir(), 'disease_dataset')
        
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_data(_self):
        """Load dataset from Kaggle with caching"""
        try:
            # Show loading spinner
            with st.spinner('📥 Downloading dataset from Kaggle...'):
                # Download dataset
                path = dataset_download(_self.dataset_path)
                
                # Find CSV file
                csv_file = os.path.join(path, _self.filename)
                
                if not os.path.exists(csv_file):
                    # Search for any CSV file in the downloaded path
                    for file in os.listdir(path):
                        if file.endswith('.csv'):
                            csv_file = os.path.join(path, file)
                            break
                    else:
                        raise FileNotFoundError(f"No CSV file found in {path}")
                
                # Load data
                df = pd.read_csv(csv_file)
                
                # Cache to local temp directory
                os.makedirs(_self.cache_path, exist_ok=True)
                cache_file = os.path.join(_self.cache_path, 'dataset.pkl')
                df.to_pickle(cache_file)
                
                return df
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            # Try loading from cache if available
            cache_file = os.path.join(_self.cache_path, 'dataset.pkl')
            if os.path.exists(cache_file):
                st.info("📀 Loading from local cache...")
                return pd.read_pickle(cache_file)
            raise e
    
    def get_data_info(self, df):
        """Get basic information about the dataset"""
        info = {
            'shape': df.shape,
            'diseases': df['diseases'].nunique(),
            'symptoms': len(df.columns) - 1,
            'missing_values': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        return info
    
    def get_disease_distribution(self, df):
        """Get disease distribution statistics"""
        distribution = df['diseases'].value_counts()
        return distribution
    
    def get_symptom_statistics(self, df):
        """Get symptom statistics"""
        symptom_cols = [col for col in df.columns if col != 'diseases']
        symptom_presence = df[symptom_cols].sum()
        
        stats = {
            'avg_symptoms_per_patient': df[symptom_cols].sum(axis=1).mean(),
            'most_common_symptom': symptom_presence.idxmax(),
            'most_common_count': symptom_presence.max(),
            'rarest_symptom': symptom_presence.idxmin(),
            'rarest_count': symptom_presence.min(),
            'total_symptoms': len(symptom_cols)
        }
        return stats, symptom_presence

# Initialize global loader
data_loader = KaggleDataLoader()