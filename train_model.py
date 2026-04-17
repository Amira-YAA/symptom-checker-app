#!/usr/bin/env python
"""
Standalone script to train and save the disease prediction model
Run this script before deploying to Streamlit Cloud
"""

import os
import sys
import pandas as pd
from src.data_loader import data_loader
from src.model_trainer import DiseaseModelTrainer

def main():
    print("="*60)
    print("🏥 Disease Prediction Model Training")
    print("="*60)
    
    # Load data
    print("\n📥 Loading dataset from Kaggle...")
    try:
        df = data_loader.load_data()
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        sys.exit(1)
    
    # Prepare data
    print("\n🔧 Preparing data for training...")
    trainer = DiseaseModelTrainer()
    X, y = trainer.prepare_data(df)
    print(f"✅ Features: {X.shape[1]}, Classes: {len(y.unique())}")
    
    # Train model
    print("\n🚀 Training Random Forest model...")
    results = trainer.train_model(X, y)
    print(f"✅ Model trained! Accuracy: {results['accuracy']*100:.2f}%")
    
    # Save model
    print("\n💾 Saving model...")
    model_path = trainer.save_model()
    print(f"✅ Model saved to: {model_path}")
    
    # Feature importance
    importance_df = trainer.get_feature_importance_df()
    print("\n⭐ Top 10 Most Important Symptoms:")
    for i, row in importance_df.head(10).iterrows():
        print(f"   {row['symptom']}: {row['importance']:.4f}")
    
    print("\n" + "="*60)
    print("✅ Training complete! Ready for deployment.")
    print("="*60)

if __name__ == "__main__":
    main()