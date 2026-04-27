"""Model training and prediction"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import streamlit as st

@st.cache_resource
def train_model(df):
    """Train Random Forest model"""
    
    X = df.drop('diseases', axis=1)
    y = df['diseases']
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=25,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        'model': model,
        'features': X.columns.tolist(),
        'encoder': encoder,
        'accuracy': accuracy
    }, "models/disease_model.pkl")
    
    return model, X.columns.tolist(), encoder, accuracy

@st.cache_resource
def load_saved_model():
    """Load saved model if exists"""
    model_path = "models/disease_model.pkl"
    if os.path.exists(model_path):
        data = joblib.load(model_path)
        return data['model'], data['features'], data['encoder'], data['accuracy']
    return None, None, None, None

def predict_disease(selected_symptoms, model, features, encoder):
    """Make prediction"""
    input_vector = np.zeros(len(features))
    for symptom in selected_symptoms:
        if symptom in features:
            input_vector[features.index(symptom)] = 1
    
    probabilities = model.predict_proba([input_vector])[0]
    
    # Top 7
    top_7_idx = np.argsort(probabilities)[-7:][::-1]
    top_7_diseases = encoder.inverse_transform(top_7_idx)
    top_7_probs = probabilities[top_7_idx]
    
    # Primary
    primary_idx = np.argmax(probabilities)
    primary_disease = encoder.inverse_transform([primary_idx])[0]
    primary_confidence = probabilities[primary_idx]
    
    return {
        'primary': primary_disease,
        'primary_confidence': primary_confidence,
        'top_7': list(zip(top_7_diseases, top_7_probs)),
        'all_probabilities': probabilities,
        'all_diseases': encoder.classes_
    }