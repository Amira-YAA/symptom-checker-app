import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from tqdm import tqdm

class DiseaseModelTrainer:
    def __init__(self, model_path='models/'):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        self.model = None
        self.symptom_columns = None
        
    def prepare_data(self, df):
        """Prepare data for training"""
        X = df.drop('diseases', axis=1)
        y = df['diseases']
        self.symptom_columns = X.columns.tolist()
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train Random Forest model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model with progress bar
        print("Training model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'model': self.model,
            'accuracy': accuracy,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_names': X.columns.tolist()
        }
    
    def save_model(self, filename='disease_model.joblib'):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        model_data = {
            'model': self.model,
            'symptom_columns': self.symptom_columns,
            'feature_importance': self.model.feature_importances_
        }
        
        filepath = os.path.join(self.model_path, filename)
        joblib.dump(model_data, filepath)
        return filepath
    
    def load_model(self, filename='disease_model.joblib'):
        """Load model from disk"""
        filepath = os.path.join(self.model_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found at {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.symptom_columns = model_data['symptom_columns']
        return model_data
    
    def predict(self, symptom_vector):
        """Make prediction for a single patient"""
        if self.model is None:
            raise ValueError("Model not loaded. Load or train model first.")
        
        prediction = self.model.predict([symptom_vector])[0]
        probabilities = self.model.predict_proba([symptom_vector])[0]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3_diseases = [self.model.classes_[i] for i in top_3_idx]
        top_3_probs = [probabilities[i] for i in top_3_idx]
        
        return {
            'primary': prediction,
            'primary_confidence': top_3_probs[0],
            'top_3': list(zip(top_3_diseases, top_3_probs))
        }
    
    def get_feature_importance_df(self):
        """Get feature importance as DataFrame"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        importance_df = pd.DataFrame({
            'symptom': self.symptom_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df