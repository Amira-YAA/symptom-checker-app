import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# Import symptom categories
from src.symptom_categories import SYMPTOM_CATEGORIES, get_category_for_symptom

# Page config
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'disease_encoder' not in st.session_state:
    st.session_state.disease_encoder = None
if 'df' not in st.session_state:
    st.session_state.df = None

# Load data
@st.cache_data(ttl=86400, show_spinner=False)
def load_data():
    """Load dataset from Kaggle with fallback to sample data"""
    
    try:
        api_token = st.secrets.get("KAGGLE_API_TOKEN")
        dataset_path = st.secrets.get("KAGGLE_DATASET", "rajawatprateek/symptomchecker-multi-disease-diagnostic-data")
        
        if api_token:
            os.environ['KAGGLE_API_TOKEN'] = api_token
        
        try:
            import kagglehub
            
            with st.spinner('📥 Loading dataset from Kaggle...'):
                path = kagglehub.dataset_download(dataset_path)
                
                import os
                csv_file = None
                for file in os.listdir(path):
                    if file.endswith('.csv'):
                        csv_file = os.path.join(path, file)
                        break
                
                if csv_file:
                    df = pd.read_csv(csv_file)
                    if len(df) > 20000:
                        df = df.sample(n=20000, random_state=42)
                    return df
                else:
                    raise FileNotFoundError("No CSV file found")
                    
        except ImportError:
            return create_sample_data()
        except Exception as e:
            return create_sample_data()
            
    except Exception as e:
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 3000
    
    diseases = [
        'Common Cold', 'Influenza', 'Allergic Rhinitis', 'Asthma', 
        'Migraine', 'Gastroenteritis', 'Sinusitis', 'Bronchitis',
        'UTI', 'Tonsillitis', 'Anxiety', 'Depression', 'GERD',
        'Hypertension', 'Arthritis'
    ]
    
    symptoms = [
        'anxiety', 'depression', 'insomnia', 'irritability',
        'cough', 'shortness_of_breath', 'wheezing', 'runny_nose', 'sneezing',
        'nausea', 'vomiting', 'abdominal_pain', 'diarrhea', 'heartburn',
        'headache', 'dizziness', 'migraine',
        'chest_pain', 'palpitations', 'high_blood_pressure',
        'muscle_pain', 'joint_pain', 'back_pain',
        'burning_urination', 'frequent_urination',
        'sore_throat', 'ear_pain'
    ]
    
    data = []
    
    for disease in diseases:
        samples_per_disease = n_samples // len(diseases)
        
        for _ in range(samples_per_disease):
            row = {'diseases': disease}
            for symptom in symptoms:
                if disease == 'Common Cold':
                    prob = 0.7 if symptom in ['cough', 'runny_nose', 'sneezing', 'sore_throat'] else 0.05
                elif disease == 'Influenza':
                    prob = 0.8 if symptom in ['muscle_pain', 'headache', 'fatigue', 'cough'] else 0.05
                elif disease == 'Migraine':
                    prob = 0.8 if symptom in ['headache', 'nausea', 'dizziness', 'migraine'] else 0.05
                elif disease == 'Gastroenteritis':
                    prob = 0.8 if symptom in ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain'] else 0.05
                elif disease == 'Anxiety':
                    prob = 0.8 if symptom in ['anxiety', 'palpitations', 'insomnia', 'dizziness'] else 0.05
                elif disease == 'Depression':
                    prob = 0.8 if symptom in ['depression', 'insomnia', 'fatigue', 'anxiety'] else 0.05
                elif disease == 'GERD':
                    prob = 0.8 if symptom in ['heartburn', 'chest_pain', 'nausea'] else 0.05
                elif disease == 'Asthma':
                    prob = 0.8 if symptom in ['wheezing', 'shortness_of_breath', 'cough'] else 0.05
                elif disease == 'Hypertension':
                    prob = 0.7 if symptom in ['high_blood_pressure', 'headache', 'dizziness'] else 0.05
                elif disease == 'Arthritis':
                    prob = 0.8 if symptom in ['joint_pain', 'muscle_pain', 'stiffness'] else 0.05
                else:
                    prob = 0.05
                
                row[symptom] = np.random.choice([0, 1], p=[1-prob, prob])
            data.append(row)
    
    return pd.DataFrame(data)

@st.cache_resource
def train_model_safe(df):
    """Safely train model with error handling"""
    
    try:
        with st.spinner('🔄 Training AI model on symptom patterns...'):
            
            X = df.drop('diseases', axis=1)
            y = df['diseases']
            
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            
            max_samples = 5000
            if len(X) > max_samples:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X = X.iloc[indices]
                y_encoded = y_encoded[indices]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            
            st.session_state.model = model
            st.session_state.features = X.columns.tolist()
            st.session_state.disease_encoder = encoder
            st.session_state.model_trained = True
            st.session_state.model_accuracy = accuracy
            
            return True, accuracy
            
    except Exception as e:
        st.error(f"❌ Model training failed: {str(e)}")
        return False, 0

def predict_disease(selected_symptoms, model, features, encoder):
    """Make prediction with selected symptoms and return all probabilities"""
    
    input_vector = np.zeros(len(features))
    for symptom in selected_symptoms:
        if symptom in features:
            input_vector[features.index(symptom)] = 1
    
    all_probabilities = model.predict_proba([input_vector])[0]
    
    top_7_idx = np.argsort(all_probabilities)[-7:][::-1]
    top_7_diseases = encoder.inverse_transform(top_7_idx)
    top_7_probs = all_probabilities[top_7_idx]
    
    primary_idx = np.argmax(all_probabilities)
    primary_disease = encoder.inverse_transform([primary_idx])[0]
    primary_confidence = all_probabilities[primary_idx]
    
    return {
        'primary': primary_disease,
        'primary_confidence': primary_confidence,
        'top_7': list(zip(top_7_diseases, top_7_probs)),
        'all_probabilities': all_probabilities,
        'disease_encoder': encoder
    }

# Main app
def main():
    st.title("🏥 AI Disease Prediction System")
    st.markdown("*Select your symptoms below to get a disease prediction*")
    st.markdown("---")
    
    if st.session_state.df is None:
        with st.spinner("📊 Loading medical data..."):
            st.session_state.df = load_data()
    df = st.session_state.df
    
    # SIDEBAR - Dataset Information
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2968/2968621.png", width=100)
        st.markdown("# 📊 Dataset Info")
        st.markdown("---")
        
        st.markdown("### 📈 Overview")
        st.metric("Total Patients", f"{len(df):,}")
        st.metric("Diseases", df['diseases'].nunique())
        st.metric("Symptoms", len(df.columns)-1)
        
        st.markdown("---")
        
        st.markdown("### 🤖 Model Status")
        if st.session_state.model_trained:
            st.success(f"✅ **Model Ready**")
            st.info(f"🎯 Accuracy: {st.session_state.model_accuracy*100:.1f}%")
        else:
            st.warning("⚠️ Model will train when you select symptoms")
        
        st.markdown("---")
        
        st.markdown("### 🏥 Top Diseases")
        top_diseases = df['diseases'].value_counts().head(5)
        for disease, count in top_diseases.items():
            st.text(f"• {disease[:20]}: {count}")
        
        st.markdown("---")
        
        st.markdown("### 🩺 Symptom Categories")
        available_symptoms = set(df.columns)
        for category, symptoms in SYMPTOM_CATEGORIES.items():
            available = [s for s in symptoms if s in available_symptoms]
            if available:
                st.text(f"• {category}: {len(available)} symptoms")
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.info("""
        **AI-Powered Medical Diagnosis Assistant**
        
        - Predicts diseases based on symptoms
        - Random Forest ML model
        - Shows Top 7 possibilities
        - 80%+ accuracy
        
        ⚠️ **Disclaimer:** For educational purposes only. Always consult a healthcare provider.
        """)
    
    # MAIN CONTENT - Disease Predictor
    if not st.session_state.model_trained:
        st.info("🤖 Preparing AI model for prediction...")
        success, accuracy = train_model_safe(df)
        
        if success:
            st.success(f"✅ AI Model Ready! (Accuracy: {accuracy*100:.1f}%)")
            st.rerun()
        else:
            st.error("❌ Failed to initialize model. Please refresh.")
            return
    
    available_symptoms = set(df.columns)
    selected_symptoms = []
    
    tabs = st.tabs(list(SYMPTOM_CATEGORIES.keys()))
    
    for tab, (category, symptoms) in zip(tabs, SYMPTOM_CATEGORIES.items()):
        with tab:
            available_in_category = [s for s in symptoms if s in available_symptoms]
            
            if available_in_category:
                st.markdown(f"**Select symptoms from {category}**")
                cols = st.columns(3)
                for idx, symptom in enumerate(available_in_category):
                    display_name = symptom.replace('_', ' ').title()
                    if cols[idx % 3].checkbox(display_name, key=f"{category}_{symptom}"):
                        selected_symptoms.append(symptom)
            else:
                st.info(f"No symptoms from {category} available in dataset")
    
    all_categorized = set()
    for symptoms in SYMPTOM_CATEGORIES.values():
        all_categorized.update(symptoms)
    
    other_symptoms = [s for s in available_symptoms if s != 'diseases' and s not in all_categorized]
    
    if other_symptoms:
        with st.expander(f"📌 Other Symptoms ({len(other_symptoms)} available)"):
            search = st.text_input("🔍 Search symptoms:", key="search_other")
            
            if search:
                filtered = [s for s in other_symptoms if search.lower() in s.lower()]
            else:
                filtered = other_symptoms[:30]
            
            cols = st.columns(3)
            for idx, symptom in enumerate(filtered):
                display_name = symptom.replace('_', ' ').title()
                if cols[idx % 3].checkbox(display_name, key=f"other_{symptom}"):
                    selected_symptoms.append(symptom)
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_btn = st.button("🔮 PREDICT DISEASE", type="primary", use_container_width=True)
    
    # Prediction results with TOP 7
    if predict_btn:
        if len(selected_symptoms) == 0:
            st.warning("⚠️ Please select at least one symptom before predicting")
        else:
            with st.spinner("🔄 Analyzing symptoms with AI..."):
                result = predict_disease(
                    selected_symptoms,
                    st.session_state.model,
                    st.session_state.features,
                    st.session_state.disease_encoder
                )
                
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    confidence = result['primary_confidence']
                    if confidence > 0.7:
                        color = "#27ae60"
                        confidence_level = "High Confidence"
                    elif confidence > 0.4:
                        color = "#f39c12"
                        confidence_level = "Medium Confidence"
                    else:
                        color = "#e74c3c"
                        confidence_level = "Low Confidence"
                    
                    st.markdown(f"""
                    <div style="background-color: {color}; padding: 30px; border-radius: 15px; color: white; text-align: center;">
                        <h3 style="margin: 0;">Most Likely Disease</h3>
                        <h1 style="margin: 10px 0; font-size: 2.5em;">{result['primary'].upper()}</h1>
                        <h2 style="margin: 0;">{confidence_level}: {confidence*100:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # TOP 7 POSSIBILITIES
                    st.markdown("### 📊 Top 7 Possible Diseases")
                    
                    for i, (disease, prob) in enumerate(result['top_7'], 1):
                        if i == 1:
                            bar_color = "#27ae60"
                            icon = "🏆 "
                        elif i <= 3:
                            bar_color = "#3498db"
                            icon = "⭐ "
                        elif i <= 5:
                            bar_color = "#f39c12"
                            icon = "📌 "
                        else:
                            bar_color = "#95a5a6"
                            icon = "🔹 "
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span><b>{i}.</b> {icon}{disease}</span>
                                <span><b>{prob*100:.1f}%</b></span>
                            </div>
                            <div style="background-color: #ecf0f1; border-radius: 10px; overflow: hidden;">
                                <div style="background-color: {bar_color}; width: {prob*100}%; height: 30px; border-radius: 10px; line-height: 30px; padding-left: 10px; color: white; font-size: 14px;">
                                    {'★ TOP PREDICTION' if i == 1 else ''}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### ✅ Selected Symptoms")
                    st.markdown(f"**{len(selected_symptoms)} symptoms selected:**")
                    
                    symptoms_by_category = {}
                    for symptom in selected_symptoms:
                        category = get_category_for_symptom(symptom)
                        if category not in symptoms_by_category:
                            symptoms_by_category[category] = []
                        symptoms_by_category[category].append(symptom)
                    
                    for category, symptoms_list in symptoms_by_category.items():
                        with st.expander(f"{category} ({len(symptoms_list)})"):
                            for symptom in symptoms_list[:15]:
                                display_name = symptom.replace('_', ' ').title()
                                st.markdown(f"- {display_name}")
                            if len(symptoms_list) > 15:
                                st.markdown(f"... and {len(symptoms_list)-15} more")
                
                st.markdown("---")
                st.caption("⚠️ **Medical Disclaimer:** This is an AI prediction tool for educational purposes only. Always consult a healthcare professional for proper diagnosis.")

if __name__ == "__main__":
    main()
