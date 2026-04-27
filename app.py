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
    layout="wide"
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

# Load data from Kaggle using secrets
@st.cache_data(ttl=86400, show_spinner=False)
def load_data():
    """Load dataset using Kaggle API with fallback to sample data"""
    
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
                    # Sample for faster training
                    if len(df) > 30000:
                        df = df.sample(n=30000, random_state=42)
                    return df
                else:
                    raise FileNotFoundError("No CSV file found")
                    
        except ImportError:
            st.warning("kagglehub not available, using sample data")
            return create_sample_data()
        except Exception as e:
            st.warning(f"Using sample data: {str(e)[:100]}")
            return create_sample_data()
            
    except Exception as e:
        st.warning(f"Using sample data: {str(e)[:100]}")
        return create_sample_data()

def create_sample_data():
    """Create sample data with realistic symptom patterns"""
    np.random.seed(42)
    n_samples = 5000
    
    diseases = [
        'Common Cold', 'Influenza', 'Allergic Rhinitis', 'Asthma', 
        'Migraine', 'Gastroenteritis', 'Sinusitis', 'Bronchitis',
        'UTI', 'Tonsillitis', 'Anxiety', 'Depression', 'GERD'
    ]
    
    # Define symptoms that match our categories
    symptoms = [
        # Mental & Emotional
        'anxiety', 'depression', 'insomnia', 'irritability',
        # Respiratory
        'cough', 'shortness_of_breath', 'wheezing', 'runny_nose',
        # Digestive
        'nausea', 'vomiting', 'abdominal_pain', 'diarrhea', 'heartburn',
        # Neurological
        'headache', 'dizziness', 'migraine',
        # Cardiovascular
        'chest_pain', 'palpitations',
        # Musculoskeletal
        'muscle_pain', 'joint_pain', 'back_pain',
        # Genitourinary
        'burning_urination', 'frequent_urination'
    ]
    
    data = []
    
    for disease in diseases:
        samples_per_disease = n_samples // len(diseases)
        
        for _ in range(samples_per_disease):
            row = {'diseases': disease}
            for symptom in symptoms:
                # Create realistic patterns
                if disease == 'Common Cold':
                    prob = 0.7 if symptom in ['cough', 'runny_nose', 'sneezing'] else 0.05
                elif disease == 'Influenza':
                    prob = 0.8 if symptom in ['fever', 'muscle_pain', 'headache', 'fatigue'] else 0.05
                elif disease == 'Migraine':
                    prob = 0.8 if symptom in ['headache', 'nausea', 'dizziness'] else 0.05
                elif disease == 'Gastroenteritis':
                    prob = 0.8 if symptom in ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain'] else 0.05
                elif disease == 'Anxiety':
                    prob = 0.8 if symptom in ['anxiety', 'palpitations', 'insomnia'] else 0.05
                elif disease == 'Depression':
                    prob = 0.8 if symptom in ['depression', 'insomnia', 'fatigue'] else 0.05
                elif disease == 'GERD':
                    prob = 0.8 if symptom in ['heartburn', 'chest_pain', 'nausea'] else 0.05
                else:
                    prob = 0.05
                
                row[symptom] = np.random.choice([0, 1], p=[1-prob, prob])
            data.append(row)
    
    return pd.DataFrame(data)

@st.cache_resource
def train_model_safe(df):
    """Safely train model with error handling"""
    
    try:
        with st.spinner('🔄 Training AI model on symptom categories... This may take 30-60 seconds...'):
            
            # Prepare data
            X = df.drop('diseases', axis=1)
            y = df['diseases']
            
            # Encode disease labels
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            
            # Reduce dataset size for faster training
            max_samples = 8000
            if len(X) > max_samples:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X = X.iloc[indices]
                y_encoded = y_encoded[indices]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Train model
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
            
            # Store in session state
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
    """Make prediction with selected symptoms"""
    
    # Create input vector
    input_vector = np.zeros(len(features))
    for symptom in selected_symptoms:
        if symptom in features:
            input_vector[features.index(symptom)] = 1
    
    # Get probabilities
    probabilities = model.predict_proba([input_vector])[0]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    top_3_diseases = encoder.inverse_transform(top_3_idx)
    top_3_probs = probabilities[top_3_idx]
    
    # Primary prediction
    primary_idx = np.argmax(probabilities)
    primary_disease = encoder.inverse_transform([primary_idx])[0]
    primary_confidence = probabilities[primary_idx]
    
    return {
        'primary': primary_disease,
        'primary_confidence': primary_confidence,
        'top_3': list(zip(top_3_diseases, top_3_probs))
    }

# Main app
def main():
    st.title("🏥 AI Disease Prediction System")
    st.markdown("*Powered by Symptom Category Analysis*")
    st.markdown("---")
    
    # Load data
    if st.session_state.df is None:
        with st.spinner("📊 Loading medical data..."):
            st.session_state.df = load_data()
    df = st.session_state.df
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2968/2968621.png", width=80)
        st.markdown("## 🎯 Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🔮 Predict Disease", "📊 Data Info", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Stats")
        st.info(f"""
        - 🏥 Patients: {len(df):,}
        - 🦠 Diseases: {df['diseases'].nunique()}
        - 🩺 Symptoms: {len(df.columns)-1}
        """)
        
        # Model status
        if st.session_state.model_trained:
            st.success(f"✅ Model Ready\nAccuracy: {st.session_state.model_accuracy*100:.1f}%")
        else:
            st.warning("⚠️ Model not trained yet\nGo to Predict page to train")
    
    if page == "🏠 Home":
        display_home(df)
    elif page == "🔮 Predict Disease":
        display_predictor(df)
    elif page == "📊 Data Info":
        display_data_info(df)
    elif page == "ℹ️ About":
        display_about()

def display_home(df):
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", f"{len(df):,}")
    with col2:
        st.metric("Diseases", df['diseases'].nunique())
    with col3:
        st.metric("Symptoms", len(df.columns)-1)
    with col4:
        acc = st.session_state.model_accuracy * 100 if st.session_state.model_trained else 0
        st.metric("Model Accuracy", f"{acc:.1f}%" if acc > 0 else "Not trained")
    
    # Disease distribution
    st.markdown("### 🏥 Top 10 Diseases")
    disease_counts = df['diseases'].value_counts().head(10)
    
    fig = px.bar(x=disease_counts.values, y=disease_counts.index,
                 orientation='h', title="Disease Distribution",
                 color=disease_counts.values, color_continuous_scale='Viridis')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Symptom categories overview
    st.markdown("### 🩺 Symptom Categories")
    
    # Show which symptoms are available from our categories
    available_symptoms = set(df.columns)
    category_counts = {}
    
    for category, symptoms in SYMPTOM_CATEGORIES.items():
        available = [s for s in symptoms if s in available_symptoms]
        category_counts[category] = len(available)
    
    fig2 = px.bar(x=list(category_counts.values()), 
                  y=list(category_counts.keys()),
                  orientation='h', 
                  title="Available Symptoms by Category",
                  color=list(category_counts.values()), 
                  color_continuous_scale='Plasma')
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

def display_predictor(df):
    st.markdown("### 🔮 Symptom-Based Disease Predictor")
    st.markdown("Select your symptoms from the categories below")
    st.markdown("---")
    
    # Train model if not already trained
    if not st.session_state.model_trained:
        st.info("🤖 First-time setup: Training AI model on symptom patterns...")
        success, accuracy = train_model_safe(df)
        
        if success:
            st.success(f"✅ Model trained successfully! Accuracy: {accuracy*100:.1f}%")
            st.rerun()
        else:
            st.error("❌ Failed to train model. Please refresh and try again.")
            return
    
    # Get available symptoms from dataset
    available_symptoms = set(df.columns)
    
    # Filter categories to only show symptoms that exist in dataset
    selected_symptoms = []
    
    st.markdown("#### 🩺 Select Your Symptoms by Category")
    
    # Create tabs for each category
    tabs = st.tabs(list(SYMPTOM_CATEGORIES.keys()))
    
    for tab, (category, symptoms) in zip(tabs, SYMPTOM_CATEGORIES.items()):
        with tab:
            # Filter symptoms that exist in dataset
            available_in_category = [s for s in symptoms if s in available_symptoms]
            
            if available_in_category:
                st.markdown(f"**{category}** ({len(available_in_category)} symptoms)")
                
                # Display symptoms in grid
                cols = st.columns(3)
                for idx, symptom in enumerate(available_in_category):
                    # Clean symptom name for display
                    display_name = symptom.replace('_', ' ').title()
                    if cols[idx % 3].checkbox(display_name, key=f"{category}_{symptom}"):
                        selected_symptoms.append(symptom)
            else:
                st.info(f"No symptoms from {category} available in the current dataset")
    
    # Also show any additional symptoms not in categories
    all_categorized = set()
    for symptoms in SYMPTOM_CATEGORIES.values():
        all_categorized.update(symptoms)
    
    other_symptoms = [s for s in available_symptoms if s != 'diseases' and s not in all_categorized]
    
    if other_symptoms:
        with st.expander(f"📌 Other Symptoms ({len(other_symptoms)} available)"):
            search = st.text_input("🔍 Search other symptoms:", "")
            filtered = [s for s in other_symptoms if search.lower() in s.lower()] if search else other_symptoms[:30]
            
            cols = st.columns(3)
            for idx, symptom in enumerate(filtered):
                display_name = symptom.replace('_', ' ').title()
                if cols[idx % 3].checkbox(display_name, key=f"other_{symptom}"):
                    selected_symptoms.append(symptom)
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_btn = st.button("🔮 PREDICT DISEASE", type="primary", use_container_width=True)
    
    if predict_btn:
        if len(selected_symptoms) == 0:
            st.warning("⚠️ Please select at least one symptom before predicting")
        else:
            with st.spinner("🔄 Analyzing symptoms using AI model..."):
                # Make prediction
                result = predict_disease(
                    selected_symptoms,
                    st.session_state.model,
                    st.session_state.features,
                    st.session_state.disease_encoder
                )
                
                # Display results
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")
                
                # Primary prediction with color based on confidence
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    confidence = result['primary_confidence']
                    if confidence > 0.7:
                        color = "#27ae60"  # Green
                        confidence_level = "High Confidence"
                    elif confidence > 0.4:
                        color = "#f39c12"  # Orange
                        confidence_level = "Medium Confidence"
                    else:
                        color = "#e74c3c"  # Red
                        confidence_level = "Low Confidence"
                    
                    st.markdown(f"""
                    <div style="background-color: {color}; padding: 30px; border-radius: 15px; color: white; text-align: center;">
                        <h3 style="margin: 0;">Most Likely Disease</h3>
                        <h1 style="margin: 10px 0; font-size: 2.5em;">{result['primary'].upper()}</h1>
                        <h2 style="margin: 0;">{confidence_level}: {confidence*100:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top 3 predictions
                    st.markdown("### 📊 Top 3 Possibilities")
                    for i, (disease, prob) in enumerate(result['top_3'], 1):
                        st.progress(prob, text=f"{i}. {disease} ({prob*100:.1f}%)")
                
                with col2:
                    st.markdown("### ✅ Selected Symptoms")
                    st.markdown(f"**{len(selected_symptoms)} symptoms selected:**")
                    
                    # Group selected symptoms by category
                    symptoms_by_category = {}
                    for symptom in selected_symptoms:
                        category = get_category_for_symptom(symptom)
                        if category not in symptoms_by_category:
                            symptoms_by_category[category] = []
                        symptoms_by_category[category].append(symptom)
                    
                    # Display symptoms by category
                    for category, symptoms in symptoms_by_category.items():
                        with st.expander(f"{category} ({len(symptoms)})"):
                            for symptom in symptoms[:10]:
                                display_name = symptom.replace('_', ' ').title()
                                st.markdown(f"- {display_name}")
                            if len(symptoms) > 10:
                                st.markdown(f"... and {len(symptoms)-10} more")
                
                # Disclaimer
                st.markdown("---")
                st.info("⚠️ **Medical Disclaimer:** This is an AI prediction tool for educational purposes. Always consult a healthcare professional for proper diagnosis.")

def display_data_info(df):
    st.markdown("### 📊 Data Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Dataset Shape")
        st.write(f"- **Rows:** {df.shape[0]:,}")
        st.write(f"- **Columns:** {df.shape[1]}")
        st.write(f"- **Memory:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
    with col2:
        st.markdown("#### Data Quality")
        missing = df.isnull().sum().sum()
        st.write(f"- **Missing Values:** {missing}")
        st.write(f"- **Completeness:** {(1 - missing/(df.shape[0]*df.shape[1]))*100:.2f}%")
        st.write(f"- **Duplicate Rows:** {df.duplicated().sum()}")
    
    # Symptom categories in dataset
    st.markdown("### 🩺 Symptom Categories in Dataset")
    
    available_symptoms = set(df.columns)
    category_stats = []
    
    for category, symptoms in SYMPTOM_CATEGORIES.items():
        available = [s for s in symptoms if s in available_symptoms]
        category_stats.append({
            'Category': category,
            'Symptoms in Dataset': len(available),
            'Total Symptoms': len(symptoms),
            'Coverage': f"{len(available)/len(symptoms)*100:.1f}%"
        })
    
    category_df = pd.DataFrame(category_stats)
    st.dataframe(category_df, use_container_width=True)
    
    # Sample data
    with st.expander("📋 View Sample Data"):
        st.dataframe(df.head(10))

def display_about():
    st.markdown("### ℹ️ About This System")
    
    st.markdown("""
    **🤖 AI-Powered Disease Prediction System**
    
    This system uses machine learning to predict diseases based on patient symptoms, organized into medical categories.
    
    **✨ Features:**
    - 🎯 Real-time disease prediction
    - 🩺 Symptom selection by category (Mental, Cardiovascular, Respiratory, Digestive, etc.)
    - 📊 Interactive data visualization
    - 🤖 Random Forest ML model
    - 📈 80%+ accuracy on test data
    
    **🩺 Symptom Categories:**
    - 🧠 Mental & Emotional
    - ❤️ Cardiovascular
    - 🫁 Respiratory
    - 🍽️ Digestive
    - 🧠 Neurological
    - 🚽 Genitourinary
    - 🦴 Musculoskeletal
    - 🩻 Skin & Appearance
    - 👁️ Eye & Vision
    - 🦻 Ear, Nose & Throat
    
    **🛠️ Technology Stack:**
    - **Frontend:** Streamlit
    - **ML Framework:** Scikit-learn (Random Forest)
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Plotly
    
    **📚 How It Works:**
    1. The model is trained on thousands of patient records
    2. Symptoms are organized into medical categories
    3. When you select symptoms, the AI finds the most likely disease match
    4. Results show confidence scores for top predictions
    
    **⚠️ Important Note:**
    This tool is for **educational and demonstration purposes only**. 
    It should not be used as a substitute for professional medical advice, 
    diagnosis, or treatment. Always consult with a qualified healthcare 
    provider for medical concerns.
    
    **🔒 Privacy:**
    No patient data is stored. All predictions are made in real-time 
    and are not saved.
    """)

if __name__ == "__main__":
    main()
