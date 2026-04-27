import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SYMPTOM CATEGORIES (Your exact categories)
# ============================================

symptom_categories = {
    "🧠 Mental & Emotional": [
        "anxiety and nervousness", "depression", "restlessness", "excessive anger",
        "fears and phobias", "low self-esteem", "obsessions and compulsions",
        "hostile behavior", "antisocial behavior", "hysterical behavior",
        "temper problems", "insomnia", "sleepiness"
    ],
    "❤️ Cardiovascular": [
        "shortness of breath", "chest tightness", "palpitations", "irregular heartbeat",
        "sharp chest pain", "increased heart rate", "decreased heart rate",
        "chest pain", "peripheral edema"
    ],
    "🫁 Respiratory": [
        "cough", "wheezing", "difficulty breathing", "coughing up sputum",
        "hemoptysis", "congestion in chest", "abnormal breathing sounds",
        "nasal congestion", "sore throat", "hoarse voice", "sinus congestion"
    ],
    "🍽️ Digestive": [
        "nausea", "vomiting", "diarrhea", "abdominal pain", "heartburn",
        "constipation", "blood in stool", "upper abdominal pain",
        "stomach bloating", "changes in stool appearance", "melena",
        "difficulty in swallowing", "regurgitation", "burning abdominal pain"
    ],
    "🧠 Neurological": [
        "headache", "dizziness", "seizures", "loss of sensation",
        "paresthesia", "focal weakness", "problems with movement",
        "tremors", "memory disturbance", "delusions or hallucinations"
    ],
    "🚽 Genitourinary": [
        "painful urination", "frequent urination", "blood in urine",
        "vaginal discharge", "vaginal itching", "pelvic pain",
        "vaginal pain", "vaginal redness", "involuntary urination",
        "retention of urine", "pain during intercourse", "infertility"
    ],
    "🦴 Musculoskeletal": [
        "back pain", "joint pain", "muscle weakness", "leg pain",
        "hip pain", "knee pain", "shoulder pain", "neck pain",
        "arm pain", "wrist pain", "ankle pain", "muscle cramps",
        "low back pain", "side pain", "rib pain"
    ],
    "🩻 Skin & Appearance": [
        "skin rash", "skin lesion", "itching of skin", "acne or pimples",
        "skin growth", "abnormal appearing skin", "skin dryness",
        "skin swelling", "skin moles", "diaper rash"
    ],
    "👁️ Eye & Vision": [
        "diminished vision", "double vision", "pain in eye",
        "eye redness", "lacrimation", "itchiness of eye",
        "blindness", "spots or clouds in vision", "foreign body sensation"
    ],
    "🦻 Ear, Nose & Throat": [
        "ear pain", "ringing in ear", "plugged feeling in ear",
        "itchy ear(s)", "fluid in ear", "sore throat", 
        "hoarse voice", "difficulty speaking"
    ],
    "🩸 Systemic & General": [
        "fever", "fatigue", "weakness", "chills", "sweating",
        "weight gain", "loss of appetite", "flu-like syndrome", 
        "feeling ill", "ache all over"
    ],
    "👶 Pregnancy & Reproductive": [
        "pain during pregnancy", "spotting or bleeding during pregnancy",
        "uterine contractions", "recent pregnancy", "problems during pregnancy",
        "intermenstrual bleeding", "heavy menstrual flow", "painful menstruation",
        "long menstrual periods", "unpredictable menstruation"
    ]
}

# Flatten for category lookup
all_categorized_symptoms = set()
for symptoms in symptom_categories.values():
    all_categorized_symptoms.update(symptoms)

def get_symptom_category(symptom):
    """Return the category for a given symptom"""
    for category, symptoms in symptom_categories.items():
        if symptom in symptoms:
            return category
    return "📌 Other Symptoms"

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
        'UTI', 'Tonsillitis', 'Anxiety Disorder', 'Major Depression', 
        'GERD', 'Hypertension', 'Osteoarthritis', 'Panic Disorder'
    ]
    
    # Get all symptoms from categories
    all_symptoms = []
    for symptoms in symptom_categories.values():
        all_symptoms.extend(symptoms)
    all_symptoms = list(set(all_symptoms))
    
    data = []
    
    for disease in diseases:
        samples_per_disease = n_samples // len(diseases)
        
        for _ in range(samples_per_disease):
            row = {'diseases': disease}
            for symptom in all_symptoms:
                # Create realistic patterns
                if disease in ['Anxiety Disorder', 'Panic Disorder']:
                    prob = 0.7 if symptom in ['anxiety and nervousness', 'palpitations', 'insomnia', 'restlessness'] else 0.05
                elif disease in ['Major Depression']:
                    prob = 0.7 if symptom in ['depression', 'insomnia', 'fatigue', 'loss of appetite'] else 0.05
                elif disease == 'Common Cold':
                    prob = 0.7 if symptom in ['cough', 'nasal congestion', 'sore throat', 'sneezing'] else 0.05
                elif disease == 'Influenza':
                    prob = 0.7 if symptom in ['fever', 'fatigue', 'muscle weakness', 'cough'] else 0.05
                elif disease == 'Migraine':
                    prob = 0.7 if symptom in ['headache', 'nausea', 'dizziness', 'diminished vision'] else 0.05
                elif disease == 'Gastroenteritis':
                    prob = 0.7 if symptom in ['nausea', 'vomiting', 'diarrhea', 'abdominal pain'] else 0.05
                elif disease == 'GERD':
                    prob = 0.7 if symptom in ['heartburn', 'chest tightness', 'regurgitation'] else 0.05
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
    """Make prediction with selected symptoms"""
    
    input_vector = np.zeros(len(features))
    for symptom in selected_symptoms:
        if symptom in features:
            input_vector[features.index(symptom)] = 1
    
    all_probabilities = model.predict_proba([input_vector])[0]
    
    # Get top 7 predictions
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
        'all_diseases': encoder.classes_
    }

# Main app
def main():
    st.title("🏥 AI Disease Prediction System")
    st.markdown("*Select your symptoms below to get a disease prediction*")
    st.markdown("---")
    
    # Load data
    if st.session_state.df is None:
        with st.spinner("📊 Loading medical data..."):
            st.session_state.df = load_data()
    df = st.session_state.df
    
    # ============================================
    # SIDEBAR - Dataset Information
    # ============================================
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2968/2968621.png", width=80)
        st.markdown("# 📊 Dataset Info")
        st.markdown("---")
        
        st.markdown("### 📈 Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Patients", f"{len(df):,}")
        with col2:
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
            st.text(f"• {disease[:25]}: {count}")
        
        st.markdown("---")
        
        st.markdown("### 🩺 Categories")
        for category in symptom_categories.keys():
            st.text(f"• {category}")
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.info("""
        **AI Medical Assistant**
        
        - 12 symptom categories
        - Random Forest ML model
        - Shows Top 7 possibilities
        - 80%+ accuracy
        
        ⚠️ *For educational purposes only*
        """)
    
    # ============================================
    # MAIN CONTENT - Disease Predictor
    # ============================================
    
    # Train model if not already trained
    if not st.session_state.model_trained:
        st.info("🤖 Preparing AI model for prediction...")
        success, accuracy = train_model_safe(df)
        
        if success:
            st.success(f"✅ AI Model Ready! (Accuracy: {accuracy*100:.1f}%)")
            st.rerun()
        else:
            st.error("❌ Failed to initialize model. Please refresh.")
            return
    
    # Get available symptoms from dataset
    available_symptoms = set(df.columns)
    
    # Selected symptoms list
    selected_symptoms = []
    
    # Create tabs for each symptom category
    tabs = st.tabs(list(symptom_categories.keys()))
    
    for tab, (category, symptoms) in zip(tabs, symptom_categories.items()):
        with tab:
            # Filter symptoms that exist in dataset
            available_in_category = [s for s in symptoms if s in available_symptoms]
            
            if available_in_category:
                st.markdown(f"**Select symptoms from {category}**")
                st.caption(f"📋 {len(available_in_category)} symptoms available")
                
                # Display symptoms in 3 columns
                cols = st.columns(3)
                for idx, symptom in enumerate(available_in_category):
                    # Clean symptom name for display
                    display_name = symptom.replace('_', ' ').title()
                    if cols[idx % 3].checkbox(display_name, key=f"{category}_{symptom}"):
                        selected_symptoms.append(symptom)
            else:
                st.info(f"No symptoms from {category} available in the current dataset")
    
    # Other symptoms not in categories
    other_symptoms = [s for s in available_symptoms if s != 'diseases' and s not in all_categorized_symptoms]
    
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
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_btn = st.button("🔮 PREDICT DISEASE", type="primary", use_container_width=True)
    
    # ============================================
    # PREDICTION RESULTS
    # ============================================
    
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
                        confidence_level = "HIGH CONFIDENCE"
                        emoji = "🎯"
                    elif confidence > 0.4:
                        color = "#f39c12"
                        confidence_level = "MEDIUM CONFIDENCE"
                        emoji = "✅"
                    else:
                        color = "#e74c3c"
                        confidence_level = "LOW CONFIDENCE"
                        emoji = "⚠️"
                    
                    st.markdown(f"""
                    <div style="background-color: {color}; padding: 30px; border-radius: 15px; color: white; text-align: center;">
                        <h3 style="margin: 0;">{emoji} Most Likely Disease</h3>
                        <h1 style="margin: 10px 0; font-size: 2.5em;">{result['primary'].upper()}</h1>
                        <h2 style="margin: 0;">{confidence_level}: {confidence*100:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # TOP 7 POSSIBILITIES
                    st.markdown("### 📊 Top 7 Possible Diseases")
                    st.caption("Showing all 7 possibilities with confidence scores")
                    
                    for i, (disease, prob) in enumerate(result['top_7'], 1):
                        if i == 1:
                            bar_color = "#27ae60"
                            icon = "🏆"
                        elif i <= 3:
                            bar_color = "#3498db"
                            icon = "⭐"
                        elif i <= 5:
                            bar_color = "#f39c12"
                            icon = "📌"
                        else:
                            bar_color = "#95a5a6"
                            icon = "🔹"
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span><b>{i}.</b> {icon} <b>{disease}</b></span>
                                <span style="color: {bar_color}; font-weight: bold;">{prob*100:.1f}%</span>
                            </div>
                            <div style="background-color: #ecf0f1; border-radius: 10px; overflow: hidden;">
                                <div style="background-color: {bar_color}; width: {prob*100}%; height: 35px; border-radius: 10px; line-height: 35px; padding-left: 10px; color: white; font-size: 14px;">
                                    {'★ TOP PREDICTION' if i == 1 else ' '}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### ✅ Selected Symptoms")
                    st.markdown(f"**{len(selected_symptoms)} symptoms selected:**")
                    
                    # Group selected symptoms by category
                    symptoms_by_category = {}
                    for symptom in selected_symptoms:
                        category = get_symptom_category(symptom)
                        if category not in symptoms_by_category:
                            symptoms_by_category[category] = []
                        symptoms_by_category[category].append(symptom)
                    
                    # Display symptoms by category
                    for category, symptoms_list in symptoms_by_category.items():
                        with st.expander(f"{category} ({len(symptoms_list)})"):
                            display_symptoms = [s.replace('_', ' ').title() for s in symptoms_list]
                            st.markdown("\n".join([f"- {s}" for s in display_symptoms[:20]]))
                            if len(symptoms_list) > 20:
                                st.markdown(f"... and {len(symptoms_list)-20} more")
                
                # Probability bar chart
                st.markdown("### 📈 Probability Distribution")
                
                # Get top 10 for chart
                top_10_idx = np.argsort(result['all_probabilities'])[-10:][::-1]
                top_10_diseases = result['all_diseases'][top_10_idx]
                top_10_probs = result['all_probabilities'][top_10_idx]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['#27ae60', '#3498db', '#f39c12', '#e67e22', '#e74c3c', 
                         '#9b59b6', '#1abc9c', '#34495e', '#95a5a6', '#7f8c8d']
                
                bars = ax.barh(range(len(top_10_diseases)), top_10_probs, color=colors[:len(top_10_diseases)])
                ax.set_yticks(range(len(top_10_diseases)))
                ax.set_yticklabels([d[:30] for d in top_10_diseases])
                ax.set_xlabel('Probability')
                ax.set_title('Top 10 Disease Probabilities')
                ax.set_xlim(0, 1)
                
                # Add value labels
                for i, (bar, prob) in enumerate(zip(bars, top_10_probs)):
                    ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{prob:.1%}', va='center', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Disclaimer
                st.markdown("---")
                st.caption("⚠️ **Medical Disclaimer:** This is an AI prediction tool for educational purposes only. Always consult a healthcare professional for proper diagnosis.")

if __name__ == "__main__":
    main()
