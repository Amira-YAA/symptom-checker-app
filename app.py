import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SYMPTOM CATEGORIES
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
    for category, symptoms in symptom_categories.items():
        if symptom in symptoms:
            return category
    return "📌 Other Symptoms"

# Page config
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state with persistence flags
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
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
if 'reset_trigger' not in st.session_state:
    st.session_state.reset_trigger = 0
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = []
if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = 0

# ============================================
# DATA LOADING - FULL DATASET
# ============================================

@st.cache_data(ttl=86400)
def load_data():
    """Load FULL dataset from Kaggle (no sampling)"""
    
    api_token = None
    try:
        api_token = st.secrets.get("KAGGLE_API_TOKEN")
    except:
        pass
    
    if not api_token:
        api_token = os.environ.get("KAGGLE_API_TOKEN")
    
    if not api_token:
        st.error("❌ Kaggle API token not found!")
        st.markdown("""
        ### Please add your Kaggle API token:
        
        **For Streamlit Cloud:**
        1. Go to Settings → Secrets
        2. Add: `KAGGLE_API_TOKEN = "your_token_here"`
        """)
        st.stop()
    
    os.environ['KAGGLE_API_TOKEN'] = api_token
    
    try:
        import kagglehub
        
        with st.spinner('📥 Loading FULL dataset (96,088 records) from Kaggle...'):
            path = kagglehub.dataset_download("rajawatprateek/symptomchecker-multi-disease-diagnostic-data")
            
            csv_file = None
            for file in os.listdir(path):
                if file.endswith('.csv'):
                    csv_file = os.path.join(path, file)
                    break
            
            if csv_file:
                df = pd.read_csv(csv_file)
                # NO SAMPLING - use full dataset
                st.success(f"✅ Loaded {len(df):,} records with {len(df.columns)-1} symptoms")
                return df
            else:
                st.error("CSV file not found")
                st.stop()
                
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.stop()

# ============================================
# MODEL TRAINING - PERSISTENT
# ============================================

def train_and_save_model(df):
    """Train model once and save to disk"""
    
    with st.spinner('🔄 Training AI model on 96,088 records (this will take 3-5 minutes)...'):
        X = df.drop('diseases', axis=1)
        y = df['diseases']
        
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        # Use ALL data for training
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
        
        # Save to disk
        os.makedirs("models", exist_ok=True)
        joblib.dump({
            'model': model,
            'features': X.columns.tolist(),
            'encoder': encoder,
            'accuracy': accuracy
        }, "models/disease_model.pkl")
        
        return model, X.columns.tolist(), encoder, accuracy

def load_or_train_model(df):
    """Load saved model or train new one"""
    
    model_path = "models/disease_model.pkl"
    
    # Try to load existing model
    if os.path.exists(model_path):
        try:
            data = joblib.load(model_path)
            st.session_state.model = data['model']
            st.session_state.features = data['features']
            st.session_state.disease_encoder = data['encoder']
            st.session_state.model_accuracy = data['accuracy']
            st.session_state.model_trained = True
            st.session_state.model_loaded = True
            return True, data['accuracy']
        except:
            pass
    
    # Train new model if not exists or corrupt
    model, features, encoder, accuracy = train_and_save_model(df)
    st.session_state.model = model
    st.session_state.features = features
    st.session_state.disease_encoder = encoder
    st.session_state.model_accuracy = accuracy
    st.session_state.model_trained = True
    st.session_state.model_loaded = True
    return True, accuracy

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

# ============================================
# SYMPTOM PATTERN ANALYZER
# ============================================

def symptom_pattern_analyzer(df, symptom_cols):
    """Compare two diseases"""
    
    st.markdown("## 🔬 Symptom Pattern Analyzer")
    st.markdown("Compare symptom patterns between two diseases")
    st.markdown("---")
    
    disease_options = sorted(df['diseases'].unique().tolist())
    
    col1, col2 = st.columns(2)
    with col1:
        disease1 = st.selectbox("**Disease A**", disease_options, index=0, key="disease_a")
    with col2:
        disease2 = st.selectbox("**Disease B**", disease_options, index=min(1, len(disease_options)-1), key="disease_b")
    
    if st.button("🔍 COMPARE SYMPTOMS", type="primary", use_container_width=True):
        if disease1 == disease2:
            st.warning("Select two different diseases")
            return
        
        with st.spinner("Analyzing symptom patterns..."):
            data1 = df[df['diseases'] == disease1]
            data2 = df[df['diseases'] == disease2]
            
            freq1 = data1[symptom_cols].sum() / len(data1)
            freq2 = data2[symptom_cols].sum() / len(data2)
            
            diff_df = pd.DataFrame({
                'symptom': symptom_cols,
                'disease1_freq': freq1.values,
                'disease2_freq': freq2.values,
                'difference': freq1.values - freq2.values,
                'abs_diff': np.abs(freq1.values - freq2.values)
            }).sort_values('abs_diff', ascending=False)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;">
                <h3>{disease1.upper()} vs {disease2.upper()}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🔍 Top 20 Distinguishing Symptoms")
            display_df = diff_df.head(20).copy()
            display_df['disease1_freq'] = display_df['disease1_freq'].apply(lambda x: f"{x:.1%}")
            display_df['disease2_freq'] = display_df['disease2_freq'].apply(lambda x: f"{x:.1%}")
            display_df['difference'] = display_df['difference'].apply(lambda x: f"{x:+.1%}")
            display_df = display_df[['symptom', 'disease1_freq', 'disease2_freq', 'difference']]
            display_df.columns = ['Symptom', disease1[:25], disease2[:25], 'Difference']
            st.dataframe(display_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 📈 Top Symptoms in {disease1[:20]}")
                top1 = diff_df.nlargest(10, 'disease1_freq')
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(range(len(top1)), top1['disease1_freq'].values, color='#2ecc71')
                ax.set_yticks(range(len(top1)))
                ax.set_yticklabels([s[:30] for s in top1['symptom'].values])
                ax.set_xlabel('Prevalence')
                ax.set_xlim(0, 1)
                for i, val in enumerate(top1['disease1_freq'].values):
                    ax.text(val + 0.02, i, f'{val:.1%}', va='center')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown(f"### 📉 Top Symptoms in {disease2[:20]}")
                top2 = diff_df.nlargest(10, 'disease2_freq')
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(range(len(top2)), top2['disease2_freq'].values, color='#e74c3c')
                ax.set_yticks(range(len(top2)))
                ax.set_yticklabels([s[:30] for s in top2['symptom'].values])
                ax.set_xlabel('Prevalence')
                ax.set_xlim(0, 1)
                for i, val in enumerate(top2['disease2_freq'].values):
                    ax.text(val + 0.02, i, f'{val:.1%}', va='center')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            st.markdown("### 🗺️ Symptom Comparison Heatmap")
            heatmap_data = diff_df.head(15).set_index('symptom')[['disease1_freq', 'disease2_freq']]
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values.T,
                x=heatmap_data.index,
                y=[disease1[:20], disease2[:20]],
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(heatmap_data.values.T * 100, 1),
                texttemplate='%{text}%'
            ))
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Symptoms", len(symptom_cols))
            col2.metric(f"{disease1[:10]} Patients", len(data1))
            col3.metric(f"{disease2[:10]} Patients", len(data2))
            col4.metric("Avg Difference", f"{diff_df['abs_diff'].mean():.1%}")
            
            st.download_button("📥 Download CSV", diff_df.to_csv(index=False), 
                              f"comparison_{disease1}_{disease2}.csv", "text/csv")

# ============================================
# DISEASE PREDICTOR PAGE
# ============================================

def display_predictor(df):
    """Main prediction interface"""
    
    st.markdown("*Select your symptoms below to get a prediction*")
    st.markdown("---")
    
    # Load or train model ONLY ONCE
    if not st.session_state.model_loaded:
        success, accuracy = load_or_train_model(df)
        if not success:
            st.error("Failed to load/train model")
            return
    
    # Controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("🗑️ Reset All", use_container_width=True):
            st.session_state.reset_trigger += 1
            st.rerun()
    with col2:
        selected_count = len(st.session_state.get('selected_symptoms', []))
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                    padding: 10px; border-radius: 10px; text-align: center; color: white;">
            ✅ SELECTED: {selected_count} symptoms
        </div>
        """, unsafe_allow_html=True)
    with col3:
        if st.button("Clear All", use_container_width=True):
            st.session_state.reset_trigger += 1
            st.rerun()
    
    st.markdown("---")
    
    available_symptoms = set(df.columns)
    reset_key = f"reset_{st.session_state.reset_trigger}"
    selected_symptoms = []
    
    category_order = [
        "🧠 Mental & Emotional", "❤️ Cardiovascular", "🫁 Respiratory",
        "🍽️ Digestive", "🧠 Neurological", "🚽 Genitourinary",
        "🦴 Musculoskeletal", "🩻 Skin & Appearance", "👁️ Eye & Vision",
        "🦻 Ear, Nose & Throat", "🩸 Systemic & General", "👶 Pregnancy & Reproductive"
    ]
    
    col1, col2 = st.columns(2)
    mid = len(category_order) // 2
    
    for idx, category in enumerate(category_order):
        with col1 if idx < mid else col2:
            if category in symptom_categories:
                available = [s for s in symptom_categories[category] if s in available_symptoms]
                if available:
                    with st.expander(f"{category} ({len(available)})", expanded=False):
                        cols = st.columns(2)
                        for i, symptom in enumerate(available):
                            name = symptom.replace('_', ' ').title()
                            key = f"{category}_{symptom}_{reset_key}"
                            if cols[i % 2].checkbox(name, key=key):
                                selected_symptoms.append(symptom)
    
    other = [s for s in available_symptoms if s != 'diseases' and s not in all_categorized_symptoms]
    if other:
        with st.expander(f"📌 Other Symptoms ({len(other)})", expanded=False):
            search = st.text_input("🔍 Search", key=f"search_{reset_key}")
            filtered = [s for s in other if search.lower() in s.lower()] if search else other[:50]
            cols = st.columns(3)
            for i, symptom in enumerate(filtered):
                name = symptom.replace('_', ' ').title()
                key = f"other_{symptom}_{reset_key}"
                if cols[i % 3].checkbox(name, key=key):
                    selected_symptoms.append(symptom)
    
    st.session_state.selected_symptoms = selected_symptoms
    
    st.markdown("---")
    if st.button("🔮 PREDICT DISEASE", type="primary", use_container_width=True):
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom")
        else:
            with st.spinner("🔄 Analyzing symptoms with AI..."):
                result = predict_disease(selected_symptoms, st.session_state.model,
                                        st.session_state.features, st.session_state.disease_encoder)
                
                st.markdown("## 🎯 Prediction Results")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    conf = result['primary_confidence']
                    if conf > 0.7:
                        color = "#27ae60"
                        level = "HIGH"
                        emoji = "🎯"
                    elif conf > 0.4:
                        color = "#f39c12"
                        level = "MEDIUM"
                        emoji = "✅"
                    else:
                        color = "#e74c3c"
                        level = "LOW"
                        emoji = "⚠️"
                    
                    st.markdown(f"""
                    <div style="background: {color}; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                        <h2>{emoji} Most Likely Disease</h2>
                        <h1>{result['primary'].upper()}</h1>
                        <h3>{level} CONFIDENCE: {conf*100:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### Top 7 Possible Diseases")
                    
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
                        <div style="margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between;">
                                <b>{i}. {icon} {disease}</b>
                                <span style="color: {bar_color};">{prob*100:.1f}%</span>
                            </div>
                            <div style="background: #ecf0f1; border-radius: 5px;">
                                <div style="background: {bar_color}; width: {prob*100}%; height: 25px; border-radius: 5px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if st.button("🔄 New Prediction"):
                        st.rerun()
                
                with col2:
                    st.markdown(f"### Selected Symptoms ({len(selected_symptoms)})")
                    by_cat = {}
                    for s in selected_symptoms:
                        cat = get_symptom_category(s)
                        by_cat.setdefault(cat, []).append(s)
                    
                    for cat, syms in by_cat.items():
                        with st.expander(f"{cat} ({len(syms)})"):
                            for s in syms[:15]:
                                st.markdown(f"- {s.replace('_', ' ').title()}")
                
                st.markdown("### 📈 Probability Distribution")
                
                top10_idx = np.argsort(result['all_probabilities'])[-10:][::-1]
                top10_diseases = result['all_diseases'][top10_idx]
                top10_probs = result['all_probabilities'][top10_idx]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['#27ae60', '#3498db', '#f39c12', '#e67e22', '#e74c3c',
                         '#9b59b6', '#1abc9c', '#34495e', '#95a5a6', '#7f8c8d']
                
                ax.barh(range(len(top10_diseases)), top10_probs, color=colors[:len(top10_diseases)])
                ax.set_yticks(range(len(top10_diseases)))
                ax.set_yticklabels([d[:30] for d in top10_diseases])
                ax.set_xlabel('Probability')
                ax.set_xlim(0, 1)
                
                for i, prob in enumerate(top10_probs):
                    ax.text(prob + 0.01, i, f'{prob:.1%}', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.caption("⚠️ **Medical Disclaimer:** For educational purposes only.")

# ============================================
# ABOUT PAGE
# ============================================

def display_about():
    st.markdown("""
    ## ℹ️ About This System
    
    **🤖 AI-Powered Disease Prediction System**
    
    This system uses machine learning to predict diseases based on patient symptoms.
    
    ### 📊 Model Performance
    - **Accuracy:** 82.2% on test data
    - **Training Data:** 96,088 patient records
    - **Symptoms:** 230 features
    - **Diseases:** 100 conditions
    
    ### 🩺 Symptom Categories (12 categories, 113 symptoms)
    - 🧠 Mental & Emotional (13)
    - ❤️ Cardiovascular (8)
    - 🫁 Respiratory (11)
    - 🍽️ Digestive (13)
    - 🧠 Neurological (8)
    - 🚽 Genitourinary (12)
    - 🦴 Musculoskeletal (13)
    - 🩻 Skin & Appearance (9)
    - 👁️ Eye & Vision (8)
    - 🦻 Ear, Nose & Throat (8)
    - 🩸 Systemic & General (10)
    - 👶 Pregnancy & Reproductive (10)
    
    ⚠️ **Disclaimer:** For educational purposes only.
    """)

# ============================================
# MAIN APP
# ============================================

def main():
    st.title("🏥 AI Disease Prediction System")
    st.markdown("*Powered by Machine Learning | 96,088 Patient Records*")
    
    # Load data
    if st.session_state.df is None:
        st.session_state.df = load_data()
    df = st.session_state.df
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2968/2968621.png", width=80)
        st.markdown("# Navigation")
        
        page = st.radio("Select Page", ["🔮 Disease Predictor", "🔬 Symptom Pattern Analyzer", "ℹ️ About"])
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.metric("Total Patients", f"{len(df):,}")
        st.metric("Diseases", df['diseases'].nunique())
        st.metric("Symptoms", len(df.columns)-1)
        
        st.markdown("---")
        
        if st.session_state.model_trained:
            st.success(f"✅ Model Ready\n{st.session_state.model_accuracy*100:.1f}% accuracy")
        else:
            st.info("⚡ Loading model...")
        
        st.markdown("---")
        st.markdown("### 🩺 Categories")
        for category in list(symptom_categories.keys())[:8]:  # Show first 8
            st.markdown(f"- {category}")
    
    # Page routing
    if page == "🔮 Disease Predictor":
        display_predictor(df)
    elif page == "🔬 Symptom Pattern Analyzer":
        symptom_cols = [c for c in df.columns if c != 'diseases']
        symptom_pattern_analyzer(df, symptom_cols)
    else:
        display_about()

if __name__ == "__main__":
    main()
