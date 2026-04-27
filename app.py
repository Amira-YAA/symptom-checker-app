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
    page_title="Disease Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
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
# DATA LOADING - WITH MANUAL TOKEN INPUT
# ============================================

def get_kaggle_token():
    """Get Kaggle token from multiple sources"""
    
    # Try Streamlit secrets
    try:
        token = st.secrets.get("KAGGLE_API_TOKEN")
        if token:
            return token
    except:
        pass
    
    # Try environment variable
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return token
    
    # Try .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.environ.get("KAGGLE_API_TOKEN")
        if token:
            return token
    except:
        pass
    
    return None

@st.cache_data(ttl=86400)
def load_data():
    """Load FULL dataset from Kaggle"""
    
    # Get token
    api_token = get_kaggle_token()
    
    # If no token found, show input box
    if not api_token:
        st.warning("🔑 Kaggle API token required to load the dataset")
        
        # Create a text input for manual token entry
        with st.form("kaggle_token_form"):
            st.markdown("""
            ### Please enter your Kaggle API Token:
            
            You can get your token from: https://www.kaggle.com/settings
            Click "Create New Token" and copy the token value.
            """)
            
            user_token = st.text_input("KAGGLE_API_TOKEN:", type="password", 
                                       placeholder="KGAT_xxxxxxxxxxxxxxxxxxxx")
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("✅ Load Dataset", use_container_width=True)
            with col2:
                st.markdown("[Get Kaggle Token](https://www.kaggle.com/settings)")
            
            if submitted and user_token:
                # Store the token in session state for this session
                st.session_state.user_token = user_token
                os.environ['KAGGLE_API_TOKEN'] = user_token
                st.rerun()
        
        st.stop()
    
    # Set the token
    os.environ['KAGGLE_API_TOKEN'] = api_token
    
    try:
        import kagglehub
        
        with st.spinner('📥 Loading FULL dataset (96,088 records) from Kaggle... This may take 2-3 minutes...'):
            path = kagglehub.dataset_download("rajawatprateek/symptomchecker-multi-disease-diagnostic-data")
            
            csv_file = None
            for file in os.listdir(path):
                if file.endswith('.csv'):
                    csv_file = os.path.join(path, file)
                    break
            
            if csv_file:
                df = pd.read_csv(csv_file)
                st.success(f"✅ Loaded {len(df):,} records with {len(df.columns)-1} symptoms")
                return df
            else:
                st.error("CSV file not found in downloaded dataset")
                st.stop()
                
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.info("Please check your Kaggle API token and internet connection.")
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
    """Interactive symptom pattern analyzer for disease comparison"""
    
    st.markdown("## 🔬 Symptom Pattern Analyzer")
    st.markdown("Compare symptom patterns between two diseases to identify distinguishing features")
    st.markdown("---")
    
    disease_options = sorted(df['diseases'].unique().tolist())
    
    col1, col2 = st.columns(2)
    
    with col1:
        disease1 = st.selectbox(
            "**Disease A**",
            options=disease_options,
            index=0,
            key="disease_a"
        )
    
    with col2:
        disease2 = st.selectbox(
            "**Disease B**",
            options=disease_options,
            index=min(1, len(disease_options)-1),
            key="disease_b"
        )
    
    compare_clicked = st.button("🔍 COMPARE SYMPTOMS", type="primary", use_container_width=True)
    
    if compare_clicked:
        if disease1 == disease2:
            st.warning("⚠️ Please select two different diseases for comparison")
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
                <h2>📊 Comparison Results</h2>
                <h3>{disease1.upper()} vs {disease2.upper()}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🔍 Top 20 Distinguishing Symptoms")
            
            top_distinguishing = diff_df.head(20)
            display_df = top_distinguishing.copy()
            display_df['disease1_freq'] = display_df['disease1_freq'].apply(lambda x: f"{x:.1%}")
            display_df['disease2_freq'] = display_df['disease2_freq'].apply(lambda x: f"{x:.1%}")
            display_df['difference'] = display_df['difference'].apply(lambda x: f"{x:+.1%}")
            display_df = display_df[['symptom', 'disease1_freq', 'disease2_freq', 'difference']]
            display_df.columns = ['Symptom', disease1[:30], disease2[:30], 'Difference']
            
            st.dataframe(display_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 📈 Top Symptoms in {disease1}")
                top1 = diff_df.nlargest(15, 'disease1_freq')
                fig1, ax1 = plt.subplots(figsize=(10, 8))
                colors1 = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(top1)))
                bars1 = ax1.barh(range(len(top1)), top1['disease1_freq'].values, color=colors1)
                ax1.set_yticks(range(len(top1)))
                ax1.set_yticklabels([s[:35] for s in top1['symptom'].values])
                ax1.set_xlabel('Prevalence')
                ax1.set_title(f'Most Common Symptoms in {disease1[:30]}')
                ax1.set_xlim(0, 1)
                
                for i, (bar, val) in enumerate(zip(bars1, top1['disease1_freq'].values)):
                    ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                            f'{val:.1%}', va='center', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close()
            
            with col2:
                st.markdown(f"### 📉 Top Symptoms in {disease2}")
                top2 = diff_df.nlargest(15, 'disease2_freq')
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                colors2 = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(top2)))
                bars2 = ax2.barh(range(len(top2)), top2['disease2_freq'].values, color=colors2)
                ax2.set_yticks(range(len(top2)))
                ax2.set_yticklabels([s[:35] for s in top2['symptom'].values])
                ax2.set_xlabel('Prevalence')
                ax2.set_title(f'Most Common Symptoms in {disease2[:30]}')
                ax2.set_xlim(0, 1)
                
                for i, (bar, val) in enumerate(zip(bars2, top2['disease2_freq'].values)):
                    ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                            f'{val:.1%}', va='center', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
            
            st.markdown("### 🗺️ Symptom Prevalence Comparison Heatmap")
            
            heatmap_data = diff_df.head(20).set_index('symptom')[['disease1_freq', 'disease2_freq']]
            
            fig3 = go.Figure(data=go.Heatmap(
                z=heatmap_data.values.T,
                x=heatmap_data.index,
                y=[disease1[:30], disease2[:30]],
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(heatmap_data.values.T * 100, 1),
                texttemplate='%{text}%',
                textfont={"size": 11}
            ))
            
            fig3.update_layout(
                title=f"Symptom Pattern Comparison",
                xaxis_title="Symptoms",
                yaxis_title="Disease",
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Unique Symptoms", len(symptom_cols))
            with col2:
                st.metric(f"{disease1[:15]} Patients", len(data1))
            with col3:
                st.metric(f"{disease2[:15]} Patients", len(data2))
            with col4:
                avg_diff = diff_df['abs_diff'].mean()
                st.metric("Avg Symptom Difference", f"{avg_diff:.1%}")
            
            most_distinctive = diff_df.iloc[0]
            if most_distinctive['difference'] > 0:
                distinctive_for = disease1
            else:
                distinctive_for = disease2
            
            st.info(f"""
            **💡 Key Insight:** Most distinguishing symptom is **'{most_distinctive['symptom']}'**  
            - Present in {max(most_distinctive['disease1_freq'], most_distinctive['disease2_freq']):.1%} of {distinctive_for} patients
            - Present in only {min(most_distinctive['disease1_freq'], most_distinctive['disease2_freq']):.1%} of the other disease
            - Difference: {abs(most_distinctive['difference']):.1%}
            """)
            
            csv = diff_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison Results (CSV)",
                data=csv,
                file_name=f"symptom_comparison_{disease1}_{disease2}.csv",
                mime="text/csv"
            )

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
        st.success(f"✅ Model Ready! Accuracy: {accuracy*100:.1f}%")
    
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
    st.markdown("### ℹ️ About This Project")
    
    st.markdown("""
    **🤖 Disease Prediction App**
    
    This project uses machine learning to predict diseases based on patient symptoms, organized into medical categories.
    
    **✨ Features:**
    - 🎯 Real-time disease prediction with Top 7 possibilities
    - 🔬 Symptom Pattern Analyzer to compare diseases
    - 🩺 12 symptom categories with expandable sections
    - 📊 Interactive visualizations
    - 🤖 ML model performance (82.2% accuracy on test data)
    
    **🩺 Symptom Categories:**
    - 🧠 Mental & Emotional (13 symptoms)
    - ❤️ Cardiovascular (8 symptoms)
    - 🫁 Respiratory (11 symptoms)
    - 🍽️ Digestive (13 symptoms)
    - 🧠 Neurological (8 symptoms)
    - 🚽 Genitourinary (12 symptoms)
    - 🦴 Musculoskeletal (13 symptoms)
    - 🩻 Skin & Appearance (9 symptoms)
    - 👁️ Eye & Vision (8 symptoms)
    - 🦻 Ear, Nose & Throat (8 symptoms)
    - 🩸 Systemic & General (10 symptoms)
    - 👶 Pregnancy & Reproductive (10 symptoms)
    
    **🛠️ Technology Stack:**
    - **Frontend:** Streamlit
    - **ML Framework:** Scikit-learn (Random Forest)
    - **Visualization:** Plotly, Matplotlib
    - **Data Source:** Kaggle API 
    
    **⚠️ Important Note:**
    This tool is for **educational and demonstration purposes only**. 
    It should not be used as a substitute for professional medical advice.
    """)


# ============================================
# MAIN APP
# ============================================

def main():
    st.title("🏥 Disease Prediction App")
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
        for category in list(symptom_categories.keys())[:8]:
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
