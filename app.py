import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SYMPTOM CATEGORIES (Complete)
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
if 'reset_trigger' not in st.session_state:
    st.session_state.reset_trigger = 0
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = []

# ============================================
# DATA LOADING
# ============================================

@st.cache_data(ttl=86400, show_spinner=False)
def load_data():
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
    np.random.seed(42)
    n_samples = 3000
    
    diseases = [
        'Common Cold', 'Influenza', 'Allergic Rhinitis', 'Asthma', 
        'Migraine', 'Gastroenteritis', 'Sinusitis', 'Bronchitis',
        'UTI', 'Tonsillitis', 'Anxiety Disorder', 'Major Depression', 
        'GERD', 'Hypertension', 'Osteoarthritis', 'Panic Disorder'
    ]
    
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
                if disease in ['Anxiety Disorder', 'Panic Disorder']:
                    prob = 0.7 if symptom in ['anxiety and nervousness', 'palpitations', 'insomnia', 'restlessness'] else 0.05
                elif disease in ['Major Depression']:
                    prob = 0.7 if symptom in ['depression', 'insomnia', 'fatigue', 'loss of appetite'] else 0.05
                elif disease == 'Common Cold':
                    prob = 0.7 if symptom in ['cough', 'nasal congestion', 'sore throat', 'sneezing'] else 0.05
                elif disease == 'Influenza':
                    prob = 0.7 if symptom in ['fever', 'fatigue', 'muscle weakness', 'cough'] else 0.05
                elif disease == 'Migraine':
                    prob = 0.7 if symptom in ['headache', 'dizziness', 'nausea', 'diminished vision'] else 0.05
                elif disease == 'Gastroenteritis':
                    prob = 0.7 if symptom in ['nausea', 'vomiting', 'diarrhea', 'abdominal pain'] else 0.05
                elif disease == 'GERD':
                    prob = 0.7 if symptom in ['heartburn', 'chest tightness', 'regurgitation'] else 0.05
                else:
                    prob = 0.05
                
                row[symptom] = np.random.choice([0, 1], p=[1-prob, prob])
            data.append(row)
    
    return pd.DataFrame(data)

# ============================================
# MODEL TRAINING
# ============================================

@st.cache_resource
def train_model_safe(df):
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
    """Display the disease predictor interface with expandable categories"""
    
    st.markdown("*Select your symptoms below to get a disease prediction*")
    st.markdown("---")
    
    if not st.session_state.model_trained:
        st.info("🤖 Preparing AI model for prediction...")
        success, accuracy = train_model_safe(df)
        if success:
            st.success(f"✅ AI Model Ready! (Accuracy: {accuracy*100:.1f}%)")
            st.rerun()
        else:
            st.error("❌ Failed to initialize model. Please refresh.")
            return
    
    col_reset, col_counter, col_clear = st.columns([1, 2, 1])
    
    with col_reset:
        if st.button("🗑️ Reset All Symptoms", type="secondary", use_container_width=True):
            st.session_state.reset_trigger += 1
            st.rerun()
    
    with col_counter:
        selected_count = len(st.session_state.get('selected_symptoms', []))
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 10px; border-radius: 10px; text-align: center; color: white;">
            <b>✅ SELECTED:</b> {selected_count} symptoms
        </div>
        """, unsafe_allow_html=True)
    
    with col_clear:
        if st.button("Clear All", type="secondary", use_container_width=True):
            st.session_state.reset_trigger += 1
            st.rerun()
    
    st.markdown("---")
    
    available_symptoms = set(df.columns)
    reset_key = f"reset_{st.session_state.reset_trigger}"
    selected_symptoms = []
    
    category_order = [
        "🧠 Mental & Emotional",
        "❤️ Cardiovascular", 
        "🫁 Respiratory",
        "🍽️ Digestive",
        "🧠 Neurological",
        "🚽 Genitourinary",
        "🦴 Musculoskeletal",
        "🩻 Skin & Appearance",
        "👁️ Eye & Vision",
        "🦻 Ear, Nose & Throat",
        "🩸 Systemic & General",
        "👶 Pregnancy & Reproductive"
    ]
    
    col1, col2 = st.columns(2)
    mid_point = len(category_order) // 2
    left_categories = category_order[:mid_point]
    right_categories = category_order[mid_point:]
    
    with col1:
        for category in left_categories:
            if category in symptom_categories:
                category_symptoms = symptom_categories[category]
                available_in_category = [s for s in category_symptoms if s in available_symptoms]
                
                if available_in_category:
                    with st.expander(f"{category} ({len(available_in_category)} symptoms)", expanded=False):
                        sym_cols = st.columns(2)
                        for idx, symptom in enumerate(available_in_category):
                            display_name = symptom.replace('_', ' ').title()
                            checkbox_key = f"{category}_{symptom}_{reset_key}"
                            
                            if sym_cols[idx % 2].checkbox(display_name, key=checkbox_key):
                                selected_symptoms.append(symptom)
    
    with col2:
        for category in right_categories:
            if category in symptom_categories:
                category_symptoms = symptom_categories[category]
                available_in_category = [s for s in category_symptoms if s in available_symptoms]
                
                if available_in_category:
                    with st.expander(f"{category} ({len(available_in_category)} symptoms)", expanded=False):
                        sym_cols = st.columns(2)
                        for idx, symptom in enumerate(available_in_category):
                            display_name = symptom.replace('_', ' ').title()
                            checkbox_key = f"{category}_{symptom}_{reset_key}"
                            
                            if sym_cols[idx % 2].checkbox(display_name, key=checkbox_key):
                                selected_symptoms.append(symptom)
    
    other_symptoms = [s for s in available_symptoms if s != 'diseases' and s not in all_categorized_symptoms]
    
    if other_symptoms:
        with st.expander(f"📌 Other Symptoms ({len(other_symptoms)} available)", expanded=False):
            search = st.text_input("🔍 Search symptoms:", key=f"search_{reset_key}")
            
            if search:
                filtered = [s for s in other_symptoms if search.lower() in s.lower()]
            else:
                filtered = other_symptoms[:50]
            
            st.caption(f"Showing {len(filtered)} of {len(other_symptoms)} symptoms")
            
            cols = st.columns(3)
            for idx, symptom in enumerate(filtered):
                display_name = symptom.replace('_', ' ').title()
                checkbox_key = f"other_{symptom}_{reset_key}"
                
                if cols[idx % 3].checkbox(display_name, key=checkbox_key):
                    selected_symptoms.append(symptom)
    
    st.session_state.selected_symptoms = selected_symptoms
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_btn = st.button("🔮 PREDICT DISEASE", type="primary", use_container_width=True)
    
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
                        <h1 style="margin: 10px 0; font-size: 2em;">{result['primary'].upper()}</h1>
                        <h2 style="margin: 0;">{confidence_level}: {confidence*100:.1f}%</h2>
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
                        <div style="margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span><b>{i}.</b> {icon} <b>{disease}</b></span>
                                <span style="color: {bar_color}; font-weight: bold;">{prob*100:.1f}%</span>
                            </div>
                            <div style="background-color: #ecf0f1; border-radius: 10px; overflow: hidden;">
                                <div style="background-color: {bar_color}; width: {prob*100}%; height: 35px; border-radius: 10px; line-height: 35px; padding-left: 10px; color: white; font-size: 14px;">
                                    {'★ TOP PREDICTION' if i == 1 else ''}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if st.button("🔄 New Prediction", type="secondary"):
                        st.rerun()
                
                with col2:
                    st.markdown("### Selected Symptoms")
                    st.markdown(f"**{len(selected_symptoms)} symptoms selected:**")
                    
                    symptoms_by_category = {}
                    for symptom in selected_symptoms:
                        category = get_symptom_category(symptom)
                        if category not in symptoms_by_category:
                            symptoms_by_category[category] = []
                        symptoms_by_category[category].append(symptom)
                    
                    for category, symptoms_list in symptoms_by_category.items():
                        with st.expander(f"{category} ({len(symptoms_list)})"):
                            display_symptoms = [s.replace('_', ' ').title() for s in symptoms_list]
                            for s in display_symptoms[:20]:
                                st.markdown(f"- {s}")
                            if len(symptoms_list) > 20:
                                st.markdown(f"... and {len(symptoms_list)-20} more")
                
                st.markdown("### 📈 Probability Distribution")
                
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
                
                for i, (bar, prob) in enumerate(zip(bars, top_10_probs)):
                    ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{prob:.1%}', va='center', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.markdown("---")
                st.caption("⚠️ **Medical Disclaimer:** This is an ML prediction tool for educational purposes only.")

# ============================================
# ABOUT PAGE
# ============================================

def display_about():
    st.markdown("### ℹ️ About This Project")
    
    st.markdown("""
    **🤖 Disease Prediction System**
    
    This project uses machine learning to predict diseases based on patient symptoms, organized into medical categories.
    
    **✨ Features:**
    - 🎯 Real-time disease prediction with Top 7 possibilities
    - 🔬 Symptom Pattern Analyzer to compare diseases
    - 🩺 12 symptom categories with expandable sections
    - 📊 Interactive visualizations
    - 🤖 Random Forest ML model (70-80% average accuracy)
    
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
    - **Data Source:** Kaggle API / Sample Data of 96,088 Pateints and 231 Symptoms
    
    **⚠️ Important Note:**
    This tool is for **educational and demonstration purposes only**. 
    It should not be used as a substitute for professional medical advice.
    """)

# ============================================
# MAIN APP
# ============================================

def main():
    st.title("🏥 Disease Prediction System")
    st.markdown("*AI-Powered Medical Diagnosis Assistant*")
    
    if st.session_state.df is None:
        with st.spinner("📊 Loading medical data..."):
            st.session_state.df = load_data()
    df = st.session_state.df
    
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2968/2968621.png", width=80)
        st.markdown("#  Navigation")
        
        page = st.radio(
            "Select Page",
            ["🔮 Disease Predictor", "🔬 Symptom Pattern Analyzer", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.metric("Total Patients", f"{len(df):,}")
        st.metric("Diseases", df['diseases'].nunique())
        st.metric("Symptoms", len(df.columns)-1)
        
        st.markdown("---")
        
        if st.session_state.model_trained:
            st.success(f"✅ Model Ready\nAccuracy: {st.session_state.model_accuracy*100:.1f}%")
        else:
            st.warning("⚠️ Model will train on Predictor page")
        
        st.markdown("---")
        st.markdown("### 🩺 Categories")
        for category in symptom_categories.keys():
            st.markdown(f"- {category}")
    
    if page == "🔮 Disease Predictor":
        display_predictor(df)
    elif page == "🔬 Symptom Pattern Analyzer":
        symptom_cols = [col for col in df.columns if col != 'diseases']
        symptom_pattern_analyzer(df, symptom_cols)
    elif page == "ℹ️ About":
        display_about()

if __name__ == "__main__":
    main()
